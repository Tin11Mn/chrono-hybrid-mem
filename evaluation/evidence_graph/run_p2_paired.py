"""Fixed paired P1/P2 set-aware reranking evaluation on one prepared DB.

Plans are frozen before either arm searches.  P2 may only reorder the existing
P1 final ranking; this runner fails closed if its trace proves otherwise.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.model import MemoryModel
from app.storage import MemoryStore
from evaluation.evidence_graph.run_locomo_paired import (
    _FrozenSearchModel, _aggregate_arm, _atomic_write_json, _required_text,
    _sha256_json, _validate_source_map, _verify_source_map_database,
    freeze_structured_plans, load_dataset_manifest, load_prepared_database,
    logical_database_digest, score_question_rankings,
)


def _top_ks(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted(set(int(item) for item in value.split(","))))
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-k must be comma-separated integers") from error
    if not result or result[0] < 1 or result[-1] > 100:
        raise argparse.ArgumentTypeError("top-k values must be between 1 and 100")
    return result


def _audit_p2_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Verify P2's trace is a permutation-only, plan-derived diagnostic."""
    violations: list[str] = []
    if trace.get("p2_enabled") is not True:
        violations.append("P2 is not recorded as enabled")
    before = [str(item) for item in trace.get("p2_pre_rerank_ids", [])]
    after = [str(item) for item in trace.get("p2_post_rerank_ids", [])]
    if len(before) != len(after) or set(before) != set(after) or len(before) != len(set(before)):
        violations.append("P2 changed candidate membership")
    needs = trace.get("p2_evidence_need_tokens", [])
    if not isinstance(needs, list) or any(not isinstance(item, str) for item in needs):
        violations.append("P2 evidence needs are malformed")
        needs = []
    need_set = set(needs)
    covered: set[str] = set()
    new_tokens = trace.get("p2_newly_covered_tokens", [])
    if not isinstance(new_tokens, list):
        violations.append("P2 coverage diagnostic is malformed")
        new_tokens = []
    seen_candidate_ids: set[str] = set()
    for item in new_tokens:
        if not isinstance(item, Mapping):
            violations.append("P2 coverage diagnostic entry is malformed")
            continue
        candidate_id = item.get("candidate_id")
        values = item.get("newly_covered_tokens")
        if not isinstance(candidate_id, str) or candidate_id not in after or candidate_id in seen_candidate_ids:
            violations.append("P2 coverage diagnostic names a non-candidate")
            continue
        seen_candidate_ids.add(candidate_id)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            violations.append("P2 newly covered tokens are malformed")
            continue
        value_set = set(values)
        if not value_set <= need_set or value_set & covered:
            violations.append("P2 coverage diagnostic is not incremental")
        covered.update(value_set)
    return {"candidate_count": len(after), "need_token_count": len(needs), "violations": violations}


def run_p2_paired_evaluation(manifest: Mapping[str, Any], database_path: str | Path, source_map_records: Sequence[Mapping[str, Any]], *, frozen_plans: Mapping[str, Mapping[str, Any]], rank_model: object, top_ks: Sequence[int] = (1, 3, 10)) -> dict[str, Any]:
    top_ks = tuple(sorted(set(int(value) for value in top_ks)))
    if not top_ks or top_ks[0] < 1 or top_ks[-1] > 100:
        raise ValueError("top_ks must be between 1 and 100")
    records, source_by_mem = _validate_source_map(manifest, source_map_records)
    _verify_source_map_database(manifest, database_path, records)
    questions = list(manifest["questions"])
    if {str(item["query_key"]) for item in questions} - set(frozen_plans):
        raise ValueError("frozen plan set is missing selected questions")
    database_path = str(Path(database_path).resolve())
    digest_before = logical_database_digest(database_path)
    baseline_model = _FrozenSearchModel(frozen_plans, rank_model)
    p2_model = _FrozenSearchModel(frozen_plans, rank_model)
    baseline_store = MemoryStore(database_path, model=baseline_model, structured_query_plan=True)
    p2_store = MemoryStore(database_path, model=p2_model, structured_query_plan=True, set_aware_rerank=True)
    baseline_scores: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    p2_scores: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    traces: list[dict[str, Any]] = []
    for question in questions:
        query = _required_text(question.get("question"), "question.question")
        options = list(question.get("options", []))
        user_id = _required_text(question.get("user_id"), "question.user_id")
        expected_plan = frozen_plans[str(question["query_key"])]
        baseline = baseline_store.search(user_id=user_id, query=query, options=options, top_k=max(top_ks))
        baseline_trace = baseline_store.last_retrieval_trace
        p2 = p2_store.search(user_id=user_id, query=query, options=options, top_k=max(top_ks))
        p2_trace = p2_store.last_retrieval_trace
        if baseline_trace.get("plan") != expected_plan or p2_trace.get("plan") != expected_plan:
            raise RuntimeError("arm did not use its frozen structured plan")
        for field in ("p1_channels", "p1_union_ids", "p1_pre_rerank_ids", "p1_counterfactual_top30_ids", "rerank_pool_ids"):
            if baseline_trace.get(field) != p2_trace.get(field):
                raise RuntimeError("P1 path differs before P2: {}".format(field))
        if baseline_trace.get("p2_enabled") is not False:
            raise RuntimeError("baseline unexpectedly enabled P2")
        audit = _audit_p2_trace(p2_trace)
        if audit["violations"]:
            raise RuntimeError("P2 trace audit failed: {}".format(audit["violations"][:3]))
        baseline_score = score_question_rankings(question, [str(item.id) for item in baseline], source_by_mem=source_by_mem, top_ks=top_ks)
        p2_score = score_question_rankings(question, [str(item.id) for item in p2], source_by_mem=source_by_mem, top_ks=top_ks)
        baseline_scores.append((question, baseline_score)); p2_scores.append((question, p2_score))
        traces.append({"case_id": question["case_id"], "frozen_plan_sha256": _sha256_json(expected_plan), "baseline": baseline_score, "p2_on": {**p2_score, "trace": copy.deepcopy(p2_trace), "audit": audit}})
    if logical_database_digest(database_path) != digest_before:
        raise RuntimeError("paired Search mutated prepared DB")
    expected_calls = len(questions)
    if any(value != expected_calls for value in (baseline_model.logical_plan_calls, p2_model.logical_plan_calls, baseline_model.rank_calls, p2_model.rank_calls)):
        raise RuntimeError("paired local model call count mismatch")
    baseline_arm = _aggregate_arm(baseline_scores, top_ks=top_ks); p2_arm = _aggregate_arm(p2_scores, top_ks=top_ks)
    return {"questions": len(questions), "database_unchanged": True, "configuration": {"set_aware_rerank": True, "rerank_limit": MemoryStore.MODEL_RERANK_LIMIT}, "baseline": baseline_arm, "p2_on": p2_arm, "delta_hit_at_k": {str(k): p2_arm["hit_at_k"][str(k)] - baseline_arm["hit_at_k"][str(k)] for k in top_ks}, "delta_mrr": p2_arm["mrr"] - baseline_arm["mrr"], "calls": {"baseline_plan": baseline_model.logical_plan_calls, "p2_plan": p2_model.logical_plan_calls, "baseline_rank": baseline_model.rank_calls, "p2_rank": p2_model.rank_calls}, "traces": traces}


def _require_loopback_base_url(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("P2 requires an explicit local loopback base URL")
    result = value.strip().rstrip("/")
    if not result.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("P2 local model must use a loopback base URL")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True); parser.add_argument("--prepared-db", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int, default=20); parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument("--search-base-url", required=True); parser.add_argument("--search-model", default="local"); parser.add_argument("--top-k", type=_top_ks, default=(1, 3, 10))
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise SystemExit("refusing to overwrite output")
    try: base_url = _require_loopback_base_url(args.search_base_url)
    except ValueError as error: raise SystemExit(str(error)) from error
    manifest = load_dataset_manifest(args.dataset, max_questions=args.max_questions, question_offset=args.question_offset)
    # P2 is Search-only.  A P2-only storage change alters the broad historical
    # materialization fingerprint but cannot alter this already-audited DB.
    # All database/source-map/support/mention digest checks still run.
    prepared = load_prepared_database(
        manifest, args.prepared_db, allow_materialization_contract_drift=True,
    )
    model = MemoryModel("local-only", model_name=args.search_model, base_url=base_url, disable_thinking=True)
    plans = freeze_structured_plans(list(manifest["questions"]), model)
    paired = run_p2_paired_evaluation(manifest, args.prepared_db, prepared["source_map"], frozen_plans=plans, rank_model=model, top_ks=args.top_k)
    report = {"stage": "p2_fixed20_paired", "frozen_plan_external_calls": len(plans), "model_calls": model.call_count, "truncated_calls": model.truncated_calls, "paired": paired}
    _atomic_write_json(output, report); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
