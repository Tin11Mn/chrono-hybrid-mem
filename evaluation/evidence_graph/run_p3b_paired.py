"""Fixed-config paired P1/P3-B1 retrieval evaluation on a prepared v4 DB.

P3-B1 is intentionally separate from P3-A: it compares normal P1 with the
source-local entity-mention channel, never enables semantic graph traversal,
and retains only raw ``mem_*`` rows as candidates.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence

from app.model import MemoryModel
from app.storage import MemoryStore
from evaluation.evidence_graph.evidence_mention_audit import (
    audit_entity_mention_database,
)
from evaluation.evidence_graph.run_locomo_paired import (
    _FrozenSearchModel,
    _aggregate_arm,
    _atomic_write_json,
    _mem_numeric_id,
    _required_text,
    _sha256_json,
    _validate_source_map,
    _verify_source_map_database,
    freeze_structured_plans,
    load_dataset_manifest,
    load_prepared_database,
    logical_database_digest,
    score_question_rankings,
)


ANCHOR_SEED_LIMIT = 6
ANCHOR_CANDIDATES_PER_SEED = 20
ANCHOR_MAX_CANDIDATES = 20
ANCHOR_RRF_WEIGHT = 0.025
ANCHOR_RERANK_QUOTA = 4
MENTION_COVERAGE_GATE = 0.80


def _parse_top_ks(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(item) for item in value.split(","))))
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-k must be comma-separated integers") from error
    if not values or values[0] < 1 or values[-1] > 100:
        raise argparse.ArgumentTypeError("top-k values must be between 1 and 100")
    return values


def _require_loopback_base_url(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("P3-B1 requires an explicit local loopback base URL")
    result = value.strip().rstrip("/")
    if not result.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("P3-B1 local model must use a loopback base URL")
    return result


def _anchor_trace_audit(
    database_path: str | Path,
    *,
    user_id: str,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every anchor-channel diagnostic ID to an audited raw witness."""

    violations: list[str] = []
    candidate_ids = [str(value) for value in trace.get("anchor_candidate_ids", [])]
    reserved_ids = [str(value) for value in trace.get("reserved_anchor_ids", [])]
    rerank_ids = [str(value) for value in trace.get("rerank_pool_ids", [])]
    diagnostics = trace.get("anchor_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        violations.append("anchor diagnostics are absent")
        diagnostics = {}
    expected = {
        "enabled": True,
        "seed_limit": ANCHOR_SEED_LIMIT,
        "candidate_limit_per_seed": ANCHOR_CANDIDATES_PER_SEED,
        "candidate_limit": ANCHOR_MAX_CANDIDATES,
    }
    for key, value in expected.items():
        if diagnostics.get(key) != value:
            violations.append("anchor configuration mismatch: {}".format(key))
    if len(candidate_ids) != len(set(candidate_ids)) or len(candidate_ids) > ANCHOR_MAX_CANDIDATES:
        violations.append("anchor candidates violate the fixed total cap")
    if len(reserved_ids) > ANCHOR_RERANK_QUOTA:
        violations.append("anchor reservation exceeds the fixed quota")
    if len(rerank_ids) > MemoryStore.MODEL_RERANK_LIMIT:
        violations.append("rerank pool exceeds the model limit")
    resolved = diagnostics.get("resolved_seeds", [])
    labels = {
        str(item.get("entity_label"))
        for item in resolved
        if isinstance(item, Mapping) and isinstance(item.get("entity_label"), str)
    }
    visits = diagnostics.get("candidate_visit_seed_labels", [])
    if not isinstance(visits, list) or any(str(item) not in labels for item in visits):
        violations.append("anchor visit sequence contains an unresolved seed")
    verified = 0
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        for candidate_id in candidate_ids:
            try:
                raw_id = _mem_numeric_id(candidate_id)
            except ValueError:
                violations.append("invalid anchor candidate ID: {}".format(candidate_id))
                continue
            row = connection.execute(
                """SELECT raw.id
                   FROM raw_messages AS raw
                   JOIN graph_entity_mentions AS mention
                     ON mention.source_message_id = raw.id
                    AND mention.user_id = raw.user_id
                   JOIN graph_entity_declarations AS declaration
                     ON declaration.id = mention.declaration_id
                    AND declaration.user_id = mention.user_id
                    AND declaration.source_message_id = mention.source_message_id
                   WHERE raw.id = ? AND raw.user_id = ?
                     AND declaration.entity_label = mention.entity_label""",
                (raw_id, user_id),
            ).fetchone()
            if row is None:
                violations.append("anchor candidate lacks a raw mention witness: {}".format(candidate_id))
            else:
                verified += 1
    finally:
        connection.close()
    return {"verified_anchor_candidates": verified, "violations": violations}


def run_p3b_paired_evaluation(
    manifest: Mapping[str, Any],
    database_path: str | Path,
    source_map_records: Sequence[Mapping[str, Any]],
    *,
    frozen_plans: Mapping[str, Mapping[str, Any]],
    rank_model: object,
    top_ks: Sequence[int] = (1, 3, 10),
) -> dict[str, Any]:
    """Evaluate baseline P1 against exactly one P3-B1 anchor channel."""

    top_ks = tuple(sorted(set(int(value) for value in top_ks)))
    if not top_ks or top_ks[0] < 1 or top_ks[-1] > 100:
        raise ValueError("top_ks must be between 1 and 100")
    mention_audit = audit_entity_mention_database(database_path)
    if (
        mention_audit["violations"]
        or not mention_audit["all_mentions_witnessed"]
        or float(mention_audit["source_coverage"]) < MENTION_COVERAGE_GATE
    ):
        raise RuntimeError("P3-B1 pre-Search mention audit or coverage gate failed")
    records, source_by_mem = _validate_source_map(manifest, source_map_records)
    _verify_source_map_database(manifest, database_path, records)
    questions = list(manifest["questions"])
    if {str(question["query_key"]) for question in questions} - set(frozen_plans):
        raise ValueError("frozen plan set is missing selected questions")
    database_path = str(Path(database_path).resolve())
    digest_before = logical_database_digest(database_path)
    baseline_model = _FrozenSearchModel(frozen_plans, rank_model)
    anchor_model = _FrozenSearchModel(frozen_plans, rank_model)
    baseline_store = MemoryStore(
        database_path, model=baseline_model, structured_query_plan=True,
    )
    anchor_store = MemoryStore(
        database_path, model=anchor_model, structured_query_plan=True,
        evidence_anchors=True, anchor_seed_limit=ANCHOR_SEED_LIMIT,
        anchor_max_candidates=ANCHOR_MAX_CANDIDATES,
        anchor_rrf_weight=ANCHOR_RRF_WEIGHT,
        anchor_rerank_quota=ANCHOR_RERANK_QUOTA,
    )
    baseline_scores: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    anchor_scores: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    traces: list[dict[str, Any]] = []
    for question in questions:
        query = _required_text(question.get("question"), "question.question")
        options = list(question.get("options", []))
        user_id = _required_text(question.get("user_id"), "question.user_id")
        expected_plan = frozen_plans[str(question["query_key"])]
        started = time.perf_counter()
        baseline_results = baseline_store.search(
            user_id=user_id, query=query, options=options, top_k=max(top_ks)
        )
        baseline_elapsed = time.perf_counter() - started
        baseline_trace = baseline_store.last_retrieval_trace
        started = time.perf_counter()
        anchor_results = anchor_store.search(
            user_id=user_id, query=query, options=options, top_k=max(top_ks)
        )
        anchor_elapsed = time.perf_counter() - started
        anchor_trace = anchor_store.last_retrieval_trace
        if baseline_trace.get("plan") != expected_plan or anchor_trace.get("plan") != expected_plan:
            raise RuntimeError("paired arm did not use the frozen structured plan")
        for field in ("p1_channels", "p1_union_ids", "p1_pre_rerank_ids", "p1_counterfactual_top30_ids"):
            if baseline_trace.get(field) != anchor_trace.get(field):
                raise RuntimeError("paired P1 pre-rerank trace differs: {}".format(field))
        anchor_audit = _anchor_trace_audit(database_path, user_id=user_id, trace=anchor_trace)
        if anchor_audit["violations"]:
            raise RuntimeError("P3-B1 trace audit failed: {}".format(anchor_audit["violations"][:3]))
        baseline_score = score_question_rankings(
            question, [str(item.id) for item in baseline_results],
            source_by_mem=source_by_mem, top_ks=top_ks,
        )
        anchor_score = score_question_rankings(
            question, [str(item.id) for item in anchor_results],
            source_by_mem=source_by_mem, top_ks=top_ks,
        )
        baseline_scores.append((question, baseline_score))
        anchor_scores.append((question, anchor_score))
        traces.append({
            "case_id": question["case_id"],
            "frozen_plan_sha256": _sha256_json(expected_plan),
            "baseline": {**baseline_score, "latency_seconds": baseline_elapsed},
            "anchor_on": {**anchor_score, "latency_seconds": anchor_elapsed,
                          "anchor_trace": copy.deepcopy(anchor_trace),
                          "trace_audit": anchor_audit},
        })
    digest_after = logical_database_digest(database_path)
    if digest_before != digest_after:
        raise RuntimeError("paired Search mutated the prepared database")
    if baseline_model.logical_plan_calls != len(questions) or anchor_model.logical_plan_calls != len(questions):
        raise RuntimeError("paired logical planner call count mismatch")
    if baseline_model.rank_calls != len(questions) or anchor_model.rank_calls != len(questions):
        raise RuntimeError("paired ranker call count mismatch")
    baseline = _aggregate_arm(baseline_scores, top_ks=top_ks)
    anchor_on = _aggregate_arm(anchor_scores, top_ks=top_ks)
    return {
        "scope": "paired shared-Add LoCoMo retrieval; P1 versus source-local P3-B1 anchors",
        "questions": len(questions), "messages": manifest["message_count"],
        "top_ks": list(top_ks), "database_unchanged": True,
        "p3b_configuration": {"seed_limit": ANCHOR_SEED_LIMIT,
            "candidates_per_seed": ANCHOR_CANDIDATES_PER_SEED,
            "max_candidates": ANCHOR_MAX_CANDIDATES,
            "rrf_weight": ANCHOR_RRF_WEIGHT, "rerank_quota": ANCHOR_RERANK_QUOTA},
        "mention_audit": mention_audit,
        "baseline": baseline, "anchor_on": anchor_on,
        "delta_hit_at_k": {str(k): anchor_on["hit_at_k"][str(k)] - baseline["hit_at_k"][str(k)] for k in top_ks},
        "delta_mrr": anchor_on["mrr"] - baseline["mrr"],
        "search_calls": {"baseline_logical_plan_calls": baseline_model.logical_plan_calls,
            "anchor_logical_plan_calls": anchor_model.logical_plan_calls,
            "baseline_rank_calls": baseline_model.rank_calls, "anchor_rank_calls": anchor_model.rank_calls,
            "additional_anchor_search_calls": 0},
        "question_traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed P3-B1 paired LoCoMo retrieval")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prepared-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument("--search-base-url", required=True)
    parser.add_argument("--search-model", default="local")
    parser.add_argument("--top-k", type=_parse_top_ks, default=(1, 3, 10))
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit("refusing to overwrite existing output: {}".format(output_path))
    base_url = _require_loopback_base_url(args.search_base_url)
    manifest = load_dataset_manifest(args.dataset, max_questions=args.max_questions, question_offset=args.question_offset)
    prepared = load_prepared_database(manifest, args.prepared_db)
    model = MemoryModel("local-only", model_name=args.search_model, base_url=base_url, disable_thinking=True)
    plans = freeze_structured_plans(list(manifest["questions"]), model)
    report = run_p3b_paired_evaluation(manifest, args.prepared_db, prepared["source_map"], frozen_plans=plans, rank_model=model, top_ks=args.top_k)
    result = {"stage": "p3b_fixed_config_paired_evaluation", "prepared_database": {key: value for key, value in prepared.items() if key != "source_map"}, "frozen_plan_external_calls": len(plans), "search_model": {"name": args.search_model, "base_url": base_url, "call_count": model.call_count, "truncated_calls": model.truncated_calls}, "paired": report}
    _atomic_write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
