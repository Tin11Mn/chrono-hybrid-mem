"""Fixed paired P1 versus P1.1 same-session-neighbor evaluation."""
from __future__ import annotations

import argparse, copy, json, sqlite3, time
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.model import MemoryModel
from app.storage import MemoryStore
from evaluation.evidence_graph.run_locomo_paired import (
    _FrozenSearchModel, _aggregate_arm, _atomic_write_json, _mem_numeric_id,
    _required_text, _sha256_json, _validate_source_map,
    _verify_source_map_database, freeze_structured_plans, load_dataset_manifest,
    load_prepared_database, logical_database_digest, score_question_rankings,
)

SEED_LIMIT = CANDIDATE_LIMIT = 4


def _top_ks(value: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(part) for part in value.split(","))))
    if not values or values[0] < 1 or values[-1] > 100:
        raise argparse.ArgumentTypeError("top-k must be between 1 and 100")
    return values


def _adjacent_audit(database: str | Path, *, user_id: str, trace: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    seeds = [str(value) for value in trace.get("adjacent_seed_ids", [])]
    candidates = [str(value) for value in trace.get("adjacent_candidate_ids", [])]
    reserved = [str(value) for value in trace.get("reserved_adjacent_ids", [])]
    pool = [str(value) for value in trace.get("rerank_pool_ids", [])]
    diagnostics = trace.get("adjacent_diagnostics", {})
    if not isinstance(diagnostics, Mapping) or diagnostics.get("enabled") is not True or diagnostics.get("seed_limit") != SEED_LIMIT or diagnostics.get("candidate_limit") != CANDIDATE_LIMIT:
        violations.append("adjacent configuration mismatch")
    if len(seeds) > SEED_LIMIT or len(candidates) > CANDIDATE_LIMIT or len(candidates) != len(set(candidates)):
        violations.append("adjacent candidate cap mismatch")
    if len(reserved) > CANDIDATE_LIMIT or len(pool) > MemoryStore.MODEL_RERANK_LIMIT:
        violations.append("adjacent rerank reservation/pool mismatch")
    connection = sqlite3.connect(str(database)); connection.row_factory = sqlite3.Row
    try:
        seed_rows = {}
        for item in seeds:
            try: raw_id = _mem_numeric_id(item)
            except ValueError: violations.append("invalid adjacent seed ID"); continue
            row = connection.execute("SELECT id, session_id, sequence FROM raw_messages WHERE id=? AND user_id=?", (raw_id, user_id)).fetchone()
            if row is None: violations.append("adjacent seed is not a user raw row")
            else: seed_rows[raw_id] = row
        verified = 0
        for item in candidates:
            try: raw_id = _mem_numeric_id(item)
            except ValueError: violations.append("invalid adjacent candidate ID"); continue
            row = connection.execute("SELECT id, session_id, sequence FROM raw_messages WHERE id=? AND user_id=?", (raw_id, user_id)).fetchone()
            if row is None or not any(row["session_id"] == seed["session_id"] and abs(int(row["sequence"]) - int(seed["sequence"])) == 1 for seed in seed_rows.values()):
                violations.append("candidate is not an immediate same-session raw neighbor")
            else: verified += 1
    finally: connection.close()
    return {"verified_adjacent_candidates": verified, "violations": violations}


def run(manifest: Mapping[str, Any], database: str | Path, source_map: Sequence[Mapping[str, Any]], *, plans: Mapping[str, Mapping[str, Any]], rank_model: object, top_ks: Sequence[int]) -> dict[str, Any]:
    records, by_mem = _validate_source_map(manifest, source_map); _verify_source_map_database(manifest, database, records)
    questions = list(manifest["questions"]); digest = logical_database_digest(database)
    baseline_model, adjacent_model = _FrozenSearchModel(plans, rank_model), _FrozenSearchModel(plans, rank_model)
    baseline = MemoryStore(str(database), model=baseline_model, structured_query_plan=True)
    adjacent = MemoryStore(str(database), model=adjacent_model, structured_query_plan=True, adjacent_turn_expansion=True, adjacent_seed_limit=SEED_LIMIT, adjacent_candidate_limit=CANDIDATE_LIMIT)
    base_scores=[]; adjacent_scores=[]; traces=[]
    for question in questions:
        query=_required_text(question.get("question"), "question.question"); options=list(question.get("options", [])); user_id=_required_text(question.get("user_id"), "question.user_id"); expected=plans[str(question["query_key"])]
        first=baseline.search(user_id=user_id, query=query, options=options, top_k=max(top_ks)); first_trace=baseline.last_retrieval_trace
        second=adjacent.search(user_id=user_id, query=query, options=options, top_k=max(top_ks)); second_trace=adjacent.last_retrieval_trace
        if first_trace.get("plan") != expected or second_trace.get("plan") != expected: raise RuntimeError("arm did not use frozen plan")
        for field in ("p1_channels", "p1_union_ids", "p1_pre_rerank_ids", "p1_counterfactual_top30_ids"):
            if first_trace.get(field) != second_trace.get(field): raise RuntimeError("P1 first-stage path differs: {}".format(field))
        audit=_adjacent_audit(database, user_id=user_id, trace=second_trace)
        if audit["violations"]: raise RuntimeError("P1.1 trace audit failed: {}".format(audit["violations"][:3]))
        base_score=score_question_rankings(question, [str(item.id) for item in first], source_by_mem=by_mem, top_ks=top_ks); adjacent_score=score_question_rankings(question, [str(item.id) for item in second], source_by_mem=by_mem, top_ks=top_ks)
        base_scores.append((question,base_score)); adjacent_scores.append((question,adjacent_score)); traces.append({"case_id":question["case_id"],"frozen_plan_sha256":_sha256_json(expected),"baseline":base_score,"adjacent_on":{**adjacent_score,"trace":copy.deepcopy(second_trace),"audit":audit}})
    if logical_database_digest(database) != digest: raise RuntimeError("paired Search mutated prepared DB")
    if any(value != len(questions) for value in (baseline_model.logical_plan_calls, adjacent_model.logical_plan_calls, baseline_model.rank_calls, adjacent_model.rank_calls)): raise RuntimeError("paired local model call count mismatch")
    base, on = _aggregate_arm(base_scores, top_ks=top_ks), _aggregate_arm(adjacent_scores, top_ks=top_ks)
    return {"questions":len(questions),"database_unchanged":True,"configuration":{"seed_limit":4,"candidate_limit":4,"rerank_limit":MemoryStore.MODEL_RERANK_LIMIT},"baseline":base,"adjacent_on":on,"delta_hit_at_k":{str(k):on["hit_at_k"][str(k)]-base["hit_at_k"][str(k)] for k in top_ks},"delta_mrr":on["mrr"]-base["mrr"],"calls":{"baseline_plan":baseline_model.logical_plan_calls,"adjacent_plan":adjacent_model.logical_plan_calls,"baseline_rank":baseline_model.rank_calls,"adjacent_rank":adjacent_model.rank_calls},"traces":traces}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",required=True); parser.add_argument("--prepared-db",required=True); parser.add_argument("--output",required=True); parser.add_argument("--max-questions",type=int,default=20); parser.add_argument("--search-base-url",required=True); parser.add_argument("--search-model",default="local"); parser.add_argument("--top-k",type=_top_ks,default=(1,3,10)); args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise SystemExit("refusing to overwrite output")
    if not args.search_base_url.startswith("http://127.0.0.1:"): raise SystemExit("local loopback URL required")
    manifest=load_dataset_manifest(args.dataset,max_questions=args.max_questions); prepared=load_prepared_database(manifest,args.prepared_db); model=MemoryModel("local-only",model_name=args.search_model,base_url=args.search_base_url,disable_thinking=True); plans=freeze_structured_plans(list(manifest["questions"]),model); paired=run(manifest,args.prepared_db,prepared["source_map"],plans=plans,rank_model=model,top_ks=args.top_k)
    report={"stage":"p1_1_fixed20_paired","frozen_plan_external_calls":len(plans),"model_calls":model.call_count,"truncated_calls":model.truncated_calls,"paired":paired}; _atomic_write_json(output,report); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
