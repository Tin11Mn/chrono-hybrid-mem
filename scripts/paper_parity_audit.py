"""Paper-freeze audit: rerank-pool stats, evidence parity, source recovery.

Frozen-artifact audit for the paper/locomo-e2e branch. Read-only over the
frozen per-question retrieval artifacts and the official LoCoMo dataset:

1. Rerank-pool audit (task: "20 or 30?") — per method, from every question
   record: pre-rerank pool size, actual LLM rerank candidate count, final
   top-k, with min/max/mean/median. No summary files are trusted.
2. Evidence parity — recompute Hit@1/3/10, MRR and pooled Evidence Recall@10
   from ``result_ids`` + ``gold_mem_ids`` via score_locomo_answers
   (evidence_metrics/aggregate_evidence) and compare with the published
   numbers. Also cross-checks the artifact's stored ``first_gold_rank``.
3. Source recovery — rebuild mem_N -> original message (1-based per
   conversation, sessions_and_evidence order) directly from the dataset,
   recompute every question's gold mem ids through the same casefolded
   content -> first-mem-id mapping the runner used, and require an exact
   match with the artifact's gold_mem_ids. Prints a deterministic 20-question
   recovery sample with conversation/session/speaker/timestamp/content.

Usage:
    python scripts/paper_parity_audit.py [--json OUT.json]

Exit code 0 = all parity checks pass; 1 = any mismatch.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "scripts"))
import score_locomo_answers as sc  # noqa: E402

DATASET = WORKSPACE / "chrono-hybrid-mem" / ".locomo" / "locomo10.json"
ELR_PATH = REPO / "scripts" / "evaluate_locomo_retrieval.py"

ARTIFACT_SETS = {
    "p1": sorted(glob.glob(str(
        WORKSPACE / "chrono-hybrid-mem-p5-diagnostics" / ".locomo"
        / "p1-local-proxy-structured-diagnostics-chunk-*.json"))),
    "p4a_bm25": sorted(glob.glob(str(
        WORKSPACE / "chrono-hybrid-mem-p3" / ".locomo"
        / "newmethod-chunk-*.json"))),
    "sfv2_4b": sorted(glob.glob(str(
        WORKSPACE / "chrono-hybrid-mem-p3" / ".locomo" / "sfv2-full-*.json"))),
}

# Published 1976-question numbers (docs/EVALUATION_NEW_METHOD_1976.md for
# P1/P4-A+bm25; README section B + docs/SESSION_FACT_SEMANTIC_LAYER_EXPERIMENT.md
# for SF v2). Rounded to 4 decimals as published.
EXPECTED = {
    "p1": {"hit_at_1": 0.5779, "hit_at_3": 0.7176, "hit_at_10": 0.7601,
           "mrr": 0.6497},
    "p4a_bm25": {"hit_at_1": 0.5850, "hit_at_3": 0.7323, "hit_at_10": 0.7809,
                 "mrr": 0.6618},
    "sfv2_4b": {"hit_at_1": 0.6108, "hit_at_3": 0.7677, "hit_at_10": 0.8219,
                "mrr": 0.6929},
}
EXCLUDED_OFFSET = 758  # conv-43 hangs the local rank server; excluded for all methods


def _load_elr():
    spec = importlib.util.spec_from_file_location("elr", ELR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_records(paths: list[str]) -> tuple[dict[int, dict], dict[str, int]]:
    """Merge per-question records by global eligible-question offset.

    Dedup policy is byte-identical to evaluate_locomo_e2e.load_frozen_diags:
    sorted-glob order, keep-first via setdefault. Returns records plus a
    per-file duplicate count so the audit can report the known sfv2-full
    overlap (offset 800 appears in both -0600b and -0800).
    """
    records: dict[int, dict] = {}
    duplicates: dict[str, int] = {}
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        for qd in data.get("question_diagnostics", []):
            offset = int(qd["question_offset"])
            if offset in records:
                duplicates[Path(path).name] = duplicates.get(Path(path).name, 0) + 1
                continue
            records[offset] = qd
    return records, duplicates


def dataset_question_index(elr) -> tuple[list[dict], dict[int, dict], dict]:
    """Rebuild the eligible-question index exactly like the runner.

    Returns (index list, offset -> index entry, per-conversation recovery map).
    Eligibility mirrors evaluate_locomo_retrieval.evaluate(): question text
    present AND at least one resolvable evidence item.
    """
    samples = json.load(open(DATASET, encoding="utf-8"))
    index: list[dict] = []
    per_sample: dict[str, dict] = {}
    for sample in samples:
        sessions, evidence_text = elr.sessions_and_evidence(sample)
        if not sessions:
            continue
        # NOTE: MemoryStore.add() validates Message.content through Pydantic
        # constr(strip_whitespace=True), so the stored raw message (and hence
        # _load_mem_by_content's keys and every Search result) is the
        # whitespace-STRIPPED content. evidence_text keeps the dataset text
        # verbatim. Gold matching at run time looks up the UNstripped evidence
        # content against the STRIPPED DB keys, so evidence turns whose text
        # has leading/trailing whitespace are silently dropped from gold.
        # The audit mirrors both sides exactly (runtime-faithful semantics).
        mem2content: dict[str, str] = {}
        mem2meta: dict[str, dict] = {}
        n = 0
        mem_by_content: dict[str, str] = {}
        for _session_key, msgs in sessions:
            for msg in msgs:
                n += 1
                mem_id = "mem_{}".format(n)
                content = str(msg["content"]).strip()
                mem2content[mem_id] = content
                mem2meta[mem_id] = {
                    "session_key": _session_key,
                    "role": msg.get("role", ""),
                    "timestamp": msg.get("timestamp"),
                }
                mem_by_content.setdefault(content.casefold(), mem_id)
        sample_id = str(sample.get("sample_id"))
        per_sample[sample_id] = {
            "sample_id": sample_id,
            "mem2content": mem2content,
            "mem2meta": mem2meta,
            "n_messages": n,
            "evidence_text": evidence_text,
            "mem_by_content": mem_by_content,
        }
        for qa_index, qa in enumerate(sample.get("qa", [])):
            if not isinstance(qa, dict):
                continue
            expected = [evidence_text[i] for i in qa.get("evidence", [])
                        if i in evidence_text]
            if not expected or not qa.get("question"):
                continue
            index.append({
                "offset": len(index),
                "sample_id": sample_id,
                "qa_index": qa_index,
                "question": qa["question"],
                "category": qa.get("category"),
                "answer": qa.get("answer"),
                "evidence_dia_ids": [i for i in qa.get("evidence", [])
                                     if i in evidence_text],
            })
    by_offset = {entry["offset"]: entry for entry in index}
    return index, by_offset, per_sample


def recomputed_expected_gold(entry: dict, smap: dict) -> tuple[list[str], int]:
    """Gold mem ids recomputed from the dataset alone (runner semantics).

    Mirrors the runner exactly: evidence content is looked up UNstripped
    (casefolded) against the STRIPPED DB-content keys, so whitespace-padded
    evidence turns drop out. Returns (gold ids, dropped item count).
    """
    mem_by_content = smap["mem_by_content"]
    gold = []
    dropped = 0
    for dia in entry["evidence_dia_ids"]:
        content = smap["evidence_text"][dia]
        mem_id = mem_by_content.get(str(content).casefold())
        if mem_id is None:
            dropped += 1
            continue
        gold.append(mem_id)
    return gold, dropped


def pool_stats(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
        "histogram": {
            str(v): values.count(v) for v in sorted(set(values))
        } if len(set(values)) <= 8 else None,
    }


def audit_method(name: str, paths: list[str], by_offset: dict,
                 per_sample: dict, elr) -> dict:
    records, duplicates = load_records(paths)
    offsets = sorted(records)
    excluded_present = EXCLUDED_OFFSET in records
    if excluded_present:
        usable = [o for o in offsets if o != EXCLUDED_OFFSET]
    else:
        usable = offsets
    per_q: list[dict] = []
    rank_mismatches = []
    gold_mismatches = []
    gold_dropped_total = 0
    gold_dropped_questions = 0
    pool_sizes = []
    llm_counts = []
    final_ks = []
    fusion_top30 = []
    categories: dict[int, int] = {}
    for offset in usable:
        qd = records[offset]
        entry = by_offset.get(offset)
        if entry is None:
            gold_mismatches.append((offset, "offset-not-in-dataset-index", None))
            continue
        result_ids = list(qd["result_ids"])
        artifact_gold = list(qd["gold_mem_ids"])
        smap = per_sample[entry["sample_id"]]
        expected_gold, dropped = recomputed_expected_gold(entry, smap)
        gold_dropped_total += dropped
        gold_dropped_questions += 1 if dropped else 0
        if sorted(artifact_gold) != sorted(expected_gold):
            gold_mismatches.append((offset, artifact_gold, expected_gold))
        em = sc.evidence_metrics(result_ids, artifact_gold)
        stored_rank = qd.get("first_gold_rank")
        if stored_rank is not None and stored_rank != em["first_gold_rank"]:
            rank_mismatches.append((offset, stored_rank, em["first_gold_rank"]))
        per_q.append(em)
        pool_sizes.append(len(qd.get("rerank_pool_ids", [])))
        if "llm_rank_candidate_count" in qd:
            llm_counts.append(int(qd["llm_rank_candidate_count"]))
        final_ks.append(len(result_ids))
        if "p1_counterfactual_top30_ids" in qd:
            fusion_top30.append(len(qd["p1_counterfactual_top30_ids"]))
        cat = int(entry["category"])
        categories[cat] = categories.get(cat, 0) + 1
        # source-recovery sanity: every id must exist in the conversation
        n_messages = smap["n_messages"]
        for rid in set(result_ids) | set(artifact_gold):
            num = int(rid.split("_", 1)[1])
            if not 1 <= num <= n_messages:
                gold_mismatches.append((offset, rid, "id out of range 1..{}".format(n_messages)))
    agg = sc.aggregate_evidence(per_q)
    expected = EXPECTED[name]
    parity = {
        "n": agg["n"],
        "offset_758_present_in_artifacts": excluded_present,
        "hit_at_1": round(agg["hit_at_1"], 4),
        "hit_at_3": round(agg["hit_at_3"], 4),
        "hit_at_10": round(agg["hit_at_10"], 4),
        "mrr": round(agg["mrr"], 4),
        "evidence_recall_at_10": round(agg["evidence_recall_at_10"], 4),
        "n_gold_total": agg["n_gold_total"],
        "category_distribution": {str(k): categories[k] for k in sorted(categories)},
        "expected": expected,
        "hit1_match": round(agg["hit_at_1"], 4) == expected["hit_at_1"],
        "hit3_match": round(agg["hit_at_3"], 4) == expected["hit_at_3"],
        "hit10_match": round(agg["hit_at_10"], 4) == expected["hit_at_10"],
        "mrr_match": round(agg["mrr"], 4) == expected["mrr"],
        "stored_first_gold_rank_mismatches": rank_mismatches[:10],
        "stored_first_gold_rank_mismatch_count": len(rank_mismatches),
        "gold_source_recovery_mismatch_count": len(gold_mismatches),
        "gold_source_recovery_mismatch_samples": gold_mismatches[:5],
        "gold_items_dropped_whitespace": gold_dropped_total,
        "gold_questions_dropped_whitespace": gold_dropped_questions,
        "rerank_pool": pool_stats(pool_sizes),
        "llm_rank_candidate_count": pool_stats(llm_counts) if llm_counts else None,
        "final_top_k": pool_stats(final_ks),
        "p1_counterfactual_top30_len": pool_stats(fusion_top30) if fusion_top30 else None,
        "artifact_files": [Path(p).name for p in paths],
        "duplicate_offsets_kept_first": duplicates,
    }
    return parity


def recovery_sample(by_offset: dict, per_sample: dict, records: dict,
                    size: int = 20, seed: int = 20260911) -> list[dict]:
    """Deterministic random sample: recover mem ids to original messages."""
    rng = random.Random(seed)
    usable = [o for o in sorted(records) if o != EXCLUDED_OFFSET and o in by_offset]
    chosen = sorted(rng.sample(usable, size))
    out = []
    for offset in chosen:
        qd = records[offset]
        entry = by_offset[offset]
        smap = per_sample[entry["sample_id"]]
        gold = qd["gold_mem_ids"]
        expected_gold, _dropped = recomputed_expected_gold(entry, smap)
        ok = sorted(expected_gold) == sorted(gold)
        first_hit = next((rid for rid in qd["result_ids"] if rid in gold), None)
        rid = first_hit or (qd["result_ids"][0] if qd["result_ids"] else None)
        rec = None
        if rid is not None:
            meta = smap["mem2meta"].get(rid, {})
            rec = {
                "mem_id": rid,
                "session_key": meta.get("session_key"),
                "role": meta.get("role"),
                "timestamp": meta.get("timestamp"),
                "content": smap["mem2content"].get(rid),
                "recovered_from": "first retrieved gold" if first_hit else "first retrieved",
            }
        out.append({
            "offset": offset,
            "conversation_id": entry["sample_id"],
            "category": entry["category"],
            "question": entry["question"],
            "gold_mem_ids": gold,
            "gold_recovery_exact_match": ok,
            "evidence_dia_ids": entry["evidence_dia_ids"],
            "recovered_example": rec,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="write full audit JSON here")
    ap.add_argument("--recovery-sample", type=int, default=20)
    args = ap.parse_args()

    elr = _load_elr()
    _index, by_offset, per_sample = dataset_question_index(elr)
    print("dataset questions indexed: {}".format(len(by_offset)))
    print("offset 758 in index (must be False for eligible set): {}".format(
        758 in by_offset))

    report = {
        "dataset": str(DATASET),
        "dataset_sha256": None,
        "eligible_questions_in_index": len(by_offset),
        "offset_758_in_index": 758 in by_offset,
        "methods": {},
    }
    import hashlib
    report["dataset_sha256"] = hashlib.sha256(DATASET.read_bytes()).hexdigest()

    all_ok = True
    for name, paths in ARTIFACT_SETS.items():
        if not paths:
            print("ERROR: no artifacts for {}".format(name))
            all_ok = False
            continue
        print("\n=== {} ({} files) ===".format(name, len(paths)))
        parity = audit_method(name, paths, by_offset, per_sample, elr)
        report["methods"][name] = parity
        print("n={n} (758 present: {o}) duplicates_kept_first: {d}".format(
            n=parity["n"], o=parity["offset_758_present_in_artifacts"],
            d=parity["duplicate_offsets_kept_first"]))
        print("Hit@1 {h1} (exp {eh1})  Hit@3 {h3} (exp {eh3})  Hit@10 {h10} (exp {eh10})  MRR {mrr} (exp {emrr})".format(
            h1=parity["hit_at_1"], eh1=parity["expected"]["hit_at_1"],
            h3=parity["hit_at_3"], eh3=parity["expected"]["hit_at_3"],
            h10=parity["hit_at_10"], eh10=parity["expected"]["hit_at_10"],
            mrr=parity["mrr"], emrr=parity["expected"]["mrr"]))
        print("Evidence Recall@10 = {} (pooled, n_gold_total={})".format(
            parity["evidence_recall_at_10"], parity["n_gold_total"]))
        print("categories: {}".format(parity["category_distribution"]))
        print("rerank_pool: {}".format(parity["rerank_pool"]))
        print("llm_rank_candidate_count: {}".format(parity["llm_rank_candidate_count"]))
        print("final_top_k: {}".format(parity["final_top_k"]))
        if parity["p1_counterfactual_top30_len"]:
            print("p1_counterfactual_top30 len: {}".format(
                parity["p1_counterfactual_top30_len"]))
        ok = (parity["hit1_match"] and parity["hit3_match"]
              and parity["hit10_match"] and parity["mrr_match"]
              and parity["stored_first_gold_rank_mismatch_count"] == 0
              and parity["gold_source_recovery_mismatch_count"] == 0)
        parity["parity_pass"] = ok
        all_ok = all_ok and ok
        print("PARITY: {}".format("PASS" if ok else "FAIL"))

    print("\n=== source recovery sample ({} questions) ===".format(
        args.recovery_sample))
    sfv2_records, _sfv2_dups = load_records(ARTIFACT_SETS["sfv2_4b"])
    sample = recovery_sample(by_offset, per_sample, sfv2_records,
                             size=args.recovery_sample)
    exact = sum(1 for s in sample if s["gold_recovery_exact_match"])
    print("exact gold recovery: {}/{}".format(exact, len(sample)))
    report["recovery_sample_size"] = len(sample)
    report["recovery_exact_matches"] = exact
    report["recovery_sample"] = sample
    for s in sample[:5]:
        rec = s["recovered_example"]
        print("  offset {} conv {} cat {} gold={} -> {}".format(
            s["offset"], s["conversation_id"], s["category"],
            s["gold_mem_ids"], s["recovered_example"]["mem_id"] if rec else None))
        if rec:
            print("    [{}|{}] {}".format(rec["role"], rec["timestamp"],
                                          str(rec["content"])[:90]))
    report["all_parity_pass"] = all_ok and exact == len(sample)
    print("\nOVERALL: {}".format("PASS" if report["all_parity_pass"] else "FAIL"))
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("audit json written to {}".format(out_path))
    return 0 if report["all_parity_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
