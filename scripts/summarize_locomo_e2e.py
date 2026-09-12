"""Summarize a single LoCoMo E2E run into a JSON summary.

Formal Identity Policy v1.0:
- accepted = model_identity ∈ {snapshot_verified, alias_only_snapshot_unverified}
- explicit_wrong_model is the ONLY identity-related rejection reason
- gpt-4o-mini (alias) and gpt-4o-mini-2024-07-18 (snapshot) both indicate
  valid ChatAnywhere GPT-4o-mini access
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

METRICS = ["f1_mem0", "f1_official", "f1_memoryart", "f1_memoryos",
           "bleu1_m1", "bleu1_m4"]

ACCEPTED_IDENTITIES = frozenset({"snapshot_verified", "alias_only_snapshot_unverified"})
EXPECTED_SNAPSHOT = "gpt-4o-mini-2024-07-18"
REQUESTED_ALIAS = "gpt-4o-mini"


def _count(values):
    c = Counter(values)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _row_cost(r):
    ans = r.get("estimated_answer_cost", 0.0) or 0.0
    j = r.get("estimated_judge_cost", 0.0) or 0.0
    return ans + j


def classify_model_identity(returned_model) -> str:
    norm = str(returned_model).strip() if returned_model is not None else ""
    if norm == EXPECTED_SNAPSHOT:
        return "snapshot_verified"
    if norm == REQUESTED_ALIAS:
        return "alias_only_snapshot_unverified"
    if norm:
        return "explicit_wrong_model"
    return "unavailable"


def _attempt_accepted(attempt: dict) -> bool:
    stored = attempt.get("accepted")
    if stored is not None:
        return bool(stored)
    mid = attempt.get("model_identity") or classify_model_identity(
        attempt.get("returned_model"))
    return mid in ACCEPTED_IDENTITIES


def _attempt_outcome(attempt: dict) -> str:
    return attempt.get("attempt_outcome") or "unknown"


def _is_rejected(attempt: dict) -> bool:
    return not _attempt_accepted(attempt)


def summarize_rows(rows):
    """Aggregate over the full subset. Denominators stated."""
    n = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    out = {
        "n_total": n, "n_ok": len(ok),
        "n_failed": n - len(ok),
        "status_counts": dict(_count(r.get("status") for r in rows)),
        "judge": {
            "n": sum(1 for r in ok if r.get("judge_result") in ("CORRECT", "WRONG")),
            "correct": sum(1 for r in ok if r.get("judge_result") == "CORRECT"),
        },
        "cost_usd": round(sum(_row_cost(r) for r in rows), 6),
        "tokens": {
            "answer_input": sum(r.get("answer_input_tokens", 0) or 0 for r in rows),
            "answer_output": sum(r.get("answer_output_tokens", 0) or 0 for r in rows),
            "judge_input": sum(r.get("judge_input_tokens", 0) or 0 for r in rows),
            "judge_output": sum(r.get("judge_output_tokens", 0) or 0 for r in rows),
        },
    }
    j = out["judge"]
    j["accuracy"] = (j["correct"] / j["n"]) if j["n"] else None
    for m in METRICS:
        out[m] = statistics.mean([r.get(m) for r in ok]) if ok else None
    hits = sum(r.get("evidence_hits_at_10", 0) or 0 for r in ok)
    gold = sum(r.get("n_gold", 0) or 0 for r in ok)
    out["evidence_recall_at_10"] = (hits / gold) if gold else None
    out["n_gold_total"] = gold

    def _judge_attempts_of(r):
        if r.get("judge_attempts"):
            return r["judge_attempts"]
        if r.get("judge_returned_model"):
            return [{"attempt": 1, "requested_model": r.get("judge_model_requested"),
                     "returned_model": r.get("judge_returned_model"),
                     "model_identity": classify_model_identity(
                         r.get("judge_returned_model")),
                     "accepted": (classify_model_identity(
                         r.get("judge_returned_model")) in ACCEPTED_IDENTITIES),
                     "system_fingerprint": r.get("judge_system_fingerprint"),
                     "attempt_outcome": "accepted" if
                         (classify_model_identity(r.get("judge_returned_model"))
                          in ACCEPTED_IDENTITIES) else "explicit_wrong_model",
                     }]
        return []

    def _answer_attempts_of(r):
        if r.get("answer_attempts"):
            return r["answer_attempts"]
        if r.get("answer_returned_model"):
            mid = classify_model_identity(r.get("answer_returned_model"))
            return [{"attempt": 1, "requested_model": r.get("answer_model_requested"),
                     "returned_model": r.get("answer_returned_model"),
                     "model_identity": mid,
                     "accepted": mid in ACCEPTED_IDENTITIES,
                     "attempt_outcome": "accepted" if mid in ACCEPTED_IDENTITIES
                        else ("explicit_wrong_model" if mid == "explicit_wrong_model"
                              else "transport_error"),
                     }]
        return []

    # model verification: new two-field taxonomy
    def _model_identity_of(attempt: dict) -> str:
        return attempt.get("model_identity") or classify_model_identity(
            attempt.get("returned_model"))

    out["model_verification"] = {
        "gateway": sorted({r.get("gateway") for r in ok if r.get("gateway")}),
        "api_base_url": sorted({r.get("api_base_url") for r in ok if r.get("api_base_url")}),
        "answer_returned_models": dict(_count(r.get("answer_returned_model") for r in ok)),
        "answer_system_fingerprints": dict(_count(r.get("answer_system_fingerprint") for r in ok)),
        "judge_returned_models": dict(_count(r.get("judge_returned_model") for r in ok
                                             if r.get("judge_returned_model"))),
        "judge_system_fingerprints": dict(_count(r.get("judge_system_fingerprint") for r in ok
                                                 if r.get("judge_system_fingerprint"))),
        # formal identity counts (answer side)
        "answer_model_identities": dict(_count(
            _model_identity_of(a)
            for r in rows for a in _answer_attempts_of(r)
        )),
        "answer_snapshot_verified": sum(
            1 for r in rows for a in _answer_attempts_of(r)
            if _model_identity_of(a) == "snapshot_verified"),
        "answer_alias_only": sum(
            1 for r in rows for a in _answer_attempts_of(r)
            if _model_identity_of(a) == "alias_only_snapshot_unverified"),
        "answer_explicit_wrong_model": sum(
            1 for r in rows for a in _answer_attempts_of(r)
            if _model_identity_of(a) == "explicit_wrong_model"),
        "answer_transport_error": sum(
            1 for r in rows for a in _answer_attempts_of(r)
            if _model_identity_of(a) == "unavailable"
            and not a.get("returned_model")),
        "answer_rate_limit": sum(
            1 for r in rows for a in _answer_attempts_of(r)
            if _attempt_outcome(a) == "rate_limit"),
        "answer_malformed": sum(
            1 for r in rows for a in _answer_attempts_of(r)
            if _attempt_outcome(a) == "malformed_response"),
        "answer_evaluator_invariant_violation": sum(
            1 for r in rows if r.get("evaluator_invariant_violation")),
        "accepted_model_scope_ok": all(
            _model_identity_of(a) in ACCEPTED_IDENTITIES
            for r in rows for a in _answer_attempts_of(r)
            if _attempt_accepted(a)),
        # snapshot verification rate (formal: alias-only is valid but NOT snapshot)
        "snapshot_verification_rate": (
            sum(1 for r in rows for a in _answer_attempts_of(r)
                if _model_identity_of(a) == "snapshot_verified")
            / max(1, sum(1 for r in rows for a in _answer_attempts_of(r)
                         if _attempt_accepted(a)))
        ),
        # formal identity counts (judge side)
        "judge_model_identities": dict(_count(
            _model_identity_of(a)
            for r in rows for a in _judge_attempts_of(r)
        )),
        "judge_explicit_wrong_model": sum(
            1 for r in rows for a in _judge_attempts_of(r)
            if _model_identity_of(a) == "explicit_wrong_model"),
        "judge_rate_limit": sum(
            1 for r in rows for a in _judge_attempts_of(r)
            if _attempt_outcome(a) == "rate_limit"),
        "judge_malformed_response": sum(
            1 for r in rows for a in _judge_attempts_of(r)
            if _attempt_outcome(a) == "malformed_response"),
    }

    # model routing reliability
    judge_attempts = [a for r in rows for a in _judge_attempts_of(r)]
    judge_rejected = [a for a in judge_attempts if _is_rejected(a)]
    judge_explicit_wrong = sum(1 for a in judge_rejected
                                if _model_identity_of(a) == "explicit_wrong_model")
    judge_rate_limit = sum(1 for a in judge_attempts
                           if _attempt_outcome(a) == "rate_limit")
    judge_malformed = sum(1 for a in judge_attempts
                          if _attempt_outcome(a) == "malformed_response")
    out["model_routing_reliability"] = {
        "total_judge_attempts": len(judge_attempts),
        "judge_accepted": len([a for a in judge_attempts if _attempt_accepted(a)]),
        "judge_explicit_wrong_model": judge_explicit_wrong,
        "judge_rate_limit_attempts": judge_rate_limit,
        "judge_malformed_response_attempts": judge_malformed,
        "formal_pass": judge_explicit_wrong == 0,
    }

    # answer-side model routing
    answer_attempts = [a for r in rows for a in _answer_attempts_of(r)]
    answer_accepted = [a for a in answer_attempts if _attempt_accepted(a)]
    answer_explicit_wrong = sum(1 for a in answer_attempts
                                 if _model_identity_of(a) == "explicit_wrong_model")
    answer_rate_limit = sum(1 for a in answer_attempts
                            if _attempt_outcome(a) == "rate_limit")
    answer_malformed = sum(1 for a in answer_attempts
                           if _attempt_outcome(a) == "malformed_response")
    out["answer_model_routing"] = {
        "total_answer_attempts": len(answer_attempts),
        "answer_accepted_attempts": len(answer_accepted),
        "snapshot_verified": sum(1 for a in answer_accepted
                                  if _model_identity_of(a) == "snapshot_verified"),
        "alias_only_snapshot_unverified": sum(1 for a in answer_accepted
                                               if _model_identity_of(a) == "alias_only_snapshot_unverified"),
        "explicit_wrong_model": answer_explicit_wrong,
        "rate_limit_attempts": answer_rate_limit,
        "malformed_response_attempts": answer_malformed,
        "formal_pass": answer_explicit_wrong == 0,
    }

    return out


def main(argv=None):
    """CLI: --run-dir DIR, optional --full-n (sensitivity denominator),
    --baseline-dir for a paired bootstrap (delegated to the stats plan).
    Writes metric_summary.json + category_summary.json into the run dir."""
    import argparse
    ap = argparse.ArgumentParser(description="LoCoMo E2E run summarizer.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--full-n", type=int, default=1540)
    ap.add_argument("--baseline-dir", default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    pq_path = run_dir / "per_question.jsonl"
    if not pq_path.exists():
        print("ERROR: no per_question.jsonl in {}".format(run_dir))
        return 2
    rows = [json.loads(l) for l in pq_path.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    summary = summarize_rows(rows)
    summary["sensitivity_full_set"] = {
        "full_n": args.full_n,
        "evaluated_ok": summary["n_ok"],
        "missing_or_failed": max(0, args.full_n - summary["n_ok"]),
        "note": "missing/failed counted as wrong; conservative lower bound",
    }
    # per-category
    from collections import defaultdict
    per_cat = defaultdict(list)
    for r in rows:
        per_cat[r.get("category_id")].append(r)
    cat_summary = {}
    names = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
    for cat in sorted(per_cat, key=lambda c: (c is None, c)):
        cat_summary[names.get(cat, str(cat))] = summarize_rows(per_cat[cat])
    cat_summary["_by_id"] = {str(k): len(v) for k, v in per_cat.items()}

    (run_dir / "metric_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "category_summary.json").write_text(
        json.dumps(cat_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summarized {}".format(run_dir))
    print("  n_total={} n_ok={} n_failed={}".format(
        summary["n_total"], summary["n_ok"], summary["n_failed"]))
    print("  judge_acc={} f1_mem0={} f1_official={}".format(
        summary["judge"]["accuracy"], summary.get("f1_mem0"),
        summary.get("f1_official")))
    if args.baseline_dir:
        print("  (baseline-dir bootstrapping is a separate step)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
