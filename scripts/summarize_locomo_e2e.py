"""Summarize LoCoMo E2E runs: per-category metrics, pooled evidence recall,
paired bootstrap, and the 1539-question sensitivity.

Reads one or two run dirs (per_question.jsonl) and emits metric_summary.json,
category_summary.json, and paired_bootstrap.json. All denominators are stated
explicitly. Bootstrap uses paired resampling (10000 draws, seed 20260826) and
reports Pr(delta>0) as a "bootstrap probability", never a p-value.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import score_locomo_answers as sc  # noqa: E402

BOOT_SEED = 20260826
BOOT_DRAWS = 10000
METRICS = ["f1_official", "f1_mem0", "f1_memoryart", "f1_memoryos",
           "bleu1_m1", "bleu1_m4", "hit1", "hit3", "hit10", "mrr"]
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


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
        out[m] = mean([r.get(m) for r in ok])
    # pooled evidence recall
    hits = sum(r.get("evidence_hits_at_10", 0) or 0 for r in ok)
    gold = sum(r.get("n_gold", 0) or 0 for r in ok)
    out["evidence_recall_at_10"] = (hits / gold) if gold else None
    out["n_gold_total"] = gold
    def _judge_attempts_of(r):
        """Normalized attempt records; legacy rows count as one accepted."""
        if r.get("judge_attempts"):
            return r["judge_attempts"]
        if r.get("judge_returned_model"):
            return [{"attempt": 1, "requested_model": r.get("judge_model_requested"),
                     "returned_model": r.get("judge_returned_model"),
                     "system_fingerprint": r.get("judge_system_fingerprint"),
                     "valid_model_identity": True, "status": "accepted"}]
        return []

    # gateway model verification: returned-model and system-fingerprint
    # distributions (fingerprints may vary freely; only recorded, never failed).
    out["model_verification"] = {
        "gateway": sorted({r.get("gateway") for r in ok if r.get("gateway")}),
        "api_base_url": sorted({r.get("api_base_url") for r in ok if r.get("api_base_url")}),
        "answer_returned_models": dict(_count(r.get("answer_returned_model") for r in ok)),
        "answer_system_fingerprints": dict(_count(r.get("answer_system_fingerprint") for r in ok)),
        "judge_returned_models": dict(_count(r.get("judge_returned_model") for r in ok
                                             if r.get("judge_returned_model"))),
        "judge_system_fingerprints": dict(_count(r.get("judge_system_fingerprint") for r in ok
                                                 if r.get("judge_system_fingerprint"))),
        "answer_drift_errors": sum(1 for r in rows if r.get("model_drift")),
        "judge_drift_errors": len([
            a for r in rows for a in _judge_attempts_of(r)
            if a.get("status") == "model_drift"]),
    }

    # Model routing reliability across ALL judge attempts (fixed-100 protocol:
    # formal metrics require the accepted returned-model distribution to be
    # 100% gpt-4o-mini-2024-07-18).
    attempts = [a for r in rows for a in _judge_attempts_of(r)]
    acc = [a for a in attempts if a.get("status") == "accepted"]
    drift = [a for a in attempts if a.get("status") == "model_drift"]
    transport = [a for a in attempts if a.get("status") == "transport_error"]
    per_q_attempts = [len(_judge_attempts_of(r)) for r in rows
                      if r.get("judge_returned_model") or r.get("judge_attempts")]
    out["model_routing_reliability"] = {
        "total_judge_attempts": len(attempts),
        "valid_judge_responses": len(acc),
        "model_drift_attempts": len(drift),
        "transport_error_attempts": len(transport),
        "drift_rate_per_attempt": (round(len(drift) / len(attempts), 4)
                                   if attempts else None),
        "returned_model_distribution_all_attempts": dict(
            _count(a.get("returned_model") for a in attempts)),
        "accepted_returned_model_distribution": dict(
            _count(a.get("returned_model") for a in acc)),
        "system_fingerprint_distribution_all_attempts": dict(
            _count(a.get("system_fingerprint") for a in attempts)),
        "questions_requiring_retry": sum(1 for n in per_q_attempts if n > 1),
        "max_attempts_for_any_question": max(per_q_attempts, default=0),
        "accepted_purity_ok": (
            all(a.get("returned_model") == "gpt-4o-mini-2024-07-18"
                for a in acc) if acc else False),
    }
    # Retrieval x answer-generation diagnostic matrix (judged questions with
    # EVALUABLE evidence metrics only — evidence-N/A rows are excluded from
    # these denominators while remaining in every E2E denominator).
    judged = [r for r in ok if r.get("judge_result") in ("CORRECT", "WRONG")
              and r.get("hit10") is not None]
    def _judge_correct(r):
        return r.get("judge_result") == "CORRECT"
    def _hit(r, k):
        return bool(r.get("hit{}".format(k)))
    matrix = {}
    for name, hit in (("hit10", lambda r: _hit(r, 10)),
                      ("miss10", lambda r: not _hit(r, 10))):
        subset = [r for r in judged if hit(r)]
        matrix[name] = {
            "n": len(subset),
            "judge_correct": sum(1 for r in subset if _judge_correct(r)),
            "judge_wrong": sum(1 for r in subset if not _judge_correct(r)),
        }
    out["retrieval_answer_matrix"] = matrix
    out["judge_accuracy_by_retrieval"] = {}
    for k in (1, 3, 10):
        subset = [r for r in judged if _hit(r, k)]
        out["judge_accuracy_by_retrieval"]["hit{}".format(k)] = round(
            sum(1 for r in subset if _judge_correct(r)) / len(subset), 4) if subset else None
    miss10 = [r for r in judged if not _hit(r, 10)]
    out["judge_accuracy_by_retrieval"]["miss10"] = round(
        sum(1 for r in miss10 if _judge_correct(r)) / len(miss10), 4) if miss10 else None
    # Notable-phenomenon counters (diagnostic; no on-the-fly fixes allowed).
    out["phenomena"] = {
        "retrieval_hit_but_judge_wrong": matrix["hit10"]["judge_wrong"],
        "retrieval_miss_but_judge_correct": matrix["miss10"]["judge_correct"],
        "f1_mem0_zero_but_judge_correct": sum(
            1 for r in judged if _judge_correct(r) and not (r.get("f1_mem0") or 0.0)),
        "f1_official_zero_but_judge_correct": sum(
            1 for r in judged if _judge_correct(r) and not (r.get("f1_official") or 0.0)),
    }
    return out


def _row_cost(r):
    ai = r.get("answer_input_tokens", 0) or 0
    ao = r.get("answer_output_tokens", 0) or 0
    ji = r.get("judge_input_tokens", 0) or 0
    jo = r.get("judge_output_tokens", 0) or 0
    return (ai * 0.15 + ao * 0.60 + ji * 0.15 + jo * 0.60) / 1e6


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


def paired_bootstrap(rows_a, rows_b, seed=BOOT_SEED, draws=BOOT_DRAWS):
    """Paired bootstrap over question-aligned metric deltas (A - B).

    Aligns on question_id; missing questions in either run count as 0 for the
    metric (failure kept in the denominator), per protocol. Returns for each
    metric: mean delta, 95% CI, and Pr(delta>0) as a bootstrap probability.
    """
    by_b = {r["question_id"]: r for r in rows_b}
    rng = random.Random(seed)
    results = {}
    # build aligned metric vectors
    aligned = [r for r in rows_a if r.get("status") == "ok"]
    n = len(aligned)
    if n == 0:
        return {"n": 0}
    for m in METRICS:
        vec_a = [r.get(m, 0.0) or 0.0 for r in aligned]
        vec_b = [(by_b.get(r["question_id"], {}).get(m, 0.0) or 0.0) for r in aligned]
        deltas = [a - b for a, b in zip(vec_a, vec_b)]
        mean_delta = sum(deltas) / n
        boot = []
        for _ in range(draws):
            s = 0.0
            for _ in range(n):
                s += deltas[rng.randrange(n)]
            boot.append(s / n)
        boot.sort()
        lo = boot[int(0.025 * draws)]
        hi = boot[int(0.975 * draws)]
        p_gt0 = sum(1 for x in boot if x > 0) / draws
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        results[m] = {
            "mean_delta": round(mean_delta, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "bootstrap_probability_gt0": round(p_gt0, 4),
            "wins": wins, "losses": losses,
        }
    return {"n": n, "seed": seed, "draws": draws, "note":
            "bootstrap_probability_gt0 is NOT a p-value", "metrics": results}


def main():
    ap = argparse.ArgumentParser(description="Summarize LoCoMo E2E runs.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline-dir", default=None,
                    help="Optional second run dir for paired bootstrap")
    ap.add_argument("--full-n", type=int, default=1540,
                    help="Full non-adversarial question count for the sensitivity")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = read_jsonl(run_dir / "per_question.jsonl")
    ok = [r for r in rows if r.get("status") == "ok"]

    summary = summarize_rows(rows)
    # 1539-style sensitivity: judge/metrics over ok questions (already done) AND
    # a conservative sensitivity treating every missing-of-full-set as failed.
    missing = max(0, args.full_n - len(ok))
    summary["sensitivity_full_set"] = {
        "full_n": args.full_n,
        "evaluated_ok": len(ok),
        "missing_or_failed": missing + summary["n_failed"],
        "note": "missing/failed counted as wrong; conservative lower bound",
    }
    j = summary["judge"]
    if j["n"]:
        denom = args.full_n
        summary["sensitivity_full_set"]["judge_accuracy"] = round(j["correct"] / denom, 4)

    # per-category
    per_cat = defaultdict(list)
    for r in rows:
        per_cat[r.get("category_id")].append(r)
    cat_summary = {}
    for cat, crows in sorted(per_cat.items(), key=lambda kv: (kv[0] is None, kv[0])):
        cat_summary[CATEGORY_NAMES.get(cat, str(cat))] = summarize_rows(crows)
    cat_summary["_by_id"] = {str(k): len(v) for k, v in per_cat.items()}

    (run_dir / "metric_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "category_summary.json").write_text(
        json.dumps(cat_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("run:", run_dir.name)
    print("  n_total={} n_ok={} n_failed={}".format(
        summary["n_total"], summary["n_ok"], summary["n_failed"]))
    print("  f1_official={} f1_mem0={} bleu1_m1={}".format(
        round(summary["f1_official"], 4), round(summary["f1_mem0"], 4),
        round(summary["bleu1_m1"], 4)))
    print("  judge_acc={} (n={})".format(summary["judge"]["accuracy"], summary["judge"]["n"]))
    print("  hit1={} hit3={} hit10={} mrr={} evrec10={}".format(
        round(summary["hit1"], 4), round(summary["hit3"], 4),
        round(summary["hit10"], 4), round(summary["mrr"], 4),
        round(summary["evidence_recall_at_10"], 4) if summary["evidence_recall_at_10"] is not None else None))
    print("  cost=${:.4f}  tokens(ans_in={} ans_out={} judge_in={} judge_out={})".format(
        summary["cost_usd"], summary["tokens"]["answer_input"],
        summary["tokens"]["answer_output"], summary["tokens"]["judge_input"],
        summary["tokens"]["judge_output"]))
    print("  per-category n:", {CATEGORY_NAMES.get(int(k), k): v
                                 for k, v in cat_summary["_by_id"].items() if k != 'None'})

    if args.baseline_dir:
        base_rows = read_jsonl(Path(args.baseline_dir) / "per_question.jsonl")
        boot = paired_bootstrap(ok, [r for r in base_rows if r.get("status") == "ok"])
        (run_dir / "paired_bootstrap.json").write_text(
            json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")
        print("  paired bootstrap vs {}: n={}".format(Path(args.baseline_dir).name, boot.get("n")))
        for m in ("f1_official", "hit1", "mrr"):
            b = boot["metrics"][m]
            print("    {}: delta={} ci95={} P(>0)={} wins={} losses={}".format(
                m, b["mean_delta"], b["ci95"], b["bootstrap_probability_gt0"],
                b["wins"], b["losses"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
