"""Paired bootstrap for Full-1540 formal E2E (Formal stats plan).

Primary: P4-A+BM25 vs SF v2+4B; Secondary: P1 vs SF v2+4B.
10000 resamples, seed 20260826, metrics f1_mem0 / bleu1_m1 /
judge(CORRECT=1). Outputs Δ, 95% CI, Pr(Δ>0) named 'bootstrap probability'
(never p-value). Also N=1539 sensitivity (exclude offset 758) for the primary
pair.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
RUNS = REPO / "results" / "locomo_e2e"

SEED = 20260826
DRAWS = 10000
OFFSET758_MARKER = "conv-43:4"  # canonical qid for retrieval_offset 758


def load_rows(run_id):
    p = RUNS / run_id / "per_question.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def metric_map(rows):
    out = {}
    for r in rows:
        ok = r.get("status") == "ok"
        f1m = r.get("f1_mem0")
        b1 = r.get("bleu1_m1")
        j = 1 if r.get("judge_result") == "CORRECT" else 0
        out[r["question_id"]] = {
            "f1_mem0": float(f1m) if ok and f1m is not None else 0.0,
            "bleu1_m1": float(b1) if ok and b1 is not None else 0.0,
            "judge": j,
        }
    return out


def bootstrap(a_map, b_map, shared_ids, metric):
    """Paired bootstrap over shared question_ids, order = shared_ids."""
    rng = random.Random(SEED + hash(metric) % (2**31))
    n = len(shared_ids)
    deltas = [a_map[qid][metric] - b_map[qid][metric] for qid in shared_ids]
    mean_delta = sum(deltas) / n
    boot = []
    for _ in range(DRAWS):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        boot.append(s / n)
    boot.sort()
    lo = boot[int(0.025 * DRAWS)]
    hi = boot[int(0.975 * DRAWS)]
    p_gt0 = sum(1 for x in boot if x > 0) / DRAWS
    return {
        "metric": metric,
        "n": n,
        "mean_delta": round(mean_delta, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "bootstrap_probability_gt0": round(p_gt0, 4),
        "note": "bootstrap_probability_gt0 is NOT a p-value",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "full1540_bootstrap.json"))
    args = ap.parse_args()

    sf2 = metric_map(load_rows("full1540-formal-sfv2-4b"))
    p4a = metric_map(load_rows("full1540-formal-p4a-bm25"))
    p1 = metric_map(load_rows("full1540-formal-p1"))

    # shared ids across ALL three (should equal 1540)
    shared = [q for q in sf2 if q in p4a and q in p1]
    shared_1539 = [q for q in shared if q != OFFSET758_MARKER]

    result = {
        "seed": SEED, "draws": DRAWS,
        "n_shared": len(shared),
        "n_1539": len(shared_1539),
        "offset758_qid": OFFSET758_MARKER,
        "comparisons": {},
    }

    for label, a_map, b_map, ids in [
        ("p4a_bm25_vs_sfv2_4b", p4a, sf2, shared),
        ("p1_vs_sfv2_4b", p1, sf2, shared),
        ("p4a_bm25_vs_sfv2_4b_N1539", p4a, sf2, shared_1539),
    ]:
        result["comparisons"][label] = {
            "n": len(ids),
            "metrics": [bootstrap(a_map, b_map, ids, m)
                        for m in ("f1_mem0", "bleu1_m1", "judge")],
        }

    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()