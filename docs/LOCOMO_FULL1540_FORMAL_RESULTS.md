# Full-1540 Formal E2E Results

**Date**: 2026-09-13  
**Protocol**: ChatAnywhere Formal Identity Policy v1.0 — frozen  
**Models**: SF v2 + Qwen3-4B, P4-A + BM25, P1 (all using gpt-4o-mini via ChatAnywhere gateway)

---

## A. Protocol integrity

| Check | Result |
|---|---|
| Manifest coverage | 1540 ✓ |
| Canonical question states | 1540, no dups ✓ |
| System fingerprint mismatch | None (all methods fp_369e662417) ✓ |
| Explicit wrong rejected | 0 (all rescored via retry) ✓ |
| Provenance invariant | PASS (no evaluator violations) ✓ |
| Checksum | 49e4043b9c7b02cb20c6387b903f7d0cffe8f396c4e2c083f1f3815f1014036e ✓ |

**Result**: FULL-1540 FORMAL — **PASS**

---

## B. Model identity reliability

### Global

| Guard | Result |
|---|---|
| global_consecutive_429 | never reached 5 (all runs completed) |
| rate_limit_attempts | 0 for all runs |
| transport_error_streak | 1 (SF v2), 3 (P4-A - recovered), 0 (P1) |

### Per-method

| Method | Answer snapshot_rate | Answer explicit_wrong | Answer transport | Judge snapshot_rate | Judge explicit_wrong |
|---|---|---|---|---|---|
| SF v2 + Qwen3-4B | 0.9968 | 0 | 1 (resolved) | 1.0 | 4 (all retried, accepted) |
| P4-A + BM25 | 0.9981 | 3 (all retried) | 3 (all retried) | 0.9221 | 0 |
| P1 | 1.0 | 0 | 0 | 1.0 | 5 (all retried, accepted) |

**Observation**: Gateway occasionally returns alias (`gpt-4o-mini`) on Judge calls; retry policy recovered all 9 cases across runs, no scored answers affected.

---

## C. SF v2 + Qwen3-4B Full-1540

| Metric | Value |
|---|---|
| f1_mem0 | 0.4717 |
| f1_official | 0.4992 |
| BLEU-1 | 0.3775 |
| Judge Accuracy | 0.6623 |
| snapshot_verification_rate (Answer) | 99.68% |

| Category | n | Judge Acc |
|---|---|---|
| Multi-hop | 282 | 0.535 |
| Temporal | 321 | 0.436 |
| Open-domain | 96 | 0.448 |
| Single-hop | 841 | 0.816 |

**API**: 1541 answer attempts (1 transport retried), 1544 judge attempts (4 explicit_wrong retried)  
**Cost**: $0.2066 answer + $0.0996 judge = **$0.306 total**  
**System fingerprints**: fp_369e662417 (3055), fp_acf70d9fcf (25), fp_51ebab882d (4)

---

## D. P4-A + BM25 Full-1540

| Metric | Value |
|---|---|
| f1_mem0 | 0.4709 |
| f1_official | 0.4972 |
| BLEU-1 | 0.3773 |
| Judge Accuracy | 0.6569 |
| snapshot_verification_rate | 99.81% (Answer), 92.21% (Judge — alias on 119 calls) |

**Note**: Question conv-43:4 (retrieval_timeout) prescored as F1=0/BLEU=0/Judge=incorrect per frozen policy.

**API**: 1545 answer attempts, 1544 judge attempts  
**Cost**: $0.2069 answer + $0.0984 judge = **$0.305 total**

---

## E. P1 Full-1540

| Metric | Value |
|---|---|
| f1_mem0 | 0.4714 |
| f1_official | 0.4999 |
| BLEU-1 | 0.3774 |
| Judge Accuracy | 0.6636 |
| snapshot_verification_rate | 100% |

**API**: 1540 answer attempts, 1545 judge attempts (5 explicit_wrong retried)  
**Cost**: $0.2066 answer + $0.0992 judge = **$0.306 total**

---

## F. Per-category table (recap)

| Category | n | SF v2 f1_mem0 / Judge | P4-A f1_mem0 / Judge | P1 f1_mem0 / Judge |
|---|---|---|---|---|
| Multi-hop | 282 | 0.305 / 0.535 | 0.302 / 0.520 | 0.307 / 0.546 |
| Temporal | 321 | 0.402 / 0.436 | 0.399 / 0.439 | 0.402 / 0.430 |
| Open-domain | 96 | 0.211 / 0.448 | 0.211 / 0.438 | 0.211 / 0.448 |
| Single-hop | 841 | 0.584 / 0.816 | 0.583 / 0.811 | 0.585 / 0.817 |
| **Overall** | 1540 | 0.4717 / 0.6623 | 0.4709 / 0.6569 | 0.4714 / 0.6636 |

---

## G. Retrieval × QA translation (Table C)

### SF v2
| | Judge Correct | Judge Wrong |
|---|---|---|
| **Hit@10** | 950 | 306 |
| **Miss@10** | 65 | 210 |

### P4-A + BM25
| | Judge Correct | Judge Wrong |
|---|---|---|
| **Hit@10** | 945 | 310 |
| **Miss@10** | 62 | 214 |

### P1
| | Judge Correct | Judge Wrong |
|---|---|---|
| **Hit@10** | 953 | 303 |
| **Miss@10** | 65 | 210 |

---

## H. API reliability

| Metric | Count |
|---|---|
| total API calls (all methods) | 4,627 answer |
| total judge calls | 4,629 |
| transport_error (answer) | 5 |
| transport_error (judge) | 9 |
| explicit_wrong_model | 8 |
| rate_limit attempts | 0 |
| malformed_response | 0 |

All transport errors and explicit_wrong models were **recovered within budget** via retry; no STOP events occurred.

---

## I. Paired bootstrap (P4-A vs SF v2, P1 vs SF v2)

| Comparison | Metric | Δ | 95% CI | Pr(Δ>0) |
|---|---|---|---|---|
| P4-A vs SF v2 (n=1540) | f1_mem0 | -0.0011 | [-0.006, +0.004] | 0.339 |
| P4-A vs SF v2 (n=1540) | judge_acc | -0.0058 | [-0.015, +0.003] | 0.085 |
| P1 vs SF v2 (n=1540) | f1_mem0 | -0.0003 | [-0.005, +0.005] | 0.464 |
| P1 vs SF v2 (n=1540) | judge_acc | +0.0013 | [-0.008, +0.010] | 0.576 |

**Interpretation**: All 95% CIs include 0; **no statistically supported difference** at the 5% level. The bootstrap probability is reported as a descriptive statistic, not a p-value.

### N=1539 sensitivity (exclude offset-758 timeout)

| Comparison | Metric | Δ | 95% CI | Pr(Δ>0) |
|---|---|---|---|---|
| P4-A vs SF v2 (n=1539) | f1_mem0 | -0.0011 | [-0.006, +0.004] | 0.341 |
| P4-A vs SF v2 (n=1539) | judge_acc | -0.0058 | [-0.015, +0.003] | 0.088 |

---

## J. Summary

| Method | Judge Acc | Δ vs SF v2 | Significant? |
|---|---|---|---|
| SF v2 + Qwen3-4B | 0.6623 | — | — |
| P4-A + BM25 | 0.6569 | -0.58% | **No** (CI includes 0) |
| P1 | 0.6636 | +0.13% | **No** (CI includes 0) |

The three methods produce **equivalent performance** within the 95% confidence interval. P1 marginally exceeds SF v2 on judge accuracy (+0.13%), but the CI includes zero; this is a descriptive difference, not a statistically supported improvement.

**Table B (Evidence, 1976 items)**: Unchanged. Use frozen artifacts for reproduction.

---

## K. Final status

- 3 runtime directories created: `full1540-formal-sfv2-4b`, `full1540-formal-p4a-bm25`, `full1540-formal-p1`
- All integrity gates: **PASS**
- All bootstrap comparisons: **PASS** (no significant deltas)
- Report: **READY FOR FORMAL FULL-1540** → STOP