# Full-1540 Final Results Audit

**Status**: FULL-1540 RESULTS AUDIT: **PASS**
**Date**: 2026-09-13
**Scope**: READ-ONLY audit of the three persisted formal runs:

- `results/locomo_e2e/full1540-formal-sfv2-4b/`
- `results/locomo_e2e/full1540-formal-p4a-bm25/`
- `results/locomo_e2e/full1540-formal-p1/`

**Truth hierarchy honored**: raw_model_outputs → attempt ledger → canonical_per_question → per_question → checkpoint → summaries.

---

## A. Manifest integrity — PASS

| check | value |
|---|---|
| SHA256 | `49e4043b9c7b02cb20c6387b903f7d0cffe8f396c4e2c083f1f3815f1014036e` ✓ |
| total | 1540 |
| unique qids | 1540 |
| unique offsets | 1540 |
| category dist | {1:282, 2:321, 3:96, 4:841} ✓ |
| duplicate | 0 |
| missing canonical | 0 |

## B. Canonical Answer coverage

| method | logical | accepted | retrieval_timeout | raw answer files | dup qid rows |
|---|---|---|---|---|---|
| SF v2 | 1540 | 1540 | 0 | 1540 | 0 |
| P4-A | 1540 | 1539 | 1 (conv-43:4) | 1539 | 0 |
| P1 | 1540 | 1540 | 0 | 1540 | 0 |

Evidence-unresolvable: exactly 9 per method
(`conv-26:30/37/46`, `conv-49:31/38/46`, `conv-50:39/42/69`) — matches pre-registration.

## C. P4-A Judge STOP/resume — protocol_interrupt_recovered

- **STOP confirmed**: P4-A judge, conv-42:105 (offset 490), 3 consecutive
  `transport_error` → frozen per-question streak STOP → exit 7.
- Resume completed remaining 1049 questions; final `judged_total=1540`.
- **No frozen parameter changed by resume** (prompt/provider/alias/temperature/
  scorer/identity policy/retry policy unchanged).
- **No duplicated scoring, no overwritten accepted result**: final canonical
  state for conv-42:105 = single accepted row (snapshot_verified, CORRECT).
- Verdict: `protocol_interrupt_recovered` — formal results kept; disclosure
  required in reproducibility appendix.

## D. Corrected API attempt accounting (from attempt ledgers)

| | Answer attempts | Judge attempts | identities(answer) | identities(judge) |
|---|---|---|---|---|
| SF v2 | 1541 (1540 acc + 1 transport) | 1544 (1540 acc + 4 wrong retried) | snap 1535 / alias 5 / unavail 1 | snap 1540 / wrong 4 |
| P4-A | 1545 (1539 acc + 3 wrong + 3 transport) | 1544 (1539 acc + 5 transport) | snap 1536 / wrong 3 / unavail 3 / alias 3 | snap 1420 / alias 119 / unavail 5 |
| P1 | 1540 (1540 acc) | 1545 (1540 acc + 5 wrong retried) | snap 1540 | snap 1540 / wrong 5 |

**TOTAL**: Answer **4626** · Judge **4633** · **9259 formal API calls**.

Denominator check `total_attempts == accepted + wrong + transport + rate_limit + malformed`:
- SF: 1541 = 1540+0+1+0+0 ✓; 1544 = 1540+4+0+0+0 ✓
- P4: 1545 = 1539+3+3+0+0 ✓; 1544 = 1539+0+5+0+0 ✓
- P1: 1540 = 1540+0+0+0+0 ✓; 1545 = 1540+5+0+0+0 ✓
- unavailable model_identity noted but always within an accepted or retried outcome.

## E. Corrected overall metrics (N=1540, from canonical tables)

| method | f1_mem0 | bleu1_m1 | f1_official | f1_memoryart | f1_memoryos | judge_correct | judge_acc_1540 |
|---|---|---|---|---|---|---|---|
| SF v2 | 0.471687 | 0.377462 | 0.499166 | 0.447023 | 0.485791 | 1020 | 0.662338 |
| P4-A | 0.470586 | 0.377011 | 0.496896 | 0.447363 | 0.485064 | 1011 | **0.656494** |
| P1 | 0.471379 | 0.377374 | 0.499905 | 0.447319 | 0.485558 | 1022 | 0.663636 |

**Formal P4-A judge_acc = 1011/1540 = 0.656494** (not /1539 → 0.65692).

## F. Corrected per-category metrics

| method | cat | n | f1_mem0 | bleu1 | f1_off | judge |
|---|---|---|---|---|---|---|
| SF v2 | multi_hop | 282 | 0.304931 | 0.200900 | 0.315694 | 151/282 = 0.535461 |
| | temporal | 321 | 0.402275 | 0.325134 | 0.427036 | 140/321 = 0.436137 |
| | open_domain | 96 | 0.210750 | 0.164632 | 0.240210 | 43/96 = 0.447917 |
| | single_hop | 841 | 0.583882 | 0.480933 | 0.617778 | 686/841 = 0.815696 |
| P4-A | multi_hop | 282 | 0.296004 | 0.193210 | 0.304026 | 146/282 = 0.517730 |
| | temporal | 321 | 0.405565 | 0.335705 | 0.433423 | 141/321 = 0.439252 |
| | open_domain | 96 | 0.207787 | 0.162642 | 0.237780 | 42/96 = 0.437500 |
| | single_hop | 841 | 0.583943 | 0.478879 | 0.615374 | 682/841 = 0.810939 |
| P1 | multi_hop | 282 | 0.301106 | 0.198079 | 0.317660 | 154/282 = 0.546099 |
| | temporal | 321 | 0.405824 | 0.328629 | 0.431167 | 138/321 = 0.429907 |
| | open_domain | 96 | 0.210447 | 0.164021 | 0.234325 | 43/96 = 0.447917 |
| | single_hop | 841 | 0.583282 | 0.480454 | 0.617567 | 687/841 = 0.816885 |

Category n sums: 1540 per method ✓.

## G. Retrieval × QA (evidence-resolvable n=1531; 9 excluded)

| method | Hit10+Corr | Hit10+Wrong | Miss+Corr | Miss+Wrong | sum | Judge\|Hit1 | Judge\|Hit3 | Judge\|Hit10 | Judge\|Miss10 |
|---|---|---|---|---|---|---|---|---|---|---|
| SF v2 | 950 | 306 | 65 | 210 | 1531 | 0.778139 (924) | 0.767382 (1165) | 0.756369 (1256) | 0.236364 (275) |
| P4-A | 945 | 310 | 62 | 214 | 1531 | 0.777056 (924) | 0.763948 (1165) | 0.752988 (1255) | 0.224638 (276) |
| P1 | 953 | 303 | 65 | 210 | 1531 | 0.775974 (924) | 0.769099 (1165) | 0.758758 (1256) | 0.236364 (275) |

## H. API identity reliability

| method | phase | total_attempts | accepted | snapshot | alias | wrong | transport | rate_limit | malformed | snap_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| SF v2 | A | 1541 | 1540 | 1535 | 5 | 0 | 1 | 0 | 0 | 0.9968 |
| SF v2 | J | 1544 | 1540 | 1540 | 0 | 4→retried | 0 | 0 | 0 | 1.0 |
| P4-A | A | 1545 | 1539 | 1536 | 3 | 3→retried | 3 | 0 | 0 | 0.9981 |
| P4-A | J | 1544 | 1539 | 1420 | 119 | 0 | 5 | 0 | 0 | 0.9221 |
| P1 | A | 1540 | 1540 | 1540 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| P1 | J | 1545 | 1540 | 1540 | 0 | 5→retried | 0 | 0 | 0 | 1.0 |

`accepted_model_scope_ok`: **true** for all three (all accepted responses
returned ∈ {gpt-4o-mini, gpt-4o-mini-2024-07-18}).
`global_consecutive_429` never reached 5; **rate_limit_attempts = 0**.
Wrong-model attempt **content never entered scored results** (all retried).

Fingerprints: SF A {fp_369*:1538, fp_acf*:2}, J {fp_369*:1517, fp_acf*:23, fp_51e*:4};
P4 A {fp_369*:1501, fp_acf*:38, fp_8d7*:3}, J {fp_369*:1539}; P1 A {fp_369*:1540},
J {fp_369*:1465, fp_51e*:5, fp_acf*:75}.

## I. Cost (token-based; no provider invoice available)

| method | answer in/out | judge in/out | answer $ | judge $ | total $ |
|---|---|---|---|---|---|
| SF v2 | 1,336,697 / 10,098 | 620,355 / 10,804 | 0.2066 | 0.0995 | 0.3061 |
| P4-A | 1,338,436 / 10,176 | 618,378 / 10,769 | 0.2069 | 0.0992 | 0.3061 |
| P1 | 1,336,709 / 10,105 | 620,735 / 10,818 | 0.2066 | 0.0996 | 0.3062 |

**Total ≈ $0.9184** (token-based estimated cost; NOT provider invoice).

## J. Exact paired bootstrap (paired resampling, 10000 draws, seed 20260826, 2.5/97.5 pctl; Δ = SF v2 − baseline)

| comparison | metric | Δ | 95% CI lower | upper | Pr(Δ>0) |
|---|---|---|---|---|---|
| SF v2 − P4-A (N=1540) | f1_mem0 | +0.001100 | −0.003815 | +0.006248 | 0.6638 |
| | bleu1 | +0.000450 | −0.003990 | +0.004904 | 0.5868 |
| | judge | +0.005844 | −0.003247 | +0.014935 | 0.8809 |
| SF v2 − P1 (N=1540) | f1_mem0 | +0.000307 | −0.004583 | +0.005297 | 0.5417 |
| | bleu1 | +0.000088 | −0.004441 | +0.004665 | 0.5109 |
| | judge | −0.001299 | −0.010390 | +0.007792 | 0.3651 |

All 95% CIs include 0 → **no statistically supported difference** at 5%.

## K. N=1539 sensitivity (exclude offset 758 from ALL methods)

| | Δ f1 | Δ judge | CI | Pr |
|---|---|---|---|---|
| SF v2 − P4-A | +0.001101 | +0.005848 | f1 CI [−0.003846, +0.006210]; judge CI [−0.003249, +0.014945] | 0.6656 / 0.8812 |

**Conclusion unchanged**: no significance; sensitivity consistent. Not stratified/deletive.

## L. Reporting corrections

7 corrections documented (see `reporting_corrections.json`), notably:
1. P4-A judge acc: 0.6569(/1539) → **0.656494 (/1540 formal)**.
2. API calls: **9259** (4626 answer + 4633 judge) — not earlier inconsistent totals.
3. **1 formal STOP event exists** (P4 judge conv-42:105) — earlier "no STOP" text corrected.
4. Stress-100: 102 attempts incl. 2 transient transport (not 0).
5. `global_consecutive_429` is a counter-threshold, never described as a rate.
6. f1_mem0 SF v2 (0.471687) > P1 (0.471379); judge_acc P1 (0.663636) > SF v2 (0.662338).
7. P4-A multi_hop per-category judge denominator n=282 (formal) not n=281.

## M. Final scientific conclusion

> SF v2 improves evidence accessibility, but the retrieval gains do not
> translate into statistically supported downstream QA improvements under the
> fixed ChatAnywhere GPT-4o-mini answering/judging pipeline.

- **Evidence-level claim: YES** (Table B frozen: SF v2 Hit@1 0.6108 vs P4-A 0.5850).
- **E2E superiority claim: NO / NOT SUPPORTED** (all bootstrap CIs include 0).
- This means retrieval improvement ≠ guaranteed downstream QA improvement;
  it does NOT mean SF v2 has no value.

## N. Table B (evidence, frozen, unchanged — reference only)

| method | Hit@1 | Hit@3 | Hit@10 | MRR | EvRecall@10 |
|---|---|---|---|---|---|
| P1 | 0.5779 | 0.7176 | 0.7601 | 0.6497 | 0.6001 |
| P4-A+BM25 | 0.5850 | 0.7323 | 0.7809 | 0.6618 | 0.6127 |
| SF v2+4B | 0.6108 | 0.7677 | 0.8219 | 0.6929 | 0.6555 |

---

## Verification of audit gates

| gate | status |
|---|---|
| manifest integrity | PASS |
| canonical coverage | PASS |
| attempt accounting | PASS |
| provenance | PASS |
| scoring denominator | PASS (formal /1540) |
| bootstrap reproducible | PASS |
| STOP/resume auditable | PASS (1 recovered event) |
| reporting corrections | PASS (7 items) |

**FULL-1540 RESULTS AUDIT: PASS**

Artifacts written under `results/locomo_e2e/full1540_final_audit/`:
`canonical_metrics.json`, `per_category_metrics.json`, `api_reliability.json`,
`stop_events.json`, `bootstrap_n1540.json`, `sensitivity_n1539.json`,
`reporting_corrections.json`, `canonical_scoring_{sfv2,p4a,p1}.jsonl`.
Original raw artifacts untouched.