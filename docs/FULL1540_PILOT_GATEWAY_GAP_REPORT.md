# Full-1540 Pilot Gateway Gap Report

Status: **PILOT / NON-FORMAL / NOT PAPER RESULT** — 2026-09-12 forensic audit.

## 1. Frozen protocol (reference)

- Manifest SHA256 `49e4043b9c7b02cb20c6387b903f7d0cffe8f396c4e2c083f1f3815f1014036e`
  (1540 questions, 282/321/96/841, three methods share it).
- Gateway ChatAnywhere, request `gpt-4o-mini`, accepted ONLY
  `gpt-4o-mini-2024-07-18`, temperature 0, top-k 10, prompts frozen
  (`748eb924…` / `6d33ff20…`).
- Identity policy: max 3 attempts per question per phase; only valid-snapshot
  responses are scored; per-question 3-attempt exhaustion → STOP whole run;
  30-minute-window identity-mismatch rate > 1% → STOP + report.

## 2. Execution timeline

| Event | time (2026-09-12) |
|---|---|
| SF v2 answer chunk 1 start | ~07:5x |
| First exhaustion candidate | conv-47:29 (3 snapshot_identity_unverified attempts) |
| SF v2 pause + resume with retry policy | third resume auto→completed 262/262 |
| SF v2 answer final | 1539/1540 accepted, conv-48:125 exhausted |
| P4-A accidental parallel start | killed immediately; 7xx rows existed |
| LLM/local logs | no further API |

(Exact timestamps are in the canonical attempt ledger; `raw_model_outputs`
files carry per-question mtime.)

## 3. Canonical coverage (rebuilt from raw files + attempt metadata)

| State | count |
|---|---:|
| accepted (raw provenance + identity_valid) | **1539** |
| identity_exhausted (3 unverified attempts) | **1** (conv-48:125) |
| transport_exhausted / rate_limit_exhausted / evaluator_invariant_violation | 0 |
| not_attempted | 0 |
| provenance_missing | **0** |
| **total** | **1540** |

P4-A accidental: unique questions attempted = 709; accepted = 708;
1 question with any identity mismatch = 6 attempt-level mismatches
(all `snapshot_identity_unverified`).

## 4. Canonical attempt ledger

`results/locomo_e2e/full1540-sfv2-4b/canonical_attempt_ledger.jsonl`
(rebuilt 1542 rows, sha256 `22491adcf65bce89…`):

- accepted attempts: 1539
- rejected: 3 — all conv-48:125, all `snapshot_identity_unverified`
- transport errors: 1 (transient, recovered within attempts)
- rate-limit (429) errors: **0** observed this phase

## 5. Identity mismatch taxonomy (exact)

For conv-48:125, each of the 3 attempts:

| attempt | returned_model | categorized |
|---|---|---|
| 1 | `gpt-4o-mini` (alias) | snapshot_identity_unverified |
| 2 | `gpt-4o-mini` (alias) | snapshot_identity_unverified |
| 3 | `gpt-4o-mini` (alias) | snapshot_identity_unverified |

`stored_valid_model_identity` == `recomputed_valid_model_identity` for all
2028 answer-attempt records scanned → **no evaluator bug**. The exhaustion is
a gateway behavior: ChatAnywhere occasionally serves requests for the alias
`gpt-4o-mini` with the unbounded alias string instead of the pinned snapshot
id, so the identity gate cannot verify the snapshot.

## 6. Was `gpt-4.1-mini-2025-04-14` ever returned by the gateway?

**No.** Global search:

| source | occurrences |
|---|---|
| raw_model_outputs/ | 0 |
| canonical_attempt_ledger.jsonl | 0 |
| per_question.jsonl | 0 |
| docs/*.md | 2 (historical narrative only) |

CORRECTION: earlier statements claiming an explicit `gpt-4.1-mini` model
substitution were not supported by any persisted raw API response and are
**retracted**. All gateway mismatches that were previously labeled "drift"
are re-classified as `snapshot_identity_unverified` (alias returned) or
`transport_error`.

## 7. Why the run is not formal

1. Execution spanned three stop/resume cycles while the identity/provenance
   logging semantics were being changed (identity_category introduced
   mid-campaign; earlier rows have `status=model_drift` legacy labels).
2. One automatic resume occurred after a stop state (protocol discipline:
   a formal run must not auto-resume after a NO-GO stop).
3. Duplicate historical rows exist in the raw per_question JSONL (resume
   upsert replaced error rows but some windows left dual rows).
4. Identity policy naming changed between the first and last attempts
   (`model_drift` → `snapshot_identity_unverified`).

→ The 1539 accepted answers are treated as **pilot data only** (stress
validation, evaluator debugging, cost estimation, gateway behavior audit).

## 8. What remains usable

- `canonical_attempt_ledger.jsonl` / `canonical_per_question.jsonl`
- All raw_model_outputs (1539 accepted + rejected) — provenance chain intact
- Gateway reliability stats (see §9)
- Cost profile (§11)
- Retry-mechanism validation through tests

## 9. Drift / mismatch rates (correct denominators)

- attempt-level identity mismatch rate =
  `snapshot_identity_unverified_attempts / total_attempts`
  = 3 / 1542 ≈ **0.19%** (SF v2 pilot; includes the 3 rejected attempts;
  P4-A accidental = 6/714 ≈ 0.84% — separate run)
- question-level incidence =
  `questions_with_any_mismatch / questions_attempted`
  = 1 / 1540 SF v2; 6 / 709 P4-A accidental
- 30-minute-window rate: computed from the ledger timestamps — the mismatch
  events are sparse and isolated; **the >1%-per-30-min rule was NOT
  triggered** (all windows ≤ ~0.2%). The STOP was caused solely by the
  **per-question three-attempt exhaustion condition** (conv-48:125).

## 10. Excluded from paper

- All Full-1540 pilot metrics (partial run; only 1539/1540 answered and
  0 judged).
- Any Fixed-100 number is NOT re-reported as Full-1540 (unchanged policy).

## 11. Exact cost consumed

| item | value |
|---|---|
| SF v2 pilot answer | ~$0.344 (raw usage summed from accepted + rejected attempts) |
| P4-A accidental answer | ~$0.16 (690 rows) |
| Fixed-100 + all earlier phases | unchanged prior records |
| **campaign-to-date (this pilot addendum)** | ≈ **$0.50** (SF v2 + P4-A accidental) |

(Exact per-question usage is in the raw files; no judge step ran for
Full-1540, so judge tokens = 0.)

## 12. Corrective actions before the next formal run

1. **Canonical attempt logging**: `canonical_reduce.py` + ledger tests
   (tests/test_canonical_reduce.py) — PASS now.
2. **Resume semantics**: a formal resume may NOT auto-run after an
   exhaustion stop; it must wait for explicit approval.
3. **Identity naming**: `model_drift` removed from taxonomies; now
   `identity_valid` / `snapshot_identity_unverified` / `explicit_wrong_model`
   / `transport_error` / `evaluator_invariant_violation`.
4. **Provenance invariant**: accepted rows must trace to a raw response with
   the exact snapshot id (verified: 1539/1539).
5. **Gateway choice for the next run**: direct exact-snapshot endpoint
   (OpenAI official or OpenRouter with provider pinned) — Option A; if
   ChatAnywhere must be used, freeze the retry/backoff policy before the run
   and start a NEW run id (Option B).

## 13. Files touched (this audit)

- `docs/FULL1540_PILOT_GATEWAY_GAP_REPORT.md` (this file)
- `scripts/canonical_reduce.py` (new)
- `tests/test_canonical_reduce.py` (new)
- `scripts/evaluate_locomo_e2e.py`, `scripts/judge_locomo_answers.py`,
  `scripts/summarize_locomo_e2e.py` (identity taxonomy)
- `results/locomo_e2e/full1540-sfv2-4b/canonical_attempt_ledger.jsonl`
- `results/locomo_e2e/full1540-sfv2-4b/canonical_per_question.jsonl`