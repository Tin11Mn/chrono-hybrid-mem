# ChatAnywhere Formal Identity Policy v1.0

Status: **FROZEN** (2026-09-13). This is the final API-layer protocol for the
formal Full-1540 E2E runs. It supersedes every earlier "drift" convention and
the forensic-pilot taxonomy. Parameters below (retry threshold, 429 policy,
MAX_TOTAL_ATTEMPTS_PER_QUESTION=9, alias acceptance, Smoke-20 questions,
Stress-100 PASS semantics) are frozen experiment infrastructure, not
tunable knobs.

## 1. Provider and request

| Item | Value |
|---|---|
| Provider | ChatAnywhere (OpenAI-compatible gateway) |
| Base URL | `https://api.chatanywhere.tech/v1` |
| Key env | `CHATANYWHERE_API_KEY` (never printed/persisted) |
| Requested model | `gpt-4o-mini` (fixed alias) |
| Answer temperature | 0 |
| Judge temperature | 0 |
| Top-k evidence | 10 |
| Prompts | frozen (answer `748eb924…`, judge `6d33ff20…`) |

## 2. Model identity taxonomy (orthogonal, two fields per attempt)

Every API attempt records TWO orthogonal fields (never mixed):

```
model_identity:
  snapshot_verified                returned_model == "gpt-4o-mini-2024-07-18"
  alias_only_snapshot_unverified   returned_model == "gpt-4o-mini"
  explicit_wrong_model             any other non-empty model string
  unavailable                      returned_model missing/empty

attempt_outcome:
  accepted                         model_identity ∈ {snapshot_verified,
                                                     alias_only_snapshot_unverified}
  explicit_wrong_model             rejected, retryable
  transport_error                  network/timeout/exception (NO valid API response;
                                   model unavailable)
  rate_limit                       HTTP 429 (counted separately from transport)
  malformed_response               HTTP/API response SUCCEEDED but response.model
                                   missing/null/"" (model unavailable)
  evaluator_invariant_violation    stored classification contradicts recomputed
```

The `unavailable` model_identity maps to TWO distinct outcomes by whether an
API response was actually received:
- no valid response (network/timeout/connection failure) → `transport_error`
- response succeeded but `response.model` missing/null/"" → `malformed_response`

These two are NEVER mixed.

## 3. Accepted condition

```
accepted_model_identity:
  returned_model in {"gpt-4o-mini", "gpt-4o-mini-2024-07-18"}
```

- `snapshot_verified` and `alias_only_snapshot_unverified` are BOTH valid
  ChatAnywhere GPT-4o-mini responses → ACCEPT.
- Alias-only is NEVER a wrong model, NEVER a stop condition, NEVER an
  identity failure.
- Only `explicit_wrong_model` (e.g. `gpt-4.1-mini-2025-04-14`) is a genuine
  model-identity violation → reject + retry.

## 4. Retry budget (frozen semantics)

Per question, maintain in the attempt record:

```
total_attempts          (count of ALL real API calls, INCLUDING the accepted one)
wrong_model_streak
transport_streak
rate_limit_streak
malformed_streak
```

Rules:

1. **`total_attempts` counts every real API request** — an accepted response
   ends the question (no further retry) but the call still counts. E.g.
   `attempt1: explicit_wrong_model`, `attempt2: alias_only_snapshot_unverified`
   → `total_attempts = 2`, `accepted = true`, `accepted_attempt = 2`.
2. **MAX_TOTAL_ATTEMPTS_PER_QUESTION = 9**: the 9th request IS allowed. If
   the 9th is accepted → question succeeds. Only if the 9th is still a
   retryable failure → STOP. Never stop before the 9th is sent.
3. **Streaks reset on any different outcome**: only consecutive failures of
   the SAME kind accumulate. `wrong_model, wrong_model, transport_error` →
   `wrong_model_streak` = 2, `transport_streak` = 1 (the transport_error
   resets the wrong streak to 0).
4. **STOP (exit 7) when**: any single streak reaches 3, OR `total_attempts`
   reaches 9 with the 9th still retryable-failed.
5. `snapshot_verified` / `alias_only_snapshot_unverified` → accepted, question
   ends.

## 5. HTTP 429 handling

- An attempt is `rate_limit` iff the underlying OpenAI exception carries
  `status_code == 429` (never classified as transport_error).
- Backoff: honor `Retry-After` header first, accepting both `Retry-After: 30`
  (seconds) and HTTP-date formats; else exponential 10s → 30s → 60s.
- `rate_limit_streak` accumulates only on consecutive 429s; any non-429
  attempt resets it to 0.
- Global guard: `global_consecutive_429` increments ONLY when the current
  attempt is a 429; **any non-429 API response or transport result resets it
  to 0** (never a whole-run cumulative count). When it reaches 5 → checkpoint
  + STOP (exit 7).

## 6. Invariant (replaces snapshot-equality acceptance)

The invariant is NO LONGER `returned_model == dated snapshot`:

```
ACCEPTED_IDENTITIES = {"snapshot_verified", "alias_only_snapshot_unverified"}

recomputed_identity = classify_model_identity(raw_returned_model)
recomputed_accepted = recomputed_identity in ACCEPTED_IDENTITIES

assert stored_model_identity == recomputed_identity
assert stored_accepted       == recomputed_accepted
if attempt_outcome == "accepted":
    assert recomputed_accepted
```

Any violation → `attempt_outcome = evaluator_invariant_violation` → STOP
(exit 7), never retry. A response with HTTP 200 and
`returned_model == "gpt-4o-mini"` MUST classify as
`alias_only_snapshot_unverified / accepted=true`; classifying it as a failure
is an evaluator bug.

## 7. Raw response retention; scored-field isolation

- Every rejected-but-HTTP-successful response (`explicit_wrong_model`,
  `malformed_response`) AND every transport/429 attempt is persisted in
  `raw_model_outputs/` for audit.
- The `content` of any non-accepted attempt NEVER appears in scored fields:
  `generated_answer` (answer) and `judge_result`/`judge_label` (judge). Only
  `attempt_outcome == accepted` writes those fields.
- `accepted` canonical state requires an accepted raw response whose
  `returned_model ∈ {gpt-4o-mini, gpt-4o-mini-2024-07-18}` (provenance chain,
  recomputed from raw).

## 8. Stop conditions (formal runs — exhaustive)

STOP only when:
1. same question: 3 consecutive `explicit_wrong_model`
2. same question: 3 consecutive `transport_error`
3. same question: 3 consecutive `malformed_response`
4. persistent 429: `global_consecutive_429 ≥ 5`
5. provenance invariant failure (`accepted` without raw)
6. manifest / config / prompt hash mismatch
7. gold leakage
8. evaluator_invariant_violation
9. `total_attempts` reaches 9 with the 9th still retryable-failed

NEVER STOP because of `returned_model == "gpt-4o-mini"`.

## 9. Reporting fields (per method, answer and judge separately)

```
logical_probes                    (questions asked)
total_attempts                    (all real API calls)
total_responses                   (responses actually received as objects)
accepted_responses                (accepted count)
snapshot_verified
alias_only_snapshot_unverified
explicit_wrong_model
transport_error
rate_limit
malformed_response
accepted_model_scope_ok           (all accepted returned in {gpt-4o-mini, gpt-4o-mini-2024-07-18})
snapshot_verification_rate        = snapshot_verified / accepted_responses
```

`accepted_model_scope_ok` replaces the old `accepted_purity_ok` (which
wrongly implied every accepted response was the dated snapshot).

## 10. Paper wording (frozen)

> We accessed GPT-4o-mini through the ChatAnywhere OpenAI-compatible gateway,
> with the requested model fixed to `gpt-4o-mini`. Returned model identifiers
> were logged for every API call. Responses exposing the dated
> `gpt-4o-mini-2024-07-18` identifier were recorded as snapshot-verified,
> while alias-only `gpt-4o-mini` responses were retained and reported
> separately.

Do NOT claim "all calls used gpt-4o-mini-2024-07-18".

## 11. Comparability

Internal three-way comparison (P1 / P4-A+BM25 / SF v2+4B) stays strictly
fair: identical ChatAnywhere provider, alias, prompts, temperature, and
scorers. Comparisons against external papers (Mem0 et al.) remain
**as-reported** with the explicit caveat that the API provider differs.

## 12. Stress-100 PASS criterion (frozen)

```
Stress-100 PASS iff ALL:
  logical_probes == 100
  accepted_probes == 100
  explicit_wrong_model == 0
  provenance_mismatch == 0
  evaluator_invariant_violation == 0
  identity_exhausted == 0
  transport_exhausted == 0
  rate_limit_exhausted == 0
  malformed_exhausted == 0
  canonical_state_count == 100
  orphan_attempts == 0
  orphan_raw_responses == 0
```

Accepted may be ANY combination of snapshot_verified + alias_only =
100. Transient transport/malformed/429 may occur and be retried, as long as
no exhaustion happens and 100/100 probes end accepted.
`snapshot_verification_rate` (including 0%) must be reported; never claim
exact-snapshot reproduction.

## 13. Smoke-20 PASS criterion (frozen)

Frozen Smoke-20 offsets:
`[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 22, 27, 40, 79, 80, 81, 82, 83]`

```
Smoke-20 PASS iff ALL:
  exact frozen offsets match == true
  Answer canonical accepted == 20/20
  Judge canonical accepted == 20/20
  explicit_wrong_model == 0
  provenance_gap == 0
  parser_failure == 0
  evaluator_invariant_violation == 0
  answer canonical duplicates == 0
  judge canonical duplicates == 0
  accepted_model_scope_ok == true
```

alias_only responses are allowed and counted separately.