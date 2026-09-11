# Full-1540 Gateway Capacity Plan (ChatAnywhere)

Status: PLANNING (2026-09-12). No formal Full-1540 request has been sent.

## 1. Required request budget

| Phase | Requests |
|---|---:|
| Answer: 1540 × 3 methods | 4,620 |
| Judge: 1540 × 3 methods | 4,620 |
| Preflights (per method × phases) | ~12 |
| Retry budget (identity/transport/rate-limit, ~2% observed headroom) | ~200 |
| **Total planning budget** | **≈ 9,450** |

(Fixed-100 actually consumed 204 calls including all preflights, one drift
attempt and zero transport failures — the 2% retry budget is conservative.)

## 2. Observed gateway behavior (empirical, this campaign)

| Observation | Value |
|---|---|
| Sustained burst (Fixed-100 answer phase) | ~30 requests/minute for 45 consecutive calls, no 429, no throttling |
| Volume on the current key in one day | **150+ requests without any quota error** — already above the free tier's 100 requests/24 h shared cap |
| Free-tier 429 signature (first key) | `当前24小时窗口的免费请求额度已用完；所有免费模型共享100次请求` |
| Judge drift observed | 1 (`gpt-4.1-mini-2025-04-14` substituted mid-run) — intercepted by the identity gate |

## 3. Account tier verification

- Billing endpoints (`/v1/dashboard/billing/subscription|usage`) return
  **HTTP 302** (web-login redirect) — the account's tier/balance is **not
  API-queryable** from here.
- Empirical inference: the current key sustained 150+ requests in one day
  with 30/min bursts and zero 429s — behavior inconsistent with the free
  100/24 h cap, consistent with a paid/upgraded key.
- **Residual requirement before execution**: the user confirms in the
  ChatAnywhere console that the key's tier removes the 100/day cap and that
  the balance covers ≈ $1 (ChatAnywhere's own token pricing; their paid
  gpt-4o-mini rate is ~30% of OpenAI's list price, so the token-based $0.92
  estimate implies roughly ¥3–6 of gateway balance).

## 4. Rate/latency model and wall-time estimate

| Phase | Measured rate (Fixed-100) | Full-1540 serial estimate |
|---|---:|---:|
| Answer | ~2.4 s/question | 4,620 × 2.4 s ≈ **3.1 h** |
| Judge | ~1.5 s/question | 4,620 × 1.5 s ≈ **1.9 h** |
| Total (3 methods, serial) | — | **≈ 5 h** + preflights/retries ⇒ plan **6–8 h** |

Per-method runs can be split across resumes (checkpointed per question);
a multi-day serial schedule is NOT the default plan (long spans increase
gateway backend-drift exposure, as already observed once). If the tier
confirmation fails, the run does NOT start (NO-GO on capacity).

## 5. Concurrency and stability controls

- llama-server concurrency constraint is irrelevant during E2E execution
  (the loopback model is only used by the already-completed supplementation).
- Serial request flow (the harness is single-threaded) keeps us far below
  any plausible RPM ceiling; no `--parallel`-style acceleration is planned.
- Every response passes the model-identity gate; every row is checkpointed;
  resume is idempotent and never re-bills completed questions.

## 6. Stop conditions (gateway-related)

1. Any 3-attempt identity exhaustion → STOP whole run (NO-GO signal).
2. Two consecutive 429 windows (i.e., quota signals despite the paid tier)
   → STOP + report; do not degrade to a multi-week schedule without an
   explicit user decision.
3. Gateway-side model substitution recurring at > 1% of attempts within any
   30-minute window → STOP + report (drift-rate watch, per Fixed-100).
