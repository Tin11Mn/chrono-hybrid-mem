# Full-1540 Missing-Retrieval Audit (10 questions)

> **Purpose statement**: the supplementation in this document exists ONLY to
> give the 1540-question E2E run complete coverage. It is NOT a re-run of the
> evidence benchmark, must never be merged into the frozen 1976-question
> artifact set, and does not change any evidence-level published number.

- Date: 2026-09-12
- Manifest: `results/locomo_e2e/full1540/question_manifest.json`
  (SHA256 `49e4043b…036e`)
- Products: `results/locomo_e2e/full1540_missing_retrieval/{p1,p4a_bm25,sfv2_4b}.jsonl`
- Tool: `scripts/full1540_supplement_retrieval.py`
- Retrieval configs: the three methods' frozen flags (P1 = structured
  planning; P4-A+BM25 = + evidence-need quota 2 + bm25 selection;
  SF v2+4B = + session-fact layer with the frozen
  `session-facts-full.json` cache + FastEmbed bge-small, dense weight 0),
  loopback Qwen3-4B llama-server (b9637, ctx 16384, single slot),
  per-call timeout 120 s (historical), overall wall cap 600 s per search.

## 1. The 9 evidence-unresolvable questions (dataset defects)

These qa entries are valid questions with reference answers, but their gold
evidence lists cannot be mapped to raw messages by the runner's
`dia_id -> content -> mem_id` chain — so they never entered the 1976-question
retrieval-eligible set:

| manifest offset | question_id | cat | Root cause | gold evidence field |
|---|---|---|---|---|
| 30 | conv-26:30 | 3 | **empty evidence list** | `[]` |
| 37 | conv-26:37 | 1 | semicolon-joined ids in one string | `["D8:6; D9:17"]` |
| 46 | conv-26:46 | 3 | **empty evidence list** | `[]` |
| 1257 | conv-49:31 | 3 | space-joined ids in one string | `["D9:1 D4:4 D4:6"]` |
| 1264 | conv-49:38 | 3 | space-joined ids in one string | `["D22:1 D22:2 D9:10 D9:11"]` |
| 1272 | conv-49:46 | 3 | space-joined ids in one string | `["D21:18 D21:22 D11:15 D11:19"]` |
| 1421 | conv-50:39 | 3 | **empty evidence list** | `[]` |
| 1424 | conv-50:42 | 3 | **empty evidence list** | `[]` |
| 1451 | conv-50:69 | 2 | leading-zero turn id (no `D30:05` exists; turn ids are `D30:5`) | `["D30:05"]` |

All 9 have `answer_present = true` → E2E Answer/Judge are fully computable.
Observation (recorded, NOT acted on): the 5 malformed-id rows could be
resolved by id normalization (splitting/zero-stripping); doing so would be a
gold-mapping code change and is out of scope for this freeze. Evidence
metrics for all 9 rows are therefore reported as **N/A / unavailable**, never
as 0.

## 2. offset 758 (conv-43:4, multi-hop)

conv-43 = 29 sessions / 680 raw messages. Its question is eligible (gold
evidence resolves), but the historical rank call hung the llama-server
(openai.APITimeoutError at the 120 s per-call limit), so 758 is absent from
the P4-A/SF v2 frozen artifacts (present-but-excluded in the P1 artifacts).

Handling: one fresh frozen-config Search per method, wall-capped. If it
succeeds, the row's evidence metrics ARE evaluable (gold maps correctly) but
are used only for E2E-side diagnostics — the frozen 1976 evidence table stays
untouched. If it times out: `retrieval_timeout` → F1=0, BLEU=0,
Judge=incorrect, kept in the 1540 denominator.

## 3. Supplementation execution results

Executed 2026-09-12 (30 searches = 10 questions × 3 methods):

| Method | ok | retrieval_timeout | latency min/median/max |
|---|---:|---:|---|
| p1 | 10 | 0 | 65 / 126 / 271 s |
| p4a_bm25 | 9 | **1 (offset 758, 399 s → wall/transport timeout)** | 49 / 88 / 399 s |
| sfv2_4b | 10 | 0 | 60 / 90 / 119 s |

**offset 758 outcome**: retrieval SUCCEEDED for P1 (126 s) and SF v2+4B
(77 s) — the historical hang did not reproduce under the same frozen config
and server (b9637, ctx 16384) — but timed out for P4-A+BM25 (399 s). Per the
frozen policy: P1/SF v2 rows enter E2E normally; the P4-A+BM25 row is
`retrieval_timeout` → F1=0, BLEU=0, Judge=incorrect, kept in the 1540
denominator. Its `evidence_metrics_evaluable` stays true (gold maps), but it
is used only for E2E-side diagnostics, never merged into the frozen 1976
table.

Result rows (per method) carry: question_id, manifest offset, retrieval
offset, config flags, commit (`f6c1f98`), retrieval_status, top-10
`retrieval_ids` + `raw_evidence`, `plan` (and SF v2 session-fact channel
diagnostics), latency, timeouts, errors, and `evidence_metrics_evaluable`
(false for the 9 unresolvable rows; true for offset 758 where retrieval
succeeded).

## 4. Integrity rules applied

- Fresh per-conversation temporary stores, exactly mirroring the runner
  (Add order = sessions_and_evidence order; SF facts injected through the
  same `_inject_session_facts` helper with the same frozen cache).
- No gold answers, observations, or qa_clues enter any query or prompt.
- No retrieval parameter differs from the historical frozen configs.
- The supplement outputs are consumed only by the Full-1540 E2E harness
  through the manifest's explicit flags; they are never treated as frozen
  artifacts.
