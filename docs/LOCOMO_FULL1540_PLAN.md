# LoCoMo Full-1540 Formal Experiment Plan

Status: **PLANNED — NOT STARTED** (2026-09-12). No formal Full-1540
Answer/Judge API request may be sent before explicit approval.

## 1. Method matrix (frozen)

| Row | Method | Retrieval config | Purpose |
|---|---|---|---|
| 1 | P1 | structured query planning only | method-evolution baseline |
| 2 | P4-A + BM25 | P1 + evidence-need channels (quota 2, bm25 selection) | controlled baseline |
| 3 | SF v2 + Qwen3-4B | P4-A+BM25 + Session-Fact semantic layer (top_n 10) | **paper core method** |

Core comparison: **P4-A+BM25 vs SF v2+4B**. Excluded from this run: SF v2+8B
(later Reranker Scaling), P4-A q2, SF v3, graph, temporal channel, P5
variants. The Reranker Scaling row (Table D) is a separate future run.

## 2. Question set (frozen)

- Manifest: `results/locomo_e2e/full1540/question_manifest.json`
- **manifest_sha256: `49e4043b9c7b02cb20c6387b903f7d0cffe8f396c4e2c083f1f3815f1014036e`**
- N = **1540** = every LoCoMo qa entry with category != 5, dataset order
- Natural distribution (verified at generation): multi-hop **282**, temporal
  **321**, open-domain **96**, single-hop **841**
- 0 duplicate question_ids; 0 missing reference answers
- `retrieval_offset` set for 1531 questions (the eligible-index key space of
  the frozen artifacts); null for the 9 evidence-unresolvable questions
- The three methods share this manifest file verbatim. No stratified
  sampling, no deletions, no result-based replacement.

## 3. The 10 questions without frozen retrieval

Policy (details in docs/LOCOMO_FULL1540_MISSING_RETRIEVAL_AUDIT.md):

| Group | n | Handling |
|---|---|---|
| A. questions with frozen retrieval | 1530 | Reuse frozen artifacts verbatim. Re-running is forbidden. |
| B. 9 evidence-unresolvable QA | 9 | One frozen-config Search per method (executed 2026-09-12, see audit doc). E2E Answer/Judge computed normally. Hit@K / MRR / Evidence Recall marked **N/A** for these rows (gold evidence cannot be mapped to mem ids) — never faked as 0. |
| C. offset 758 (conv-43:4, multi-hop) | 1 | One frozen-config Search per method with a bounded wall time (per-call 120 s historical + 600 s overall backstop). Success → normal E2E. Timeout → `retrieval_timeout`: F1=0, BLEU=0, Judge=incorrect, question kept in the N=1540 denominator. Never deleted. |

Supplementation products: `results/locomo_e2e/full1540_missing_retrieval/
{p1,p4a_bm25,sfv2_4b}.jsonl` (one row per question × method with retrieval
ids, raw evidence, latency, timeouts, errors, evidence-metric evaluability).
The 10 rows are E2E coverage repair only — they never re-open the 1976
evidence track and are excluded from the conditional evidence diagnostics
denominators (§7).

## 4. Shared downstream protocol (all three methods identical)

| Item | Value |
|---|---|
| Gateway | ChatAnywhere (`https://api.chatanywhere.tech/v1`) |
| Requested model | `gpt-4o-mini` |
| Accepted returned model | `gpt-4o-mini-2024-07-18` (identity-checked per response) |
| Answer temperature | 0 |
| Judge temperature | 0 |
| Answer prompt | frozen, SHA256 `748eb924…` |
| Judge prompt | frozen, SHA256 `6d33ff20…` |
| Scorers | identical offline scorers for all methods |
| Top-k evidence | 10 |

The only experimental variable is the retrieved evidence produced by the
memory method. Different methods never share generated answers; each
generated answer is scored once by every offline scorer; each answer is
judged once (no regeneration for different metrics).

## 5. Model identity retry policy (frozen, as validated in Fixed-100)

- Max **3 attempts** per question per phase (answer / judge).
- A response is VALID iff `response.model == "gpt-4o-mini-2024-07-18"`.
- Invalid attempts: metadata logged (`judge_attempts[]` / answer attempt
  fields), content never persisted to scored fields, identical inputs
  retried, never counted as completed.
- Retry triggers: model identity mismatch, transport/API failure, explicit
  rate-limit retry. Never based on judge/answer content.
- 3 failures on one question → checkpoint, STOP whole run (NO-GO signal).
- Never accept `gpt-4.1-mini` or any other model; never switch gateway.

## 6. Execution layout and order

```
results/locomo_e2e/full1540/
    question_manifest.json          (shared, frozen)
    full1540_missing_retrieval/     (see audit doc)
    p1/      {run_config.json, per_question.jsonl, checkpoint.json,
    p4a_bm25/ metric_summary.json, category_summary.json,
    sfv2_4b/  raw_model_outputs/}
```

Run order: **SF v2+4B → P4-A+BM25 → P1** (paper method first, then
controlled baseline, then low baseline). All three share the manifest and
protocol; no configuration may be adjusted after seeing another method's
results.

Resume discipline: identical commands with `--resume`; completed questions
are never re-billed; config/manifest hashes verified per launch.

## 7. Metrics and denominators

- **Primary E2E** (per method, per category + Overall): f1_mem0, BLEU-1
  (m1), LLM Judge Accuracy. Overall = natural 1540 micro/overall (NOT the
  Fixed-100 macro).
- **Diagnostic E2E**: f1_official, f1_memoryart, f1_memoryos.
- **Evidence track (Table B)**: the frozen 1976-question results
  (Hit@1/3/10, MRR, Evidence Recall@10) are reused untouched — Full-1540
  does NOT rewrite them.
- **Retrieval → QA translation (Table C)**: Hit@10, Judge Acc | Hit,
  Judge Acc | Miss, E2E F1, Judge Overall — per method. Conditional
  evidence diagnostics use only questions with evaluable evidence metrics
  (1530 + offset-758-if-successful); the 9 unresolvable rows stay in the
  E2E 1540 denominators but out of the conditional evidence denominators.

## 8. Stop conditions

1. Model identity: 3 failed attempts on one question → STOP (NO-GO signal).
2. Cost: per-run `--cost-cap-usd 5.0` hard guard (token-based estimate).
3. Quota: repeated 429s after backoff → checkpoint + STOP + report.
4. Manifest/config hash mismatch on any resume → fail closed (exit 6/3).
5. Any gold leakage detected in a prompt → STOP + quarantine the run.

## 9. Planned tables

- **Table A** — LoCoMo E2E Main Results: P1 / P4-A+BM25 / SF v2+4B internal
  rows; external rows (Mem0, MemoryART, MemoryOS, A-Mem, LightMem) marked
  **as-reported** with protocol caveats.
- **Table B** — Evidence Retrieval Diagnostics: frozen 1976 numbers.
- **Table C** — Retrieval → QA Translation.
- **Table D** — Reranker Scaling (SF v2+8B): future separate run.

## 10. Statistical analysis

See docs/LOCOMO_FULL1540_STATISTICS_PLAN.md (paired bootstrap, 10,000
resamples, seed 20260826, "bootstrap probability" naming discipline).

## 11. Execution prerequisites (from the capacity plan)

1. ChatAnywhere account tier confirmed to sustain ~9,300+ requests without
   the free 100/day cap (empirical evidence + user console confirmation).
2. llama-server needed only for the (already completed) supplementation —
   not during E2E execution.
3. Preflight: Answer + Judge identity checks immediately before each
   method's run.

## 12. Amendments (2026-09-12, approved at execution start — reporting only)

1. **Wording for the 9 evidence-unresolvable questions**: the paper describes
   them as **4 questions with no annotated evidence** and **5 questions with
   malformed evidence identifiers** — never as a blanket "dataset defects".
2. **N=1539 sensitivity analysis**: the formal primary result keeps the
   established protocol (P4-A+BM25 offset 758 `retrieval_timeout` → F1=0,
   BLEU=0, Judge=incorrect, kept in N=1540). After all three methods
   complete, an additional zero-API-cost sensitivity re-computation excludes
   offset 758 from ALL three methods (N=1539): f1_mem0 / BLEU-1 / Judge
   accuracy deltas for P4-A+BM25 vs SF v2+4B with paired bootstrap 95% CI
   and Pr(Δ>0). N=1539 is a robustness check only; N=1540 remains the
   primary.
3. The 9 evidence-unresolvable questions enter E2E N=1540 normally; their
   evidence metrics are always **N/A**; they never enter Hit/Miss conditional
   evidence diagnostics.

No protocol, method, prompt, scorer, manifest, or retrieval-artifact change
is authorized beyond these reporting-only items.
