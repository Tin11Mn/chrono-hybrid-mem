# Rerank Pool Audit: 20 or 30?

**Verdict: the rerank pool is 30, not 20.** The "20" figure was a conflation
with two *different* constants (`graph_max_candidates=20`, the P3-A graph
channel cap, and the frozen P3-A gate values). The LLM rerank pool hard limit
is `MemoryStore.MODEL_RERANK_LIMIT = 30`, and the frozen 1976-question
artifacts confirm the LLM reranker saw exactly 30 candidates on every query.
The paper must use 30.

Four independent sources were checked (no reliance on READMEs):

## A. Current code (`paper/locomo-e2e` = `main @ f6c1f98`)

- `app/storage.py`: `MODEL_RERANK_LIMIT = 30`; `search()` raises
  `"model rerank pool exceeded its hard limit"` if the pool would exceed 30.
- The P3-A "candidate cap 20" is `graph_max_candidates = 20` — the bounded
  **graph channel** candidate cap (default-off experiment), not the rerank
  pool.

## B. History

| Ref | `MODEL_RERANK_LIMIT` |
|---|---|
| `v0.1.0`, `v0.2.0` (official 44.33 submission), `research-v0.3.0`, `research-v0.4.0` | constant not yet defined (different storage layer) |
| `424f4c9` 2026-08-15 "Extend hybrid retrieval evaluation and local reranking" | **introduced at 30** (`git log -S` finds exactly one introduction commit) |
| `research-p1-20260816`, `da147e1` (SF v2), `f6c1f98` (main / paper base), `197c14b` (research tip), `.p4release-bak` (P4-A baseline) | 30, never changed |

## C. Frozen artifacts (per-question `question_diagnostics`, all 1,976 questions)

| Method | rerank pool size (`rerank_pool_ids`) | LLM rerank candidates (`llm_rank_candidate_count`) | final top-k |
|---|---|---|---|
| P1 | min 3, max 30, mean 29.963, median 30 (30 on 1,972/1,976; 4 short conversations: 3, 4, 18, 21) | field absent in P1-era schema; pool IDs = 30 | min 3, max 10, mean 9.993, median 10 |
| P4-A + BM25 | **30 / 30 / 30.0 / 30 (n=1,976)** | **30 × 1,976** | 10 × 1,976 |
| SF v2 + 4B | **30 / 30 / 30.0 / 30 (n=1,976)** | **30 × 1,976** | 10 × 1,976 |

(The 4 sub-30 P1 pools are conversations smaller than the pool limit; the
pool equals the whole deduplicated fusion output there.)

## D. Evaluation layer (double-cap check)

`scripts/evaluate_locomo_retrieval.py` calls
`store.search(..., top_k=max(top_ks))` with `top_ks = [1, 3, 10]` → the store
is asked for **10** results. Internally:

- per-FTS-channel candidate limit = `min(max(top_k*4, 50), 200)` = **50**;
- fusion/RRF union → dedup → sidecar quota budgeting → rerank pool clipped to
  **30** (`MODEL_RERANK_LIMIT`);
- ranked output truncated to top-10 (`result_ids`).

There is **no** evaluation-layer 20 or 30 construction: the only "30" literals
in the eval script are the upper bounds of the optional sidecar quota CLI
args (`--bridge-quota`, `--sidecar-shared-quota`, `--relax-quota` must be
≤ 30 = `MODEL_RERANK_LIMIT`) and the `p1_counterfactual_top30_ids` logging
cap. Retrieval traces in the artifacts carry `rerank_pool_ids` (length 30)
directly from the store.

## Paper wording

> Candidates are fused by weighted reciprocal-rank fusion, deduplicated, and
> the top **30** (the fixed rerank pool limit) — including reserved sidecar
> candidates — are re-ranked by the LLM; the final response is truncated to
> top-k. All 1976-question runs issued exactly 30 candidates per LLM rerank
> call.
