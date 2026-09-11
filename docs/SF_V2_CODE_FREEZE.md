# SF v2 Code Freeze (paper/locomo-e2e)

Status: **FROZEN** (2026-09-11). This document is the authoritative mapping
between the paper's core method (Session-Fact Semantic Layer, **SF v2**) and
the exact code, artifacts, and protocol that produced the published numbers.

## 1. Frozen branch

| Item | Value |
|---|---|
| Paper branch | `paper/locomo-e2e` (worktree `chrono-hybrid-mem-paper/`) |
| Base | `main` @ `f6c1f98` ("Merge research/p3-evidence-graph into main: complete research progression table in README") |
| Method code commit | **`f6c1f98`** — every later commit on the paper branch touches ONLY evaluation tooling, prompts, tests, and docs (`app/` is untouched after f6c1f98) |
| Evaluation-port commit | `3613f13` (E2E harness + prompts + tests ported from the untracked main-worktree files) |
| `main` untouched | Yes — `main` remains at `f6c1f98`; all work happened in a separate worktree/branch |

## 2. What code produced "SF v2 + Qwen3-4B = Hit@1 0.6108"?

**Answer: `main @ f6c1f98` is byte-identical to the code that ran the full
1976-question SF v2 evaluation.**

Evidence chain:

1. **Source commit of SF v2**: `da147e1` "feat(session-facts): adopt SF v2
   semantic fact layer after full-1976 significance" (2026-09-04 20:14 +0800).
2. `git diff da147e1 f6c1f98` touches ONLY the five README translations
   (README*.md). `app/storage.py`, `app/model.py`,
   `scripts/evaluate_locomo_retrieval.py` are byte-identical between
   `da147e1` and `f6c1f98`.
3. The frozen artifacts (`sfv2-full-*.json`, 11 files) have mtimes
   2026-09-04 14:01–20:09 +0800 — the run finished 5 minutes before
   `da147e1` was committed (the classic run-then-commit pattern; the working
   tree that produced the artifacts is what `da147e1` then committed).
4. **Empirical verification**: the frozen-artifact parity test
   (docs/LOCOMO_E2E_PARITY_CHECK.md) recomputes Hit@1 0.6108 / Hit@3 0.7677 /
   Hit@10 0.8219 / MRR 0.6929 bit-exactly from the artifacts using the
   `f6c1f98` code lineage. Code↔result mapping is therefore verified, not
   assumed.

## 3. Is `f37804f` required? — **No.**

`f37804f` ("fix(session-facts): restore v2 flat-cache injection",
2026-09-05 20:42) is **excluded** from the paper branch and is **not needed**:

1. **What it actually does**: `c024e07` (SF v3 exploration, 2026-09-05 18:42)
   reworked `_inject_session_facts` in `scripts/evaluate_locomo_retrieval.py`
   and placed the dict-only guard *before* the SF v2 flat-list branch, so v2
   caches injected 0 rows under the SF v3 code. `f37804f` moves the flat-list
   branch back before the guard. Code delta: 7 lines in that file + docs.
2. **The paper branch never contains `c024e07`**, so the regression it fixes
   does not exist on `paper/locomo-e2e`. `main @ f6c1f98`'s
   `_inject_session_facts` already handles the SF v2 flat-list cache
   correctly — it is the *same* code path that ran the 0.6108 experiment.
3. **Timeline proof**: the 0.6108 artifacts were produced 2026-09-04
   (14:01–20:09); `c024e07` (which introduced the breakage) was committed
   2026-09-05 18:42 and `f37804f` 2026-09-05 20:42 — both **after** the run.
   The fix restores the pre-`c024e07` behavior that the paper branch already
   has.
4. Cherry-picking `f37804f` onto the paper branch would additionally drag in
   SF v3 diagnosis documentation while changing no behavior.

## 4. Excluded commits (REJECT experiments)

`research/p3-evidence-graph` is 4 commits ahead of `main`; **none** are in the
paper branch:

| Commit | Content | Status |
|---|---|---|
| `c024e07` | SF v3 bridging/profile layer (storage.py ±149, eval script ±154) | REJECT (full-1976 not significant) |
| `f37804f` | Repair of the c024e07 v2-injection regression (+ miss/loss diagnosis, top_n convergence) | Not needed (see §3); its diagnosis closes the direction: **SF v2 top_n=10 is locally optimal** (top_n=60 scored 0.5850 < 0.6050 on conv-26, 4W/8L) |
| `b9f09cf` | SIMPLE log-scaled temporal channel | REJECT (conv-26 A/B no gain) |
| `197c14b` | REM diagnosticity annotation + RANK_PROMPT_V2 re-check | REJECT (no gain) |

## 5. Frozen retrieval artifacts (SF v2 + Qwen3-4B)

Location: `chrono-hybrid-mem-p3/.locomo/` (11 files). Runner:
`chrono-hybrid-mem-p3/.locomo/_run_sfv2_full.py` (drives
`python -m scripts.evaluate_locomo_retrieval` with the flags below, per
segment; `_sfv2_full.log` records the timeline, including the seg-0800
timeouts that were re-run in 50-question windows and merged into
`sfv2-full-0800.json`).

Run configuration (exact flags):

```
--dataset ..\chrono-hybrid-mem\.locomo\locomo10.json
--local-search-model-url http://127.0.0.1:8081/v1   (Qwen3-4B rerank proxy, llama.cpp)
--local-search-model-name local
--structured-query-plan
--include-question-diagnostics
--evidence-need-retrieval --evidence-need-quota 2 --need-select-by-bm25
--session-fact-layer
--session-fact-cache .locomo\session-facts-full.json
--local-embedding-model BAAI/bge-small-en-v1.5      (loads the dense encoder solely
                                                     to score the SF fact channel;
                                                     --dense-weight 0 keeps the dense
                                                     evidence channel off)
--local-cache-dir ..\chrono-hybrid-mem\.model-cache
--model-timeout 120
```

| File | SHA256 | mtime (2026-09-04, +0800) |
|---|---|---|
| sfv2-full-0000.json | `60392b01f7837a4fb25b1c30af3364eb60b4fa9941786a91860a52dd62ef104d` | 14:01:16 |
| sfv2-full-0200.json | `64a8d9e10f9ca09ee2252e3fdf5624907bfa7b2a0c8d848f0325e7e15299f5ff` | 14:30:19 |
| sfv2-full-0400.json | `8439897cae4351bfc65f139f14154b0cd286c2878c5c0b77f787386539fddff6` | 14:58:55 |
| sfv2-full-0600a.json | `923bb3e43de3bfaae3eecfe138862c2856fbe243c057cd476a4b8b15ca88df2b` | 15:21:58 |
| sfv2-full-0600b.json | `7ede8b8aa61c94ea8cd5fc8a1442cbc11cdf5b5ae5a946ae51398b1c0c14156f` | 15:29:25 |
| sfv2-full-0800.json | `9fd3eca1e9532c5553aeca83e999ccaed4aeb828ed8f4a27d50357ffbaf77d96` | 20:09:08 |
| sfv2-full-1000.json | `41bb4823a6ce5c79786dee51d1a51daa0f6f06f755a7a2491ac83779b9c71979` | 16:13:37 |
| sfv2-full-1200.json | `e5d65d892f80e7f8c8c89544b550639ef0e884c7c69e3be35f94547dfcc8a77d` | 16:43:19 |
| sfv2-full-1400.json | `71776a3627f00dd8c1721442ecfde51637722d63afc09bb756d83a944748b878` | 17:16:39 |
| sfv2-full-1600.json | `95b1c0ea72a3891d3a30bdc287e0188d3a0ecff2a11ab6c1f9fe65bb23122e3c` | 17:51:09 |
| sfv2-full-1800.json | `c4828a93e46640c5054f88f1f5a5d8277056e4b21b0fb073d8ca9c2802157a9f` | 18:21:09 |
| session-facts-full.json (offline SF cache) | `f52438ddae4a2387f1d324eeccc74f87aeb74d4d8a194ebb33b9e61aa0693b84` | 03:35:52 |

Dataset: `chrono-hybrid-mem/.locomo/locomo10.json`, SHA256
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
(byte-identical to the official release).

Dedup policy when merging segments: sorted-glob order, keep-first
(`evaluate_locomo_e2e.load_frozen_diags`). Offset 800 appears in both
`sfv2-full-0600b.json` (759–800) and `sfv2-full-0800.json` (800–999); the
`0600b` record wins.

## 6. Paper method identity (to cite)

```
Paper Branch:        paper/locomo-e2e
Paper Method:        ChronoHybridMem + SF v2 (P1 structured planning
                     + P4-A q2 evidence-need retrieval + bm25 selection
                     + Session-Fact semantic layer, session_fact_top_n=10)
Paper Method Commit: f6c1f98 (app/ and evaluation-runner code identical to da147e1)
Retrieval Artifact:  chrono-hybrid-mem-p3/.locomo/sfv2-full-*.json (11 files, SHA256 above)
LoCoMo Dataset:      chrono-hybrid-mem/.locomo/locomo10.json (79fa87e9…8ff4)
Evidence Evaluation: 1976 historical track (offset 758 excluded for all methods)
E2E Evaluation:      1540 non-adversarial track (category != 5)
```
