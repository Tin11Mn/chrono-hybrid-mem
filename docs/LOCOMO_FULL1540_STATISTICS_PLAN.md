# Full-1540 Statistics Plan

Status: PLANNING (2026-09-12). All tests run AFTER the three Full-1540 runs
complete; nothing here executes now.

## 1. Design

- Unit: the **shared 1540-question manifest** (identical question_ids across
  the three methods; per-question pairing by `question_id`).
- Tests: **paired bootstrap**, 10,000 resamples, **seed 20260826**
  (identical to the published evidence-track procedure).
- Statistic: per-resample mean of the per-question metric delta (A − B);
  report Δ (mean), 95% CI ([2.5%, 97.5%] percentile), and
  **Pr(Δ > 0)** — always named **"bootstrap probability"**, never a p-value.
- Decision rule: a comparison is treated as supported only when the 95% CI
  excludes 0 (bootstrap probability reported alongside, not substituted).

## 2. Pairwise comparisons

Primary: **P4-A+BM25 vs SF v2+4B** (the paper's controlled claim).
Secondary: P1 vs SF v2+4B (evolution context). P1 vs P4-A+BM25 may be
reported descriptively.

## 3. Metrics

E2E (primary): f1_mem0, BLEU-1 (m1), LLM Judge Accuracy (0/1 per question).
Diagnostic E2E (secondary tables): f1_official, f1_memoryart, f1_memoryos.
Evidence track (Table B, frozen 1976): Hit@1, Hit@10, MRR — the same paired
procedure over the 1,976 shared questions (already computed; re-reported
under the new tables, never re-run).

## 4. Missing-data policy inside the tests

- The 9 evidence-unresolvable rows and any `retrieval_timeout` row stay in
  the **E2E paired tests** (their F1/BLEU/Judge values exist: 0/0/incorrect
  for timeouts, real scores otherwise).
- They are **excluded from every conditional evidence diagnostic** that needs
  gold mem ids (Hit@10 × Judge matrix, Judge|Hit/Miss accuracies) — those
  denominators are 1530 (+offset 758 if its fresh retrieval succeeded) and
  are stated next to every number.
- Questions missing from one method's run (should be none; protocol treats
  missing as failed) enter the pairing as metric 0 for that method — the
  conservative convention already used by the summarizer.

## 5. Retrieval → QA translation table (Table C)

Per method: Hit@10, Judge Acc | Hit@10, Judge Acc | Miss@10, E2E f1_mem0,
Judge overall. Denominator note as in §4. This table is diagnostic; it must
not be used to tune anything post hoc.

## 6. Reporting discipline

- Every Fixed-100-style "stratified" number is banned from Full-1540
  summaries; Overall is the natural-distribution 1540 micro/overall.
- Full-1540 does not rewrite the frozen 1976 evidence table (Table B).
- External baselines (Mem0, MemoryART, MemoryOS, A-Mem, LightMem) appear in
  Table A as **as-reported**, with protocol-difference caveats.
- The bootstrap outputs land in
  `results/locomo_e2e/full1540/<method>/paired_bootstrap.json` and the
  paper-results doc; the Fixed-100 ↔ Full-1540 comparison is descriptive
  only (overlapping subset — no significance claims).
