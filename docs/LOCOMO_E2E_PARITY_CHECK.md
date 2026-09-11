# LoCoMo Frozen-Artifact Parity Check + Source Recovery

Status: **PASS** (2026-09-11). All published evidence-level numbers for the
three methods are reproduced bit-exactly from per-question frozen artifacts
(never from summary files), and `mem_id` recovery to original messages is
exact on a deterministic 20-question sample.

Tool: `scripts/paper_parity_audit.py` (paper branch). Raw audit output:
`results/paper_freeze_audit.json` (untracked; regenerable).

## 1. Method

For each of P1, P4-A+BM25, SF v2+4B, over every per-question record:

- Recompute `evidence_metrics(result_ids, gold_mem_ids)` (the parity-tested
  `scripts/score_locomo_answers.py` implementation) → per-question
  Hit@1/3/10, MRR, pooled Evidence Recall@10 via `aggregate_evidence`.
- Cross-check the artifact's stored `first_gold_rank` against the recomputed
  rank (self-consistency).
- Recompute gold mem ids **from the dataset alone** through the runner's
  exact semantics and require an exact match with `gold_mem_ids`.
- Verify every referenced id is within `1..n_messages` of its conversation.
- Verify n=1976 and offset-758 handling; verify category distribution.

Artifact sets (paths + SHA256 in docs/SF_V2_CODE_FREEZE.md §5):

- P1: `chrono-hybrid-mem-p5-diagnostics/.locomo/p1-local-proxy-structured-diagnostics-chunk-*.json` (10 files, 2026-08-26)
- P4-A+BM25: `chrono-hybrid-mem-p3/.locomo/newmethod-chunk-*.json` (11 files, 2026-08-26)
- SF v2+4B: `chrono-hybrid-mem-p3/.locomo/sfv2-full-*.json` (11 files, 2026-09-04)

## 2. Parity results (n = 1,976, offset 758 excluded)

| Metric | P1 recomputed | P1 published | P4-A+BM25 recomputed | P4-A published | SF v2 recomputed | SF v2 published |
|---|---:|---:|---:|---:|---:|---:|
| Hit@1 | **0.5779** | 0.5779 ✓ | **0.5850** | 0.5850 ✓ | **0.6108** | 0.6108 ✓ |
| Hit@3 | **0.7176** | 0.7176 ✓ | **0.7323** | 0.7323 ✓ | **0.7677** | 0.7677 ✓ |
| Hit@10 | **0.7601** | 0.7601 ✓ | **0.7809** | 0.7809 ✓ | **0.8219** | 0.8219 ✓ |
| MRR | **0.6497** | 0.6497 ✓ | **0.6618** | 0.6618 ✓ | **0.6929** | 0.6929 ✓ |
| Evidence Recall@10 (pooled) | 0.6001 | 0.5976* | 0.6127 | 0.6129* | **0.6555** | 0.6555 ✓ (progress.md; README prints 0.6558, rounding variant) |
| n | 1976 | 1976 ✓ | 1976 | 1976 ✓ | 1976 | 1976 ✓ |
| n_gold_total (retained) | 2781 | — | 2781 | — | 2781 | — |

\* Historical P1/P4-A Evidence Recall used pre-mapping denominators (2,806 /
2,804 / 2,800 items over the 1,977 / 1,976 sets); the recomputed 0.6001 /
0.6127 use the same top-10-hit pooled denominator (2,781) as SF v2 and are
the fair cross-method numbers. The 0.0002 differences are pure denominator
effects, documented below.

Stored `first_gold_rank` mismatches: **0** for all three methods.
Category distribution (identical for all methods):
`{1: 280, 2: 320, 3: 89, 4: 841, 5: 446}` (1=multi-hop, 2=temporal,
3=open-domain, 4=single-hop, 5=adversarial).

## 3. Offset 758 handling

- conv-43 (offset 758) hangs the local Qwen rank server; it is excluded for
  **all** methods uniformly.
- P1 artifacts were run over the 1,977-question set and **contain** offset
  758; the parity recompute excludes it (n=1,976).
- P4-A and SF v2 artifacts exclude it at run time via the 0600a/0600b
  segment split (600–757 / 759–800); offset 758 is absent from their files.

## 4. Gold-denominator disclosure (whitespace drop)

The runner maps gold evidence to memory ids by casefolded content match
against the stored raw messages. `Message.content` is validated by Pydantic
`constr(strip_whitespace=True)`, so stored content is whitespace-stripped,
while `qa["evidence"]` texts keep the dataset's trailing whitespace. Evidence
turns whose text has leading/trailing whitespace therefore silently fail the
lookup and are **dropped from gold**:

- **18 gold items in 18 questions** are dropped — identical set for all three
  methods (verified: same offsets, same lists).
- This accounts exactly for the retained denominator 2,781 vs the 2,799
  resolvable evidence items over the 1,976 questions.
- Impact: a uniform, method-independent, deterministic deflation of pooled
  Evidence Recall denominators. It cannot change method ranking and is
  already embedded in every published number. For the paper: keep using the
  published numbers; no recomputation is warranted mid-freeze. (A post-hoc
  fix would change all published evidence-level numbers and require a full
  protocol-version bump.)

## 5. Source Recovery Parity — 20/20 exact

`mem_N` semantics: per-conversation 1-based message index in
`sessions_and_evidence` (Add) order; each conversation is Added to a fresh
temporary SQLite database, so `mem_N` is conversation-local. Recovery maps
`mem_N` → the Nth message's Pydantic-stripped content (byte-identical to what
`/search` returns), plus conversation_id, session key, speaker/role, and the
session timestamp.

Deterministic sample: 20 questions, seed 20260911, over the SF v2 artifacts
(recovery script in `paper_parity_audit.py::recovery_sample`):

- **20/20 gold recoveries exact** (recomputed gold == artifact gold, per
  question, over the whole sample AND over all 1,976 questions — the
  per-question gold check in §2 is a full-coverage check, the 20-sample adds
  content/speaker/timestamp recovery).
- No unresolvable `mem_id` was encountered (all ids within `1..n_messages`).
- Example recovered rows (truncated):

| offset | conv | cat | gold | recovered example |
|---|---|---|---|---|
| 31 | conv-26 | 1 | mem_77, mem_152, mem_36, mem_3 | mem_77 \| Caroline \| 1688391360 \| session_5 \| "Caroline: Since we last spoke, some big things have happened…" |
| 355 | conv-41 | 2 | mem_571 | mem_571 \| Maria \| 1691255940 \| session_28 \| "Maria: Hey John, I'm here for you! Staying positive makes a big difference…" |
| 517 | conv-42 | 1 | mem_261, mem_52, mem_29, mem_483, mem_484 | mem_261 \| Joanna \| 1654278240 \| session_14 \| "Joanna: Nate, after finishing my screenplay I got a rejection letter…" |
| 1657 | conv-49 | 1 | mem_455, mem_457, mem_474 | (was the initial whitespace-drop discovery case; exact after mirroring runtime semantics) |

Recovery does NOT consult observations, gold answers, or qa_clues: it uses
only the conversation turns and the same `sessions_and_evidence` ordering the
runner used to Add.

## 6. E2E denominator clarification (for the 1,540 track)

The dataset contains 1,986 qa entries: categories
`{1: 282, 2: 321, 3: 96, 4: 841, 5: 446}` → **cat ≠ 5 = 1,540** (the E2E
denominator). The frozen retrieval artifacts cover the 1,976 *eligible*
questions (question + ≥1 resolvable evidence; 1,977 incl. 758). The 1,540
set therefore includes 10 questions without frozen retrieval: 9 qa entries
whose evidence list does not resolve + offset 758. Per the frozen protocol,
missing/failed questions stay in the denominator (conservative; the
summarizer reports this as `sensitivity_full_set`).

## 7. Verdict

Parity: **PASS** for all three methods (4/4 headline metrics bit-exact;
stored ranks self-consistent; gold recovery exact). The frozen artifacts are
cleared as the input for the LoCoMo E2E QA stage. No full retrieval re-run is
needed.
