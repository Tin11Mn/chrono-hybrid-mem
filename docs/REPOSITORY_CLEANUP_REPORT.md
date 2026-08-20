# ChronoHybridMem Repository Cleanup Report

Date: 2026-08-20
Repository: `Tin11Mn/chrono-hybrid-mem`

This report separates the stable P1 repository cleanup from the concurrently running P3 Evidence Graph experiment. All algorithmic P3 files and uncommitted P3 work were left untouched.

## 1. Final Repository State

### Stable and research refs

| Ref | SHA at cleanup audit | Role |
|---|---|---|
| `main` | `94d43ea2d907eb0ef28983e82461da24f6c141f8` | Stable P1 code and normalized CI baseline |
| `research/p3-evidence-graph` | `9b4e95f1b6769b8e9eabf4a5ffd2847b01d7879e` | Remotely preserved P3 planning checkpoint |
| local `agent/p3-evidence-graph` | `dfb775221e46b072c783179847e944508ae5b6a1` plus active uncommitted work | Isolated P3 implementation worktree |

`94d43ea` is the stable P1/CI SHA observed before publishing this documentation-only report and multilingual README set. The final documentation commit may advance the `main` branch head without changing the P1 retrieval algorithm; the stable P1/CI baseline remains identifiable by this SHA and PR #8.

At the audit checkpoint, the only remote branches were:

```text
main
research/p3-evidence-graph
```

Three registered worktrees remained:

| Worktree | Branch / SHA | Reason retained |
|---|---|---|
| `chrono-hybrid-mem` | `agent/p1-main-cleanup@abc925a` | Contains three uncommitted P3 planning records from the parallel research workflow; not altered |
| `chrono-hybrid-mem-cleanup` | documentation branch based on `main@94d43ea` | Isolated repository-cleanup publication worktree |
| `chrono-hybrid-mem-p3` | `agent/p3-evidence-graph@dfb7752` plus active changes | Live P3 implementation; protected |

The historical `stash@{0}` (`pre-late-interaction-local-experiments`) was preserved. A stash is independent of the deleted local branch refs and was not treated as disposable cleanup data.

### Tags

All five long-term tags are annotated tags and were remotely dereferenced to their intended commit targets:

```text
v0.1.0
v0.2.0
research-v0.3.0
research-v0.4.0
research-p1-20260816
```

## 2. P1 Cleanup

The pre-cleanup `main@0691afa` contained two duplicate-definition defects:

- `app/main.py` defined `structured_query_plan_from_environment()` twice; the later copy silently changed the no-environment default to `false` even though the README and `.env.example` described P1 as default-on.
- `app/model.py` defined `plan_query_structured()` twice.

The cleanup retained the implementation and configuration used by the recorded P1 experiment, removed the duplicate definitions, and aligned the API-service behavior:

```text
MEMORY_STRUCTURED_QUERY_PLAN=true
```

The API service now defaults P1 on and supports an explicit `false` opt-out. A regression test covers both cases. This does not mean every direct caller defaults P1 on: direct `MemoryStore` construction remains default-off, and the LoCoMo evaluator still requires `--structured-query-plan`. That distinction is documented in all README languages.

No retrieval weights, candidate-pool sizes, prompts, dense parameters, model-call counts, or P3 code were changed by the P1 cleanup.

## 3. Tests

Local verification after the multilingual README and consistency test were added:

| Check | Result |
|---|---|
| `python -m pytest` | **97 passed**, 0 failed, 1 warning |
| `python -m compileall -q app tests scripts` | Passed |
| `python scripts/check_readme_consistency.py` | Passed for 5 languages and 5 annotated tags |
| `python scripts/compare_release_v020.py --cases examples/release_comparison_diagnostic.json` | Passed; delta Recall@1 `0.0`, delta MRR `0.0` |
| `git diff --check` | Passed |

The single warning is an existing Pydantic v2 deprecation: `min_items` in `app/schemas.py` should eventually become `min_length`. It did not cause a test failure and was not expanded into this repository-hygiene task.

The full 1,977-question P1 metrics were **not rerun during cleanup**. They are historical recorded results from 2026-08-16, preserved in `docs/P1_LOCAL_EVALUATION.md` and `research-p1-20260816`.

## 4. CI

The previous `Verify` workflow installed only `requirements.txt` and ran all tests. Two HTTP embedding adapter tests imported NumPy, so both PR #7 and the preceding `main` run failed with `ModuleNotFoundError: numpy` even though the P1 diff was unrelated.

Dependency audit found:

- 94 of the original 96 tests needed only the core dependencies.
- 2 mocked loopback HTTP embedding tests additionally needed NumPy.
- 0 tests instantiated FastEmbed or downloaded a model.

The final structure is:

### Core verification

- Workflow/job: `Verify / core-verification`
- Installs `requirements-test.txt`, which extends `requirements.txt` with pinned NumPy.
- Runs all tests, Python compilation, the frozen v0.2 diagnostic comparison, and the standard Docker build.
- This is the primary lightweight PR check.

### Local research smoke

- Workflow/job: `Local Research Smoke / dependency-and-image-smoke`
- Triggered manually or only when local-research paths change.
- Installs `requirements-local.txt`, verifies FastEmbed/NumPy imports with `HF_HUB_OFFLINE=1`, runs the two dependency-bound tests, and builds `Dockerfile.local`.
- It does not instantiate BGE/ColBERT or download model weights.

### External evaluation

- `External LoCoMo Retrieval Evaluation`: manual-only.
- `LoCoMo gpt-4o-mini Search Evaluation`: manual-only, with an explicit missing-key failure message.

This avoids automatic full-dataset network evaluation and automatic paid API calls on `main` pushes. The repository currently has no branch-protection rule, so GitHub does not technically mark any status check as required. If branch protection is enabled later, only the core job should be required by default; research and external evaluations should remain path-scoped or optional.

## 5. Tags

| Tag | Dereferenced commit | Purpose | Result status |
|---|---|---|---|
| `v0.1.0` | `5fd77045c74a5b17876abca30812888587628eaa` | Minimal reliable baseline | Historical release |
| `v0.2.0` | `7cf45c76ea7998554a13386b924627b83aeb3134` | Official Rank 5 / 44.33 version | Organizer-confirmed on 2026-08-20 |
| `research-v0.3.0` | `3a0ba8c06722fe53b07d8d251ada1729c390bcdc` | BGE + ColBERT local hybrid milestone | Local research; not official |
| `research-v0.4.0` | `1b013b93a0f3e6e12366208f20eae1d245889909` | Qwen reranker + time-aware key milestone | Local research; not official |
| `research-p1-20260816` | `0691afafe4cede21f973efb996b86a29d441ff88` | P1 structured-planning full local milestone | Post-submission local research; not official |

No existing tag was deleted, moved, or force-updated.

## 6. PR Cleanup

| PR | Action | Reason |
|---|---|---|
| #1 `agent/local-memory-v0-3` | Commented and closed | Superseded; exact state preserved by `research-v0.3.0` |
| #2 `agent/qwen-reranker-v0-4` | Commented and closed | Superseded; exact state preserved by `research-v0.4.0` |
| #7 P1 consistency fix | Commented and closed | Superseded by the broader, CI-verified PR #8 |
| #8 P1 + CI normalization | Checks passed and squash-merged | Established `main@94d43ea` stable baseline |

Open PR count after cleanup: **0**.

## 7. Remote Branch Cleanup

Deleted after tag/PR/tree-equivalence checks:

```text
agent/bilingual-readme
agent/gpt4o-mini-rubric-candidate
agent/late-interaction-fusion
agent/local-memory-v0-3
agent/qwen-reranker-v0-4
agent/structured-plan-p1
agent/p1-main-cleanup
agent/repository-cleanup
```

Kept:

```text
main
research/p3-evidence-graph
```

The active local `agent/p3-evidence-graph` was never pushed or deleted by cleanup. Its lifecycle belongs to the parallel P3 workflow.

## 8. Local Branch and Worktree Cleanup

Deleted local branches after verifying their content was preserved by main, merged PRs, or research tags:

```text
agent/bilingual-readme
agent/bilingual-readme-clean
agent/gpt4o-mini-rubric-candidate
agent/hierarchical-retrieval
agent/late-interaction-fusion
agent/local-memory-v0-3
agent/qwen-reranker-v0-4
agent/repository-cleanup
agent/structured-plan-p1
```

The clean `chrono-hybrid-mem-p1-submit` worktree was unregistered. Its initial non-force removal encountered a permission-restricted `.pytest_tmp` directory, so cleanup stopped rather than escalating deletion. The user subsequently removed the residual directory, which was then verified absent.

Local branches/worktrees tied to active P3 records were deliberately retained:

```text
agent/p1-main-cleanup
agent/p3-evidence-graph
research/p3-evidence-graph
```

## 9. Repository Files

- Preserved all source code, tests, evaluation scripts, stable docs, fixtures, `findings.md`, `progress.md`, and `task_plan.md` provenance.
- Added `assets/chronohybridmem-logo.png`, using the logo supplied by the repository owner.
- Added `.gitignore` coverage for pytest/evaluation temp-name variants, common Python tool caches, local virtual environments, runtime SQLite files, logs, coverage output, IDE metadata, and OS metadata.
- Confirmed no cache, database, `.env`, model weight, LoCoMo dataset, or temporary log was tracked.
- Did not delete active `.locomo`, model-cache, evaluation-cache, or P3 worktree data because those directories may be in use by the parallel experiment.

## 10. GitHub About

Repository description:

> Evidence-grounded hybrid long-term memory retrieval for AI agents. Developed for the Agent Memory Challenge Academic Textual Memory track (Rank 5).

Topics:

```text
agent-memory
ai-agents
evidence-retrieval
fastapi
fts5
locomo
long-term-memory
memory-retrieval
sqlite
agent-memory-challenge
```

- Homepage: left empty because no verified project website was available.
- Default branch: `main`.
- Branch protection: not enabled at the audit checkpoint.

## 11. Multilingual README

Completed:

```text
README.md          English
README.zh-CN.md    简体中文
README.es.md       Español
README.ja.md       日本語
README.ko.md       한국어
```

Each README uses the same owner-supplied logo, language navigation, architecture, official/local result separation, version map, API examples, commands, dependency boundaries, P3 status, and safety disclosures. The default GitHub page is English.

## 12. README Consistency

`scripts/check_readme_consistency.py` and `tests/test_readme_consistency.py` verify:

- all five files and the shared logo exist;
- the current language is bold and the other four navigation links resolve;
- local documentation and asset links exist and do not escape the repository;
- fenced code blocks, commands, JSON, inline technical tokens, numeric facts, and content links match the English source;
- Rank 5, Overall 44.33, the **CONFIRMED** v0.2 mapping, P1 metrics, stable/P3 status marker, branch/tag names, configuration flags, and dependency files remain consistent;
- all five version tags remain annotated and point to the audited commits.

Result: **passed**.

### Post-cleanup official confirmation

On 2026-08-20, the competition organizer confirmed that the official result
used `v0.2.0` at commit
`7cf45c76ea7998554a13386b924627b83aeb3134`. This resolves the uncertainty
recorded during cleanup without changing any tag target or historical commit.
See [Official Evaluation Confirmation](OFFICIAL_EVALUATION_CONFIRMATION.md).

## 13. P3 Safety Verification

- Remote `research/p3-evidence-graph@9b4e95f` still exists.
- Active `chrono-hybrid-mem-p3` worktree still exists on local `agent/p3-evidence-graph@dfb7752`.
- The active P3 worktree contains ongoing Evidence Graph implementation changes; cleanup did not stage, restore, switch, rebase, merge, delete, or otherwise mutate them.
- The original worktree's three P3 planning-note changes were also left untouched.
- No force push, hard reset, clean, P3 rebase, or P3-to-main merge occurred.
- P3's base includes the P1 duplicate/default fix, preserving a meaningful future P1 flag-off versus P1+P3 comparison.

Therefore P3 can continue through its planned P3-0, P3-A, P3-B, P3-C, fixed-200, and full gates. The active uncommitted P3 implementation should be checkpointed and pushed by the P3 workflow when its own validation boundary is reached; repository cleanup did not claim ownership of that experiment.

## 14. Final Recommended Structure

```text
branches
├── main                         stable validated P1
└── research/p3-evidence-graph  active experimental research checkpoint

annotated tags
├── v0.1.0
├── v0.2.0
├── research-v0.3.0
├── research-v0.4.0
└── research-p1-20260816

CI
├── Verify / core-verification          primary lightweight PR verification
├── Local Research Smoke                path-scoped/manual dependency and image check
├── External LoCoMo Retrieval Evaluation manual
└── LoCoMo gpt-4o-mini Search Evaluation manual

documentation
├── README.md
├── README.zh-CN.md
├── README.es.md
├── README.ja.md
├── README.ko.md
├── docs/
└── assets/chronohybridmem-logo.png
```

Long-term rule:

> `main` stores the stable research implementation, annotated tags freeze historical milestones, `research/*` carries unfinished research, and `agent/*` is temporary development state that is removed after integration or archival.
