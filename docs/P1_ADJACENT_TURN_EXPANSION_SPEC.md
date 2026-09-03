# Spec: P1.1 Post-Hit Adjacent-Turn Evidence Expansion

## Objective

Improve evidence completeness for the fixed downstream `gpt-4o-mini`
Answer/Eval pipeline without changing the answer model. After the existing P1
hybrid retrieval has produced its first-stage raw-message candidates, P1.1 may
add a small number of immediately adjacent turns from the same session to the
rerank pool. This targets conversation structures where the seed names a topic
and the answer appears in the next or previous pronoun/context turn.

This is intentionally distinct from P3-A semantic graph retrieval and from
P3-B1 entity mentions. It is a standalone, feature-gated ablation so a later
gain or loss is attributable. It returns only original `mem_*` rows and their
verbatim raw text; no context concatenation becomes a new memory and no new
model call is made.

## Tech Stack

- Existing Python 3.12/SQLite `MemoryStore` and FastAPI configuration.
- Existing structured plan, P1 hybrid fusion, and ranker; no dependency,
  external service, model, graph database, or schema migration.
- Existing `raw_messages(user_id, session_id, id, content, ...)` ordering and
  `MemoryStore.MODEL_RERANK_LIMIT == 30`.

## Commands

From `chrono-hybrid-mem-p3`, run focused tests:

```powershell
& ..\chrono-hybrid-mem\.venv-local\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .pytest-p1-adjacent-unique `
  tests\test_hybrid_retrieval.py `
  tests\test_model_mode.py
```

Then run the normal local regression suite for affected retrieval behavior:

```powershell
& ..\chrono-hybrid-mem\.venv-local\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .pytest-p1-adjacent-full-unique `
  tests\test_hybrid_retrieval.py `
  tests\test_model_mode.py `
  tests\test_graph_paired_evaluation.py
```

Only after mechanics tests pass, run a predeclared flag-off/flag-on local
paired evaluation over a fixed scope, model and cache. The platform remains the
only source for a final AML score.

## Project Structure

```text
app/main.py
  feature-flag and bounded configuration parsing
app/storage.py
  post-P1 adjacent candidate lookup, slot accounting, retrieval trace
tests/test_hybrid_retrieval.py
  lexical/semantic seed, session boundary, order and raw-evidence tests
tests/test_model_mode.py
  default-off/config validation regressions
docs/P1_ADJACENT_TURN_EXPANSION_SPEC.md
  this approved source of truth
```

## Code Style

The expansion is request-local, parameterized, and raw-message preserving:

```python
if not self.adjacent_turn_expansion:
    return []
neighbors = self._adjacent_raw_candidates(
    connection,
    user_id=user_id,
    seed_ids=p1_seed_ids[:self.adjacent_seed_limit],
    candidate_limit=self.adjacent_candidate_limit,
)
return [row for row in neighbors if row.id not in first_stage_ids]
```

Use explicit `adjacent_*` names, a deterministic seed/neighbor ordering, and
plain JSON-safe trace data. Never mutate shared `last_*` state during lookup;
publish diagnostics only with the completed response.

## Testing Strategy

1. Lexical and semantic seed tests: a topic-bearing seed is retrieved while the
   answer is only in its immediate same-session neighbor; the returned candidate
   is the neighbor's original raw ID and exact raw text.
2. Boundary tests: no cross-session traversal, no missing neighbor, no
   duplicate ID/content, no image-special-case leakage, and no expansion from
   a candidate that does not belong to the user.
3. Determinism tests: fixed first-stage seed order yields fixed neighbor order;
   the hard seed/candidate caps apply; rerank pool stays at or below 30.
4. Isolation tests: feature-off yields byte-for-byte identical candidate IDs and
   result order to the current P1 path; feature-on makes zero additional
   planner/ranker/model calls.
5. Interaction tests: P3-A graph reservation is neither silently removed nor
   exceeded. The first P1.1 outcome experiment runs graph-off so mechanisms
   remain attributable.
6. Evaluation: fixed-20 is mechanics-only. A fixed-200 flag-off/on comparison
   reports Hit@1, MRR, evidence recall, category metrics, call parity and
   cross-user leakage but does not tune parameters from those outcomes.

## Boundaries

- Always: return only original raw messages, stay within one `user_id` and one
  session per neighbor, preserve the 30-candidate limit, expose trace fields,
  and keep the feature default-off.
- Ask first: change the frozen seed/candidate limits after they are set,
  combine this feature with P3-A/P3-B1 in an outcome-bearing experiment, alter
  platform API behavior, start a model, or submit/push/merge.
- Never: create synthetic concatenated memories, use QA/gold IDs to select
  seeds/neighbors/limits, cross session boundaries, add an unbounded iterative
  agent loop, or claim a local score is an AML score.

## Success Criteria

- Default-off is behavior-identical to the existing P1 path.
- Feature-on adds only unique immediate same-session original raw candidates,
  with a fixed, auditable cap and no added model calls.
- Trace contains `adjacent_seed_ids`, `adjacent_candidate_ids`,
  `adjacent_deduped_ids`, and slot-accounting fields.
- Focused and affected regression tests pass.
- The predeclared full paired evaluation, not a tuned fixed-20 slice,
  determines whether P1.1 advances.

## Open Questions

1. The initial limits are now approved and frozen at 4 first-stage seeds and 4
   adjacent candidates. Any change requires a new predeclared experiment.
2. Define the exact allocation rule when P3-A is enabled. The approved first
   experiment is graph-off P1.1; interaction is a later separate task.

## Implementation Plan

1. Add default-off, bounded P1.1 configuration parsing in `app/main.py` and
   thread it to `MemoryStore` without changing existing defaults.
2. Add a request-local SQLite helper in `app/storage.py` which turns the
   already-fused P1 ranking into at most four unique immediate same-session raw
   neighbors, with deterministic seed/neighbor order and no cross-user lookup.
3. Reserve at most four P1.1 candidates in the model's 30-item rerank pool
   only when graph retrieval is disabled; retain original P1 candidate order
   otherwise. Emit complete adjacent-candidate trace diagnostics.
4. Add feature-off, bounds, session boundary, raw-ID/content, determinism, and
   model-call parity regressions. Run the focused suite, then the affected full
   retrieval suite.
5. Run a zero-model trace/mechanics smoke check. Only after it passes may a
   separate, fixed-scope local paired outcome evaluation be proposed.

## Tasks

- [x] Task: Add validated P1.1 configuration.
  - Acceptance: default is disabled; values outside `0..30` fail explicitly.
  - Verify: `tests/test_model_mode.py`.
  - Files: `app/main.py`, `app/storage.py`, `tests/test_model_mode.py`.
- [x] Task: Implement bounded adjacent raw-message lookup and pool allocation.
  - Acceptance: fixed 4/4 caps, same-session/user only, no synthetic text,
    deterministic trace, graph-on path unchanged.
  - Verify: focused retrieval tests and trace assertions.
  - Files: `app/storage.py`, `tests/test_hybrid_retrieval.py`.
- [x] Task: Verify and record the mechanics gate.
  - Acceptance: default-off parity, zero extra model calls, pool <=30, and all
    affected regressions pass.
  - Verify: commands in this specification and zero-model smoke trace.
  - Files: `progress.md`, `task_plan.md`.

## Approval Gate

Implementation begins only after the user approves P1.1. Approval does not
authorize model launch, an outcome evaluation, Git commit/push/merge, or the
separate schema-v4 P3-B1 implementation.
