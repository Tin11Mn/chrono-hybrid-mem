# Round 2: grounded specialist memory indexes

Approved scope: continue the weaknesses listed in the prior diagnosis. Baseline
for this round is 660a08f. Keep the existing Add/Search raw-source contract and
the current isolated text worktree. No paid test calls or production changes.

## Acceptance criteria
1. Add a reversible `MEMORY_STRUCTURED_MEMORY` flag, requiring evidence selection.
   Reuse the one Add extraction call to return bounded, source-witnessed records.
   Never trigger model backfill or extra planning/selection calls automatically.
2. Persist facts/states/rules/observations/events/explicit directed links with exact
   source spans, tenant/session/source IDs, normalized event timestamps, and
   extraction-completeness status. Invalid annotations cannot modify raw evidence.
3. Resolve state slots by normalized subject/attribute/scope. Explicit updates with
   known event times supersede earlier values; late ingestion is not recency.
   Same-time conflicts, missing times, unsupported scopes, future/hypothetical and
   retracted values must remain distinguishable. Historical/as-of queries keep
   relevant versions; do not delete old sources. Authoritative exclusions require
   complete indexing for the user, otherwise report a partial-index limitation.
4. Maintain incremental extractive session/topic rollup pages of <=16 records,
   each retaining every supporting source ID/span. This is a source-grounded
   compression/index layer, not free-form conclusions or fabricated summaries.
5. For explicit literal conditions (`weight<=20`, `fragile=true`), evaluate rules
   deterministically, include triggered exceptions, and report unknown inputs.
   Natural-language rules outside the supported grammar stay retrieval-only.
6. Numeric observation summaries (count/range/mean/trend and exact affine fits)
   retain all contributing source IDs and counterexamples. Contradictory pairs,
   missing values and extrapolation cannot become established facts. Transfer
   recalls compatible source schemas across sessions without inventing target facts.
7. Traverse only witnessed directed causes/before/enables edges, bounded to3hops
   and16outgoing edges; reject negation/speculation, preserve full source paths,
   and never confuse correlation or reverse order with causation.
8. State/rule/path/observation evidence obligations must survive selection when
   the whole required bundle fits the <=30pool. If it does not fit, report coverage
   limits instead of claiming completeness. History remains available. Privacy
   improves by source minimization; field-level alteration of required raw sources
   is outside the formal raw-source contract and will not be silently introduced.

## Implementation
`app/structured_memory.py`: source validation, SQLite ledger/rollups, state/rule/
numeric/path operators, bounded candidate/obligation assembly.
`app/model.py`: one-call structured extraction and catalog-aware query planning.
`app/storage.py`: atomic Add integration and Search obligations/candidate hints.
`app/main.py` and `.env.text-optimized.example`: explicit runtime activation.
Tests use deterministic extraction fixtures and actual SQLite stores, never hidden
benchmark gold. No new package dependencies, eval(), or instruction execution.

## Commands and verification
Windows Python: `../chrono-hybrid-mem-coding/.venv/Scripts/python.exe`.
Tests: `scripts/test_offline.py -q tests/test_structured_memory.py` then full
`scripts/test_offline.py -q -m "not local_research"`.
Static: `git diff --check`; `python -m compileall -q app`.
Outputs stay in `.planning/2026-10-28-text-memory-optimization/` with round2 prefixes.

## Tasks
- [x] Source-witnessed ledger, completeness, extractive rollups and migration.
- [x] State version solver, safe as-of routing and query obligations.
- [x] Literal rule, numeric observation and directed path operators.
- [x] Runtime integration and independent adversarial/regression tests.
- [x] Document actual coverage, remaining limits, reproducible evidence and release.

Verification:41new specialist cases passed. Final offline full run781pass/2fail/2deselected:
one pre-existing README/tag failure and one Windows cache-file lock failure, latter
passed unchanged on targeted retry. Compile and diff checks passed. See ROUND2_REPORT.md
in the workflow output folder. Runtime semantic extraction quality and official scores remain unmeasured.
