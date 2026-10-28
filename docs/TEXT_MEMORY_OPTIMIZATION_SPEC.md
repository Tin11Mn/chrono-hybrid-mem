# Text memory optimization, 2026-10-28

## Objective and authorization
Implement the user-approved diagnosis priorities in an isolated text-track worktree.
Preserve raw evidence and Add/Search schemas. No production deployment or paid calls
are part of local verification. Baseline is deployment commit 924db24 (runtime d92c63b).

## Design and acceptance criteria
1. Rebuild neighbor indexes across Add chunk boundaries, within the same user/session;
   retrying an ingestion remains idempotent. Upgrade existing databases' context indexes
   once using an explicit schema migration marker. No model calls for indexing.
2. Opt-in evidence selection replaces (does not supplement) the ranking model call.
   Validate selected/excluded IDs against the actual pool, evidence-need coverage, and
   explicit current/superseded state decisions. Apply filtering only to complete valid
   decisions; incomplete/malformed decisions preserve original lexical ordering and
   never trigger another model call. Never globally delete historical messages.
3. Rules retain conditions, definitions, exceptions and worked examples; learning retains
   multiple applicable observations; summaries sample across sessions with original
   provenance. Add a bounded coverage channel for governance/rule/learning/summary,
   preserving at least 18 normal candidates in a 30-message pool. Sampling is a bounded
   retrieval aid, not a claim of exhaustive history compression.
4. Keep successful selections in model order, enforce top_k as an upper bound, and
   avoid appending unrelated/unreviewed evidence. For top_k below selected count,
   trace truncated coverage explicitly. No synthetic answers in responses.
5. Optional trace JSONL stores IDs/counts/status, not memory/query text or model prose;
   failures to write traces must not fail Search. Trace publication uses each call's
   local decision, not the mutable last-query snapshot.
6. New selection/coverage flags default off for rollback compatibility. Provide an
   explicit optimized preset and operation instructions. No additional dependencies.

## Commands
From this worktree on Windows:
`../chrono-hybrid-mem-coding/.venv/Scripts/python.exe scripts/test_offline.py -q -m "not local_research"`
Focused: same Python, `scripts/test_offline.py -q tests/test_text_memory_optimization.py`.
Static: `git diff --check`; Python `-m compileall -q app`.

## Structure and style
`app/evidence_selection.py`: pure validation and metadata-only trace serialization.
`app/model.py`: single-call JSON selector and query intents.
`app/storage.py`: chunk indexing, candidate coverage, selection integration.
`app/main.py`: explicit runtime flags. Tests under `tests/`; report under
`.planning/2026-10-28-text-memory-optimization/`.
Follow existing typed functions and parameterized SQL. Example: `selected_ids: List[str]`.

## Verification and boundaries
Test actual SQLite stores with deterministic model doubles: false current/old states,
historical requests, rules with exceptions, multi-hop coverage, unrelated sensitive
memories, chunk boundaries, user isolation, malformed/empty outputs, failure fallback,
no extra model calls, legacy flag-off behavior, existing-DB migration and trace redaction.
Always run non-local regression suite; never use benchmark gold to tune or fabricate
official gains. Schema work here is internal index migration and uses the user-authorized
implementation scope; no API schema changes, no dependencies, no deployment required.

## Tasks
- [x] Cross-chunk indexing and migration.
- [x] Selector protocol and bounded coverage channel.
- [x] Runtime flags, redacted traces, optimized preset.
- [x] Offline failure/regression tests, comparison report and delivery.

## Verification findings and scoped repair
The baseline's optional `add_session_facts` and semantic reader still reference
`session_facts.source_message_id`, although the current schema uses a separate
source-mapping table. Baseline reproduces four failures. Align these paths with
that table, reject foreign user/session sources, preserve all source links for a
multi-source fact, and explicitly remove replaced links when FK cascades are off.
This does not enable SF or add model calls to the optimized preset.
The new offline runner removes inherited runtime credentials and blocks WAN
sockets. Use it for all remaining regression work.
