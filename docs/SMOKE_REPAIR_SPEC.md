# Repair failed smoke ingestion and real model protocol

User approved cancellation of teval_1b423908456bf011 and repair before another smoke.
Base/source preserved8352433. Do not amend its published tag or evaluation record.

1. Cancel via platform UI and verify cancelled status; preserve partial database/traces.
2. Move all network/model extraction out of SQLite write transactions. Keep atomic,
   idempotent raw/index publication and retry-safe handling, including duplicate concurrent
   requests and user isolation. Short writes must tolerate16concurrent synthetic Add calls
   without holding a writer lock during model waits. Never solve this by merely lengthening
   the busy timeout or silently acknowledging uncommitted ingestion.
3. Capture only synthetic real-model request/response protocol examples. Determine exact
   schema mismatch before changing validators. Preserve provenance/ID/sufficiency checks;
   do not declare success by relabeling malformed responses as complete.
4. Make actual provider output conform with the minimal documented protocol, normalize
   only lossless structural variants. Keep one normal extraction/planning/selection call,
   explicitly bounded failure behavior and private diagnostics. No hidden gold inspection.
5. Verify meaningful concurrent writer/reader/duplicate/error tests, replay captured
   synthetic output through validators, existing tests and small real staging checks.
6. Publish a new immutable repair version and deploy only after real validated extraction
   and selection are observed. Keep8352433and its partial DB available for audit/rollback.
   Platform smoke restart is separate from this repair unless user explicitly requests it.

Files:app/storage.py,app/model.py,app/evidence_selection.py,app/structured_memory.py,
tests undertests/,private workflow artifacts under.planning/2026-10-28-smoke-repair/.
Test via../chrono-hybrid-mem-coding/.venv/Scripts/python.exe scripts/test_offline.py.
No new dependencies. No arbitrary code execution or source-message rewriting.
