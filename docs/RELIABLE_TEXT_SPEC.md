# Reliable text retrieval and new smoke

## Objective and observed faults

Fix the 62.02 smoke regression without benchmark-specific answers: preserve fallback
ranking, increase validated extraction coverage, retain explicit corrections/dates,
retrieve natural-language rules with exceptions and all observation counterexamples,
and make cross-session summary coverage larger than the previous 30-source pool.
Deploy an exact new immutable version to an empty database and start Smoke only.

Independent live captures show invented/noncontiguous spans, resolved names missing
from first-person witnesses, invented symbolic conditions, wrong kind=update and
omitted effective dates. Strict grounding correctly rejects these. The remedy must
prevent malformed source references rather than make grounding permissive.

## Architecture and boundaries

- Keep API/auth/model and raw-source response schema. No new dependency or service.
- Build immutable sentence/paragraph source units. The model references supplied IDs;
  the program copies contiguous original source text as spans. No synthesized spans.
- Use @speaker for explicit first person. Unit all is available only when bounded;
  referential context must be included in a literal witness rather than invented.
- Bounded extraction pages for long sources and one repair attempt for incomplete or
  invalid annotation. No model I/O under SQLite writes. Complete means all pages were
  certified and all records validated; incomplete annotations remain visibly partial.
- Preserve witnessed transition cues and explicit event/effective dates. No inferred
  relative dates, silent assertion supersession, or cross-scope merging.
- Natural-language rules remain original-source evidence for downstream reasoning;
  do not translate them into guessed executable expressions. Retain exceptions.
  Parse observation measurements from their exact witnessed value before broad span,
  preserving multiple rows and counterexamples even in one source.
- Separate evidence sufficiency from literal answer presence: complete premises can
  support downstream calculation without storing a fabricated answer. Unknown inputs
  and missing premises still prevent completeness.
- Expand bounded specialist bundles to96sources when necessary, preserving30for
  ordinary retrieval. Summary candidates cover interior stages and sessions, with
  explicit limits. Required evidence cannot disappear via a30-item truncation.
- Keep independent fallback ranking from768428d. Never relax tenant/source/privacy
  obligations or claim unvalidated completeness. Preserve old tags,DBs and scores.

## Files/style

app/source_annotations.py owns source-unit protocol, app/model.py bounded model calls,
app/structured_memory.py source ledger/operators, app/storage.py candidate publication.
Tests in tests/test_reliable_text.py; immutable synthetic captures in tests/fixtures.
Use small typed Python helpers, e.g. `def source_units(text: str) -> List[dict]:`.

## Commands and verification

`../chrono-hybrid-mem-coding/.venv/Scripts/python.exe scripts/test_offline.py -q tests/test_reliable_text.py tests/test_ranking_fallback.py tests/test_smoke_repair.py tests/test_structured_memory.py tests/test_text_memory_optimization.py`

`../chrono-hybrid-mem-coding/.venv/Scripts/python.exe scripts/test_offline.py -q -m "not local_research"`

`git diff --check`

Before activation, actual provider tests must validate complex extraction and current/
history/time/rules/observations/summary queries, plus16simultaneous HTTPAdds. Document
all failures and resource limits. Old README baseline failure remains separate.
New Smoke must visibly show the new version/new writes; capture UI proof and initial
servercounts. No Full, hidden answers, guessed gold, credentials in logs, or promise
of an improved official score before the new evaluation completes.
