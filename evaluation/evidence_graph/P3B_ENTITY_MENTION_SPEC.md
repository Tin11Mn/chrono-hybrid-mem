# Spec: P3-B1 Provenance-Anchored Entity Mention Index

## Objective

P3-A's strict semantic graph remains intact but is not a viable retrieval arm:
the fresh schema-v3, zero-model materialization has only 3 witnessed relation
edges for 419 raw messages. P3-B1 adds a separate, high-coverage evidence
index:

```text
Entity label --exactly mentioned in--> Mention witness --> Raw message
```

The index asserts only that an entity label occurs at a precise span in a raw
message. It never asserts a semantic relation, does not generate an answer,
and passes only the original raw message to the existing ranker. Its purpose is
to give a frozen query-plan entity a bounded, provenance-backed route to
candidate evidence.

The cache preflight establishes the opportunity without QA/gold data: 1,307 of
1,596 source-local extracted entity declarations can be independently anchored
in raw text, spanning 363 of 419 messages (86.6%). This is motivation, not a
retrieval score claim.

## Tech Stack

- Python 3.12, SQLite and the existing FastAPI application.
- Existing `MemoryStore`, structured query plans, and reranker; no new model,
  dependency, remote service, graph database, or vector-store dependency.
- Existing Unicode contract: NFKC -> casefold -> all-whitespace collapse, with
  explicit codepoint-offset remapping to raw-message text.

## Commands

Run focused tests from `chrono-hybrid-mem-p3`:

```powershell
& ..\chrono-hybrid-mem\.venv-local\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .pytest-p3b-unique `
  tests\test_evidence_mention_storage.py `
  tests\test_evidence_mention_audit.py `
  tests\test_graph_paired_evaluation.py
```

Run the existing hard-gate regression suite before a fresh materialization:

```powershell
& ..\chrono-hybrid-mem\.venv-local\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .pytest-p3b-full-unique `
  tests\test_evidence_graph_core.py `
  tests\test_evidence_graph_storage.py `
  tests\test_graph_paired_evaluation.py `
  tests\test_graph_p3a_gate.py
```

The first materialization must be cache-reuse/prepare-only with a new schema
path and an explicit local loopback URL. It must report zero model calls. A
paired Search command is not authorized until the pre-Search audit criteria
below pass.

## Project Structure

```text
app/storage.py
  schema migration, source-local declaration and mention materialization,
  bounded anchor lookup
evaluation/evidence_graph/evidence_mention_audit.py
  evaluator-owned mention witness/schema/trace audit
evaluation/evidence_graph/run_locomo_paired.py
  separate P3-B1 paired protocol and frozen trace contract
tests/test_evidence_mention_storage.py
  storage/schema/span/round-robin regressions
tests/test_evidence_mention_audit.py
  independent adversarial verifier regressions
tests/test_graph_paired_evaluation.py
  pre-Search rejection and paired-arm invariants
evaluation/evidence_graph/P3B_ENTITY_MENTION_SPEC.md
  this approved source of truth
```

## Code Style

Every persisted anchor is fail-closed and preserves its proof rather than
recreating it from an in-memory model object:

```python
mention = find_exact_source_mention(source_content, entity_label)
if mention is None:
    return None
return MentionWitness(
    normalization_id=MENTION_NORMALIZATION_ID,
    source_span=(0, len(normalized_source)),
    mention_span=mention.span,
    source_span_sha256=sha256(normalized_source),
)
```

Names use explicit nouns (`mention_witness`, `anchor_paths`), all SQL is
parameterized, spans are half-open Python codepoint offsets, and diagnostics
are JSON-safe plain data. The semantic graph's `graph_paths` and relation
support types must not be reused for mention anchors.

## Testing Strategy

1. Unit tests cover NFKC/casefold/whitespace handling, Latin word boundaries,
   CJK/non-Latin exact matching, repeated mentions, source-less entities,
   hallucinated labels, and span/hash remapping failures.
2. Storage tests verify compound foreign keys, one row per exact occurrence,
   `user_id` isolation, idempotent Add, no retroactive attachment to legacy
   rows, and no mutation of the P3-A semantic-edge behavior.
3. Evaluator tests forge each sidecar field (entity ID, source ID, span,
   hash, label, normalization ID) and prove that the independent audit rejects
   it without importing a production matching helper.
4. Retrieval tests verify per-seed cap, total cap, round-robin visit order,
   raw-message-only candidates, identical frozen plan calls across off/on arms,
   and zero additional Search calls.
5. A zero-model preflight on the frozen cache must show all persisted mentions
   independently audited and source-message coverage of at least 80%; otherwise
   P3-B1 is rejected before paired Search. This threshold is a safety/coverage
   gate fixed before outcomes, not a QA/gold-derived tuning choice.
6. Only then run a fixed-config paired evaluation. Fixed-20 remains mechanics
   only; fixed-200 is the first outcome-bearing decision. No local result is
   presented as an AML leaderboard score.

## Boundaries

- Always: retain raw-message provenance; validate every persisted anchor before
  Search; keep P3-A semantic edges unchanged; use user-scoped compound keys;
  enforce bounded candidate/seed traversal; run focused and hard-gate tests.
- Ask first: introduce the schema migration and a new P3-B1 candidate channel;
  add any dependency; change platform-facing API behavior; start a model or
  run a paired outcome evaluation after preflight.
- Never: loosen the P3-A S-P-O grammar to increase count; treat an entity label
  as a fact; use QA answers/gold IDs/Hit outcomes to select anchors or tune
  caps; send data to a non-loopback service; overwrite an existing prepared
  artifact; commit, push, or merge without user direction.

## Success Criteria

- A fresh, versioned database contains only independently witnessed entity
  mentions and retains the existing 3 semantic relation edges unchanged.
- Every mention revalidates from SQLite raw content with zero audit violations.
- The frozen-cache preflight reaches at least 80% raw-message mention coverage
  with zero model calls and no QA/gold access.
- The P3-B1 trace can prove candidate cap, per-seed cap, round-robin sequence,
  and that each candidate is an original raw message with an audited anchor.
- Paired arms preserve frozen-plan/model-call/database invariants.
- No score claim is made until a separately predeclared full evaluation passes.

## Open Questions

1. Use a separate schema-v4 database versus a sidecar database dedicated to
   P3-B1; the default recommendation is schema v4 so transaction and FK
   integrity remain atomic.
2. Freeze P3-B1's candidate cap, per-seed cap, RRF weight, and rerank quota in
   the implementation plan before any outcome-bearing Search.
3. Keep session-neighbor expansion as a separate P1 completeness ablation. It
   is promising from public leaderboard designs, but combining it with P3-B1
   in the first experiment would prevent attribution of any gain or loss.

## Implementation Plan

The user approved this scope after P1.1. The following protocol decisions are
fixed before any implementation outcome is observed:

- Materialize schema-v4 source-local, de-duplicated label declarations and
  their `graph_entity_mentions` children in the same prepared SQLite database
  as raw messages and P3-A's unchanged semantic edges.  A declaration's
  `anchor_entity_id` is a deterministic hash of its user, source and canonical
  label; it is not a model identifier or a `graph_entities` identity, and
  P3-B1 never treats it as one.
- Each mention child is bound by a compound FK to that declaration and to the
  same raw source.  It stores canonical source and mention half-open spans,
  independently remapped raw-message codepoint spans, and a canonical-source
  hash.  A match is stored only when both canonical endpoints map safely to raw
  boundaries and normalizing that raw slice reproduces the label.  Source-local
  declaration persistence occurs before P3-A's global entity linking, so the
  latter's deliberate same-name ambiguity/identity fail-closed rules cannot
  reduce P3-B1 mention coverage.
- Match an entity label only against its own source's canonical raw text; store
  all valid occurrences, never a model-proposed span.
- Keep P3-B1 disabled by default and mutually exclusive with P3-A for the
  first ablation. Its initial bounded retrieval configuration is 6 seed labels,
  at most 20 source candidates inspected per seed, 20 anchor candidates in
  total, RRF `0.025`, and a 4-item rerank reservation. These values reuse the
  already audited P3-A mechanical bounds and cannot be tuned from fixed-20
  outcomes.
- Add an evaluator-owned mention auditor and a P3-B1 trace contract before any
  database is materialized. Only then rebuild a fresh cache-reuse schema-v4 DB
  and run the zero-model coverage preflight.

## Tasks

- [x] Task: Define and persist source-exact declaration/mention witnesses.
  - Acceptance: schema-v4 source-local declaration parent plus composite
    FK/uniqueness/check constraints; exact Unicode-normalized matching; all
    occurrences per source; P3-A edges retain their current semantics.
  - Verify: new storage/schema/span regressions and a fresh cache-reuse build.
  - Files: `app/storage.py`, `tests/test_evidence_mention_storage.py`.
- [x] Task: Add evaluator-owned mention audit and materialization contract.
  - Acceptance: audit reloads SQLite raw text and rejects forged entity/source,
    span, hash, label and schema fields without production matcher imports.
  - Verify: adversarial audit tests and zero-model DB audit.
  - Files: `evaluation/evidence_graph/evidence_mention_audit.py`,
    `evaluation/evidence_graph/run_locomo_paired.py`,
    `tests/test_evidence_mention_audit.py`.
- [x] Task: Add bounded, traceable anchor candidate retrieval.
  - Acceptance: graph-off-only feature, 6/20/0.025/4 frozen bounds,
    user-scoped round-robin candidates, original raw rows only, pool <=30.
  - Verify: retrieval trace/call-parity regressions; no model launch.
  - Files: `app/main.py`, `app/storage.py`, paired-evaluator tests.
- [x] Task: Run zero-model P3-B1 preflight.
  - Acceptance: fresh schema-v4 cache-reuse materialization, all anchor rows
    independently audited, >=80% raw-message coverage, zero model calls.
  - Verify: saved prepare report and independent audit output.
  - Files: `.locomo/p3-b1/` ignored artifacts, `progress.md`, `task_plan.md`.

### 2026-08-25 zero-model preflight record

- Fresh artifact: `.locomo/p3-b1/fixed20-v4.db` plus its immutable manifest
  and prepare-only report.  It replayed the existing 419-record extraction
  cache with `419/419` cache hits and `0` model calls; the local endpoint was
  not listening and the runner reports `extraction_model.constructed=false`.
- Independent read-only audit: `1,274` source-local declarations, `1,306`
  exact mention occurrences, `363/419` witnessed raw sources
  (`0.8663484486873508`), zero violations.  The de-duplicated declaration
  count is intentionally lower than the occurrence count.
- P3-A preservation check in the same v4 artifact: `3` graph edges, `3`
  support rows, and every edge independently witnessed.  No Search, QA/gold
  access, outcome metric, commit, push, or merge occurred.

### 2026-08-25 fixed-20 paired decision

- The authorized local-Qwen proxy comparison used one frozen plan per question,
  two otherwise identical Search arms, and the frozen `6/20/0.025/4` anchor
  contract. The baseline reached Hit@1 `0.40`, Hit@10 `0.55`, MRR `0.45`; the
  anchor arm reached `0.35`, `0.50`, `0.41`. All 60 model calls completed with
  zero truncations and the prepared database digest was unchanged.
- Decision: **REJECT P3-B1 as configured.** Do not tune from this 20-question
  result or promote it to fixed-200/full evaluation. This is a local proxy
  result, not a platform score.

## Approval Gate

This document specifies a new schema and retrieval channel. Implementation
begins only after the user approves this P3-B1 scope; approval does not
authorize a model launch, an outcome evaluation, a commit, or a push.
