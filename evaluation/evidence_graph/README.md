# P3 evidence-graph diagnostics

This directory contains offline diagnostic and evaluation runners. The metric
and P3-0 fixture modules remain independent of the production memory service.
The paired LoCoMo runner deliberately imports the production store so cached
composite extractions are replayed through the exact Add/Search implementation.
No LoCoMo questions, answers, caches, or derived traces are checked in.

`generate_cases.py` deterministically creates 140 synthetic P3-0 cases (10 in
each category): person-person, person-location, person-organization,
preference, employment, membership, rules, temporal update, parallel temporal
states, correction, retraction, same-name isolation, cross-user isolation, and
adversarial memory text. Every gold relation uses the production controlled
predicate vocabulary and points to an original `source_message_id`.

Generate or verify the checked-in fixture:

```powershell
python -m evaluation.evidence_graph.generate_cases
```

Run model extraction against a loopback proxy (exactly one call per message):

```powershell
python -m evaluation.evidence_graph.run_extraction `
  --base-url http://127.0.0.1:8081/v1 `
  --model local `
  --max-cases 20 `
  --output .eval-tmp/p3-0-local-20.json
```

Omit `--base-url` to use the formal model named by `--model`; that path requires
`OPENAI_API_KEY`. The output caches raw per-message extraction, the sanitized
graph, per-case metrics, category metrics, latency, and call counts.

`metrics.py` contains pure extraction and retrieval metrics. Extraction
metrics deliberately separate semantic edge precision, predicate mapping,
source provenance, temporal state, entity-link clustering, unsupported edges,
and cross-user leakage. Retrieval metrics implement strict all-gold Chain
Recall@K, fractional Evidence Coverage@K, Bridge Recall@K, and graph-only
recovered source evidence.

Entity-link mentions use an accepted-endpoint policy: a mention is one
user/source/entity occurrence referenced by a relation that survives the
fail-closed graph parser. Dangling extracted entities remain part of entity
precision, but quoted, injected, or otherwise rejected relation endpoints do
not enter link clustering. Prediction occurrence IDs are aligned to gold only
for a unique user/source/canonical-name match, so ontology typing is not charged
twice. Cluster identity remains entirely prediction-derived, case-scoped, and
type/identity-hint aware, so a same-name false merge still creates false pairs.

An existing extraction report can be deterministically reparsed with
`--rescore-report`. This path reads each cached `raw_extraction`, joins it to the
authoritative fixture message by case/source/user ID, and makes zero model calls;
it is an audit aid, not a replacement for an independent model-backed rerun.

Retrieval metrics reject empty gold lists and non-positive K values so an
invalid/no-answer case cannot silently inflate an aggregate.

## Fair paired P3-A LoCoMo evaluation

`run_locomo_paired.py` prevents an Add-time confound by separating the run into
three explicit stages:

1. hash the authoritative dataset and select the same full-conversation scope
   used by the legacy evaluator;
2. call the composite Fact+Graph extractor exactly once per selected message,
   cache every raw payload, and replay those payloads through session-bundled
   production `MemoryStore.add` calls into one prepared SQLite database;
3. freeze one structured plan per question and search that same prepared DB
   with graph off and graph on, scoring returned IDs through a strict
   `dia_id <-> mem_id` map.

Session-bundled Add is intentional. The store still calls `extract_memory`
once for each message, while preserving the existing previous/current/next
context index. Splitting a session into one Add request per message would
silently change the P1 baseline.

Use two explicit processes for a fixed-20 screen. First configure the local
runtime as 4 slots x 2048 context and run extraction/materialization only:

The cache identity must include the SHA-256 of the exact ingest artifact, not
just the server's generic `local` model name. For the currently verified Qwen
GGUF that digest is
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
Recompute it with `Get-FileHash <model.gguf> -Algorithm SHA256` whenever the
GGUF or quantization changes. The CLI requires explicit loopback ingest/search
base URLs and fails rather than falling back to an external OpenAI endpoint.

```powershell
python -m evaluation.evidence_graph.run_locomo_paired `
  --dataset C:\path\to\locomo10.json `
  --max-questions 20 `
  --cache-mode build `
  --prepared-mode build `
  --extraction-cache .locomo\p3-a\composite-extractions.json `
  --prepared-db .locomo\p3-a\fixed20-v3.db `
  --ingest-base-url http://127.0.0.1:8081/v1 `
  --ingest-model local `
  --ingest-artifact-fingerprint 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 `
  --workers 4 `
  --prepare-only `
  --output .locomo\p3-a\fixed20-v3-prepare.json
```

`--prepare-only` writes the dataset, extraction-cache, and prepared-database
manifests with `stage: "prepare_only"`, then returns before constructing a
search model, freezing a plan, or calling Search. Stop and manually reconfigure
the runtime as 1 slot x 8192 context; the runner deliberately does not restart
or reconfigure it. Then evaluate by reusing both artifacts. Keep the ingest
model/base URL identical because those values are part of the cache
fingerprint, and pass the identical artifact SHA, even though this stage makes
no ingest calls:

Both stage reports include model diagnostics (`call_count`,
`finish_reason_counts`, and `truncated_calls`). The prepare report contains only
the ingest-model diagnostics; the evaluation report additionally records the
search model after plan freezing and both Search arms. A completed paired report
hard-checks exactly `N` plan calls, `2N` arm calls (one baseline rank plus one
graph-on rank per question), `3N` total search-model calls, and zero truncated
responses. Any mismatch aborts before the result JSON is written.

```powershell
python -m evaluation.evidence_graph.run_locomo_paired `
  --dataset C:\path\to\locomo10.json `
  --max-questions 20 `
  --cache-mode reuse `
  --prepared-mode reuse `
  --extraction-cache .locomo\p3-a\composite-extractions.json `
  --prepared-db .locomo\p3-a\fixed20-v3.db `
  --ingest-base-url http://127.0.0.1:8081/v1 `
  --ingest-model local `
  --ingest-artifact-fingerprint 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 `
  --search-base-url http://127.0.0.1:8081/v1 `
  --search-model local `
  --graph-rrf-weight 0.025 `
  --graph-rerank-quota 4 `
  --graph-max-candidates 20 `
  --output .locomo\p3-a\fixed20-v3-paired.json
```

The output path must be new. The runner refuses to overwrite an earlier
result. Cache fingerprints bind the raw extraction payloads to the dataset,
model configuration, and checked-out extraction contract. `reuse` requires a
complete matching cache and makes zero Add-time model requests. `extend`
reuses existing records and calls the extractor only for newly selected
messages, which is the intended fixed-20 to fixed-200 path:

Extraction completion is checkpointed atomically by the main process at least
once per `--workers` successful payloads and again before propagating a model
failure. A failed build therefore leaves a valid fingerprinted partial cache;
rerun with `--cache-mode extend` to pay only for the remaining messages.

```powershell
python -m evaluation.evidence_graph.run_locomo_paired `
  --dataset C:\path\to\locomo10.json `
  --max-questions 200 `
  --cache-mode extend `
  --prepared-mode build `
  --extraction-cache .locomo\p3-a\composite-extractions.json `
  --prepared-db .locomo\p3-a\fixed200-v3.db `
  --ingest-base-url http://127.0.0.1:8081/v1 `
  --ingest-model local `
  --ingest-artifact-fingerprint 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 `
  --workers 4 `
  --prepare-only `
  --output .locomo\p3-a\fixed200-v3-prepare.json
```

After the same manual 1-slot x 8192 runtime switch, run:

```powershell
python -m evaluation.evidence_graph.run_locomo_paired `
  --dataset C:\path\to\locomo10.json `
  --max-questions 200 `
  --cache-mode reuse `
  --prepared-mode reuse `
  --extraction-cache .locomo\p3-a\composite-extractions.json `
  --prepared-db .locomo\p3-a\fixed200-v3.db `
  --ingest-base-url http://127.0.0.1:8081/v1 `
  --ingest-model local `
  --ingest-artifact-fingerprint 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 `
  --search-base-url http://127.0.0.1:8081/v1 `
  --search-model local `
  --graph-rrf-weight 0.025 `
  --graph-rerank-quota 4 `
  --graph-max-candidates 20 `
  --output .locomo\p3-a\fixed200-v3-paired.json
```

Prepared databases use schema v3: every traversable `graph_edges` row must
have exactly one composite-FK-bound `graph_edge_support` witness row. The
sidecar manifest contains the dataset/scope hash, logical SQLite digest, exact
source map, and a materialization-contract hash of `app/storage.py`,
`app/evidence_graph.py`, `app/graph_routing.py`, and
`evaluation/evidence_graph/support_witness_audit.py`. Use
`--prepared-mode reuse` only with the identical question scope and checked-out
materialization code; legacy or mismatched sidecars are rejected. Missing cache
records, duplicate source keys, fingerprint mismatches, source-order
mismatches, cross-user IDs, graph
paths without persisted provenance, or any Search-time database mutation abort
the run.

Before a `MemoryStore` is created or either Search arm begins, the evaluator-owned
`support_witness_audit.py` independently verifies every persisted graph edge.
It does not import or trust the production boolean verifier: it recomputes the
NFKC/casefold/whitespace canonical source, validates the sidecar's composite
FK and one-to-one binding, spans, hash, frozen frame/binding/state, and the
full-source scope. Any invalid persisted witness aborts the run before Search.
The later trace audit reloads the authoritative sidecar for every traversed
path; `unsupported_traversed_edges == 0` is therefore necessary but not
sufficient. Each arm also requires unique rerank-pool IDs, unique pool content
under the store's exact `casefold()` deduplication rule, exact agreement
between traced and returned final IDs, and byte-for-text equality between every
returned result and its `raw_messages.content` row.

The formal CLI fixes the graph mechanics exactly at RRF weight `0.025`, rerank
quota `4`, and candidate cap `20`; a different value is rejected before Search.
It also requires the production store to expose a complete pre-rerank trace:
P1 channels and counterfactual top-30, Graph candidates,
requested/resolved/unresolved seed accounting, reserved Graph IDs, final rerank
pool/final IDs, source paths, and edge diagnostics. The latter prove candidate
limit/count, per-seed fetch limits, and the consumed-edge round-robin order.
Until those fields are available, it fails rather than emitting a misleading
paired score. The `--allow-incomplete-storage-trace` switch exists only for
offline fake-model or mechanics diagnostics; such reports set
`storage_trace_complete` to `false` and cannot be used for the P3-A promotion
gate.

Dense score fusion is explicitly rejected in this runner because the current
production dense-fusion branch returns before evidence-graph fusion. P3-A uses
the cleaned P1 RRF protocol. The report records ID-based Hit@K, evidence
Recall@K, MRR, per-question graph-off/on traces, graph-only recovered and lost
gold evidence, Search call parity, and the database digest before and after
both arms.

After the persisted-witness audit and frozen-config validation, but before
either Search arm starts, the runner freezes the outcome-independent
direct-relation subset. Membership uses only valid plan entities plus the
controlled predicates implied by the question/plan; prepared-graph resolution
and gold adjacency are retained only as diagnostics, never as membership
selectors. The final JSON includes both
`relation_subset_manifest` (including `subset_manifest_sha256`) and `p3a_gate`.
The gate's formal scope is exactly 200 questions: a fixed-20 screen can at most
return `MECHANICS_PASS`, while fixed-200 is evaluated against the formal P3-A
promotion criteria. Gate evaluation requires `--top-k` to include both 1 and
10 (the default `1,3,10` does).
