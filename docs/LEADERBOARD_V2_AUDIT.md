# ChronoHybridMem Leaderboard V2 Audit

Date: 2026-08-15  
Scope: Agent Memory Challenge, Academic / Textual track  
Formal evaluator model: `gpt-4o-mini`

## Executive verdict

The score `Overall 44.33` is most plausibly associated with the submitted
`v0.2.0` release, whose annotated tag resolves to
`7cf45c76ea7998554a13386b924627b83aeb3134`. The submission record names
`ChronoHybridMem v0.2.0`, but the platform does not expose the deployed image
digest or checked-out Git SHA. Therefore the only defensible declaration is:

```text
LEADERBOARD_BASELINE_COMMIT=
7cf45c76ea7998554a13386b924627b83aeb3134 (UNVERIFIED)
```

The current public `main` is
`c7df0cb3f436953e450f966fbc60c22727ce806a`. Its code path includes material
retrieval and ranking changes made after the leaderboard submission. The
current local branch is based on that later code and also contains uncommitted
research changes. Neither is a valid substitute for the 44.33 baseline.

The main technical diagnosis is not “use a larger local model.” The formal
answer model is fixed, and the memory service controls only which original
evidence it puts first. The largest addressable weakness is that the formal
search path turns the question and model-generated expansions into one flat,
equally weighted OR query. This loses intent, evidence-hop, entity-role,
temporal, state, and rule structure before candidate retrieval. The first
experiment should therefore be a feature-gated structured retrieval plan that
reuses the existing query-planning call and changes no dependency or API.

## 1. Version reconstruction

| State | Revision | Evidence and status |
|---|---|---|
| Leaderboard submission | `7cf45c76…` | `v0.2.0` resolves to this commit; application commit `02f9a9f…` says `ChronoHybridMem v0.2.0` was submitted. Exact platform checkout is **UNVERIFIED**. |
| Post-submission ranking change | `4c4e0d3…` | Merge of PR #3, “Strengthen gpt-4o-mini evidence ranking rubric,” on 2026-08-11, after the 2026-08-07 submission. |
| Current public `main` | `c7df0cb…` | PR #4 documentation commit on top of `4c4e0d3…`; application code is effectively the PR #3 code. |
| Current local worktree | `11a3b00…` plus uncommitted files | Contains additional prompt, metadata, bounded-rerank, evaluator, and local-model experiments. It is not an attributable release result. |

The current local `origin/main` reference is stale, so public GitHub state—not
that local tracking ref—is used for `CURRENT_MAIN_COMMIT`.

The post-submission model workflow did not establish a score: its GitHub run
failed because `OPENAI_API_KEY` was empty. The test/build and retrieval-only
workflow passed, but this does not validate the fixed-model ranking quality.

## 2. Actual execution paths

### Competition path

```text
Add
  -> persist original raw messages (exact user_id ownership)
  -> one existing gpt-4o-mini extraction call per message
  -> index raw text and source-linked retrieval annotations

Search
  -> one existing gpt-4o-mini query-planning call
  -> lexical candidate retrieval and RRF fusion
  -> one existing gpt-4o-mini candidate-ranking call
  -> return original raw messages only
```

At the likely leaderboard baseline, extraction was factual-only, the query
planner and later RRF/context channels were absent, and fact rows themselves
could be returned. Current `main` corrects provenance by mapping annotations
back to original messages and adds Unicode/Porter raw, fact, and local-context
channels. Its entity channel is deliberately weighted zero and temporal bonus
defaults to zero.

The formal Dockerfile installs only `requirements.txt`; it does not install or
run BGE, ColBERT, Qwen, FastEmbed, GGUF files, or a second local service. With
only `OPENAI_API_KEY`, the formal path remains `gpt-4o-mini` plus SQLite FTS.

### Local research path

```text
LoCoMo adapter
  -> the same SQLite/raw-message base
  -> optional local embeddings, session fusion, cross-encoders,
     Qwen selectors/rerankers, query rewriting, and diagnostic channels
  -> retrieval-only evidence metrics
```

These local backends are useful for diagnosis, but they are disabled by
default, use `requirements-local.txt`/`Dockerfile.local`, and are not evidence
that the deployable competition path improved. The best full local result,
Hit@1 `0.5225`, is therefore a research result rather than a leaderboard
baseline or formal-model result.

## 3. Leaderboard score diagnosis

ChronoHybridMem is fifth with:

| Overall | A | B | C | D | E | G | H |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 44.33 | 55.72 | 44.44 | 23.30 | 32.94 | 57.13 | 28.31 | 35.00 |

Against ranks 1–4, the method is already strongest or near strongest on C, E,
and H, while trailing the best displayed values by 2.15 on A, 2.37 on B,
5.01 on D, and 2.46 on G. Those are not equal-priority gaps: C/E/H are
preservation gates; A and B are the safest first targets; D and G need deeper
storage semantics.

`Overall` is not the simple mean of the seven displayed capabilities. An
observational least-squares fit over the visible top 20 rows predicts the
scores reasonably well and associates most variation with A and G, but the
capability columns are correlated and the official aggregation is undisclosed.
Negative/tiny fitted coefficients for D/C are artifacts, not causal weights.
This fit is useful only to reject the assumption of equal weighting; it must
not be used as an official scoring formula.

## 4. Why the current code produces this capability profile

### A — explicit factual memory: 55.72

Strengths are raw-message preservation, lexical indexing, source-linked
annotations on current `main`, and exact speaker/date metadata in the current
worktree. The main bottleneck is query construction: raw tokens and up to 16
generated terms are flattened into an equally weighted OR expression. A weak
synonym can create many topical distractors, while the decisive entity–relation
pair receives no privileged channel. Local full-set results show both stages
matter: among 1,263 Hit@1 misses, 659 already have evidence in the top 10,
391 first recover it at ranks 11–100, and 204 miss top 100 entirely. Therefore
A is a mixed recall/reranking problem, with a low-risk recall fix available
first.

### B — multi-hop reasoning: 44.44

The service ranks a list of candidates, but retrieval has no representation of
separate evidence needs. A single broad OR query and RRF favor repeated
mentions of one popular hop. The ranking prompt on current `main` mentions
multi-step relations, yet it receives no machine-readable hop plan or coverage
state. B therefore has both candidate-recall and evidence-set-coverage failure;
the more distinctive missing mechanism is coverage across complementary hops.

### C — temporal reasoning: 23.30

This is relatively strong versus the top five. Timestamps are stored, explicit
current/historical cue patterns exist, and later ranking instructions preserve
current/previous distinctions. However the numeric temporal bonus defaults to
zero, and there is no explicit state lifecycle. C must be treated as a
non-regression gate while D is improved.

### D — memory governance: 32.94

Facts/annotations are append-only strings. There is no typed state key, no
`supersedes` edge, no retraction target, and no distinction among current,
historical, uncertain, or invalidated states. Recency can break a tie but cannot
prove that two differently worded statements update the same property. This is
consistent with the D gap, although only an ablation can establish causality.

### E — personalization: 57.13

Exact `user_id` isolation, speaker-aware raw evidence, and extracted personal
facts explain the strong result. Generic entity expansion or aggressive
deduplication can damage it by confusing the query subject with the speaker or
discarding preference changes. E is a preservation gate.

### G — rules: 28.31

The likely leaderboard Add prompt extracted only factual memory statements.
Current `main` still stores a flat list of facts rather than typed rules,
exceptions, authority, scope, and precedence. The uncommitted prompt asks for
rule-like annotations, but the schema still collapses them to strings, so it
does not solve rule retrieval or precedence. The low G score is therefore
consistent with an Add-stage representation gap plus a search/ranking gap.

### H — safety and privacy: 35.00

Exact user filtering, original-evidence return, supplied-ID validation, and
instructions not to execute candidate text are strong properties. Typed
privacy/consent boundaries may improve H later, but no A/B experiment may
weaken cross-user isolation or allow generated text to replace raw evidence.

## 5. Specific hypothesis audit

### Flat OR expansion

Current `plan_query` returns only `terms`. Search merges them with filtered raw
tokens and applies one OR match across all principal lexical indexes. This
cannot express required versus optional terms, distinct hops, entity roles, or
intent. It also makes every expansion equally strong. The defect is directly
observable in code; its score impact still requires a flag-off/flag-on test.

### Historical entity-channel regression

Commit `8a91c00…` added a weight-1 entity-bound channel by combining capitalized
query terms with content terms using `entity AND content`. Commit `38ced23…`
disabled it after full LoCoMo regression. The likely mechanism is false binding:
the person named in a question is not necessarily the evidence speaker or even
the correct premise. Cross-speaker and adversarial questions then promote the
wrong person's topical messages. Conditional gating can avoid many of these
cases only if it distinguishes high-confidence subject–attribute lookup from
false-premise, comparison, quoted, and cross-speaker intents. This is plausible,
not yet proven, and is intentionally not the first experiment.

### Reranker duplication

The current prompt requests the smallest useful evidence set and multiple
links, but its output is only an ordered ID list. It does not label which
`evidence_need` each candidate covers, penalize semantic duplicates, or verify
all-hop coverage. Repeated same-hop selection is therefore a code-supported
risk. No formal `gpt-4o-mini` trace set currently quantifies its rate, so the
claim remains unverified pending instrumentation.

### Typed sidecar

A typed sidecar can preserve raw evidence while adding `kind`, subject,
predicate, object, validity interval, polarity, confidence, authority, scope,
exceptions, and source ID. It should help D/G if used only as a retrieval aid
and always mapped back to source messages. It could still reduce C/E through
bad extraction, overconfident supersession, or speaker conflation; flag-off
temporal/personalization regression tests are mandatory.

## 6. Answers to the 12 required questions

1. **Which commit produced 44.33?** Most likely `7cf45c76…` (`v0.2.0`), but
   exact deployment identity is **UNVERIFIED**.
2. **Does current main contain unreflected improvements?** Yes. It contains
   post-submission hybrid retrieval/provenance work and PR #3's stronger
   `gpt-4o-mini` ranking rubric; no valid formal model score demonstrates their
   leaderboard effect yet.
3. **Is A mainly recall or reranking?** Both. Local rank accounting shows large
   reranking headroom inside top 10 and material recall loss beyond top 10/100.
   Structured retrieval is the lower-risk first intervention.
4. **Is B mainly recall or evidence-set coverage?** Both, with absent explicit
   evidence-set coverage the clearest mechanism unique to multi-hop queries.
5. **Does D come from missing state/supersession?** The code strongly supports
   that hypothesis, but an isolated state-index ablation is required for a
   causal claim.
6. **Does G come from factual-only extraction?** Partly and plausibly largely.
   The baseline extraction was factual-only and no typed rule/precedence index
   exists; retrieval and ranking also contribute.
7. **Why did the entity channel regress?** It hard-bound capitalized query
   entities to evidence, which is unsafe for wrong-person premises,
   cross-speaker relations, comparisons, and quoted entities.
8. **Can conditional gating avoid it?** It can reduce the known mechanism, but
   only a gated adversarial regression suite can establish a net gain.
9. **Does the current reranker repeat one hop?** The interface permits it and
   lacks coverage accounting; the formal frequency is currently unmeasured.
10. **Can structured planning plus set-aware reranking improve A/B without more
    calls?** Yes architecturally: replace the payload/contract of the existing
    planning and ranking calls. They must be evaluated as separate P1 and P2
    ablations, not bundled.
11. **Can a typed sidecar improve D/G without lowering C/E?** It is a strong
    hypothesis if source-linked and non-authoritative, but not guaranteed; state,
    temporal, speaker, and personalization preservation gates are required.
12. **Which single change is most likely to raise Overall by more than 0.7?** A
    structured query plan with tiered retrieval is the best first bet because it
    can affect A/B and route C/D/G using the existing call. “More than 0.7” is a
    leaderboard-point hypothesis, not a promise, because the official formula
    and formal evaluation are unavailable.

## 7. Evaluation and preservation protocol

The repository needs a generated, non-hidden AML-like suite with at least 30
cases each for A/B/C/D/E/G/H. Each case records memories, query, options,
required evidence IDs, forbidden evidence IDs, and category. Report Hit@1/3/5/10
and MRR; add all-evidence coverage for B, temporal/current/historical state
metrics for C/D, rule recall for G, cross-user leakage for H, duplicate ratio,
candidate-pool size, latency, call count, and token usage.

Every feature is default-off and independently selectable. The planned gates
are `MEMORY_STRUCTURED_QUERY_PLAN`, `MEMORY_INTENT_ENTITY_GATE`,
`MEMORY_MULTIHOP_COVERAGE`, `MEMORY_TYPED_SIDECAR`, `MEMORY_STATE_INDEX`,
`MEMORY_RULE_INDEX`, `MEMORY_INTENT_SESSION_EXPANSION`, and
`MEMORY_SET_AWARE_RERANK`.

A feature is `KEEP` only if its declared target improves, full retrieval does
not materially regress, C/E/H preservation gates hold, exact cross-user
leakage remains zero, original evidence/provenance remains intact, API and
formal Docker dependencies are unchanged, and latency/call counts remain
within budget. Otherwise it is `REJECT` regardless of implementation effort.

The present environment has no `OPENAI_API_KEY`, so deterministic unit and
retrieval-only tests can be run locally, but a formal `gpt-4o-mini` baseline and
ablation cannot honestly be declared complete here.

## 8. Prioritized decision

### Top 5 likely bottlenecks

1. Leaderboard/current-main/worktree version ambiguity obscures attribution.
2. Flat, equally weighted OR query expansion loses intent and creates noise.
3. Multi-hop retrieval/ranking has no explicit complementary-evidence coverage.
4. Flat factual annotations have no typed state, supersession, rule, or
   precedence semantics.
5. Entity and session signals are not safely intent-gated; unconditional forms
   have already regressed broad retrieval.

### Top 3 lowest-risk improvements

1. Structured query planning with tiered retrieval, behind
   `MEMORY_STRUCTURED_QUERY_PLAN`, reusing the existing planning call.
2. Set-aware reranking, behind `MEMORY_SET_AWARE_RERANK`, reusing the existing
   ranking call and consuming P1 evidence needs.
3. Conditional entity binding, behind `MEMORY_INTENT_ENTITY_GATE`, enabled only
   for high-confidence subject–attribute intent.

### Top 3 highest-upside improvements

1. Source-linked typed sidecar annotations using the existing Add call.
2. Explicit current/historical/retracted state and supersession indexing.
3. Dedicated rule/exception/authority/scope/precedence retrieval.

### Features that must NOT be changed

- Original raw evidence must remain the returned object; generated annotations
  are never evidence substitutes.
- Exact `user_id` isolation and zero cross-user leakage.
- Public Add/Search API and supplied-candidate-ID validation.
- Formal `gpt-4o-mini` model choice and the existing call-count budget.
- Formal Docker dependency footprint unless a later deployability review proves
  a change is allowed and necessary.
- Existing C/E/H strengths without explicit non-regression evidence.
- Disabled local-model features must not silently enter the competition path.

### First experiment to run

Implement only P1: a default-off structured plan returned by the existing
query-planning call (`intent`, `core_terms`, `expansion_terms`, `entities`,
`temporal_cues`, and `evidence_needs`). Use core terms in the principal lexical
channels and route expansion/evidence-need terms through lower-weight channels.
Do not add a model call, change Add, add typed storage, re-enable entity binding,
or implement set-aware ranking in this experiment. Compare flag off versus on
on deterministic synthetic regressions and the same external retrieval set;
run formal `gpt-4o-mini` baseline/ablation when a secret is available. Then make
an explicit `KEEP` or `REJECT` decision before starting P2.

## 9. P1 implementation status

P1 is implemented default-off. The structured contract is bounded and routes
core/entity/temporal terms to the existing primary retrieval channels, while
optional expansions and evidence needs enter lower-weight Porter raw/fact
channels. A fixture-plan sweep rejected the initial `0.35` support weight:
Hit@1 fell from the flat arm's `0.357143` to `0.214286`, and forbidden@1 rose
from `0.357143` to `0.642857`. Weights from `0` through `0.02` instead reached Hit@1
`0.785714`, MRR `0.892857`, and forbidden@1 `0.071429`; `0.05` and above
reproduced the regression. The retained P1 weight is therefore the conservative
`0.01`: a focused near-neighbor regression test showed that `0.02` could still
flip a primary-evidence lead. This isolates core intent while allowing only a recall-level support
signal. The ranking prompt and Add schema were not changed as part of P1.

The repository contains 210 generated AML-like cases (30 per category). The
no-model preservation baseline is Hit@1/3/5/10 `0.785714/1/1/1`, MRR
`0.892857`, B all-evidence coverage@10 `1`, and cross-user leakage `0`.
This is deliberately not treated as an official-score proxy. A deterministic
flag-off/flag-on test verifies the intended retrieval effect and exactly one
planning call. The full suite and compilation pass with 94 tests.

Decision: **PENDING**, not `KEEP`. The retrieval mechanics pass the fixture-plan
ablation at weight `0.01`, but same-suite `gpt-4o-mini` baseline and P1
arms cannot run because `OPENAI_API_KEY` is not configured. Docker CLI is also
unavailable locally. P2 must not start until the formal arms establish the
target gain and C/E/H preservation.

The automated gate records the mechanical comparison as `MECHANICS_PASS`:
Hit@1 delta `+0.428571`, A/B deltas `+1/+1`, C/E/H deltas `+1/0/0`,
forbidden@1 delta `-0.285714`, mean Search latency ratio `1.167183`, and GPT-call
delta `0`. The decision label is intentionally not `ADVANCE` or `KEEP`, because
annotated fixture plans do not measure `gpt-4o-mini` planning accuracy.
