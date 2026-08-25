# ChronoHybridMem implementation plan

## 2026-08-19 P3 Evidence Graph execution

- [x] Phase 1 — audit current `main`, P1 code, tests, model-call budget, candidate pools, and evaluation protocol.
- [ ] Phase 2 — reproduce the current P1 baseline in the clean P3 worktree. **IN PROGRESS** (fixed-20 reproduced; fixed-200 rerun remains)
- [x] Phase 3 — implement and gate P3-0 extraction diagnostics (provenance 1.00, leakage 0, unsupported edges 0, relation precision/recall >=0.95, every category recall >=0.90, entity-link precision/recall >=0.98, parse/truncation failures 0), including a graph-capable cached Add-ingestion protocol distinct from the historical Search-only P1 evaluator. **COMPLETE**
- [x] Phase 4 — implement and gate conservative P3-A one-hop retrieval with bounded graph quota/weight ablations. **SAFETY COMPLETE / COVERAGE REJECT**: the fresh schema-v3 zero-model rebuild has 3 independently witnessed edges across 419 messages and 1,052 entities, so it is not eligible for a paired score run.
- [x] Phase 5 — implement and validate P1.1 bounded same-session neighbor expansion. **MECHANICS COMPLETE**: default-off, frozen 4/4 bounds, raw-only session isolation, trace and call-parity gates pass. A local outcome evaluation remains separately unauthorized.
- [x] Phase 5B — implement P3-B1 provenance-anchored entity mentions. **REJECTED AFTER FIXED-20**: source-local schema-v4 index and independent audit pass (363/419 sources, 86.63%), but the frozen paired local-Qwen test fell from Hit@1 `0.40` to `0.35` and MRR `0.45` to `0.41` (Hit@10 `0.55` to `0.50`). Do not tune or promote P3-B1 to fixed-200/full; keep P1.1 isolated.
- [ ] Phase 6 — implement P3-C explicit temporal state only if A or B passes; require zero false supersedes.
- [ ] Phase 7 — compare passing configurations on fixed 200; run paired full 1,977 only if the predeclared promotion gate passes.
- [ ] Phase 8 — document ACCEPT/CONDITIONAL/REJECT, commit stable stages, push `agent/p3-evidence-graph`, and report CI.

Hard constraints: graph output always resolves to original `source_message_id`; exact `user_id` isolation; no generated benchmark answer; zero added Search LLM calls; feature-flag-off reproduces P1; no automatic merge to `main`.

## 2026-08-15 leaderboard V2 audit and controlled ablation

- [x] Reconstruct the likely 44.33 leaderboard baseline and label the exact deployment SHA `UNVERIFIED`.
- [x] Separate the v0.2.0 competition path, current public main, current worktree, and local research path.
- [x] Audit A/B/C/D/E/G/H mechanisms and publish `docs/LEADERBOARD_V2_AUDIT.md`.
- [x] Build the non-hidden AML-like seven-category synthetic regression harness.
- [x] Implement only P1 structured query planning behind `MEMORY_STRUCTURED_QUERY_PLAN`.
- [ ] Run flag-off baseline, flag-on ablation, preservation, latency, and call-count checks (fixture mechanics gate passes; formal model arms require `OPENAI_API_KEY`).
- [ ] Decide `KEEP` or `REJECT` before implementing P2.

## 2026-08-15 leaderboard-targeted iteration

- [x] Read the live Academic / Textual leaderboard and compare ChronoHybridMem with ranks 1-4.
- [x] Stop answer-model scaling; the formal path is fixed to `gpt-4o-mini`.
- [x] Supply speaker, event date, extracted annotations, and adjacent context to evidence ranking.
- [x] Strengthen update/retraction, rule/process, multi-hop, and evidence/privacy ranking rules.
- [x] Run unit, compile, and diff validation (89 passed).
- [ ] Run a formal `gpt-4o-mini` Smoke/Full evaluation and verify D/G/B/A plus overall score.
- [ ] Verify strict Hit@1 >= 0.70 on all 1,977 local evidence questions.

The previous strict full-set model-enhanced best was 0.5225. The current local Qwen3-4B P1 proxy is 0.5761/0.7157/0.7618/0.6479 at Hit@1/3/10/MRR. The new no-model base is 0.3612/0.5468/0.6990/0.8968 at Hit@1/3/10/100. `OPENAI_API_KEY` is not configured in this environment, so the P1 result is a local proxy and must not be reported as an official or 0.70 result.

## Goal

Deliver the v0.1.0 Docker-ready evidence retrieval baseline described in `docs/SPEC.md`.

- [x] Create independent local Git repository and approved implementation specification.
- [x] Implement schemas, SQLite persistence and FTS retrieval.
- [x] Implement FastAPI routes and configuration.
- [x] Add contract, isolation and idempotency tests.
- [x] Add Docker packaging, README and verify the full build/test path (GitHub Actions passed pytest and Docker build on 2026-08-07).
- [x] Add release metadata, MIT license and v0.1.0 tag for competition submission (completed).
- [x] Add offline retrieval-evaluation harness and documentation (GitHub Actions passed on 2026-08-07).
- [x] Integrate `gpt-4o-mini` fact extraction and evidence reranking for academic-track compliance (GitHub Actions passed on 2026-08-07).
- [x] Submit the `v0.2.0` academic Docker-deployment application to Agent Memory Challenge (submitted on 2026-08-07).
- [x] Improve hybrid retrieval with RRF fusion, evidence preservation, and explicit temporal ranking (GitHub Actions run 31170561376 passed tests and Docker build on 2026-08-07).
- [x] Extend temporal retrieval to explicit historical queries and re-measure against v0.2.0 (GitHub Actions run 31172617528 passed tests and Docker build on 2026-08-07).
- [x] Add a local-only LoCoMo evidence-retrieval evaluator and run a public-data retrieval baseline. The dataset and its detailed results are not stored in this repository.
- [x] Compare current retrieval against v0.2.0 over the same full LoCoMo evidence set; broad retrieval regression detected and retained only as an external run result.
- [x] Calibrate temporal ranking against the full LoCoMo comparison; disable it by default because it did not demonstrate a general gain.
- [x] Evaluate content-term query normalization against the full LoCoMo comparison; retain it after a positive external retrieval result. Detailed third-party evaluation output is not stored in this repository.
- [ ] Add `gpt-4o-mini` query-term planning and verify its API contract; model-backed external evaluation requires a runtime secret (in progress).
- [x] Add neighbor-context indexing as a lower-weight RRF channel and retain it after a positive full LoCoMo result.
- [x] Add parallel Porter-tokenized FTS channels and retain them after a positive full LoCoMo result.
- [x] Evaluate an entity-bound structured retrieval channel on full LoCoMo; disable it by default after detecting an overall regression.
- [ ] Run a bounded 100-question Search-only `gpt-4o-mini` LoCoMo ablation; requires the GitHub `OPENAI_API_KEY` secret (in progress).
- [x] Build a local-model experimentation backend that preserves the existing Add/Search API and can later be mapped to `gpt-4o-mini`.
- [x] Reproduce the current no-model full LoCoMo baseline before changing ranking.
- [x] Add local dense retrieval and lexical-dense score fusion without committing evaluation data or model weights.
- [x] Run bounded ablations, then a full LoCoMo evaluation; retain only changes that improve Hit@1.
- [ ] Continue method iterations toward full-set Hit@1 >= 0.70, recording limitations and avoiding evaluation-data leakage.
- [x] Preserve LoCoMo's original session boundaries and implement an optional paper-derived session-to-turn fusion signal.
- [x] Combine the positive session signal with bounded turn reranking; reject it from the default path because fixed 200-question Hit@1 did not exceed 0.440.
- [x] Replace hard ColBERT ordering with calibrated score fusion; reject it because fixed 200-question Hit@1 did not exceed hard reranking's 0.440.
- [ ] Add a constrained local instruction-model selector over existing candidate IDs and test whether top-10 oracle coverage can be converted into Hit@1.
- [x] Validate the constrained local selector protocol and select a top-3 candidate pool on a fixed 20-question smoke test.
- [x] Verify and reject two-pass Qwen top-10 then top-3 refinement: it tied the single-pass Hit@1 0.55.
- [x] Verify API tests and Docker build in GitHub Actions for the v0.3.0 milestone.
- [x] Publish the verified Hit@1 0.4355 local-method milestone as isolated draft PR #1 without moving v0.2.0.
- [x] Gate and reject Qwen3-8B Q4: Hit@1 tied 0.55, MRR fell to 0.6292, latency rose to 333 seconds.
- [x] Implement and gate a leakage-free context-augmented dense key; it improved Hit@10 but not Hit@1.
- [x] Test and reject context weight 0.5 with ColBERT pools 5/10/20; best Hit@1 remained below 0.440.
- [x] Gate and reject BAAI/bge-reranker-base on 20 questions: Hit@1 0.40.
- [x] Check E5 support and reject the available multilingual variant at Hit@1 0.50; e5-large-v2 is not native.
- [x] Audit speaker identity: it is already embedded in each turn; no change needed.
- [x] Preserve timestamps and gate time-aware dense keys; weight 0.5 gained +0.005 Hit@1 on 200 questions.
- [x] Pair baseline vs time weight 0.5 on 500 questions; gain persisted at Hit@1 0.448 vs 0.444.
- [x] Run full time-aware validation and reject it: Hit@1 0.4345 < 0.4355.
- [x] Gate and reject Qwen3-4B Chain-of-Note: Hit@1 0.50 < 0.55.
- [x] Implement and reject hard top-session filtering: Top-2/3 tied Hit@1 0.440, Top-1 fell to 0.390.
- [x] Implement and reject local query rewriting: Hit@1 0.50 < 0.55.
- [x] Re-run the Qwen3-Reranker-4B Q3_K_M 20-question gate after fixing yes/no token-variant normalization; valid Hit@1=0.60, advance to fixed 200 questions.
- [x] Run the Qwen3-Reranker-4B Q3_K_M fixed-200 gate; Hit@1=0.485 > 0.440, advance unchanged to full validation.
- [x] Run full 1,977-question Qwen3-Reranker-4B Q3_K_M validation; Hit@1=0.5195 > 0.4355.
- [ ] Calibrate first-stage/Qwen score fusion on the fixed-200 set; advance only a better weight to full validation.
- [x] Calibrate first-stage/Qwen score fusion on the fixed-200 set; reject all tested weights (best 0.470 < 0.485 hard Qwen).
- [x] Gate Qwen hard reranking with the previously recall-positive time-aware dense key (weight 0.5) on fixed 200 questions; Hit@1=0.505 > 0.485, advance to full.
- [x] Run full Qwen + time-aware-key validation; Hit@1=0.5225 > 0.5195 (historical v0.4 local best).
- [x] Gate Qwen hard reranking with the previously high-recall context-aware dense key (weight 0.5) on fixed 200 questions; tie Hit@1=0.505, reject alone.
- [x] Gate the combined time-aware + context-aware dense keys with hard Qwen on fixed 200 questions; reject Hit@1=0.500.
- [ ] Select a new non-overlapping retrieval mechanism and gate it against the current P1 local proxy best Hit@1=0.5761.
- [x] Publish the verified v0.4 local research milestone to an isolated GitHub branch and draft PR.
- [ ] Diagnose the remaining exact-turn errors from the current P1 local proxy full-set run (Hit@1=0.5761) and identify a leakage-free mechanism with enough oracle headroom (in progress).
- [ ] Gate a conflict-routed cascade: retain the dedicated 4B pairwise first choice by default and invoke comparative top-1 verification only on likely wrong-person premises.
- [x] Gate and reject conflict-routed and conservative-verifier cascades; neither preserves normal-query Hit@1 while improving false-premise selection.
- [ ] Add sentence-level latent retrieval keys that aggregate back to the original evidence turn, and gate candidate recall before reranking.
- [x] Gate and reject sentence-level latent dense keys; candidate coverage improved by at most one fixed-200 question.
- [ ] Measure dedicated-reranker score saturation, ties, and evidence score rank; test a principled tie-breaking or score-calibration change if diagnosed.
- [x] Diagnose and gate near-tie calibration; reject it after fixed-200 Hit@1 0.500 failed the 0.505 advancement threshold.
- [ ] Gate generic Qwen3-8B comparative-top1 on adversarial and open-domain slices; advance only if it materially exceeds generic 4B listwise selection.
- [x] Gate and reject unconditional generic 8B comparative-top1; gains were only +1/20 per hard slice at prohibitive latency.
- [ ] Measure a speaker-mismatch router against fixed-200 categories, then invoke comparative selection only if the router isolates hard wrong-person premises with useful precision.
- [x] Gate and reject Jina turbo cross-encoding after fixed-200 Hit@1 0.370 failed to generalize.
- [ ] Quantify Qwen/Jina/first-stage expert complementarity and the observable routing ceiling before implementing another cascade.
- [ ] Generate a shared fixed-50 Qwen bitmap and stop the expert-routing path unless first-stage∨Qwen oracle reaches at least 0.70.
- [x] Reject expert routing after first-stage∨Qwen fixed-50 oracle reached only 0.560.
- [ ] Add speaker-bound first-person coreference latent keys and gate fixed-200 candidate recall before reranking.
- [x] Gate official asymmetric Qwen3 embeddings at 0.6B and 4B; reject both after fixed-20 candidate recall regressed.
- [ ] Audit fixed-200 candidate/evidence error structure and gate a stronger joint candidate reader that can reason over speaker identity and temporal relations.
- [x] Gate structured 8B joint reading and speaker-swapped latent queries; reject unconditional expansion after fixed-50 Hit@1 reached only 0.52.
- [ ] Measure constraint/comparative strategy complementarity and implement an observable arbiter only if the oracle ceiling exceeds 0.70.
- [ ] Gate each new mechanism on the fixed 200-question subset, then run full 1,977-question validation only for a material gain.
- [ ] Continue bounded iterations until strict full-set Hit@1 >= 0.70, or document a genuine external blocker after exhausting safe local paths.

## Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Unicode delimiter intermediate scan treated terminal possessive apostrophes (`students'` / `students’`) inside an outer single-quoted scope as the quote closer, allowing later quoted content to escape | 1 | The parser sub-audit rejected the intermediate snapshot. Refine lexical apostrophe classification for plural/terminal possessives while keeping true quote closers and contractions distinct; rerun independent Unicode fuzz before freezing. |
| The prior final independent parser-audit agent stopped because of an agent usage-limit interruption before issuing a verdict | 1 | This is not a code failure or an ACCEPT. Start a new independent read-only audit of the exact final SHA before enabling P3-A materialization or Search. |
| Initial witness-carrier SHA `7CBC28D1...` dropped source-backed relations when the parser retained an ordinary relation from conflicting non-assert state witnesses | 1 | Independent audit REJECTED the snapshot. Preserve the historical parser output with `support_witness=None` in such conflicts; storage must fail closed and not materialize that edge. Add regression coverage for historical/correction and re-audit a new exact SHA. |
| First storage/evaluator combined suite found `test_support_sidecar_remaps_legacy_witness_offsets_to_v1_codepoints` had no persisted support row | 1 | Treat as a real Unicode-normalization integration failure, not a test relaxation opportunity. Storage owner must diagnose/remap safely, then evaluator reruns independent combined verification. |
| Root review found evaluator semantic validation initially checked only the stored clause, allowing a tampered support row to potentially cite an earlier assertion while later source-level withdrawal/quote/meta context invalidated it | 1 | Add evaluator-owned, fail-closed full-source structural scope audit with adversarial tamper regressions; do not import or call production parser support logic. Re-run combined hard-gate tests before any rematerialization. |
| Read-only P3-A gate audit found formal config was CLI-tunable, direct paired evaluation could defer an all-edge witness failure until after Search, and trace diversity/cap mechanics were not enforced by the formal gate | 1 | Add explicit formal config enforcement, pre-Search all-edge audit with zero-call tamper test, and trace candidate-cap/per-seed accounting validation before schema-v3 rematerialization. Keep diagnostics clearly non-formal. |
| Token-scaffold SHA `E82F62CD...` enumerated quote/bracket characters and missed Unicode paired containers (`「」`, `『』`, `‹›`, `《》`, `【】`, `⟦⟧`), allowing a middle quoted sentence to escape | 1 | Independent review rejected the SHA. Use Unicode punctuation categories `Pi/Ps` and `Pf/Pe` for fail-closed container scopes, preserving only word-internal apostrophe/contraction exceptions. |
| Marker-span SHA `EC37A5B3...` still derived marker clauses from a finite separator list, so em-dash/colon/`yet`/`while` adjacency caused the marker mask to swallow independent withdrawal text | 1 | Independent review rejected the SHA. Derive the minimal absolute token span of the marker scaffold itself and scan all text outside that span, independent of punctuation or conjunction separators. |
| Initial-subject SHA `CF7BE348...` marked an entire forward-marker sentence governed, so an independent negation/retraction elsewhere in that same sentence could not withdraw an earlier edge | 1 | Independent review rejected the SHA. Track/erase only the forward-marker clause span during backward-withdrawal analysis; the governed target sentence remains fully excluded, while marker-external withdrawal material still applies. |
| Initial-sequence SHA `04A61081...` had a directional parser bug: the sentence-initial article guard rejected explicit nonmatching subjects `A-X`, `A X`, `A.X`, `A, X` before identity comparison | 1 | Independent review rejected the SHA. Recognize two-or-more separated uppercase single-letter initials before applying the `A` article guard; ordinary `A company/person` remains non-explicit. |
| Ordered-acronym SHA `7DAB543B...` normalized only one-token abbreviations and missed spaced/punctuated initial sequences (`A S`, `I B M`) in the reverse identity direction | 1 | Independent review rejected the SHA. Collapse two-or-more single-character name tokens into an abbreviation representation and apply the same bidirectional ordered-initial comparison; retain `A X` and single-character controls. |
| Acronym SHA `02CEDEC0...` required exact initials and missed conventional acronyms that omit middle names, stopwords, or legal suffixes (`ABS→AS`, `UOC→UC`, `UOCLA→UCLA`, `IBMC→IBM`) | 1 | Independent review rejected the SHA. Treat a length>=2 single-token acronym as overlapping when it is an ordered subsequence of the other name's initial sequence; retain nonmatching and out-of-order controls. |
| Identity-overlap SHA `B2050036...` did not recognize compact/acronym references (`AS`, `MJW`, `IBM`, `AliceSmith`) as possible aliases of a multi-token candidate name | 1 | Independent review rejected the SHA. Add bidirectional normalized acronym and compact-name equality to the overlap contract while retaining boundary-safe disjoint-name positives. |
| Governed-sentence SHA `1E4B4F17...` treated a partial first name or surname (`Alice` / `Smith`) as an explicitly different subject from `Alice Smith`, so the shortened-name denial did not retract the earlier edge | 1 | Independent review rejected the SHA. A different-subject exemption now requires normalized name tokens (and non-space-script substrings) to be disjoint; any partial/alias-like overlap remains ambiguous and fail-closed. |
| Structural-scan SHA `23DDB9CA...` skipped a forward-meta marker but not the sentence it governed, so an untrusted negative target sentence incorrectly retracted an earlier true relation | 1 | Independent review rejected the SHA on the required preservation boundary. Precompute governed sentence spans; marker and target sentences may neither produce witnesses nor participate in backward withdrawal scans. |
| Container SHA `863B117E...` still attributed cross-sentence withdrawals/meta/reporting only when a sentence began with enumerated leads, so arbitrary discourse adverbs, filler sentences, and paraphrased attribution bypassed it | 1 | Independent review rejected the SHA. Remove lead-dependent attribution: scan all later top-level sentences for any negation/retraction that cannot be proven to have a different explicit subject, and block a candidate when any prior sentence contains token-class meta/directive/nonfactual/reporting markers. |
| Scope-fixed SHA `D9D56EC0...` still relied on enumerated withdrawal/meta regexes and tracked only double quotes, so adverb-inserted denials, smart/ASCII single quotes, guillemets, reporting parentheses, and meta paraphrases bypassed the gate | 1 | Independent review rejected the SHA. Replace subclause acceptance with top-level sentence/container full-match, a lexical delimiter stack that distinguishes word-internal apostrophes, structural adjacent-withdrawal detection, and token-class meta governance; do not continue expanding phrase blacklists. |
| Morphology-fixed SHA `9C135B2D...` lost source scope when splitting clauses: post-assertion negation, a previous-sentence hypothetical/ignore marker, and multi-sentence reported quotes could all leave a false positive witness | 1 | Independent review rejected the SHA. Move quote/meta/contradiction scope tracking to normalized full-source absolute spans before clause validation; retain dedicated correction/retraction wrappers and ensure a later untrusted segment does not erase an earlier independent true relation. |
| Frozen unified-grammar SHA `6E0EF21D...` accepted first-person agreement errors such as `I works`, `I lives`, and `I written` across controlled predicates | 1 | Independent review rejected the SHA despite green fixture/full tests. Add predicate-frame-specific first-person morphology to the same positive grammar, cover all controlled predicates with valid/invalid pairs, then freeze a new SHA and rerun every gate from scratch. |
| A read-only PowerShell artifact check piped directly after a `foreach` block and failed with `An empty pipe element is not allowed` | 1 | No state changed. Accumulate objects into an array first, then format the completed array; do not repeat the invalid block-pipeline form. |
| The parser sub-agent created local commit `613fd7d` before the method-level P3-A gate decision, contrary to the working convention to defer commits | 1 | Nothing was pushed or merged. Preserve the local commit and avoid destructive history rewrites; continue independent review and add subsequent work normally, reporting the local-only commit state before user handoff. |
| A read-only `rg` alternation was double-quoted through PowerShell and reached ripgrep with an unclosed group | 1 | No file changed. Re-ran the symbol lookup as a plain alternation without embedded quoted literals. |
| Independent parser review read a concurrently edited intermediate file and hit `NameError: _STRICT_INTERROGATIVE_PREFIX_PATTERN is not defined` | 1 | No benchmark/model call or artifact was accepted from that transient SHA. Paused the reviewer until the parser owner reports a stable SHA, then rerun the entire adversarial matrix and 419-record replay from scratch. |
| A read-only SQLite identity-hint query used backslash-escaped quotes that PowerShell does not honor | 1 | No state changed. Remove SQL string literals/COALESCE from the one-liner and use a single-quoted Python SQL string inside PowerShell's double-quoted `-c` argument. |
| `Get-NetTCPConnection` was access-denied while resolving the exact loopback llama-server PID | 1 | Use read-only `netstat -ano` scoped to `127.0.0.1:8081`, then validate the resulting PID's process name and executable path before stopping only that process. |
| Fixed-20 LoCoMo prepare at 4x2048 produced a server-side `truncated=1` on a long composite extraction | 1 | Stopped only evaluation session `82088`; 28 successful records remain atomically cached and no DB/report was created. Reconfigure the unchanged model to 2x4096 and resume with `cache-mode extend`; reject the 2048-slot run as invalid. |
| PowerShell rejected piping directly after a `foreach` block during the fixed-20 artifact preflight | 1 | No state changed. Accumulate rows in an array, then format the completed array; do not repeat the invalid pipeline form. |
| P3-A production review reproduced cross-request Search state contamination and cross-session same-role identity merging | 1 | Block live evaluation; convert all retrieval-driving graph state to request-local variables and restrict non-speaker identity hints to session-local reuse, then require concurrency/identity regressions before fixed-20. |
| P3-0 fixed-20 rerun produced six empty correction payloads and temporal accuracy `0.4` | 1 | llama.cpp logs proved each failed request reached the four-slot server's 1024-token per-slot limit with `truncated=1`; classify that run as an invalid local-runtime diagnostic and restart the same model with total context 8192 (about 2048 per slot) before rerunning. |
| A read-only Python one-liner for printing fixed-20 gold turns had an unclosed f-string/list expression because PowerShell quoting split the command | 1 | No file changed. Re-ran the inspection with native PowerShell JSON parsing and explicit loops. |
| A planning-file patch targeted the P3-0 error-table row in `progress.md` instead of `task_plan.md` | 1 | The patch failed atomically. Split the update and applied the error entry under the actual table. |
| Polling the historical fixed-200 evaluator session ID returned `Unknown process id` | 1 | The old process is no longer managed and produced no saved result. Do not infer completion; the new paired fixed-200 run will establish the valid baseline instead. |
| The first P3-A gate unit expected fixed-20 mechanics to require a graph-new gold case, so the outcome-free screen returned `REJECT_P3_A` | 1 | Kept graph-new gold as a formal fixed-200 hard gate, but made fixed-20 mechanics depend only on trace/DB/integrity/call/quota/seed invariants; it must not consume gold outcomes for parameter choice. |
| Hidden `Start-Process` launch of the local Qwen server was rejected by the execution policy | 1 | No child was started. Switch to a foreground unified exec session, keep its session ID, health-check port 8081, and stop that exact session after evaluation. |
| Python dependency installation timed out; initial virtualenv pip was corrupted | 2 | Used static compilation for this session; Docker/pytest verification remains pending in an environment with working package access. |
| Docker CLI unavailable | 1 | Dockerfile is complete; image build must run on a machine with Docker installed. |
| PowerShell parsed `v0.1.0^{commit}` incorrectly while reading the tag target | 1 | Tag creation and push succeeded; verified the release through GitHub CLI instead. |
| Official OpenAI documentation connector could not be installed because `codex.exe` is access-denied | 2 | User must run the documented MCP installation command and restart Codex; then implementation can safely continue. |
| Temporal-ranking test returned a later-ingested, older event | 1 | Expanded each retrieval channel's candidate pool before RRF fusion, so temporal reranking can compare evidence beyond the final `top_k`. |
| Context retrieval changed a legacy API test's expected result count and exposed an over-broad fake fact extractor | 1 | Updated the idempotency assertion to verify stable source IDs and constrained the fake extractor to fact-bearing text. |
| Windows CIM memory query returned access denied | 1 | GPU/VRAM information was available through `nvidia-smi`; system RAM is nonessential for selecting the first bounded experiment. |
| System Python could not run pytest (`No module named pytest`) | 1 | Use an isolated `.venv-local` created from the bundled Python 3.12 runtime and install `requirements-local.txt`. |
| Bulk `requirements-local.txt` install timed out after 10 minutes and left two stalled child processes | 1 | Verified that no packages were installed, stopped only the two task-owned process IDs, and switched to staged dependency installation with visible output. |
| Pytest could not create temporary directories in either the Windows profile temp or `C:\tmp` under the managed sandbox | 2 | Use a gitignored `.pytest-tmp` directory inside the writable repository workspace. |
| SentenceTransformers/PyTorch pip installation timed out after 20 minutes with no ML packages installed | 1 | Stopped only the two verified task-owned child processes; inspect existing Conda ML environments before switching to a lighter ONNX backend. |
| PowerShell environment-variable enumeration raised a duplicate-key error while checking local LLM runtimes | 1 | The direct `ollama` executable check completed and showed it is absent; skip the nonessential environment enumeration path. |
| ONNX Runtime GPU wheel did not install within 15 minutes and left the experimental venv without an ONNX runtime | 1 | Stopped only the two verified installer processes and restored the known-good CPU `onnxruntime==1.28.0`; proceed with caching/algorithmic optimization instead of repeating the GPU install. |
| First multi-file rerank-sweep patch referenced a documentation line under the wrong file section | 1 | Inspected exact anchors and reapplied the changes with separate file headers. |
| Read-only GitHub CLI verification was blocked by sandbox networking | 1 | Re-ran the exact read-only release/tag check with approved network access; no repository state was changed. |
| GitHub CLI publish prerequisite reported both inherited and keyring credentials invalid | 2 | Clearing only the process-level token confirmed the keyring credential itself is invalid; local release preparation can finish, but `gh auth login -h github.com --web` is required before push. |
| `rg` was given a Windows wildcard path that PowerShell did not resolve | 1 | Query the `.github/workflows` directory directly instead of using a wildcard. |
| Optional `tests/test_config.py` inspection target did not exist | 1 | Located configuration coverage in `tests/test_model_mode.py` and added validation tests there. |
| Connected GitHub App rejected Git tree and branch creation with `Resource not accessible by integration` | 2 | The app remains useful for read/PR operations but cannot replace Git transport; reauthenticate GitHub CLI before pushing the prepared branch. |
| Academic-search workflow fragment was requested by its shorthand name instead of the manifest path | 1 | Loaded `references/workflows/wf1-multi-source-search.md` exactly as declared by `manifest.yaml`; do not retry the shorthand path. |
| Initial hierarchy test ranked the session's lexical topic cue above the exact evidence turn | 1 | Kept session fusion scoped to session selection and revised the fixture so within-session evidence selection is independently supported by the turn signal; exact-turn ordering remains an explicit evaluation target. |
| Official GitHub/Hugging Face API request was blocked by sandbox networking | 1 | Re-ran the read-only metadata request with approved network access. |
| Unauthenticated GitHub release API was rate-limited on the shared egress IP | 1 | Do not retry the API; validate a versioned official release asset URL directly from the primary release page. Hugging Face metadata succeeded and confirmed `Qwen3-4B-Q4_K_M.gguf`. |
| First combined download left the GitHub `curl` process open after the complete 38.6 MB ZIP was written | 1 | Verified the exact task-owned PID and complete file length, stopped that process with approved privileges, and separated the resumable model download from runtime download. |
| Sandboxed `Stop-Process` could not terminate the escalated download child | 1 | Re-ran only the exact verified PID termination with approved privileges; no unrelated process was touched. |
| Stopping the first download caused the combined parent command to terminate during the subsequent model transfer | 1 | The GGUF has a valid 357,322,752-byte partial file; resume it in a standalone `curl --continue-at -` command rather than repeating the combined workflow. |
| `Start-Process` could not build a child environment because the host contains duplicate case-insensitive `Path/PATH` keys | 1 | No server started. Retry once with `-UseNewEnvironment`, which avoids copying the malformed inherited environment and does not alter system variables. |
| `Start-Process -UseNewEnvironment` still hit the duplicate `Path/PATH` dictionary error | 2 | Abandoned `Start-Process`; used .NET `ProcessStartInfo` with `CreateNoWindow=true`, which started the loopback-only server successfully. |
| The .NET-started server inherited the launcher output handles, keeping the launcher cell open | 1 | Keep the verified task-owned launcher cell while evaluating, then terminate it after stopping the exact llama-server PID; future helper can redirect via a dedicated wrapper. |
| Qwen thinking-mode smoke consumed all 256 output tokens in `reasoning_content` and returned empty final content | 1 | Keep no-thinking unchanged; allow 768 tokens only in thinking mode, request brief reasoning, and revalidate one request before any dataset run. |
| Qwen thinking-mode real-candidate retry still consumed all 768 tokens and returned empty content after 88 seconds | 2 | Reject thinking mode for this 4B runtime; do not increase the budget again. Continue only with the validated no-thinking constrained JSON path. |
| PowerShell's current .NET `ProcessStartInfo.ArgumentList` property was null when launching Qwen3-8B | 1 | The argument-less child exited; use the supported `Arguments` string property with fully resolved paths and verify the exact child PID/health. |
| Qwen3-8B server disappeared 120 seconds after launch during evaluation | 1 | The launcher command hit its own 120-second timeout and terminated the inherited child; restart with a 30-minute task-owned launcher timeout before rerunning the unchanged gate. |
| Qwen3-8B emitted one non-standard JSON response and aborted the 20-question gate | 1 | Parse fenced/embedded JSON safely; when no valid supplied-ID list exists, preserve the original candidate order and count the fallback instead of failing the API. |
| Initial multi-file cross-encoder patch missed the compact local-reranker factory formatting | 1 | Inspected the exact factory and applied smaller anchored patches; no partial code change was made by the failed patch. |
| FastEmbed could not fetch BAAI/bge-reranker-base through the managed network sandbox | 1 | Re-run the unchanged 20-question gate with approved Hugging Face access; subsequent runs use the ignored local cache. |
| Initial hard-session multi-file patch missed the expanded `MemoryStore` constructor layout | 1 | Inspected the current constructor and applied smaller exact-anchor patches; the failed patch made no partial change. |
| Qwen3-Reranker-4B Q4_K_M download escalation was rejected because the current Codex external-operation usage limit was reached | 1 | Do not bypass through another channel. Retry the exact official Hugging Face download after 12:32 or after the user explicitly resumes/approves it. |
| Approved Qwen3-Reranker-4B Q4_K_M download could not connect to `huggingface.co:443` | 2 | No model bytes were written. Do not repeat the identical transfer; first diagnose the endpoint with a small approved HEAD request, then resume only if reachable. |
| Hugging Face no-body HEAD request also timed out at 20 seconds | 3 | Treat the download as externally unavailable for this session. Continue only offline integration/tests; do not retry the network transfer until connectivity changes. |
| A later approved recovery check again timed out at `huggingface.co:443` | 4 | Network remains unavailable. Leave the model gate pending and wait for an external connectivity change rather than repeating the request. |
| Original Q4_K_M model URL returned HTTP 404 through the functioning proxy | 1 | Query the public repository file list; use the currently available Q3_K_M quantization and verify its published size/hash before evaluation. |
| Qwen reranker first evaluation failed because the installed OpenAI SDK rejects `top_logprobs` on `/v1/completions` | 1 | Use the text-completions API form `logprobs=20`; retain the response parser for llama.cpp's content-shaped logprobs. |
| First Qwen reranker adapter treated llama.cpp's `logprobs.content` entries only as objects, but this SDK returned dictionaries | 1 | Support both shapes and cover the dictionary form with a unit test before rerunning the exact same gate. |
| Full Qwen reranker evaluation lost the local server connection | 1 | The launcher tool's 30-minute timeout terminated its child server before full evaluation could finish. Launch the verified loopback server as a detached hidden process, health-check it, then restart the unchanged full gate. |
| Caption-aware fixed-200 Qwen gate exceeded the 20-minute command limit without producing metrics | 1 | Treat it as a runtime failure, not a quality result. Switch to a fixed-20 gate and add resumable per-question diagnostics before any larger retry. Do not stop unrelated GPU processes. |
| Initial image-followup multi-file patch expected identical evaluator argument blocks | 1 | No partial changes were applied. Split storage/tests from CLI wiring and patch the evaluator's distinct branches at exact anchors. |
| Resume-offset test expected a hit for the query `Second?`, which had no matching content term | 1 | Keep the production offset logic unchanged and use `What does Milo prefer, coffee?` in the retrieval fixture. |
| Manual batched-completion probe used single-quoted PowerShell strings, leaving literal backtick-newline text in the prompt | 1 | Use the probe only to confirm indexed batch response support. Implement batching through the existing Python `prompt()` method and require exact fixed-20 metric parity before accepting it. |
| Comparative-audit patch placed an evaluator constructor anchor under the model-file patch section | 1 | No partial changes were applied. Patch the model/test and evaluator CLI as separate exact file edits. |
