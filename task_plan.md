# ChronoHybridMem implementation plan

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
- [x] Run full Qwen + time-aware-key validation; Hit@1=0.5225 > 0.5195, retain as current local best.
- [x] Gate Qwen hard reranking with the previously high-recall context-aware dense key (weight 0.5) on fixed 200 questions; tie Hit@1=0.505, reject alone.
- [x] Gate the combined time-aware + context-aware dense keys with hard Qwen on fixed 200 questions; reject Hit@1=0.500.
- [ ] Select a new non-overlapping retrieval mechanism and gate it against the current full-set best Hit@1=0.5225.
- [x] Publish the verified v0.4 local research milestone to an isolated GitHub branch and draft PR.

## Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
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
