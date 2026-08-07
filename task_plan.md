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
- [ ] Verify API tests and Docker build, then document how the validated local method maps to the GPT-backed path.
- [ ] Publish the verified Hit@1 0.4355 local-method milestone as an isolated v0.3.0 draft PR/release without moving v0.2.0.

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
