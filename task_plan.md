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
- [ ] Add neighbor-context indexing as a lower-weight RRF channel and evaluate it on full LoCoMo (in progress).

## Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Python dependency installation timed out; initial virtualenv pip was corrupted | 2 | Used static compilation for this session; Docker/pytest verification remains pending in an environment with working package access. |
| Docker CLI unavailable | 1 | Dockerfile is complete; image build must run on a machine with Docker installed. |
| PowerShell parsed `v0.1.0^{commit}` incorrectly while reading the tag target | 1 | Tag creation and push succeeded; verified the release through GitHub CLI instead. |
| Official OpenAI documentation connector could not be installed because `codex.exe` is access-denied | 2 | User must run the documented MCP installation command and restart Codex; then implementation can safely continue. |
| Temporal-ranking test returned a later-ingested, older event | 1 | Expanded each retrieval channel's candidate pool before RRF fusion, so temporal reranking can compare evidence beyond the final `top_k`. |
