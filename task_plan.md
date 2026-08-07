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

## Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Python dependency installation timed out; initial virtualenv pip was corrupted | 2 | Used static compilation for this session; Docker/pytest verification remains pending in an environment with working package access. |
| Docker CLI unavailable | 1 | Dockerfile is complete; image build must run on a machine with Docker installed. |
| PowerShell parsed `v0.1.0^{commit}` incorrectly while reading the tag target | 1 | Tag creation and push succeeded; verified the release through GitHub CLI instead. |
| Official OpenAI documentation connector could not be installed because `codex.exe` is access-denied | 2 | User must run the documented MCP installation command and restart Codex; then implementation can safely continue. |
