# Progress

## 2026-08-07

- Created the independent local Git repository.
- Documented the implementation specification and first delivery plan.
- Implemented FastAPI Add/Search endpoints, SQLite FTS5 persistence and API tests.
- Corrected Add idempotency to rely on SQLite's unique constraint under concurrent retries.
- Dependency installation could not complete locally (two timeouts; one initial pip installation was corrupted), so runtime tests remain pending.
- Python static compilation passed for all application and test modules.
- Docker CLI is not installed on this workstation; Docker image verification is pending elsewhere.
- Removed the failed `.venv` created during dependency setup.
- Created the public repository `Tin11Mn/chrono-hybrid-mem` and pushed the initial `main` branch.
- Added GitHub Actions verification; run 31167009857 passed pytest and Docker image build.
- Received submission metadata: 孟昊轩, 郑州大学, hxmeng@gs.zzu.edu.cn.
- Added MIT License, pushed tag `v0.1.0`, and published the corresponding GitHub release.
- Added offline retrieval evaluation with fictional examples; GitHub Actions run 31168359021 passed tests and Docker build.
- Began `gpt-4o-mini` integration planning; paused because the required official-docs MCP connector cannot be installed from this Codex session.
- Implemented `gpt-4o-mini` source-linked fact extraction and candidate-only reranking; GitHub Actions run 31169046309 passed all tests and Docker build without an API call.
- Submitted the academic Docker-deployment application for ChronoHybridMem v0.2.0. The platform confirmed receipt and will deploy, validate the API contract, and initiate evaluation without issuing a leaderboard key.
- Diagnosed the first hybrid-retrieval CI failure: each channel had been truncated at the final response size before temporal ranking. Expanded the candidate pool before fusion; verification is in progress.
- Completed hybrid retrieval verification: GitHub Actions run 31170561376 passed API tests and the Docker image build. The implementation now gathers a wider per-channel candidate pool before RRF fusion and temporal ranking.
