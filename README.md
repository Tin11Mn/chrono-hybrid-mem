# ChronoHybridMem

ChronoHybridMem v0.1.0 is a Docker-deployable textual-memory baseline for the Agent Memory Challenge academic track. It synchronously stores raw conversation evidence, retrieves lexical matches through SQLite FTS5, and returns ranked memory records only. It does not generate benchmark answers.

## Method

```text
Add: validate -> persist raw messages -> index in SQLite FTS5 -> return success
Search: strict user_id filter -> FTS5/BM25 ranking -> return evidence records
```

The v0.1.0 baseline deliberately uses no external model, API key, embedding service, or evaluation data. A future source-linked fact layer can be added only after the competition's model rules are confirmed in writing.

## Run with Docker

```bash
docker build -t chrono-hybrid-mem:0.1.0 .
docker run --rm -p 8000:8000 -v chrono-memory-data:/data chrono-hybrid-mem:0.1.0
```

Health check: `GET http://localhost:8000/health`

## API

`POST /add` persists data before returning a success response. Retrying the same `request_id` is idempotent.

```json
{"request_id":"run:1:chunk:0","user_id":"run:1:conversation:0","session_id":"run:1:session:0","messages":[{"role":"user","content":"Alice prefers tea."}]}
```

```json
{"success":true,"request_id":"run:1:chunk:0","user_id":"run:1:conversation:0","session_id":"run:1:session:0"}
```

`POST /search` searches only within the exact `user_id` and returns at most `top_k` records (1–100).

```json
{"query":"What does Alice prefer?","user_id":"run:1:conversation:0","top_k":100}
```

```json
{"data":[{"id":"mem_1","content":"Alice prefers tea.","score":1.0,"created_at":"2026-08-07T00:00:00Z"}]}
```

## Local development and verification

Requires Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m pytest
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Operational notes

- `MEMORY_DB_PATH` controls the SQLite location; Docker defaults to `/data/chrono_hybrid_mem.db`.
- The service is safe for retried Add calls using `request_id`; SQLite WAL supports concurrent readers and serialized writers.
- Evaluation data must not be used for training or analytics and should be removed within the competition's required retention period. The deployment operator is responsible for deleting the mounted data volume.
- No credentials belong in this repository. The file `.env.example` contains names only.

## Attribution and versioning

This initial implementation is original project code using FastAPI, Uvicorn and SQLite FTS5. Before submission, create a release tag and include its commit SHA, author/team details, and any added upstream attribution in this README.
