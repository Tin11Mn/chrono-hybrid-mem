# ChronoHybridMem

ChronoHybridMem v0.2.0 is a Docker-deployable textual-memory system for the Agent Memory Challenge academic track. It synchronously stores raw conversation evidence, derives source-linked facts with `gpt-4o-mini` in competition mode, retrieves candidates through SQLite FTS5, and returns ranked memory records only. It does not generate benchmark answers.

## System metadata

- Author: 孟昊轩 (Haoxuan Meng)
- Institution: Zhengzhou University (郑州大学)
- Contact: hxmeng@gs.zzu.edu.cn
- Track / Division: Textual Memory / Academic Methods
- Submission route: Public GitHub repository with maintainer Docker deployment

## Method

```text
Add: validate -> persist raw messages -> gpt-4o-mini fact extraction -> FTS5 indexing -> return success
Search: strict user_id filter -> raw/fact FTS5 candidates -> RRF fusion and temporal boost -> gpt-4o-mini candidate ordering -> return evidence records
```

Competition mode uses only `gpt-4o-mini` for fact extraction and candidate ordering. It requires a private `OPENAI_API_KEY` at runtime; the key is not in this repository or Docker image. Local development can leave model mode disabled and use the lexical baseline. No evaluation data is stored in the repository.

Raw-message and source-linked-fact results are merged with Reciprocal Rank Fusion rather than comparing their separate BM25 scales. For queries explicitly asking for the latest/current state, event timestamps apply a small recency bonus. Returned content remains the original source message; facts only improve retrieval.

## Run with Docker

```bash
docker build -t chrono-hybrid-mem:0.2.0 .
docker run --rm -p 8000:8000 -v chrono-memory-data:/data \
  -e MEMORY_REQUIRE_MODEL=true -e OPENAI_API_KEY=your_runtime_secret \
  chrono-hybrid-mem:0.2.0
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

## Offline retrieval evaluation

Use only your own or otherwise permitted, non-official data. The included fictional examples are not competition data.

```bash
python scripts/evaluate_retrieval.py --cases examples/demo_eval.json
```

The report includes evidence-level `Recall@K`, per-case evidence coverage, MRR, and average Search latency. Add local cases with `case_id`, `user_id`, `session_id`, `messages`, `query`, and `expected_evidence`; every expected-evidence string is matched against returned memory content.

`examples/release_comparison_diagnostic.json` is a small fictional regression diagnostic. CI compares its `Recall@1` and MRR against the immutable `v0.2.0` release using `scripts/compare_release_v020.py`. It measures this implementation change only; it is not an official leaderboard score or a proxy for the challenge's hidden test set.

## Operational notes

- `MEMORY_DB_PATH` controls the SQLite location; Docker defaults to `/data/chrono_hybrid_mem.db`.
- Competition deployment must set `MEMORY_REQUIRE_MODEL=true` and inject `OPENAI_API_KEY` through a secret manager. The service fails fast if model mode is required but no key is available.
- `gpt-4o-mini` extracts concise source-linked facts during Add and returns only an ordering of existing evidence IDs during Search. It never produces a benchmark answer.
- The service is safe for retried Add calls using `request_id`; SQLite WAL supports concurrent readers and serialized writers.
- Evaluation data must not be used for training or analytics and should be removed within the competition's required retention period. The deployment operator is responsible for deleting the mounted data volume.
- No credentials belong in this repository. The file `.env.example` contains names only.

## Attribution and versioning

This implementation is original project code using FastAPI, Uvicorn, SQLite FTS5, and the OpenAI Python SDK with `gpt-4o-mini`. Before submission, create a release tag and include its commit SHA, license, and any added upstream attribution in this README.

## License

Released under the [MIT License](LICENSE).
