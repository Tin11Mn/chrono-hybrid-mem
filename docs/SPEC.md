# Spec: ChronoHybridMem v0.1.0

## Objective

Build a reproducible, Docker-deployable textual-memory service for the Agent Memory Challenge academic track. It accepts conversation chunks synchronously and returns ranked evidence only. It never generates benchmark answers.

## Tech stack

- Python 3.11 target runtime
- FastAPI and Uvicorn for HTTP
- SQLite with FTS5 for persistent lexical retrieval
- OpenAI Python SDK using `gpt-4o-mini` for source-linked fact extraction and candidate-only reranking in evaluation mode
- Pytest and HTTPX for contract and isolation tests

## Commands

```text
docker build -t chrono-hybrid-mem:0.1.0 .
docker run --rm -p 8000:8000 -v chrono-data:/data chrono-hybrid-mem:0.1.0
python -m pytest
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Project structure

```text
app/       HTTP routes, schemas, SQLite storage and retrieval
tests/     API contract, idempotency and isolation tests
docs/      implementation specification
```

## Code style

Use typed, side-effect-light functions and parameterised SQL.

```python
def search_messages(*, user_id: str, query: str, top_k: int) -> list[MemoryResult]:
    """Return evidence belonging only to the requested user."""
```

## Testing strategy

Pytest exercises health, synchronous Add, retry idempotency, strict `user_id` isolation, Search response shape and empty results. Tests use a temporary SQLite database.

## Boundaries

- Always: validate input, persist before returning success, parameterise SQL, run tests.
- Ask first: add external models/services, change API contract, publish or push to GitHub.
- Never: commit secrets or evaluation data; generate answers in Search; mix users' memories.

## Model deployment mode

Local development may omit `OPENAI_API_KEY` and use lexical retrieval only. Competition deployment must set `MEMORY_REQUIRE_MODEL=true` and inject `OPENAI_API_KEY` through the platform's secret mechanism. In that mode the service refuses to start without the key, uses only `gpt-4o-mini`, stores source-linked extracted facts, and asks the model to return only an ordering of existing candidate IDs. Search never requests or returns a final answer.

## Hybrid ranking

Search retrieves through two channels: raw-message FTS5 and source-linked fact FTS5. It merges the two ordered lists with Reciprocal Rank Fusion (RRF, constant 60) rather than comparing their independent BM25 values. Facts are retrieval aids only: results always return the original source message. When a query explicitly asks for the current, latest, recent, or equivalent Chinese temporal state, a small event-timestamp bonus resolves otherwise similar evidence; no time bias is applied to ordinary queries.

## Success criteria

Historical queries such as `before` and `previous` receive the corresponding older-event bonus. Ordinary queries receive no time bias.
The temporal bonus is disabled by default because external evaluation did not establish a general gain. It may be enabled experimentally with `MEMORY_TEMPORAL_BONUS` (maximum `0.01`) only when validated on a permitted development set.
For English natural-language questions, common function words are removed from the FTS query when content terms remain; the original terms are retained for all-stopword queries and non-English text.

- `GET /health` returns 200 without authentication.
- `POST /add` persists each message and echoes the three request identifiers with `success: true`.
- Repeating an Add `request_id` is safe and does not duplicate records.
- `POST /search` returns `{ "data": [...] }`, bounded by `top_k`, with non-empty IDs and content.
- Docker builds and launches the service with a persistent mounted data directory.
- In evaluation mode, Add extracts source-linked facts with `gpt-4o-mini` and Search uses `gpt-4o-mini` only to reorder existing evidence candidates.
- Hybrid ranking must preserve source evidence, favour relevant multi-channel matches, and use event time only for explicit temporal queries.

## Open questions

- The deployment operator must provide the OpenAI API key as a runtime secret; it must never be stored in the repository or submission text.
