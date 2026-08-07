# Spec: ChronoHybridMem v0.1.0

## Objective

Build a reproducible, Docker-deployable textual-memory service for the Agent Memory Challenge academic track. It accepts conversation chunks synchronously and returns ranked evidence only. It never generates benchmark answers.

## Tech stack

- Python 3.11 target runtime
- FastAPI and Uvicorn for HTTP
- SQLite with FTS5 for persistent lexical retrieval
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

## Success criteria

- `GET /health` returns 200 without authentication.
- `POST /add` persists each message and echoes the three request identifiers with `success: true`.
- Repeating an Add `request_id` is safe and does not duplicate records.
- `POST /search` returns `{ "data": [...] }`, bounded by `top_k`, with non-empty IDs and content.
- Docker builds and launches the service with a persistent mounted data directory.

## Open questions

- A future version may add `gpt-4o-mini` fact extraction after written confirmation of the competition's model-use rules.
