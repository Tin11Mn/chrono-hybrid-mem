# Cycle 2 deployment checklist

This document is for the current lightweight Textual Memory / Open-source
Methods submission path. The official contract is published at
<https://agentmemories.ai/rules>.

## Runtime configuration

Inject secrets through the deployment platform, never through Git, the image,
the URL, or screenshots:

```text
MEMORY_REQUIRE_MODEL=true
OPENAI_API_KEY=<runtime-secret>
OPENAI_BASE_URL=<optional-compatible-https-endpoint>
MEMORY_SYSTEM_KEY=<participant-api-token>
MEMORY_DB_PATH=/data/chrono_hybrid_mem.db
MEMORY_STRUCTURED_QUERY_PLAN=true
MEMORY_EVIDENCE_NEED_RETRIEVAL=true
MEMORY_NEED_SELECT_BY_BM25=true
```

The model path must use `gpt-4o-mini` for both Add and Search for an
Open-source Methods entry. `MEMORY_REQUIRE_MODEL=true` prevents an accidental
fallback to the no-model lexical path.

For ChatAnywhere's OpenAI-compatible endpoint, set
`OPENAI_BASE_URL=https://api.chatanywhere.tech/v1`; keep its API key in
`OPENAI_API_KEY`. This model-provider URL is separate from the public Add/Search
URLs submitted to AML.

## HTTP contract

- `GET /health` is unauthenticated and must return any 2xx response.
- `POST /add` and `POST /search` use the submitted Bearer authentication.
- Add must finish persistence before returning HTTP 200.
- Search returns `{ "data": [...] }` containing original memory evidence only.
- Search must return no more than the requested `top_k` (formal evaluation uses
  `top_k=100`).
- The database volume must survive process/container restarts.
- For a NAT VPS, map the provider-generated public external port to the
  service's internal port (8000 for the app, or 443 when using a local HTTPS
  reverse proxy). Do not depend on security-group rules when the provider
  requires NAT forwarding. Record the generated external port and include it
  in the submitted public URL when it is not the standard port.

## Freeze procedure

1. Rotate any credential that has ever appeared in repository history.
2. Run the focused API tests and a live Add → Search smoke test.
3. Record the exact commit and create the final tag.
4. Build and deploy the image from that exact commit.
5. Submit the same commit/tag, endpoint URLs, auth scheme, capacity, timeout,
   and operational limits in the evaluation request.

The current standard submission does not claim Session-Fact v2. That research
path requires additional runtime wiring and semantic retrieval dependencies;
it must be evaluated separately before being declared as a submitted feature.
