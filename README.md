# ChronoHybridMem

ChronoHybridMem is a Docker-deployable textual-memory system for the Agent Memory Challenge academic track. It stores conversation evidence synchronously and returns ranked source records; it does not generate benchmark answers.

- **v0.2.0** is the immutable competition submission using only `gpt-4o-mini` in model-backed mode.
- **v0.3.0** is a local-method research milestone that adds lexical-dense score fusion and bounded late-interaction reranking. It does not replace the submitted tag.
- **v0.4.0-local** adds an optional purpose-trained Qwen yes/no reranker and time-aware dense keys for local research. It does not replace the submitted tag and requires a separately launched loopback model server; model weights are never committed or bundled into Docker.

## System metadata

- Author: 孟昊轩 (Haoxuan Meng)
- Institution: 郑州大学 (Zhengzhou University)
- Contact: hxmeng@gs.zzu.edu.cn
- Track / Division: Textual Memory / Academic Methods
- Submission route: Public GitHub repository with maintainer Docker deployment

## Method

The stable competition path is:

```text
Add: validate -> persist raw messages -> gpt-4o-mini fact extraction -> FTS5 indexing
Search: gpt-4o-mini query planning -> user isolation -> FTS5 candidates -> RRF -> gpt-4o-mini evidence ordering
```

The optional v0.3 local path preserves the same Add/Search API:

```text
Search: Porter-BM25 + BGE-large dense scores -> candidate-set z-score fusion (alpha=0.3)
        -> answerai-colbert-small MaxSim reranking of the top 5 -> original evidence records
```

The score-fusion formula follows the training-free lexical-dense method described in [Training-Free Lexical–Dense Fusion for Conversational-Memory Retrieval](https://arxiv.org/html/2606.04194). The local reranker is a token-level late-interaction stage, not an answer generator. Model files, evaluation data, and credentials are excluded from Git.

### P1 structured query plan (opt-in)

Set `MEMORY_STRUCTURED_QUERY_PLAN=true` to replace the flat list produced by the existing
`gpt-4o-mini` query-planning call with a bounded plan that separates core/entity/temporal
terms from low-weight expansion and evidence-need terms. This does not add a model call,
change the Add/Search contract, or permit generated answers. The default remains `false`.

On a fixed 20-question LoCoMo development slice, using local Qwen3-4B only as a
`gpt-4o-mini` planning/ranking proxy, P1 improved Hit@1 from 0.55 to 0.60 and MRR from
0.5917 to 0.6125. Hit@10 stayed at 0.65, while exact evidence recall@10 decreased from
0.4839 to 0.4194. These small-slice proxy measurements are ablation evidence, not an
official leaderboard result; the feature therefore remains isolated behind the flag.

## Validated retrieval result

The selected v0.3 configuration was evaluated on all 1,977 LoCoMo questions using strict exact annotated evidence-turn matching:

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Immutable v0.2.0 retrieval | 0.2671 | 0.4355 | 0.5695 | 0.3567 |
| Latest no-model lexical baseline | 0.3359 | 0.5169 | 0.6591 | 0.4328 |
| v0.3 local hybrid | **0.4355** | **0.6186** | **0.7577** | **0.5183** |
| v0.4 local Qwen + time key | **0.5225** | **0.6808** | **0.7653** | **0.5856** |

This is a retrieval-only external result, not an official leaderboard score. The cited paper reports session-level retrieval, whereas this repository uses the stricter exact evidence-turn criterion, so its numbers are not directly comparable. The project target remains Hit@1 ≥ 0.70.

## Run the competition-compatible image

```bash
docker build -t chrono-hybrid-mem:0.2.0 .
docker run --rm -p 8000:8000 -v chrono-memory-data:/data \
  -e MEMORY_REQUIRE_MODEL=true -e MEMORY_STRUCTURED_QUERY_PLAN=true \
  -e OPENAI_API_KEY=your_runtime_secret \
  chrono-hybrid-mem:0.2.0
```

Competition mode requires a private `OPENAI_API_KEY` at runtime. The secret is not stored in this repository or image.

## Run the local v0.3 method

```bash
docker build -f Dockerfile.local -t chrono-hybrid-mem:0.3.0-local .
docker run --rm -p 8000:8000 \
  -v chrono-memory-data:/data -v chrono-local-models:/models \
  chrono-hybrid-mem:0.3.0-local
```

The first start downloads the configured BGE and ColBERT weights into the persistent `/models` volume. The default is CPU inference; set `MEMORY_LOCAL_EMBEDDING_DEVICE` only when the container has a compatible runtime.

Health check: `GET http://localhost:8000/health`

## Run the optional v0.4 local reranker

Start a compatible Qwen3-Reranker model through a loopback-only OpenAI-compatible server, then set these optional variables before launching the same API service:

```text
MEMORY_LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
MEMORY_DENSE_FUSION_ALPHA=0.3
MEMORY_DENSE_TIME_WEIGHT=0.5
MEMORY_LOCAL_YES_NO_RERANK_BASE_URL=http://127.0.0.1:8081/v1
MEMORY_LOCAL_YES_NO_RERANK_MODEL=local
MEMORY_LOCAL_RERANK_TOP_N=10
```

The reranker compares only supplied query–evidence pairs through the Qwen yes/no logprob protocol and returns the original records; it never produces a benchmark answer. This local research path is intentionally opt-in because the model server and weights are not part of the Docker image.

## API

`POST /add` persists data before returning success. Retrying the same `request_id` is idempotent.

```json
{"request_id":"run:1:chunk:0","user_id":"run:1:conversation:0","session_id":"run:1:session:0","messages":[{"role":"user","content":"Alice prefers tea."}]}
```

`POST /search` searches only within the exact `user_id` and returns at most `top_k` records (1–100).

```json
{"query":"What does Alice prefer?","user_id":"run:1:conversation:0","top_k":100}
```

```json
{"data":[{"id":"mem_1","content":"Alice prefers tea.","score":1.0,"created_at":"2026-08-07T00:00:00Z"}]}
```

## Local development and tests

Requires Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m pytest
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For the optional local retrieval backend, install `requirements-local.txt` instead. Copy `.env.example` values through your environment; the validated configuration is documented there.

## Offline evaluation

The included examples are fictional and may be evaluated with:

```bash
python scripts/evaluate_retrieval.py --cases examples/demo_eval.json
```

The CI regression diagnostic compares current lexical behavior with immutable v0.2.0. It is mechanism-level evidence, not a challenge score. For the public LoCoMo protocol and the exact v0.3 command, see [docs/EXTERNAL_EVALUATION.md](docs/EXTERNAL_EVALUATION.md). Third-party data is supplied by a local path and excluded from Git.

## Operational notes

- `MEMORY_DB_PATH` controls the SQLite path; Docker defaults to `/data/chrono_hybrid_mem.db`.
- `MEMORY_REQUIRE_MODEL=true` fails fast when competition model mode has no API key.
- Model-backed Search only orders existing evidence IDs and never generates a benchmark answer.
- SQLite WAL supports concurrent readers and serialized writes; `request_id` makes Add retries idempotent.
- Evaluation data must not be used for training or analytics and must be deleted according to applicable rules and licenses.
- No credentials belong in this repository; `.env.example` contains configuration names only.

## License

Released under the [MIT License](LICENSE).
