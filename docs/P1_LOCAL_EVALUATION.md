# P1 Local-Model Evaluation

## Scope

This report records a full 1,977-question LoCoMo retrieval evaluation of P1
structured query planning using the local Qwen3-4B model through the loopback
OpenAI-compatible llama.cpp server. The local model replaced `gpt-4o-mini` for
both Search planning and evidence ordering. Add-time extraction was skipped,
matching the existing Search-only evaluation protocol.

The evaluation uses strict exact annotated evidence-turn matching. It is a
local method result, not an official Agent Memory Challenge leaderboard score.
The LoCoMo source file and model weights remain outside Git.

## Method

P1 was enabled with `--structured-query-plan`. The planner returns bounded
fields for intent, core terms, expansions, entities, temporal cues, and
evidence needs. Core/entity/temporal fields feed the primary lexical channels;
expansions and evidence needs use the calibrated support channel. The ranker
still receives only retrieved candidates and returns existing evidence IDs.

The local server was Qwen3-4B Q4_K_M served by llama.cpp on `127.0.0.1:8081`.
Because a single long run previously lost its final aggregate after a late
JSON truncation, the validated run used ten resumable chunks and merged only
successfully written aggregate results. The local adapter now tolerates fenced
or embedded JSON and falls back to safe first-stage behavior when a local
response is incomplete; the strict remote model path still raises on malformed
JSON.

## Full result

| Metric | P1 local Qwen3-4B |
|---|---:|
| Questions | 1,977 |
| Hit@1 | **0.5761** |
| Hit@3 | **0.7157** |
| Hit@10 | **0.7618** |
| MRR | **0.6479** |
| Evidence Recall@10 | 0.5976 |

Category Hit@1:

| Category | Hit@1 |
|---|---:|
| 1 | 0.3879 |
| 2 | 0.6031 |
| 3 | 0.2697 |
| 4 | 0.6445 |
| 5 | 0.6076 |

## Fixed-200 gate

The predeclared local screening comparison was:

| Configuration | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Local flat planner | 0.545 | 0.670 | 0.740 | 0.6145 |
| Local P1 structured planner | **0.565** | **0.685** | 0.740 | **0.6292** |

P1 passed the gate because Hit@1 and MRR improved while Hit@10 remained
unchanged. This triggered the full local P1 run. A full 1,977-question local
flat control was not run, so the full P1 result must not be described as a
full-set flat-to-P1 delta.

## Reproduction

The source dataset must be supplied from an approved local path. Start the
Qwen3-4B llama.cpp server on `127.0.0.1:8081`, then run the Search-only
evaluation in resumable chunks:

```powershell
python scripts/evaluate_locomo_retrieval.py `
  --dataset .locomo/locomo10.json `
  --max-questions 200 `
  --question-offset 0 `
  --local-search-model-url http://127.0.0.1:8081/v1 `
  --structured-query-plan `
  --output .locomo/p1-local-proxy-structured-chunk-0000.json
```

After all offsets are complete, merge the aggregate-only chunk files:

```powershell
python scripts/merge_locomo_chunks.py `
  .locomo/p1-local-proxy-structured-chunk-*.json `
  --output .locomo/p1-local-proxy-structured-full.json
```

The generated full JSON is intentionally ignored by Git because it is derived
from local evaluation data. This report contains the reproducible aggregate
metrics without publishing the dataset.
