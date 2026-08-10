# External evaluation protocol

This repository contains no third-party evaluation examples or results. External data is downloaded to a temporary local directory, evaluated, and deleted according to the competition rules and the dataset licence.

## LoCoMo retrieval evaluation

LoCoMo is a public long-term conversation benchmark with question-level evidence dialog IDs. This project evaluates retrieval only: each conversation is added under an isolated `user_id`, then each question is searched and returned source-message text is matched to its annotated evidence dialogs.

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -OutFile C:\tmp\locomo10.json
python scripts/evaluate_locomo_retrieval.py --dataset C:\tmp\locomo10.json --compare-v020
Remove-Item C:\tmp\locomo10.json
```

The report gives question hit rate and evidence recall at K, MRR, and category-level Hit@1. With `--compare-v020`, it also runs the v0.2.0 no-model raw-message retrieval logic over exactly the same conversations and prints deltas. It does not call an answer model and does not reproduce LoCoMo's end-to-end QA score.

For a bounded Search-only model ablation, set `OPENAI_API_KEY` through a secret manager and add `--search-model --max-questions 100`. This sends the public questions and retrieved candidate evidence to `gpt-4o-mini` for query planning and candidate ordering. It deliberately skips Add-time fact extraction to bound calls and isolate Search behavior.

For a fully local dense-retrieval ablation, install `requirements-local.txt` and run:

```powershell
python scripts/evaluate_locomo_retrieval.py --dataset C:\tmp\locomo10.json `
  --local-embedding-model BAAI/bge-small-en-v1.5 --local-device cpu `
  --local-cache-dir .model-cache --dense-weight 1 --max-questions 100 --compare-v020
```

The model cache is ignored by Git. Local dense retrieval is a method-development backend, not a declaration that non-`gpt-4o-mini` models are accepted by the competition.

The selected v0.3 full-set configuration is reproducible with:

```powershell
python scripts/evaluate_locomo_retrieval.py --dataset C:\tmp\locomo10.json `
  --local-embedding-model BAAI/bge-large-en-v1.5 --local-device cpu `
  --local-cache-dir .model-cache --fusion-alpha 0.3 `
  --local-reranker-model answerdotai/answerai-colbert-small-v1 `
  --rerank-top-n 5 --compare-v020
```

The v0.4 local research milestone uses a purpose-trained loopback Qwen reranker. After starting a compatible local server at `http://127.0.0.1:8081/v1`, run:

```powershell
python scripts/evaluate_locomo_retrieval.py --dataset C:\path\to\locomo10.json `
  --top-k 1,3,10 --local-embedding-model BAAI/bge-large-en-v1.5 `
  --local-device cpu --local-cache-dir .model-cache --fusion-alpha 0.3 `
  --dense-time-weight 0.5 --local-yes-no-reranker-url http://127.0.0.1:8081/v1 `
  --local-yes-no-reranker-model local --rerank-top-n 10
```

On the local full 1,977-question exact-evidence protocol, this configuration reached Hit@1 0.5225, Hit@3 0.6808, Hit@10 0.7653, and MRR 0.5856. These are local research measurements, not a platform leaderboard result. Do not commit model weights, LoCoMo data, or generated evaluation artifacts.

On all 1,977 questions, this run produced Hit@1/3/10 of 0.4355/0.6186/0.7577 and MRR 0.5183. The immutable v0.2.0 comparison produced Hit@1 0.2671 and MRR 0.3567. These are aggregate retrieval-only results with strict exact evidence-turn matching, not official leaderboard results. The referenced fusion paper reports a different session-level retrieval metric, so its values are not directly comparable.

Use `--fusion-alphas 0.3,0.4,0.5,0.6,0.7` for a bounded sweep that reuses document and query embeddings inside one process. This avoids recomputing the model for every fusion weight.

After selecting an alpha, use `--rerank-top-ns 5,10,20` together with a local late-interaction model to compare rerank-pool sizes with the same shared-cache mechanism.

LoCoMo is licensed CC BY-NC 4.0. It may be used only for non-commercial evaluation with required attribution; do not add its content or derived records to this repository. See the [upstream repository](https://github.com/snap-research/locomo) and its [license](https://raw.githubusercontent.com/snap-research/locomo/main/LICENSE.txt).
