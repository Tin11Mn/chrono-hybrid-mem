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

LoCoMo is licensed CC BY-NC 4.0. It may be used only for non-commercial evaluation with required attribution; do not add its content or derived records to this repository. See the [upstream repository](https://github.com/snap-research/locomo) and its [license](https://raw.githubusercontent.com/snap-research/locomo/main/LICENSE.txt).
