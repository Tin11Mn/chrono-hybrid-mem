# AML-like synthetic retrieval suite

This directory contains 210 deterministic, non-hidden evidence-retrieval cases:
30 each for A, B, C, D, E, G, and H. They are generated templates and do not
copy or infer any hidden leaderboard item.

Regenerate and validate the dataset with:

```powershell
python evaluation/aml_synthetic/generate_cases.py
```

`required_evidence_ids` defines the source memories needed by an answer model;
`forbidden_evidence_ids` marks conflicting, wrong-person, obsolete, example-only,
or cross-user evidence that must not displace the required source. Category B
requires all listed evidence IDs for all-evidence coverage metrics.

Run a deterministic lexical baseline:

```powershell
python -m evaluation.aml_synthetic.evaluate
```

Run the two formal-path arms with the same `OPENAI_API_KEY` and suite:

```powershell
python -m evaluation.aml_synthetic.evaluate --model --output baseline.json
python -m evaluation.aml_synthetic.evaluate --model --structured --output p1.json
```

The evaluator reports Hit@1/3/5/10, MRR, per-category Hit@K, B all-evidence
coverage, forbidden-evidence rate, cross-user leakage, duplicate ratio,
candidate pool size, Add/Search latency, GPT call count, and token usage.

Without an API key, isolate only the retrieval-routing effect using the same
annotated plan and an identity ranker in both arms:

```powershell
python -m evaluation.aml_synthetic.evaluate --fixture-plans --output fixture-flat.json
python -m evaluation.aml_synthetic.evaluate --fixture-plans --structured --output fixture-structured.json
```

Run the P2 mechanical paired comparison using the same frozen fixture plans:

```powershell
python -m evaluation.aml_synthetic.evaluate --fixture-plans --structured --output p2-p1.json
python -m evaluation.aml_synthetic.evaluate --fixture-plans --structured --set-aware-rerank --output p2-on.json
python -m evaluation.aml_synthetic.compare p2-p1.json p2-on.json --p2-non-degradation
```

For a local-model smoke test that is balanced across all seven public
categories, use `--per-category-limit 1` (or a higher value); do not use an
unstratified prefix as a cross-category claim.

Fixture-plan numbers are an engineering ablation, never a substitute for the
formal `gpt-4o-mini` arms.

Apply the predeclared score-preservation gate with:

```powershell
python -m evaluation.aml_synthetic.compare fixture-flat.json fixture-structured.json
```

The gate rejects any non-improving result, A/B regression, C/E/H regression,
coverage loss, forbidden-evidence increase, cross-user leakage, extra GPT call,
or mean Search latency above 1.25×. Fixture runs can only return
`MECHANICS_PASS`; formal-model runs return `ADVANCE`, never an automatic
leaderboard `KEEP` decision.

For method screening only, a local OpenAI-compatible model can replace the two
Search calls without changing the competition implementation:

```powershell
python -m evaluation.aml_synthetic.evaluate --local-base-url http://127.0.0.1:8081/v1 --skip-extraction --limit 21 --output local-flat.json
python -m evaluation.aml_synthetic.evaluate --local-base-url http://127.0.0.1:8081/v1 --skip-extraction --structured --limit 21 --output local-p1.json
```

The endpoint is restricted to loopback. `--skip-extraction` is a fast P1 screen,
not a full competition-path result.
