# Hybrid retrieval diagnostic results

Run: GitHub Actions `31171050875` on 2026-08-07  
Comparison: immutable `v0.2.0` versus current `main` at commit `49b2ada`  
Data: two fictional, hand-authored diagnostic cases in `examples/release_comparison_diagnostic.json`  
Configuration: `top_k=1`, no external model call

| Metric | v0.2.0 | Current hybrid retrieval | Change |
|---|---:|---:|---:|
| Recall@1 | 0.50 | 1.00 | +0.50 |
| Case coverage@1 | 0.50 | 1.00 | +0.50 |
| MRR | 0.50 | 1.00 | +0.50 |
| Average Search latency | 0.586 ms | 0.616 ms | +0.030 ms |

The case missed by `v0.2.0` contains two conflicting preferences: the newer event was ingested first and the older event second. The released version returned the later-ingested old evidence at rank one. Current code retrieves a broader candidate pool before Reciprocal Rank Fusion and applies the explicit-current-query temporal bonus, returning the newer source evidence at rank one.

These figures establish a regression result for this specific temporal-conflict mechanism. They are not a leaderboard score, do not use challenge data, and are too small to estimate generalization. The competition result remains subject to the platform's hidden evaluation.
