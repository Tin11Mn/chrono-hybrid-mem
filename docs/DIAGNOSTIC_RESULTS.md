# Hybrid retrieval diagnostic results

Run: GitHub Actions `31172617528` on 2026-08-07  
Comparison: immutable `v0.2.0` versus current `main` at commit `491af0b`  
Data: three fictional, hand-authored diagnostic cases in `examples/release_comparison_diagnostic.json`  
Configuration: `top_k=1`, no external model call

| Metric | v0.2.0 | Current hybrid retrieval | Change |
|---|---:|---:|---:|
| Recall@1 | 0.3333 | 1.0000 | +0.6667 |
| Case coverage@1 | 0.3333 | 1.0000 | +0.6667 |
| MRR | 0.3333 | 1.0000 | +0.6667 |
| Average Search latency | 0.723 ms | 0.767 ms | +0.044 ms |

The two cases missed by `v0.2.0` contain conflicting preferences. One asks for the current state when the newer event was ingested first; the other asks for the historical state when the older event was ingested first. The released version returned the later-ingested evidence at rank one in both cases. Current code retrieves a broader candidate pool before Reciprocal Rank Fusion and applies an explicit current- or historical-query event-time bonus, returning the intended source evidence at rank one.

These figures establish a regression result for this specific temporal-conflict mechanism. They are not a leaderboard score, do not use challenge data, and are too small to estimate generalization. The competition result remains subject to the platform's hidden evaluation.
