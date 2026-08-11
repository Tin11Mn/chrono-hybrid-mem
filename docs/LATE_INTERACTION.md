# Late-interaction retrieval (local research option)

This repository now exposes an optional token-level late-interaction first-stage retriever. Set `MEMORY_LOCAL_LATE_INTERACTION_MODEL` to a FastEmbed-compatible late-interaction model and set `MEMORY_DENSE_FUSION_ALPHA` to enable z-score lexical/dense fusion. The late-interaction model is mutually exclusive with `MEMORY_LOCAL_EMBEDDING_MODEL`.

The retriever computes token-level MaxSim scores between the query and each stored memory, then uses the existing FTS5/Porter lexical channel for score-level fusion. It is local-development only; the competition deployment remains the platform-provided `gpt-4o-mini` path.

Example:

```text
MEMORY_LOCAL_LATE_INTERACTION_MODEL=answerdotai/answerai-colbert-small-v1
MEMORY_DENSE_FUSION_ALPHA=0.4
MEMORY_LOCAL_EMBEDDING_DEVICE=cpu
```

The `0.4` value is a starting point motivated by the recent LoCoMo late-interaction/BM25 study, not a result measured by this repository. Run the fixed smoke and 200-question gates before any full-set claim. The published study reports Hit@1=0.752 for its own e5-large-v2/session-level protocol; that number is not copied into this project's metrics.

Initial local smoke testing with cached `colbert-ir/colbertv2.0` on 20 LoCoMo questions reached Hit@1=0.50 at the best tested fusion weight (`alpha=0.2`). This is below the existing Qwen3-Reranker smoke gate (0.60), so this candidate is not promoted to a 200-question run without a stronger model or revised turn/session aggregation.
