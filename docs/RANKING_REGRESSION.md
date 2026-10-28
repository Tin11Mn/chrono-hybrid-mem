# Preserve ranking when evidence coverage is incomplete

The selection-enabled retrieval path replaced the existing model ranking call.
When its coverage certificate was incomplete or rejected by specialist obligations,
it restored the lexical results but discarded all model ordering. This changed the
legacy path from model reranking to fusion order, even when the model could identify
useful evidence for part of the query.

The selector now returns `ordered_ids` independently of its completeness decision.
The fallback validates these IDs against the actual candidate allowlist, then reorders
only the existing legacy results. Sources outside that result set are not added;
excluded IDs are not removed; the remaining tail keeps its previous relative order.
The normal `top_k` limit applies after sorting. Legacy providers may express a leading
order in `selected_ids` when they omit `ordered_ids`. An explicitly malformed order
is rejected rather than silently repaired.

This makes no coverage or state-resolution claim. Successful evidence selection still
requires the complete certificate and all specialist obligations. Errors or missing
ordering retain fusion order. Selection-disabled retrieval keeps the existing rank call.
The change introduces no additional model call and returns only original memory text.

The private redacted retrieval trace records `fallback_order_applied` and
`fallback_ordered_ids` separately from `applied`, which continues to mean that evidence
filtering passed validation. These diagnostics contain allowlisted source IDs only.

Independent synthetic tests first reproduced the lost ordering, then verify incomplete
coverage, missing needs, specialist rejection, malformed/unknown IDs, coverage-only
sources, stable membership, output limits and provider errors. These mechanics checks
do not establish an official evaluation score improvement.

The observed smoke comparison (63.54 to 62.02) prompted the investigation. Production
had 9 complete decisions and 37 fallbacks among 46 nonempty searches. This establishes
that the affected path was common; aggregate scores and ID-only traces cannot prove
which individual score differences it caused. Incomplete structured indexing remains
a separate unresolved limitation and is not bypassed by the ranking correction.
