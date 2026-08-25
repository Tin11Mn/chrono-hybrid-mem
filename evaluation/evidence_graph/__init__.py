"""Offline diagnostics for the P3 evidence graph.

The metrics exported here deliberately have no dependency on :mod:`app`, so
cached model output and synthetic fixtures remain independently scoreable.
The package also contains optional production-integration runners that import
the graph parser and storage implementation explicitly.
"""

from .metrics import (
    bridge_recall_at_k,
    chain_recall_at_k,
    cross_user_leakage,
    entity_link_precision,
    entity_link_recall,
    entity_precision,
    entity_recall,
    evaluate_extraction_quality,
    evaluate_retrieval_cases,
    evidence_coverage_at_k,
    false_temporal_annotation_count,
    false_temporal_annotation_rate,
    graph_only_recovered_count,
    graph_only_recovered_evidence,
    predicate_normalization_accuracy,
    provenance_accuracy,
    relation_f1,
    relation_precision,
    relation_recall,
    state_aware_relation_f1,
    state_aware_relation_precision,
    state_aware_relation_recall,
    temporal_state_accuracy,
    unsupported_edge_rate,
)

__all__ = [
    "bridge_recall_at_k",
    "chain_recall_at_k",
    "cross_user_leakage",
    "entity_link_precision",
    "entity_link_recall",
    "entity_precision",
    "entity_recall",
    "evaluate_extraction_quality",
    "evaluate_retrieval_cases",
    "evidence_coverage_at_k",
    "false_temporal_annotation_count",
    "false_temporal_annotation_rate",
    "graph_only_recovered_count",
    "graph_only_recovered_evidence",
    "predicate_normalization_accuracy",
    "provenance_accuracy",
    "relation_f1",
    "relation_precision",
    "relation_recall",
    "state_aware_relation_f1",
    "state_aware_relation_precision",
    "state_aware_relation_recall",
    "temporal_state_accuracy",
    "unsupported_edge_rate",
]
