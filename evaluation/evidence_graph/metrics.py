"""Pure metrics for evidence-graph extraction and retrieval.

All functions accept ordinary mappings and sequences and perform no I/O.  The
relation functions optionally resolve ``subject_entity_id`` and
``object_entity_id`` through the supplied entity records, so independently
generated database IDs do not have to match the fixture IDs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import statistics
from typing import Iterable, Mapping, Sequence
import unicodedata


Record = Mapping[str, object]


def _text(value: object) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.strip().casefold().split())


def _first(record: Record, *names: str) -> object:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return ""


def _entity_signature(entity: Record) -> tuple[str, str, str]:
    return (
        _text(_first(entity, "user_id")),
        _text(_first(entity, "canonical_name", "name", "display_name")),
        _text(_first(entity, "entity_type", "type")),
    )


def _entity_index(entities: Sequence[Record]) -> dict[str, tuple[str, str, str]]:
    index: dict[str, tuple[str, str, str]] = {}
    for entity in entities:
        entity_id = _text(_first(entity, "entity_id", "id"))
        if entity_id:
            index[entity_id] = _entity_signature(entity)
    return index


def _endpoint_signature(
    relation: Record,
    endpoint: str,
    entities: Mapping[str, tuple[str, str, str]],
) -> tuple[str, str, str]:
    entity_id = _text(_first(relation, f"{endpoint}_entity_id", f"{endpoint}_id"))
    if entity_id and entity_id in entities:
        return entities[entity_id]

    if endpoint == "object":
        name = _first(relation, "object", "object_value", "target")
        entity_type = _first(relation, "object_type", "target_type")
    else:
        name = _first(relation, "subject", "source")
        entity_type = _first(relation, "subject_type", "source_type")
    return (
        _text(_first(relation, f"{endpoint}_user_id", "user_id")),
        _text(name),
        _text(entity_type),
    )


def _predicate(relation: Record) -> str:
    return _text(_first(relation, "predicate", "relation"))


def _semantic_key(
    relation: Record,
    entities: Mapping[str, tuple[str, str, str]],
) -> tuple[object, ...]:
    return (
        _text(_first(relation, "user_id")),
        _endpoint_signature(relation, "subject", entities),
        _predicate(relation),
        _endpoint_signature(relation, "object", entities),
    )


def _alignment_key(
    relation: Record,
    entities: Mapping[str, tuple[str, str, str]],
) -> tuple[object, ...]:
    return (
        _text(_first(relation, "user_id")),
        _endpoint_signature(relation, "subject", entities),
        _endpoint_signature(relation, "object", entities),
    )


def _provenance_key(
    relation: Record,
    entities: Mapping[str, tuple[str, str, str]],
) -> tuple[object, ...]:
    return _semantic_key(relation, entities) + (
        _text(_first(relation, "source_message_id")),
    )


def _precision_from_counters(
    gold: Counter[object], predicted: Counter[object]
) -> float:
    predicted_count = predicted.total()
    if predicted_count == 0:
        return 1.0 if gold.total() == 0 else 0.0
    supported = sum((gold & predicted).values())
    return supported / predicted_count


def _recall_from_counters(
    gold: Counter[object], predicted: Counter[object]
) -> float:
    """Return multiset recall without allowing duplicate predictions free credit."""

    gold_count = gold.total()
    if gold_count == 0:
        return 1.0
    supported = sum((gold & predicted).values())
    return supported / gold_count


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def entity_precision(
    gold_entities: Sequence[Record], predicted_entities: Sequence[Record]
) -> float:
    """Return extraction precision over user/name/type entity signatures.

    Counters, rather than sets, are used so duplicate or over-merged same-name
    entities cannot receive free credit.
    """

    gold = Counter(_entity_signature(item) for item in gold_entities)
    predicted = Counter(_entity_signature(item) for item in predicted_entities)
    return _precision_from_counters(gold, predicted)


def entity_recall(
    gold_entities: Sequence[Record], predicted_entities: Sequence[Record]
) -> float:
    """Return multiset recall over user/name/type entity signatures."""

    gold = Counter(_entity_signature(item) for item in gold_entities)
    predicted = Counter(_entity_signature(item) for item in predicted_entities)
    return _recall_from_counters(gold, predicted)


def relation_precision(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Return precision of semantic subject/predicate/object relations.

    Provenance and temporal fields are intentionally scored by separate
    metrics.  Entity IDs are resolved to normalized user/name/type signatures.
    """

    gold_index = _entity_index(gold_entities)
    predicted_index = _entity_index(predicted_entities)
    gold = Counter(_semantic_key(item, gold_index) for item in gold_relations)
    predicted = Counter(
        _semantic_key(item, predicted_index) for item in predicted_relations
    )
    return _precision_from_counters(gold, predicted)


def _relation_counters(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> tuple[Counter[object], Counter[object]]:
    gold_index = _entity_index(gold_entities)
    predicted_index = _entity_index(predicted_entities)
    return (
        Counter(_semantic_key(item, gold_index) for item in gold_relations),
        Counter(
            _semantic_key(item, predicted_index)
            for item in predicted_relations
        ),
    )


def relation_recall(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Return multiset recall of semantic subject/predicate/object relations."""

    gold, predicted = _relation_counters(
        gold_relations,
        predicted_relations,
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    )
    return _recall_from_counters(gold, predicted)


def relation_f1(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Return the harmonic mean of strict multiset relation precision/recall."""

    gold, predicted = _relation_counters(
        gold_relations,
        predicted_relations,
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    )
    return _f1(
        _precision_from_counters(gold, predicted),
        _recall_from_counters(gold, predicted),
    )


def unsupported_edge_rate(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Return the fraction of predicted edges unsupported by the gold text."""

    if not predicted_relations:
        return 0.0
    return 1.0 - relation_precision(
        gold_relations,
        predicted_relations,
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    )


def predicate_normalization_accuracy(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Score predicates only where the predicted endpoints match a gold pair.

    Predictions with unsupported endpoints are handled by relation precision and
    unsupported-edge rate.  If gold contains relations but no predicted endpoint
    pair can be aligned, the accuracy is zero.
    """

    gold_index = _entity_index(gold_entities)
    predicted_index = _entity_index(predicted_entities)
    gold_by_alignment: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    for relation in gold_relations:
        gold_by_alignment[_alignment_key(relation, gold_index)][_predicate(relation)] += 1

    predicted_by_alignment: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    for relation in predicted_relations:
        key = _alignment_key(relation, predicted_index)
        if key in gold_by_alignment:
            predicted_by_alignment[key][_predicate(relation)] += 1

    denominator = sum(counter.total() for counter in predicted_by_alignment.values())
    if denominator == 0:
        return 1.0 if not gold_relations else 0.0
    correct = sum(
        sum((gold_by_alignment[key] & predicted).values())
        for key, predicted in predicted_by_alignment.items()
    )
    return correct / denominator


def provenance_accuracy(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Score source-message provenance among semantically supported edges."""

    gold_index = _entity_index(gold_entities)
    predicted_index = _entity_index(predicted_entities)
    gold_semantics = Counter(_semantic_key(item, gold_index) for item in gold_relations)
    aligned_predictions = [
        item
        for item in predicted_relations
        if _semantic_key(item, predicted_index) in gold_semantics
    ]
    if not aligned_predictions:
        return 1.0 if not gold_relations else 0.0
    gold = Counter(_provenance_key(item, gold_index) for item in gold_relations)
    predicted = Counter(
        _provenance_key(item, predicted_index) for item in aligned_predictions
    )
    return sum((gold & predicted).values()) / len(aligned_predictions)


def _linked_pairs(mentions: Sequence[Record]) -> set[tuple[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        mention_id = _text(_first(mention, "mention_id", "id"))
        entity_id = _text(_first(mention, "entity_id", "link_group"))
        if mention_id and entity_id:
            grouped[entity_id].append(mention_id)
    pairs: set[tuple[str, str]] = set()
    for mention_ids in grouped.values():
        unique = sorted(set(mention_ids))
        for left_index, left in enumerate(unique):
            for right in unique[left_index + 1 :]:
                pairs.add((left, right))
    return pairs


def entity_link_precision(
    gold_mentions: Sequence[Record], predicted_mentions: Sequence[Record]
) -> float:
    """Return pairwise precision of mention-to-entity clustering.

    Mention IDs must be stable between gold and prediction.  A predicted entity
    ID shared across users therefore creates false pairs rather than being hidden
    by per-user grouping.
    """

    for label, mentions in (
        ("gold", gold_mentions), ("predicted", predicted_mentions)
    ):
        mention_ids = [
            _text(_first(mention, "mention_id", "id")) for mention in mentions
        ]
        populated_ids = [mention_id for mention_id in mention_ids if mention_id]
        if len(populated_ids) != len(set(populated_ids)):
            raise ValueError(f"{label} mention IDs must be unique")

    gold_pairs = _linked_pairs(gold_mentions)
    predicted_pairs = _linked_pairs(predicted_mentions)
    if not predicted_pairs:
        return 1.0 if not gold_pairs else 0.0
    return len(gold_pairs & predicted_pairs) / len(predicted_pairs)


def entity_link_recall(
    gold_mentions: Sequence[Record], predicted_mentions: Sequence[Record]
) -> float:
    """Return pairwise recall of mention-to-entity clustering.

    A prediction that emits too few mentions or splits a gold cluster therefore
    cannot pass merely because every pair it did predict was correct.
    """

    for label, mentions in (
        ("gold", gold_mentions), ("predicted", predicted_mentions)
    ):
        mention_ids = [
            _text(_first(mention, "mention_id", "id")) for mention in mentions
        ]
        populated_ids = [mention_id for mention_id in mention_ids if mention_id]
        if len(populated_ids) != len(set(populated_ids)):
            raise ValueError(f"{label} mention IDs must be unique")

    gold_pairs = _linked_pairs(gold_mentions)
    if not gold_pairs:
        return 1.0
    predicted_pairs = _linked_pairs(predicted_mentions)
    return len(gold_pairs & predicted_pairs) / len(gold_pairs)


def _temporal_key(
    relation: Record,
    entities: Mapping[str, tuple[str, str, str]],
    relation_sources: Mapping[str, str],
    *,
    include_supersedes: bool,
) -> tuple[object, ...]:
    supersedes_source = _text(_first(relation, "supersedes_source_message_id"))
    if not supersedes_source:
        supersedes_id = _text(
            _first(relation, "supersedes_relation_id", "supersedes_edge_id")
        )
        supersedes_source = relation_sources.get(supersedes_id, supersedes_id)
    state = (
        _text(_first(relation, "state_change") or "assert"),
        _text(_first(relation, "temporal_status")),
    )
    if include_supersedes:
        state += (supersedes_source,)
    return _provenance_key(relation, entities) + state


def _is_temporal(relation: Record) -> bool:
    return bool(
        _text(_first(relation, "temporal_status"))
        or _text(_first(relation, "state_change")) not in {"", "assert"}
        or _first(
            relation,
            "supersedes_source_message_id",
            "supersedes_relation_id",
            "supersedes_edge_id",
        )
    )


def _state_key(
    relation: Record,
    entities: Mapping[str, tuple[str, str, str]],
) -> tuple[object, ...]:
    """Provenance-bound relation identity plus state fields for every edge."""

    return _provenance_key(relation, entities) + (
        _text(_first(relation, "state_change") or "assert"),
        _text(_first(relation, "temporal_status")),
    )


def _state_counters(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> tuple[Counter[object], Counter[object]]:
    gold_index = _entity_index(gold_entities)
    predicted_index = _entity_index(predicted_entities)
    return (
        Counter(_state_key(item, gold_index) for item in gold_relations),
        Counter(_state_key(item, predicted_index) for item in predicted_relations),
    )


def state_aware_relation_precision(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Score semantic relations and state fields over *all* predictions."""

    gold, predicted = _state_counters(
        gold_relations,
        predicted_relations,
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    )
    return _precision_from_counters(gold, predicted)


def state_aware_relation_recall(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Score semantic relations and state fields over *all* gold relations."""

    gold, predicted = _state_counters(
        gold_relations,
        predicted_relations,
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    )
    return _recall_from_counters(gold, predicted)


def state_aware_relation_f1(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Harmonic mean of state-aware multiset precision and recall."""

    gold, predicted = _state_counters(
        gold_relations,
        predicted_relations,
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    )
    return _f1(
        _precision_from_counters(gold, predicted),
        _recall_from_counters(gold, predicted),
    )


def false_temporal_annotation_count(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> int:
    """Count supported non-temporal relation occurrences marked as temporal.

    Counter intersection makes this occurrence-aware: duplicate predictions do
    not consume more gold support than exists.  Unsupported temporal edges are
    already penalized by relation precision and are not double-counted here.
    """

    gold_index = _entity_index(gold_entities)
    predicted_index = _entity_index(predicted_entities)
    non_temporal_gold = Counter(
        _provenance_key(item, gold_index)
        for item in gold_relations
        if not _is_temporal(item)
    )
    temporal_predictions = Counter(
        _provenance_key(item, predicted_index)
        for item in predicted_relations
        if _is_temporal(item)
    )
    return sum((non_temporal_gold & temporal_predictions).values())


def false_temporal_annotation_rate(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
) -> float:
    """Return false temporal annotations as a fraction of all predictions."""

    if not predicted_relations:
        return 0.0
    return false_temporal_annotation_count(
        gold_relations,
        predicted_relations,
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) / len(predicted_relations)


def temporal_state_accuracy(
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    *,
    gold_entities: Sequence[Record] = (),
    predicted_entities: Sequence[Record] = (),
    include_supersedes: bool = False,
) -> float:
    """Return exact temporal-state accuracy over temporal gold relations.

    P3-0 extraction evaluates ``state_change`` and ``temporal_status``.  Set
    ``include_supersedes`` for a later storage-level P3-C audit, where independent
    edge IDs are resolved through their source-message provenance before comparison.
    """

    temporal_gold = [item for item in gold_relations if _is_temporal(item)]
    if not temporal_gold:
        return 1.0
    gold_index = _entity_index(gold_entities)
    predicted_index = _entity_index(predicted_entities)
    gold_sources = {
        _text(_first(item, "relation_id", "edge_id", "id")): _text(
            _first(item, "source_message_id")
        )
        for item in gold_relations
    }
    predicted_sources = {
        _text(_first(item, "relation_id", "edge_id", "id")): _text(
            _first(item, "source_message_id")
        )
        for item in predicted_relations
    }
    gold = Counter(
        _temporal_key(
            item, gold_index, gold_sources,
            include_supersedes=include_supersedes,
        )
        for item in temporal_gold
    )
    predicted = Counter(
        _temporal_key(
            item, predicted_index, predicted_sources,
            include_supersedes=include_supersedes,
        )
        for item in predicted_relations
        if _is_temporal(item)
    )
    return sum((gold & predicted).values()) / len(temporal_gold)


def cross_user_leakage(
    predicted_relations: Sequence[Record],
    *,
    predicted_entities: Sequence[Record] = (),
    expected_user_id: str | None = None,
) -> int:
    """Count graph edges that cross or escape their user boundary.

    An edge is counted once if its own ``user_id`` is not the requested user or
    either resolved endpoint belongs to another user.
    """

    entity_users = {
        _text(_first(entity, "entity_id", "id")): _text(_first(entity, "user_id"))
        for entity in predicted_entities
        if _first(entity, "entity_id", "id")
    }
    expected = _text(expected_user_id) if expected_user_id is not None else None
    leaked = 0
    for relation in predicted_relations:
        edge_user = _text(_first(relation, "user_id"))
        relation_leaks = expected is not None and edge_user != expected
        for endpoint in ("subject", "object"):
            entity_id = _text(
                _first(relation, f"{endpoint}_entity_id", f"{endpoint}_id")
            )
            endpoint_user = entity_users.get(entity_id) or _text(
                _first(relation, f"{endpoint}_user_id")
            )
            if endpoint_user and endpoint_user != edge_user:
                relation_leaks = True
        leaked += int(relation_leaks)
    return leaked


def evaluate_extraction_quality(
    *,
    gold_entities: Sequence[Record],
    predicted_entities: Sequence[Record],
    gold_relations: Sequence[Record],
    predicted_relations: Sequence[Record],
    gold_mentions: Sequence[Record] = (),
    predicted_mentions: Sequence[Record] = (),
    expected_user_id: str | None = None,
) -> dict[str, float | int]:
    """Compute the complete P3-0 quality report for one diagnostic batch."""

    common = {
        "gold_entities": gold_entities,
        "predicted_entities": predicted_entities,
    }
    return {
        "entity_precision": entity_precision(gold_entities, predicted_entities),
        "entity_recall": entity_recall(gold_entities, predicted_entities),
        "relation_precision": relation_precision(
            gold_relations, predicted_relations, **common
        ),
        "relation_recall": relation_recall(
            gold_relations, predicted_relations, **common
        ),
        "relation_f1": relation_f1(
            gold_relations, predicted_relations, **common
        ),
        "predicate_normalization_accuracy": predicate_normalization_accuracy(
            gold_relations, predicted_relations, **common
        ),
        "provenance_accuracy": provenance_accuracy(
            gold_relations, predicted_relations, **common
        ),
        "entity_link_precision": entity_link_precision(
            gold_mentions, predicted_mentions
        ),
        "entity_link_recall": entity_link_recall(
            gold_mentions, predicted_mentions
        ),
        "unsupported_edge_rate": unsupported_edge_rate(
            gold_relations, predicted_relations, **common
        ),
        "temporal_state_accuracy": temporal_state_accuracy(
            gold_relations, predicted_relations, **common
        ),
        "state_aware_relation_precision": state_aware_relation_precision(
            gold_relations, predicted_relations, **common
        ),
        "state_aware_relation_recall": state_aware_relation_recall(
            gold_relations, predicted_relations, **common
        ),
        "state_aware_relation_f1": state_aware_relation_f1(
            gold_relations, predicted_relations, **common
        ),
        "false_temporal_annotation_count": false_temporal_annotation_count(
            gold_relations, predicted_relations, **common
        ),
        "false_temporal_annotation_rate": false_temporal_annotation_rate(
            gold_relations, predicted_relations, **common
        ),
        "cross_user_leakage": cross_user_leakage(
            predicted_relations,
            predicted_entities=predicted_entities,
            expected_user_id=expected_user_id,
        ),
    }


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _id_set(values: Iterable[object]) -> set[str]:
    ids: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            identifier = _first(
                value, "source_message_id", "evidence_id", "message_id", "id"
            )
        else:
            identifier = value
        normalized = str(identifier).strip() if identifier is not None else ""
        if normalized:
            ids.add(normalized)
    return ids


def _top_ids(ranked: Sequence[object], k: int | None) -> set[str]:
    selected = ranked if k is None else ranked[:k]
    return _id_set(selected)


def _require_gold(gold_evidence_ids: Iterable[object]) -> set[str]:
    gold = _id_set(gold_evidence_ids)
    if not gold:
        raise ValueError("gold evidence must not be empty")
    return gold


def chain_recall_at_k(
    gold_evidence_ids: Iterable[object], ranked_evidence_ids: Sequence[object], k: int
) -> float:
    """Return 1 only when *all* gold evidence is present in the top ``k``."""

    _validate_k(k)
    gold = _require_gold(gold_evidence_ids)
    return float(gold <= _top_ids(ranked_evidence_ids, k))


def evidence_coverage_at_k(
    gold_evidence_ids: Iterable[object], ranked_evidence_ids: Sequence[object], k: int
) -> float:
    """Return the fraction of distinct gold evidence present in the top ``k``."""

    _validate_k(k)
    gold = _require_gold(gold_evidence_ids)
    return len(gold & _top_ids(ranked_evidence_ids, k)) / len(gold)


def bridge_recall_at_k(
    bridge_evidence_ids: Iterable[object], ranked_evidence_ids: Sequence[object], k: int
) -> float:
    """Return coverage of the annotated first-hop/bridge evidence at ``k``."""

    _validate_k(k)
    bridges = _require_gold(bridge_evidence_ids)
    return len(bridges & _top_ids(ranked_evidence_ids, k)) / len(bridges)


def graph_only_recovered_evidence(
    gold_evidence_ids: Iterable[object],
    baseline_evidence_ids: Sequence[object],
    graph_evidence_ids: Sequence[object],
    *,
    k: int | None = None,
) -> frozenset[str]:
    """Return gold IDs found by graph retrieval but absent from P1 retrieval."""

    if k is not None:
        _validate_k(k)
    gold = _require_gold(gold_evidence_ids)
    baseline = _top_ids(baseline_evidence_ids, k)
    graph = _top_ids(graph_evidence_ids, k)
    return frozenset((gold & graph) - baseline)


def graph_only_recovered_count(
    gold_evidence_ids: Iterable[object],
    baseline_evidence_ids: Sequence[object],
    graph_evidence_ids: Sequence[object],
    *,
    k: int | None = None,
) -> int:
    """Count gold source messages uniquely recovered through the graph channel."""

    return len(
        graph_only_recovered_evidence(
            gold_evidence_ids,
            baseline_evidence_ids,
            graph_evidence_ids,
            k=k,
        )
    )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def evaluate_retrieval_cases(
    cases: Sequence[Record], *, k_values: Sequence[int] = (3, 10)
) -> dict[str, object]:
    """Aggregate evidence-chain metrics over retrieval-result records.

    Required case fields are ``gold_evidence_ids`` and ``ranked_evidence_ids``.
    ``bridge_evidence_ids`` is optional and bridge averages exclude unannotated
    cases.  Graph-only recovery uses the optional ``baseline_evidence_ids`` and
    ``graph_evidence_ids`` fields and counts case-scoped evidence identities.
    """

    if not cases:
        raise ValueError("at least one retrieval case is required")
    unique_k = tuple(dict.fromkeys(k_values))
    if not unique_k:
        raise ValueError("at least one k value is required")
    for k in unique_k:
        _validate_k(k)

    chains: dict[int, list[float]] = {k: [] for k in unique_k}
    coverage: dict[int, list[float]] = {k: [] for k in unique_k}
    bridges: dict[int, list[float]] = {k: [] for k in unique_k}
    recovered: set[tuple[str, str]] = set()
    recovered_cases: set[str] = set()

    for index, case in enumerate(cases):
        case_id = str(_first(case, "case_id", "id") or f"case-{index}")
        gold = case.get("gold_evidence_ids", ())
        ranked = case.get("ranked_evidence_ids", ())
        if not isinstance(gold, Iterable) or isinstance(gold, (str, bytes)):
            raise TypeError("gold_evidence_ids must be an iterable of IDs")
        if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
            raise TypeError("ranked_evidence_ids must be a sequence of IDs")
        gold = tuple(gold)
        for k in unique_k:
            chains[k].append(chain_recall_at_k(gold, ranked, k))
            coverage[k].append(evidence_coverage_at_k(gold, ranked, k))

        bridge_ids = tuple(case.get("bridge_evidence_ids", ()))
        if bridge_ids:
            for k in unique_k:
                bridges[k].append(bridge_recall_at_k(bridge_ids, ranked, k))

        baseline_ids = case.get("baseline_evidence_ids", ())
        graph_ids = case.get("graph_evidence_ids", ())
        if graph_ids:
            for evidence_id in graph_only_recovered_evidence(
                gold, baseline_ids, graph_ids
            ):
                recovered.add((case_id, evidence_id))
                recovered_cases.add(case_id)

    return {
        "cases": len(cases),
        "chain_recall_at_k": {str(k): _mean(chains[k]) for k in unique_k},
        "evidence_coverage_at_k": {
            str(k): _mean(coverage[k]) for k in unique_k
        },
        "bridge_recall_at_k": {
            str(k): _mean(bridges[k]) for k in unique_k
        },
        "bridge_annotated_cases": max((len(values) for values in bridges.values()), default=0),
        "graph_only_recovered_evidence_count": len(recovered),
        "graph_only_recovered_case_count": len(recovered_cases),
    }
