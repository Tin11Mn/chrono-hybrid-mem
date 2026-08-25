import json
from pathlib import Path
import threading

import pytest

from evaluation.evidence_graph.generate_cases import CATEGORIES, generate, validate
from evaluation.evidence_graph.metrics import (
    bridge_recall_at_k,
    chain_recall_at_k,
    cross_user_leakage,
    entity_link_precision,
    entity_link_recall,
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
from evaluation.evidence_graph.run_extraction import evaluate_predictions, run_cases


FIXTURE_PATH = (
    Path(__file__).parents[1] / "evaluation" / "evidence_graph" / "cases.json"
)


def _entities(prefix="gold"):
    return [
        {
            "entity_id": f"{prefix}-person",
            "user_id": "user-a",
            "canonical_name": "Alice",
            "entity_type": "person",
        },
        {
            "entity_id": f"{prefix}-place",
            "user_id": "user-a",
            "canonical_name": "Cedar Vale",
            "entity_type": "location",
        },
    ]


def _edge(prefix="gold", **overrides):
    value = {
        "relation_id": f"{prefix}-relation",
        "user_id": "user-a",
        "subject_entity_id": f"{prefix}-person",
        "predicate": "lives_in",
        "object_entity_id": f"{prefix}-place",
        "source_message_id": "message-1",
        "state_change": "update",
        "temporal_status": "current",
        "supersedes_edge_id": "old-edge",
    }
    value.update(overrides)
    return value


def test_relation_metrics_resolve_independent_entity_ids_and_separate_error_types():
    gold_entities = _entities("gold")
    predicted_entities = _entities("pred")
    gold = [_edge("gold")]
    correct = [_edge("pred")]

    assert relation_precision(
        gold, correct, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 1.0
    assert unsupported_edge_rate(
        gold, correct, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 0.0
    assert provenance_accuracy(
        gold, correct, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 1.0

    wrong_predicate = [_edge("pred", predicate="located_in")]
    assert predicate_normalization_accuracy(
        gold, wrong_predicate, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 0.0
    assert relation_precision(
        gold, wrong_predicate, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 0.0
    assert unsupported_edge_rate(
        gold, wrong_predicate, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 1.0

    wrong_source = [_edge("pred", source_message_id="message-2")]
    assert relation_precision(
        gold, wrong_source, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 1.0
    assert provenance_accuracy(
        gold, wrong_source, gold_entities=gold_entities,
        predicted_entities=predicted_entities,
    ) == 0.0


def test_relation_recall_and_f1_prevent_sparse_high_precision_predictions():
    gold_entities = _entities("gold")
    predicted_entities = _entities("pred")
    gold = [
        _edge("gold", relation_id=f"gold-{index}")
        for index in range(4)
    ]
    predicted = [_edge("pred")]

    common = {
        "gold_entities": gold_entities,
        "predicted_entities": predicted_entities,
    }
    assert relation_precision(gold, predicted, **common) == 1.0
    assert relation_recall(gold, predicted, **common) == 0.25
    assert relation_f1(gold, predicted, **common) == pytest.approx(0.4)

    # Counter semantics also penalize a duplicate prediction when only one
    # occurrence exists in gold.
    assert relation_precision(
        [gold[0]], [predicted[0], predicted[0]], **common
    ) == 0.5
    assert relation_recall(
        [gold[0]], [predicted[0], predicted[0]], **common
    ) == 1.0


def test_entity_and_pairwise_link_recall_measure_omitted_gold_items():
    gold_entities = _entities("gold")
    predicted_entities = [_entities("pred")[0]]
    assert entity_recall(gold_entities, predicted_entities) == 0.5

    gold_mentions = [
        {"mention_id": mention_id, "entity_id": "gold-cluster"}
        for mention_id in ("m1", "m2", "m3")
    ]
    predicted_mentions = [
        {"mention_id": "m1", "entity_id": "pred-cluster"},
        {"mention_id": "m2", "entity_id": "pred-cluster"},
        {"mention_id": "m3", "entity_id": "split-away"},
    ]

    assert entity_link_precision(gold_mentions, predicted_mentions) == 1.0
    assert entity_link_recall(gold_mentions, predicted_mentions) == pytest.approx(1 / 3)


def test_state_aware_metrics_penalize_false_current_annotation_on_plain_gold():
    gold_entities = _entities("gold")
    predicted_entities = _entities("pred")
    gold = [_edge(
        "gold",
        state_change="assert",
        temporal_status=None,
        supersedes_edge_id=None,
    )]
    predicted = [_edge(
        "pred",
        state_change="assert",
        temporal_status="current",
        supersedes_edge_id=None,
    )]
    common = {
        "gold_entities": gold_entities,
        "predicted_entities": predicted_entities,
    }

    # The semantic edge is correct, but adding a temporal label unsupported by
    # the gold state must not receive state-aware credit.
    assert relation_precision(gold, predicted, **common) == 1.0
    assert temporal_state_accuracy(gold, predicted, **common) == 1.0
    assert state_aware_relation_precision(gold, predicted, **common) == 0.0
    assert state_aware_relation_recall(gold, predicted, **common) == 0.0
    assert state_aware_relation_f1(gold, predicted, **common) == 0.0
    assert false_temporal_annotation_count(gold, predicted, **common) == 1
    assert false_temporal_annotation_rate(gold, predicted, **common) == 1.0


def test_state_metrics_bind_each_state_to_its_source_provenance():
    gold_entities = _entities("gold")
    predicted_entities = _entities("pred")
    gold = [
        _edge(
            "gold", source_message_id="message-1", state_change="assert",
            temporal_status=None, supersedes_edge_id=None,
        ),
        _edge(
            "gold", source_message_id="message-2", state_change="retraction",
            temporal_status="previous", supersedes_edge_id=None,
        ),
    ]
    predicted = [
        _edge(
            "pred", source_message_id="message-1", state_change="assert",
            temporal_status=None, supersedes_edge_id=None,
        ),
        _edge(
            "pred", source_message_id="message-2", state_change="retraction",
            temporal_status="previous", supersedes_edge_id=None,
        ),
    ]
    common = {
        "gold_entities": gold_entities,
        "predicted_entities": predicted_entities,
    }

    assert state_aware_relation_precision(gold, predicted, **common) == 1.0
    assert state_aware_relation_recall(gold, predicted, **common) == 1.0
    assert false_temporal_annotation_count(gold, predicted, **common) == 0

    swapped = [
        dict(predicted[0], state_change="retraction", temporal_status="previous"),
        dict(predicted[1], state_change="assert", temporal_status=None),
    ]
    assert state_aware_relation_precision(gold, swapped, **common) == 0.0
    assert state_aware_relation_recall(gold, swapped, **common) == 0.0
    assert false_temporal_annotation_count(gold, swapped, **common) == 1


def test_extraction_report_scores_temporal_link_and_cross_user_boundaries():
    gold_entities = _entities("gold")
    predicted_entities = _entities("pred")
    gold_mentions = [
        {"mention_id": "m1", "entity_id": "person-a", "user_id": "user-a"},
        {"mention_id": "m2", "entity_id": "person-a", "user_id": "user-a"},
        {"mention_id": "m3", "entity_id": "person-b", "user_id": "user-a"},
    ]
    predicted_mentions = [
        {"mention_id": "m1", "entity_id": "merged", "user_id": "user-a"},
        {"mention_id": "m2", "entity_id": "merged", "user_id": "user-a"},
        {"mention_id": "m3", "entity_id": "merged", "user_id": "user-a"},
    ]
    predicted_edge = _edge("pred", temporal_status="previous")
    report = evaluate_extraction_quality(
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
        gold_relations=[_edge("gold")],
        predicted_relations=[predicted_edge],
        gold_mentions=gold_mentions,
        predicted_mentions=predicted_mentions,
        expected_user_id="user-a",
    )

    assert report["entity_precision"] == 1.0
    assert report["entity_recall"] == 1.0
    assert report["relation_precision"] == 1.0
    assert report["relation_recall"] == 1.0
    assert report["relation_f1"] == 1.0
    assert report["temporal_state_accuracy"] == 0.0
    assert report["entity_link_precision"] == pytest.approx(1 / 3)
    assert report["entity_link_recall"] == 1.0
    assert report["state_aware_relation_precision"] == 0.0
    assert report["state_aware_relation_recall"] == 0.0
    assert report["state_aware_relation_f1"] == 0.0
    assert report["false_temporal_annotation_count"] == 0
    assert report["false_temporal_annotation_rate"] == 0.0
    assert report["cross_user_leakage"] == 0

    leaking_entities = [
        *predicted_entities,
        {"entity_id": "other", "user_id": "user-b", "canonical_name": "Bob", "entity_type": "person"},
    ]
    leaking_edge = _edge("pred", object_entity_id="other")
    assert cross_user_leakage(
        [leaking_edge], predicted_entities=leaking_entities,
        expected_user_id="user-a",
    ) == 1


def test_link_precision_rejects_duplicate_occurrence_ids():
    mentions = [
        {"mention_id": "m1", "entity_id": "person-a"},
        {"mention_id": "m1", "entity_id": "person-b"},
    ]

    with pytest.raises(ValueError, match="predicted mention IDs must be unique"):
        entity_link_precision([], mentions)


def test_empty_prediction_conventions_do_not_pass_positive_extraction_cases():
    gold_entities = _entities("gold")
    gold_edges = [_edge("gold")]
    assert relation_precision(gold_edges, [], gold_entities=gold_entities) == 0.0
    assert relation_recall(gold_edges, [], gold_entities=gold_entities) == 0.0
    assert relation_f1(gold_edges, [], gold_entities=gold_entities) == 0.0
    assert entity_recall(gold_entities, []) == 0.0
    assert provenance_accuracy(gold_edges, [], gold_entities=gold_entities) == 0.0
    assert predicate_normalization_accuracy(
        gold_edges, [], gold_entities=gold_entities
    ) == 0.0
    assert unsupported_edge_rate(gold_edges, [], gold_entities=gold_entities) == 0.0
    assert temporal_state_accuracy(gold_edges, [], gold_entities=gold_entities) == 0.0
    assert state_aware_relation_recall(
        gold_edges, [], gold_entities=gold_entities
    ) == 0.0
    assert false_temporal_annotation_rate(
        gold_edges, [], gold_entities=gold_entities
    ) == 0.0


def test_category_relation_recall_exposes_collapse_hidden_by_global_average():
    generated = generate()
    strong_cases = [
        case for case in generated if case["category"] == "person_location"
    ][:9]
    collapsed_case = next(
        case for case in generated if case["category"] == "membership"
    )
    cases = [*strong_cases, collapsed_case]
    predictions = []
    for case in strong_cases:
        predictions.append({
            "case_id": case["case_id"],
            "entities": case["gold"]["entities"],
            "relations": case["gold"]["relations"],
            "mentions": case["gold"]["mentions"],
        })
    predictions.append({
        "case_id": collapsed_case["case_id"],
        "entities": [],
        "relations": [],
        "mentions": [],
    })

    overall, by_category = evaluate_predictions(cases, predictions)

    assert overall["relation_recall"] == 0.9
    assert by_category["person_location"]["relation_recall"] == 1.0
    assert by_category["membership"]["relation_recall"] == 0.0


def test_chain_coverage_bridge_and_graph_only_formulas():
    gold = ["m1", "m2"]
    ranked = ["m1", "noise", "m2", "m2"]

    assert chain_recall_at_k(gold, ranked, 2) == 0.0
    assert chain_recall_at_k(gold, ranked, 3) == 1.0
    assert evidence_coverage_at_k(gold, ranked, 2) == 0.5
    assert evidence_coverage_at_k(gold, ranked, 3) == 1.0
    assert bridge_recall_at_k(["m1"], ranked, 1) == 1.0
    assert graph_only_recovered_evidence(
        gold, ["m1", "noise"], ["m2", "other"]
    ) == frozenset({"m2"})
    assert graph_only_recovered_count(
        gold, ["m1", "noise"], ["m2", "other"]
    ) == 1
    assert graph_only_recovered_count(
        gold, ["m1", "noise"], ["noise", "m2"], k=1
    ) == 0


@pytest.mark.parametrize("function", [chain_recall_at_k, evidence_coverage_at_k, bridge_recall_at_k])
def test_retrieval_metrics_reject_empty_gold_and_invalid_k(function):
    with pytest.raises(ValueError, match="gold evidence"):
        function([], ["m1"], 1)
    with pytest.raises(ValueError, match="positive integer"):
        function(["m1"], ["m1"], 0)


def test_retrieval_case_aggregation_uses_case_scoped_recovery_counts():
    report = evaluate_retrieval_cases(
        [
            {
                "case_id": "one",
                "gold_evidence_ids": ["m1", "m2"],
                "ranked_evidence_ids": ["m1", "x", "m2"],
                "bridge_evidence_ids": ["m1"],
                "baseline_evidence_ids": ["m1"],
                "graph_evidence_ids": ["m2"],
            },
            {
                "case_id": "two",
                "gold_evidence_ids": ["m2"],
                "ranked_evidence_ids": ["m2"],
                "baseline_evidence_ids": [],
                "graph_evidence_ids": ["m2"],
            },
        ],
        k_values=(1, 3),
    )

    assert report["chain_recall_at_k"] == {"1": 0.5, "3": 1.0}
    assert report["evidence_coverage_at_k"] == {"1": 0.75, "3": 1.0}
    assert report["bridge_recall_at_k"] == {"1": 1.0, "3": 1.0}
    assert report["bridge_annotated_cases"] == 1
    assert report["graph_only_recovered_evidence_count"] == 2
    assert report["graph_only_recovered_case_count"] == 2


def test_generated_p3_zero_suite_is_deterministic_complete_and_checked_in():
    first = generate()
    second = generate()
    counts = validate(first)
    checked_in = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert first == second == checked_in
    assert len(first) == 140
    assert counts == {category: 10 for category in sorted(CATEGORIES)}
    assert {case["category"] for case in first} == set(CATEGORIES)
    assert all(case["synthetic"] is True for case in first)
    assert all(case["gold"]["expected_cross_user_leakage"] == 0 for case in first)


def test_diagnostic_suite_contains_required_adversarial_and_temporal_structures():
    cases = generate()
    by_category = {
        category: [case for case in cases if case["category"] == category]
        for category in CATEGORIES
    }

    assert all(case["gold"]["forbidden_edges"] for case in by_category["adversarial"])
    assert all("prompt-injection" in case["tags"] for case in by_category["adversarial"])
    assert all(len({message["user_id"] for message in case["messages"]}) == 2 for case in by_category["cross_user"])
    assert all(len({entity["entity_id"] for entity in case["gold"]["entities"] if entity["entity_type"] == "person"}) == 2 for case in by_category["same_name"])
    assert all(any(edge["supersedes_edge_id"] for edge in case["gold"]["relations"]) for case in by_category["temporal_update"])
    assert all(not any(edge["supersedes_edge_id"] for edge in case["gold"]["relations"]) for case in by_category["temporal_parallel"])


def test_extraction_runner_calls_model_once_per_message_and_preserves_source_ids():
    case = next(item for item in generate() if item["category"] == "person_location")
    gold_entities = {item["entity_id"]: item for item in case["gold"]["entities"]}
    gold_relation = case["gold"]["relations"][0]
    subject = gold_entities[gold_relation["subject_entity_id"]]
    obj = gold_entities[gold_relation["object_entity_id"]]

    class FakeModel:
        def __init__(self):
            self.call_count = 0

        def extract_memory(self, content, speaker="", timestamp=None):
            self.call_count += 1
            return {
                "facts": [],
                "entities": [
                    {"name": subject["display_name"], "type": subject["entity_type"]},
                    {"name": obj["display_name"], "type": obj["entity_type"]},
                ],
                "relations": [{
                    "subject": subject["display_name"],
                    "subject_type": subject["entity_type"],
                    "relation": gold_relation["predicate"],
                    "object": obj["display_name"],
                    "object_type": obj["entity_type"],
                    "explicit": True,
                    "state_change": "assert",
                    "temporal_status": None,
                }],
            }

    report = run_cases([case], FakeModel())
    prediction = report["predictions"][0]

    assert report["model_calls"] == report["messages"] == 1
    assert report["one_call_per_message"] is True
    assert report["metrics"]["relation_precision"] == 1.0
    assert report["metrics"]["provenance_accuracy"] == 1.0
    assert {edge["source_message_id"] for edge in prediction["relations"]} == {
        case["messages"][0]["id"]
    }


def test_runner_aligns_unique_occurrences_without_changing_predicted_clusters():
    correction = next(
        item for item in generate() if item["category"] == "correction"
    )

    class SequenceModel:
        def __init__(self, payloads):
            self.payloads = payloads
            self.call_count = 0

        def extract_memory(self, content, speaker="", timestamp=None):
            payload = self.payloads[self.call_count]
            self.call_count += 1
            return payload

    payloads = [
        {
            "facts": [],
            "entities": [
                {"name": "Amina", "type": "person"},
                {"name": "Aster Works", "type": "organization"},
            ],
            "relations": [{
                "subject": "Amina",
                "subject_type": "person",
                "relation": "works_at",
                "object": "Aster Works",
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        },
        {
            "facts": [],
            "entities": [
                {"name": "Amina", "type": "person"},
                {"name": "Beacon Lab", "type": "organization"},
            ],
            "relations": [{
                "subject": "Amina",
                "subject_type": "person",
                "relation": "works_at",
                "object": "Beacon Lab",
                "object_type": "organization",
                "explicit": True,
                "state_change": "correction",
                "temporal_status": "current",
            }],
        },
    ]

    report = run_cases([correction], SequenceModel(payloads))
    prediction = report["predictions"][0]

    assert report["metrics"]["entity_link_precision"] == 1.0
    assert {mention["mention_id"] for mention in prediction["mentions"]} == {
        mention["mention_id"] for mention in correction["gold"]["mentions"]
    }
    amina_groups = {
        mention["entity_id"]
        for mention in prediction["mentions"]
        if mention["surface"] == "Amina"
    }
    assert len(amina_groups) == 1


def test_runner_uses_source_verified_appositives_to_split_same_name_endpoints():
    same_name = next(
        item for item in generate() if item["category"] == "same_name"
    )

    class SequenceModel:
        def __init__(self, payloads):
            self.payloads = payloads
            self.call_count = 0

        def extract_memory(self, content, speaker="", timestamp=None):
            payload = self.payloads[self.call_count]
            self.call_count += 1
            return payload

    merged_payloads = [
        {
            "entities": [
                {"name": "Jordan 1", "type": "person"},
                {"name": "Aster Works", "type": "organization"},
            ],
            "relations": [{
                "subject": "Jordan 1",
                "subject_type": "person",
                "relation": "works_at",
                "object": "Aster Works",
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        },
        {
            "entities": [
                {"name": "Jordan 1", "type": "person"},
                {"name": "Beacon Lab", "type": "organization"},
            ],
            "relations": [{
                "subject": "Jordan 1",
                "subject_type": "person",
                "relation": "works_at",
                "object": "Beacon Lab",
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        },
    ]
    distinct_payloads = [
        {
            "entities": [
                {
                    "name": "Jordan 1",
                    "type": "person",
                    "identity_hint": "designer",
                },
                {"name": "Aster Works", "type": "organization"},
            ],
            "relations": merged_payloads[0]["relations"],
        },
        {
            "entities": [
                {
                    "name": "Jordan 1",
                    "type": "person",
                    "identity_hint": "chemist",
                },
                {"name": "Beacon Lab", "type": "organization"},
            ],
            "relations": merged_payloads[1]["relations"],
        },
    ]

    source_verified = run_cases([same_name], SequenceModel(merged_payloads))
    distinct = run_cases([same_name], SequenceModel(distinct_payloads))

    # The authoritative fixture text says "the designer" / "the chemist" in
    # the same clause.  The parser derives those hints rather than trusting or
    # requiring the model to provide them.
    assert source_verified["metrics"]["entity_link_precision"] == 1.0
    assert source_verified["metrics"]["entity_link_recall"] == 1.0
    assert distinct["metrics"]["entity_link_precision"] == 1.0


def test_extraction_runner_parallel_workers_preserve_order_and_call_count():
    cases = generate()[:4]

    class FakeModel:
        def __init__(self):
            self.call_count = 0
            self.lock = threading.Lock()

        def extract_memory(self, content, speaker="", timestamp=None):
            with self.lock:
                self.call_count += 1
            return {"facts": [], "entities": [], "relations": []}

    report = run_cases(cases, FakeModel(), workers=4)

    assert report["workers"] == 4
    assert report["one_call_per_message"] is True
    assert report["model_calls"] == report["messages"]
    assert [item["case_id"] for item in report["predictions"]] == [
        item["case_id"] for item in cases
    ]
