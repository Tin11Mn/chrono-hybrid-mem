from copy import deepcopy

import pytest

from evaluation.evidence_graph.generate_cases import generate
from evaluation.evidence_graph.run_extraction import (
    _aligned_predicted_mentions,
    _event_timestamp,
    _score_case,
    rescore_cached_report,
    run_cases,
)


def _entity(entity_id, name, entity_type, source):
    return {
        "entity_id": entity_id,
        "user_id": "user-1",
        "canonical_name": name.casefold(),
        "display_name": name,
        "entity_type": entity_type,
        "first_source_message_id": source,
    }


def _relation(relation_id, subject_id, object_id, source):
    return {
        "relation_id": relation_id,
        "user_id": "user-1",
        "subject_entity_id": subject_id,
        "predicate": "works_at",
        "object_entity_id": object_id,
        "source_message_id": source,
        "state_change": "assert",
        "temporal_status": None,
        "explicit": True,
    }


def _mention(mention_id, entity_id, source, surface):
    return {
        "mention_id": mention_id,
        "entity_id": entity_id,
        "user_id": "user-1",
        "source_message_id": source,
        "surface": surface,
    }


def test_link_alignment_ignores_type_but_entity_metric_still_penalizes_it():
    gold_person = _entity("gold-person", "Alice", "person", "m1")
    gold_org = _entity("gold-org", "Acme", "organization", "m1")
    case = {
        "case_id": "type-separation",
        "category": "unit",
        "gold": {
            "entities": [gold_person, gold_org],
            "relations": [
                _relation("gold-r1", "gold-person", "gold-org", "m1"),
                _relation("gold-r2", "gold-person", "gold-org", "m2"),
            ],
            "mentions": [
                _mention("gold-a1", "gold-person", "m1", "Alice"),
                _mention("gold-o1", "gold-org", "m1", "Acme"),
                _mention("gold-a2", "gold-person", "m2", "Alice"),
                _mention("gold-o2", "gold-org", "m2", "Acme"),
            ],
        },
    }
    predicted_entities = [
        _entity("pred-a1", "Alice", "person", "m1"),
        _entity("pred-o1", "Acme", "group", "m1"),
        _entity("pred-a2", "Alice", "person", "m2"),
        _entity("pred-o2", "Acme", "group", "m2"),
    ]
    predicted_relations = [
        _relation("pred-r1", "pred-a1", "pred-o1", "m1"),
        _relation("pred-r2", "pred-a2", "pred-o2", "m2"),
    ]
    prediction = {
        "entities": predicted_entities,
        "relations": predicted_relations,
        "mentions": _aligned_predicted_mentions(
            case, predicted_entities, predicted_relations
        ),
    }

    metrics = _score_case(case, prediction)

    assert metrics["entity_link_precision"] == 1.0
    assert metrics["entity_precision"] == 0.5
    assert {item["mention_id"] for item in prediction["mentions"]} == {
        "gold-a1",
        "gold-o1",
        "gold-a2",
        "gold-o2",
    }


def test_type_independent_occurrence_alignment_still_penalizes_same_name_merge():
    jordan_one = _entity("gold-jordan-1", "Jordan", "person", "m1")
    jordan_two = _entity("gold-jordan-2", "Jordan", "person", "m2")
    org_one = _entity("gold-org-1", "Aster Works", "organization", "m1")
    org_two = _entity("gold-org-2", "Beacon Lab", "organization", "m2")
    case = {
        "case_id": "same-name-separation",
        "category": "unit",
        "gold": {
            "entities": [jordan_one, jordan_two, org_one, org_two],
            "relations": [
                _relation("gold-r1", "gold-jordan-1", "gold-org-1", "m1"),
                _relation("gold-r2", "gold-jordan-2", "gold-org-2", "m2"),
            ],
            "mentions": [
                _mention("gold-j1", "gold-jordan-1", "m1", "Jordan"),
                _mention("gold-o1", "gold-org-1", "m1", "Aster Works"),
                _mention("gold-j2", "gold-jordan-2", "m2", "Jordan"),
                _mention("gold-o2", "gold-org-2", "m2", "Beacon Lab"),
            ],
        },
    }
    predicted_entities = [
        _entity("pred-j1", "Jordan", "person", "m1"),
        _entity("pred-o1", "Aster Works", "organization", "m1"),
        _entity("pred-j2", "Jordan", "person", "m2"),
        _entity("pred-o2", "Beacon Lab", "organization", "m2"),
    ]
    predicted_relations = [
        _relation("pred-r1", "pred-j1", "pred-o1", "m1"),
        _relation("pred-r2", "pred-j2", "pred-o2", "m2"),
    ]
    prediction = {
        "entities": predicted_entities,
        "relations": predicted_relations,
        "mentions": _aligned_predicted_mentions(
            case, predicted_entities, predicted_relations
        ),
    }

    metrics = _score_case(case, prediction)

    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_link_precision"] == 0.0


def _gold_payload(case):
    relation = case["gold"]["relations"][0]
    entities = {
        item["entity_id"]: item for item in case["gold"]["entities"]
    }
    subject = entities[relation["subject_entity_id"]]
    obj = entities[relation["object_entity_id"]]
    return {
        "facts": [],
        "entities": [
            {"name": subject["display_name"], "type": subject["entity_type"]},
            {"name": obj["display_name"], "type": obj["entity_type"]},
        ],
        "relations": [{
            "subject": subject["display_name"],
            "subject_type": subject["entity_type"],
            "relation": relation["predicate"],
            "object": obj["display_name"],
            "object_type": obj["entity_type"],
            "explicit": True,
            "state_change": relation.get("state_change", "assert"),
            "temporal_status": relation.get("temporal_status"),
        }],
    }


def test_cached_report_rescore_reparses_raw_without_model_calls():
    case = next(item for item in generate() if item["category"] == "employment")

    class CountingModel:
        def __init__(self):
            self.call_count = 0

        def extract_memory(self, content, speaker="", timestamp=None):
            self.call_count += 1
            return _gold_payload(case)

    model = CountingModel()
    cached = run_cases([case], model)
    assert model.call_count == 1
    cached["predictions"][0]["entities"] = []
    cached["predictions"][0]["relations"] = []
    cached["predictions"][0]["mentions"] = []
    cached["predictions"][0]["messages"][0]["parsed_graph"] = {
        "stale": True
    }
    cached["model"] = "cached-model"
    cached["base_url"] = "http://cached.invalid/v1"
    raw_before = deepcopy(
        cached["predictions"][0]["messages"][0]["raw_extraction"]
    )

    rescored = rescore_cached_report(
        generate(), cached, rescored_from="cached-report.json"
    )

    assert model.call_count == 1
    assert rescored["rescored_from"] == "cached-report.json"
    assert rescored["model_calls_delta"] == 0
    assert rescored["model_calls"] == 1
    assert rescored["original_runtime"]["model_calls"] == 1
    assert rescored["original_runtime"]["model"] == "cached-model"
    assert "zero model calls" in rescored["runtime_note"]
    assert rescored["metrics"]["relation_precision"] == 1.0
    assert rescored["metrics"]["provenance_accuracy"] == 1.0
    message = rescored["predictions"][0]["messages"][0]
    relation = rescored["predictions"][0]["relations"][0]
    assert message["raw_extraction"] == raw_before
    assert message["parsed_graph"].get("stale") is None
    assert relation["source_message_id"] == case["messages"][0]["id"]
    assert relation["user_id"] == case["messages"][0]["user_id"]
    assert relation["event_ts"] == _event_timestamp(
        case["messages"][0]["timestamp"]
    )


def test_cached_report_rescore_fails_loudly_when_raw_extraction_is_missing():
    case = next(item for item in generate() if item["category"] == "employment")

    class OneShotModel:
        call_count = 0

        def extract_memory(self, content, speaker="", timestamp=None):
            self.call_count += 1
            return _gold_payload(case)

    cached = run_cases([case], OneShotModel())
    del cached["predictions"][0]["messages"][0]["raw_extraction"]

    with pytest.raises(ValueError, match="missing raw_extraction"):
        rescore_cached_report(
            [case], cached, rescored_from="broken-report.json"
        )
