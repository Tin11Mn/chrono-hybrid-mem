# -*- coding: utf-8 -*-
"""Selective evidence-graph gate tests (default-off, P3-A regression fix)."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_locomo_retrieval.py"
SPEC = importlib.util.spec_from_file_location("locomo_evaluation", SCRIPT)
locomo_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locomo_evaluation)


class GraphPlanModel:
    """Structured plan carrying entities + intent."""

    def __init__(self, intent="fact", entities=None):
        self.intent = intent
        self.entities = entities or ["Alice"]

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query(self, query, options):
        return []

    def plan_query_structured(self, query, options):
        return {
            "intent": self.intent,
            "core_terms": ["works"],
            "entities": list(self.entities),
            "temporal_cues": [],
            "expansion_terms": [],
            "evidence_needs": ["find the employer"],
        }

    def rank_candidates(self, query, options, candidates):
        return [candidate["id"] for candidate in candidates]


SAMPLE_GRAPH = [{
    "sample_id": "sample-graph",
    "conversation": {"session_1": [
        {"speaker": "Ari", "dia_id": "D1:1", "text": "Alice works at Acme."},
        {"speaker": "Bela", "dia_id": "D1:2", "text": "Bob works at Beta."},
        {"speaker": "Cid", "dia_id": "D1:3", "text": "Gamma unrelated filler."},
    ]},
    "qa": [{
        "question": "Where does the person who knows Alice work?",
        "evidence": ["D1:1", "D1:2"], "category": 3,
    }],
}]


def _run(evidence_graph=False, graph_selective=False, intent="fact", entities=None):
    return locomo_evaluation.evaluate(
        SAMPLE_GRAPH, [1, 3, 10], None,
        model=GraphPlanModel(intent=intent, entities=entities),
        structured_query_plan=True,
        evidence_graph=evidence_graph,
        graph_selective=graph_selective,
        include_question_diagnostics=True,
    )


def test_graph_default_off_preserves_p1():
    report = _run()
    record = report["question_diagnostics"][0]
    assert record["graph_selective"]["enabled"] is False
    assert record["graph_selective"]["graph_candidate_ids"] == []


def test_graph_unconditional_runs_on_any_plan():
    report = _run(evidence_graph=True, intent="fact")
    record = report["question_diagnostics"][0]
    assert record["graph_selective"]["enabled"] is False
    assert record["graph_selective"]["triggered"] is True
    assert record["graph_selective"]["reason"] == "unconditional"


def test_graph_selective_triggers_on_multi_hop():
    report = _run(evidence_graph=True, graph_selective=True, intent="multi_hop")
    record = report["question_diagnostics"][0]
    assert record["graph_selective"]["triggered"] is True
    assert record["graph_selective"]["reason"] == "multi_hop_intent"


def test_graph_selective_skips_fact_intent_with_few_entities():
    report = _run(evidence_graph=True, graph_selective=True, intent="fact",
                  entities=["Alice"])
    record = report["question_diagnostics"][0]
    assert record["graph_selective"]["triggered"] is False
    assert record["graph_selective"]["reason"] == "not_multi_hop_not_entity_dense"


def test_graph_selective_triggers_on_entity_dense():
    report = _run(evidence_graph=True, graph_selective=True, intent="fact",
                  entities=["Alice", "Bob"])
    record = report["question_diagnostics"][0]
    assert record["graph_selective"]["triggered"] is True
    assert record["graph_selective"]["reason"] == "entity_dense"


def test_graph_selective_requires_evidence_graph():
    import pytest

    with pytest.raises(ValueError):
        _run(graph_selective=True)


def test_evidence_graph_requires_structured_plan():
    import pytest

    with pytest.raises(ValueError):
        locomo_evaluation.evaluate(
            SAMPLE_GRAPH, [1], None,
            model=GraphPlanModel(),
            evidence_graph=True,
        )
