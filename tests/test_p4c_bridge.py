import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.storage import extract_bridge_terms  # noqa: E402

SCRIPT = ROOT / "scripts" / "evaluate_locomo_retrieval.py"
SPEC = importlib.util.spec_from_file_location("locomo_evaluation", SCRIPT)
locomo_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locomo_evaluation)


class BridgePlanModel:
    """Search model whose structured plan is multi-hop with evidence needs."""

    def __init__(self, intent="multi_hop", needs=None):
        self.intent = intent
        self.needs = needs or ["identify who gave Alice the book", "find that person's workplace"]

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query(self, query, options):
        return []

    def plan_query_structured(self, query, options):
        return {
            "intent": self.intent,
            "core_terms": [],
            "entities": [],
            "temporal_cues": [],
            "expansion_terms": [],
            "evidence_needs": list(self.needs),
        }

    def rank_candidates(self, query, options, candidates):
        return [candidate["id"] for candidate in candidates]


SAMPLE_BRIDGE = [{
    "sample_id": "sample-bridge",
    "conversation": {
        "session_1": [
            {"speaker": "Ari", "dia_id": "D1:1", "text": "Alpha giver Bob gave Alice the book."},
        ],
        "session_2": [
            {"speaker": "Bela", "dia_id": "D1:2", "text": "Bob is at the Microsoft workplace."},
        ],
        "session_3": [
            {"speaker": "Cid", "dia_id": "D1:3", "text": "Gamma unrelated filler note."},
        ],
    },
    "qa": [{
        "question": "Where does the person who gave Alice the book work?",
        "evidence": ["D1:1", "D1:2"], "category": 3,
    }],
}]


def test_p4c_default_off_preserves_p1_behavior():
    model = BridgePlanModel()
    report = locomo_evaluation.evaluate(
        SAMPLE_BRIDGE, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        bridge_retrieval=False,
    )
    assert report["questions"] == 1


def test_p4c_triggered_on_multi_hop_with_bridge_terms():
    model = BridgePlanModel()
    report = locomo_evaluation.evaluate(
        SAMPLE_BRIDGE, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        bridge_retrieval=True, bridge_max_terms=3,
        include_question_diagnostics=True,
    )

    record = report["question_diagnostics"][0]
    # Bridge extracted "Bob" (speaker in first-pass evidence, absent from query).
    assert "bob" in [term.casefold() for term in record["bridge_terms"]]
    # Second pass must have surfaced the second-hop evidence.
    assert record["gold_bridge_positions"]
    gold = set(record["gold_mem_ids"])
    assert gold.intersection(record["promoted_bridge_ids"])
    # Pool respects the hard limit.
    assert len(record["rerank_pool_ids"]) <= 30


def test_p4c_not_triggered_without_multi_hop_or_needs():
    model = BridgePlanModel(intent="fact", needs=["single simple need"])
    report = locomo_evaluation.evaluate(
        SAMPLE_BRIDGE, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        bridge_retrieval=True, bridge_max_terms=3,
        include_question_diagnostics=True,
    )

    record = report["question_diagnostics"][0]
    assert record["bridge_terms"] == []
    assert record["promoted_bridge_ids"] == []
    assert record["gold_bridge_positions"] == {}


def test_p4c_requires_structured_plan():
    import pytest

    with pytest.raises(ValueError):
        locomo_evaluation.evaluate(
            SAMPLE_BRIDGE, [1], None,
            model=BridgePlanModel(),
            bridge_retrieval=True,
        )


def test_extract_bridge_terms_excludes_speakers_and_extracts_capitalized_spans():
    extract = extract_bridge_terms

    # Known participant names are prefix noise and are excluded.
    terms = extract(
        "Where does the giver work?",
        ["Bob gave Alice the book.", "Bob works at Microsoft."],
        known_speakers=["Bob", "Alice"],
        max_terms=3,
    )
    assert "Bob" not in terms
    assert "Alice" not in terms

    # Capitalized non-participant spans are extracted even from one candidate.
    terms = extract(
        "where does the giver work",
        ["the giver works at Microsoft."],
        known_speakers=[],
        max_terms=3,
    )
    assert "Microsoft" in terms

    # Terms already in the query are excluded.
    terms = extract(
        "Microsoft office",
        ["Bob works at Microsoft.", "Alice likes Microsoft."],
        known_speakers=[],
        max_terms=3,
    )
    assert "Microsoft" not in terms

    # Sentence-initial common words are not extracted.
    terms = extract(
        "question",
        ["The giver gave a book.", "The giver mentioned the plan."],
        known_speakers=[],
        max_terms=3,
    )
    assert not any(term.casefold() in {"the", "a"} for term in terms)
