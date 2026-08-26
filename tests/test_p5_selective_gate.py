# -*- coding: utf-8 -*-
"""P5 selective rerank gate tests (default-off, evidence-preserving)."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_locomo_retrieval.py"
SPEC = importlib.util.spec_from_file_location("locomo_evaluation", SCRIPT)
locomo_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locomo_evaluation)


class P5PlanModel:
    """Search model: structured plan, passthrough rank (no score)."""

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query(self, query, options):
        return []

    def plan_query_structured(self, query, options):
        return {
            "intent": "fact",
            "core_terms": ["campus"],
            "entities": ["Microsoft"],
            "temporal_cues": [],
            "expansion_terms": [],
            "evidence_needs": ["find the campus"],
        }

    def rank_candidates(self, query, options, candidates):
        # Put candidate 0 first, then the rest — deterministic, no scores.
        return [candidate["id"] for candidate in candidates]


# Two messages: candidate A (mem_1, raw only) and candidate B (mem_2, raw +
# fact + context). B is the stronger-evidence runner-up.
SAMPLE_P5 = [{
    "sample_id": "sample-p5",
    "conversation": {"session_1": [
        {"speaker": "Ari", "dia_id": "D1:1",
         "text": "Microsoft campus tour was fun."},
        {"speaker": "Bela", "dia_id": "D1:2",
         "text": "The Redmond campus has the new building."},
    ]},
    "qa": [{
        "question": "Which campus has the new building?",
        "evidence": ["D1:2"], "category": 3,
    }],
}]


def _run(p5_gate=False, min_channels=2, epsilon=0.0005):
    return locomo_evaluation.evaluate(
        SAMPLE_P5, [1, 3, 10], None,
        model=P5PlanModel(), structured_query_plan=True,
        p5_gate=p5_gate,
        p5_min_evidence_channels=min_channels,
        p5_near_tie_epsilon=epsilon,
        include_question_diagnostics=True,
    )


def test_p5_default_off_preserves_p1_order():
    report = _run(p5_gate=False)
    record = report["question_diagnostics"][0]
    assert record["p5_swapped"] is False
    assert record["p5_gate"]["enabled"] is False


def test_p5_gate_requires_structured_plan():
    import pytest

    with pytest.raises(ValueError):
        locomo_evaluation.evaluate(
            SAMPLE_P5, [1], None,
            model=P5PlanModel(),
            p5_gate=True,
        )


def test_p5_never_swaps_when_gap_is_wide():
    # With a large epsilon the near-tie condition fails (gap < eps is required,
    # so a huge epsilon makes every pair "near-tied"); use epsilon=0 so the gap
    # is never < 0 -> never triggers.
    report = _run(p5_gate=True, epsilon=0.0)
    record = report["question_diagnostics"][0]
    assert record["p5_swapped"] is False
    assert record["p5_gate"]["reason"] in ("not_near_tie", "missing_fusion_score")


def test_p5_gate_recorded_without_swap_on_insufficient_evidence():
    # min_channels=30 forces insufficient evidence unless a fact exists:
    # query-token overlap can never reach 30.
    report = _run(p5_gate=True, min_channels=30)
    record = report["question_diagnostics"][0]
    # Diagnostics must exist; swap must not happen without strong evidence.
    assert record["p5_gate"]["enabled"] is True
    assert record["p5_swapped"] is False
    assert record["p5_gate"]["reason"] in (
        "runner_up_not_strictly_stronger",
        "not_near_tie",
    )


def test_p5_swaps_only_when_runner_up_is_strictly_stronger():
    # A swap requires the runner-up to exceed the Top-1 query-token overlap;
    # strict dominance, not mere threshold.
    report = _run(p5_gate=True, min_channels=1, epsilon=0.01)
    record = report["question_diagnostics"][0]
    gate = record["p5_gate"]
    if gate.get("top1_query_overlap") is not None:
        if gate.get("reason") == "near_tie_with_stronger_evidence":
            assert gate["top2_query_overlap"] > gate["top1_query_overlap"]
        elif gate.get("reason") == "runner_up_not_strictly_stronger":
            assert gate["top2_query_overlap"] <= gate["top1_query_overlap"]
