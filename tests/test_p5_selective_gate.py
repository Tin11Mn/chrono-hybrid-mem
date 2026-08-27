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

    def __init__(self, confidence=None):
        self.confidence = confidence

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

    def rank_candidates_with_confidence(self, query, options, candidates):
        ordered = [candidate["id"] for candidate in candidates]
        if self.confidence is None:
            return ordered, {}
        return ordered, dict(self.confidence)


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
    # epsilon=0 with no confidence -> no near-tie signal -> no swap.
    report = _run(p5_gate=True, epsilon=0.0)
    record = report["question_diagnostics"][0]
    assert record["p5_swapped"] is False
    assert record["p5_gate"]["reason"] in (
        "no_confidence_signal",
        "not_near_tie",
        "missing_fusion_score",
        "strata_excluded",
    )


def test_p5_gate_recorded_without_swap_on_insufficient_evidence():
    # min_channels is irrelevant when confidence is absent: the gate must NOT
    # fall back to retrieval-layer proxies (empirically disproven).
    report = _run(p5_gate=True, min_channels=30)
    record = report["question_diagnostics"][0]
    assert record["p5_gate"]["enabled"] is True
    assert record["p5_swapped"] is False
    assert record["p5_gate"]["reason"] in (
        "no_confidence_signal",
        "not_near_tie",
        "strata_excluded",
    )


def test_p5_swaps_only_when_runner_up_is_strictly_stronger():
    # Without confidence the gate never swaps (no fallback to disproven
    # retrieval-layer proxies).
    report = _run(p5_gate=True, min_channels=1, epsilon=0.01)
    record = report["question_diagnostics"][0]
    gate = record["p5_gate"]
    assert record["p5_swapped"] is False
    assert gate["reason"] in (
        "no_confidence_signal",
        "strata_excluded",
    )


def test_p5_strata_excludes_non_matching_queries():
    # Default strata = temporal,correction. The test query ("Which campus has
    # the new building?") carries no temporal/correction language, so the gate
    # must be excluded by strata before any evidence check.
    report = _run(p5_gate=True, min_channels=1, epsilon=0.01)
    record = report["question_diagnostics"][0]
    gate = record["p5_gate"]
    assert gate["strata_matched"] is False
    assert gate["reason"] == "strata_excluded"
    assert record["p5_swapped"] is False


def test_p5_strata_all_allows_gate():
    # --p5-strata all removes the strata restriction.
    report = locomo_evaluation.evaluate(
        SAMPLE_P5, [1, 3, 10], None,
        model=P5PlanModel(), structured_query_plan=True,
        p5_gate=True, p5_min_evidence_channels=1,
        p5_near_tie_epsilon=0.01, p5_strata="all",
        include_question_diagnostics=True,
    )
    record = report["question_diagnostics"][0]
    gate = record["p5_gate"]
    assert gate["strata_matched"] is True
    assert gate["reason"] not in ("strata_excluded",)


def test_p5_strata_rejects_unknown_values():
    import pytest

    with pytest.raises(ValueError):
        locomo_evaluation.evaluate(
            SAMPLE_P5, [1], None,
            model=P5PlanModel(), structured_query_plan=True,
            p5_gate=True, p5_strata="bogus",
        )


def test_p5_confidence_drives_swap_when_runner_up_is_more_confident():
    # The model ranks mem_2 first (it restates the question) but is LESS sure
    # of it than of mem_1 (0.6 vs 0.8): the runner-up mem_1 is more confident
    # by 0.2 > margin 0.05, so the gate swaps mem_1 to the top.
    model = P5PlanModel(confidence={
        "mem_1": 0.8,
        "mem_2": 0.6,
    })
    report = locomo_evaluation.evaluate(
        SAMPLE_P5, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        p5_gate=True, p5_confidence_margin=0.05, p5_strata="all",
        include_question_diagnostics=True,
    )
    record = report["question_diagnostics"][0]
    gate = record["p5_gate"]
    assert gate["strata_matched"] is True
    # mem_2 is the model's Top-1; mem_1 (runner-up) is more confident -> swap.
    assert gate["top1_id"] == "mem_2"
    assert gate["reason"] == "near_tie_with_stronger_evidence"
    assert record["p5_swapped"] is True
    assert gate["swapped_ids"] == ["mem_1", "mem_2"]


def test_p5_confidence_below_margin_does_not_swap():
    # Runner-up advantage 0.02 < margin 0.05 -> no swap.
    model = P5PlanModel(confidence={
        "mem_1": 0.60,
        "mem_2": 0.62,
    })
    report = locomo_evaluation.evaluate(
        SAMPLE_P5, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        p5_gate=True, p5_confidence_margin=0.05, p5_strata="all",
        include_question_diagnostics=True,
    )
    record = report["question_diagnostics"][0]
    gate = record["p5_gate"]
    assert record["p5_swapped"] is False
    assert gate["reason"] == "runner_up_not_strictly_stronger"
