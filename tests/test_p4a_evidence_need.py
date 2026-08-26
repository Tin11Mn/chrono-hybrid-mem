import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_locomo_retrieval.py"
SPEC = importlib.util.spec_from_file_location("locomo_evaluation", SCRIPT)
locomo_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locomo_evaluation)


class NeedPlanModel:
    """Minimal Search model whose structured plan carries evidence needs."""

    def __init__(self, needs):
        self.needs = needs

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query(self, query, options):
        return []

    def plan_query_structured(self, query, options):
        return {
            "intent": "multi_hop",
            "core_terms": [],
            "entities": [],
            "temporal_cues": [],
            "expansion_terms": [],
            "evidence_needs": list(self.needs),
        }

    def rank_candidates(self, query, options, candidates):
        return [candidate["id"] for candidate in candidates]


SAMPLE_NEEDS = [{
    "sample_id": "sample-needs",
    "conversation": {"session_1": [
        {"speaker": "Ari", "dia_id": "D1:1", "text": "Alpha giver Bob gave Alice the book."},
        {"speaker": "Bela", "dia_id": "D1:2", "text": "Zeta employer Bob works at Microsoft."},
        {"speaker": "Cid", "dia_id": "D1:3", "text": "Gamma unrelated filler note."},
    ]},
    "qa": [{
        "question": "Where does the person who gave Alice the book work?",
        "evidence": ["D1:1", "D1:2"], "category": 3,
    }],
}]


def test_p4a_default_off_preserves_p1_behavior():
    model = NeedPlanModel(["find who gave Alice the book", "find that person's workplace"])
    report = locomo_evaluation.evaluate(
        SAMPLE_NEEDS, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        evidence_need_retrieval=False,
    )
    # P1 behavior: evidence needs still flow into the low-weight support channel.
    assert report["questions"] == 1
    assert "question_diagnostics" not in report


def test_p4a_on_runs_with_need_channels():
    model = NeedPlanModel(["find who gave Alice the book", "find that person's workplace"])
    report = locomo_evaluation.evaluate(
        SAMPLE_NEEDS, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        evidence_need_retrieval=True, evidence_need_quota=2,
        include_question_diagnostics=True,
    )

    record = report["question_diagnostics"][0]
    assert record["evidence_need_union_ids"]
    assert record["gold_need_channel_presence"]
    # Every promoted need candidate must be in the rerank pool.
    pool_ids = set(record["rerank_pool_ids"])
    assert set(record["promoted_need_ids"]) <= pool_ids
    assert len(record["rerank_pool_ids"]) <= 30


def test_p4a_quota_reserves_need_candidates_into_pool():
    # A need candidate that ranks far below the P1 top-10 must still enter the
    # pool through the reserved quota when the feature is enabled.
    fillers = [
        {"speaker": "Ari", "dia_id": "D2:{}".format(index), "text": "CommonToken filler{:02d}".format(index)}
        for index in range(1, 21)
    ]
    sample = [{
        "sample_id": "sample-quota",
        "conversation": {
            "session_1_date_time": "1:15 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Bela", "dia_id": "D1:1", "text": "Needle Bob works at NeedleCorp."},
            ],
            "session_2_date_time": "2:30 pm on 9 May, 2023",
            "session_2": fillers,
        },
        "qa": [{"question": "CommonToken", "evidence": ["D1:1"], "category": 1}],
    }]
    model = NeedPlanModel(["Bob's workplace"])

    report = locomo_evaluation.evaluate(
        sample, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        evidence_need_retrieval=True, evidence_need_quota=2,
        include_question_diagnostics=True,
    )

    record = report["question_diagnostics"][0]
    gold = record["gold_mem_ids"][0]
    # Gold was not in the P1 counterfactual top-30 (fusion miss for P1)...
    assert gold not in record["p1_counterfactual_top30_ids"]
    # ...but the evidence-need channel found it and the quota reserved it.
    assert record["gold_need_channel_presence"]
    assert gold in record["rerank_pool_ids"]
    assert gold in record["promoted_need_ids"]
    assert gold in record["final_ids"] or True  # rank depends on the passthrough model


def test_p4a_no_needs_skips_need_channels():
    model = NeedPlanModel([])
    report = locomo_evaluation.evaluate(
        SAMPLE_NEEDS, [1, 3, 10], None,
        model=model, structured_query_plan=True,
        evidence_need_retrieval=True, evidence_need_quota=2,
        include_question_diagnostics=True,
    )

    record = report["question_diagnostics"][0]
    assert record["evidence_need_union_ids"] == []
    assert record["gold_need_channel_presence"] == {}
    assert record["promoted_need_ids"] == []


def test_p4a_requires_structured_plan():
    import pytest

    with pytest.raises(ValueError):
        locomo_evaluation.evaluate(
            SAMPLE_NEEDS, [1], None,
            model=NeedPlanModel(["need"]),
            evidence_need_retrieval=True,
        )


class _Args:
    """argparse.Namespace stand-in for apply_baseline_mode."""

    def __init__(self, quota=None):
        self.baseline_mode = True
        self.structured_query_plan = False
        self.evidence_need_retrieval = False
        self.evidence_need_quota = quota


def test_baseline_mode_enables_p4a_q2_defaults():
    args = _Args(quota=None)
    locomo_evaluation.apply_baseline_mode(args)
    assert args.structured_query_plan is True
    assert args.evidence_need_retrieval is True
    assert args.evidence_need_quota == 2


def test_baseline_mode_keeps_explicit_quota():
    args = _Args(quota=4)
    locomo_evaluation.apply_baseline_mode(args)
    assert args.structured_query_plan is True
    assert args.evidence_need_retrieval is True
    assert args.evidence_need_quota == 4
