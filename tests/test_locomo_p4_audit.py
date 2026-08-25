import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_locomo_retrieval.py"
SPEC = importlib.util.spec_from_file_location("locomo_evaluation", SCRIPT)
locomo_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locomo_evaluation)


SAMPLE_SINGLE = [{
    "sample_id": "sample-diag",
    "conversation": {"session_1": [
        {"speaker": "Ari", "dia_id": "D1:1", "text": "Milo prefers jasmine tea."},
    ]},
    "qa": [{
        "question": "What tea does Milo prefer?", "answer": "jasmine tea",
        "evidence": ["D1:1"], "category": 1,
    }],
}]


def test_locomo_question_diagnostics_default_off_keeps_aggregate_identical():
    without = locomo_evaluation.evaluate(SAMPLE_SINGLE, [1, 3], None)
    with_flag = locomo_evaluation.evaluate(
        SAMPLE_SINGLE, [1, 3], None, include_question_diagnostics=True
    )

    assert "question_diagnostics" not in without
    aggregate = dict(with_flag)
    diagnostics = aggregate.pop("question_diagnostics")
    assert aggregate == without
    assert len(diagnostics) == 1


def test_locomo_question_diagnostics_exports_per_question_records():
    report = locomo_evaluation.evaluate(
        SAMPLE_SINGLE, [1, 3], None, include_question_diagnostics=True
    )

    diagnostics = report["question_diagnostics"]
    assert len(diagnostics) == report["questions"] == 1
    record = diagnostics[0]
    assert record["question_offset"] == 0
    assert record["category"] == "1"
    assert record["first_gold_rank"] == 1
    assert record["failure_bucket"] == "top1_hit"
    assert record["recall_bucket"] is None
    assert record["result_ids"]
    assert record["gold_mem_ids"] == ["mem_1"]
    assert record["gold_channel_presence"]
    assert record["gold_counterfactual_positions"] == {"mem_1": 1}
    assert record["final_ids"]


def test_locomo_question_diagnostics_classifies_reranker_drop():
    fillers = [
        {
            "speaker": "Ari",
            "dia_id": "D2:{}".format(index),
            "text": "CommonWord filler{:02d}".format(index),
        }
        for index in range(1, 12)
    ]
    sample = [{
        "sample_id": "sample-drop",
        "conversation": {
            "session_1_date_time": "1:15 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Bela", "dia_id": "D1:1", "text": "CommonWord gold evidence."},
            ],
            "session_2_date_time": "2:30 pm on 9 May, 2023",
            "session_2": fillers,
        },
        "qa": [{"question": "CommonWord", "evidence": ["D1:1"], "category": 1}],
    }]

    report = locomo_evaluation.evaluate(
        sample, [1, 3, 10], None, include_question_diagnostics=True
    )

    record = report["question_diagnostics"][0]
    assert record["first_gold_rank"] is None
    assert record["failure_bucket"] == "recall_miss_top10"
    assert record["recall_bucket"] == "reranker_drop"
    assert record["gold_mem_ids"] == ["mem_1"]
    assert record["gold_counterfactual_positions"]["mem_1"] == 12
    assert record["gold_pool_positions"]["mem_1"] == 12
    assert "mem_1" not in record["final_ids"]


def test_locomo_question_diagnostics_classifies_channel_miss():
    sample = [{
        "sample_id": "sample-miss",
        "conversation": {
            "session_1_date_time": "1:15 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Bela", "dia_id": "D1:1", "text": "Unrelated gold text."},
            ],
            "session_2_date_time": "2:30 pm on 9 May, 2023",
            "session_2": [
                {"speaker": "Ari", "dia_id": "D1:2", "text": "Zqzx marker text."},
            ],
        },
        "qa": [{"question": "Zqzx", "evidence": ["D1:1"], "category": 1}],
    }]

    report = locomo_evaluation.evaluate(
        sample, [1, 3, 10], None, include_question_diagnostics=True
    )

    record = report["question_diagnostics"][0]
    assert record["first_gold_rank"] is None
    assert record["failure_bucket"] == "recall_miss_top10"
    assert record["recall_bucket"] == "channel_miss"
    assert record["gold_channel_presence"] == {}


def test_recall_failure_classifier_all_buckets():
    channel_found = {
        "raw": ["mem_1", "mem_7", "mem_2"],
        "raw_porter": ["mem_1", "mem_7"],
    }

    assert locomo_evaluation._classify_recall_failure(
        ["mem_9"], {"p1_channels": channel_found}
    ) == "channel_miss"
    assert locomo_evaluation._classify_recall_failure(
        ["mem_7"], {
            "p1_channels": channel_found,
            "p1_counterfactual_top30_ids": ["mem_1", "mem_2"],
        }
    ) == "fusion_miss"
    assert locomo_evaluation._classify_recall_failure(
        ["mem_7"], {
            "p1_channels": channel_found,
            "p1_counterfactual_top30_ids": ["mem_7", "mem_1"],
            "rerank_pool_ids": ["mem_1", "mem_2"],
        }
    ) == "quota_displacement"
    assert locomo_evaluation._classify_recall_failure(
        ["mem_7"], {
            "p1_channels": channel_found,
            "p1_counterfactual_top30_ids": ["mem_7", "mem_1"],
            "rerank_pool_ids": ["mem_7", "mem_1"],
        }
    ) == "reranker_drop"
    assert locomo_evaluation._classify_recall_failure(
        [], {"p1_channels": channel_found}
    ) is None
    assert locomo_evaluation._classify_recall_failure(
        ["mem_7"], {}
    ) is None


def test_failure_bucket_mapping():
    assert locomo_evaluation._failure_bucket(1) == "top1_hit"
    assert locomo_evaluation._failure_bucket(2) == "ranking_top3"
    assert locomo_evaluation._failure_bucket(3) == "ranking_top3"
    assert locomo_evaluation._failure_bucket(4) == "ranking_top10"
    assert locomo_evaluation._failure_bucket(10) == "ranking_top10"
    assert locomo_evaluation._failure_bucket(11) == "recall_miss_top10"
    assert locomo_evaluation._failure_bucket(None) == "recall_miss_top10"
