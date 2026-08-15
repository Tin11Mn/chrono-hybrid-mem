import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_locomo_retrieval.py"
SPEC = importlib.util.spec_from_file_location("locomo_evaluation", SCRIPT)
locomo_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locomo_evaluation)


def test_locomo_evaluator_uses_annotated_dialog_evidence():
    report = locomo_evaluation.evaluate(
        [{
            "sample_id": "sample-1",
            "conversation": {
                "session_1_date_time": "ignored",
                "session_1": [
                    {"speaker": "Ari", "dia_id": "D1:1", "text": "Milo prefers jasmine tea."},
                    {"speaker": "Bela", "dia_id": "D1:2", "text": "Bela likes hiking."},
                ],
            },
            "qa": [{
                "question": "What tea does Milo prefer?",
                "answer": "jasmine tea",
                "evidence": ["D1:1"],
                "category": 1,
            }],
        }],
        [1, 3],
        None,
    )

    assert report["questions"] == 1
    assert report["hit_at_k"] == {"1": 1.0, "3": 1.0}
    assert report["evidence_recall_at_k"] == {"1": 1.0, "3": 1.0}
    assert report["mrr"] == 1.0


def test_locomo_released_baseline_is_available_for_comparison():
    sample = [{
        "sample_id": "sample-2",
        "conversation": {"session_1": [{"speaker": "Ari", "dia_id": "D1:1", "text": "Milo prefers tea."}]},
        "qa": [{"question": "What does Milo prefer?", "evidence": ["D1:1"], "category": 1}],
    }]

    report = locomo_evaluation.evaluate(sample, [1], None, retriever="v0.2.0")

    assert report["hit_at_k"] == {"1": 1.0}


def test_locomo_evaluator_preserves_original_session_boundaries():
    sessions, evidence = locomo_evaluation.sessions_and_evidence({
        "conversation": {
            "session_2_date_time": "2:30 pm on 9 May, 2023",
            "session_2": [
                {"speaker": "Ari", "dia_id": "D2:1", "text": "Second session."},
            ],
            "session_1_date_time": "1:15 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ari", "dia_id": "D1:1", "text": "First session."},
            ],
        },
    })

    assert [session_id for session_id, _ in sessions] == ["session_1", "session_2"]
    assert [messages[0]["timestamp"] for _, messages in sessions] == [1683551700, 1683642600]
    assert set(evidence) == {"D1:1", "D2:1"}


def test_locomo_evaluator_preserves_official_image_caption_but_not_search_query():
    sessions, evidence = locomo_evaluation.sessions_and_evidence({
        "conversation": {
            "session_1": [{
                "speaker": "Ari",
                "dia_id": "D1:1",
                "text": "Take a look at this.",
                "blip_caption": "a sunrise painting over a lake",
                "query": "private image search terms",
            }],
        },
    })

    expected = "Ari: Take a look at this. Shared image: a sunrise painting over a lake"
    assert sessions[0][1][0]["content"] == expected
    assert evidence["D1:1"] == expected
    assert "private image search terms" not in expected


def test_locomo_evaluator_can_resume_from_an_eligible_question_offset():
    sample = [{
        "sample_id": "sample-offset",
        "conversation": {"session_1": [
            {"speaker": "Ari", "dia_id": "D1:1", "text": "Milo prefers tea."},
            {"speaker": "Ari", "dia_id": "D1:2", "text": "Milo prefers coffee."},
        ]},
        "qa": [
            {"question": "First?", "evidence": ["D1:1"], "category": 1},
            {"question": "What does Milo prefer, coffee?", "evidence": ["D1:2"], "category": 2},
        ],
    }]

    report = locomo_evaluation.evaluate(sample, [1], 1, question_offset=1)

    assert report["questions"] == 1
    assert report["question_offset"] == 1
    assert report["next_question_offset"] == 2
    assert report["raw_counts"]["category_counts"] == {
        "2": {"questions": 1, "hit_at_1": 1, "hit_counts": {"1": 1}}
    }
    assert report["category_hit_at_k"] == {"2": {"1": 1.0}}


def test_locomo_mrr_uses_the_best_rank_across_multiple_evidence_turns():
    report = locomo_evaluation.evaluate([{
        "sample_id": "sample-multi",
        "conversation": {"session_1": [
            {"speaker": "Ari", "dia_id": "D1:1", "text": "Rareword weak evidence."},
            {"speaker": "Ari", "dia_id": "D1:2", "text": "Rareword direct evidence."},
        ]},
        "qa": [{
            "question": "Rareword direct?", "evidence": ["D1:1", "D1:2"], "category": 1,
        }],
    }], [1, 2], None)

    assert report["hit_at_k"]["1"] == 1.0
    assert report["mrr"] == 1.0
    assert report["raw_counts"]["hit_counts"]["1"] == 1
