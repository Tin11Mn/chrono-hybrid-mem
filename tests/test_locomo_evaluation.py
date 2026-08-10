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
