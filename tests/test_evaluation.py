from app.evaluation import evaluate_cases


def test_offline_evaluator_reports_perfect_retrieval(tmp_path):
    report = evaluate_cases(
        [{
            "case_id": "case-1",
            "user_id": "user-a",
            "session_id": "session-a",
            "messages": [{"role": "user", "content": "Mina prefers tea."}],
            "query": "What does Mina prefer?",
            "expected_evidence": ["Mina prefers tea"],
        }],
        str(tmp_path / "evaluation.db"),
        [1, 10],
    )
    assert report["recall_at_k"] == {"1": 1.0, "10": 1.0}
    assert report["case_coverage_at_k"] == {"1": 1.0, "10": 1.0}
    assert report["mrr"] == 1.0
