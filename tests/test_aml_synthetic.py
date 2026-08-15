from evaluation.aml_synthetic.generate_cases import generate, validate
from evaluation.aml_synthetic.compare import compare


def test_aml_synthetic_suite_has_thirty_valid_cases_per_category():
    cases = generate()
    counts = validate(cases)

    assert len(cases) == 210
    assert counts == {category: 30 for category in "ABCDEGH"}
    assert all(case["structured_plan"]["core_terms"] for case in cases)


def test_aml_comparison_gate_requires_gain_and_preserves_c_e_h():
    baseline = {
        "cases": 210,
        "model": "fixture-plan-identity-ranker",
        "cross_user_leakage": 0,
        "gpt_calls": 420,
        "hit_at_k": {"1": 0.4},
        "category_hit_at_k": {
            category: {"1": 0.4} for category in "ABCDEGH"
        },
        "multi_hop_all_evidence_coverage_at_10": 0.8,
        "forbidden_evidence_at_1": 0.2,
        "search_latency_seconds_mean": 1.0,
    }
    experiment = {
        **baseline,
        "hit_at_k": {"1": 0.5},
        "category_hit_at_k": {
            **baseline["category_hit_at_k"],
            "A": {"1": 0.5},
            "B": {"1": 0.4},
        },
        "forbidden_evidence_at_1": 0.1,
        "search_latency_seconds_mean": 1.1,
    }

    assert compare(baseline, experiment)["decision"] == "MECHANICS_PASS"
    experiment["category_hit_at_k"]["H"] = {"1": 0.3}
    rejected = compare(baseline, experiment)
    assert rejected["decision"] == "REJECT"
    assert "C, E, or H Hit@1 regressed" in rejected["failures"]
