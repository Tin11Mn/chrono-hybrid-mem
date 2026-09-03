from evaluation.aml_synthetic.evaluate import FixturePlanModel
from evaluation.aml_synthetic.run_p2_paired import freeze_plans, run_p2_paired


def test_frozen_plan_paired_runner_reuses_one_plan_per_case(tmp_path):
    cases = [
        {"id": "a", "category": "A", "query": "Which tea?", "options": [],
         "structured_plan": {"core_terms": ["tea"], "expansion_terms": [], "entities": [], "temporal_cues": [], "evidence_needs": ["tea"]},
         "memories": [{"id": "mem_1", "content": "Tea is jasmine."}], "required_evidence_ids": ["mem_1"], "forbidden_evidence_ids": []},
        {"id": "b", "category": "B", "query": "Which trail?", "options": [],
         "structured_plan": {"core_terms": ["trail"], "expansion_terms": [], "entities": [], "temporal_cues": [], "evidence_needs": ["trail"]},
         "memories": [{"id": "mem_1", "content": "The trail is north."}], "required_evidence_ids": ["mem_1"], "forbidden_evidence_ids": []},
    ]
    delegate = FixturePlanModel(cases); plans = freeze_plans(cases, delegate)
    result = run_p2_paired(cases, work_dir=tmp_path, rank_model=delegate, plans=plans)
    assert delegate.call_count == 6
    assert result["calls"] == {"baseline_plan_reads": 2, "p2_plan_reads": 2, "baseline_rank": 2, "p2_rank": 2}
    assert result["baseline"]["gpt_calls"] == result["p2_on"]["gpt_calls"] == 2
