import pytest

from app.main import (
    create_app,
    dense_fusion_alpha_from_environment,
    dense_context_weight_from_environment,
    dense_time_weight_from_environment,
    rerank_top_n_from_environment,
    rerank_fusion_weight_from_environment,
    session_fusion_weight_from_environment,
    session_top_n_from_environment,
    instruction_rerank_top_n_from_environment,
    instruction_refine_top_n_from_environment,
    structured_query_plan_from_environment,
)
from app.model import MemoryModel, model_from_environment
from app.local_semantic import local_reranker_from_environment
from app.schemas import AddRequest
from app.storage import MemoryStore


class FakeMemoryModel:
    def extract_facts(self, content):
        return ["Mina prefers tea in the afternoon."] if "tea" in content else []

    def rank_candidates(self, query, options, candidates):
        return [candidate["id"] for candidate in candidates]

    def plan_query(self, query, options):
        return ["Mina", "prefers tea"]


class FakeStructuredMemoryModel(FakeMemoryModel):
    def __init__(self):
        self.flat_plan_calls = 0
        self.structured_plan_calls = 0

    def plan_query(self, query, options):
        self.flat_plan_calls += 1
        return ["stamps"]

    def plan_query_structured(self, query, options):
        self.structured_plan_calls += 1
        return {
            "intent": "fact",
            "core_terms": ["Mina", "tea"],
            "expansion_terms": ["beverage"],
            "entities": ["Mina"],
            "temporal_cues": [],
            "evidence_needs": [],
        }


def test_competition_mode_requires_a_secret(monkeypatch):
    monkeypatch.setenv("MEMORY_REQUIRE_MODEL", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        model_from_environment()


def test_structured_query_plan_defaults_on_and_reads_false(monkeypatch):
    monkeypatch.delenv("MEMORY_STRUCTURED_QUERY_PLAN", raising=False)
    assert structured_query_plan_from_environment() is True
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "FaLsE")
    assert structured_query_plan_from_environment() is False


def test_structured_query_plan_is_bounded_and_sanitized():
    model = MemoryModel.__new__(MemoryModel)
    model._json_response = lambda system_prompt, user_payload: {
        "intent": "unsupported",
        "core_terms": [" Tea ", "tea", None] + [f"term-{index}" for index in range(10)],
        "expansion_terms": "not-a-list",
        "entities": ["Mina"],
        "temporal_cues": ["latest"],
        "evidence_needs": [f"need-{index}" for index in range(6)],
    }

    plan = model.plan_query_structured("query", [])

    assert plan["intent"] == "other"
    assert plan["core_terms"] == ["Tea"] + [f"term-{index}" for index in range(7)]
    assert plan["expansion_terms"] == []
    assert plan["entities"] == ["Mina"]
    assert plan["evidence_needs"] == [f"need-{index}" for index in range(4)]
    assert set(plan) == {
        "intent", "core_terms", "expansion_terms", "entities",
        "temporal_cues", "evidence_needs",
    }


def test_local_score_fusion_rejects_an_invalid_alpha(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_FUSION_ALPHA", "1.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_fusion_alpha_from_environment()


def test_dense_context_rejects_an_invalid_weight(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_CONTEXT_WEIGHT", "1.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_context_weight_from_environment()


def test_dense_time_rejects_an_invalid_weight(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_TIME_WEIGHT", "-0.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_time_weight_from_environment()


def test_local_reranker_rejects_an_unbounded_pool(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_TOP_N", "101")
    with pytest.raises(RuntimeError, match="between 1 and 100"):
        rerank_top_n_from_environment()


def test_local_reranker_modes_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_MODEL", "answerai/colbert")
    monkeypatch.setenv("MEMORY_LOCAL_CROSS_ENCODER_MODEL", "BAAI/cross-encoder")
    with pytest.raises(RuntimeError, match="either"):
        local_reranker_from_environment()


def test_yes_no_reranker_cannot_be_combined_with_fastembed_reranking(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_MODEL", "answerai/colbert")
    monkeypatch.setenv("MEMORY_LOCAL_YES_NO_RERANK_BASE_URL", "http://127.0.0.1:8081/v1")
    with pytest.raises(RuntimeError, match="not both"):
        create_app(str(tmp_path / "memory.db"))


def test_session_fusion_rejects_an_unbounded_weight(monkeypatch):
    monkeypatch.setenv("MEMORY_SESSION_FUSION_WEIGHT", "11")
    with pytest.raises(RuntimeError, match="between 0 and 10"):
        session_fusion_weight_from_environment()


def test_session_filter_rejects_an_unbounded_pool(monkeypatch):
    monkeypatch.setenv("MEMORY_SESSION_TOP_N", "101")
    with pytest.raises(RuntimeError, match="between 0 and 100"):
        session_top_n_from_environment()


def test_rerank_fusion_rejects_an_invalid_weight(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_FUSION_WEIGHT", "-0.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        rerank_fusion_weight_from_environment()


def test_instruction_rerank_rejects_an_unbounded_pool(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_INSTRUCTION_TOP_N", "0")
    with pytest.raises(RuntimeError, match="between 1 and 100"):
        instruction_rerank_top_n_from_environment()


def test_instruction_refine_pool_cannot_exceed_first_pass(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_INSTRUCTION_TOP_N", "3")
    monkeypatch.setenv("MEMORY_LOCAL_INSTRUCTION_REFINE_TOP_N", "4")
    with pytest.raises(RuntimeError, match="cannot exceed"):
        instruction_refine_top_n_from_environment()


def test_model_facts_retrieve_their_original_source_evidence(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"), model=FakeMemoryModel())
    store.initialize()
    store.add(AddRequest(
        request_id="request-1", user_id="user-a", session_id="session-a",
        messages=[{"role": "user", "content": "Mina says tea is her preferred afternoon drink."}],
    ))
    results = store.search(user_id="user-a", query="Mina tea", options=[], top_k=10)
    assert results[0].id.startswith("mem_")
    assert results[0].content == "Mina says tea is her preferred afternoon drink."


def test_model_query_plan_expands_retrieval_terms(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"), model=FakeMemoryModel())
    store.initialize()
    store.add(AddRequest(
        request_id="request-2", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina prefers tea."},
            {"role": "user", "content": "Mina likes collecting stamps."},
        ],
    ))

    results = store.search(
        user_id="user-a", query="Which beverage does Mina favor?", options=[], top_k=1
    )

    assert results[0].content == "Mina prefers tea."


def test_structured_plan_changes_retrieval_without_an_extra_planner_call(tmp_path):
    messages = [
        {"role": "user", "content": "Mina prefers tea."},
        {"role": "user", "content": "Mina likes collecting stamps."},
    ]
    flat_model = FakeStructuredMemoryModel()
    flat_store = MemoryStore(str(tmp_path / "flat.db"), model=flat_model)
    flat_store.initialize()
    flat_store.add(AddRequest(
        request_id="flat", user_id="user-a", session_id="session-a", messages=messages,
    ))

    structured_model = FakeStructuredMemoryModel()
    structured_store = MemoryStore(
        str(tmp_path / "structured.db"),
        model=structured_model,
        structured_query_plan=True,
    )
    structured_store.initialize()
    structured_store.add(AddRequest(
        request_id="structured", user_id="user-a", session_id="session-a", messages=messages,
    ))

    flat = flat_store.search(
        user_id="user-a", query="Which beverage does she favor?", options=[], top_k=1
    )
    structured = structured_store.search(
        user_id="user-a", query="Which beverage does she favor?", options=[], top_k=1
    )

    assert flat[0].content == "Mina likes collecting stamps."
    assert structured[0].content == "Mina prefers tea."
    assert (flat_model.flat_plan_calls, flat_model.structured_plan_calls) == (1, 0)
    assert (structured_model.flat_plan_calls, structured_model.structured_plan_calls) == (0, 1)
    assert MemoryStore.STRUCTURED_SUPPORT_RRF_WEIGHT == 0.01
