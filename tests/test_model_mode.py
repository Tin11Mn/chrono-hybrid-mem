import pytest

from app.main import (
    create_app,
    dense_fusion_alpha_from_environment,
    dense_context_weight_from_environment,
    dense_speaker_mask_max_from_environment,
    dense_speaker_conflict_margin_from_environment,
    dense_speaker_conflict_gate_only_from_environment,
    dense_sentence_weight_from_environment,
    dense_image_carry_weight_from_environment,
    dense_speaker_coref_weight_from_environment,
    dense_time_weight_from_environment,
    rerank_top_n_from_environment,
    rerank_image_followups_from_environment,
    rerank_fusion_weight_from_environment,
    rerank_near_tie_epsilon_from_environment,
    session_fusion_weight_from_environment,
    session_top_n_from_environment,
    instruction_rerank_top_n_from_environment,
    instruction_refine_top_n_from_environment,
    instruction_speaker_conflict_only_from_environment,
    structured_query_plan_from_environment,
)
from app.model import MemoryModel, model_from_environment
from app.local_semantic import (
    local_reranker_from_environment,
    local_semantic_retriever_from_environment,
)
from app.schemas import AddRequest
from app.storage import MemoryStore


class FakeMemoryModel:
    def __init__(self):
        self.ranked_candidates = []

    def extract_facts(self, content, speaker="", timestamp=None):
        return ["Mina prefers tea in the afternoon."] if "tea" in content else []

    def rank_candidates(self, query, options, candidates):
        self.ranked_candidates = candidates
        return [candidate["id"] for candidate in candidates]

    def plan_query(self, query, options):
        return ["Mina", "prefers tea"]


class FakeStructuredMemoryModel(FakeMemoryModel):
    def __init__(self):
        super().__init__()
        self.flat_plan_calls = 0
        self.structured_plan_calls = 0

    def plan_query(self, query, options):
        self.flat_plan_calls += 1
        return ["Mina", "oolong", "hiking", "preferred drink"]

    def plan_query_structured(self, query, options):
        self.structured_plan_calls += 1
        return {
            "intent": "fact",
            "core_terms": ["Mina", "oolong"],
            "expansion_terms": ["hiking"],
            "entities": ["Mina"],
            "temporal_cues": [],
            "evidence_needs": ["preferred drink"],
        }


def test_competition_mode_requires_a_secret(monkeypatch):
    monkeypatch.setenv("MEMORY_REQUIRE_MODEL", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        model_from_environment()


def test_custom_memory_model_endpoint_must_be_loopback():
    with pytest.raises(RuntimeError, match="loopback"):
        MemoryModel("test", base_url="https://example.com/v1")


def test_structured_query_plan_defaults_on_and_reads_false(monkeypatch):
    monkeypatch.delenv("MEMORY_STRUCTURED_QUERY_PLAN", raising=False)
    assert structured_query_plan_from_environment() is True
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "FaLsE")
    assert structured_query_plan_from_environment() is False


def test_structured_query_plan_contract_is_bounded_and_sanitized():
    model = MemoryModel.__new__(MemoryModel)
    model._json_response = lambda system_prompt, payload: {
        "intent": "multi_hop",
        "core_terms": [" Mina ", "Mina", 7] + ["term{}".format(i) for i in range(20)],
        "expansion_terms": "not-a-list",
        "entities": ["Mina"],
        "temporal_cues": ["previous"],
        "evidence_needs": ["workplace", "city"],
    }

    plan = model.plan_query_structured("Where does Mina work?", [])

    assert plan["intent"] == "multi_hop"
    assert plan["core_terms"][0] == "Mina"
    assert len(plan["core_terms"]) == 8
    assert plan["expansion_terms"] == []
    assert plan["evidence_needs"] == ["workplace", "city"]




def test_local_score_fusion_rejects_an_invalid_alpha(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_FUSION_ALPHA", "1.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_fusion_alpha_from_environment()


def test_dense_context_rejects_an_invalid_weight(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_CONTEXT_WEIGHT", "1.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_context_weight_from_environment()


def test_late_interaction_and_dense_embedding_modes_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("MEMORY_LOCAL_LATE_INTERACTION_MODEL", "answerai/colbert")
    with pytest.raises(RuntimeError, match="either"):
        local_semantic_retriever_from_environment()


def test_dense_time_rejects_an_invalid_weight(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_TIME_WEIGHT", "-0.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_time_weight_from_environment()


def test_dense_speaker_mask_max_reads_boolean_flag(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_SPEAKER_MASK_MAX", "true")

    assert dense_speaker_mask_max_from_environment() is True


def test_dense_speaker_conflict_rejects_a_negative_margin(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_SPEAKER_CONFLICT_MARGIN", "-0.01")

    with pytest.raises(RuntimeError, match="non-negative"):
        dense_speaker_conflict_margin_from_environment()


def test_speaker_conflict_gate_flags_read_boolean_values(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_SPEAKER_CONFLICT_GATE_ONLY", "true")
    monkeypatch.setenv("MEMORY_LOCAL_INSTRUCTION_SPEAKER_CONFLICT_ONLY", "true")

    assert dense_speaker_conflict_gate_only_from_environment() is True
    assert instruction_speaker_conflict_only_from_environment() is True


def test_dense_sentence_weight_rejects_an_invalid_value(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_SENTENCE_WEIGHT", "1.1")

    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_sentence_weight_from_environment()


def test_dense_image_carry_weight_rejects_an_invalid_value(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_IMAGE_CARRY_WEIGHT", "-0.1")

    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_image_carry_weight_from_environment()


def test_dense_speaker_coref_weight_rejects_an_invalid_value(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_SPEAKER_COREF_WEIGHT", "2")

    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_speaker_coref_weight_from_environment()


def test_rerank_near_tie_rejects_an_invalid_epsilon(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_NEAR_TIE_EPSILON", "1.1")

    with pytest.raises(RuntimeError, match="between 0 and 1"):
        rerank_near_tie_epsilon_from_environment()


def test_local_reranker_rejects_an_unbounded_pool(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_TOP_N", "101")
    with pytest.raises(RuntimeError, match="between 1 and 100"):
        rerank_top_n_from_environment()


def test_image_followup_context_rejects_an_unbounded_window(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_IMAGE_FOLLOWUPS", "5")
    with pytest.raises(RuntimeError, match="between 0 and 4"):
        rerank_image_followups_from_environment()


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
    model = FakeMemoryModel()
    store = MemoryStore(str(tmp_path / "memory.db"), model=model)
    store.initialize()
    store.add(AddRequest(
        request_id="request-1", user_id="user-a", session_id="session-a",
        messages=[{"role": "user", "content": "Mina says tea is her preferred afternoon drink."}],
    ))
    results = store.search(user_id="user-a", query="Mina tea", options=[], top_k=10)
    assert results[0].id.startswith("mem_")
    assert results[0].content == "Mina says tea is her preferred afternoon drink."
    ranking_text = model.ranked_candidates[0]["content"]
    assert "Source speaker: user" in ranking_text
    assert "Mina prefers tea in the afternoon." in ranking_text


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


def test_structured_query_plan_routes_support_terms_without_an_extra_plan_call(tmp_path):
    model = FakeStructuredMemoryModel()
    baseline_model = FakeStructuredMemoryModel()
    baseline_store = MemoryStore(
        str(tmp_path / "baseline.db"), model=baseline_model,
        structured_query_plan=False,
    )
    baseline_store.initialize()
    store = MemoryStore(
        str(tmp_path / "memory.db"), model=model, structured_query_plan=True
    )
    store.initialize()
    request = AddRequest(
        request_id="structured-plan", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina chose oolong."},
            {"role": "user", "content": "Mina discussed hiking."},
        ],
    )
    baseline_store.add(request)
    store.add(request)

    baseline_results = baseline_store.search(
        user_id="user-a", query="Which beverage does Mina favor?", options=[], top_k=1
    )
    results = store.search(
        user_id="user-a", query="Which beverage does Mina favor?", options=[], top_k=1
    )

    assert baseline_results[0].content == "Mina discussed hiking."
    assert results[0].content == "Mina chose oolong."
    assert model.structured_plan_calls == 1
    assert model.flat_plan_calls == 0
    assert baseline_model.flat_plan_calls == 1
    assert baseline_model.structured_plan_calls == 0


def test_model_ranking_pool_is_bounded_while_search_still_returns_top_k(tmp_path):
    model = FakeMemoryModel()
    store = MemoryStore(str(tmp_path / "memory.db"), model=model)
    store.initialize()
    store.add(AddRequest(
        request_id="bounded-model-pool",
        user_id="user-a",
        session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina tea memory {}.".format(index)}
            for index in range(40)
        ],
    ))

    results = store.search(
        user_id="user-a", query="Mina tea", options=[], top_k=100
    )

    assert len(model.ranked_candidates) == MemoryStore.MODEL_RERANK_LIMIT
    assert len(results) == 40
