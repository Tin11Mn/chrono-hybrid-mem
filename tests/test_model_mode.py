import importlib

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
    set_aware_rerank_from_environment,
    evidence_need_retrieval_from_environment,
    adjacent_turn_expansion_from_environment,
    adjacent_seed_limit_from_environment,
    adjacent_candidate_limit_from_environment,
    evidence_graph_from_environment,
    evidence_anchors_from_environment,
    anchor_seed_limit_from_environment,
    anchor_max_candidates_from_environment,
    anchor_rrf_weight_from_environment,
    anchor_rerank_quota_from_environment,
    graph_max_hops_from_environment,
    graph_temporal_from_environment,
    graph_max_candidates_from_environment,
    graph_rerank_quota_from_environment,
    graph_rrf_weight_from_environment,
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


def test_set_aware_rerank_defaults_off(monkeypatch):
    monkeypatch.delenv("MEMORY_SET_AWARE_RERANK", raising=False)
    assert set_aware_rerank_from_environment() is False


def test_evidence_need_retrieval_defaults_to_p4a_q2_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("MEMORY_EVIDENCE_NEED_RETRIEVAL", raising=False)
    assert evidence_need_retrieval_from_environment() is True
    monkeypatch.setenv("MEMORY_EVIDENCE_NEED_RETRIEVAL", "false")
    assert evidence_need_retrieval_from_environment() is False


def test_adjacent_turn_expansion_defaults_off_with_frozen_bounded_defaults(monkeypatch):
    for key in (
        "MEMORY_ADJACENT_TURN_EXPANSION",
        "MEMORY_ADJACENT_SEED_LIMIT",
        "MEMORY_ADJACENT_CANDIDATE_LIMIT",
    ):
        monkeypatch.delenv(key, raising=False)

    assert adjacent_turn_expansion_from_environment() is False
    assert adjacent_seed_limit_from_environment() == 4
    assert adjacent_candidate_limit_from_environment() == 4


@pytest.mark.parametrize(
    "name,value,reader",
    [
        ("MEMORY_ADJACENT_SEED_LIMIT", "-1", adjacent_seed_limit_from_environment),
        ("MEMORY_ADJACENT_SEED_LIMIT", "31", adjacent_seed_limit_from_environment),
        (
            "MEMORY_ADJACENT_CANDIDATE_LIMIT",
            "-1",
            adjacent_candidate_limit_from_environment,
        ),
        (
            "MEMORY_ADJACENT_CANDIDATE_LIMIT",
            "31",
            adjacent_candidate_limit_from_environment,
        ),
        (
            "MEMORY_ADJACENT_CANDIDATE_LIMIT",
            "not-an-integer",
            adjacent_candidate_limit_from_environment,
        ),
    ],
)
def test_adjacent_turn_expansion_rejects_unbounded_environment_values(
    monkeypatch, name, value, reader
):
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=name):
        reader()


def test_adjacent_turn_expansion_flag_off_ignores_invalid_limits(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ADJACENT_TURN_EXPANSION", "false")
    monkeypatch.setenv("MEMORY_ADJACENT_SEED_LIMIT", "not-an-integer")
    monkeypatch.setenv("MEMORY_ADJACENT_CANDIDATE_LIMIT", "31")
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "false")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")

    create_app(str(tmp_path / "adjacent-flag-off.db"))


def test_adjacent_turn_expansion_wires_frozen_defaults_to_memory_store(
    monkeypatch, tmp_path
):
    main_module = importlib.import_module("app.main")
    captured = {}

    class CapturingMemoryStore:
        MODEL_RERANK_LIMIT = 30

        def __init__(self, database_path, **kwargs):
            captured["database_path"] = database_path
            captured.update(kwargs)

        def initialize(self):
            pass

    monkeypatch.setattr(main_module, "MemoryStore", CapturingMemoryStore)
    monkeypatch.setenv("MEMORY_ADJACENT_TURN_EXPANSION", "true")
    monkeypatch.delenv("MEMORY_ADJACENT_SEED_LIMIT", raising=False)
    monkeypatch.delenv("MEMORY_ADJACENT_CANDIDATE_LIMIT", raising=False)
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "false")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")

    main_module.create_app(str(tmp_path / "adjacent-wired.db"))

    assert captured["adjacent_turn_expansion"] is True
    assert captured["adjacent_seed_limit"] == 4
    assert captured["adjacent_candidate_limit"] == 4


def test_adjacent_turn_expansion_enabled_initializes_the_real_memory_store(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MEMORY_ADJACENT_TURN_EXPANSION", "true")
    monkeypatch.delenv("MEMORY_ADJACENT_SEED_LIMIT", raising=False)
    monkeypatch.delenv("MEMORY_ADJACENT_CANDIDATE_LIMIT", raising=False)
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "false")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")

    database_path = tmp_path / "adjacent-enabled.db"
    create_app(str(database_path))

    assert database_path.is_file()


def test_evidence_graph_defaults_off_and_has_bounded_defaults(monkeypatch):
    for key in (
        "MEMORY_EVIDENCE_GRAPH",
        "MEMORY_GRAPH_MAX_HOPS",
        "MEMORY_GRAPH_TEMPORAL",
        "MEMORY_GRAPH_RRF_WEIGHT",
        "MEMORY_GRAPH_MAX_CANDIDATES",
        "MEMORY_GRAPH_RERANK_QUOTA",
    ):
        monkeypatch.delenv(key, raising=False)

    assert evidence_graph_from_environment() is False
    assert graph_max_hops_from_environment() == 1
    assert graph_temporal_from_environment() is False
    assert graph_rrf_weight_from_environment() == 0.025
    assert graph_max_candidates_from_environment() == 20
    assert graph_rerank_quota_from_environment() == 4


@pytest.mark.parametrize(
    "name,value,reader",
    [
        ("MEMORY_GRAPH_MAX_HOPS", "3", graph_max_hops_from_environment),
        ("MEMORY_GRAPH_MAX_HOPS", "2", graph_max_hops_from_environment),
        ("MEMORY_GRAPH_TEMPORAL", "true", graph_temporal_from_environment),
        ("MEMORY_GRAPH_RRF_WEIGHT", "1.1", graph_rrf_weight_from_environment),
        ("MEMORY_GRAPH_RRF_WEIGHT", "nan", graph_rrf_weight_from_environment),
        (
            "MEMORY_GRAPH_MAX_CANDIDATES",
            "101",
            graph_max_candidates_from_environment,
        ),
        ("MEMORY_GRAPH_RERANK_QUOTA", "31", graph_rerank_quota_from_environment),
    ],
)
def test_evidence_graph_rejects_unbounded_environment_values(
    monkeypatch, name, value, reader
):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError):
        reader()


def test_evidence_graph_requires_the_structured_planner(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "true")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "false")
    with pytest.raises(RuntimeError, match="STRUCTURED_QUERY_PLAN"):
        create_app(str(tmp_path / "memory.db"))


def test_graph_flag_off_ignores_invalid_graph_environment_and_creates_no_sidecar(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "false")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")
    monkeypatch.setenv("MEMORY_GRAPH_MAX_HOPS", "not-an-integer")
    monkeypatch.setenv("MEMORY_GRAPH_TEMPORAL", "true")
    monkeypatch.setenv("MEMORY_GRAPH_RRF_WEIGHT", "nan")
    monkeypatch.setenv("MEMORY_GRAPH_MAX_CANDIDATES", "not-an-integer")
    monkeypatch.setenv("MEMORY_GRAPH_RERANK_QUOTA", "not-an-integer")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_REQUIRE_MODEL", raising=False)
    database_path = tmp_path / "flag-off.db"

    create_app(str(database_path))

    import sqlite3
    connection = sqlite3.connect(database_path)
    try:
        graph_objects = connection.execute(
            """SELECT name FROM sqlite_master
               WHERE name LIKE 'graph_%' OR name = 'schema_migrations'"""
        ).fetchall()
    finally:
        connection.close()
    assert graph_objects == []


def test_evidence_graph_rejects_dense_score_fusion(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "true")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")
    monkeypatch.setenv("MEMORY_DENSE_FUSION_ALPHA", "0.5")
    with pytest.raises(RuntimeError, match="DENSE_FUSION_ALPHA"):
        create_app(str(tmp_path / "memory.db"))


def test_evidence_anchors_defaults_off_with_frozen_defaults(monkeypatch):
    for key in (
        "MEMORY_EVIDENCE_ANCHORS",
        "MEMORY_ANCHOR_SEED_LIMIT",
        "MEMORY_ANCHOR_MAX_CANDIDATES",
        "MEMORY_ANCHOR_RRF_WEIGHT",
        "MEMORY_ANCHOR_RERANK_QUOTA",
    ):
        monkeypatch.delenv(key, raising=False)

    assert evidence_anchors_from_environment() is False
    assert anchor_seed_limit_from_environment() == 6
    assert anchor_max_candidates_from_environment() == 20
    assert anchor_rrf_weight_from_environment() == 0.025
    assert anchor_rerank_quota_from_environment() == 4


@pytest.mark.parametrize(
    "name,value,reader",
    [
        ("MEMORY_ANCHOR_SEED_LIMIT", "5", anchor_seed_limit_from_environment),
        ("MEMORY_ANCHOR_MAX_CANDIDATES", "21", anchor_max_candidates_from_environment),
        ("MEMORY_ANCHOR_RRF_WEIGHT", "0.05", anchor_rrf_weight_from_environment),
        ("MEMORY_ANCHOR_RRF_WEIGHT", "nan", anchor_rrf_weight_from_environment),
        ("MEMORY_ANCHOR_RERANK_QUOTA", "3", anchor_rerank_quota_from_environment),
    ],
)
def test_evidence_anchors_reject_non_frozen_environment_values(
    monkeypatch, name, value, reader
):
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=name):
        reader()


def test_anchor_flag_off_ignores_invalid_anchor_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_EVIDENCE_ANCHORS", "false")
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "false")
    monkeypatch.setenv("MEMORY_ADJACENT_TURN_EXPANSION", "false")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")
    monkeypatch.setenv("MEMORY_ANCHOR_SEED_LIMIT", "not-an-integer")
    monkeypatch.setenv("MEMORY_ANCHOR_MAX_CANDIDATES", "not-an-integer")
    monkeypatch.setenv("MEMORY_ANCHOR_RRF_WEIGHT", "nan")
    monkeypatch.setenv("MEMORY_ANCHOR_RERANK_QUOTA", "not-an-integer")

    create_app(str(tmp_path / "anchor-flag-off.db"))


def test_evidence_anchors_wire_frozen_defaults_to_memory_store(monkeypatch, tmp_path):
    main_module = importlib.import_module("app.main")
    captured = {}

    class CapturingMemoryStore:
        MODEL_RERANK_LIMIT = 30
        ANCHOR_SEED_LIMIT = 6
        ANCHOR_MAX_CANDIDATES = 20
        ANCHOR_RRF_WEIGHT = 0.025
        ANCHOR_RERANK_QUOTA = 4

        def __init__(self, database_path, **kwargs):
            captured["database_path"] = database_path
            captured.update(kwargs)

        def initialize(self):
            pass

    monkeypatch.setattr(main_module, "MemoryStore", CapturingMemoryStore)
    monkeypatch.setenv("MEMORY_EVIDENCE_ANCHORS", "true")
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "false")
    monkeypatch.setenv("MEMORY_ADJACENT_TURN_EXPANSION", "false")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")
    monkeypatch.delenv("MEMORY_ANCHOR_SEED_LIMIT", raising=False)
    monkeypatch.delenv("MEMORY_ANCHOR_MAX_CANDIDATES", raising=False)
    monkeypatch.delenv("MEMORY_ANCHOR_RRF_WEIGHT", raising=False)
    monkeypatch.delenv("MEMORY_ANCHOR_RERANK_QUOTA", raising=False)

    main_module.create_app(str(tmp_path / "anchors-wired.db"))

    assert captured["evidence_anchors"] is True
    assert captured["anchor_seed_limit"] == 6
    assert captured["anchor_max_candidates"] == 20
    assert captured["anchor_rrf_weight"] == 0.025
    assert captured["anchor_rerank_quota"] == 4


@pytest.mark.parametrize(
    "name,value,error_pattern",
    [
        ("MEMORY_STRUCTURED_QUERY_PLAN", "false", "STRUCTURED_QUERY_PLAN"),
        ("MEMORY_EVIDENCE_GRAPH", "true", "EVIDENCE_GRAPH"),
        ("MEMORY_ADJACENT_TURN_EXPANSION", "true", "ADJACENT_TURN_EXPANSION"),
        ("MEMORY_DENSE_FUSION_ALPHA", "0.5", "DENSE_FUSION_ALPHA"),
    ],
)
def test_evidence_anchors_reject_incompatible_feature_combinations(
    monkeypatch, tmp_path, name, value, error_pattern
):
    monkeypatch.setenv("MEMORY_EVIDENCE_ANCHORS", "true")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "true")
    monkeypatch.setenv("MEMORY_EVIDENCE_GRAPH", "false")
    monkeypatch.setenv("MEMORY_ADJACENT_TURN_EXPANSION", "false")
    monkeypatch.delenv("MEMORY_DENSE_FUSION_ALPHA", raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=error_pattern):
        create_app(str(tmp_path / "anchors-incompatible.db"))


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
