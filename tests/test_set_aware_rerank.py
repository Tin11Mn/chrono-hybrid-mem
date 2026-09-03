import pytest

from app.main import create_app, set_aware_rerank_from_environment
from app.schemas import AddRequest
from app.storage import MemoryStore


class SetAwareModel:
    def __init__(self, evidence_needs):
        self.evidence_needs = evidence_needs
        self.plan_calls = 0
        self.rank_calls = 0

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query_structured(self, query, options):
        self.plan_calls += 1
        return {
            "intent": "multi_hop",
            "core_terms": ["Mina"],
            "expansion_terms": [],
            "entities": ["Mina"],
            "temporal_cues": [],
            "evidence_needs": list(self.evidence_needs),
        }

    def rank_candidates(self, query, options, candidates):
        self.rank_calls += 1
        preferred = []
        for marker in ("TEA_PRIMARY", "TEA_REDUNDANT", "HIKING_COMPLEMENT"):
            preferred.extend(
                candidate["id"]
                for candidate in candidates
                if marker in candidate["content"]
            )
        return preferred + [
            candidate["id"] for candidate in candidates
            if candidate["id"] not in preferred
        ]


def _seed_fixture(store):
    for request_id, content in (
        ("tea-primary", "TEA_PRIMARY: Mina prefers tea."),
        ("tea-redundant", "TEA_REDUNDANT: Mina drinks tea daily."),
        ("hiking-complement", "HIKING_COMPLEMENT: Mina goes hiking on Sundays."),
    ):
        store.add(AddRequest(
            request_id=request_id,
            user_id="user-a",
            session_id="session-a",
            messages=[{"role": "user", "content": content}],
        ))


def test_set_aware_rerank_defaults_off_and_preserves_p1_order(tmp_path):
    database_path = tmp_path / "p2-default-off.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    _seed_fixture(seed_store)

    default_model = SetAwareModel(["tea", "hiking"])
    explicit_off_model = SetAwareModel(["tea", "hiking"])
    default_store = MemoryStore(
        str(database_path), model=default_model, structured_query_plan=True
    )
    explicit_off_store = MemoryStore(
        str(database_path), model=explicit_off_model,
        structured_query_plan=True, set_aware_rerank=False,
    )

    default_results = default_store.search(user_id="user-a", query="Mina", top_k=3)
    explicit_off_results = explicit_off_store.search(
        user_id="user-a", query="Mina", top_k=3
    )

    assert [result.id for result in default_results] == [
        result.id for result in explicit_off_results
    ]
    assert default_model.plan_calls == explicit_off_model.plan_calls == 1
    assert default_model.rank_calls == explicit_off_model.rank_calls == 1
    assert default_store.last_retrieval_trace["p2_enabled"] is False
    assert default_store.last_retrieval_trace["p2_pre_rerank_ids"] == (
        default_store.last_retrieval_trace["p2_post_rerank_ids"]
    )


def test_set_aware_rerank_promotes_complementary_original_evidence(tmp_path):
    database_path = tmp_path / "p2-complementary.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    _seed_fixture(seed_store)

    model = SetAwareModel(["tea", "hiking"])
    store = MemoryStore(
        str(database_path), model=model, structured_query_plan=True,
        set_aware_rerank=True,
    )

    results = store.search(user_id="user-a", query="Mina", top_k=3)
    trace = store.last_retrieval_trace

    assert [result.content.split(":", 1)[0] for result in results] == [
        "TEA_PRIMARY", "HIKING_COMPLEMENT", "TEA_REDUNDANT"
    ]
    assert trace["p2_enabled"] is True
    assert trace["p2_evidence_need_tokens"] == ["tea", "hiking"]
    assert trace["p2_pre_rerank_ids"][0] == results[0].id
    assert trace["p2_post_rerank_ids"] == [result.id for result in results]
    assert trace["p2_newly_covered_tokens"] == [
        {"candidate_id": results[0].id, "newly_covered_tokens": ["tea"]},
        {"candidate_id": results[1].id, "newly_covered_tokens": ["hiking"]},
    ]
    assert model.plan_calls == 1
    assert model.rank_calls == 1
    assert all(result.id in trace["p2_pre_rerank_ids"] for result in results)


def test_set_aware_rerank_without_usable_needs_keeps_original_order(tmp_path):
    database_path = tmp_path / "p2-empty-needs.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    _seed_fixture(seed_store)

    store = MemoryStore(
        str(database_path), model=SetAwareModel([]), structured_query_plan=True,
        set_aware_rerank=True,
    )
    results = store.search(user_id="user-a", query="Mina", top_k=3)
    trace = store.last_retrieval_trace

    assert trace["p2_enabled"] is True
    assert trace["p2_evidence_need_tokens"] == []
    assert trace["p2_pre_rerank_ids"] == trace["p2_post_rerank_ids"]
    assert [result.id for result in results] == trace["p2_pre_rerank_ids"][:3]


def test_set_aware_rerank_requires_structured_query_plan(tmp_path):
    with pytest.raises(ValueError, match="structured_query_plan"):
        MemoryStore(str(tmp_path / "invalid-p2.db"), set_aware_rerank=True)


def test_set_aware_rerank_environment_defaults_off_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMORY_SET_AWARE_RERANK", raising=False)
    assert set_aware_rerank_from_environment() is False

    monkeypatch.setenv("MEMORY_SET_AWARE_RERANK", "true")
    monkeypatch.setenv("MEMORY_STRUCTURED_QUERY_PLAN", "false")
    with pytest.raises(RuntimeError, match="MEMORY_STRUCTURED_QUERY_PLAN"):
        create_app(str(tmp_path / "invalid-p2-app.db"))
