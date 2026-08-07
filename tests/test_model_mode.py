import pytest

from app.main import dense_fusion_alpha_from_environment, rerank_top_n_from_environment
from app.model import model_from_environment
from app.schemas import AddRequest
from app.storage import MemoryStore


class FakeMemoryModel:
    def extract_facts(self, content):
        return ["Mina prefers tea in the afternoon."] if "tea" in content else []

    def rank_candidates(self, query, options, candidates):
        return [candidate["id"] for candidate in candidates]

    def plan_query(self, query, options):
        return ["Mina", "prefers tea"]


def test_competition_mode_requires_a_secret(monkeypatch):
    monkeypatch.setenv("MEMORY_REQUIRE_MODEL", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        model_from_environment()


def test_local_score_fusion_rejects_an_invalid_alpha(monkeypatch):
    monkeypatch.setenv("MEMORY_DENSE_FUSION_ALPHA", "1.1")
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        dense_fusion_alpha_from_environment()


def test_local_reranker_rejects_an_unbounded_pool(monkeypatch):
    monkeypatch.setenv("MEMORY_LOCAL_RERANK_TOP_N", "101")
    with pytest.raises(RuntimeError, match="between 1 and 100"):
        rerank_top_n_from_environment()


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
