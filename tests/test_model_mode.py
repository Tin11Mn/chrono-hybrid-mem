import pytest

from app.model import model_from_environment
from app.schemas import AddRequest
from app.storage import MemoryStore


class FakeMemoryModel:
    def extract_facts(self, content):
        return ["Mina prefers tea in the afternoon."] if "Mina" in content else []

    def rank_candidates(self, query, options, candidates):
        return [candidate["id"] for candidate in reversed(candidates)]


def test_competition_mode_requires_a_secret(monkeypatch):
    monkeypatch.setenv("MEMORY_REQUIRE_MODEL", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        model_from_environment()


def test_model_facts_are_persisted_and_only_candidate_ids_are_ranked(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"), model=FakeMemoryModel())
    store.initialize()
    store.add(AddRequest(
        request_id="request-1", user_id="user-a", session_id="session-a",
        messages=[{"role": "user", "content": "Mina says tea is her preferred afternoon drink."}],
    ))
    results = store.search(user_id="user-a", query="Mina tea", options=[], top_k=10)
    assert any(result.id.startswith("fact_") for result in results)
    assert any(result.content == "Mina prefers tea in the afternoon." for result in results)
