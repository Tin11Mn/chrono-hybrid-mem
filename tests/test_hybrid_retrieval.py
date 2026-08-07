from app.schemas import AddRequest
from app.storage import MemoryStore


def test_temporal_query_prefers_newer_event_over_later_ingestion(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"), temporal_bonus=0.001)
    store.initialize()
    store.add(AddRequest(
        request_id="new-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 200,
            "content": "Ravi's preferred drink is tea.",
        }],
    ))
    store.add(AddRequest(
        request_id="old-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 100,
            "content": "Ravi's preferred drink is coffee.",
        }],
    ))

    results = store.search(
        user_id="user-a", query="What is Ravi's current preferred drink?", top_k=1
    )

    assert results[0].content == "Ravi's preferred drink is tea."


def test_historical_query_prefers_older_event_over_later_ingestion(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"), temporal_bonus=0.001)
    store.initialize()
    store.add(AddRequest(
        request_id="old-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 100,
            "content": "Ravi's preferred drink is hot coffee.",
        }],
    ))
    store.add(AddRequest(
        request_id="new-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 200,
            "content": "Ravi's preferred drink is black tea.",
        }],
    ))

    results = store.search(
        user_id="user-a", query="What did Ravi prefer before?", top_k=1
    )

    assert results[0].content == "Ravi's preferred drink is hot coffee."


def test_query_stop_words_do_not_hide_content_terms(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.initialize()
    store.add(AddRequest(
        request_id="content", user_id="user-a", session_id="session-a",
        messages=[{"role": "user", "content": "Milo prefers jasmine tea."}],
    ))

    results = store.search(user_id="user-a", query="What does Milo prefer?", top_k=1)

    assert results[0].content == "Milo prefers jasmine tea."


def test_neighbor_context_retrieves_pronominal_evidence(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.initialize()
    store.add(AddRequest(
        request_id="context", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina is choosing a drink for breakfast."},
            {"role": "user", "content": "She settles on jasmine tea."},
            {"role": "user", "content": "Bela plans to hike tomorrow."},
        ],
    ))

    results = store.search(
        user_id="user-a", query="Which tea did Mina choose?", top_k=3
    )

    assert any(result.content == "She settles on jasmine tea." for result in results)
