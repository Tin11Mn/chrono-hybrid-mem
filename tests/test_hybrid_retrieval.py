from app.schemas import AddRequest
from app.storage import MemoryStore


def test_temporal_query_prefers_newer_event_over_later_ingestion(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
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
