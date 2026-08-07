from fastapi.testclient import TestClient

from app.main import create_app


def make_client(tmp_path):
    return TestClient(create_app(str(tmp_path / "memory.db")))


def add_payload(request_id="req-1", user_id="user-a"):
    return {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": "session-1",
        "messages": [
            {"role": "user", "timestamp": 1704067200000, "content": "Alice prefers tea in the morning."},
            {"role": "assistant", "content": "Alice lives in Hong Kong."},
        ],
    }


def test_health_requires_no_authentication(tmp_path):
    response = make_client(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_add_is_synchronous_and_idempotent(tmp_path):
    client = make_client(tmp_path)
    payload = add_payload()
    first = client.post("/add", json=payload)
    second = client.post("/add", json=payload)

    assert first.status_code == 200
    assert first.json() == {
        "success": True, "request_id": "req-1", "user_id": "user-a", "session_id": "session-1",
    }
    assert second.status_code == 200
    results = client.post("/search", json={"query": "tea", "user_id": "user-a", "top_k": 100})
    data = results.json()["data"]
    assert {item["id"] for item in data} == {"mem_1", "mem_2"}


def test_search_enforces_user_isolation_and_response_shape(tmp_path):
    client = make_client(tmp_path)
    client.post("/add", json=add_payload("req-a", "user-a"))
    client.post("/add", json={
        **add_payload("req-b", "user-b"),
        "messages": [{"role": "user", "content": "Bob prefers tea in the evening."}],
    })

    own = client.post("/search", json={"query": "tea", "user_id": "user-a", "top_k": 1})
    other = client.post("/search", json={"query": "Alice", "user_id": "user-b", "top_k": 100})
    empty = client.post("/search", json={"query": "unrelated", "user_id": "user-a", "top_k": 100})

    assert own.status_code == 200
    assert len(own.json()["data"]) == 1
    result = own.json()["data"][0]
    assert result["id"].startswith("mem_")
    assert result["content"]
    assert isinstance(result["score"], float)
    assert result["created_at"].endswith("Z")
    assert other.json() == {"data": []}
    assert empty.json() == {"data": []}
