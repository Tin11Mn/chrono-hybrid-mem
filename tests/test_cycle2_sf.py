"""Cycle 2 SFv2 compliance tests."""
import pytest
import sqlite3
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.storage import MemoryStore
from app.schemas import AddRequest, Message


def _make_store(db_path: str = ":memory:", session_fact_layer: bool = True):
    """Create a MemoryStore with mock model for testing."""
    mock_model = MagicMock()
    mock_model.model_name = "gpt-4o-mini"
    mock_model.generate_session_facts.return_value = [
        {"fact_text": "Alice prefers tea over coffee.", "source_message_ids": ["mem_1", "mem_3"]},
        {"fact_text": "Bob works at Acme Corp.", "source_message_ids": ["mem_2"]},
    ]
    mock_model.extract_facts.return_value = ["Alice likes tea."]
    mock_model.extract_memory.return_value = {"facts": ["Alice likes tea."], "entities": [], "relations": []}
    mock_model.plan_query_structured.return_value = {"core_terms": [], "entities": [], "temporal_cues": []}
    mock_model.rank_candidates.return_value = []
    return MemoryStore(db_path, model=mock_model, session_fact_layer=session_fact_layer)


def test_online_session_facts_generated_on_add():
    """Test that session facts are generated synchronously during Add."""
    store = _make_store()
    store.initialize()
    
    request = AddRequest(
        request_id="test-001",
        user_id="user-alice",
        session_id="session-001",
        messages=[
            Message(role="user", content="I love tea.", timestamp=1000),
            Message(role="assistant", content="Tea is healthy.", timestamp=1001),
            Message(role="user", content="I prefer green tea.", timestamp=1002),
        ],
    )
    
    store.add(request)
    
    # Verify session_facts table has entries
    with store._connection() as conn:
        rows = conn.execute(
            "SELECT id, fact_text, version FROM session_facts WHERE user_id = ? AND session_id = ?",
            ("user-alice", "session-001"),
        ).fetchall()
    
    assert len(rows) == 2, f"Expected 2 session facts, got {len(rows)}"
    assert rows[0]["fact_text"] == "Alice prefers tea over coffee."
    assert rows[1]["fact_text"] == "Bob works at Acme Corp."
    
    # Verify provenance
    for row in rows:
        src_rows = conn.execute(
            "SELECT source_message_id FROM session_fact_sources WHERE session_fact_id = ?",
            (row["id"],),
        ).fetchall()
        assert len(src_rows) > 0, f"Fact {row['id']} has no provenance"


def test_session_facts_are_replaced_not_accumulated():
    """Test that adding a second chunk replaces stale facts."""
    store = _make_store()
    store.initialize()
    
    # First chunk
    request1 = AddRequest(
        request_id="test-002a",
        user_id="user-bob",
        session_id="session-002",
        messages=[
            Message(role="user", content="I like apples.", timestamp=2000),
        ],
    )
    store.add(request1)
    
    # Second chunk - model returns updated facts
    store.model.generate_session_facts.return_value = [
        {"fact_text": "Bob likes apples and oranges.", "source_message_ids": ["mem_1", "mem_2"]},
    ]
    
    request2 = AddRequest(
        request_id="test-002b",
        user_id="user-bob",
        session_id="session-002",
        messages=[
            Message(role="assistant", content="Oranges are good too.", timestamp=2001),
        ],
    )
    store.add(request2)
    
    # Verify only 1 fact (replaced, not accumulated)
    with store._connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM session_facts WHERE user_id = ? AND session_id = ?",
            ("user-bob", "session-002"),
        ).fetchone()["cnt"]
    
    assert count == 1, f"Expected 1 session fact after replacement, got {count}"


def test_cross_user_isolation():
    """Test that session facts are isolated by user_id."""
    store = _make_store()
    store.initialize()
    
    request1 = AddRequest(
        request_id="test-003a",
        user_id="user-alice",
        session_id="session-003",
        messages=[Message(role="user", content="Hello.", timestamp=3000)],
    )
    request2 = AddRequest(
        request_id="test-003b",
        user_id="user-bob",
        session_id="session-003",
        messages=[Message(role="user", content="Hi.", timestamp=3001)],
    )
    
    store.add(request1)
    store.add(request2)
    
    with store._connection() as conn:
        alice_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM session_facts WHERE user_id = 'user-alice'",
        ).fetchone()["cnt"]
        bob_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM session_facts WHERE user_id = 'user-bob'",
        ).fetchone()["cnt"]
    
    assert alice_count == 2
    assert bob_count == 2


def test_request_id_idempotency():
    """Test that duplicate request_id is idempotent."""
    store = _make_store()
    store.initialize()
    
    request = AddRequest(
        request_id="test-004",
        user_id="user-alice",
        session_id="session-004",
        messages=[Message(role="user", content="Test.", timestamp=4000)],
    )
    
    # First add
    store.add(request)
    
    # Second add with same request_id should not duplicate raw messages
    store.add(request)
    
    with store._connection() as conn:
        msg_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM raw_messages WHERE request_id = 'test-004'",
        ).fetchone()["cnt"]
        # Note: request_id is in ingestions, not raw_messages directly
        # But we can check raw_messages count
        raw_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM raw_messages WHERE user_id = 'user-alice'",
        ).fetchone()["cnt"]
    
    assert raw_count == 1, f"Expected 1 message, got {raw_count}"


def test_provenance_chain_valid():
    """Test that every session fact traces back to a real raw message."""
    store = _make_store()
    store.initialize()
    
    request = AddRequest(
        request_id="test-005",
        user_id="user-alice",
        session_id="session-005",
        messages=[
            Message(role="user", content="I work at Google.", timestamp=5000),
        ],
    )
    store.add(request)
    
    with store._connection() as conn:
        facts = conn.execute(
            "SELECT id, fact_text FROM session_facts WHERE user_id = ? AND session_id = ?",
            ("user-alice", "session-005"),
        ).fetchall()
        
        for fact in facts:
            sources = conn.execute(
                """SELECT sfs.source_message_id
                   FROM session_fact_sources sfs
                   JOIN raw_messages rm ON rm.id = sfs.source_message_id
                   WHERE sfs.session_fact_id = ? AND rm.user_id = ?""",
                (fact["id"], "user-alice"),
            ).fetchall()
            assert len(sources) > 0, f"Fact {fact['id']} has no valid provenance"


def test_session_facts_not_returned_in_search():
    """Test that search returns raw messages, not session fact text."""
    store = _make_store()
    store.initialize()
    
    request = AddRequest(
        request_id="test-006",
        user_id="user-alice",
        session_id="session-006",
        messages=[Message(role="user", content="I like Python.", timestamp=6000)],
    )
    store.add(request)
    
    results = store.search(
        user_id="user-alice",
        query="What does Alice like?",
        top_k=10,
    )
    
    # Results should be raw messages, not session fact text
    for r in results:
        assert "Alice prefers tea" not in r.content, "Session fact text should not appear in search results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
