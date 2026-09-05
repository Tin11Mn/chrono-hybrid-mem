"""Tests for the MemoryART-inspired session-fact semantic layer (default off).

The offline per-session observations (B direction) are injected through
``MemoryStore.add_session_facts`` and, when ``session_fact_layer`` is enabled
with a semantic retriever, dense-matched facts re-enter the rerank pool and
their source messages carry the observations into the Search rerank input.
All new behavior is gated off by default and must not change existing
retrieval.
"""

import importlib.util
import json
import sqlite3
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_locomo_retrieval.py"
SPEC = importlib.util.spec_from_file_location("locomo_evaluation", SCRIPT)
locomo_evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(locomo_evaluation)

from app.schemas import AddRequest
from app.storage import MemoryStore


class RankModel:
    """Deterministic Search model: keeps fusion order, no plan extraction."""

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query(self, query, options):
        return []

    def rank_candidates(self, query, options, candidates):
        return [candidate["id"] for candidate in candidates]


class FixedRetriever:
    """Semantic retriever stub that always ranks session facts by id order."""

    def __init__(self, top_first_ids=None):
        self.top_first_ids = top_first_ids or []

    def score(self, query, options, candidates):
        # Deterministic: score by descending id suffix so ordering is stable.
        result = {}
        for candidate in candidates:
            cid = str(candidate["id"])
            suffix = cid.split("_")[-1]
            try:
                result[cid] = float(int(suffix)) * 0.001
            except ValueError:
                result[cid] = 0.5
        return result

    def rank(self, query, options, candidates, limit):
        scored = self.score(query, options, candidates)
        ordered = sorted(scored, key=scored.get, reverse=True)
        if self.top_first_ids:
            head = [cid for cid in self.top_first_ids if cid in scored]
            ordered = head + [cid for cid in ordered if cid not in head]
        return ordered[:limit]


SAMPLE = {
    "sample_id": "sample-sf",
    "conversation": {"session_1": [
        {"speaker": "Ari", "dia_id": "D1:1", "text": "Alpha giver Bob gave Alice the book."},
        {"speaker": "Bela", "dia_id": "D1:2", "text": "Zeta employer Bob works at Microsoft."},
        {"speaker": "Cid", "dia_id": "D1:3", "text": "Gamma unrelated filler note."},
    ]},
    "qa": [{
        "question": "Where does the person who gave Alice the book work?",
        "evidence": ["D1:1", "D1:2"], "category": 3,
    }],
}


def _build_store(tmp_path, *, session_fact_layer=False, retriever=None):
    db_path = str(tmp_path / "sample.db")
    store = MemoryStore(
        db_path,
        model=RankModel(),
        semantic_retriever=retriever,
        dense_rrf_weight=0.0 if retriever else 1.0,
        structured_query_plan=False,
        session_fact_layer=session_fact_layer,
        session_fact_rrf_weight=0.05,
        session_fact_quota=2,
        session_fact_top_n=3,
    )
    store.initialize()
    messages = [
        {"role": "Ari", "content": "Ari: Alpha giver Bob gave Alice the book.", "timestamp": 0},
        {"role": "Bela", "content": "Bela: Zeta employer Bob works at Microsoft.", "timestamp": 0},
        {"role": "Cid", "content": "Cid: Gamma unrelated filler note.", "timestamp": 0},
    ]
    store.add(AddRequest(
        request_id="r1", user_id="u1", session_id="s1", messages=messages,
    ))
    return store


def test_session_facts_table_exists_after_initialize(tmp_path):
    store = _build_store(tmp_path)
    db_path = store.database_path
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session_facts'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None


def test_add_session_facts_idempotent_and_queryable(tmp_path):
    store = _build_store(tmp_path)
    facts = [
        {"session_id": "s1", "source_message_id": 1, "fact_text": "Ari gave Alice a book."},
        {"session_id": "s1", "source_message_id": 2, "fact_text": "Bob works at Microsoft."},
    ]
    store.add_session_facts(user_id="u1", facts=facts)
    store.add_session_facts(user_id="u1", facts=facts)  # duplicate insert must be idempotent
    db_path = store.database_path
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM session_facts WHERE user_id = ?", ("u1",)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 2


def test_session_fact_layer_default_off_does_not_touch_facts(tmp_path):
    store = _build_store(tmp_path)  # session_fact_layer defaults to False
    store.add_session_facts(user_id="u1", facts=[
        {"session_id": "s1", "source_message_id": 2, "fact_text": "Bob works at Microsoft."},
    ])
    results = store.search(user_id="u1", query="where does Bob work", top_k=5)
    assert results  # retrieval still works
    # Session facts are stored but the search channel is off: no reserved ids.
    trace = store.last_retrieval_trace
    assert trace.get("session_fact_union_ids", []) == []
    assert trace.get("reserved_session_fact_ids", []) == []


def test_session_fact_layer_with_retriever_reserves_matches(tmp_path):
    retriever = FixedRetriever()
    store = _build_store(tmp_path, session_fact_layer=True, retriever=retriever)
    store.add_session_facts(user_id="u1", facts=[
        {"session_id": "s1", "source_message_id": 2, "fact_text": "Bob works at Microsoft."},
        {"session_id": "s1", "source_message_id": 3, "fact_text": "Cid wrote a note."},
    ])
    results = store.search(user_id="u1", query="where does Bob work", top_k=5)
    trace = store.last_retrieval_trace
    diag = trace.get("session_fact_diagnostics", {})
    assert diag.get("enabled") is True
    # With the stub retriever every fact scores by id, so both facts are
    # matched; source message ids 2 and 3 are the union.
    union = trace.get("session_fact_union_ids", [])
    assert "mem_2" in union or "mem_3" in union


def test_session_fact_layer_requires_retriever_for_channel(tmp_path):
    # Layer enabled but no semantic retriever: the channel must stay inert and
    # search must not raise.
    store = _build_store(tmp_path, session_fact_layer=True, retriever=None)
    store.add_session_facts(user_id="u1", facts=[
        {"session_id": "s1", "source_message_id": 2, "fact_text": "Bob works at Microsoft."},
    ])
    results = store.search(user_id="u1", query="where does Bob work", top_k=5)
    trace = store.last_retrieval_trace
    diag = trace.get("session_fact_diagnostics", {})
    assert diag.get("enabled") is False
    assert results


def test_bridge_fact_stores_multi_source_links(tmp_path):
    """A bridge note fans out to every supporting source row."""
    store = _build_store(tmp_path)
    store.add_session_facts(user_id="u1", facts=[{
        "session_id": "s1", "source_message_id": 1, "kind": "bridge",
        "fact_text": "Bob, who gave Alice the book, works at Microsoft.",
        "source_message_ids": [1, 2],
    }])
    db_path = store.database_path
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT source_message_id FROM session_fact_sources "
            "ORDER BY source_message_id"
        ).fetchall()
        kinds = conn.execute(
            "SELECT kind FROM session_facts"
        ).fetchall()
        count = conn.execute(
            "SELECT COUNT(*) FROM session_fact_sources"
        ).fetchone()[0]
    finally:
        conn.close()
    assert kinds == [("bridge",)]
    assert count == 2
    assert [r[0] for r in rows] == [1, 2]


def test_bridge_hit_brings_all_sources_into_channel(tmp_path):
    """A matched bridge fact must surface both linked messages."""
    retriever = FixedRetriever()
    store = _build_store(tmp_path, session_fact_layer=True, retriever=retriever)
    store.add_session_facts(user_id="u1", facts=[
        {"session_id": "s1", "source_message_id": 1, "kind": "bridge",
         "fact_text": "Bob, who gave Alice the book, works at Microsoft.",
         "source_message_ids": [1, 2]},
    ])
    store.search(user_id="u1", query="where does the book giver work", top_k=5)
    trace = store.last_retrieval_trace
    union = trace.get("session_fact_union_ids", [])
    # Both ends of the bridge (mem_1 = gave book, mem_2 = works at MS) surface.
    assert "mem_1" in union and "mem_2" in union
    diag = trace.get("session_fact_diagnostics", {})
    # One distinct fact scored, not one per expanded source row.
    assert diag.get("facts") == 1
    assert diag.get("triggered") is True


def test_mixed_fact_and_bridge_kinds_coexist(tmp_path):
    retriever = FixedRetriever()
    store = _build_store(tmp_path, session_fact_layer=True, retriever=retriever)
    store.add_session_facts(user_id="u1", facts=[
        {"session_id": "s1", "source_message_id": 3, "fact_text": "Cid wrote a note."},
        {"session_id": "s1", "source_message_id": 1, "kind": "profile",
         "fact_text": "Bob is a book giver.",
         "source_message_ids": [1]},
    ])
    store.search(user_id="u1", query="anything", top_k=5)
    trace = store.last_retrieval_trace
    diag = trace.get("session_fact_diagnostics", {})
    assert diag.get("facts") == 2
    union = trace.get("session_fact_union_ids", [])
    assert "mem_3" in union
    assert "mem_1" in union
