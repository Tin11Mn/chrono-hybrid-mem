"""Retrieval-level regressions for the P3-B1 source-local anchor channel."""

from __future__ import annotations

from app.schemas import AddRequest
from app.storage import MemoryStore


class _AnchorRetrievalModel:
    """Deterministic composite model fake with no relation output."""

    def __init__(self, entities_by_content, planned_entities):
        self.entities_by_content = entities_by_content
        self.planned_entities = planned_entities
        self.extract_memory_calls = 0
        self.plan_calls = 0
        self.rank_calls = 0
        self.ranked_candidate_ids = []

    def extract_memory(self, content, speaker=None, timestamp=None):
        del speaker, timestamp
        self.extract_memory_calls += 1
        return {
            "facts": [],
            "entities": self.entities_by_content.get(content, []),
            "relations": [],
        }

    def extract_facts(self, content, speaker=None, timestamp=None):
        del content, speaker, timestamp
        return []

    def plan_query_structured(self, query, options):
        del query, options
        self.plan_calls += 1
        return {
            "intent": "fact",
            "core_terms": ["unfindable"],
            "expansion_terms": [],
            "entities": list(self.planned_entities),
            "temporal_cues": [],
            "evidence_needs": [],
            "bridge_needed": False,
        }

    def rank_candidates(self, query, options, candidates):
        del query, options
        self.rank_calls += 1
        self.ranked_candidate_ids = [candidate["id"] for candidate in candidates]
        return list(self.ranked_candidate_ids)


def _store(tmp_path, *, model):
    store = MemoryStore(
        str(tmp_path / "anchor-retrieval.db"),
        model=model,
        structured_query_plan=True,
        evidence_anchors=True,
    )
    store.initialize()
    return store


def _add(store, *, request_id, content, user_id="user-a"):
    store.add(AddRequest(
        request_id=request_id,
        user_id=user_id,
        session_id="session-{}".format(request_id),
        messages=[{
            "role": "speaker",
            "content": content,
            "timestamp": 1,
        }],
    ))


def test_anchor_channel_uses_source_local_mentions_without_p3a_edges(tmp_path):
    content = "Ann recorded the blue-file evidence."
    model = _AnchorRetrievalModel(
        {content: [{"name": "Ann", "type": "person"}]}, ["Ann"],
    )
    store = _store(tmp_path, model=model)
    _add(store, request_id="ann", content=content)

    results = store.search(
        user_id="user-a", query="Where is the evidence?", top_k=1
    )
    trace = store.last_retrieval_trace
    with store._connection() as connection:
        edge_count = connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]

    assert [result.content for result in results] == [content]
    assert edge_count == 0
    assert trace["graph_candidate_ids"] == []
    assert trace["graph_paths"] == []
    assert len(trace["anchor_candidate_ids"]) == 1
    assert trace["anchor_diagnostics"]["resolved_seeds"] == [{
        "entity_label": "ann", "candidate_count": 1,
    }]
    assert model.plan_calls == 1
    assert model.rank_calls == 1


def test_anchor_channel_fails_closed_on_a_tampered_mention_witness(tmp_path):
    content = "Ann recorded the tamper-proof evidence."
    model = _AnchorRetrievalModel(
        {content: [{"name": "Ann", "type": "person"}]}, ["Ann"],
    )
    store = _store(tmp_path, model=model)
    _add(store, request_id="ann", content=content)
    with store._connection() as connection:
        connection.execute(
            "UPDATE graph_entity_mentions SET source_span_sha256 = ?", ("0" * 64,)
        )

    store.search(user_id="user-a", query="Where is the evidence?", top_k=1)
    trace = store.last_retrieval_trace

    # P1 can still retrieve the raw message by its label; P3-B1 must not treat
    # the corrupted sidecar as evidence or reserve it in the anchor channel.
    assert trace["anchor_candidate_ids"] == []
    assert trace["anchor_diagnostics"]["invalid_witness_candidates_skipped"] == 1
    assert trace["reserved_anchor_ids"] == []


def test_anchor_channel_round_robins_and_caps_source_candidates(tmp_path):
    entities_by_content = {}
    for index in range(24):
        ann_content = "Ann source-local witness {:02d}.".format(index)
        bob_content = "Bob source-local witness {:02d}.".format(index)
        entities_by_content[ann_content] = [{"name": "Ann", "type": "person"}]
        entities_by_content[bob_content] = [{"name": "Bob", "type": "person"}]
    model = _AnchorRetrievalModel(entities_by_content, ["Ann", "Bob"])
    store = _store(tmp_path, model=model)
    for index in range(24):
        _add(
            store,
            request_id="ann-{:02d}".format(index),
            content="Ann source-local witness {:02d}.".format(index),
        )
        _add(
            store,
            request_id="bob-{:02d}".format(index),
            content="Bob source-local witness {:02d}.".format(index),
        )

    store.search(user_id="user-a", query="retrieve the witnesses", top_k=20)
    trace = store.last_retrieval_trace
    diagnostics = trace["anchor_diagnostics"]

    assert len(trace["anchor_candidate_ids"]) == MemoryStore.ANCHOR_MAX_CANDIDATES
    assert len(set(trace["anchor_candidate_ids"])) == MemoryStore.ANCHOR_MAX_CANDIDATES
    assert diagnostics["candidate_count"] == MemoryStore.ANCHOR_MAX_CANDIDATES
    assert diagnostics["candidate_cap_reached"] is True
    assert diagnostics["candidate_visit_seed_labels"] == ["ann", "bob"] * 10
    assert [item["count"] for item in diagnostics["candidates_fetched_by_seed"]] == [20, 20]
    assert len(trace["rerank_pool_ids"]) <= MemoryStore.MODEL_RERANK_LIMIT
    assert model.rank_calls == 1
