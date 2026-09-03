import hashlib
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from app.evidence_graph import SupportWitness
from app.schemas import AddRequest
from app.storage import (
    SUPPORT_NORMALIZATION_ID,
    SUPPORT_SCHEMA_VERSION,
    MemoryStore,
)


class FakeGraphModel:
    def __init__(self, *, seed="Alice", core="unfindable"):
        self.seed = seed
        self.core = core
        self.extract_memory_calls = 0
        self.extract_facts_calls = 0
        self.plan_calls = 0
        self.rank_calls = 0
        self.ranked_candidates = []

    @staticmethod
    def _graph_payload():
        relation = {
            "subject": "Alice",
            "subject_type": "person",
            "relation": "works_at",
            "object": "Acme",
            "object_type": "organization",
            "explicit": True,
            "state_change": "assert",
            "temporal_status": None,
        }
        return {
            "facts": ["Alice works at Acme."],
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Acme", "type": "organization"},
            ],
            # Duplicate model output must not create a duplicate source edge.
            "relations": [relation, dict(relation)],
        }

    def extract_memory(self, content, speaker=None, timestamp=None):
        self.extract_memory_calls += 1
        if "works at Acme" in content:
            return self._graph_payload()
        return {"facts": [], "entities": [], "relations": []}

    def extract_facts(self, content, speaker=None, timestamp=None):
        self.extract_facts_calls += 1
        return []

    def plan_query_structured(self, query, options):
        self.plan_calls += 1
        return {
            "intent": "relation",
            "core_terms": [self.core],
            "expansion_terms": [],
            "entities": [self.seed],
            "temporal_cues": [],
            "evidence_needs": ["find the direct relationship"],
            "bridge_needed": False,
        }

    def rank_candidates(self, query, options, candidates):
        self.rank_calls += 1
        self.ranked_candidates = list(candidates)
        return [candidate["id"] for candidate in candidates]


def add_messages(
    store,
    *,
    user_id="user-a",
    request_id="request-1",
    session_id=None,
    role="speaker",
    contents=None,
):
    contents = contents or ["Alice works at Acme."]
    store.add(AddRequest(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id or "session-{}".format(request_id),
        messages=[
            {"role": role, "content": content, "timestamp": index + 1}
            for index, content in enumerate(contents)
        ],
    ))


def graph_store(tmp_path, model=None, **kwargs):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        model=model or FakeGraphModel(),
        structured_query_plan=True,
        evidence_graph=True,
        **kwargs,
    )
    store.initialize()
    return store


def storage_only_graph_store(tmp_path):
    store = MemoryStore(
        str(tmp_path / "storage-only.db"),
        structured_query_plan=True,
        evidence_graph=True,
    )
    store.initialize()
    return store


def exact_support_payload(source_message_id, *, clause="alice works at acme.",
                          witness=None):
    subject = SimpleNamespace(
        entity_id="subject",
        canonical_name="alice",
        display_name="Alice",
        entity_type="person",
        identity_hint=None,
    )
    object_ = SimpleNamespace(
        entity_id="object",
        canonical_name="acme",
        display_name="Acme",
        entity_type="organization",
        identity_hint=None,
    )
    if witness is None:
        subject_start = clause.index("alice")
        predicate_start = clause.index("works")
        object_start = clause.index("acme")
        witness = SupportWitness(
            spec_id="works_at:direct",
            clause=clause,
            subject_span=(subject_start, subject_start + len("alice")),
            predicate_span=(predicate_start, predicate_start + len("works at")),
            object_span=(object_start, object_start + len("acme")),
            binding="named",
            state_change="assert",
            temporal_status=None,
            source_span=(0, len(clause)),
        )
    relation = SimpleNamespace(
        user_id="user-a",
        source_message_id="mem_{}".format(source_message_id),
        subject_entity_id="subject",
        object_entity_id="object",
        predicate="works_at",
        state_change="assert",
        temporal_status=None,
        support_witness=witness,
    )
    return SimpleNamespace(entities=(subject, object_), relations=(relation,))


def jordan_designer_payload(content, speaker=None, timestamp=None):
    organization = "Acme" if "Acme" in content else "Beta"
    return {
        "facts": [],
        "entities": [
            {"name": "Jordan", "type": "person"},
            {"name": organization, "type": "organization"},
        ],
        "relations": [{
            "subject": "Jordan",
            "subject_type": "person",
            "relation": "works_at",
            "object": organization,
            "object_type": "organization",
            "explicit": True,
            "state_change": "assert",
            "temporal_status": None,
        }],
    }


def test_graph_add_uses_one_composite_call_and_persists_source_provenance(tmp_path):
    model = FakeGraphModel()
    store = graph_store(tmp_path, model)

    add_messages(store)

    assert model.extract_memory_calls == 1
    assert model.extract_facts_calls == 0
    with store._connection() as connection:
        entities = connection.execute(
            "SELECT * FROM graph_entities ORDER BY id"
        ).fetchall()
        edges = connection.execute("SELECT * FROM graph_edges").fetchall()
        support = connection.execute(
            "SELECT * FROM graph_edge_support"
        ).fetchall()
        raw = connection.execute("SELECT * FROM raw_messages").fetchone()
        assert len(entities) == 2
        assert len(edges) == 1
        assert len(support) == 1
        assert edges[0]["source_message_id"] == raw["id"]
        assert edges[0]["user_id"] == raw["user_id"] == "user-a"
        assert edges[0]["event_ts"] == raw["event_ts"] == 1
        assert support[0]["edge_id"] == edges[0]["id"]
        assert support[0]["source_message_id"] == raw["id"]
        assert support[0]["support_schema_version"] == SUPPORT_SCHEMA_VERSION
        assert support[0]["normalization_id"] == SUPPORT_NORMALIZATION_ID
        assert support[0]["spec_id"] == "works_at:direct"
        assert support[0]["binding"] == "named"
        normalized_source = "alice works at acme."
        assert support[0]["source_start"] == 0
        assert support[0]["source_end"] == len(normalized_source)
        assert support[0]["clause_start"] == 0
        assert support[0]["clause_end"] == len(normalized_source)
        assert support[0]["subject_start"] == 0
        assert support[0]["subject_end"] == len("alice")
        assert support[0]["predicate_start"] == len("alice ")
        assert support[0]["predicate_end"] == len("alice works at")
        assert support[0]["object_start"] == len("alice works at ")
        assert support[0]["object_end"] == len("alice works at acme")
        assert support[0]["source_span_sha256"] == hashlib.sha256(
            normalized_source.encode("utf-8")
        ).hexdigest()
        assert [row["version"] for row in connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        )] == [1, 2, 3, 4]
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_support_sidecar_remaps_legacy_witness_offsets_to_v1_codepoints(tmp_path):
    model = FakeGraphModel()
    model.extract_memory = lambda content, speaker=None, timestamp=None: (
        model._graph_payload()
    )
    store = graph_store(tmp_path, model)
    source_text = "\r\n\r\n\u00a0Alice\u2028\u2028works\tat\u00a0Acme.\r\n\r\n"
    store.add(AddRequest(
        request_id="unicode-support",
        user_id="user-a",
        session_id="unicode-session",
        messages=[{"role": "speaker", "content": source_text, "timestamp": 7}],
    ))
    legacy_clause = store._legacy_parser_support_source(source_text)
    assert legacy_clause == "alice\u2028\u2028works at acme."
    with store._connection() as connection:
        support = connection.execute(
            "SELECT * FROM graph_edge_support"
        ).fetchone()

    assert support is not None
    normalized = "alice works at acme."
    assert support["normalization_id"] == "nfkc-casefold-ws-v1"
    assert support["source_start"] == 0
    assert support["source_end"] == len(normalized)
    assert support["clause_start"] == 0
    assert support["clause_end"] == len(normalized)
    assert normalized[support["subject_start"]:support["subject_end"]] == "alice"
    assert normalized[support["predicate_start"]:support["predicate_end"]] == "works at"
    assert normalized[support["object_start"]:support["object_end"]] == "acme"
    assert support["source_span_sha256"] == hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def test_support_sidecar_rejects_unmappable_legacy_whitespace_boundary(tmp_path):
    store = storage_only_graph_store(tmp_path)
    source_text = "Alice\u2028\u2028works at Acme. Trailing text."
    store.add(AddRequest(
        request_id="unmappable-support",
        user_id="user-a",
        session_id="unmappable-session",
        messages=[{"role": "speaker", "content": source_text}],
    ))
    legacy_clause = store._legacy_parser_support_source(source_text)
    assert legacy_clause == source_text.casefold()
    clause = "alice works at acme."
    # Position 6 is inside the two-codepoint U+2028 run. Its canonical v1
    # position is intentionally non-unique, so storage must fail closed even
    # though the carried clause and local spans themselves are otherwise valid.
    witness = SupportWitness(
        spec_id="works_at:direct",
        clause=clause,
        subject_span=(0, len("alice")),
        predicate_span=(len("alice "), len("alice works at")),
        object_span=(len("alice works at "), len("alice works at acme")),
        binding="named",
        state_change="assert",
        temporal_status=None,
        source_span=(6, 6 + len(clause)),
    )
    with store._connection() as connection:
        raw = connection.execute(
            "SELECT id, created_at FROM raw_messages WHERE user_id = ?",
            ("user-a",),
        ).fetchone()
        assert raw is not None
        store._store_graph_payload(
            connection,
            user_id="user-a",
            session_id="unmappable-session",
            source_message_id=int(raw["id"]),
            event_ts=None,
            created_at=str(raw["created_at"]),
            payload=exact_support_payload(
                int(raw["id"]), clause=clause, witness=witness
            ),
        )
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_edge_support"
        ).fetchone()[0] == 0


def test_support_sidecar_requires_exact_witness_and_never_overwrites_one(tmp_path):
    store = storage_only_graph_store(tmp_path)
    source_text = "Alice works at Acme."
    store.add(AddRequest(
        request_id="support-idempotency",
        user_id="user-a",
        session_id="support-session",
        messages=[{"role": "speaker", "content": source_text}],
    ))
    with store._connection() as connection:
        raw = connection.execute(
            "SELECT id, created_at FROM raw_messages WHERE user_id = ?",
            ("user-a",),
        ).fetchone()
        assert raw is not None
        source_id = int(raw["id"])
        payload = exact_support_payload(source_id)
        store._store_graph_payload(
            connection,
            user_id="user-a",
            session_id="support-session",
            source_message_id=source_id,
            event_ts=None,
            created_at=str(raw["created_at"]),
            payload=payload,
        )
        # Retrying the same payload is a no-op.
        store._store_graph_payload(
            connection,
            user_id="user-a",
            session_id="support-session",
            source_message_id=source_id,
            event_ts=None,
            created_at=str(raw["created_at"]),
            payload=payload,
        )
        original = connection.execute(
            "SELECT id, spec_id FROM graph_edge_support"
        ).fetchone()
        assert original is not None
        original_witness = payload.relations[0].support_witness
        different_witness = SupportWitness(
            spec_id="works_at:coordinated",
            clause=original_witness.clause,
            subject_span=original_witness.subject_span,
            predicate_span=original_witness.predicate_span,
            object_span=original_witness.object_span,
            binding=original_witness.binding,
            state_change=original_witness.state_change,
            temporal_status=original_witness.temporal_status,
            source_span=original_witness.source_span,
        )
        store._store_graph_payload(
            connection,
            user_id="user-a",
            session_id="support-session",
            source_message_id=source_id,
            event_ts=None,
            created_at=str(raw["created_at"]),
            payload=exact_support_payload(source_id, witness=different_witness),
        )
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1
        support_rows = connection.execute(
            "SELECT id, spec_id FROM graph_edge_support"
        ).fetchall()
        assert [(row["id"], row["spec_id"]) for row in support_rows] == [
            (original["id"], "works_at:direct")
        ]

        no_witness_payload = exact_support_payload(source_id)
        no_witness_payload.relations[0].support_witness = None
        # A legacy edge may not be re-attested by an absent witness, either.
        store._store_graph_payload(
            connection,
            user_id="user-a",
            session_id="support-session",
            source_message_id=source_id,
            event_ts=None,
            created_at=str(raw["created_at"]),
            payload=no_witness_payload,
        )
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_edge_support"
        ).fetchone()[0] == 1


def test_one_hop_graph_returns_original_message_for_outgoing_and_incoming_seed(tmp_path):
    model = FakeGraphModel(seed="Alice")
    store = graph_store(tmp_path, model)
    add_messages(store)

    outgoing = store.search(
        user_id="user-a", query="unfindable", options=[], top_k=1
    )
    assert [item.content for item in outgoing] == ["Alice works at Acme."]
    assert store.last_graph_candidate_ids == [outgoing[0].id]
    assert store.last_graph_paths[0]["hop_count"] == 1
    assert store.last_graph_paths[0]["relations"] == ["works_at"]
    support_path = store.last_graph_paths[0]
    assert isinstance(support_path["support_id"], int)
    assert support_path["support_schema_version"] == SUPPORT_SCHEMA_VERSION
    assert support_path["support_normalization_id"] == SUPPORT_NORMALIZATION_ID
    assert support_path["support_spec_id"] == "works_at:direct"
    assert support_path["support_binding"] == "named"
    assert support_path["support_source_span"] == [0, len("alice works at acme.")]
    assert support_path["support_clause_span"] == [0, len("alice works at acme.")]
    assert support_path["support_subject_span"] == [0, len("alice")]
    assert support_path["support_predicate_span"] == [
        len("alice "), len("alice works at")
    ]
    assert support_path["support_object_span"] == [
        len("alice works at "), len("alice works at acme")
    ]
    assert support_path["support_source_span_sha256"] == hashlib.sha256(
        b"alice works at acme."
    ).hexdigest()

    model.seed = "Acme"
    incoming = store.search(
        user_id="user-a", query="unfindable", options=[], top_k=1
    )
    assert [item.id for item in incoming] == [outgoing[0].id]
    assert model.plan_calls == 2
    assert model.rank_calls == 2


def test_graph_traversal_is_exactly_user_scoped(tmp_path):
    model = FakeGraphModel()
    store = graph_store(tmp_path, model)
    add_messages(store, user_id="user-a", request_id="a", contents=["Alice works at Acme."])
    add_messages(store, user_id="user-b", request_id="b", contents=["Alice works at Acme too."])

    results = store.search(
        user_id="user-a", query="unfindable", options=[], top_k=10
    )

    assert len(results) == 1
    assert results[0].content == "Alice works at Acme."
    with store._connection() as connection:
        source_users = {
            row["user_id"]
            for row in connection.execute(
                """SELECT raw.user_id
                   FROM graph_edges AS edge
                   JOIN raw_messages AS raw
                     ON raw.id = edge.source_message_id
                    AND raw.user_id = edge.user_id
                   WHERE edge.user_id = ?""",
                ("user-a",),
            ).fetchall()
        }
    assert source_users == {"user-a"}


def test_ambiguous_same_user_seed_fails_closed(tmp_path):
    model = FakeGraphModel()
    store = graph_store(tmp_path, model)
    add_messages(store)
    with store._connection() as connection:
        source_id = connection.execute(
            "SELECT id FROM raw_messages WHERE user_id = ?", ("user-a",)
        ).fetchone()[0]
        connection.execute(
            """INSERT INTO graph_entities(
                   user_id, canonical_name, display_name, entity_type,
                   first_source_message_id, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("user-a", "alice", "Another Alice", "person", source_id, "now"),
        )

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)

    assert store.last_graph_candidate_ids == []
    assert store.last_graph_paths == []


def test_same_payload_same_name_entities_remain_ambiguous_in_storage(tmp_path):
    model = FakeGraphModel(seed="Alex")
    model.extract_memory = lambda content, speaker=None, timestamp=None: {
        "facts": [],
        "entities": [
            {"name": "Alex", "type": "person"},
            {"name": " ALEX ", "type": "person"},
            {"name": "Tea", "type": "food"},
        ],
        "relations": [{
            "subject": "Alex",
            "subject_type": "person",
            "relation": "prefers",
            "object": "Tea",
            "object_type": "food",
            "explicit": True,
            "state_change": "assert",
            "temporal_status": None,
        }],
    }
    store = graph_store(tmp_path, model)
    add_messages(store, contents=["Two different people are both called Alex."])

    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?""",
            ("user-a", "alex", "person"),
        ).fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 0

    model.extract_memory = lambda content, speaker=None, timestamp=None: {
        "facts": [],
        "entities": [
            {"name": "Alex", "type": "person"},
            {"name": "Tea", "type": "food"},
        ],
        "relations": [{
            "subject": "Alex",
            "subject_type": "person",
            "relation": "prefers",
            "object": "Tea",
            "object_type": "food",
            "explicit": True,
            "state_change": "assert",
            "temporal_status": None,
        }],
    }
    add_messages(
        store,
        request_id="ambiguous-followup",
        session_id="session-request-1",
        contents=["Alex prefers Tea."],
    )
    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?""",
            ("user-a", "alex", "person"),
        ).fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 0

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    assert store.last_graph_candidate_ids == []


def test_explicit_hints_keep_same_name_identities_distinct_across_adds(tmp_path):
    model = FakeGraphModel(seed="Jordan")

    def extract(content, speaker=None, timestamp=None):
        hint = "designer" if "design" in content else "chemist"
        organization = "Org A" if hint == "designer" else "Org B"
        return {
            "facts": [],
            "entities": [
                {"name": "Jordan", "type": "person", "identity_hint": hint},
                {"name": organization, "type": "organization"},
            ],
            "relations": [{
                "subject": "Jordan",
                "subject_type": "person",
                "relation": "works_at",
                "object": organization,
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        }

    model.extract_memory = extract
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        request_id="designer",
        contents=["Jordan the designer works at Org A."],
    )
    add_messages(
        store,
        request_id="chemist",
        contents=["Jordan the chemist works at Org B."],
    )

    with store._connection() as connection:
        jordans = connection.execute(
            """SELECT identity_hint FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?
               ORDER BY identity_hint""",
            ("user-a", "jordan", "person"),
        ).fetchall()
        assert [row["identity_hint"] for row in jordans] == ["chemist", "designer"]
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 2

    add_messages(
        store,
        request_id="generic-jordan",
        contents=["Jordan works at Org B."],
    )
    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?""",
            ("user-a", "jordan", "person"),
        ).fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 2

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    assert store.last_graph_candidate_ids == []


def test_cross_session_unhinted_same_name_does_not_create_false_bridge(tmp_path):
    model = FakeGraphModel(seed="Jordan")

    def extract(content, speaker=None, timestamp=None):
        organization = "Grove Clinic" if "Grove" in content else "Harbor Press"
        return {
            "facts": [],
            "entities": [
                {"name": "Jordan", "type": "person"},
                {"name": organization, "type": "organization"},
            ],
            "relations": [{
                "subject": "Jordan",
                "subject_type": "person",
                "relation": "works_at",
                "object": organization,
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        }

    model.extract_memory = extract
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        request_id="jordan-grove",
        session_id="session-grove",
        contents=["Jordan works at Grove Clinic."],
    )
    add_messages(
        store,
        request_id="jordan-harbor",
        session_id="session-harbor",
        contents=["Jordan works at Harbor Press."],
    )

    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?""",
            ("user-a", "jordan", "person"),
        ).fetchone()[0] == 2
        assert connection.execute(
            """SELECT COUNT(DISTINCT subject_entity_id) FROM graph_edges
               WHERE user_id = ? AND predicate = ?""",
            ("user-a", "works_at"),
        ).fetchone()[0] == 2

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    assert store.last_graph_candidate_ids == []
    assert store.last_graph_paths == []


def test_cross_session_role_hint_does_not_merge_explicitly_different_people(
    tmp_path,
):
    model = FakeGraphModel(seed="Jordan")
    model.extract_memory = jordan_designer_payload
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        request_id="jordan-acme",
        session_id="session-acme",
        contents=["Jordan, the designer, works at Acme."],
    )
    add_messages(
        store,
        request_id="jordan-beta",
        session_id="session-beta",
        contents=["A different Jordan, the designer, works at Beta."],
    )

    with store._connection() as connection:
        jordans = connection.execute(
            """SELECT id, identity_hint FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?
               ORDER BY id""",
            ("user-a", "jordan", "person"),
        ).fetchall()
        assert len(jordans) == 2
        assert [row["identity_hint"] for row in jordans] == [
            "designer", "designer",
        ]
        assert connection.execute(
            """SELECT COUNT(DISTINCT subject_entity_id) FROM graph_edges
               WHERE user_id = ? AND predicate = ?""",
            ("user-a", "works_at"),
        ).fetchone()[0] == 2

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    assert store.last_graph_candidate_ids == []
    assert store.last_retrieval_trace["unresolved_seeds"][0][
        "reason"
    ] == "ambiguous"


def test_same_session_role_hint_reuses_one_local_identity(tmp_path):
    model = FakeGraphModel(seed="Jordan")
    model.extract_memory = jordan_designer_payload
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        request_id="jordan-acme",
        session_id="shared-designer-session",
        contents=["Jordan, the designer, works at Acme."],
    )
    add_messages(
        store,
        request_id="jordan-beta",
        session_id="shared-designer-session",
        contents=["Jordan, the designer, works at Beta."],
    )

    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?
                 AND identity_hint = ?""",
            ("user-a", "jordan", "person", "designer"),
        ).fetchone()[0] == 1
        assert connection.execute(
            """SELECT COUNT(DISTINCT subject_entity_id) FROM graph_edges
               WHERE user_id = ? AND predicate = ?""",
            ("user-a", "works_at"),
        ).fetchone()[0] == 1

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    assert len(store.last_graph_candidate_ids) == 2


def test_same_session_unhinted_exact_identity_reuses_one_entity(tmp_path):
    model = FakeGraphModel(seed="Jordan")

    def extract(content, speaker=None, timestamp=None):
        organization = "Grove Clinic" if "Grove" in content else "Harbor Press"
        return {
            "facts": [],
            "entities": [
                {"name": "Jordan", "type": "person"},
                {"name": organization, "type": "organization"},
            ],
            "relations": [{
                "subject": "Jordan",
                "subject_type": "person",
                "relation": "works_at",
                "object": organization,
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        }

    model.extract_memory = extract
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        request_id="same-session-1",
        session_id="shared-session",
        contents=["Jordan works at Grove Clinic."],
    )
    add_messages(
        store,
        request_id="same-session-2",
        session_id="shared-session",
        contents=["Jordan works at Harbor Press."],
    )

    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?""",
            ("user-a", "jordan", "person"),
        ).fetchone()[0] == 1
        assert connection.execute(
            """SELECT COUNT(DISTINCT subject_entity_id) FROM graph_edges
               WHERE user_id = ? AND predicate = ?""",
            ("user-a", "works_at"),
        ).fetchone()[0] == 1

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    assert len(store.last_graph_candidate_ids) == 2
    assert len(store.last_graph_paths) == 2


def test_trusted_api_speaker_duplicates_persist_once_and_reuse_across_adds(
    tmp_path,
):
    model = FakeGraphModel(seed="Melanie")

    def extract(content, speaker=None, timestamp=None):
        object_name = "blue bowl" if "bowl" in content else "clay vase"
        speaker_entities = [
            {"name": "Melanie", "type": "person"}
            for _ in range(5 if "bowl" in content else 1)
        ]
        return {
            "facts": [],
            "entities": speaker_entities + [
                {"name": object_name, "type": "object"},
            ],
            "relations": [{
                "subject": "Melanie",
                "subject_type": "person",
                "relation": "created",
                "object": object_name,
                "object_type": "object",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        }

    model.extract_memory = extract
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        request_id="melanie-bowl",
        session_id="melanie-session-1",
        role="Melanie",
        contents=["I created the blue bowl."],
    )

    with store._connection() as connection:
        speakers = connection.execute(
            """SELECT id, identity_hint FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?""",
            ("user-a", "melanie", "person"),
        ).fetchall()
        assert [row["identity_hint"] for row in speakers] == [
            "api-speaker:melanie",
        ]
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_edges"
        ).fetchone()[0] == 1
        speaker_id = int(speakers[0]["id"])

    add_messages(
        store,
        request_id="melanie-vase",
        session_id="melanie-session-2",
        role="Melanie",
        contents=["I created the clay vase."],
    )

    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ? AND entity_type = ?
                 AND identity_hint = ?""",
            ("user-a", "melanie", "person", "api-speaker:melanie"),
        ).fetchone()[0] == 1
        edges = connection.execute(
            """SELECT edge.subject_entity_id,
                      object.canonical_name AS object_name
               FROM graph_edges AS edge
               JOIN graph_entities AS object
                 ON object.id = edge.object_entity_id
                AND object.user_id = edge.user_id
               WHERE edge.user_id = ? AND edge.predicate = ?
               ORDER BY edge.id""",
            ("user-a", "created"),
        ).fetchall()
        assert [row["object_name"] for row in edges] == [
            "blue bowl", "clay vase",
        ]
        assert {int(row["subject_entity_id"]) for row in edges} == {speaker_id}


def test_exact_api_speaker_hint_reuses_identity_across_sessions_and_users_isolate(
    tmp_path,
):
    model = FakeGraphModel(seed="Alice")

    def extract(content, speaker=None, timestamp=None):
        return {
            "facts": [],
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Acme", "type": "organization"},
            ],
            "relations": [{
                "subject": "Alice",
                "subject_type": "person",
                "relation": "works_at",
                "object": "Acme",
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        }

    model.extract_memory = extract
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        user_id="user-a",
        request_id="alice-a-1",
        session_id="alice-session-1",
        role="Alice",
        contents=["I work at Acme."],
    )
    add_messages(
        store,
        user_id="user-a",
        request_id="alice-a-2",
        session_id="alice-session-2",
        role="Alice",
        contents=["I still work at Acme."],
    )
    add_messages(
        store,
        user_id="user-b",
        request_id="alice-b-1",
        session_id="alice-session-3",
        role="Alice",
        contents=["I work at Acme."],
    )

    with store._connection() as connection:
        rows = connection.execute(
            """SELECT user_id, COUNT(*) AS entity_count,
                      MIN(identity_hint) AS identity_hint
               FROM graph_entities
               WHERE canonical_name = ? AND entity_type = ?
               GROUP BY user_id ORDER BY user_id""",
            ("alice", "person"),
        ).fetchall()
        assert [
            (row["user_id"], row["entity_count"], row["identity_hint"])
            for row in rows
        ] == [
            ("user-a", 1, "api-speaker:alice"),
            ("user-b", 1, "api-speaker:alice"),
        ]

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    assert len(store.last_graph_candidate_ids) == 2
    assert len(store.last_graph_paths) == 2
    with store._connection() as connection:
        assert {
            row["user_id"]
            for row in connection.execute(
                """SELECT raw.user_id
                   FROM graph_edges AS edge
                   JOIN raw_messages AS raw
                     ON raw.id = edge.source_message_id
                    AND raw.user_id = edge.user_id
                   WHERE edge.user_id = ?""",
                ("user-a",),
            ).fetchall()
        } == {"user-a"}


def test_alias_resolution_requires_user_scoped_explicit_alias_source(tmp_path):
    model = FakeGraphModel(seed="A. Example")
    store = graph_store(tmp_path, model)
    add_messages(store)
    with store._connection() as connection:
        entity_id = connection.execute(
            """SELECT id FROM graph_entities
               WHERE user_id = ? AND canonical_name = ?""",
            ("user-a", "alice"),
        ).fetchone()[0]
        source_id = connection.execute(
            "SELECT id FROM raw_messages WHERE user_id = ?", ("user-a",)
        ).fetchone()[0]
        connection.execute(
            """INSERT INTO graph_aliases(
                   user_id, entity_id, normalized_alias, display_alias,
                   source_message_id, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("user-a", entity_id, "a. example", "A. Example", source_id, "now"),
        )

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)

    assert len(store.last_graph_candidate_ids) == 1


def test_graph_foreign_keys_reject_nonexistent_and_cross_user_provenance(tmp_path):
    store = graph_store(tmp_path)
    add_messages(store)
    with pytest.raises(sqlite3.IntegrityError), store._connection() as connection:
        connection.execute(
            """INSERT INTO graph_entities(
                   user_id, canonical_name, display_name, entity_type,
                   first_source_message_id, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("user-b", "mallory", "Mallory", "person", 999, "now"),
        )


def test_initialize_rejects_graph_tables_with_columns_but_no_constraints(tmp_path):
    database_path = tmp_path / "malformed.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE graph_entities (
            id INTEGER PRIMARY KEY, user_id TEXT, canonical_name TEXT,
            display_name TEXT, entity_type TEXT, identity_hint TEXT,
            first_source_message_id INTEGER, created_at TEXT
        );
        CREATE TABLE graph_aliases (
            id INTEGER PRIMARY KEY, user_id TEXT, entity_id INTEGER,
            normalized_alias TEXT, display_alias TEXT,
            source_message_id INTEGER, created_at TEXT
        );
        CREATE TABLE graph_edges (
            id INTEGER PRIMARY KEY, user_id TEXT, subject_entity_id INTEGER,
            predicate TEXT, object_entity_id INTEGER, object_value TEXT,
            source_message_id INTEGER, event_ts INTEGER, state_change TEXT,
            temporal_status TEXT, supersedes_edge_id INTEGER, created_at TEXT
        );
        """
    )
    connection.close()

    store = MemoryStore(
        str(database_path), structured_query_plan=True, evidence_graph=True
    )
    with pytest.raises(RuntimeError, match="incompatible graph_"):
        store.initialize()


def test_graph_only_candidate_is_reserved_inside_thirty_item_rerank_pool(tmp_path):
    model = FakeGraphModel(core="needle", seed="A. Example")
    store = graph_store(
        tmp_path,
        model,
        graph_rrf_weight=0.01,
        graph_rerank_quota=1,
    )
    add_messages(
        store,
        request_id="lexical",
        contents=["needle memory {}".format(index) for index in range(35)],
    )
    add_messages(
        store, request_id="graph", contents=["Alice works at Acme."]
    )
    with store._connection() as connection:
        entity_id = connection.execute(
            """SELECT id FROM graph_entities
               WHERE user_id = ? AND canonical_name = ?""",
            ("user-a", "alice"),
        ).fetchone()[0]
        source_id = connection.execute(
            """SELECT id FROM raw_messages
               WHERE user_id = ? AND content = ?""",
            ("user-a", "Alice works at Acme."),
        ).fetchone()[0]
        connection.execute(
            """INSERT INTO graph_aliases(
                   user_id, entity_id, normalized_alias, display_alias,
                   source_message_id, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("user-a", entity_id, "a. example", "A. Example", source_id, "now"),
        )

    store.search(user_id="user-a", query="needle", options=[], top_k=10)

    assert len(model.ranked_candidates) == MemoryStore.MODEL_RERANK_LIMIT
    graph_id = store.last_graph_candidate_ids[0]
    ranked_ids = [candidate["id"] for candidate in model.ranked_candidates]
    assert graph_id in ranked_ids
    assert "Untrusted graph retrieval metadata" in next(
        candidate["content"]
        for candidate in model.ranked_candidates
        if candidate["id"] == graph_id
    )
    assert model.plan_calls == 1
    assert model.rank_calls == 1
    trace = store.last_retrieval_trace
    assert trace["graph_channel_only_ids"] == [graph_id]
    assert trace["reserved_graph_ids"] == [graph_id]
    assert trace["promoted_graph_ids"] == [graph_id]
    assert len(trace["displaced_p1_ids"]) == 1
    assert len(trace["rerank_pool_ids"]) == MemoryStore.MODEL_RERANK_LIMIT


def test_low_rank_graph_candidate_also_seen_by_p1_still_gets_quota(tmp_path):
    model = FakeGraphModel(core="needle")
    store = graph_store(
        tmp_path,
        model,
        graph_rrf_weight=0.01,
        graph_rerank_quota=1,
    )
    contents = ["Alice works at Acme. Needle."]
    contents.extend("needle newer {}".format(index) for index in range(40))
    add_messages(store, contents=contents)

    store.search(user_id="user-a", query="needle", options=[], top_k=10)

    graph_id = store.last_graph_candidate_ids[0]
    assert graph_id not in store.last_graph_only_candidate_ids
    assert graph_id in {
        candidate["id"] for candidate in model.ranked_candidates
    }


def test_graph_flag_off_uses_legacy_extractor_and_writes_no_graph_rows(tmp_path):
    model = FakeGraphModel(core="needle")
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        model=model,
        structured_query_plan=True,
        evidence_graph=False,
    )
    store.initialize()

    add_messages(store, contents=["needle memory"])
    results = store.search(user_id="user-a", query="needle", options=[], top_k=1)

    assert results[0].content == "needle memory"
    assert model.extract_facts_calls == 1
    assert model.extract_memory_calls == 0
    assert store.last_graph_candidate_ids == []
    with store._connection() as connection:
        graph_objects = connection.execute(
            """SELECT name FROM sqlite_master
               WHERE name LIKE 'graph_%' OR name = 'schema_migrations'"""
        ).fetchall()
        assert graph_objects == []


def test_graph_flag_off_ignores_every_graph_only_constructor_value(tmp_path):
    model = FakeGraphModel(core="needle")
    baseline_model = FakeGraphModel(core="needle")
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        model=model,
        structured_query_plan=True,
        evidence_graph=False,
        dense_fusion_alpha=None,
        graph_max_hops=999,
        graph_temporal=True,
        graph_rrf_weight=float("nan"),
        graph_max_candidates=-99,
        graph_rerank_quota=999,
    )
    baseline = MemoryStore(
        str(tmp_path / "baseline.db"),
        model=baseline_model,
        structured_query_plan=True,
        evidence_graph=False,
    )
    store.initialize()
    baseline.initialize()
    contents = ["needle memory {}".format(index) for index in range(40)]
    add_messages(store, contents=contents)
    add_messages(baseline, contents=contents)

    results = store.search(user_id="user-a", query="needle", options=[], top_k=100)
    baseline_results = baseline.search(
        user_id="user-a", query="needle", options=[], top_k=100
    )

    assert [
        (item.id, item.content, item.score) for item in results
    ] == [
        (item.id, item.content, item.score) for item in baseline_results
    ]
    assert model.extract_facts_calls == baseline_model.extract_facts_calls == 40
    assert model.extract_memory_calls == 0
    assert baseline_model.extract_memory_calls == 0
    assert (model.plan_calls, model.rank_calls) == (
        baseline_model.plan_calls, baseline_model.rank_calls
    ) == (1, 1)


def test_graph_rejects_dense_fusion_and_unimplemented_p3_modes(tmp_path):
    common = {
        "database_path": str(tmp_path / "memory.db"),
        "structured_query_plan": True,
        "evidence_graph": True,
    }
    with pytest.raises(ValueError, match="dense_fusion_alpha"):
        MemoryStore(**common, dense_fusion_alpha=0.5)
    with pytest.raises(ValueError, match="graph_max_hops=1"):
        MemoryStore(**common, graph_max_hops=2)
    with pytest.raises(ValueError, match="graph_temporal=false"):
        MemoryStore(**common, graph_temporal=True)


def test_zero_graph_quota_reserves_no_candidate_and_pool_stays_bounded(tmp_path):
    model = FakeGraphModel(core="needle", seed="A. Example")
    store = graph_store(
        tmp_path,
        model,
        graph_rrf_weight=0.0,
        graph_rerank_quota=0,
    )
    add_messages(store, request_id="graph", contents=["Alice works at Acme."])
    add_messages(
        store,
        request_id="lexical",
        contents=["needle memory {}".format(index) for index in range(100)],
    )
    with store._connection() as connection:
        entity_id = connection.execute(
            """SELECT id FROM graph_entities
               WHERE user_id = ? AND canonical_name = ?""",
            ("user-a", "alice"),
        ).fetchone()[0]
        source_id = connection.execute(
            """SELECT id FROM raw_messages
               WHERE user_id = ? AND content = ?""",
            ("user-a", "Alice works at Acme."),
        ).fetchone()[0]
        connection.execute(
            """INSERT INTO graph_aliases(
                   user_id, entity_id, normalized_alias, display_alias,
                   source_message_id, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            ("user-a", entity_id, "a. example", "A. Example", source_id, "now"),
        )

    store.search(user_id="user-a", query="needle", options=[], top_k=10)

    graph_id = store.last_graph_candidate_ids[0]
    rerank_ids = [candidate["id"] for candidate in model.ranked_candidates]
    assert len(rerank_ids) == MemoryStore.MODEL_RERANK_LIMIT
    assert graph_id not in rerank_ids
    trace = store.last_retrieval_trace
    assert trace["reserved_graph_ids"] == []
    assert trace["promoted_graph_ids"] == []


def test_graph_candidates_round_robin_across_resolved_seeds(tmp_path):
    model = FakeGraphModel(core="unfindable")

    def extract(content, speaker=None, timestamp=None):
        person = "Bob" if content.startswith("Bob") else "Alice"
        organization = content.split(" at ", 1)[1].rstrip(".")
        return {
            "facts": [],
            "entities": [
                {"name": person, "type": "person"},
                {"name": organization, "type": "organization"},
            ],
            "relations": [{
                "subject": person,
                "subject_type": "person",
                "relation": "works_at",
                "object": organization,
                "object_type": "organization",
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        }

    model.extract_memory = extract
    model.plan_query_structured = lambda query, options: {
        "intent": "multi_hop",
        "core_terms": ["unfindable"],
        "expansion_terms": [],
        "entities": ["Alice", "Bob"],
        "temporal_cues": [],
        "evidence_needs": ["both workplaces"],
    }
    store = graph_store(tmp_path, model, graph_max_candidates=4)
    add_messages(
        store,
        contents=[
            "Alice works at Org {}.".format(index) for index in range(20)
        ] + ["Bob works at Bob Org."],
    )
    with store._connection() as connection:
        bob_message_id = connection.execute(
            "SELECT id FROM raw_messages WHERE content = ?",
            ("Bob works at Bob Org.",),
        ).fetchone()[0]

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)

    assert "mem_{}".format(bob_message_id) in store.last_graph_candidate_ids
    trace = store.last_retrieval_trace
    assert len(trace["resolved_seeds"]) == 2
    assert [item["count"] for item in trace["edge_diagnostics"]["edges_fetched_by_seed"]] == [
        MemoryStore.GRAPH_EDGE_LIMIT_PER_SEED,
        1,
    ]
    assert trace["edge_diagnostics"]["candidate_cap_reached"] is True


def test_frozen_plan_predicates_prioritize_relevant_edges_before_seed_cap(tmp_path):
    model = FakeGraphModel(core="unfindable")

    def extract(content, speaker=None, timestamp=None):
        if " works at " in content:
            predicate = "works_at"
            object_type = "organization"
            object_name = content.split(" works at ", 1)[1].rstrip(".")
        else:
            predicate = "owns"
            object_type = "object"
            object_name = content.split(" owns ", 1)[1].rstrip(".")
        return {
            "facts": [],
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": object_name, "type": object_type},
            ],
            "relations": [{
                "subject": "Alice",
                "subject_type": "person",
                "relation": predicate,
                "object": object_name,
                "object_type": object_type,
                "explicit": True,
                "state_change": "assert",
                "temporal_status": None,
            }],
        }

    model.extract_memory = extract
    model.plan_query_structured = lambda query, options: {
        "intent": "fact",
        "core_terms": ["unfindable"],
        "expansion_terms": [],
        "entities": ["Alice"],
        "temporal_cues": [],
        "evidence_needs": ["find Alice's employer"],
    }
    store = graph_store(tmp_path, model, graph_max_candidates=1)
    relevant = "Alice works at Priority Org."
    add_messages(
        store,
        contents=["Alice owns Item {}.".format(index) for index in range(20)]
        + [relevant],
    )
    with store._connection() as connection:
        relevant_id = connection.execute(
            "SELECT id FROM raw_messages WHERE content = ?", (relevant,)
        ).fetchone()[0]

    store.search(
        user_id="user-a", query="unfindable employer", options=[], top_k=10
    )

    assert store.last_graph_candidate_ids == ["mem_{}".format(relevant_id)]
    assert store.last_retrieval_trace["edge_diagnostics"][
        "preferred_predicates"
    ][:2] == ["works_at", "role_at"]


def test_duplicate_planned_seed_is_accounted_but_never_traversed_twice(tmp_path):
    model = FakeGraphModel(core="unfindable")
    model.plan_query_structured = lambda query, options: {
        "intent": "fact",
        "core_terms": ["unfindable"],
        "expansion_terms": [],
        "entities": ["Alice", " Alice "],
        "temporal_cues": [],
        "evidence_needs": ["find the employer"],
    }
    store = graph_store(tmp_path, model)
    add_messages(store)

    store.search(user_id="user-a", query="unfindable", options=[], top_k=10)

    trace = store.last_retrieval_trace
    assert len(trace["requested_seeds"]) == 2
    assert len(trace["resolved_seeds"]) == 1
    assert trace["unresolved_seeds"] == [{
        "seed": " Alice ",
        "normalized": "alice",
        "reason": "duplicate",
        "candidate_count": 1,
    }]


def test_same_content_graph_sources_deduplicate_to_one_graph_provenance(tmp_path):
    model = FakeGraphModel(core="Alice", seed="Alice")
    store = graph_store(tmp_path, model)
    add_messages(
        store,
        request_id="duplicates",
        contents=["Alice works at Acme.", "Alice works at Acme."],
    )

    results = store.search(user_id="user-a", query="Alice", options=[], top_k=10)

    assert len(store.last_graph_candidate_ids) == 2
    assert [item.content for item in results] == ["Alice works at Acme."]
    assert results[0].id in store.last_graph_candidate_ids
    assert store.last_retrieval_trace["graph_candidate_ids"] == (
        store.last_graph_candidate_ids
    )


def test_retrieval_trace_is_complete_read_only_and_resets_per_search(tmp_path):
    model = FakeGraphModel(core="unfindable", seed="Alice")
    store = graph_store(tmp_path, model)
    add_messages(store)

    first = store.search(user_id="user-a", query="unfindable", options=[], top_k=10)
    trace = store.last_retrieval_trace

    required = {
        "plan", "resolved_seeds", "unresolved_seeds", "p1_union_ids",
        "p1_channels", "p1_pre_rerank_ids",
        "p1_counterfactual_top30_ids", "graph_candidate_ids", "graph_paths",
        "graph_channel_only_ids", "reserved_graph_ids", "promoted_graph_ids",
        "displaced_p1_ids", "rerank_pool_ids", "final_ids",
        "edge_diagnostics",
    }
    assert required.issubset(trace)
    assert trace["resolved_seeds"]
    assert set(trace["p1_channels"]) == {
        "raw", "raw_porter", "fact", "fact_porter", "context",
        "context_porter", "support_raw", "support_fact", "entity_raw",
        "entity_porter", "entity_context", "dense",
    }
    assert trace["p1_counterfactual_top30_ids"] == trace[
        "p1_pre_rerank_ids"
    ][:MemoryStore.MODEL_RERANK_LIMIT]
    assert trace["graph_paths"] == store.last_graph_paths
    assert trace["final_ids"] == [item.id for item in first]
    trace["plan"]["mutated"] = True
    assert "mutated" not in store.last_retrieval_trace["plan"]

    model.seed = "Missing Entity"
    store.search(user_id="user-a", query="nothing", options=[], top_k=10)
    second_trace = store.last_retrieval_trace
    assert second_trace["graph_candidate_ids"] == []
    assert second_trace["resolved_seeds"] == []
    assert second_trace["unresolved_seeds"][0]["reason"] == "not_found"
    assert second_trace["reserved_graph_ids"] == []
    assert second_trace["graph_paths"] == []


def test_graph_flag_only_changes_graph_trace_not_p1_pre_rerank_trace(tmp_path):
    baseline_model = FakeGraphModel(core="needle", seed="Alice")
    graph_model = FakeGraphModel(core="needle", seed="Alice")
    baseline_model.extract_facts = lambda content, speaker=None, timestamp=None: (
        ["Alice works at Acme."] if "works at Acme" in content else []
    )
    graph_store_instance = MemoryStore(
        str(tmp_path / "graph.db"),
        model=graph_model,
        structured_query_plan=True,
        evidence_graph=True,
    )
    baseline_store = MemoryStore(
        str(tmp_path / "baseline.db"),
        model=baseline_model,
        structured_query_plan=True,
        evidence_graph=False,
    )
    graph_store_instance.initialize()
    baseline_store.initialize()
    contents = ["needle memory", "Alice works at Acme."]
    add_messages(graph_store_instance, contents=contents)
    add_messages(baseline_store, contents=contents)

    graph_store_instance.search(
        user_id="user-a", query="needle", options=[], top_k=10
    )
    baseline_store.search(
        user_id="user-a", query="needle", options=[], top_k=10
    )

    graph_trace = graph_store_instance.last_retrieval_trace
    baseline_trace = baseline_store.last_retrieval_trace
    for key in (
        "p1_channels",
        "p1_union_ids",
        "p1_pre_rerank_ids",
        "p1_counterfactual_top30_ids",
    ):
        assert graph_trace[key] == baseline_trace[key]
    assert baseline_trace["graph_paths"] == []
    assert graph_trace["graph_paths"]


def test_concurrent_search_keeps_graph_quota_and_trace_request_local(tmp_path):
    alice_waiting = threading.Event()
    bob_completed = threading.Event()

    class ConcurrentModel:
        @staticmethod
        def extract_memory(content, speaker=None, timestamp=None):
            if " works at " not in content:
                return {"facts": [], "entities": [], "relations": []}
            person, organization = content.split(" works at ", 1)
            organization = organization.rstrip(".")
            return {
                "facts": [],
                "entities": [
                    {"name": person, "type": "person"},
                    {"name": organization, "type": "organization"},
                ],
                "relations": [{
                    "subject": person,
                    "subject_type": "person",
                    "relation": "works_at",
                    "object": organization,
                    "object_type": "organization",
                    "explicit": True,
                    "state_change": "assert",
                    "temporal_status": None,
                }],
            }

        @staticmethod
        def plan_query_structured(query, options):
            return {
                "intent": "fact",
                "core_terms": [query],
                "expansion_terms": [],
                "entities": ["Alice" if query == "qa" else "Bob"],
                "temporal_cues": [],
                "evidence_needs": [],
            }

        @staticmethod
        def rank_candidates(query, options, candidates):
            ordered = sorted(
                candidates,
                key=lambda item: (
                    "works at" not in item["content"], item["id"]
                ),
            )
            return [item["id"] for item in ordered]

    class BarrierRetriever:
        pause_alice = False

        def rank(self, query, options, candidates, limit):
            if self.pause_alice and query == "qa":
                alice_waiting.set()
                if not bob_completed.wait(timeout=5):
                    raise RuntimeError("Bob Search did not complete")
            return []

    retriever = BarrierRetriever()
    store = graph_store(
        tmp_path,
        ConcurrentModel(),
        semantic_retriever=retriever,
        graph_rrf_weight=0.0,
        graph_rerank_quota=1,
    )
    add_messages(
        store,
        user_id="user-a",
        request_id="alice-graph",
        session_id="alice-graph-session",
        role="Alice",
        contents=["Alice works at Acme."],
    )
    add_messages(
        store,
        user_id="user-a",
        request_id="alice-lexical",
        session_id="alice-lexical-session",
        contents=["Alice qa lexical {}".format(index) for index in range(100)],
    )
    add_messages(
        store,
        user_id="user-b",
        request_id="bob-graph",
        session_id="bob-graph-session",
        role="Bob",
        contents=["Bob works at Beta."],
    )

    control = store.search(
        user_id="user-a", query="qa", options=[], top_k=1
    )
    control_trace = store.last_retrieval_trace
    assert [item.content for item in control] == ["Alice works at Acme."]
    assert control_trace["promoted_graph_ids"] == [control[0].id]

    retriever.pause_alice = True
    results = {}
    errors = []

    def search_alice():
        try:
            results["alice"] = store.search(
                user_id="user-a", query="qa", options=[], top_k=1
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    def search_bob():
        try:
            results["bob"] = store.search(
                user_id="user-b", query="qb", options=[], top_k=1
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            bob_completed.set()

    alice_thread = threading.Thread(target=search_alice)
    alice_thread.start()
    assert alice_waiting.wait(timeout=5)
    bob_thread = threading.Thread(target=search_bob)
    bob_thread.start()
    alice_thread.join(timeout=10)
    bob_thread.join(timeout=10)

    assert not alice_thread.is_alive()
    assert not bob_thread.is_alive()
    assert errors == []
    assert [item.content for item in results["alice"]] == [
        "Alice works at Acme."
    ]
    assert [item.content for item in results["bob"]] == [
        "Bob works at Beta."
    ]

    # Alice is deliberately the last completing request. Its published
    # snapshot must be internally consistent and contain no Bob state.
    trace = store.last_retrieval_trace
    assert trace["plan"]["entities"] == ["Alice"]
    assert trace["graph_candidate_ids"] == [control[0].id]
    assert trace["promoted_graph_ids"] == [control[0].id]
    assert trace["final_ids"] == [control[0].id]
    assert store.last_graph_candidate_ids == trace["graph_candidate_ids"]
    assert store.last_graph_paths == trace["graph_paths"]


def test_concurrent_search_keeps_speaker_conflict_gate_request_local(tmp_path):
    alice_waiting = threading.Event()
    bob_completed = threading.Event()

    class SpeakerScorer:
        @staticmethod
        def score(query, options, candidates):
            masked = "[PERSON]" in query
            return {
                item["id"]: (
                    (0.95 if "adoption" in item["content"] else 0.2)
                    if masked
                    else (0.1 if "adoption" in item["content"] else 0.9)
                )
                for item in candidates
            }

    class BarrierReranker:
        @staticmethod
        def rank(query, options, candidates):
            if query == "What did Caroline book?":
                alice_waiting.set()
                if not bob_completed.wait(timeout=5):
                    raise RuntimeError("conflicting Search did not complete")
            return [item["id"] for item in candidates]

    class RecordingInstructionReranker:
        def __init__(self):
            self.queries = []
            self.lock = threading.Lock()

        def rank(self, query, options, candidates):
            with self.lock:
                self.queries.append(query)
            return [item["id"] for item in candidates]

    instruction = RecordingInstructionReranker()
    store = MemoryStore(
        str(tmp_path / "speaker-conflict.db"),
        semantic_retriever=SpeakerScorer(),
        dense_fusion_alpha=0.0,
        dense_speaker_conflict_margin=0.05,
        dense_speaker_conflict_gate_only=True,
        local_reranker=BarrierReranker(),
        local_instruction_reranker=instruction,
        instruction_speaker_conflict_only=True,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="speaker-memories",
        user_id="user-a",
        session_id="speaker-session",
        messages=[
            {
                "role": "Caroline",
                "content": "Caroline: I chose the adoption agency.",
            },
            {
                "role": "Melanie",
                "content": "Melanie: I booked a travel agency.",
            },
        ],
    ))

    errors = []

    def search_without_conflict():
        try:
            store.search(
                user_id="user-a",
                query="What did Caroline book?",
                options=[],
                top_k=1,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    def search_with_conflict():
        try:
            store.search(
                user_id="user-a",
                query="Why did Melanie choose the adoption agency?",
                options=[],
                top_k=1,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            bob_completed.set()

    no_conflict_thread = threading.Thread(target=search_without_conflict)
    no_conflict_thread.start()
    assert alice_waiting.wait(timeout=5)
    conflict_thread = threading.Thread(target=search_with_conflict)
    conflict_thread.start()
    no_conflict_thread.join(timeout=10)
    conflict_thread.join(timeout=10)

    assert not no_conflict_thread.is_alive()
    assert not conflict_thread.is_alive()
    assert errors == []
    assert instruction.queries == [
        "Why did Melanie choose the adoption agency?"
    ]
    assert store.speaker_conflict_trigger_count == 1
    # The no-conflict Search deliberately finishes last and publishes false.
    assert store.last_speaker_conflict is False
