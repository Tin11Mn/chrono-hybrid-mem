"""Storage regressions for the P3-B1 source-exact entity mention index.

These tests intentionally exercise the persisted witnesses rather than a
production matching helper.  The independent evaluator must be able to reload
the same rows and prove their claims from SQLite raw-message content alone.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata

import pytest

from app.evidence_graph import Entity, GraphPayload
from app.schemas import AddRequest
from app.storage import (
    MENTION_NORMALIZATION_ID,
    MENTION_SCHEMA_VERSION,
    MemoryStore,
)


def _canonical(value: str) -> str:
    """P3-B1's frozen canonical text contract, owned independently by tests."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class AnchorModel:
    """Small composite-extraction stub; it never provides relation evidence."""

    def __init__(self, entities_by_content):
        self.entities_by_content = entities_by_content
        self.extract_memory_calls = 0
        self.extract_facts_calls = 0

    def extract_memory(self, content, speaker=None, timestamp=None):
        del speaker, timestamp
        self.extract_memory_calls += 1
        return {
            "facts": [],
            "entities": list(self.entities_by_content.get(content, [])),
            "relations": [],
        }

    def extract_facts(self, content, speaker=None, timestamp=None):
        del content, speaker, timestamp
        self.extract_facts_calls += 1
        return []


class RelationModel(AnchorModel):
    """One P3-A-valid relation, used only to prove its behavior is unchanged."""

    def __init__(self):
        super().__init__({})

    def extract_memory(self, content, speaker=None, timestamp=None):
        del content, speaker, timestamp
        self.extract_memory_calls += 1
        return {
            "facts": ["Alice works at Acme."],
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


def anchor_store(tmp_path, *, model=None, database_name="anchors.db"):
    store = MemoryStore(
        str(tmp_path / database_name),
        model=model,
        structured_query_plan=True,
        evidence_anchors=True,
    )
    store.initialize()
    return store


def _add_raw_message(
    store,
    *,
    content: str,
    user_id: str = "user-a",
    request_id: str = "request-1",
    session_id: str = "session-1",
):
    store.add(AddRequest(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        messages=[{
            "role": "speaker",
            "content": content,
            "timestamp": 1,
        }],
    ))
    with store._connection() as connection:
        row = connection.execute(
            """SELECT id, user_id, session_id, content, event_ts, created_at
               FROM raw_messages
               WHERE user_id = ? AND session_id = ?
               ORDER BY id DESC LIMIT 1""",
            (user_id, session_id),
        ).fetchone()
    assert row is not None
    return row


def _payload_for_labels(*, user_id: str, source_message_id: int, labels):
    return GraphPayload(entities=tuple(
        Entity(
            entity_id="payload-{}".format(index),
            user_id=user_id,
            canonical_name=_canonical(label),
            display_name=label,
            entity_type="person",
            first_source_message_id="mem_{}".format(source_message_id),
            created_at="2026-08-24T00:00:00Z",
        )
        for index, label in enumerate(labels)
    ))


def _store_local_payload(store, raw_row, *, labels):
    payload = _payload_for_labels(
        user_id=str(raw_row["user_id"]),
        source_message_id=int(raw_row["id"]),
        labels=labels,
    )
    with store._connection() as connection:
        store._store_graph_payload(
            connection,
            user_id=str(raw_row["user_id"]),
            session_id=str(raw_row["session_id"]),
            source_message_id=int(raw_row["id"]),
            event_ts=raw_row["event_ts"],
            created_at=str(raw_row["created_at"]),
            payload=payload,
        )
    return payload


def _mention_rows(store, *, user_id="user-a"):
    with store._connection() as connection:
        return connection.execute(
            """SELECT declaration.id AS declaration_id,
                      declaration.user_id AS declaration_user_id,
                      declaration.source_message_id AS declaration_source_message_id,
                      declaration.anchor_entity_id,
                      declaration.declaration_schema_version,
                      declaration.normalization_id AS declaration_normalization_id,
                      declaration.entity_label AS declaration_label,
                      mention.id AS mention_id,
                      mention.user_id AS mention_user_id,
                      mention.source_message_id AS mention_source_message_id,
                      mention.mention_schema_version,
                      mention.normalization_id AS mention_normalization_id,
                      mention.entity_label AS mention_label,
                      mention.source_start,
                      mention.source_end,
                      mention.mention_start,
                      mention.mention_end,
                      mention.raw_source_start,
                      mention.raw_source_end,
                      mention.raw_mention_start,
                      mention.raw_mention_end,
                      mention.source_span_sha256
               FROM graph_entity_declarations AS declaration
               JOIN graph_entity_mentions AS mention
                 ON mention.declaration_id = declaration.id
                AND mention.user_id = declaration.user_id
                AND mention.source_message_id = declaration.source_message_id
               WHERE declaration.user_id = ?
               ORDER BY declaration.entity_label, mention.mention_start,
                        mention.raw_mention_start, mention.id""",
            (user_id,),
        ).fetchall()


def _unique_index_columns(connection, table_name: str):
    result = set()
    for index in connection.execute("PRAGMA index_list({})".format(table_name)):
        if not int(index["unique"]):
            continue
        name = str(index["name"]).replace('"', '""')
        result.add(tuple(
            str(column["name"])
            for column in connection.execute(
                'PRAGMA index_info("{}")'.format(name)
            )
        ))
    return result


def _foreign_key_columns(connection, table_name: str):
    grouped = {}
    for row in connection.execute("PRAGMA foreign_key_list({})".format(table_name)):
        grouped.setdefault(int(row["id"]), []).append(row)
    return {
        (
            str(rows[0]["table"]),
            tuple(str(row["from"]) for row in sorted(rows, key=lambda item: item["seq"])),
            tuple(str(row["to"]) for row in sorted(rows, key=lambda item: item["seq"])),
        )
        for rows in grouped.values()
    }


def _clone_row_without_id(connection, table_name: str, row, **changes):
    values = {
        column: row[column]
        for column in row.keys()
        if column != "id"
    }
    values.update(changes)
    columns = list(values)
    connection.execute(
        "INSERT INTO {}({}) VALUES ({})".format(
            table_name,
            ", ".join(columns),
            ", ".join("?" for _ in columns),
        ),
        tuple(values[column] for column in columns),
    )


def test_source_local_declaration_survives_ambiguous_global_identity_linking(tmp_path):
    """P3-B1 must not inherit P3-A's intentionally sparse global identity map."""

    store = anchor_store(tmp_path)
    raw = _add_raw_message(store, content="Alex arrived.")

    # P3-A would reject a generic Alex once two specifically hinted global
    # identities exist.  The source-local declaration must still persist.
    with store._connection() as connection:
        for hint in ("chemist", "designer"):
            connection.execute(
                """INSERT INTO graph_entities(
                       user_id, canonical_name, display_name, entity_type,
                       identity_hint, first_source_message_id, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    "user-a", "alex", "Alex", "person", hint,
                    int(raw["id"]), str(raw["created_at"]),
                ),
            )

    _store_local_payload(store, raw, labels=["Alex"])
    rows = _mention_rows(store)

    assert len(rows) == 1
    row = rows[0]
    assert row["declaration_label"] == row["mention_label"] == "alex"
    assert row["declaration_source_message_id"] == int(raw["id"])
    assert re.fullmatch(r"[0-9a-f]{64}", row["anchor_entity_id"])
    with store._connection() as connection:
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entities
               WHERE user_id = ? AND canonical_name = ?""",
            ("user-a", "alex"),
        ).fetchone()[0] == 2
        assert connection.execute(
            """SELECT COUNT(*) FROM graph_entity_declarations
               WHERE user_id = ? AND source_message_id = ? AND entity_label = ?""",
            ("user-a", int(raw["id"]), "alex"),
        ).fetchone()[0] == 1


def test_exact_mentions_use_nfkc_casefold_and_unicode_whitespace_in_both_offset_domains(tmp_path):
    store = anchor_store(tmp_path)
    source = "\uff21\uff2e\uff2e\u00a0met\tANN.\u2028ann"
    raw = _add_raw_message(store, content=source)

    _store_local_payload(store, raw, labels=["Ann"])
    rows = _mention_rows(store)

    canonical_source = _canonical(source)
    assert canonical_source == "ann met ann. ann"
    assert len(rows) == 3
    assert [(row["mention_start"], row["mention_end"]) for row in rows] == [
        (0, 3), (8, 11), (13, 16),
    ]
    assert [(row["raw_mention_start"], row["raw_mention_end"]) for row in rows] == [
        (0, 3), (8, 11), (13, 16),
    ]
    for row in rows:
        assert row["declaration_schema_version"] == 1
        assert row["mention_schema_version"] == MENTION_SCHEMA_VERSION
        assert row["declaration_normalization_id"] == MENTION_NORMALIZATION_ID
        assert row["mention_normalization_id"] == MENTION_NORMALIZATION_ID
        assert row["source_start"] == row["raw_source_start"] == 0
        assert row["source_end"] == len(canonical_source)
        assert row["raw_source_end"] == len(source)
        assert _canonical(
            source[row["raw_mention_start"]:row["raw_mention_end"]]
        ) == "ann"
        assert canonical_source[row["mention_start"]:row["mention_end"]] == "ann"
        assert row["source_span_sha256"] == hashlib.sha256(
            canonical_source.encode("utf-8")
        ).hexdigest()


def test_latin_boundaries_and_cjk_literal_overlap_are_exact_and_complete(tmp_path):
    store = anchor_store(tmp_path)
    latin = "Ann, Anna, Ann_1; ann."
    cjk = "\u54c8\u54c8\u54c8"
    latin_raw = _add_raw_message(
        store,
        content=latin,
        request_id="latin-request",
        session_id="latin-session",
    )
    cjk_raw = _add_raw_message(
        store,
        content=cjk,
        request_id="cjk-request",
        session_id="cjk-session",
    )

    _store_local_payload(store, latin_raw, labels=["Ann"])
    _store_local_payload(store, cjk_raw, labels=["\u54c8\u54c8"])

    with store._connection() as connection:
        latin_rows = connection.execute(
            """SELECT mention.raw_mention_start, mention.raw_mention_end
               FROM graph_entity_mentions AS mention
               JOIN graph_entity_declarations AS declaration
                 ON declaration.id = mention.declaration_id
                AND declaration.user_id = mention.user_id
                AND declaration.source_message_id = mention.source_message_id
               WHERE declaration.source_message_id = ?
               ORDER BY mention.raw_mention_start""",
            (int(latin_raw["id"]),),
        ).fetchall()
        cjk_rows = connection.execute(
            """SELECT mention.mention_start, mention.mention_end,
                      mention.raw_mention_start, mention.raw_mention_end
               FROM graph_entity_mentions AS mention
               JOIN graph_entity_declarations AS declaration
                 ON declaration.id = mention.declaration_id
                AND declaration.user_id = mention.user_id
                AND declaration.source_message_id = mention.source_message_id
               WHERE declaration.source_message_id = ?
               ORDER BY mention.mention_start""",
            (int(cjk_raw["id"]),),
        ).fetchall()

    assert [(row["raw_mention_start"], row["raw_mention_end"]) for row in latin_rows] == [
        (0, 3), (18, 21),
    ]
    # CJK labels intentionally use literal matching rather than a Latin word
    # boundary rule, so overlapping source occurrences are retained.
    assert [tuple(row) for row in cjk_rows] == [(0, 2, 0, 2), (1, 3, 1, 3)]


def test_fullwidth_and_combining_mentions_remap_to_raw_codepoint_offsets(tmp_path):
    store = anchor_store(tmp_path)
    source = "\uff21\uff4e\uff4e cafe\u0301 / CAF\u00c9"
    raw = _add_raw_message(store, content=source)

    _store_local_payload(store, raw, labels=["Ann", "caf\u00e9"])
    rows = _mention_rows(store)

    by_label = {}
    for row in rows:
        by_label.setdefault(row["mention_label"], []).append(row)
    assert [(row["raw_mention_start"], row["raw_mention_end"])
            for row in by_label["ann"]] == [(0, 3)]

    cafe_rows = by_label["caf\u00e9"]
    first_start = source.index("cafe\u0301")
    second_start = source.index("CAF\u00c9")
    assert [(row["raw_mention_start"], row["raw_mention_end"])
            for row in cafe_rows] == [
                (first_start, first_start + len("cafe\u0301")),
                (second_start, second_start + len("CAF\u00c9")),
            ]
    canonical_source = _canonical(source)
    assert [(row["mention_start"], row["mention_end"])
            for row in cafe_rows] == [
                (canonical_source.index("caf\u00e9"), canonical_source.index("caf\u00e9") + 4),
                (canonical_source.rindex("caf\u00e9"), canonical_source.rindex("caf\u00e9") + 4),
            ]
    for row in cafe_rows:
        raw_slice = source[row["raw_mention_start"]:row["raw_mention_end"]]
        assert _canonical(raw_slice) == "caf\u00e9"


@pytest.mark.parametrize(
    ("source", "partial_label"),
    [
        ("\u00df", "s"),
        ("\ufb03", "f"),
    ],
)
def test_internal_nfkc_or_casefold_expansion_boundaries_fail_closed(
    tmp_path, source, partial_label
):
    """No raw span may begin/end inside a one-codepoint normalization expansion."""

    store = anchor_store(tmp_path)
    raw = _add_raw_message(store, content=source)

    _store_local_payload(store, raw, labels=[partial_label])

    with store._connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_entity_declarations"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_entity_mentions"
        ).fetchone()[0] == 0


def test_schema_v4_enforces_compound_provenance_foreign_keys_uniqueness_and_checks(tmp_path):
    store = anchor_store(tmp_path)
    raw = _add_raw_message(store, content="Ann met Ann.")
    _store_local_payload(store, raw, labels=["Ann"])

    with store._connection() as connection:
        migration_versions = {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        assert 4 in migration_versions

        declaration_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(graph_entity_declarations)")
        }
        mention_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(graph_entity_mentions)")
        }
        assert {
            "id", "user_id", "source_message_id", "anchor_entity_id",
            "declaration_schema_version", "normalization_id", "entity_label",
            "created_at",
        }.issubset(declaration_columns)
        assert {
            "id", "declaration_id", "user_id", "source_message_id",
            "mention_schema_version", "normalization_id", "entity_label",
            "source_start", "source_end", "mention_start", "mention_end",
            "raw_source_start", "raw_source_end", "raw_mention_start",
            "raw_mention_end", "source_span_sha256",
        }.issubset(mention_columns)

        assert (
            "raw_messages", ("source_message_id", "user_id"), ("id", "user_id")
        ) in _foreign_key_columns(connection, "graph_entity_declarations")
        mention_foreign_keys = _foreign_key_columns(connection, "graph_entity_mentions")
        assert (
            "graph_entity_declarations",
            ("declaration_id", "user_id", "source_message_id"),
            ("id", "user_id", "source_message_id"),
        ) in mention_foreign_keys
        assert (
            "raw_messages", ("source_message_id", "user_id"), ("id", "user_id")
        ) in mention_foreign_keys

        declaration_unique = _unique_index_columns(
            connection, "graph_entity_declarations"
        )
        mention_unique = _unique_index_columns(connection, "graph_entity_mentions")
        assert ("id", "user_id", "source_message_id") in declaration_unique
        assert ("user_id", "source_message_id", "entity_label") in declaration_unique
        assert (
            "declaration_id", "user_id", "source_message_id",
            "mention_start", "mention_end",
        ) in mention_unique

        declaration_sql = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type = 'table' AND name = 'graph_entity_declarations'"""
        ).fetchone()["sql"]
        mention_sql = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type = 'table' AND name = 'graph_entity_mentions'"""
        ).fetchone()["sql"]
        declaration_sql = re.sub(r"\s+", "", declaration_sql).upper()
        mention_sql = re.sub(r"\s+", "", mention_sql).upper()
        assert "CHECK(DECLARATION_SCHEMA_VERSION=1)" in declaration_sql
        assert "CHECK(NORMALIZATION_ID='NFKC-CASEFOLD-WS-V1')" in declaration_sql
        assert "CHECK(MENTION_SCHEMA_VERSION=1)" in mention_sql
        assert "CHECK(NORMALIZATION_ID='NFKC-CASEFOLD-WS-V1')" in mention_sql
        assert "CHECK(SOURCE_START=0)" in mention_sql

        mention = connection.execute(
            "SELECT * FROM graph_entity_mentions ORDER BY id LIMIT 1"
        ).fetchone()
        assert mention is not None
        with pytest.raises(sqlite3.IntegrityError):
            _clone_row_without_id(connection, "graph_entity_mentions", mention)
        with pytest.raises(sqlite3.IntegrityError):
            _clone_row_without_id(
                connection,
                "graph_entity_mentions",
                mention,
                user_id="other-user",
            )
        with pytest.raises(sqlite3.IntegrityError):
            _clone_row_without_id(
                connection,
                "graph_entity_mentions",
                mention,
                source_start=1,
            )
        with pytest.raises(sqlite3.IntegrityError):
            _clone_row_without_id(
                connection,
                "graph_entity_mentions",
                mention,
                source_span_sha256="not-a-sha256",
            )


def test_duplicate_local_declarations_are_idempotent_per_source_and_isolated_by_user(tmp_path):
    source = "Ann met Ann."
    model = AnchorModel({
        source: [
            {"name": "Ann", "type": "person"},
            {"name": "ann", "type": "person"},
        ],
    })
    store = anchor_store(tmp_path, model=model)
    request = AddRequest(
        request_id="same-request",
        user_id="user-a",
        session_id="same-session",
        messages=[{"role": "speaker", "content": source, "timestamp": 1}],
    )

    store.add(request)
    store.add(request)  # idempotent Add must not duplicate raw/declaration/mention rows.
    assert model.extract_memory_calls == 1
    assert model.extract_facts_calls == 0
    with store._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_entity_declarations"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_entity_mentions"
        ).fetchone()[0] == 2
        first_anchor_id = connection.execute(
            "SELECT anchor_entity_id FROM graph_entity_declarations"
        ).fetchone()[0]

    # A different user with identical source text/label gets no shared parent
    # or child provenance.  The source-scoped hash must consequently differ.
    store.add(AddRequest(
        request_id="user-b-request",
        user_id="user-b",
        session_id="same-session",
        messages=[{"role": "speaker", "content": source, "timestamp": 1}],
    ))
    with store._connection() as connection:
        declarations = connection.execute(
            """SELECT user_id, source_message_id, anchor_entity_id, entity_label
               FROM graph_entity_declarations ORDER BY user_id"""
        ).fetchall()
        user_b_mentions = connection.execute(
            """SELECT COUNT(*) FROM graph_entity_mentions
               WHERE user_id = ?""",
            ("user-b",),
        ).fetchone()[0]
    assert [(row["user_id"], row["entity_label"]) for row in declarations] == [
        ("user-a", "ann"), ("user-b", "ann"),
    ]
    assert declarations[0]["anchor_entity_id"] == first_anchor_id
    assert declarations[0]["anchor_entity_id"] != declarations[1]["anchor_entity_id"]
    assert user_b_mentions == 2


def test_p3a_semantic_edge_count_is_preserved_and_anchor_mode_writes_no_relation_edges(tmp_path):
    source = "Alice works at Acme."

    p3a_model = RelationModel()
    p3a_store = MemoryStore(
        str(tmp_path / "p3a-v4.db"),
        model=p3a_model,
        structured_query_plan=True,
        evidence_graph=True,
    )
    p3a_store.initialize()
    _add_raw_message(
        p3a_store,
        content=source,
        request_id="p3a-request",
        session_id="p3a-session",
    )
    with p3a_store._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_edge_support"
        ).fetchone()[0] == 1
        assert 4 in {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations")
        }

    anchor_model = RelationModel()
    anchor_store_instance = anchor_store(
        tmp_path,
        model=anchor_model,
        database_name="anchors-only-v4.db",
    )
    _add_raw_message(
        anchor_store_instance,
        content=source,
        request_id="anchor-request",
        session_id="anchor-session",
    )
    with anchor_store_instance._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_edge_support"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_entity_declarations"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM graph_entity_mentions"
        ).fetchone()[0] == 2
