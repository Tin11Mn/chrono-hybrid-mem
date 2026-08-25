"""Independent adversarial checks for P3-B1 mention-sidecar auditing."""

from __future__ import annotations

import sqlite3

import pytest

from app.schemas import AddRequest
from app.storage import MemoryStore
from evaluation.evidence_graph.evidence_mention_audit import (
    audit_entity_mention_database,
    audit_entity_mentions,
)


class _AnchorModel:
    def __init__(self, entities_by_content):
        self.entities_by_content = entities_by_content

    def extract_memory(self, content, speaker=None, timestamp=None):
        del speaker, timestamp
        return {
            "facts": [],
            "entities": self.entities_by_content.get(content, []),
            "relations": [],
        }


def _store(tmp_path, *, include_second=False):
    database_path = tmp_path / "mentions.db"
    model = _AnchorModel({
        "Ann met Ann.": [{"name": "Ann", "type": "person"}],
        "Bob arrived.": [{"name": "Bob", "type": "person"}],
    })
    store = MemoryStore(
        str(database_path),
        model=model,
        structured_query_plan=True,
        evidence_anchors=True,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="one",
        user_id="user-a",
        session_id="session-a",
        messages=[{"role": "Ann", "content": "Ann met Ann.", "timestamp": 1}],
    ))
    if include_second:
        store.add(AddRequest(
            request_id="two",
            user_id="user-a",
            session_id="session-b",
            messages=[{"role": "Bob", "content": "Bob arrived.", "timestamp": 2}],
        ))
    return store, database_path


def _audit_store(store):
    with store._connection() as connection:
        return audit_entity_mentions(connection)


def test_auditor_reconstructs_every_valid_witness_and_read_only_database_path(tmp_path):
    store, database_path = _store(tmp_path)

    report = _audit_store(store)
    assert report["all_mentions_witnessed"] is True
    assert report["declaration_count"] == 1
    assert report["mention_count"] == 2
    assert report["raw_message_count"] == 1
    assert report["distinct_witnessed_source_count"] == 1
    assert report["source_coverage"] == 1.0
    assert report["violations"] == []

    read_only_report = audit_entity_mention_database(database_path)
    assert read_only_report == report


@pytest.mark.parametrize(
    "table,column,value,expected_reason",
    [
        ("graph_entity_mentions", "entity_label", "bob", "invalid_declaration_or_label"),
        ("graph_entity_mentions", "source_start", 1, "unwitnessed_mention_span"),
        ("graph_entity_mentions", "mention_end", 2, "unwitnessed_mention_span"),
        ("graph_entity_mentions", "raw_mention_end", 2, "unwitnessed_mention_span"),
        ("graph_entity_mentions", "source_span_sha256", "0" * 64, "unwitnessed_mention_span"),
        ("graph_entity_mentions", "normalization_id", "forged-v2", "invalid_declaration_or_label"),
        ("graph_entity_declarations", "anchor_entity_id", "0" * 64, "invalid_declaration_or_label"),
        ("graph_entity_declarations", "normalization_id", "forged-v2", "invalid_declaration_or_label"),
    ],
)
def test_auditor_rejects_forged_sidecar_fields(
    tmp_path, table, column, value, expected_reason
):
    store, _ = _store(tmp_path)
    with store._connection() as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        # Model a database that was tampered with outside the schema contract.
        # The production CHECK constraints are intentionally kept in place; the
        # independent auditor must nevertheless detect invalid persisted data.
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            "UPDATE {} SET {} = ? WHERE id = 1".format(table, column),
            (value,),
        )
        connection.execute("PRAGMA ignore_check_constraints=OFF")
        connection.execute("PRAGMA foreign_keys=ON")
        report = audit_entity_mentions(connection)

    assert report["all_mentions_witnessed"] is False
    assert expected_reason in {item["reason"] for item in report["violations"]}


def test_auditor_rejects_child_source_rebinding_even_if_target_raw_exists(tmp_path):
    store, _ = _store(tmp_path, include_second=True)
    with store._connection() as connection:
        second_source_id = connection.execute(
            "SELECT id FROM raw_messages WHERE content = ?", ("Bob arrived.",)
        ).fetchone()[0]
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "UPDATE graph_entity_mentions SET source_message_id = ? WHERE id = 1",
            (second_source_id,),
        )
        connection.execute("PRAGMA foreign_keys=ON")
        report = audit_entity_mentions(connection)

    assert report["all_mentions_witnessed"] is False
    assert "missing_parent_or_raw_source" in {
        item["reason"] for item in report["violations"]
    }


def test_empty_v4_sidecar_reports_zero_coverage_without_certifying_a_fake_anchor(tmp_path):
    store = MemoryStore(
        str(tmp_path / "empty.db"),
        model=_AnchorModel({}),
        structured_query_plan=True,
        evidence_anchors=True,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="empty",
        user_id="user-a",
        session_id="session-a",
        messages=[{"role": "speaker", "content": "No extracted entity.", "timestamp": 1}],
    ))

    report = _audit_store(store)
    assert report["all_mentions_witnessed"] is True
    assert report["mention_count"] == 0
    assert report["distinct_witnessed_source_count"] == 0
    assert report["source_coverage"] == 0.0
