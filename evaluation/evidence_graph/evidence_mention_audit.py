"""Evaluator-owned verification for P3-B1 source-exact mention anchors.

This module deliberately does not import production storage or graph-parser
helpers.  A prepared database is accepted only when this independent matcher
can reconstruct every persisted canonical and raw codepoint span directly from
the authoritative ``raw_messages`` row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Optional
import unicodedata


DECLARATION_SCHEMA_VERSION = 1
MENTION_SCHEMA_VERSION = 1
NORMALIZATION_ID = "nfkc-casefold-ws-v1"
ANCHOR_ID_NAMESPACE = "p3b-anchor-v1"
ENTITY_LABEL_MAX_CHARS = 128
_UNSEGMENTED_SCRIPT_MARKERS = (
    "CJK UNIFIED IDEOGRAPH",
    "CJK COMPATIBILITY IDEOGRAPH",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "THAI",
    "LAO",
    "KHMER",
    "MYANMAR",
)


def normalize_mention_source(value: object) -> Optional[str]:
    """Use the frozen NFKC/casefold/all-whitespace-collapse contract."""

    if not isinstance(value, str):
        return None
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def source_local_anchor_entity_id(
    *, user_id: str, source_message_id: int, entity_label: str
) -> str:
    material = "\0".join((
        ANCHOR_ID_NAMESPACE,
        str(user_id),
        str(source_message_id),
        str(entity_label),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _is_unsegmented_character(value: str) -> bool:
    name = unicodedata.name(value, "")
    return any(marker in name for marker in _UNSEGMENTED_SCRIPT_MARKERS)


def _is_word_character(value: str) -> bool:
    category = unicodedata.category(value)
    return (
        value == "_"
        or category[0] in {"L", "N", "M"}
        or category in {"Pc", "Cf"}
    )


def _requires_boundaries(entity_label: str) -> bool:
    word_characters = [
        character for character in entity_label if _is_word_character(character)
    ]
    return bool(word_characters) and not all(
        _is_unsegmented_character(character) for character in word_characters
    )


def _token_boundary_map(
    raw_token: str, *, raw_offset: int
) -> Optional[tuple[str, list[Optional[int]]]]:
    canonical = unicodedata.normalize("NFKC", raw_token).casefold()
    if not canonical or any(character.isspace() for character in canonical):
        return None
    candidates: dict[int, list[int]] = {}
    for raw_index in range(len(raw_token) + 1):
        prefix = unicodedata.normalize("NFKC", raw_token[:raw_index]).casefold()
        if canonical.startswith(prefix):
            candidates.setdefault(len(prefix), []).append(raw_offset + raw_index)
    boundaries: list[Optional[int]] = [None] * (len(canonical) + 1)
    for canonical_index, raw_indexes in candidates.items():
        if len(raw_indexes) == 1:
            boundaries[canonical_index] = raw_indexes[0]
    if boundaries[0] != raw_offset or boundaries[-1] != raw_offset + len(raw_token):
        return None
    return canonical, boundaries


def _canonical_source_with_raw_boundaries(
    source_content: str,
) -> Optional[tuple[str, list[Optional[int]]]]:
    expected = normalize_mention_source(source_content)
    if not expected:
        return None
    runs: list[tuple[bool, int, int]] = []
    start = 0
    is_space = source_content[0].isspace()
    for index, character in enumerate(source_content[1:], start=1):
        next_is_space = character.isspace()
        if next_is_space != is_space:
            runs.append((is_space, start, index))
            start = index
            is_space = next_is_space
    runs.append((is_space, start, len(source_content)))

    parts: list[str] = []
    boundaries: list[Optional[int]] = []
    pending_whitespace: Optional[tuple[int, int]] = None
    for is_space_run, run_start, run_end in runs:
        if is_space_run:
            if parts:
                pending_whitespace = (run_start, run_end)
            continue
        token = _token_boundary_map(
            source_content[run_start:run_end], raw_offset=run_start
        )
        if token is None:
            return None
        canonical_token, token_boundaries = token
        if parts:
            if pending_whitespace is None:
                return None
            whitespace_start, whitespace_end = pending_whitespace
            if boundaries[-1] != whitespace_start:
                return None
            parts.append(" ")
            boundaries.append(whitespace_end)
        else:
            boundaries.append(token_boundaries[0])
        if boundaries[-1] != token_boundaries[0]:
            return None
        parts.append(canonical_token)
        boundaries.extend(token_boundaries[1:])
        pending_whitespace = None
    canonical_source = "".join(parts)
    if canonical_source != expected or len(boundaries) != len(canonical_source) + 1:
        return None
    return canonical_source, boundaries


def exact_source_entity_mentions(
    source_content: str, entity_label: object
) -> list[dict[str, Any]]:
    """Reconstruct every valid P3-B1 witness without production code."""

    normalized_label = normalize_mention_source(entity_label)
    source = _canonical_source_with_raw_boundaries(source_content)
    if (
        not normalized_label
        or len(normalized_label) > ENTITY_LABEL_MAX_CHARS
        or source is None
    ):
        return []
    canonical_source, boundaries = source
    source_hash = hashlib.sha256(canonical_source.encode("utf-8")).hexdigest()
    requires_boundaries = _requires_boundaries(normalized_label)
    result: list[dict[str, Any]] = []
    start = canonical_source.find(normalized_label)
    while start >= 0:
        end = start + len(normalized_label)
        previous = canonical_source[start - 1] if start else None
        following = canonical_source[end] if end < len(canonical_source) else None
        boundary_blocked = requires_boundaries and (
            (previous is not None and _is_word_character(previous))
            or (following is not None and _is_word_character(following))
        )
        if not boundary_blocked:
            raw_start, raw_end = boundaries[start], boundaries[end]
            if (
                raw_start is not None
                and raw_end is not None
                and raw_start < raw_end
                and normalize_mention_source(source_content[raw_start:raw_end])
                == normalized_label
            ):
                result.append({
                    "source_start": 0,
                    "source_end": len(canonical_source),
                    "mention_start": start,
                    "mention_end": end,
                    "raw_source_start": 0,
                    "raw_source_end": len(source_content),
                    "raw_mention_start": raw_start,
                    "raw_mention_end": raw_end,
                    "source_span_sha256": source_hash,
                })
        start = canonical_source.find(normalized_label, start + 1)
    return result


_DECLARATION_COLUMNS = {
    "id", "user_id", "source_message_id", "anchor_entity_id",
    "declaration_schema_version", "normalization_id", "entity_label",
    "created_at",
}
_MENTION_COLUMNS = {
    "id", "declaration_id", "user_id", "source_message_id",
    "mention_schema_version", "normalization_id", "entity_label",
    "source_start", "source_end", "mention_start", "mention_end",
    "raw_source_start", "raw_source_end", "raw_mention_start",
    "raw_mention_end", "source_span_sha256",
}


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info({})".format(table_name))
    }


def _violation(
    violations: list[dict[str, Any]],
    *,
    reason: str,
    mention_id: object = None,
    declaration_id: object = None,
) -> None:
    if len(violations) >= 100:
        return
    payload: dict[str, Any] = {"reason": reason}
    if mention_id is not None:
        payload["mention_id"] = int(mention_id)
    if declaration_id is not None:
        payload["declaration_id"] = int(declaration_id)
    violations.append(payload)


def audit_entity_mentions(connection: sqlite3.Connection) -> dict[str, Any]:
    """Audit every P3-B1 declaration and occurrence in an open SQLite DB."""

    connection.row_factory = sqlite3.Row
    violations: list[dict[str, Any]] = []
    table_names = {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    for table_name, required in (
        ("graph_entity_declarations", _DECLARATION_COLUMNS),
        ("graph_entity_mentions", _MENTION_COLUMNS),
        ("raw_messages", {"id", "user_id", "content"}),
    ):
        if table_name not in table_names or not required.issubset(
            _table_columns(connection, table_name)
        ):
            _violation(violations, reason="missing_or_incompatible_{}".format(table_name))
    if violations:
        return {
            "declaration_schema_version": DECLARATION_SCHEMA_VERSION,
            "mention_schema_version": MENTION_SCHEMA_VERSION,
            "normalization_id": NORMALIZATION_ID,
            "declaration_count": 0,
            "mention_count": 0,
            "raw_message_count": 0,
            "distinct_mentioned_source_count": 0,
            "distinct_witnessed_source_count": 0,
            "source_coverage": 0.0,
            "all_mentions_witnessed": False,
            "violations": violations,
        }

    raw_message_count = int(connection.execute(
        "SELECT COUNT(*) FROM raw_messages"
    ).fetchone()[0])
    declaration_count = int(connection.execute(
        "SELECT COUNT(*) FROM graph_entity_declarations"
    ).fetchone()[0])
    mention_count = int(connection.execute(
        "SELECT COUNT(*) FROM graph_entity_mentions"
    ).fetchone()[0])
    mentioned_sources = {
        int(row[0]) for row in connection.execute(
            "SELECT DISTINCT source_message_id FROM graph_entity_mentions"
        )
    }
    witnessed_sources: set[int] = set()

    orphan_rows = connection.execute(
        """SELECT declaration.id
           FROM graph_entity_declarations AS declaration
           LEFT JOIN graph_entity_mentions AS mention
             ON mention.declaration_id = declaration.id
            AND mention.user_id = declaration.user_id
            AND mention.source_message_id = declaration.source_message_id
           WHERE mention.id IS NULL"""
    ).fetchall()
    for row in orphan_rows:
        _violation(violations, reason="declaration_without_mention", declaration_id=row[0])

    rows = connection.execute(
        """SELECT mention.*, declaration.id AS declaration_actual_id,
                  declaration.user_id AS declaration_user_id,
                  declaration.source_message_id AS declaration_source_message_id,
                  declaration.anchor_entity_id,
                  declaration.declaration_schema_version,
                  declaration.normalization_id AS declaration_normalization_id,
                  declaration.entity_label AS declaration_entity_label,
                  raw.id AS raw_id, raw.user_id AS raw_user_id,
                  raw.content AS raw_content
           FROM graph_entity_mentions AS mention
           LEFT JOIN graph_entity_declarations AS declaration
             ON declaration.id = mention.declaration_id
            AND declaration.user_id = mention.user_id
            AND declaration.source_message_id = mention.source_message_id
           LEFT JOIN raw_messages AS raw
             ON raw.id = mention.source_message_id
            AND raw.user_id = mention.user_id
           ORDER BY mention.id"""
    ).fetchall()
    for row in rows:
        mention_id = row["id"]
        declaration_id = row["declaration_id"]
        if row["declaration_actual_id"] is None or row["raw_id"] is None:
            _violation(
                violations,
                reason="missing_parent_or_raw_source",
                mention_id=mention_id,
                declaration_id=declaration_id,
            )
            continue
        label = row["entity_label"]
        if (
            not isinstance(label, str)
            or normalize_mention_source(label) != label
            or not label
            or len(label) > ENTITY_LABEL_MAX_CHARS
            or row["declaration_entity_label"] != label
            or row["declaration_user_id"] != row["user_id"]
            or row["declaration_source_message_id"] != row["source_message_id"]
            or row["raw_user_id"] != row["user_id"]
            or int(row["declaration_schema_version"])
            != DECLARATION_SCHEMA_VERSION
            or int(row["mention_schema_version"]) != MENTION_SCHEMA_VERSION
            or row["declaration_normalization_id"] != NORMALIZATION_ID
            or row["normalization_id"] != NORMALIZATION_ID
            or row["anchor_entity_id"]
            != source_local_anchor_entity_id(
                user_id=str(row["user_id"]),
                source_message_id=int(row["source_message_id"]),
                entity_label=label,
            )
        ):
            _violation(
                violations,
                reason="invalid_declaration_or_label",
                mention_id=mention_id,
                declaration_id=declaration_id,
            )
            continue
        expected_witnesses = exact_source_entity_mentions(
            str(row["raw_content"]), label
        )
        fields = (
            "source_start", "source_end", "mention_start", "mention_end",
            "raw_source_start", "raw_source_end", "raw_mention_start",
            "raw_mention_end", "source_span_sha256",
        )
        if not any(
            all(witness[field] == row[field] for field in fields)
            for witness in expected_witnesses
        ):
            _violation(
                violations,
                reason="unwitnessed_mention_span",
                mention_id=mention_id,
                declaration_id=declaration_id,
            )
            continue
        witnessed_sources.add(int(row["source_message_id"]))

    duplicate_rows = connection.execute(
        """SELECT source_message_id, user_id, entity_label,
                  mention_start, mention_end, COUNT(*) AS count
           FROM graph_entity_mentions
           GROUP BY source_message_id, user_id, entity_label,
                    mention_start, mention_end
           HAVING COUNT(*) > 1"""
    ).fetchall()
    for row in duplicate_rows:
        _violation(violations, reason="duplicate_mention_occurrence")

    coverage = (
        len(witnessed_sources) / raw_message_count if raw_message_count else 0.0
    )
    return {
        "declaration_schema_version": DECLARATION_SCHEMA_VERSION,
        "mention_schema_version": MENTION_SCHEMA_VERSION,
        "normalization_id": NORMALIZATION_ID,
        "declaration_count": declaration_count,
        "mention_count": mention_count,
        "raw_message_count": raw_message_count,
        "distinct_mentioned_source_count": len(mentioned_sources),
        "distinct_witnessed_source_count": len(witnessed_sources),
        "source_coverage": coverage,
        "all_mentions_witnessed": not violations,
        "violations": violations,
    }


def audit_entity_mention_database(database_path: str | Path) -> dict[str, Any]:
    """Open one immutable artifact read-only and return its audit report."""

    path = Path(database_path).resolve()
    connection = sqlite3.connect("{}?mode=ro".format(path.as_uri()), uri=True)
    try:
        return audit_entity_mentions(connection)
    finally:
        connection.close()
