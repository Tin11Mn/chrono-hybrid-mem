"""Fair, cache-backed paired P1/P3-A retrieval evaluation on LoCoMo.

This runner deliberately separates the expensive Add-time extraction from the
two Search arms.  Every selected source message is extracted exactly once,
the resulting composite payload is replayed through the production
``MemoryStore.add`` path, and graph-off/graph-on stores then search the same
prepared SQLite database without calling Add again.

The module is usable as a library with fake models for an entirely offline
mechanics test.  The CLI is stricter: unless explicitly relaxed for a local
diagnostic it requires the production store to expose a complete pre-rerank
trace, so a graph score cannot be reported without proving the 30-item quota
mechanics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from app.model import MODEL_NAME, MemoryModel
from app.schemas import AddRequest
from app.storage import MemoryStore
from evaluation.evidence_graph.p3a_gate import (
    build_relation_subset_manifest,
    evaluate_p3a_gate,
    validate_p3a_formal_configuration,
)
from evaluation.evidence_graph.support_witness_audit import (
    FROZEN_SPEC_REGISTRY_FINGERPRINT,
    NORMALIZATION_ID as SUPPORT_NORMALIZATION_ID,
    SUPPORT_SCHEMA_VERSION,
    audit_graph_trace_paths,
    audit_persisted_graph_support,
)
from evaluation.evidence_graph.evidence_mention_audit import (
    DECLARATION_SCHEMA_VERSION as MENTION_DECLARATION_SCHEMA_VERSION,
    MENTION_SCHEMA_VERSION,
    NORMALIZATION_ID as MENTION_NORMALIZATION_ID,
    audit_entity_mention_database,
)
from scripts.evaluate_locomo_retrieval import session_number, session_timestamp


CACHE_SCHEMA_VERSION = 1
PREPARED_SCHEMA_VERSION = 4
TRACE_SCHEMA_VERSION = 2
MAX_WORKERS = 8
P3A_FORMAL_QUESTION_COUNT = 200


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _artifact_sha256(value: object, label: str) -> str:
    """Normalize an explicit model-artifact SHA-256 fingerprint."""

    fingerprint = _required_text(value, label).casefold()
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return fingerprint


def _record_sequence(value: object, label: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ValueError(f"{label} must be a sequence of records")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{index}] must be a record")
        records.append(item)
    return records


def _message_content(turn: Mapping[str, Any]) -> str:
    """Render exactly the same source text as the legacy LoCoMo evaluator."""

    speaker = _required_text(turn.get("speaker"), "turn.speaker")
    text = _required_text(turn.get("text"), "turn.text")
    content = f"{speaker}: {text}"
    caption = str(turn.get("blip_caption", "")).strip()
    if caption:
        content += f" Shared image: {caption}"
    return content


def _source_key(sample_id: str, session_key: str, dia_id: str) -> str:
    # NUL is forbidden by JSON text inputs and avoids delimiter ambiguity.
    return "\0".join((sample_id, session_key, dia_id))


def _query_key(question: str, options: Sequence[str]) -> str:
    return _sha256_json({"question": question, "options": list(options)})


def _manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "dataset_sha256": manifest["dataset_sha256"],
        "question_offset": manifest["question_offset"],
        "max_questions": manifest["max_questions"],
        "samples": manifest["samples"],
        "questions": manifest["questions"],
    }


def build_dataset_manifest(
    samples: object,
    *,
    dataset_sha256: str,
    max_questions: int | None,
    question_offset: int = 0,
) -> dict[str, Any]:
    """Build the authoritative, ordered ingestion/question scope.

    Selection intentionally mirrors ``evaluate_locomo_retrieval.evaluate``:
    a conversation is ingested in full before its eligible questions are
    consumed, and a prefix that ends inside a conversation still includes all
    of that conversation's source messages.
    """

    dataset_sha256 = _required_text(dataset_sha256, "dataset_sha256")
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be positive when supplied")
    if question_offset < 0:
        raise ValueError("question_offset must be non-negative")
    if not isinstance(samples, list):
        raise ValueError("LoCoMo dataset must be a JSON array")

    selected_samples: list[dict[str, Any]] = []
    selected_questions: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    eligible_index = 0

    for sample_index, sample in enumerate(samples):
        if max_questions is not None and len(selected_questions) >= max_questions:
            break
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample[{sample_index}] must be a record")
        sample_id = _required_text(
            sample.get("sample_id", f"sample-{sample_index}"),
            f"sample[{sample_index}].sample_id",
        )
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        conversation = sample.get("conversation")
        if not isinstance(conversation, Mapping):
            raise ValueError(f"conversation for {sample_id} must be a record")

        user_id = f"locomo:{sample_id}"
        evidence_ids: set[str] = set()
        seen_dia_ids: set[str] = set()
        manifest_sessions: list[dict[str, Any]] = []
        session_items = sorted(
            (
                (str(key), value)
                for key, value in conversation.items()
                if isinstance(value, list)
            ),
            key=lambda item: session_number(item[0]),
        )
        for session_key, turns in session_items:
            event_ts = session_timestamp(dict(conversation), session_key)
            session_messages: list[dict[str, Any]] = []
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, Mapping) or not all(
                    field in turn for field in ("dia_id", "speaker", "text")
                ):
                    continue
                dia_id = _required_text(
                    turn.get("dia_id"),
                    f"{sample_id}.{session_key}[{turn_index}].dia_id",
                )
                if dia_id in seen_dia_ids:
                    raise ValueError(
                        f"duplicate dia_id within sample {sample_id}: {dia_id}"
                    )
                seen_dia_ids.add(dia_id)
                evidence_ids.add(dia_id)
                speaker = _required_text(turn.get("speaker"), "turn.speaker")
                content = _message_content(turn)
                session_messages.append(
                    {
                        "source_key": _source_key(sample_id, session_key, dia_id),
                        "sample_id": sample_id,
                        "session_key": session_key,
                        "dia_id": dia_id,
                        "user_id": user_id,
                        "speaker": speaker,
                        "content": content,
                        "content_sha256": _sha256_bytes(content.encode("utf-8")),
                        "event_ts": event_ts,
                        "sequence": len(session_messages),
                    }
                )
            if session_messages:
                manifest_sessions.append(
                    {
                        "session_key": session_key,
                        "session_id": f"locomo:{sample_id}:{session_key}",
                        "request_id": f"locomo:{sample_id}:{session_key}",
                        "messages": session_messages,
                    }
                )

        if not manifest_sessions:
            continue
        selected_samples.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "user_id": user_id,
                "sessions": manifest_sessions,
            }
        )

        qa_items = sample.get("qa", [])
        if not isinstance(qa_items, list):
            raise ValueError(f"qa for {sample_id} must be a list")
        for qa_index, qa in enumerate(qa_items):
            if max_questions is not None and len(selected_questions) >= max_questions:
                break
            if not isinstance(qa, Mapping):
                continue
            question = str(qa.get("question", "")).strip()
            raw_evidence = qa.get("evidence", [])
            if not isinstance(raw_evidence, list):
                continue
            gold_dia_ids = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_evidence
                    if str(item).strip() in evidence_ids
                )
            )
            if not question or not gold_dia_ids:
                continue
            current_eligible = eligible_index
            eligible_index += 1
            if current_eligible < question_offset:
                continue
            raw_options = qa.get("options", [])
            options = (
                [str(item) for item in raw_options]
                if isinstance(raw_options, list)
                else []
            )
            selected_questions.append(
                {
                    "case_id": f"{sample_id}:qa:{qa_index}",
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "qa_index": qa_index,
                    "eligible_index": current_eligible,
                    "question": question,
                    "options": options,
                    "query_key": _query_key(question, options),
                    "category": str(qa.get("category", "unknown")),
                    "gold_dia_ids": gold_dia_ids,
                }
            )

    if not selected_questions:
        raise ValueError("the selected LoCoMo scope contains no evaluable questions")

    all_messages = [
        message
        for sample in selected_samples
        for session in sample["sessions"]
        for message in session["messages"]
    ]
    seen_source_keys: set[str] = set()
    for message in all_messages:
        source_key = str(message["source_key"])
        if source_key in seen_source_keys:
            raise ValueError(f"duplicate source key: {source_key!r}")
        seen_source_keys.add(source_key)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "question_offset": question_offset,
        "max_questions": max_questions,
        "samples": selected_samples,
        "questions": selected_questions,
        "message_count": len(all_messages),
        "question_count": len(selected_questions),
    }
    manifest["manifest_sha256"] = _sha256_json(_manifest_hash_payload(manifest))
    return manifest


def load_dataset_manifest(
    dataset_path: str | Path,
    *,
    max_questions: int | None,
    question_offset: int = 0,
) -> dict[str, Any]:
    path = Path(dataset_path)
    raw = path.read_bytes()
    try:
        samples = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON dataset: {path}") from error
    manifest = build_dataset_manifest(
        samples,
        dataset_sha256=_sha256_bytes(raw),
        max_questions=max_questions,
        question_offset=question_offset,
    )
    manifest["dataset_path"] = str(path.resolve())
    return manifest


def manifest_messages(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for sample in _record_sequence(manifest.get("samples"), "manifest.samples"):
        for session in _record_sequence(sample.get("sessions"), "sample.sessions"):
            result.extend(_record_sequence(session.get("messages"), "session.messages"))
    if len(result) != int(manifest.get("message_count", -1)):
        raise ValueError("manifest message_count does not match its records")
    return result


def extraction_contract_fingerprint() -> str:
    """Bind raw extraction caches to the checked-out model contract."""

    model_path = Path(__file__).resolve().parents[2] / "app" / "model.py"
    return _sha256_bytes(model_path.read_bytes())


def materialization_contract_fingerprint() -> str:
    """Bind prepared DBs to the code that sanitizes and materializes them."""

    repository_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "app/storage.py",
        "app/evidence_graph.py",
        "app/graph_routing.py",
        "evaluation/evidence_graph/support_witness_audit.py",
        "evaluation/evidence_graph/evidence_mention_audit.py",
    )
    components = {
        relative_path: _sha256_bytes(
            (repository_root / relative_path).read_bytes()
        )
        for relative_path in relative_paths
    }
    return _sha256_json(
        {
            "schema_version": 3,
            "component_sha256": components,
            "support_schema_version": SUPPORT_SCHEMA_VERSION,
            "support_normalization_id": SUPPORT_NORMALIZATION_ID,
            "support_spec_registry_fingerprint": FROZEN_SPEC_REGISTRY_FINGERPRINT,
            "mention_declaration_schema_version": MENTION_DECLARATION_SCHEMA_VERSION,
            "mention_schema_version": MENTION_SCHEMA_VERSION,
            "mention_normalization_id": MENTION_NORMALIZATION_ID,
        }
    )


def extraction_cache_fingerprint(
    manifest: Mapping[str, Any],
    *,
    model_fingerprint: str,
    ingest_artifact_fingerprint: str,
    contract_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_sha256": _required_text(
            manifest.get("dataset_sha256"), "manifest.dataset_sha256"
        ),
        "model_fingerprint": _required_text(
            model_fingerprint, "model_fingerprint"
        ),
        "ingest_artifact_fingerprint": _artifact_sha256(
            ingest_artifact_fingerprint, "ingest_artifact_fingerprint"
        ),
        "contract_fingerprint": _required_text(
            contract_fingerprint, "contract_fingerprint"
        ),
    }


def _cache_record_index(
    cache: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported extraction cache schema")
    records = _record_sequence(cache.get("records"), "cache.records")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        source_key = _required_text(
            record.get("source_key"), f"cache.records[{index}].source_key"
        )
        if source_key in indexed:
            raise ValueError(f"duplicate extraction cache source_key: {source_key!r}")
        if not isinstance(record.get("raw_extraction"), Mapping):
            raise ValueError(
                f"cache record {source_key!r} raw_extraction must be a record"
            )
        indexed[source_key] = record
    return indexed


def _read_extraction_cache(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid extraction cache: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("extraction cache must be a JSON object")
    _cache_record_index(value)
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_cached_message(
    message: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    source_key = str(message["source_key"])
    expected = {
        "source_key": source_key,
        "sample_id": message["sample_id"],
        "session_key": message["session_key"],
        "dia_id": message["dia_id"],
        "user_id": message["user_id"],
        "speaker": message["speaker"],
        "event_ts": message["event_ts"],
        "content_sha256": message["content_sha256"],
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(
                f"extraction cache metadata mismatch for {source_key!r}: {field}"
            )


def ensure_extraction_cache(
    manifest: Mapping[str, Any],
    cache_path: str | Path,
    *,
    fingerprint: Mapping[str, Any],
    model: object | None,
    mode: str = "reuse",
    workers: int = 1,
) -> dict[str, Any]:
    """Build, reuse, or monotonically extend a composite extraction cache."""

    if mode not in {"build", "reuse", "require", "extend"}:
        raise ValueError("cache mode must be build, reuse, require, or extend")
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    path = Path(cache_path)
    expected_fingerprint = dict(fingerprint)
    messages = manifest_messages(manifest)

    if path.exists():
        if mode == "build":
            raise FileExistsError(f"extraction cache already exists: {path}")
        cache = _read_extraction_cache(path)
        if cache.get("fingerprint") != expected_fingerprint:
            raise ValueError("extraction cache fingerprint mismatch")
    else:
        if mode in {"reuse", "require"}:
            raise FileNotFoundError(f"extraction cache does not exist: {path}")
        cache = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": expected_fingerprint,
            "records": [],
        }

    indexed = _cache_record_index(cache)
    for message in messages:
        cached = indexed.get(str(message["source_key"]))
        if cached is not None:
            _validate_cached_message(message, cached)
    missing = [
        message for message in messages if str(message["source_key"]) not in indexed
    ]
    if missing and mode in {"reuse", "require"}:
        preview = [str(item["source_key"]) for item in missing[:5]]
        raise ValueError(
            f"extraction cache is missing {len(missing)} selected messages: {preview}"
        )
    if missing and model is None:
        raise ValueError("a graph-capable extraction model is required for cache misses")
    if missing and not callable(getattr(model, "extract_memory", None)):
        raise TypeError("extraction model must expose extract_memory")

    starting_calls = getattr(model, "call_count", None) if model is not None else None

    def extract(message: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        raw = model.extract_memory(  # type: ignore[union-attr]
            str(message["content"]),
            speaker=str(message["speaker"]),
            timestamp=message.get("event_ts"),
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"extract_memory returned a non-record for {message['source_key']!r}"
            )
        return {
            "source_key": message["source_key"],
            "sample_id": message["sample_id"],
            "session_key": message["session_key"],
            "dia_id": message["dia_id"],
            "user_id": message["user_id"],
            "speaker": message["speaker"],
            "event_ts": message["event_ts"],
            "content_sha256": message["content_sha256"],
            "raw_extraction": copy.deepcopy(dict(raw)),
            "latency_seconds": time.perf_counter() - started,
        }

    pending_since_checkpoint = 0

    def checkpoint() -> None:
        nonlocal pending_since_checkpoint
        cache["records"] = [indexed[key] for key in sorted(indexed)]
        cache["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(path, cache)
        pending_since_checkpoint = 0

    def accept(record: Mapping[str, Any]) -> None:
        nonlocal pending_since_checkpoint
        source_key = str(record["source_key"])
        if source_key in indexed:
            raise RuntimeError(
                f"cache extension produced duplicate key: {source_key!r}"
            )
        indexed[source_key] = dict(record)
        pending_since_checkpoint += 1
        if pending_since_checkpoint >= workers:
            checkpoint()

    if missing:
        # Persist the cache identity before model work, then checkpoint completed
        # payloads from the main thread.  A late failure therefore leaves a
        # valid monotonic prefix/set that ``extend`` can resume without paying
        # for successful extractions again.
        if not path.exists():
            checkpoint()
        failure: Exception | None = None
        if workers == 1:
            for message in missing:
                try:
                    accept(extract(message))
                except Exception as error:
                    failure = error
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(extract, message) for message in missing]
                for future in as_completed(futures):
                    try:
                        accept(future.result())
                    except Exception as error:
                        # Continue draining: other already-submitted calls may
                        # succeed and every such paid payload must be retained.
                        if failure is None:
                            failure = error
        if pending_since_checkpoint:
            checkpoint()
        if failure is not None:
            raise failure

    ending_calls = getattr(model, "call_count", None) if model is not None else None
    if starting_calls is not None and ending_calls is not None:
        observed_calls = int(ending_calls) - int(starting_calls)
        if observed_calls != len(missing):
            raise RuntimeError(
                f"expected {len(missing)} extraction calls, observed {observed_calls}"
            )
    else:
        observed_calls = len(missing)

    # Re-read after an update so duplicate/corruption checks cover bytes on disk.
    cache = _read_extraction_cache(path)
    indexed = _cache_record_index(cache)
    for message in messages:
        cached = indexed.get(str(message["source_key"]))
        if cached is None:
            raise RuntimeError("cache write lost a selected extraction record")
        _validate_cached_message(message, cached)
    return {
        "path": str(path.resolve()),
        "fingerprint": expected_fingerprint,
        "selected_messages": len(messages),
        "cache_records": len(indexed),
        "cache_hits": len(messages) - len(missing),
        "model_calls": observed_calls,
        "one_call_per_missing_message": observed_calls == len(missing),
        "sha256": _sha256_bytes(path.read_bytes()),
        "records_by_source_key": indexed,
    }


class _SessionReplayModel:
    """Replay cached payloads in the exact order used by one bundled Add."""

    def __init__(self) -> None:
        self._expected: list[Mapping[str, Any]] = []
        self._position = 0
        self.extract_memory_calls = 0

    def begin(self, records: Sequence[Mapping[str, Any]]) -> None:
        if self._position != len(self._expected):
            raise RuntimeError("previous replay session was not fully consumed")
        self._expected = list(records)
        self._position = 0

    def extract_memory(
        self, content: str, speaker: str = "", timestamp: int | None = None
    ) -> dict[str, Any]:
        if self._position >= len(self._expected):
            raise RuntimeError("Add requested an unexpected extraction payload")
        record = self._expected[self._position]
        self._position += 1
        self.extract_memory_calls += 1
        if record.get("speaker") != speaker:
            raise ValueError("replay speaker does not match the authoritative manifest")
        if record.get("event_ts") != timestamp:
            raise ValueError("replay timestamp does not match the authoritative manifest")
        if record.get("content_sha256") != _sha256_bytes(content.encode("utf-8")):
            raise ValueError("replay content hash does not match the authoritative manifest")
        raw = record.get("raw_extraction")
        if not isinstance(raw, Mapping):
            raise ValueError("cached raw_extraction must be a record")
        return copy.deepcopy(dict(raw))

    def finish(self) -> None:
        if self._position != len(self._expected):
            raise RuntimeError(
                f"Add consumed {self._position}/{len(self._expected)} replay payloads"
            )


def _normalize_sql_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def logical_database_digest(database_path: str | Path) -> str:
    """Hash schema and table contents without depending on SQLite page layout."""

    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload: list[object] = []
    connection = sqlite3.connect(str(path))
    try:
        tables = connection.execute(
            """SELECT name, sql FROM sqlite_master
               WHERE type = 'table'
                 AND (name NOT LIKE 'sqlite_%' OR name = 'sqlite_sequence')
               ORDER BY name"""
        ).fetchall()
        for table_name, schema_sql in tables:
            quoted = '"{}"'.format(str(table_name).replace('"', '""'))
            columns = [
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({quoted})").fetchall()
            ]
            rows = [
                [_normalize_sql_value(value) for value in row]
                for row in connection.execute(f"SELECT * FROM {quoted}").fetchall()
            ]
            rows.sort(key=_canonical_json)
            payload.append(
                {
                    "table": table_name,
                    "schema": schema_sql,
                    "columns": columns,
                    "rows": rows,
                }
            )
    finally:
        connection.close()
    return _sha256_json(payload)


def _prepared_sidecar_path(database_path: Path) -> Path:
    return database_path.with_name(database_path.name + ".manifest.json")


def _prepared_support_invariants(database_path: str | Path) -> dict[str, Any]:
    """Return evaluator-owned witness counts for a materialized graph DB.

    Prepared artifacts are accepted only when every persisted graph edge has a
    valid one-to-one support witness.  The evaluator reopens SQLite itself;
    this is intentionally not derived from a storage diagnostic or a parser
    boolean.
    """

    audit = audit_persisted_graph_support(database_path)
    if audit["violations"] or not audit["all_edges_witnessed"]:
        raise RuntimeError(
            "prepared graph support audit failed: {}".format(
                list(audit["violations"])[:5]
            )
        )
    return {
        "support_schema_version": SUPPORT_SCHEMA_VERSION,
        "support_normalization_id": SUPPORT_NORMALIZATION_ID,
        "support_spec_registry_fingerprint": FROZEN_SPEC_REGISTRY_FINGERPRINT,
        "graph_edge_count": int(audit["edge_count"]),
        "graph_edge_support_count": int(audit["support_count"]),
        "all_graph_edges_witnessed": bool(audit["all_edges_witnessed"]),
    }


def _prepared_mention_invariants(database_path: str | Path) -> dict[str, Any]:
    """Return independent P3-B1 source-mention evidence for a prepared DB.

    This remains a structural/materialization audit, not a retrieval outcome
    gate.  A database with no extracted entities is valid but reports zero
    source coverage; P3-B1's separate preflight applies its fixed coverage
    threshold before any paired Search.
    """

    audit = audit_entity_mention_database(database_path)
    if audit["violations"] or not audit["all_mentions_witnessed"]:
        raise RuntimeError(
            "prepared entity-mention audit failed: {}".format(
                list(audit["violations"])[:5]
            )
        )
    return {
        "mention_declaration_schema_version": (
            MENTION_DECLARATION_SCHEMA_VERSION
        ),
        "mention_schema_version": MENTION_SCHEMA_VERSION,
        "mention_normalization_id": MENTION_NORMALIZATION_ID,
        "graph_entity_declaration_count": int(audit["declaration_count"]),
        "graph_entity_mention_count": int(audit["mention_count"]),
        "mention_source_coverage": float(audit["source_coverage"]),
        "all_entity_mentions_witnessed": bool(audit["all_mentions_witnessed"]),
    }


def _validate_source_map(
    manifest: Mapping[str, Any], source_map_records: object
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    records = _record_sequence(source_map_records, "source_map")
    messages = manifest_messages(manifest)
    expected_keys = {str(message["source_key"]) for message in messages}
    by_source: dict[str, Mapping[str, Any]] = {}
    by_mem: dict[str, Mapping[str, Any]] = {}
    dia_keys: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        source_key = _required_text(record.get("source_key"), f"source_map[{index}]")
        mem_id = _required_text(record.get("mem_id"), f"source_map[{index}].mem_id")
        sample_id = _required_text(record.get("sample_id"), "source_map.sample_id")
        dia_id = _required_text(record.get("dia_id"), "source_map.dia_id")
        if source_key in by_source:
            raise ValueError(f"duplicate source_map source_key: {source_key!r}")
        if mem_id in by_mem:
            raise ValueError(f"duplicate source_map mem_id: {mem_id!r}")
        if (sample_id, dia_id) in dia_keys:
            raise ValueError(f"duplicate source_map dia_id: {(sample_id, dia_id)!r}")
        dia_keys.add((sample_id, dia_id))
        by_source[source_key] = record
        by_mem[mem_id] = record
    if set(by_source) != expected_keys:
        missing = sorted(expected_keys - set(by_source))[:5]
        extra = sorted(set(by_source) - expected_keys)[:5]
        raise ValueError(f"source_map/manifest mismatch; missing={missing}, extra={extra}")
    return records, by_mem


def _verify_source_map_database(
    manifest: Mapping[str, Any],
    database_path: str | Path,
    source_map_records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind every sidecar mapping back to its exact authoritative DB row."""

    expected_by_source = {
        str(message["source_key"]): message for message in manifest_messages(manifest)
    }
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        for record in source_map_records:
            source_key = str(record["source_key"])
            expected = expected_by_source[source_key]
            mem_id = _required_text(record.get("mem_id"), "source_map.mem_id")
            row = connection.execute(
                """SELECT id, user_id, session_id, role, content, event_ts, sequence
                   FROM raw_messages WHERE id = ?""",
                (_mem_numeric_id(mem_id),),
            ).fetchone()
            if row is None:
                raise ValueError(f"source_map memory row is missing: {mem_id}")
            expected_session_id = (
                f"locomo:{expected['sample_id']}:{expected['session_key']}"
            )
            checks = {
                "user_id": expected["user_id"],
                "session_id": expected_session_id,
                "role": expected["speaker"],
                "content": expected["content"],
                "event_ts": expected["event_ts"],
                "sequence": expected["sequence"],
            }
            for field, expected_value in checks.items():
                if row[field] != expected_value:
                    raise ValueError(
                        f"source_map DB mismatch for {source_key!r}: {field}"
                    )
            record_checks = {
                "sample_id": expected["sample_id"],
                "session_key": expected["session_key"],
                "dia_id": expected["dia_id"],
                "user_id": expected["user_id"],
                "session_id": expected_session_id,
                "sequence": expected["sequence"],
            }
            for field, expected_value in record_checks.items():
                if record.get(field) != expected_value:
                    raise ValueError(
                        f"source_map manifest mismatch for {source_key!r}: {field}"
                    )
    finally:
        connection.close()


def build_prepared_database(
    manifest: Mapping[str, Any],
    extraction_records: Mapping[str, Mapping[str, Any]],
    database_path: str | Path,
    *,
    cache_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay cached extractions through session-bundled production Add."""

    path = Path(database_path)
    sidecar = _prepared_sidecar_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"prepared database or sidecar already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    replay = _SessionReplayModel()
    store = MemoryStore(
        str(path),
        model=replay,  # type: ignore[arg-type]
        structured_query_plan=True,
        evidence_graph=True,
        graph_max_hops=1,
        graph_temporal=False,
    )
    store.initialize()
    source_map: list[dict[str, Any]] = []
    expected_replay_calls = 0

    for sample in _record_sequence(manifest.get("samples"), "manifest.samples"):
        user_id = _required_text(sample.get("user_id"), "sample.user_id")
        sample_id = _required_text(sample.get("sample_id"), "sample.sample_id")
        for session in _record_sequence(sample.get("sessions"), "sample.sessions"):
            messages = _record_sequence(session.get("messages"), "session.messages")
            cached_records: list[Mapping[str, Any]] = []
            for message in messages:
                source_key = str(message["source_key"])
                record = extraction_records.get(source_key)
                if record is None:
                    raise ValueError(f"prepared DB cache missing {source_key!r}")
                _validate_cached_message(message, record)
                cached_records.append(record)
            replay.begin(cached_records)
            request = AddRequest(
                request_id=_required_text(session.get("request_id"), "request_id"),
                user_id=user_id,
                session_id=_required_text(session.get("session_id"), "session_id"),
                messages=[
                    {
                        "role": message["speaker"],
                        "content": message["content"],
                        "timestamp": message["event_ts"],
                    }
                    for message in messages
                ],
            )
            store.add(request)
            replay.finish()
            expected_replay_calls += len(messages)

            with store._connection() as connection:
                rows = connection.execute(
                    """SELECT id, role, content, event_ts, sequence
                       FROM raw_messages
                       WHERE user_id = ? AND session_id = ?
                       ORDER BY sequence, id""",
                    (user_id, request.session_id),
                ).fetchall()
            if len(rows) != len(messages):
                raise RuntimeError(
                    f"prepared session row count mismatch for {request.session_id}"
                )
            for expected, row in zip(messages, rows):
                if int(row["sequence"]) != int(expected["sequence"]):
                    raise RuntimeError("prepared source sequence mismatch")
                if str(row["role"]) != str(expected["speaker"]):
                    raise RuntimeError("prepared source speaker mismatch")
                if str(row["content"]) != str(expected["content"]):
                    raise RuntimeError("prepared source content mismatch")
                if row["event_ts"] != expected["event_ts"]:
                    raise RuntimeError("prepared source timestamp mismatch")
                source_map.append(
                    {
                        "source_key": expected["source_key"],
                        "sample_id": sample_id,
                        "session_key": expected["session_key"],
                        "dia_id": expected["dia_id"],
                        "user_id": user_id,
                        "session_id": request.session_id,
                        "sequence": expected["sequence"],
                        "mem_id": f"mem_{int(row['id'])}",
                    }
                )

    if replay.extract_memory_calls != expected_replay_calls:
        raise RuntimeError(
            f"expected {expected_replay_calls} replay calls, "
            f"observed {replay.extract_memory_calls}"
        )
    _validate_source_map(manifest, source_map)
    support_invariants = _prepared_support_invariants(path)
    mention_invariants = _prepared_mention_invariants(path)
    digest = logical_database_digest(path)
    prepared = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cache_fingerprint": dict(cache_fingerprint),
        "materialization_contract_fingerprint": (
            materialization_contract_fingerprint()
        ),
        "database_digest": digest,
        "message_count": len(source_map),
        "replay_extract_calls": replay.extract_memory_calls,
        "one_replay_call_per_message": replay.extract_memory_calls == len(source_map),
        "source_map": source_map,
        **support_invariants,
        **mention_invariants,
    }
    _atomic_write_json(sidecar, prepared)
    return {
        "database_path": str(path.resolve()),
        "sidecar_path": str(sidecar.resolve()),
        **prepared,
    }


def load_prepared_database(
    manifest: Mapping[str, Any],
    database_path: str | Path,
    *,
    expected_cache_fingerprint: Mapping[str, Any] | None = None,
    allow_materialization_contract_drift: bool = False,
) -> dict[str, Any]:
    path = Path(database_path)
    sidecar = _prepared_sidecar_path(path)
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError("prepared database and sidecar are both required")
    try:
        prepared = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prepared database sidecar: {sidecar}") from error
    if not isinstance(prepared, dict):
        raise ValueError("prepared database sidecar must be an object")
    if prepared.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise ValueError("unsupported prepared database schema")
    if prepared.get("dataset_sha256") != manifest.get("dataset_sha256"):
        raise ValueError("prepared database dataset fingerprint mismatch")
    if prepared.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("prepared database scope manifest mismatch")
    if (
        not allow_materialization_contract_drift
        and prepared.get("materialization_contract_fingerprint")
        != materialization_contract_fingerprint()
    ):
        raise ValueError("prepared database materialization contract mismatch")
    if (
        expected_cache_fingerprint is not None
        and prepared.get("cache_fingerprint") != dict(expected_cache_fingerprint)
    ):
        raise ValueError("prepared database extraction fingerprint mismatch")
    try:
        support_invariants = _prepared_support_invariants(path)
    except RuntimeError as error:
        raise ValueError("prepared database graph support audit mismatch") from error
    for field, expected_value in support_invariants.items():
        if prepared.get(field) != expected_value:
            raise ValueError(
                "prepared database graph support invariant mismatch: {}".format(
                    field
                )
            )
    try:
        mention_invariants = _prepared_mention_invariants(path)
    except RuntimeError as error:
        raise ValueError("prepared database entity-mention audit mismatch") from error
    for field, expected_value in mention_invariants.items():
        if prepared.get(field) != expected_value:
            raise ValueError(
                "prepared database entity-mention invariant mismatch: {}".format(
                    field
                )
            )
    records, _ = _validate_source_map(manifest, prepared.get("source_map"))
    _verify_source_map_database(manifest, path, records)
    digest = logical_database_digest(path)
    if digest != prepared.get("database_digest"):
        raise ValueError("prepared database logical digest mismatch")
    return {
        "database_path": str(path.resolve()),
        "sidecar_path": str(sidecar.resolve()),
        **prepared,
        "source_map": records,
    }


def freeze_structured_plans(
    questions: Sequence[Mapping[str, Any]], model: object
) -> dict[str, dict[str, Any]]:
    if not callable(getattr(model, "plan_query_structured", None)):
        raise TypeError("search model must expose plan_query_structured")
    plans: dict[str, dict[str, Any]] = {}
    for question in questions:
        key = _required_text(question.get("query_key"), "question.query_key")
        if key in plans:
            continue
        raw = model.plan_query_structured(
            _required_text(question.get("question"), "question.question"),
            list(question.get("options", [])),
        )
        if not isinstance(raw, Mapping):
            raise ValueError(f"structured plan for {key} must be a record")
        # JSON round-trip also guarantees the plan can be recorded reproducibly.
        plans[key] = json.loads(_canonical_json(dict(raw)))
    return plans


class _FrozenSearchModel:
    def __init__(
        self,
        plans: Mapping[str, Mapping[str, Any]],
        rank_model: object,
    ) -> None:
        if not callable(getattr(rank_model, "rank_candidates", None)):
            raise TypeError("search model must expose rank_candidates")
        self.plans = plans
        self.rank_model = rank_model
        self.logical_plan_calls = 0
        self.rank_calls = 0

    def plan_query_structured(
        self, query: str, options: Sequence[str]
    ) -> dict[str, Any]:
        self.logical_plan_calls += 1
        key = _query_key(query, options)
        plan = self.plans.get(key)
        if plan is None:
            raise RuntimeError(f"no frozen structured plan for query {key}")
        return copy.deepcopy(dict(plan))

    def rank_candidates(
        self,
        query: str,
        options: Sequence[str],
        candidates: list[dict[str, str]],
    ) -> list[str]:
        self.rank_calls += 1
        result = self.rank_model.rank_candidates(query, list(options), candidates)
        if not isinstance(result, list):
            raise ValueError("rank_candidates must return a list")
        return result


_TRACE_ALIASES: dict[str, tuple[str, ...]] = {
    "requested_seeds": ("requested_seeds",),
    "resolved_seeds": ("resolved_seeds",),
    "unresolved_seeds": ("unresolved_seeds",),
    "p1_channels": ("p1_channels", "p1_channel_ids"),
    "p1_candidate_union_ids": (
        "p1_candidate_union_ids",
        "p1_candidate_ids",
        "p1_union_ids",
    ),
    "p1_pre_rerank_ids": (
        "p1_pre_rerank_ids",
        "p1_counterfactual_pre_rerank_ids",
        "p1_pre_rerank_order_ids",
    ),
    "p1_top30_ids": (
        "p1_top30_ids",
        "p1_counterfactual_top30_ids",
        "p1_rerank_pool_ids",
    ),
    "graph_candidate_ids": ("graph_candidate_ids",),
    "graph_reserved_ids": (
        "graph_reserved_ids",
        "reserved_graph_ids",
    ),
    "rerank_pool_ids": ("rerank_pool_ids",),
    "final_ids": ("final_ids",),
    "graph_paths": ("graph_paths",),
    "edge_diagnostics": ("edge_diagnostics",),
}


def _normalize_storage_trace(
    store: object, *, require_complete: bool
) -> dict[str, Any]:
    raw = getattr(store, "last_retrieval_trace", None)
    normalized: dict[str, Any] = {}
    missing: list[str] = []
    if isinstance(raw, Mapping):
        for canonical, aliases in _TRACE_ALIASES.items():
            found = False
            for alias in aliases:
                if alias in raw:
                    normalized[canonical] = copy.deepcopy(raw[alias])
                    found = True
                    break
            if not found:
                missing.append(canonical)
    else:
        missing = list(_TRACE_ALIASES)

    if missing and require_complete:
        raise RuntimeError(
            "formal paired evaluation requires complete MemoryStore retrieval "
            f"trace fields; missing={missing}"
        )
    if missing:
        normalized = {
            "requested_seeds": [],
            "resolved_seeds": [],
            "unresolved_seeds": [],
            "p1_channels": {},
            "p1_candidate_union_ids": None,
            "p1_pre_rerank_ids": None,
            "p1_top30_ids": None,
            "graph_candidate_ids": list(
                getattr(store, "last_graph_candidate_ids", [])
            ),
            "graph_reserved_ids": None,
            "rerank_pool_ids": None,
            "final_ids": None,
            "graph_paths": copy.deepcopy(
                list(getattr(store, "last_graph_paths", []))
            ),
            "edge_diagnostics": {},
        }
    normalized["complete"] = not missing
    normalized["missing_fields"] = missing
    normalized["query_plan"] = copy.deepcopy(
        dict(getattr(store, "last_query_plan", {}) or {})
    )
    normalized["graph_channel_only_ids"] = list(
        getattr(store, "last_graph_only_candidate_ids", [])
    )
    return normalized


def _mem_numeric_id(mem_id: str) -> int:
    if not mem_id.startswith("mem_") or not mem_id[4:].isdigit():
        raise ValueError(f"invalid memory ID: {mem_id!r}")
    return int(mem_id[4:])


def _audit_graph_trace(
    database_path: str | Path,
    *,
    user_id: str,
    trace: Mapping[str, Any],
    source_by_mem: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit trace paths against evaluator-reloaded DB witness records.

    The trace's witness payload is intentionally treated as a tamperable
    diagnostic.  ``audit_graph_trace_paths`` reloads the support sidecar,
    reconstructs the source proof with the evaluator-owned frozen registry,
    then compares every trace field to that authoritative row.
    """

    violations: list[str] = []
    graph_ids = [str(item) for item in trace.get("graph_candidate_ids", []) or []]
    paths = _record_sequence(trace.get("graph_paths", []), "graph_paths")
    path_audit = audit_graph_trace_paths(
        database_path,
        user_id=user_id,
        paths=paths,
    )
    violations.extend(path_audit["violations"])
    # Persisted-row audit is deliberately separate from trace audit: an edge
    # must remain witnessed even when no particular query traverses it.
    violations.extend(
        f"persisted support: {item}"
        for item in path_audit["persisted_violations"]
    )
    sources_with_paths: set[str] = set()
    verified_graph_candidates = 0
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        for path in paths:
            raw_source_ids = path.get("source_message_ids", [])
            if not isinstance(raw_source_ids, list):
                violations.append("graph path source_message_ids must be a list")
                continue
            for source_id in (str(item) for item in raw_source_ids):
                sources_with_paths.add(source_id)
                mapped = source_by_mem.get(source_id)
                if mapped is None or mapped.get("user_id") != user_id:
                    violations.append(f"cross-user or unknown graph source {source_id}")
        if len(graph_ids) != len(set(graph_ids)):
            violations.append("duplicate graph candidate IDs")
        for graph_id in graph_ids:
            if graph_id not in sources_with_paths:
                violations.append(f"graph candidate has no source path: {graph_id}")
            mapped = source_by_mem.get(graph_id)
            if mapped is None or mapped.get("user_id") != user_id:
                violations.append(f"cross-user or unknown graph candidate {graph_id}")
                continue
            try:
                message_id = _mem_numeric_id(graph_id)
            except ValueError:
                violations.append(f"invalid graph candidate ID {graph_id!r}")
                continue
            raw = connection.execute(
                "SELECT user_id FROM raw_messages WHERE id = ?", (message_id,)
            ).fetchone()
            if raw is None or raw["user_id"] != user_id:
                violations.append(f"graph candidate raw row mismatch: {graph_id}")
            else:
                verified_graph_candidates += 1
    finally:
        connection.close()
    return {
        "traversed_edges": int(path_audit["traversed_edges"]),
        "supported_traversed_edges": int(path_audit["supported_traversed_edges"]),
        "unsupported_traversed_edges": int(path_audit["unsupported_traversed_edges"]),
        "unsupported_edge_ids": list(path_audit["unsupported_edge_ids"]),
        "graph_candidates_verified": verified_graph_candidates,
        "persisted_edge_count": int(path_audit["persisted_edge_count"]),
        "persisted_support_count": int(path_audit["persisted_support_count"]),
        "persisted_all_edges_witnessed": bool(
            path_audit["persisted_all_edges_witnessed"]
        ),
        "violations": violations,
    }


def _audit_arm_integrity(
    database_path: str | Path,
    *,
    user_id: str,
    trace: Mapping[str, Any],
    results: Sequence[object],
    source_by_mem: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit quota-pool identity, storage deduplication, and final content."""

    violations: list[str] = []
    raw_pool = trace.get("rerank_pool_ids")
    if raw_pool is None:
        pool_ids: list[str] | None = None
    elif not isinstance(raw_pool, list):
        pool_ids = None
        violations.append("rerank_pool_ids must be a list")
    else:
        pool_ids = [str(item) for item in raw_pool]

    raw_final_ids = trace.get("final_ids")
    if raw_final_ids is None:
        traced_final_ids: list[str] | None = None
    elif not isinstance(raw_final_ids, list):
        traced_final_ids = None
        violations.append("final_ids must be a list")
    else:
        traced_final_ids = [str(item) for item in raw_final_ids]

    result_ids = [str(getattr(result, "id", "")) for result in results]
    if len(result_ids) != len(set(result_ids)):
        violations.append("duplicate final result IDs")
    if traced_final_ids is not None and traced_final_ids != result_ids:
        violations.append("storage final_ids differ from returned result IDs")
    if pool_ids is not None and len(pool_ids) != len(set(pool_ids)):
        violations.append("rerank_pool_ids are not unique")

    ids_to_load = set(result_ids)
    if pool_ids is not None:
        ids_to_load.update(pool_ids)
    rows_by_mem: dict[str, sqlite3.Row] = {}
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        for mem_id in sorted(ids_to_load):
            mapped = source_by_mem.get(mem_id)
            if mapped is None or mapped.get("user_id") != user_id:
                violations.append(f"cross-user or unknown candidate {mem_id!r}")
                continue
            try:
                message_id = _mem_numeric_id(mem_id)
            except ValueError:
                violations.append(f"invalid candidate ID {mem_id!r}")
                continue
            row = connection.execute(
                "SELECT id, user_id, content FROM raw_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if row is None or row["user_id"] != user_id:
                violations.append(f"candidate raw row mismatch: {mem_id}")
                continue
            rows_by_mem[mem_id] = row
    finally:
        connection.close()

    normalized_pool_content: dict[str, str] = {}
    if pool_ids is not None:
        for mem_id in pool_ids:
            row = rows_by_mem.get(mem_id)
            if row is None:
                continue
            normalized = str(row["content"]).casefold()
            previous = normalized_pool_content.get(normalized)
            if previous is not None and previous != mem_id:
                violations.append(
                    "rerank pool contains casefold-equivalent content: "
                    f"{previous}, {mem_id}"
                )
            else:
                normalized_pool_content[normalized] = mem_id

    final_content_verified = 0
    for result, mem_id in zip(results, result_ids):
        row = rows_by_mem.get(mem_id)
        if row is None:
            continue
        if getattr(result, "content", None) != row["content"]:
            violations.append(f"final result content differs from raw: {mem_id}")
        else:
            final_content_verified += 1

    return {
        "rerank_pool_checked": pool_ids is not None,
        "rerank_pool_ids": len(pool_ids or []),
        "rerank_pool_unique_ids": (
            pool_ids is not None and len(pool_ids) == len(set(pool_ids))
        ),
        "rerank_pool_unique_casefold_content": (
            pool_ids is not None
            and len(normalized_pool_content) == len(pool_ids)
        ),
        "final_results": len(results),
        "final_content_matches_raw": final_content_verified,
        "violations": violations,
    }


def _ranked_dia_ids(
    ranked_mem_ids: Sequence[str],
    *,
    user_id: str,
    source_by_mem: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    result: list[str] = []
    seen_mem: set[str] = set()
    for mem_id in ranked_mem_ids:
        if mem_id in seen_mem:
            raise ValueError(f"duplicate ranked memory ID: {mem_id}")
        seen_mem.add(mem_id)
        mapped = source_by_mem.get(mem_id)
        if mapped is None:
            raise ValueError(f"ranked memory ID is absent from source map: {mem_id}")
        if mapped.get("user_id") != user_id:
            raise ValueError(f"cross-user ranked memory ID: {mem_id}")
        result.append(_required_text(mapped.get("dia_id"), "source_map.dia_id"))
    return result


def score_question_rankings(
    question: Mapping[str, Any],
    ranked_mem_ids: Sequence[str],
    *,
    source_by_mem: Mapping[str, Mapping[str, Any]],
    top_ks: Sequence[int],
) -> dict[str, Any]:
    user_id = _required_text(question.get("user_id"), "question.user_id")
    gold = list(dict.fromkeys(str(item) for item in question["gold_dia_ids"]))
    if not gold:
        raise ValueError("question gold evidence must not be empty")
    ranked_dia = _ranked_dia_ids(
        ranked_mem_ids, user_id=user_id, source_by_mem=source_by_mem
    )
    ranks = [
        next(
            (index + 1 for index, dia_id in enumerate(ranked_dia) if dia_id == item),
            None,
        )
        for item in gold
    ]
    present = [rank for rank in ranks if rank is not None]
    first_rank = min(present) if present else None
    return {
        "ranked_mem_ids": list(ranked_mem_ids),
        "ranked_dia_ids": ranked_dia,
        "first_gold_rank": first_rank,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "hit_at_k": {
            str(k): int(first_rank is not None and first_rank <= k) for k in top_ks
        },
        "evidence_hits_at_k": {
            str(k): sum(rank is not None and rank <= k for rank in ranks)
            for k in top_ks
        },
    }


def _aggregate_arm(
    question_scores: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    question_count = len(question_scores)
    evidence_total = sum(
        len(set(str(item) for item in question["gold_dia_ids"]))
        for question, _ in question_scores
    )
    hit_counts = {
        str(k): sum(int(score["hit_at_k"][str(k)]) for _, score in question_scores)
        for k in top_ks
    }
    evidence_hits = {
        str(k): sum(
            int(score["evidence_hits_at_k"][str(k)])
            for _, score in question_scores
        )
        for k in top_ks
    }
    categories: dict[str, list[Mapping[str, Any]]] = {}
    for question, score in question_scores:
        categories.setdefault(str(question["category"]), []).append(score)
    return {
        "questions": question_count,
        "evidence_items": evidence_total,
        "hit_at_k": {
            str(k): hit_counts[str(k)] / question_count if question_count else 0.0
            for k in top_ks
        },
        "evidence_recall_at_k": {
            str(k): evidence_hits[str(k)] / evidence_total if evidence_total else 0.0
            for k in top_ks
        },
        "mrr": (
            sum(float(score["reciprocal_rank"]) for _, score in question_scores)
            / question_count
            if question_count
            else 0.0
        ),
        "category_hit_at_k": {
            category: {
                str(k): sum(int(item["hit_at_k"][str(k)]) for item in scores)
                / len(scores)
                for k in top_ks
            }
            for category, scores in sorted(categories.items())
        },
        "raw_counts": {
            "hit_counts": hit_counts,
            "evidence_hit_counts": evidence_hits,
            "reciprocal_rank_sum": sum(
                float(score["reciprocal_rank"]) for _, score in question_scores
            ),
        },
    }


def run_paired_evaluation(
    manifest: Mapping[str, Any],
    database_path: str | Path,
    source_map_records: Sequence[Mapping[str, Any]],
    *,
    frozen_plans: Mapping[str, Mapping[str, Any]],
    rank_model: object,
    top_ks: Sequence[int] = (1, 3, 10),
    graph_rrf_weight: float = 0.025,
    graph_rerank_quota: int = 4,
    graph_max_candidates: int = 20,
    require_complete_trace: bool = True,
    store_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Search one prepared DB with graph off/on and score by authoritative IDs."""

    unique_top_ks = tuple(sorted(set(int(value) for value in top_ks)))
    if not unique_top_ks or unique_top_ks[0] < 1 or unique_top_ks[-1] > 100:
        raise ValueError("top_ks must be between 1 and 100")
    options = dict(store_options or {})
    if options.get("dense_fusion_alpha") is not None:
        raise ValueError("P3-A paired evaluation rejects dense score fusion")
    forbidden_options = {
        "database_path",
        "model",
        "structured_query_plan",
        "evidence_graph",
        "graph_max_hops",
        "graph_temporal",
        "graph_rrf_weight",
        "graph_max_candidates",
        "graph_rerank_quota",
    }
    overlap = forbidden_options & set(options)
    if overlap:
        raise ValueError(f"paired runner owns store options: {sorted(overlap)}")

    validated_source_map, source_by_mem = _validate_source_map(
        manifest, source_map_records
    )
    _verify_source_map_database(manifest, database_path, validated_source_map)
    questions = _record_sequence(manifest.get("questions"), "manifest.questions")
    expected_plan_keys = {str(question["query_key"]) for question in questions}
    if not expected_plan_keys <= set(frozen_plans):
        raise ValueError("frozen plan set is missing selected questions")

    database_path = str(Path(database_path).resolve())
    digest_before = logical_database_digest(database_path)
    # Audit the whole prepared graph before creating a MemoryStore or making a
    # Search call.  A later trace can diagnose traversal, but it must never
    # turn an already-invalid persisted edge into an eligible experiment.
    prepared_support_audit = audit_persisted_graph_support(database_path)
    if (
        prepared_support_audit["violations"]
        or not prepared_support_audit["all_edges_witnessed"]
    ):
        raise RuntimeError(
            "formal prepared support audit failed: {}".format(
                list(prepared_support_audit["violations"])[:5]
            )
        )
    baseline_model = _FrozenSearchModel(frozen_plans, rank_model)
    graph_model = _FrozenSearchModel(frozen_plans, rank_model)
    baseline_store = MemoryStore(
        database_path,
        model=baseline_model,  # type: ignore[arg-type]
        structured_query_plan=True,
        evidence_graph=False,
        graph_max_hops=1,
        graph_temporal=False,
        graph_rrf_weight=graph_rrf_weight,
        graph_max_candidates=graph_max_candidates,
        graph_rerank_quota=graph_rerank_quota,
        **options,
    )
    graph_store = MemoryStore(
        database_path,
        model=graph_model,  # type: ignore[arg-type]
        structured_query_plan=True,
        evidence_graph=True,
        graph_max_hops=1,
        graph_temporal=False,
        graph_rrf_weight=graph_rrf_weight,
        graph_max_candidates=graph_max_candidates,
        graph_rerank_quota=graph_rerank_quota,
        **options,
    )

    baseline_scores: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    graph_scores: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    traces: list[dict[str, Any]] = []
    baseline_latencies: list[float] = []
    graph_latencies: list[float] = []
    recovered_counts = {str(k): 0 for k in unique_top_ks}
    lost_counts = {str(k): 0 for k in unique_top_ks}
    recovered_cases = {str(k): 0 for k in unique_top_ks}
    lost_cases = {str(k): 0 for k in unique_top_ks}
    traversed_edges_total = 0
    unsupported_traversed_edges_total = 0
    rerank_pool_candidates_verified = 0
    final_results_content_verified = 0

    for question in questions:
        query = _required_text(question.get("question"), "question.question")
        query_options = list(question.get("options", []))
        user_id = _required_text(question.get("user_id"), "question.user_id")
        started = time.perf_counter()
        baseline_results = baseline_store.search(
            user_id=user_id,
            query=query,
            options=query_options,
            top_k=max(unique_top_ks),
        )
        baseline_latencies.append(time.perf_counter() - started)
        baseline_trace = _normalize_storage_trace(
            baseline_store, require_complete=require_complete_trace
        )

        started = time.perf_counter()
        graph_results = graph_store.search(
            user_id=user_id,
            query=query,
            options=query_options,
            top_k=max(unique_top_ks),
        )
        graph_latencies.append(time.perf_counter() - started)
        graph_trace = _normalize_storage_trace(
            graph_store, require_complete=require_complete_trace
        )

        expected_plan = frozen_plans[str(question["query_key"])]
        if baseline_trace["query_plan"] != expected_plan:
            raise RuntimeError("baseline did not use the frozen structured plan")
        if graph_trace["query_plan"] != expected_plan:
            raise RuntimeError("graph arm did not use the frozen structured plan")
        if baseline_trace["complete"] and graph_trace["complete"]:
            for field in (
                "p1_channels",
                "p1_candidate_union_ids",
                "p1_pre_rerank_ids",
                "p1_top30_ids",
            ):
                if baseline_trace[field] != graph_trace[field]:
                    raise RuntimeError(
                        f"paired P1 pre-rerank trace differs between arms: {field}"
                    )

        baseline_ids = [str(result.id) for result in baseline_results]
        graph_ids = [str(result.id) for result in graph_results]
        baseline_score = score_question_rankings(
            question,
            baseline_ids,
            source_by_mem=source_by_mem,
            top_ks=unique_top_ks,
        )
        graph_score = score_question_rankings(
            question,
            graph_ids,
            source_by_mem=source_by_mem,
            top_ks=unique_top_ks,
        )
        baseline_scores.append((question, baseline_score))
        graph_scores.append((question, graph_score))

        baseline_integrity = _audit_arm_integrity(
            database_path,
            user_id=user_id,
            trace=baseline_trace,
            results=baseline_results,
            source_by_mem=source_by_mem,
        )
        graph_integrity = _audit_arm_integrity(
            database_path,
            user_id=user_id,
            trace=graph_trace,
            results=graph_results,
            source_by_mem=source_by_mem,
        )
        graph_audit = _audit_graph_trace(
            database_path,
            user_id=user_id,
            trace=graph_trace,
            source_by_mem=source_by_mem,
        )
        violations = [
            *(f"baseline: {item}" for item in baseline_integrity["violations"]),
            *(f"graph_on: {item}" for item in graph_integrity["violations"]),
            *(f"graph_trace: {item}" for item in graph_audit["violations"]),
            *(
                f"prepared_support: {item}"
                for item in prepared_support_audit["violations"]
            ),
        ]
        if violations:
            raise RuntimeError(f"formal paired integrity audit failed: {violations[:5]}")
        traversed_edges_total += int(graph_audit["traversed_edges"])
        unsupported_traversed_edges_total += int(
            graph_audit["unsupported_traversed_edges"]
        )
        rerank_pool_candidates_verified += int(
            baseline_integrity["rerank_pool_ids"]
        ) + int(graph_integrity["rerank_pool_ids"])
        final_results_content_verified += int(
            baseline_integrity["final_content_matches_raw"]
        ) + int(graph_integrity["final_content_matches_raw"])

        gold = set(str(item) for item in question["gold_dia_ids"])
        recovered_at_k: dict[str, list[str]] = {}
        lost_at_k: dict[str, list[str]] = {}
        for k in unique_top_ks:
            baseline_top = set(baseline_score["ranked_dia_ids"][:k])
            graph_top = set(graph_score["ranked_dia_ids"][:k])
            recovered = sorted((gold & graph_top) - baseline_top)
            lost = sorted((gold & baseline_top) - graph_top)
            recovered_at_k[str(k)] = recovered
            lost_at_k[str(k)] = lost
            recovered_counts[str(k)] += len(recovered)
            lost_counts[str(k)] += len(lost)
            recovered_cases[str(k)] += int(bool(recovered))
            lost_cases[str(k)] += int(bool(lost))

        p1_top30 = graph_trace.get("p1_top30_ids")
        rerank_pool = graph_trace.get("rerank_pool_ids")
        graph_candidates = set(graph_trace.get("graph_candidate_ids") or [])
        graph_quota_promoted = None
        graph_quota_displaced = None
        if isinstance(p1_top30, list) and isinstance(rerank_pool, list):
            graph_quota_promoted = sorted(
                (set(rerank_pool) & graph_candidates) - set(p1_top30)
            )
            graph_quota_displaced = sorted(set(p1_top30) - set(rerank_pool))
            if len(rerank_pool) > MemoryStore.MODEL_RERANK_LIMIT:
                raise RuntimeError("rerank pool exceeds the 30-item model limit")
            reserved = graph_trace.get("graph_reserved_ids")
            if isinstance(reserved, list) and len(reserved) > graph_rerank_quota:
                raise RuntimeError("graph reserved IDs exceed configured quota")

        traces.append(
            {
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "case_id": question["case_id"],
                "sample_id": question["sample_id"],
                "category": question["category"],
                "gold_dia_ids": list(question["gold_dia_ids"]),
                "frozen_plan_sha256": _sha256_json(expected_plan),
                "baseline": {**baseline_score, "storage": baseline_trace},
                "graph_on": {**graph_score, "storage": graph_trace},
                "paired_recovered_gold_at_k": recovered_at_k,
                "paired_lost_gold_at_k": lost_at_k,
                "graph_quota_promoted_mem_ids": graph_quota_promoted,
                "graph_quota_displaced_p1_mem_ids": graph_quota_displaced,
                "formal_integrity_audit": {
                    "baseline": baseline_integrity,
                    "graph_on": graph_integrity,
                    "graph_provenance": graph_audit,
                },
            }
        )

    digest_after = logical_database_digest(database_path)
    if digest_after != digest_before:
        raise RuntimeError("paired Search mutated the prepared database")
    if baseline_model.logical_plan_calls != len(questions):
        raise RuntimeError("baseline logical planner call count mismatch")
    if graph_model.logical_plan_calls != len(questions):
        raise RuntimeError("graph logical planner call count mismatch")
    if baseline_model.rank_calls != len(questions):
        raise RuntimeError("baseline ranker call count mismatch")
    if graph_model.rank_calls != len(questions):
        raise RuntimeError("graph ranker call count mismatch")

    baseline_report = _aggregate_arm(baseline_scores, top_ks=unique_top_ks)
    graph_report = _aggregate_arm(graph_scores, top_ks=unique_top_ks)
    baseline_report["search_latency_seconds_total"] = sum(baseline_latencies)
    baseline_report["search_latency_seconds_mean"] = statistics.fmean(
        baseline_latencies
    )
    graph_report["search_latency_seconds_total"] = sum(graph_latencies)
    graph_report["search_latency_seconds_mean"] = statistics.fmean(graph_latencies)
    return {
        "scope": "paired shared-Add LoCoMo retrieval; exact dia_id scoring",
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "question_offset": manifest["question_offset"],
        "max_questions": manifest["max_questions"],
        "questions": len(questions),
        "messages": manifest["message_count"],
        "top_ks": list(unique_top_ks),
        "graph_configuration": {
            "max_hops": 1,
            "temporal": False,
            "rrf_weight": graph_rrf_weight,
            "max_candidates": graph_max_candidates,
            "rerank_quota": graph_rerank_quota,
            "rerank_limit": MemoryStore.MODEL_RERANK_LIMIT,
        },
        "storage_trace_complete": all(
            trace["baseline"]["storage"]["complete"]
            and trace["graph_on"]["storage"]["complete"]
            for trace in traces
        ),
        "database_digest_before": digest_before,
        "database_digest_after": digest_after,
        "database_unchanged": digest_before == digest_after,
        "formal_integrity_audit": {
            "unsupported_traversed_edges": unsupported_traversed_edges_total,
            "traversed_edges": traversed_edges_total,
            "persisted_graph_edges": int(prepared_support_audit["edge_count"]),
            "persisted_graph_edge_supports": int(
                prepared_support_audit["support_count"]
            ),
            "all_persisted_graph_edges_witnessed": bool(
                prepared_support_audit["all_edges_witnessed"]
            ),
            "support_schema_version": SUPPORT_SCHEMA_VERSION,
            "support_normalization_id": SUPPORT_NORMALIZATION_ID,
            "support_spec_registry_fingerprint": FROZEN_SPEC_REGISTRY_FINGERPRINT,
            "rerank_pool_candidates_verified": rerank_pool_candidates_verified,
            "final_results_content_verified": final_results_content_verified,
            "violations": 0,
        },
        "baseline": baseline_report,
        "graph_on": graph_report,
        "delta_hit_at_k": {
            str(k): graph_report["hit_at_k"][str(k)]
            - baseline_report["hit_at_k"][str(k)]
            for k in unique_top_ks
        },
        "delta_evidence_recall_at_k": {
            str(k): graph_report["evidence_recall_at_k"][str(k)]
            - baseline_report["evidence_recall_at_k"][str(k)]
            for k in unique_top_ks
        },
        "delta_mrr": graph_report["mrr"] - baseline_report["mrr"],
        "paired_recovered_gold_evidence_count_at_k": recovered_counts,
        "paired_lost_gold_evidence_count_at_k": lost_counts,
        "paired_recovered_case_count_at_k": recovered_cases,
        "paired_lost_case_count_at_k": lost_cases,
        "search_calls": {
            "frozen_plan_external_calls_during_arms": 0,
            "baseline_logical_plan_calls": baseline_model.logical_plan_calls,
            "graph_logical_plan_calls": graph_model.logical_plan_calls,
            "baseline_rank_calls": baseline_model.rank_calls,
            "graph_rank_calls": graph_model.rank_calls,
            "additional_graph_search_calls": (
                graph_model.logical_plan_calls
                + graph_model.rank_calls
                - baseline_model.logical_plan_calls
                - baseline_model.rank_calls
            ),
        },
        "question_traces": traces,
    }


def run_paired_evaluation_with_gate(
    manifest: Mapping[str, Any],
    database_path: str | Path,
    source_map_records: Sequence[Mapping[str, Any]],
    *,
    frozen_plans: Mapping[str, Mapping[str, Any]],
    rank_model: object,
    top_ks: Sequence[int] = (1, 3, 10),
    graph_rrf_weight: float = 0.025,
    graph_rerank_quota: int = 4,
    graph_max_candidates: int = 20,
    require_complete_trace: bool = True,
    store_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the outcome-independent subset, then Search and evaluate P3-A."""

    if not {1, 10} <= {int(value) for value in top_ks}:
        raise ValueError("P3-A gate evaluation requires top_ks to include 1 and 10")
    validate_p3a_formal_configuration(
        graph_rrf_weight=graph_rrf_weight,
        graph_rerank_quota=graph_rerank_quota,
        graph_max_candidates=graph_max_candidates,
    )
    validated_source_map, _ = _validate_source_map(manifest, source_map_records)
    _verify_source_map_database(manifest, database_path, validated_source_map)
    digest_before_subset = logical_database_digest(database_path)
    relation_subset_manifest = build_relation_subset_manifest(
        database_path,
        manifest,
        validated_source_map,
        frozen_plans,
    )
    digest_after_subset = logical_database_digest(database_path)
    if digest_after_subset != digest_before_subset:
        raise RuntimeError("relation-subset freeze mutated the prepared database")

    # This call contains the first and only Search operations in this function.
    paired = run_paired_evaluation(
        manifest,
        database_path,
        validated_source_map,
        frozen_plans=frozen_plans,
        rank_model=rank_model,
        top_ks=top_ks,
        graph_rrf_weight=graph_rrf_weight,
        graph_rerank_quota=graph_rerank_quota,
        graph_max_candidates=graph_max_candidates,
        require_complete_trace=require_complete_trace,
        store_options=store_options,
    )
    gate = evaluate_p3a_gate(
        paired,
        relation_subset_manifest,
        formal_question_count=P3A_FORMAL_QUESTION_COUNT,
    )
    return {
        "relation_subset_manifest": relation_subset_manifest,
        "paired": paired,
        "p3a_gate": gate,
    }


def _parse_top_ks(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(item) for item in value.split(","))))
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-k must be comma-separated integers") from error
    if not values or values[0] < 1 or values[-1] > 100:
        raise argparse.ArgumentTypeError("top-k values must be between 1 and 100")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fair graph-off/on LoCoMo retrieval over one shared Add DB."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--extraction-cache", required=True)
    parser.add_argument("--prepared-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument(
        "--cache-mode", choices=["build", "reuse", "require", "extend"],
        default="reuse",
    )
    parser.add_argument(
        "--prepared-mode", choices=["build", "reuse"], default="build"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ingest-model", default=MODEL_NAME)
    parser.add_argument("--ingest-base-url")
    parser.add_argument(
        "--ingest-artifact-fingerprint",
        required=True,
        help=(
            "SHA-256 of the exact local ingest model artifact (for example the "
            "GGUF file); cache build/reuse/extend is bound to this value."
        ),
    )
    parser.add_argument("--search-model", default=MODEL_NAME)
    parser.add_argument("--search-base-url")
    parser.add_argument("--top-k", type=_parse_top_ks, default=(1, 3, 10))
    parser.add_argument("--graph-rrf-weight", type=float, default=0.025)
    parser.add_argument("--graph-rerank-quota", type=int, default=4)
    parser.add_argument("--graph-max-candidates", type=int, default=20)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build/reuse extraction cache and prepared DB, then stop before Search.",
    )
    parser.add_argument(
        "--allow-incomplete-storage-trace",
        action="store_true",
        help="Diagnostic only; formal paired reports require complete pre-rerank trace.",
    )
    return parser.parse_args()


def _require_loopback_base_url(base_url: object, label: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise SystemExit(
            f"{label} requires an explicit loopback base URL; "
            "external OpenAI evaluation is disabled"
        )
    normalized = base_url.strip().rstrip("/")
    if not normalized.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise SystemExit(
            f"{label} must use http://127.0.0.1:<port> or "
            "http://localhost:<port>"
        )
    return normalized


def _runtime_model(
    *, model_name: str, base_url: str | None, purpose: str
) -> MemoryModel:
    local_base_url = _require_loopback_base_url(base_url, f"{purpose} model")
    return MemoryModel(
        "local-only",
        model_name=model_name,
        base_url=local_base_url,
        disable_thinking=True,
    )


def _model_diagnostics(
    model: object | None, *, model_name: str, base_url: str | None
) -> dict[str, Any]:
    raw_finish_reasons = getattr(model, "finish_reason_counts", {})
    finish_reasons = (
        {str(key): int(value) for key, value in raw_finish_reasons.items()}
        if isinstance(raw_finish_reasons, Mapping)
        else {}
    )
    return {
        "name": model_name,
        "base_url": base_url,
        "constructed": model is not None,
        "call_count": int(getattr(model, "call_count", 0) or 0),
        "finish_reason_counts": finish_reasons,
        "truncated_calls": int(getattr(model, "truncated_calls", 0) or 0),
    }


def main() -> None:
    args = parse_args()
    try:
        validate_p3a_formal_configuration(
            graph_rrf_weight=args.graph_rrf_weight,
            graph_rerank_quota=args.graph_rerank_quota,
            graph_max_candidates=args.graph_max_candidates,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")
    args.ingest_base_url = _require_loopback_base_url(
        args.ingest_base_url, "--ingest-base-url"
    )
    if not args.prepare_only:
        args.search_base_url = _require_loopback_base_url(
            args.search_base_url, "--search-base-url"
        )
    manifest = load_dataset_manifest(
        args.dataset,
        max_questions=args.max_questions,
        question_offset=args.question_offset,
    )
    model_fingerprint = _sha256_json(
        {
            "model": args.ingest_model,
            "base_url": args.ingest_base_url,
            "disable_thinking": bool(args.ingest_base_url),
        }
    )
    fingerprint = extraction_cache_fingerprint(
        manifest,
        model_fingerprint=model_fingerprint,
        ingest_artifact_fingerprint=args.ingest_artifact_fingerprint,
        contract_fingerprint=extraction_contract_fingerprint(),
    )
    extraction_model: object | None = None
    if args.cache_mode in {"build", "extend"}:
        extraction_model = _runtime_model(
            model_name=args.ingest_model,
            base_url=args.ingest_base_url,
            purpose="ingest",
        )
    cache = ensure_extraction_cache(
        manifest,
        args.extraction_cache,
        fingerprint=fingerprint,
        model=extraction_model,
        mode=args.cache_mode,
        workers=args.workers,
    )
    if args.prepared_mode == "build":
        prepared = build_prepared_database(
            manifest,
            cache["records_by_source_key"],
            args.prepared_db,
            cache_fingerprint=fingerprint,
        )
    else:
        prepared = load_prepared_database(
            manifest,
            args.prepared_db,
            expected_cache_fingerprint=fingerprint,
        )

    common_report = {
        "dataset": {
            "path": manifest["dataset_path"],
            "sha256": manifest["dataset_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "question_offset": manifest["question_offset"],
            "max_questions": manifest["max_questions"],
            "messages": manifest["message_count"],
            "questions": manifest["question_count"],
        },
        "extraction_cache": {
            key: value
            for key, value in cache.items()
            if key != "records_by_source_key"
        },
        "prepared_database": {
            key: value for key, value in prepared.items() if key != "source_map"
        },
        "ingest_artifact_fingerprint": fingerprint[
            "ingest_artifact_fingerprint"
        ],
        "extraction_model": _model_diagnostics(
            extraction_model,
            model_name=args.ingest_model,
            base_url=args.ingest_base_url,
        ),
    }
    if args.prepare_only:
        report = {"stage": "prepare_only", **common_report}
        _atomic_write_json(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    search_model = _runtime_model(
        model_name=args.search_model,
        base_url=args.search_base_url,
        purpose="search",
    )
    questions = _record_sequence(manifest["questions"], "manifest.questions")
    calls_before_plans = search_model.call_count
    truncated_before_plans = search_model.truncated_calls
    plans = freeze_structured_plans(questions, search_model)
    plan_external_calls = search_model.call_count - calls_before_plans
    expected_plan_external_calls = len(questions)
    if plan_external_calls != expected_plan_external_calls:
        raise RuntimeError("frozen plan external call count mismatch")
    plan_truncated_calls = search_model.truncated_calls - truncated_before_plans
    if plan_truncated_calls != 0:
        raise RuntimeError(
            "structured planning observed truncated model responses: "
            f"{plan_truncated_calls}"
        )
    calls_before_arms = search_model.call_count
    truncated_before_arms = search_model.truncated_calls
    integrated = run_paired_evaluation_with_gate(
        manifest,
        args.prepared_db,
        prepared["source_map"],
        frozen_plans=plans,
        rank_model=search_model,
        top_ks=args.top_k,
        graph_rrf_weight=args.graph_rrf_weight,
        graph_rerank_quota=args.graph_rerank_quota,
        graph_max_candidates=args.graph_max_candidates,
        require_complete_trace=not args.allow_incomplete_storage_trace,
    )
    paired_arm_external_calls = search_model.call_count - calls_before_arms
    expected_paired_arm_external_calls = 2 * len(questions)
    if paired_arm_external_calls != expected_paired_arm_external_calls:
        raise RuntimeError(
            "paired Search external call count mismatch: expected "
            f"{expected_paired_arm_external_calls}, observed "
            f"{paired_arm_external_calls}"
        )
    paired_arm_truncated_calls = (
        search_model.truncated_calls - truncated_before_arms
    )
    if paired_arm_truncated_calls != 0:
        raise RuntimeError(
            "paired Search observed truncated model responses: "
            f"{paired_arm_truncated_calls}"
        )
    total_external_calls = search_model.call_count - calls_before_plans
    expected_total_external_calls = (
        expected_plan_external_calls + expected_paired_arm_external_calls
    )
    if total_external_calls != expected_total_external_calls:
        raise RuntimeError("total search-model external call count mismatch")
    total_truncated_calls = search_model.truncated_calls - truncated_before_plans
    if total_truncated_calls != 0:
        raise RuntimeError(
            "search evaluation observed truncated model responses: "
            f"{total_truncated_calls}"
        )
    paired = integrated["paired"]
    report = {
        "stage": "paired_evaluation",
        **common_report,
        "frozen_plan_external_calls": plan_external_calls,
        "search_model_call_audit": {
            "question_count": len(questions),
            "plan_external_calls": plan_external_calls,
            "expected_plan_external_calls": expected_plan_external_calls,
            "plan_truncated_calls": plan_truncated_calls,
            "paired_arm_external_calls": paired_arm_external_calls,
            "expected_paired_arm_external_calls": (
                expected_paired_arm_external_calls
            ),
            "total_external_calls": total_external_calls,
            "expected_total_external_calls": expected_total_external_calls,
            "paired_arm_truncated_calls": paired_arm_truncated_calls,
            "total_truncated_calls": total_truncated_calls,
        },
        "search_model": _model_diagnostics(
            search_model,
            model_name=args.search_model,
            base_url=args.search_base_url,
        ),
        "relation_subset_manifest": integrated["relation_subset_manifest"],
        "paired": paired,
        "p3a_gate": integrated["p3a_gate"],
    }
    _atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
