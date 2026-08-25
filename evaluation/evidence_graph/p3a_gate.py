"""Predeclared, count-based P3-A subset and promotion gates.

This module deliberately consumes no answer text and never defines a subset
from graph-on retrieval outcomes.  Relation membership is frozen from the
query plan, the prepared graph, and authoritative gold source IDs before any
paired Search arm is run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from app.evidence_graph import normalize_entity_name
from app.graph_routing import preferred_graph_predicates


RELATION_SUBSET_SCHEMA_VERSION = 1
P3A_GATE_SCHEMA_VERSION = 1
P3A_FORMAL_DATASET_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
P3A_FORMAL_QUESTION_OFFSET = 0
P3A_FORMAL_QUESTION_COUNT = 200
P3A_FORMAL_GRAPH_RRF_WEIGHT = 0.025
P3A_FORMAL_GRAPH_RERANK_QUOTA = 4
P3A_FORMAL_GRAPH_MAX_CANDIDATES = 20
P3A_FORMAL_GRAPH_SEED_LIMIT = 6
P3A_FORMAL_GRAPH_EDGE_LIMIT_PER_SEED = 20
RELATION_SUBSET_POLICY = "plan_entity_and_controlled_predicate_v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def validate_p3a_formal_configuration(
    *,
    graph_rrf_weight: object,
    graph_rerank_quota: object,
    graph_max_candidates: object,
) -> None:
    """Reject a P3-A run whose retrieval mechanics differ from the protocol."""

    try:
        rrf_weight = float(graph_rrf_weight)
    except (TypeError, ValueError) as error:
        raise ValueError("P3-A graph RRF weight must be numeric") from error
    if isinstance(graph_rerank_quota, bool) or not isinstance(
        graph_rerank_quota, int
    ):
        raise ValueError("P3-A graph rerank quota must be an integer")
    if isinstance(graph_max_candidates, bool) or not isinstance(
        graph_max_candidates, int
    ):
        raise ValueError("P3-A graph maximum candidates must be an integer")
    expected = {
        "rrf_weight": P3A_FORMAL_GRAPH_RRF_WEIGHT,
        "rerank_quota": P3A_FORMAL_GRAPH_RERANK_QUOTA,
        "max_candidates": P3A_FORMAL_GRAPH_MAX_CANDIDATES,
    }
    actual = {
        "rrf_weight": rrf_weight,
        "rerank_quota": graph_rerank_quota,
        "max_candidates": graph_max_candidates,
    }
    if actual != expected:
        raise ValueError(
            "formal P3-A graph configuration is fixed at "
            f"{expected}; received {actual}"
        )


def relation_subset_manifest_sha256(value: Mapping[str, Any]) -> str:
    """Hash a subset manifest without trusting its embedded digest."""

    return _sha256({
        key: item for key, item in value.items()
        if key != "subset_manifest_sha256"
    })


def build_relation_subset_manifest(
    database_path: str | Path,
    manifest: Mapping[str, Any],
    source_map_records: Sequence[Mapping[str, Any]],
    frozen_plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze an outcome-independent direct-relation question subset."""

    source_by_dia: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in source_map_records:
        key = (str(record.get("sample_id", "")), str(record.get("dia_id", "")))
        if not all(key) or key in source_by_dia:
            raise ValueError("source map contains a missing or duplicate dia identity")
        source_by_dia[key] = record

    questions = manifest.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("manifest.questions must be a non-empty list")
    records: list[dict[str, Any]] = []
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        for question in questions:
            if not isinstance(question, Mapping):
                raise ValueError("manifest question must be a record")
            case_id = str(question.get("case_id", ""))
            sample_id = str(question.get("sample_id", ""))
            user_id = str(question.get("user_id", ""))
            query_key = str(question.get("query_key", ""))
            if not all((case_id, sample_id, user_id, query_key)):
                raise ValueError("manifest question identity is incomplete")
            plan = frozen_plans.get(query_key)
            if not isinstance(plan, Mapping):
                raise ValueError(f"missing frozen plan for {case_id}")
            raw_entities = plan.get("entities", [])
            if not isinstance(raw_entities, list):
                raw_entities = []

            requested: list[dict[str, Any]] = []
            resolved_entity_ids: list[int] = []
            seen_normalized: set[str] = set()
            valid_plan_seed_count = 0
            for raw_seed in raw_entities[:6]:
                if not isinstance(raw_seed, str):
                    continue
                normalized = normalize_entity_name(raw_seed)
                if normalized is None:
                    requested.append({
                        "seed": raw_seed, "normalized": None,
                        "status": "invalid", "candidate_entity_ids": [],
                    })
                    continue
                if normalized in seen_normalized:
                    requested.append({
                        "seed": raw_seed, "normalized": normalized,
                        "status": "duplicate", "candidate_entity_ids": [],
                    })
                    continue
                seen_normalized.add(normalized)
                valid_plan_seed_count += 1
                rows = connection.execute(
                    """SELECT DISTINCT entity_id FROM (
                           SELECT id AS entity_id
                           FROM graph_entities
                           WHERE user_id = ? AND canonical_name = ?
                           UNION ALL
                           SELECT entity_id
                           FROM graph_aliases
                           WHERE user_id = ? AND normalized_alias = ?
                       ) ORDER BY entity_id""",
                    (user_id, normalized, user_id, normalized),
                ).fetchall()
                candidates = sorted({int(row["entity_id"]) for row in rows})
                status = (
                    "resolved" if len(candidates) == 1
                    else "not_found" if not candidates
                    else "ambiguous"
                )
                requested.append({
                    "seed": raw_seed,
                    "normalized": normalized,
                    "status": status,
                    "candidate_entity_ids": [str(item) for item in candidates],
                })
                if len(candidates) == 1:
                    resolved_entity_ids.append(candidates[0])

            gold_mem_ids: list[str] = []
            for dia_id in question.get("gold_dia_ids", []):
                mapped = source_by_dia.get((sample_id, str(dia_id)))
                if mapped is None or str(mapped.get("user_id", "")) != user_id:
                    raise ValueError(f"gold source mapping mismatch for {case_id}")
                mem_id = str(mapped.get("mem_id", ""))
                if not mem_id.startswith("mem_") or not mem_id[4:].isdigit():
                    raise ValueError(f"invalid gold memory ID for {case_id}")
                if mem_id not in gold_mem_ids:
                    gold_mem_ids.append(mem_id)
            gold_numeric = {int(item[4:]) for item in gold_mem_ids}

            adjacent_edges: list[dict[str, Any]] = []
            seen_edge_ids: set[int] = set()
            for entity_id in resolved_entity_ids:
                rows = connection.execute(
                    """SELECT edge.id, edge.source_message_id, edge.predicate
                       FROM graph_edges AS edge
                       JOIN raw_messages AS raw
                         ON raw.id = edge.source_message_id
                        AND raw.user_id = edge.user_id
                       WHERE edge.user_id = ? AND raw.user_id = ?
                         AND (edge.subject_entity_id = ? OR edge.object_entity_id = ?)
                       ORDER BY edge.id""",
                    (user_id, user_id, entity_id, entity_id),
                ).fetchall()
                for row in rows:
                    edge_id = int(row["id"])
                    if edge_id in seen_edge_ids:
                        continue
                    seen_edge_ids.add(edge_id)
                    adjacent_edges.append({
                        "edge_id": str(edge_id),
                        "source_mem_id": f"mem_{int(row['source_message_id'])}",
                        "predicate": str(row["predicate"]),
                    })
            supporting = [
                edge for edge in adjacent_edges
                if int(edge["source_mem_id"][4:]) in gold_numeric
            ]
            planned_predicates = list(preferred_graph_predicates(
                str(question.get("question", "")), plan
            ))
            records.append({
                "case_id": case_id,
                "sample_id": sample_id,
                "user_id": user_id,
                "query_key": query_key,
                "plan_sha256": _sha256(dict(plan)),
                "requested_seeds": requested,
                "resolved_entity_ids": [
                    str(item) for item in sorted(set(resolved_entity_ids))
                ],
                "gold_mem_ids": gold_mem_ids,
                "planned_predicates": planned_predicates,
                "supporting_edges": supporting,
                # Membership is intentionally independent of graph extraction,
                # graph resolution, candidate movement, and gold adjacency.
                # Those remain diagnostics, never a post-treatment selector.
                "relation_subset": bool(
                    valid_plan_seed_count and planned_predicates
                ),
            })
    finally:
        connection.close()

    payload: dict[str, Any] = {
        "schema_version": RELATION_SUBSET_SCHEMA_VERSION,
        "source_manifest_sha256": str(manifest.get("manifest_sha256", "")),
        "policy": RELATION_SUBSET_POLICY,
        "questions": len(records),
        "relation_questions": sum(
            int(record["relation_subset"]) for record in records
        ),
        "records": records,
    }
    payload["subset_manifest_sha256"] = relation_subset_manifest_sha256(payload)
    return payload


def _formal_graph_configuration_violations(
    paired_report: Mapping[str, Any],
) -> list[str]:
    configuration = paired_report.get("graph_configuration")
    if not isinstance(configuration, Mapping):
        return ["graph_configuration is missing or invalid"]
    expected = {
        "max_hops": 1,
        "temporal": False,
        "rrf_weight": P3A_FORMAL_GRAPH_RRF_WEIGHT,
        "max_candidates": P3A_FORMAL_GRAPH_MAX_CANDIDATES,
        "rerank_quota": P3A_FORMAL_GRAPH_RERANK_QUOTA,
        "rerank_limit": 30,
    }
    violations: list[str] = []
    for key, required in expected.items():
        actual = configuration.get(key)
        if key == "rrf_weight":
            valid = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and float(actual) == required
            )
        elif key in {"max_hops", "max_candidates", "rerank_quota", "rerank_limit"}:
            valid = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual == required
            )
        else:
            valid = actual is required
        if not valid:
            violations.append(
                f"graph_configuration.{key}={actual!r}; expected {required!r}"
            )
    return violations


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _graph_trace_mechanics_violations(
    *, case_id: str, storage: Mapping[str, Any]
) -> list[str]:
    """Validate cap and round-robin diagnostics emitted by ``MemoryStore``."""

    violations: list[str] = []
    diagnostics = storage.get("edge_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return [f"{case_id}: missing edge_diagnostics"]

    raw_candidates = storage.get("graph_candidate_ids")
    raw_paths = storage.get("graph_paths")
    candidates = raw_candidates if isinstance(raw_candidates, list) else None
    paths = raw_paths if isinstance(raw_paths, list) else None
    if candidates is None:
        violations.append(f"{case_id}: graph_candidate_ids is not a list")
        candidates = []
    if paths is None:
        violations.append(f"{case_id}: graph_paths is not a list")
        paths = []

    numeric_fields = (
        "seed_limit",
        "edge_limit_per_seed",
        "candidate_limit",
        "candidate_count",
        "path_count",
        "edges_considered",
        "duplicate_edges_skipped",
    )
    for field in numeric_fields:
        if not _nonnegative_int(diagnostics.get(field)):
            violations.append(f"{case_id}: invalid edge_diagnostics.{field}")
    if diagnostics.get("seed_limit") != P3A_FORMAL_GRAPH_SEED_LIMIT:
        violations.append(f"{case_id}: unexpected graph seed limit")
    if diagnostics.get("edge_limit_per_seed") != P3A_FORMAL_GRAPH_EDGE_LIMIT_PER_SEED:
        violations.append(f"{case_id}: unexpected graph edge limit per seed")
    if diagnostics.get("candidate_limit") != P3A_FORMAL_GRAPH_MAX_CANDIDATES:
        violations.append(f"{case_id}: unexpected graph candidate limit")

    candidate_count = diagnostics.get("candidate_count")
    path_count = diagnostics.get("path_count")
    edges_considered = diagnostics.get("edges_considered")
    duplicate_edges = diagnostics.get("duplicate_edges_skipped")
    candidate_limit = diagnostics.get("candidate_limit")
    if _nonnegative_int(candidate_count):
        if candidate_count != len(candidates):
            violations.append(f"{case_id}: candidate count does not match IDs")
        if _nonnegative_int(candidate_limit) and candidate_count > candidate_limit:
            violations.append(f"{case_id}: candidate cap exceeded")
    if len(candidates) != len(set(str(item) for item in candidates)):
        violations.append(f"{case_id}: duplicate graph candidate ID")
    if _nonnegative_int(path_count) and path_count != len(paths):
        violations.append(f"{case_id}: path count does not match paths")
    if (
        _nonnegative_int(candidate_count)
        and _nonnegative_int(path_count)
        and candidate_count > path_count
    ):
        violations.append(f"{case_id}: candidate count exceeds path count")
    if (
        _nonnegative_int(path_count)
        and _nonnegative_int(edges_considered)
        and path_count > edges_considered
    ):
        violations.append(f"{case_id}: path count exceeds considered edges")
    if (
        _nonnegative_int(duplicate_edges)
        and _nonnegative_int(edges_considered)
        and duplicate_edges > edges_considered
    ):
        violations.append(f"{case_id}: duplicate count exceeds considered edges")

    path_source_ids: set[str] = set()
    for path in paths:
        if not isinstance(path, Mapping):
            violations.append(f"{case_id}: graph path is not a record")
            continue
        source_ids = path.get("source_message_ids")
        if not isinstance(source_ids, list) or not source_ids:
            violations.append(f"{case_id}: graph path lacks source message IDs")
            continue
        path_source_ids.update(str(item) for item in source_ids)
    if set(str(item) for item in candidates) != path_source_ids:
        violations.append(f"{case_id}: graph candidate/path source IDs differ")

    resolved = storage.get("resolved_seeds")
    fetched = diagnostics.get("edges_fetched_by_seed")
    visits = diagnostics.get("edge_visit_seed_ids")
    resolved_rows = resolved if isinstance(resolved, list) else None
    fetched_rows = fetched if isinstance(fetched, list) else None
    visit_ids = visits if isinstance(visits, list) else None
    if resolved_rows is None:
        violations.append(f"{case_id}: resolved_seeds is not a list")
        resolved_rows = []
    if fetched_rows is None:
        violations.append(f"{case_id}: edges_fetched_by_seed is not a list")
        fetched_rows = []
    if visit_ids is None:
        violations.append(f"{case_id}: edge_visit_seed_ids is not a list")
        visit_ids = []
    resolved_ids: list[str] = []
    for item in resolved_rows:
        if not isinstance(item, Mapping) or not str(item.get("entity_id", "")):
            violations.append(f"{case_id}: invalid resolved seed record")
            continue
        resolved_ids.append(str(item["entity_id"]))
    if len(fetched_rows) != len(resolved_ids):
        violations.append(f"{case_id}: fetched seed accounting length mismatch")
    fetched_counts: list[int] = []
    for index, item in enumerate(fetched_rows):
        if not isinstance(item, Mapping):
            violations.append(f"{case_id}: invalid fetched-seed record")
            continue
        count = item.get("count")
        if not _nonnegative_int(count):
            violations.append(f"{case_id}: invalid fetched-seed count")
            continue
        if index < len(resolved_ids):
            fetched_counts.append(count)
        if count > P3A_FORMAL_GRAPH_EDGE_LIMIT_PER_SEED:
            violations.append(f"{case_id}: per-seed edge cap exceeded")
        if item.get("limit_reached") is not (
            count == P3A_FORMAL_GRAPH_EDGE_LIMIT_PER_SEED
        ):
            violations.append(f"{case_id}: fetched-seed limit flag is inconsistent")
        if index >= len(resolved_ids) or str(item.get("entity_id", "")) != resolved_ids[index]:
            violations.append(f"{case_id}: fetched-seed order differs from resolution")
    if (
        _nonnegative_int(edges_considered)
        and edges_considered > sum(fetched_counts)
    ):
        violations.append(f"{case_id}: considered edges exceed fetched edges")

    expected_visits: list[str] = []
    positions = [0 for _ in fetched_counts]
    while any(position < count for position, count in zip(positions, fetched_counts)):
        for index, count in enumerate(fetched_counts):
            if positions[index] < count:
                expected_visits.append(resolved_ids[index])
                positions[index] += 1
    normalized_visits = [str(item) for item in visit_ids]
    if _nonnegative_int(edges_considered) and len(normalized_visits) != edges_considered:
        violations.append(f"{case_id}: visit count differs from considered edges")
    if normalized_visits != expected_visits[:len(normalized_visits)]:
        violations.append(f"{case_id}: graph edge visits are not round-robin")

    cap_reached = diagnostics.get("candidate_cap_reached")
    if not isinstance(cap_reached, bool):
        violations.append(f"{case_id}: candidate cap flag is not boolean")
    elif cap_reached:
        if candidate_count != candidate_limit:
            violations.append(f"{case_id}: cap flag set before candidate limit")
        if len(normalized_visits) >= len(expected_visits):
            violations.append(f"{case_id}: cap flag set without unvisited edge")
    elif len(normalized_visits) != len(expected_visits):
        violations.append(f"{case_id}: traversal stopped before fetched edges were visited")
    return violations


def evaluate_p3a_gate(
    paired_report: Mapping[str, Any],
    relation_subset_manifest: Mapping[str, Any],
    *,
    formal_question_count: int = 200,
) -> dict[str, Any]:
    """Evaluate the predeclared P3-A gate from raw per-question counts."""

    traces = paired_report.get("question_traces")
    subset_records = relation_subset_manifest.get("records")
    if not isinstance(traces, list) or not isinstance(subset_records, list):
        raise ValueError("paired report and relation subset need question records")
    subset_by_case = {
        str(record["case_id"]): record
        for record in subset_records if isinstance(record, Mapping)
    }
    if len(subset_by_case) != len(subset_records):
        raise ValueError("relation subset contains duplicate case IDs")
    if len(traces) != len(subset_records):
        raise ValueError("paired report and relation subset question counts differ")
    if relation_subset_manifest.get("schema_version") != RELATION_SUBSET_SCHEMA_VERSION:
        raise ValueError("unsupported relation subset manifest schema")
    if relation_subset_manifest.get("policy") != RELATION_SUBSET_POLICY:
        raise ValueError("unsupported relation subset policy")
    embedded_subset_hash = str(
        relation_subset_manifest.get("subset_manifest_sha256", "")
    )
    if embedded_subset_hash != relation_subset_manifest_sha256(
        relation_subset_manifest
    ):
        raise ValueError("relation subset manifest digest mismatch")
    if str(relation_subset_manifest.get("source_manifest_sha256", "")) != str(
        paired_report.get("manifest_sha256", "")
    ):
        raise ValueError("relation subset and paired scope manifest mismatch")

    wins = losses = 0
    relation_wins = relation_losses = relation_n = 0
    oracle_off_hits = oracle_on_hits = 0
    oracle_off_evidence = oracle_on_evidence = 0
    total_evidence = 0
    graph_new_gold_questions = 0
    lost_gold_questions = 0
    seed_requested = seed_accounted = 0
    queries_with_plan_seeds = queries_with_resolved_seed = 0
    quota_violations: list[str] = []
    configuration_violations = _formal_graph_configuration_violations(
        paired_report
    )
    raw_configuration = paired_report.get("graph_configuration", {})
    configured_quota = (
        raw_configuration.get("rerank_quota", -1)
        if isinstance(raw_configuration, Mapping)
        and isinstance(raw_configuration.get("rerank_quota", -1), int)
        and not isinstance(raw_configuration.get("rerank_quota", -1), bool)
        else -1
    )
    graph_trace_mechanics_violations: list[str] = []

    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("question trace must be a record")
        case_id = str(trace.get("case_id", ""))
        subset = subset_by_case.get(case_id)
        if subset is None:
            raise ValueError(f"missing relation subset record for {case_id}")
        baseline = trace.get("baseline", {})
        graph_on = trace.get("graph_on", {})
        baseline_hit = int(baseline.get("hit_at_k", {}).get("1", 0))
        graph_hit = int(graph_on.get("hit_at_k", {}).get("1", 0))
        wins += int(not baseline_hit and graph_hit)
        losses += int(baseline_hit and not graph_hit)
        if bool(subset.get("relation_subset")):
            relation_n += 1
            relation_wins += int(not baseline_hit and graph_hit)
            relation_losses += int(baseline_hit and not graph_hit)

        storage = graph_on.get("storage", {})
        baseline_storage = baseline.get("storage", {})
        if not isinstance(storage, Mapping):
            graph_trace_mechanics_violations.append(
                f"{case_id}: graph storage trace is not a record"
            )
            storage = {}
        graph_trace_mechanics_violations.extend(
            _graph_trace_mechanics_violations(case_id=case_id, storage=storage)
        )
        p1_pool = list(storage.get("p1_top30_ids") or [])
        on_pool = list(storage.get("rerank_pool_ids") or [])
        baseline_pool = list(baseline_storage.get("rerank_pool_ids") or [])
        graph_candidates = set(storage.get("graph_candidate_ids") or [])
        reserved = list(storage.get("graph_reserved_ids") or [])
        gold = set(str(item) for item in subset.get("gold_mem_ids", []))
        total_evidence += len(gold)
        oracle_off_hits += int(bool(gold & set(p1_pool)))
        oracle_on_hits += int(bool(gold & set(on_pool)))
        oracle_off_evidence += len(gold & set(p1_pool))
        oracle_on_evidence += len(gold & set(on_pool))
        graph_new_gold_questions += int(not gold & set(p1_pool) and bool(gold & set(on_pool)))
        lost_gold_questions += int(bool(gold & set(p1_pool)) and not gold & set(on_pool))

        if baseline_pool != p1_pool:
            quota_violations.append(f"{case_id}: baseline pool differs from P1 top30")
        if len(on_pool) > 30 or len(on_pool) != len(set(on_pool)):
            quota_violations.append(f"{case_id}: invalid rerank pool")
        if len(reserved) > configured_quota:
            quota_violations.append(f"{case_id}: quota exceeded")
        if not set(reserved) <= graph_candidates & set(on_pool):
            quota_violations.append(f"{case_id}: invalid reserved candidate")
        if configured_quota == 0 and reserved:
            quota_violations.append(f"{case_id}: quota zero reserved a candidate")
        if len(p1_pool) == len(on_pool) == 30:
            if len(set(on_pool) - set(p1_pool)) != len(set(p1_pool) - set(on_pool)):
                quota_violations.append(f"{case_id}: quota movement is not balanced")

        requested = list(storage.get("requested_seeds") or [])
        resolved = list(storage.get("resolved_seeds") or [])
        unresolved = list(storage.get("unresolved_seeds") or [])
        seed_requested += len(requested)
        seed_accounted += len(resolved) + len(unresolved)
        queries_with_plan_seeds += int(bool(requested))
        queries_with_resolved_seed += int(bool(resolved))

    baseline_report = paired_report.get("baseline", {})
    graph_report = paired_report.get("graph_on", {})
    delta_hit10 = (
        float(graph_report.get("hit_at_k", {}).get("10", 0.0))
        - float(baseline_report.get("hit_at_k", {}).get("10", 0.0))
    )
    delta_evidence10 = (
        float(graph_report.get("evidence_recall_at_k", {}).get("10", 0.0))
        - float(baseline_report.get("evidence_recall_at_k", {}).get("10", 0.0))
    )
    integrity = paired_report.get("formal_integrity_audit", {})
    calls = paired_report.get("search_calls", {})
    hard_checks = {
        "storage_trace_complete": bool(paired_report.get("storage_trace_complete")),
        "database_unchanged": bool(paired_report.get("database_unchanged")),
        "integrity_violations_zero": int(integrity.get("violations", -1)) == 0,
        "unsupported_traversed_edges_zero": int(
            integrity.get("unsupported_traversed_edges", -1)
        ) == 0,
        "additional_search_calls_zero": int(
            calls.get("additional_graph_search_calls", -1)
        ) == 0,
        "frozen_graph_configuration": not configuration_violations,
        "graph_trace_mechanics": not graph_trace_mechanics_violations,
        "quota_invariants": not quota_violations,
        "seed_accounting_complete": seed_requested == seed_accounted,
        "oracle_hit30_non_regression": oracle_on_hits >= oracle_off_hits,
        "oracle_evidence30_non_regression": (
            oracle_on_evidence >= oracle_off_evidence
        ),
        "strict_graph_new_gold_observed": graph_new_gold_questions >= 1,
    }
    preservation = delta_hit10 >= -0.01 and delta_evidence10 >= -0.01
    overall_net = wins - losses
    relation_net = relation_wins - relation_losses
    overall_branch = overall_net >= math.ceil(0.02 * len(traces))
    relation_branch = (
        relation_n >= 30
        and relation_net >= math.ceil(0.05 * relation_n)
        and overall_net >= -math.floor(0.005 * len(traces))
    )
    formal_scope_checks = {
        "question_count": (
            len(traces) == formal_question_count == P3A_FORMAL_QUESTION_COUNT
        ),
        "question_offset": int(
            paired_report.get("question_offset", -1)
        ) == P3A_FORMAL_QUESTION_OFFSET,
        "max_questions": int(
            paired_report.get("max_questions", -1)
        ) == P3A_FORMAL_QUESTION_COUNT,
        "dataset_sha256": str(paired_report.get("dataset_sha256", ""))
        == P3A_FORMAL_DATASET_SHA256,
    }
    formal_scope = all(formal_scope_checks.values())
    promoted = (
        formal_scope and all(hard_checks.values()) and preservation
        and (overall_branch or relation_branch)
    )
    mechanics_checks = (
        "storage_trace_complete",
        "database_unchanged",
        "integrity_violations_zero",
        "unsupported_traversed_edges_zero",
        "additional_search_calls_zero",
        "frozen_graph_configuration",
        "graph_trace_mechanics",
        "quota_invariants",
        "seed_accounting_complete",
    )
    decision = (
        "PROMOTE_P3_A" if promoted
        else "MECHANICS_PASS" if (
            len(traces) < P3A_FORMAL_QUESTION_COUNT
            and all(hard_checks[key] for key in mechanics_checks)
        )
        else "REJECT_P3_A"
    )
    return {
        "schema_version": P3A_GATE_SCHEMA_VERSION,
        "decision": decision,
        "formal_scope": formal_scope,
        "formal_scope_checks": formal_scope_checks,
        "questions": len(traces),
        "wins": wins,
        "losses": losses,
        "net_hit_at_1": overall_net,
        "relation_questions": relation_n,
        "relation_wins": relation_wins,
        "relation_losses": relation_losses,
        "relation_net_hit_at_1": relation_net,
        "overall_branch_pass": overall_branch,
        "relation_branch_pass": relation_branch,
        "preservation_pass": preservation,
        "hard_checks": hard_checks,
        "configuration_violations": configuration_violations,
        "graph_trace_mechanics_violations": graph_trace_mechanics_violations,
        "quota_violations": quota_violations,
        "candidate_oracle": {
            "hit_at_30_off": oracle_off_hits / len(traces),
            "hit_at_30_on": oracle_on_hits / len(traces),
            "evidence_coverage_at_30_off": (
                oracle_off_evidence / total_evidence if total_evidence else 0.0
            ),
            "evidence_coverage_at_30_on": (
                oracle_on_evidence / total_evidence if total_evidence else 0.0
            ),
            "graph_new_gold_questions": graph_new_gold_questions,
            "lost_gold_questions": lost_gold_questions,
        },
        "seed_diagnostics": {
            "requested": seed_requested,
            "accounted": seed_accounted,
            "resolved_query_coverage": (
                queries_with_resolved_seed / queries_with_plan_seeds
                if queries_with_plan_seeds else 0.0
            ),
            "coverage_label": (
                "LOW_GRAPH_COVERAGE"
                if queries_with_plan_seeds
                and queries_with_resolved_seed / queries_with_plan_seeds < 0.5
                else "ADEQUATE_GRAPH_COVERAGE"
            ),
        },
    }
