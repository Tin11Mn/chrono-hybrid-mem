"""Offline retrieval metrics for non-official, user-provided evaluation cases."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List

from .schemas import AddRequest
from .storage import MemoryStore


def _validate_case(case: Dict[str, object]) -> None:
    required = ("case_id", "user_id", "session_id", "messages", "query", "expected_evidence")
    missing = [field for field in required if field not in case]
    if missing:
        raise ValueError("Case is missing required fields: {}".format(", ".join(missing)))
    if not case["expected_evidence"]:
        raise ValueError("Case {} has no expected_evidence".format(case["case_id"]))


def evaluate_cases(
    cases: Iterable[Dict[str, object]], database_path: str, top_ks: List[int]
) -> Dict[str, object]:
    """Measure evidence retrieval without invoking an answer model.

    Expected evidence entries are case-insensitive substrings that must occur in
    a returned memory record. The input must contain only data the caller may use.
    """
    if not top_ks or min(top_ks) < 1 or max(top_ks) > 100:
        raise ValueError("top_ks must be between 1 and 100")

    store = MemoryStore(database_path)
    store.initialize()
    materialized_cases = list(cases)
    for case in materialized_cases:
        _validate_case(case)
        store.add(AddRequest(
            request_id="offline:{}".format(case["case_id"]),
            user_id=str(case["user_id"]),
            session_id=str(case["session_id"]),
            messages=case["messages"],
        ))

    recall_hits = {top_k: 0 for top_k in top_ks}
    coverage_hits = {top_k: 0 for top_k in top_ks}
    reciprocal_rank_sum = 0.0
    evidence_total = 0
    elapsed_ms = 0.0

    for case in materialized_cases:
        expected = [str(item).casefold() for item in case["expected_evidence"]]
        evidence_total += len(expected)
        started = time.perf_counter()
        results = store.search(
            user_id=str(case["user_id"]), query=str(case["query"]), top_k=max(top_ks)
        )
        elapsed_ms += (time.perf_counter() - started) * 1000
        contents = [result.content.casefold() for result in results]
        ranks = [
            next((index + 1 for index, content in enumerate(contents) if evidence in content), None)
            for evidence in expected
        ]
        first_rank = next((rank for rank in ranks if rank is not None), None)
        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank
        for top_k in top_ks:
            matches = [rank for rank in ranks if rank is not None and rank <= top_k]
            recall_hits[top_k] += len(matches)
            if len(matches) == len(expected):
                coverage_hits[top_k] += 1

    case_count = len(materialized_cases)
    return {
        "cases": case_count,
        "evidence_items": evidence_total,
        "recall_at_k": {
            str(top_k): round(recall_hits[top_k] / evidence_total, 4) if evidence_total else 0.0
            for top_k in top_ks
        },
        "case_coverage_at_k": {
            str(top_k): round(coverage_hits[top_k] / case_count, 4) if case_count else 0.0
            for top_k in top_ks
        },
        "mrr": round(reciprocal_rank_sum / case_count, 4) if case_count else 0.0,
        "average_search_ms": round(elapsed_ms / case_count, 3) if case_count else 0.0,
    }
