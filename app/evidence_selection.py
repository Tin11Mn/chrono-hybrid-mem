"""Validate source-only evidence decisions; never synthesize answer content."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def validate_selection(payload: Any, allowed_ids: Sequence[str],
                       evidence_needs: Sequence[str]) -> Dict[str, Any]:
    """Accept a complete coverage certificate, or preserve the legacy retrieval path.

    This validates referential integrity and declared coverage, not the model's
    semantic correctness. No text supplied by the model becomes response content.
    """
    invalid = {"applied": False, "reason": "invalid_decision", "selected_ids": []}
    if not isinstance(payload, dict):
        return invalid
    if payload.get("complete") is not True:
        return {**invalid, "reason": "incomplete_decision"}
    allowed = set(allowed_ids)

    def id_list(value: Any, nonempty: bool = False) -> Optional[List[str]]:
        if not isinstance(value, list) or (nonempty and not value):
            return None
        if any(not isinstance(item, str) or item not in allowed for item in value):
            return None
        if len(set(value)) != len(value):
            return None
        return list(value)

    selected = id_list(payload.get("selected_ids"), nonempty=True)
    excluded = id_list(payload.get("excluded_ids", []))
    if selected is None or excluded is None or set(selected) & set(excluded):
        return invalid
    groups = payload.get("evidence_groups")
    if not isinstance(groups, list) or not groups:
        return {**invalid, "reason": "missing_coverage"}
    covered_needs = set()
    covered_ids = set()
    normalized_groups = []
    for group in groups:
        if not isinstance(group, dict):
            return invalid
        ids = id_list(group.get("ids"), nonempty=True)
        need_index = group.get("need_index")
        if ids is None or not set(ids).issubset(selected):
            return {**invalid, "reason": "missing_coverage"}
        if need_index is not None:
            if (isinstance(need_index, bool) or not isinstance(need_index, int)
                    or not 0 <= need_index < len(evidence_needs)):
                return invalid
            covered_needs.add(need_index)
        covered_ids.update(ids)
        normalized_groups.append({"need_index": need_index, "ids": ids})
    if covered_needs != set(range(len(evidence_needs))) or covered_ids != set(selected):
        return {**invalid, "reason": "missing_coverage"}
    states = payload.get("state_updates", [])
    if not isinstance(states, list):
        return invalid
    normalized_states = []
    for state in states:
        if not isinstance(state, dict):
            return invalid
        current = state.get("current_id")
        old = id_list(state.get("superseded_ids"), nonempty=True)
        if not isinstance(current, str) or current not in selected or old is None:
            return invalid
        if current in old or not set(old).issubset(excluded):
            return {**invalid, "reason": "state_conflict"}
        normalized_states.append({"current_id": current, "superseded_ids": old})
    return {"applied": True, "reason": "complete", "selected_ids": selected,
            "excluded_ids": excluded, "evidence_groups": normalized_groups,
            "state_updates": normalized_states}


def append_redacted_trace(path: str, trace: Dict[str, Any]) -> None:
    """Append only whitelisted diagnostics; never serialize query/plan/source text.

    A single O_APPEND write avoids sharing the mutable last-query snapshot. The
    caller serializes writes inside its diagnostics lock. Keep the file private.
    """
    record = {"version": 1, "final_ids": trace.get("final_ids", []),
              "rerank_pool_ids": trace.get("rerank_pool_ids", []),
              "coverage_candidate_ids": trace.get("coverage_candidate_ids", []),
              "selection": trace.get("evidence_selection", {})}
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=True) + "\n").encode("utf-8")
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if os.write(descriptor, line) != len(line):
                logging.getLogger(__name__).warning("retrieval_trace_write_incomplete")
        finally:
            os.close(descriptor)
    except OSError:
        logging.getLogger(__name__).warning("retrieval_trace_write_failed")
