"""Canonical attempt/state reducers for Full-1540 pilots.

Implements the 2026-09-12 CANONICAL FORENSIC RECONSTRUCTION rules:

- canonical state per question_id derived ONLY from persisted evidence
  (raw_model_outputs first, attempt metadata second), never "last line wins";
- states: accepted / identity_exhausted / transport_exhausted /
  rate_limit_exhausted / evaluator_invariant_violation / not_attempted /
  provenance_missing — they must sum exactly to the manifest total;
- an accepted row is only trusted when a real raw response with
  returned_model == expected snapshot exists (provenance chain);
- stored identity verdicts are cross-checked against the recomputed verdict;
  any mismatch is evaluator_invariant_violation, not gateway drift.

The 16-row overlap seen in the first ledger pass (a raw accepted row plus the
same attempt re-added from per_question answer_attempts[]) is handled here:
canonical accepted rows come from the RAW files; rejected attempts come from
attempts[] metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

EXPECTED_SNAPSHOT = "gpt-4o-mini-2024-07-18"
REQUESTED_ALIAS = "gpt-4o-mini"


def classify_returned_model(returned_model) -> dict:
    norm = str(returned_model).strip() if returned_model is not None else ""
    if norm == EXPECTED_SNAPSHOT:
        return {"identity_valid": True, "identity_category": "identity_valid"}
    if norm == REQUESTED_ALIAS:
        return {"identity_valid": False,
                "identity_category": "snapshot_identity_unverified"}
    if norm:
        return {"identity_valid": False,
                "identity_category": "explicit_wrong_model"}
    return {"identity_valid": False, "identity_category": "transport_error"}


def read_jsonl(path: Path) -> list:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(
        encoding="utf-8").splitlines() if l.strip()]


def load_raw_answers(run_dir: Path) -> dict:
    """qid -> list of raw answer payloads (highest-priority source)."""
    raw_dir = Path(run_dir) / "raw_model_outputs"
    out = {}
    if not raw_dir.exists():
        return out
    for f in sorted(raw_dir.glob("*.answer.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        qid = r.get("question_id")
        if qid:
            out.setdefault(qid, []).append(r)
    return out


def canonical_state_for_question(qid: str, pq_rows: list,
                                 raw_rows: list) -> dict:
    """Derive the canonical state from persisted evidence only."""
    # accepted requires BOTH an ok per_question row AND a raw response whose
    # returned model equals the expected snapshot (provenance chain).
    raw_accepted = [r for r in raw_rows
                    if r.get("status") == "ok"
                    and (r.get("returned_model") or "").strip() == EXPECTED_SNAPSHOT]
    ok_rows = [r for r in pq_rows if r.get("status") == "ok"]
    err_rows = [r for r in pq_rows if r.get("status") == "error"]
    timeout_rows = [r for r in pq_rows
                    if r.get("status") == "retrieval_timeout"]
    if ok_rows and raw_accepted:
        state = "accepted"
    elif ok_rows and not raw_accepted:
        state = "provenance_missing"
    elif err_rows:
        r = err_rows[-1]
        cats = {a.get("identity_category") or classify_returned_model(
            a.get("returned_model"))["identity_category"]
            for a in (r.get("answer_attempts") or [])}
        # invariant scan over stored attempts
        for a in (r.get("answer_attempts") or []):
            recomputed = classify_returned_model(
                a.get("returned_model"))["identity_valid"]
            stored = a.get("valid_model_identity",
                           a.get("identity_valid"))
            if stored is not None and bool(stored) != recomputed:
                state = "evaluator_invariant_violation"
                break
        else:
            if "transport_error" in cats and len(cats) == 1:
                state = "transport_exhausted"
            else:
                state = "identity_exhausted"
    elif timeout_rows:
        state = "retrieval_timeout"
    else:
        state = "not_attempted"
    n_attempts = sum(len(r.get("answer_attempts") or []) for r in pq_rows)
    # rows without an attempts[] array (first-run schema) still consumed one call
    n_attempts += sum(1 for r in pq_rows
                      if r.get("answer_attempts") is None
                      and r.get("status") in ("ok", "error"))
    return {"question_id": qid, "state": state,
            "raw_accepted_files": len(raw_accepted),
            "attempt_rows": n_attempts,
            "pq_rows": len(pq_rows)}


def reduce_manifest(manifest: dict, per_question_path: Path,
                    run_dir: Path) -> dict:
    """Full canonical reduction for one pilot run.

    Returns {coverage: {...}, states: {qid: state}, per_question: [...]} with
    the hard invariant coverage sum == manifest total enforced.
    """
    pq_rows = read_jsonl(per_question_path)
    by_qid = {}
    for r in pq_rows:
        by_qid.setdefault(r.get("question_id"), []).append(r)
    raw = load_raw_answers(Path(run_dir))
    per_question = []
    for entry in manifest["questions"]:
        qid = entry["question_id"]
        rec = canonical_state_for_question(qid, by_qid.get(qid, []),
                                           raw.get(qid, []))
        rec["manifest_offset"] = entry["offset"]
        rec["category_id"] = entry["category_id"]
        per_question.append(rec)
    coverage = {}
    for rec in per_question:
        coverage[rec["state"]] = coverage.get(rec["state"], 0) + 1
    total = sum(coverage.values())
    assert total == len(manifest["questions"]), (
        "canonical coverage invariant violated: {} != {}".format(
            total, len(manifest["questions"])))
    # accepted rows must all carry provenance
    for rec in per_question:
        if rec["state"] == "accepted":
            assert rec["raw_accepted_files"] >= 1, (
                "accepted without raw provenance: {}".format(rec["question_id"]))
    return {"coverage": coverage, "total": total,
            "per_question": per_question}
