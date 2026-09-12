"""Canonical attempt/state reducers for LoCoMo E2E runs.

Implements the ChatAnywhere Formal Identity Policy v1.0 (2026-09-13):

- canonical state per question_id derived ONLY from persisted evidence
  (raw_model_outputs first, attempt metadata second), never "last line wins";
- states: accepted / identity_exhausted / transport_exhausted /
  rate_limit_exhausted / malformed_exhausted / evaluator_invariant_violation /
  retrieval_timeout / not_attempted / provenance_missing — they must sum
  exactly to the manifest total;
- accepted = model_identity ∈ {snapshot_verified, alias_only_snapshot_unverified}
  (BOTH are valid ChatAnywhere GPT-4o-mini responses). Only explicit_wrong_model
  is a genuine violation;
- an accepted row is only trusted when a real raw response with
  returned_model ∈ {gpt-4o-mini, gpt-4o-mini-2024-07-18} exists (provenance
  chain);
- stored model_identity/accepted are cross-checked against the recomputed
  verdict; any mismatch is evaluator_invariant_violation, not gateway drift.
"""
from __future__ import annotations

import json
from pathlib import Path

EXPECTED_SNAPSHOT = "gpt-4o-mini-2024-07-18"
REQUESTED_ALIAS = "gpt-4o-mini"
ACCEPTED_MODEL_IDS = frozenset({EXPECTED_SNAPSHOT, REQUESTED_ALIAS})


def classify_model_identity(returned_model) -> str:
    norm = str(returned_model).strip() if returned_model is not None else ""
    if norm == EXPECTED_SNAPSHOT:
        return "snapshot_verified"
    if norm == REQUESTED_ALIAS:
        return "alias_only_snapshot_unverified"
    if norm:
        return "explicit_wrong_model"
    return "unavailable"


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


def _attempt_identity(attempt: dict) -> str:
    """model_identity of one attempt: stored field else recomputed from raw."""
    stored = attempt.get("model_identity")
    if stored:
        return stored
    return classify_model_identity(attempt.get("returned_model"))


def _attempt_accepted(attempt: dict) -> bool:
    """accepted flag of one attempt: stored field else recomputed."""
    stored = attempt.get("accepted")
    if stored is not None:
        return bool(stored)
    return _attempt_identity(attempt) in {"snapshot_verified",
                                          "alias_only_snapshot_unverified"}


def _attempt_outcome(attempt: dict) -> str:
    return attempt.get("attempt_outcome") or "unknown"


def canonical_state_for_question(qid: str, pq_rows: list,
                                 raw_rows: list) -> dict:
    """Derive the canonical state from persisted evidence only."""
    raw_accepted = [r for r in raw_rows
                    if r.get("status") == "ok"
                    and (r.get("returned_model") or "").strip()
                        in ACCEPTED_MODEL_IDS]
    ok_rows = [r for r in pq_rows if r.get("status") == "ok"]
    err_rows = [r for r in pq_rows if r.get("status") == "error"]
    timeout_rows = [r for r in pq_rows
                    if r.get("status") == "retrieval_timeout"]

    def _exhaustion_state(r) -> str:
        attempts = r.get("answer_attempts") or []
        # invariant scan: stored == recomputed for every attempt
        for a in attempts:
            recomputed_id = classify_model_identity(a.get("returned_model"))
            recomputed_acc = recomputed_id in {"snapshot_verified",
                                               "alias_only_snapshot_unverified"}
            stored_id = a.get("model_identity")
            stored_acc = a.get("accepted")
            if (stored_id is not None and stored_id != recomputed_id) or \
               (stored_acc is not None and bool(stored_acc) != recomputed_acc):
                return "evaluator_invariant_violation"
        if not attempts:
            return "identity_exhausted"
        outcomes = [_attempt_outcome(a) for a in attempts]
        # exhaustion classification by the repeated kind that caused the stop
        if any(o == "transport_error" for o in outcomes):
            return "transport_exhausted"
        if any(o == "rate_limit" for o in outcomes):
            return "rate_limit_exhausted"
        if any(o == "malformed_response" for o in outcomes):
            return "malformed_exhausted"
        return "identity_exhausted"

    if ok_rows and raw_accepted:
        state = "accepted"
    elif ok_rows and not raw_accepted:
        state = "provenance_missing"
    elif err_rows:
        state = _exhaustion_state(err_rows[-1])
    elif timeout_rows:
        state = "retrieval_timeout"
    else:
        state = "not_attempted"

    n_attempts = sum(len(r.get("answer_attempts") or []) for r in pq_rows)
    n_attempts += sum(1 for r in pq_rows
                      if r.get("answer_attempts") is None
                      and r.get("status") in ("ok", "error"))
    return {"question_id": qid, "state": state,
            "raw_accepted_files": len(raw_accepted),
            "attempt_rows": n_attempts,
            "pq_rows": len(pq_rows)}


def reduce_manifest(manifest: dict, per_question_path: Path,
                    run_dir: Path) -> dict:
    """Full canonical reduction for one run.

    Returns {coverage: {...}, total, per_question: [...]} with the hard
    invariant coverage sum == manifest total enforced.
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