"""Canonical reducer + identity taxonomy tests (2026-09-12 forensic rules).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import canonical_reduce as cr  # noqa: E402
import evaluate_locomo_e2e as e2e  # noqa: E402


@pytest.fixture
def workdir():
    """Workspace-local temp dir (pytest tmp_path is sandbox-blocked)."""
    with tempfile.TemporaryDirectory(prefix="canonical-test-", dir=str(REPO)) as d:
        yield Path(d)


# --------------------------------------------------------------------------- #
# identity classification                                                    #
# --------------------------------------------------------------------------- #
def test_classify_snapshot_is_valid():
    v = cr.classify_returned_model("gpt-4o-mini-2024-07-18")
    assert v["identity_valid"] is True
    assert v["identity_category"] == "identity_valid"


def test_classify_alias_is_unverified():
    v = cr.classify_returned_model("gpt-4o-mini")
    assert v["identity_valid"] is False
    assert v["identity_category"] == "snapshot_identity_unverified"


def test_classify_other_model_is_wrong():
    v = cr.classify_returned_model("gpt-4.1-mini-2025-04-14")
    assert v["identity_valid"] is False
    assert v["identity_category"] == "explicit_wrong_model"


def test_classify_none_is_transport():
    v = cr.classify_returned_model(None)
    assert v["identity_valid"] is False
    assert v["identity_category"] == "transport_error"


def test_e2e_verify_invariant_raises_on_stored_mismatch():
    with pytest.raises(RuntimeError) as exc:
        e2e.verify_identity_invariant(
            "q1", 1, "gpt-4o-mini-2024-07-18", stored_valid=False, status="ok")
    assert "evaluator_invariant_violation" in str(exc.value)


def test_e2e_verify_invariant_passes_when_consistent():
    cat = e2e.verify_identity_invariant(
        "q1", 1, "gpt-4o-mini", stored_valid=False, status="ok")
    assert cat == "snapshot_identity_unverified"


# --------------------------------------------------------------------------- #
# canonical reducer                                                          #
# --------------------------------------------------------------------------- #
MANIFEST = {"questions": [
    {"question_id": "c1:0", "offset": 0, "category_id": 1},
    {"question_id": "c1:1", "offset": 1, "category_id": 2},
    {"question_id": "c2:0", "offset": 2, "category_id": 3},
]}


def _write(p: Path, rows, raw_single_json=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    if raw_single_json:
        # raw_model_outputs files are single JSON objects (not JSONL)
        payload = rows if isinstance(rows, (dict, list)) else {}
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                           for r in rows) + "\n", encoding="utf-8")


def test_reduce_accepted_provenance_chain(workdir):
    run = Path(workdir) / "run"
    per_q = run / "per_question.jsonl"
    _write(per_q, [
        {"question_id": "c1:0", "status": "ok", "answer_returned_model": e2e.EXPECTED_RETURNED_MODEL,
         "answer_attempts": None},
        {"question_id": "c1:1", "status": "ok",
         "answer_attempts": [{"attempt": 1, "returned_model": e2e.EXPECTED_RETURNED_MODEL,
                              "valid_model_identity": True, "identity_category": "identity_valid",
                              "status": "accepted"}]},
        {"question_id": "c2:0", "status": "error", "answer_attempts": [
            {"attempt": 1, "returned_model": "gpt-4o-mini", "valid_model_identity": False,
             "identity_category": "snapshot_identity_unverified", "status": "snapshot_identity_unverified"},
            {"attempt": 2, "returned_model": "gpt-4o-mini", "valid_model_identity": False,
             "identity_category": "snapshot_identity_unverified", "status": "snapshot_identity_unverified"},
            {"attempt": 3, "returned_model": "gpt-4o-mini", "valid_model_identity": False,
             "identity_category": "snapshot_identity_unverified", "status": "snapshot_identity_unverified"},
        ]},
    ])
    # raw responses for the two accepted questions
    raw = run / "raw_model_outputs"
    _write(raw / "c1_0.answer.json", {"question_id": "c1:0", "status": "ok",
                                      "returned_model": e2e.EXPECTED_RETURNED_MODEL},
           raw_single_json=True)
    _write(raw / "c1_1.answer.json", {"question_id": "c1:1", "status": "ok",
                                      "returned_model": e2e.EXPECTED_RETURNED_MODEL},
           raw_single_json=True)
    res = cr.reduce_manifest(MANIFEST, per_q, run)
    assert res["total"] == 3
    by_qid = {r["question_id"]: r["state"] for r in res["per_question"]}
    assert by_qid == {"c1:0": "accepted", "c1:1": "accepted", "c2:0": "identity_exhausted"}
    assert res["coverage"] == {"accepted": 2, "identity_exhausted": 1}


def test_reduce_ok_without_raw_is_provenance_missing(workdir):
    run = Path(workdir) / "run2"
    per_q = run / "per_question.jsonl"
    _write(per_q, [{"question_id": "c1:0", "status": "ok",
                    "answer_attempts": None}])
    # NO raw file: accepted claim without provenance
    res = cr.reduce_manifest(MANIFEST, per_q, run)
    by_qid = {r["question_id"]: r["state"] for r in res["per_question"]}
    assert by_qid["c1:0"] == "provenance_missing"


def test_reduce_invariant_violation_detected(workdir):
    run = Path(workdir) / "run3"
    per_q = run / "per_question.jsonl"
    # error row with a single attempt whose stored identity contradicts the
    # recomputed verdict -> evaluator_invariant_violation
    _write(per_q, [{"question_id": "c2:0", "status": "error", "answer_attempts": [
        {"attempt": 1, "returned_model": e2e.EXPECTED_RETURNED_MODEL,
         "valid_model_identity": False,  # stored false but returned IS snapshot
         "identity_category": "identity_valid", "status": "accepted"},
    ]}])
    res = cr.reduce_manifest(MANIFEST, per_q, run)
    by_qid = {r["question_id"]: r["state"] for r in res["per_question"]}
    assert by_qid["c2:0"] == "evaluator_invariant_violation"