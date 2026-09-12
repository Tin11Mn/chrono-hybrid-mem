"""Canonical reducer + identity taxonomy tests (Formal Identity Policy v1.0).
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
import judge_locomo_answers as jd  # noqa: E402

SNAPSHOT = e2e.EXPECTED_RETURNED_MODEL
ALIAS = e2e.REQUESTED_MODEL


@pytest.fixture
def workdir():
    """Workspace-local temp dir (pytest tmp_path is sandbox-blocked)."""
    with tempfile.TemporaryDirectory(prefix="canonical-test-", dir=str(REPO)) as d:
        yield Path(d)


# --------------------------------------------------------------------------- #
# identity classification (Formal policy: alias IS valid)                    #
# --------------------------------------------------------------------------- #
def test_classify_snapshot_is_snapshot_verified():
    assert cr.classify_model_identity(SNAPSHOT) == "snapshot_verified"


def test_classify_alias_is_alias_only_snapshot_unverified():
    assert cr.classify_model_identity(ALIAS) == "alias_only_snapshot_unverified"


def test_classify_other_model_is_explicit_wrong():
    assert cr.classify_model_identity("gpt-4.1-mini-2025-04-14") == "explicit_wrong_model"


def test_classify_none_is_unavailable():
    assert cr.classify_model_identity(None) == "unavailable"


def test_classify_empty_is_unavailable():
    assert cr.classify_model_identity("") == "unavailable"


def test_alias_is_accepted_identity():
    assert ALIAS in cr.ACCEPTED_MODEL_IDS


def test_e2e_invariant_raises_on_stored_mismatch():
    with pytest.raises(RuntimeError) as exc:
        e2e.verify_attempt_invariant(
            "q1", 1, SNAPSHOT, model_identity="explicit_wrong_model",
            accepted=False)
    assert "evaluator_invariant_violation" in str(exc.value)


def test_e2e_invariant_passes_when_stored_and_recomputed_match():
    cat = e2e.verify_attempt_invariant(
        "q1", 1, ALIAS, model_identity="alias_only_snapshot_unverified",
        accepted=True)
    assert cat == "alias_only_snapshot_unverified"


def test_judge_classifier_alias_is_valid():
    assert jd.classify_model_identity(ALIAS) == "alias_only_snapshot_unverified"
    assert jd.classify_model_identity(ALIAS) in jd.ACCEPTED_IDENTITIES


# --------------------------------------------------------------------------- #
# canonical reducer (alias is now ACCEPTED)                                  #
# --------------------------------------------------------------------------- #
MANIFEST = {"questions": [
    {"question_id": "c1:0", "offset": 0, "category_id": 1},
    {"question_id": "c1:1", "offset": 1, "category_id": 2},
    {"question_id": "c2:0", "offset": 2, "category_id": 3},
]}


def _write(p: Path, rows, raw_single_json=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    if raw_single_json:
        payload = rows if isinstance(rows, (dict, list)) else {}
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                           for r in rows) + "\n", encoding="utf-8")


def test_reduce_snapshot_and_alias_both_accepted(workdir):
    run = Path(workdir) / "run"
    per_q = run / "per_question.jsonl"
    # c1:0 snapshot-accepted, c1:1 alias-accepted (BOTH legal), c2:0 3x
    # explicit_wrong_model -> identity_exhausted
    _write(per_q, [
        {"question_id": "c1:0", "status": "ok",
         "answer_attempts": [{"attempt": 1, "returned_model": SNAPSHOT,
                              "model_identity": "snapshot_verified",
                              "accepted": True, "attempt_outcome": "accepted"}]},
        {"question_id": "c1:1", "status": "ok",
         "answer_attempts": [{"attempt": 1, "returned_model": ALIAS,
                              "model_identity": "alias_only_snapshot_unverified",
                              "accepted": True, "attempt_outcome": "accepted"}]},
        {"question_id": "c2:0", "status": "error", "answer_attempts": [
            {"attempt": 1, "returned_model": "wrong-model-a",
             "model_identity": "explicit_wrong_model", "accepted": False,
             "attempt_outcome": "explicit_wrong_model"},
            {"attempt": 2, "returned_model": "wrong-model-a",
             "model_identity": "explicit_wrong_model", "accepted": False,
             "attempt_outcome": "explicit_wrong_model"},
            {"attempt": 3, "returned_model": "wrong-model-a",
             "model_identity": "explicit_wrong_model", "accepted": False,
             "attempt_outcome": "explicit_wrong_model"},
        ]},
    ])
    raw = run / "raw_model_outputs"
    _write(raw / "c1_0.answer.json", {"question_id": "c1:0", "status": "ok",
                                      "returned_model": SNAPSHOT},
           raw_single_json=True)
    _write(raw / "c1_1.answer.json", {"question_id": "c1:1", "status": "ok",
                                      "returned_model": ALIAS},
           raw_single_json=True)
    res = cr.reduce_manifest(MANIFEST, per_q, run)
    assert res["total"] == 3
    by_qid = {r["question_id"]: r["state"] for r in res["per_question"]}
    assert by_qid == {"c1:0": "accepted", "c1:1": "accepted", "c2:0": "identity_exhausted"}


def test_reduce_ok_without_raw_is_provenance_missing(workdir):
    run = Path(workdir) / "run2"
    per_q = run / "per_question.jsonl"
    _write(per_q, [{"question_id": "c1:0", "status": "ok",
                    "answer_attempts": None}])
    res = cr.reduce_manifest(MANIFEST, per_q, run)
    by_qid = {r["question_id"]: r["state"] for r in res["per_question"]}
    assert by_qid["c1:0"] == "provenance_missing"


def test_reduce_invariant_violation_detected(workdir):
    run = Path(workdir) / "run3"
    per_q = run / "per_question.jsonl"
    # error row whose stored identity contradicts recomputed -> violation
    _write(per_q, [{"question_id": "c2:0", "status": "error", "answer_attempts": [
        {"attempt": 1, "returned_model": SNAPSHOT,
         "model_identity": "explicit_wrong_model",  # stored contradicts
         "accepted": False, "attempt_outcome": "accepted"},
    ]}])
    res = cr.reduce_manifest(MANIFEST, per_q, run)
    by_qid = {r["question_id"]: r["state"] for r in res["per_question"]}
    assert by_qid["c2:0"] == "evaluator_invariant_violation"


def test_reduce_rate_limit_and_malformed_states(workdir):
    run = Path(workdir) / "run4"
    per_q = run / "per_question.jsonl"
    # all 3 attempts are 429s -> rate_limit_exhausted
    _write(per_q, [{"question_id": "c2:0", "status": "error", "answer_attempts": [
        {"attempt": i, "returned_model": None, "model_identity": "unavailable",
         "accepted": False, "attempt_outcome": "rate_limit"}
        for i in (1, 2, 3)]}])
    res = cr.reduce_manifest(MANIFEST, per_q, run)
    assert res["per_question"][-1]["state"] == "rate_limit_exhausted"

    run2 = Path(workdir) / "run5"
    per_q2 = run2 / "per_question.jsonl"
    # all 3 attempts are malformed responses -> malformed_exhausted
    _write(per_q2, [{"question_id": "c2:0", "status": "error", "answer_attempts": [
        {"attempt": i, "returned_model": None, "model_identity": "unavailable",
         "accepted": False, "attempt_outcome": "malformed_response"}
        for i in (1, 2, 3)]}])
    res2 = cr.reduce_manifest(MANIFEST, per_q2, run2)
    assert res2["per_question"][-1]["state"] == "malformed_exhausted"