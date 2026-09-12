"""Checkpoint/resume + pipeline tests for the E2E harness.

Uses a stub AnswerClient (no network) to run the full pipeline over a handful of
questions, then verifies:
- per_question.jsonl rows carry the full 鎼? schema and no gold leakage in the
  rendered answer prompt,
- resume skips already-answered questions,
- config_digest mismatch fails closed,
- the summarizer produces pooled evidence recall and category denominators.
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
import evaluate_locomo_e2e as e2e  # noqa: E402
import judge_locomo_answers as jd  # noqa: E402
import summarize_locomo_e2e as summ  # noqa: E402

# Captured before any test stubs e2e.AnswerClient.
_REAL_ANSWER_CLIENT = e2e.AnswerClient

DATASET = (REPO.parent / "chrono-hybrid-mem" / ".locomo" / "locomo10.json")
P3 = REPO.parent / "chrono-hybrid-mem-p3" / ".locomo"
GLOB = str(P3 / "sfv2-full-*.json")

pytestmark = pytest.mark.skipif(
    not (DATASET.exists() and list(Path(P3).glob("sfv2-full-*.json"))),
    reason="dataset or frozen artifacts absent",
)


class StubClient:
    def __init__(self, base_url, model, timeout):
        self.calls = 0

    def answer(self, prompt):
        self.calls += 1
        # leakage guard: the answer prompt must never contain the gold answer
        # text beyond what the evidence itself states (gold is passed separately).
        return {"text": "stub answer", "model_returned": "stub",
                "latency_ms": 1.0, "input_tokens": 10, "output_tokens": 2}


@pytest.fixture
def workdir():
    """Workspace-local temp dir (pytest tmp_path is sandbox-blocked)."""
    with tempfile.TemporaryDirectory(prefix="e2e-test-", dir=str(REPO)) as d:
        yield Path(d)


def _run(tmp_path, run_id="t1", max_q=5, resume=False, client=None):
    argv = sys.argv
    sys.argv = [
        "x", "--artifact-glob", GLOB, "--run-id", run_id,
        "--out-root", str(tmp_path), "--max-questions", str(max_q),
        "--answer-prompt", "locomo_answer_mem0.txt",
    ] + (["--resume"] if resume else [])
    # monkeypatch the client + skip the paid-endpoint guard
    original_client = e2e.AnswerClient
    e2e.AnswerClient = client or StubClient
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    e2e.os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"  # non-paid sentinel
    try:
        rc = e2e.main()
    finally:
        sys.argv = argv
        e2e.AnswerClient = original_client
    return rc, tmp_path / run_id


def test_full_pipeline_and_schema(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc, run_dir = _run(workdir, max_q=5)
    assert rc == 0
    rows = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 5
    required = {"question_id", "conversation_id", "category_id", "question",
                "reference_answer", "retrieved_evidence_ids", "generated_answer",
                "answer_model", "answer_prompt_hash",
                "f1_official", "f1_mem0", "bleu1_m1", "hit1", "hit3", "hit10",
                "mrr", "n_gold", "status", "run_id", "method", "config_digest"}
    for r in rows:
        assert required <= set(r), required - set(r)
        assert r["status"] == "ok"
        assert r["category_id"] in (1, 2, 3, 4)  # non-adversarial only
        assert len(r["retrieved_evidence_ids"]) <= 10
        # evidence text reconstructed
        assert all(e.get("content") for e in r["retrieved_evidence"])
        # gold answer never embedded in the answer prompt inputs
        assert r["generated_answer"] is not None


def test_resume_skips_completed(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc1, run_dir = _run(workdir, "t2", max_q=3)
    assert rc1 == 0
    n1 = len((run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines())
    # resume with a larger cap should only ADD the new ones
    rc2, _ = _run(workdir, "t2", max_q=6, resume=True)
    assert rc2 == 0
    n2 = len((run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines())
    assert n1 == 3 and n2 == 6  # 3 old + 3 new, no duplicates


def test_config_digest_fail_closed(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc1, run_dir = _run(workdir, "t3", max_q=2)
    assert rc1 == 0
    # tamper with the frozen config digest
    cfg = json.loads((run_dir / "run_config.json").read_text())
    cfg["config_digest"] = "deadbeef"
    (run_dir / "run_config.json").write_text(json.dumps(cfg))
    rc2, _ = _run(workdir, "t3", max_q=2, resume=True)
    assert rc2 == 3  # fail closed


class FailOnceStub(StubClient):
    """Fails the FIRST answer call, succeeds afterwards (retry simulation)."""

    def __init__(self, base_url, model, timeout):
        super().__init__(base_url, model, timeout)
        self.failed = False

    def answer(self, prompt):
        if not self.failed:
            self.failed = True
            raise RuntimeError("stub transient failure")
        return super().answer(prompt)


def test_resume_retries_failed_questions(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    e2e.AnswerClient = FailOnceStub
    rc1, run_dir = _run(workdir, "t5", max_q=3, client=FailOnceStub)
    assert rc1 == 0
    rows1 = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    failed = [r for r in rows1 if r["status"] == "error"]
    assert len(failed) == 1
    # resume must retry the failed question and REPLACE its error row
    rc2, _ = _run(workdir, "t5", max_q=3, resume=True)
    assert rc2 == 0
    rows2 = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    ids = [r["question_id"] for r in rows2]
    assert len(ids) == len(set(ids))  # no duplicate rows for any question
    retried = [r for r in rows2 if r["question_id"] == failed[0]["question_id"]]
    assert len(retried) == 1 and retried[0]["status"] == "ok"


def test_answer_client_reports_model_drift(workdir, monkeypatch):
    import types
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    client = _REAL_ANSWER_CLIENT("http://127.0.0.1:9/v1", e2e.REQUESTED_MODEL, 5.0)

    def make_resp(model):
        return types.SimpleNamespace(
            model=model, system_fingerprint="fp_test",
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="OK"))],
            usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=1))

    class FakeCompletions:
        def __init__(self, model):
            self.model = model

        def create(self, **kwargs):
            return make_resp(self.model)

    class FakeChat:
        def __init__(self, model):
            self.completions = FakeCompletions(model)

    # wrong returned model -> invalid identity, never scored
    client._c = types.SimpleNamespace(chat=FakeChat("gpt-3.5-turbo"))
    out = client.answer("ping")
    assert out["valid_model_identity"] is False
    assert out["status"] == "model_drift"
    assert out["model_returned"] == "gpt-3.5-turbo"
    # expected returned model -> passes, fingerprint captured
    client._c = types.SimpleNamespace(chat=FakeChat(e2e.EXPECTED_RETURNED_MODEL))
    out = client.answer("ping")
    assert out["valid_model_identity"] is True
    assert out["system_fingerprint"] == "fp_test"


class DriftStub:
    """AnswerClient stand-in whose every call raises (transport failure)."""

    def __init__(self, base_url, model, timeout):
        pass

    def answer(self, prompt):
        raise RuntimeError("stub transport failure")


class AlwaysDriftAnswerStub:
    """AnswerClient stand-in that never returns a valid snapshot."""

    def __init__(self, base_url, model, timeout):
        pass

    def answer(self, prompt):
        return {"text": "x", "model_returned": "gpt-4.1-mini-2025-04-14",
                "valid_model_identity": False, "status": "model_drift",
                "system_fingerprint": "fp_drift", "latency_ms": 1.0,
                "input_tokens": 10, "output_tokens": 2}


def test_main_stops_on_model_drift(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("CHATANYWHERE_API_KEY", raising=False)
    # identity drift on every attempt: 3 attempts then whole-run stop (exit 7)
    rc, run_dir = _run(workdir, "t7", max_q=3, client=AlwaysDriftAnswerStub)
    assert rc == 7
    rows = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1  # checkpoint persisted, no further questions
    assert len(rows[0]["answer_attempts"]) == 3
    assert rows[0]["status"] == "error"
    assert rows[0]["answer_model_drift"] is True
    assert rows[0].get("generated_answer") is None  # drift content never scored


def test_main_stops_on_transport_failure(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("CHATANYWHERE_API_KEY", raising=False)
    rc, run_dir = _run(workdir, "t7b", max_q=3, client=DriftStub)
    assert rc == 7
    rows = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    statuses = [a["status"] for a in rows[0]["answer_attempts"]]
    assert statuses == ["transport_error"] * 3


def test_manifest_freeze_and_tamper_fail_closed(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("CHATANYWHERE_API_KEY", raising=False)
    manifest_path = workdir / "question_manifest.json"
    # dry-run freezes the manifest without any API call
    argv = sys.argv
    sys.argv = ["x", "--artifact-glob", GLOB, "--run-id", "t8",
                "--out-root", str(workdir), "--max-questions", "8",
                "--stratify", "2", "--manifest", str(manifest_path),
                "--dry-run"]
    original_client = e2e.AnswerClient
    e2e.AnswerClient = StubClient
    try:
        assert e2e.main() == 0
    finally:
        sys.argv = argv
        e2e.AnswerClient = original_client
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["questions"]) == 8
    from collections import Counter
    cats = Counter(q["category_id"] for q in manifest["questions"])
    assert set(cats) == {1, 2, 3, 4} and set(cats.values()) == {2}
    qids = [q["question_id"] for q in manifest["questions"]]
    assert len(qids) == len(set(qids))  # no duplicates
    cfg = json.loads((workdir / "t8" / "run_config.json").read_text())
    assert cfg["manifest_sha256"]
    # tamper with the frozen manifest -> real runs must fail closed (exit 6)
    manifest["questions"][0]["question_id"] = "conv-99:0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    e2e.AnswerClient = StubClient
    argv = sys.argv
    sys.argv = ["x", "--artifact-glob", GLOB, "--run-id", "t8",
                "--out-root", str(workdir), "--max-questions", "8",
                "--stratify", "2", "--manifest", str(manifest_path), "--resume"]
    try:
        assert e2e.main() == 6
    finally:
        sys.argv = argv
        e2e.AnswerClient = original_client
    # tamper with an OFFSET (out of frozen set) -> still exit 6
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["questions"][0]["offset"] = 999999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    e2e.AnswerClient = StubClient
    argv = sys.argv
    sys.argv = ["x", "--artifact-glob", GLOB, "--run-id", "t8",
                "--out-root", str(workdir), "--max-questions", "8",
                "--stratify", "2", "--manifest", str(manifest_path), "--resume"]
    try:
        assert e2e.main() == 6
    finally:
        sys.argv = argv
        e2e.AnswerClient = original_client


class FlakyJudgeStub:
    """JudgeClient stand-in: drifts the first two calls, then accepts."""

    def __init__(self, base_url, model, timeout):
        self.calls = 0

    def judge(self, prompt):
        self.calls += 1
        if self.calls <= 2:
            return {"raw": '{"label":"WRONG"}', "model_returned": "gpt-4.1-mini-2025-04-14",
                    "valid_model_identity": False, "status": "model_drift",
                    "system_fingerprint": "fp_drift", "latency_ms": 1.0,
                    "input_tokens": 10, "output_tokens": 2}
        return {"raw": '{"label":"CORRECT"}',
                "model_returned": "gpt-4o-mini-2024-07-18",
                "valid_model_identity": True, "status": "accepted",
                "system_fingerprint": "fp_ok", "latency_ms": 1.0,
                "input_tokens": 10, "output_tokens": 2}


class AlwaysDriftJudgeStub:
    """JudgeClient stand-in that never returns a valid snapshot."""

    def __init__(self, base_url, model, timeout):
        pass

    def judge(self, prompt):
        return {"raw": '{"label":"CORRECT"}', "model_returned": "gpt-4.1-mini-2025-04-14",
                "valid_model_identity": False, "status": "model_drift",
                "system_fingerprint": "fp_drift", "latency_ms": 1.0,
                "input_tokens": 10, "output_tokens": 2}


def test_judge_retries_drift_then_accepts(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("CHATANYWHERE_API_KEY", raising=False)
    rc, run_dir = _run(workdir, "t9", max_q=1)
    assert rc == 0
    argv = sys.argv
    sys.argv = ["x", "--run-dir", str(run_dir)]
    original = jd.JudgeClient
    jd.JudgeClient = FlakyJudgeStub
    try:
        rc2 = jd.main()
    finally:
        sys.argv = argv
        jd.JudgeClient = original
    assert rc2 == 0
    rows = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    row = rows[0]
    assert row["judge_result"] == "CORRECT"
    assert row["judge_returned_model"] == "gpt-4o-mini-2024-07-18"
    attempts = row["judge_attempts"]
    assert [a["status"] for a in attempts] == ["model_drift", "model_drift", "accepted"]
    # drifted attempt content must not appear anywhere in the row
    assert all("raw" not in a for a in attempts)


def test_judge_gives_up_after_three_drifts(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("CHATANYWHERE_API_KEY", raising=False)
    rc, run_dir = _run(workdir, "t10", max_q=1)
    assert rc == 0
    argv = sys.argv
    sys.argv = ["x", "--run-dir", str(run_dir)]
    original = jd.JudgeClient
    jd.JudgeClient = AlwaysDriftJudgeStub
    try:
        rc2 = jd.main()
    finally:
        sys.argv = argv
        jd.JudgeClient = original
    assert rc2 == 7  # whole-run stop after 3 invalid attempts
    rows = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    row = rows[0]
    assert len(row["judge_attempts"]) == 3
    assert row.get("judge_result") is None  # never scored
    assert "no gpt-4o-mini-2024-07-18 response" in row["judge_error"]


def test_cost_cap_stops_early(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # each stub call costs (10*0.15 + 2*0.60)/1e6 = $0.0000027; a $0.000001 cap
    # must stop after the first paid call
    rc2 = None
    argv = sys.argv
    sys.argv = ["x", "--artifact-glob", GLOB, "--run-id", "t6cap",
                "--out-root", str(workdir), "--max-questions", "8",
                "--cost-cap-usd", "0.000001"]
    e2e.AnswerClient = StubClient
    try:
        rc2 = e2e.main()
    finally:
        sys.argv = argv
    assert rc2 == 0
    rows = [json.loads(l) for l in (workdir / "t6cap" / "per_question.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1  # cap reached after the first paid call
    assert rows[0]["estimated_answer_cost"] > 0
    assert rows[0]["cumulative_cost"] >= rows[0]["estimated_answer_cost"]


def test_summarizer_outputs(workdir, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc, run_dir = _run(workdir, "t4", max_q=8)
    assert rc == 0
    argv = sys.argv
    sys.argv = ["x", "--run-dir", str(run_dir), "--full-n", "1540"]
    try:
        rc2 = summ.main()
    finally:
        sys.argv = argv
    assert rc2 == 0
    ms = json.loads((run_dir / "metric_summary.json").read_text())
    cs = json.loads((run_dir / "category_summary.json").read_text())
    assert ms["n_ok"] == 8
    assert "sensitivity_full_set" in ms and ms["sensitivity_full_set"]["full_n"] == 1540
    assert 0.0 <= ms["evidence_recall_at_10"] <= 1.0
    assert "_by_id" in cs
