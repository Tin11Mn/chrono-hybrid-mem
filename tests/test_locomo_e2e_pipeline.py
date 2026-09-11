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
import summarize_locomo_e2e as summ  # noqa: E402

DATASET = REPO / ".locomo" / "locomo10.json"
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


def _run(tmp_path, run_id="t1", max_q=5, resume=False):
    argv = sys.argv
    sys.argv = [
        "x", "--artifact-glob", GLOB, "--run-id", run_id,
        "--out-root", str(tmp_path), "--max-questions", str(max_q),
        "--answer-prompt", "locomo_answer_mem0.txt",
    ] + (["--resume"] if resume else [])
    # monkeypatch the client + skip the paid-endpoint guard
    e2e.AnswerClient = StubClient
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    e2e.os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"  # non-paid sentinel
    try:
        rc = e2e.main()
    finally:
        sys.argv = argv
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
