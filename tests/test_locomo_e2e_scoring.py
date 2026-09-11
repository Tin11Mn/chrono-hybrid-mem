"""Parity + exactness tests for scripts/score_locomo_answers.py.

Two layers:
1. Exactness vs the official LoCoMo evaluator (imported with light stubs).
2. Parity vs the published frozen SF v2 artifact: re-derive Hit@1/3/10, MRR and
   pooled Evidence Recall@10 from result_ids + gold_mem_ids and require an exact
   match to the numbers in docs/CHRONOHYBRIDMEM_RESULTS_FOR_PAPER.md.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import score_locomo_answers as sc  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Official evaluator exactness                                              #
# --------------------------------------------------------------------------- #
def _load_official():
    off = REPO.parent / "_audit-src" / "locomo-official" / "task_eval"
    if not off.exists():
        return None
    for m in ("bert_score", "rouge"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["rouge"].Rouge = lambda: None
    sys.modules["bert_score"].score = lambda *a, **k: None
    # Import under a unique module name: a plain `import evaluation` would
    # collide with the repo's `evaluation` namespace package if another test
    # module (e.g. test_graph_paired_evaluation) already cached it.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "locomo_official_evaluation", str(off / "evaluation.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not (REPO.parent / "_audit-src" / "locomo-official").exists(),
    reason="official evaluator clone not present",
)
def test_official_f1_matches_reference():
    e = _load_official()
    cases = [
        ("Adoption agencies", "Adoption agencies"),
        ("the cat and the dog", "cat dog"),
        ("A, B, C", "A, B, C"),
        ("A, B", "A, B, C"),
        ("Paris, France", "Paris"),
        ("", "something"),
    ]
    for pred, gold in cases:
        assert sc._official_f1_score(pred, gold) == pytest.approx(
            e.f1_score(pred, gold), abs=1e-12)
    assert sc._official_f1_multi("A, B, C", "A, B, C") == pytest.approx(
        e.f1("A, B, C", "A, B, C"), abs=1e-12)
    assert sc._official_f1_multi("A, B", "A, B, C") == pytest.approx(
        e.f1("A, B", "A, B, C"), abs=1e-12)
    assert sc._official_normalize("the cat and the dog") == e.normalize_answer(
        "the cat and the dog")


def test_official_f1_known_values():
    # Whole-answer cat4 exact match.
    assert sc.f1_official("Adoption agencies", "Adoption agencies", 4) == 1.0
    # cat1 multi-answer is per-gold best-match, NOT a plain multiset over the
    # whole string; with single-token preds, gold "C" never matches.
    assert sc.f1_official("A, B, C", "A, B, C", 1) == pytest.approx(2 / 3)
    # 'and'/'the' are dropped -> pred collapses to gold.
    assert sc.f1_official("the cat and the dog", "cat dog", 4) == 1.0
    # cat3 ';'-truncation
    assert sc.f1_official("x", "x; trailing", 3) == 1.0


def test_lineage_f1s_are_set_based_and_category_agnostic():
    p, g = "the cat and the dog", "cat dog"
    # Mem0/MemoryART/MemoryOS keep 'and' -> set F1 != 1.0
    for fn in (sc.f1_mem0, sc.f1_memoryart, sc.f1_memoryos):
        v = fn(p, g)
        assert 0.0 < v < 1.0
    # exact-match collapses to 1.0 for all
    for fn in (sc.f1_mem0, sc.f1_memoryart, sc.f1_memoryos):
        assert fn("yes", "yes") == 1.0


def test_bleu1_variants_and_empty():
    assert sc.bleu1_m1("adoption agencies", "adoption agencies") == pytest.approx(1.0)
    assert sc.bleu1_m4("adoption agencies", "adoption agencies") == pytest.approx(1.0)
    assert sc.bleu1_m1("", "x") == 0.0
    # partial unigram overlap < 1.0
    assert 0.0 < sc.bleu1_m1("adoption", "adoption agencies") < 1.0


def test_score_answer_shape():
    out = sc.score_answer("yes", "yes", 4)
    assert set(out) == {
        "f1_official", "f1_mem0", "f1_memoryart", "f1_memoryos",
        "bleu1_m1", "bleu1_m4",
    }
    assert all(0.0 <= v <= 1.0 for v in out.values())


# --------------------------------------------------------------------------- #
# 2. Frozen-artifact parity                                                    #
# --------------------------------------------------------------------------- #
def _frozen_chunk_paths() -> list[str]:
    p3 = REPO.parent / "chrono-hybrid-mem-p3" / ".locomo"
    return sorted(glob.glob(str(p3 / "sfv2-full-*.json")))


@pytest.mark.skipif(not _frozen_chunk_paths(), reason="frozen SF v2 artifacts absent")
def test_parity_with_published_sfv2_numbers():
    seen = {}
    for f in _frozen_chunk_paths():
        for qd in json.load(open(f, encoding="utf-8"))["question_diagnostics"]:
            seen.setdefault(qd["question_offset"], qd)
    perq = [sc.evidence_metrics(qd["result_ids"], qd["gold_mem_ids"])
            for qd in seen.values()]
    agg = sc.aggregate_evidence(perq)
    assert agg["n"] == 1976
    assert agg["hit_at_1"] == pytest.approx(0.6108, abs=5e-5)
    assert agg["hit_at_3"] == pytest.approx(0.7677, abs=5e-5)
    assert agg["hit_at_10"] == pytest.approx(0.8219, abs=5e-5)
    assert agg["mrr"] == pytest.approx(0.6929, abs=5e-5)
    assert agg["evidence_recall_at_10"] == pytest.approx(0.6555, abs=5e-5)
