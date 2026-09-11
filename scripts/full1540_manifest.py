"""Generate and freeze the official Full-1540 question manifest.

Full-1540 = every LoCoMo qa entry with category != 5, in dataset order
(natural distribution: multi-hop 282 / temporal 321 / open-domain 96 /
single-hop 841 = 1540). No sampling, no replacement, no result-based
selection of any kind.

Each entry carries:
- offset            position in this manifest (0..1539) — the E2E key space
- question_id       "sample_id:qa_index" (stable identity, resume key)
- conversation_id / category_id / category_name
- retrieval_offset  position in the ELIGIBLE-question index (0..1976) that
                    the frozen retrieval artifacts are keyed by, or null for
                    questions that were never retrieval-eligible (their gold
                    evidence list does not resolve to raw messages)
- evidence_resolvable / answer_present / frozen_retrieval_available

The manifest hash is the SHA256 of the file text (normalized newlines), also
written into the summary. The three methods MUST share this file verbatim.

Usage:
    python scripts/full1540_manifest.py [--out results/locomo_e2e/full1540/question_manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_locomo_e2e as e2e  # noqa: E402

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
EXPECTED_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "results" / "locomo_e2e"
                                         / "full1540" / "question_manifest.json"))
    args = ap.parse_args()

    elr = e2e._load_elr()
    samples = json.load(open(e2e.DEFAULT_DATASET, encoding="utf-8"))
    # Eligible-index offsets (the frozen-artifact key space), by question_id.
    idx, per_sample = e2e.build_question_index(samples, elr)
    eligible_offset_by_qid: dict[str, int] = {}
    if isinstance(idx, list):
        for pos, entry in enumerate(idx):
            qid = "{}:{}".format(entry["sample_id"], entry["qa_index"])
            eligible_offset_by_qid[qid] = pos
    else:
        eligible_offset_by_qid = {
            "{}:{}".format(v["sample_id"], v["qa_index"]): k
            for k, v in idx.items()}

    questions = []
    counts: dict[int, int] = {}
    answer_missing = []
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        smap = per_sample.get(sample_id)
        for qa_index, qa in enumerate(sample.get("qa", [])):
            if not isinstance(qa, dict):
                continue
            try:
                cat = int(qa.get("category"))
            except (TypeError, ValueError):
                continue
            if cat == 5:
                continue
            qid = "{}:{}".format(sample_id, qa_index)
            entry = {
                "offset": len(questions),
                "question_id": qid,
                "conversation_id": sample_id,
                "category_id": cat,
                "category_name": CATEGORY_NAMES.get(cat, str(cat)),
                "retrieval_offset": eligible_offset_by_qid.get(qid),
                "evidence_resolvable": qid in eligible_offset_by_qid,
                "answer_present": bool(qa.get("answer")),
                "question_present": bool(qa.get("question")),
            }
            if not entry["answer_present"]:
                answer_missing.append(qid)
            questions.append(entry)
            counts[cat] = counts.get(cat, 0) + 1

    counts_ok = counts == EXPECTED_COUNTS
    manifest = {
        "run_id": "full1540",
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": str(e2e.DEFAULT_DATASET),
        "dataset_sha256": hashlib.sha256(
            Path(e2e.DEFAULT_DATASET).read_bytes()).hexdigest(),
        "selection_rule": (
            "ALL LoCoMo qa entries with category != 5 in dataset order "
            "(natural distribution); no sampling, no replacement, no "
            "result-based selection; shared verbatim by p1 / p4a_bm25 / "
            "sfv2_4b"),
        "expected_category_counts": EXPECTED_COUNTS,
        "questions": questions,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    out_path.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(
        out_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()

    print("manifest written: {}".format(out_path))
    print("total questions: {} (expected 1540)".format(len(questions)))
    print("category counts: {} match={}".format(
        dict(sorted(counts.items())), counts_ok))
    print("unique question_ids: {}".format(
        len({q['question_id'] for q in questions}) == len(questions)))
    print("with frozen retrieval (retrieval_offset set): {}".format(
        sum(1 for q in questions if q["retrieval_offset"] is not None)))
    print("evidence-unresolvable (no retrieval_offset): {}".format(
        sum(1 for q in questions if not q["evidence_resolvable"])))
    print("answer missing: {}".format(len(answer_missing)))
    print("manifest_sha256: {}".format(sha))
    return 0 if (counts_ok and len(questions) == 1540
                 and not answer_missing) else 1


if __name__ == "__main__":
    sys.exit(main())
