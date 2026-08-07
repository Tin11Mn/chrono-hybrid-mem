"""Evaluate evidence retrieval on a local LoCoMo JSON file.

The dataset is intentionally supplied by path and is never copied into this
repository. Use only where the dataset licence and competition rules permit.
"""

import argparse
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import AddRequest
from app.storage import MemoryStore


def session_number(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else -1


def messages_and_evidence(sample: Dict[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    conversation = sample["conversation"]
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo conversation must be an object")
    messages: List[Dict[str, object]] = []
    evidence_text: Dict[str, str] = {}
    for session_key in sorted(
        (key for key, value in conversation.items() if isinstance(value, list)), key=session_number
    ):
        for turn in conversation[session_key]:
            if not isinstance(turn, dict) or not all(field in turn for field in ("dia_id", "speaker", "text")):
                continue
            content = "{}: {}".format(turn["speaker"], turn["text"])
            messages.append({"role": str(turn["speaker"]), "content": content, "timestamp": session_number(session_key)})
            evidence_text[str(turn["dia_id"])] = content
    return messages, evidence_text


def released_v020_search(store: MemoryStore, user_id: str, query: str, top_k: int) -> List[str]:
    """Faithfully reproduce v0.2.0's no-model raw-message retrieval path."""
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        return []
    match_query = " OR ".join('"{}"'.format(term.replace('"', '')) for term in terms)
    with store._connection() as connection:
        rows = connection.execute(
            """SELECT raw.content
               FROM messages_fts
               JOIN raw_messages AS raw ON raw.id = messages_fts.message_id
               WHERE messages_fts MATCH ? AND messages_fts.user_id = ?
               ORDER BY bm25(messages_fts), raw.id DESC
               LIMIT ?""",
            (match_query, user_id, top_k),
        ).fetchall()
    return [str(row["content"]) for row in rows]


def evaluate(samples: Iterable[Dict[str, object]], top_ks: List[int], max_questions: int | None,
             retriever: str = "current") -> Dict[str, object]:
    if retriever not in {"current", "v0.2.0"}:
        raise ValueError("retriever must be current or v0.2.0")
    hit_counts = {top_k: 0 for top_k in top_ks}
    evidence_hits = {top_k: 0 for top_k in top_ks}
    category_counts = defaultdict(lambda: {"questions": 0, "hit_at_1": 0})
    question_count = evidence_total = 0
    reciprocal_rank_sum = 0.0

    with tempfile.TemporaryDirectory(prefix="chrono-locomo-") as temporary_directory:
        for sample_index, sample in enumerate(samples):
            if max_questions is not None and question_count >= max_questions:
                break
            messages, evidence_text = messages_and_evidence(sample)
            if not messages:
                continue
            store = MemoryStore(str(Path(temporary_directory) / "sample_{}.db".format(sample_index)))
            store.initialize()
            user_id = "locomo:{}".format(sample.get("sample_id", sample_index))
            store.add(AddRequest(
                request_id="locomo:{}".format(sample_index), user_id=user_id,
                session_id="locomo-session:{}".format(sample_index), messages=messages,
            ))
            for qa in sample.get("qa", []):
                if max_questions is not None and question_count >= max_questions:
                    break
                if not isinstance(qa, dict):
                    continue
                expected = [evidence_text[item] for item in qa.get("evidence", []) if item in evidence_text]
                if not expected or not qa.get("question"):
                    continue
                question_count += 1
                evidence_total += len(expected)
                if retriever == "current":
                    results = store.search(user_id=user_id, query=str(qa["question"]), top_k=max(top_ks))
                    contents = [result.content.casefold() for result in results]
                else:
                    contents = [content.casefold() for content in released_v020_search(
                        store, user_id, str(qa["question"]), max(top_ks)
                    )]
                ranks = [
                    next((index + 1 for index, content in enumerate(contents) if item.casefold() == content), None)
                    for item in expected
                ]
                first_rank = next((rank for rank in ranks if rank is not None), None)
                category = str(qa.get("category", "unknown"))
                category_counts[category]["questions"] += 1
                if first_rank == 1:
                    category_counts[category]["hit_at_1"] += 1
                if first_rank is not None:
                    reciprocal_rank_sum += 1.0 / first_rank
                for top_k in top_ks:
                    matches = [rank for rank in ranks if rank is not None and rank <= top_k]
                    evidence_hits[top_k] += len(matches)
                    if matches:
                        hit_counts[top_k] += 1

    return {
        "questions": question_count,
        "evidence_items": evidence_total,
        "hit_at_k": {str(k): round(hit_counts[k] / question_count, 4) if question_count else 0.0 for k in top_ks},
        "evidence_recall_at_k": {str(k): round(evidence_hits[k] / evidence_total, 4) if evidence_total else 0.0 for k in top_ks},
        "mrr": round(reciprocal_rank_sum / question_count, 4) if question_count else 0.0,
        "category_hit_at_1": {
            category: round(values["hit_at_1"] / values["questions"], 4)
            for category, values in sorted(category_counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-only LoCoMo evaluation.")
    parser.add_argument("--dataset", required=True, help="Local path to LoCoMo locomo10.json")
    parser.add_argument("--top-k", default="1,3,10", help="Comma-separated retrieval depths")
    parser.add_argument("--max-questions", type=int, help="Optional cap for a smoke test")
    parser.add_argument("--compare-v020", action="store_true", help="Also reproduce v0.2.0 no-model retrieval")
    args = parser.parse_args()
    top_ks = sorted({int(value) for value in args.top_k.split(",")})
    if not top_ks or min(top_ks) < 1 or max(top_ks) > 100:
        raise ValueError("--top-k values must be between 1 and 100")
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo dataset must be a JSON array")
    current = evaluate(data, top_ks, args.max_questions)
    if not args.compare_v020:
        print(json.dumps(current, ensure_ascii=False, indent=2))
        return
    baseline = evaluate(data, top_ks, args.max_questions, retriever="v0.2.0")
    print(json.dumps({
        "scope": "retrieval-only, no external model call",
        "v0.2.0": baseline,
        "current": current,
        "delta_hit_at_1": round(current["hit_at_k"]["1"] - baseline["hit_at_k"]["1"], 4),
        "delta_mrr": round(current["mrr"] - baseline["mrr"], 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
