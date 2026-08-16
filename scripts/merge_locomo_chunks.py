"""Merge aggregate-only LoCoMo evaluation chunks into one result."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.chunks]
    if not records:
        raise SystemExit("no chunk results")
    expected = sum(int(record["questions"]) for record in records)
    top_ks = [str(key) for key in records[0]["hit_at_k"]]
    category_names = sorted({name for record in records for name in record["category_hit_at_k"]})

    def total(path):
        return sum(float(record[path]) * int(record["questions"]) for record in records)

    raw_hit = {
        key: sum(int(record["raw_counts"]["hit_counts"][key]) for record in records)
        for key in top_ks
    }
    raw_evidence = {
        key: sum(int(record["raw_counts"]["evidence_hit_counts"][key]) for record in records)
        for key in top_ks
    }
    category_counts = {}
    for category in category_names:
        category_counts[category] = {
            "questions": sum(
                int(record["raw_counts"]["category_counts"].get(category, {}).get("questions", 0))
                for record in records
            ),
            "hit_at_1": sum(
                int(record["raw_counts"]["category_counts"].get(category, {}).get("hit_at_1", 0))
                for record in records
            ),
            "hit_counts": {
                key: sum(
                    int(record["raw_counts"]["category_counts"].get(category, {}).get("hit_counts", {}).get(key, 0))
                    for record in records
                )
                for key in top_ks
            },
        }

    category_hit = {}
    for category, values in category_counts.items():
        count = values["questions"]
        category_hit[category] = {
            key: round(values["hit_counts"][key] / count, 4) if count else 0.0
            for key in top_ks
        }

    result = {
        "questions": expected,
        "evidence_items": sum(int(record.get("evidence_items", 0)) for record in records),
        "hit_at_k": {key: round(raw_hit[key] / expected, 4) for key in top_ks},
        "evidence_recall_at_k": {key: round(raw_evidence[key] / sum(int(record.get("evidence_items", 0)) for record in records), 4) for key in top_ks},
        "mrr": round(sum(float(record["raw_counts"]["reciprocal_rank_sum"]) for record in records) / expected, 4),
        "category_hit_at_1": {category: values["hit_at_1"] / values["questions"] if values["questions"] else 0.0 for category, values in category_counts.items()},
        "category_hit_at_k": category_hit,
        "raw_counts": {
            "hit_counts": raw_hit,
            "evidence_hit_counts": raw_evidence,
            "reciprocal_rank_sum": sum(float(record["raw_counts"]["reciprocal_rank_sum"]) for record in records),
            "category_counts": category_counts,
        },
        "source_chunks": [str(path) for path in args.chunks],
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
