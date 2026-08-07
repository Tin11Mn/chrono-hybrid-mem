"""Run offline evidence-retrieval metrics on a caller-supplied JSON case file."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import evaluate_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChronoHybridMem retrieval offline.")
    parser.add_argument("--cases", required=True, help="JSON array of local, permitted evaluation cases")
    parser.add_argument("--top-k", default="10,30,100", help="Comma-separated K values (1-100)")
    parser.add_argument("--database", help="Optional SQLite path; a temporary database is used by default")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("--cases must contain a JSON array")
    top_ks = sorted({int(value) for value in args.top_k.split(",")})
    with tempfile.TemporaryDirectory() as temporary_directory:
        database = args.database or str(Path(temporary_directory) / "offline_eval.db")
        report = evaluate_cases(cases, database, top_ks)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
