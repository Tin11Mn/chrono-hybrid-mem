"""Compare the released v0.2.0 retriever with the current checkout.

The supplied diagnostic file must contain only fictional or otherwise permitted
data. This is a regression diagnostic, not an estimate of the challenge's
hidden-test score.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_evaluator(source_root: Path, cases: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(source_root / "scripts" / "evaluate_retrieval.py"),
            "--cases",
            str(cases),
            "--top-k",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v0.2.0 with the current retriever.")
    parser.add_argument("--cases", required=True, help="Permitted fictional diagnostic JSON")
    args = parser.parse_args()
    cases = Path(args.cases).resolve()

    with tempfile.TemporaryDirectory(prefix="chrono-v020-") as temporary_directory:
        baseline_root = Path(temporary_directory) / "v0.2.0"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(baseline_root), "v0.2.0"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            baseline = run_evaluator(baseline_root, cases)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(baseline_root)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

    current = run_evaluator(ROOT, cases)
    report = {
        "dataset": cases.name,
        "scope": "fictional diagnostic only; not the competition hidden test set",
        "top_k": 1,
        "v0.2.0": baseline,
        "current": current,
        "delta_recall_at_1": round(
            current["recall_at_k"]["1"] - baseline["recall_at_k"]["1"], 4
        ),
        "delta_mrr": round(current["mrr"] - baseline["mrr"], 4),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
