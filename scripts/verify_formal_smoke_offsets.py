"""Verify the formal Smoke-20 selection reproduces the frozen 20 offsets.

Frozen Smoke-20 offsets (by manifest order across the eligible set):
  [0,1,2,3,4,5,6,7,8,11,13,14,22,27,40,79,80,81,82,83]
(4 categories x 5, deterministic offset prefix per category, blind to
answers/retrieval outcomes/judge results).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_locomo_e2e as e2e  # noqa: E402

FROZEN_OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 22, 27, 40,
                  79, 80, 81, 82, 83]


def frozen_smoke_offsets() -> list[int]:
    elr = e2e._load_elr()
    samples = json.load(open(e2e.DEFAULT_DATASET, encoding="utf-8"))
    index, _ = e2e.build_question_index(samples, elr)
    frozen = e2e.load_frozen_diags(str(e2e.P3_LOCOMO / "sfv2-full-*.json"))
    selected = []
    remaining = {cat: 5 for cat in (1, 2, 3, 4)}
    for offset in sorted(frozen):
        meta = index[offset]
        try:
            cat = int(meta["category"])
        except Exception:
            cat = None
        if cat == 5 or cat not in remaining or remaining[cat] <= 0:
            continue
        remaining[cat] -= 1
        selected.append(offset)
        if len(selected) >= 20:
            break
    return selected


def main() -> int:
    selected = frozen_smoke_offsets()
    ok = selected == FROZEN_OFFSETS
    print("frozen expected: {}".format(FROZEN_OFFSETS))
    print("actual selected: {}".format(selected))
    print("exact match: {}".format(ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())