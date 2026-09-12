"""One-shot ChatAnywhere preflight for the LoCoMo E2E runs.

Makes exactly ONE Answer-side and ONE Judge-side request through the SAME
client classes the real runs use (no separate code path). Per formal policy:
- accepted = model_identity ∈ {snapshot_verified, alias_only_snapshot_unverified}
- alias-only is VALID for ChatAnywhere GPT-4o-mini access
- never_print/persist the API key; exit 0 = both sides pass.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_locomo_e2e as e2e  # noqa: E402
import judge_locomo_answers as jd  # noqa: E402


def main() -> int:
    if not os.environ.get(e2e.API_KEY_ENV):
        print("ERROR: {} is not set; cannot preflight.".format(e2e.API_KEY_ENV))
        return 2
    base = e2e.DEFAULT_BASE_URL
    print("preflight: gateway={} base_url={} requested_model={}".format(
        e2e.GATEWAY, base, e2e.REQUESTED_MODEL))

    # Answer preflight
    try:
        ac = e2e.AnswerClient(base, e2e.REQUESTED_MODEL, 60.0)
        out = ac.answer("Reply with exactly: OK")
    except RuntimeError as exc:
        if "evaluator_invariant" in str(exc):
            print("ANSWER PREFLIGHT FAIL (invariant): {}".format(exc))
            return 1
        print("ANSWER PREFLIGHT FAIL: {}".format(exc))
        return 1
    except Exception as exc:
        print("ANSWER PREFLIGHT FAIL: {}".format(exc))
        return 1
    answer_identity = e2e.classify_model_identity(out["model_returned"])
    answer_accepted = out["accepted"]
    print("answer: model_identity={} accepted={} returned_model={} fp={} "
          "tok={}+{} text={!r}".format(
              answer_identity, answer_accepted, out["model_returned"],
              out.get("system_fingerprint"), out["input_tokens"],
              out["output_tokens"], out["text"][:20]))

    # Judge preflight
    try:
        jc = jd.JudgeClient(base, jd.REQUESTED_MODEL, 60.0)
        jout = jc.judge(jd.load_prompt("locomo_judge_mem0.txt").format(
            question="Where does Alice work?",
            gold_answer="Microsoft",
            generated_answer="Alice works at Microsoft."))
    except Exception as exc:
        print("JUDGE PREFLIGHT FAIL: {}".format(exc))
        return 1
    judge_identity = jd.classify_model_identity(jout["model_returned"])
    judge_accepted = jout["accepted"]
    label = jd.parse_label(jout["raw"])
    print("judge: model_identity={} accepted={} returned_model={} fp={} "
          "tok={}+{} label={}".format(
              judge_identity, judge_accepted, jout["model_returned"],
              jout.get("system_fingerprint"), jout["input_tokens"],
              jout["output_tokens"], label))

    # Verdicts: per formal policy, alias-only is ACCEPTED (no snapshot
    # equality check). Both sides must be accepted, parseable, and
    # non-zero usage.
    answer_ok = answer_accepted and out["input_tokens"] > 0
    judge_ok = judge_accepted and label in ("CORRECT", "WRONG") and jout["input_tokens"] > 0
    passed = answer_ok and judge_ok
    print("preflight: {} (snapshot={}, alias_only={}, wrong_model={})".format(
        "PASS" if passed else "FAIL",
        answer_identity, judge_identity,
        "none"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
