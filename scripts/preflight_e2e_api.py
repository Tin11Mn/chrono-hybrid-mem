"""One-shot ChatAnywhere preflight for the LoCoMo E2E runs.

Makes exactly ONE Answer-side and ONE Judge-side request through the SAME
client classes the real runs use (no separate code path). Verifies:

- authentication succeeds (non-empty usage metadata back),
- requested model == "gpt-4o-mini",
- returned model == "gpt-4o-mini-2024-07-18" (the AnswerClient/JudgeClient
  drift gate itself enforces this and raises ModelDriftError),
- judge reply parses to a CORRECT/WRONG label.

Never prints or persists the API key. Exit 0 = both sides pass.
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

    try:
        ac = e2e.AnswerClient(base, e2e.REQUESTED_MODEL, 60.0)
        out = ac.answer("Reply with exactly: OK")
    except e2e.ModelDriftError as exc:
        print("ANSWER PREFLIGHT FAIL (model drift): {}".format(exc))
        return 1
    except Exception as exc:
        print("ANSWER PREFLIGHT FAIL: {}".format(exc))
        return 1
    print("answer: returned_model={} fingerprint={} usage={}+{} tok text={!r}".format(
        out["model_returned"], out.get("system_fingerprint"),
        out["input_tokens"], out["output_tokens"], out["text"][:20]))

    try:
        jc = jd.JudgeClient(base, jd.REQUESTED_MODEL, 60.0)
        jout = jc.judge(jd.load_prompt("locomo_judge_mem0.txt").format(
            question="Where does Alice work?",
            gold_answer="Microsoft",
            generated_answer="Alice works at Microsoft."))
    except Exception as exc:
        print("JUDGE PREFLIGHT FAIL: {}".format(exc))
        return 1
    label = jd.parse_label(jout["raw"])
    print("judge: returned_model={} fingerprint={} usage={}+{} tok label={}".format(
        jout["model_returned"], jout.get("system_fingerprint"),
        jout["input_tokens"], jout["output_tokens"], label))

    usage_ok = out["input_tokens"] > 0 and jout["input_tokens"] > 0
    label_ok = label in ("CORRECT", "WRONG")
    identity_ok = (out.get("valid_model_identity") is True
                   and jout.get("valid_model_identity") is True)
    passed = usage_ok and label_ok and identity_ok
    print("preflight: {}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
