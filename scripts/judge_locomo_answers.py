"""Binary CORRECT/WRONG judge for LoCoMo generated answers.

Reads a run's per_question.jsonl (produced by evaluate_locomo_e2e.py) and fills
the judge_* fields for every question that has a generated_answer but no
judge_result yet. Resumable: completed questions are never re-judged, and the
run's config digest is enforced so a prompt/model change cannot silently mix
judge results.

The judge model is a separate [OI]-compatible endpoint; the gold answer appears
ONLY here (never in retrieval or answer generation), satisfying the leakage
constraint. Default model is gpt-4o-mini-2024-07-18 (user decision #6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent  # chrono-hybrid-mem/
sys.path.insert(0, str(REPO / "scripts"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_prompt(name: str) -> str:
    return (REPO / "prompts" / name).read_text(encoding="utf-8")


def extract_json(text: str) -> str:
    """Mem0-style extract_json: pull the first {...} block from the reply."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def parse_label(raw: str) -> str | None:
    try:
        label = json.loads(extract_json(raw)).get("label", "")
    except Exception:
        return None
    label = str(label).strip().upper()
    return label if label in {"CORRECT", "WRONG"} else None


class JudgeClient:
    """Thin [OI]-compatible chat client. Constructs lazily so --dry-run works
    without the SDK or an endpoint."""

    def __init__(self, base_url: str | None, model: str, timeout: float):
        from openai import OpenAI  # local import; only needed when actually judging
        self._client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=0,
        )
        self.model = model

    def judge(self, prompt: str) -> dict:
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        latency = (time.time() - t0) * 1000
        choice = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return {
            "raw": choice,
            "model_returned": getattr(resp, "model", None),
            "latency_ms": latency,
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Binary judge for LoCoMo answers.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--judge-base-url", default=None,
                    help="[OI]-compatible endpoint; default uses OPENAI_BASE_URL/api.openai.com")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--price-in-per-1m", type=float, default=0.15)
    ap.add_argument("--price-out-per-1m", type=float, default=0.60)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print how many would be judged; no API calls.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per_q_path = run_dir / "per_question.jsonl"
    cfg_path = run_dir / "run_config.json"
    if not per_q_path.exists():
        print("ERROR: per_question.jsonl not found in {}".format(run_dir))
        return 2

    rows = read_jsonl(per_q_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    judge_prompt = load_prompt("locomo_judge_mem0.txt")
    judge_prompt_hash = sha256_text(judge_prompt)

    # Fail-closed: enforce judge prompt/model consistency with the frozen config.
    if cfg:
        exp_hash = cfg.get("judge_prompt_hash")
        exp_model = cfg.get("judge_model")
        if exp_hash and exp_hash != judge_prompt_hash:
            print("ERROR: judge prompt hash changed ({} -> {}). New run-dir required."
                  .format(exp_hash, judge_prompt_hash))
            return 3
        if exp_model and exp_model != args.judge_model:
            print("ERROR: judge model changed ({} -> {}). New run-dir required."
                  .format(exp_model, args.judge_model))
            return 3

    base_url = args.judge_base_url or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1")

    todo = [r for r in rows
            if r.get("status") == "ok"
            and r.get("generated_answer")
            and not r.get("judge_result")]
    if args.max_questions is not None:
        todo = todo[: args.max_questions]
    print("run={} total={} to_judge={} model={} endpoint={}".format(
        cfg.get("run_id", run_dir.name), len(rows), len(todo),
        args.judge_model, base_url))
    if args.dry_run:
        print("dry-run: no API calls.")
        return 0

    if not os.environ.get("OPENAI_API_KEY") and "api.openai.com" in base_url:
        print("ERROR: OPENAI_API_KEY not set; refusing to call the paid endpoint.")
        return 4

    client = JudgeClient(base_url, args.judge_model, args.timeout)
    done = 0
    answer_spent = sum(
        r.get("estimated_answer_cost", 0.0) or 0.0
        for r in rows if r.get("status") == "ok"
    )
    for row in rows:
        if row not in todo:
            # still ensure judge_prompt metadata present on already-judged rows
            continue
        prompt = judge_prompt.format(
            question=row.get("question", ""),
            gold_answer=row.get("reference_answer", ""),
            generated_answer=row.get("generated_answer", ""),
        )
        try:
            out = client.judge(prompt)
            label = parse_label(out["raw"])
            judge_cost = (out["input_tokens"] * args.price_in_per_1m
                          + out["output_tokens"] * args.price_out_per_1m) / 1e6
            answer_spent += judge_cost
            row["judge_result"] = label
            row["judge_raw_response"] = out["raw"]
            row["judge_model"] = args.judge_model
            row["judge_model_requested"] = args.judge_model
            row["judge_model_returned"] = out["model_returned"]
            row["judge_prompt_version"] = "mem0.v1"
            row["judge_prompt_hash"] = judge_prompt_hash
            row["judge_latency_ms"] = round(out["latency_ms"], 1)
            row["judge_input_tokens"] = out["input_tokens"]
            row["judge_output_tokens"] = out["output_tokens"]
            row["estimated_judge_cost"] = round(judge_cost, 6)
            row["cumulative_cost"] = round(answer_spent, 6)
            if label is None:
                row["judge_parse_error"] = True
        except Exception as exc:  # record, keep going; resumable
            row["judge_error"] = str(exc)[:200]
        done += 1
        if done % 25 == 0 or done == len(todo):
            write_jsonl_atomic(per_q_path, rows)
            print("  judged {}/{}".format(done, len(todo)))
    write_jsonl_atomic(per_q_path, rows)
    judged = sum(1 for r in rows if r.get("judge_result") == "CORRECT")
    total = sum(1 for r in rows if r.get("judge_result") in {"CORRECT", "WRONG"})
    print("done. judged_total={} correct={} acc={}".format(
        total, judged, round(judged / total, 4) if total else "n/a"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
