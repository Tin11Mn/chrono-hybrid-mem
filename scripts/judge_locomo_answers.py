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

# Gateway configuration: ChatAnywhere-mediated GPT-4o-mini-2024-07-18 access,
# identical to the Answer path. Request "gpt-4o-mini"; every successful
# response must report model == EXPECTED_RETURNED_MODEL or the run fails closed.
GATEWAY = "chatanywhere"
DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"
API_KEY_ENV = "CHATANYWHERE_API_KEY"
REQUESTED_MODEL = "gpt-4o-mini"
EXPECTED_RETURNED_MODEL = "gpt-4o-mini-2024-07-18"
# Identity-validated retry budget per question (frozen 2026-09-12): a judge
# response is accepted only when returned_model == EXPECTED_RETURNED_MODEL;
# after 3 failed attempts for one question the whole run stops (NO-GO signal).
MAX_JUDGE_ATTEMPTS = 3


class ModelDriftError(RuntimeError):
    """The gateway returned a model id other than EXPECTED_RETURNED_MODEL."""


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
            api_key=os.environ.get(API_KEY_ENV, "EMPTY"),
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
        returned_model = getattr(resp, "model", None)
        # Identity gate: the response is VALID only when the gateway returned
        # the frozen snapshot id. Invalid responses are reported via
        # valid_model_identity/status; the caller owns the per-question retry
        # policy and must never score them.
        valid = returned_model == EXPECTED_RETURNED_MODEL
        choice = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return {
            "raw": choice,
            "model_returned": returned_model,
            "valid_model_identity": valid,
            "status": "accepted" if valid else "model_drift",
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
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


def save_raw_output(run_dir: Path, question_id: str, phase: str,
                    payload: dict) -> None:
    """Persist one raw model response under raw_model_outputs/."""
    raw_dir = run_dir / "raw_model_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_id = question_id.replace(":", "_")
    (raw_dir / "{}.{}.json".format(safe_id, phase)).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_checkpoint(run_dir: Path, per_q_path: Path, rows: list[dict]) -> None:
    """Refresh checkpoint.json from the current per_question.jsonl state."""
    checkpoint = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "answered_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "answered_error": [r["question_id"] for r in rows
                           if r.get("status") != "ok"],
        "judged": sum(1 for r in rows
                      if r.get("judge_result") in ("CORRECT", "WRONG")),
        "spend_estimated_usd": round(sum(
            (r.get("estimated_answer_cost") or 0.0)
            + (r.get("estimated_judge_cost") or 0.0) for r in rows), 6),
        "per_question_path": str(per_q_path),
    }
    write_jsonl_atomic(run_dir / "checkpoint.json", [checkpoint])


def main() -> int:
    ap = argparse.ArgumentParser(description="Binary judge for LoCoMo answers.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge-model", default=REQUESTED_MODEL)
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
        "OPENAI_BASE_URL", DEFAULT_BASE_URL)

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

    if (not os.environ.get(API_KEY_ENV)
            and not base_url.startswith(("http://127.0.0.1:", "http://localhost:"))):
        print("ERROR: {} not set; refusing to call the paid endpoint {}.".format(
            API_KEY_ENV, base_url))
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
        # A retried row (e.g. after a model-drift stop) must reflect only the
        # FINAL accepted attempt; stale fields from aborted attempts are
        # cleared here. The drift history itself is preserved below in
        # judge_attempts and in the run log.
        for key in ("judge_error", "judge_model_drift", "judge_parse_error",
                    "judge_result", "judge_raw_response", "judge_input_tokens",
                    "judge_output_tokens", "estimated_judge_cost",
                    "judge_system_fingerprint", "judge_returned_model",
                    "judge_latency_ms"):
            row.pop(key, None)
        prompt = judge_prompt.format(
            question=row.get("question", ""),
            gold_answer=row.get("reference_answer", ""),
            generated_answer=row.get("generated_answer", ""),
        )
        # Identity-validated retry: identical inputs every attempt; the ONLY
        # retry triggers are returned-model identity mismatch or a transport
        # failure, never the judge content. Max 3 attempts, then STOP.
        attempts = []
        accepted = None
        for attempt_no in range(1, MAX_JUDGE_ATTEMPTS + 1):
            try:
                out = client.judge(prompt)
            except Exception as exc:
                attempts.append({
                    "attempt": attempt_no,
                    "requested_model": args.judge_model,
                    "returned_model": None,
                    "system_fingerprint": None,
                    "valid_model_identity": False,
                    "status": "transport_error",
                    "error": str(exc)[:200],
                })
                continue
            attempts.append({
                "attempt": attempt_no,
                "requested_model": args.judge_model,
                "returned_model": out.get("model_returned"),
                "system_fingerprint": out.get("system_fingerprint"),
                "valid_model_identity": bool(out.get("valid_model_identity")),
                "status": out.get("status"),
                "input_tokens": out.get("input_tokens", 0),
                "output_tokens": out.get("output_tokens", 0),
            })
            answer_spent += (out.get("input_tokens", 0) * args.price_in_per_1m
                             + out.get("output_tokens", 0) * args.price_out_per_1m) / 1e6
            if out.get("valid_model_identity") is True:
                accepted = out
                break
        row["judge_attempts"] = attempts
        drift_cost = sum(
            (a.get("input_tokens") or 0) * args.price_in_per_1m
            + (a.get("output_tokens") or 0) * args.price_out_per_1m
            for a in attempts if a.get("status") != "accepted") / 1e6
        if drift_cost:
            row["judge_drift_cost"] = round(drift_cost, 6)
        if accepted is None:
            # 3 attempts without a single valid snapshot -> NO-GO signal.
            row["judge_error"] = (
                "no {} response after {} attempts ({})".format(
                    EXPECTED_RETURNED_MODEL, len(attempts),
                    ",".join(a["status"] for a in attempts)))
            write_jsonl_atomic(per_q_path, rows)
            write_checkpoint(run_dir, per_q_path, rows)
            print("JUDGE GIVE-UP: {} after {} attempts ({}); checkpoint "
                  "persisted; STOPPING WHOLE RUN.".format(
                      row["question_id"], len(attempts),
                      ",".join(a["status"] for a in attempts)))
            return 7
        # Only the accepted attempt feeds any field used by metrics; drifted
        # response content is never persisted to the row.
        label = parse_label(accepted["raw"])
        judge_cost = (accepted["input_tokens"] * args.price_in_per_1m
                      + accepted["output_tokens"] * args.price_out_per_1m) / 1e6
        row["judge_result"] = label
        row["judge_raw_response"] = accepted["raw"]
        row["judge_model"] = args.judge_model
        row["judge_model_requested"] = args.judge_model
        row["judge_model_returned"] = accepted["model_returned"]
        row["judge_returned_model"] = accepted["model_returned"]
        row["judge_gateway"] = GATEWAY
        row["judge_api_base_url"] = base_url
        row["judge_temperature"] = 0.0
        row["judge_system_fingerprint"] = accepted.get("system_fingerprint")
        row["judge_prompt_version"] = "mem0.v1"
        row["judge_prompt_hash"] = judge_prompt_hash
        row["judge_latency_ms"] = round(accepted["latency_ms"], 1)
        row["judge_input_tokens"] = accepted["input_tokens"]
        row["judge_output_tokens"] = accepted["output_tokens"]
        row["estimated_judge_cost"] = round(judge_cost, 6)
        row["cumulative_cost"] = round(answer_spent, 6)
        if label is None:
            row["judge_parse_error"] = True
        save_raw_output(run_dir, row["question_id"], "judge", {
            "question_id": row["question_id"],
            "phase": "judge",
            "requested_model": args.judge_model,
            "returned_model": row.get("judge_returned_model"),
            "system_fingerprint": row.get("judge_system_fingerprint"),
            "temperature": row.get("judge_temperature"),
            "raw": row.get("judge_raw_response"),
            "parsed_label": row.get("judge_result"),
            "input_tokens": row.get("judge_input_tokens"),
            "output_tokens": row.get("judge_output_tokens"),
            "latency_ms": row.get("judge_latency_ms"),
            "judge_error": row.get("judge_error"),
            "judge_attempts": attempts,
        })
        done += 1
        if done % 25 == 0 or done == len(todo):
            write_jsonl_atomic(per_q_path, rows)
            write_checkpoint(run_dir, per_q_path, rows)
            print("  judged {}/{}".format(done, len(todo)))
    write_jsonl_atomic(per_q_path, rows)
    write_checkpoint(run_dir, per_q_path, rows)
    judged = sum(1 for r in rows if r.get("judge_result") == "CORRECT")
    total = sum(1 for r in rows if r.get("judge_result") in {"CORRECT", "WRONG"})
    print("done. judged_total={} correct={} acc={}".format(
        total, judged, round(judged / total, 4) if total else "n/a"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
