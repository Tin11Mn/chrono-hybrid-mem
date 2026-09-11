"""LoCoMo end-to-end QA harness for ChronoHybridMem (frozen-retrieval path).

Per user directive, this harness does NOT re-run retrieval. It reads the frozen
1976-question retrieval artifacts (produced by scripts/evaluate_locomo_retrieval.py
with --include-question-diagnostics), reconstructs the original evidence text from
mem IDs (verified: mem_N == Nth message of sessions_and_evidence(sample), 1-based),
filters to the non-adversarial set (category != 5), renders the Mem0 answer prompt,
calls an [OI]-compatible answer model, and writes a resumable per_question.jsonl.

Leakage controls (enforced):
- Answer prompt contains ONLY retrieved original evidence + timestamps + question.
- qa.answer / qa.adversarial_answer / qa.evidence / observation / session_summary /
  event_summary never enter the answer prompt (gold appears only in the judge).
- category is used ONLY to select the evaluation subset (cat != 5), never for routing.

Checkpoint/resume: per-question append + fsync; a run_config.json with a
config_digest fails closed if the frozen artifact set, prompt, or model changes.

This module is import-safe (no dataset load, no LLM) until main() runs.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent  # chrono-hybrid-mem/
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "scripts"))
import score_locomo_answers as sc  # noqa: E402

DEFAULT_DATASET = WORKSPACE / "chrono-hybrid-mem" / ".locomo" / "locomo10.json"
P3_LOCOMO = WORKSPACE / "chrono-hybrid-mem-p3" / ".locomo"
ELR_PATH = REPO / "scripts" / "evaluate_locomo_retrieval.py"

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_elr():
    """Import evaluate_locomo_retrieval for sessions_and_evidence (read-only)."""
    spec = importlib.util.spec_from_file_location("elr", ELR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reconstruct_mem_content(sample, elr) -> dict:
    """mem_N -> content for one conversation (1-based, Add order)."""
    sessions, _ = elr.sessions_and_evidence(sample)
    mem2content = {}
    n = 0
    for _sk, msgs in sessions:
        for m in msgs:
            n += 1
            mem2content["mem_{}".format(n)] = m["content"]
    return mem2content, sessions


def session_timestamps(sample, sessions) -> dict:
    """Map each message content -> its session timestamp (for prompt rendering)."""
    # We attach timestamps by replaying sessions_and_evidence order and zipping
    # with the session keys' timestamps via the conversation structure.
    elr = _load_elr()
    conv = sample["conversation"]
    mem2ts = {}
    n = 0
    import re
    def sessno(k):
        m = re.search(r"(\d+)", str(k))
        return int(m.group(1)) if m else 0
    for session_key in sorted((k for k, v in conv.items() if isinstance(v, list)),
                              key=sessno):
        ts = elr.session_timestamp(conv, session_key)
        for _turn in conv[session_key]:
            n += 1
            mem2ts["mem_{}".format(n)] = ts
    return mem2ts


def build_question_index(samples, elr):
    """Global eligible-question index -> (sample, qa, expected_texts, mem2content).

    Mirrors evaluate_locomo_retrieval.py exactly: a question is eligible iff it has
    a question AND at least one resolvable evidence item. Returns a list in the same
    order the frozen artifacts were produced, plus per-conversation mem maps.
    """
    index = []
    per_sample = {}
    for sample in samples:
        sessions, evidence_text = elr.sessions_and_evidence(sample)
        if not sessions:
            continue
        mem2content, _ = reconstruct_mem_content(sample, elr)
        mem2ts = session_timestamps(sample, sessions)
        per_sample[sample.get("sample_id")] = {
            "mem2content": mem2content, "mem2ts": mem2ts,
            "evidence_text": evidence_text,
        }
        for qa_index, qa in enumerate(sample.get("qa", [])):
            if not isinstance(qa, dict):
                continue
            expected = [evidence_text[i] for i in qa.get("evidence", []) if i in evidence_text]
            if not expected or not qa.get("question"):
                continue
            index.append({
                "sample_id": sample.get("sample_id"),
                "qa_index": qa_index,
                "question": qa["question"],
                "category": qa.get("category"),
                "answer": qa.get("answer"),
                "adversarial_answer": qa.get("adversarial_answer"),
                "expected": expected,
            })
    return index, per_sample


def load_frozen_diags(pattern: str) -> dict:
    """offset -> question_diagnostic, dedup keep-first across chunk files."""
    seen = {}
    for f in sorted(glob.glob(pattern)):
        data = json.load(open(f, encoding="utf-8"))
        for qd in data.get("question_diagnostics", []):
            seen.setdefault(qd["question_offset"], qd)
    return seen


def config_digest(artifact_files, dataset_path, answer_prompt_hash, answer_model,
                  profile, top_k):
    payload = {
        "artifact_sha256": [sha256_file(Path(f)) for f in sorted(artifact_files)],
        "dataset_sha256": sha256_file(Path(dataset_path)),
        "answer_prompt_hash": answer_prompt_hash,
        "answer_model": answer_model,
        "profile": profile,
        "top_k_evidence": top_k,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def render_answer_prompt(template, mem_entries, speakers, question):
    """Fill the Mem0 Jinja-style ANSWER_PROMPT.

    mem_entries: list of (mem_id, content, timestamp). We split by speaker name
    (the part before ':' in content) into speaker_1/speaker_2 memories, matching
    Mem0's per-speaker rendering (json.dumps list of "<ts>: <memory>").
    """
    by_speaker = {}
    order = []
    for _mid, content, ts in mem_entries:
        sp = content.split(":", 1)[0].strip() if ":" in content else "speaker"
        if sp not in by_speaker:
            by_speaker[sp] = []
            order.append(sp)
        body = content.split(":", 1)[1].strip() if ":" in content else content
        by_speaker[sp].append("{}: {}".format(ts, body))
    s1 = order[0] if order else (speakers[0] if speakers else "speaker 1")
    s2 = order[1] if len(order) > 1 else (speakers[1] if len(speakers) > 1 else "speaker 2")
    out = template
    out = out.replace("{{speaker_1_user_id}}", s1).replace("{{speaker_2_user_id}}", s2)
    out = out.replace("{{speaker_1_memories}}",
                      json.dumps(by_speaker.get(s1, []), indent=4, ensure_ascii=False))
    out = out.replace("{{speaker_2_memories}}",
                      json.dumps(by_speaker.get(s2, []), indent=4, ensure_ascii=False))
    out = out.replace("{{question}}", question)
    return out


class AnswerClient:
    def __init__(self, base_url, model, timeout):
        from openai import OpenAI  # noqa
        self._c = OpenAI(base_url=base_url,
                         api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                         timeout=timeout, max_retries=0)
        self.model = model

    def answer(self, prompt):
        t0 = time.time()
        resp = self._c.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        latency = (time.time() - t0) * 1000
        usage = getattr(resp, "usage", None)
        return {
            "text": (resp.choices[0].message.content or "").strip(),
            "model_returned": getattr(resp, "model", None),
            "latency_ms": latency,
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description="LoCoMo E2E QA (frozen retrieval).")
    ap.add_argument("--artifact-glob", default=str(P3_LOCOMO / "sfv2-full-*.json"))
    ap.add_argument("--method", default="sfv2_qwen3_4b")
    ap.add_argument("--profile", default="mem0", choices=["mem0"])
    ap.add_argument("--answer-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--answer-base-url", default=None)
    ap.add_argument("--answer-prompt", default="locomo_answer_mem0.txt")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-root", default=str(REPO / "results" / "locomo_e2e"))
    ap.add_argument("--top-k-evidence", type=int, default=10)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--question-offset", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--cost-cap-usd", type=float, default=5.0)
    ap.add_argument("--price-in-per-1m", type=float, default=0.15)
    ap.add_argument("--price-out-per-1m", type=float, default=0.60)
    args = ap.parse_args()

    run_id = args.run_id or "{}_{}".format(args.method, time.strftime("%Y%m%d-%H%M%S"))
    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = run_dir / "per_question.jsonl"
    cfg_path = run_dir / "run_config.json"

    answer_prompt_path = REPO / "prompts" / args.answer_prompt
    answer_template = answer_prompt_path.read_text(encoding="utf-8")
    answer_prompt_hash = sha256_text(answer_template)
    artifact_files = sorted(glob.glob(args.artifact_glob))
    if not artifact_files:
        print("ERROR: no artifacts match {}".format(args.artifact_glob))
        return 2
    digest = config_digest(artifact_files, args.dataset, answer_prompt_hash,
                           args.answer_model, args.profile, args.top_k_evidence)

    # Fail-closed resume check.
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text(encoding="utf-8"))
        if prev.get("config_digest") != digest:
            print("ERROR: config_digest changed ({} -> {}). Use a new --run-id."
                  .format(prev.get("config_digest"), digest))
            return 3
    base_url = args.answer_base_url or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1")

    elr = _load_elr()
    samples = json.load(open(args.dataset, encoding="utf-8"))
    index, per_sample = build_question_index(samples, elr)
    frozen = load_frozen_diags(args.artifact_glob)

    # Select eligible questions: cat != 5, offset >= question_offset, capped.
    selected = []
    for offset in sorted(frozen):
        if offset < args.question_offset:
            continue
        qd = frozen[offset]
        meta = index[offset]
        cat = meta["category"]
        try:
            cat_int = int(cat)
        except Exception:
            cat_int = None
        if cat_int == 5:
            continue
        selected.append((offset, qd, meta, cat_int))
        if args.max_questions is not None and len(selected) >= args.max_questions:
            break

    already = {r["question_id"] for r in read_jsonl(per_q_path)} if args.resume else set()
    todo = [s for s in selected if "{}:{}".format(s[2]["sample_id"], s[2]["qa_index"]) not in already]

    print("run={} method={} profile={} artifacts={} eligible_total={} selected={} todo={}".format(
        run_id, args.method, args.profile, len(artifact_files),
        len(index), len(selected), len(todo)))
    print("digest={} answer_model={} endpoint={}".format(digest[:12], args.answer_model, base_url))

    # Write/refresh run_config.json
    cfg = {
        "run_id": run_id, "method": args.method, "profile": args.profile,
        "config_digest": digest,
        "artifact_glob": args.artifact_glob, "artifact_files": [Path(f).name for f in artifact_files],
        "dataset": str(args.dataset), "dataset_sha256": sha256_file(Path(args.dataset)),
        "answer_model": args.answer_model, "answer_prompt": args.answer_prompt,
        "answer_prompt_hash": answer_prompt_hash,
        "judge_model": "gpt-4o-mini-2024-07-18",
        "judge_prompt_hash": sha256_text((REPO / "prompts" / "locomo_judge_mem0.txt").read_text(encoding="utf-8")),
        "top_k_evidence": args.top_k_evidence,
        "question_subset": "category != 5 (non-adversarial)",
        "frozen_retrieval": True,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print("dry-run: would process {} questions; no API calls.".format(len(todo)))
        # Show one fully-rendered prompt for inspection (first todo item).
        if todo:
            offset, qd, meta, cat = todo[0]
            smap = per_sample[meta["sample_id"]]
            entries = [(rid, smap["mem2content"].get(rid, ""), smap["mem2ts"].get(rid, ""))
                       for rid in qd["result_ids"][: args.top_k_evidence]]
            p = render_answer_prompt(answer_template, entries,
                                     [], meta["question"])
            print("----- sample rendered answer prompt (offset {}) -----".format(offset))
            print(p[:1200])
        return 0

    if not os.environ.get("OPENAI_API_KEY") and "api.openai.com" in base_url:
        print("ERROR: OPENAI_API_KEY not set; refusing paid endpoint.")
        return 4

    client = AnswerClient(base_url, args.answer_model, args.timeout)
    spent = 0.0
    done = 0
    for offset, qd, meta, cat in todo:
        smap = per_sample[meta["sample_id"]]
        result_ids = qd["result_ids"][: args.top_k_evidence]
        entries = [(rid, smap["mem2content"].get(rid, ""), smap["mem2ts"].get(rid, ""))
                   for rid in result_ids]
        prompt = render_answer_prompt(answer_template, entries, [], meta["question"])
        em = sc.evidence_metrics(qd["result_ids"], qd["gold_mem_ids"])
        reference = meta["answer"] if meta["answer"] is not None else ""
        row = {
            "question_id": "{}:{}".format(meta["sample_id"], meta["qa_index"]),
            "conversation_id": meta["sample_id"], "qa_index": meta["qa_index"],
            "question_offset": offset, "category_id": cat,
            "category_name": CATEGORY_NAMES.get(cat),
            "question": meta["question"], "reference_answer": reference,
            "retrieved_evidence_ids": result_ids,
            "retrieved_evidence": [
                {"mem_id": rid, "timestamp": smap["mem2ts"].get(rid),
                 "content": smap["mem2content"].get(rid)} for rid in result_ids],
            "answer_model": args.answer_model, "answer_prompt_version": "mem0.v1",
            "answer_prompt_hash": answer_prompt_hash, "answer_temperature": 0.0,
            "top_k_evidence": args.top_k_evidence,
            "hit1": em["hit_at_1"], "hit3": em["hit_at_3"], "hit10": em["hit_at_10"],
            "mrr": em["mrr"], "n_gold": em["n_gold"],
            "evidence_hits_at_10": em["evidence_hits_at_10"],
            "first_gold_rank": em["first_gold_rank"],
            "status": "ok", "attempts": 1, "error": None,
            "run_id": run_id, "method": args.method, "config_digest": digest,
        }
        try:
            out = client.answer(prompt)
            row["generated_answer"] = out["text"]
            row["answer_model_returned"] = out["model_returned"]
            row["answer_latency_ms"] = round(out["latency_ms"], 1)
            row["answer_input_tokens"] = out["input_tokens"]
            row["answer_output_tokens"] = out["output_tokens"]
            spent += (out["input_tokens"] * args.price_in_per_1m
                      + out["output_tokens"] * args.price_out_per_1m) / 1e6
            row.update(sc.score_answer(out["text"], reference, cat))
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)[:200]
        append_jsonl(per_q_path, row)
        done += 1
        if done % 10 == 0 or done == len(todo):
            print("  answered {}/{}  spent=${:.4f}".format(done, len(todo), spent))
        if spent >= args.cost_cap_usd:
            print("COST CAP REACHED (${:.4f} >= ${:.2f}); stopping.".format(spent, args.cost_cap_usd))
            break
    print("done. answered={} spent=${:.4f}".format(done, spent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
