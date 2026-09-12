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

# Gateway configuration: ChatAnywhere-mediated GPT-4o-mini-2024-07-18 access.
# The request model is "gpt-4o-mini"; every successful response must report
# model == EXPECTED_RETURNED_MODEL or the run fails closed.
GATEWAY = "chatanywhere"
DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"
API_KEY_ENV = "CHATANYWHERE_API_KEY"
REQUESTED_MODEL = "gpt-4o-mini"
EXPECTED_RETURNED_MODEL = "gpt-4o-mini-2024-07-18"
# Identity-validated retry budget per question per phase (frozen 2026-09-12,
# mirrors MAX_JUDGE_ATTEMPTS): 3 attempts, then STOP the whole run.
MAX_ANSWER_ATTEMPTS = 3


class ModelDriftError(RuntimeError):
    """The gateway returned a model id other than EXPECTED_RETURNED_MODEL."""


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
                  profile, top_k, extra=None):
    payload = {
        "artifact_sha256": [sha256_file(Path(f)) for f in sorted(artifact_files)],
        "dataset_sha256": sha256_file(Path(dataset_path)),
        "answer_prompt_hash": answer_prompt_hash,
        "answer_model": answer_model,
        "profile": profile,
        "top_k_evidence": top_k,
    }
    if extra:
        payload["extra"] = extra
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
                         api_key=os.environ.get(API_KEY_ENV, "EMPTY"),
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
        returned_model = getattr(resp, "model", None)
        # Identity gate: VALID only when the gateway returned the frozen
        # snapshot id. Invalid responses are reported via
        # valid_model_identity/status; the caller owns the per-question retry
        # policy and must never score them.
        valid = returned_model == EXPECTED_RETURNED_MODEL
        usage = getattr(resp, "usage", None)
        return {
            "text": (resp.choices[0].message.content or "").strip(),
            "model_returned": returned_model,
            "valid_model_identity": valid,
            "status": "accepted" if valid else "model_drift",
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
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


def write_jsonl_atomic(path, rows):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    os.replace(tmp, path)


def upsert_row(path, row):
    """Append a new question row, or atomically replace a prior failed row.

    Resume retries previously failed questions; the retry result must replace
    the old error row so the JSONL never carries two rows for one question.
    """
    replaced = False
    if path.exists():
        rows = read_jsonl(path)
        for index, existing in enumerate(rows):
            if (existing.get("question_id") == row["question_id"]
                    and existing.get("status") != "ok"):
                rows[index] = row
                replaced = True
                break
        if replaced:
            write_jsonl_atomic(path, rows)
    if not replaced:
        append_jsonl(path, row)


def save_raw_output(run_dir: Path, question_id: str, phase: str,
                    payload: dict) -> None:
    """Persist one raw model response under raw_model_outputs/."""
    raw_dir = run_dir / "raw_model_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_id = question_id.replace(":", "_")
    (raw_dir / "{}.{}.json".format(safe_id, phase)).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_checkpoint(run_dir: Path, per_q_path: Path, spent: float) -> None:
    """Refresh checkpoint.json from the current per_question.jsonl state."""
    rows = read_jsonl(per_q_path) if per_q_path.exists() else []
    checkpoint = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "answered_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "answered_error": [r["question_id"] for r in rows
                           if r.get("status") != "ok"],
        "judged": sum(1 for r in rows if r.get("judge_result") in ("CORRECT", "WRONG")),
        "spend_estimated_usd": round(spent, 6),
        "per_question_path": str(per_q_path),
    }
    write_jsonl_atomic(run_dir / "checkpoint.json", [checkpoint])


def main():
    ap = argparse.ArgumentParser(description="LoCoMo E2E QA (frozen retrieval).")
    ap.add_argument("--artifact-glob", default=str(P3_LOCOMO / "sfv2-full-*.json"))
    ap.add_argument("--method", default="sfv2_qwen3_4b")
    ap.add_argument("--profile", default="mem0", choices=["mem0"])
    ap.add_argument("--answer-model", default=REQUESTED_MODEL)
    ap.add_argument("--answer-base-url", default=None)
    ap.add_argument("--answer-prompt", default="locomo_answer_mem0.txt")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-root", default=str(REPO / "results" / "locomo_e2e"))
    ap.add_argument("--top-k-evidence", type=int, default=10)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--stratify", type=int, default=None,
                    help="Take the first N eligible questions of EACH category "
                         "1-4 (deterministic offset order) instead of the "
                         "global first-N. Smoke-20 uses --stratify 5.")
    ap.add_argument("--question-offset", type=int, default=0)
    ap.add_argument("--manifest", default=None,
                    help="Path to question_manifest.json. First run freezes the "
                         "stratified selection into this file (deterministic "
                         "offset prefix per category, chosen before any API "
                         "call and without looking at answers or retrieval "
                         "outcomes); every later run verifies its SHA256 "
                         "against run_config.json and fails closed on drift.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--e2e-manifest", default=None,
                    help="Full-1540 mode: the question set comes from this "
                         "manifest (all cat!=5 entries). Frozen artifacts are "
                         "used where retrieval_offset exists; the 9 "
                         "evidence-unresolvable questions and offset 758 take "
                         "their retrieval from --supplement-dir for this "
                         "--method. A supplement retrieval_timeout row is "
                         "prescored F1=0/BLEU=0/Judge=incorrect with no API "
                         "call, per the frozen protocol.")
    ap.add_argument("--supplement-dir", default=None,
                    help="Directory holding {p1,p4a_bm25,sfv2_4b}.jsonl "
                         "supplement rows (required with --e2e-manifest)")
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
    full1540_extra = None
    if args.e2e_manifest:
        mk = {"sfv2_qwen3_4b": "sfv2_4b"}.get(args.method, args.method)
        full1540_extra = "e2e-manifest:{}:{}".format(
            mk, Path(args.supplement_dir or "").name)
    digest = config_digest(artifact_files, args.dataset, answer_prompt_hash,
                           args.answer_model, args.profile, args.top_k_evidence,
                           extra=full1540_extra)

    # Fail-closed resume check.
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text(encoding="utf-8"))
        if prev.get("config_digest") != digest:
            print("ERROR: config_digest changed ({} -> {}). Use a new --run-id."
                  .format(prev.get("config_digest"), digest))
            return 3
    base_url = args.answer_base_url or os.environ.get(
        "OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if (not args.dry_run
            and not os.environ.get(API_KEY_ENV)
            and not base_url.startswith(("http://127.0.0.1:", "http://localhost:"))):
        print("ERROR: {} not set; refusing paid endpoint {}.".format(
            API_KEY_ENV, base_url))
        return 4

    elr = _load_elr()
    samples = json.load(open(args.dataset, encoding="utf-8"))
    index, per_sample = build_question_index(samples, elr)
    frozen = load_frozen_diags(args.artifact_glob)

    already = {
        r["question_id"] for r in read_jsonl(per_q_path)
        if r.get("status") == "ok"
    } if args.resume else set()

    manifest_sha = None
    if args.e2e_manifest:
        # Full-1540 mode: the manifest defines the question set; retrieval is
        # taken from the frozen artifacts where available and from the
        # supplement JSONL for the 9 evidence-unresolvable questions and
        # offset 758. The manifest hash is verified on every launch.
        import hashlib as _hashlib
        manifest_path = Path(args.e2e_manifest)
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha = _hashlib.sha256(
            manifest_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        if cfg_path.exists():
            prev = json.loads(cfg_path.read_text(encoding="utf-8"))
            stored = prev.get("manifest_sha256")
            if stored and stored != manifest_sha:
                print("ERROR: full1540 manifest hash changed ({} -> {}).".format(
                    stored[:16], manifest_sha[:16]))
                return 6
        method_key = {"sfv2_qwen3_4b": "sfv2_4b"}.get(args.method, args.method)
        if not args.supplement_dir:
            print("ERROR: --e2e-manifest requires --supplement-dir.")
            return 2
        supp_path = Path(args.supplement_dir) / "{}.jsonl".format(method_key)
        supplement = {}
        if supp_path.exists():
            for line in supp_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    supplement[r["question_id"]] = r
        meta_by_qid = {}
        for sample in samples:
            sid = str(sample.get("sample_id"))
            for qa_index, qa in enumerate(sample.get("qa", [])):
                if isinstance(qa, dict):
                    meta_by_qid["{}:{}".format(sid, qa_index)] = {
                        "sample_id": sid, "qa_index": qa_index,
                        "question": qa.get("question"),
                        "answer": qa.get("answer"),
                        "category": qa.get("category")}
        selected = []
        for entry in manifest_data["questions"]:
            qid = entry["question_id"]
            meta = meta_by_qid.get(qid)
            if meta is None:
                print("ERROR: manifest question {} not found in dataset.".format(qid))
                return 6
            cat = int(entry["category_id"])
            ro = entry["retrieval_offset"]
            if ro is not None and ro != 758:
                qd = frozen.get(ro)
                if qd is None:
                    print("ERROR: frozen retrieval missing for offset {} ({}).".format(
                        ro, qid))
                    return 6
                selected.append((entry["offset"], qd, meta, cat))
                continue
            supp = supplement.get(qid)
            if supp is None:
                print("ERROR: no supplement row for {} (method {}).".format(
                    qid, method_key))
                return 6
            qd = {
                "result_ids": list(supp.get("retrieval_ids", [])),
                "gold_mem_ids": list(supp.get("gold_mem_ids_dedup_first", [])),
                "question_offset": None,
            }
            if supp.get("retrieval_status") != "ok":
                # Frozen protocol: retrieval_timeout -> prescored failure row,
                # no API call, kept in the N=1540 denominator.
                qd["_retrieval_timeout"] = True
            if ro is None:
                # Evidence-unresolvable: gold cannot map to mem ids.
                qd["_evidence_na"] = True
            selected.append((entry["offset"], qd, meta, cat))
        print("full1540 manifest mode: {} questions (frozen {} / supplement {})".format(
            len(selected),
            sum(1 for q in manifest_data["questions"]
                if q["retrieval_offset"] is not None and q["retrieval_offset"] != 758),
            sum(1 for q in manifest_data["questions"]
                if q["retrieval_offset"] is None or q["retrieval_offset"] == 758)))
    else:
        # Select eligible questions: cat != 5, offset >= question_offset, capped.
        # --stratify N picks the first N of each category 1-4 in offset order so a
        # smoke run covers all four classes regardless of their global frequency.
        selected = []
        stratified_remaining = (
            {cat: args.stratify for cat in (1, 2, 3, 4)}
            if args.stratify is not None else None
        )
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
            if stratified_remaining is not None:
                if cat_int not in stratified_remaining:
                    continue
                if stratified_remaining[cat_int] <= 0:
                    continue
                stratified_remaining[cat_int] -= 1
            selected.append((offset, qd, meta, cat_int))
            if args.max_questions is not None and len(selected) >= args.max_questions:
                break

    # Question manifest: freeze the exact selection before any API call and
    # verify it (SHA256) on every subsequent run including resumes.
    manifest_sha = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_sha = sha256_text(manifest_path.read_text(encoding="utf-8"))
            if cfg_path.exists():
                prev_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                stored_sha = prev_cfg.get("manifest_sha256")
                if stored_sha and stored_sha != manifest_sha:
                    print("ERROR: manifest hash changed ({} -> {}). New run-id "
                          "required; the frozen question set must not change."
                          .format(stored_sha[:16], manifest_sha[:16]))
                    return 6
            manifest_selected = []
            for entry in prev_manifest.get("questions", []):
                offset = int(entry["offset"])
                # NOTE: build_question_index returns a LIST whose position IS
                # the offset; membership tests (`offset in index`) are
                # meaningless, so validate by bounds + exact question_id.
                if offset not in frozen or not 0 <= offset < len(index):
                    print("ERROR: manifest offset {} not in frozen artifacts "
                          "or dataset index.".format(offset))
                    return 6
                qid = "{}:{}".format(index[offset]["sample_id"],
                                     index[offset]["qa_index"])
                if qid != entry["question_id"]:
                    print("ERROR: manifest question_id mismatch at offset {} "
                          "({} != {}).".format(offset, qid, entry["question_id"]))
                    return 6
                manifest_selected.append(
                    (offset, frozen[offset], index[offset],
                     int(index[offset]["category"])))
            if len(manifest_selected) != len(prev_manifest["questions"]):
                print("ERROR: manifest verification failed.")
                return 6
            selected = manifest_selected
        else:
            manifest = {
                "run_id": run_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "selection_rule": (
                    "deterministic stratified prefix: first N eligible "
                    "questions per category 1-4 in frozen-artifact offset "
                    "order (N={}); chosen before any API call, blind to "
                    "answers, retrieval outcomes, and judge results".format(
                        args.stratify)),
                "questions": [
                    {
                        "offset": offset,
                        "question_id": "{}:{}".format(
                            meta["sample_id"], meta["qa_index"]),
                        "conversation_id": meta["sample_id"],
                        "category_id": cat,
                        "category_name": CATEGORY_NAMES.get(cat),
                    }
                    for offset, _qd, meta, cat in selected
                ],
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8")
            manifest_sha = sha256_text(manifest_path.read_text(encoding="utf-8"))
            print("manifest frozen: {} questions -> {} (sha256 {})".format(
                len(selected), manifest_path, manifest_sha[:16]))
    todo = [s for s in selected if "{}:{}".format(s[2]["sample_id"], s[2]["qa_index"]) not in already]

    print("run={} method={} profile={} artifacts={} eligible_total={} selected={} todo={}".format(
        run_id, args.method, args.profile, len(artifact_files),
        len(index), len(selected), len(todo)))
    print("digest={} answer_model={} endpoint={}".format(digest[:12], args.answer_model, base_url))

    # Write/refresh run_config.json
    cfg = {
        "run_id": run_id, "method": args.method, "profile": args.profile,
        "config_digest": digest,
        "gateway": GATEWAY,
        "api_base_url": base_url,
        "requested_model": args.answer_model,
        "expected_returned_model": EXPECTED_RETURNED_MODEL,
        "access_note": "ChatAnywhere gateway-mediated GPT-4o-mini-2024-07-18 access",
        "artifact_glob": args.artifact_glob, "artifact_files": [Path(f).name for f in artifact_files],
        "dataset": str(args.dataset), "dataset_sha256": sha256_file(Path(args.dataset)),
        "answer_model": args.answer_model, "answer_prompt": args.answer_prompt,
        "answer_prompt_hash": answer_prompt_hash,
        "judge_model": "gpt-4o-mini",
        "judge_expected_returned_model": EXPECTED_RETURNED_MODEL,
        "judge_prompt_hash": sha256_text((REPO / "prompts" / "locomo_judge_mem0.txt").read_text(encoding="utf-8")),
        "top_k_evidence": args.top_k_evidence,
        "question_subset": "category != 5 (non-adversarial)",
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "e2e_manifest": args.e2e_manifest,
        "supplement_dir": args.supplement_dir,
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
        ev_na = bool(qd.get("_evidence_na"))
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
            "answer_model": args.answer_model, "answer_model_requested": args.answer_model,
            "gateway": GATEWAY, "api_base_url": base_url,
            "answer_returned_model": None, "answer_system_fingerprint": None,
            "answer_prompt_version": "mem0.v1",
            "answer_prompt_hash": answer_prompt_hash, "answer_temperature": 0.0,
            "top_k_evidence": args.top_k_evidence,
            "hit1": None if ev_na else em["hit_at_1"],
            "hit3": None if ev_na else em["hit_at_3"],
            "hit10": None if ev_na else em["hit_at_10"],
            "mrr": None if ev_na else em["mrr"], "n_gold": em["n_gold"],
            "evidence_hits_at_10": None if ev_na else em["evidence_hits_at_10"],
            "first_gold_rank": None if ev_na else em["first_gold_rank"],
            "evidence_metrics": "N/A" if ev_na else "ok",
            "status": "ok", "attempts": 1, "error": None,
            "run_id": run_id, "method": args.method, "config_digest": digest,
        }
        if qd.get("_retrieval_timeout"):
            # Frozen protocol: retrieval_timeout -> prescored failure row with
            # NO API call, kept in the N=1540 denominator.
            row.update({
                "status": "retrieval_timeout",
                "generated_answer": None,
                "f1_mem0": 0.0, "f1_official": 0.0, "f1_memoryart": 0.0,
                "f1_memoryos": 0.0, "bleu1_m1": 0.0, "bleu1_m4": 0.0,
                "judge_result": "WRONG", "judge_source": "retrieval_timeout_policy",
                "error": "retrieval_timeout per frozen protocol; no API call",
            })
            upsert_row(per_q_path, row)
            write_checkpoint(run_dir, per_q_path, spent)
            done += 1
            print("  [timeout-prescored] {} (no API call)".format(row["question_id"]))
            continue
        try:
            # Identity-validated retry (frozen Full-1540 policy): identical
            # inputs every attempt; retry ONLY on returned-model identity
            # mismatch or transport failure, never on content. Max 3 attempts,
            # then STOP the whole run.
            attempts = []
            accepted = None
            for attempt_no in range(1, MAX_ANSWER_ATTEMPTS + 1):
                try:
                    out = client.answer(prompt)
                except Exception as exc:
                    attempts.append({
                        "attempt": attempt_no,
                        "requested_model": args.answer_model,
                        "returned_model": None,
                        "system_fingerprint": None,
                        "valid_model_identity": False,
                        "status": "transport_error",
                        "error": str(exc)[:200],
                    })
                    continue
                attempts.append({
                    "attempt": attempt_no,
                    "requested_model": args.answer_model,
                    "returned_model": out.get("model_returned"),
                    "system_fingerprint": out.get("system_fingerprint"),
                    "valid_model_identity": bool(out.get("valid_model_identity")),
                    "status": out.get("status"),
                    "input_tokens": out.get("input_tokens", 0),
                    "output_tokens": out.get("output_tokens", 0),
                })
                spent += (out.get("input_tokens", 0) * args.price_in_per_1m
                          + out.get("output_tokens", 0) * args.price_out_per_1m) / 1e6
                if out.get("valid_model_identity") is True:
                    accepted = out
                    break
            row["answer_attempts"] = attempts
            drift_cost = sum(
                (a.get("input_tokens") or 0) * args.price_in_per_1m
                + (a.get("output_tokens") or 0) * args.price_out_per_1m
                for a in attempts if a.get("status") != "accepted") / 1e6
            if drift_cost:
                row["answer_drift_cost"] = round(drift_cost, 6)
            if accepted is None:
                # 3 attempts without a valid snapshot -> NO-GO signal.
                row["status"] = "error"
                row["answer_model_drift"] = any(
                    a.get("status") == "model_drift" for a in attempts)
                row["error"] = (
                    "no {} response after {} attempts ({})".format(
                        EXPECTED_RETURNED_MODEL, len(attempts),
                        ",".join(a["status"] for a in attempts)))
                upsert_row(per_q_path, row)
                save_raw_output(run_dir, row["question_id"], "answer", {
                    "question_id": row["question_id"], "phase": "answer",
                    "judge_attempts": attempts,
                    "status": row["status"], "error": row["error"]})
                write_checkpoint(run_dir, per_q_path, spent)
                print("ANSWER GIVE-UP: {} after {} attempts ({}); checkpoint "
                      "persisted; STOPPING WHOLE RUN.".format(
                          row["question_id"], len(attempts),
                          ",".join(a["status"] for a in attempts)))
                return 7
            out = accepted
            row["generated_answer"] = out["text"]
            row["answer_model_returned"] = out["model_returned"]
            row["answer_returned_model"] = out["model_returned"]
            row["answer_system_fingerprint"] = out.get("system_fingerprint")
            row["answer_latency_ms"] = round(out["latency_ms"], 1)
            row["answer_input_tokens"] = out["input_tokens"]
            row["answer_output_tokens"] = out["output_tokens"]
            row_cost = (out["input_tokens"] * args.price_in_per_1m
                        + out["output_tokens"] * args.price_out_per_1m) / 1e6
            row["estimated_answer_cost"] = round(row_cost, 6)
            row["attempts"] = len(attempts)
            row["cumulative_cost"] = round(spent, 6)
            row.update(sc.score_answer(out["text"], reference, cat))
        except ModelDriftError as exc:
            # Retained for any residual raise path: fail closed.
            row["status"] = "error"
            row["model_drift"] = True
            row["error"] = str(exc)[:200]
            upsert_row(per_q_path, row)
            save_raw_output(run_dir, row["question_id"], "answer",
                            {"error": str(exc)[:500], "model_drift": True})
            write_checkpoint(run_dir, per_q_path, spent)
            print("MODEL DRIFT: {}; checkpoint persisted at {}; stopping.".format(
                exc, per_q_path))
            return 5
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)[:200]
        upsert_row(per_q_path, row)
        save_raw_output(run_dir, row["question_id"], "answer", {
            "question_id": row["question_id"],
            "phase": "answer",
            "requested_model": args.answer_model,
            "returned_model": row.get("answer_returned_model"),
            "system_fingerprint": row.get("answer_system_fingerprint"),
            "temperature": row.get("answer_temperature"),
            "raw": row.get("generated_answer"),
            "input_tokens": row.get("answer_input_tokens"),
            "output_tokens": row.get("answer_output_tokens"),
            "latency_ms": row.get("answer_latency_ms"),
            "status": row.get("status"),
            "error": row.get("error"),
        })
        write_checkpoint(run_dir, per_q_path, spent)
        done += 1
        if done % 10 == 0 or done == len(todo):
            print("  answered {}/{}  spent=${:.4f}".format(done, len(todo), spent))
        if spent >= args.cost_cap_usd:
            print("COST CAP REACHED (${:.4f} >= ${:.2f}); stopping.".format(spent, args.cost_cap_usd))
            break
    write_checkpoint(run_dir, per_q_path, spent)
    print("done. answered={} spent=${:.4f}".format(done, spent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
