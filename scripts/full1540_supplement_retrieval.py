"""Supplement the 10 Full-1540 questions that lack frozen retrieval.

Covers exactly:
- the 9 evidence-unresolvable questions (manifest retrieval_offset = null):
  their gold evidence lists are empty or malformed in the dataset, so they
  never entered the 1976-question retrieval track — but the questions
  themselves are valid and belong to the 1540 E2E set;
- offset 758 (conv-43:4, multi-hop): the only eligible question whose
  historical retrieval timed out; it is absent from the P4-A/SF v2 frozen
  artifacts.

For each question x method (p1 / p4a_bm25 / sfv2_4b) this script replays the
method's FROZEN retrieval configuration (same flags as the historical full
runs, same loopback Qwen3-4B rank server, same SF fact cache, FastEmbed
bge-small) on a fresh per-conversation store and records the top-10 raw
messages. No gold answers, no observations, and no tuning of any kind are
used; Search returns raw original messages only.

This is E2E coverage repair. It does NOT re-run the 1976 evidence benchmark
and must not be merged into the frozen artifact set.

Wall-time policy (recorded per row): per-LLM-call timeout 120s (historical
--model-timeout); overall search wall cap 600s (new backstop for the
conv-43 rank hang). On timeout -> status=retrieval_timeout, no results.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_locomo_e2e as e2e  # noqa: E402
import evaluate_locomo_retrieval as elr  # noqa: E402
from evaluate_locomo_retrieval import SearchOnlyModel  # noqa: E402
from app.schemas import AddRequest  # noqa: E402
from app.storage import MemoryStore  # noqa: E402

DATASET = e2e.DEFAULT_DATASET
MANIFEST = REPO / "results" / "locomo_e2e" / "full1540" / "question_manifest.json"
OUT_DIR = REPO / "results" / "locomo_e2e" / "full1540_missing_retrieval"
SF_CACHE = WORKSPACE / "chrono-hybrid-mem-p3" / ".locomo" / "session-facts-full.json"
EMBED_CACHE = WORKSPACE / "chrono-hybrid-mem" / ".model-cache"
LOCAL_URL = "http://127.0.0.1:8081/v1"
MODEL_TIMEOUT_S = 120.0          # historical --model-timeout
SEARCH_WALL_TIMEOUT_S = 600.0    # new wall-clock backstop (recorded)
METHOD_COMMIT = "f6c1f98"

METHOD_CONFIGS = {
    "p1": {
        "structured_query_plan": True,
        "config_flags": ["--structured-query-plan"],
    },
    "p4a_bm25": {
        "structured_query_plan": True,
        "evidence_need_retrieval": True,
        "evidence_need_quota": 2,
        "need_select_by_bm25": True,
        "config_flags": ["--structured-query-plan", "--evidence-need-retrieval",
                         "--evidence-need-quota", "2", "--need-select-by-bm25"],
    },
    "sfv2_4b": {
        "structured_query_plan": True,
        "evidence_need_retrieval": True,
        "evidence_need_quota": 2,
        "need_select_by_bm25": True,
        "session_fact_layer": True,
        "session_fact_cache": str(SF_CACHE),
        "local_embedding_model": "BAAI/bge-small-en-v1.5",
        "local_cache_dir": str(EMBED_CACHE),
        "dense_weight": 0.0,
        "config_flags": ["--structured-query-plan", "--evidence-need-retrieval",
                         "--evidence-need-quota", "2", "--need-select-by-bm25",
                         "--session-fact-layer",
                         "--session-fact-cache", str(SF_CACHE),
                         "--local-embedding-model", "BAAI/bge-small-en-v1.5",
                         "--local-cache-dir", str(EMBED_CACHE),
                         "--dense-weight", "0"],
    },
}


def build_store(method: str, db_path: str, user_id: str, sample):
    """Mirror the historical runner's store construction for one method."""
    cfg = METHOD_CONFIGS[method]
    model = SearchOnlyModel(
        "local-only", model_name="local", base_url=LOCAL_URL,
        disable_thinking=True, timeout_seconds=MODEL_TIMEOUT_S)
    kwargs = {"model": model, "structured_query_plan": cfg["structured_query_plan"]}
    if cfg.get("evidence_need_retrieval"):
        kwargs.update(evidence_need_retrieval=True,
                      evidence_need_quota=cfg["evidence_need_quota"],
                      need_select_by_bm25=True)
    semantic_retriever = None
    if cfg.get("session_fact_layer"):
        from app.local_semantic import LocalSemanticRetriever
        semantic_retriever = LocalSemanticRetriever(
            cfg["local_embedding_model"], device="cpu",
            cache_dir=cfg["local_cache_dir"])
        kwargs.update(session_fact_layer=True,
                      dense_rrf_weight=cfg["dense_weight"])
    store = MemoryStore(db_path, semantic_retriever=semantic_retriever, **kwargs)
    store.initialize()
    sessions, evidence_text = elr.sessions_and_evidence(sample)
    for i, (session_key, messages) in enumerate(sessions):
        store.add(AddRequest(
            request_id="supp:{}:{}".format(user_id, session_key),
            user_id=user_id, session_id="supp:{}".format(session_key),
            messages=messages))
    if cfg.get("session_fact_layer"):
        mem_by_content = elr._load_mem_by_content(db_path, user_id)
        elr._inject_session_facts(
            store=store, user_id=user_id, cache_path=cfg["session_fact_cache"],
            sample_id=str(sample.get("sample_id")),
            evidence_text=evidence_text, mem_by_content=mem_by_content)
    return store, sessions, evidence_text


def run_one(method: str, entry: dict, samples_by_id: dict) -> dict:
    sample = samples_by_id[entry["conversation_id"]]
    sample_id = str(sample.get("sample_id"))
    row = {
        "question_id": entry["question_id"],
        "manifest_offset": entry["offset"],
        "retrieval_offset": entry["retrieval_offset"],
        "category_id": entry["category_id"],
        "method": method,
        "method_commit": METHOD_COMMIT,
        "config_flags": METHOD_CONFIGS[method]["config_flags"],
        "local_search_url": LOCAL_URL,
        "model_timeout_s": MODEL_TIMEOUT_S,
        "search_wall_timeout_s": SEARCH_WALL_TIMEOUT_S,
        "retrieval_status": None,
        "retrieval_ids": [],
        "raw_evidence": [],
        "latency_ms": None,
        "error": None,
        "evidence_metrics_evaluable": False,
        "note": None,
    }
    with tempfile.TemporaryDirectory(prefix="full1540-supp-") as tmp:
        db_path = str(Path(tmp) / "sample.db")
        user_id = "locomo:{}".format(sample_id)
        store, sessions, evidence_text = build_store(
            method, db_path, user_id, sample)
        # gold evidence is resolvable for offset 758 only
        qa = sample["qa"][int(entry["question_id"].rsplit(":", 1)[1])]
        gold_resolvable = bool(entry["retrieval_offset"] is not None)
        row["evidence_metrics_evaluable"] = bool(gold_resolvable)
        if gold_resolvable:
            gold_mem = []
            seen = set()
            n = 0
            for _sk, msgs in sessions:
                for msg in msgs:
                    n += 1
                    key = str(msg["content"]).strip().casefold()
                    if key not in seen:
                        seen.add(key)
                        for dia in qa.get("evidence", []):
                            if dia in evidence_text and \
                                    str(evidence_text[dia]).strip().casefold() == key:
                                gold_mem.append("mem_{}".format(n))
            row["gold_mem_ids_dedup_first"] = sorted(set(gold_mem))
        executor = ThreadPoolExecutor(max_workers=1)

        def _search():
            return store.search(user_id=user_id, query=str(qa["question"]),
                                top_k=10)

        future = executor.submit(_search)
        started = time.time()
        try:
            results = future.result(timeout=SEARCH_WALL_TIMEOUT_S)
            row["latency_ms"] = round((time.time() - started) * 1000, 1)
            row["retrieval_status"] = "ok"
            row["retrieval_ids"] = [r.id for r in results]
            row["raw_evidence"] = [r.content for r in results]
            trace = store.last_retrieval_trace
            row["plan"] = trace.get("plan", {})
            if method == "sfv2_4b":
                row["session_fact_diagnostics"] = trace.get(
                    "session_fact_diagnostics", {})
        except TimeoutError:
            row["retrieval_status"] = "retrieval_timeout"
            row["error"] = ("search exceeded wall timeout of {}s".format(
                SEARCH_WALL_TIMEOUT_S))
        except Exception as exc:  # APITimeoutError and any transport failure
            row["latency_ms"] = round((time.time() - started) * 1000, 1)
            row["retrieval_status"] = "retrieval_timeout"
            row["error"] = str(exc)[:300]
        finally:
            executor.shutdown(wait=False)
    return row


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = json.load(open(DATASET, encoding="utf-8"))
    samples_by_id = {str(s.get("sample_id")): s for s in samples}
    targets = [q for q in manifest["questions"]
               if q["retrieval_offset"] is None or q["retrieval_offset"] == 758]
    print("supplement targets: {} questions x {} methods".format(
        len(targets), len(METHOD_CONFIGS)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for method in ("p1", "p4a_bm25", "sfv2_4b"):
        out_path = OUT_DIR / "{}.jsonl".format(method)
        done_ids = set()
        if out_path.exists():  # resumable
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("retrieval_status") == "ok":
                        done_ids.add(r["question_id"])
        with open(out_path, "a", encoding="utf-8") as out:
            for entry in targets:
                if entry["question_id"] in done_ids:
                    continue
                t0 = time.time()
                row = run_one(method, entry, samples_by_id)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                status = row["retrieval_status"]
                if status != "ok":
                    failures += 1
                print("[{}+] {} {} {} latency={} ids={}".format(
                    method, entry["question_id"], status,
                    "ev_ok" if row["evidence_metrics_evaluable"] else "ev_na",
                    row["latency_ms"], row["retrieval_ids"][:3]),
                    flush=True)
    print("done. non-ok searches: {}".format(failures))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
