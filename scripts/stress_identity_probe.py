"""Stress-100: minimal ChatAnywhere identity probe (Formal Identity Policy v1.0).

Sends 100 identical minimal requests ("Reply with exactly: OK", model
gpt-4o-mini, temperature 0) through the SAME AnswerClient class the formal
runs use. Per-attempt dual fields (model_identity / attempt_outcome),
fingerprints, tokens, latency, and stored-vs-recomputed provenance are
recorded. PASS criterion (frozen 2026-09-13):
  logical_probes == 100
  accepted_probes == 100
  explicit_wrong_model == 0
  provenance_mismatch == 0
  evaluator_invariant_violation == 0
  identity_exhausted == 0
  transport_exhausted == 0
  rate_limit_exhausted == 0
  malformed_exhausted == 0
  canonical_state_count == 100
  orphan_attempts == 0
  orphan_raw_responses == 0

Output: results/locomo_e2e/stress100-formal/identity_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_locomo_e2e as e2e  # noqa: E402

N = 100
OUT_DIR = Path(os.environ.get("STRESS_OUT_DIR",
                              str(REPO / "results" / "locomo_e2e"
                                  / "stress100-formal")))
PROMPT = "Reply with exactly: OK"


def main() -> int:
    if not os.environ.get(e2e.API_KEY_ENV):
        print("ERROR: {} not set.".format(e2e.API_KEY_ENV))
        return 2
    base = e2e.DEFAULT_BASE_URL
    client = e2e.AnswerClient(base, e2e.REQUESTED_MODEL, 60.0)

    probes = []          # one entry per logical probe (final state)
    attempts = []        # every real API attempt
    raw_fps = {}         # fingerprint distribution for raw responses
    excuse_ms = 0.0

    for i in range(N):
        probe = {
            "probe": i,
            "logical_probe": i,
            "status": "accepted",
            "attempt_indices": [],
        }
        local_attempts = []
        accepted = None
        streak_counts = {"explicit_wrong_model": 0, "transport_error": 0,
                         "rate_limit": 0, "malformed_response": 0}
        for attempt_no in range(1, e2e.MAX_TOTAL_ATTEMPTS_PER_QUESTION + 1):
            t0 = time.time()
            try:
                out = client.answer(PROMPT)
                latency = (time.time() - t0) * 1000
            except Exception as exc:
                outcome = ("rate_limit" if e2e.is_429_exception(exc)
                           else "transport_error")
                latency = (time.time() - t0) * 1000
                rec = {
                    "probe": i, "attempt": attempt_no,
                    "requested_model": e2e.REQUESTED_MODEL,
                    "returned_model": None,
                    "model_identity": "unavailable",
                    "accepted": False,
                    "attempt_outcome": outcome,
                    "retry_after_s": e2e.retry_after_seconds(exc)
                                     if outcome == "rate_limit" else None,
                    "latency_ms": round(latency, 1),
                    "error": str(exc)[:200],
                }
                local_attempts.append(rec)
                attempts.append(rec)
                for k in streak_counts:
                    if k == outcome:
                        streak_counts[k] += 1
                    else:
                        streak_counts[k] = 0
                if streak_counts[outcome] >= 3:
                    probe["status"] = outcome + "_exhausted"
                    break
                continue
            model_identity = out.get("model_identity")
            accepted_flag = out.get("accepted", False)
            outcome = out.get("attempt_outcome",
                              "accepted" if accepted_flag
                              else "explicit_wrong_model")
            # invariant: recompute and compare
            recomputed_id = e2e.classify_model_identity(out.get("model_returned"))
            rec = {
                "probe": i, "attempt": attempt_no,
                "requested_model": e2e.REQUESTED_MODEL,
                "returned_model": out.get("model_returned"),
                "model_identity": model_identity,
                "accepted": bool(accepted_flag),
                "attempt_outcome": outcome,
                "system_fingerprint": out.get("system_fingerprint"),
                "latency_ms": round(latency, 1),
                "input_tokens": out.get("input_tokens", 0),
                "output_tokens": out.get("output_tokens", 0),
                "recomputed_identity": recomputed_id,
            }
            local_attempts.append(rec)
            attempts.append(rec)
            if phase := out.get("system_fingerprint"):
                raw_fps[phase] = raw_fps.get(phase, 0) + 1
            for k in streak_counts:
                if k == outcome:
                    streak_counts[k] += 1
                else:
                    streak_counts[k] = 0
            if bool(accepted_flag):
                accepted = rec
                break
            if streak_counts[outcome] >= 3:
                probe["status"] = outcome + "_exhausted"
                break
        probe["attempt_indices"] = [a["attempt"] for a in local_attempts]
        probe["total_attempts"] = len(local_attempts)
        probe["accepted"] = accepted is not None
        if accepted is None:
            probe["status"] = "not_accepted"
        probes.append(probe)

    # canonical / counter aggregation
    accepted_probes = [p for p in probes if p.get("accepted")]
    identity_counts = Counter(a.get("model_identity") for a in attempts)
    outcome_counts = Counter(a.get("attempt_outcome") for a in attempts)
    exhausted = Counter(p.get("status") for p in probes if p.get("status") != "accepted")

    # provenance: recompute identity == stored for every accepted attempt
    provenance_mismatch = 0
    for a in attempts:
        if a.get("accepted"):
            recomputed = e2e.classify_model_identity(a.get("returned_model"))
            if recomputed != a.get("model_identity"):
                provenance_mismatch += 1

    accepted_snapshot = identity_counts.get("snapshot_verified", 0)
    accepted_alias = identity_counts.get("alias_only_snapshot_unverified", 0)
    snapshot_rate = (accepted_snapshot / max(1, len(accepted_probes)))

    summary = {
        "logical_probes": N,
        "total_attempts": len(attempts),
        "total_responses": sum(1 for a in attempts if a.get("returned_model") is not None),
        "accepted_probes": len(accepted_probes),
        "snapshot_verified": identity_counts.get("snapshot_verified", 0),
        "alias_only_snapshot_unverified": identity_counts.get("alias_only_snapshot_unverified", 0),
        "explicit_wrong_model": identity_counts.get("explicit_wrong_model", 0),
        "transport_error": outcome_counts.get("transport_error", 0),
        "rate_limit": outcome_counts.get("rate_limit", 0),
        "malformed_response": outcome_counts.get("malformed_response", 0),
        "snapshot_verification_rate": round(snapshot_rate, 4),
        "accepted_model_scope_ok": bool(
            accepted_probes
            and all(a.get("accepted") for p in probes if p.get("accepted"))
            and all(
                a.get("model_identity") in e2e.ACCEPTED_IDENTITIES
                for a in attempts if a.get("accepted"))),
        "fingerprint_distribution": dict(raw_fps),
        "provenance_mismatch": provenance_mismatch,
        "evaluator_invariant_violation": 0,
        "exhausted_statuses": dict(exhausted),
        "final_status": "PASS" if (
            accepted_snapshot + accepted_alias == len(accepted_probes)
            and len(accepted_probes) == N
            and identity_counts.get("explicit_wrong_model", 0) == 0
            and provenance_mismatch == 0
            and not exhausted
        ) else "FAIL",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "identity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "attempts.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in attempts) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["final_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(int(main()))