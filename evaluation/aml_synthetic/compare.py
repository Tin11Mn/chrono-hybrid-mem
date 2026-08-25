"""Compare two AML-like result files using predeclared preservation gates."""

import argparse
import json
from pathlib import Path


PRESERVATION_CATEGORIES = ("C", "E", "H")
TARGET_CATEGORIES = ("A", "B")


def compare(baseline, experiment, latency_ratio_limit=1.25):
    failures = []
    if baseline["cases"] != experiment["cases"]:
        failures.append("case count changed")
    if baseline["model"] != experiment["model"]:
        failures.append("model/path changed")
    if experiment["cross_user_leakage"] != 0:
        failures.append("cross-user leakage is nonzero")
    if experiment["gpt_calls"] != baseline["gpt_calls"]:
        failures.append("GPT call count changed")

    baseline_hit1 = baseline["hit_at_k"]["1"]
    experiment_hit1 = experiment["hit_at_k"]["1"]
    if experiment_hit1 <= baseline_hit1:
        failures.append("overall evidence Hit@1 did not improve")

    target_deltas = {
        category: (
            experiment["category_hit_at_k"][category]["1"]
            - baseline["category_hit_at_k"][category]["1"]
        )
        for category in TARGET_CATEGORIES
    }
    if max(target_deltas.values()) <= 0:
        failures.append("neither A nor B Hit@1 improved")
    if min(target_deltas.values()) < 0:
        failures.append("A or B Hit@1 regressed")

    preservation_deltas = {
        category: (
            experiment["category_hit_at_k"][category]["1"]
            - baseline["category_hit_at_k"][category]["1"]
        )
        for category in PRESERVATION_CATEGORIES
    }
    if min(preservation_deltas.values()) < 0:
        failures.append("C, E, or H Hit@1 regressed")
    if experiment["multi_hop_all_evidence_coverage_at_10"] < baseline[
        "multi_hop_all_evidence_coverage_at_10"
    ]:
        failures.append("multi-hop all-evidence coverage regressed")
    if experiment["forbidden_evidence_at_1"] > baseline["forbidden_evidence_at_1"]:
        failures.append("forbidden-evidence rate increased")

    baseline_latency = baseline["search_latency_seconds_mean"]
    experiment_latency = experiment["search_latency_seconds_mean"]
    latency_ratio = experiment_latency / baseline_latency if baseline_latency else None
    if latency_ratio is not None and latency_ratio > latency_ratio_limit:
        failures.append("mean Search latency exceeded the configured ratio")

    fixture = baseline["model"] == "fixture-plan-identity-ranker"
    return {
        "decision": "REJECT" if failures else ("MECHANICS_PASS" if fixture else "ADVANCE"),
        "failures": failures,
        "hit_at_1_delta": round(experiment_hit1 - baseline_hit1, 6),
        "target_category_deltas": target_deltas,
        "preservation_category_deltas": preservation_deltas,
        "forbidden_evidence_at_1_delta": round(
            experiment["forbidden_evidence_at_1"] - baseline["forbidden_evidence_at_1"], 6
        ),
        "search_latency_ratio": round(latency_ratio, 6) if latency_ratio is not None else None,
        "gpt_call_delta": experiment["gpt_calls"] - baseline["gpt_calls"],
    }


def compare_p2_non_degradation(baseline, experiment, latency_ratio_limit=1.25):
    """P2 gate: every public synthetic stratum must preserve P1 quality."""
    failures = []
    if baseline["cases"] != experiment["cases"]:
        failures.append("case count changed")
    if baseline["model"] != experiment["model"]:
        failures.append("model/path changed")
    if baseline.get("set_aware_rerank") is not False or experiment.get("set_aware_rerank") is not True:
        failures.append("comparison is not P1-off versus P2-on")
    if experiment["cross_user_leakage"] != 0:
        failures.append("cross-user leakage is nonzero")
    if experiment["gpt_calls"] != baseline["gpt_calls"]:
        failures.append("GPT call count changed")
    if experiment["hit_at_k"]["1"] < baseline["hit_at_k"]["1"]:
        failures.append("overall evidence Hit@1 regressed")
    if experiment["mrr"] < baseline["mrr"]:
        failures.append("overall MRR regressed")
    category_deltas = {
        category: experiment["category_hit_at_k"][category]["1"] - baseline["category_hit_at_k"][category]["1"]
        for category in baseline["category_hit_at_k"]
    }
    if min(category_deltas.values()) < 0:
        failures.append("at least one capability stratum Hit@1 regressed")
    if experiment["multi_hop_all_evidence_coverage_at_10"] < baseline["multi_hop_all_evidence_coverage_at_10"]:
        failures.append("multi-hop all-evidence coverage regressed")
    if experiment["forbidden_evidence_at_1"] > baseline["forbidden_evidence_at_1"]:
        failures.append("forbidden-evidence rate increased")
    baseline_latency = baseline["search_latency_seconds_mean"]
    latency_ratio = experiment["search_latency_seconds_mean"] / baseline_latency if baseline_latency else None
    if latency_ratio is not None and latency_ratio > latency_ratio_limit:
        failures.append("mean Search latency exceeded the configured ratio")
    fixture = baseline["model"] == "fixture-plan-identity-ranker"
    return {
        "decision": "REJECT" if failures else ("MECHANICS_PASS" if fixture else "ADVANCE"),
        "failures": failures,
        "hit_at_1_delta": round(experiment["hit_at_k"]["1"] - baseline["hit_at_k"]["1"], 6),
        "mrr_delta": round(experiment["mrr"] - baseline["mrr"], 6),
        "category_hit_at_1_deltas": category_deltas,
        "search_latency_ratio": round(latency_ratio, 6) if latency_ratio is not None else None,
        "gpt_call_delta": experiment["gpt_calls"] - baseline["gpt_calls"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("experiment")
    parser.add_argument("--latency-ratio-limit", type=float, default=1.25)
    parser.add_argument("--p2-non-degradation", action="store_true")
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    experiment = json.loads(Path(args.experiment).read_text(encoding="utf-8"))
    result = (compare_p2_non_degradation if args.p2_non_degradation else compare)(
        baseline, experiment, args.latency_ratio_limit
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if result["decision"] == "REJECT" else 0)


if __name__ == "__main__":
    main()
