"""Frozen-plan paired P1/P2 evaluation for the public AML-like suite."""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.model import MemoryModel
from evaluation.aml_synthetic.compare import compare_p2_non_degradation
from evaluation.aml_synthetic.evaluate import SearchOnlyLocalProxy, evaluate


class FrozenPlanRankModel:
    """Expose fixed structured plans while counting only this arm's ranks."""

    def __init__(self, plans: Mapping[str, Mapping[str, Any]], rank_model: object) -> None:
        self._plans = plans
        self._rank_model = rank_model
        self.logical_plan_calls = 0
        self.rank_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def call_count(self) -> int:
        return self.rank_calls

    def extract_facts(self, content, speaker="", timestamp=None):
        return self._rank_model.extract_facts(content, speaker, timestamp)

    def plan_query_structured(self, query, options):
        self.logical_plan_calls += 1
        if query not in self._plans:
            raise RuntimeError("no frozen plan for synthetic query")
        return copy.deepcopy(dict(self._plans[query]))

    def rank_candidates(self, query, options, candidates):
        self.rank_calls += 1
        return self._rank_model.rank_candidates(query, options, candidates)


def select_cases(cases: Sequence[Mapping[str, Any]], *, limit: int | None, per_category_limit: int | None) -> list[Mapping[str, Any]]:
    if limit and per_category_limit:
        raise ValueError("limit and per_category_limit are mutually exclusive")
    if limit:
        return list(cases[:limit])
    if per_category_limit is None:
        return list(cases)
    if per_category_limit <= 0:
        raise ValueError("per_category_limit must be positive")
    counts: dict[str, int] = defaultdict(int); selected = []
    for item in cases:
        category = str(item["category"])
        if counts[category] < per_category_limit:
            selected.append(item); counts[category] += 1
    return selected


def freeze_plans(cases: Sequence[Mapping[str, Any]], model: object) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in cases:
        query = str(item["query"])
        if query in result:
            continue
        plan = model.plan_query_structured(query, list(item["options"]))
        if not isinstance(plan, Mapping):
            raise ValueError("structured plan must be an object")
        result[query] = json.loads(json.dumps(dict(plan), ensure_ascii=False, sort_keys=True))
    return result


def run_p2_paired(cases: Sequence[Mapping[str, Any]], *, work_dir: Path, rank_model: object, plans: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline_model = FrozenPlanRankModel(plans, rank_model)
    p2_model = FrozenPlanRankModel(plans, rank_model)
    baseline = evaluate(cases, work_dir / "baseline", baseline_model, True, set_aware_rerank=False, include_case_diagnostics=True)
    p2_on = evaluate(cases, work_dir / "p2-on", p2_model, True, set_aware_rerank=True, include_case_diagnostics=True)
    for arm, model, enabled in ((baseline, baseline_model, False), (p2_on, p2_model, True)):
        arm["model"] = "local-search-proxy"
        if arm["set_aware_rerank"] is not enabled or model.logical_plan_calls != len(cases) or model.rank_calls != len(cases):
            raise RuntimeError("paired arm call/configuration invariant failed")
    return {
        "questions": len(cases), "frozen_plan_external_calls": len(plans),
        "baseline": baseline, "p2_on": p2_on,
        "calls": {"baseline_plan_reads": baseline_model.logical_plan_calls, "p2_plan_reads": p2_model.logical_plan_calls, "baseline_rank": baseline_model.rank_calls, "p2_rank": p2_model.rank_calls},
        "plan_sha256": {query: __import__("hashlib").sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() for query, plan in plans.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("cases.json")))
    parser.add_argument("--work-dir", default=".eval-tmp/aml-p2-paired")
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-base-url", required=True); parser.add_argument("--local-model", default="local")
    parser.add_argument("--limit", type=int); parser.add_argument("--per-category-limit", type=int)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise SystemExit("refusing to overwrite output")
    if not args.local_base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise SystemExit("local loopback URL required")
    cases = select_cases(json.loads(Path(args.cases).read_text(encoding="utf-8")), limit=args.limit, per_category_limit=args.per_category_limit)
    if not cases: raise SystemExit("no cases selected")
    delegate = SearchOnlyLocalProxy(MemoryModel("local-only", model_name=args.local_model, base_url=args.local_base_url, disable_thinking=True))
    plans = freeze_plans(cases, delegate)
    paired = run_p2_paired(cases, work_dir=Path(args.work_dir), rank_model=delegate, plans=plans)
    result = {"stage": "aml_synthetic_p2_frozen_plan_paired", "model": args.local_model, "paired": paired, "gate": compare_p2_non_degradation(paired["baseline"], paired["p2_on"]), "external_model_calls": delegate.call_count}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
