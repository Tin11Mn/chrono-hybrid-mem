"""Evaluate original-evidence retrieval on the public AML-like suite."""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app.model import MemoryModel
from app.schemas import AddRequest
from app.storage import MemoryStore


K_VALUES = (1, 3, 5, 10)


class FixturePlanModel:
    """Isolate flat-vs-structured retrieval mechanics without claiming model quality."""

    def __init__(self, cases):
        self.plans = {item["query"]: item["structured_plan"] for item in cases}
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query(self, query, options):
        self.call_count += 1
        plan_item = self.plans[query]
        flattened = []
        for key in (
            "core_terms", "expansion_terms", "entities", "temporal_cues", "evidence_needs"
        ):
            flattened.extend(plan_item[key])
        return flattened[:16]

    def plan_query_structured(self, query, options):
        self.call_count += 1
        return self.plans[query]

    def rank_candidates(self, query, options, candidates):
        self.call_count += 1
        return [candidate["id"] for candidate in candidates]


class SearchOnlyLocalProxy:
    """Use the local proxy for Search calls while omitting unchanged Add extraction."""

    def __init__(self, delegate):
        self.delegate = delegate

    @property
    def call_count(self):
        return self.delegate.call_count

    @property
    def prompt_tokens(self):
        return self.delegate.prompt_tokens

    @property
    def completion_tokens(self):
        return self.delegate.completion_tokens

    def extract_facts(self, content, speaker="", timestamp=None):
        return []

    def plan_query(self, query, options):
        return self.delegate.plan_query(query, options)

    def plan_query_structured(self, query, options):
        return self.delegate.plan_query_structured(query, options)

    def rank_candidates(self, query, options, candidates):
        return self.delegate.rank_candidates(query, options, candidates)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("cases.json")),
    )
    parser.add_argument("--work-dir", default=".eval-tmp/aml-synthetic")
    parser.add_argument("--model", action="store_true", help="Use formal gpt-4o-mini path")
    parser.add_argument(
        "--fixture-plans", action="store_true",
        help="Use annotated plans and identity ranking to isolate retrieval routing",
    )
    parser.add_argument("--local-base-url", help="Loopback OpenAI-compatible proxy")
    parser.add_argument("--local-model", default="local")
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Proxy smoke only: omit the unchanged Add extraction calls",
    )
    parser.add_argument("--structured", action="store_true")
    parser.add_argument(
        "--support-weight", type=float, default=MemoryStore.STRUCTURED_SUPPORT_RRF_WEIGHT,
        help="Structured expansion/evidence-need RRF weight for P1 calibration",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def mean(values):
    return round(statistics.fmean(values), 6) if values else 0.0


def evaluate(
    cases, work_dir, model, structured,
    support_weight=MemoryStore.STRUCTURED_SUPPORT_RRF_WEIGHT,
):
    work_dir = work_dir / f"run-{time.time_ns()}"
    work_dir.mkdir(parents=True, exist_ok=False)
    hits = {k: [] for k in K_VALUES}
    category_hits = defaultdict(lambda: {k: [] for k in K_VALUES})
    reciprocal_ranks = []
    all_evidence_coverage = []
    forbidden_top1 = []
    duplicate_ratios = []
    pool_sizes = []
    add_latencies = []
    search_latencies = []
    cross_user_leaks = 0

    for position, item in enumerate(cases):
        database_path = work_dir / f"case-{position:04d}.db"
        store = MemoryStore(
            str(database_path), model=model, structured_query_plan=structured
        )
        store.STRUCTURED_SUPPORT_RRF_WEIGHT = support_weight
        store.initialize()
        started = time.perf_counter()
        for memory_position, memory in enumerate(item["memories"]):
            store.add(AddRequest(
                request_id=f"{item['id']}-{memory_position}",
                user_id=memory.get("user_id", "user-main"),
                session_id=memory.get("session_id", "session-main"),
                messages=[{
                    "role": memory.get("role", "user"),
                    "content": memory["content"],
                    **({"timestamp": memory["timestamp"]} if "timestamp" in memory else {}),
                }],
            ))
        add_latencies.append(time.perf_counter() - started)

        started = time.perf_counter()
        results = store.search(
            user_id="user-main",
            query=item["query"],
            options=item["options"],
            top_k=10,
        )
        search_latencies.append(time.perf_counter() - started)
        ranked_ids = [result.id for result in results]
        required = set(item["required_evidence_ids"])
        forbidden = set(item["forbidden_evidence_ids"])
        for k in K_VALUES:
            value = float(bool(required & set(ranked_ids[:k])))
            hits[k].append(value)
            category_hits[item["category"]][k].append(value)
        ranks = [ranked_ids.index(evidence_id) + 1 for evidence_id in required if evidence_id in ranked_ids]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        if item["category"] == "B":
            all_evidence_coverage.append(float(required <= set(ranked_ids[:10])))
        forbidden_top1.append(float(bool(ranked_ids[:1] and ranked_ids[0] in forbidden)))
        pool_sizes.append(len(ranked_ids))
        duplicate_ratios.append(
            1.0 - len({result.content.casefold() for result in results}) / len(results)
            if results else 0.0
        )
        other_user_ids = {
            memory["id"] for memory in item["memories"]
            if memory.get("user_id", "user-main") != "user-main"
        }
        cross_user_leaks += len(other_user_ids & set(ranked_ids))

    return {
        "cases": len(cases),
        "model": (
            "fixture-plan-identity-ranker" if isinstance(model, FixturePlanModel)
            else "local-search-proxy" if isinstance(model, SearchOnlyLocalProxy)
            else "local-full-proxy" if getattr(model, "model_name", None) == "local"
            else "gpt-4o-mini" if model else None
        ),
        "structured_query_plan": structured,
        "structured_support_rrf_weight": support_weight if structured else None,
        "hit_at_k": {str(k): mean(values) for k, values in hits.items()},
        "mrr": mean(reciprocal_ranks),
        "category_hit_at_k": {
            category: {str(k): mean(values) for k, values in category_values.items()}
            for category, category_values in sorted(category_hits.items())
        },
        "multi_hop_all_evidence_coverage_at_10": mean(all_evidence_coverage),
        "forbidden_evidence_at_1": mean(forbidden_top1),
        "cross_user_leakage": cross_user_leaks,
        "duplicate_ratio": mean(duplicate_ratios),
        "candidate_pool_size_mean": mean(pool_sizes),
        "add_latency_seconds_mean": mean(add_latencies),
        "search_latency_seconds_mean": mean(search_latencies),
        "gpt_calls": getattr(model, "call_count", 0),
        "prompt_tokens": getattr(model, "prompt_tokens", 0),
        "completion_tokens": getattr(model, "completion_tokens", 0),
    }


def main():
    args = parse_args()
    selected_models = sum(bool(value) for value in (
        args.model, args.fixture_plans, args.local_base_url
    ))
    if selected_models > 1:
        raise SystemExit("use exactly one of --model, --fixture-plans, or --local-base-url")
    if args.structured and selected_models == 0:
        raise SystemExit("--structured requires a planner model")
    if args.skip_extraction and not args.local_base_url:
        raise SystemExit("--skip-extraction is only valid with --local-base-url")
    api_key = os.getenv("OPENAI_API_KEY")
    if args.model and not api_key:
        raise SystemExit("--model requires OPENAI_API_KEY")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]
    if args.model:
        model = MemoryModel(api_key)
    elif args.fixture_plans:
        model = FixturePlanModel(cases)
    elif args.local_base_url:
        local_model = MemoryModel(
            "local-only", model_name=args.local_model,
            base_url=args.local_base_url, disable_thinking=True,
        )
        model = SearchOnlyLocalProxy(local_model) if args.skip_extraction else local_model
    else:
        model = None
    if args.support_weight < 0 or args.support_weight > 1:
        raise SystemExit("--support-weight must be between 0 and 1")
    metrics = evaluate(
        cases, Path(args.work_dir), model, args.structured, args.support_weight
    )
    rendered = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
