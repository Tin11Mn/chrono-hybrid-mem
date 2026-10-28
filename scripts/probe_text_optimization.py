"""Compare evidence-output mechanics on synthetic data; no network or answer gold."""

import argparse
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimized", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from app.schemas import AddRequest, Message
    from app.storage import MemoryStore

    class Model:
        def __init__(self):
            self.plan_calls = self.rank_calls = self.selection_calls = 0
            self.pool_count = 0
            self.selected_ids = []

        def extract_facts(self, content, **kwargs):
            return []

        def plan_query_structured(self, query, options):
            self.plan_calls += 1
            return {"intent": "governance", "core_terms": ["project", "status"],
                    "entities": [], "temporal_cues": [], "expansion_terms": [], "evidence_needs": []}

        def rank_candidates(self, query, options, candidates):
            self.rank_calls += 1
            self.pool_count = len(candidates)
            self.selected_ids = [candidates[0]["id"]]
            return self.selected_ids

        def select_evidence(self, query, options, candidates, plan):
            self.selection_calls += 1
            self.pool_count = len(candidates)
            self.selected_ids = [candidates[0]["id"]]
            return {"complete": True, "selected_ids": self.selected_ids, "excluded_ids": [],
                    "evidence_groups": [{"need_index": None, "ids": self.selected_ids}], "state_updates": []}

    model = Model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="probe-", dir=args.output.parent) as folder:
        options = {"evidence_selection": True, "coverage_retrieval": True} if args.optimized else {}
        store = MemoryStore(str(Path(folder) / "memory.db"), model=model,
                            structured_query_plan=True, evidence_need_retrieval=True,
                            need_select_by_bm25=True, **options)
        store.initialize()
        for i in range(40):
            store.add(AddRequest(request_id=str(i), user_id="synthetic", session_id="s1", messages=[
                Message(role="user", content=f"Project status record {i}: value {i}.", timestamp=1700000000+i)
            ]))
        results = store.search(user_id="synthetic", query="current project status", top_k=100)
        with store._connection() as connection:
            context = connection.execute("SELECT content FROM context_fts WHERE message_id=40").fetchone()[0]
        record = {"source": str(args.source.resolve()), "optimized": args.optimized,
                  "messages": 40, "requested_top_k": 100, "rank_pool": model.pool_count,
                  "model_selected": len(model.selected_ids), "returned": len(results),
                  "returned_unselected": sum(item.id not in model.selected_ids for item in results),
                  "cross_batch_neighbor_indexed": "record 38:" in context,
                  "query_model_calls": model.plan_calls + model.rank_calls + model.selection_calls,
                  "official_accuracy": None,
                  "scope": "Synthetic protocol/index comparison; model semantics and official scores unmeasured."}
        args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
