"""Independent synthetic mechanics tests, without benchmark gold or API calls."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from app.evidence_selection import validate_selection
from app.model import MemoryModel
from app.schemas import AddRequest, Message
from app.storage import MemoryStore


def decision(ids, excluded=None, needs=0, states=None):
    return {"complete": True, "selected_ids": ids, "excluded_ids": excluded or [],
            "evidence_groups": [{"need_index": i if needs else None, "ids": ids}
                                for i in range(max(1, needs))],
            "state_updates": states or []}


class FixtureModel:
    def __init__(self, choose=None, intent="fact", needs=None):
        self.choose = choose
        self.intent = intent
        self.needs = needs or []
        self.plans = self.ranks = self.selections = 0
        self.candidates = []

    def extract_facts(self, content, **kwargs):
        return []

    def plan_query_structured(self, query, options):
        self.plans += 1
        return {"intent": self.intent, "core_terms": ["project"], "entities": [],
                "temporal_cues": [], "expansion_terms": [], "evidence_needs": self.needs}

    def rank_candidates(self, query, options, candidates):
        self.ranks += 1
        return [candidates[0]["id"]]

    def select_evidence(self, query, options, candidates, plan):
        self.selections += 1
        self.candidates = candidates
        if self.choose:
            return self.choose(candidates)
        return decision([candidates[0]["id"]], needs=len(self.needs))


def add(store, text, request_id, session="s1", user="u1", timestamp=None):
    store.add(AddRequest(request_id=str(request_id), user_id=user, session_id=session,
                         messages=[Message(role="user", content=text, timestamp=timestamp)]))


def make_store(tmp_path, model=None, **kwargs):
    store = MemoryStore(str(tmp_path / "memory.db"), model=model,
                        structured_query_plan=True, **kwargs)
    store.initialize()
    return store


def test_selection_filters_unselected_sensitive_candidates_without_extra_calls(tmp_path):
    model = FixtureModel(lambda pool: decision(["mem_1"]))
    store = make_store(tmp_path, model, evidence_selection=True)
    add(store, "Project status is approved.", 0)
    for i in range(1, 40):
        add(store, f"Project private medical detail {i} should not be disclosed.", i)
    results = store.search(user_id="u1", query="What is the project status?", top_k=100)
    assert [row.id for row in results] == ["mem_1"]
    assert results[0].content == "Project status is approved."
    assert (model.plans, model.selections, model.ranks) == (1, 1, 0)
    assert len(model.candidates) == 30
    assert store.last_retrieval_trace["evidence_selection"]["applied"] is True


def test_flag_off_keeps_legacy_results_and_single_rank_call(tmp_path):
    model = FixtureModel()
    store = make_store(tmp_path, model)
    for i in range(40):
        add(store, f"Project status record {i}", i)
    results = store.search(user_id="u1", query="project status", top_k=100)
    assert len(results) == 40
    assert (model.plans, model.selections, model.ranks) == (1, 0, 1)


@pytest.mark.parametrize("mode", ["incomplete", "unknown_id", "empty", "error", "missing_need"])
def test_invalid_or_failed_selection_preserves_lexical_results_once(tmp_path, mode):
    def choose(pool):
        if mode == "error":
            raise RuntimeError("secret model output must not be logged")
        if mode == "incomplete":
            return {"complete": False}
        if mode == "unknown_id":
            return decision(["mem_other_user"])
        if mode == "empty":
            return decision([])
        return decision([pool[0]["id"]])
    model = FixtureModel(choose, needs=["project current state", "project conflicting state"])
    store = make_store(tmp_path, model, evidence_selection=True)
    add(store, "Project status is pending.", 1)
    add(store, "Project status is approved.", 2)
    add(store, "Project status belongs to a different user.", 3, user="other")
    result = store.search(user_id="u1", query="project status", top_k=100)
    assert {row.id for row in result} == {"mem_1", "mem_2"}
    assert (model.selections, model.ranks) == (1, 0)
    assert not store.last_retrieval_trace["evidence_selection"]["applied"]


def test_current_update_excludes_old_value_but_history_remains_queryable(tmp_path):
    current = True
    def choose(pool):
        if current:
            return decision(["mem_2"], excluded=["mem_1", "mem_3"], states=[
                {"current_id": "mem_2", "superseded_ids": ["mem_1"]}])
        return decision(["mem_1", "mem_2"])
    model = FixtureModel(choose, intent="governance")
    store = make_store(tmp_path, model, evidence_selection=True, coverage_retrieval=True)
    add(store, "Project status is pending.", 1, timestamp=1700000000)
    add(store, "Correction: project status is approved, effective now.", 2, timestamp=1700086400)
    add(store, "Project status might be archived next year; this is only a plan.", 3, timestamp=1700172800)
    assert [row.id for row in store.search(user_id="u1", query="current project status", top_k=100)] == ["mem_2"]
    current = False
    assert [row.id for row in store.search(user_id="u1", query="project status change history", top_k=100)] == ["mem_1", "mem_2"]
    assert "Event date:" in model.candidates[0]["content"]
    with store._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 3


@pytest.mark.parametrize("intent", ["rule", "multi_hop", "learning", "summary"])
def test_all_required_evidence_is_kept_in_order_and_topk_is_an_upper_bound(tmp_path, intent):
    ids = ["mem_{}".format(i) for i in range(1, 16)]
    model = FixtureModel(lambda pool: decision(ids, needs=3), intent=intent,
                         needs=["project definitions", "project exception", "project observations"])
    store = make_store(tmp_path, model, evidence_selection=True)
    for i in range(16):
        add(store, f"Project authoritative evidence step {i}", i)
    result = store.search(user_id="u1", query="project all necessary steps", top_k=100)
    assert [row.id for row in result] == ids  # No arbitrary 12-ID cut.
    result = store.search(user_id="u1", query="project all necessary steps", top_k=2)
    assert [row.id for row in result] == ids[:2]
    assert store.last_retrieval_trace["evidence_selection"]["coverage_truncated"]


def test_cross_chunk_neighbors_repair_both_sides_and_keep_sessions_and_users_separate(tmp_path):
    store = make_store(tmp_path)
    add(store, "Project Orchid is our destination.", 1)
    add(store, "Different session secret.", 2, session="another")
    add(store, "Different user secret.", 3, user="other")
    add(store, "It opens on Saturday.", 4)
    add(store, "The route is via the harbor.", 5)
    add(store, "The route is via the harbor.", 5)  # Retry must not duplicate.
    with store._connection() as connection:
        for table in ("context_fts", "context_porter_fts"):
            rows = connection.execute("SELECT message_id,content FROM {} WHERE user_id='u1'".format(table)).fetchall()
            contexts = {row["message_id"]: row["content"] for row in rows}
            assert "Saturday" in contexts[1]
            assert "Orchid" in contexts[4] and "harbor" in contexts[4]
            assert "Saturday" in contexts[5]
            assert "secret" not in contexts[4]
            assert len(rows) == 4
    assert "mem_4" in {r.id for r in store.search(user_id="u1", query="Orchid opening", top_k=100)}


def test_existing_database_index_migration_is_idempotent_and_source_preserving(tmp_path):
    store = make_store(tmp_path)
    add(store, "Project Orchid location", 1)
    add(store, "It opens Saturday", 2)
    with store._connection() as connection:
        connection.execute("DELETE FROM text_index_migrations")
        connection.execute("UPDATE context_fts SET content='old batch-only index'")
        connection.execute("UPDATE context_porter_fts SET content='old batch-only index'")
    store.initialize()
    store.initialize()
    with store._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM context_fts").fetchone()[0] == 2
        assert "Saturday" in connection.execute("SELECT content FROM context_fts WHERE message_id=1").fetchone()[0]


def test_summary_can_recall_multiple_sessions_without_query_word_overlap(tmp_path):
    model = FixtureModel(lambda pool: decision(["mem_1", "mem_2", "mem_3"]), intent="summary")
    store = make_store(tmp_path, model, evidence_selection=True, coverage_retrieval=True)
    for i, text in enumerate(["Joined the robotics lab.", "Built a sensor prototype.", "Published the measurements."]):
        add(store, text, i, session=str(i))
    add(store, "Other user's confidential notes.", 4, session="0", user="other")
    results = store.search(user_id="u1", query="Summarize my journey", top_k=100)
    assert [row.id for row in results] == ["mem_1", "mem_2", "mem_3"]
    assert len(model.candidates) == 3
    assert "confidential" not in str(model.candidates)


def test_summary_failure_does_not_return_unfiltered_coverage_memories(tmp_path):
    model = FixtureModel(lambda pool: {"complete": False}, intent="summary")
    store = make_store(tmp_path, model, evidence_selection=True, coverage_retrieval=True)
    add(store, "Private unrelated notes", 1)
    assert store.search(user_id="u1", query="Summarize my journey", top_k=100) == []
    assert model.selections == 1


def test_rule_coverage_includes_nonlexical_exception_and_bounded_pool(tmp_path):
    model = FixtureModel(intent="rule")
    store = make_store(tmp_path, model, evidence_selection=True, coverage_retrieval=True)
    add(store, "Project requests must pass review.", 0)
    for i in range(1, 80):
        add(store, f"Project unrelated progress record {i}.", i)
    add(store, "Unless the office is closed, use the emergency exception.", 80)
    store.search(user_id="u1", query="project approval rule", top_k=100)
    assert "mem_81" in {item["id"] for item in model.candidates}
    assert len(model.candidates) == 30
    assert len(store.last_retrieval_trace["reserved_coverage_ids"]) <= 8


def test_redacted_trace_drops_free_text_and_trace_write_failure_is_nonfatal(tmp_path):
    def choose(pool):
        result = decision(["mem_1"])
        result["reason"] = "private-output-secret"
        result["evidence_groups"][0]["explanation"] = "private-output-secret"
        return result
    model = FixtureModel(choose)
    trace_file = tmp_path / "trace.jsonl"
    store = make_store(tmp_path, model, evidence_selection=True, retrieval_trace_path=str(trace_file))
    add(store, "Project secret content", 1)
    store.search(user_id="u1", query="project secret query", top_k=100)
    text = trace_file.read_text(encoding="utf-8")
    assert "secret" not in text
    assert json.loads(text)["selection"]["selected_ids"] == ["mem_1"]
    store.retrieval_trace_path = str(tmp_path)  # A directory cannot be appended.
    assert len(store.search(user_id="u1", query="project", top_k=100)) == 1


@pytest.mark.parametrize("payload", [
    decision(["a"], states=[{"current_id": "a", "superseded_ids": ["b"]}]),
    decision(["a"], excluded=["a"]),
    decision(["a", "a"]),
    decision(["a"], excluded=["untrusted"]),
])
def test_unwitnessed_or_inconsistent_decisions_do_not_apply(payload):
    assert not validate_selection(payload, ["a", "b"], [])["applied"]


def test_selector_json_contract_uses_one_model_call_and_no_answer_text(monkeypatch):
    model = object.__new__(MemoryModel)
    calls = []
    def response(prompt, payload, max_tokens):
        calls.append((prompt, payload, max_tokens))
        return decision(["mem_1"])
    monkeypatch.setattr(model, "_json_response", response)
    result = model.select_evidence("current status", [], [{"id": "mem_1", "content": "raw"}], {"evidence_needs": []})
    assert result["selected_ids"] == ["mem_1"]
    assert len(calls) == 1
    assert "future plan" in calls[0][0] and "exceptions" in calls[0][0]
    assert calls[0][2] == 2048


def test_runtime_flags_are_explicit_and_reject_incompatible_settings(tmp_path, monkeypatch):
    from app.main import create_app, text_optimization_flag
    monkeypatch.delenv("MEMORY_EVIDENCE_SELECTION", raising=False)
    assert not text_optimization_flag("MEMORY_EVIDENCE_SELECTION")
    monkeypatch.setenv("MEMORY_EVIDENCE_SELECTION", "TRUE")
    monkeypatch.setenv("MEMORY_COVERAGE_RETRIEVAL", "true")
    captured = []
    original = MemoryStore.__init__
    def capture(self, *args, **kwargs):
        captured.append(kwargs)
        original(self, *args, **kwargs)
    monkeypatch.setattr(MemoryStore, "__init__", capture)
    create_app(str(tmp_path / "api.db"))
    assert captured[0]["evidence_selection"] and captured[0]["coverage_retrieval"]
    monkeypatch.setenv("MEMORY_EVIDENCE_SELECTION", "typo")
    with pytest.raises(RuntimeError, match="true or false"):
        text_optimization_flag("MEMORY_EVIDENCE_SELECTION")
    with pytest.raises(ValueError, match="requires evidence_selection"):
        MemoryStore(str(tmp_path / "invalid.db"), coverage_retrieval=True)


def test_api_optimized_path_returns_selected_original_source(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main_module
    model = FixtureModel(lambda pool: decision(["mem_1"]))
    monkeypatch.setattr(main_module, "model_from_environment", lambda: model)
    monkeypatch.setenv("MEMORY_EVIDENCE_SELECTION", "true")
    monkeypatch.setenv("MEMORY_COVERAGE_RETRIEVAL", "true")
    with TestClient(main_module.create_app(str(tmp_path / "api.db"))) as client:
        for i, text in enumerate(["Project current status: approved.", "Project private unrelated notes."]):
            assert client.post("/add", json={"request_id": str(i), "user_id": "u1", "session_id": "s1",
                                             "messages": [{"role": "user", "content": text}]}).status_code == 200
        response = client.post("/search", json={"query": "project status", "user_id": "u1", "top_k": 100})
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data) == 1 and data[0]["content"] == "Project current status: approved."
        assert set(data[0]) == {"id", "content", "score", "created_at"}
    assert model.selections == 1 and model.ranks == 0


def test_concurrent_traces_use_each_query_local_result(tmp_path):
    model = FixtureModel()
    trace_file = tmp_path / "concurrent.jsonl"
    store = make_store(tmp_path, model, evidence_selection=True, retrieval_trace_path=str(trace_file))
    for i in range(8):
        add(store, f"Project status for private user {i}", i, user=str(i), timestamp=1700000000 + i)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda user: store.search(user_id=str(user), query="project status", top_k=100), range(8)))
    records = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    assert {tuple(row["final_ids"]) for row in records} == {tuple(item.id for item in result) for result in results}
    assert len(records) == 8
    assert all(row["selection"]["selected_ids"] == row["final_ids"] for row in records)


def test_enhanced_planner_is_opt_in_and_uses_one_call(monkeypatch):
    model = object.__new__(MemoryModel)
    calls = []
    def response(prompt, payload):
        calls.append(prompt)
        return {"intent": "summary", "evidence_needs": ["earlier stage", "later stage"]}
    monkeypatch.setattr(model, "_json_response", response)
    assert model.plan_query_structured("Summarize", [])["intent"] == "other"
    assert "whole-history synthesis" not in calls[0]
    plan = model.plan_evidence_query("Summarize", [])
    assert plan["intent"] == "summary" and len(plan["evidence_needs"]) == 2
    assert len(calls) == 2


def test_session_fact_provenance_rejects_foreign_sources_and_cleans_replaced_links(tmp_path):
    store = make_store(tmp_path)
    add(store, "Project source one", 1)
    add(store, "Other session source", 2, session="s2")
    add(store, "Other user source", 3, user="other")
    store.add_session_facts(user_id="u1", facts=[
        {"session_id": "s1", "source_message_id": 1, "fact_text": "Valid fact"},
        {"session_id": "s1", "source_message_id": 2, "fact_text": "Wrong session"},
        {"session_id": "s1", "source_message_id": 3, "fact_text": "Wrong user"}])
    store.replace_session_facts(user_id="u1", session_id="s1", new_facts=[
        {"fact_text": "Updated fact", "source_message_ids": ["mem_1", "mem_1", "mem_2", "mem_3"]}])
    with store._connection() as connection:
        assert [row[0] for row in connection.execute("SELECT fact_text FROM session_facts")] == ["Updated fact"]
        assert [row[0] for row in connection.execute("SELECT source_message_id FROM session_fact_sources")] == [1]
