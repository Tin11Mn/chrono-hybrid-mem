"""Regression gates for actual smoke failures; no network/model credentials."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading
import time

import pytest

from app.model import MemoryModel
from app.schemas import AddRequest
from app.storage import MemoryStore
from app.structured_memory import grounded_records, resolve_states
from app.structured_memory import route_memory_requests
from app.evidence_selection import validate_selection


def request(index=1, user="tenant", messages=None):
    return AddRequest(request_id=str(index), user_id=user, session_id="session",
                      messages=messages or [{"role": "user", "content": "Message {}".format(index)}])


def test_sixteen_extractions_run_without_holding_a_sqlite_writer(tmp_path):
    release = threading.Event()
    reached = threading.Event()
    lock = threading.Lock()
    class BlockingModel:
        calls = 0
        def extract_facts(self, *args, **kwargs):
            with lock:
                self.calls += 1
                if self.calls == 16:
                    reached.set()
            assert release.wait(8), "extraction gate timed out"
            return []
    model = BlockingModel()
    store = MemoryStore(str(tmp_path / "parallel.db"), model=model)
    store.initialize()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(store.add, request(i)) for i in range(16)]
        try:
            assert reached.wait(6), "not all16 model calls could start concurrently"
            with sqlite3.connect(store.database_path, timeout=0.05) as connection:
                connection.execute("BEGIN IMMEDIATE")
                assert connection.execute("SELECT COUNT(*) FROM ingestions").fetchone()[0] == 0
                assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 0
                connection.rollback()
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        for future in futures:
            future.result(timeout=10)
    with store._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 16
        assert connection.execute("SELECT COUNT(*) FROM ingestions").fetchone()[0] == 16
        assert connection.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 16
    assert model.calls == 16
    assert not MemoryStore._ingestion_registry


def test_duplicate_requests_coalesce_across_store_instances_and_users_are_isolated(tmp_path):
    class SlowModel:
        calls = 0
        def extract_facts(self, *args, **kwargs):
            self.calls += 1
            time.sleep(0.03)
            return []
    model = SlowModel()
    path = str(tmp_path / "duplicates.db")
    stores = [MemoryStore(path, model=model), MemoryStore(path, model=model)]
    stores[0].initialize()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(stores[i % 2].add, request("same")) for i in range(8)]
        for future in futures:
            future.result(timeout=5)
    assert model.calls == 1
    stores[1].add(request("same", user="other-tenant"))
    assert model.calls == 2
    with stores[0]._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 2
    assert not MemoryStore._ingestion_registry


def test_model_failure_leaves_no_partial_ingestion_and_retry_is_safe(tmp_path):
    class FailingModel:
        fail = True
        def extract_facts(self, content, **kwargs):
            if content == "second" and self.fail:
                raise RuntimeError("upstream unavailable")
            return []
    model = FailingModel()
    store = MemoryStore(str(tmp_path / "retry.db"), model=model)
    store.initialize()
    payload = request(messages=[{"role": "user", "content": x} for x in ["first", "second"]])
    with pytest.raises(RuntimeError, match="upstream"):
        store.add(payload)
    with store._connection() as connection:
        for table in ["ingestions", "raw_messages", "messages_fts", "context_fts"]:
            assert connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] == 0
    model.fail = False
    store.add(payload)
    store.add(payload)
    with store._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 2


def test_publish_failure_rolls_back_all_indices_and_marker(tmp_path, monkeypatch):
    store = MemoryStore(str(tmp_path / "publish.db"))
    store.initialize()
    original = store._refresh_session_context
    def fail(*args):
        raise RuntimeError("publish failed")
    monkeypatch.setattr(store, "_refresh_session_context", fail)
    with pytest.raises(RuntimeError):
        store.add(request())
    with store._connection() as connection:
        for table in ["ingestions", "raw_messages", "messages_fts", "messages_porter_fts"]:
            assert connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] == 0
    monkeypatch.setattr(store, "_refresh_session_context", original)
    store.add(request())
    assert not MemoryStore._ingestion_registry


def test_old_ingestion_primary_key_migrates_without_losing_history(tmp_path):
    store = MemoryStore(str(tmp_path / "old.db"))
    store.initialize()
    with store._connection() as connection:
        connection.execute("DROP TABLE ingestions")
        connection.execute("CREATE TABLE ingestions(request_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,session_id TEXT NOT NULL,completed_at TEXT NOT NULL)")
        connection.execute("INSERT INTO ingestions VALUES ('same','first','session','old-time')")
    store.initialize()
    store.initialize()
    store.add(request("same", user="second"))
    with store._connection() as connection:
        assert connection.execute("SELECT completed_at FROM ingestions WHERE user_id='first'").fetchone()[0] == "old-time"
        assert connection.execute("SELECT COUNT(*) FROM ingestions").fetchone()[0] == 2


def captured():
    return json.loads((Path(__file__).parent / "fixtures" / "synthetic_protocol_8352433.json").read_text(encoding="utf-8"))


def test_actual_provider_uncued_update_becomes_assertion_without_supersession():
    records, complete = grounded_records(captured()["pending"], "ProtocolSample status is pending.", "user", 100)
    assert complete and records[0]["modality"] == "assert"
    records[0]["source_id"] = 2
    older = dict(records[0], source_id=1, effective_ts=50, value="approved")
    result = resolve_states([older] + records)
    assert result["status"] == "unresolved" and result["blocked"] == []
    updated, valid = grounded_records(captured()["approved"], "ProtocolSample status is now approved.", "user", 200)
    assert valid and updated[0]["modality"] == "update"
    updated[0]["source_id"] = 3
    assert resolve_states(records + updated)["required"] == [3]


def test_actual_provider_zero_need_index_is_general_evidence_not_fake_coverage():
    payload = captured()["zero_need_selection"]
    result = validate_selection(payload, ["mem_1", "mem_2"], [])
    assert result["applied"] and result["evidence_groups"] == [{"need_index": None, "ids": ["mem_2"]}]
    assert not validate_selection(payload, ["mem_1", "mem_2"], ["one", "two"])["applied"]
    payload["selected_ids"] = ["other-users-memory"]
    assert not validate_selection(payload, ["mem_1", "mem_2"], [])["applied"]


def test_prompt_fact_scaffolding_is_never_indexed_as_an_actual_fact(monkeypatch):
    model = object.__new__(MemoryModel)
    monkeypatch.setattr(model, "_json_response", lambda *args, **kwargs: captured()["approved"])
    result = model.extract_structured_memory("ProtocolSample status is now approved.", "user", 100)
    assert result["facts"] == []
    assert result["records"][0]["value"] == "approved"


def test_selector_contract_uses_actual_need_count_instead_of_copied_index(monkeypatch):
    model = object.__new__(MemoryModel)
    prompts = []
    def response(prompt, payload, **kwargs):
        prompts.append(prompt)
        return {}
    monkeypatch.setattr(model, "_json_response", response)
    model.select_evidence("q", [], [], {"evidence_needs": []})
    model.select_evidence("q", [], [], {"evidence_needs": ["a", "b"]})
    assert "ZERO evidence_needs" in prompts[0] and "MUST be []" in prompts[0]
    assert "[0, 1]" in prompts[1]
    model.plan_indexed_query("q", [], {"groups": []})
    assert "memory_requests is an object" in prompts[2]
    assert "Every value except intent must be an array" not in prompts[2]


def test_captured_wrong_kind_and_attribute_label_route_only_to_known_literal_slots():
    catalog = {"groups": [{"kind": "state", "group_id": "known-state", "subject": "candidatesample",
                            "attribute": "status", "scope": "", "authority": "user"}]}
    wrong = {"state_slots": ["status"], "observation_groups": ["known-state"], "temporal_mode": "current"}
    routed = route_memory_requests(catalog, wrong, "CandidateSample current status")
    assert routed["state_slots"] == ["known-state"] and routed["observation_groups"] == []
    assert route_memory_requests(catalog, {}, "List status changes of CandidateSample")["temporal_mode"] == "history"
    assert route_memory_requests(catalog, {"state_slots": ["status", "foreign-slot"]}, "Unrelated question")["state_slots"] == []


def test_literal_slot_routing_does_not_guess_unmentioned_scope():
    catalog = {"groups": [{"kind": "state", "group_id": str(i), "subject": "alice", "attribute": "status",
                            "scope": "project" + str(i), "authority": "user"} for i in [1, 2]]}
    assert route_memory_requests(catalog, {"state_slots": ["status"]}, "Alice status")["state_slots"] == []
    assert route_memory_requests(catalog, {}, "Alice project2 status")["state_slots"] == ["2"]


def test_fact_and_state_records_share_versions_without_changing_assertion_authority():
    from app.structured_memory import group_id
    data = captured()
    data["pending"]["records"][0]["kind"] = "fact"
    old, complete = grounded_records(data["pending"], "ProtocolSample status is pending.", "user", 100)
    new, valid = grounded_records(data["approved"], "ProtocolSample status is now approved.", "user", 200)
    assert complete and valid and old[0]["modality"] == "assert"
    assert old[0]["group_id"] == new[0]["group_id"]
    old[0]["source_id"], new[0]["source_id"] = 1, 2
    assert resolve_states(old + new)["required"] == [2]
    assert resolve_states(old + new, "history")["required"] == [1, 2]


def test_compact_explicit_coverage_cannot_claim_missing_needs_or_unknown_sources():
    data = {"complete": True, "selected_ids": ["a", "b"], "excluded_ids": [], "covered_need_indices": [0, 1]}
    assert validate_selection(data, ["a", "b"], ["first", "second"])["applied"]
    assert not validate_selection(dict(data, covered_need_indices=[0]), ["a", "b"], ["first", "second"])["applied"]
    assert not validate_selection(dict(data, covered_need_indices=[0, 0]), ["a", "b"], ["first"])["applied"]
    assert not validate_selection(dict(data, selected_ids=["foreign"]), ["a", "b"], ["first", "second"])["applied"]
    assert validate_selection(dict(data, covered_need_indices=[]), ["a", "b"], [])["applied"]


def test_legacy_fact_slot_migration_merges_state_versions_without_touching_sources(tmp_path):
    data = captured()
    data["pending"]["records"][0]["kind"] = "fact"
    class ReplayModel:
        def extract_structured_memory(self, content, **kwargs):
            return data["approved"] if "now" in content else data["pending"]
    store = MemoryStore(str(tmp_path / "slot-migration.db"), model=ReplayModel(), structured_query_plan=True,
                        evidence_selection=True, structured_memory=True)
    store.initialize()
    for i,text in enumerate(["ProtocolSample status is pending.", "ProtocolSample status is now approved."]):
        store.add(request(i, messages=[{"role":"user","content":text,"timestamp":100+i*100}]))
    with store._connection() as connection:
        row=connection.execute("SELECT id,record_json FROM structured_records WHERE kind='fact'").fetchone()
        old=json.loads(row["record_json"]);old["group_id"]="legacy-fact-group"
        connection.execute("UPDATE structured_records SET group_id=?,record_json=? WHERE id=?",
                           ("legacy-fact-group",json.dumps(old),row["id"]))
        connection.execute("DELETE FROM structured_index_migrations")
    store.initialize()
    store.initialize()
    with store._connection() as connection:
        assert connection.execute("SELECT COUNT(DISTINCT group_id) FROM structured_records").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 2
