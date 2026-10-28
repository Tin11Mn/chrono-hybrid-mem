"""Independent source fixtures for deterministic specialist mechanisms, no APIs."""

from datetime import datetime, timezone
import json

import pytest

from app.model import MemoryModel
from app.schemas import AddRequest, Message
from app.storage import MemoryStore
from app.structured_memory import (StructuredMemoryIndex, directed_path, enforce_obligations,
    evaluate_rules, grounded_records, observation_summary, resolve_states)


def packet(text, value, *, kind="state", subject="Alice", attribute="status", scope="",
           mode="assert", conditions=None, date=None):
    record = {"kind": kind, "subject": subject, "attribute": attribute, "scope": scope,
              "value": value, "span": text, "modality": mode, "conditions": conditions or []}
    if date:
        record["effective_date"] = date
    return {"complete": True, "facts": [], "records": [record]}


def record(text, value, source=1, timestamp=1700000000, **kwargs):
    rows, complete = grounded_records(packet(text, value, **kwargs), text, "user", timestamp)
    assert complete and len(rows) == 1
    rows[0].update(source_id=source, session_id="s1")
    return rows[0]


class IndexedModel:
    def __init__(self, packets, intent="governance", mode="current", choose=None, subjects=None, as_of=None):
        self.packets, self.intent, self.mode, self.choose = packets, intent, mode, choose
        self.subjects = subjects
        self.as_of = as_of
        self.extract_calls = self.plan_calls = self.select_calls = 0
        self.path = None

    def extract_structured_memory(self, content, **kwargs):
        self.extract_calls += 1
        return self.packets.get(content, {"complete": True, "facts": [], "records": []})

    def plan_indexed_query(self, query, options, catalog):
        self.plan_calls += 1
        self.catalog = catalog
        request = {"temporal_mode": self.mode, "as_of": self.as_of,
                   "state_slots": [], "rule_groups": [], "observation_groups": []}
        keys = {"state": "state_slots", "rule": "rule_groups", "observation": "observation_groups"}
        for group in catalog["groups"]:
            if self.subjects is None or group["subject"] in self.subjects:
                request[keys[group["kind"]]].append(group["group_id"])
        if self.path:
            request["path"] = self.path
        return {"intent": self.intent, "core_terms": [], "expansion_terms": [], "entities": self.subjects or [],
                "temporal_cues": [], "evidence_needs": [], "memory_requests": request}

    def select_evidence(self, query, options, candidates, plan):
        self.select_calls += 1
        self.pool = candidates
        self.plan = plan
        ids = self.choose(candidates) if self.choose else list(dict.fromkeys(
            source for group in plan.get("required_source_groups", []) for source in group))
        ids = ids or [candidates[0]["id"]]
        return {"complete": True, "selected_ids": ids, "excluded_ids": [],
                "evidence_groups": [{"need_index": None, "ids": ids}], "state_updates": []}

    def plan_query_structured(self, query, options):
        raise AssertionError("one indexed planner required")

    def extract_facts(self, content, **kwargs):
        raise AssertionError("structured extraction must replace the ordinary Add call")

    def rank_candidates(self, *args):
        raise AssertionError("selection must replace the ordinary ranking call")


def store_for(tmp_path, model, trace=None):
    store = MemoryStore(str(tmp_path / "specialist.db"), model=model, structured_query_plan=True,
                        evidence_selection=True, coverage_retrieval=True, structured_memory=True,
                        retrieval_trace_path=str(trace) if trace else None)
    store.initialize()
    return store


def add(store, text, index, timestamp=1700000000, session="s1", user="u1"):
    store.add(AddRequest(request_id=str(index), user_id=user, session_id=session,
                         messages=[Message(role="user", content=text, timestamp=timestamp)]))


def test_state_solver_uses_event_time_not_ingestion_and_preserves_asof_and_history():
    latest = record("Alice status is now approved.", "approved", 1, 300, mode="update")
    older = record("Alice status is pending.", "pending", 2, 200)
    assert resolve_states([latest, older])["required"] == [1]
    assert resolve_states([latest, older])["blocked"] == [2]
    assert resolve_states([latest, older], "as_of", 250)["required"] == [2]
    assert resolve_states([latest, older], "history")["required"] == [2, 1]


def test_state_solver_does_not_invent_supersession_from_assertion_or_missing_time():
    old = record("Alice status pending.", "pending", 1, None)
    new = record("Alice status is now approved.", "approved", 2, 200, mode="update")
    result = resolve_states([old, new])
    assert result["status"] == "unresolved" and result["required"] == [1, 2]
    assertion = record("Alice status approved.", "approved", 3, 300)
    assert resolve_states([new, assertion])["blocked"] == []
    same_time = record("Alice status now rejected.", "rejected", 4, 200, mode="update")
    assert resolve_states([new, same_time])["status"] == "unresolved"


def test_future_and_hypothesis_cannot_replace_current_even_if_model_labels_update():
    old = record("Alice status approved.", "approved")
    future = record("Alice status will be archived next year.", "archived", 2, 1700000100, mode="update")
    hypothetical = record("Maybe Alice status is now rejected.", "rejected", 3, 1700000200, mode="update")
    dated = record("Alice status now closed, effective 2099-01-01.", "closed", 4, 1700000300,
                   mode="update", date="2099-01-01")
    result = resolve_states([old, future, hypothetical, dated])
    assert result["required"] == [1] and set(result["blocked"]) == {2, 3, 4}


def test_retraction_is_evidence_without_resurrecting_removed_value():
    old = record("Alice status approved.", "approved", 1, 100)
    retracted = record("Alice retracts status approved.", "approved", 2, 200, mode="retraction")
    result = resolve_states([old, retracted])
    assert result["required"] == [2] and result["blocked"] == [1]


@pytest.mark.parametrize("mutation", [
    {"span": "invented span"}, {"value": "invented value"}, {"subject": "Bob"},
    {"scope": "invented scope"}, {"effective_date": "2026-10-28"}, {"kind": []},
    {"modality": "exception"}, {"conditions": ["score>=99"]},
])
def test_unwitnessed_records_are_rejected_and_index_cannot_claim_complete(mutation):
    text = "Alice status approved."
    data = packet(text, "approved")
    data["records"][0].update(mutation)
    records, complete = grounded_records(data, text, "user", 100)
    assert not records and not complete


def test_source_offsets_and_timestamp_units_are_preserved():
    text = "前言。Alice status approved. 后记。"
    data = packet("Alice status approved.", "approved")
    rows, complete = grounded_records(data, text, "user", 1700000000123)
    assert complete and rows[0]["effective_ts"] == 1700000000
    assert text[rows[0]["start"]:rows[0]["end"]] == "Alice status approved."


def test_index_enforces_current_state_when_model_selects_obsolete_source(tmp_path):
    old, new = "Alice status pending.", "Alice status now approved."
    model = IndexedModel({old: packet(old, "pending"), new: packet(new, "approved", mode="update")},
                         choose=lambda pool: ["mem_1"])
    store = store_for(tmp_path, model)
    add(store, old, 1, 100)
    add(store, new, 2, 200)
    result = store.search(user_id="u1", query="Alice current status", top_k=100)
    assert [r.id for r in result] == ["mem_2"]
    trace = store.last_retrieval_trace
    assert trace["evidence_selection"]["reason"] == "specialist_obligation_missing"
    assert trace["structured_memory"]["blocked_ids"] == ["mem_1"]
    assert (model.extract_calls, model.plan_calls, model.select_calls) == (2, 1, 1)
    model.mode = "history"
    assert {r.id for r in store.search(user_id="u1", query="Alice status history", top_k=100)} == {"mem_1", "mem_2"}


def test_scopes_and_authorities_do_not_share_state_slot(tmp_path):
    a, b = "Alice projectA status pending.", "Alice projectB status now approved."
    model = IndexedModel({a: packet(a, "pending", scope="projectA"),
                          b: packet(b, "approved", scope="projectB", mode="update")})
    store = store_for(tmp_path, model)
    add(store, a, 1, 100)
    add(store, b, 2, 200)
    store.search(user_id="u1", query="Alice status", top_k=100)
    assert len(model.catalog["groups"]) == 2
    assert store.last_retrieval_trace["structured_memory"]["blocked_ids"] == []


def test_partial_existing_database_never_claims_authoritative_state_exclusion(tmp_path):
    from app.schemas import AddRequest
    db = str(tmp_path / "specialist.db")
    old_store = MemoryStore(db)
    old_store.initialize()
    add(old_store, "Alice previously worked elsewhere.", 0, 50)
    text = "Alice status now approved."
    model = IndexedModel({text: packet(text, "approved", mode="update")})
    store = store_for(tmp_path, model)
    add(store, text, 1, 100)
    store.search(user_id="u1", query="Alice status", top_k=100)
    assert not model.catalog["complete"]
    assert store.last_retrieval_trace["structured_memory"]["blocked_ids"] == []
    assert {r["status"] for r in store.last_retrieval_trace["structured_memory"]["operators"].values()} == {"partial_index"}
    assert model.extract_calls == 1  # No paid backfill during initialization.


def test_rollup_pages_cover_every_original_record_and_keep_counterexamples(tmp_path):
    texts = [f"Sensor x={i}; y={2*i+1}." for i in range(35)]
    packets = {text: packet(text, text, subject="Sensor", kind="observation", attribute="measurements") for text in texts}
    model = IndexedModel(packets, intent="summary")
    store = store_for(tmp_path, model)
    for i, text in enumerate(texts):
        add(store, text, i, 100+i, session=f"session-{i//12}")
    add(store, texts[0], 0)  # Request retry cannot duplicate the ledger.
    with store._connection() as connection:
        rows = connection.execute("SELECT * FROM structured_rollups WHERE level='topic' AND user_id='u1'").fetchall()
        assert len(rows) == 3
        assert all(len(json.loads(row["record_ids"])) <= 16 for row in rows)
        assert set(source for row in rows for source in json.loads(row["source_ids"])) == set(range(1, 36))
        assert connection.execute("SELECT COUNT(*) FROM structured_records").fetchone()[0] == 35
        assert "x=17" in "\n".join(row["extractive_text"] for row in rows)
    store.search(user_id="u1", query="Summarize Sensor measurements", top_k=100)
    assert store.last_retrieval_trace["structured_memory"]["pool_coverage_complete"]
    assert store.last_retrieval_trace["evidence_selection"]["reason"] == "complete"
    assert len(model.pool) == 35


def test_summary_subject_scope_does_not_force_other_topics_or_privacy_sources(tmp_path):
    a, b, secret = "Alice joined the lab.", "Bob won a race.", "Alice password=private."
    model = IndexedModel({a: packet(a, "lab", kind="fact", attribute="work"),
                          b: packet(b, "race", kind="fact", subject="Bob", attribute="sport"),
                          secret: packet(secret, "private", kind="privacy", attribute="password")}, intent="summary", subjects=["alice"])
    store = store_for(tmp_path, model)
    for i, text in enumerate([a, b, secret]):
        add(store, text, i)
    result = store.search(user_id="u1", query="Summarize Alice work", top_k=100)
    assert [r.id for r in result] == ["mem_1"]
    assert model.plan["required_source_groups"] == [["mem_1"]]


def test_literal_rule_conditions_and_exceptions_are_computed_without_eval():
    base = record("Shipping weight<=20: fee=standard.", "fee=standard", kind="rule", subject="Shipping",
                  attribute="fee", conditions=["weight<=20"])
    exception = record("Shipping exception fragile=true: fee=special.", "fee=special", 2, kind="rule", subject="Shipping",
                       attribute="fee", mode="exception", conditions=["fragile=true"])
    result = evaluate_rules([base, exception], "weight=10; fragile=true")
    assert result["required"] == [1, 2] and result["applicable"] == [2]
    missing = evaluate_rules([base, exception], "weight=10")
    assert missing["status"] == "unknown_inputs" and 2 in missing["unknown"]
    assert evaluate_rules([base], "weight=99")["required"] == []


def test_rule_exception_cannot_be_dropped_by_the_model(tmp_path):
    a, b = "Shipping weight<=20: fee=standard.", "Shipping exception fragile=true: fee=special."
    model = IndexedModel({a: packet(a, "fee=standard", subject="Shipping", kind="rule", attribute="fee", conditions=["weight<=20"]),
                          b: packet(b, "fee=special", subject="Shipping", kind="rule", attribute="fee", mode="exception", conditions=["fragile=true"])},
                         intent="rule", choose=lambda pool: ["mem_1"])
    store = store_for(tmp_path, model)
    add(store, a, 1)
    add(store, b, 2)
    result = store.search(user_id="u1", query="Shipping weight=10; fragile=true", top_k=100)
    assert {r.id for r in result} == {"mem_1", "mem_2"}
    assert store.last_retrieval_trace["evidence_selection"]["reason"] == "specialist_obligation_missing"


def test_numeric_observations_preserve_outliers_and_do_not_validate_extrapolation():
    data = [record(f"Sensor x={i}; y={2*i+1}.", f"x={i}; y={2*i+1}", i, 100+i,
                   kind="observation", subject="Sensor", attribute="measurement") for i in (1, 2, 3)]
    summary = observation_summary(data)
    assert summary["exact_affine_fit"]["slope"] == "2"
    assert summary["exact_affine_fit"]["extrapolation_validated"] is False
    assert summary["statistics"]["y"]["trend"] == "increasing"
    outlier = record("Sensor x=2; y=999.", "x=2; y=999", 4, 200, kind="observation", subject="Sensor", attribute="measurement")
    summary = observation_summary(data + [outlier])
    assert summary["exact_affine_fit"] is None
    assert set(summary["required"]) == {1, 2, 3, 4}


def test_numeric_schema_transfers_evidence_across_sessions_without_target_facts(tmp_path):
    texts = [f"Sensor x={i}; y={3*i}." for i in (1, 2, 3)]
    model = IndexedModel({t: packet(t, t, kind="observation", subject="Sensor", attribute="measurement") for t in texts}, intent="learning")
    store = store_for(tmp_path, model)
    for i, text in enumerate(texts):
        add(store, text, i, 100+i, session=str(i))
    results = store.search(user_id="u1", query="New instrument x=2 based on Sensor calibration", top_k=100)
    assert {r.id for r in results} == {"mem_1", "mem_2", "mem_3"}
    assert all("New instrument" not in r.content for r in results)


def test_directed_path_requires_every_link_and_does_not_reverse_causation():
    links = [record("Alpha causes Beta", "Beta", 1, kind="link", subject="Alpha", attribute="causes"),
             record("Beta causes Gamma", "Gamma", 2, kind="link", subject="Beta", attribute="causes")]
    assert directed_path(links, "Alpha", "Gamma", "causes")["required"] == [1, 2]
    assert directed_path(links, "Gamma", "Alpha", "causes")["status"] == "not_found"
    assert directed_path(links, "Alpha", "Gamma", "correlates")["status"] == "unsupported"
    assert directed_path(links, "Alpha", "Gamma", "causes", max_hops=1)["status"] == "not_found"


@pytest.mark.parametrize("text", ["Alpha does not cause Beta", "Maybe Alpha causes Beta", "Alpha correlates with Beta", "Alpha because of Beta"])
def test_negated_speculative_or_reversed_causal_links_are_not_traversable(text):
    records, complete = grounded_records(packet(text, "Beta", kind="link", subject="Alpha", attribute="causes"), text, "user", 100)
    assert records == [] and not complete


def test_path_obligations_include_intermediate_source_and_tenant_filter(tmp_path):
    a, b, foreign = "Alpha causes Beta", "Beta causes Gamma", "Gamma causes Delta"
    model = IndexedModel({t: packet(t, target, kind="link", subject=subject, attribute="causes")
                          for t, subject, target in [(a, "Alpha", "Beta"), (b, "Beta", "Gamma"), (foreign, "Gamma", "Delta")]}, intent="multi_hop")
    model.path = {"start": "Alpha", "end": "Gamma", "relation": "causes"}
    store = store_for(tmp_path, model)
    add(store, a, 1, 100)
    add(store, b, 2, 200, session="s2")
    add(store, foreign, 3, 300, user="foreign")
    result = store.search(user_id="u1", query="Why Alpha causes Gamma", top_k=100)
    assert {r.id for r in result} == {"mem_1", "mem_2"}
    model.path["end"] = "Delta"
    store.search(user_id="u1", query="Alpha to Delta", top_k=100)
    assert store.last_retrieval_trace["structured_memory"]["operators"]["path"]["status"] == "not_found"


def test_one_call_extraction_and_index_planner_contracts(monkeypatch):
    model = object.__new__(MemoryModel)
    calls = []
    def response(prompt, payload, **kwargs):
        calls.append((prompt, payload, kwargs))
        return {"complete": True, "facts": [], "records": [], "memory_requests": {"state_slots": ["known"]}}
    monkeypatch.setattr(model, "_json_response", response)
    model.extract_structured_memory("Alice status approved.", "user", 100)
    plan = model.plan_indexed_query("Alice status", [], {"groups": [{"group_id": "known"}]})
    assert len(calls) == 2
    assert "memory_catalog" in calls[1][1]
    assert plan["memory_requests"]["state_slots"] == ["known"]
    assert calls[0][2]["max_tokens"] == 4096


def test_specialist_trace_contains_no_raw_values_or_operator_statistics(tmp_path):
    text = "Alice status now secret-approved."
    trace = tmp_path / "trace.jsonl"
    model = IndexedModel({text: packet(text, "secret-approved", mode="update")})
    store = store_for(tmp_path, model, trace)
    add(store, text, 1, 100)
    store.search(user_id="u1", query="Alice status", top_k=100)
    data = trace.read_text(encoding="utf-8")
    assert "Alice" not in data and "secret" not in data
    assert "structured_memory" in json.loads(data)


def test_structured_flag_is_reversible_and_rejects_incompatible_paths(tmp_path):
    with pytest.raises(ValueError, match="requires evidence_selection"):
        MemoryStore(str(tmp_path / "bad.db"), structured_memory=True)
    store = MemoryStore(str(tmp_path / "legacy.db"))
    store.initialize()
    with store._connection() as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='structured_records'").fetchone() is None


def test_clipped_quote_and_hypothesis_context_cannot_create_an_effective_state():
    inner = "Alice status now approved"
    for text in ['"' + inner + '"', "Maybe " + inner, "The note says: " + inner, "It is not true: " + inner]:
        rows, complete = grounded_records(packet(inner, "approved", mode="update"), text, "user", 100)
        assert complete and rows[0]["modality"] == "hypothetical"
        rows[0].update(source_id=1)
        assert resolve_states(rows)["required"] == []


def test_before_cycle_is_reported_as_conflict_instead_of_asserted_order():
    links = [record("Alpha before Beta", "Beta", 1, kind="link", subject="Alpha", attribute="before"),
             record("Beta before Alpha", "Alpha", 2, kind="link", subject="Beta", attribute="before")]
    result = directed_path(links, "Alpha", "Beta", "before")
    assert result["status"] == "conflicting_order" and set(result["required"]) == {1, 2}


def test_inverted_or_disjunctive_rule_is_retrieval_only():
    for text in ["Shipping unless fragile=true: fee=standard.", "Shipping weight<=20 or fragile=true: fee=standard."]:
        rule = record(text, "fee=standard", kind="rule", subject="Shipping", attribute="fee", conditions=["fragile=true"])
        assert rule["conditions"] == []
        assert evaluate_rules([rule], "fragile=true")["status"] == "unknown_inputs"


def test_unsupported_numbers_and_ambiguous_observations_cannot_manufacture_exact_fit():
    from app.structured_memory import assignments
    assert assignments("x=1e9; y=20kg") == {}
    assert assignments("x=1; x=2; y=3.") == {"y": "3"}
    data = [record(f"Sensor x={i}; y={2*i}.", f"x={i}; y={2*i}", i, kind="observation", subject="Sensor", attribute="measurement") for i in (1, 2, 3)]
    bad = record("Sensor x=NaN; y=6.", "x=NaN; y=6", 4, kind="observation", subject="Sensor", attribute="measurement")
    result = observation_summary(data + [bad])
    assert result["exact_affine_fit"] is None and 4 in result["required"]
    rule = record("Shipping weight<=20: fee=standard.", "fee=standard", kind="rule", subject="Shipping", attribute="fee", conditions=["weight<=20"])
    assert evaluate_rules([rule], "weight=NaN")["status"] == "unknown_inputs"


def test_asof_requires_literal_query_date_and_cannot_include_later_sources(tmp_path):
    old, new = "Alice status pending.", "Alice status now approved."
    model = IndexedModel({old: packet(old, "pending"), new: packet(new, "approved", mode="update")},
                         mode="as_of", as_of="2024-01-01")
    store = store_for(tmp_path, model)
    add(store, old, 1, int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()))
    add(store, new, 2, int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()))
    assert [r.id for r in store.search(user_id="u1", query="Alice status as of 2024-01-01", top_k=100)] == ["mem_1"]
    # A hallucinated date absent from the query cannot authorize exclusions.
    assert {r.id for r in store.search(user_id="u1", query="Alice status before some earlier point", top_k=100)} == {"mem_1", "mem_2"}


def test_bad_structured_annotation_preserves_raw_ingestion_without_retry(tmp_path):
    class BrokenModel(IndexedModel):
        def extract_structured_memory(self, *args, **kwargs):
            self.extract_calls += 1
            raise RuntimeError("truncated output")
    model = BrokenModel({})
    store = store_for(tmp_path, model)
    add(store, "Alice status approved.", 1)
    with store._connection() as connection:
        assert connection.execute("SELECT content FROM raw_messages").fetchone()[0] == "Alice status approved."
        assert connection.execute("SELECT complete FROM structured_sources").fetchone()[0] == 0
    assert model.extract_calls == 1


def test_state_authority_is_part_of_the_slot_and_foreign_records_never_enter_rollup(tmp_path):
    text = "Alice status approved."
    model = IndexedModel({text: packet(text, "approved")})
    store = store_for(tmp_path, model)
    add(store, text, 1)
    store.add(AddRequest(request_id="assistant-source", user_id="u1", session_id="s1",
                        messages=[Message(role="assistant", content=text, timestamp=200)]))
    add(store, text, 3, user="other")
    with store._connection() as connection:
        catalog = StructuredMemoryIndex.catalog(connection, "u1")
        assert len(catalog["groups"]) == 2
        assert {g["authority"] for g in catalog["groups"]} == {"user", "assistant"}
        sources = {source for row in connection.execute("SELECT source_ids FROM structured_rollups WHERE user_id='u1'") for source in json.loads(row[0])}
        assert sources == {1, 2}


def test_privacy_only_unrequested_source_is_not_reintroduced_on_selector_failure(tmp_path):
    fact, secret = "Alice joined the lab.", "Alice password=private."
    class FailingSelector(IndexedModel):
        def select_evidence(self, *args):
            self.select_calls += 1
            raise RuntimeError("unavailable")
    model = FailingSelector({fact: packet(fact, "lab", kind="fact", attribute="work"),
                             secret: packet(secret, "private", kind="privacy", attribute="password")},
                            intent="summary", subjects=["alice"])
    store = store_for(tmp_path, model)
    add(store, fact, 1)
    add(store, secret, 2)
    results = store.search(user_id="u1", query="Summarize Alice work", top_k=100)
    assert [r.id for r in results] == ["mem_1"]
    assert store.last_retrieval_trace["structured_memory"]["blocked_ids"] == ["mem_2"]
    # Explicitly naming a field makes it eligible, not automatically selected.
    model.intent = "fact"
    store.search(user_id="u1", query="Alice password", top_k=100)
    assert store.last_retrieval_trace["structured_memory"]["blocked_ids"] == []


def test_mixed_necessary_source_remains_original_and_summary_limits_are_explicit(tmp_path):
    text = "Alice status approved; password=private."
    data = packet(text, "approved")
    private = packet(text, "private", kind="privacy", attribute="password")["records"][0]
    data["records"].append(private)
    model = IndexedModel({text: data})
    store = store_for(tmp_path, model)
    add(store, text, 1)
    results = store.search(user_id="u1", query="Alice current status", top_k=100)
    assert results[0].content == text  # No hidden field-level redaction of originals.
    assert store.last_retrieval_trace["structured_memory"]["blocked_ids"] == []
    decision = {"applied": True, "selected_ids": ["mem_1"]}
    assert enforce_obligations(decision, {"coverage_limited": True}, ["mem_1"])["reason"] == "specialist_catalog_incomplete"


def test_query_relevant_slot_survives_catalog_cap(tmp_path):
    texts = [f"Actor{i} status approved." for i in range(140)]
    model = IndexedModel({text: packet(text, "approved", subject=f"Actor{i}") for i, text in enumerate(texts)})
    store = store_for(tmp_path, model)
    for i, text in enumerate(texts):
        add(store, text, i)
    with store._connection() as connection:
        default = StructuredMemoryIndex.catalog(connection, "u1")
        omitted = next(text.split()[0] for text in texts if normalize_name(text.split()[0]) not in
                       {group["subject"] for group in default["groups"]})
        targeted = StructuredMemoryIndex.catalog(connection, "u1", f"{omitted} current status")
        assert len(targeted["groups"]) <= 128
        assert normalize_name(omitted) in {group["subject"] for group in targeted["groups"]}


def normalize_name(value):
    return value.casefold()
