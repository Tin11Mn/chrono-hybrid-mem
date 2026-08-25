import sqlite3

from evaluation.evidence_graph.p3a_gate import (
    build_relation_subset_manifest,
    evaluate_p3a_gate,
    P3A_FORMAL_DATASET_SHA256,
    RELATION_SUBSET_POLICY,
    relation_subset_manifest_sha256,
)


def test_relation_subset_is_frozen_from_unique_uncapped_gold_adjacency(tmp_path):
    database = tmp_path / "prepared.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE raw_messages(id INTEGER PRIMARY KEY, user_id TEXT, content TEXT);
        CREATE TABLE graph_entities(
            id INTEGER PRIMARY KEY, user_id TEXT, canonical_name TEXT
        );
        CREATE TABLE graph_aliases(
            entity_id INTEGER, user_id TEXT, normalized_alias TEXT
        );
        CREATE TABLE graph_edges(
            id INTEGER PRIMARY KEY, user_id TEXT, subject_entity_id INTEGER,
            object_entity_id INTEGER, source_message_id INTEGER, predicate TEXT
        );
        INSERT INTO raw_messages VALUES (1, 'user-a', 'Alice works at Acme.');
        INSERT INTO raw_messages VALUES (2, 'user-a', 'Another Jordan memory.');
        INSERT INTO graph_entities VALUES (10, 'user-a', 'alice');
        INSERT INTO graph_entities VALUES (11, 'user-a', 'acme');
        INSERT INTO graph_entities VALUES (12, 'user-a', 'jordan');
        INSERT INTO graph_entities VALUES (13, 'user-a', 'jordan');
        INSERT INTO graph_edges VALUES (20, 'user-a', 10, 11, 1, 'works_at');
        """
    )
    connection.commit()
    connection.close()
    manifest = {
        "manifest_sha256": "manifest",
        "questions": [
            {
                "case_id": "case-a", "sample_id": "sample", "user_id": "user-a",
                "query_key": "query-a", "gold_dia_ids": ["D1:1"],
                "question": "Where does Alice work?",
            },
            {
                "case_id": "case-b", "sample_id": "sample", "user_id": "user-a",
                "query_key": "query-b", "gold_dia_ids": ["D1:2"],
                "question": "Where does Jordan work?",
            },
        ],
    }
    source_map = [
        {"sample_id": "sample", "dia_id": "D1:1", "user_id": "user-a", "mem_id": "mem_1"},
        {"sample_id": "sample", "dia_id": "D1:2", "user_id": "user-a", "mem_id": "mem_2"},
    ]
    plans = {
        "query-a": {"entities": ["Alice"], "evidence_needs": ["find employer"]},
        "query-b": {"entities": ["Jordan"], "evidence_needs": ["find employer"]},
    }

    result = build_relation_subset_manifest(database, manifest, source_map, plans)

    assert result["policy"] == RELATION_SUBSET_POLICY
    assert result["relation_questions"] == 2
    assert result["records"][0]["relation_subset"] is True
    assert result["records"][0]["supporting_edges"][0]["edge_id"] == "20"
    # Ambiguity is a coverage failure to measure, not grounds for selecting the
    # hard case out of the relation subset.
    assert result["records"][1]["relation_subset"] is True
    assert result["records"][1]["requested_seeds"][0]["status"] == "ambiguous"
    serialized = str(result)
    assert "rerank_pool" not in serialized
    assert "final_ids" not in serialized


def _paired_fixture(question_count=200):
    traces = []
    subset_records = []
    for index in range(question_count):
        case_id = f"case-{index}"
        gold_mem = f"mem_{index + 1}"
        baseline_hit = int(index < 100)
        graph_hit = int(index < 104)
        p1_pool = [gold_mem] if index < 150 else ["mem_9999"]
        on_pool = list(p1_pool)
        graph_candidates = []
        reserved = []
        if index == 150:
            on_pool = [gold_mem]
            graph_candidates = [gold_mem]
            reserved = [gold_mem]
        storage_common = {
            "p1_top30_ids": p1_pool,
            "rerank_pool_ids": p1_pool,
            "graph_candidate_ids": [],
            "graph_reserved_ids": [],
            "requested_seeds": ["Alice"],
            "resolved_seeds": [{"entity_id": "1"}],
            "unresolved_seeds": [],
        }
        graph_storage = dict(storage_common)
        graph_storage.update({
            "rerank_pool_ids": on_pool,
            "graph_candidate_ids": graph_candidates,
            "graph_reserved_ids": reserved,
            "graph_paths": (
                [{"source_message_ids": [gold_mem]}]
                if graph_candidates else []
            ),
            "edge_diagnostics": {
                "seed_limit": 6,
                "edge_limit_per_seed": 20,
                "candidate_limit": 20,
                "candidate_count": len(graph_candidates),
                "path_count": len(graph_candidates),
                "edges_considered": 1,
                "duplicate_edges_skipped": int(not graph_candidates),
                "candidate_cap_reached": False,
                "edges_fetched_by_seed": [{
                    "entity_id": "1",
                    "normalized": "alice",
                    "count": 1,
                    "limit_reached": False,
                }],
                "edge_visit_seed_ids": ["1"],
            },
        })
        traces.append({
            "case_id": case_id,
            "baseline": {"hit_at_k": {"1": baseline_hit}, "storage": storage_common},
            "graph_on": {"hit_at_k": {"1": graph_hit}, "storage": graph_storage},
        })
        subset_records.append({
            "case_id": case_id,
            "relation_subset": index < 40,
            "gold_mem_ids": [gold_mem],
        })
    paired = {
        "manifest_sha256": "manifest",
        "dataset_sha256": P3A_FORMAL_DATASET_SHA256,
        "question_offset": 0,
        "max_questions": 200,
        "question_traces": traces,
        "graph_configuration": {
            "max_hops": 1,
            "temporal": False,
            "rrf_weight": 0.025,
            "max_candidates": 20,
            "rerank_quota": 4,
            "rerank_limit": 30,
        },
        "storage_trace_complete": True,
        "database_unchanged": True,
        "formal_integrity_audit": {
            "violations": 0, "unsupported_traversed_edges": 0,
        },
        "search_calls": {"additional_graph_search_calls": 0},
        "baseline": {"hit_at_k": {"10": 0.74}, "evidence_recall_at_k": {"10": 0.60}},
        "graph_on": {"hit_at_k": {"10": 0.74}, "evidence_recall_at_k": {"10": 0.60}},
    }
    subset = {
        "schema_version": 1,
        "source_manifest_sha256": "manifest",
        "policy": RELATION_SUBSET_POLICY,
        "records": subset_records,
    }
    subset["subset_manifest_sha256"] = relation_subset_manifest_sha256(subset)
    return paired, subset


def test_formal_gate_uses_raw_win_loss_counts_and_candidate_oracle():
    paired, subset = _paired_fixture()

    gate = evaluate_p3a_gate(paired, subset)

    assert gate["decision"] == "PROMOTE_P3_A"
    assert (gate["wins"], gate["losses"], gate["net_hit_at_1"]) == (4, 0, 4)
    assert gate["overall_branch_pass"] is True
    assert gate["hard_checks"]["strict_graph_new_gold_observed"] is True
    assert gate["candidate_oracle"]["hit_at_30_on"] > gate["candidate_oracle"]["hit_at_30_off"]


def test_fixed20_can_only_pass_mechanics_not_promote():
    paired, subset = _paired_fixture(question_count=20)

    gate = evaluate_p3a_gate(paired, subset)

    assert gate["decision"] == "MECHANICS_PASS"
    assert gate["formal_scope"] is False


def test_gate_rejects_non_frozen_graph_configuration():
    paired, subset = _paired_fixture()
    paired["graph_configuration"]["rrf_weight"] = 0.026

    gate = evaluate_p3a_gate(paired, subset)

    assert gate["decision"] == "REJECT_P3_A"
    assert gate["hard_checks"]["frozen_graph_configuration"] is False
    assert any("rrf_weight" in item for item in gate["configuration_violations"])


def test_gate_rejects_candidate_cap_or_round_robin_trace_tampering():
    paired, subset = _paired_fixture()
    storage = paired["question_traces"][0]["graph_on"]["storage"]
    storage["edge_diagnostics"]["candidate_count"] = 21

    gate = evaluate_p3a_gate(paired, subset)

    assert gate["decision"] == "REJECT_P3_A"
    assert gate["hard_checks"]["graph_trace_mechanics"] is False
    assert any("candidate" in item for item in gate["graph_trace_mechanics_violations"])

    paired, subset = _paired_fixture()
    storage = paired["question_traces"][0]["graph_on"]["storage"]
    storage["edge_diagnostics"]["edge_visit_seed_ids"] = ["not-the-seed"]

    gate = evaluate_p3a_gate(paired, subset)

    assert gate["decision"] == "REJECT_P3_A"
    assert gate["hard_checks"]["graph_trace_mechanics"] is False
    assert any("round-robin" in item for item in gate["graph_trace_mechanics_violations"])


def test_gate_rejects_sparse_relation_subset_or_integrity_failure():
    paired, subset = _paired_fixture()
    for index, record in enumerate(subset["records"]):
        record["relation_subset"] = index < 20
    subset["subset_manifest_sha256"] = relation_subset_manifest_sha256(subset)
    paired["formal_integrity_audit"]["unsupported_traversed_edges"] = 1

    gate = evaluate_p3a_gate(paired, subset)

    assert gate["decision"] == "REJECT_P3_A"
    assert gate["relation_branch_pass"] is False
    assert gate["hard_checks"]["unsupported_traversed_edges_zero"] is False


def test_gate_rejects_tampered_or_cross_scope_relation_manifest():
    paired, subset = _paired_fixture()
    subset["records"][0]["relation_subset"] = False

    try:
        evaluate_p3a_gate(paired, subset)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("tampered relation subset was accepted")

    subset["subset_manifest_sha256"] = relation_subset_manifest_sha256(subset)
    paired["manifest_sha256"] = "other-scope"
    try:
        evaluate_p3a_gate(paired, subset)
    except ValueError as error:
        assert "scope manifest mismatch" in str(error)
    else:
        raise AssertionError("cross-scope relation subset was accepted")


def test_two_hundred_question_cherry_picked_window_cannot_be_formal():
    paired, subset = _paired_fixture()
    paired["question_offset"] = 20

    gate = evaluate_p3a_gate(paired, subset)

    assert gate["formal_scope"] is False
    assert gate["formal_scope_checks"]["question_offset"] is False
    assert gate["decision"] == "REJECT_P3_A"
