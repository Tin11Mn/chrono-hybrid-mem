import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

import evaluation.evidence_graph.run_locomo_paired as paired_module
from app.storage import MemoryStore
from evaluation.evidence_graph.run_locomo_paired import (
    _audit_graph_trace,
    _audit_arm_integrity,
    _normalize_storage_trace,
    build_prepared_database,
    ensure_extraction_cache,
    extraction_cache_fingerprint,
    freeze_structured_plans,
    load_dataset_manifest,
    load_prepared_database,
    logical_database_digest,
    materialization_contract_fingerprint,
    run_paired_evaluation,
    run_paired_evaluation_with_gate,
    score_question_rankings,
)
from evaluation.evidence_graph.support_witness_audit import (
    audit_graph_trace_paths,
    audit_persisted_graph_support,
    normalize_witness_source,
    normalized_source_span_sha256,
)


FAKE_INGEST_ARTIFACT_SHA256 = "a" * 64


def synthetic_locomo():
    return [
        {
            "sample_id": "sample-a",
            "conversation": {
                "session_1_date_time": "1:15 PM on 8 May, 2023",
                "session_1": [
                    {
                        "speaker": "Ari",
                        "dia_id": "D1:1",
                        "text": "Alice works at Acme.",
                    },
                    {
                        "speaker": "Ari",
                        "dia_id": "D1:2",
                        "text": "A quiet filler follows the relation.",
                        "blip_caption": "a small blue notebook",
                        "query": "private crawler terms",
                    },
                ],
                "session_2_date_time": "2:30 PM on 9 May, 2023",
                "session_2": [
                    {
                        "speaker": "Bela",
                        "dia_id": "D2:1",
                        "text": "Needle distractor only.",
                    }
                ],
            },
            "qa": [
                {
                    "question": "Needle relation lookup?",
                    "answer": "Acme",
                    "evidence": ["D1:1"],
                    "category": 1,
                }
            ],
        }
    ]


def write_dataset(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(synthetic_locomo()), encoding="utf-8")
    return path


class FakeExtractionModel:
    def __init__(self):
        self.call_count = 0
        self.seen = []
        self._lock = threading.Lock()

    def extract_memory(self, content, speaker="", timestamp=None):
        with self._lock:
            self.call_count += 1
            self.seen.append((content, speaker, timestamp))
        if "Alice works at Acme" not in content:
            return {"facts": [], "entities": [], "relations": []}
        return {
            "facts": ["Alice works at Acme."],
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Acme", "type": "organization"},
            ],
            "relations": [
                {
                    "subject": "Alice",
                    "subject_type": "person",
                    "relation": "works_at",
                    "object": "Acme",
                    "object_type": "organization",
                    "explicit": True,
                    "state_change": "assert",
                    "temporal_status": None,
                }
            ],
        }


class FailOnceExtractionModel(FakeExtractionModel):
    def __init__(self, failing_text):
        super().__init__()
        self.failing_text = failing_text
        self.failed = False

    def extract_memory(self, content, speaker="", timestamp=None):
        if self.failing_text in content and not self.failed:
            with self._lock:
                self.call_count += 1
                self.seen.append((content, speaker, timestamp))
                self.failed = True
            raise RuntimeError("synthetic extraction failure")
        return super().extract_memory(content, speaker=speaker, timestamp=timestamp)


class FakePlanRankModel:
    def __init__(self):
        self.plan_calls = 0
        self.rank_calls = 0
        self.call_count = 0
        self.truncated_calls = 0
        self.finish_reason_counts = {}

    def plan_query_structured(self, query, options):
        self.plan_calls += 1
        self.call_count += 1
        return {
            "intent": "fact",
            "core_terms": ["needle"],
            "expansion_terms": [],
            "entities": ["Alice"],
            "temporal_cues": [],
            "evidence_needs": ["direct workplace relation"],
        }

    def rank_candidates(self, query, options, candidates):
        self.rank_calls += 1
        self.call_count += 1

        def priority(candidate):
            content = candidate["content"]
            if "Untrusted graph retrieval metadata" in content:
                return 0
            if "Needle distractor" in content:
                return 1
            return 2

        return [item["id"] for item in sorted(candidates, key=priority)]


def build_cache_and_manifest(tmp_path, *, workers=2):
    dataset = write_dataset(tmp_path)
    manifest = load_dataset_manifest(dataset, max_questions=1)
    fingerprint = extraction_cache_fingerprint(
        manifest,
        model_fingerprint="fake-extractor-v1",
        ingest_artifact_fingerprint=FAKE_INGEST_ARTIFACT_SHA256,
        contract_fingerprint="fake-contract-v1",
    )
    model = FakeExtractionModel()
    cache_path = tmp_path / "extractions.json"
    cache = ensure_extraction_cache(
        manifest,
        cache_path,
        fingerprint=fingerprint,
        model=model,
        mode="build",
        workers=workers,
    )
    return manifest, fingerprint, model, cache_path, cache


def test_authoritative_manifest_hashes_dataset_and_preserves_full_sample(tmp_path):
    dataset = write_dataset(tmp_path)
    manifest = load_dataset_manifest(dataset, max_questions=1)

    assert manifest["message_count"] == 3
    assert manifest["question_count"] == 1
    assert manifest["dataset_sha256"]
    assert manifest["manifest_sha256"]
    messages = manifest["samples"][0]["sessions"][0]["messages"]
    assert [item["sequence"] for item in messages] == [0, 1]
    assert "Shared image: a small blue notebook" in messages[1]["content"]
    assert "private crawler terms" not in messages[1]["content"]
    assert manifest["questions"][0]["gold_dia_ids"] == ["D1:1"]


def test_artifact_fingerprint_and_local_base_url_fail_closed(tmp_path):
    manifest = load_dataset_manifest(write_dataset(tmp_path), max_questions=1)
    with pytest.raises(ValueError, match="64-character SHA-256"):
        extraction_cache_fingerprint(
            manifest,
            model_fingerprint="fake-extractor-v1",
            ingest_artifact_fingerprint="generic-local-name",
            contract_fingerprint="fake-contract-v1",
        )
    with pytest.raises(SystemExit, match="explicit loopback"):
        paired_module._require_loopback_base_url(None, "--search-base-url")
    with pytest.raises(SystemExit, match="must use http://127.0.0.1"):
        paired_module._require_loopback_base_url(
            "https://api.openai.com/v1", "--search-base-url"
        )


def test_composite_cache_builds_once_and_reuses_without_a_model(tmp_path):
    manifest, fingerprint, model, cache_path, cache = build_cache_and_manifest(
        tmp_path
    )

    assert model.call_count == manifest["message_count"] == 3
    assert cache["model_calls"] == 3
    assert cache["one_call_per_missing_message"] is True
    assert len({item[0] for item in model.seen}) == 3

    reused = ensure_extraction_cache(
        manifest,
        cache_path,
        fingerprint=fingerprint,
        model=None,
        mode="reuse",
        workers=2,
    )
    assert reused["model_calls"] == 0
    assert reused["cache_hits"] == 3


def test_composite_cache_fails_loudly_on_fingerprint_missing_and_duplicates(tmp_path):
    manifest, fingerprint, _, cache_path, _ = build_cache_and_manifest(tmp_path)

    wrong = dict(fingerprint)
    wrong["model_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ensure_extraction_cache(
            manifest,
            cache_path,
            fingerprint=wrong,
            model=None,
            mode="reuse",
        )

    wrong_artifact = dict(fingerprint)
    wrong_artifact["ingest_artifact_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ensure_extraction_cache(
            manifest,
            cache_path,
            fingerprint=wrong_artifact,
            model=None,
            mode="reuse",
        )

    value = json.loads(cache_path.read_text(encoding="utf-8"))
    original_records = list(value["records"])
    value["records"] = original_records[:-1]
    cache_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 1 selected messages"):
        ensure_extraction_cache(
            manifest,
            cache_path,
            fingerprint=fingerprint,
            model=None,
            mode="reuse",
        )

    value["records"] = original_records + [dict(original_records[0])]
    cache_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate extraction cache"):
        ensure_extraction_cache(
            manifest,
            cache_path,
            fingerprint=fingerprint,
            model=None,
            mode="reuse",
        )


def test_failed_extraction_checkpoints_successes_and_extend_only_fills_missing(
    tmp_path,
):
    dataset = write_dataset(tmp_path)
    manifest = load_dataset_manifest(dataset, max_questions=1)
    fingerprint = extraction_cache_fingerprint(
        manifest,
        model_fingerprint="fake-extractor-v1",
        ingest_artifact_fingerprint=FAKE_INGEST_ARTIFACT_SHA256,
        contract_fingerprint="fake-contract-v1",
    )
    cache_path = tmp_path / "recoverable-extractions.json"
    failing = FailOnceExtractionModel("quiet filler")

    with pytest.raises(RuntimeError, match="synthetic extraction failure"):
        ensure_extraction_cache(
            manifest,
            cache_path,
            fingerprint=fingerprint,
            model=failing,
            mode="build",
            workers=2,
        )

    checkpoint = json.loads(cache_path.read_text(encoding="utf-8"))
    assert failing.call_count == manifest["message_count"] == 3
    assert len(checkpoint["records"]) == 2
    failed_source_key = next(
        message["source_key"]
        for sample in manifest["samples"]
        for session in sample["sessions"]
        for message in session["messages"]
        if "quiet filler" in message["content"]
    )
    assert failed_source_key not in {
        record["source_key"] for record in checkpoint["records"]
    }

    resumed = FakeExtractionModel()
    completed = ensure_extraction_cache(
        manifest,
        cache_path,
        fingerprint=fingerprint,
        model=resumed,
        mode="extend",
        workers=2,
    )

    assert resumed.call_count == 1
    assert "quiet filler" in resumed.seen[0][0]
    assert completed["cache_hits"] == 2
    assert completed["model_calls"] == 1
    assert completed["cache_records"] == 3
    assert completed["one_call_per_missing_message"] is True


def test_session_bundled_replay_builds_strict_source_map_and_context(tmp_path):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )

    assert prepared["message_count"] == 3
    assert prepared["replay_extract_calls"] == 3
    assert prepared["one_replay_call_per_message"] is True
    assert [item["dia_id"] for item in prepared["source_map"]] == [
        "D1:1",
        "D1:2",
        "D2:1",
    ]
    assert len({item["mem_id"] for item in prepared["source_map"]}) == 3
    assert prepared["database_digest"] == logical_database_digest(database)
    assert prepared["materialization_contract_fingerprint"] == (
        materialization_contract_fingerprint()
    )
    assert prepared["graph_edge_count"] == 1
    assert prepared["graph_edge_support_count"] == 1
    assert prepared["all_graph_edges_witnessed"] is True
    assert prepared["graph_entity_declaration_count"] == 2
    assert prepared["graph_entity_mention_count"] == 2
    assert prepared["mention_source_coverage"] == pytest.approx(1 / 3)
    assert prepared["all_entity_mentions_witnessed"] is True
    assert load_prepared_database(manifest, database)["database_digest"] == prepared[
        "database_digest"
    ]

    first_mem = int(prepared["source_map"][0]["mem_id"].split("_", 1)[1])
    connection = sqlite3.connect(database)
    try:
        context = connection.execute(
            "SELECT content FROM context_fts WHERE message_id = ?", (first_mem,)
        ).fetchone()[0]
        assert "Alice works at Acme" in context
        assert "quiet filler" in context
        assert connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
    finally:
        connection.close()


def test_prepared_database_reuse_rejects_logical_mutation(tmp_path):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE raw_messages SET created_at = 'tampered' WHERE id = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="logical digest mismatch"):
        load_prepared_database(manifest, database)


def test_prepared_database_rejects_tampered_entity_mention_before_reuse(tmp_path):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE graph_entity_mentions SET source_span_sha256 = ? WHERE id = 1",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="entity-mention audit mismatch"):
        load_prepared_database(manifest, database)


def test_prepared_database_rejects_tampered_or_legacy_materialization_contract(
    tmp_path,
):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    sidecar = Path(prepared["sidecar_path"])
    original = json.loads(sidecar.read_text(encoding="utf-8"))

    tampered = dict(original)
    tampered["materialization_contract_fingerprint"] = "tampered"
    sidecar.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="materialization contract mismatch"):
        load_prepared_database(manifest, database)

    legacy = dict(original)
    legacy.pop("materialization_contract_fingerprint")
    sidecar.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="materialization contract mismatch"):
        load_prepared_database(manifest, database)


def test_paired_runner_uses_frozen_plan_ids_and_leaves_database_unchanged(tmp_path):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    search_model = FakePlanRankModel()
    plans = freeze_structured_plans(manifest["questions"], search_model)
    assert search_model.plan_calls == 1

    report = run_paired_evaluation(
        manifest,
        database,
        prepared["source_map"],
        frozen_plans=plans,
        rank_model=search_model,
        top_ks=(1, 3),
        require_complete_trace=False,
    )

    assert report["baseline"]["hit_at_k"]["1"] == 0.0
    assert report["graph_on"]["hit_at_k"]["1"] == 1.0
    assert report["delta_hit_at_k"]["1"] == 1.0
    assert report["dataset_sha256"] == manifest["dataset_sha256"]
    assert report["question_offset"] == 0
    assert report["max_questions"] == 1
    assert report["paired_recovered_gold_evidence_count_at_k"]["1"] == 1
    assert report["paired_lost_gold_evidence_count_at_k"]["1"] == 0
    assert report["database_unchanged"] is True
    assert report["database_digest_before"] == report["database_digest_after"]
    assert report["search_calls"]["additional_graph_search_calls"] == 0
    assert report["search_calls"]["baseline_logical_plan_calls"] == 1
    assert report["search_calls"]["graph_logical_plan_calls"] == 1
    assert report["formal_integrity_audit"]["unsupported_traversed_edges"] == 0
    assert report["formal_integrity_audit"]["traversed_edges"] == 1
    assert report["formal_integrity_audit"]["persisted_graph_edges"] == 1
    assert report["formal_integrity_audit"]["persisted_graph_edge_supports"] == 1
    assert (
        report["formal_integrity_audit"]["all_persisted_graph_edges_witnessed"]
        is True
    )
    assert report["formal_integrity_audit"]["violations"] == 0
    # Diagnostic mode must report whether the installed storage exposes the
    # complete formal trace contract.  This may become true once the production
    # storage implementation gains every required channel/provenance field.
    assert isinstance(report["storage_trace_complete"], bool)
    trace = report["question_traces"][0]
    assert trace["paired_recovered_gold_at_k"]["1"] == ["D1:1"]
    assert trace["paired_lost_gold_at_k"]["1"] == []
    assert trace["baseline"]["ranked_dia_ids"][0] == "D2:1"
    assert trace["graph_on"]["ranked_dia_ids"][0] == "D1:1"
    audit = trace["formal_integrity_audit"]
    assert trace["graph_on"]["storage"]["requested_seeds"] == ["Alice"]
    assert len(trace["graph_on"]["storage"]["resolved_seeds"]) == 1
    assert trace["graph_on"]["storage"]["unresolved_seeds"] == []
    edge_diagnostics = trace["graph_on"]["storage"]["edge_diagnostics"]
    assert edge_diagnostics["candidate_limit"] == 20
    assert edge_diagnostics["candidate_count"] <= edge_diagnostics["candidate_limit"]
    assert len(edge_diagnostics["edge_visit_seed_ids"]) == (
        edge_diagnostics["edges_considered"]
    )
    assert all(
        item["count"] <= edge_diagnostics["edge_limit_per_seed"]
        for item in edge_diagnostics["edges_fetched_by_seed"]
    )
    assert audit["baseline"]["rerank_pool_unique_ids"] is True
    assert audit["baseline"]["rerank_pool_unique_casefold_content"] is True
    assert audit["graph_on"]["rerank_pool_unique_ids"] is True
    assert audit["graph_on"]["rerank_pool_unique_casefold_content"] is True
    assert audit["graph_provenance"]["unsupported_traversed_edges"] == 0
    assert audit["baseline"]["final_content_matches_raw"] == len(
        trace["baseline"]["ranked_mem_ids"]
    )
    assert audit["graph_on"]["final_content_matches_raw"] == len(
        trace["graph_on"]["ranked_mem_ids"]
    )
    assert search_model.rank_calls == 2


def test_integrated_gate_freezes_relation_subset_before_any_search(
    tmp_path, monkeypatch
):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    search_model = FakePlanRankModel()
    plans = freeze_structured_plans(manifest["questions"], search_model)
    events = []
    original_subset_builder = paired_module.build_relation_subset_manifest
    original_search = MemoryStore.search

    def tracked_subset_builder(*args, **kwargs):
        events.append("subset")
        return original_subset_builder(*args, **kwargs)

    def tracked_search(*args, **kwargs):
        assert events == ["subset"] or events == ["subset", "search"]
        events.append("search")
        return original_search(*args, **kwargs)

    monkeypatch.setattr(
        paired_module,
        "build_relation_subset_manifest",
        tracked_subset_builder,
    )
    monkeypatch.setattr(MemoryStore, "search", tracked_search)

    integrated = run_paired_evaluation_with_gate(
        manifest,
        database,
        prepared["source_map"],
        frozen_plans=plans,
        rank_model=search_model,
        top_ks=(1, 3, 10),
        require_complete_trace=True,
    )

    assert events == ["subset", "search", "search"]
    subset = integrated["relation_subset_manifest"]
    assert subset["source_manifest_sha256"] == manifest["manifest_sha256"]
    assert len(subset["subset_manifest_sha256"]) == 64
    assert subset["relation_questions"] == 1
    assert integrated["p3a_gate"]["questions"] == 1
    assert integrated["p3a_gate"]["formal_scope"] is False
    assert integrated["p3a_gate"]["decision"] == "MECHANICS_PASS"
    assert integrated["p3a_gate"]["seed_diagnostics"]["requested"] == 1
    assert integrated["p3a_gate"]["seed_diagnostics"]["accounted"] == 1
    assert integrated["paired"]["database_unchanged"] is True
    assert search_model.plan_calls == 1
    assert search_model.rank_calls == 2


def test_integrated_gate_rejects_non_frozen_config_before_subset_or_search(
    tmp_path, monkeypatch
):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    search_model = FakePlanRankModel()
    plans = freeze_structured_plans(manifest["questions"], search_model)
    events = []

    def forbidden_subset(*args, **kwargs):
        events.append("subset")
        raise AssertionError("non-frozen configuration reached subset freeze")

    def forbidden_search(*args, **kwargs):
        events.append("search")
        raise AssertionError("non-frozen configuration reached Search")

    monkeypatch.setattr(
        paired_module, "build_relation_subset_manifest", forbidden_subset
    )
    monkeypatch.setattr(MemoryStore, "search", forbidden_search)

    with pytest.raises(ValueError, match="formal P3-A graph configuration"):
        run_paired_evaluation_with_gate(
            manifest,
            database,
            prepared["source_map"],
            frozen_plans=plans,
            rank_model=search_model,
            graph_rrf_weight=0.026,
        )

    assert events == []
    assert search_model.rank_calls == 0


def test_prepare_only_cli_writes_stage_without_constructing_search(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    cache_path = tmp_path / "prepare-cache.json"
    database = tmp_path / "prepare.db"
    output = tmp_path / "prepare-report.json"
    runtime_purposes = []
    search_calls = []
    plan_calls = []

    args = SimpleNamespace(
        dataset=str(dataset),
        extraction_cache=str(cache_path),
        prepared_db=str(database),
        output=str(output),
        max_questions=1,
        question_offset=0,
        cache_mode="build",
        prepared_mode="build",
        workers=2,
        ingest_model="fake-ingest",
        ingest_base_url="http://127.0.0.1:8081/v1",
        ingest_artifact_fingerprint=FAKE_INGEST_ARTIFACT_SHA256,
        search_model="must-not-construct",
        search_base_url="http://127.0.0.1:8082/v1",
        top_k=(1, 3, 10),
        graph_rrf_weight=0.025,
        graph_rerank_quota=4,
        graph_max_candidates=20,
        prepare_only=True,
        allow_incomplete_storage_trace=False,
    )

    def fake_runtime_model(*, model_name, base_url, purpose):
        runtime_purposes.append(purpose)
        if purpose != "ingest":
            raise AssertionError("prepare-only constructed a search model")
        return FakeExtractionModel()

    def forbidden_search(*args, **kwargs):
        search_calls.append((args, kwargs))
        raise AssertionError("prepare-only called Search")

    def forbidden_plans(*args, **kwargs):
        plan_calls.append((args, kwargs))
        raise AssertionError("prepare-only froze search plans")

    monkeypatch.setattr(paired_module, "parse_args", lambda: args)
    monkeypatch.setattr(paired_module, "_runtime_model", fake_runtime_model)
    monkeypatch.setattr(paired_module, "freeze_structured_plans", forbidden_plans)
    monkeypatch.setattr(MemoryStore, "search", forbidden_search)

    paired_module.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert runtime_purposes == ["ingest"]
    assert search_calls == []
    assert plan_calls == []
    assert report["stage"] == "prepare_only"
    assert set(report) == {
        "stage", "dataset", "extraction_cache", "prepared_database",
        "ingest_artifact_fingerprint", "extraction_model",
    }
    assert report["dataset"]["question_offset"] == 0
    assert report["dataset"]["max_questions"] == 1
    assert report["prepared_database"]["materialization_contract_fingerprint"] == (
        materialization_contract_fingerprint()
    )
    assert (
        report["ingest_artifact_fingerprint"]
        == FAKE_INGEST_ARTIFACT_SHA256
    )
    assert report["extraction_model"] == {
        "name": "fake-ingest",
        "base_url": "http://127.0.0.1:8081/v1",
        "constructed": True,
        "call_count": 3,
        "finish_reason_counts": {},
        "truncated_calls": 0,
    }
    assert "search_model" not in report
    assert "paired" not in report
    assert "p3a_gate" not in report


def _paired_main_args(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        dataset=str(write_dataset(tmp_path)),
        extraction_cache=str(tmp_path / "paired-cache.json"),
        prepared_db=str(tmp_path / "paired.db"),
        output=str(tmp_path / "paired-report.json"),
        max_questions=1,
        question_offset=0,
        cache_mode="build",
        prepared_mode="build",
        workers=1,
        ingest_model="fake-ingest",
        ingest_base_url="http://127.0.0.1:8081/v1",
        ingest_artifact_fingerprint=FAKE_INGEST_ARTIFACT_SHA256,
        search_model="fake-search",
        search_base_url="http://127.0.0.1:8082/v1",
        top_k=(1, 3, 10),
        graph_rrf_weight=0.025,
        graph_rerank_quota=4,
        graph_max_candidates=20,
        prepare_only=False,
        allow_incomplete_storage_trace=False,
    )


def test_main_hard_audits_plan_and_two_arm_external_model_calls(
    tmp_path, monkeypatch
):
    args = _paired_main_args(tmp_path)
    search_model = FakePlanRankModel()
    runtime_purposes = []

    def fake_runtime_model(*, model_name, base_url, purpose):
        runtime_purposes.append(purpose)
        return FakeExtractionModel() if purpose == "ingest" else search_model

    monkeypatch.setattr(paired_module, "parse_args", lambda: args)
    monkeypatch.setattr(paired_module, "_runtime_model", fake_runtime_model)

    paired_module.main()

    report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert runtime_purposes == ["ingest", "search"]
    assert report["search_model_call_audit"] == {
        "question_count": 1,
        "plan_external_calls": 1,
        "expected_plan_external_calls": 1,
        "plan_truncated_calls": 0,
        "paired_arm_external_calls": 2,
        "expected_paired_arm_external_calls": 2,
        "total_external_calls": 3,
        "expected_total_external_calls": 3,
        "paired_arm_truncated_calls": 0,
        "total_truncated_calls": 0,
    }
    assert report["search_model"]["call_count"] == 3
    assert report["search_model"]["truncated_calls"] == 0


def test_main_rejects_unaccounted_or_truncated_paired_arm_calls(
    tmp_path, monkeypatch
):
    class BadCallAccountingModel(FakePlanRankModel):
        def rank_candidates(self, query, options, candidates):
            previous_calls = self.call_count
            result = super().rank_candidates(query, options, candidates)
            self.call_count = previous_calls
            return result

    args = _paired_main_args(tmp_path / "uncounted")
    bad_calls = BadCallAccountingModel()

    def uncounted_runtime(*, model_name, base_url, purpose):
        return FakeExtractionModel() if purpose == "ingest" else bad_calls

    monkeypatch.setattr(paired_module, "parse_args", lambda: args)
    monkeypatch.setattr(paired_module, "_runtime_model", uncounted_runtime)
    with pytest.raises(RuntimeError, match="external call count mismatch"):
        paired_module.main()

    class TruncatedAccountingModel(FakePlanRankModel):
        def rank_candidates(self, query, options, candidates):
            result = super().rank_candidates(query, options, candidates)
            if self.rank_calls == 1:
                self.truncated_calls += 1
            return result

    args = _paired_main_args(tmp_path / "truncated")
    truncated = TruncatedAccountingModel()

    def truncated_runtime(*, model_name, base_url, purpose):
        return FakeExtractionModel() if purpose == "ingest" else truncated

    monkeypatch.setattr(paired_module, "parse_args", lambda: args)
    monkeypatch.setattr(paired_module, "_runtime_model", truncated_runtime)
    with pytest.raises(RuntimeError, match="truncated model responses"):
        paired_module.main()


def test_presearch_graph_audit_rejects_unsupported_edge_without_search(
    tmp_path, monkeypatch
):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    connection = sqlite3.connect(database)
    try:
        # Keep the edge structurally valid while making its predicate false in
        # the persisted source clause.  Search can still traverse it, so only
        # the independent lexical provenance guard can catch this mutation.
        connection.execute("UPDATE graph_edges SET predicate = 'likes'")
        connection.commit()
    finally:
        connection.close()

    model = FakePlanRankModel()
    plans = freeze_structured_plans(manifest["questions"], model)
    search_calls = []

    def forbidden_search(*args, **kwargs):
        search_calls.append((args, kwargs))
        raise AssertionError("invalid prepared graph reached Search")

    monkeypatch.setattr(MemoryStore, "search", forbidden_search)
    with pytest.raises(RuntimeError, match="formal prepared support audit failed"):
        run_paired_evaluation(
            manifest,
            database,
            prepared["source_map"],
            frozen_plans=plans,
            rank_model=model,
            top_ks=(1, 3),
            require_complete_trace=True,
        )
    assert search_calls == []
    assert model.rank_calls == 0


def test_independent_witness_audit_rejects_tampered_support_hash(tmp_path, monkeypatch):
    """A production boolean cannot certify a forged persisted witness."""

    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    before = audit_persisted_graph_support(database)
    assert before["all_edges_witnessed"] is True
    assert normalize_witness_source("Alice\u00a0Works\r\nAt\u2028Acme") == (
        "alice works at acme"
    )

    # The sidecar CHECK accepts any lower-case 64-hex value.  A forged hash is
    # therefore structurally plausible and must be rejected by the evaluator's
    # own canonical-source recomputation rather than any production decision.
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE graph_edge_support SET source_span_sha256 = ?",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()

    # This mirrors a compromised production helper.  The evaluator does not
    # import or consult it, so the forged witness remains rejected.
    import app.evidence_graph as production_graph

    monkeypatch.setattr(
        production_graph,
        "relation_is_textually_supported",
        lambda **_kwargs: True,
    )
    audit = audit_persisted_graph_support(database)
    assert audit["all_edges_witnessed"] is False
    assert audit["unsupported_edge_ids"] == ["edge_1"]
    assert any("SHA-256 mismatch" in item for item in audit["violations"])


def test_independent_witness_audit_rejects_source_scope_forgery(tmp_path):
    """A locally valid clause is unsafe when full-source scope withdraws it."""

    cases = (
        (
            "later-withdrawal",
            "Alice works at Acme. Alice does not.",
            ("later source sentence withdraws the same subject assertion",),
        ),
        (
            "forward-governance",
            "Ignore the next sentence. Alice works at Acme.",
            ("claimed clause is governed by a prior forward marker",),
        ),
        (
            "non-immediate-prior-governance",
            "Ignore the following statement. This is neutral context. Alice works at Acme.",
            (
                "claimed clause has source-level meta/reporting governance",
                "claimed clause is governed by a prior forward marker",
            ),
        ),
        (
            "partial-name-withdrawal",
            "Alice works at Acme. Ali does not.",
            ("later source sentence contains an explicit withdrawal/negation marker",),
        ),
        (
            "quoted-source",
            '"Alice works at Acme."',
            ("claimed clause lies in a quoted/bracketed source container",),
        ),
        (
            "unicode-angle-container",
            "〈Alice works at Acme.〉",
            ("claimed clause lies in a quoted/bracketed source container",),
        ),
    )
    clause = "alice works at acme."
    for label, raw_source, expected_violations in cases:
        case_path = tmp_path / label
        case_path.mkdir()
        manifest, fingerprint, _, _, cache = build_cache_and_manifest(case_path)
        database = case_path / "prepared.db"
        build_prepared_database(
            manifest,
            cache["records_by_source_key"],
            database,
            cache_fingerprint=fingerprint,
        )
        canonical_source = normalize_witness_source(raw_source)
        assert canonical_source is not None
        clause_start = canonical_source.index(clause)
        subject_start = canonical_source.index("alice", clause_start)
        predicate_start = canonical_source.index("works at", clause_start)
        object_start = canonical_source.index("acme", clause_start)
        connection = sqlite3.connect(database)
        try:
            connection.execute("UPDATE raw_messages SET content = ?", (raw_source,))
            connection.execute(
                """UPDATE graph_edge_support
                   SET source_end = ?, clause_start = ?, clause_end = ?,
                       subject_start = ?, subject_end = ?,
                       predicate_start = ?, predicate_end = ?,
                       object_start = ?, object_end = ?, source_span_sha256 = ?""",
                (
                    len(canonical_source),
                    clause_start,
                    clause_start + len(clause),
                    subject_start,
                    subject_start + len("alice"),
                    predicate_start,
                    predicate_start + len("works at"),
                    object_start,
                    object_start + len("acme"),
                    normalized_source_span_sha256(canonical_source),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        audit = audit_persisted_graph_support(database)
        assert audit["all_edges_witnessed"] is False
        for expected_violation in expected_violations:
            assert any(expected_violation in item for item in audit["violations"])


def test_trace_witness_is_reloaded_from_db_and_rejects_tampering(tmp_path):
    """A copied trace support ID/spec is diagnostic data, never a proof."""

    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    plans = freeze_structured_plans(manifest["questions"], FakePlanRankModel())

    class FrozenTraceModel(FakePlanRankModel):
        def plan_query_structured(self, query, options):
            return plans[paired_module._query_key(query, options)]

    store = MemoryStore(
        str(database),
        model=FrozenTraceModel(),
        structured_query_plan=True,
        evidence_graph=True,
        graph_max_hops=1,
        graph_temporal=False,
    )
    question = manifest["questions"][0]
    store.search(
        user_id=question["user_id"],
        query=question["question"],
        options=question["options"],
        top_k=3,
    )
    trace = store.last_retrieval_trace
    assert trace["graph_paths"]
    source_by_mem = {item["mem_id"]: item for item in prepared["source_map"]}

    clean = audit_graph_trace_paths(
        database,
        user_id=question["user_id"],
        paths=trace["graph_paths"],
    )
    assert clean["violations"] == []

    tampered = json.loads(json.dumps(trace))
    tampered["graph_paths"][0]["support_id"] += 1000
    direct_audit = _audit_graph_trace(
        database,
        user_id=question["user_id"],
        trace=tampered,
        source_by_mem=source_by_mem,
    )
    assert any(
        "trace witness mismatch" in item and "support_id" in item
        for item in direct_audit["violations"]
    )


def test_arm_audit_rejects_duplicate_pool_content_and_tampered_final(tmp_path):
    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    source_by_mem = {
        str(item["mem_id"]): item for item in prepared["source_map"]
    }
    first_id = str(prepared["source_map"][0]["mem_id"])
    last_id = str(prepared["source_map"][-1]["mem_id"])
    first_numeric = int(first_id.split("_", 1)[1])
    last_numeric = int(last_id.split("_", 1)[1])
    connection = sqlite3.connect(database)
    try:
        content = connection.execute(
            "SELECT content FROM raw_messages WHERE id = ?", (first_numeric,)
        ).fetchone()[0]
        connection.execute(
            "UPDATE raw_messages SET content = ? WHERE id = ?",
            (str(content).swapcase(), last_numeric),
        )
        connection.commit()
    finally:
        connection.close()

    audit = _audit_arm_integrity(
        database,
        user_id="locomo:sample-a",
        trace={
            "rerank_pool_ids": [first_id, last_id, first_id],
            "final_ids": [first_id],
        },
        results=[SimpleNamespace(id=first_id, content="tampered")],
        source_by_mem=source_by_mem,
    )

    assert audit["rerank_pool_unique_ids"] is False
    assert audit["rerank_pool_unique_casefold_content"] is False
    assert audit["final_content_matches_raw"] == 0
    assert any("rerank_pool_ids are not unique" in item for item in audit["violations"])
    assert any("casefold-equivalent content" in item for item in audit["violations"])
    assert any("final result content differs from raw" in item for item in audit["violations"])


def test_id_scoring_does_not_treat_duplicate_text_as_the_same_evidence():
    question = {"user_id": "user-a", "gold_dia_ids": ["D2"]}
    source_map = {
        "mem_1": {"user_id": "user-a", "dia_id": "D1"},
        "mem_2": {"user_id": "user-a", "dia_id": "D2"},
    }

    wrong = score_question_rankings(
        question, ["mem_1"], source_by_mem=source_map, top_ks=(1,)
    )
    right = score_question_rankings(
        question, ["mem_2"], source_by_mem=source_map, top_ks=(1,)
    )

    assert wrong["hit_at_k"]["1"] == 0
    assert right["hit_at_k"]["1"] == 1


def test_formal_trace_and_dense_fusion_fail_closed(tmp_path):
    legacy_store = SimpleNamespace(
        last_graph_candidate_ids=[],
        last_graph_only_candidate_ids=[],
        last_graph_paths=[],
        last_query_plan={},
    )
    with pytest.raises(RuntimeError, match="requires complete MemoryStore retrieval trace"):
        _normalize_storage_trace(legacy_store, require_complete=True)

    manifest, fingerprint, _, _, cache = build_cache_and_manifest(tmp_path)
    database = tmp_path / "prepared.db"
    prepared = build_prepared_database(
        manifest,
        cache["records_by_source_key"],
        database,
        cache_fingerprint=fingerprint,
    )
    plans = freeze_structured_plans(manifest["questions"], FakePlanRankModel())
    with pytest.raises(ValueError, match="rejects dense score fusion"):
        run_paired_evaluation(
            manifest,
            database,
            prepared["source_map"],
            frozen_plans=plans,
            rank_model=FakePlanRankModel(),
            store_options={"dense_fusion_alpha": 0.5},
            require_complete_trace=False,
        )
