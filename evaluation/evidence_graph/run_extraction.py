"""Run the P3-0 extraction diagnostic with one model call per source message.

The runner is intentionally separate from the production Add/Search evaluator.
It can target the formal model or a loopback OpenAI-compatible local proxy and
stores every raw extraction plus its fail-closed parsed graph for auditability.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Mapping, Sequence

from app.evidence_graph import (
    normalize_entity_name,
    normalize_entity_type,
    parse_graph_payload,
)
from app.model import MODEL_NAME, MemoryModel
from evaluation.evidence_graph.generate_cases import validate
from evaluation.evidence_graph.metrics import evaluate_extraction_quality


def _event_timestamp(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    rendered = value.strip().replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(rendered).timestamp())
    except ValueError:
        return None


def _gold_entity_occurrences(case: Mapping[str, object]) -> list[dict]:
    gold = case["gold"]
    assert isinstance(gold, Mapping)
    entities = gold["entities"]
    mentions = gold["mentions"]
    assert isinstance(entities, Sequence) and isinstance(mentions, Sequence)
    by_id = {entity["entity_id"]: entity for entity in entities}
    occurrences = []
    for mention in mentions:
        entity = by_id[mention["entity_id"]]
        occurrences.append(dict(entity))
    return occurrences


def _occurrence_key(
    *, user_id: object, source_message_id: object, name: object
) -> tuple[str, str, str] | None:
    """Identify one textual occurrence independently of ontology typing.

    Entity type is deliberately excluded here: type quality is already scored
    by the entity and relation metrics, while link quality should measure only
    whether two uniquely located mentions were clustered together.  Predicted
    cluster IDs remain type- and identity-hint-aware below.
    """

    canonical = normalize_entity_name(name)
    rendered_user = str(user_id).strip()
    rendered_source = str(source_message_id).strip()
    if not rendered_user or not rendered_source or not canonical:
        return None
    return rendered_user, rendered_source, canonical


def _aligned_predicted_mentions(
    case: Mapping[str, object],
    entities: Sequence[Mapping[str, object]],
    relations: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Align accepted graph-endpoint occurrences to stable gold mention IDs.

    The gold ID is used solely as an occurrence label required by the pairwise
    clustering metric.  Link-group identity remains prediction-derived.  A
    duplicate, missing, or ambiguous occurrence receives a predicted-only ID,
    so alignment cannot turn an incorrect cluster into a correct one.  Dangling
    entities excluded from every accepted relation stay visible to entity
    precision but do not become link mentions; this matches the fixture policy.
    """

    gold = case["gold"]
    assert isinstance(gold, Mapping)
    gold_entities = gold["entities"]
    gold_mentions = gold["mentions"]
    assert isinstance(gold_entities, Sequence)
    assert isinstance(gold_mentions, Sequence)
    gold_by_id = {str(entity["entity_id"]): entity for entity in gold_entities}
    gold_occurrences: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for mention in gold_mentions:
        entity = gold_by_id.get(str(mention["entity_id"]))
        if entity is None:
            continue
        key = _occurrence_key(
            user_id=mention.get("user_id", entity.get("user_id")),
            source_message_id=mention.get("source_message_id"),
            name=entity.get("canonical_name", entity.get("display_name")),
        )
        if key is not None:
            gold_occurrences[key].append(str(mention["mention_id"]))

    consumed_gold_ids: set[str] = set()
    case_id = str(case["case_id"])
    predicted_mentions = []
    endpoint_ids = {
        str(relation[field])
        for relation in relations
        for field in ("subject_entity_id", "object_entity_id")
        if relation.get(field) is not None
    }
    endpoint_entities = [
        entity for entity in entities if str(entity.get("entity_id")) in endpoint_ids
    ]
    predicted_occurrence_keys = [
        _occurrence_key(
            user_id=entity.get("user_id"),
            source_message_id=entity.get("first_source_message_id"),
            name=entity.get("canonical_name", entity.get("display_name")),
        )
        for entity in endpoint_entities
    ]
    predicted_key_counts = Counter(predicted_occurrence_keys)
    for index, entity in enumerate(endpoint_entities):
        user_id = str(entity["user_id"])
        source = str(entity["first_source_message_id"])
        canonical = normalize_entity_name(
            entity.get("canonical_name", entity.get("display_name"))
        ) or ""
        entity_type = normalize_entity_type(entity.get("entity_type")) or ""
        identity_hint = normalize_entity_name(entity.get("identity_hint")) or ""
        key = _occurrence_key(
            user_id=user_id,
            source_message_id=source,
            name=canonical,
        )
        candidates = gold_occurrences.get(key, []) if key is not None else []
        if (
            key is not None
            and predicted_key_counts[key] == 1
            and len(candidates) == 1
            and candidates[0] not in consumed_gold_ids
        ):
            mention_id = candidates[0]
            consumed_gold_ids.add(mention_id)
        else:
            mention_id = f"predicted:{case_id}:{entity['entity_id']}:{index}"
        cluster_material = "\0".join(
            (case_id, user_id, canonical, entity_type, identity_hint)
        ).encode("utf-8")
        predicted_mentions.append({
            "mention_id": mention_id,
            # Exact user/name/type/hint clustering is the conservative P3-0
            # baseline. Same-name counterexamples expose false merges.
            "entity_id": "exact:" + hashlib.sha256(cluster_material).hexdigest()[:24],
            "user_id": user_id,
            "source_message_id": source,
            "surface": entity.get("display_name", canonical),
        })
    return predicted_mentions


def _score_case(case: Mapping[str, object], prediction: Mapping[str, object]) -> dict:
    gold = case["gold"]
    assert isinstance(gold, Mapping)
    return evaluate_extraction_quality(
        gold_entities=_gold_entity_occurrences(case),
        predicted_entities=prediction["entities"],
        gold_relations=gold["relations"],
        predicted_relations=prediction["relations"],
        gold_mentions=gold["mentions"],
        predicted_mentions=prediction["mentions"],
    )


def _flatten_and_score(
    cases: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> dict:
    gold_entities: list[dict] = []
    predicted_entities: list[dict] = []
    gold_relations: list[dict] = []
    predicted_relations: list[dict] = []
    gold_mentions: list[dict] = []
    predicted_mentions: list[dict] = []
    for case, prediction in zip(cases, predictions):
        gold = case["gold"]
        assert isinstance(gold, Mapping)
        gold_entities.extend(_gold_entity_occurrences(case))
        predicted_entities.extend(prediction["entities"])
        gold_relations.extend(gold["relations"])
        predicted_relations.extend(prediction["relations"])
        gold_mentions.extend(gold["mentions"])
        predicted_mentions.extend(prediction["mentions"])
    return evaluate_extraction_quality(
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
        gold_relations=gold_relations,
        predicted_relations=predicted_relations,
        gold_mentions=gold_mentions,
        predicted_mentions=predicted_mentions,
    )


def evaluate_predictions(
    cases: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> tuple[dict, dict[str, dict]]:
    """Score cached predictions globally and by diagnostic category."""

    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have equal length")
    for case, prediction in zip(cases, predictions):
        if case["case_id"] != prediction["case_id"]:
            raise ValueError("case/prediction order or IDs do not match")

    overall = _flatten_and_score(cases, predictions)
    grouped: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = defaultdict(list)
    for case, prediction in zip(cases, predictions):
        grouped[str(case["category"])].append((case, prediction))
    by_category = {
        category: _flatten_and_score(
            [item[0] for item in items], [item[1] for item in items]
        )
        for category, items in sorted(grouped.items())
    }
    return overall, by_category


_RUNTIME_FIELDS = (
    "cases",
    "messages",
    "model_calls",
    "one_call_per_message",
    "workers",
    "latency_seconds_total",
    "latency_seconds_mean",
    "model",
    "base_url",
)


def _record_sequence(value: object, label: str) -> list[Mapping[str, object]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ValueError(f"{label} must be a sequence of records")
    records = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{index}] must be a record")
        records.append(item)
    return records


def _required_text(record: Mapping[str, object], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{field} must be a non-empty string")
    return value.strip()


def _cases_for_cached_predictions(
    cases: Sequence[Mapping[str, object]],
    cached_predictions: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    fixture_by_id: dict[str, Mapping[str, object]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"cases[{index}] must be a record")
        case_id = _required_text(case, "case_id", f"cases[{index}]")
        if case_id in fixture_by_id:
            raise ValueError(f"duplicate fixture case_id: {case_id}")
        fixture_by_id[case_id] = case

    selected = []
    seen_ids: set[str] = set()
    for index, prediction in enumerate(cached_predictions):
        case_id = _required_text(
            prediction, "case_id", f"cached predictions[{index}]"
        )
        if case_id in seen_ids:
            raise ValueError(f"duplicate cached case_id: {case_id}")
        seen_ids.add(case_id)
        case = fixture_by_id.get(case_id)
        if case is None:
            raise ValueError(f"cached case_id is absent from fixture: {case_id}")
        selected.append(case)
    return selected


def rescore_cached_report(
    cases: Sequence[Mapping[str, object]],
    cached_report: Mapping[str, object],
    *,
    rescored_from: str | Path,
) -> dict[str, object]:
    """Reparse and rescore cached raw extractions without calling a model.

    Cached model output is paired to fixture messages only through exact case
    and source-message IDs.  The authoritative source text, event timestamp,
    speaker, and user provenance always come from the fixture.  Missing,
    duplicate, extra, or mismatched records raise instead of being inferred.
    """

    if not isinstance(cached_report, Mapping):
        raise ValueError("cached report must be a record")
    rendered_source = str(rescored_from).strip()
    if not rendered_source:
        raise ValueError("rescored_from must identify the cached report")
    if "predictions" not in cached_report:
        raise ValueError("cached report is missing predictions")
    cached_predictions = _record_sequence(
        cached_report["predictions"], "cached predictions"
    )
    if not cached_predictions:
        raise ValueError("cached report predictions must not be empty")
    selected_cases = _cases_for_cached_predictions(cases, cached_predictions)

    rebuilt_predictions: list[dict] = []
    rescored_messages = 0
    for case_index, (case, cached_prediction) in enumerate(
        zip(selected_cases, cached_predictions)
    ):
        case_id = _required_text(case, "case_id", f"cases[{case_index}]")
        category = _required_text(case, "category", f"case {case_id}")
        cached_category = _required_text(
            cached_prediction, "category", f"cached prediction {case_id}"
        )
        if cached_category != category:
            raise ValueError(
                f"cached category mismatch for {case_id}: "
                f"{cached_category!r} != {category!r}"
            )

        fixture_messages = _record_sequence(
            case.get("messages"), f"fixture messages for {case_id}"
        )
        cached_messages = _record_sequence(
            cached_prediction.get("messages"),
            f"cached messages for {case_id}",
        )
        fixture_by_source: dict[str, Mapping[str, object]] = {}
        for message_index, message in enumerate(fixture_messages):
            label = f"fixture message {case_id}[{message_index}]"
            source_id = _required_text(message, "id", label)
            if source_id in fixture_by_source:
                raise ValueError(f"duplicate fixture source_message_id: {source_id}")
            _required_text(message, "user_id", label)
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{label}.content must be a string")
            fixture_by_source[source_id] = message

        cached_by_source: dict[str, Mapping[str, object]] = {}
        for message_index, message in enumerate(cached_messages):
            label = f"cached message {case_id}[{message_index}]"
            source_id = _required_text(message, "source_message_id", label)
            if source_id in cached_by_source:
                raise ValueError(f"duplicate cached source_message_id: {source_id}")
            cached_by_source[source_id] = message
        if set(cached_by_source) != set(fixture_by_source):
            missing = sorted(set(fixture_by_source) - set(cached_by_source))
            extra = sorted(set(cached_by_source) - set(fixture_by_source))
            raise ValueError(
                f"cached/fixture message mismatch for {case_id}; "
                f"missing={missing}, extra={extra}"
            )

        entities: list[dict] = []
        relations: list[dict] = []
        rebuilt_messages = []
        for fixture_message in fixture_messages:
            source_id = _required_text(
                fixture_message, "id", f"fixture message in {case_id}"
            )
            user_id = _required_text(
                fixture_message, "user_id", f"fixture message {source_id}"
            )
            cached_message = cached_by_source[source_id]
            cached_user = _required_text(
                cached_message, "user_id", f"cached message {source_id}"
            )
            if cached_user != user_id:
                raise ValueError(
                    f"cached user_id mismatch for {source_id}: "
                    f"{cached_user!r} != {user_id!r}"
                )
            if "raw_extraction" not in cached_message:
                raise ValueError(
                    f"cached message {source_id} is missing raw_extraction"
                )
            raw = cached_message["raw_extraction"]
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"cached message {source_id}.raw_extraction must be a record"
                )
            graph = parse_graph_payload(
                raw,
                user_id=user_id,
                source_message_id=source_id,
                event_ts=_event_timestamp(fixture_message.get("timestamp")),
                source_text=fixture_message["content"],
                speaker=fixture_message.get(
                    "speaker", fixture_message.get("role", "")
                ),
            ).to_dict()
            entities.extend(graph["entities"])
            relations.extend(graph["relations"])
            rebuilt_message = dict(cached_message)
            rebuilt_message["source_message_id"] = source_id
            rebuilt_message["user_id"] = user_id
            rebuilt_message["raw_extraction"] = raw
            rebuilt_message["parsed_graph"] = graph
            rebuilt_messages.append(rebuilt_message)
            rescored_messages += 1

        rebuilt_prediction = dict(cached_prediction)
        rebuilt_prediction.update({
            "case_id": case_id,
            "category": category,
            "messages": rebuilt_messages,
            "entities": entities,
            "relations": relations,
            "mentions": _aligned_predicted_mentions(case, entities, relations),
        })
        rebuilt_prediction["metrics"] = _score_case(case, rebuilt_prediction)
        rebuilt_predictions.append(rebuilt_prediction)

    overall, by_category = evaluate_predictions(
        selected_cases, rebuilt_predictions
    )
    original_runtime = {
        field: cached_report[field]
        for field in _RUNTIME_FIELDS
        if field in cached_report
    }
    report = dict(cached_report)
    report.update({
        "cases": len(selected_cases),
        "messages": rescored_messages,
        "metrics": overall,
        "category_metrics": by_category,
        "predictions": rebuilt_predictions,
        "rescored_from": rendered_source,
        "model_calls_delta": 0,
        "rescored_messages": rescored_messages,
        "original_runtime": original_runtime,
        "runtime_note": (
            "Original extraction runtime is retained in original_runtime; "
            "this deterministic rescore made zero model calls."
        ),
    })
    return report


def run_cases(
    cases: Sequence[Mapping[str, object]], model: object, *, workers: int = 1
) -> dict[str, object]:
    """Extract and parse every fixture message exactly once."""

    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    starting_calls = getattr(model, "call_count", None)

    def run_case(case: Mapping[str, object]) -> tuple[dict, list[float], int]:
        entities: list[dict] = []
        relations: list[dict] = []
        message_predictions = []
        case_latencies = []
        case_calls = 0
        for message in case["messages"]:
            started = time.perf_counter()
            raw = model.extract_memory(
                message["content"],
                speaker=message.get("speaker", message.get("role", "")),
                timestamp=_event_timestamp(message.get("timestamp")),
            )
            case_calls += 1
            latency = time.perf_counter() - started
            case_latencies.append(latency)
            graph = parse_graph_payload(
                raw,
                user_id=message["user_id"],
                source_message_id=message["id"],
                event_ts=_event_timestamp(message.get("timestamp")),
                source_text=message["content"],
                speaker=message.get("speaker", message.get("role", "")),
            ).to_dict()
            entities.extend(graph["entities"])
            relations.extend(graph["relations"])
            message_predictions.append({
                "source_message_id": message["id"],
                "user_id": message["user_id"],
                "raw_extraction": raw,
                "parsed_graph": graph,
                "latency_seconds": latency,
            })
        prediction = {
            "case_id": case["case_id"],
            "category": case["category"],
            "messages": message_predictions,
            "entities": entities,
            "relations": relations,
            "mentions": _aligned_predicted_mentions(case, entities, relations),
        }
        prediction["metrics"] = _score_case(case, prediction)
        return prediction, case_latencies, case_calls

    if workers == 1:
        completed = [run_case(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            completed = list(executor.map(run_case, cases))
    predictions = [item[0] for item in completed]
    latencies = [latency for item in completed for latency in item[1]]
    attempted_calls = sum(item[2] for item in completed)

    ending_calls = getattr(model, "call_count", None)
    if starting_calls is not None and ending_calls is not None:
        observed_calls = ending_calls - starting_calls
        if observed_calls != attempted_calls:
            raise RuntimeError(
                f"expected exactly {attempted_calls} extraction model calls, "
                f"observed {observed_calls}"
            )
    else:
        observed_calls = attempted_calls

    overall, by_category = evaluate_predictions(cases, predictions)
    return {
        "cases": len(cases),
        "messages": attempted_calls,
        "model_calls": observed_calls,
        "one_call_per_message": observed_calls == attempted_calls,
        "workers": workers,
        "latency_seconds_total": sum(latencies),
        "latency_seconds_mean": statistics.fmean(latencies) if latencies else 0.0,
        "metrics": overall,
        "category_metrics": by_category,
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", default=str(Path(__file__).with_name("cases.json"))
    )
    parser.add_argument("--base-url", help="Loopback OpenAI-compatible endpoint")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--rescore-report",
        help=(
            "Reparse raw_extraction records from an existing report and "
            "rescore them without any model requests"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        raise SystemExit("--max-cases must be a positive integer")
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    validate(cases)
    if args.rescore_report:
        if args.max_cases is not None:
            raise SystemExit(
                "--max-cases cannot be combined with --rescore-report; "
                "cached case IDs select the exact fixture subset"
            )
        if args.base_url:
            raise SystemExit("--base-url cannot be combined with --rescore-report")
        cached_path = Path(args.rescore_report)
        cached_report = json.loads(cached_path.read_text(encoding="utf-8"))
        report = rescore_cached_report(
            cases,
            cached_report,
            rescored_from=str(cached_path.resolve()),
        )
    else:
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        api_key = "local-only" if args.base_url else os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("formal extraction requires OPENAI_API_KEY")
        model = MemoryModel(
            api_key,
            model_name=args.model,
            base_url=args.base_url,
            disable_thinking=bool(args.base_url),
        )
        report = run_cases(cases, model, workers=args.workers)
        report["model"] = args.model
        report["base_url"] = args.base_url
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
