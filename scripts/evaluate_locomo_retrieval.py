"""Evaluate evidence retrieval on a local LoCoMo JSON file.

The dataset is intentionally supplied by path and is never copied into this
repository. Use only where the dataset licence and competition rules permit.
"""

import argparse
import json
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import AddRequest
from app.model import MemoryModel
from app.local_semantic import (
    LocalCrossEncoderReranker,
    LocalHTTPEmbeddingRetriever,
    LocalLateInteractionReranker,
    LocalSemanticRetriever,
)
from app.local_instruction import (
    LocalDualStrategyReranker,
    LocalInstructionReranker,
    LocalQueryExpander,
    LocalYesNoReranker,
)
from app.storage import MemoryStore


class SearchOnlyModel:
    """Use the configured model for query planning/ranking without Add-time calls."""

    def __init__(
        self, api_key: str, model_name: str = "gpt-4o-mini",
        base_url: str | None = None, disable_thinking: bool = False,
        rank_prompt_v2: bool = False, timeout_seconds: float = 120.0,
    ) -> None:
        self.model = MemoryModel(
            api_key, model_name=model_name, base_url=base_url,
            disable_thinking=disable_thinking,
            rank_prompt_v2=rank_prompt_v2,
            timeout_seconds=timeout_seconds,
        )

    def extract_facts(
        self, content: str, speaker: str = "", timestamp: int | None = None
    ) -> List[str]:
        return []

    def plan_query(self, query: str, options: List[str]) -> List[str]:
        return self.model.plan_query(query, options)

    def plan_query_structured(self, query: str, options: List[str]) -> Dict[str, object]:
        return self.model.plan_query_structured(query, options)

    def rank_candidates(self, query: str, options: List[str], candidates: List[Dict[str, str]]) -> List[str]:
        return self.model.rank_candidates(query, options, candidates)

    def rank_candidates_with_confidence(
        self, query: str, options: List[str], candidates: List[Dict[str, str]]
    ) -> Tuple[List[str], Dict[str, float]]:
        return self.model.rank_candidates_with_confidence(query, options, candidates)


def session_number(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else -1


def session_timestamp(conversation: Dict[str, object], session_key: str) -> int:
    raw_value = str(conversation.get("{}_date_time".format(session_key), "")).strip()
    if raw_value:
        try:
            parsed = datetime.strptime(raw_value, "%I:%M %p on %d %B, %Y")
            return int(parsed.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    return session_number(session_key)


def sessions_and_evidence(
    sample: Dict[str, object],
) -> Tuple[List[Tuple[str, List[Dict[str, object]]]], Dict[str, str]]:
    conversation = sample["conversation"]
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo conversation must be an object")
    sessions: List[Tuple[str, List[Dict[str, object]]]] = []
    evidence_text: Dict[str, str] = {}
    for session_key in sorted(
        (key for key, value in conversation.items() if isinstance(value, list)), key=session_number
    ):
        event_timestamp = session_timestamp(conversation, session_key)
        messages: List[Dict[str, object]] = []
        for turn in conversation[session_key]:
            if not isinstance(turn, dict) or not all(field in turn for field in ("dia_id", "speaker", "text")):
                continue
            content = "{}: {}".format(turn["speaker"], turn["text"])
            # LoCoMo's official dialog-RAG path treats the released BLIP image
            # caption as part of the source turn. Preserve that public source
            # content here too; the image-search `query` is generation metadata
            # and is intentionally not indexed.
            caption = str(turn.get("blip_caption", "")).strip()
            if caption:
                content += " Shared image: {}".format(caption)
            messages.append({
                "role": str(turn["speaker"]),
                "content": content,
                "timestamp": event_timestamp,
            })
            evidence_text[str(turn["dia_id"])] = content
        if messages:
            sessions.append((session_key, messages))
    return sessions, evidence_text


def released_v020_search(store: MemoryStore, user_id: str, query: str, top_k: int) -> List[str]:
    """Faithfully reproduce v0.2.0's no-model raw-message retrieval path."""
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        return []
    match_query = " OR ".join('"{}"'.format(term.replace('"', '')) for term in terms)
    with store._connection() as connection:
        rows = connection.execute(
            """SELECT raw.content
               FROM messages_fts
               JOIN raw_messages AS raw ON raw.id = messages_fts.message_id
               WHERE messages_fts MATCH ? AND messages_fts.user_id = ?
               ORDER BY bm25(messages_fts), raw.id DESC
               LIMIT ?""",
            (match_query, user_id, top_k),
        ).fetchall()
    return [str(row["content"]) for row in rows]


def _failure_bucket(first_rank: int | None) -> str:
    """Map the first gold rank to the deterministic P4-0 bucket."""
    if first_rank == 1:
        return "top1_hit"
    if first_rank is not None and first_rank <= 3:
        return "ranking_top3"
    if first_rank is not None and first_rank <= 10:
        return "ranking_top10"
    return "recall_miss_top10"


def _gold_positions(
    gold_mem_ids: List[str], ordered_ids: Sequence[str]
) -> Dict[str, int]:
    """Return 1-based positions of gold memory IDs inside an ordered ID list."""
    positions = {
        candidate_id: index + 1 for index, candidate_id in enumerate(ordered_ids)
    }
    return {
        candidate_id: positions[candidate_id]
        for candidate_id in gold_mem_ids
        if candidate_id in positions
    }


def _classify_recall_failure(
    gold_mem_ids: List[str], retrieval_trace: Dict[str, object]
) -> str | None:
    """Classify why gold evidence stayed outside the final Top-10.

    Buckets (mutually exclusive, defined on the P1 pipeline):
    - channel_miss: no P1 retrieval channel returned the gold memory ID.
    - fusion_miss: at least one channel returned it, but RRF kept it out of the
      P1-only counterfactual Top-30 (``p1_counterfactual_top30_ids``).
    - quota_displacement: it reached the counterfactual Top-30, but a graph /
      anchor / adjacent quota reservation displaced it from the actual rerank
      pool (``rerank_pool_ids``).
    - reranker_drop: it entered the rerank pool but the ranker kept it out of
      the final Top-10. Without a Search model this reduces to the structural
      pool-to-Top-K cut, because no reordering happens.

    Returns None when no gold memory ID can be located or the trace has no data.
    """
    if not gold_mem_ids:
        return None
    gold_set = set(gold_mem_ids)
    p1_channels = retrieval_trace.get("p1_channels", {})
    if not isinstance(p1_channels, dict) or not p1_channels:
        # No channel data means the trace is absent (no search ran): the
        # failure cause is unknown, not a channel miss.
        return None
    gold_in_channel = any(
        gold_set.intersection(channel_ids)
        for channel_ids in p1_channels.values()
        if isinstance(channel_ids, list)
    )
    if not gold_in_channel:
        return "channel_miss"
    counterfactual_ids = set(retrieval_trace.get("p1_counterfactual_top30_ids", []))
    if not gold_set.intersection(counterfactual_ids):
        return "fusion_miss"
    pool_ids = set(retrieval_trace.get("rerank_pool_ids", []))
    if not gold_set.intersection(pool_ids):
        return "quota_displacement"
    return "reranker_drop"


def _load_mem_by_content(
    database_path: str, user_id: str
) -> Dict[str, str]:
    """Map casefolded raw message content to its ``mem_<id>`` identifier.

    Read-only, diagnostics-only helper: the evaluation pipeline matches gold
    evidence by content while the retrieval trace identifies candidates by
    memory ID, so the audit needs this bridge.
    """
    mem_by_content: Dict[str, str] = {}
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT id, content FROM raw_messages WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    finally:
        connection.close()
    for row_id, content in rows:
        content_key = str(content).casefold()
        if content_key not in mem_by_content:
            mem_by_content[content_key] = "mem_{}".format(int(row_id))
    return mem_by_content


def evaluate(samples: Iterable[Dict[str, object]], top_ks: List[int], max_questions: int | None,
             question_offset: int = 0,
             retriever: str = "current", model: object = None,
             semantic_retriever: object = None, dense_rrf_weight: float = 1.0,
             dense_fusion_alpha: float | None = None, local_reranker: object = None,
             dense_context_weight: float = 0.0,
             dense_time_weight: float = 0.0,
             dense_speaker_mask_max: bool = False,
             dense_speaker_conflict_margin: float | None = None,
             dense_speaker_conflict_gate_only: bool = False,
             dense_sentence_weight: float = 0.0,
             dense_image_carry_weight: float = 0.0,
             dense_speaker_coref_weight: float = 0.0,
             dense_speaker_swap_max: bool = False,
             rerank_top_n: int = 10, session_fusion_weight: float = 0.0,
             rerank_image_followups: int = 0,
             session_top_n: int = 0,
             rerank_fusion_weight: float | None = None,
             rerank_near_tie_epsilon: float = 0.0,
             local_instruction_reranker: object = None,
             instruction_speaker_conflict_only: bool = False,
             local_query_expander: object = None,
             instruction_rerank_top_n: int = 10,
             instruction_refine_top_n: int = 0,
             include_hit_bitmap: bool = False,
             structured_query_plan: bool = False,
             set_aware_rerank: bool = False,
             include_question_diagnostics: bool = False,
             evidence_need_retrieval: bool = False,
             evidence_need_quota: int = 2,
             evidence_need_rrf_weight: float = 0.01,
             need_select_by_bm25: bool = False,
             adjacent_turn_expansion: bool = False,
             bridge_retrieval: bool = False,
             bridge_max_terms: int = 3,
             bridge_rrf_weight: float = 0.01,
             bridge_rerank_quota: int = 2,
             sidecar_shared_quota: int = 0,
             query_relaxation: bool = False,
             relax_rrf_weight: float = 0.01,
             relax_quota: int = 2,
             p5_gate: bool = False,
             p5_near_tie_epsilon: float = 0.0005,
             p5_min_evidence_channels: int = 2,
             p5_confidence_margin: float = 0.05,
             p5_strata: str = "temporal,correction",
             llm_rerank_top_n: int = 0,
             evidence_graph: bool = False,
             graph_selective: bool = False,
             graph_rrf_weight: float = 0.025,
             graph_max_candidates: int = 20,
             graph_rerank_quota: int = 4) -> Dict[str, object]:
    if retriever not in {"current", "v0.2.0"}:
        raise ValueError("retriever must be current or v0.2.0")
    hit_counts = {top_k: 0 for top_k in top_ks}
    evidence_hits = {top_k: 0 for top_k in top_ks}
    category_counts = defaultdict(
        lambda: {
            "questions": 0,
            "hit_at_1": 0,
            "hit_counts": {top_k: 0 for top_k in top_ks},
        }
    )
    question_count = evidence_total = eligible_question_count = 0
    reciprocal_rank_sum = 0.0
    speaker_conflict_triggers = 0
    hit_at_1_bitmap = []
    question_diagnostics = []
    instruction_failures_before = getattr(
        local_instruction_reranker, "invalid_response_count", 0
    )

    with tempfile.TemporaryDirectory(prefix="chrono-locomo-") as temporary_directory:
        for sample_index, sample in enumerate(samples):
            if max_questions is not None and question_count >= max_questions:
                break
            sessions, evidence_text = sessions_and_evidence(sample)
            if not sessions:
                continue
            store = MemoryStore(
                str(Path(temporary_directory) / "sample_{}.db".format(sample_index)),
                model=model,
                semantic_retriever=semantic_retriever,
                dense_rrf_weight=dense_rrf_weight,
                dense_fusion_alpha=dense_fusion_alpha,
                dense_context_weight=dense_context_weight,
                dense_time_weight=dense_time_weight,
                dense_speaker_mask_max=dense_speaker_mask_max,
                dense_speaker_conflict_margin=dense_speaker_conflict_margin,
                dense_speaker_conflict_gate_only=dense_speaker_conflict_gate_only,
                dense_sentence_weight=dense_sentence_weight,
                dense_image_carry_weight=dense_image_carry_weight,
                dense_speaker_coref_weight=dense_speaker_coref_weight,
                dense_speaker_swap_max=dense_speaker_swap_max,
                local_reranker=local_reranker,
                rerank_top_n=rerank_top_n,
                rerank_image_followups=rerank_image_followups,
                session_fusion_weight=session_fusion_weight,
                session_top_n=session_top_n,
                rerank_fusion_weight=rerank_fusion_weight,
                rerank_near_tie_epsilon=rerank_near_tie_epsilon,
                local_instruction_reranker=local_instruction_reranker,
                instruction_speaker_conflict_only=instruction_speaker_conflict_only,
                local_query_expander=local_query_expander,
                instruction_rerank_top_n=instruction_rerank_top_n,
                instruction_refine_top_n=instruction_refine_top_n,
                structured_query_plan=structured_query_plan,
                set_aware_rerank=set_aware_rerank,
                evidence_need_retrieval=evidence_need_retrieval,
                evidence_need_quota=evidence_need_quota,
                evidence_need_rrf_weight=evidence_need_rrf_weight,
                need_select_by_bm25=need_select_by_bm25,
                adjacent_turn_expansion=adjacent_turn_expansion,
                bridge_retrieval=bridge_retrieval,
                bridge_max_terms=bridge_max_terms,
                bridge_rrf_weight=bridge_rrf_weight,
                bridge_rerank_quota=bridge_rerank_quota,
                sidecar_shared_quota=sidecar_shared_quota,
                query_relaxation=query_relaxation,
                relax_rrf_weight=relax_rrf_weight,
                relax_quota=relax_quota,
                p5_gate=p5_gate,
                p5_near_tie_epsilon=p5_near_tie_epsilon,
                p5_min_evidence_channels=p5_min_evidence_channels,
                p5_confidence_margin=p5_confidence_margin,
                p5_strata=p5_strata,
                llm_rerank_top_n=llm_rerank_top_n,
                evidence_graph=evidence_graph,
                graph_selective=graph_selective,
                graph_rrf_weight=graph_rrf_weight,
                graph_max_candidates=graph_max_candidates,
                graph_rerank_quota=graph_rerank_quota,
            )
            store.initialize()
            user_id = "locomo:{}".format(sample.get("sample_id", sample_index))
            for session_key, messages in sessions:
                store.add(AddRequest(
                    request_id="locomo:{}:{}".format(sample_index, session_key),
                    user_id=user_id,
                    session_id="locomo:{}:{}".format(sample_index, session_key),
                    messages=messages,
                ))
            mem_by_content: Dict[str, str] = {}
            if include_question_diagnostics and retriever == "current":
                mem_by_content = _load_mem_by_content(
                    str(Path(temporary_directory) / "sample_{}.db".format(sample_index)),
                    user_id,
                )
            for qa in sample.get("qa", []):
                if max_questions is not None and question_count >= max_questions:
                    break
                if not isinstance(qa, dict):
                    continue
                expected = [evidence_text[item] for item in qa.get("evidence", []) if item in evidence_text]
                if not expected or not qa.get("question"):
                    continue
                if eligible_question_count < question_offset:
                    eligible_question_count += 1
                    continue
                eligible_question_count += 1
                question_count += 1
                evidence_total += len(expected)
                if retriever == "current":
                    results = store.search(user_id=user_id, query=str(qa["question"]), top_k=max(top_ks))
                    contents = [result.content.casefold() for result in results]
                else:
                    contents = [content.casefold() for content in released_v020_search(
                        store, user_id, str(qa["question"]), max(top_ks)
                    )]
                ranks = [
                    next((index + 1 for index, content in enumerate(contents) if item.casefold() == content), None)
                    for item in expected
                ]
                present_ranks = [rank for rank in ranks if rank is not None]
                first_rank = min(present_ranks) if present_ranks else None
                category = str(qa.get("category", "unknown"))
                category_counts[category]["questions"] += 1
                if first_rank == 1:
                    category_counts[category]["hit_at_1"] += 1
                if include_hit_bitmap:
                    hit_at_1_bitmap.append(1 if first_rank == 1 else 0)
                if first_rank is not None:
                    reciprocal_rank_sum += 1.0 / first_rank
                for top_k in top_ks:
                    matches = [rank for rank in ranks if rank is not None and rank <= top_k]
                    evidence_hits[top_k] += len(matches)
                    if matches:
                        hit_counts[top_k] += 1
                        category_counts[category]["hit_counts"][top_k] += 1
                if include_question_diagnostics and retriever == "current":
                    gold_mem_ids = [
                        mem_by_content[item.casefold()]
                        for item in expected
                        if item.casefold() in mem_by_content
                    ]
                    retrieval_trace = store.last_retrieval_trace
                    p1_channels = retrieval_trace.get("p1_channels", {})
                    gold_channel_presence: Dict[str, Dict[str, int]] = {}
                    for channel_name, channel_ids in p1_channels.items():
                        if not isinstance(channel_ids, list):
                            continue
                        positions = _gold_positions(gold_mem_ids, channel_ids)
                        if positions:
                            gold_channel_presence[channel_name] = positions
                    sidecar_ids = (
                        set(retrieval_trace.get("graph_candidate_ids", []))
                        | set(retrieval_trace.get("anchor_candidate_ids", []))
                        | set(retrieval_trace.get("adjacent_candidate_ids", []))
                    )
                    question_diagnostics.append({
                        "question_offset": eligible_question_count - 1,
                        "category": category,
                        "first_gold_rank": first_rank,
                        "failure_bucket": _failure_bucket(first_rank),
                        "recall_bucket": (
                            _classify_recall_failure(gold_mem_ids, retrieval_trace)
                            if max(top_ks) >= 10
                            and (first_rank is None or first_rank > 10)
                            else None
                        ),
                        "result_ids": [result.id for result in results],
                        "gold_mem_ids": gold_mem_ids,
                        "gold_channel_presence": gold_channel_presence,
                        "gold_counterfactual_positions": _gold_positions(
                            gold_mem_ids,
                            retrieval_trace.get("p1_counterfactual_top30_ids", []),
                        ),
                        "gold_pool_positions": _gold_positions(
                            gold_mem_ids,
                            retrieval_trace.get("rerank_pool_ids", []),
                        ),
                        "recovered_by_sidecar": bool(
                            set(gold_mem_ids).intersection(sidecar_ids)
                        ),
                        "gold_need_channel_presence": {
                            channel_name: positions
                            for channel_name, channel_ids in (
                                retrieval_trace.get(
                                    "evidence_need_channels", {}
                                ).items()
                            )
                            if (
                                positions := _gold_positions(
                                    gold_mem_ids, channel_ids
                                )
                            )
                        },
                        "gold_adjacent_positions": _gold_positions(
                            gold_mem_ids,
                            retrieval_trace.get("adjacent_candidate_ids", []),
                        ),
                        "promoted_adjacent_ids": retrieval_trace.get(
                            "promoted_adjacent_ids", []
                        ),
                        "displaced_p1_for_adjacent_ids": retrieval_trace.get(
                            "displaced_p1_for_adjacent_ids", []
                        ),
                        "gold_bridge_positions": _gold_positions(
                            gold_mem_ids,
                            retrieval_trace.get("bridge_union_ids", []),
                        ),
                        "bridge_terms": retrieval_trace.get("bridge_terms", []),
                        "promoted_bridge_ids": retrieval_trace.get(
                            "promoted_bridge_ids", []
                        ),
                        "displaced_p1_for_bridge_ids": retrieval_trace.get(
                            "displaced_p1_for_bridge_ids", []
                        ),
                        "evidence_need_union_ids": retrieval_trace.get(
                            "evidence_need_union_ids", []
                        ),
                        "evidence_need_diagnostics": retrieval_trace.get(
                            "evidence_need_diagnostics", {}
                        ),
                        "relax_union_ids": retrieval_trace.get(
                            "relax_union_ids", []
                        ),
                        "reserved_relax_ids": retrieval_trace.get(
                            "reserved_relax_ids", []
                        ),
                        "promoted_relax_ids": retrieval_trace.get(
                            "promoted_relax_ids", []
                        ),
                        "gold_relax_positions": _gold_positions(
                            gold_mem_ids,
                            retrieval_trace.get("relax_union_ids", []),
                        ),
                        "p5_gate": retrieval_trace.get("p5_diagnostics", {}),
                        "p5_swapped": bool(
                            retrieval_trace.get("p5_diagnostics", {}).get(
                                "swapped_ids", []
                            )
                        ),
                        "graph_selective": {
                            "enabled": retrieval_trace.get(
                                "edge_diagnostics", {}
                            ).get("selective_enabled", False),
                            "triggered": retrieval_trace.get(
                                "edge_diagnostics", {}
                            ).get("selective_triggered", False),
                            "reason": retrieval_trace.get(
                                "edge_diagnostics", {}
                            ).get("selective_reason"),
                            "graph_candidate_ids": retrieval_trace.get(
                                "graph_candidate_ids", []
                            ),
                        },
                        "llm_rank_candidate_count": retrieval_trace.get(
                            "llm_rank_candidate_count", 0
                        ),
                        "reserved_need_ids": retrieval_trace.get(
                            "reserved_need_ids", []
                        ),
                        "reserved_bridge_ids": retrieval_trace.get(
                            "reserved_bridge_ids", []
                        ),
                        "promoted_need_ids": retrieval_trace.get(
                            "promoted_need_ids", []
                        ),
                        "displaced_p1_for_need_ids": retrieval_trace.get(
                            "displaced_p1_for_need_ids", []
                        ),
                        "plan": retrieval_trace.get("plan", {}),
                        "p1_counterfactual_top30_ids": retrieval_trace.get(
                            "p1_counterfactual_top30_ids", []
                        ),
                        "rerank_pool_ids": retrieval_trace.get("rerank_pool_ids", []),
                        "final_ids": retrieval_trace.get("final_ids", []),
                    })
            speaker_conflict_triggers += store.speaker_conflict_trigger_count

    result = {
        "questions": question_count,
        "evidence_items": evidence_total,
        "hit_at_k": {str(k): round(hit_counts[k] / question_count, 4) if question_count else 0.0 for k in top_ks},
        "evidence_recall_at_k": {str(k): round(evidence_hits[k] / evidence_total, 4) if evidence_total else 0.0 for k in top_ks},
        "mrr": round(reciprocal_rank_sum / question_count, 4) if question_count else 0.0,
        "category_hit_at_1": {
            category: round(values["hit_at_1"] / values["questions"], 4)
            for category, values in sorted(category_counts.items())
        },
        "category_hit_at_k": {
            category: {
                str(top_k): round(
                    values["hit_counts"][top_k] / values["questions"], 4
                )
                for top_k in top_ks
            }
            for category, values in sorted(category_counts.items())
        },
        "question_offset": question_offset,
        "next_question_offset": question_offset + question_count,
        "raw_counts": {
            "hit_counts": {str(k): hit_counts[k] for k in top_ks},
            "evidence_hit_counts": {str(k): evidence_hits[k] for k in top_ks},
            "reciprocal_rank_sum": reciprocal_rank_sum,
            "category_counts": {
                category: {
                    "questions": values["questions"],
                    "hit_at_1": values["hit_at_1"],
                    "hit_counts": {
                        str(top_k): values["hit_counts"][top_k]
                        for top_k in top_ks
                    },
                }
                for category, values in sorted(category_counts.items())
            },
        },
    }
    if local_instruction_reranker is not None:
        result["instruction_format_fallbacks"] = (
            getattr(local_instruction_reranker, "invalid_response_count", 0)
            - instruction_failures_before
        )
    if dense_speaker_conflict_margin is not None:
        result["speaker_conflict_triggers"] = speaker_conflict_triggers
    if include_hit_bitmap:
        result["hit_at_1_bitmap"] = hit_at_1_bitmap
    if include_question_diagnostics:
        result["question_diagnostics"] = question_diagnostics
    return result


def apply_baseline_mode(args) -> None:
    """Enable the validated P4-A q2 baseline flags in place.

    Structured planning plus evidence-need retrieval at quota 2 (weight 0.01).
    Explicit --evidence-need-quota / --evidence-need-rrf-weight were parsed
    before this runs, so their values always win over the shorthand defaults.
    """
    args.structured_query_plan = True
    args.evidence_need_retrieval = True
    if args.evidence_need_quota is None:
        args.evidence_need_quota = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-only LoCoMo evaluation.")
    parser.add_argument("--dataset", required=True, help="Local path to LoCoMo locomo10.json")
    parser.add_argument("--top-k", default="1,3,10", help="Comma-separated retrieval depths")
    parser.add_argument("--max-questions", type=int, help="Optional cap for a smoke test")
    parser.add_argument(
        "--question-offset", type=int, default=0,
        help="Skip eligible questions before evaluating a resumable chunk",
    )
    parser.add_argument(
        "--output", help="Optional local JSON path for the aggregate-only result",
    )
    parser.add_argument(
        "--include-hit-bitmap", action="store_true",
        help="Include only per-question Hit@1 bits for local complementarity analysis",
    )
    parser.add_argument(
        "--include-question-diagnostics", action="store_true",
        help=(
            "Export default-off per-question retrieval diagnostics: rank bucket, "
            "trace-derived recall bucket (channel_miss / fusion_miss / "
            "quota_displacement / reranker_drop) and gold channel presence"
        ),
    )
    parser.add_argument("--compare-v020", action="store_true", help="Also reproduce v0.2.0 no-model retrieval")
    parser.add_argument(
        "--search-model", action="store_true",
        help="Use OPENAI_API_KEY for query planning/ranking, but skip Add-time fact calls",
    )
    parser.add_argument(
        "--local-search-model-url",
        help="Use a loopback OpenAI-compatible model for Search planning/ranking only",
    )
    parser.add_argument("--local-search-model-name", default="local")
    parser.add_argument(
        "--rank-prompt-v2", action="store_true",
        help=(
            "Use the v2 rank prompt (mention != answer rule + few-shot). "
            "Default-off; v1 is the operational baseline."
        ),
    )
    parser.add_argument(
        "--model-timeout", type=float, default=120.0,
        help="Per-request timeout seconds for the Search model (default 120)",
    )
    parser.add_argument(
        "--structured-query-plan", action="store_true",
        help="Enable P1 structured planning on the selected Search model",
    )
    parser.add_argument(
        "--set-aware-rerank", action="store_true",
        help="Enable default-off P2 set-aware ordering after candidate ranking",
    )
    parser.add_argument(
        "--evidence-need-retrieval", action="store_true",
        help=(
            "Enable default-off P4-A per-evidence-need retrieval channels with a "
            "reserved rerank-pool quota; requires --structured-query-plan"
        ),
    )
    parser.add_argument(
        "--evidence-need-quota", type=int, default=2,
        help="P4-A reserved rerank-pool slots for evidence-need candidates (default 2)",
    )
    parser.add_argument(
        "--evidence-need-rrf-weight", type=float, default=0.01,
        help="P4-A low RRF weight for evidence-need channels (default 0.01)",
    )
    parser.add_argument(
        "--need-select-by-bm25", action="store_true",
        help=(
            "P4-A: order evidence-need union candidates by best bm25 score "
            "instead of channel insertion order before the quota picks (default off)"
        ),
    )
    parser.add_argument(
        "--baseline-mode", action="store_true",
        help=(
            "One-flag shorthand for the validated P4-A q2 baseline: enables "
            "--structured-query-plan and --evidence-need-retrieval with quota 2. "
            "Explicit --evidence-need-quota / --evidence-need-rrf-weight still win."
        ),
    )
    parser.add_argument(
        "--adjacent-turn-expansion", action="store_true",
        help=(
            "Enable default-off P1.1/P4-B same-session +/-1 neighbor recovery "
            "with a reserved rerank-pool quota"
        ),
    )
    parser.add_argument(
        "--bridge-retrieval", action="store_true",
        help=(
            "Enable default-off P4-C bridge second-pass retrieval for multi-hop "
            "plans; requires --structured-query-plan"
        ),
    )
    parser.add_argument(
        "--bridge-max-terms", type=int, default=3,
        help="P4-C max deterministic bridge terms (default 3)",
    )
    parser.add_argument(
        "--bridge-rrf-weight", type=float, default=0.01,
        help="P4-C low RRF weight for bridge candidates (default 0.01)",
    )
    parser.add_argument(
        "--bridge-quota", type=int, default=2,
        help="P4-C reserved rerank-pool slots for bridge candidates (default 2)",
    )
    parser.add_argument(
        "--sidecar-shared-quota", type=int, default=0,
        help=(
            "Shared sidecar pool: when >0, P4-A need and P4-C bridge candidates "
            "share ONE total rerank-pool reservation instead of each holding its "
            "own fixed quota (default 0 = disabled, independent quotas)"
        ),
    )
    parser.add_argument(
        "--query-relaxation", action="store_true",
        help=(
            "Enable default-off P4-D query relaxation: when every evidence-need "
            "channel returns zero hits, run one bounded FTS5 prefix (word*) pass "
            "over raw channels; requires --evidence-need-retrieval"
        ),
    )
    parser.add_argument(
        "--relax-rrf-weight", type=float, default=0.01,
        help="P4-D low RRF weight for relaxed candidates (default 0.01)",
    )
    parser.add_argument(
        "--relax-quota", type=int, default=2,
        help="P4-D reserved rerank-pool slots for relaxed candidates (default 2)",
    )
    parser.add_argument(
        "--p5-gate", action="store_true",
        help=(
            "Enable default-off P5 selective rerank gate: swap Top-1/Top-2 only "
            "when near-tied on fusion score AND the runner-up carries strictly "
            "stronger P1-channel evidence (evidence-preserving, never unconditional)"
        ),
    )
    parser.add_argument(
        "--p5-near-tie-epsilon", type=float, default=0.0005,
        help="P5 fusion-score gap below which Top-1/Top-2 count as near-tied (default 0.0005)",
    )
    parser.add_argument(
        "--p5-min-evidence-channels", type=int, default=2,
        help="P5 minimum P1-channel hits required for the runner-up to be promoted (default 2)",
    )
    parser.add_argument(
        "--p5-confidence-margin", type=float, default=0.05,
        help=(
            "P5 minimum confidence advantage (0-1) the runner-up needs over "
            "Top-1 to justify a swap when the model supplies confidence scores "
            "(default 0.05)"
        ),
    )
    parser.add_argument(
        "--p5-strata", default="temporal,correction",
        help=(
            "P5 gate strata: comma-separated subset of all/temporal/correction "
            "(default temporal,correction); the gate only fires for queries "
            "matching the strata language"
        ),
    )
    parser.add_argument(
        "--evidence-graph", action="store_true",
        help="Enable the P3 evidence-graph one-hop channel (default off)",
    )
    parser.add_argument(
        "--graph-selective", action="store_true",
        help=(
            "Gate the graph channel to multi-hop plans or entity-dense queries "
            "instead of running it unconditionally (P3-A global defaulting "
            "regressed on a 20-question slice)"
        ),
    )
    parser.add_argument(
        "--graph-rrf-weight", type=float, default=0.025,
        help="RRF weight for graph candidates (default 0.025)",
    )
    parser.add_argument(
        "--graph-max-candidates", type=int, default=20,
        help="Max graph candidates (default 20)",
    )
    parser.add_argument(
        "--graph-quota", type=int, default=4,
        help="Reserved rerank-pool slots for graph candidates (default 4)",
    )
    parser.add_argument(
        "--llm-rerank-top-n", type=int, default=0,
        help=(
            "Direction-3 candidate compression: only the first N rerank-pool "
            "candidates (fusion order) are sent to the Search model for ranking "
            "(default 0 = all candidates, up to 30)"
        ),
    )
    parser.add_argument(
        "--local-embedding-model",
        help="Use a local FastEmbed model for dense retrieval, for example BAAI/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--local-embedding-url",
        help="Loopback OpenAI-compatible embedding base URL",
    )
    parser.add_argument("--local-embedding-http-model", default="local")
    parser.add_argument(
        "--local-embedding-instruction",
        help=(
            "Optional one-sentence Qwen3 query instruction; applied to queries only "
            "using the model-card Instruct/Query format"
        ),
    )
    parser.add_argument(
        "--local-late-interaction-model",
        help="Use a local FastEmbed late-interaction model as the first-stage retriever",
    )
    parser.add_argument("--local-device", choices=["auto", "cpu", "cuda"], help="FastEmbed device")
    parser.add_argument(
        "--local-cache-dir",
        help="Directory for local model files; keep it outside version control",
    )
    parser.add_argument(
        "--dense-weight", type=float, default=1.0,
        help="RRF weight for the local dense retrieval channel (default: 1.0)",
    )
    parser.add_argument(
        "--fusion-alpha", type=float,
        help="Use z-score fusion: alpha*BM25 + (1-alpha)*dense",
    )
    parser.add_argument(
        "--fusion-alphas",
        help="Comma-separated z-score fusion sweep evaluated with shared embedding caches",
    )
    parser.add_argument("--dense-context-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-context-weights",
        help="Comma-separated previous+current dense-key weights for a shared-cache sweep",
    )
    parser.add_argument("--dense-time-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-time-weights",
        help="Comma-separated timestamp-augmented dense-key weights for a shared-cache sweep",
    )
    parser.add_argument(
        "--dense-speaker-mask-max", action="store_true",
        help="Max-fuse exact and speaker-anonymized dense scores",
    )
    parser.add_argument("--dense-speaker-conflict-margin", type=float)
    parser.add_argument("--dense-speaker-conflict-gate-only", action="store_true")
    parser.add_argument(
        "--dense-speaker-conflict-margins",
        help="Comma-separated conditional speaker-conflict margins for a shared-cache sweep",
    )
    parser.add_argument("--dense-sentence-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-sentence-weights",
        help="Comma-separated latent sentence-key weights for a shared-cache sweep",
    )
    parser.add_argument("--dense-image-carry-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-image-carry-weights",
        help="Comma-separated prior-image carry weights for a shared-cache sweep",
    )
    parser.add_argument("--dense-speaker-coref-weight", type=float, default=0.0)
    parser.add_argument(
        "--dense-speaker-coref-weights",
        help="Comma-separated speaker-coreference weights for a shared-cache sweep",
    )
    parser.add_argument(
        "--dense-speaker-swap-max", action="store_true",
        help="Max-fuse the original query with a two-speaker name-swapped latent query",
    )
    parser.add_argument("--local-reranker-model", help="FastEmbed late-interaction model")
    parser.add_argument("--local-cross-encoder-model", help="FastEmbed cross-encoder reranker")
    parser.add_argument(
        "--local-yes-no-reranker-url",
        help="Loopback URL for a Qwen-style yes/no logprob reranker",
    )
    parser.add_argument("--local-yes-no-reranker-model", default="local")
    parser.add_argument(
        "--local-yes-no-reranker-instruction",
        choices=["web", "memory", "evidence_audit"], default="web",
    )
    parser.add_argument("--local-yes-no-reranker-batch-size", type=int, default=1)
    parser.add_argument("--local-yes-no-mask-speakers", action="store_true")
    parser.add_argument("--local-yes-no-max-masked-score", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--rerank-image-followups", type=int, default=0)
    parser.add_argument(
        "--rerank-top-ns",
        help="Comma-separated rerank-pool sweep evaluated with shared embedding caches",
    )
    parser.add_argument(
        "--rerank-fusion-weight", type=float,
        help="Blend first-stage and local reranker z-scores with a weight from 0 to 1",
    )
    parser.add_argument(
        "--rerank-fusion-weights",
        help="Comma-separated reranker-fusion sweep evaluated with shared caches",
    )
    parser.add_argument("--rerank-near-tie-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--rerank-near-tie-epsilons",
        help="Comma-separated reranker near-tie thresholds for a shared-cache sweep",
    )
    parser.add_argument(
        "--session-weight", type=float, default=0.0,
        help="Weight for hierarchical session-level lexical/dense fusion (default: 0)",
    )
    parser.add_argument(
        "--session-weights",
        help="Comma-separated session-weight sweep evaluated with shared embedding caches",
    )
    parser.add_argument("--session-top-n", type=int, default=0)
    parser.add_argument(
        "--session-top-ns",
        help="Comma-separated hard session-pool sizes for a shared-cache sweep",
    )
    parser.add_argument(
        "--local-instruction-url",
        help="Loopback OpenAI-compatible base URL, for example http://127.0.0.1:8081/v1",
    )
    parser.add_argument("--local-query-expansion-url")
    parser.add_argument("--local-query-expansion-model", default="local")
    parser.add_argument("--local-instruction-model", default="local")
    parser.add_argument("--instruction-rerank-top-n", type=int, default=10)
    parser.add_argument("--instruction-refine-top-n", type=int, default=0)
    parser.add_argument("--instruction-speaker-conflict-only", action="store_true")
    parser.add_argument("--instruction-thinking", action="store_true")
    parser.add_argument(
        "--instruction-strategy",
        choices=[
            "direct", "chain_of_note", "comparative_audit", "comparative_top1",
            "conservative_verify_top1", "constraint_first_top1", "answer_first_top1",
            "dual_arbitrate_top1",
        ],
        default="direct",
    )
    args = parser.parse_args()
    if args.baseline_mode:
        apply_baseline_mode(args)
    if args.question_offset < 0:
        raise ValueError("--question-offset must be non-negative")
    if sum(bool(value) for value in (
        args.local_late_interaction_model,
        args.local_embedding_model,
        args.local_embedding_url,
    )) > 1:
        raise ValueError(
            "Use only one of --local-embedding-model, --local-late-interaction-model, "
            "or --local-embedding-url"
        )
    late_interaction_first_stage = bool(args.local_late_interaction_model)
    if late_interaction_first_stage:
        # Reuse the existing validation and reporting paths while selecting the
        # late-interaction implementation at construction time below.
        args.local_embedding_model = args.local_late_interaction_model
    top_ks = sorted({int(value) for value in args.top_k.split(",")})
    if not top_ks or min(top_ks) < 1 or max(top_ks) > 100:
        raise ValueError("--top-k values must be between 1 and 100")
    if args.dense_weight < 0 or args.dense_weight > 10:
        raise ValueError("--dense-weight must be between 0 and 10")
    if args.fusion_alpha is not None and not 0 <= args.fusion_alpha <= 1:
        raise ValueError("--fusion-alpha must be between 0 and 1")
    if args.fusion_alpha is not None and args.fusion_alphas:
        raise ValueError("Use either --fusion-alpha or --fusion-alphas, not both")
    if sum(bool(value) for value in (
        args.fusion_alphas, args.rerank_top_ns, args.session_weights,
        args.rerank_fusion_weights, args.dense_context_weights, args.dense_time_weights,
        args.session_top_ns, args.dense_speaker_conflict_margins,
        args.dense_sentence_weights,
        args.rerank_near_tie_epsilons,
        args.dense_image_carry_weights,
        args.dense_speaker_coref_weights,
    )) > 1:
        raise ValueError("Use only one shared-cache sweep at a time")
    if args.question_offset and any((
        args.fusion_alphas, args.rerank_top_ns, args.session_weights,
        args.rerank_fusion_weights, args.dense_context_weights,
        args.dense_time_weights, args.session_top_ns,
        args.dense_speaker_conflict_margins,
        args.dense_sentence_weights,
        args.rerank_near_tie_epsilons,
        args.dense_image_carry_weights,
        args.dense_speaker_coref_weights,
    )):
        raise ValueError("--question-offset is supported for one configuration, not sweeps")
    if args.include_question_diagnostics and any((
        args.fusion_alphas, args.rerank_top_ns, args.session_weights,
        args.rerank_fusion_weights, args.dense_context_weights,
        args.dense_time_weights, args.session_top_ns,
        args.dense_speaker_conflict_margins,
        args.dense_sentence_weights,
        args.rerank_near_tie_epsilons,
        args.dense_image_carry_weights,
        args.dense_speaker_coref_weights,
    )):
        raise ValueError(
            "--include-question-diagnostics is supported for one configuration, not sweeps"
        )
    fusion_alphas = None
    if args.fusion_alphas:
        fusion_alphas = [float(value) for value in args.fusion_alphas.split(",")]
        if not fusion_alphas or any(value < 0 or value > 1 for value in fusion_alphas):
            raise ValueError("--fusion-alphas values must be between 0 and 1")
    has_embedding = bool(args.local_embedding_model or args.local_embedding_url)
    if args.fusion_alpha is not None and not has_embedding:
        raise ValueError("--fusion-alpha requires --local-embedding-model")
    if fusion_alphas is not None and not has_embedding:
        raise ValueError("--fusion-alphas requires --local-embedding-model")
    if not 0 <= args.dense_context_weight <= 1:
        raise ValueError("--dense-context-weight must be between 0 and 1")
    dense_context_weights = None
    if args.dense_context_weights:
        dense_context_weights = [
            float(value) for value in args.dense_context_weights.split(",")
        ]
        if not dense_context_weights or any(
            value < 0 or value > 1 for value in dense_context_weights
        ):
            raise ValueError("--dense-context-weights values must be between 0 and 1")
        if not args.local_embedding_model or args.fusion_alpha is None:
            raise ValueError(
                "--dense-context-weights requires --local-embedding-model and --fusion-alpha"
            )
    if not 0 <= args.dense_time_weight <= 1:
        raise ValueError("--dense-time-weight must be between 0 and 1")
    dense_time_weights = None
    if args.dense_time_weights:
        dense_time_weights = [float(value) for value in args.dense_time_weights.split(",")]
        if not dense_time_weights or any(
            value < 0 or value > 1 for value in dense_time_weights
        ):
            raise ValueError("--dense-time-weights values must be between 0 and 1")
        if not args.local_embedding_model or args.fusion_alpha is None:
            raise ValueError(
                "--dense-time-weights requires --local-embedding-model and --fusion-alpha"
            )
    if args.local_reranker_model and not args.local_embedding_model:
        raise ValueError("--local-reranker-model requires --local-embedding-model")
    if args.dense_speaker_mask_max and (
        not args.local_embedding_model or args.fusion_alpha is None
    ):
        raise ValueError(
            "--dense-speaker-mask-max requires --local-embedding-model and --fusion-alpha"
        )
    if args.dense_speaker_conflict_margin is not None and (
        args.dense_speaker_conflict_margin < 0
        or not args.local_embedding_model
        or args.fusion_alpha is None
    ):
        raise ValueError(
            "--dense-speaker-conflict-margin must be non-negative and requires "
            "--local-embedding-model and --fusion-alpha"
        )
    if args.dense_speaker_conflict_gate_only and (
        args.dense_speaker_conflict_margin is None
    ):
        raise ValueError(
            "--dense-speaker-conflict-gate-only requires "
            "--dense-speaker-conflict-margin"
        )
    dense_speaker_conflict_margins = None
    if args.dense_speaker_conflict_margins:
        dense_speaker_conflict_margins = [
            float(value) for value in args.dense_speaker_conflict_margins.split(",")
        ]
        if (
            not dense_speaker_conflict_margins
            or any(value < 0 for value in dense_speaker_conflict_margins)
            or not args.local_embedding_model
            or args.fusion_alpha is None
        ):
            raise ValueError(
                "--dense-speaker-conflict-margins must be non-negative and requires "
                "--local-embedding-model and --fusion-alpha"
            )
    if not 0 <= args.dense_sentence_weight <= 1:
        raise ValueError("--dense-sentence-weight must be between 0 and 1")
    dense_sentence_weights = None
    if args.dense_sentence_weights:
        dense_sentence_weights = [
            float(value) for value in args.dense_sentence_weights.split(",")
        ]
        if (
            not dense_sentence_weights
            or any(value < 0 or value > 1 for value in dense_sentence_weights)
            or not args.local_embedding_model
            or args.fusion_alpha is None
        ):
            raise ValueError(
                "--dense-sentence-weights must be between 0 and 1 and requires "
                "--local-embedding-model and --fusion-alpha"
            )
    if not 0 <= args.dense_image_carry_weight <= 1:
        raise ValueError("--dense-image-carry-weight must be between 0 and 1")
    dense_image_carry_weights = None
    if args.dense_image_carry_weights:
        dense_image_carry_weights = [
            float(value) for value in args.dense_image_carry_weights.split(",")
        ]
        if (
            not dense_image_carry_weights
            or any(value < 0 or value > 1 for value in dense_image_carry_weights)
            or not args.local_embedding_model
            or args.fusion_alpha is None
        ):
            raise ValueError(
                "--dense-image-carry-weights must be between 0 and 1 and requires "
                "--local-embedding-model and --fusion-alpha"
            )
    if not 0 <= args.dense_speaker_coref_weight <= 1:
        raise ValueError("--dense-speaker-coref-weight must be between 0 and 1")
    dense_speaker_coref_weights = None
    if args.dense_speaker_coref_weights:
        dense_speaker_coref_weights = [
            float(value) for value in args.dense_speaker_coref_weights.split(",")
        ]
        if (
            not dense_speaker_coref_weights
            or any(value < 0 or value > 1 for value in dense_speaker_coref_weights)
            or not args.local_embedding_model
            or args.fusion_alpha is None
        ):
            raise ValueError(
                "--dense-speaker-coref-weights must be between 0 and 1 and requires "
                "--local-embedding-model and --fusion-alpha"
            )
    if args.local_cross_encoder_model and not args.local_embedding_model:
        raise ValueError("--local-cross-encoder-model requires --local-embedding-model")
    if args.local_cross_encoder_model and args.local_reranker_model:
        raise ValueError(
            "Use either --local-cross-encoder-model or --local-reranker-model, not both"
        )
    if args.local_yes_no_reranker_url and (
        args.local_reranker_model or args.local_cross_encoder_model
    ):
        raise ValueError(
            "--local-yes-no-reranker-url cannot be combined with FastEmbed rerankers"
        )
    if args.local_yes_no_reranker_url and (
        not args.local_embedding_model or args.fusion_alpha is None
    ):
        raise ValueError(
            "--local-yes-no-reranker-url requires --local-embedding-model and "
            "--fusion-alpha"
        )
    if not 1 <= args.local_yes_no_reranker_batch_size <= 32:
        raise ValueError("--local-yes-no-reranker-batch-size must be between 1 and 32")
    if args.local_yes_no_mask_speakers and args.local_yes_no_max_masked_score:
        raise ValueError(
            "Use either --local-yes-no-mask-speakers or "
            "--local-yes-no-max-masked-score, not both"
        )
    if args.rerank_top_n < 1 or args.rerank_top_n > 100:
        raise ValueError("--rerank-top-n must be between 1 and 100")
    if args.rerank_image_followups < 0 or args.rerank_image_followups > 4:
        raise ValueError("--rerank-image-followups must be between 0 and 4")
    if args.rerank_fusion_weight is not None and not 0 <= args.rerank_fusion_weight <= 1:
        raise ValueError("--rerank-fusion-weight must be between 0 and 1")
    if args.rerank_fusion_weight is not None and not (
        args.local_reranker_model or args.local_yes_no_reranker_url
    ):
        raise ValueError(
            "--rerank-fusion-weight requires --local-reranker-model or "
            "--local-yes-no-reranker-url"
        )
    if not 0 <= args.rerank_near_tie_epsilon <= 1:
        raise ValueError("--rerank-near-tie-epsilon must be between 0 and 1")
    rerank_near_tie_epsilons = None
    if args.rerank_near_tie_epsilons:
        rerank_near_tie_epsilons = [
            float(value) for value in args.rerank_near_tie_epsilons.split(",")
        ]
        if (
            not rerank_near_tie_epsilons
            or any(value <= 0 or value > 1 for value in rerank_near_tie_epsilons)
            or not args.local_yes_no_reranker_url
        ):
            raise ValueError(
                "--rerank-near-tie-epsilons must be in (0, 1] and requires "
                "--local-yes-no-reranker-url"
            )
    if args.session_weight < 0 or args.session_weight > 10:
        raise ValueError("--session-weight must be between 0 and 10")
    if args.session_weight > 0 and (
        not args.local_embedding_model or args.fusion_alpha is None
    ):
        raise ValueError(
            "--session-weight requires --local-embedding-model and --fusion-alpha"
        )
    if args.session_top_n < 0 or args.session_top_n > 100:
        raise ValueError("--session-top-n must be between 0 and 100")
    session_top_ns = None
    if args.session_top_ns:
        session_top_ns = [int(value) for value in args.session_top_ns.split(",")]
        if not session_top_ns or any(value < 1 or value > 100 for value in session_top_ns):
            raise ValueError("--session-top-ns values must be between 1 and 100")
        if not args.local_embedding_model or args.fusion_alpha is None:
            raise ValueError(
                "--session-top-ns requires --local-embedding-model and --fusion-alpha"
            )
    if args.instruction_rerank_top_n < 1 or args.instruction_rerank_top_n > 100:
        raise ValueError("--instruction-rerank-top-n must be between 1 and 100")
    if (
        args.instruction_refine_top_n == 1
        or args.instruction_refine_top_n < 0
        or args.instruction_refine_top_n > args.instruction_rerank_top_n
    ):
        raise ValueError(
            "--instruction-refine-top-n must be 0 or between 2 and "
            "--instruction-rerank-top-n"
        )
    if args.local_instruction_url and (
        not args.local_embedding_model or args.fusion_alpha is None
    ):
        raise ValueError(
            "--local-instruction-url requires --local-embedding-model and --fusion-alpha"
        )
    rerank_top_ns = None
    if args.rerank_top_ns:
        rerank_top_ns = [int(value) for value in args.rerank_top_ns.split(",")]
        if not rerank_top_ns or any(value < 1 or value > 100 for value in rerank_top_ns):
            raise ValueError("--rerank-top-ns values must be between 1 and 100")
        if not args.local_reranker_model or args.fusion_alpha is None:
            raise ValueError(
                "--rerank-top-ns requires --local-reranker-model and --fusion-alpha"
            )
    session_weights = None
    if args.session_weights:
        session_weights = [float(value) for value in args.session_weights.split(",")]
        if not session_weights or any(value < 0 or value > 10 for value in session_weights):
            raise ValueError("--session-weights values must be between 0 and 10")
        if not args.local_embedding_model or args.fusion_alpha is None:
            raise ValueError(
                "--session-weights requires --local-embedding-model and --fusion-alpha"
            )
    rerank_fusion_weights = None
    if args.rerank_fusion_weights:
        rerank_fusion_weights = [
            float(value) for value in args.rerank_fusion_weights.split(",")
        ]
        if not rerank_fusion_weights or any(
            value < 0 or value > 1 for value in rerank_fusion_weights
        ):
            raise ValueError("--rerank-fusion-weights values must be between 0 and 1")
        if not (
            args.local_reranker_model or args.local_yes_no_reranker_url
        ) or args.fusion_alpha is None:
            raise ValueError(
                "--rerank-fusion-weights requires a local reranker and --fusion-alpha"
            )
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo dataset must be a JSON array")
    model = None
    if args.search_model and args.local_search_model_url:
        raise ValueError("Use either --search-model or --local-search-model-url")
    if args.structured_query_plan and not (args.search_model or args.local_search_model_url):
        raise ValueError("--structured-query-plan requires a Search model")
    if args.set_aware_rerank and not args.structured_query_plan:
        raise ValueError("--set-aware-rerank requires --structured-query-plan")
    if args.evidence_need_retrieval and not args.structured_query_plan:
        raise ValueError("--evidence-need-retrieval requires --structured-query-plan")
    if args.evidence_need_retrieval and args.fusion_alpha is not None:
        raise ValueError("--evidence-need-retrieval cannot use --fusion-alpha")
    if args.bridge_retrieval and not args.structured_query_plan:
        raise ValueError("--bridge-retrieval requires --structured-query-plan")
    if args.bridge_retrieval and args.fusion_alpha is not None:
        raise ValueError("--bridge-retrieval cannot use --fusion-alpha")
    if (
        isinstance(args.bridge_max_terms, bool)
        or not 1 <= args.bridge_max_terms <= 5
    ):
        raise ValueError("--bridge-max-terms must be an integer between 1 and 5")
    if (
        isinstance(args.bridge_rrf_weight, bool)
        or not 0 <= args.bridge_rrf_weight <= 1
    ):
        raise ValueError("--bridge-rrf-weight must be between 0 and 1")
    if (
        isinstance(args.bridge_quota, bool)
        or not 0 <= args.bridge_quota <= 30
    ):
        raise ValueError("--bridge-quota must be an integer between 0 and 30")
    if (
        isinstance(args.sidecar_shared_quota, bool)
        or not 0 <= args.sidecar_shared_quota <= 30
    ):
        raise ValueError("--sidecar-shared-quota must be an integer between 0 and 30")
    if (
        args.sidecar_shared_quota > 0
        and not args.evidence_need_retrieval
        and not args.bridge_retrieval
    ):
        raise ValueError("--sidecar-shared-quota requires --evidence-need-retrieval or --bridge-retrieval")
    if args.query_relaxation and not args.evidence_need_retrieval:
        raise ValueError("--query-relaxation requires --evidence-need-retrieval")
    if (
        isinstance(args.relax_rrf_weight, bool)
        or not 0 <= args.relax_rrf_weight <= 1
    ):
        raise ValueError("--relax-rrf-weight must be between 0 and 1")
    if (
        isinstance(args.relax_quota, bool)
        or not 0 <= args.relax_quota <= 30
    ):
        raise ValueError("--relax-quota must be an integer between 0 and 30")
    if args.p5_gate and not args.structured_query_plan:
        raise ValueError("--p5-gate requires --structured-query-plan")
    if (
        isinstance(args.p5_near_tie_epsilon, bool)
        or not 0 <= args.p5_near_tie_epsilon <= 1
    ):
        raise ValueError("--p5-near-tie-epsilon must be between 0 and 1")
    if (
        isinstance(args.p5_min_evidence_channels, bool)
        or not 1 <= args.p5_min_evidence_channels <= 30
    ):
        raise ValueError("--p5-min-evidence-channels must be an integer between 1 and 30")
    if (
        isinstance(args.p5_confidence_margin, bool)
        or not 0 <= args.p5_confidence_margin <= 1
    ):
        raise ValueError("--p5-confidence-margin must be between 0 and 1")
    if not isinstance(args.p5_strata, str) or not args.p5_strata.strip():
        raise ValueError("--p5-strata must be a non-empty string")
    strata_parts = {part.strip() for part in args.p5_strata.split(",") if part.strip()}
    if not strata_parts or not strata_parts.issubset({"all", "temporal", "correction"}):
        raise ValueError("--p5-strata must be a comma-separated subset of all/temporal/correction")
    if args.evidence_graph and not args.structured_query_plan:
        raise ValueError("--evidence-graph requires --structured-query-plan")
    if args.graph_selective and not args.evidence_graph:
        raise ValueError("--graph-selective requires --evidence-graph")
    if (
        isinstance(args.graph_rrf_weight, bool)
        or not 0 <= args.graph_rrf_weight <= 1
    ):
        raise ValueError("--graph-rrf-weight must be between 0 and 1")
    if (
        isinstance(args.graph_max_candidates, bool)
        or not 0 <= args.graph_max_candidates <= 100
    ):
        raise ValueError("--graph-max-candidates must be between 0 and 100")
    if (
        isinstance(args.graph_quota, bool)
        or not 0 <= args.graph_quota <= 30
    ):
        raise ValueError("--graph-quota must be an integer between 0 and 30")
    if (
        isinstance(args.llm_rerank_top_n, bool)
        or not 0 <= args.llm_rerank_top_n <= 30
    ):
        raise ValueError("--llm-rerank-top-n must be an integer between 0 and 30")
    if (
        isinstance(args.evidence_need_quota, bool)
        or not 1 <= args.evidence_need_quota <= 30
    ):
        raise ValueError("--evidence-need-quota must be an integer between 1 and 30")
    if (
        isinstance(args.evidence_need_rrf_weight, bool)
        or not 0 <= args.evidence_need_rrf_weight <= 1
    ):
        raise ValueError("--evidence-need-rrf-weight must be between 0 and 1")
    if args.need_select_by_bm25 and not args.evidence_need_retrieval:
        raise ValueError("--need-select-by-bm25 requires --evidence-need-retrieval")
    if args.search_model:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("--search-model requires OPENAI_API_KEY")
        model = SearchOnlyModel(
            api_key, rank_prompt_v2=args.rank_prompt_v2,
            timeout_seconds=args.model_timeout,
        )
    elif args.local_search_model_url:
        model = SearchOnlyModel(
            "local-only", model_name=args.local_search_model_name,
            base_url=args.local_search_model_url, disable_thinking=True,
            rank_prompt_v2=args.rank_prompt_v2,
            timeout_seconds=args.model_timeout,
        )
    semantic_retriever = None
    if args.local_embedding_url:
        semantic_retriever = LocalHTTPEmbeddingRetriever(
            args.local_embedding_url,
            model_name=args.local_embedding_http_model,
            query_instruction=args.local_embedding_instruction,
        )
    elif args.local_embedding_model:
        semantic_retriever = (
            LocalLateInteractionReranker(
                args.local_embedding_model,
                device=args.local_device,
                cache_dir=args.local_cache_dir,
            )
            if late_interaction_first_stage
            else LocalSemanticRetriever(
                args.local_embedding_model,
                device=args.local_device,
                cache_dir=args.local_cache_dir,
            )
        )
    if args.instruction_speaker_conflict_only and (
        not args.local_instruction_url
        or args.dense_speaker_conflict_margin is None
        or not args.dense_speaker_conflict_gate_only
    ):
        raise ValueError(
            "--instruction-speaker-conflict-only requires --local-instruction-url, "
            "--dense-speaker-conflict-margin, and --dense-speaker-conflict-gate-only"
        )
    local_reranker = None
    if args.local_reranker_model:
        local_reranker = LocalLateInteractionReranker(
            args.local_reranker_model,
            device=args.local_device,
            cache_dir=args.local_cache_dir,
        )
    elif args.local_cross_encoder_model:
        local_reranker = LocalCrossEncoderReranker(
            args.local_cross_encoder_model,
            device=args.local_device,
            cache_dir=args.local_cache_dir,
        )
    elif args.local_yes_no_reranker_url:
        local_reranker = LocalYesNoReranker(
            args.local_yes_no_reranker_url,
            model_name=args.local_yes_no_reranker_model,
            instruction=args.local_yes_no_reranker_instruction,
            batch_size=args.local_yes_no_reranker_batch_size,
            mask_speakers=args.local_yes_no_mask_speakers,
            max_masked_score=args.local_yes_no_max_masked_score,
        )
    local_instruction_reranker = None
    if args.local_instruction_url:
        local_instruction_reranker = (
            LocalDualStrategyReranker(
                args.local_instruction_url,
                model_name=args.local_instruction_model,
            )
            if args.instruction_strategy == "dual_arbitrate_top1"
            else LocalInstructionReranker(
                args.local_instruction_url,
                model_name=args.local_instruction_model,
                thinking=args.instruction_thinking,
                strategy=args.instruction_strategy,
            )
        )
    local_query_expander = None
    if args.local_query_expansion_url:
        if not args.local_embedding_model or args.fusion_alpha is None:
            raise ValueError(
                "--local-query-expansion-url requires --local-embedding-model and --fusion-alpha"
            )
        local_query_expander = LocalQueryExpander(
            args.local_query_expansion_url,
            model_name=args.local_query_expansion_model,
        )
    if dense_context_weights is not None:
        sweep = {
            str(weight): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_context_weight=weight,
                local_reranker=local_reranker,
                rerank_top_n=args.rerank_top_n,
                rerank_image_followups=args.rerank_image_followups,
                session_fusion_weight=args.session_weight,
                rerank_fusion_weight=args.rerank_fusion_weight,
                local_instruction_reranker=local_instruction_reranker,
                instruction_rerank_top_n=args.instruction_rerank_top_n,
                instruction_refine_top_n=args.instruction_refine_top_n,
            )
            for weight in dense_context_weights
        }
        print(json.dumps({
            "scope": "retrieval-only context-augmented dense-key sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "local_reranker_model": args.local_reranker_model,
            "dense_context_weight_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if dense_sentence_weights is not None:
        sweep = {
            str(weight): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_time_weight=args.dense_time_weight,
                dense_sentence_weight=weight,
            )
            for weight in dense_sentence_weights
        }
        print(json.dumps({
            "scope": "retrieval-only latent sentence-key sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "dense_time_weight": args.dense_time_weight,
            "dense_sentence_weight_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if dense_image_carry_weights is not None:
        sweep = {
            str(weight): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_time_weight=args.dense_time_weight,
                dense_image_carry_weight=weight,
            )
            for weight in dense_image_carry_weights
        }
        print(json.dumps({
            "scope": "retrieval-only prior-image carry dense-key sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "dense_time_weight": args.dense_time_weight,
            "dense_image_carry_weight_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if dense_speaker_coref_weights is not None:
        sweep = {
            str(weight): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_time_weight=args.dense_time_weight,
                dense_speaker_coref_weight=weight,
            )
            for weight in dense_speaker_coref_weights
        }
        print(json.dumps({
            "scope": "retrieval-only speaker-coreference dense-key sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "dense_time_weight": args.dense_time_weight,
            "dense_speaker_coref_weight_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if dense_speaker_conflict_margins is not None:
        sweep = {
            str(margin): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_time_weight=args.dense_time_weight,
                dense_speaker_conflict_margin=margin,
            )
            for margin in dense_speaker_conflict_margins
        }
        print(json.dumps({
            "scope": "retrieval-only conditional speaker-conflict sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "dense_time_weight": args.dense_time_weight,
            "speaker_conflict_margin_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if session_top_ns is not None:
        sweep = {
            str(top_n): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                local_reranker=local_reranker,
                rerank_top_n=args.rerank_top_n,
                rerank_image_followups=args.rerank_image_followups,
                session_fusion_weight=args.session_weight,
                session_top_n=top_n,
            )
            for top_n in session_top_ns
        }
        print(json.dumps({
            "scope": "retrieval-only hard session-to-turn sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "local_reranker_model": args.local_reranker_model,
            "session_top_n_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if dense_time_weights is not None:
        sweep = {
            str(weight): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_context_weight=args.dense_context_weight,
                dense_time_weight=weight,
                local_reranker=local_reranker,
                rerank_top_n=args.rerank_top_n,
                rerank_image_followups=args.rerank_image_followups,
                session_fusion_weight=args.session_weight,
                rerank_fusion_weight=args.rerank_fusion_weight,
            )
            for weight in dense_time_weights
        }
        print(json.dumps({
            "scope": "retrieval-only time-aware dense-key sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "local_reranker_model": args.local_reranker_model,
            "dense_time_weight_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if rerank_top_ns is not None:
        sweep = {
            str(top_n): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_context_weight=args.dense_context_weight,
                local_reranker=local_reranker,
                rerank_top_n=top_n,
                rerank_image_followups=args.rerank_image_followups,
                session_fusion_weight=args.session_weight,
                rerank_fusion_weight=args.rerank_fusion_weight,
                local_instruction_reranker=local_instruction_reranker,
                instruction_rerank_top_n=args.instruction_rerank_top_n,
                instruction_refine_top_n=args.instruction_refine_top_n,
            )
            for top_n in rerank_top_ns
        }
        print(json.dumps({
            "scope": "retrieval-only local rerank-pool sweep; shared embedding caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "local_reranker_model": args.local_reranker_model,
            "rerank_top_n_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if rerank_fusion_weights is not None:
        sweep = {
            str(weight): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_context_weight=args.dense_context_weight,
                local_reranker=local_reranker,
                rerank_top_n=args.rerank_top_n,
                rerank_image_followups=args.rerank_image_followups,
                session_fusion_weight=args.session_weight,
                rerank_fusion_weight=weight,
                local_instruction_reranker=local_instruction_reranker,
                instruction_rerank_top_n=args.instruction_rerank_top_n,
                instruction_refine_top_n=args.instruction_refine_top_n,
            )
            for weight in rerank_fusion_weights
        }
        print(json.dumps({
            "scope": "retrieval-only first-stage/reranker score-fusion sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "local_reranker_model": args.local_reranker_model,
            "rerank_top_n": args.rerank_top_n,
            "session_fusion_weight": args.session_weight,
            "rerank_fusion_weight_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if rerank_near_tie_epsilons is not None:
        sweep = {
            str(epsilon): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_context_weight=args.dense_context_weight,
                dense_time_weight=args.dense_time_weight,
                local_reranker=local_reranker,
                rerank_top_n=args.rerank_top_n,
                rerank_near_tie_epsilon=epsilon,
            )
            for epsilon in rerank_near_tie_epsilons
        }
        print(json.dumps({
            "scope": "retrieval-only reranker near-tie sweep; shared caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "dense_time_weight": args.dense_time_weight,
            "rerank_top_n": args.rerank_top_n,
            "rerank_near_tie_epsilon_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if session_weights is not None:
        sweep = {
            str(weight): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=args.fusion_alpha,
                dense_context_weight=args.dense_context_weight,
                local_reranker=local_reranker,
                rerank_top_n=args.rerank_top_n,
                rerank_image_followups=args.rerank_image_followups,
                session_fusion_weight=weight,
                rerank_fusion_weight=args.rerank_fusion_weight,
                local_instruction_reranker=local_instruction_reranker,
                instruction_rerank_top_n=args.instruction_rerank_top_n,
                instruction_refine_top_n=args.instruction_refine_top_n,
            )
            for weight in session_weights
        }
        print(json.dumps({
            "scope": "retrieval-only hierarchical session-weight sweep; shared embedding caches",
            "local_embedding_model": args.local_embedding_model,
            "dense_fusion_alpha": args.fusion_alpha,
            "local_reranker_model": args.local_reranker_model,
            "rerank_top_n": args.rerank_top_n if args.local_reranker_model else None,
            "session_weight_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    if fusion_alphas is not None:
        sweep = {
            str(alpha): evaluate(
                data,
                top_ks,
                args.max_questions,
                model=model,
                semantic_retriever=semantic_retriever,
                dense_fusion_alpha=alpha,
                dense_context_weight=args.dense_context_weight,
                local_reranker=local_reranker,
                rerank_top_n=args.rerank_top_n,
                rerank_image_followups=args.rerank_image_followups,
                session_fusion_weight=args.session_weight,
                rerank_fusion_weight=args.rerank_fusion_weight,
                local_instruction_reranker=local_instruction_reranker,
                instruction_rerank_top_n=args.instruction_rerank_top_n,
                instruction_refine_top_n=args.instruction_refine_top_n,
            )
            for alpha in fusion_alphas
        }
        print(json.dumps({
            "scope": "retrieval-only local fusion sweep; shared embedding caches",
            "local_embedding_model": args.local_embedding_model,
            "local_reranker_model": args.local_reranker_model,
            "rerank_top_n": args.rerank_top_n if args.local_reranker_model else None,
            "fusion_sweep": sweep,
        }, ensure_ascii=False, indent=2))
        return
    current = evaluate(
        data,
        top_ks,
        args.max_questions,
        question_offset=args.question_offset,
        model=model,
        semantic_retriever=semantic_retriever,
        dense_rrf_weight=args.dense_weight,
        dense_fusion_alpha=args.fusion_alpha,
        dense_context_weight=args.dense_context_weight,
        dense_time_weight=args.dense_time_weight,
        dense_speaker_mask_max=args.dense_speaker_mask_max,
        dense_speaker_conflict_margin=args.dense_speaker_conflict_margin,
        dense_speaker_conflict_gate_only=args.dense_speaker_conflict_gate_only,
        dense_sentence_weight=args.dense_sentence_weight,
        dense_image_carry_weight=args.dense_image_carry_weight,
        dense_speaker_coref_weight=args.dense_speaker_coref_weight,
        dense_speaker_swap_max=args.dense_speaker_swap_max,
        local_reranker=local_reranker,
        rerank_top_n=args.rerank_top_n,
        rerank_image_followups=args.rerank_image_followups,
        session_fusion_weight=args.session_weight,
        session_top_n=args.session_top_n,
        rerank_fusion_weight=args.rerank_fusion_weight,
        rerank_near_tie_epsilon=args.rerank_near_tie_epsilon,
        local_instruction_reranker=local_instruction_reranker,
        instruction_speaker_conflict_only=args.instruction_speaker_conflict_only,
        local_query_expander=local_query_expander,
        instruction_rerank_top_n=args.instruction_rerank_top_n,
        instruction_refine_top_n=args.instruction_refine_top_n,
        include_hit_bitmap=args.include_hit_bitmap,
        structured_query_plan=args.structured_query_plan,
        set_aware_rerank=args.set_aware_rerank,
        include_question_diagnostics=args.include_question_diagnostics,
        evidence_need_retrieval=args.evidence_need_retrieval,
        evidence_need_quota=args.evidence_need_quota,
        evidence_need_rrf_weight=args.evidence_need_rrf_weight,
        need_select_by_bm25=args.need_select_by_bm25,
        adjacent_turn_expansion=args.adjacent_turn_expansion,
        bridge_retrieval=args.bridge_retrieval,
        bridge_max_terms=args.bridge_max_terms,
        bridge_rrf_weight=args.bridge_rrf_weight,
        bridge_rerank_quota=args.bridge_quota,
        sidecar_shared_quota=args.sidecar_shared_quota,
        query_relaxation=args.query_relaxation,
        relax_rrf_weight=args.relax_rrf_weight,
        relax_quota=args.relax_quota,
        p5_gate=args.p5_gate,
        p5_near_tie_epsilon=args.p5_near_tie_epsilon,
        p5_min_evidence_channels=args.p5_min_evidence_channels,
        p5_confidence_margin=args.p5_confidence_margin,
        p5_strata=args.p5_strata,
        llm_rerank_top_n=args.llm_rerank_top_n,
        evidence_graph=args.evidence_graph,
        graph_selective=args.graph_selective,
        graph_rrf_weight=args.graph_rrf_weight,
        graph_max_candidates=args.graph_max_candidates,
        graph_rerank_quota=args.graph_quota,
    )
    if not args.compare_v020:
        rendered = json.dumps(current, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return
    baseline = evaluate(
        data, top_ks, args.max_questions,
        question_offset=args.question_offset, retriever="v0.2.0",
    )
    comparison = {
        "scope": "retrieval-only, no external API call",
        "local_embedding_model": args.local_embedding_model,
        "dense_rrf_weight": args.dense_weight if args.local_embedding_model else None,
        "dense_fusion_alpha": args.fusion_alpha,
        "dense_context_weight": args.dense_context_weight,
        "dense_time_weight": args.dense_time_weight,
        "dense_speaker_mask_max": args.dense_speaker_mask_max,
        "local_reranker_model": args.local_reranker_model,
        "rerank_top_n": args.rerank_top_n if args.local_reranker_model else None,
        "session_fusion_weight": args.session_weight,
        "rerank_fusion_weight": args.rerank_fusion_weight,
        "local_instruction_model": (
            args.local_instruction_model if args.local_instruction_url else None
        ),
        "instruction_rerank_top_n": (
            args.instruction_rerank_top_n if args.local_instruction_url else None
        ),
        "instruction_refine_top_n": (
            args.instruction_refine_top_n if args.local_instruction_url else None
        ),
        "instruction_thinking": args.instruction_thinking if args.local_instruction_url else None,
        "instruction_strategy": (
            args.instruction_strategy if args.local_instruction_url else None
        ),
        "v0.2.0": baseline,
        "current": current,
        "delta_hit_at_1": round(current["hit_at_k"]["1"] - baseline["hit_at_k"]["1"], 4),
        "delta_mrr": round(current["mrr"] - baseline["mrr"], 4),
    }
    rendered = json.dumps(comparison, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
