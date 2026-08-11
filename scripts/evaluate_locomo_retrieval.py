"""Evaluate evidence retrieval on a local LoCoMo JSON file.

The dataset is intentionally supplied by path and is never copied into this
repository. Use only where the dataset licence and competition rules permit.
"""

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import AddRequest
from app.model import MemoryModel
from app.local_semantic import (
    LocalCrossEncoderReranker,
    LocalLateInteractionReranker,
    LocalSemanticRetriever,
)
from app.local_instruction import (
    LocalInstructionReranker,
    LocalQueryExpander,
    LocalYesNoReranker,
)
from app.storage import MemoryStore


class SearchOnlyModel:
    """Use the configured model for query planning/ranking without Add-time calls."""

    def __init__(self, api_key: str) -> None:
        self.model = MemoryModel(api_key)

    def extract_facts(self, content: str) -> List[str]:
        return []

    def plan_query(self, query: str, options: List[str]) -> List[str]:
        return self.model.plan_query(query, options)

    def rank_candidates(self, query: str, options: List[str], candidates: List[Dict[str, str]]) -> List[str]:
        return self.model.rank_candidates(query, options, candidates)


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


def evaluate(samples: Iterable[Dict[str, object]], top_ks: List[int], max_questions: int | None,
             retriever: str = "current", model: object = None,
             semantic_retriever: object = None, dense_rrf_weight: float = 1.0,
             dense_fusion_alpha: float | None = None, local_reranker: object = None,
             dense_context_weight: float = 0.0,
             dense_time_weight: float = 0.0,
             rerank_top_n: int = 10, session_fusion_weight: float = 0.0,
             session_top_n: int = 0,
             rerank_fusion_weight: float | None = None,
             local_instruction_reranker: object = None,
             local_query_expander: object = None,
             instruction_rerank_top_n: int = 10,
             instruction_refine_top_n: int = 0) -> Dict[str, object]:
    if retriever not in {"current", "v0.2.0"}:
        raise ValueError("retriever must be current or v0.2.0")
    hit_counts = {top_k: 0 for top_k in top_ks}
    evidence_hits = {top_k: 0 for top_k in top_ks}
    category_counts = defaultdict(lambda: {"questions": 0, "hit_at_1": 0})
    question_count = evidence_total = 0
    reciprocal_rank_sum = 0.0
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
                local_reranker=local_reranker,
                rerank_top_n=rerank_top_n,
                session_fusion_weight=session_fusion_weight,
                session_top_n=session_top_n,
                rerank_fusion_weight=rerank_fusion_weight,
                local_instruction_reranker=local_instruction_reranker,
                local_query_expander=local_query_expander,
                instruction_rerank_top_n=instruction_rerank_top_n,
                instruction_refine_top_n=instruction_refine_top_n,
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
            for qa in sample.get("qa", []):
                if max_questions is not None and question_count >= max_questions:
                    break
                if not isinstance(qa, dict):
                    continue
                expected = [evidence_text[item] for item in qa.get("evidence", []) if item in evidence_text]
                if not expected or not qa.get("question"):
                    continue
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
                first_rank = next((rank for rank in ranks if rank is not None), None)
                category = str(qa.get("category", "unknown"))
                category_counts[category]["questions"] += 1
                if first_rank == 1:
                    category_counts[category]["hit_at_1"] += 1
                if first_rank is not None:
                    reciprocal_rank_sum += 1.0 / first_rank
                for top_k in top_ks:
                    matches = [rank for rank in ranks if rank is not None and rank <= top_k]
                    evidence_hits[top_k] += len(matches)
                    if matches:
                        hit_counts[top_k] += 1

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
    }
    if local_instruction_reranker is not None:
        result["instruction_format_fallbacks"] = (
            getattr(local_instruction_reranker, "invalid_response_count", 0)
            - instruction_failures_before
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-only LoCoMo evaluation.")
    parser.add_argument("--dataset", required=True, help="Local path to LoCoMo locomo10.json")
    parser.add_argument("--top-k", default="1,3,10", help="Comma-separated retrieval depths")
    parser.add_argument("--max-questions", type=int, help="Optional cap for a smoke test")
    parser.add_argument("--compare-v020", action="store_true", help="Also reproduce v0.2.0 no-model retrieval")
    parser.add_argument(
        "--search-model", action="store_true",
        help="Use OPENAI_API_KEY for query planning/ranking, but skip Add-time fact calls",
    )
    parser.add_argument(
        "--local-embedding-model",
        help="Use a local FastEmbed model for dense retrieval, for example BAAI/bge-small-en-v1.5",
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
    parser.add_argument("--local-reranker-model", help="FastEmbed late-interaction model")
    parser.add_argument("--local-cross-encoder-model", help="FastEmbed cross-encoder reranker")
    parser.add_argument(
        "--local-yes-no-reranker-url",
        help="Loopback URL for a Qwen-style yes/no logprob reranker",
    )
    parser.add_argument("--local-yes-no-reranker-model", default="local")
    parser.add_argument("--rerank-top-n", type=int, default=10)
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
    parser.add_argument("--instruction-thinking", action="store_true")
    parser.add_argument(
        "--instruction-strategy", choices=["direct", "chain_of_note"], default="direct"
    )
    args = parser.parse_args()
    if args.local_late_interaction_model and args.local_embedding_model:
        raise ValueError(
            "Use either --local-embedding-model or --local-late-interaction-model, not both"
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
        args.session_top_ns,
    )) > 1:
        raise ValueError("Use only one shared-cache sweep at a time")
    fusion_alphas = None
    if args.fusion_alphas:
        fusion_alphas = [float(value) for value in args.fusion_alphas.split(",")]
        if not fusion_alphas or any(value < 0 or value > 1 for value in fusion_alphas):
            raise ValueError("--fusion-alphas values must be between 0 and 1")
    if args.fusion_alpha is not None and not args.local_embedding_model:
        raise ValueError("--fusion-alpha requires --local-embedding-model")
    if fusion_alphas is not None and not args.local_embedding_model:
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
    if args.rerank_top_n < 1 or args.rerank_top_n > 100:
        raise ValueError("--rerank-top-n must be between 1 and 100")
    if args.rerank_fusion_weight is not None and not 0 <= args.rerank_fusion_weight <= 1:
        raise ValueError("--rerank-fusion-weight must be between 0 and 1")
    if args.rerank_fusion_weight is not None and not (
        args.local_reranker_model or args.local_yes_no_reranker_url
    ):
        raise ValueError(
            "--rerank-fusion-weight requires --local-reranker-model or "
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
    if args.search_model:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("--search-model requires OPENAI_API_KEY")
        model = SearchOnlyModel(api_key)
    semantic_retriever = None
    if args.local_embedding_model:
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
        )
    local_instruction_reranker = None
    if args.local_instruction_url:
        local_instruction_reranker = LocalInstructionReranker(
            args.local_instruction_url,
            model_name=args.local_instruction_model,
            thinking=args.instruction_thinking,
            strategy=args.instruction_strategy,
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
        model=model,
        semantic_retriever=semantic_retriever,
        dense_rrf_weight=args.dense_weight,
        dense_fusion_alpha=args.fusion_alpha,
        dense_context_weight=args.dense_context_weight,
        dense_time_weight=args.dense_time_weight,
        local_reranker=local_reranker,
        rerank_top_n=args.rerank_top_n,
        session_fusion_weight=args.session_weight,
        session_top_n=args.session_top_n,
        rerank_fusion_weight=args.rerank_fusion_weight,
        local_instruction_reranker=local_instruction_reranker,
        local_query_expander=local_query_expander,
        instruction_rerank_top_n=args.instruction_rerank_top_n,
        instruction_refine_top_n=args.instruction_refine_top_n,
    )
    if not args.compare_v020:
        print(json.dumps(current, ensure_ascii=False, indent=2))
        return
    baseline = evaluate(data, top_ks, args.max_questions, retriever="v0.2.0")
    print(json.dumps({
        "scope": "retrieval-only, no external API call",
        "local_embedding_model": args.local_embedding_model,
        "dense_rrf_weight": args.dense_weight if args.local_embedding_model else None,
        "dense_fusion_alpha": args.fusion_alpha,
        "dense_context_weight": args.dense_context_weight,
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
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
