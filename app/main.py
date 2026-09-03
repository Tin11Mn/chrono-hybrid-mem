import math
import os

from fastapi import FastAPI

from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .model import model_from_environment
from .local_semantic import (
    local_reranker_from_environment,
    local_semantic_retriever_from_environment,
)
from .local_instruction import (
    local_instruction_reranker_from_environment,
    local_query_expander_from_environment,
    local_yes_no_reranker_from_environment,
)
from .storage import MemoryStore


def temporal_bonus_from_environment() -> float:
    value = float(os.getenv("MEMORY_TEMPORAL_BONUS", "0"))
    if value < 0 or value > 0.01:
        raise RuntimeError("MEMORY_TEMPORAL_BONUS must be between 0 and 0.01")
    return value


def structured_query_plan_from_environment() -> bool:
    return os.getenv("MEMORY_STRUCTURED_QUERY_PLAN", "true").lower() == "true"


def set_aware_rerank_from_environment() -> bool:
    return os.getenv("MEMORY_SET_AWARE_RERANK", "false").lower() == "true"


def evidence_need_retrieval_from_environment() -> bool:
    """Return the validated P4-A q2 baseline switch (enabled by default)."""

    return os.getenv("MEMORY_EVIDENCE_NEED_RETRIEVAL", "true").lower() == "true"


def adjacent_turn_expansion_from_environment() -> bool:
    return os.getenv("MEMORY_ADJACENT_TURN_EXPANSION", "false").lower() == "true"


def _adjacent_turn_limit_from_environment(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            "{} must be an integer between 0 and {}".format(
                name, MemoryStore.MODEL_RERANK_LIMIT
            )
        ) from error
    if value < 0 or value > MemoryStore.MODEL_RERANK_LIMIT:
        raise RuntimeError(
            "{} must be between 0 and {}".format(
                name, MemoryStore.MODEL_RERANK_LIMIT
            )
        )
    return value


def adjacent_seed_limit_from_environment() -> int:
    return _adjacent_turn_limit_from_environment("MEMORY_ADJACENT_SEED_LIMIT", 4)


def adjacent_candidate_limit_from_environment() -> int:
    return _adjacent_turn_limit_from_environment(
        "MEMORY_ADJACENT_CANDIDATE_LIMIT", 4
    )


def evidence_graph_from_environment() -> bool:
    return os.getenv("MEMORY_EVIDENCE_GRAPH", "false").lower() == "true"


def evidence_anchors_from_environment() -> bool:
    """Return whether the isolated P3-B1 mention-anchor experiment is enabled."""

    return os.getenv("MEMORY_EVIDENCE_ANCHORS", "false").lower() == "true"


def _frozen_anchor_int_from_environment(name: str, expected: int) -> int:
    raw_value = os.getenv(name, str(expected))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError("P3-B1 requires {}={}".format(name, expected)) from error
    if value != expected:
        raise RuntimeError("P3-B1 requires {}={}".format(name, expected))
    return value


def anchor_seed_limit_from_environment() -> int:
    return _frozen_anchor_int_from_environment(
        "MEMORY_ANCHOR_SEED_LIMIT", MemoryStore.ANCHOR_SEED_LIMIT
    )


def anchor_max_candidates_from_environment() -> int:
    return _frozen_anchor_int_from_environment(
        "MEMORY_ANCHOR_MAX_CANDIDATES", MemoryStore.ANCHOR_MAX_CANDIDATES
    )


def anchor_rrf_weight_from_environment() -> float:
    expected = MemoryStore.ANCHOR_RRF_WEIGHT
    raw_value = os.getenv("MEMORY_ANCHOR_RRF_WEIGHT", str(expected))
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(
            "P3-B1 requires MEMORY_ANCHOR_RRF_WEIGHT={}".format(expected)
        ) from error
    if not math.isfinite(value) or value != expected:
        raise RuntimeError(
            "P3-B1 requires MEMORY_ANCHOR_RRF_WEIGHT={}".format(expected)
        )
    return value


def anchor_rerank_quota_from_environment() -> int:
    return _frozen_anchor_int_from_environment(
        "MEMORY_ANCHOR_RERANK_QUOTA", MemoryStore.ANCHOR_RERANK_QUOTA
    )


def graph_max_hops_from_environment() -> int:
    value = int(os.getenv("MEMORY_GRAPH_MAX_HOPS", "1"))
    if value != 1:
        raise RuntimeError("P3-A requires MEMORY_GRAPH_MAX_HOPS=1")
    return value


def graph_temporal_from_environment() -> bool:
    value = os.getenv("MEMORY_GRAPH_TEMPORAL", "false").lower() == "true"
    if value:
        raise RuntimeError("P3-A requires MEMORY_GRAPH_TEMPORAL=false")
    return False


def graph_rrf_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_GRAPH_RRF_WEIGHT", "0.025"))
    if not math.isfinite(value) or value < 0 or value > 1:
        raise RuntimeError("MEMORY_GRAPH_RRF_WEIGHT must be finite and between 0 and 1")
    return value


def graph_max_candidates_from_environment() -> int:
    value = int(os.getenv("MEMORY_GRAPH_MAX_CANDIDATES", "20"))
    if value < 0 or value > 100:
        raise RuntimeError("MEMORY_GRAPH_MAX_CANDIDATES must be between 0 and 100")
    return value


def graph_rerank_quota_from_environment() -> int:
    value = int(os.getenv("MEMORY_GRAPH_RERANK_QUOTA", "4"))
    if value < 0 or value > MemoryStore.MODEL_RERANK_LIMIT:
        raise RuntimeError(
            "MEMORY_GRAPH_RERANK_QUOTA must be between 0 and {}".format(
                MemoryStore.MODEL_RERANK_LIMIT
            )
        )
    return value


def dense_rrf_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_DENSE_RRF_WEIGHT", "1"))
    if value < 0 or value > 10:
        raise RuntimeError("MEMORY_DENSE_RRF_WEIGHT must be between 0 and 10")
    return value


def dense_fusion_alpha_from_environment():
    raw_value = os.getenv("MEMORY_DENSE_FUSION_ALPHA", "").strip()
    if not raw_value:
        return None
    value = float(raw_value)
    if value < 0 or value > 1:
        raise RuntimeError("MEMORY_DENSE_FUSION_ALPHA must be between 0 and 1")
    return value


def dense_context_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_DENSE_CONTEXT_WEIGHT", "0"))
    if value < 0 or value > 1:
        raise RuntimeError("MEMORY_DENSE_CONTEXT_WEIGHT must be between 0 and 1")
    return value


def dense_time_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_DENSE_TIME_WEIGHT", "0"))
    if value < 0 or value > 1:
        raise RuntimeError("MEMORY_DENSE_TIME_WEIGHT must be between 0 and 1")
    return value


def dense_speaker_mask_max_from_environment() -> bool:
    return os.getenv("MEMORY_DENSE_SPEAKER_MASK_MAX", "false").lower() == "true"


def dense_speaker_conflict_margin_from_environment():
    raw_value = os.getenv("MEMORY_DENSE_SPEAKER_CONFLICT_MARGIN", "").strip()
    if not raw_value:
        return None
    value = float(raw_value)
    if value < 0:
        raise RuntimeError("MEMORY_DENSE_SPEAKER_CONFLICT_MARGIN must be non-negative")
    return value


def dense_speaker_conflict_gate_only_from_environment() -> bool:
    return (
        os.getenv("MEMORY_DENSE_SPEAKER_CONFLICT_GATE_ONLY", "false").lower()
        == "true"
    )


def dense_sentence_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_DENSE_SENTENCE_WEIGHT", "0"))
    if value < 0 or value > 1:
        raise RuntimeError("MEMORY_DENSE_SENTENCE_WEIGHT must be between 0 and 1")
    return value


def dense_image_carry_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_DENSE_IMAGE_CARRY_WEIGHT", "0"))
    if value < 0 or value > 1:
        raise RuntimeError("MEMORY_DENSE_IMAGE_CARRY_WEIGHT must be between 0 and 1")
    return value


def dense_speaker_coref_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_DENSE_SPEAKER_COREF_WEIGHT", "0"))
    if value < 0 or value > 1:
        raise RuntimeError("MEMORY_DENSE_SPEAKER_COREF_WEIGHT must be between 0 and 1")
    return value


def instruction_speaker_conflict_only_from_environment() -> bool:
    return (
        os.getenv("MEMORY_LOCAL_INSTRUCTION_SPEAKER_CONFLICT_ONLY", "false").lower()
        == "true"
    )


def rerank_top_n_from_environment() -> int:
    value = int(os.getenv("MEMORY_LOCAL_RERANK_TOP_N", "5"))
    if value < 1 or value > 100:
        raise RuntimeError("MEMORY_LOCAL_RERANK_TOP_N must be between 1 and 100")
    return value


def rerank_image_followups_from_environment() -> int:
    value = int(os.getenv("MEMORY_LOCAL_RERANK_IMAGE_FOLLOWUPS", "0"))
    if value < 0 or value > 4:
        raise RuntimeError("MEMORY_LOCAL_RERANK_IMAGE_FOLLOWUPS must be between 0 and 4")
    return value


def session_fusion_weight_from_environment() -> float:
    value = float(os.getenv("MEMORY_SESSION_FUSION_WEIGHT", "0"))
    if value < 0 or value > 10:
        raise RuntimeError("MEMORY_SESSION_FUSION_WEIGHT must be between 0 and 10")
    return value


def session_top_n_from_environment() -> int:
    value = int(os.getenv("MEMORY_SESSION_TOP_N", "0"))
    if value < 0 or value > 100:
        raise RuntimeError("MEMORY_SESSION_TOP_N must be between 0 and 100")
    return value


def rerank_fusion_weight_from_environment():
    raw_value = os.getenv("MEMORY_LOCAL_RERANK_FUSION_WEIGHT", "").strip()
    if not raw_value:
        return None
    value = float(raw_value)
    if value < 0 or value > 1:
        raise RuntimeError("MEMORY_LOCAL_RERANK_FUSION_WEIGHT must be between 0 and 1")
    return value


def rerank_near_tie_epsilon_from_environment() -> float:
    value = float(os.getenv("MEMORY_LOCAL_RERANK_NEAR_TIE_EPSILON", "0"))
    if value < 0 or value > 1:
        raise RuntimeError(
            "MEMORY_LOCAL_RERANK_NEAR_TIE_EPSILON must be between 0 and 1"
        )
    return value


def instruction_rerank_top_n_from_environment() -> int:
    value = int(os.getenv("MEMORY_LOCAL_INSTRUCTION_TOP_N", "10"))
    if value < 1 or value > 100:
        raise RuntimeError("MEMORY_LOCAL_INSTRUCTION_TOP_N must be between 1 and 100")
    return value


def instruction_refine_top_n_from_environment() -> int:
    value = int(os.getenv("MEMORY_LOCAL_INSTRUCTION_REFINE_TOP_N", "0"))
    if value == 1 or value < 0 or value > 100:
        raise RuntimeError(
            "MEMORY_LOCAL_INSTRUCTION_REFINE_TOP_N must be 0 or between 2 and 100"
        )
    first_pass = instruction_rerank_top_n_from_environment()
    if value > first_pass:
        raise RuntimeError(
            "MEMORY_LOCAL_INSTRUCTION_REFINE_TOP_N cannot exceed "
            "MEMORY_LOCAL_INSTRUCTION_TOP_N"
        )
    return value


def create_app(database_path: str = None) -> FastAPI:
    path = database_path or os.getenv("MEMORY_DB_PATH", "data/chrono_hybrid_mem.db")
    structured_query_plan = structured_query_plan_from_environment()
    set_aware_rerank = set_aware_rerank_from_environment()
    evidence_need_retrieval = evidence_need_retrieval_from_environment()
    adjacent_turn_expansion = adjacent_turn_expansion_from_environment()
    evidence_graph = evidence_graph_from_environment()
    evidence_anchors = evidence_anchors_from_environment()

    if evidence_graph and evidence_anchors:
        raise RuntimeError(
            "MEMORY_EVIDENCE_ANCHORS cannot be combined with MEMORY_EVIDENCE_GRAPH"
        )
    if set_aware_rerank and not structured_query_plan:
        raise RuntimeError(
            "MEMORY_SET_AWARE_RERANK requires MEMORY_STRUCTURED_QUERY_PLAN=true"
        )
    if evidence_need_retrieval and not structured_query_plan:
        raise RuntimeError(
            "MEMORY_EVIDENCE_NEED_RETRIEVAL requires "
            "MEMORY_STRUCTURED_QUERY_PLAN=true"
        )
    if evidence_anchors and adjacent_turn_expansion:
        raise RuntimeError(
            "MEMORY_EVIDENCE_ANCHORS cannot be combined with "
            "MEMORY_ADJACENT_TURN_EXPANSION"
        )
    if evidence_graph and not structured_query_plan:
        raise RuntimeError(
            "MEMORY_EVIDENCE_GRAPH requires MEMORY_STRUCTURED_QUERY_PLAN=true"
        )
    if evidence_anchors and not structured_query_plan:
        raise RuntimeError(
            "MEMORY_EVIDENCE_ANCHORS requires MEMORY_STRUCTURED_QUERY_PLAN=true"
        )

    adjacent_options = {}
    if adjacent_turn_expansion:
        adjacent_options = {
            "adjacent_turn_expansion": True,
            "adjacent_seed_limit": adjacent_seed_limit_from_environment(),
            "adjacent_candidate_limit": adjacent_candidate_limit_from_environment(),
        }
    graph_options = {}
    if evidence_graph:
        graph_options = {
            "graph_max_hops": graph_max_hops_from_environment(),
            "graph_temporal": graph_temporal_from_environment(),
            "graph_rrf_weight": graph_rrf_weight_from_environment(),
            "graph_max_candidates": graph_max_candidates_from_environment(),
            "graph_rerank_quota": graph_rerank_quota_from_environment(),
        }
    anchor_options = {}
    if evidence_anchors:
        anchor_options = {
            "anchor_seed_limit": anchor_seed_limit_from_environment(),
            "anchor_max_candidates": anchor_max_candidates_from_environment(),
            "anchor_rrf_weight": anchor_rrf_weight_from_environment(),
            "anchor_rerank_quota": anchor_rerank_quota_from_environment(),
        }
    dense_fusion_alpha = dense_fusion_alpha_from_environment()
    if evidence_graph and dense_fusion_alpha is not None:
        raise RuntimeError(
            "MEMORY_EVIDENCE_GRAPH cannot be combined with MEMORY_DENSE_FUSION_ALPHA"
        )
    if evidence_anchors and dense_fusion_alpha is not None:
        raise RuntimeError(
            "MEMORY_EVIDENCE_ANCHORS cannot be combined with "
            "MEMORY_DENSE_FUSION_ALPHA"
        )
    if evidence_need_retrieval and dense_fusion_alpha is not None:
        raise RuntimeError(
            "MEMORY_EVIDENCE_NEED_RETRIEVAL cannot be combined with "
            "MEMORY_DENSE_FUSION_ALPHA"
        )
    yes_no_reranker = local_yes_no_reranker_from_environment()
    has_fastembed_reranker = bool(
        os.getenv("MEMORY_LOCAL_RERANK_MODEL", "").strip()
        or os.getenv("MEMORY_LOCAL_CROSS_ENCODER_MODEL", "").strip()
    )
    if yes_no_reranker and has_fastembed_reranker:
        raise RuntimeError(
            "Use either MEMORY_LOCAL_RERANK_MODEL/MEMORY_LOCAL_CROSS_ENCODER_MODEL "
            "or MEMORY_LOCAL_YES_NO_RERANK_BASE_URL, not both"
        )
    local_reranker = local_reranker_from_environment()
    store = MemoryStore(
        path,
        model=model_from_environment(),
        temporal_bonus=temporal_bonus_from_environment(),
        semantic_retriever=local_semantic_retriever_from_environment(),
        dense_rrf_weight=dense_rrf_weight_from_environment(),
        dense_fusion_alpha=dense_fusion_alpha,
        dense_context_weight=dense_context_weight_from_environment(),
        dense_time_weight=dense_time_weight_from_environment(),
        dense_speaker_mask_max=dense_speaker_mask_max_from_environment(),
        dense_speaker_conflict_margin=dense_speaker_conflict_margin_from_environment(),
        dense_speaker_conflict_gate_only=(
            dense_speaker_conflict_gate_only_from_environment()
        ),
        dense_sentence_weight=dense_sentence_weight_from_environment(),
        dense_image_carry_weight=dense_image_carry_weight_from_environment(),
        dense_speaker_coref_weight=dense_speaker_coref_weight_from_environment(),
        local_reranker=yes_no_reranker or local_reranker,
        rerank_top_n=rerank_top_n_from_environment(),
        rerank_image_followups=rerank_image_followups_from_environment(),
        session_fusion_weight=session_fusion_weight_from_environment(),
        session_top_n=session_top_n_from_environment(),
        rerank_fusion_weight=rerank_fusion_weight_from_environment(),
        rerank_near_tie_epsilon=rerank_near_tie_epsilon_from_environment(),
        local_instruction_reranker=local_instruction_reranker_from_environment(),
        instruction_speaker_conflict_only=(
            instruction_speaker_conflict_only_from_environment()
        ),
        local_query_expander=local_query_expander_from_environment(),
        instruction_rerank_top_n=instruction_rerank_top_n_from_environment(),
        instruction_refine_top_n=instruction_refine_top_n_from_environment(),
        structured_query_plan=structured_query_plan,
        set_aware_rerank=set_aware_rerank,
        evidence_need_retrieval=evidence_need_retrieval,
        evidence_graph=evidence_graph,
        evidence_anchors=evidence_anchors,
        **graph_options,
        **anchor_options,
        **adjacent_options,
    )
    store.initialize()
    app = FastAPI(title="ChronoHybridMem", version="0.4.0-local")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/add", response_model=AddResponse)
    def add(request: AddRequest) -> AddResponse:
        store.add(request)
        return AddResponse(
            request_id=request.request_id, user_id=request.user_id, session_id=request.session_id
        )

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        return SearchResponse(data=store.search(
            user_id=request.user_id, query=request.query, options=request.options, top_k=request.top_k
        ))

    return app


app = create_app()
