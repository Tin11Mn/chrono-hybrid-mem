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


def structured_query_plan_from_environment() -> bool:
    return os.getenv("MEMORY_STRUCTURED_QUERY_PLAN", "false").lower() == "true"


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
        dense_fusion_alpha=dense_fusion_alpha_from_environment(),
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
        structured_query_plan=structured_query_plan_from_environment(),
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
