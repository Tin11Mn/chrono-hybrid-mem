import os

from fastapi import FastAPI

from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .model import model_from_environment
from .local_semantic import (
    local_reranker_from_environment,
    local_semantic_retriever_from_environment,
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


def rerank_top_n_from_environment() -> int:
    value = int(os.getenv("MEMORY_LOCAL_RERANK_TOP_N", "5"))
    if value < 1 or value > 100:
        raise RuntimeError("MEMORY_LOCAL_RERANK_TOP_N must be between 1 and 100")
    return value


def create_app(database_path: str = None) -> FastAPI:
    path = database_path or os.getenv("MEMORY_DB_PATH", "data/chrono_hybrid_mem.db")
    store = MemoryStore(
        path,
        model=model_from_environment(),
        temporal_bonus=temporal_bonus_from_environment(),
        semantic_retriever=local_semantic_retriever_from_environment(),
        dense_rrf_weight=dense_rrf_weight_from_environment(),
        dense_fusion_alpha=dense_fusion_alpha_from_environment(),
        local_reranker=local_reranker_from_environment(),
        rerank_top_n=rerank_top_n_from_environment(),
    )
    store.initialize()
    app = FastAPI(title="ChronoHybridMem", version="0.3.0")

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
