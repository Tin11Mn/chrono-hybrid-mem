import os

from fastapi import FastAPI

from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .model import model_from_environment
from .storage import MemoryStore


def create_app(database_path: str = None) -> FastAPI:
    path = database_path or os.getenv("MEMORY_DB_PATH", "data/chrono_hybrid_mem.db")
    store = MemoryStore(path, model=model_from_environment())
    store.initialize()
    app = FastAPI(title="ChronoHybridMem", version="0.2.0")

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
