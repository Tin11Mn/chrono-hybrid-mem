"""Optional local embedding retrieval for method development.

This module is imported safely without ML dependencies. FastEmbed is loaded
only when local semantic retrieval is explicitly enabled.
"""

import os
from typing import Dict, List, Optional


DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


class LocalSemanticRetriever:
    """Ranks raw memory records with a locally hosted embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            import numpy as np
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-local.txt to enable local semantic retrieval"
            ) from error
        if device not in {None, "auto", "cpu", "cuda"}:
            raise RuntimeError("Local embedding device must be auto, cpu, or cuda")
        self.model_name = model_name
        self.batch_size = batch_size
        self._np = np
        model_options = {"model_name": model_name, "cache_dir": cache_dir}
        if device == "cpu":
            model_options["cuda"] = False
        elif device == "cuda":
            model_options["cuda"] = True
        self.model = TextEmbedding(**model_options)
        self._document_cache = {}
        self._query_cache = {}

    def _normalized_matrix(self, vectors: object) -> object:
        matrix = self._np.asarray(list(vectors), dtype=self._np.float32)
        norms = self._np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / self._np.maximum(norms, 1e-12)

    def rank(
        self,
        query: str,
        options: List[str],
        candidates: List[Dict[str, str]],
        limit: int,
    ) -> List[str]:
        scores = self.score(query, options, candidates)
        return sorted(scores, key=scores.get, reverse=True)[:limit]

    def score(
        self,
        query: str,
        options: List[str],
        candidates: List[Dict[str, str]],
    ) -> Dict[str, float]:
        if not candidates:
            return {}
        document_key = tuple((item["id"], item["content"]) for item in candidates)
        document_embeddings = self._document_cache.get(document_key)
        if document_embeddings is None:
            passages = [item["content"] for item in candidates]
            document_embeddings = self._normalized_matrix(
                self.model.passage_embed(passages, batch_size=self.batch_size)
            )
            self._document_cache[document_key] = document_embeddings
        option_text = " ".join(option for option in options if option.strip())
        query_text = "{} {}".format(query, option_text).strip()
        query_embedding = self._query_cache.get(query_text)
        if query_embedding is None:
            query_embedding = self._normalized_matrix(
                self.model.query_embed([query_text], batch_size=1)
            )[0]
            self._query_cache[query_text] = query_embedding
        similarities = document_embeddings @ query_embedding
        return {
            candidate["id"]: float(similarities[index])
            for index, candidate in enumerate(candidates)
        }


def local_semantic_retriever_from_environment() -> Optional[LocalSemanticRetriever]:
    late_model = os.getenv("MEMORY_LOCAL_LATE_INTERACTION_MODEL", "").strip()
    model_name = os.getenv("MEMORY_LOCAL_EMBEDDING_MODEL", "").strip()
    if late_model and model_name:
        raise RuntimeError(
            "Use either MEMORY_LOCAL_EMBEDDING_MODEL or "
            "MEMORY_LOCAL_LATE_INTERACTION_MODEL, not both"
        )
    if late_model:
        device = os.getenv("MEMORY_LOCAL_EMBEDDING_DEVICE") or None
        batch_size = int(os.getenv("MEMORY_LOCAL_EMBEDDING_BATCH_SIZE", "32"))
        if batch_size < 1:
            raise RuntimeError("MEMORY_LOCAL_EMBEDDING_BATCH_SIZE must be positive")
        return LocalLateInteractionReranker(
            late_model,
            device=device,
            batch_size=batch_size,
            cache_dir=os.getenv("MEMORY_LOCAL_MODEL_CACHE") or None,
        )
    if not model_name:
        return None
    device = os.getenv("MEMORY_LOCAL_EMBEDDING_DEVICE") or None
    batch_size = int(os.getenv("MEMORY_LOCAL_EMBEDDING_BATCH_SIZE", "64"))
    cache_dir = os.getenv("MEMORY_LOCAL_MODEL_CACHE") or None
    if batch_size < 1:
        raise RuntimeError("MEMORY_LOCAL_EMBEDDING_BATCH_SIZE must be positive")
    return LocalSemanticRetriever(
        model_name, device=device, batch_size=batch_size, cache_dir=cache_dir
    )


class LocalLateInteractionReranker:
    """Token-level ColBERT-style MaxSim reranking over a bounded candidate set."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            import numpy as np
            from fastembed import LateInteractionTextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-local.txt to enable local late-interaction reranking"
            ) from error
        if device not in {None, "auto", "cpu", "cuda"}:
            raise RuntimeError("Local reranker device must be auto, cpu, or cuda")
        model_options = {"model_name": model_name, "cache_dir": cache_dir}
        if device == "cpu":
            model_options["cuda"] = False
        elif device == "cuda":
            model_options["cuda"] = True
        self.model_name = model_name
        self.batch_size = batch_size
        self._np = np
        self.model = LateInteractionTextEmbedding(**model_options)
        self._passage_cache = {}
        self._query_cache = {}

    def _normalize_tokens(self, matrix: object) -> object:
        values = self._np.asarray(matrix, dtype=self._np.float32)
        norms = self._np.linalg.norm(values, axis=1, keepdims=True)
        return values / self._np.maximum(norms, 1e-12)

    def rank(
        self,
        query: str,
        options: List[str],
        candidates: List[Dict[str, str]],
        limit: Optional[int] = None,
    ) -> List[str]:
        scores = self.score(query, options, candidates)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked if limit is None else ranked[:limit]

    def score(
        self, query: str, options: List[str], candidates: List[Dict[str, str]]
    ) -> Dict[str, float]:
        if not candidates:
            return {}
        missing = [
            item for item in candidates if item["content"] not in self._passage_cache
        ]
        if missing:
            embeddings = self.model.passage_embed(
                [item["content"] for item in missing], batch_size=self.batch_size
            )
            for item, embedding in zip(missing, embeddings):
                self._passage_cache[item["content"]] = self._normalize_tokens(embedding)
        option_text = " ".join(option for option in options if option.strip())
        query_text = "{} {}".format(query, option_text).strip()
        query_embedding = self._query_cache.get(query_text)
        if query_embedding is None:
            query_embedding = self._normalize_tokens(
                next(iter(self.model.query_embed([query_text], batch_size=1)))
            )
            self._query_cache[query_text] = query_embedding
        scores = {}
        for candidate in candidates:
            passage = self._passage_cache[candidate["content"]]
            similarities = query_embedding @ passage.T
            scores[candidate["id"]] = float(similarities.max(axis=1).sum())
        return scores


class LocalCrossEncoderReranker:
    """Joint query-passage scoring over a bounded candidate pool."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-local.txt to enable local cross-encoder reranking"
            ) from error
        if device not in {None, "auto", "cpu", "cuda"}:
            raise RuntimeError("Local reranker device must be auto, cpu, or cuda")
        model_options = {"model_name": model_name, "cache_dir": cache_dir}
        if device == "cpu":
            model_options["cuda"] = False
        elif device == "cuda":
            model_options["cuda"] = True
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextCrossEncoder(**model_options)
        self._score_cache = {}

    def score(
        self, query: str, options: List[str], candidates: List[Dict[str, str]]
    ) -> Dict[str, float]:
        option_text = " ".join(option for option in options if option.strip())
        query_text = "{} {}".format(query, option_text).strip()
        cache_key = (
            query_text,
            tuple((item["id"], item["content"]) for item in candidates),
        )
        cached = self._score_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        values = list(self.model.rerank(
            query_text,
            [item["content"] for item in candidates],
            batch_size=self.batch_size,
        ))
        scores = {
            candidate["id"]: float(values[index])
            for index, candidate in enumerate(candidates)
        }
        self._score_cache[cache_key] = scores
        return dict(scores)

    def rank(
        self, query: str, options: List[str], candidates: List[Dict[str, str]]
    ) -> List[str]:
        scores = self.score(query, options, candidates)
        return sorted(scores, key=scores.get, reverse=True)


def local_reranker_from_environment() -> Optional[object]:
    model_name = os.getenv("MEMORY_LOCAL_RERANK_MODEL", "").strip()
    cross_encoder_name = os.getenv("MEMORY_LOCAL_CROSS_ENCODER_MODEL", "").strip()
    if model_name and cross_encoder_name:
        raise RuntimeError(
            "Use either MEMORY_LOCAL_RERANK_MODEL or "
            "MEMORY_LOCAL_CROSS_ENCODER_MODEL, not both"
        )
    selected_model = model_name or cross_encoder_name
    if not selected_model:
        return None
    device = os.getenv("MEMORY_LOCAL_EMBEDDING_DEVICE") or None
    batch_size = int(os.getenv("MEMORY_LOCAL_RERANK_BATCH_SIZE", "32"))
    cache_dir = os.getenv("MEMORY_LOCAL_MODEL_CACHE") or None
    if batch_size < 1:
        raise RuntimeError("MEMORY_LOCAL_RERANK_BATCH_SIZE must be positive")
    reranker_class = (
        LocalCrossEncoderReranker if cross_encoder_name else LocalLateInteractionReranker
    )
    return reranker_class(
        selected_model, device=device, batch_size=batch_size, cache_dir=cache_dir
    )
