from __future__ import annotations

from typing import Dict, List

from src.retrievers.lexical_retriever import LexicalRetriever
from src.retrievers.semantic_retriever import LocalSemanticRetriever, SemanticFaissRetriever


class HybridRetriever:
    """Hybrid retriever that fuses lexical and semantic scores."""

    def __init__(
        self,
        semantic_backend: str = "local",
        keyword_weight: float = 0.55,
        semantic_weight: float = 0.45,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.semantic_backend = semantic_backend
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.lexical = LexicalRetriever()
        if semantic_backend == "faiss":
            self.semantic = SemanticFaissRetriever(embedding_model_name=embedding_model_name)
        elif semantic_backend == "local":
            self.semantic = LocalSemanticRetriever()
        else:
            raise ValueError(f"Unsupported semantic_backend: {semantic_backend}")
        self.chunks = None

    @staticmethod
    def _chunk_key(chunk: Dict[str, object]) -> tuple:
        return (str(chunk.get("source", "")), str(chunk.get("text", "")))

    @staticmethod
    def _normalize(scores: Dict[tuple, float]) -> Dict[tuple, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score:
            normalized_value = 1.0 if max_score > 0 else 0.0
            return {key: normalized_value for key in scores}
        return {
            key: (value - min_score) / (max_score - min_score)
            for key, value in scores.items()
        }

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        self.chunks = chunks
        self.lexical.build_index(chunks)
        self.semantic.build_index(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        if self.chunks is None:
            raise ValueError("Index has not been built yet.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, len(self.chunks))
        search_k = len(self.chunks)
        lexical_results = self.lexical.search(query, top_k=search_k)
        semantic_results = self.semantic.search(query, top_k=search_k)

        lexical_scores = {
            self._chunk_key(result): float(result.get("similarity_score", 0.0))
            for result in lexical_results
        }
        semantic_scores = {
            self._chunk_key(result): float(result.get("similarity_score", 0.0))
            for result in semantic_results
        }
        normalized_lexical = self._normalize(lexical_scores)
        normalized_semantic = self._normalize(semantic_scores)

        chunks_by_key = {self._chunk_key(chunk): chunk for chunk in self.chunks}
        fused_scores = {}
        for key in chunks_by_key:
            fused_scores[key] = (
                self.keyword_weight * normalized_lexical.get(key, 0.0)
                + self.semantic_weight * normalized_semantic.get(key, 0.0)
            )

        ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        method = (
            "hybrid_lexical_faiss_score_fusion"
            if self.semantic_backend == "faiss"
            else "hybrid_lexical_local_score_fusion"
        )
        results = []
        for key in ranked_keys:
            chunk = chunks_by_key[key].copy()
            chunk["keyword_score"] = float(normalized_lexical.get(key, 0.0))
            chunk["semantic_score"] = float(normalized_semantic.get(key, 0.0))
            chunk["similarity_score"] = float(fused_scores[key])
            chunk["retrieval_mode"] = "hybrid"
            chunk["retrieval_method"] = method
            results.append(chunk)
        return results
