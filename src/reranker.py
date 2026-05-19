"""
Part 2: Cross-Encoder Re-Ranker
--------------------------------
Re-ranks retrieved documents using a cross-encoder model for more
accurate relevance scoring. Cross-encoders process (query, document)
pairs jointly, giving better accuracy than bi-encoder similarity.
"""

import time
from typing import List
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from config import CONFIG
from src.retriever import RetrievalResult


class DocumentReranker:
    """Re-ranks retrieval results using a cross-encoder model."""

    def __init__(self, config=None):
        self.config = config or CONFIG.model
        self.cross_encoder = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._loaded:
            return

        print(f"[Reranker] Loading cross-encoder: {self.config.reranker_model}")
        self.cross_encoder = CrossEncoder(self.config.reranker_model)
        self._loaded = True
        print("[Reranker] Cross-encoder loaded.")

    def rerank(self, query: str, results: List[RetrievalResult],
               top_n: int = None) -> tuple:
        """
        Re-rank retrieval results using the cross-encoder.

        Args:
            query: The original query
            results: List of RetrievalResult from the retriever
            top_n: Number of results to keep after re-ranking

        Returns:
            (re-ranked results, rerank_time in seconds)
        """
        if not results:
            return results, 0.0

        self._load_model()

        top_n = top_n or CONFIG.retrieval.rerank_top_n

        start_time = time.time()

        # Create (query, document) pairs for cross-encoder
        pairs = [(query, r.text) for r in results]

        # Score all pairs
        scores = self.cross_encoder.predict(pairs)

        # Assign rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
            result.final_score = float(score)  # Override final score with rerank score

        # Sort by rerank score (descending) and take top_n
        reranked = sorted(results, key=lambda r: r.rerank_score, reverse=True)[:top_n]

        rerank_time = time.time() - start_time
        return reranked, rerank_time

    def get_stats(self) -> dict:
        return {
            "model": self.config.reranker_model,
            "loaded": self._loaded,
        }


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reranker = DocumentReranker()

    # Simulate retrieval results
    mock_results = [
        RetrievalResult(doc_index=0, text="Python is a programming language.", final_score=0.9),
        RetrievalResult(doc_index=1, text="The Eiffel Tower is in Paris.", final_score=0.85),
        RetrievalResult(doc_index=2, text="Guido van Rossum created Python in 1991.", final_score=0.7),
    ]

    reranked, t = reranker.rerank("Who created Python?", mock_results, top_n=2)
    print(f"Reranking took {t:.4f}s")
    for r in reranked:
        print(f"  [{r.rerank_score:.4f}] {r.text}")
