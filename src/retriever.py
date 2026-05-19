"""
Part 2: Hybrid Retriever
-------------------------
Combines vector search (FAISS) and keyword search (BM25) with
Reciprocal Rank Fusion (RRF) for hybrid retrieval.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from rank_bm25 import BM25Okapi

from config import CONFIG
from src.indexer import VectorIndexer


@dataclass
class RetrievalResult:
    """A single retrieved document with scores."""
    doc_index: int
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    source: str = ""  # "vector", "bm25", or "hybrid"


@dataclass
class RetrievalOutput:
    """Complete retrieval output with metadata."""
    results: List[RetrievalResult]
    retrieval_time: float
    strategy_used: str
    top_k_used: int
    vector_time: float = 0.0
    bm25_time: float = 0.0
    fusion_time: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval combining FAISS vector search and BM25 keyword search.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(self, vector_indexer: VectorIndexer, config=None):
        self.vector_indexer = vector_indexer
        self.config = config or CONFIG.retrieval
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        self._corpus_texts: List[str] = []

    def build_bm25_index(self, texts: List[str]) -> None:
        """Build the BM25 keyword index from document texts."""
        print("[Retriever] Building BM25 index...")
        self._corpus_texts = texts
        self._tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"[Retriever] BM25 index built with {len(texts)} documents.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, remove stopwords."""
        # Basic stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "it", "its", "this", "that", "these", "those", "and", "but",
            "or", "nor", "not", "so", "very", "just", "than", "too", "also",
        }
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def retrieve(self, query: str, top_k: int = None,
                 strategy: str = None) -> RetrievalOutput:
        """
        Retrieve documents using the specified strategy.

        Args:
            query: Search query
            top_k: Number of results (overrides config default)
            strategy: "vector", "bm25", or "hybrid" (overrides config default)

        Returns:
            RetrievalOutput with ranked results and timing
        """
        k = top_k or self.config.default_top_k
        strat = strategy or self.config.retrieval_strategy

        start_time = time.time()
        vector_time = 0.0
        bm25_time = 0.0
        fusion_time = 0.0

        if strat == "vector":
            results, vector_time = self._vector_search(query, k)
        elif strat == "bm25":
            results, bm25_time = self._bm25_search(query, k)
        elif strat == "hybrid":
            results, vector_time, bm25_time, fusion_time = self._hybrid_search(query, k)
        else:
            raise ValueError(f"Unknown strategy: {strat}")

        total_time = time.time() - start_time

        return RetrievalOutput(
            results=results,
            retrieval_time=total_time,
            strategy_used=strat,
            top_k_used=k,
            vector_time=vector_time,
            bm25_time=bm25_time,
            fusion_time=fusion_time,
        )

    def _vector_search(self, query: str, top_k: int) -> Tuple[List[RetrievalResult], float]:
        """Pure vector (FAISS) search."""
        start = time.time()
        raw_results = self.vector_indexer.search(query, top_k)
        elapsed = time.time() - start

        results = []
        for idx, score, text in raw_results:
            results.append(RetrievalResult(
                doc_index=idx,
                text=text,
                vector_score=score,
                final_score=score,
                source="vector",
            ))
        return results, elapsed

    def _bm25_search(self, query: str, top_k: int) -> Tuple[List[RetrievalResult], float]:
        """Pure BM25 keyword search."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_bm25_index() first.")

        start = time.time()
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        elapsed = time.time() - start

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(RetrievalResult(
                doc_index=int(idx),
                text=self._corpus_texts[idx],
                bm25_score=float(scores[idx]),
                final_score=float(scores[idx]),
                source="bm25",
            ))
        return results, elapsed

    def _hybrid_search(self, query: str, top_k: int) -> Tuple[List[RetrievalResult], float, float, float]:
        """
        Hybrid search combining vector and BM25 via Reciprocal Rank Fusion (RRF).
        """
        # Fetch more candidates for fusion (2x top_k from each source)
        fetch_k = top_k * 2

        # Vector search
        vector_results, vector_time = self._vector_search(query, fetch_k)

        # BM25 search
        bm25_results, bm25_time = self._bm25_search(query, fetch_k)

        # Reciprocal Rank Fusion
        fusion_start = time.time()
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)
        fusion_time = time.time() - fusion_start

        return fused, vector_time, bm25_time, fusion_time

    def _reciprocal_rank_fusion(self, vector_results: List[RetrievalResult],
                                 bm25_results: List[RetrievalResult],
                                 top_k: int, rrf_k: int = 60) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion: combines rankings from multiple sources.
        Score = sum over sources of: weight / (rrf_k + rank)
        """
        doc_scores: Dict[int, RetrievalResult] = {}
        vector_weight = self.config.vector_weight
        bm25_weight = self.config.bm25_weight

        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_idx = result.doc_index
            rrf_score = vector_weight / (rrf_k + rank + 1)

            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = RetrievalResult(
                    doc_index=doc_idx,
                    text=result.text,
                    vector_score=result.vector_score,
                    source="hybrid",
                )
            doc_scores[doc_idx].vector_score = result.vector_score
            doc_scores[doc_idx].fused_score += rrf_score

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_idx = result.doc_index
            rrf_score = bm25_weight / (rrf_k + rank + 1)

            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = RetrievalResult(
                    doc_index=doc_idx,
                    text=result.text,
                    source="hybrid",
                )
            doc_scores[doc_idx].bm25_score = result.bm25_score
            doc_scores[doc_idx].fused_score += rrf_score

        # Sort by fused score and take top_k
        ranked = sorted(doc_scores.values(), key=lambda r: r.fused_score, reverse=True)[:top_k]

        # Set final scores
        for r in ranked:
            r.final_score = r.fused_score

        return ranked


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.indexer import VectorIndexer

    docs = [
        "Python is a high-level programming language created by Guido van Rossum.",
        "FAISS is Facebook's library for fast approximate nearest neighbor search.",
        "The Eiffel Tower stands 330 meters tall in Paris, France.",
        "Machine learning algorithms learn patterns from training data.",
        "DNA stores genetic information using four nucleotide bases.",
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]

    indexer = VectorIndexer()
    indexer.build_index(docs, ids, force_rebuild=True)

    retriever = HybridRetriever(indexer)
    retriever.build_bm25_index(docs)

    output = retriever.retrieve("What is Python programming?", top_k=3, strategy="hybrid")
    print(f"\nStrategy: {output.strategy_used} | Time: {output.retrieval_time:.4f}s")
    for r in output.results:
        print(f"  [{r.final_score:.4f}] (vec={r.vector_score:.3f}, bm25={r.bm25_score:.3f}) {r.text[:80]}")
