"""
Pipeline Orchestrator
---------------------
Ties all components together into a single coherent RAG pipeline.
Handles initialization, query processing, and coordinates the
adaptive + feedback systems.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Dict

from config import CONFIG
from src.ingestion import DocumentIngestion
from src.indexer import VectorIndexer
from src.retriever import HybridRetriever
from src.reranker import DocumentReranker
from src.generator import TextGenerator, GenerationResult
from src.adaptive import AdaptiveDecisionLayer, AdaptiveDecision
from src.feedback import FeedbackLoop, QueryRecord
from src.metrics import PerformanceMetrics


@dataclass
class PipelineResponse:
    """Complete response from the RAG pipeline."""
    query: str
    answer: str
    confidence: float
    total_latency: float
    retrieval_time: float
    rerank_time: float
    generation_time: float
    top_k_used: int
    strategy_used: str
    used_reranking: bool
    complexity_score: float
    adaptive_reason: str
    num_docs_retrieved: int
    retrieved_texts: List[str]


class AdaptiveRAGPipeline:
    """
    Full Adaptive RAG Pipeline orchestrator.

    Flow:
    1. Analyze query complexity
    2. Get feedback-driven overrides
    3. Make adaptive decisions (K, strategy, reranking)
    4. Retrieve documents (vector / BM25 / hybrid)
    5. Optionally re-rank
    6. Generate answer
    7. Record feedback
    """

    def __init__(self):
        self.ingestion = DocumentIngestion()
        self.indexer = VectorIndexer()
        self.retriever: Optional[HybridRetriever] = None
        self.reranker = DocumentReranker()
        self.generator = TextGenerator()
        self.adaptive = AdaptiveDecisionLayer()
        self.feedback = FeedbackLoop()
        self.metrics = PerformanceMetrics()
        self._initialized = False
        self._documents = []

    def initialize(self, force_reload: bool = False) -> None:
        """
        Initialize the full pipeline:
        1. Ingest documents
        2. Build FAISS index
        3. Build BM25 index
        """
        print("\n=== Initializing Adaptive RAG Pipeline ===\n")

        # Step 1: Ingest documents
        print("Step 1/3: Document ingestion...")
        self._documents = self.ingestion.load_and_process(force_reload=force_reload)
        texts = self.ingestion.get_all_texts()
        doc_ids = [doc.doc_id for doc in self._documents]

        stats = self.ingestion.get_stats()
        print(f"  [OK] {stats['total_chunks']} document chunks from {stats['unique_titles']} topics")

        # Step 2: Build FAISS index
        print("Step 2/3: Building vector index (FAISS)...")
        self.indexer.build_index(texts, doc_ids, force_rebuild=force_reload)
        idx_stats = self.indexer.get_stats()
        print(f"  [OK] {idx_stats['total_vectors']} vectors indexed ({idx_stats.get('index_file_size_mb', 0):.1f} MB)")

        # Step 3: Build BM25 index
        print("Step 3/3: Building keyword index (BM25)...")
        self.retriever = HybridRetriever(self.indexer)
        self.retriever.build_bm25_index(texts)
        print("  [OK] BM25 index ready")

        # Load feedback state if exists
        self.feedback.load_state()

        self._initialized = True
        print("\n[OK] Pipeline ready!\n")

    def query(self, query_text: str, verbose: bool = False) -> PipelineResponse:
        """
        Process a query through the full adaptive pipeline.

        Args:
            query_text: The user's question
            verbose: If True, print step-by-step info

        Returns:
            PipelineResponse with answer and all metadata
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.time()

        # ── Step 1: Adaptive Decision ────────────────────────────────────────
        overrides = self.feedback.get_overrides()
        decision = self.adaptive.decide(query_text, feedback_overrides=overrides)

        if verbose:
            print(f"\nAdaptive Decision:")
            print(f"   K={decision.top_k}, Strategy={decision.strategy}, Rerank={decision.use_reranking}")
            print(f"   Reason: {decision.reason}")

        # ── Step 2: Retrieve ─────────────────────────────────────────────────
        retrieval_output = self.retriever.retrieve(
            query=query_text,
            top_k=decision.top_k,
            strategy=decision.strategy,
        )
        retrieval_time = retrieval_output.retrieval_time

        if verbose:
            print(f"\nRetrieved {len(retrieval_output.results)} documents in {retrieval_time:.4f}s")

        # ── Step 3: Re-rank (if enabled) ─────────────────────────────────────
        rerank_time = 0.0
        if decision.use_reranking and retrieval_output.results:
            reranked, rerank_time = self.reranker.rerank(
                query=query_text,
                results=retrieval_output.results,
                top_n=CONFIG.retrieval.rerank_top_n,
            )
            final_results = reranked
            if verbose:
                print(f"Re-ranked to {len(final_results)} results in {rerank_time:.4f}s")
        else:
            final_results = retrieval_output.results[:CONFIG.retrieval.rerank_top_n]

        # ── Step 4: Generate ─────────────────────────────────────────────────
        context_texts = [r.text for r in final_results]
        gen_result = self.generator.generate(query_text, context_texts)
        generation_time = gen_result.generation_time

        if verbose:
            print(f"Generated answer in {generation_time:.4f}s (confidence={gen_result.confidence:.4f})")

        total_latency = time.time() - start_time

        # ── Step 5: Record Feedback ──────────────────────────────────────────
        avg_score = (
            sum(r.final_score for r in final_results) / len(final_results)
            if final_results else 0.0
        )

        record = QueryRecord(
            timestamp=time.time(),
            query=query_text,
            complexity_score=decision.complexity_score,
            top_k_used=decision.top_k,
            strategy_used=decision.strategy,
            use_reranking=decision.use_reranking,
            total_latency=total_latency,
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            generation_time=generation_time,
            answer_length=len(gen_result.answer),
            confidence=gen_result.confidence,
            avg_retrieval_score=avg_score,
            num_results_returned=len(final_results),
            answer=gen_result.answer,
        )
        self.feedback.record_query(record)

        return PipelineResponse(
            query=query_text,
            answer=gen_result.answer,
            confidence=gen_result.confidence,
            total_latency=total_latency,
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            generation_time=generation_time,
            top_k_used=decision.top_k,
            strategy_used=decision.strategy,
            used_reranking=decision.use_reranking,
            complexity_score=decision.complexity_score,
            adaptive_reason=decision.reason,
            num_docs_retrieved=len(final_results),
            retrieved_texts=context_texts,
        )

    def get_stats(self) -> Dict:
        """Get comprehensive pipeline statistics."""
        self.metrics.set_records(self.feedback.get_all_records())
        return {
            "ingestion": self.ingestion.get_stats(),
            "indexer": self.indexer.get_stats(),
            "generator": self.generator.get_stats(),
            "feedback": self.feedback.get_rolling_stats(),
        }

    def print_stats(self) -> None:
        """Print comprehensive performance report."""
        self.metrics.set_records(self.feedback.get_all_records())
        self.metrics.print_report()

    def save_report(self) -> None:
        """Save performance report to disk."""
        self.metrics.set_records(self.feedback.get_all_records())
        self.metrics.save_report()
        self.feedback.save_state()

    def reset_feedback(self) -> None:
        """Reset the feedback loop state."""
        self.feedback.reset()
