"""
Central configuration for the Adaptive RAG Inference System.
All tunable parameters are defined here as dataclass fields.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os

# ── Project Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
REPORT_DIR = PROJECT_ROOT / "report"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model selection and parameters."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    generator_model: str = "google/flan-t5-base"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_new_tokens: int = 128
    generator_device: str = "cpu"  # "cpu" or "cuda"


@dataclass
class IngestionConfig:
    """Document ingestion parameters."""
    dataset_name: str = "rajpurkar/squad"
    dataset_split: str = "train"
    chunk_size: int = 512        # characters per chunk
    chunk_overlap: int = 50      # overlap between chunks
    min_chunk_length: int = 30   # discard very short chunks


@dataclass
class RetrievalConfig:
    """Retrieval parameters (defaults — adaptive layer overrides these)."""
    default_top_k: int = 5
    min_top_k: int = 2
    max_top_k: int = 15
    bm25_weight: float = 0.3           # weight for BM25 in hybrid fusion
    vector_weight: float = 0.7         # weight for vector search in hybrid fusion
    rerank_top_n: int = 5              # how many to keep after re-ranking
    use_reranking: bool = True
    retrieval_strategy: str = "hybrid"  # "vector", "bm25", or "hybrid"


@dataclass
class AdaptiveConfig:
    """Adaptive decision layer thresholds."""
    # Query complexity thresholds
    short_query_max_words: int = 4
    medium_query_max_words: int = 10
    complex_indicators: list = field(default_factory=lambda: [
        "how", "why", "explain", "compare", "difference",
        "relationship", "describe", "analyze", "what are the",
        "in what way", "to what extent"
    ])

    # Dynamic K mapping
    simple_k: int = 3
    medium_k: int = 5
    complex_k: int = 10

    # Latency thresholds (seconds)
    latency_threshold_high: float = 5.0
    latency_threshold_critical: float = 10.0

    # Feedback-driven adjustments
    confidence_low_threshold: float = 0.3
    confidence_high_threshold: float = 0.7
    quality_window_size: int = 20  # rolling window for feedback


@dataclass
class FeedbackConfig:
    """Feedback loop parameters."""
    rolling_window_size: int = 50
    k_adjustment_step: int = 1
    min_queries_before_adapt: int = 5
    latency_spike_multiplier: float = 2.0  # spike = 2x the rolling avg


@dataclass
class AppConfig:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)


# ── Global config instance ─────────────────────────────────────────────────────
CONFIG = AppConfig()
