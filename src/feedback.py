"""
Part 4: Feedback Loop
---------------------
Tracks per-query performance metrics (latency, quality proxies) and
adaptively adjusts retrieval parameters. No ML training — pure rule-based
heuristic adaptation.

Tracked metrics per query:
  - Total latency, retrieval time, generation time
  - Answer confidence, answer length
  - Retrieval scores
  - Adaptive decisions made

Rolling window analysis drives adjustments:
  - High latency → reduce K, disable re-ranking
  - Low confidence → increase K
  - Consistently good quality → maintain current settings
"""

import time
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

from config import CONFIG, DATA_DIR


@dataclass
class QueryRecord:
    """Record of a single query execution."""
    timestamp: float
    query: str
    complexity_score: float
    top_k_used: int
    strategy_used: str
    use_reranking: bool

    # Timing
    total_latency: float = 0.0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    generation_time: float = 0.0

    # Quality proxies
    answer_length: int = 0
    confidence: float = 0.0
    avg_retrieval_score: float = 0.0
    num_results_returned: int = 0

    # Answer
    answer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackState:
    """Current state of the feedback-driven adaptive system."""
    k_adjustment: int = 0
    disable_reranking: bool = False
    force_strategy: Optional[str] = None
    latency_reduction: bool = False
    total_queries: int = 0
    adaptations_made: int = 0
    last_adaptation_reason: str = ""


class FeedbackLoop:
    """
    Tracks query performance and adaptively adjusts retrieval parameters.
    Uses a rolling window of recent queries to detect trends.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG.feedback
        self.state = FeedbackState()
        self._history: deque = deque(maxlen=self.config.rolling_window_size)
        self._all_records: List[QueryRecord] = []
        self._state_path = DATA_DIR / "feedback_state.json"

    def record_query(self, record: QueryRecord) -> None:
        """Record a completed query and update adaptive state."""
        self._history.append(record)
        self._all_records.append(record)
        self.state.total_queries += 1

        # Run adaptive logic after minimum queries
        if len(self._history) >= self.config.min_queries_before_adapt:
            self._update_adaptive_state()

    def get_overrides(self) -> Dict:
        """Get current feedback-driven overrides for the adaptive layer."""
        overrides = {}

        if self.state.k_adjustment != 0:
            overrides["k_adjustment"] = self.state.k_adjustment

        if self.state.disable_reranking:
            overrides["disable_reranking"] = True

        if self.state.force_strategy:
            overrides["force_strategy"] = self.state.force_strategy

        if self.state.latency_reduction:
            overrides["latency_reduction"] = True

        return overrides

    def _update_adaptive_state(self) -> None:
        """Analyze recent history and adjust parameters."""
        records = list(self._history)
        if not records:
            return

        # Compute rolling metrics
        latencies = [r.total_latency for r in records]
        confidences = [r.confidence for r in records]
        avg_latency = sum(latencies) / len(latencies)
        avg_confidence = sum(confidences) / len(confidences)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 5 else max(latencies)

        old_state = FeedbackState(
            k_adjustment=self.state.k_adjustment,
            disable_reranking=self.state.disable_reranking,
            latency_reduction=self.state.latency_reduction,
        )

        # ── Rule 1: Latency spike detection ──────────────────────────────────
        if p95_latency > CONFIG.adaptive.latency_threshold_critical:
            self.state.latency_reduction = True
            self.state.disable_reranking = True
            self.state.k_adjustment = -2
            self.state.last_adaptation_reason = (
                f"Critical latency: P95={p95_latency:.2f}s > {CONFIG.adaptive.latency_threshold_critical}s"
            )
        elif p95_latency > CONFIG.adaptive.latency_threshold_high:
            self.state.latency_reduction = False
            self.state.disable_reranking = True
            self.state.k_adjustment = -1
            self.state.last_adaptation_reason = (
                f"High latency: P95={p95_latency:.2f}s > {CONFIG.adaptive.latency_threshold_high}s"
            )
        else:
            # Latency is fine, remove latency-based constraints
            self.state.latency_reduction = False
            self.state.disable_reranking = False
            if self.state.k_adjustment < 0:
                self.state.k_adjustment = 0
                self.state.last_adaptation_reason = "Latency normalized, removing constraints"

        # ── Rule 2: Confidence-based K adjustment ────────────────────────────
        if avg_confidence < CONFIG.adaptive.confidence_low_threshold:
            # Low confidence → retrieve more documents
            if self.state.k_adjustment < 3:
                self.state.k_adjustment += 1
                self.state.last_adaptation_reason = (
                    f"Low confidence: avg={avg_confidence:.3f} < {CONFIG.adaptive.confidence_low_threshold}"
                )
        elif avg_confidence > CONFIG.adaptive.confidence_high_threshold:
            # High confidence → we can reduce K for speed
            if self.state.k_adjustment > -2:
                self.state.k_adjustment = max(self.state.k_adjustment - 1, -2)
                self.state.last_adaptation_reason = (
                    f"High confidence: avg={avg_confidence:.3f} > {CONFIG.adaptive.confidence_high_threshold}"
                )

        # ── Rule 3: Strategy adjustment based on retrieval quality ───────────
        avg_retrieval_scores = [r.avg_retrieval_score for r in records if r.avg_retrieval_score > 0]
        if avg_retrieval_scores:
            mean_ret_score = sum(avg_retrieval_scores) / len(avg_retrieval_scores)
            if mean_ret_score < 0.3:
                # Poor retrieval quality → switch to hybrid
                self.state.force_strategy = "hybrid"
                self.state.last_adaptation_reason = (
                    f"Poor retrieval quality: avg_score={mean_ret_score:.3f}, forcing hybrid"
                )
            else:
                self.state.force_strategy = None

        # Track if we made an adaptation
        if (old_state.k_adjustment != self.state.k_adjustment or
            old_state.disable_reranking != self.state.disable_reranking or
            old_state.latency_reduction != self.state.latency_reduction):
            self.state.adaptations_made += 1

    def get_rolling_stats(self) -> Dict:
        """Get statistics from the rolling window."""
        records = list(self._history)
        if not records:
            return {"status": "no data yet"}

        latencies = [r.total_latency for r in records]
        confidences = [r.confidence for r in records]
        retrieval_times = [r.retrieval_time for r in records]
        generation_times = [r.generation_time for r in records]
        answer_lengths = [r.answer_length for r in records]

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return {
            "window_size": len(records),
            "total_queries": self.state.total_queries,
            "latency": {
                "mean": sum(latencies) / n,
                "p50": sorted_lat[n // 2],
                "p95": sorted_lat[int(n * 0.95)] if n >= 5 else sorted_lat[-1],
                "p99": sorted_lat[int(n * 0.99)] if n >= 10 else sorted_lat[-1],
                "min": min(latencies),
                "max": max(latencies),
            },
            "retrieval_time_avg": sum(retrieval_times) / n,
            "generation_time_avg": sum(generation_times) / n,
            "confidence_avg": sum(confidences) / n,
            "answer_length_avg": sum(answer_lengths) / n,
            "current_adjustments": {
                "k_adjustment": self.state.k_adjustment,
                "disable_reranking": self.state.disable_reranking,
                "latency_reduction": self.state.latency_reduction,
                "force_strategy": self.state.force_strategy,
            },
            "adaptations_made": self.state.adaptations_made,
            "last_adaptation_reason": self.state.last_adaptation_reason,
        }

    def get_all_records(self) -> List[QueryRecord]:
        """Return all recorded query results."""
        return self._all_records

    def reset(self) -> None:
        """Reset feedback state and history."""
        self.state = FeedbackState()
        self._history.clear()
        self._all_records.clear()
        print("[Feedback] State reset.")

    def save_state(self) -> None:
        """Save feedback history to disk."""
        data = {
            "state": asdict(self.state),
            "records": [r.to_dict() for r in self._all_records[-200:]],  # Keep last 200
        }
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self) -> None:
        """Load feedback history from disk."""
        if self._state_path.exists():
            with open(self._state_path, "r") as f:
                data = json.load(f)
            self.state = FeedbackState(**data.get("state", {}))
            for r in data.get("records", []):
                record = QueryRecord(**r)
                self._history.append(record)
                self._all_records.append(record)


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fb = FeedbackLoop()

    # Simulate some queries
    for i in range(10):
        record = QueryRecord(
            timestamp=time.time(),
            query=f"Test query {i}",
            complexity_score=0.5,
            top_k_used=5,
            strategy_used="hybrid",
            use_reranking=True,
            total_latency=2.0 + i * 0.5,
            retrieval_time=0.5,
            generation_time=1.5 + i * 0.3,
            answer_length=50,
            confidence=0.6 - i * 0.03,
            avg_retrieval_score=0.7,
        )
        fb.record_query(record)

    stats = fb.get_rolling_stats()
    print("\n--- Feedback Stats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    overrides = fb.get_overrides()
    print(f"\nCurrent overrides: {overrides}")
