"""
Part 5: Performance Metrics
-----------------------------
Comprehensive performance measurement and reporting for the RAG pipeline.
Reports latency (P50/P95/P99), retrieval vs generation breakdown,
quality metrics, and impact of adaptive logic.
"""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from config import REPORT_DIR
from src.feedback import QueryRecord


class PerformanceMetrics:
    """Computes and reports performance metrics from query records."""

    def __init__(self):
        self._records: List[QueryRecord] = []

    def set_records(self, records: List[QueryRecord]) -> None:
        """Set the records to analyze."""
        self._records = records

    def compute_latency_metrics(self) -> Dict:
        """Compute latency percentiles and breakdown."""
        if not self._records:
            return {"status": "no data"}

        total_lats = sorted([r.total_latency for r in self._records])
        ret_times = [r.retrieval_time for r in self._records]
        gen_times = [r.generation_time for r in self._records]
        rerank_times = [r.rerank_time for r in self._records]
        n = len(total_lats)

        def percentile(data, p):
            sorted_d = sorted(data)
            idx = int(len(sorted_d) * p / 100)
            return sorted_d[min(idx, len(sorted_d) - 1)]

        return {
            "total_queries": n,
            "total_latency": {
                "mean": sum(total_lats) / n,
                "p50": percentile(total_lats, 50),
                "p95": percentile(total_lats, 95),
                "p99": percentile(total_lats, 99),
                "min": min(total_lats),
                "max": max(total_lats),
            },
            "retrieval_time": {
                "mean": sum(ret_times) / n,
                "p50": percentile(ret_times, 50),
                "p95": percentile(ret_times, 95),
            },
            "generation_time": {
                "mean": sum(gen_times) / n,
                "p50": percentile(gen_times, 50),
                "p95": percentile(gen_times, 95),
            },
            "reranking_time": {
                "mean": sum(rerank_times) / n,
                "p50": percentile(rerank_times, 50),
                "p95": percentile(rerank_times, 95),
            },
            "time_breakdown_pct": {
                "retrieval": (sum(ret_times) / sum(total_lats) * 100) if sum(total_lats) > 0 else 0,
                "generation": (sum(gen_times) / sum(total_lats) * 100) if sum(total_lats) > 0 else 0,
                "reranking": (sum(rerank_times) / sum(total_lats) * 100) if sum(total_lats) > 0 else 0,
            },
        }

    def compute_quality_metrics(self) -> Dict:
        """Compute quality-related metrics."""
        if not self._records:
            return {"status": "no data"}

        confidences = [r.confidence for r in self._records]
        answer_lengths = [r.answer_length for r in self._records]
        retrieval_scores = [r.avg_retrieval_score for r in self._records if r.avg_retrieval_score > 0]
        n = len(self._records)

        return {
            "confidence": {
                "mean": sum(confidences) / n,
                "min": min(confidences),
                "max": max(confidences),
            },
            "answer_length": {
                "mean": sum(answer_lengths) / n,
                "min": min(answer_lengths),
                "max": max(answer_lengths),
            },
            "retrieval_score": {
                "mean": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0,
            },
        }

    def compute_adaptive_impact(self) -> Dict:
        """Measure the impact of adaptive decisions."""
        if not self._records:
            return {"status": "no data"}

        # Group by strategy
        strategies = {}
        for r in self._records:
            s = r.strategy_used
            if s not in strategies:
                strategies[s] = {"count": 0, "latencies": [], "confidences": []}
            strategies[s]["count"] += 1
            strategies[s]["latencies"].append(r.total_latency)
            strategies[s]["confidences"].append(r.confidence)

        strategy_stats = {}
        for s, data in strategies.items():
            strategy_stats[s] = {
                "count": data["count"],
                "avg_latency": sum(data["latencies"]) / len(data["latencies"]),
                "avg_confidence": sum(data["confidences"]) / len(data["confidences"]),
            }

        # Group by K value
        k_values = {}
        for r in self._records:
            k = r.top_k_used
            if k not in k_values:
                k_values[k] = {"count": 0, "latencies": [], "confidences": []}
            k_values[k]["count"] += 1
            k_values[k]["latencies"].append(r.total_latency)
            k_values[k]["confidences"].append(r.confidence)

        k_stats = {}
        for k, data in k_values.items():
            k_stats[k] = {
                "count": data["count"],
                "avg_latency": sum(data["latencies"]) / len(data["latencies"]),
                "avg_confidence": sum(data["confidences"]) / len(data["confidences"]),
            }

        # Reranking impact
        with_rerank = [r for r in self._records if r.use_reranking]
        without_rerank = [r for r in self._records if not r.use_reranking]

        rerank_impact = {}
        if with_rerank:
            rerank_impact["with_reranking"] = {
                "count": len(with_rerank),
                "avg_latency": sum(r.total_latency for r in with_rerank) / len(with_rerank),
                "avg_confidence": sum(r.confidence for r in with_rerank) / len(with_rerank),
            }
        if without_rerank:
            rerank_impact["without_reranking"] = {
                "count": len(without_rerank),
                "avg_latency": sum(r.total_latency for r in without_rerank) / len(without_rerank),
                "avg_confidence": sum(r.confidence for r in without_rerank) / len(without_rerank),
            }

        # Complexity distribution
        complexity_dist = {"simple": 0, "medium": 0, "complex": 0}
        for r in self._records:
            if r.complexity_score < 0.3:
                complexity_dist["simple"] += 1
            elif r.complexity_score < 0.6:
                complexity_dist["medium"] += 1
            else:
                complexity_dist["complex"] += 1

        return {
            "strategy_breakdown": strategy_stats,
            "k_value_breakdown": k_stats,
            "reranking_impact": rerank_impact,
            "complexity_distribution": complexity_dist,
        }

    def generate_full_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        return {
            "latency": self.compute_latency_metrics(),
            "quality": self.compute_quality_metrics(),
            "adaptive_impact": self.compute_adaptive_impact(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save_report(self, filename: str = "performance_report.json") -> Path:
        """Save the full report to disk."""
        report = self.generate_full_report()
        path = REPORT_DIR / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[Metrics] Report saved to {path}")
        return path

    def print_report(self) -> None:
        """Pretty-print the performance report to console."""
        report = self.generate_full_report()

        lat = report["latency"]
        if "total_latency" in lat:
            print("\n--- Latency Metrics ---")
            print(f"  {'Metric':<10} {'Total':>10} {'Retrieval':>12} {'Generation':>12} {'Reranking':>12}")
            print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
            for pct in ["mean", "p50", "p95"]:
                print(f"  {pct.upper():<10} {lat['total_latency'][pct]:>9.3f}s {lat['retrieval_time'][pct]:>11.3f}s {lat['generation_time'][pct]:>11.3f}s {lat['reranking_time'][pct]:>11.3f}s")

            breakdown = lat["time_breakdown_pct"]
            print(f"\n  Time Breakdown: Retrieval {breakdown['retrieval']:.1f}% | Generation {breakdown['generation']:.1f}% | Reranking {breakdown['reranking']:.1f}%")

        qual = report["quality"]
        if "confidence" in qual:
            print("\n--- Quality Metrics ---")
            print(f"  {'Metric':<15} {'Mean':>10} {'Min':>10} {'Max':>10}")
            print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
            print(f"  {'Confidence':<15} {qual['confidence']['mean']:>10.4f} {qual['confidence']['min']:>10.4f} {qual['confidence']['max']:>10.4f}")
            print(f"  {'Answer Length':<15} {qual['answer_length']['mean']:>10.0f} {qual['answer_length']['min']:>10} {qual['answer_length']['max']:>10}")

        adaptive = report["adaptive_impact"]
        if "strategy_breakdown" in adaptive:
            print("\n--- Adaptive Impact: Strategy Breakdown ---")
            print(f"  {'Strategy':<12} {'Count':>8} {'Avg Latency':>14} {'Avg Confidence':>16}")
            print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*16}")
            for strat, data in adaptive["strategy_breakdown"].items():
                print(f"  {strat:<12} {data['count']:>8} {data['avg_latency']:>13.3f}s {data['avg_confidence']:>16.4f}")

        if "k_value_breakdown" in adaptive:
            print("\n--- Adaptive Impact: K-Value Breakdown ---")
            print(f"  {'K':>4} {'Count':>8} {'Avg Latency':>14} {'Avg Confidence':>16}")
            print(f"  {'-'*4} {'-'*8} {'-'*14} {'-'*16}")
            for k in sorted(adaptive["k_value_breakdown"].keys(), key=lambda x: int(x)):
                data = adaptive["k_value_breakdown"][k]
                print(f"  {str(k):>4} {data['count']:>8} {data['avg_latency']:>13.3f}s {data['avg_confidence']:>16.4f}")

        print(f"\nTotal queries analyzed: {lat.get('total_queries', 0)}")
        print(f"Report generated: {report['generated_at']}")
        print()

