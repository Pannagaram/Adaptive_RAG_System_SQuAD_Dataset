"""
Benchmark Suite
----------------
Runs automated benchmarks comparing fixed vs adaptive RAG pipeline.
Uses sample queries from SQuAD to measure performance.
"""

import sys
import os
import time
import json
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG, REPORT_DIR
from src.pipeline import AdaptiveRAGPipeline
from src.feedback import QueryRecord


# ── Benchmark Queries ────────────────────────────────────────────────────────
# Mix of simple, medium, and complex queries
BENCHMARK_QUERIES = [
    # Simple (short, factual)
    "What is oxygen?",
    "Who wrote Hamlet?",
    "Where is Paris?",
    "What is DNA?",
    "Who invented the telephone?",
    "What is gravity?",
    "Where is the Nile?",
    "What is photosynthesis?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "What is a molecule?",
    "Who discovered penicillin?",
    "What is the capital of France?",
    "What is an atom?",
    "Who was Aristotle?",
    "What is the sun?",
    "What is evolution?",
    "What is a cell?",
    "Who is Newton?",
    "What is electricity?",

    # Medium (moderate length, some specificity)
    "How does the immune system protect the body?",
    "What was the cause of World War I?",
    "How do computers store information in memory?",
    "What role does chlorophyll play in photosynthesis?",
    "How did the Roman Empire fall?",
    "What is the difference between a virus and a bacterium?",
    "How does natural selection drive evolution?",
    "What are the main functions of the United Nations?",
    "How does the heart pump blood through the body?",
    "What were the key events of the French Revolution?",
    "How do vaccines work to prevent disease?",
    "What is the structure of the solar system?",
    "How did the Industrial Revolution change society?",
    "What processes cause earthquakes?",
    "How does the brain process visual information?",
    "What factors contribute to climate change?",
    "How do plants absorb water and nutrients?",
    "What was the significance of the printing press?",
    "How does the digestive system break down food?",
    "What are the properties of electromagnetic waves?",

    # Complex (long, multi-faceted, analytical)
    "Explain the relationship between supply and demand in economics and how government intervention affects market equilibrium",
    "How did the development of the steam engine influence both industrial manufacturing and transportation systems in the 19th century?",
    "Describe the process of DNA replication and explain why errors in this process can lead to genetic mutations",
    "What are the main differences between renewable and non-renewable energy sources and how do they impact the environment?",
    "Explain how the three branches of government in the United States check and balance each other's power",
    "How does the water cycle work and what role do human activities play in disrupting natural water patterns?",
    "Compare and contrast the philosophical approaches of empiricism and rationalism in the history of Western thought",
    "What are the key factors that led to the decline of ancient civilizations and how do they relate to modern challenges?",
    "Explain how antibiotics work at a molecular level and why antibiotic resistance is becoming a major public health concern",
    "How do gravitational forces between celestial bodies influence the formation and evolution of planetary systems?",
]


def load_squad_queries(n: int = 50) -> list:
    """Load sample queries from the SQuAD dataset for benchmarking."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("rajpurkar/squad", split="validation")

        # Sample diverse queries
        indices = random.sample(range(len(dataset)), min(n, len(dataset)))
        queries = []
        for i in indices:
            queries.append({
                "question": dataset[i]["question"],
                "ground_truth": dataset[i]["answers"]["text"][0] if dataset[i]["answers"]["text"] else "",
                "context_title": dataset[i].get("title", ""),
            })
        return queries
    except Exception as e:
        print(f"[Benchmark] Could not load SQuAD queries: {e}")
        return []


def run_benchmark(pipeline: AdaptiveRAGPipeline, num_squad_queries: int = 50) -> dict:
    """
    Run the full benchmark suite.
    """
    from tqdm import tqdm

    print("\n=== Running Benchmark Suite ===\n")

    results = {
        "benchmark_queries": [],
        "squad_queries": [],
        "summary": {},
    }

    # Phase 1: Hardcoded Benchmark Queries
    print(f"Phase 1: Running {len(BENCHMARK_QUERIES)} benchmark queries...\n")

    for query in tqdm(BENCHMARK_QUERIES, desc="Benchmark queries"):
        try:
            response = pipeline.query(query, verbose=False)
            results["benchmark_queries"].append({
                "query": query,
                "answer": response.answer,
                "confidence": response.confidence,
                "total_latency": response.total_latency,
                "retrieval_time": response.retrieval_time,
                "generation_time": response.generation_time,
                "rerank_time": response.rerank_time,
                "top_k": response.top_k_used,
                "strategy": response.strategy_used,
                "reranking": response.used_reranking,
                "complexity": response.complexity_score,
            })
        except Exception as e:
            print(f"  Error on '{query[:50]}': {e}")

    # Phase 2: SQuAD Queries
    print(f"\nPhase 2: Running {num_squad_queries} SQuAD queries...\n")
    squad_queries = load_squad_queries(num_squad_queries)

    if squad_queries:
        for sq in tqdm(squad_queries, desc="SQuAD queries"):
            try:
                response = pipeline.query(sq["question"], verbose=False)
                f1 = compute_f1(response.answer, sq["ground_truth"]) if sq["ground_truth"] else 0.0

                results["squad_queries"].append({
                    "query": sq["question"],
                    "answer": response.answer,
                    "ground_truth": sq["ground_truth"],
                    "f1_score": f1,
                    "confidence": response.confidence,
                    "total_latency": response.total_latency,
                    "top_k": response.top_k_used,
                    "strategy": response.strategy_used,
                    "complexity": response.complexity_score,
                })
            except Exception as e:
                print(f"  Error: {e}")

    # Phase 3: Compute Summary
    all_results = results["benchmark_queries"] + results["squad_queries"]

    if all_results:
        latencies = [r["total_latency"] for r in all_results]
        confidences = [r["confidence"] for r in all_results]
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        results["summary"] = {
            "total_queries": n,
            "latency_p50": sorted_lat[n // 2],
            "latency_p95": sorted_lat[int(n * 0.95)] if n >= 5 else sorted_lat[-1],
            "latency_mean": sum(latencies) / n,
            "confidence_mean": sum(confidences) / n,
        }

        if results["squad_queries"]:
            f1_scores = [r["f1_score"] for r in results["squad_queries"]]
            results["summary"]["squad_f1_mean"] = sum(f1_scores) / len(f1_scores)

        strat_counts = {}
        for r in all_results:
            s = r["strategy"]
            strat_counts[s] = strat_counts.get(s, 0) + 1
        results["summary"]["strategy_distribution"] = strat_counts

        k_counts = {}
        for r in all_results:
            k = r["top_k"]
            k_counts[k] = k_counts.get(k, 0) + 1
        results["summary"]["k_distribution"] = k_counts

    # Print Results
    print("\n=== Benchmark Results ===\n")
    summary = results["summary"]
    print(f"  Total Queries   : {summary.get('total_queries', 0)}")
    print(f"  Latency P50     : {summary.get('latency_p50', 0):.3f}s")
    print(f"  Latency P95     : {summary.get('latency_p95', 0):.3f}s")
    print(f"  Latency Mean    : {summary.get('latency_mean', 0):.3f}s")
    print(f"  Confidence Mean : {summary.get('confidence_mean', 0):.4f}")
    if "squad_f1_mean" in summary:
        print(f"  SQuAD F1 Mean   : {summary['squad_f1_mean']:.4f}")

    if "strategy_distribution" in summary:
        print(f"\n  Strategy Distribution: {summary['strategy_distribution']}")
    if "k_distribution" in summary:
        print(f"  K Distribution: {summary['k_distribution']}")

    # Save benchmark results
    report_path = REPORT_DIR / "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {report_path}")

    # Also save the pipeline performance report
    pipeline.save_report()

    return results


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = AdaptiveRAGPipeline()
    pipeline.initialize()
    run_benchmark(pipeline, num_squad_queries=50)
    pipeline.print_stats()
