"""
Adaptive RAG Inference System — Main Entry Point
-------------------------------------------------
Interactive CLI for querying, benchmarking, and inspecting the pipeline.

Usage:
    python main.py                  → Start interactive CLI
    python main.py --benchmark      → Run benchmark suite
    python main.py --query "..."    → Single query mode
"""

import sys
import os
import argparse
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from config import CONFIG
    from src.pipeline import AdaptiveRAGPipeline
except Exception as e:
    print(f"Import error: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


console = Console()


def print_banner():
    """Print the application banner."""
    print("=" * 63)
    print("   Adaptive RAG Inference System")
    print()
    print("   An intelligent retrieval-augmented generation pipeline")
    print("   that optimizes itself at inference time.")
    print()
    print("   Dataset: SQuAD v1.1 (~19K Wikipedia paragraphs)")
    print("   Models:  all-MiniLM-L6-v2 + FLAN-T5-base")
    print("=" * 63)
    print()


def print_help():
    """Print available commands."""
    print("\n--- Available Commands ---")
    print("  query <text>      Ask a question (or just type your question)")
    print("  verbose <text>    Ask with verbose output showing adaptive decisions")
    print("  benchmark [n]     Run benchmark suite (n = num SQuAD queries, default 50)")
    print("  stats             Show performance statistics")
    print("  config            Show current configuration")
    print("  feedback          Show feedback loop state")
    print("  reset             Reset feedback state")
    print("  help              Show this help")
    print("  quit / exit       Exit the program")
    print()


def display_response(response):
    """Display a query response."""
    print()
    print("-" * 50)
    print(f"  ANSWER: {response.answer}")
    print("-" * 50)
    print(f"  Total Latency   : {response.total_latency:.3f}s")
    print(f"  Retrieval Time  : {response.retrieval_time:.3f}s")
    if response.used_reranking:
        print(f"  Re-ranking Time : {response.rerank_time:.3f}s")
    print(f"  Generation Time : {response.generation_time:.3f}s")
    print(f"  Confidence      : {response.confidence:.4f}")
    print(f"  Top-K Used      : {response.top_k_used}")
    print(f"  Strategy        : {response.strategy_used}")
    print(f"  Reranking       : {'Yes' if response.used_reranking else 'No'}")
    print(f"  Complexity      : {response.complexity_score:.2f}")
    print(f"  Docs Retrieved  : {response.num_docs_retrieved}")
    print(f"  Adaptive Reason : {response.adaptive_reason}")
    print()


def interactive_mode(pipeline: AdaptiveRAGPipeline):
    """Run the interactive CLI loop."""
    print_help()

    while True:
        try:
            user_input = input("RAG> ").strip()

            if not user_input:
                continue

            # Parse commands
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()

            if command in ("quit", "exit", "q"):
                print("\nSaving state and exiting...")
                pipeline.save_report()
                break

            elif command == "help":
                print_help()

            elif command == "stats":
                pipeline.print_stats()

            elif command == "config":
                print(f"\n--- Configuration ---")
                print(f"  Embedding Model : {CONFIG.model.embedding_model}")
                print(f"  Generator Model : {CONFIG.model.generator_model}")
                print(f"  Reranker Model  : {CONFIG.model.reranker_model}")
                print(f"  Default K       : {CONFIG.retrieval.default_top_k}")
                print(f"  K Range         : [{CONFIG.retrieval.min_top_k}, {CONFIG.retrieval.max_top_k}]")
                print(f"  Strategy        : {CONFIG.retrieval.retrieval_strategy}")
                print(f"  Reranking       : {'enabled' if CONFIG.retrieval.use_reranking else 'disabled'}")
                print(f"  Chunk Size      : {CONFIG.ingestion.chunk_size}")
                print(f"  BM25 Weight     : {CONFIG.retrieval.bm25_weight}")
                print(f"  Vector Weight   : {CONFIG.retrieval.vector_weight}")
                print()

            elif command == "feedback":
                stats = pipeline.feedback.get_rolling_stats()
                print("\n--- Feedback Loop State ---")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                print()

            elif command == "reset":
                pipeline.reset_feedback()
                print("Feedback state reset.")

            elif command == "benchmark":
                n = int(parts[1]) if len(parts) > 1 else 50
                from benchmarks.run_benchmark import run_benchmark
                run_benchmark(pipeline, num_squad_queries=n)

            elif command == "verbose":
                if len(parts) < 2:
                    print("Please provide a query after 'verbose'")
                    continue
                query_text = parts[1]
                response = pipeline.query(query_text, verbose=True)
                display_response(response)

            elif command == "query":
                if len(parts) < 2:
                    print("Please provide a query after 'query'")
                    continue
                query_text = parts[1]
                response = pipeline.query(query_text, verbose=False)
                display_response(response)

            else:
                # Treat entire input as a query
                response = pipeline.query(user_input, verbose=False)
                display_response(response)

        except KeyboardInterrupt:
            print("\nSaving state...")
            pipeline.save_report()
            break
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Adaptive RAG Inference System")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--query", "-q", type=str, help="Single query mode")
    parser.add_argument("--force-reload", action="store_true", help="Force re-download and re-index")
    parser.add_argument("--num-queries", "-n", type=int, default=50, help="Number of SQuAD queries for benchmark")

    args = parser.parse_args()

    print_banner()

    # Initialize pipeline
    pipeline = AdaptiveRAGPipeline()
    pipeline.initialize(force_reload=args.force_reload)

    if args.benchmark:
        from benchmarks.run_benchmark import run_benchmark
        run_benchmark(pipeline, num_squad_queries=args.num_queries)
        pipeline.print_stats()

    elif args.query:
        response = pipeline.query(args.query, verbose=True)
        display_response(response)

    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

