"""
Flask Web Server for the Adaptive RAG Inference System.
Provides a beautiful web UI and REST API endpoints.
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from config import CONFIG, REPORT_DIR
from src.pipeline import AdaptiveRAGPipeline

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global pipeline instance
pipeline = None
init_status = {"state": "not_started", "message": ""}


def get_pipeline():
    """Get or initialize the pipeline."""
    global pipeline, init_status
    if pipeline is None:
        try:
            init_status = {"state": "initializing", "message": "Loading documents..."}
            pipeline = AdaptiveRAGPipeline()
            pipeline.initialize()
            init_status = {"state": "ready", "message": "Pipeline ready!"}
        except Exception as e:
            init_status = {"state": "error", "message": str(e)}
            traceback.print_exc()
            raise
    return pipeline


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Get pipeline initialization status."""
    return jsonify(init_status)


@app.route("/api/query", methods=["POST"])
def api_query():
    """Process a query through the RAG pipeline."""
    try:
        p = get_pipeline()
        data = request.get_json()
        query_text = data.get("query", "").strip()
        verbose = data.get("verbose", False)
        history = data.get("history", [])

        if not query_text:
            return jsonify({"error": "Empty query"}), 400

        response = p.query(query_text, verbose=verbose, history=history)

        return jsonify({
            "answer": response.answer,
            "confidence": round(response.confidence, 4),
            "total_latency": round(response.total_latency, 3),
            "retrieval_time": round(response.retrieval_time, 3),
            "generation_time": round(response.generation_time, 3),
            "rerank_time": round(response.rerank_time, 3),
            "top_k_used": response.top_k_used,
            "strategy_used": response.strategy_used,
            "used_reranking": response.used_reranking,
            "complexity_score": round(response.complexity_score, 2),
            "num_docs": response.num_docs_retrieved,
            "adaptive_reason": response.adaptive_reason,
            "sources": [
                {"text": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                 "title": doc.title,
                 "score": round(doc.score, 4) if doc.score else 0}
                for doc in (response.source_documents[:5] if response.source_documents else [])
            ],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback")
def api_feedback():
    """Get feedback loop state."""
    try:
        p = get_pipeline()
        state = p.feedback.get_state_summary()
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Get performance statistics."""
    try:
        p = get_pipeline()
        p.metrics.set_records(p.feedback.history)
        report = p.metrics.generate_full_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config")
def api_config():
    """Get current configuration."""
    return jsonify({
        "embedding_model": CONFIG.embedding_model,
        "generator_model": CONFIG.generator_model,
        "reranker_model": CONFIG.reranker_model,
        "default_top_k": CONFIG.default_top_k,
        "min_k": CONFIG.min_k,
        "max_k": CONFIG.max_k,
        "chunk_size": CONFIG.chunk_size,
    })


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """Run a mini benchmark."""
    try:
        p = get_pipeline()
        data = request.get_json() or {}
        queries = data.get("queries", [
            "What is oxygen?",
            "Who wrote Hamlet?",
            "How does the immune system protect the body?",
            "What is photosynthesis?",
            "Explain the relationship between supply and demand",
        ])

        results = []
        for q in queries:
            try:
                resp = p.query(q, verbose=False)
                results.append({
                    "query": q,
                    "answer": resp.answer,
                    "confidence": round(resp.confidence, 4),
                    "latency": round(resp.total_latency, 3),
                    "strategy": resp.strategy_used,
                    "top_k": resp.top_k_used,
                    "complexity": round(resp.complexity_score, 2),
                })
            except Exception as e:
                results.append({"query": q, "error": str(e)})

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Adaptive RAG - Web Interface")
    print("  Starting server at http://localhost:5000")
    print("=" * 60 + "\n")

    # Pre-initialize the pipeline
    print("[Server] Initializing RAG pipeline...")
    try:
        get_pipeline()
        print("[Server] Pipeline ready! Opening web UI...")
    except Exception as e:
        print(f"[Server] Warning: Pipeline init failed: {e}")
        print("[Server] Will retry on first request.")

    app.run(host="0.0.0.0", port=5000, debug=False)
