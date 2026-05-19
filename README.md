# 🧠 Adaptive RAG Inference System

An intelligent Retrieval-Augmented Generation (RAG) pipeline that **optimizes itself at inference time**. The system dynamically adjusts retrieval depth, strategy, and re-ranking based on query complexity and real-time performance feedback.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Design Decisions](#design-decisions)
- [Tradeoffs](#tradeoffs)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Benchmarks](#benchmarks)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Query Analyzer                                 │
│  • Word count, question words, complex indicators                │
│  • Named entity heuristic, multi-hop detection                   │
│  • Keyword density → Complexity Score (0.0 – 1.0)                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Adaptive Decision Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐      │
│  │ Dynamic K   │  │  Strategy    │  │  Latency-Aware     │      │
│  │ simple → 3  │  │  simple→vec  │  │  high lat → ↓K     │      │
│  │ medium → 5  │  │  medium→hyb  │  │  critical → no     │      │
│  │ complex→ 10 │  │  complex→hyb │  │    rerank           │      │
│  └─────────────┘  └──────────────┘  └────────────────────┘      │
│                                                                  │
│  + Feedback Overrides (K adjustment, strategy forcing)           │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌────────────────┐
│ FAISS Vector │ │   BM25   │ │   Reciprocal   │
│   Search     │ │ Keyword  │ │   Rank Fusion  │
│ (cosine sim) │ │  Search  │ │   (RRF)        │
└──────────────┘ └──────────┘ └────────┬───────┘
                                       │
                                       ▼
                            ┌────────────────────┐
                            │  Cross-Encoder     │
                            │  Re-Ranker         │
                            │  (ms-marco-MiniLM) │
                            └────────┬───────────┘
                                     │
                                     ▼
                            ┌────────────────────┐
                            │  FLAN-T5 Generator │
                            │  (local, no API)   │
                            └────────┬───────────┘
                                     │
                                     ▼
                            ┌────────────────────┐
                            │    Response +      │
                            │    Confidence      │
                            └────────┬───────────┘
                                     │
                                     ▼
                            ┌────────────────────┐
                            │  Feedback Loop     │
                            │  • Track latency   │
                            │  • Track quality   │
                            │  • Adjust K        │
                            │  • Adjust strategy │
                            └────────────────────┘
```

---

## ✨ Features

| Part | Feature | Description |
|------|---------|-------------|
| **1** | Document Ingestion | SQuAD v1.1 dataset (~19K+ Wikipedia paragraphs, ~87K QA pairs) |
| **1** | Vector Index | FAISS with cosine similarity (384-dim normalized vectors) |
| **1** | Generation | Local FLAN-T5-base (no API keys needed) |
| **2** | Hybrid Retrieval | FAISS vector search + BM25 keyword search |
| **2** | Rank Fusion | Reciprocal Rank Fusion (RRF) for merging results |
| **2** | Re-ranking | Cross-encoder (ms-marco-MiniLM) for precise relevance |
| **3** | Query Analysis | Complexity scoring (word count, indicators, entities) |
| **3** | Dynamic K | K=3 (simple) / K=5 (medium) / K=10 (complex) |
| **3** | Strategy Selection | Vector-only, BM25-only, or Hybrid based on query type |
| **4** | Latency Tracking | Rolling window P50/P95/P99 monitoring |
| **4** | Quality Proxy | Confidence estimation from token probabilities |
| **4** | Adaptive Feedback | Auto-adjusts K and strategy based on performance trends |
| **5** | Performance Report | Latency breakdown, quality metrics, adaptive impact |
| **5** | Benchmark Suite | 50+ hardcoded + SQuAD validation queries with F1 scoring |
| **6** | Web Interface | Flask-based modern glassmorphic chat interface |
| **6** | Transparency UI | Real-time expandable view of pipeline latency, scores, and retrieved sources |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- ~4GB disk space (for models and dataset)
- ~4GB RAM minimum (8GB+ recommended)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd "Adaptive RAG"

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Start the Web Interface (Recommended)
python app.py
# This will open a local server at http://localhost:5000 with a modern chat UI

# Interactive CLI mode
python main.py

# Single CLI query
python main.py --query "What is photosynthesis?"

# Run benchmarks
python main.py --benchmark

# Force re-download dataset/models
python main.py --force-reload
```

**First run will automatically:**
1. Download SQuAD v1.1 dataset from HuggingFace (~30MB)
2. Process and chunk ~19,000+ paragraphs
3. Download embedding model all-MiniLM-L6-v2 (~80MB)
4. Encode all documents into FAISS index (~5-10 min)
5. Download FLAN-T5-base (~250MB) on first query
6. Download cross-encoder reranker (~80MB) on first rerank

Subsequent runs use cached data and are much faster.

---

## 🔧 How It Works

### Part 1: Basic Pipeline
1. **Ingestion**: Downloads SQuAD v1.1, extracts unique paragraphs, chunks with sliding window (512 chars, 50 overlap)
2. **Indexing**: Encodes chunks with `all-MiniLM-L6-v2`, stores in FAISS `IndexFlatIP` (cosine similarity)
3. **Generation**: FLAN-T5-base generates answers from retrieved context with structured prompts

### Part 2: Retrieval Optimization
- **Vector Search**: FAISS nearest-neighbor search on embedded queries
- **BM25 Search**: TF-IDF keyword matching for lexical recall
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) combines both rankings with configurable weights (70% vector, 30% BM25)
- **Re-ranking**: Cross-encoder scores (query, document) pairs jointly for precise relevance

### Part 3: Adaptive Decision Layer
The system analyzes each query and adapts:
- **Simple queries** (≤4 words): K=3, vector-only, no reranking → fast
- **Medium queries** (5-10 words): K=5, hybrid, with reranking → balanced
- **Complex queries** (>10 words, complex indicators): K=10, hybrid, with reranking → thorough

### Part 4: Feedback Loop
Tracks rolling-window metrics and adjusts:
- **High latency** → reduce K, disable reranking
- **Low confidence** → increase K to find better context
- **Poor retrieval scores** → force hybrid strategy
- **All rule-based**, no ML training required

### Part 5: Performance Measurement
Reports: P50/P95/P99 latency, retrieval vs generation breakdown, confidence distribution, adaptive decision impact, F1 scores against ground truth.

### Part 6: Web Interface
A modern, dark-themed Flask frontend provides a chat-like experience:
- **Real-Time Transparency**: Users can expand an "Analysis & Metrics" panel on every response to see exactly what the adaptive pipeline decided (K size, strategy, reranking) and exactly which source chunks were used to generate the answer.
- **Session Management**: Chat sessions are grouped dynamically in the sidebar.

---

## 🎯 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FLAN-T5-base over API** | Fully local, no API keys, reproducible, zero cost |
| **FAISS IndexFlatIP** | Exact search for correctness; dataset size (~19K) is manageable |
| **Sentence-transformers** | Fast, high-quality embeddings in 384 dimensions |
| **BM25 for keyword search** | Captures lexical matches that vector search can miss |
| **RRF over weighted sum** | Rank-based fusion is robust to score distribution differences |
| **Cross-encoder reranker** | Joint (query, doc) scoring is more accurate than bi-encoder |
| **Rule-based adaptation** | Transparent, debuggable, no training data needed |
| **Rolling window feedback** | Adapts to recent trends without being swayed by old data |
| **SQuAD v1.1 dataset** | Large (19K+ docs), diverse topics, has ground-truth for evaluation |

---

## ⚖️ Tradeoffs

| Tradeoff | Pro | Con |
|----------|-----|-----|
| **Local models** | No cost, privacy, reproducible | Slower, lower quality than GPT-4/Claude |
| **FLAN-T5-base** | Fast (~1-3s/query), 250MB | 512 token context limit, shorter answers |
| **Exact FAISS search** | 100% recall | O(n) search; would need IVF for millions of docs |
| **Rule-based adaptation** | Transparent, no training | Less flexible than learned policies |
| **BM25 stopword list** | Faster matching | May miss relevant function words |
| **Sliding window chunking** | Sentence boundary preservation | Overlap increases index size by ~10% |

---

## 📁 Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Central configuration
├── main.py                      # CLI entry point
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py             # Part 1: Document ingestion & chunking
│   ├── indexer.py               # Part 1: FAISS vector index
│   ├── generator.py             # Part 1: FLAN-T5 text generation
│   ├── retriever.py             # Part 2: Hybrid retrieval (Vector + BM25 + RRF)
│   ├── reranker.py              # Part 2: Cross-encoder re-ranking
│   ├── adaptive.py              # Part 3: Adaptive decision layer
│   ├── feedback.py              # Part 4: Feedback loop & adaptation
│   ├── pipeline.py              # Orchestrator: full pipeline
│   └── metrics.py               # Part 5: Performance measurement
│
├── benchmarks/
│   └── run_benchmark.py         # Benchmark suite
│
├── data/                        # Generated data (auto-created)
│   ├── processed_documents.json # Cached document chunks
│   └── faiss_index/             # Persisted FAISS index
│
└── report/
    ├── performance_report.json  # Performance metrics
    └── benchmark_results.json   # Benchmark results
```

---

## 💻 Usage

### Interactive Mode
```bash
python main.py
```

Commands:
- **Type any question** → Get an adaptive RAG answer
- `verbose <question>` → See adaptive decisions step-by-step
- `benchmark` → Run full benchmark suite
- `stats` → Show performance metrics
- `config` → Show configuration
- `feedback` → Show feedback loop state
- `reset` → Reset feedback history
- `quit` → Exit

### Single Query
```bash
python main.py --query "How does the immune system work?"
```

### Benchmarks
```bash
# Default: 50 hardcoded + 50 SQuAD queries
python main.py --benchmark

# Custom number of SQuAD queries
python main.py --benchmark --num-queries 100
```

---

## 📊 Benchmarks

After running `python main.py --benchmark`, find results in:
- `report/benchmark_results.json` — Raw results with per-query data
- `report/performance_report.json` — Aggregated performance metrics

The benchmark measures:
- **Latency**: P50/P95/P99 end-to-end, with retrieval vs generation breakdown
- **Quality**: Confidence scores, answer lengths, F1 against SQuAD ground truth
- **Adaptive Impact**: How different K values and strategies affect performance

---

# 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
Built with ❤️ by Pannagaram

