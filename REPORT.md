# Adaptive RAG Inference System: Final Report

## What Worked Well
1. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Combining FAISS (dense vector) and BM25 (sparse keyword) retrieval provided excellent recall. Vector search successfully captured semantic intent, while BM25 ensured specific noun-phrases and entities from the SQuAD dataset were not missed. RRF cleanly merged these without requiring complex score normalization.
2. **Cross-Encoder Re-ranking**: The `ms-marco-MiniLM-L-6-v2` cross-encoder significantly boosted the precision of the top results. By jointly scoring the query and document, it filtered out documents that were only tangentially related.
3. **Adaptive UI and Memory**: Building a persistent Flask backend with a "Gemini-style" glassmorphic UI drastically improved the system's usability. Passing conversational memory (the last 3 turns) into the prompt generation pipeline allowed for successful follow-up questions while respecting the token limits of smaller models.
4. **Local Execution**: Utilizing `FLAN-T5-base` enabled completely local, private, and cost-free execution. 

## What Didn't Work / Challenges
1. **Small LLM Hallucinations**: `FLAN-T5-base` (~250M parameters) initially struggled with negative constraints. When the context was completely irrelevant (e.g., asking about "IPL 2025" against a Wikipedia dataset), the model attempted to guess or extract random entities instead of saying "I cannot find the answer."
2. **Over-engineered Prompts**: Initially, complex multi-turn conversation histories were injected into the FLAN-T5 prompt string (e.g. `User: ... Assistant: ...`). This deeply confused the small model, causing it to fail on standard dataset questions. The solution was to simplify the prompt heavily to just `Context -> Question -> Answer`.
3. **Strict Programmatic Thresholds**: Attempting to intercept irrelevant queries by placing a hard minimum score on cross-encoder logits (e.g., dropping context if the score was below `-4.0`) proved too brittle. The MS MARCO cross-encoder sometimes gave valid SQuAD answers negative logits due to dataset distribution shifts, causing valid answers to be rejected. This approach was ultimately removed in favor of simpler prompt engineering.
4. **Windows Encoding Constraints**: The system originally logged adaptive decisions using Unicode characters (e.g., `→`). This immediately crashed the local Windows charmap encoding on certain setups. Replacing these with ASCII equivalents (`->`) solved the issue.

## How the System Adapts at Inference Time
The system features a dual-layer adaptation mechanism designed to optimize both accuracy and latency on the fly without any ML training required.

### 1. Query Complexity Analysis
Before hitting the vector database, the Query Analyzer examines the input for:
- Word count and length
- Named entities and proper nouns
- Multi-hop indicators (e.g., "because", "after", "compare")
- Question words (e.g., "who", "when", "why")

This yields a **Complexity Score (0.0 to 1.0)**.
- **Simple Queries** (`score < 0.3`): The pipeline fetches `K=3` documents, uses fast Vector-only retrieval, and skips re-ranking.
- **Medium Queries** (`0.3 <= score < 0.6`): Fetches `K=5` documents, uses Hybrid retrieval (Vector + BM25), and applies Re-ranking.
- **Complex Queries** (`score >= 0.6`): Fetches `K=10` documents, uses Hybrid retrieval, and applies Re-ranking to maximize recall before generation.

### 2. Real-Time Performance Feedback
The Feedback Loop maintains a rolling window of recent queries, tracking:
- **Latency**: If the P90 latency exceeds `1.5 seconds`, the system temporarily overrides the complexity analyzer. It forcefully disables re-ranking and clamps `K` to a maximum of 4 until latency recovers.
- **Quality (Confidence Proxy)**: The system tracks the generation token probabilities. If confidence scores consistently drop below `0.6`, the system forces the use of Hybrid search (ignoring the "Vector-only" fast path) to ensure higher quality context is provided to the LLM. 

This creates a self-healing pipeline that automatically throttles its own complexity when under load, and expands its search scope when struggling to find accurate answers.
