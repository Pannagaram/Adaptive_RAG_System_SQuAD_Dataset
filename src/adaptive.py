"""
Part 3: Adaptive Decision Layer
---------------------------------
Analyzes query complexity at runtime and dynamically selects:
  - top-K value
  - retrieval strategy (vector / bm25 / hybrid)
  - whether to use re-ranking
  - latency-aware adjustments

All decisions are rule-based — no ML training required.
"""

import re
from dataclasses import dataclass
from typing import Optional

from config import CONFIG


@dataclass
class QueryAnalysis:
    """Results of analyzing a query's complexity."""
    query: str
    word_count: int
    complexity_score: float        # 0.0 = simple, 1.0 = very complex
    complexity_level: str          # "simple", "medium", "complex"
    has_question_word: bool
    has_complex_indicator: bool
    has_named_entities: bool       # heuristic
    is_multi_hop: bool            # heuristic
    keyword_density: float         # ratio of content words to total words


@dataclass
class AdaptiveDecision:
    """The adaptive layer's decision for a query."""
    top_k: int
    strategy: str                  # "vector", "bm25", "hybrid"
    use_reranking: bool
    reason: str                    # Human-readable explanation
    complexity_score: float
    latency_adjusted: bool = False


class AdaptiveDecisionLayer:
    """
    Runtime selector that analyzes queries and adapts retrieval parameters.
    Combines query complexity analysis with latency-aware adjustments.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG.adaptive
        self._recent_latencies = []  # Populated by feedback loop
        self._current_overrides = {}  # Feedback-driven overrides

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to determine its complexity.

        Factors considered:
          - Word count
          - Presence of question words (what, why, how, etc.)
          - Complex indicators (explain, compare, analyze, etc.)
          - Named entity heuristic (capitalized words)
          - Multi-hop indicators (and, also, both, relationship, etc.)
          - Keyword density (content words vs function words)
        """
        words = query.strip().split()
        word_count = len(words)
        lower_query = query.lower().strip()

        # Question word detection
        question_words = {"what", "where", "when", "who", "whom", "which",
                          "why", "how", "is", "are", "was", "were", "do", "does", "did",
                          "can", "could", "would", "should"}
        has_question_word = any(lower_query.startswith(qw) for qw in question_words)

        # Complex indicator detection
        has_complex_indicator = any(
            ind in lower_query for ind in self.config.complex_indicators
        )

        # Named entity heuristic: words starting with uppercase (not at sentence start)
        has_named_entities = any(
            w[0].isupper() for w in words[1:] if len(w) > 1
        ) if len(words) > 1 else False

        # Multi-hop indicators
        multi_hop_words = {"and", "also", "both", "relationship", "between",
                           "compared", "versus", "vs", "affect", "influence",
                           "relate", "connect", "impact"}
        is_multi_hop = any(w.lower() in multi_hop_words for w in words)

        # Keyword density: ratio of non-stopword to total words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                     "to", "for", "on", "with", "at", "by", "from", "it", "its",
                     "this", "that", "and", "or", "but", "not"}
        content_words = [w for w in words if w.lower() not in stopwords]
        keyword_density = len(content_words) / max(word_count, 1)

        # ── Compute complexity score (0.0 – 1.0) ────────────────────────────
        score = 0.0

        # Length factor (longer = more complex)
        if word_count <= self.config.short_query_max_words:
            score += 0.1
        elif word_count <= self.config.medium_query_max_words:
            score += 0.3
        else:
            score += 0.5

        # Complexity indicators
        if has_complex_indicator:
            score += 0.2
        if is_multi_hop:
            score += 0.15
        if has_named_entities:
            score += 0.05
        if keyword_density > 0.6:
            score += 0.1

        score = min(1.0, score)

        # Classify
        if score < 0.3:
            level = "simple"
        elif score < 0.6:
            level = "medium"
        else:
            level = "complex"

        return QueryAnalysis(
            query=query,
            word_count=word_count,
            complexity_score=score,
            complexity_level=level,
            has_question_word=has_question_word,
            has_complex_indicator=has_complex_indicator,
            has_named_entities=has_named_entities,
            is_multi_hop=is_multi_hop,
            keyword_density=keyword_density,
        )

    def decide(self, query: str,
               feedback_overrides: Optional[dict] = None) -> AdaptiveDecision:
        """
        Make adaptive decisions for a query.

        Args:
            query: The user's query
            feedback_overrides: Optional overrides from the feedback loop

        Returns:
            AdaptiveDecision with top_k, strategy, and re-ranking settings
        """
        analysis = self.analyze_query(query)

        # ── Base decisions from complexity ───────────────────────────────────
        if analysis.complexity_level == "simple":
            top_k = self.config.simple_k
            strategy = "vector"           # Fast vector-only for simple queries
            use_reranking = False          # Skip re-ranking for speed
            reason = f"Simple query ({analysis.word_count} words, score={analysis.complexity_score:.2f}) → small K, vector-only, no rerank"

        elif analysis.complexity_level == "medium":
            top_k = self.config.medium_k
            strategy = "hybrid"           # Use hybrid for medium complexity
            use_reranking = True
            reason = f"Medium query ({analysis.word_count} words, score={analysis.complexity_score:.2f}) → moderate K, hybrid, with rerank"

        else:  # complex
            top_k = self.config.complex_k
            strategy = "hybrid"           # Full hybrid for complex queries
            use_reranking = True
            reason = f"Complex query ({analysis.word_count} words, score={analysis.complexity_score:.2f}) → large K, hybrid, with rerank"

        latency_adjusted = False

        # ── Apply feedback overrides ─────────────────────────────────────────
        if feedback_overrides:
            k_adj = feedback_overrides.get("k_adjustment", 0)
            if k_adj != 0:
                old_k = top_k
                top_k = max(CONFIG.retrieval.min_top_k,
                           min(CONFIG.retrieval.max_top_k, top_k + k_adj))
                reason += f" | Feedback: K adjusted {old_k}→{top_k}"

            if feedback_overrides.get("disable_reranking", False):
                use_reranking = False
                reason += " | Feedback: reranking disabled (latency)"
                latency_adjusted = True

            if feedback_overrides.get("force_strategy"):
                strategy = feedback_overrides["force_strategy"]
                reason += f" | Feedback: forced strategy={strategy}"

            if feedback_overrides.get("latency_reduction", False):
                # Reduce retrieval depth for speed
                top_k = max(CONFIG.retrieval.min_top_k, top_k - 2)
                use_reranking = False
                latency_adjusted = True
                reason += f" | Latency reduction: K→{top_k}, rerank off"

        return AdaptiveDecision(
            top_k=top_k,
            strategy=strategy,
            use_reranking=use_reranking,
            reason=reason,
            complexity_score=analysis.complexity_score,
            latency_adjusted=latency_adjusted,
        )


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    layer = AdaptiveDecisionLayer()

    queries = [
        "What is DNA?",
        "How does photosynthesis work in plants?",
        "Explain the relationship between quantum mechanics and general relativity and how they affect our understanding of black holes",
    ]

    for q in queries:
        analysis = layer.analyze_query(q)
        decision = layer.decide(q)
        print(f"\nQuery: {q}")
        print(f"  Analysis: {analysis.complexity_level} (score={analysis.complexity_score:.2f})")
        print(f"  Decision: K={decision.top_k}, strategy={decision.strategy}, rerank={decision.use_reranking}")
        print(f"  Reason: {decision.reason}")
