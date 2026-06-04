"""
Retrieval and generation quality metrics.

Implements standard IR metrics for measuring retrieval quality
and LLM-based metrics for generation quality.

These metrics are the core of the evaluation framework — they let you
make data-driven decisions about chunking strategies, retrieval methods,
and prompt engineering.
"""

import math
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_k: float = 0.0  # Normalized Discounted Cumulative Gain
    hit_rate: float = 0.0  # Did we retrieve at least one relevant doc?
    k: int = 5


@dataclass
class GenerationMetrics:
    """Metrics for evaluating generation quality."""

    faithfulness: float = 0.0  # Fraction of claims supported by context
    answer_relevance: float = 0.0  # How relevant is the answer to the query
    citation_accuracy: float = 0.0  # Are citations pointing to correct sources
    completeness: float = 0.0  # Does the answer cover all aspects of the query


@dataclass
class LatencyMetrics:
    """System performance metrics."""

    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    hallucination_check_ms: float = 0.0


@dataclass
class EvalResult:
    """Complete evaluation result for a single query."""

    query: str
    expected_answer: str | None = None
    actual_answer: str = ""
    retrieval_metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation_metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    latency_metrics: LatencyMetrics = field(default_factory=LatencyMetrics)
    token_usage: int = 0
    cost_usd: float = 0.0


class MetricsCalculator:
    """
    Calculate retrieval and generation quality metrics.

    Standard IR metrics implementation for evaluating RAG systems.
    """

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Precision@k: fraction of top-k retrieved docs that are relevant.

        P@k = |relevant ∩ retrieved[:k]| / k
        """
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Recall@k: fraction of relevant docs that appear in top-k.

        R@k = |relevant ∩ retrieved[:k]| / |relevant|
        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_top_k / len(relevant_ids)

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """
        MRR: 1/rank of the first relevant document.

        MRR = 1 / rank_of_first_relevant
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        NDCG@k: Normalized Discounted Cumulative Gain.

        Accounts for position of relevant documents (higher is better).
        Uses binary relevance (1 if relevant, 0 if not).
        """
        if not relevant_ids or k == 0:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0

        # Ideal DCG (all relevant docs at top)
        ideal_relevant_count = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant_count))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Hit Rate@k: 1 if any relevant doc in top-k, else 0.

        Simple binary metric — did we find anything useful?
        """
        top_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0

    def compute_retrieval_metrics(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 5,
    ) -> RetrievalMetrics:
        """Compute all retrieval metrics at once."""
        return RetrievalMetrics(
            precision_at_k=self.precision_at_k(retrieved_ids, relevant_ids, k),
            recall_at_k=self.recall_at_k(retrieved_ids, relevant_ids, k),
            mrr=self.mean_reciprocal_rank(retrieved_ids, relevant_ids),
            ndcg_at_k=self.ndcg_at_k(retrieved_ids, relevant_ids, k),
            hit_rate=self.hit_rate(retrieved_ids, relevant_ids, k),
            k=k,
        )

    @staticmethod
    def compute_latency_percentiles(latencies: list[float]) -> dict:
        """
        Compute latency percentiles from a list of latency measurements.

        Returns p50, p95, p99 in milliseconds.
        """
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(p / 100 * (n - 1))
            return sorted_latencies[idx]

        return {
            "p50": round(percentile(50), 2),
            "p95": round(percentile(95), 2),
            "p99": round(percentile(99), 2),
            "mean": round(sum(latencies) / n, 2),
            "count": n,
        }
