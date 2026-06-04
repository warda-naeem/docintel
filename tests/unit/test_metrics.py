"""
Unit tests for evaluation metrics.

Tests the core IR metrics calculations without any external dependencies.
These tests validate the mathematical correctness of our metrics.
"""

import pytest

from src.eval.metrics import MetricsCalculator


class TestPrecisionAtK:
    """Test Precision@k metric."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_perfect_precision(self):
        """All retrieved docs are relevant."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c", "d", "e"}
        assert self.calc.precision_at_k(retrieved, relevant, k=5) == 1.0

    def test_zero_precision(self):
        """No retrieved docs are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert self.calc.precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_precision(self):
        """Some retrieved docs are relevant."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        # 3 relevant in top 5 = 0.6
        assert self.calc.precision_at_k(retrieved, relevant, k=5) == 0.6

    def test_precision_at_k_smaller_than_results(self):
        """k is smaller than total retrieved."""
        retrieved = ["a", "b", "x", "y", "z"]
        relevant = {"a", "b"}
        # 2 relevant in top 3 = 0.667
        assert round(self.calc.precision_at_k(retrieved, relevant, k=3), 3) == 0.667

    def test_k_zero(self):
        """Edge case: k=0."""
        assert self.calc.precision_at_k(["a"], {"a"}, k=0) == 0.0


class TestRecallAtK:
    """Test Recall@k metric."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_perfect_recall(self):
        """All relevant docs are retrieved."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert self.calc.recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self):
        """Only some relevant docs retrieved."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        # 1 out of 3 relevant = 0.333
        assert round(self.calc.recall_at_k(retrieved, relevant, k=3), 3) == 0.333

    def test_empty_relevant_set(self):
        """No relevant docs exist."""
        retrieved = ["a", "b"]
        relevant = set()
        assert self.calc.recall_at_k(retrieved, relevant, k=2) == 0.0


class TestMRR:
    """Test Mean Reciprocal Rank."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_first_result_relevant(self):
        """First result is relevant → MRR = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert self.calc.mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_result_relevant(self):
        """Second result is relevant → MRR = 0.5."""
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert self.calc.mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_result_relevant(self):
        """Third result is relevant → MRR = 0.333."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert round(self.calc.mean_reciprocal_rank(retrieved, relevant), 3) == 0.333

    def test_no_relevant_found(self):
        """No relevant docs in results → MRR = 0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert self.calc.mean_reciprocal_rank(retrieved, relevant) == 0.0


class TestNDCG:
    """Test Normalized Discounted Cumulative Gain."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_perfect_ndcg(self):
        """All relevant docs at top positions."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        assert self.calc.ndcg_at_k(retrieved, relevant, k=5) == 1.0

    def test_zero_ndcg(self):
        """No relevant docs retrieved."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert self.calc.ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_ndcg(self):
        """Relevant docs not at ideal positions."""
        retrieved = ["x", "a", "y", "b", "z"]
        relevant = {"a", "b"}
        ndcg = self.calc.ndcg_at_k(retrieved, relevant, k=5)
        # Should be less than 1.0 since relevant docs aren't at top
        assert 0.0 < ndcg < 1.0

    def test_empty_relevant(self):
        """No relevant docs defined."""
        retrieved = ["a", "b"]
        relevant = set()
        assert self.calc.ndcg_at_k(retrieved, relevant, k=2) == 0.0


class TestHitRate:
    """Test Hit Rate metric."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_hit(self):
        """At least one relevant doc in top-k."""
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        assert self.calc.hit_rate(retrieved, relevant, k=3) == 1.0

    def test_miss(self):
        """No relevant doc in top-k."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert self.calc.hit_rate(retrieved, relevant, k=3) == 0.0


class TestLatencyPercentiles:
    """Test latency percentile calculations."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_basic_percentiles(self):
        """Test with known distribution."""
        latencies = list(range(1, 101))  # 1 to 100
        stats = self.calc.compute_latency_percentiles(latencies)

        assert stats["p50"] == 50.0
        assert stats["p95"] == 95.0
        assert stats["p99"] == 99.0
        assert stats["count"] == 100

    def test_empty_latencies(self):
        """Empty list returns zeros."""
        stats = self.calc.compute_latency_percentiles([])
        assert stats["p50"] == 0.0

    def test_single_value(self):
        """Single value returns that value for all percentiles."""
        stats = self.calc.compute_latency_percentiles([42.0])
        assert stats["p50"] == 42.0
        assert stats["p95"] == 42.0
