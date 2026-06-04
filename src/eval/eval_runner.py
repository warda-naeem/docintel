"""
Evaluation runner — executes eval datasets against the pipeline.

Runs a set of test queries with known-good answers and measures
retrieval quality, generation faithfulness, and latency.

Usage:
    python -m src.eval.eval_runner --dataset tests/eval/test_dataset.json
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from src.eval.metrics import (
    MetricsCalculator,
    RetrievalMetrics,
    GenerationMetrics,
    LatencyMetrics,
    EvalResult,
)


@dataclass
class EvalCase:
    """A single evaluation test case."""

    query: str
    expected_chunk_ids: list[str]  # Ground truth relevant chunks
    expected_answer: str | None = None  # Optional reference answer
    tags: list[str] | None = None  # For filtering (e.g., "factual", "multi-hop")


@dataclass
class EvalSummary:
    """Summary of an evaluation run."""

    total_cases: int
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_mrr: float
    avg_ndcg: float
    avg_hit_rate: float
    avg_faithfulness: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    total_tokens: int
    total_cost_usd: float
    timestamp: str
    config: dict


class EvalRunner:
    """
    Runs evaluation datasets against the RAG pipeline.

    Workflow:
    1. Load test dataset (queries + ground truth)
    2. Run each query through the pipeline
    3. Compare retrieved chunks against ground truth
    4. Measure generation quality
    5. Aggregate metrics and produce report
    """

    def __init__(self) -> None:
        self.calculator = MetricsCalculator()
        self.results: list[EvalResult] = []

    def load_dataset(self, path: str) -> list[EvalCase]:
        """Load evaluation dataset from JSON file."""
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(dataset_path) as f:
            data = json.load(f)

        cases = []
        for item in data.get("cases", []):
            cases.append(EvalCase(
                query=item["query"],
                expected_chunk_ids=item.get("expected_chunk_ids", []),
                expected_answer=item.get("expected_answer"),
                tags=item.get("tags"),
            ))

        return cases

    def evaluate_retrieval(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
        k: int = 5,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality for a single query."""
        relevant_set = set(expected_ids)
        return self.calculator.compute_retrieval_metrics(retrieved_ids, relevant_set, k)

    def run_eval(self, cases: list[EvalCase], k: int = 5) -> EvalSummary:
        """
        Run full evaluation on a dataset.

        This is a simplified version that evaluates retrieval metrics.
        In production, this would call the full pipeline.
        """
        all_latencies = []
        total_tokens = 0
        total_cost = 0.0

        retrieval_metrics_list = []

        for case in cases:
            start = time.time()

            # In a full implementation, this would call the pipeline
            # For now, we demonstrate the metrics calculation structure
            retrieved_ids = self._simulate_retrieval(case.query)
            latency = (time.time() - start) * 1000

            # Calculate metrics
            r_metrics = self.evaluate_retrieval(retrieved_ids, case.expected_chunk_ids, k)
            retrieval_metrics_list.append(r_metrics)
            all_latencies.append(latency)

            # Store result
            self.results.append(EvalResult(
                query=case.query,
                expected_answer=case.expected_answer,
                retrieval_metrics=r_metrics,
                latency_metrics=LatencyMetrics(total_ms=latency),
            ))

        # Aggregate
        latency_stats = self.calculator.compute_latency_percentiles(all_latencies)

        n = len(retrieval_metrics_list) or 1
        summary = EvalSummary(
            total_cases=len(cases),
            avg_precision_at_k=sum(m.precision_at_k for m in retrieval_metrics_list) / n,
            avg_recall_at_k=sum(m.recall_at_k for m in retrieval_metrics_list) / n,
            avg_mrr=sum(m.mrr for m in retrieval_metrics_list) / n,
            avg_ndcg=sum(m.ndcg_at_k for m in retrieval_metrics_list) / n,
            avg_hit_rate=sum(m.hit_rate for m in retrieval_metrics_list) / n,
            avg_faithfulness=0.0,  # Computed separately with LLM
            latency_p50_ms=latency_stats["p50"],
            latency_p95_ms=latency_stats["p95"],
            latency_p99_ms=latency_stats["p99"],
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            config={"k": k, "model": "gpt-4o-mini"},
        )

        return summary

    def save_results(self, summary: EvalSummary, output_path: str) -> None:
        """Save evaluation results to JSON for tracking over time."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        results_data = {
            "summary": asdict(summary),
            "detailed_results": [asdict(r) for r in self.results],
        }

        with open(output, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

    def _simulate_retrieval(self, query: str) -> list[str]:
        """
        Placeholder for actual retrieval call.

        In production, this calls the hybrid retriever.
        For unit testing the eval framework itself, returns empty.
        """
        return []
