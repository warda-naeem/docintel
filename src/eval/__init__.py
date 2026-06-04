"""
Evaluation framework for measuring RAG pipeline quality.

Metrics tracked:
- Retrieval: Precision@k, Recall@k, MRR, NDCG
- Generation: Faithfulness, Answer Relevance, Citation Accuracy
- System: Latency p50/p95/p99, Token Usage, Cost
"""
