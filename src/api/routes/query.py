"""
Query endpoint — the main RAG pipeline.

Orchestrates the full pipeline:
Query → Cache Check → Retrieval → Re-ranking → Generation → Hallucination Check → Response

This is the core endpoint that demonstrates the full system working together.
"""

import time
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from src.ingestion.embedder import Embedder
from src.retrieval.vector_search import VectorSearch
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse_search import SparseSearch
from src.generation.llm_client import LLMClient
from src.generation.citation_linker import CitationLinker
from src.generation.hallucination_guard import HallucinationGuard
from src.cache.semantic_cache import SemanticCache

router = APIRouter()


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    """Query request body."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer based on ingested documents",
    )
    document_id: str | None = Field(
        None,
        description="Optional: restrict search to a specific document",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of source chunks to retrieve",
    )
    enable_reranking: bool = Field(
        True,
        description="Whether to apply cross-encoder re-ranking",
    )
    enable_hallucination_check: bool = Field(
        True,
        description="Whether to verify answer faithfulness",
    )


class SourceChunk(BaseModel):
    """A source chunk cited in the answer."""
    source_index: int
    chunk_id: str
    document_id: str
    page_number: int
    content_preview: str
    relevance_score: float


class PipelineMetrics(BaseModel):
    """Timing and quality metrics for the pipeline."""
    total_time_ms: float
    retrieval_time_ms: float
    reranking_time_ms: float
    generation_time_ms: float
    hallucination_check_time_ms: float
    cache_hit: bool
    faithfulness_score: float | None
    confidence: float
    token_usage: dict


class QueryResponse(BaseModel):
    """Full query response with answer, sources, and metrics."""
    answer: str
    confidence: str  # HIGH, MEDIUM, LOW
    sources: list[SourceChunk]
    metrics: PipelineMetrics
    is_faithful: bool
    hallucination_rate: float | None


# --- Endpoint ---

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Answer a question using the full RAG pipeline.

    Pipeline stages:
    1. Semantic cache check (skip pipeline if hit)
    2. Hybrid retrieval (dense + sparse + RRF fusion)
    3. Cross-encoder re-ranking
    4. LLM generation with citations
    5. Hallucination verification
    6. Cache storage

    Returns answer with citations, confidence, and full pipeline metrics.
    """
    total_start = time.time()
    metrics = {
        "retrieval_time_ms": 0.0,
        "reranking_time_ms": 0.0,
        "generation_time_ms": 0.0,
        "hallucination_check_time_ms": 0.0,
        "cache_hit": False,
        "token_usage": {},
    }

    # Initialize components
    embedder = Embedder()
    cache = SemanticCache()

    # Step 0: Embed query (needed for both cache and retrieval)
    query_embedding = embedder.embed_query(request.question)

    # Step 1: Cache check
    cache_result = cache.lookup(query_embedding)
    if cache_result.hit:
        metrics["cache_hit"] = True
        total_time = (time.time() - total_start) * 1000

        return QueryResponse(
            answer=cache_result.response or "",
            confidence="HIGH",
            sources=[],  # Cached responses don't include full sources
            metrics=PipelineMetrics(
                total_time_ms=round(total_time, 2),
                retrieval_time_ms=0.0,
                reranking_time_ms=0.0,
                generation_time_ms=0.0,
                hallucination_check_time_ms=0.0,
                cache_hit=True,
                faithfulness_score=None,
                confidence=cache_result.confidence or 0.0,
                token_usage={},
            ),
            is_faithful=True,
            hallucination_rate=None,
        )

    # Step 2: Retrieval
    retrieval_start = time.time()

    vector_search = VectorSearch()
    sparse_search = SparseSearch()
    hybrid = HybridRetriever(vector_search, sparse_search, embedder)

    hybrid_results = hybrid.search(
        query=request.question,
        top_k=request.top_k * 3,  # Over-retrieve for re-ranking
        document_id=request.document_id,
    )

    metrics["retrieval_time_ms"] = (time.time() - retrieval_start) * 1000

    if not hybrid_results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please ingest documents first.",
        )

    # Step 3: Re-ranking
    reranking_start = time.time()

    if request.enable_reranking:
        reranker = Reranker()
        ranked_results = reranker.rerank_batch(
            query=request.question,
            candidates=hybrid_results,
            top_k=request.top_k,
        )
    else:
        # Convert hybrid results to ranked format without re-ranking
        from src.retrieval.reranker import RankedResult
        ranked_results = [
            RankedResult(
                chunk_id=r.chunk_id,
                content=r.content,
                relevance_score=r.score,
                original_score=r.score,
                document_id=r.document_id,
                page_number=r.page_number,
                rank=i + 1,
                metadata=r.metadata,
            )
            for i, r in enumerate(hybrid_results[:request.top_k])
        ]

    metrics["reranking_time_ms"] = (time.time() - reranking_start) * 1000

    # Step 4: Generation
    generation_start = time.time()

    llm = LLMClient()
    context_chunks = [r.content for r in ranked_results]
    llm_response = llm.generate_with_context(
        query=request.question,
        context_chunks=context_chunks,
    )

    metrics["generation_time_ms"] = (time.time() - generation_start) * 1000
    metrics["token_usage"] = {
        "prompt_tokens": llm_response.prompt_tokens,
        "completion_tokens": llm_response.completion_tokens,
        "total_tokens": llm_response.total_tokens,
        "estimated_cost_usd": llm_response.estimated_cost_usd,
    }

    # Link citations
    citation_linker = CitationLinker()
    cited_answer = citation_linker.link_citations(llm_response.content, ranked_results)

    # Step 5: Hallucination check
    hallucination_start = time.time()
    faithfulness_score = None
    hallucination_rate = None
    is_faithful = True

    if request.enable_hallucination_check:
        guard = HallucinationGuard()
        full_context = "\n\n".join(context_chunks)
        faithfulness_result = guard.verify(llm_response.content, full_context)

        faithfulness_score = faithfulness_result.faithfulness_score
        hallucination_rate = faithfulness_result.hallucination_rate
        is_faithful = faithfulness_result.is_faithful

        # If unfaithful and should refuse, use filtered answer
        if not is_faithful and guard.should_refuse(faithfulness_result, cited_answer.confidence):
            answer_text = faithfulness_result.filtered_answer or llm_response.content
        else:
            answer_text = llm_response.content
    else:
        answer_text = llm_response.content

    metrics["hallucination_check_time_ms"] = (time.time() - hallucination_start) * 1000

    # Determine confidence level
    confidence_level = citation_linker.extract_confidence_level(answer_text) or "MEDIUM"
    if cited_answer.has_insufficient_info:
        confidence_level = "LOW"

    # Step 6: Cache the result
    cache.store(
        query=request.question,
        query_embedding=query_embedding,
        response=answer_text,
        citations=[{"chunk_id": c.chunk_id, "source_index": c.source_index} for c in cited_answer.citations],
        confidence=cited_answer.confidence,
    )

    # Build response
    total_time = (time.time() - total_start) * 1000

    sources = [
        SourceChunk(
            source_index=i + 1,
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            page_number=r.page_number,
            content_preview=r.content[:200],
            relevance_score=r.relevance_score,
        )
        for i, r in enumerate(ranked_results)
    ]

    return QueryResponse(
        answer=answer_text,
        confidence=confidence_level,
        sources=sources,
        metrics=PipelineMetrics(
            total_time_ms=round(total_time, 2),
            retrieval_time_ms=round(metrics["retrieval_time_ms"], 2),
            reranking_time_ms=round(metrics["reranking_time_ms"], 2),
            generation_time_ms=round(metrics["generation_time_ms"], 2),
            hallucination_check_time_ms=round(metrics["hallucination_check_time_ms"], 2),
            cache_hit=False,
            faithfulness_score=faithfulness_score,
            confidence=cited_answer.confidence,
            token_usage=metrics["token_usage"],
        ),
        is_faithful=is_faithful,
        hallucination_rate=hallucination_rate,
    )
