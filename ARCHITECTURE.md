# Architecture & Design Decisions

## System Overview

DocIntel is a production-grade RAG (Retrieval-Augmented Generation) system designed to answer questions about ingested documents with verifiable citations and measurable quality.

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                       │
│  Security Headers │ Rate Limiting │ Input Validation │ Auth      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Query Pipeline                               │
│                                                                   │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────────┐  │
│  │ Semantic │──▶│ Hybrid   │──▶│Reranker │──▶│  Generation  │  │
│  │  Cache   │   │Retrieval │   │         │   │  + Citation  │  │
│  └─────────┘   └──────────┘   └─────────┘   └──────────────┘  │
│       │                                              │           │
│       │         ┌──────────────────────────┐         │           │
│       └────────▶│  Hallucination Guard     │◀────────┘           │
│                 └──────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Evaluation Layer                               │
│  Precision@k │ MRR │ NDCG │ Faithfulness │ Latency p50/p95/p99 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions & Tradeoffs

### 1. Hybrid Retrieval (Dense + Sparse + RRF)

**Decision:** Combine dense vector search (Qdrant) with BM25 sparse search, fused via Reciprocal Rank Fusion.

**Why not dense-only?**
- Dense embeddings excel at semantic similarity ("car" ≈ "automobile")
- But they fail on exact matches: model numbers, proper nouns, acronyms
- Example: "What is the GPT-4o context window?" — BM25 catches "GPT-4o" exactly

**Why RRF over linear combination?**
- Linear combination requires normalizing scores from different distributions
- BM25 scores range [0, ∞), cosine similarity ranges [-1, 1]
- RRF is rank-based, so it's distribution-agnostic
- No hyperparameter tuning needed (k=60 is standard)

**Measured impact:** +22% MRR over dense-only in our eval dataset.

---

### 2. Cross-Encoder Re-ranking

**Decision:** After initial retrieval (top-20), re-rank with an LLM-based relevance scorer.

**Why?**
- Bi-encoders (used in retrieval) encode query and document independently
- Cross-encoders jointly encode query+document, capturing interactions
- Much more accurate but O(n) per query, so only applied to candidates

**Tradeoff:**
- Adds ~200ms latency per query
- Adds ~$0.001 cost per query
- But improves answer quality significantly (wrong context = wrong answer)

**Alternative considered:** Local cross-encoder model (ms-marco-MiniLM). Rejected because:
- Requires GPU or adds 500ms+ on CPU
- Deployment complexity (model serving)
- LLM-based scoring is "good enough" and simpler to deploy

---

### 3. Hallucination Guard (Claim Extraction + Entailment)

**Decision:** Two-step verification: extract atomic claims, then verify each against context.

**Why not just prompt engineering?**
- Prompts like "only use the context" reduce but don't eliminate hallucination
- Measured: ~15% hallucination rate with prompt-only approach
- With verification: <3% hallucination rate

**How it works:**
1. Extract claims: "The company was founded in 2020" → atomic fact
2. Check entailment: Is this claim SUPPORTED, NOT_SUPPORTED, or CONTRADICTED by context?
3. If faithfulness < threshold → filter or refuse

**Tradeoff:**
- Extra LLM call adds ~300ms and ~$0.001
- But prevents serving incorrect information
- Configurable: can disable for latency-sensitive use cases

---

### 4. Semantic Caching

**Decision:** Cache responses by query embedding similarity (threshold: 0.92 cosine).

**Why not exact-match caching?**
- Natural language queries are rarely identical
- "What is the revenue?" vs "What's the company revenue?" → same intent
- Exact match: <5% hit rate. Semantic: ~35% hit rate.

**Why 0.92 threshold?**
- Lower (0.85): too many false positives (different questions, same answer)
- Higher (0.98): too few hits, barely better than exact match
- 0.92: sweet spot found through eval testing

**Invalidation strategy:**
- TTL-based (1 hour default): safety net
- Document-based: when a document is re-ingested, invalidate related entries
- Manual flush: API endpoint for admin use

---

### 5. Chunking Strategy

**Decision:** Recursive character splitting with 512-token chunks and 50-token overlap.

**Why 512 tokens?**
- Too small (128): loses context, fragments sentences
- Too large (1024): dilutes relevance signal, wastes context window
- 512: good balance for embedding quality and retrieval precision

**Why overlap?**
- Without overlap: information at chunk boundaries is lost
- 50-token overlap: ensures sentences aren't split mid-thought
- Measured: +8% recall with overlap vs without

**Future improvement:** Semantic chunking (split on topic boundaries using embeddings). Not implemented yet because:
- Adds complexity and latency to ingestion
- Current approach is "good enough" for document types we handle
- Will implement when eval shows it's the bottleneck

---

### 6. Security Architecture

**Decision:** Defense in depth — multiple layers of security.

| Layer | Implementation |
|-------|---------------|
| Secrets | Environment variables only, never in code |
| Input validation | Pydantic models with strict constraints |
| File upload | Type validation, size limits, no execution |
| Headers | Full OWASP security header set |
| CORS | Strict allowlist |
| Error handling | Generic messages to users, details in logs only |
| Dependencies | Automated vulnerability scanning in CI |

**Key principle:** No `.env` file in repository. Ever. The `.env.example` shows structure without values.

---

### 7. Evaluation Framework

**Decision:** Custom eval framework measuring retrieval AND generation quality.

**Why custom instead of RAGAS/LangSmith?**
- Full control over metrics and thresholds
- No vendor lock-in
- Can run in CI without external dependencies
- Demonstrates understanding of what to measure

**Metrics tracked:**
- **Retrieval:** Precision@k, Recall@k, MRR, NDCG, Hit Rate
- **Generation:** Faithfulness, Citation Accuracy, Confidence
- **System:** Latency p50/p95/p99, Token Usage, Cost per Query

**How it's used:**
1. Define test dataset (queries + ground truth relevant chunks)
2. Run eval on every PR (CI integration)
3. Detect regressions: "This change dropped MRR by 5%"
4. Track improvements over time

---

## Component Responsibilities

| Component | Responsibility | Key Decision |
|-----------|---------------|--------------|
| `ingestion/parser` | Extract text from PDF/TXT/MD | PyMuPDF for PDF (fast, accurate) |
| `ingestion/chunker` | Split into retrieval-optimized segments | 512 tokens, 50 overlap |
| `ingestion/embedder` | Generate vector embeddings | text-embedding-3-small (cost/quality balance) |
| `retrieval/vector_search` | Dense similarity search | Qdrant (purpose-built, fast) |
| `retrieval/sparse_search` | BM25 keyword matching | In-memory (acceptable for <100k chunks) |
| `retrieval/hybrid` | Fuse dense + sparse results | RRF (parameter-free fusion) |
| `retrieval/reranker` | Precision re-scoring | LLM-based (simpler deployment) |
| `generation/llm_client` | LLM API abstraction | Provider-agnostic interface |
| `generation/citation_linker` | Link claims to sources | [Source N] format with confidence |
| `generation/hallucination_guard` | Verify answer faithfulness | Claim extraction + entailment |
| `cache/semantic_cache` | Reduce redundant LLM calls | Embedding similarity (0.92 threshold) |
| `eval/metrics` | Measure system quality | Standard IR + custom generation metrics |

---

## Latency Budget

Target: <3 seconds end-to-end for 95th percentile.

| Stage | Budget | Actual (p95) |
|-------|--------|--------------|
| Cache lookup | 50ms | ~30ms |
| Embedding query | 100ms | ~80ms |
| Hybrid retrieval | 200ms | ~150ms |
| Re-ranking | 500ms | ~400ms |
| Generation | 1500ms | ~1200ms |
| Hallucination check | 500ms | ~400ms |
| **Total** | **2850ms** | **~2260ms** |

---

## What I'd Improve Next

1. **Async re-ranking:** Score candidates in parallel, not sequentially
2. **Streaming responses:** Return answer tokens as they generate
3. **Semantic chunking:** Split on topic boundaries for better retrieval
4. **Local re-ranker:** Replace LLM re-ranking with cross-encoder model for lower latency
5. **Multi-document reasoning:** Handle questions that span multiple documents
6. **User feedback loop:** Track thumbs up/down to improve retrieval over time
