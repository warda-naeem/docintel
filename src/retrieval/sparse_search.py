"""
BM25 sparse retrieval.

Implements keyword-based search using the BM25 algorithm.
Complements dense vector search by capturing exact keyword matches
that semantic embeddings might miss (e.g., model numbers, proper nouns).

Design Decision:
- BM25 excels at exact term matching where dense search fails
- Combined with dense search via Reciprocal Rank Fusion (RRF)
- In-memory index rebuilt on startup (acceptable for <100k chunks)
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from src.retrieval.vector_search import SearchResult


@dataclass
class IndexedDocument:
    """A document stored in the BM25 index."""

    chunk_id: str
    content: str
    document_id: str
    page_number: int
    tokens: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SparseSearch:
    """
    BM25-based sparse retrieval for keyword matching.

    BM25 (Best Matching 25) is a probabilistic ranking function that scores
    documents based on term frequency, inverse document frequency, and
    document length normalization.

    Why BM25 alongside dense search:
    - Dense embeddings capture semantic meaning ("car" ≈ "automobile")
    - BM25 captures exact lexical matches ("GPT-4o" must match exactly)
    - Together they cover both semantic and keyword-based queries
    """

    def __init__(self) -> None:
        self._documents: list[IndexedDocument] = []
        self._bm25: BM25Okapi | None = None
        self._is_built = False

    @property
    def index_size(self) -> int:
        """Number of documents in the index."""
        return len(self._documents)

    def add_documents(self, documents: list[IndexedDocument]) -> None:
        """
        Add documents to the index.

        Note: Call build_index() after adding all documents.
        """
        self._documents.extend(documents)
        self._is_built = False

    def add_from_chunks(
        self,
        chunks: list[dict],
    ) -> None:
        """
        Add chunks from a list of dictionaries (e.g., from Qdrant payloads).

        Expected dict keys: chunk_id, content, document_id, page_number, metadata
        """
        for chunk_data in chunks:
            content = chunk_data.get("content", "")
            doc = IndexedDocument(
                chunk_id=chunk_data.get("chunk_id", ""),
                content=content,
                document_id=chunk_data.get("document_id", ""),
                page_number=chunk_data.get("page_number", 0),
                tokens=self._tokenize(content),
                metadata=chunk_data.get("metadata", {}),
            )
            self._documents.append(doc)

        self._is_built = False

    def build_index(self) -> None:
        """
        Build the BM25 index from all added documents.

        Must be called after adding documents and before searching.
        """
        if not self._documents:
            self._bm25 = None
            self._is_built = True
            return

        # Tokenize all documents
        corpus = []
        for doc in self._documents:
            if not doc.tokens:
                doc.tokens = self._tokenize(doc.content)
            corpus.append(doc.tokens)

        self._bm25 = BM25Okapi(corpus)
        self._is_built = True

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search the BM25 index with a text query.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of SearchResult ordered by BM25 score (highest first)
        """
        if not self._is_built:
            self.build_index()

        if not self._bm25 or not self._documents:
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Only include documents with non-zero relevance
                doc = self._documents[idx]
                results.append(SearchResult(
                    chunk_id=doc.chunk_id,
                    content=doc.content,
                    score=float(score),
                    document_id=doc.document_id,
                    page_number=doc.page_number,
                    metadata={**doc.metadata, "retrieval_method": "bm25"},
                ))

        return results

    def remove_document(self, document_id: str) -> int:
        """
        Remove all chunks for a document from the index.

        Returns number of chunks removed.
        Requires rebuild_index() after removal.
        """
        before_count = len(self._documents)
        self._documents = [
            doc for doc in self._documents
            if doc.document_id != document_id
        ]
        removed = before_count - len(self._documents)

        if removed > 0:
            self._is_built = False

        return removed

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Simple whitespace + punctuation tokenization with lowercasing.
        Preserves hyphenated words and numbers.
        """
        # Lowercase and split on non-alphanumeric (keep hyphens in words)
        text = text.lower()
        # Split on whitespace and punctuation, keep alphanumeric and hyphens
        tokens = re.findall(r"[a-z0-9][\w\-]*", text)
        # Filter very short tokens (likely noise)
        return [t for t in tokens if len(t) > 1]
