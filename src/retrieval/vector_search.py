"""
Dense vector search using Qdrant.

Handles:
- Collection creation and management
- Upserting embedded chunks
- Similarity search with filtering
- Batch operations for efficiency
"""

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)

from src.config import get_settings
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import EmbeddedChunk


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    chunk_id: str
    content: str
    score: float
    document_id: str
    page_number: int
    metadata: dict = field(default_factory=dict)


class VectorSearch:
    """
    Dense vector search powered by Qdrant.

    Responsibilities:
    - Manage vector collection lifecycle
    - Index embedded chunks
    - Perform similarity search
    - Support filtering by document_id, page, etc.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            timeout=30,
        )
        self.collection_name = settings.qdrant_collection_name
        self.dimension = 1536  # Default for text-embedding-3-small

    def ensure_collection(self) -> None:
        """
        Create collection if it doesn't exist.

        Uses cosine distance (normalized embeddings from OpenAI).
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def index_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """
        Upsert embedded chunks into the vector store.

        Returns the number of chunks indexed.
        Uses batch upsert for efficiency.
        """
        if not embedded_chunks:
            return 0

        self.ensure_collection()

        # Update dimension from actual embeddings
        if embedded_chunks[0].embedding_dimension != self.dimension:
            self.dimension = embedded_chunks[0].embedding_dimension

        points = []
        for i, ec in enumerate(embedded_chunks):
            point = PointStruct(
                id=i + self._get_collection_size(),
                vector=ec.embedding,
                payload={
                    "chunk_id": ec.chunk.chunk_id,
                    "content": ec.chunk.content,
                    "document_id": ec.chunk.document_id,
                    "page_number": ec.chunk.page_number,
                    "chunk_index": ec.chunk.chunk_index,
                    "token_estimate": ec.chunk.token_estimate,
                    "metadata": ec.chunk.metadata,
                },
            )
            points.append(point)

        # Batch upsert (Qdrant handles batching internally for large sets)
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        document_id: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Perform dense vector similarity search.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            document_id: Optional filter to search within a specific document
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult ordered by relevance (highest first)
        """
        self.ensure_collection()

        # Build filter if document_id specified
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            search_params=SearchParams(
                hnsw_ef=128,  # Higher = more accurate but slower
                exact=False,
            ),
        )

        return [
            SearchResult(
                chunk_id=hit.payload.get("chunk_id", ""),
                content=hit.payload.get("content", ""),
                score=hit.score,
                document_id=hit.payload.get("document_id", ""),
                page_number=hit.payload.get("page_number", 0),
                metadata=hit.payload.get("metadata", {}),
            )
            for hit in results
        ]

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Used when re-ingesting or removing documents.
        Returns approximate number of points deleted.
        """
        self.ensure_collection()

        # Get count before deletion for reporting
        before_count = self._get_collection_size()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

        after_count = self._get_collection_size()
        return before_count - after_count

    def _get_collection_size(self) -> int:
        """Get current number of points in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def health_check(self) -> bool:
        """Check if Qdrant is reachable and collection exists."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
