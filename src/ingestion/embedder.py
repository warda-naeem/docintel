"""
Embedding generation module.

Generates vector embeddings for document chunks using OpenAI's embedding API.
Includes batching, retry logic, and rate limit handling.

Design Decision:
- Using text-embedding-3-small for good quality/cost ratio
- Abstraction layer allows swapping to local models (e.g., sentence-transformers)
- Batch processing to minimize API calls and respect rate limits
"""

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from openai import OpenAI, RateLimitError, APITimeoutError

from src.config import get_settings
from src.ingestion.chunker import Chunk


@dataclass
class EmbeddedChunk:
    """A chunk with its vector embedding."""

    chunk: Chunk
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def content(self) -> str:
        return self.chunk.content


class Embedder:
    """
    Generates embeddings for text chunks.

    Features:
    - Batch processing (reduces API calls)
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Embedding dimension tracking for consistency
    """

    # OpenAI embedding dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Max batch size for OpenAI embedding API
    MAX_BATCH_SIZE = 100

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout_seconds,
        )
        self.model = settings.openai_embedding_model
        self.dimension = self.MODEL_DIMENSIONS.get(self.model, 1536)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Generate embeddings for a list of chunks.

        Processes in batches to respect API limits.
        Returns EmbeddedChunk objects with vectors attached.
        """
        if not chunks:
            return []

        all_embedded: list[EmbeddedChunk] = []

        # Process in batches
        for i in range(0, len(chunks), self.MAX_BATCH_SIZE):
            batch = chunks[i:i + self.MAX_BATCH_SIZE]
            texts = [chunk.content for chunk in batch]

            embeddings = self._get_embeddings_with_retry(texts)

            for chunk, embedding in zip(batch, embeddings):
                all_embedded.append(EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                    embedding_model=self.model,
                    embedding_dimension=self.dimension,
                ))

        return all_embedded

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query string.

        Used at query time for vector search.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        embeddings = self._get_embeddings_with_retry([query])
        return embeddings[0]

    def _get_embeddings_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]:
        """
        Call OpenAI embedding API with exponential backoff retry.

        Handles:
        - Rate limit errors (429) with backoff
        - Timeout errors with retry
        - Empty text filtering
        """
        # Sanitize inputs: replace empty strings with a space
        sanitized_texts = [t if t.strip() else " " for t in texts]

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=sanitized_texts,
                    model=self.model,
                )

                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]

                # Validate dimensions
                if embeddings and len(embeddings[0]) != self.dimension:
                    raise ValueError(
                        f"Expected dimension {self.dimension}, "
                        f"got {len(embeddings[0])}"
                    )

                return embeddings

            except RateLimitError:
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise

            except APITimeoutError:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Embedding generation failed after all retries")

    def compute_similarity(self, embedding_a: list[float], embedding_b: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Used for semantic cache similarity checking.
        Returns value between -1 and 1 (typically 0 to 1 for normalized embeddings).
        """
        a = np.array(embedding_a)
        b = np.array(embedding_b)

        # Cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
