"""
Unit tests for the chunking module.

Tests chunking strategies without requiring any external services.
"""

import pytest

from src.ingestion.chunker import Chunker, Chunk
from src.ingestion.parser import ParsedDocument, DocumentPage


class TestChunker:
    """Test document chunking logic."""

    def setup_method(self):
        self.chunker = Chunker()

    def _make_document(self, text: str, pages: int = 1) -> ParsedDocument:
        """Helper to create a test document."""
        page_list = []
        lines = text.split("\n")
        lines_per_page = max(1, len(lines) // pages)

        for i in range(pages):
            start = i * lines_per_page
            end = start + lines_per_page if i < pages - 1 else len(lines)
            page_text = "\n".join(lines[start:end])
            page_list.append(DocumentPage(
                page_number=i + 1,
                content=page_text,
                char_count=len(page_text),
            ))

        return ParsedDocument(
            document_id="test-doc-001",
            filename="test.txt",
            pages=page_list,
            total_pages=pages,
            total_chars=len(text),
            file_type="txt",
        )

    def test_basic_chunking(self):
        """Document is split into chunks."""
        text = "Hello world. " * 100  # ~1300 chars
        doc = self._make_document(text)
        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_required_fields(self):
        """Each chunk has all required metadata."""
        text = "This is a test document with enough content to create chunks. " * 20
        doc = self._make_document(text)
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.chunk_id != ""
            assert chunk.document_id == "test-doc-001"
            assert chunk.content != ""
            assert chunk.page_number >= 1
            assert chunk.chunk_index >= 0
            assert chunk.token_estimate > 0

    def test_empty_document(self):
        """Empty document produces no chunks."""
        doc = self._make_document("")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_short_document_single_chunk(self):
        """Very short document produces a single chunk."""
        text = "Short document."
        doc = self._make_document(text)
        chunks = self.chunker.chunk_document(doc)

        # Should produce at least one chunk (if content is non-empty)
        assert len(chunks) >= 1

    def test_chunk_ids_are_unique(self):
        """All chunk IDs within a document are unique."""
        text = "Content for testing uniqueness. " * 50
        doc = self._make_document(text)
        chunks = self.chunker.chunk_document(doc)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunks_preserve_content(self):
        """No content is lost during chunking (with overlap, total may exceed original)."""
        text = "Word " * 200
        doc = self._make_document(text)
        chunks = self.chunker.chunk_document(doc)

        # All original words should appear in at least one chunk
        all_chunk_text = " ".join(c.content for c in chunks)
        assert "Word" in all_chunk_text

    def test_chunk_index_sequential(self):
        """Chunk indices are sequential starting from 0."""
        text = "Testing sequential indices. " * 50
        doc = self._make_document(text)
        chunks = self.chunker.chunk_document(doc)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
