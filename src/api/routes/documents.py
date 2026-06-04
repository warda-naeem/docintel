"""
Document management endpoints.

Handles document ingestion (upload, parse, chunk, embed, index)
and document lifecycle (list, delete, re-ingest).
"""

import hashlib
import time
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from src.ingestion.parser import DocumentParser, ParsedDocument
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.retrieval.vector_search import VectorSearch

router = APIRouter()


# --- Response Models ---

class IngestResponse(BaseModel):
    """Response after document ingestion."""
    document_id: str
    filename: str
    total_chunks: int
    total_pages: int
    processing_time_ms: float
    document_hash: str


class DocumentInfo(BaseModel):
    """Document metadata."""
    document_id: str
    filename: str
    chunk_count: int


class DeleteResponse(BaseModel):
    """Response after document deletion."""
    document_id: str
    chunks_deleted: int
    message: str


# --- Endpoints ---

@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(
    file: Annotated[UploadFile, File(description="PDF or text file to ingest")],
):
    """
    Ingest a document: parse → chunk → embed → index.

    Full pipeline:
    1. Validate file type (PDF, TXT, MD only)
    2. Parse document into pages/sections
    3. Chunk into retrieval-optimized segments
    4. Generate embeddings
    5. Index in vector store

    Returns document metadata and processing stats.
    """
    start_time = time.time()

    # Validate file type
    allowed_types = {
        "application/pdf",
        "text/plain",
        "text/markdown",
    }
    allowed_extensions = {".pdf", ".txt", ".md"}

    filename = file.filename or "unknown"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Read file content
    content = await file.read()

    # Size limit: 10MB
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB.",
        )

    # Generate document hash for cache invalidation
    document_hash = hashlib.sha256(content).hexdigest()[:16]

    # Step 1: Parse
    parser = DocumentParser()
    try:
        parsed = parser.parse(content, filename)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse document: {str(e)}",
        )

    # Step 2: Chunk
    chunker = Chunker()
    chunks = chunker.chunk_document(parsed)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Document produced no chunks. It may be empty or unreadable.",
        )

    # Step 3: Embed
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    # Step 4: Index
    vector_search = VectorSearch()
    indexed_count = vector_search.index_chunks(embedded_chunks)

    processing_time = (time.time() - start_time) * 1000

    return IngestResponse(
        document_id=parsed.document_id,
        filename=filename,
        total_chunks=indexed_count,
        total_pages=parsed.total_pages,
        processing_time_ms=round(processing_time, 2),
        document_hash=document_hash,
    )


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks from the vector store.

    Also invalidates related cache entries.
    """
    vector_search = VectorSearch()
    deleted = vector_search.delete_document(document_id)

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found or already deleted.",
        )

    return DeleteResponse(
        document_id=document_id,
        chunks_deleted=deleted,
        message=f"Successfully deleted {deleted} chunks.",
    )
