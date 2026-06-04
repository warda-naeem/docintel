"""
Document parsing module.

Handles extraction of text content from various file formats:
- PDF (with page-level metadata)
- Markdown
- Plain text
- HTML

Security: Validates file types by magic numbers, not just extensions.
"""

import hashlib
import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import magic
from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.config import get_settings


@dataclass
class ParsedPage:
    """A single page/section of extracted content."""

    content: str
    page_number: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete parsed document with metadata."""

    document_id: str
    filename: str
    content_type: str
    pages: list[ParsedPage]
    total_characters: int
    content_hash: str
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all pages into full text."""
        return "\n\n".join(page.content for page in self.pages)


class DocumentParser:
    """
    Parses documents into structured text with metadata.

    Security measures:
    - File type validation via magic numbers (not extension)
    - File size limits enforced
    - No execution of embedded content
    - Sanitization of extracted text
    """

    # Magic number signatures for allowed file types
    MAGIC_SIGNATURES: dict[str, list[bytes]] = {
        "application/pdf": [b"%PDF"],
        "text/plain": [],  # No specific magic number for text
        "text/markdown": [],
        "text/html": [],
    }

    def __init__(self) -> None:
        settings = get_settings()
        self.max_file_size = settings.max_file_size_bytes
        self.allowed_types = settings.allowed_file_types_list

    def validate_file(self, file_content: bytes, filename: str) -> str:
        """
        Validate file type and size.

        Returns the detected MIME type.
        Raises ValueError if validation fails.
        """
        # Check file size
        if len(file_content) > self.max_file_size:
            raise ValueError(
                f"File size ({len(file_content)} bytes) exceeds maximum "
                f"({self.max_file_size} bytes)"
            )

        if len(file_content) == 0:
            raise ValueError("File is empty")

        # Detect MIME type using magic numbers (not extension)
        detected_type = magic.from_buffer(file_content, mime=True)

        # Allow markdown files detected as text/plain
        if detected_type == "text/plain" and filename.endswith((".md", ".markdown")):
            detected_type = "text/markdown"

        if detected_type not in self.allowed_types:
            raise ValueError(
                f"File type '{detected_type}' is not allowed. "
                f"Allowed types: {self.allowed_types}"
            )

        # Additional PDF validation: check magic bytes
        if detected_type == "application/pdf":
            if not file_content[:4] == b"%PDF":
                raise ValueError("File claims to be PDF but has invalid magic bytes")

        return detected_type

    def parse(self, file_content: bytes, filename: str) -> ParsedDocument:
        """
        Parse a document into structured text.

        Args:
            file_content: Raw file bytes
            filename: Original filename (used for type hints, not trusted)

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            ValueError: If file validation fails
        """
        content_type = self.validate_file(file_content, filename)

        # Generate content hash for deduplication and cache invalidation
        content_hash = hashlib.sha256(file_content).hexdigest()

        # Generate deterministic document ID
        document_id = hashlib.sha256(
            f"{filename}:{content_hash}".encode()
        ).hexdigest()[:16]

        # Route to appropriate parser
        if content_type == "application/pdf":
            pages = self._parse_pdf(file_content)
        elif content_type == "text/html":
            pages = self._parse_html(file_content)
        elif content_type == "text/markdown":
            pages = self._parse_markdown(file_content)
        else:
            pages = self._parse_plaintext(file_content)

        # Filter out empty pages
        pages = [p for p in pages if p.content.strip()]

        total_chars = sum(len(p.content) for p in pages)

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            pages=pages,
            total_characters=total_chars,
            content_hash=content_hash,
            metadata={
                "page_count": len(pages),
                "file_size_bytes": len(file_content),
            },
        )

    def _parse_pdf(self, content: bytes) -> list[ParsedPage]:
        """Extract text from PDF with page-level granularity."""
        reader = PdfReader(io.BytesIO(content))
        pages = []

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Sanitize: remove null bytes and excessive whitespace
            text = self._sanitize_text(text)

            if text.strip():
                pages.append(ParsedPage(
                    content=text,
                    page_number=i,
                    metadata={"source_type": "pdf"},
                ))

        return pages

    def _parse_html(self, content: bytes) -> list[ParsedPage]:
        """Extract text from HTML, stripping tags and scripts."""
        text = content.decode("utf-8", errors="replace")

        # Use BeautifulSoup to safely extract text
        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style elements (security: don't process JS)
        for element in soup(["script", "style", "iframe", "object", "embed"]):
            element.decompose()

        extracted = soup.get_text(separator="\n")
        extracted = self._sanitize_text(extracted)

        return [ParsedPage(
            content=extracted,
            page_number=1,
            metadata={"source_type": "html"},
        )]

    def _parse_markdown(self, content: bytes) -> list[ParsedPage]:
        """Parse markdown into sections based on headers."""
        text = content.decode("utf-8", errors="replace")
        text = self._sanitize_text(text)

        # Split by top-level headers for section-aware chunking
        sections = []
        current_section = []
        current_page = 1

        for line in text.split("\n"):
            if line.startswith("# ") and current_section:
                sections.append(ParsedPage(
                    content="\n".join(current_section),
                    page_number=current_page,
                    metadata={"source_type": "markdown"},
                ))
                current_section = [line]
                current_page += 1
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append(ParsedPage(
                content="\n".join(current_section),
                page_number=current_page,
                metadata={"source_type": "markdown"},
            ))

        return sections

    def _parse_plaintext(self, content: bytes) -> list[ParsedPage]:
        """Parse plain text as a single page."""
        text = content.decode("utf-8", errors="replace")
        text = self._sanitize_text(text)

        return [ParsedPage(
            content=text,
            page_number=1,
            metadata={"source_type": "plaintext"},
        )]

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """
        Sanitize extracted text.

        - Remove null bytes
        - Normalize whitespace
        - Remove control characters (except newlines/tabs)
        """
        # Remove null bytes
        text = text.replace("\x00", "")

        # Remove other control characters (keep \n, \t, \r)
        text = "".join(
            char for char in text
            if char in ("\n", "\t", "\r") or (ord(char) >= 32)
        )

        # Normalize excessive newlines (max 2 consecutive)
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text.strip()
