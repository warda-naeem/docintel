"""
Citation linking module.

Links generated answer claims back to source chunks, providing
verifiable references for every statement in the response.

Design Decision:
- Every claim in the answer should be traceable to a source chunk
- Citations use [Source N] format matching the context numbering
- Enables users to verify answers against original documents
- Supports confidence scoring based on citation coverage
"""

import re
from dataclasses import dataclass, field

from src.retrieval.reranker import RankedResult


@dataclass
class Citation:
    """A citation linking answer content to a source chunk."""

    source_index: int  # 1-indexed, matches [Source N] in answer
    chunk_id: str
    document_id: str
    page_number: int
    relevance_score: float
    content_preview: str  # First 200 chars of the source chunk

    @property
    def reference_tag(self) -> str:
        """The citation tag as it appears in the answer."""
        return f"[Source {self.source_index}]"


@dataclass
class CitedAnswer:
    """An answer with linked citations and confidence metadata."""

    answer: str
    citations: list[Citation]
    confidence: float  # 0.0 to 1.0 based on citation coverage
    cited_source_count: int
    total_source_count: int
    has_insufficient_info: bool = False

    @property
    def citation_coverage(self) -> float:
        """Fraction of provided sources that were actually cited."""
        if self.total_source_count == 0:
            return 0.0
        return self.cited_source_count / self.total_source_count


class CitationLinker:
    """
    Links generated answers to source documents.

    Responsibilities:
    - Parse [Source N] references from generated text
    - Map references to actual source chunks
    - Calculate confidence based on citation patterns
    - Detect "I don't know" responses
    """

    # Patterns that indicate insufficient information
    INSUFFICIENT_INFO_PATTERNS = [
        r"i don'?t have enough information",
        r"cannot.*answer.*based on",
        r"the context doesn'?t contain",
        r"no.*information.*available",
        r"not.*mentioned.*in.*context",
        r"cannot.*fully answer",
    ]

    def link_citations(
        self,
        answer: str,
        source_chunks: list[RankedResult],
    ) -> CitedAnswer:
        """
        Parse citations from answer and link to source chunks.

        Args:
            answer: Generated answer text (may contain [Source N] references)
            source_chunks: The ranked chunks that were provided as context

        Returns:
            CitedAnswer with linked citations and confidence score
        """
        # Check if answer indicates insufficient information
        has_insufficient_info = self._detect_insufficient_info(answer)

        # Extract cited source indices from the answer
        cited_indices = self._extract_citation_indices(answer)

        # Build citation objects
        citations = []
        for idx in cited_indices:
            if 1 <= idx <= len(source_chunks):
                chunk = source_chunks[idx - 1]
                citations.append(Citation(
                    source_index=idx,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    relevance_score=chunk.relevance_score,
                    content_preview=chunk.content[:200],
                ))

        # Calculate confidence
        confidence = self._calculate_confidence(
            answer=answer,
            citations=citations,
            total_sources=len(source_chunks),
            has_insufficient_info=has_insufficient_info,
        )

        return CitedAnswer(
            answer=answer,
            citations=citations,
            confidence=confidence,
            cited_source_count=len(citations),
            total_source_count=len(source_chunks),
            has_insufficient_info=has_insufficient_info,
        )

    def _extract_citation_indices(self, text: str) -> list[int]:
        """Extract all [Source N] indices from text."""
        pattern = r"\[Source\s+(\d+)\]"
        matches = re.findall(pattern, text, re.IGNORECASE)
        # Return unique indices in order of appearance
        seen = set()
        indices = []
        for match in matches:
            idx = int(match)
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices

    def _detect_insufficient_info(self, answer: str) -> bool:
        """Check if the answer indicates insufficient information."""
        answer_lower = answer.lower()
        return any(
            re.search(pattern, answer_lower)
            for pattern in self.INSUFFICIENT_INFO_PATTERNS
        )

    def _calculate_confidence(
        self,
        answer: str,
        citations: list[Citation],
        total_sources: int,
        has_insufficient_info: bool,
    ) -> float:
        """
        Calculate confidence score for the answer.

        Factors:
        - Number of citations (more = higher confidence)
        - Relevance scores of cited sources
        - Whether the answer admits uncertainty
        - Answer length (very short answers may indicate low confidence)
        """
        if has_insufficient_info:
            return 0.3  # Low confidence when model admits uncertainty

        if not citations:
            # No citations but also no admission of uncertainty
            # This might be a hallucination — flag as medium-low
            return 0.4

        # Base confidence from citation count
        citation_ratio = min(len(citations) / max(total_sources, 1), 1.0)

        # Average relevance of cited sources
        avg_relevance = sum(c.relevance_score for c in citations) / len(citations)

        # Weighted combination
        confidence = (0.4 * citation_ratio) + (0.6 * avg_relevance)

        # Boost if multiple sources agree (more citations = more confidence)
        if len(citations) >= 3:
            confidence = min(confidence + 0.1, 1.0)

        return round(confidence, 3)

    @staticmethod
    def extract_confidence_level(answer: str) -> str | None:
        """
        Extract explicit confidence level from answer (HIGH/MEDIUM/LOW).

        Used with RAG_ANSWER_V2 prompt that asks for confidence assessment.
        """
        # Look for confidence indicator at the end of the answer
        pattern = r"(?:confidence|level)[:\s]*(HIGH|MEDIUM|LOW)"
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Also check for standalone HIGH/MEDIUM/LOW at the end
        last_line = answer.strip().split("\n")[-1].strip()
        if last_line.upper() in ("HIGH", "MEDIUM", "LOW"):
            return last_line.upper()

        return None
