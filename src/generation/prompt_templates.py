"""
Versioned prompt templates for generation.

Centralizes all prompts used in the system for:
- Easy iteration and A/B testing
- Version tracking (which prompt produced which eval results)
- Clear separation of concerns

Design Decision:
- Prompts are code, not config — they need version control and testing
- Each prompt has a version string for tracking in eval results
"""

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A versioned prompt template."""

    name: str
    version: str
    system_prompt: str
    user_template: str
    description: str = ""


# =============================================================================
# RAG Generation Prompts
# =============================================================================

RAG_ANSWER_V1 = PromptTemplate(
    name="rag_answer",
    version="v1.0",
    description="Standard RAG answer generation with citation support",
    system_prompt="""You are a precise document analysis assistant. Your role is to answer questions based ONLY on the provided context.

Rules:
1. Only use information from the provided context to answer questions.
2. If the context doesn't contain the answer, explicitly say "I don't have enough information to answer this question based on the available documents."
3. When referencing information, indicate which source it came from using [Source N] notation.
4. Be concise and factual. Do not speculate or add information not in the context.
5. If multiple sources provide conflicting information, note the discrepancy.
6. Structure your answer clearly with proper formatting.""",
    user_template="""Context information from retrieved documents:

{context}

---

Based ONLY on the context above, answer the following question.

Question: {query}

Answer:""",
)


RAG_ANSWER_V2 = PromptTemplate(
    name="rag_answer",
    version="v2.0",
    description="Enhanced RAG with confidence signaling and structured output",
    system_prompt="""You are a precise document analysis assistant. Answer questions using ONLY the provided context.

Output format:
1. Provide your answer with [Source N] citations inline.
2. End with a confidence assessment: HIGH (directly stated in sources), MEDIUM (inferred from sources), or LOW (partially supported).

Rules:
- NEVER fabricate information not in the context.
- If unsure, say "Based on the available context, I cannot fully answer this question."
- Prefer direct quotes when the exact wording matters.
- Note any contradictions between sources.""",
    user_template="""Context from retrieved documents:

{context}

---

Question: {query}

Provide your answer with inline citations [Source N] and end with your confidence level (HIGH/MEDIUM/LOW):""",
)


# =============================================================================
# Hallucination Detection Prompts
# =============================================================================

CLAIM_EXTRACTION = PromptTemplate(
    name="claim_extraction",
    version="v1.0",
    description="Extract atomic claims from a generated answer",
    system_prompt="""You are a claim extraction system. Break down the given text into atomic, verifiable claims.

Rules:
- Each claim should be a single, self-contained factual statement.
- Remove opinions, hedging language, and meta-commentary.
- Keep claims specific and verifiable against source documents.
- Return one claim per line, numbered.""",
    user_template="""Extract all atomic factual claims from the following answer:

Answer: {answer}

Claims (one per line, numbered):""",
)


FAITHFULNESS_CHECK = PromptTemplate(
    name="faithfulness_check",
    version="v1.0",
    description="Verify if a claim is supported by the context",
    system_prompt="""You are a faithfulness verification system. Determine if a claim is supported by the given context.

For each claim, respond with EXACTLY one of:
- SUPPORTED: The claim is directly stated or clearly implied by the context.
- NOT_SUPPORTED: The claim cannot be verified from the context.
- CONTRADICTED: The context directly contradicts this claim.""",
    user_template="""Context:
{context}

---

Claim: {claim}

Verdict (SUPPORTED/NOT_SUPPORTED/CONTRADICTED):""",
)


# =============================================================================
# Re-ranking Prompts
# =============================================================================

RELEVANCE_SCORING = PromptTemplate(
    name="relevance_scoring",
    version="v1.0",
    description="Score passage relevance to a query",
    system_prompt="You are a relevance scoring system. Return only a decimal number between 0.0 and 1.0.",
    user_template="""Rate the relevance of the following passage to the query.
Return ONLY a number between 0.0 and 1.0, where:
- 0.0 = completely irrelevant
- 0.5 = somewhat relevant  
- 1.0 = highly relevant and directly answers the query

Query: {query}

Passage: {passage}

Relevance score:""",
)


# =============================================================================
# Prompt Registry
# =============================================================================

# Default prompts used by the system
ACTIVE_PROMPTS = {
    "rag_answer": RAG_ANSWER_V2,  # Currently using v2
    "claim_extraction": CLAIM_EXTRACTION,
    "faithfulness_check": FAITHFULNESS_CHECK,
    "relevance_scoring": RELEVANCE_SCORING,
}


def get_prompt(name: str) -> PromptTemplate:
    """Get the active prompt template by name."""
    if name not in ACTIVE_PROMPTS:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(ACTIVE_PROMPTS.keys())}")
    return ACTIVE_PROMPTS[name]


def format_prompt(name: str, **kwargs: str) -> tuple[str, str]:
    """
    Get and format a prompt template.

    Returns (system_prompt, formatted_user_prompt) tuple.
    """
    template = get_prompt(name)
    user_prompt = template.user_template.format(**kwargs)
    return template.system_prompt, user_prompt
