"""
LLM client abstraction layer.

Provides a clean interface for LLM interactions with:
- Model abstraction (swap providers without changing calling code)
- Token tracking for cost estimation
- Structured output support
- Error handling with graceful degradation
"""

from dataclasses import dataclass

from openai import OpenAI

from src.config import get_settings


@dataclass
class LLMResponse:
    """Structured response from LLM call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost based on token usage (GPT-4o-mini pricing)."""
        # Pricing as of 2024: $0.15/1M input, $0.60/1M output
        input_cost = (self.prompt_tokens / 1_000_000) * 0.15
        output_cost = (self.completion_tokens / 1_000_000) * 0.60
        return input_cost + output_cost


class LLMClient:
    """
    Abstraction layer for LLM interactions.

    Design decisions:
    - Single responsibility: only handles LLM API calls
    - Model-agnostic interface (can swap OpenAI for Anthropic, local, etc.)
    - Token tracking for cost monitoring
    - Temperature control for deterministic outputs in eval
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout_seconds,
        )
        self.model = settings.openai_chat_model
        self.temperature = settings.temperature
        self.max_context_tokens = settings.max_context_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System message setting behavior
            user_prompt: User message with the actual request
            temperature: Override default temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and usage metadata
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "unknown",
        )

    def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Generate a response with retrieved context (RAG pattern).

        Formats context chunks into the prompt and generates an answer.
        This is the primary method for Q&A generation.
        """
        if system_prompt is None:
            system_prompt = self._default_rag_system_prompt()

        # Format context
        formatted_context = self._format_context(context_chunks)

        user_prompt = f"""Context information from retrieved documents:

{formatted_context}

---

Based ONLY on the context above, answer the following question.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."

Question: {query}

Answer:"""

        return self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

    def _format_context(self, chunks: list[str]) -> str:
        """Format context chunks with numbering for citation reference."""
        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            formatted.append(f"[Source {i}]\n{chunk}")
        return "\n\n".join(formatted)

    @staticmethod
    def _default_rag_system_prompt() -> str:
        """Default system prompt for RAG generation."""
        return """You are a precise document analysis assistant. Your role is to answer questions based ONLY on the provided context.

Rules:
1. Only use information from the provided context to answer questions.
2. If the context doesn't contain the answer, explicitly say so.
3. When referencing information, indicate which source it came from (e.g., [Source 1]).
4. Be concise and factual. Do not speculate or add information not in the context.
5. If multiple sources provide conflicting information, note the discrepancy.
6. Structure your answer clearly with proper formatting."""
