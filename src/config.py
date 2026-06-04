"""
Configuration management with strict validation.

All configuration is loaded from environment variables and validated at startup.
The application fails fast with clear error messages if required config is missing.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation. Loaded from environment variables."""

    # --- OpenAI ---
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for embeddings and chat completions",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model for generation",
    )
    openai_max_retries: int = Field(default=3, ge=1, le=10)
    openai_timeout_seconds: int = Field(default=30, ge=5, le=120)

    # --- Qdrant ---
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: str = Field(default="", description="Qdrant API key (optional for local)")
    qdrant_collection_name: str = Field(default="docintel_documents")

    # --- Redis ---
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_cache_ttl_seconds: int = Field(default=3600, ge=60)
    semantic_cache_similarity_threshold: float = Field(
        default=0.92,
        ge=0.8,
        le=0.99,
        description="Cosine similarity threshold for semantic cache hits",
    )

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_key: str = Field(
        ...,
        description="API key for authenticating requests",
    )
    api_rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
    )

    # --- Ingestion ---
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    allowed_file_types: str = Field(
        default="application/pdf,text/markdown,text/plain,text/html",
    )
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

    # --- Generation ---
    max_context_tokens: int = Field(default=4000, ge=500, le=16000)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    faithfulness_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum faithfulness score to include a claim",
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Below this confidence, respond with 'I don't know'",
    )

    # --- Observability ---
    log_level: str = Field(default="INFO")
    enable_tracing: bool = Field(default=True)
    metrics_enabled: bool = Field(default=True)

    # --- Environment ---
    environment: str = Field(default="development")

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if v == "your-openai-api-key-here" or not v.strip():
            raise ValueError(
                "OPENAI_API_KEY must be set to a valid API key. "
                "Copy .env.example to .env and add your key."
            )
        return v.strip()

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is set and has minimum length."""
        if v == "your-api-key-here" or not v.strip():
            raise ValueError(
                "API_KEY must be set to a secure value. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        if len(v.strip()) < 16:
            raise ValueError("API_KEY must be at least 16 characters for security.")
        return v.strip()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a recognized value."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {valid_levels}")
        return v.upper()

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("QDRANT_URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must start with redis://")
        return v

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def allowed_file_types_list(self) -> list[str]:
        """Parse comma-separated file types into a list."""
        return [ft.strip() for ft in self.allowed_file_types.split(",") if ft.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


def get_settings() -> Settings:
    """
    Load and validate settings from environment.

    Raises ValidationError with clear messages if config is invalid.
    This is called once at startup — fail fast, not at runtime.
    """
    return Settings()
