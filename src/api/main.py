"""
FastAPI application entry point.

Configures the application with:
- Security middleware (CORS, rate limiting, auth)
- API routes (documents, query, health)
- Startup validation (fail fast on bad config)
- Structured error handling
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.api.middleware.security_headers import SecurityHeadersMiddleware
from src.api.routes import documents, query, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: startup and shutdown logic.

    Startup:
    - Validate configuration (fail fast)
    - Initialize connections (Qdrant, Redis)
    - Log startup info

    Shutdown:
    - Close connections gracefully
    """
    # Startup
    settings = get_settings()
    print(f"[DocIntel] Starting in {settings.environment} mode")
    print(f"[DocIntel] Qdrant: {settings.qdrant_url}")
    print(f"[DocIntel] Redis: {settings.redis_url}")
    print(f"[DocIntel] Model: {settings.openai_chat_model}")

    yield

    # Shutdown
    print("[DocIntel] Shutting down gracefully")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="DocIntel API",
        description="Production-grade AI Document Intelligence Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
    )

    # --- Middleware (order matters: last added = first executed) ---

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # --- Routes ---
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])

    return app


# Application instance
app = create_app()
