"""
Health check endpoints.

Provides liveness and readiness probes for container orchestration,
plus detailed system status for monitoring.
"""

from fastapi import APIRouter

from src.retrieval.vector_search import VectorSearch
from src.cache.semantic_cache import SemanticCache

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic liveness probe.

    Returns 200 if the application is running.
    Used by Docker/K8s for liveness checks.
    """
    return {"status": "healthy", "service": "docintel"}


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe: checks all dependencies.

    Returns 200 only if Qdrant and Redis are reachable.
    Used by load balancers to determine if traffic should be routed here.
    """
    vector_search = VectorSearch()
    cache = SemanticCache()

    qdrant_ok = vector_search.health_check()
    redis_ok = cache.health_check()

    status = "ready" if (qdrant_ok and redis_ok) else "not_ready"
    status_code = 200 if status == "ready" else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "dependencies": {
                "qdrant": "connected" if qdrant_ok else "unreachable",
                "redis": "connected" if redis_ok else "unreachable",
            },
        },
    )


@router.get("/health/details")
async def detailed_health():
    """
    Detailed system status for monitoring dashboards.

    Includes cache stats, vector store info, and configuration.
    """
    vector_search = VectorSearch()
    cache = SemanticCache()

    cache_stats = {}
    try:
        cache_stats = cache.get_stats()
    except Exception:
        cache_stats = {"error": "unable to fetch cache stats"}

    vector_info = {}
    try:
        vector_info = {
            "collection": vector_search.collection_name,
            "size": vector_search._get_collection_size(),
            "connected": vector_search.health_check(),
        }
    except Exception:
        vector_info = {"error": "unable to fetch vector store info"}

    return {
        "status": "healthy",
        "service": "docintel",
        "version": "0.1.0",
        "components": {
            "vector_store": vector_info,
            "cache": cache_stats,
        },
    }
