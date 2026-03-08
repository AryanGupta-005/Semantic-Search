"""
routes.py

FastAPI route definitions.

This file only handles HTTP concerns:
- Parsing request bodies
- Calling the query engine or cache
- Formatting responses

Zero business logic lives here.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: dict
    dominant_cluster: int
    cluster_distribution: Dict[str, float]


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    current_threshold: float
    eviction_count: int


class CacheClearResponse(BaseModel):
    message: str


class AnalyticsResponse(BaseModel):
    per_cluster: Dict[str, Any]
    most_queried_cluster: Optional[int]
    recent_hit_rate: float
    current_threshold: float
    adapt_window: int
    total_evictions: int
    cache_utilisation: str


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search with cache",
    description=(
        "Embeds the query, searches the semantic cache across the top-2 clusters, "
        "and returns a cached result or fresh FAISS result with cross-encoder reranking. "
        "On a miss, the result is stored in the cache before returning."
    )
)
async def query(request: Request, body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    engine = request.app.state.query_engine
    response = engine.handle_query(body.query.strip())

    return QueryResponse(
        query=response.query,
        cache_hit=response.cache_hit,
        matched_query=response.matched_query,
        similarity_score=response.similarity_score,
        result=response.result,
        dominant_cluster=response.dominant_cluster,
        cluster_distribution={str(k): v for k, v in response.cluster_distribution.items()}
    )


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics",
    description=(
        "Returns global cache state: total entries, hit/miss counts, hit rate, "
        "current similarity threshold (may differ from startup if adaptive threshold "
        "has adjusted it), and total evictions."
    )
)
async def cache_stats(request: Request):
    cache = request.app.state.cache
    stats = cache.get_stats()
    return CacheStatsResponse(
        total_entries=stats.total_entries,
        hit_count=stats.hit_count,
        miss_count=stats.miss_count,
        hit_rate=stats.hit_rate,
        current_threshold=stats.current_threshold,
        eviction_count=stats.eviction_count
    )


@router.delete(
    "/cache",
    response_model=CacheClearResponse,
    summary="Flush cache",
    description="Clears all cache entries and resets all statistics including per-cluster analytics."
)
async def clear_cache(request: Request):
    cache = request.app.state.cache
    cache.clear()
    return CacheClearResponse(message="Cache cleared and stats reset")


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Per-cluster cache analytics",
    description=(
        "Returns a detailed breakdown of cache performance per cluster. "
        "Shows which clusters receive the most queries, which have the best hit rates, "
        "how many entries are stored per partition, and the current adaptive threshold state. "
        "Useful for understanding which semantic regions of the corpus are queried most often."
    )
)
async def analytics(request: Request):
    cache = request.app.state.cache
    data = cache.get_analytics()
    return AnalyticsResponse(**data)


@router.post(
    "/cache/threshold",
    summary="Update similarity threshold",
    description=(
        "Manually override the similarity threshold at runtime without restarting the server. "
        "Useful for live experiments. The adaptive threshold system will continue adjusting "
        "from this new starting point."
    )
)
async def set_threshold(request: Request, threshold: float):
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    cache = request.app.state.cache
    cache.set_threshold(threshold)
    return {"message": f"Threshold updated to {threshold}", "new_threshold": threshold}
