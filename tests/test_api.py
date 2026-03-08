"""
tests/test_api.py

Integration tests for the FastAPI endpoints.

These tests mock the heavy dependencies (embedder, cluster model, FAISS)
so they run fast without requiring the full artifact pipeline.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


def make_mock_app():
    """
    Creates a FastAPI test app with all ML dependencies mocked.
    Allows testing endpoint logic without loading 130MB models.
    """
    from fastapi import FastAPI
    from app.api.routes import router
    from app.services.semantic_cache import SemanticCache
    from app.services.query_engine import QueryEngine, QueryResponse

    app = FastAPI()
    app.include_router(router)

    # Mock query engine that returns predictable responses
    mock_engine = MagicMock()
    mock_engine.handle_query.return_value = QueryResponse(
        query="test query",
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result={"top_documents": [], "cluster_distribution": {0: 0.9}},
        dominant_cluster=0,
        cluster_distribution={0: 0.9}
    )

    cache = SemanticCache(threshold=0.85)

    app.state.query_engine = mock_engine
    app.state.cache = cache

    return app


# ── /query endpoint ───────────────────────────────────────────────────


def test_query_returns_200():
    app = make_mock_app()
    client = TestClient(app)
    response = client.post("/query", json={"query": "future of space exploration"})
    assert response.status_code == 200


def test_query_response_has_required_fields():
    app = make_mock_app()
    client = TestClient(app)
    response = client.post("/query", json={"query": "future of space"})
    data = response.json()

    required_fields = [
        "query", "cache_hit", "matched_query",
        "similarity_score", "result", "dominant_cluster"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_empty_query_returns_400():
    app = make_mock_app()
    client = TestClient(app)
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 400


def test_query_cache_hit_flag_is_boolean():
    app = make_mock_app()
    client = TestClient(app)
    response = client.post("/query", json={"query": "test"})
    data = response.json()
    assert isinstance(data["cache_hit"], bool)


# ── /cache/stats endpoint ─────────────────────────────────────────────


def test_cache_stats_returns_200():
    app = make_mock_app()
    client = TestClient(app)
    response = client.get("/cache/stats")
    assert response.status_code == 200


def test_cache_stats_has_required_fields():
    app = make_mock_app()
    client = TestClient(app)
    response = client.get("/cache/stats")
    data = response.json()

    assert "total_entries" in data
    assert "hit_count" in data
    assert "miss_count" in data
    assert "hit_rate" in data


def test_initial_cache_stats_are_zero():
    app = make_mock_app()
    client = TestClient(app)
    response = client.get("/cache/stats")
    data = response.json()

    assert data["total_entries"] == 0
    assert data["hit_count"] == 0
    assert data["miss_count"] == 0
    assert data["hit_rate"] == 0.0


# ── DELETE /cache endpoint ────────────────────────────────────────────


def test_delete_cache_returns_200():
    app = make_mock_app()
    client = TestClient(app)
    response = client.delete("/cache")
    assert response.status_code == 200


def test_delete_cache_resets_stats():
    app = make_mock_app()
    client = TestClient(app)

    # First clear
    client.delete("/cache")

    # Stats should be zero
    stats = client.get("/cache/stats").json()
    assert stats["total_entries"] == 0
    assert stats["hit_count"] == 0
    assert stats["miss_count"] == 0
