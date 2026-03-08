"""
tests/test_cache.py

Unit tests for the semantic cache.

These tests run without loading the embedding model or FAISS index.
They use synthetic embeddings to verify cache logic in isolation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.semantic_cache import SemanticCache, CacheEntry


def make_embedding(seed: int) -> np.ndarray:
    """Creates a normalized random embedding for testing."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def make_similar_embedding(base: np.ndarray, noise: float = 0.05) -> np.ndarray:
    """Creates a slightly perturbed version of base embedding — simulates paraphrase."""
    rng = np.random.default_rng(99)
    perturbed = base + rng.standard_normal(base.shape).astype(np.float32) * noise
    return perturbed / np.linalg.norm(perturbed)


def make_result() -> dict:
    return {"top_documents": [{"doc_id": 1, "text_snippet": "test", "label": "sci.space", "similarity": 0.9}]}


# ── Cache hit / miss ──────────────────────────────────────────────────


def test_empty_cache_returns_miss():
    cache = SemanticCache(threshold=0.85)
    query_emb = make_embedding(1)
    result = cache.lookup(query_emb, dominant_cluster=0)
    assert result is None


def test_exact_match_is_hit():
    cache = SemanticCache(threshold=0.85)
    emb = make_embedding(1)
    cache.store("test query", emb, dominant_cluster=0, result=make_result())

    result = cache.lookup(emb, dominant_cluster=0)
    assert result is not None
    entry, score = result
    assert score > 0.99
    assert entry.query_text == "test query"


def test_similar_query_is_hit():
    cache = SemanticCache(threshold=0.85)
    base_emb = make_embedding(1)
    similar_emb = make_similar_embedding(base_emb, noise=0.02)

    cache.store("original query", base_emb, dominant_cluster=0, result=make_result())
    result = cache.lookup(similar_emb, dominant_cluster=0)

    assert result is not None, "Similar embedding should be a cache hit at low noise"


def test_different_query_is_miss():
    cache = SemanticCache(threshold=0.85)
    emb1 = make_embedding(1)
    emb2 = make_embedding(999)   # Very different seed → very different vector

    cache.store("space query", emb1, dominant_cluster=0, result=make_result())
    result = cache.lookup(emb2, dominant_cluster=0)

    assert result is None, "Very different embedding should be a cache miss"


def test_wrong_cluster_is_miss():
    """
    Verifies cluster partitioning works.
    A query stored in cluster 0 should not be found when looking in cluster 1.
    """
    cache = SemanticCache(threshold=0.85)
    emb = make_embedding(1)

    cache.store("space query", emb, dominant_cluster=0, result=make_result())
    result = cache.lookup(emb, dominant_cluster=1)   # Wrong cluster

    assert result is None, "Entry in cluster 0 should not be found in cluster 1"


# ── Statistics ────────────────────────────────────────────────────────


def test_stats_start_at_zero():
    cache = SemanticCache(threshold=0.85)
    stats = cache.get_stats()
    assert stats.total_entries == 0
    assert stats.hit_count == 0
    assert stats.miss_count == 0
    assert stats.hit_rate == 0.0


def test_stats_update_on_miss():
    cache = SemanticCache(threshold=0.85)
    cache.lookup(make_embedding(1), dominant_cluster=0)
    stats = cache.get_stats()
    assert stats.miss_count == 1
    assert stats.hit_count == 0


def test_stats_update_on_hit():
    cache = SemanticCache(threshold=0.85)
    emb = make_embedding(1)
    cache.store("query", emb, dominant_cluster=0, result=make_result())
    cache.lookup(emb, dominant_cluster=0)
    stats = cache.get_stats()
    assert stats.hit_count == 1
    assert stats.total_entries == 1


def test_hit_rate_calculation():
    cache = SemanticCache(threshold=0.85)
    emb = make_embedding(1)
    cache.store("query", emb, dominant_cluster=0, result=make_result())

    # 1 hit, 1 miss
    cache.lookup(emb, dominant_cluster=0)             # hit
    cache.lookup(make_embedding(999), dominant_cluster=0)  # miss

    stats = cache.get_stats()
    assert stats.hit_rate == pytest.approx(0.5, abs=0.01)


# ── Clear ─────────────────────────────────────────────────────────────


def test_clear_resets_everything():
    cache = SemanticCache(threshold=0.85)
    emb = make_embedding(1)
    cache.store("query", emb, dominant_cluster=0, result=make_result())
    cache.lookup(emb, dominant_cluster=0)

    cache.clear()

    stats = cache.get_stats()
    assert stats.total_entries == 0
    assert stats.hit_count == 0
    assert stats.miss_count == 0

    # After clear, the same embedding should be a miss
    result = cache.lookup(emb, dominant_cluster=0)
    assert result is None


# ── Threshold ─────────────────────────────────────────────────────────


def test_threshold_change_affects_hits():
    """
    Shows that lowering threshold turns a miss into a hit.
    Directly validates the tunable parameter behavior.
    """
    base_emb = make_embedding(1)
    slightly_different = make_similar_embedding(base_emb, noise=0.15)

    # At strict threshold this should miss
    strict_cache = SemanticCache(threshold=0.95)
    strict_cache.store("original", base_emb, dominant_cluster=0, result=make_result())
    strict_result = strict_cache.lookup(slightly_different, dominant_cluster=0)

    # At lenient threshold this should hit
    lenient_cache = SemanticCache(threshold=0.60)
    lenient_cache.store("original", base_emb, dominant_cluster=0, result=make_result())
    lenient_result = lenient_cache.lookup(slightly_different, dominant_cluster=0)

    assert strict_result is None,    "Strict threshold should miss on moderately similar query"
    assert lenient_result is not None, "Lenient threshold should hit on moderately similar query"


def test_set_threshold_updates_behavior():
    cache = SemanticCache(threshold=0.95)
    assert cache.threshold == 0.95
    cache.set_threshold(0.70)
    assert cache.threshold == 0.70


def test_invalid_threshold_raises():
    cache = SemanticCache(threshold=0.85)
    with pytest.raises(ValueError):
        cache.set_threshold(1.5)
    with pytest.raises(ValueError):
        cache.set_threshold(-0.1)
