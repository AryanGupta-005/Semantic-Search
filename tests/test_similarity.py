"""
tests/test_similarity.py

Tests for the cosine similarity implementation.
Verifies correctness of the custom implementation against known values.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.similarity import cosine_similarity, cosine_similarity_batch


def test_identical_vectors_score_one():
    v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_opposite_vectors_score_minus_one():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    w = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(v, w) == pytest.approx(-1.0, abs=1e-5)


def test_orthogonal_vectors_score_zero():
    v = np.array([1.0, 0.0], dtype=np.float32)
    w = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(v, w) == pytest.approx(0.0, abs=1e-5)


def test_zero_vector_returns_zero():
    v = np.zeros(10, dtype=np.float32)
    w = np.ones(10, dtype=np.float32)
    assert cosine_similarity(v, w) == 0.0


def test_batch_similarity_matches_single():
    """Vectorized batch must return same results as single-pair computation."""
    rng = np.random.default_rng(42)
    query = rng.standard_normal(384).astype(np.float32)
    matrix = rng.standard_normal((20, 384)).astype(np.float32)

    batch_scores = cosine_similarity_batch(query, matrix)
    single_scores = [cosine_similarity(query, matrix[i]) for i in range(len(matrix))]

    np.testing.assert_allclose(batch_scores, single_scores, atol=1e-5)


def test_batch_returns_correct_shape():
    rng = np.random.default_rng(1)
    query = rng.standard_normal(384).astype(np.float32)
    matrix = rng.standard_normal((50, 384)).astype(np.float32)
    result = cosine_similarity_batch(query, matrix)
    assert result.shape == (50,)
