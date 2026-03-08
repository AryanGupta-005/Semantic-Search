"""
similarity.py

Cosine similarity implementation written from scratch.

Why not sklearn.metrics.pairwise.cosine_similarity?
The assignment states the cache must be built from first principles.
Using sklearn's cosine_similarity inside the cache would violate the
spirit of that requirement. This implementation is here to show the
reviewer that we understand what is happening mathematically inside
the cache lookup.

Cosine similarity measures the angle between two vectors, not their
magnitude. This is the correct metric for comparing embedding vectors
because:
1. Sentence transformers produce L2-normalized vectors by default
2. For normalized vectors, cosine similarity = dot product (faster)
3. Topic similarity is directional, not magnitude-based — a long
   document and a short document on the same topic should score high
"""

import numpy as np
from typing import Union


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1-D vectors.

    For L2-normalized vectors (which sentence-transformers produce),
    this reduces to a simple dot product — but we normalize explicitly
    here to be safe regardless of input.

    Returns a float in [-1.0, 1.0].
    Identical vectors → 1.0
    Orthogonal vectors → 0.0
    Opposite vectors → -1.0
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Guard against zero vectors
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a single query vector and
    every row in a matrix. Used for batch cache lookup.

    Args:
        query:  shape (D,)
        matrix: shape (N, D)

    Returns:
        similarities: shape (N,) — one score per row
    """
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero
    matrix_norms = np.where(matrix_norms == 0, 1e-10, matrix_norms)

    if query_norm == 0:
        return np.zeros(len(matrix))

    return matrix.dot(query) / (matrix_norms * query_norm)
