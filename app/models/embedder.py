"""
embedder.py

Text embedding using sentence-transformers.

Model choice: all-MiniLM-L12-v2
- 12-layer MiniLM architecture (vs 6-layer L6 variant)
- 384-dimensional output vectors
- ~130MB model size
- Strong semantic similarity quality on sentence-level tasks
- Runs on CPU without significant performance penalty

Rejected alternatives:
- all-MiniLM-L6-v2: same architecture, fewer layers, slightly lower quality.
  L12 adds marginal cost for measurably better embeddings.
- all-mpnet-base-v2: higher quality but 3x slower and 4x larger. The quality
  gain does not justify the inference cost for a 20k document corpus demo.
- OpenAI text-embedding-ada-002: requires API key, costs per call, cannot
  run offline. Not appropriate for a reproducible local submission.
- TF-IDF: keyword overlap only, no semantic understanding. "NASA missions"
  and "space exploration" would have zero similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps sentence-transformers for both batch and single-query embedding.

    Single instance is created at app startup and reused across all requests.
    Loading the model takes ~2 seconds; inference per query is ~10ms on CPU.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.embedding_dim = settings.EMBEDDING_DIM
        logger.info("Embedding model loaded successfully")

    def encode(self, text: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        if batch_size is None:
            batch_size = settings.EMBEDDING_BATCH_SIZE
        """
        Encodes one or more texts into embedding vectors.

        Args:
            text: single string or list of strings
            batch_size: number of texts to process per forward pass.
                        64 is a good default for CPU — balances memory
                        and throughput.

        Returns:
            np.ndarray of shape (D,) for single text or (N, D) for batch
            where D = 384

        Vectors are L2-normalized by default (normalize_embeddings=True).
        Normalized vectors allow cosine similarity = dot product, which
        is what FAISS IndexFlatIP computes.
        """
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=isinstance(text, list) and len(text) > 100,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
            convert_to_numpy=True
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string. Used at API request time.
        Returns shape (384,).
        """
        return self.encode(query)
