"""
vector_store.py

FAISS vector index for semantic document retrieval.

Index type: IndexFlatIP (Inner Product on L2-normalized vectors)

Why IndexFlatIP:
    For L2-normalized vectors, inner product = cosine similarity.
    This is the mathematically correct similarity metric for comparing
    semantic embedding vectors.

Why not IndexIVFFlat (approximate search):
    IVFFlat is faster at large scale (millions of vectors) but introduces
    approximation error — it may miss some true nearest neighbors.
    At 20,000 documents, exact search is fast enough (~5ms per query
    on CPU) and there is no reason to sacrifice accuracy for speed.
    If the corpus scaled to 1M+ documents, IVFFlat would be the right
    choice.

Why not ChromaDB, Weaviate, or Pinecone:
    These are full-featured vector database services. They add operational
    complexity (running a separate server process, API calls) that is
    unnecessary for a self-contained local system. FAISS is a library,
    not a service — it runs in-process with zero overhead.
"""

import numpy as np
import faiss
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Single search result returned by FAISS lookup.
    """
    doc_id: int
    similarity: float
    text_snippet: str   # First 300 chars of document
    label: str          # Original newsgroup label (for context)


class VectorStore:
    """
    Wraps FAISS index with document metadata for human-readable results.

    The index stores embedding vectors. The metadata store maps
    doc_id → {text, label} so we can return readable content.
    """

    def __init__(self):
        logger.info("Loading FAISS index...")
        self.index = self._load_index()
        self.metadata = self._load_metadata()
        logger.info(
            f"VectorStore ready | {self.index.ntotal} documents indexed"
        )

    def _load_index(self) -> faiss.Index:
        if not settings.FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {settings.FAISS_INDEX_PATH}. "
                "Run pipeline/step4_build_index.py first."
            )
        return faiss.read_index(str(settings.FAISS_INDEX_PATH))

    def _load_metadata(self) -> dict:
        if not settings.METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {settings.METADATA_PATH}. "
                "Run pipeline/step4_build_index.py first."
            )
        with open(settings.METADATA_PATH, "r") as f:
            return json.load(f)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None
    ) -> List[SearchResult]:
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        """
        Finds the top_k most semantically similar documents.

        Args:
            query_embedding: shape (384,) — must be L2-normalized
            top_k: number of results to return

        Returns:
            List of SearchResult ordered by similarity descending
        """
        # FAISS expects shape (1, D) for single query
        query_matrix = query_embedding.reshape(1, -1).astype(np.float32)

        # Search returns (distances, indices)
        # For IndexFlatIP on normalized vectors, distances = cosine similarities
        similarities, indices = self.index.search(query_matrix, top_k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for padding when fewer results exist
                continue

            doc = self.metadata.get(str(idx), {})
            text = doc.get("text", "")

            results.append(SearchResult(
                doc_id=int(idx),
                similarity=round(float(sim), 4),
                text_snippet=text[:300] + "..." if len(text) > 300 else text,
                label=doc.get("label", "unknown")
            ))

        return results
