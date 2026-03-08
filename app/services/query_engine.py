"""
query_engine.py

Orchestrates the full query pipeline.

Pipeline:
    embed → detect clusters → cache lookup (top-2 clusters)
         → (hit) return immediately
         → (miss) FAISS search → cross-encoder rerank → store → return

Keeping this logic in a service class (not the route handler) means
the API layer stays thin. Routes handle HTTP. Business logic lives here.

Two-stage retrieval:
    Stage 1 — FAISS bi-encoder search
        Fast. Embeddings pre-computed. Returns top-k candidates.
        Limitation: bi-encoder similarity is a proxy for relevance.

    Stage 2 — Cross-encoder reranking
        Scores each (query, document) pair jointly — richer signal.
        Corrects Stage 1 ranking errors before returning to the user.
        Only runs on 5 candidates so latency cost is minimal.

Multi-cluster cache lookup:
    We get the top-2 clusters for the query and search both partitions.
    This catches cached queries that landed in the second cluster —
    particularly important for boundary documents the assignment
    explicitly calls "the most interesting cases".
"""

import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass

from app.models.embedder import Embedder
from app.models.clustering import ClusterModel
from app.models.reranker import Reranker
from app.services.vector_store import VectorStore
from app.services.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: dict
    dominant_cluster: int
    cluster_distribution: dict


class QueryEngine:
    """Stateless orchestrator — all state lives in injected services."""

    def __init__(
        self,
        embedder: Embedder,
        cluster_model: ClusterModel,
        vector_store: VectorStore,
        cache: SemanticCache,
        reranker: Optional[Reranker] = None
    ):
        self.embedder = embedder
        self.cluster_model = cluster_model
        self.vector_store = vector_store
        self.cache = cache
        self.reranker = reranker

    def handle_query(self, query_text: str) -> QueryResponse:
        """
        Full query pipeline.

        Step 1: Embed query
        Step 2: Get top-2 cluster IDs + full soft distribution
        Step 3: Search cache across both cluster partitions
        Step 4a (hit): Return cached result immediately
        Step 4b (miss): FAISS search → rerank → store → return
        """

        # Step 1: Embed
        query_embedding = self.embedder.encode_query(query_text)

        # Step 2: Cluster detection
        # top_cluster_ids → [dominant, second] for multi-cluster cache search
        top_cluster_ids = self.cluster_model.get_top_n_clusters(query_embedding, n=2)
        dominant_cluster = top_cluster_ids[0]
        cluster_distribution = self.cluster_model.get_top_clusters(query_embedding)

        # Step 3: Cache lookup across top-2 clusters
        cache_result = self.cache.lookup(query_embedding, top_cluster_ids)

        if cache_result is not None:
            cached_entry, similarity_score = cache_result
            logger.info(
                f"Cache HIT | query='{query_text[:50]}' | "
                f"matched='{cached_entry.query_text[:50]}' | "
                f"score={similarity_score:.4f} | clusters={top_cluster_ids}"
            )
            return QueryResponse(
                query=query_text,
                cache_hit=True,
                matched_query=cached_entry.query_text,
                similarity_score=similarity_score,
                result=cached_entry.result,
                dominant_cluster=dominant_cluster,
                cluster_distribution=cluster_distribution
            )

        # Step 4b: Cache miss — FAISS search
        logger.info(f"Cache MISS | query='{query_text[:50]}' | clusters={top_cluster_ids}")

        # Fetch more candidates when reranker is available so it has
        # more to work with. Without reranker, top_k=5 is sufficient.
        top_k = 10 if self.reranker else 5
        search_results = self.vector_store.search(query_embedding, top_k=top_k)

        documents = [
            {
                "doc_id": r.doc_id,
                "text_snippet": r.text_snippet,
                "label": r.label,
                "similarity": r.similarity
            }
            for r in search_results
        ]

        # Cross-encoder reranking: re-score (query, doc) pairs jointly
        # and return only the top-5 after reranking.
        if self.reranker and documents:
            documents = self.reranker.rerank(query_text, documents)
            documents = documents[:5]   # keep top-5 post-rerank
            logger.debug(f"Reranked to top-5 | top_doc_id={documents[0]['doc_id']}")

        result = {
            "top_documents": documents,
            "cluster_distribution": cluster_distribution,
            "reranked": self.reranker is not None
        }

        # Store in cache under the dominant cluster
        self.cache.store(
            query_text=query_text,
            query_embedding=query_embedding,
            dominant_cluster=dominant_cluster,
            result=result
        )

        return QueryResponse(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=result,
            dominant_cluster=dominant_cluster,
            cluster_distribution=cluster_distribution
        )
