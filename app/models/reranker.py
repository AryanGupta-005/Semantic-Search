"""
reranker.py

Cross-encoder reranker for second-stage result refinement.

Two-stage retrieval architecture:
    Stage 1 — FAISS bi-encoder search (fast, approximate)
        Embeds query and documents independently.
        Returns top-k candidates by cosine similarity.
        Fast because embeddings are pre-computed.
        Weakness: bi-encoder similarity is a proxy, not exact relevance.

    Stage 2 — Cross-encoder reranking (slower, precise)
        Takes (query, document) pairs and scores them jointly.
        The model sees both texts together — much richer signal.
        Reranks the top-k candidates by true relevance.
        Only runs on 5 candidates, so latency cost is minimal.

Why this matters:
    FAISS might return "space shuttle launch" as the top result for
    "NASA budget cuts" because both are in the space cluster and share
    vocabulary. A cross-encoder understands that budget cuts is more
    about economics than launch mechanics — and reranks accordingly.

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking (100k+ query-passage pairs)
    - MiniLM architecture: small, fast, production-appropriate
    - Strong zero-shot generalisation to new domains
    - ~85MB — acceptable for a demo service

Alternative considered: cross-encoder/ms-marco-MiniLM-L-12-v2
    - Deeper (12 layers vs 6), slightly more accurate
    - ~170MB, roughly 2x slower
    - Not worth the tradeoff for a 5-document reranking task
    - L-6 is sufficient at this scale

Why not a larger model (e.g. ms-marco-electra-base)?
    - 435MB, 3-5x slower per pair
    - Gains are marginal on 5 candidates
    - Cross-encoders scale poorly with candidate count; keep k small
"""

import logging
from typing import List, Tuple

from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Wraps a CrossEncoder model for (query, document) relevance scoring.

    Loaded once at startup. Reranks FAISS results before returning
    them to the client — the user always sees the most relevant doc first.
    """

    def __init__(self):
        logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        documents: List[dict]
    ) -> List[dict]:
        """
        Reranks a list of documents by cross-encoder relevance score.

        Args:
            query:     the user's original query string
            documents: list of dicts with keys doc_id, text_snippet, label, similarity

        Returns:
            The same list, sorted by cross-encoder score (descending).
            Each doc gets a new key: rerank_score (float).

        Note:
            cross-encoder scores are raw logits, not probabilities.
            Higher = more relevant. We don't normalise them — the
            ordering is what matters, not the absolute values.
        """
        if not documents:
            return documents

        # Build (query, passage) pairs — one per candidate document
        pairs = [(query, doc["text_snippet"]) for doc in documents]

        # Score all pairs in one forward pass (batched internally)
        scores = self.model.predict(pairs)

        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = round(float(score), 4)

        # Sort by rerank score, highest first
        reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

        logger.debug(
            f"Reranked {len(documents)} docs | "
            f"top score={reranked[0]['rerank_score']:.4f} | "
            f"query='{query[:50]}'"
        )

        return reranked
