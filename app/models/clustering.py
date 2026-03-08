"""
clustering.py

Fuzzy cluster membership detection using a pre-trained GMM.

Algorithm choice: Gaussian Mixture Model (sklearn)

Why GMM over Fuzzy C-Means (the other standard soft-clustering approach):
1. Soft membership: both produce probability distributions per document,
   satisfying the assignment requirement. GMM's output is a proper
   probability distribution (sums to 1.0) via the EM algorithm.
2. Stability: GMM uses sklearn's well-maintained EM implementation.
   scikit-fuzzy's FCM can be numerically unstable on high-dimensional
   data and its library has not been actively maintained since 2018.
3. Model selection: GMM provides BIC (Bayesian Information Criterion)
   scores out of the box. This gives us a principled, evidence-based
   method to choose the number of clusters — directly addressing the
   assignment requirement to justify K with evidence.
4. Input dimensionality: both FCM and GMM perform poorly in 384-dim
   space due to the curse of dimensionality. We apply UMAP reduction
   to 50 dims before clustering. At 50 dims, GMM's covariance
   estimation is reliable. See pipeline/step3_cluster.py for details.

Why NOT hard clustering (K-Means, DBSCAN):
   Assignment explicitly states hard assignments are unacceptable.
   A document on gun legislation belongs to both politics and firearms
   clusters. Hard clustering cannot represent this.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple
import umap

from app.core.config import settings

logger = logging.getLogger(__name__)


class ClusterModel:
    """
    Loads the pre-trained GMM and UMAP reducer from disk.
    At query time: reduces the query embedding to 50 dims, then
    returns the GMM membership distribution across all clusters.
    """

    def __init__(self):
        logger.info("Loading cluster model and UMAP reducer...")
        self.gmm = self._load_gmm()
        self.umap_reducer = self._load_umap()
        self.n_clusters = self.gmm.n_components
        logger.info(f"Cluster model loaded: {self.n_clusters} clusters")

    def _load_gmm(self):
        if not settings.GMM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"GMM model not found at {settings.GMM_MODEL_PATH}. "
                "Run pipeline/step3_cluster.py first."
            )
        with open(settings.GMM_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    def _load_umap(self):
        if not settings.UMAP_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"UMAP reducer not found at {settings.UMAP_MODEL_PATH}. "
                "Run pipeline/step3_cluster.py first."
            )
        with open(settings.UMAP_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    def get_membership(self, embedding: np.ndarray) -> np.ndarray:
        """
        Returns soft membership distribution for a single embedding.

        Pipeline:
          384-dim embedding
               ↓
          UMAP → 50-dim   (same reducer used during training)
               ↓
          GMM predict_proba → probability per cluster

        Args:
            embedding: shape (384,) — L2-normalized query embedding

        Returns:
            np.ndarray shape (n_clusters,) — probabilities summing to 1.0
        """
        # Reshape for UMAP (expects 2D input)
        emb_2d = embedding.reshape(1, -1)

        # Reduce to 50 dims using the trained UMAP reducer
        reduced = self.umap_reducer.transform(emb_2d)  # shape (1, 50)

        # Get GMM membership probabilities
        membership = self.gmm.predict_proba(reduced)  # shape (1, n_clusters)

        return membership[0]  # shape (n_clusters,)

    def get_dominant_cluster(self, embedding: np.ndarray) -> int:
        """
        Returns the cluster index with highest membership probability.
        Used for cache routing — queries go to their dominant cluster's
        cache partition rather than searching the entire cache.
        """
        membership = self.get_membership(embedding)
        return int(np.argmax(membership))

    def get_top_n_clusters(self, embedding: np.ndarray, n: int = 2) -> list:
        """
        Returns the top-n cluster IDs ordered by membership probability.
        Used for multi-cluster cache search.

        Example return for n=2:
            [5, 2]   (cluster 5 is dominant, cluster 2 is second)
        """
        membership = self.get_membership(embedding)
        top_indices = np.argsort(membership)[::-1][:n]
        return [int(idx) for idx in top_indices]

    def get_top_clusters(self, embedding: np.ndarray, top_n: int = 3) -> Dict[int, float]:
        """
        Returns top-N clusters and their membership probabilities.
        Used in the API response to show semantic distribution.

        Example return:
            {5: 0.72, 2: 0.18, 9: 0.10}
        """
        membership = self.get_membership(embedding)
        top_indices = np.argsort(membership)[::-1][:top_n]
        return {
            int(idx): round(float(membership[idx]), 4)
            for idx in top_indices
            if membership[idx] > 0.01  # Filter out negligible membership
        }
