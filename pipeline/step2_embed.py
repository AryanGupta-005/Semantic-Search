"""
pipeline/step2_embed.py

Generates sentence embeddings for every document in the cleaned corpus.
Also produces a UMAP-reduced 50-dim version used for clustering.

Two outputs:
  artifacts/embeddings/document_embeddings.npy   shape (N, 384)
  artifacts/embeddings/reduced_embeddings_50d.npy shape (N, 50)

Why two versions:
  Full 384-dim embeddings → used for FAISS vector search
      Searching at full dimension gives the best semantic accuracy.

  50-dim UMAP reduction → used for GMM clustering
      Clustering in 384-dim space suffers from the curse of dimensionality.
      Distances become less meaningful in high-dimensional space, making
      cluster centers harder to define. UMAP to 50 dims preserves the
      semantic structure of the embedding space while making distance
      metrics reliable enough for GMM to work correctly.

      Why 50 dims (not 2, not 100)?
        - 2 dims: fine for visualization, loses too much structure for clustering
        - 50 dims: strong structure preservation, dimensionality low enough
          for GMM covariance estimation to be reliable
        - 100 dims: diminishing returns on structure, still high for GMM

Run:
    python pipeline/step2_embed.py
"""

import json
import numpy as np
import logging
from pathlib import Path
import pickle
import umap

from app.models.embedder import Embedder
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/processed/clean_corpus.json")
ARTIFACTS_DIR = Path("artifacts/embeddings")
UMAP_MODEL_SAVE = Path("artifacts/clusters/umap_reducer.pkl")


def main():
    logger.info("=== Step 2: Embedding Generation ===")

    # Load cleaned corpus
    with open(CORPUS_PATH) as f:
        documents = json.load(f)

    texts = [doc["text"] for doc in documents]
    logger.info(f"Embedding {len(texts)} documents...")

    # Generate full 384-dim embeddings
    embedder = Embedder()
    embeddings = embedder.encode(texts, batch_size=64)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save full embeddings
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    full_path = ARTIFACTS_DIR / "document_embeddings.npy"
    np.save(full_path, embeddings)
    logger.info(f"Saved full embeddings → {full_path}")

    # UMAP reduction to 50 dims for clustering
    logger.info(f"Reducing to {settings.UMAP_N_COMPONENTS} dims with UMAP...")
    logger.info(
        f"UMAP params: n_neighbors={settings.UMAP_N_NEIGHBORS}, "
        f"min_dist={settings.UMAP_MIN_DIST}, n_components={settings.UMAP_N_COMPONENTS}"
    )

    reducer = umap.UMAP(
        n_components=settings.UMAP_N_COMPONENTS,
        n_neighbors=settings.UMAP_N_NEIGHBORS,
        min_dist=settings.UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
        verbose=True
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    # Save reduced embeddings
    reduced_path = ARTIFACTS_DIR / "reduced_embeddings_50d.npy"
    np.save(reduced_path, reduced_embeddings)
    logger.info(f"Saved reduced embeddings → {reduced_path}")

    # Save UMAP reducer model so it can transform new queries at runtime
    UMAP_MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    with open(UMAP_MODEL_SAVE, "wb") as f:
        pickle.dump(reducer, f)
    logger.info(f"Saved UMAP reducer → {UMAP_MODEL_SAVE}")

    logger.info("=== Step 2 complete ===")


if __name__ == "__main__":
    main()
