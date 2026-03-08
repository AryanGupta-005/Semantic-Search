"""
pipeline/step4_build_index.py

Builds the FAISS vector index from full 384-dim embeddings.
Also saves document metadata for result lookup.

Run:
    python pipeline/step4_build_index.py
"""

import json
import numpy as np
import faiss
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FULL_EMB_PATH = Path("artifacts/embeddings/document_embeddings.npy")
CORPUS_PATH   = Path("data/processed/clean_corpus.json")
FAISS_DIR     = Path("artifacts/faiss")
METADATA_PATH = Path("artifacts/metadata/documents.json")


def main():
    logger.info("=== Step 4: Building FAISS Index ===")

    # Load full embeddings
    embeddings = np.load(FULL_EMB_PATH).astype(np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Load corpus for metadata
    with open(CORPUS_PATH) as f:
        documents = json.load(f)

    # Verify embeddings are L2-normalized (should be from step2)
    # Re-normalize to be safe — FAISS IndexFlatIP requires this for
    # inner product to equal cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    logger.info("Verified L2 normalization")

    # Build FAISS index
    # IndexFlatIP: exact inner product search
    # On L2-normalized vectors, inner product = cosine similarity
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    logger.info(f"FAISS index built | {index.ntotal} vectors | dim={embedding_dim}")

    # Save index
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss_path = FAISS_DIR / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"Saved FAISS index → {faiss_path}")

    # Save metadata: doc_id → {text, label}
    metadata = {
        str(doc["id"]): {
            "text": doc["text"],
            "label": doc["label"]
        }
        for doc in documents
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    logger.info(f"Saved metadata → {METADATA_PATH}")

    # Quick sanity check
    logger.info("Running sanity check...")
    test_query = embeddings[0:1]  # Use first doc as query
    distances, indices = index.search(test_query, 3)
    logger.info(f"Top-3 results for doc[0]: indices={indices[0]}, scores={distances[0]}")
    # First result should be doc 0 itself with score ~1.0
    assert indices[0][0] == 0, "Sanity check failed: first result should be the query itself"
    logger.info("Sanity check passed")

    logger.info("=== Step 4 complete ===")


if __name__ == "__main__":
    main()
