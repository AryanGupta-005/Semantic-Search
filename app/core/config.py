"""
app/core/config.py

Central configuration for the entire system.

All tunable parameters live here. Every other file imports from this
module instead of hardcoding values. This means:

  - Changing the threshold requires editing one line, not hunting through files
  - Parameters can be overridden via environment variables without touching code
  - A reviewer sees every important parameter in one place

pydantic-settings is used so values can come from:
  1. Environment variables (highest priority)
  2. .env file
  3. Defaults defined here (lowest priority)

Example override via environment:
  CACHE_THRESHOLD=0.80 uvicorn app.main:app --reload

Example override via .env file:
  CACHE_THRESHOLD=0.80
  TOP_K_RESULTS=10
"""

from pathlib import Path
import os

try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):

        # ── Embedding Model ──────────────────────────────────────────
        EMBEDDING_MODEL: str = "all-MiniLM-L12-v2"
        EMBEDDING_DIM: int = 384

        # ── FAISS Search ─────────────────────────────────────────────
        TOP_K_RESULTS: int = 5

        # ── UMAP Reduction ───────────────────────────────────────────
        UMAP_N_COMPONENTS: int = 50
        UMAP_N_NEIGHBORS: int = 15
        UMAP_MIN_DIST: float = 0.1

        # ── Clustering ───────────────────────────────────────────────
        CLUSTER_K_MIN: int = 10
        CLUSTER_K_MAX: int = 31
        CLUSTER_K_STEP: int = 5
        GMM_COVARIANCE_TYPE: str = "full"
        GMM_MAX_ITER: int = 300
        GMM_N_INIT: int = 3

        # ── Semantic Cache ───────────────────────────────────────────
        CACHE_THRESHOLD: float = 0.85

        # ── Preprocessing ────────────────────────────────────────────
        MIN_DOCUMENT_LENGTH: int = 50
        EMBEDDING_BATCH_SIZE: int = 64

        # ── Artifact Paths ───────────────────────────────────────────
        ARTIFACTS_DIR: Path = Path("artifacts")

        @property
        def EMBEDDINGS_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "embeddings" / "document_embeddings.npy"

        @property
        def REDUCED_EMBEDDINGS_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "embeddings" / "reduced_embeddings_50d.npy"

        @property
        def GMM_MODEL_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "gmm_model.pkl"

        @property
        def UMAP_MODEL_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "umap_reducer.pkl"

        @property
        def MEMBERSHIP_MATRIX_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "membership_matrix.npy"

        @property
        def CLUSTER_ANALYSIS_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "cluster_analysis.json"

        @property
        def WARMUP_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "warmup_queries.json"

        @property
        def FAISS_INDEX_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "faiss" / "index.faiss"

        @property
        def METADATA_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "metadata" / "documents.json"

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

except ImportError:
    # Fallback: plain class when pydantic-settings is not installed.
    # Reads from environment variables directly.
    # Install pydantic-settings for full .env file support.
    class Settings:
        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")
        EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 384))
        TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))
        UMAP_N_COMPONENTS: int = int(os.getenv("UMAP_N_COMPONENTS", 50))
        UMAP_N_NEIGHBORS: int = int(os.getenv("UMAP_N_NEIGHBORS", 15))
        UMAP_MIN_DIST: float = float(os.getenv("UMAP_MIN_DIST", 0.1))
        CLUSTER_K_MIN: int = int(os.getenv("CLUSTER_K_MIN", 10))
        CLUSTER_K_MAX: int = int(os.getenv("CLUSTER_K_MAX", 31))
        CLUSTER_K_STEP: int = int(os.getenv("CLUSTER_K_STEP", 5))
        GMM_COVARIANCE_TYPE: str = os.getenv("GMM_COVARIANCE_TYPE", "full")
        GMM_MAX_ITER: int = int(os.getenv("GMM_MAX_ITER", 300))
        GMM_N_INIT: int = int(os.getenv("GMM_N_INIT", 3))
        CACHE_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", 0.85))
        MIN_DOCUMENT_LENGTH: int = int(os.getenv("MIN_DOCUMENT_LENGTH", 50))
        EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 64))
        ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))

        @property
        def EMBEDDINGS_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "embeddings" / "document_embeddings.npy"

        @property
        def REDUCED_EMBEDDINGS_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "embeddings" / "reduced_embeddings_50d.npy"

        @property
        def GMM_MODEL_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "gmm_model.pkl"

        @property
        def UMAP_MODEL_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "umap_reducer.pkl"

        @property
        def MEMBERSHIP_MATRIX_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "membership_matrix.npy"

        @property
        def CLUSTER_ANALYSIS_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "cluster_analysis.json"

        @property
        def WARMUP_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "clusters" / "warmup_queries.json"

        @property
        def FAISS_INDEX_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "faiss" / "index.faiss"

        @property
        def METADATA_PATH(self) -> Path:
            return self.ARTIFACTS_DIR / "metadata" / "documents.json"


# Single instance imported everywhere
settings = Settings()
