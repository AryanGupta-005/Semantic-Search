"""
main.py

FastAPI application entry point.

Startup sequence:
1. Load embedding model
2. Load GMM cluster model + UMAP reducer
3. Load FAISS vector index
4. Load cross-encoder reranker
5. Initialize semantic cache
6. Load persisted cache from disk (if exists)
7. Warm cache with representative queries
8. Mount routes

All expensive operations happen once at startup.
Request handlers are fast because they use pre-loaded artifacts.

Start server:
    uvicorn app.main:app --reload
"""

import logging
import json
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.models.embedder import Embedder
from app.models.clustering import ClusterModel
from app.models.reranker import Reranker
from app.services.vector_store import VectorStore
from app.services.semantic_cache import SemanticCache
from app.services.query_engine import QueryEngine
from app.api.routes import router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
CACHE_PERSIST_PATH = "artifacts/cache/cache_state.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifecycle.
    All artifacts are loaded here once, before the first request.
    On shutdown, the cache is saved to disk so the next start is warm.
    """
    logger.info("=== Starting Semantic Search Service ===")

    # 1. Embedding model
    embedder = Embedder()

    # 2. Clustering model
    cluster_model = ClusterModel()

    # 3. FAISS index
    vector_store = VectorStore()

    # 4. Cross-encoder reranker
    # Reranks FAISS candidates by true relevance before returning to user.
    # Falls back gracefully if loading fails (e.g. no internet on first run).
    reranker = None
    try:
        reranker = Reranker()
    except Exception as e:
        logger.warning(f"Reranker failed to load — running without reranking: {e}")

    # 5. Semantic cache
    cache = SemanticCache(threshold=settings.CACHE_THRESHOLD)

    # 6. Load persisted cache from previous session
    # This means the cache is warm from the moment the server starts,
    # not just after the first few queries.
    cache.load(CACHE_PERSIST_PATH)

    # 7. Warm cache with representative queries (if no persisted state)
    if cache.get_stats().total_entries == 0 and settings.WARMUP_PATH.exists():
        with open(settings.WARMUP_PATH) as f:
            warmup_data = json.load(f)
        warmup_entries = []
        for item in warmup_data:
            embedding = np.array(item["embedding"], dtype=np.float32)
            warmup_entries.append({
                "query_text": item["query_text"],
                "query_embedding": embedding,
                "dominant_cluster": item["dominant_cluster"],
                "result": item["result"]
            })
        cache.warm(warmup_entries)
        logger.info(f"Cache warmed with {len(warmup_entries)} entries")
    else:
        logger.info("Skipping warmup — cache already populated from disk or no warmup file")

    # 8. Build query engine
    query_engine = QueryEngine(
        embedder=embedder,
        cluster_model=cluster_model,
        vector_store=vector_store,
        cache=cache,
        reranker=reranker
    )

    # Attach to app state so routes can access
    app.state.embedder = embedder
    app.state.cluster_model = cluster_model
    app.state.vector_store = vector_store
    app.state.cache = cache
    app.state.query_engine = query_engine

    logger.info("=== Service ready ===")

    yield

    # Shutdown — save cache to disk for next session
    logger.info("Saving cache to disk before shutdown...")
    cache.save(CACHE_PERSIST_PATH)
    logger.info("=== Shutdown complete ===")


app = FastAPI(
    title="Semantic Search System",
    description=(
        "Lightweight semantic search over the **20 Newsgroups** corpus (~20k documents).\n\n"
        "### Features\n"
        "- 🔍 **Semantic search** via FAISS vector index (exact cosine similarity)\n"
        "- 🧠 **Fuzzy GMM clustering** with soft document membership\n"
        "- ⚡ **Cluster-partitioned semantic cache** — O(N/k) lookup, LRU eviction\n"
        "- 🔁 **Multi-cluster search** — searches top-2 clusters, handles boundary documents\n"
        "- 📈 **Adaptive threshold** — auto-adjusts based on recent hit rate\n"
        "- 🎯 **Cross-encoder reranking** — reranks FAISS results by true relevance\n"
        "- 💾 **Cache persistence** — survives server restarts\n"
        "- 📊 **Analytics endpoint** — per-cluster hit rate breakdown\n"
        "- 🐳 **Dockerized** — starts with a single command\n\n"
        "### How the cache works\n"
        "Queries are embedded and compared against cached entries within their **top-2 dominant clusters**. "
        "A hit occurs when cosine similarity exceeds the threshold (default `0.85`, adaptive). "
        "Semantically identical queries return instantly without re-querying FAISS.\n\n"
        "Built for **Trademarkia AI/ML Internship** — Aryan"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url="/redoc",
    contact={"name": "Aryan"},
    license_info={"name": "MIT"},
)

app.include_router(router)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    from fastapi.openapi.docs import get_swagger_ui_html
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Semantic Search API — Trademarkia",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={
            "docExpansion": "list",
            "defaultModelsExpandDepth": 2,
            "syntaxHighlight.theme": "monokai",
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "operationsSorter": "method",
        },
    )

    custom_css = """
    <style>
      body, .swagger-ui { background: #0f1117 !important; color: #e2e8f0 !important; }
      .swagger-ui .topbar { background: #1a1d2e !important; border-bottom: 2px solid #6366f1; }
      .swagger-ui .topbar .download-url-wrapper { display: none; }
      .swagger-ui .info { margin: 30px 0; }
      .swagger-ui .info .title { color: #a5b4fc !important; font-size: 2rem !important; font-weight: 700; }
      .swagger-ui .info p, .swagger-ui .info li { color: #cbd5e1 !important; }
      .swagger-ui .info h3 { color: #818cf8 !important; }
      .swagger-ui .info code { background: #1e293b; color: #f472b6; padding: 2px 6px; border-radius: 4px; }
      .swagger-ui .opblock { background: #1a1d2e !important; border: 1px solid #2d3148 !important; border-radius: 8px !important; margin-bottom: 12px; }
      .swagger-ui .opblock-summary { border-radius: 8px !important; }
      .swagger-ui .opblock.opblock-post { border-left: 4px solid #6366f1 !important; }
      .swagger-ui .opblock.opblock-get  { border-left: 4px solid #10b981 !important; }
      .swagger-ui .opblock.opblock-delete { border-left: 4px solid #f43f5e !important; }
      .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #6366f1 !important; border-radius: 4px; }
      .swagger-ui .opblock.opblock-get  .opblock-summary-method { background: #10b981 !important; border-radius: 4px; }
      .swagger-ui .opblock.opblock-delete .opblock-summary-method { background: #f43f5e !important; border-radius: 4px; }
      .swagger-ui .opblock-tag { color: #a5b4fc !important; border-bottom: 1px solid #2d3148 !important; font-size: 1.1rem !important; }
      .swagger-ui label, .swagger-ui .parameter__name, .swagger-ui table thead tr th,
      .swagger-ui .response-col_status { color: #94a3b8 !important; }
      .swagger-ui .opblock-summary-description, .swagger-ui .opblock-summary-path { color: #e2e8f0 !important; }
      .swagger-ui textarea, .swagger-ui input[type=text], .swagger-ui select {
        background: #0f1117 !important; color: #e2e8f0 !important;
        border: 1px solid #334155 !important; border-radius: 6px !important; }
      .swagger-ui .btn.execute { background: #6366f1 !important; border-color: #6366f1 !important;
                                  color: #fff !important; border-radius: 6px !important;
                                  font-weight: 600 !important; letter-spacing: 0.5px; }
      .swagger-ui .btn.execute:hover { background: #4f46e5 !important; }
      .swagger-ui .responses-inner { background: #1a1d2e !important; border-radius: 6px; }
      .swagger-ui .response-col_description { color: #cbd5e1 !important; }
      .swagger-ui .microlight { background: #0d1117 !important; border-radius: 6px; }
      .swagger-ui section.models { background: #1a1d2e !important; border: 1px solid #2d3148 !important; border-radius: 8px !important; }
      .swagger-ui section.models h4 { color: #a5b4fc !important; }
      .swagger-ui .model-title { color: #818cf8 !important; }
      .swagger-ui .model { color: #94a3b8 !important; }
      ::-webkit-scrollbar { width: 6px; height: 6px; }
      ::-webkit-scrollbar-track { background: #1a1d2e; }
      ::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 3px; }
    </style>
    """

    modified = html.body.decode("utf-8").replace("</head>", custom_css + "</head>")
    return HTMLResponse(modified)
