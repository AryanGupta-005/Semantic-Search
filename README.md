# Semantic Search System — Trademarkia AI/ML Internship

A production-grade semantic search engine built over the **20 Newsgroups corpus** (~20,000 documents). The system combines dense vector search, fuzzy clustering, and a custom semantic cache into a single FastAPI service.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │     20 Newsgroups Dataset    │
                        │     ~20,000 Usenet posts     │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │     Text Preprocessing       │
                        │  Remove headers, quotes,     │
                        │  signatures, short docs      │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │     Embedding Model          │
                        │  all-MiniLM-L12-v2 (384-dim) │
                        └──────────────┬──────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
           ┌──────────────────────┐   ┌──────────────────────┐
           │   FAISS Vector Index  │   │   UMAP + GMM         │
           │   19,879 documents    │   │   Fuzzy Clustering   │
           │   Exact cosine search │   │   15 soft clusters   │
           └──────────────────────┘   └──────────────────────┘
                          │                         │
                          └────────────┬────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │     FastAPI Service          │
                        │                              │
                        │  POST /query                 │
                        │    ├── Embed query           │
                        │    ├── Detect top-2 clusters │
                        │    ├── Semantic cache lookup │
                        │    │    (hit) return cached  │
                        │    │    (miss) FAISS search  │
                        │    │         + rerank        │
                        │    └── Store & return        │
                        │                              │
                        │  GET  /cache/stats           │
                        │  GET  /analytics             │
                        │  DELETE /cache               │
                        │  POST /cache/threshold       │
                        └─────────────────────────────┘
```

---

## Design Decisions

### Embedding Model — `all-MiniLM-L12-v2`

**Chosen over:** `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

L12 uses 12 transformer layers versus L6's 6. This produces meaningfully better semantic representations at the cost of ~2x inference time — acceptable for a search system where the bottleneck is FAISS retrieval, not embedding. `all-mpnet-base-v2` is more accurate but at 768 dimensions doubles memory and index size with diminishing returns at this corpus scale.

---

### Dimensionality Reduction — UMAP to 50 dimensions

**Chosen over:** PCA, no reduction

GMM clustering on 384-dimensional vectors suffers from the curse of dimensionality — distances become meaningless in high-dimensional space. UMAP is a non-linear manifold technique that preserves local neighbourhood structure far better than PCA (which is linear and only captures global variance directions). Reducing to 50 dimensions retains semantic structure while giving GMM a tractable, well-shaped input.

---

### Clustering Algorithm — Gaussian Mixture Model (GMM)

**Chosen over:** K-Means, Fuzzy C-Means, DBSCAN

The assignment explicitly requires **soft cluster membership** — a document should belong to multiple clusters with varying degrees. GMM provides this naturally through posterior probabilities. K-Means produces hard assignments. Fuzzy C-Means produces soft memberships but treats all clusters as equal-radius hyperspheres; GMM learns the shape and orientation of each cluster independently, which better models real-world topic overlap.

DBSCAN was rejected because it cannot assign membership probabilities and produces noise points — unusable for a cache routing system.

Number of clusters (15) was selected by minimising BIC across k=5 to k=30. BIC penalises model complexity, so it finds the k that best explains the data without overfitting.

---

### Vector Database — FAISS `IndexFlatIP`

**Chosen over:** HNSWlib, Chroma, Weaviate, FAISS IVF

At 19,879 documents, exact search with `IndexFlatIP` runs in under 5ms per query on CPU. There is no reason to introduce approximation error. HNSWlib uses approximate nearest-neighbour search (HNSW graph), which trades accuracy for speed — a tradeoff that only makes sense at millions of vectors. `IndexFlatIP` on normalised vectors is equivalent to exact cosine similarity.

External services (Chroma, Weaviate) were rejected because they require a running server, which contradicts the single-command startup requirement.

---

### Semantic Cache — Custom Python, Zero Libraries

**Required by assignment.** No Redis, Memcached, or caching libraries.

The cache is built from scratch using Python's `collections.OrderedDict`. Key design decisions:

**Cluster partitioning:**
Instead of scanning all N cached entries on every lookup, the cache is partitioned by dominant cluster. Lookup complexity drops from O(N) to O(N/k) where k=15. With 1000 cached entries and 15 clusters, the average partition search is ~67 entries — 15x faster.

**Multi-cluster search (top-2):**
The assignment specifically calls out boundary documents as "the most interesting cases". A query about gun legislation sits between the politics and firearms clusters. Searching only the dominant cluster misses cached queries that landed in the second cluster. Searching top-2 clusters catches these boundary cases at minimal cost.

**LRU eviction:**
`OrderedDict.move_to_end()` enables O(1) LRU tracking. When the cache reaches 1000 entries, the least recently used entry is evicted. This prevents unbounded memory growth in long-running deployments.

**Adaptive threshold:**
A fixed similarity threshold is a blunt instrument. The system tracks the hit rate over a sliding window of 25 queries. If the recent hit rate falls below 20% (cache too cold or threshold too strict), the threshold loosens by 0.03. If it exceeds 65% (threshold possibly too permissive), it tightens. Bounded between 0.70 and 0.95.

**Cache persistence:**
On server shutdown, the cache is saved to `artifacts/cache/cache_state.json`. On startup, it is restored. This means the cache is warm from the first request of every session, not just after the first few queries warm it up.

---

### Cross-Encoder Reranking — `ms-marco-MiniLM-L-6-v2`

**Chosen over:** no reranking, larger cross-encoders

Two-stage retrieval is standard in production search systems. Stage 1 (FAISS bi-encoder) is fast but approximate — it embeds query and documents independently, so the similarity score is a proxy for relevance. Stage 2 (cross-encoder) scores each (query, document) pair jointly, giving the model full attention over both texts — a much richer relevance signal.

The cross-encoder only runs on the top-10 FAISS candidates, so the latency cost is small. `L-6-v2` (6 layers, ~85MB) was chosen over `L-12-v2` (12 layers, ~170MB) because the accuracy gain on 10 candidates does not justify 2x the inference time.

---

## Project Structure

```
semantic-search-trademarkia/
│
├── app/
│   ├── main.py                  # FastAPI app, startup/shutdown lifecycle
│   ├── api/
│   │   └── routes.py            # Endpoint definitions
│   ├── core/
│   │   └── config.py            # Settings (pydantic-settings)
│   ├── models/
│   │   ├── embedder.py          # SentenceTransformer wrapper
│   │   ├── clustering.py        # GMM + UMAP cluster detection
│   │   └── reranker.py          # CrossEncoder reranker
│   ├── services/
│   │   ├── semantic_cache.py    # Custom cache (no libraries)
│   │   ├── vector_store.py      # FAISS index wrapper
│   │   └── query_engine.py      # Pipeline orchestrator
│   └── utils/
│       ├── preprocessing.py     # Text cleaning
│       └── similarity.py        # Cosine similarity utilities
│
├── pipeline/
│   ├── step1_preprocess.py      # Clean and save corpus
│   ├── step2_embed.py           # Generate and save embeddings
│   ├── step3_cluster.py         # Fit GMM, save membership matrix
│   ├── step4_build_index.py     # Build and save FAISS index
│   └── run_pipeline.sh          # Run all steps in order
│
├── artifacts/
│   ├── embeddings/              # document_embeddings.npy, reduced_embeddings_50d.npy
│   ├── clusters/                # gmm_model.pkl, umap_reducer.pkl, membership_matrix.npy
│   ├── faiss/                   # index.faiss
│   ├── metadata/                # documents.json
│   └── cache/                   # cache_state.json (auto-saved on shutdown)
│
├── experiments/
│   ├── cluster_analysis.ipynb   # BIC/AIC selection, UMAP visualisation, cluster interpretation
│   └── threshold_analysis.ipynb # Threshold vs hit rate / precision tradeoff
│
├── tests/
│   ├── test_cache.py
│   ├── test_similarity.py
│   └── test_api.py
│
├── docker/Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/AryanGupta-005/Semantic-Search
cd semantic-search-trademarkia
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

Download the 20 Newsgroups dataset from:
https://archive.uci.edu/dataset/113/twenty+newsgroups

Extract into:
```
data/raw/20_newsgroups/
```

### 3. Run the pipeline

```bash
# Windows
$env:PYTHONPATH = "."

# macOS / Linux
export PYTHONPATH=.

python pipeline/step1_preprocess.py
python pipeline/step2_embed.py
python pipeline/step3_cluster.py
python pipeline/step4_build_index.py
```

Pipeline takes approximately 25-30 minutes on CPU. Artifacts are saved to `artifacts/`.

### 4. Start the server

```bash
uvicorn app.main:app --reload
```

API available at: http://localhost:8000
Interactive docs: http://localhost:8000/docs

---

## Docker

```bash
docker build -t semantic-search -f docker/Dockerfile .
docker run -p 8000:8000 semantic-search
```

Or with docker-compose:

```bash
docker-compose up
```

---

## API Reference

### `POST /query`

Semantic search with cache lookup and cross-encoder reranking.

**Request:**
```json
{
  "query": "future of space exploration"
}
```

**Response:**
```json
{
  "query": "future of space exploration",
  "cache_hit": true,
  "matched_query": "NASA space missions",
  "similarity_score": 0.9134,
  "result": {
    "top_documents": [...],
    "cluster_distribution": {"5": 0.72, "2": 0.18, "9": 0.10},
    "reranked": true
  },
  "dominant_cluster": 5,
  "cluster_distribution": {"5": 0.72, "2": 0.18, "9": 0.10}
}
```

---

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "current_threshold": 0.85,
  "eviction_count": 0
}
```

---

### `GET /analytics`

Per-cluster breakdown of cache performance.

```json
{
  "per_cluster": {
    "5": {
      "queries": 12,
      "hits": 7,
      "misses": 5,
      "hit_rate": 0.5833,
      "entries_stored": 8
    }
  },
  "most_queried_cluster": 5,
  "recent_hit_rate": 0.48,
  "current_threshold": 0.85,
  "adapt_window": 25,
  "total_evictions": 0,
  "cache_utilisation": "42/1000"
}
```

---

### `DELETE /cache`

Flushes all cache entries and resets all statistics.

---

### `POST /cache/threshold?threshold=0.80`

Updates the similarity threshold at runtime without restarting the server.

---

## Similarity Threshold — Behaviour Analysis

The threshold is the most important tunable parameter in the system.

| Threshold | Behaviour |
|-----------|-----------|
| 0.70 | Aggressive — high hit rate, some false positives |
| 0.80 | Balanced — good hit rate, occasional false positive |
| **0.85** | **Production default — conservative, high confidence hits** |
| 0.90 | Strict — only near-identical phrasing hits |
| 0.95 | Very strict — effectively no cache |

Example: `"gun control laws in america"` vs `"firearms regulation in the united states"` produces a similarity of ~0.79. At threshold 0.85 this is a miss — these are different enough in vocabulary that returning a cached result risks giving the wrong answer. At 0.75 it would hit. The correct value depends on the cost of a false positive versus a false negative in the target application.

The **adaptive threshold** system removes the need to pick one value permanently. It observes the recent hit rate and adjusts automatically within safe bounds.

---

## Cluster Analysis

The GMM model reveals 15 semantic clusters in the corpus, which do not map 1:1 to the original 20 newsgroup labels. Key findings:

- Several newsgroups merge into single clusters (e.g. `sci.electronics` and `comp.sys.hardware` share a cluster — both are hardware discussions)
- Some newsgroups split across clusters (e.g. `talk.politics.misc` appears in a general politics cluster and a gun legislation cluster)
- Boundary documents (e.g. posts about military technology in space programs) receive meaningful non-zero membership in 2-3 clusters simultaneously

See `experiments/cluster_analysis.ipynb` for full visualisation and analysis.

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| API | FastAPI | 0.111.0 |
| Embeddings | sentence-transformers | 2.7.0 |
| Reranking | sentence-transformers CrossEncoder | 2.7.0 |
| Vector search | faiss-cpu | 1.8.0 |
| Clustering | scikit-learn GMM | 1.4.2 |
| Dim reduction | umap-learn | 0.5.6 |
| Serving | uvicorn | 0.29.0 |
| Container | Docker | — |

---

## Dataset

**20 Newsgroups** — ~20,000 Usenet discussion posts across 20 topic categories.
Source: https://archive.uci.edu/dataset/113/twenty+newsgroups

After preprocessing: **19,879 documents retained**, 118 discarded (too short after cleaning).
Noise removed: email headers, quoted reply chains, signatures, routing metadata.
