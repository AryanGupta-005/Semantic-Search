#!/bin/bash
# run_pipeline.sh
# Runs all offline pipeline steps in order.
# Execute once before starting the API server.
#
# Usage:
#   chmod +x pipeline/run_pipeline.sh
#   ./pipeline/run_pipeline.sh

set -e  # Exit immediately if any step fails

echo "================================================"
echo " Semantic Search Pipeline"
echo "================================================"

echo ""
echo "Step 1: Preprocessing dataset..."
python pipeline/step1_preprocess.py

echo ""
echo "Step 2: Generating embeddings + UMAP reduction..."
python pipeline/step2_embed.py

echo ""
echo "Step 3: Fitting GMM clusters..."
python pipeline/step3_cluster.py

echo ""
echo "Step 4: Building FAISS index..."
python pipeline/step4_build_index.py

echo ""
echo "================================================"
echo " Pipeline complete!"
echo " Start the API with:"
echo "   uvicorn app.main:app --reload"
echo "================================================"
