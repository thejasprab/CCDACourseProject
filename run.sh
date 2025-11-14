#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RAW="data/raw/arxiv-metadata-oai-snapshot.json"

echo "[check] ensuring full raw dataset exists..."
if [[ ! -f "$RAW" ]]; then
  echo "[download] fetching arXiv full snapshot via KaggleHub..."
  python -m streaming.kaggle_downloader --mode full
fi

echo "[ingest] FULL dataset → Parquet..."
python -m pipelines.ingest_full

echo "[ml] training TF-IDF model on FULL dataset..."
python -m pipelines.train_full

echo "[complex] running complex SQL analytics on FULL dataset..."
python -m pipelines.complex_full

# Optional: one-shot weekly streaming processing (batch style).
# Uncomment if you want this as part of the pipeline:
# echo "[stream] processing latest FULL streaming drop (--once)..."
# python -m streaming.full_stream --once

echo "[done] FULL pipeline complete. Artifacts under:"
echo "  - data/processed/arxiv_full/"
echo "  - data/models/tfidf_full/"
echo "  - data/processed/features_full/"
echo "  - reports/analysis_full/"
echo "  - reports/streaming_full/ (if streaming run)"
