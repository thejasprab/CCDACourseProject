#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SAMPLE="data/sample/arxiv-sample.jsonl"

echo "[check] ensuring sample dataset exists..."
if [[ ! -f "$SAMPLE" ]]; then
  echo "[download] fetching arXiv snapshot + writing sample via KaggleHub..."
  python -m streaming.kaggle_downloader --mode sample --sample-size 50000
fi

echo "[ingest] SAMPLE dataset → Parquet..."
python -m pipelines.ingest_sample

echo "[ml] training TF-IDF model on SAMPLE dataset..."
python -m pipelines.train_sample

echo "[complex] running complex SQL analytics on SAMPLE dataset..."
python -m pipelines.complex_sample

echo "[stream] preparing simulated weekly sample drops..."
python -m streaming.sample_prepare_batches --start-date "$(date +%Y-%m-%d)" --interval-seconds 1 --no-sleep --overwrite

echo "[stream] starting SAMPLE streaming job (Ctrl+C to stop)..."
python -m streaming.sample_stream

echo "[done] SAMPLE pipeline steps executed. Artifacts under:"
echo "  - data/processed/arxiv_sample/"
echo "  - data/models/tfidf_sample/"
echo "  - data/processed/features_sample/"
echo "  - reports/analysis_sample/"
echo "  - reports/streaming_sample/"
