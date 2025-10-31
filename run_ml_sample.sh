#!/usr/bin/env bash
set -euo pipefail

SPLIT_PARQUET="data/processed/arxiv_sample_split"
MODEL_DIR="data/models/tfidf_sample"
FEATS_OUT="data/processed/features_trained_sample"
OUT_DIR="reports/ml_sample"

echo "==> Step 1: Split"
python scripts/split_sample.py \
  --parquet data/processed/arxiv_parquet \
  --out "${SPLIT_PARQUET}" \
  --test-years 2007,2008,2009 \
  --test-size 1000 \
  --seed 42

echo "==> Step 2: Train & Save"
python scripts/train_ml_sample.py \
  --split-parquet "${SPLIT_PARQUET}" \
  --model-dir "${MODEL_DIR}" \
  --features-out "${FEATS_OUT}" \
  --vocab-size 80000 \
  --min-df 3 \
  --use-bigrams false \
  --extra-stopwords-topdf 200 \
  --seed 42

echo "==> Step 3: Evaluate"
python notebooks/ml_sample_week12.py \
  --mode eval \
  --model-dir "${MODEL_DIR}" \
  --split-parquet "${SPLIT_PARQUET}" \
  --features "${FEATS_OUT}" \
  --out "${OUT_DIR}" \
  --k 3 \
  --strategy exact

echo "Done. Artifacts in ${OUT_DIR}/"
