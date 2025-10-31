#!/usr/bin/env bash
set -euo pipefail

# ===== Full arXiv pipeline (split -> train -> evaluate) =====
# Assumes you already have full Parquet at data/processed/arxiv_full
# Creates: model (data/models/tfidf_full), features (data/processed/features_trained_full),
# and reports (reports/full)

SPLIT_PARQUET="data/processed/arxiv_split"
MODEL_DIR="data/models/tfidf_full"
FEATS_OUT="data/processed/features_trained_full"
OUT_DIR="reports/full"
SRC_PARQUET="data/processed/arxiv_full"

mkdir -p "${OUT_DIR}"

# 1) Split train/test (configurable)
python scripts/split.py \
  --parquet "${SRC_PARQUET}" \
  --out "${SPLIT_PARQUET}" \
  --test-years 2019,2020,2021 \
  --test-size 50000 \
  --seed 42

# 2) Train TF‑IDF model + write normalized features
python scripts/train_ml.py \
  --split-parquet "${SPLIT_PARQUET}" \
  --model-dir "${MODEL_DIR}" \
  --features-out "${FEATS_OUT}" \
  --vocab-size 250000 \
  --min-df 5 \
  --use-bigrams false \
  --extra-stopwords-topdf 500 \
  --seed 42

# 3) Evaluate (Top‑K via block-by-category to avoid full cartesian)
python notebooks/ml_week12.py \
  --mode eval \
  --model-dir "${MODEL_DIR}" \
  --split-parquet "${SPLIT_PARQUET}" \
  --features "${FEATS_OUT}" \
  --out "${OUT_DIR}" \
  --k 5 \
  --strategy block_cat \
  --eval-max-test 20000

# 4) (Optional) Ad-hoc query against TRAIN features
# python notebooks/ml_week12.py \
#   --mode query \
#   --model-dir "${MODEL_DIR}" \
#   --features-train "${FEATS_OUT}/split=train" \
#   --out "${OUT_DIR}" \
#   --query-title "Graph Neural Networks for Molecules" \
#   --query-abstract "We propose a message passing architecture ..." \
#   --k 10

echo "Done. Artifacts in ${OUT_DIR}/"