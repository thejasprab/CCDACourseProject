#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="data/models/tfidf_full"
FEATS_TRAIN="data/processed/features_trained_full/split=train"
OUT_DIR="reports/full"

python notebooks/ml_week12.py \
  --mode query \
  --model-dir "${MODEL_DIR}" \
  --features-train "${FEATS_TRAIN}" \
  --out "${OUT_DIR}" \
  --query-title "Diffusion models for conditional generation" \
  --query-abstract "We introduce a guidance method ..." \
  --k 10