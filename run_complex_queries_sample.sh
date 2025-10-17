#!/bin/bash
set -euo pipefail

SAMPLE="data/sample/arxiv-sample.jsonl"
OUT="data/processed/arxiv_parquet"

# ---- Spark-friendly settings (smaller than full run) ----
export SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-$(pwd)/data/tmp/spark-local}"
mkdir -p "$SPARK_LOCAL_DIRS"

# Smaller heap is fine for the sample
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-4g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-4g}"

# Apply confs even when launching via `python script.py`
export PYSPARK_SUBMIT_ARGS="\
 --conf spark.sql.session.timeZone=UTC \
 --conf spark.sql.adaptive.enabled=true \
 --conf spark.sql.adaptive.coalescePartitions.enabled=true \
 --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=8m \
 --conf spark.sql.files.maxPartitionBytes=8m \
 --conf spark.sql.shuffle.partitions=256 \
 --conf spark.sql.adaptive.skewJoin.enabled=true \
 --conf spark.sql.adaptive.skewedPartitionThresholdInBytes=64m \
 --conf spark.sql.adaptive.skewedPartitionMaxSplitBytes=16m \
 --conf spark.local.dir=${SPARK_LOCAL_DIRS} \
 --driver-memory ${SPARK_DRIVER_MEMORY} \
 --conf spark.executor.memory=${SPARK_EXECUTOR_MEMORY} \
 pyspark-shell"

echo "[check] ensuring sample JSONL exists..."
if [[ ! -f "$SAMPLE" ]]; then
  echo "[download] fetching arXiv metadata + writing sample via KaggleHub..."
  # Creates data/raw/... and data/sample/arxiv-sample.jsonl
  python scripts/download_arxiv.py --sample 50000
fi

echo "[check] ensuring SAMPLE Parquet exists..."
if [[ ! -d "$OUT" ]] || [[ -z "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
  echo "[ingest] JSONL sample -> Parquet (partitioned by year)…"
  python -m src.ingestion \
    --input "$SAMPLE" \
    --output "$OUT" \
    --partition-by year \
    --repartition 64 \
    --no-stats
fi

echo "[complex] running Week 9 complex queries (SAMPLE)…"
python notebooks/complex_queries_week9.py \
  --parquet "$OUT" \
  --outdir sample_complex_queries

echo "[done] complex query artifacts in reports/sample_complex_queries/ (CSVs + PNGs)."
