#!/bin/bash
set -euo pipefail

RAW="data/raw/arxiv-metadata-oai-snapshot.json"
OUT="data/processed/arxiv_full"

# ---- Spark low-memory friendly settings for BOTH ingestion & EDA ----
export SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-$(pwd)/data/tmp/spark-local}"
mkdir -p "$SPARK_LOCAL_DIRS"

# Give the local driver (and its single executor) more heap; keep tasks small
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-8g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-8g}"

# Ensure these confs are applied even when launching via `python script.py`
export PYSPARK_SUBMIT_ARGS="\
 --conf spark.sql.session.timeZone=UTC \
 --conf spark.sql.adaptive.enabled=true \
 --conf spark.sql.adaptive.coalescePartitions.enabled=true \
 --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=8m \
 --conf spark.sql.files.maxPartitionBytes=8m \
 --conf spark.sql.shuffle.partitions=800 \
 --conf spark.sql.adaptive.skewJoin.enabled=true \
 --conf spark.sql.adaptive.skewedPartitionThresholdInBytes=64m \
 --conf spark.sql.adaptive.skewedPartitionMaxSplitBytes=16m \
 --conf spark.local.dir=${SPARK_LOCAL_DIRS} \
 --driver-memory ${SPARK_DRIVER_MEMORY} \
 --conf spark.executor.memory=${SPARK_EXECUTOR_MEMORY} \
 pyspark-shell"

echo "[check] ensuring raw dataset exists..."
if [[ ! -f "$RAW" ]]; then
  echo "[download] fetching arXiv metadata via KaggleHub..."
  python scripts/download_arxiv.py --sample 0
fi

echo "[ingest] JSON/JSONL -> Parquet (full)…"
python -m src.ingestion \
  --input "$RAW" \
  --output "$OUT" \
  --partition-by year \
  --repartition 200 \
  --no-stats

echo "[eda] generating reports for FULL data…"
python notebooks/eda.py \
  --parquet "$OUT" \
  --topk 30 \
  --abslen-sample-frac 0.02 \
  --outdir full

echo "[done] artifacts in reports/full/ (CSVs + PNGs)."
