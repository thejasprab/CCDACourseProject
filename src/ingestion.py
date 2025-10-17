# src/ingestion.py
"""
OOM-resistant ingestion for Kaggle arXiv metadata → Parquet.

Strategies:
  - Align the big shuffle with the output partition column (e.g., 'year')
  - Use many small partitions to minimize per-task memory
  - Parquet writer tuned for small in-memory buffers (block/page) and fewer rows per file
"""

import argparse
from pyspark.sql import functions as F
from src.utils import get_spark
from src.transformations import transform_all


def read_arxiv_json(spark, path: str, multiline: bool = False):
    """Read JSON/JSONL. Kaggle's arxiv snapshot is JSONL (so usually multiline=False)."""
    return (
        spark.read
        .option("multiLine", "true" if multiline else "false")
        .json(path)
    )


def main():
    ap = argparse.ArgumentParser(description="Ingest Kaggle arXiv metadata into Parquet (OOM-resistant).")
    ap.add_argument("--input", required=True, help="Path or glob to arxiv metadata JSON/JSONL (local).")
    ap.add_argument("--output", required=True, help="Directory to write processed Parquet.")
    ap.add_argument("--multiline", action="store_true", help="Input is pretty-printed multi-line JSON.")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit rows for quick demo.")
    ap.add_argument("--sample-frac", type=float, default=0.0, help="Optional: sample fraction (0<frac<=1).")
    ap.add_argument("--repartition", type=int, default=0, help="Target partitions per key before write (default 512).")
    ap.add_argument("--min-abstract-len", type=int, default=40, help="Filter very short abstracts.")
    ap.add_argument("--partition-by", default="year",
                    choices=["year", "primary_category", "none"],
                    help="Partition column for Parquet (or 'none').")
    ap.add_argument("--no-stats", action="store_true",
                    help="Skip post-write stats to minimize extra jobs.")
    args = ap.parse_args()

    spark = get_spark("arxiv_week8_ingestion_lowmem")

    # --- Read raw ---
    df_raw = read_arxiv_json(spark, args.input, multiline=args.multiline)

    # Optional sampling/limiting to validate pipeline quickly
    if args.sample_frac and 0.0 < args.sample_frac <= 1.0:
        df_raw = df_raw.sample(False, args.sample_frac, seed=42)
    if args.limit and args.limit > 0:
        df_raw = df_raw.limit(args.limit)

    # --- Transform & quality filters ---
    df = transform_all(df_raw, min_abstract_len=args.min_abstract_len)

    # --- Partition-aware shuffle alignment (critical to avoid a second huge shuffle) ---
    if args.partition_by in ("year", "primary_category"):
        target = args.repartition if args.repartition and args.repartition > 0 else 512
        df = df.repartition(target, F.col(args.partition_by))
    else:
        # No partitioned write: honor numeric repartition if provided
        if args.repartition and args.repartition > 0:
            df = df.repartition(args.repartition)

    # --- Write with small Parquet buffers to reduce writer heap usage ---
    writer = (
        df.write
          .mode("overwrite")
          .option("compression", "zstd")                         # can switch to 'snappy' if preferred
          .option("parquet.block.size", 8 * 1024 * 1024)         # 8 MB blocks
          .option("parquet.page.size", 512 * 1024)               # 512 KB pages
          .option("parquet.enable.dictionary", "true")
          .option("maxRecordsPerFile", 50000)                    # cap rows per file
    )
    if args.partition_by != "none":
        writer = writer.partitionBy(args.partition_by)

    writer.parquet(args.output)

    if not args.no_stats:
        # Lightweight-ish stats (aggregations only). Skip with --no-stats if you want zero extra jobs.
        n = df.count()
        print(f"[OK] Wrote {n} rows to {args.output}")

        try:
            top_cats = (
                df.groupBy("primary_category")
                  .count()
                  .orderBy(F.desc("count"))
                  .limit(10)
                  .collect()
            )
            print("[Top categories]")
            for r in top_cats:
                print(f"  {r['primary_category']}: {r['count']}")
        except Exception as e:
            print(f"[warn] top_cats aggregation skipped: {e}")

        try:
            by_year = (
                df.groupBy("year")
                  .count()
                  .orderBy("year")
                  .collect()
            )
            print("[Counts by year]")
            for r in by_year:
                print(f"  {r['year']}: {r['count']}")
        except Exception as e:
            print(f"[warn] by_year aggregation skipped: {e}")

    spark.stop()


if __name__ == "__main__":
    main()
