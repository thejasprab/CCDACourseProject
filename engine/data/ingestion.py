"""
Generic ingestion for Kaggle arXiv metadata → Parquet.

Used by:
  - pipelines.ingest_full
  - pipelines.ingest_sample
"""

from __future__ import annotations

import argparse
from pyspark.sql import functions as F

from engine.utils.spark_utils import get_spark
from engine.data.transformations import transform_all


def read_arxiv_json(spark, path: str, multiline: bool = False):
    """Read JSON/JSONL (Kaggle snapshot is normally JSONL)."""
    return (
        spark.read.option("multiLine", "true" if multiline else "false")
        .json(path)
    )


def run_ingestion(
    input_path: str,
    output_path: str,
    multiline: bool = False,
    limit: int = 0,
    sample_frac: float = 0.0,
    repartition: int = 0,
    min_abstract_len: int = 40,
    partition_by: str = "year",  # "year" | "primary_category" | "none"
    no_stats: bool = False,
    app_name: str = "arxiv_ingestion",
):
    spark = get_spark(app_name)

    df_raw = read_arxiv_json(spark, input_path, multiline=multiline)

    if sample_frac and 0.0 < sample_frac <= 1.0:
        df_raw = df_raw.sample(False, sample_frac, seed=42)
    if limit and limit > 0:
        df_raw = df_raw.limit(limit)

    df = transform_all(df_raw, min_abstract_len=min_abstract_len)

    if partition_by in ("year", "primary_category"):
        target = repartition if repartition and repartition > 0 else 512
        df = df.repartition(target, F.col(partition_by))
    else:
        if repartition and repartition > 0:
            df = df.repartition(repartition)

    writer = (
        df.write.mode("overwrite")
        .option("compression", "zstd")
        .option("parquet.block.size", 8 * 1024 * 1024)
        .option("parquet.page.size", 512 * 1024)
        .option("parquet.enable.dictionary", "true")
        .option("maxRecordsPerFile", 50000)
    )
    if partition_by != "none":
        writer = writer.partitionBy(partition_by)

    writer.parquet(output_path)

    if not no_stats:
        n = df.count()
        print(f"[OK] Wrote {n} rows to {output_path}")

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
        except Exception as e:  # noqa: BLE001
            print(f"[warn] top_cats aggregation skipped: {e}")

        try:
            by_year = df.groupBy("year").count().orderBy("year").collect()
            print("[Counts by year]")
            for r in by_year:
                print(f"  {r['year']}: {r['count']}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] by_year aggregation skipped: {e}")

    spark.stop()


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Ingest arXiv JSON/JSONL into Parquet (generic)."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--multiline", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-frac", type=float, default=0.0)
    ap.add_argument("--repartition", type=int, default=0)
    ap.add_argument("--min-abstract-len", type=int, default=40)
    ap.add_argument(
        "--partition-by",
        default="year",
        choices=["year", "primary_category", "none"],
    )
    ap.add_argument("--no-stats", action="store_true")
    return ap.parse_args()


def main():
    args = _parse_args()
    run_ingestion(
        input_path=args.input,
        output_path=args.output,
        multiline=args.multiline,
        limit=args.limit,
        sample_frac=args.sample_frac,
        repartition=args.repartition,
        min_abstract_len=args.min_abstract_len,
        partition_by=args.partition_by,
        no_stats=args.no_stats,
    )


if __name__ == "__main__":
    main()
