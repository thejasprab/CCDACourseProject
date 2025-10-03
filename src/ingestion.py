# src/ingestion.py
import argparse
from pyspark.sql import functions as F
from src.utils import get_spark
from src.transformations import transform_all

def read_arxiv_json(spark, path: str, multiline: bool = False):
    return (
        spark.read
        .option("multiLine", "true" if multiline else "false")
        .json(path)
    )

def main():
    ap = argparse.ArgumentParser(description="Ingest Kaggle arXiv metadata into Parquet for Week 8 EDA.")
    ap.add_argument("--input", required=True, help="Path or glob to arxiv metadata JSON/JSONL (local).")
    ap.add_argument("--output", required=True, help="Directory to write processed Parquet.")
    ap.add_argument("--multiline", action="store_true", help="Input is pretty-printed multi-line JSON.")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit rows for quick demo.")
    ap.add_argument("--sample-frac", type=float, default=0.0, help="Optional: sample fraction (0<frac<=1).")
    ap.add_argument("--repartition", type=int, default=0, help="Optional: repartition before write.")
    ap.add_argument("--min-abstract-len", type=int, default=40, help="Filter very short abstracts.")
    ap.add_argument("--partition-by", default="year", choices=["year", "primary_category", "none"], help="Partition column for Parquet.")
    args = ap.parse_args()

    spark = get_spark("arxiv_week8_ingestion")

    df_raw = read_arxiv_json(spark, args.input, multiline=args.multiline)

    if args.sample_frac and args.sample_frac > 0:
        df_raw = df_raw.sample(False, args.sample_frac, seed=42)
    if args.limit and args.limit > 0:
        df_raw = df_raw.limit(args.limit)

    df = transform_all(df_raw, min_abstract_len=args.min_abstract_len)

    if args.repartition and args.repartition > 0:
        df = df.repartition(args.repartition)

    writer = df.write.mode("overwrite")
    if args.partition_by != "none":
        writer = writer.partitionBy(args.partition_by)
    writer.parquet(args.output)

    n = df.count()
    top_cats = df.groupBy("primary_category").count().orderBy(F.desc("count")).limit(10).collect()
    by_year = df.groupBy("year").count().orderBy("year").collect()

    print(f"[OK] Wrote {n} rows to {args.output}")
    print("[Top categories]")
    for r in top_cats:
        print(f"  {r['primary_category']}: {r['count']}")
    print("[Counts by year]")
    for r in by_year:
        print(f"  {r['year']}: {r['count']}")

    spark.stop()

if __name__ == "__main__":
    main()
