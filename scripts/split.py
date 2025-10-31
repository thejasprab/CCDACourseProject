#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


def build_spark():
    return (
        SparkSession.builder
        .appName("split_full")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "512")
        .config("spark.sql.files.maxPartitionBytes", 256 << 20)  # 256MB
        .getOrCreate()
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Input Parquet root (full dataset)")
    ap.add_argument("--out", required=True, help="Output dir with split=train|test partitions")
    ap.add_argument("--test-years", default="2019,2020,2021")
    ap.add_argument("--test-size", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def ensure_columns(df):
    cols = set(df.columns)
    dtypes = dict(df.dtypes)

    # Canonical arXiv IDs
    if "arxiv_id" in cols and "id" not in cols:
        df = df.withColumn("arxiv_id", F.col("arxiv_id").cast("string"))
        df = df.withColumn("id", F.col("arxiv_id"))
    elif "id" in cols:
        df = df.withColumn("id", F.col("id").cast("string"))
        if "arxiv_id" not in cols:
            df = df.withColumn("arxiv_id", F.col("id"))
    else:
        rn = F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
        df = (
            df.withColumn("_row_id", rn.cast("string"))
              .withColumn("id", F.concat_ws("_", F.lit("row"), F.col("_row_id")))
              .withColumn("arxiv_id", F.col("id"))
              .drop("_row_id")
        )

    # id_base (strip version suffix v\d+)
    df = df.withColumn("id_base", F.regexp_replace(F.col("id"), r"v\d+$", ""))

    # Strings
    df = (
        df.withColumn("title", F.coalesce(F.col("title").cast("string"), F.lit("")))
          .withColumn("abstract", F.coalesce(F.col("abstract").cast("string"), F.lit("")))
    )

    # categories => array<string>
    if "categories" in cols and dtypes.get("categories") == "string":
        df = df.withColumn("categories", F.split(F.col("categories"), r"\s+"))
    elif "categories" not in cols:
        df = df.withColumn("categories", F.array().cast("array<string>"))

    # year
    if "year" not in cols:
        if "update_date" in cols:
            df = df.withColumn("update_date", F.col("update_date").cast("string"))
            df = df.withColumn("year", F.substring(F.col("update_date"), 1, 4).cast("int"))
        elif "versions" in cols:
            created = F.coalesce(
                F.col("versions")[0]["created"],
                F.element_at(F.col("versions"), 1)["created"],
            )
            df = (
                df.withColumn("_created", created.cast("string"))
                  .withColumn("year", F.regexp_extract(F.col("_created"), r"(19|20)\d{2}", 0).cast("int"))
                  .drop("_created")
            )
        else:
            df = df.withColumn("year", F.lit(None).cast("int"))

    # abstract length filter
    df = df.withColumn("abs_len", F.length("abstract")).filter(F.col("abs_len") >= 20).drop("abs_len")

    # de-dup by id_base (keep latest year when available)
    w = Window.partitionBy("id_base").orderBy(F.desc_nulls_last("year"))
    df = df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")

    return df


def main():
    args = parse_args()
    spark = build_spark()

    test_years = [int(y) for y in args.test_years.split(",") if y.strip()]
    df = spark.read.parquet(args.parquet)
    df = ensure_columns(df)

    test_pool = df.filter(F.col("year").isin(test_years))
    test_ids = (
        test_pool.select("id_base")
        .orderBy(F.rand(args.seed))
        .limit(args.test_size)
        .withColumn("split", F.lit("test"))
    )

    labeled = (
        df.join(test_ids, on="id_base", how="left")
          .withColumn("split", F.when(F.col("split").isNull(), "train").otherwise("test"))
    )

    (
        labeled.repartition("split")
        .write.mode("overwrite")
        .partitionBy("split")
        .parquet(args.out)
    )

    print(f"[split] wrote → {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()