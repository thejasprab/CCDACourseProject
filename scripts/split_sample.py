#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window  # <-- correct import

def build_spark():
    return (SparkSession.builder
            .appName("split_sample")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "64")
            .getOrCreate())

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-years", default="2007,2008,2009")
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def ensure_columns(df):
    cols = set(df.columns)

    # id_base (strip version suffix v\d+). If no 'id', synthesize a stable id.
    if "id" in cols:
        df = df.withColumn("id", F.col("id").cast("string"))
        df = df.withColumn("id_base", F.regexp_replace(F.col("id"), r"v\d+$", ""))
    else:
        # synthesize id from input file + row number
        rn = F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
        df = df.withColumn("_row_id", rn)
        df = df.withColumn("id_synth", F.concat_ws(":", F.input_file_name(), F.col("_row_id")))
        df = df.withColumn("id_base", F.col("id_synth")).drop("_row_id", "id_synth")

    # title / abstract
    df = (df
          .withColumn("title", F.coalesce(F.col("title").cast("string"), F.lit("")))
          .withColumn("abstract", F.coalesce(F.col("abstract").cast("string"), F.lit(""))))

    # categories: support string "cs.LG cs.AI" or array<string>
    dtype_map = dict(df.dtypes)
    if "categories" in cols and dtype_map.get("categories") == "string":
        df = df.withColumn("categories", F.split(F.col("categories"), r"\s+"))
    elif "categories" not in cols:
        df = df.withColumn("categories", F.array().cast("array<string>"))

    # year: prefer explicit; else derive from update_date or versions.created
    if "year" not in cols:
        if "update_date" in cols:
            df = df.withColumn("update_date", F.col("update_date").cast("string"))
            df = df.withColumn("year", F.substring(F.col("update_date"), 1, 4).cast("int"))
        elif "versions" in cols:
            created = F.coalesce(F.col("versions")[0]["created"], F.element_at(F.col("versions"), 1)["created"])
            df = (df.withColumn("_created", created.cast("string"))
                    .withColumn("year", F.regexp_extract(F.col("_created"), r"(19|20)\d{2}", 0).cast("int"))
                    .drop("_created"))
        else:
            df = df.withColumn("year", F.lit(None).cast("int"))

    # basic text filter
    df = df.withColumn("abs_len", F.length("abstract")).filter(F.col("abs_len") >= 20).drop("abs_len")

    # de-dup by id_base keeping latest year (when available)
    w = Window.partitionBy("id_base").orderBy(F.desc_nulls_last("year"))
    df = df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")

    return df

def main():
    args = parse_args()
    spark = build_spark()

    test_years = [int(y) for y in args.test_years.split(",") if y.strip()]
    df = spark.read.parquet(args.parquet)
    df = ensure_columns(df)

    # Choose TEST from requested years; if year missing, exclude from TEST pool
    test_pool = df.filter(F.col("year").isin(test_years))
    test_ids = (test_pool.select("id_base")
                .orderBy(F.rand(args.seed))
                .limit(args.test_size)
                .withColumn("split", F.lit("test")))

    labeled = (df.join(test_ids, on="id_base", how="left")
                 .withColumn("split", F.when(F.col("split").isNull(), "train").otherwise("test")))

    (labeled
     .repartition("split")
     .write.mode("overwrite")
     .partitionBy("split")
     .parquet(args.out))

    print(f"[split] wrote → {args.out}")
    spark.stop()

if __name__ == "__main__":
    main()
