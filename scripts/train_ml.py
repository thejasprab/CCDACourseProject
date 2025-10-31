#!/usr/bin/env python3
import argparse
import json
import os
import time
from pyspark.sql import SparkSession, functions as F
from src.featurization import build_text_pipeline, compute_extra_stopwords


def make_spark():
    try:
        from src.utils import get_spark  # type: ignore
        return get_spark(app_name="train_full")
    except Exception:
        return (
            SparkSession.builder
            .appName("train_full")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "1024")
            .config("spark.sql.files.maxPartitionBytes", 256 << 20)
            .getOrCreate()
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-parquet", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--features-out", required=True)
    ap.add_argument("--vocab-size", type=int, default=250000)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--use-bigrams", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--extra-stopwords-topdf", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def ensure_id_base(df):
    if "id_base" in df.columns:
        return df
    if "id" in df.columns:
        return df.withColumn("id_base", F.regexp_replace(F.col("id").cast("string"), r"v\\d+$", ""))
    if "arxiv_id" in df.columns:
        return df.withColumn("id_base", F.regexp_replace(F.col("arxiv_id").cast("string"), r"v\\d+$", ""))
    raise ValueError("Expected 'id_base' or 'id' or 'arxiv_id' in dataset.")


def normalize_schema_for_text(df):
    dtypes = dict(df.dtypes)

    if "categories" in dtypes and dtypes["categories"] == "string":
        df = df.withColumn("categories", F.split(F.col("categories"), r"\\s+"))
    elif "categories" not in dtypes:
        df = df.withColumn("categories", F.array().cast("array<string>"))

    df = (
        df.withColumn("title", F.coalesce(F.col("title").cast("string"), F.lit("")))
          .withColumn("abstract", F.coalesce(F.col("abstract").cast("string"), F.lit("")))
    )

    if "arxiv_id" in df.columns:
        df = df.withColumn("paper_id", F.col("arxiv_id").cast("string"))
    elif "id" in df.columns:
        df = df.withColumn("paper_id", F.col("id").cast("string"))
    else:
        df = df.withColumn("paper_id", F.col("id_base"))

    df = df.withColumn("text", F.concat_ws(" ", F.lower(F.col("title")), F.lower(F.col("abstract"))))
    return df


def main():
    args = parse_args()
    spark = make_spark()

    df = spark.read.parquet(args.split_parquet)
    df = ensure_id_base(df)
    df = normalize_schema_for_text(df)

    if "split" not in df.columns:
        raise ValueError("Expected a 'split' column with values 'train'/'test' in --split-parquet")

    train = df.filter(F.col("split") == "train")
    test = df.filter(F.col("split") == "test")

    extra_sw = compute_extra_stopwords(
        spark, train.select("id_base", "abstract"), top_df=args.extra_stopwords_topdf, seed=args.seed
    )

    pipeline = build_text_pipeline(
        vocab_size=args.vocab_size,
        min_df=args.min_df,
        use_bigrams=args.use_bigrams,
        extra_stopwords=extra_sw,
    )
    model = pipeline.fit(train)

    model.write().overwrite().save(args.model_dir)
    os.makedirs(args.model_dir, exist_ok=True)
    meta = {
        "created_at": int(time.time()),
        "vocab_size": args.vocab_size,
        "min_df": args.min_df,
        "use_bigrams": args.use_bigrams,
        "extra_stopwords_topdf": args.extra_stopwords_topdf,
        "seed": args.seed,
        "spark_version": spark.version,
    }
    with open(os.path.join(args.model_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)

    feats_train = (
        model.transform(train)
        .select(
            "id_base",
            "paper_id",
            "title",
            "abstract",
            "categories",
            "year",
            F.col("features_norm").alias("features"),
        )
    )

    feats_test = (
        model.transform(test)
        .select(
            "id_base",
            "paper_id",
            "title",
            "abstract",
            "categories",
            "year",
            F.col("features_norm").alias("features"),
        )
    )

    (
        feats_train.repartition(256)
        .write.mode("overwrite")
        .parquet(os.path.join(args.features_out, "split=train"))
    )
    (
        feats_test.repartition(128)
        .write.mode("overwrite")
        .parquet(os.path.join(args.features_out, "split=test"))
    )

    print(f"[train] model saved to: {args.model_dir}")
    print(f"[train] features saved to: {args.features_out}/split={'{train|test}'}")
    spark.stop()


if __name__ == "__main__":
    main()