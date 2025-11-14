# engine/ml/train.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pyspark.sql import functions as F

from engine.utils.spark_utils import get_spark
from engine.utils.io_utils import write_json
from engine.ml.featurization import build_text_pipeline, compute_extra_stopwords


def ensure_id_base(df):
    if "id_base" in df.columns:
        return df
    if "arxiv_id" in df.columns:
        return df.withColumn(
            "id_base",
            F.regexp_replace(F.col("arxiv_id").cast("string"), r"v\d+$", ""),
        )
    if "id" in df.columns:
        return df.withColumn(
            "id_base", F.regexp_replace(F.col("id").cast("string"), r"v\d+$", "")
        )
    raise ValueError("Expected one of 'id_base', 'arxiv_id', or 'id' in dataset")


def normalize_schema_for_text(df):
    dtypes = dict(df.dtypes)

    if "categories" in dtypes and dtypes["categories"] == "string":
        df = df.withColumn("categories", F.split(F.col("categories"), r"\s+"))
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

    df = df.withColumn(
        "text",
        F.concat_ws(
            " ",
            F.lower(F.col("title")).alias(""),
            F.lower(F.col("abstract")).alias(""),
        ),
    )
    return df


def train_model(
    input_parquet: str,
    model_dir: str,
    features_out: str,
    vocab_size: int = 250_000,
    min_df: int = 5,
    use_bigrams: bool = False,
    extra_stopwords_topdf: int = 500,
    seed: int = 42,
    app_name: str = "tfidf_train",
):
    """
    Generic TF-IDF training routine used by both sample + full pipelines.

    NOTE: extra_stopwords_topdf can be set to 0 to disable the expensive
    extra-stopword computation on very large datasets (full arXiv).
    """
    spark = get_spark(app_name)
    df = spark.read.parquet(input_parquet)

    df = ensure_id_base(df)
    df = normalize_schema_for_text(df)

    # Optional extra stopwords computation (can be disabled by passing 0)
    if extra_stopwords_topdf is not None and extra_stopwords_topdf > 0:
        extra_sw = compute_extra_stopwords(
            spark,
            df.select("id_base", "abstract"),
            top_df=extra_stopwords_topdf,
            seed=seed,
        )
    else:
        extra_sw = []

    pipeline = build_text_pipeline(
        vocab_size=vocab_size,
        min_df=min_df,
        use_bigrams=use_bigrams,
        extra_stopwords=extra_sw,
    )
    model = pipeline.fit(df)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(model_dir)

    meta = {
        "created_at": int(time.time()),
        "vocab_size": vocab_size,
        "min_df": min_df,
        "use_bigrams": use_bigrams,
        "extra_stopwords_topdf": extra_stopwords_topdf,
        "seed": seed,
        "spark_version": spark.version,
    }
    write_json(meta, Path(model_dir) / "model.json")

    feats = (
        model.transform(df)
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
        feats.repartition(256)
        .write.mode("overwrite")
        .parquet(features_out)
    )

    print(f"[train] model saved to: {model_dir}")
    print(f"[train] features saved to: {features_out}")

    spark.stop()


def _parse_args():
    ap = argparse.ArgumentParser(description="Train TF-IDF model (no split).")
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--features-out", required=True)
    ap.add_argument("--vocab-size", type=int, default=250000)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument(
        "--use-bigrams", type=lambda x: str(x).lower() == "true", default=False
    )
    ap.add_argument("--extra-stopwords-topdf", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = _parse_args()
    train_model(
        input_parquet=args.input_parquet,
        model_dir=args.model_dir,
        features_out=args.features_out,
        vocab_size=args.vocab_size,
        min_df=args.min_df,
        use_bigrams=args.use_bigrams,
        extra_stopwords_topdf=args.extra_stopwords_topdf,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
