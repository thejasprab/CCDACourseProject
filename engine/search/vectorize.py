# engine/search/vectorize.py
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel

from engine.search.similarity import topk_exact


def vectorize_query(
    spark: SparkSession, model: PipelineModel, title: str, abstract: str
):
    """
    Use the trained Spark ML pipeline to turn (title + abstract) into a TF-IDF vector.

    This is cheap, even on a single row, and reuses the model trained offline.
    """
    text = (title or "") + " " + (abstract or "")
    df = spark.createDataFrame([(text,)], ["text"])
    out = model.transform(df).select(F.col("features_norm").alias("features"))
    return out.first()["features"]


def query_topk(
    spark: SparkSession,
    model: PipelineModel,
    features_train_df,
    query_title: str,
    query_abstract: str,
    k: int = 5,
):
    """
    Full query pipeline inside Spark:

      1. Vectorize the input query with the trained TF-IDF pipeline.
      2. Wrap it as a tiny DataFrame (test_df).
      3. Run brute-force cosine@K via topk_exact against the precomputed corpus
         features (train_df).

    This keeps Spark work at inference limited to:
      - 1-row transform
      - 1 crossJoin + dot-product UDF over the corpus
    """
    # noqa import to keep MLlib SparseVector type available on the driver
    from pyspark.ml.linalg import SparseVector  # noqa: F401

    qvec = vectorize_query(spark, model, query_title, query_abstract)
    qdf = spark.createDataFrame([("Q", qvec)], ["id_base", "features"])
    # query has no categories, but the schema expects the column
    qdf = qdf.withColumn("categories", F.array().cast("array<string>"))

    recs = topk_exact(qdf, features_train_df, k=k, exclude_self=False)
    return recs
