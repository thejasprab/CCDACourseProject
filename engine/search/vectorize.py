from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel

from engine.search.similarity import topk_exact


def vectorize_query(
    spark: SparkSession, model: PipelineModel, title: str, abstract: str
):
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
    from pyspark.ml.linalg import SparseVector  # noqa: F401

    qvec = vectorize_query(spark, model, query_title, query_abstract)
    qdf = spark.createDataFrame([("Q", qvec)], ["id_base", "features"])
    qdf = qdf.withColumn("categories", F.array().cast("array<string>"))
    recs = topk_exact(qdf, features_train_df, k=k, exclude_self=False)
    return recs
