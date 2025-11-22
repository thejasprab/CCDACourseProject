# engine/search/similarity.py
from typing import List

from pyspark.sql import DataFrame, functions as F, Window
from pyspark.ml.linalg import SparseVector
from pyspark.sql.types import DoubleType

# Cosine on L2-normalized vectors == dot product.


def _dot(a: SparseVector, b: SparseVector) -> float:
    if a is None or b is None:
        return 0.0
    ai = dict(zip(a.indices, a.values))
    s = 0.0
    for j, v in zip(b.indices, b.values):
        if j in ai:
            s += ai[j] * v
    return float(s)


_dot_udf = F.udf(_dot, DoubleType())


def topk_exact(
    test_df: DataFrame,
    train_df: DataFrame,
    k: int = 3,
    exclude_self: bool = True,
) -> DataFrame:
    """
    Brute-force exact cosine@K.

    Assumes 'features' are L2-normalized.

    Inputs:
      test_df : id_base, categories, features
      train_df: id_base, categories, features

    Output:
      test_id, rank, neighbor_id, score, neighbor_categories
    """
    # NOTE: do NOT broadcast the full train_df; it's millions of rows.
    # Spark will naturally broadcast the tiny test_df side.
    joined = (
        test_df.alias("q")
        .crossJoin(train_df.alias("c"))
        .select(
            F.col("q.id_base").alias("q_id_base"),
            F.col("q.categories").alias("q_categories"),
            F.col("q.features").alias("q_features"),
            F.col("c.id_base").alias("c_id_base"),
            F.col("c.categories").alias("c_categories"),
            F.col("c.features").alias("c_features"),
        )
    )

    joined = joined.withColumn(
        "score",
        _dot_udf(F.col("q_features"), F.col("c_features")),
    )

    if exclude_self:
        joined = joined.where(F.col("q_id_base") != F.col("c_id_base"))

    w = Window.partitionBy("q_id_base").orderBy(F.col("score").desc(), F.col("c_id_base"))
    top = (
        joined.withColumn("rank", F.row_number().over(w))
        .where(F.col("rank") <= k)
        .select(
            F.col("q_id_base").alias("test_id"),
            "rank",
            F.col("c_id_base").alias("neighbor_id"),
            "score",
            F.col("c_categories").alias("neighbor_categories"),
        )
    )
    return top
