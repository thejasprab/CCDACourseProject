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
    Brute-force exact cosine@K (safe for sample sizes). Assumes 'features' are L2-normalized.
    Expects columns:
      test_df : id_base, categories, features
      train_df: id_base, categories, features
    Returns: test_id, rank, neighbor_id, score, neighbor_categories
    """
    joined = (
        test_df.alias("q")
        .crossJoin(F.broadcast(train_df.alias("c")))
        .withColumn("score", _dot_udf(F.col("q.features"), F.col("c.features")))
    )

    if exclude_self:
        joined = joined.where(F.col("q.id_base") != F.col("c.id_base"))

    w = Window.partitionBy("q.id_base").orderBy(F.desc("score"), F.col("c.id_base"))
    top = (
        joined.withColumn("rank", F.row_number().over(w))
        .where(F.col("rank") <= k)
        .select(
            F.col("q.id_base").alias("test_id"),
            "rank",
            F.col("c.id_base").alias("neighbor_id"),
            "score",
            F.col("c.categories").alias("neighbor_categories"),
        )
    )
    return top
