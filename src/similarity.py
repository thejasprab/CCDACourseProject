from typing import List
from pyspark.sql import DataFrame, functions as F, Window
from pyspark.ml.linalg import VectorUDT, SparseVector

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

_dot_udf = F.udf(_dot)

def topk_exact(
    test_df: DataFrame,
    train_df: DataFrame,
    k: int = 3,
    exclude_self: bool = True,
):
    """
    Brute-force exact cosine@K (safe for sample sizes). Assumes 'features' are L2-normalized.
    Expects columns:
      test_df : id_base, categories, features
      train_df: id_base, categories, features
    Returns: test_id, rank, neighbor_id, score, neighbor_categories
    """
    # Hint broadcast if train is small (sample)
    joined = (test_df.alias("q")
              .crossJoin(F.broadcast(train_df.alias("c")))
              .withColumn("score", _dot_udf(F.col("q.features"), F.col("c.features"))))

    if exclude_self:
        joined = joined.where(F.col("q.id_base") != F.col("c.id_base"))

    w = Window.partitionBy("q.id_base").orderBy(F.desc("score"), F.col("c.id_base"))
    top = (joined
           .withColumn("rank", F.row_number().over(w))
           .where(F.col("rank") <= k)
           .select(
               F.col("q.id_base").alias("test_id"),
               "rank",
               F.col("c.id_base").alias("neighbor_id"),
               "score",
               F.col("c.categories").alias("neighbor_categories")
           ))
    return top

def pairwise_cosine_mean(df: DataFrame) -> DataFrame:
    """
    Compute mean pairwise cosine within each (test_id) recommended list for intra-list diversity (ILD).
    Lower mean cosine => more diverse list.
    Input: df with columns (test_id, rank, neighbor_id, score, features?) — features optional.
    If 'features' not present, returns mean(score) over pairs formed by ranks (proxy).
    """
    if "features" not in df.columns:
        # Use provided 'score' between query and each item; for ILD we need item-item similarity.
        # Fallback proxy: average of (score) won't reflect ILD correctly; we instead return null.
        return df.groupBy("test_id").agg(F.lit(None).alias("ild_mean"))
    # To compute pairwise item-item cosine we need features for neighbors; caller should join them.
    # This function just expects a prepared table with pairs and 'pair_cosine'.
    # Left here as placeholder; real computation is done in the notebook where we have both sides.
    return df.groupBy("test_id").agg(F.avg("pair_cosine").alias("ild_mean"))
