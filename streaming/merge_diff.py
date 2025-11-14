from __future__ import annotations

from pathlib import Path

from pyspark.sql import functions as F

from engine.utils.spark_utils import get_spark


def diff_new_papers(old_parquet: str, new_parquet: str, out_parquet: str):
    """
    Simple helper: given old and new full Parquets, write only *new* arxiv_id rows.
    """
    spark = get_spark("merge_diff")
    old = spark.read.parquet(old_parquet).select("arxiv_id").distinct()
    new = spark.read.parquet(new_parquet)

    only_new = new.join(old, on="arxiv_id", how="left_anti")
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    only_new.write.mode("overwrite").parquet(out_parquet)
    spark.stop()
