from __future__ import annotations

from typing import List

from pyspark.sql import functions as F

from app.services.spark_session import get_spark_session
from engine.ml.model_loader import load_model_and_features


def list_primary_categories(mode: str = "sample", top_k: int = 50) -> List[str]:
    spark = get_spark_session()
    _, feats = load_model_and_features(spark, mode)
    df = (
        feats.groupBy("categories")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_k)
    )
    cats = set()
    for row in df.collect():
        if isinstance(row["categories"], list):
            for c in row["categories"]:
                cats.add(c)
    return sorted(cats)
