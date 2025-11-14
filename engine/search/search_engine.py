from __future__ import annotations

from typing import List, Dict, Any

from pyspark.sql import functions as F

from engine.utils.spark_utils import get_spark
from engine.ml.model_loader import load_model_and_features
from engine.search.vectorize import query_topk


class SearchEngine:
    """
    Tiny façade around TF-IDF + cosine Top-K for either 'sample' or 'full' mode.
    """

    def __init__(self, mode: str = "sample", spark=None):
        if mode not in {"sample", "full"}:
            raise ValueError("mode must be 'sample' or 'full'")
        self.mode = mode
        self.spark = spark or get_spark(f"search_{mode}")
        self.model, self.features = load_model_and_features(self.spark, mode)

    def search(
        self, title: str = "", abstract: str = "", k: int = 10
    ) -> List[Dict[str, Any]]:
        base = self.features.select(
            "id_base",
            "paper_id",
            "title",
            "abstract",
            "categories",
            "year",
            "features",
        )

        recs = query_topk(
            self.spark,
            self.model,
            base.select("id_base", "categories", "features"),
            title,
            abstract,
            k=k,
        )

        meta = base.select(
            F.col("id_base").alias("neighbor_id_meta"),
            "paper_id",
            "title",
            "abstract",
            "categories",
            "year",
        )

        joined = (
            recs.join(
                meta, recs.neighbor_id == meta.neighbor_id_meta, "left"
            ).drop("neighbor_id_meta")
            .orderBy("rank")
        )

        out: List[Dict[str, Any]] = []
        for r in joined.collect():
            out.append(
                {
                    "rank": int(r["rank"]),
                    "score": float(r["score"]),
                    "neighbor_id": r["neighbor_id"],
                    "paper_id": r["paper_id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "categories": r["categories"],
                    "year": r["year"],
                }
            )
        return out
