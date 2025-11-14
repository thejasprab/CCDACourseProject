from pathlib import Path
from typing import Tuple

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame


def _paths_for_mode(mode: str) -> Tuple[Path, Path]:
    if mode not in {"full", "sample"}:
        raise ValueError("mode must be 'full' or 'sample'")
    if mode == "full":
        model_dir = Path("data/models/tfidf_full")
        feats = Path("data/processed/features_full")
    else:
        model_dir = Path("data/models/tfidf_sample")
        feats = Path("data/processed/features_sample")
    return model_dir, feats


def load_model_and_features(
    spark: SparkSession, mode: str
) -> Tuple[PipelineModel, DataFrame]:
    model_dir, feats_dir = _paths_for_mode(mode)
    model = PipelineModel.load(str(model_dir))
    feats = spark.read.parquet(str(feats_dir))
    return model, feats
