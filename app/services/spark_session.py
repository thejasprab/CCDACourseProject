from __future__ import annotations

from pyspark.sql import SparkSession

from engine.utils.spark_utils import get_spark

_SPARK: SparkSession | None = None


def get_spark_session() -> SparkSession:
    global _SPARK  # noqa: PLW0603
    if _SPARK is None:
        _SPARK = get_spark("ccda_app")
    return _SPARK
