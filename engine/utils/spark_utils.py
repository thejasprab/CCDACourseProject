# engine/utils/spark_utils.py
from pathlib import Path

from pyspark.sql import SparkSession


def get_spark(app_name: str = "ccda_project") -> SparkSession:
    """
    Create (or get) a SparkSession tuned for this project.

    Memory settings are modest but slightly higher than the original (6g → 8g).
    Adjust them downwards if your machine is tight on RAM, or upwards if you
    have plenty of memory.
    """
    local_tmp = Path("data/tmp/spark-local")
    local_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")

        # Memory; tweak if your machine can't handle this
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")

        # Parquet safety: smaller batches + disable vectorized reader
        .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        # Also disable vectorized in-memory columnar storage; keeps allocations saner
        .config("spark.sql.inMemoryColumnarStorage.enableVectorized", "false")

        # Shuffle / partition tuning
        .config("spark.sql.shuffle.partitions", "512")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "8m")
        .config("spark.sql.files.maxPartitionBytes", "8m")

        # Skew handling
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewedPartitionThresholdInBytes", "64m")
        .config("spark.sql.adaptive.skewedPartitionMaxSplitBytes", "16m")

        # Temp dirs
        .config("spark.local.dir", str(local_tmp))
        .config("spark.shuffle.checksum.enabled", "false")

        # Default Parquet compression
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    return spark
