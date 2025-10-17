# src/utils.py
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.column import Column


def get_spark(app_name: str = "arxiv_week8") -> SparkSession:
    """
    Create (or get) a SparkSession tuned for low-memory environments.

    Key choices to avoid OOM:
      - Stable local temp dir inside the repo (not /tmp)
      - Many small tasks (high shuffle partitions)
      - AQE with small advisory partition sizes
      - Small input split size per task
      - Skew handling to split oversized partitions
    """
    local_tmp = Path("data/tmp/spark-local")
    local_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")

        # ---- Memory / resources (tune if needed) ----
        # If your machine is tighter on RAM, drop these to 4g.
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")

        # ---- Lots of small tasks; AQE keeps them efficient ----
        .config("spark.sql.shuffle.partitions", "512")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "8m")  # small target post-shuffle
        .config("spark.sql.files.maxPartitionBytes", "8m")                # small input per task

        # ---- Skew handling: split very large partitions ----
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewedPartitionThresholdInBytes", "64m")
        .config("spark.sql.adaptive.skewedPartitionMaxSplitBytes", "16m")

        # ---- Temp / stability ----
        .config("spark.local.dir", str(local_tmp))
        .config("spark.shuffle.checksum.enabled", "false")  # avoid fragile checksum files

        # ---- Parquet default compression (can be overridden per-writer) ----
        .config("spark.sql.parquet.compression.codec", "zstd")

        # ---- Misc ----
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    return spark


# ---------- helper column transforms ----------

def clean_text(col: Column) -> Column:
    """Trim and collapse whitespace."""
    return F.regexp_replace(F.trim(col), r"\s+", " ")


def parse_year_from_datestr(col: Column) -> Column:
    """Extract the first 4-digit year as int."""
    return F.regexp_extract(col.cast("string"), r"(\d{4})", 1).cast("int")


def extract_primary_category(categories_col: Column) -> Column:
    """First whitespace-delimited token is the primary category."""
    return F.split(F.coalesce(categories_col, F.lit("")), r"\s+")[0]


def split_categories(categories_col: Column) -> Column:
    """Split on whitespace into array, dropping empties."""
    arr = F.split(F.coalesce(categories_col, F.lit("")), r"\s+")
    return F.filter(arr, lambda x: x != "")


def normalize_authors(authors_col: Column) -> Column:
    """
    Normalize authors string into an array:
      - Replace ' and ' with commas
      - Normalize comma/whitespace
      - Trim and drop empties
    """
    replaced = F.regexp_replace(F.coalesce(authors_col, F.lit("")), r"\s+and\s+", ",")
    replaced = F.regexp_replace(replaced, r"\s*,\s*", ",")
    arr = F.split(replaced, ",")
    arr = F.transform(arr, lambda x: F.trim(x))
    return F.filter(arr, lambda x: x != "")


def lower(col: Column) -> Column:
    """Lowercase after cleanup."""
    return F.lower(clean_text(col))
