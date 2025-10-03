# src/utils.py
from pyspark.sql import SparkSession, functions as F

def get_spark(app_name: str = "arxiv_week8") -> SparkSession:
    """Create (or get) a SparkSession with sane defaults for Codespaces/local."""
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    return spark

def clean_text(col: F.Column) -> F.Column:
    return F.regexp_replace(F.trim(col), r"\s+", " ")

def parse_year_from_datestr(col: F.Column) -> F.Column:
    return F.regexp_extract(col.cast("string"), r"(\d{4})", 1).cast("int")

def extract_primary_category(categories_col: F.Column) -> F.Column:
    return F.split(F.coalesce(categories_col, F.lit("")), r"\s+")[0]

def split_categories(categories_col: F.Column) -> F.Column:
    return F.expr("filter(split(coalesce(categories, ''), ' +'), x -> x <> '')")

def normalize_authors(authors_col: F.Column) -> F.Column:
    replaced = F.regexp_replace(F.coalesce(authors_col, F.lit("")), r"\s+and\s+", ",")
    replaced = F.regexp_replace(replaced, r"\s*,\s*", ",")
    return F.expr(f"transform(filter(split({replaced._jc.toString()}, ','), x -> x <> ''), x -> trim(x))")

def lower(col: F.Column) -> F.Column:
    return F.lower(clean_text(col))
