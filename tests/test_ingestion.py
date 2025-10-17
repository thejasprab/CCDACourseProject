"""
Unit tests for the ingestion pipeline (src/ingestion.py).

These tests validate that the ingestion process:
  - Produces a valid Spark DataFrame with the expected schema and types.
  - Contains derived columns like title_len, abstract_len, and has_doi.
  - Correctly parses authors, categories, and versions.
  - Writes readable parquet output without corruption.

Run:
    pytest tests/test_ingestion_pipeline.py -v
"""

import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from src import ingestion


# ---------- Spark fixture ----------

@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .appName("test_ingestion_pipeline")
        .master("local[2]")
        .getOrCreate()
    )
    yield spark
    spark.stop()


# ---------- sample raw data ----------

@pytest.fixture(scope="session")
def raw_sample_df(spark):
    """
    Creates a small synthetic DataFrame simulating raw arXiv JSON data
    (as produced by scripts/download_arxiv.py or similar).
    """
    data = [
        {
            "id": "arXiv:2101.00001",
            "title": "Deep Learning for Vision",
            "abstract": "We explore neural network architectures for computer vision.",
            "authors": "Alice, Bob",
            "categories": "cs.CV cs.LG",
            "versions": [{"v": 1}, {"v": 2}],
            "doi": "10.1000/example",
            "submitted_date": "2021-01-01T00:00:00Z",
        },
        {
            "id": "arXiv:1901.00234",
            "title": "Quantum Computing Basics",
            "abstract": "An introduction to quantum algorithms and their limitations.",
            "authors": "Carol",
            "categories": "quant-ph",
            "versions": [{"v": 1}],
            "doi": None,
            "submitted_date": "2019-02-15T00:00:00Z",
        },
    ]
    df = spark.createDataFrame(data)
    return df


# ---------- ingestion output fixture ----------

@pytest.fixture(scope="session")
def ingested_df(spark, raw_sample_df, tmp_path):
    """
    Runs the ingestion.transform() or main() function on the synthetic dataset
    to produce the processed DataFrame.
    """
    if hasattr(ingestion, "transform"):
        df_processed = ingestion.transform(raw_sample_df)
    elif hasattr(ingestion, "main"):
        df_processed = ingestion.main(raw_sample_df)
    else:
        pytest.skip("No valid ingestion entry function found in src/ingestion.py")

    # Optionally save to Parquet (to test writing)
    outpath = tmp_path / "arxiv_ingest_test.parquet"
    df_processed.write.mode("overwrite").parquet(str(outpath))
    df_reloaded = spark.read.parquet(str(outpath))
    return df_reloaded


# ---------- tests ----------

def test_schema_contains_expected_columns(ingested_df):
    expected_cols = {
        "arxiv_id", "title", "abstract", "year",
        "primary_category", "authors_list", "versions",
        "title_len", "abstract_len", "has_doi"
    }
    df_cols = set(ingested_df.columns)
    missing = expected_cols - df_cols
    assert not missing, f"Missing columns in ingestion output: {missing}"


def test_title_and_abstract_lengths_computed(ingested_df):
    lengths = ingested_df.select("title_len", "abstract_len").collect()
    for row in lengths:
        assert row.title_len > 0
        assert row.abstract_len > 0


def test_has_doi_column_valid(ingested_df):
    vals = [r.has_doi for r in ingested_df.select("has_doi").collect()]
    assert all(v in [True, False] for v in vals)


def test_authors_list_extracted(ingested_df):
    df = ingested_df.select(F.size(F.col("authors_list")).alias("n_authors"))
    assert df.filter(df.n_authors > 0).count() == df.count()


def test_primary_category_not_null(ingested_df):
    assert ingested_df.filter(F.col("primary_category").isNotNull()).count() > 0


def test_year_parsed_correctly(ingested_df):
    years = [r.year for r in ingested_df.select("year").distinct().collect()]
    assert all(isinstance(y, int) and 1900 < y < 2100 for y in years)


def test_versions_array_nonempty(ingested_df):
    counts = ingested_df.select(F.size(F.col("versions")).alias("n_versions"))
    assert counts.filter(counts.n_versions > 0).count() == counts.count()


def test_ingested_dataframe_nonempty(ingested_df):
    assert ingested_df.count() > 0


def test_unique_arxiv_ids(ingested_df):
    distinct = ingested_df.select("arxiv_id").distinct().count()
    total = ingested_df.count()
    assert distinct == total, "arxiv_id should be unique per paper"


def test_output_writable(tmp_path, ingested_df, spark):
    outpath = tmp_path / "ingestion_output.parquet"
    ingested_df.write.mode("overwrite").parquet(str(outpath))
    reloaded = spark.read.parquet(str(outpath))
    assert reloaded.count() == ingested_df.count()
    assert set(reloaded.columns) == set(ingested_df.columns)
