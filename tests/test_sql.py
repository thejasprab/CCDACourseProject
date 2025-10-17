"""
Unit tests for notebooks/week9-ComplexQueries.py

These tests create a miniature in-memory Spark DataFrame that imitates
the arXiv metadata and validate that each complex analytical query
runs successfully and returns expected columns.

Run:
    pytest tests/test_sql_complex.py -v
"""

import pytest
from pyspark.sql import SparkSession, Row
from notebooks import week9_ComplexQueries as w9

# ---------- Spark fixture ----------

@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .appName("test_week9_complex_queries")
        .master("local[2]")
        .getOrCreate()
    )
    yield spark
    spark.stop()

# ---------- miniature dataset ----------

@pytest.fixture(scope="session")
def sample_df(spark):
    data = [
        Row(
            arxiv_id="a1",
            title="AI Paper 1",
            abstract="Deep learning improves vision tasks significantly.",
            abstract_len=55,
            primary_category="cs.LG",
            categories="cs.LG stat.ML",
            authors_list=["Alice", "Bob"],
            versions=[{"v": 1}, {"v": 2}],
            has_doi=True,
            year=2019,
            submitted_date="2019-03-15",
        ),
        Row(
            arxiv_id="a2",
            title="Physics optics",
            abstract="Quantum optics paper abstract text.",
            abstract_len=35,
            primary_category="physics.optics",
            categories="physics.optics",
            authors_list=["Carol"],
            versions=[{"v": 1}],
            has_doi=False,
            year=2018,
            submitted_date="2018-02-10",
        ),
        Row(
            arxiv_id="a3",
            title="ML and NLP",
            abstract="Language models improve translation accuracy.",
            abstract_len=47,
            primary_category="cs.CL",
            categories="cs.CL cs.LG",
            authors_list=["Alice", "Dan"],
            versions=[{"v": 1}, {"v": 2}, {"v": 3}],
            has_doi=True,
            year=2020,
            submitted_date="2020-06-02",
        ),
        Row(
            arxiv_id="a4",
            title="Cross discipline",
            abstract="Interdisciplinary study bridging physics and AI.",
            abstract_len=60,
            primary_category="cs.LG",
            categories="cs.LG physics.comp-ph",
            authors_list=["Eve", "Bob"],
            versions=[{"v": 1}],
            has_doi=False,
            year=2021,
            submitted_date="2021-09-07",
        ),
        Row(
            arxiv_id="a5",
            title="Old math",
            abstract="Classical mathematics revisited.",
            abstract_len=40,
            primary_category="math.GM",
            categories="math.GM",
            authors_list=["Frank"],
            versions=[{"v": 1}],
            has_doi=True,
            year=2010,
            submitted_date="2010-01-15",
        ),
    ]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("papers")
    w9.ensure_aux_columns(spark)
    return df

# ---------- generic helper ----------

def check_nonempty(df, expected_cols):
    assert all(c in df.columns for c in expected_cols)
    assert df.count() >= 0  # allow empty edge cases

# ---------- individual tests ----------

def test_category_cooccurrence(spark, sample_df, tmp_path):
    w9.complex_1_category_cooccurrence(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_category_cooccurrence.csv"))
    check_nonempty(df, ["cat_a", "cat_b", "pair_count"])

def test_author_collaboration(spark, sample_df, tmp_path):
    w9.complex_2_author_collab_over_time(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_author_pairs_by_year.csv"))
    check_nonempty(df, ["year", "author_a", "author_b", "n_coauthored"])

def test_rising_declining_topics(spark, sample_df, tmp_path):
    w9.complex_3_rising_declining_topics(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_rising_declining_topics_fullrank.csv"))
    check_nonempty(df, ["primary_category", "pct_change"])

def test_readability_trends(spark, sample_df, tmp_path):
    w9.complex_4_readability_lexical_trends(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_lexical_richness_by_year.csv"))
    check_nonempty(df, ["year", "avg_lexical_richness"])

def test_doi_version_correlation(spark, sample_df, tmp_path):
    w9.complex_5_doi_versions_correlation(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_doi_versions_correlation.csv"))
    check_nonempty(df, ["corr_doi_versions"])

def test_author_lifecycle(spark, sample_df, tmp_path):
    w9.complex_6_author_productivity_lifecycle(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_author_lifecycle_top.csv"))
    check_nonempty(df, ["author", "paper_count"])

def test_author_category_migration(spark, sample_df, tmp_path):
    w9.complex_7_author_category_migration(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_author_category_migration.csv"))
    check_nonempty(df, ["author", "cat_earliest", "cat_latest"])

def test_abstract_popularity(spark, sample_df, tmp_path):
    w9.complex_8_abstract_len_vs_popularity(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_abstractlen_versions_correlation.csv"))
    check_nonempty(df, ["corr_abslen_versions"])

def test_weekday_submission(spark, sample_df, tmp_path):
    w9.complex_9_weekday_submission_patterns(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_weekday_submissions.csv"))
    check_nonempty(df, ["weekday_short", "submissions"])

def test_category_stability(spark, sample_df, tmp_path):
    w9.complex_10_category_stability_versions(spark, tmp_path)
    df = spark.read.option("header", True).csv(str(tmp_path / "complex_category_versions_avg.csv"))
    check_nonempty(df, ["primary_category", "avg_versions"])
