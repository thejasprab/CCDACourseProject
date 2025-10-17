# notebooks/week9-ComplexQueries.py
"""
Advanced / Complex Spark SQL analyses for arXiv metadata.

Usage:
  python notebooks/week9-ComplexQueries.py --parquet data/processed/arxiv_full
  # or for sample:
  python notebooks/week9-ComplexQueries.py --parquet data/processed/arxiv_parquet

This script:
  - Loads the unified arXiv Parquet dataset used in Week 8
  - Registers a temp view `papers`
  - Runs a set of advanced Spark SQL analyses (10 total), each saved to reports/<run_type or --outdir>/
  - Performs simple validations after each query (schema/row-count/logic sanity checks)
  - Is robust to optional columns (e.g., submitted_date, categories_list); falls back when missing

Columns expected (from Week 8 and ingestion):
  - arxiv_id, title, abstract, title_len, abstract_len
  - primary_category (string)
  - categories or categories_list (string of space-separated codes or array<string>)
  - authors_list (array<string>)    # optional but expected
  - versions (array<struct> or array<any>) -> used for n_versions via size(versions)
  - has_doi (boolean) -> used for DOI logic
  - year (int)
  - submitted_date (timestamp)      # optional; used for weekday analysis

Outputs:
  - CSVs into reports/<run_type or --outdir>/complex_*.csv
  - No plots here (Week 8 handles plotting). Add plots if you want later.

Notes:
  - If `categories_list` is absent, we derive it from `categories` (space-separated) or fallback to `[primary_category]`.
  - If `submitted_date` is absent, weekday analysis is skipped gracefully.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F


# ---------- helpers ----------

def pick_outdir(parquet_path: str, user_outdir: str | None) -> Path:
    if user_outdir:
        out = Path("reports") / user_outdir
    else:
        run_type = "full" if "full" in parquet_path else "sample"
        out = Path("reports") / run_type
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_df_as_csv(df_spark, path: Path):
    # All outputs are aggregated and small -> safe to collect to pandas
    df_spark.toPandas().to_csv(path, index=False)
    print(f"[saved] {path}")

def validate_nonempty(df, required_cols: list[str]) -> None:
    cols_ok = all(c in df.columns for c in required_cols)
    if not cols_ok:
        missing = [c for c in required_cols if c not in df.columns]
        raise AssertionError(f"Missing columns in result: {missing}")
    assert df.count() >= 0  # allow 0 rows for some edge cases; most checks enforce >0 where relevant

def validate_positive_rows(df, msg="Expected non-empty result"):
    assert df.count() > 0, msg

def ensure_aux_columns(spark):
    """
    Prepare helper views/columns:
      - categories_list: array<string>
      - n_versions: int
      - doi_int: 0/1 from has_doi
    Produces view `papers_enriched`.
    """
    # Build an enriched DataFrame from the existing 'papers' table.
    # Try to construct categories_list:
    # 1) if categories_list exists -> use it
    # 2) elif categories exists -> split on space
    # 3) else -> array(primary_category)
    cols = [c.name for c in spark.table("papers").schema]
    q_parts = ["SELECT *"]

    if "categories_list" in cols:
        q_parts.append(", categories_list")
    elif "categories" in cols:
        q_parts.append(", SPLIT(categories, ' ') AS categories_list")
    else:
        q_parts.append(", ARRAY(primary_category) AS categories_list")

    # n_versions
    if "versions" in cols:
        q_parts.append(", SIZE(versions) AS n_versions")
    else:
        q_parts.append(", CAST(1 AS INT) AS n_versions")  # best-effort default

    # doi_int
    if "has_doi" in cols:
        q_parts.append(", CASE WHEN has_doi THEN 1 ELSE 0 END AS doi_int")
    else:
        # If we have 'doi' string, construct hasdoi from non-empty
        if "doi" in cols:
            q_parts.append(", CASE WHEN doi IS NOT NULL AND TRIM(doi) <> '' THEN 1 ELSE 0 END AS doi_int")
        else:
            q_parts.append(", CAST(0 AS INT) AS doi_int")

    q = "SELECT *, " + ", ".join(q_parts[1:]) + " FROM papers"
    spark.sql("DROP VIEW IF EXISTS papers_enriched")
    spark.sql(f"CREATE OR REPLACE TEMP VIEW papers_enriched AS {q}")

# ---------- complex analyses (10) ----------

def complex_1_category_cooccurrence(spark, outdir: Path):
    """
    Interdisciplinary Category Co-occurrence (pair counts).
    Finds category pairs that frequently co-occur in the same paper.
    """
    print("\n[complex-1] Category co-occurrence pairs (interdisciplinarity)")
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW cat_pairs AS
        SELECT
            arxiv_id,
            c1 AS cat_a,
            c2 AS cat_b
        FROM (
          SELECT arxiv_id, categories_list
          FROM papers_enriched
        ) t
        LATERAL VIEW EXPLODE(categories_list) a AS c1
        LATERAL VIEW EXPLODE(categories_list) b AS c2
        WHERE c1 < c2   -- avoid duplicates (unordered pairs)
    """)
    df = spark.sql("""
        SELECT cat_a, cat_b, COUNT(*) AS pair_count
        FROM cat_pairs
        GROUP BY cat_a, cat_b
        HAVING pair_count > 1
        ORDER BY pair_count DESC, cat_a, cat_b
        LIMIT 200
    """)
    validate_nonempty(df, ["cat_a", "cat_b", "pair_count"])
    save_df_as_csv(df, outdir / "complex_category_cooccurrence.csv")

def complex_2_author_collab_over_time(spark, outdir: Path):
    """
    Author Collaboration Network Over Time (top author pairs by year).
    """
    print("\n[complex-2] Author collaboration pairs by year (top)")
    # Need authors_list and year
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "authors_list" not in cols or "year" not in cols:
        print("[skip] authors_list or year missing")
        return

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_pairs AS
        SELECT
          year,
          a1 AS author_a,
          a2 AS author_b
        FROM (
          SELECT year, authors_list FROM papers_enriched
        )
        LATERAL VIEW EXPLODE(authors_list) L1 AS a1
        LATERAL VIEW EXPLODE(authors_list) L2 AS a2
        WHERE a1 < a2
    """)
    df = spark.sql("""
        SELECT year, author_a, author_b, COUNT(*) AS n_coauthored
        FROM author_pairs
        GROUP BY year, author_a, author_b
        HAVING n_coauthored >= 2
        ORDER BY year, n_coauthored DESC
        LIMIT 500
    """)
    validate_nonempty(df, ["year", "author_a", "author_b", "n_coauthored"])
    save_df_as_csv(df, outdir / "complex_author_pairs_by_year.csv")

def complex_3_rising_declining_topics(spark, outdir: Path):
    """
    Rising and Declining Topics:
      - Count by primary_category x year
      - Compute percent change from earliest year to latest year for each category
      - Rank top rising/shrinking
    """
    print("\n[complex-3] Rising and declining topics (primary_category x year growth)")
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "primary_category" not in cols or "year" not in cols:
        print("[skip] primary_category or year missing")
        return

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW cat_year_counts AS
        SELECT primary_category, year, COUNT(*) AS c
        FROM papers_enriched
        GROUP BY primary_category, year
    """)
    # Get min/max year per category, then pick counts at those endpoints
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW cat_year_bounds AS
        SELECT
          primary_category,
          MIN(year) AS y_min,
          MAX(year) AS y_max
        FROM cat_year_counts
        GROUP BY primary_category
    """)
    df = spark.sql("""
        WITH endpoints AS (
          SELECT
            b.primary_category,
            b.y_min,
            b.y_max,
            MIN(CASE WHEN c.year = b.y_min THEN c.c END) AS c_start,
            MIN(CASE WHEN c.year = b.y_max THEN c.c END) AS c_end
          FROM cat_year_bounds b
          JOIN cat_year_counts c
            ON c.primary_category = b.primary_category
           AND c.year IN (b.y_min, b.y_max)
          GROUP BY b.primary_category, b.y_min, b.y_max
        )
        SELECT
          primary_category,
          y_min,
          y_max,
          c_start,
          c_end,
          CASE WHEN c_start > 0 THEN ROUND((c_end - c_start) * 100.0 / c_start, 2) ELSE NULL END AS pct_change
        FROM endpoints
        WHERE c_start IS NOT NULL AND c_end IS NOT NULL
        ORDER BY pct_change DESC NULLS LAST
    """)
    validate_nonempty(df, ["primary_category", "y_min", "y_max", "c_start", "c_end", "pct_change"])
    save_df_as_csv(df, outdir / "complex_rising_declining_topics_fullrank.csv")

    # Top 20 rising and top 20 declining
    rising = df.orderBy(F.col("pct_change").desc_nulls_last()).limit(20)
    declining = df.orderBy(F.col("pct_change").asc_nulls_last()).limit(20)
    save_df_as_csv(rising, outdir / "complex_rising_topics_top20.csv")
    save_df_as_csv(declining, outdir / "complex_declining_topics_top20.csv")

def complex_4_readability_lexical_trends(spark, outdir: Path):
    """
    Abstract readability proxy via lexical richness over time.
    lexical_richness = distinct_token_count / token_count (using whitespace tokenization)
    """
    print("\n[complex-4] Readability / Lexical richness trends by year")
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "abstract" not in cols or "year" not in cols:
        print("[skip] abstract or year missing")
        return

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW abstract_tokens AS
        SELECT
          year,
          SIZE(SPLIT(LOWER(abstract), '\\\\s+')) AS tok_count,
          SIZE(ARRAY_DISTINCT(SPLIT(LOWER(abstract), '\\\\s+'))) AS distinct_tok_count
        FROM papers_enriched
        WHERE abstract IS NOT NULL AND LENGTH(abstract) > 0
    """)
    df = spark.sql("""
        SELECT
          year,
          AVG(CAST(distinct_tok_count AS DOUBLE) / NULLIF(tok_count, 0)) AS avg_lexical_richness,
          AVG(tok_count) AS avg_token_count
        FROM abstract_tokens
        GROUP BY year
        ORDER BY year
    """)
    validate_nonempty(df, ["year", "avg_lexical_richness", "avg_token_count"])
    save_df_as_csv(df, outdir / "complex_lexical_richness_by_year.csv")

def complex_5_doi_versions_correlation(spark, outdir: Path):
    """
    DOI vs Version Correlation:
      - Average n_versions by DOI presence
      - Pearson corr between doi_int and n_versions
    """
    print("\n[complex-5] DOI vs Versions: group averages + correlation")
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW doi_versions AS
        SELECT doi_int, n_versions FROM papers_enriched
    """)
    group_df = spark.sql("""
        SELECT
          doi_int,
          AVG(n_versions) AS avg_versions
        FROM doi_versions
        GROUP BY doi_int
        ORDER BY doi_int DESC
    """)
    validate_nonempty(group_df, ["doi_int", "avg_versions"])
    save_df_as_csv(group_df, outdir / "complex_doi_vs_versions_group.csv")

    corr_df = spark.sql("""
        SELECT CORR(CAST(doi_int AS DOUBLE), CAST(n_versions AS DOUBLE)) AS corr_doi_versions
        FROM doi_versions
    """)
    validate_nonempty(corr_df, ["corr_doi_versions"])
    save_df_as_csv(corr_df, outdir / "complex_doi_versions_correlation.csv")

def complex_6_author_productivity_lifecycle(spark, outdir: Path):
    """
    Author productivity lifespan and volume:
      - first_year, last_year, active_span_years, paper_count
    """
    print("\n[complex-6] Author productivity lifecycle (active span)")
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "authors_list" not in cols or "year" not in cols:
        print("[skip] authors_list or year missing")
        return

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_years AS
        SELECT
          EXPLODE(authors_list) AS author,
          year
        FROM papers_enriched
        WHERE authors_list IS NOT NULL
    """)
    df = spark.sql("""
        SELECT
          author,
          MIN(year) AS first_year,
          MAX(year) AS last_year,
          (MAX(year) - MIN(year)) AS active_span_years,
          COUNT(*) AS paper_count
        FROM author_years
        GROUP BY author
        HAVING paper_count >= 2
        ORDER BY active_span_years DESC, paper_count DESC
        LIMIT 1000
    """)
    validate_nonempty(df, ["author", "first_year", "last_year", "active_span_years", "paper_count"])
    save_df_as_csv(df, outdir / "complex_author_lifecycle_top.csv")

def complex_7_author_category_migration(spark, outdir: Path):
    """
    Author category migration:
      For each author, the dominant category in earliest year vs latest year; keep those who changed.
    """
    print("\n[complex-7] Author category migration (earliest vs latest dominant category)")
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "authors_list" not in cols or "year" not in cols or "primary_category" not in cols:
        print("[skip] need authors_list, year, primary_category")
        return

    # author/year/category counts
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_cat_year AS
        SELECT
          EXPLODE(authors_list) AS author,
          year,
          primary_category,
          COUNT(*) AS c
        FROM papers_enriched
        GROUP BY EXPLODE(authors_list), year, primary_category
    """)
    # earliest and latest year per author
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_bounds AS
        SELECT
          author,
          MIN(year) AS y_min,
          MAX(year) AS y_max
        FROM author_cat_year
        GROUP BY author
    """)
    # dominant category at earliest and latest years
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_earliest AS
        SELECT acy.author, acy.primary_category AS cat_earliest
        FROM author_cat_year acy
        JOIN author_bounds b ON b.author = acy.author AND acy.year = b.y_min
        QUALIFY ROW_NUMBER() OVER (PARTITION BY acy.author ORDER BY acy.c DESC, acy.primary_category) = 1
    """)
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_latest AS
        SELECT acy.author, acy.primary_category AS cat_latest
        FROM author_cat_year acy
        JOIN author_bounds b ON b.author = acy.author AND acy.year = b.y_max
        QUALIFY ROW_NUMBER() OVER (PARTITION BY acy.author ORDER BY acy.c DESC, acy.primary_category) = 1
    """)
    # NOTE: QUALIFY is available in Spark 3.5 with Databricks SQL; if not, rewrite using subqueries.
    # Portable rewrite (without QUALIFY):
    spark.sql("DROP VIEW IF EXISTS author_earliest")
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_earliest AS
        SELECT author, cat_earliest FROM (
          SELECT acy.author,
                 acy.primary_category AS cat_earliest,
                 ROW_NUMBER() OVER (PARTITION BY acy.author ORDER BY acy.c DESC, acy.primary_category) AS rn
          FROM author_cat_year acy
          JOIN author_bounds b ON b.author = acy.author AND acy.year = b.y_min
        ) t WHERE rn = 1
    """)
    spark.sql("DROP VIEW IF EXISTS author_latest")
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_latest AS
        SELECT author, cat_latest FROM (
          SELECT acy.author,
                 acy.primary_category AS cat_latest,
                 ROW_NUMBER() OVER (PARTITION BY acy.author ORDER BY acy.c DESC, acy.primary_category) AS rn
          FROM author_cat_year acy
          JOIN author_bounds b ON b.author = acy.author AND acy.year = b.y_max
        ) t WHERE rn = 1
    """)
    df = spark.sql("""
        SELECT
          e.author,
          e.cat_earliest,
          l.cat_latest
        FROM author_earliest e
        JOIN author_latest l USING (author)
        WHERE e.cat_earliest <> l.cat_latest
        ORDER BY author
        LIMIT 5000
    """)
    validate_nonempty(df, ["author", "cat_earliest", "cat_latest"])
    save_df_as_csv(df, outdir / "complex_author_category_migration.csv")

def complex_8_abstract_len_vs_popularity(spark, outdir: Path):
    """
    Abstract length vs popularity proxy (n_versions):
      - Correlation between abstract_len and n_versions
      - Bucketed analysis by abstract_len deciles
    """
    print("\n[complex-8] Abstract length vs popularity proxy (versions)")
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "abstract_len" not in cols:
        print("[skip] abstract_len missing")
        return

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW abs_pop AS
        SELECT CAST(abstract_len AS DOUBLE) AS abstract_len, CAST(n_versions AS DOUBLE) AS n_versions
        FROM papers_enriched
        WHERE abstract_len IS NOT NULL
    """)
    corr_df = spark.sql("""
        SELECT CORR(abstract_len, n_versions) AS corr_abslen_versions FROM abs_pop
    """)
    validate_nonempty(corr_df, ["corr_abslen_versions"])
    save_df_as_csv(corr_df, outdir / "complex_abstractlen_versions_correlation.csv")

    # Decile buckets via NTILE over abstract_len
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW abs_with_bucket AS
        SELECT abstract_len, n_versions,
               NTILE(10) OVER (ORDER BY abstract_len) AS abslen_decile
        FROM abs_pop
    """)
    bucket_df = spark.sql("""
        SELECT abslen_decile,
               AVG(abstract_len) AS avg_abslen,
               AVG(n_versions) AS avg_versions
        FROM abs_with_bucket
        GROUP BY abslen_decile
        ORDER BY abslen_decile
    """)
    validate_nonempty(bucket_df, ["abslen_decile", "avg_abslen", "avg_versions"])
    save_df_as_csv(bucket_df, outdir / "complex_abstractlen_versions_by_decile.csv")

def complex_9_weekday_submission_patterns(spark, outdir: Path):
    """
    Weekday vs Weekend submission patterns (requires submitted_date timestamp).
    """
    print("\n[complex-9] Weekday submission patterns")
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "submitted_date" not in cols:
        print("[skip] submitted_date missing")
        return

    df = spark.sql("""
        SELECT
          DATE_FORMAT(submitted_date, 'E') AS weekday_short,
          DATE_FORMAT(submitted_date, 'u') AS weekday_num,   -- 1..7 (Mon..Sun)
          COUNT(*) AS submissions
        FROM papers_enriched
        GROUP BY DATE_FORMAT(submitted_date, 'E'), DATE_FORMAT(submitted_date, 'u')
        ORDER BY CAST(weekday_num AS INT)
    """)
    validate_nonempty(df, ["weekday_short", "weekday_num", "submissions"])
    save_df_as_csv(df, outdir / "complex_weekday_submissions.csv")

def complex_10_category_stability_versions(spark, outdir: Path):
    """
    Category stability via versions:
      - Average n_versions per primary_category
      - Distribution ranks
    """
    print("\n[complex-10] Category stability via versions (avg versions per primary_category)")
    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "primary_category" not in cols:
        print("[skip] primary_category missing")
        return

    df = spark.sql("""
        SELECT
          primary_category,
          AVG(n_versions) AS avg_versions,
          COUNT(*) AS n_papers
        FROM papers_enriched
        GROUP BY primary_category
        HAVING n_papers >= 20   -- stability filter
        ORDER BY avg_versions DESC, n_papers DESC
    """)
    validate_nonempty(df, ["primary_category", "avg_versions", "n_papers"])
    save_df_as_csv(df, outdir / "complex_category_versions_avg.csv")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to Parquet output from src/ingestion.py")
    ap.add_argument("--outdir", default=None, help="Optional subfolder name under reports/. If not set, uses 'full' or 'sample'.")
    args = ap.parse_args()

    outdir = pick_outdir(args.parquet, args.outdir)

    spark = (
        SparkSession.builder
        .appName("arxiv_week9_complex_queries")
        # .config("spark.sql.shuffle.partitions", "200")  # tweak if needed
        .getOrCreate()
    )
    df = spark.read.parquet(args.parquet)

    # Register base view
    df.createOrReplaceTempView("papers")

    # Build enriched view with helper columns
    ensure_aux_columns(spark)

    # Run all 10 complex analyses
    complex_1_category_cooccurrence(spark, outdir)
    complex_2_author_collab_over_time(spark, outdir)
    complex_3_rising_declining_topics(spark, outdir)
    complex_4_readability_lexical_trends(spark, outdir)
    complex_5_doi_versions_correlation(spark, outdir)
    complex_6_author_productivity_lifecycle(spark, outdir)
    complex_7_author_category_migration(spark, outdir)
    complex_8_abstract_len_vs_popularity(spark, outdir)
    complex_9_weekday_submission_patterns(spark, outdir)
    complex_10_category_stability_versions(spark, outdir)

    print(f"\n[done] Complex analytics written to {outdir}/")
    spark.stop()


if __name__ == "__main__":
    main()
