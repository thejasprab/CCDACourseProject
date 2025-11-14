#!/usr/bin/env python3
"""
Complex Spark SQL / DataFrame analyses for arXiv metadata.

This module is fully self-contained and does NOT import anything
from the old notebooks. It assumes you are running on the already
ingested Parquet produced by your ingestion pipeline.

Typical use
-----------
From Python:

    from engine.complex.complex_queries import run_complex_queries

    run_complex_queries(
        parquet_path="data/processed/arxiv_full",
        outdir="reports/analysis_full"
    )

From CLI:

    python -m engine.complex.complex_queries \
        --parquet data/processed/arxiv_full \
        --outdir reports/analysis_full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F


# ----------------------------------------------------------------------
# Helpers: filesystem + plotting
# ----------------------------------------------------------------------


def _ensure_outdir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_df_as_csv(df_spark, path: Path):
    """
    Save a small aggregated Spark DataFrame to CSV by collecting to pandas.
    """
    pdf = df_spark.toPandas()
    pdf.to_csv(path, index=False)
    print(f"[saved] {path}")
    return pdf  # return for plotting convenience


def matplotlib_savefig(path: Path):
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def _finalize_axes(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def barh_topn(
    pdf: pd.DataFrame,
    label_col: str,
    value_col: str,
    out_png: Path,
    title: str,
    topn: int = 30,
):
    if pdf.empty:
        return
    df = pdf[[label_col, value_col]].dropna()
    df = df.sort_values(value_col, ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(10, 0.35 * max(4, len(df))))
    ax.barh(df[label_col].astype(str), df[value_col])
    ax.invert_yaxis()
    _finalize_axes(ax, title, xlabel=value_col, ylabel=label_col)
    matplotlib_savefig(out_png)


def line_xy(
    pdf: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_png: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
):
    if pdf.empty:
        return
    df = pdf[[x_col, y_col]].dropna().sort_values(x_col)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x_col], df[y_col], marker="o")
    _finalize_axes(ax, title, xlabel or x_col, ylabel or y_col)
    matplotlib_savefig(out_png)


def scatter_xy(
    pdf: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_png: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
):
    if pdf.empty:
        return
    df = pdf[[x_col, y_col]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.7)
    _finalize_axes(ax, title, xlabel or x_col, ylabel or y_col)
    matplotlib_savefig(out_png)


def single_value_figure(
    value: float | int | str,
    out_png: Path,
    title: str,
    label: str = "",
):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    text = f"{label}: {value}" if label else f"{value}"
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=16)
    ax.set_title(title)
    matplotlib_savefig(out_png)


def weekday_sort(df: pd.DataFrame, day_col: str, daynum_col: str) -> pd.DataFrame:
    df[daynum_col] = pd.to_numeric(df[daynum_col], errors="coerce")
    return df.sort_values(daynum_col)


# ----------------------------------------------------------------------
# Aux view builder
# ----------------------------------------------------------------------


def ensure_aux_columns(spark: SparkSession) -> None:
    """
    Prepare helper columns on top of the `papers` view:

    Creates a new temp view `papers_enriched` with:

      - categories_list   : array<string> (from categories/primary_category)
      - n_versions        : int
      - doi_int           : 0/1 flag
    """
    cols = [c.name for c in spark.table("papers").schema]

    extra_exprs = []

    # categories_list
    if "categories_list" in cols:
        extra_exprs.append("categories_list")
    elif "categories" in cols:
        extra_exprs.append("SPLIT(categories, ' ') AS categories_list")
    else:
        # fallback to primary_category, will still work for many queries
        extra_exprs.append("ARRAY(primary_category) AS categories_list")

    # n_versions
    if "versions" in cols:
        extra_exprs.append("SIZE(versions) AS n_versions")
    else:
        extra_exprs.append("CAST(1 AS INT) AS n_versions")

    # doi_int
    if "has_doi" in cols:
        extra_exprs.append("CASE WHEN has_doi THEN 1 ELSE 0 END AS doi_int")
    elif "doi" in cols:
        extra_exprs.append(
            "CASE WHEN doi IS NOT NULL AND TRIM(doi) <> '' THEN 1 ELSE 0 END AS doi_int"
        )
    else:
        extra_exprs.append("CAST(0 AS INT) AS doi_int")

    select_extra = ", ".join(extra_exprs)
    query = f"SELECT *, {select_extra} FROM papers"

    spark.sql("DROP VIEW IF EXISTS papers_enriched")
    spark.sql(f"CREATE OR REPLACE TEMP VIEW papers_enriched AS {query}")


# ----------------------------------------------------------------------
# Complex queries (10)
# ----------------------------------------------------------------------


def complex_1_category_cooccurrence(spark: SparkSession, outdir: Path) -> None:
    """
    Interdisciplinary Category Co-occurrence:
    frequently co-occurring category pairs within the same paper.
    """
    print("\n[complex-1] Category co-occurrence pairs")

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
        WHERE c1 < c2
    """)

    df = spark.sql("""
        SELECT cat_a, cat_b, COUNT(*) AS pair_count
        FROM cat_pairs
        GROUP BY cat_a, cat_b
        HAVING pair_count > 1
        ORDER BY pair_count DESC, cat_a, cat_b
        LIMIT 200
    """)

    csv_path = outdir / "complex_category_cooccurrence.csv"
    pdf = save_df_as_csv(df, csv_path)

    if not pdf.empty:
        pdf["pair"] = pdf["cat_a"].astype(str) + " × " + pdf["cat_b"].astype(str)
        barh_topn(
            pdf,
            "pair",
            "pair_count",
            outdir / "complex_category_cooccurrence_top.png",
            "Top category co-occurrences",
            topn=30,
        )


def complex_2_author_collab_over_time(spark: SparkSession, outdir: Path) -> None:
    """
    Author collaboration network over time (author pairs per year).
    """
    print("\n[complex-2] Author collaboration pairs by year")

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

    csv_path = outdir / "complex_author_pairs_by_year.csv"
    pdf = save_df_as_csv(df, csv_path)

    if pdf.empty:
        return

    # per-year totals
    per_year = pdf.groupby("year", as_index=False)["n_coauthored"].sum()
    line_xy(
        per_year,
        "year",
        "n_coauthored",
        outdir / "complex_author_pairs_by_year_totals.png",
        "Total coauthored pairs (n>=2) per year",
        xlabel="year",
        ylabel="pair count (sum n_coauthored)",
    )

    # top 20 pairs (author_a × author_b (year))
    top_pairs = pdf.copy()
    top_pairs["pair"] = (
        top_pairs["author_a"].astype(str)
        + " × "
        + top_pairs["author_b"].astype(str)
        + " ("
        + top_pairs["year"].astype(str)
        + ")"
    )
    barh_topn(
        top_pairs,
        "pair",
        "n_coauthored",
        outdir / "complex_author_pairs_by_year_top20.png",
        "Top author pairs by year (n_coauthored ≥ 2)",
        topn=20,
    )


def complex_3_rising_declining_topics(spark: SparkSession, outdir: Path) -> None:
    """
    Rising and declining topics:
    compare earliest vs latest year volume per primary_category.
    """
    print("\n[complex-3] Rising and declining topics")

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
          CASE WHEN c_start > 0 THEN ROUND((c_end - c_start) * 100.0 / c_start, 2)
               ELSE NULL
          END AS pct_change
        FROM endpoints
        WHERE c_start IS NOT NULL AND c_end IS NOT NULL
        ORDER BY pct_change DESC NULLS LAST
    """)

    full_csv = outdir / "complex_rising_declining_topics_fullrank.csv"
    pdf_full = save_df_as_csv(df, full_csv)

    if pdf_full.empty:
        return

    # Top 20 rising & declining
    rising = df.orderBy(F.col("pct_change").desc_nulls_last()).limit(20)
    declining = df.orderBy(F.col("pct_change").asc_nulls_last()).limit(20)

    pdf_rise = save_df_as_csv(rising, outdir / "complex_rising_topics_top20.csv")
    pdf_decl = save_df_as_csv(declining, outdir / "complex_declining_topics_top20.csv")

    if not pdf_rise.empty:
        barh_topn(
            pdf_rise,
            "primary_category",
            "pct_change",
            outdir / "complex_rising_topics_top20.png",
            "Top 20 rising topics by % change",
        )

    if not pdf_decl.empty:
        pdf_decl_plot = pdf_decl.sort_values("pct_change", ascending=True)
        barh_topn(
            pdf_decl_plot,
            "primary_category",
            "pct_change",
            outdir / "complex_declining_topics_top20.png",
            "Top 20 declining topics by % change",
        )


def complex_4_readability_lexical_trends(spark: SparkSession, outdir: Path) -> None:
    """
    Abstract readability proxy via lexical richness over time:
    lexical_richness = distinct_token_count / token_count
    """
    print("\n[complex-4] Readability / Lexical richness trends")

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

    csv = outdir / "complex_lexical_richness_by_year.csv"
    pdf = save_df_as_csv(df, csv)

    if pdf.empty:
        return

    line_xy(
        pdf,
        "year",
        "avg_lexical_richness",
        outdir / "complex_lexical_richness_by_year.png",
        "Average lexical richness by year",
        xlabel="year",
        ylabel="avg lexical richness",
    )

    line_xy(
        pdf,
        "year",
        "avg_token_count",
        outdir / "complex_avg_token_count_by_year.png",
        "Average abstract length (tokens) by year",
        xlabel="year",
        ylabel="avg tokens",
    )


def complex_5_doi_versions_correlation(spark: SparkSession, outdir: Path) -> None:
    """
    DOI vs Version correlation:
    average number of versions for DOI vs non-DOI, plus Pearson correlation.
    """
    print("\n[complex-5] DOI vs versions correlation")

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

    csv_group = outdir / "complex_doi_vs_versions_group.csv"
    pdf_group = save_df_as_csv(group_df, csv_group)

    corr_df = spark.sql("""
        SELECT CORR(CAST(doi_int AS DOUBLE), CAST(n_versions AS DOUBLE)) AS corr_doi_versions
        FROM doi_versions
    """)

    csv_corr = outdir / "complex_doi_versions_correlation.csv"
    pdf_corr = save_df_as_csv(corr_df, csv_corr)

    if not pdf_group.empty:
        pdf_group["doi_label"] = np.where(
            pdf_group["doi_int"].astype(int) == 1, "DOI present", "No DOI"
        )
        barh_topn(
            pdf_group,
            "doi_label",
            "avg_versions",
            outdir / "complex_doi_vs_versions_group.png",
            "Avg # versions by DOI presence",
            topn=2,
        )

    if not pdf_corr.empty:
        val = float(pdf_corr["corr_doi_versions"].iloc[0])
        single_value_figure(
            round(val, 4),
            outdir / "complex_doi_versions_correlation.png",
            "Pearson correlation",
            label="corr(doi_int, n_versions)",
        )


def complex_6_author_productivity_lifecycle(
    spark: SparkSession, outdir: Path
) -> None:
    """
    Author productivity lifespan and volume:
    first_year, last_year, active span, and paper count.
    """
    print("\n[complex-6] Author productivity lifecycle")

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

    csv = outdir / "complex_author_lifecycle_top.csv"
    pdf = save_df_as_csv(df, csv)

    if pdf.empty:
        return

    scatter_xy(
        pdf,
        "active_span_years",
        "paper_count",
        outdir / "complex_author_lifecycle_scatter.png",
        "Author lifecycle: span vs paper count",
        xlabel="active span (years)",
        ylabel="# papers",
    )


def complex_7_author_category_migration(
    spark: SparkSession, outdir: Path
) -> None:
    """
    Author category migration:
    dominant category earliest vs latest year, for authors who change.
    """
    print("\n[complex-7] Author category migration")

    cols = [c.name for c in spark.table("papers_enriched").schema]
    if (
        "authors_list" not in cols
        or "year" not in cols
        or "primary_category" not in cols
    ):
        print("[skip] need authors_list, year, primary_category")
        return

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_cat_year AS
        SELECT
          author,
          year,
          primary_category,
          COUNT(*) AS c
        FROM (
          SELECT
            year,
            primary_category,
            author
          FROM papers_enriched
          LATERAL VIEW OUTER EXPLODE(authors_list) a AS author
        ) t
        WHERE author IS NOT NULL AND TRIM(author) <> ''
        GROUP BY author, year, primary_category
    """)

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_bounds AS
        SELECT
          author,
          MIN(year) AS y_min,
          MAX(year) AS y_max
        FROM author_cat_year
        GROUP BY author
    """)

    # earliest
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_earliest AS
        SELECT author, cat_earliest FROM (
          SELECT
            acy.author,
            acy.primary_category AS cat_earliest,
            ROW_NUMBER() OVER (
              PARTITION BY acy.author
              ORDER BY acy.c DESC, acy.primary_category
            ) AS rn
          FROM author_cat_year acy
          JOIN author_bounds b
            ON b.author = acy.author AND acy.year = b.y_min
        ) t
        WHERE rn = 1
    """)

    # latest
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW author_latest AS
        SELECT author, cat_latest FROM (
          SELECT
            acy.author,
            acy.primary_category AS cat_latest,
            ROW_NUMBER() OVER (
              PARTITION BY acy.author
              ORDER BY acy.c DESC, acy.primary_category
            ) AS rn
          FROM author_cat_year acy
          JOIN author_bounds b
            ON b.author = acy.author AND acy.year = b.y_max
        ) t
        WHERE rn = 1
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

    csv = outdir / "complex_author_category_migration.csv"
    pdf = save_df_as_csv(df, csv)

    if pdf.empty:
        return

    # top 20 transitions
    trans = (
        pdf.groupby(["cat_earliest", "cat_latest"])
        .size()
        .reset_index(name="n")
    )
    trans["transition"] = (
        trans["cat_earliest"].astype(str)
        + " → "
        + trans["cat_latest"].astype(str)
    )
    barh_topn(
        trans,
        "transition",
        "n",
        outdir / "complex_author_category_migration_top20.png",
        "Top 20 author category migrations",
        topn=20,
    )


def complex_8_abstract_len_vs_popularity(
    spark: SparkSession, outdir: Path
) -> None:
    """
    Abstract length vs popularity proxy (n_versions):
      - correlation
      - decile analysis by abstract_len
    """
    print("\n[complex-8] Abstract length vs popularity (versions)")

    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "abstract_len" not in cols:
        print("[skip] abstract_len missing")
        return

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW abs_pop AS
        SELECT
          CAST(abstract_len AS DOUBLE) AS abstract_len,
          CAST(n_versions AS DOUBLE) AS n_versions
        FROM papers_enriched
        WHERE abstract_len IS NOT NULL
    """)

    corr_df = spark.sql("""
        SELECT CORR(abstract_len, n_versions) AS corr_abslen_versions
        FROM abs_pop
    """)

    csv_corr = outdir / "complex_abstractlen_versions_correlation.csv"
    pdf_corr = save_df_as_csv(corr_df, csv_corr)

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW abs_with_bucket AS
        SELECT
          abstract_len,
          n_versions,
          NTILE(10) OVER (ORDER BY abstract_len) AS abslen_decile
        FROM abs_pop
    """)

    bucket_df = spark.sql("""
        SELECT
          abslen_decile,
          AVG(abstract_len) AS avg_abslen,
          AVG(n_versions) AS avg_versions
        FROM abs_with_bucket
        GROUP BY abslen_decile
        ORDER BY abslen_decile
    """)

    csv_bucket = outdir / "complex_abstractlen_versions_by_decile.csv"
    pdf_bucket = save_df_as_csv(bucket_df, csv_bucket)

    if not pdf_corr.empty:
        val = float(pdf_corr["corr_abslen_versions"].iloc[0])
        single_value_figure(
            round(val, 4),
            outdir / "complex_abstractlen_versions_correlation.png",
            "Pearson correlation",
            label="corr(abstract_len, n_versions)",
        )

    if not pdf_bucket.empty:
        line_xy(
            pdf_bucket,
            "abslen_decile",
            "avg_versions",
            outdir / "complex_abstractlen_versions_by_decile.png",
            "Avg # versions by abstract length decile",
            xlabel="abstract length decile (1=short)",
            ylabel="avg # versions",
        )


def complex_9_weekday_submission_patterns(
    spark: SparkSession, outdir: Path
) -> None:
    """
    Weekday vs weekend submission patterns.
    """
    print("\n[complex-9] Weekday submission patterns")

    cols = [c.name for c in spark.table("papers_enriched").schema]
    if "submitted_date" not in cols:
        print("[skip] submitted_date missing")
        return

    df = spark.sql("""
        SELECT
          DATE_FORMAT(submitted_date, 'E') AS weekday_short,
          DATE_FORMAT(submitted_date, 'u') AS weekday_num,
          COUNT(*) AS submissions
        FROM papers_enriched
        GROUP BY DATE_FORMAT(submitted_date, 'E'), DATE_FORMAT(submitted_date, 'u')
        ORDER BY CAST(weekday_num AS INT)
    """)

    csv = outdir / "complex_weekday_submissions.csv"
    pdf = save_df_as_csv(df, csv)

    if pdf.empty:
        return

    pdf_plot = weekday_sort(pdf.copy(), "weekday_short", "weekday_num")
    line_xy(
        pdf_plot,
        "weekday_short",
        "submissions",
        outdir / "complex_weekday_submissions.png",
        "Submissions by weekday",
        xlabel="weekday",
        ylabel="submissions",
    )


def complex_10_category_stability_versions(
    spark: SparkSession, outdir: Path
) -> None:
    """
    Category stability via versions:
    average # versions per primary_category.
    """
    print("\n[complex-10] Category stability via versions")

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
        HAVING n_papers >= 20
        ORDER BY avg_versions DESC, n_papers DESC
    """)

    csv = outdir / "complex_category_versions_avg.csv"
    pdf = save_df_as_csv(df, csv)

    if pdf.empty:
        return

    barh_topn(
        pdf,
        "primary_category",
        "avg_versions",
        outdir / "complex_category_versions_avg_top30.png",
        "Avg # versions by primary category (n_papers ≥ 20)",
        topn=30,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def run_complex_queries(parquet_path: str, outdir: str | Path) -> None:
    """
    Entry point used by pipelines:

    - Load parquet (already ingested arXiv data).
    - Register `papers` temp view.
    - Build `papers_enriched`.
    - Run all 10 complex analyses.
    - Save CSV + PNG into outdir.
    """
    outdir_path = _ensure_outdir(outdir)

    spark = (
        SparkSession.builder.appName("arxiv_complex_engine")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    print(f"[load] reading parquet from {parquet_path}")
    df = spark.read.parquet(parquet_path)
    df.createOrReplaceTempView("papers")

    ensure_aux_columns(spark)

    complex_1_category_cooccurrence(spark, outdir_path)
    complex_2_author_collab_over_time(spark, outdir_path)
    complex_3_rising_declining_topics(spark, outdir_path)
    complex_4_readability_lexical_trends(spark, outdir_path)
    complex_5_doi_versions_correlation(spark, outdir_path)
    complex_6_author_productivity_lifecycle(spark, outdir_path)
    complex_7_author_category_migration(spark, outdir_path)
    complex_8_abstract_len_vs_popularity(spark, outdir_path)
    complex_9_weekday_submission_patterns(spark, outdir_path)
    complex_10_category_stability_versions(spark, outdir_path)

    print(f"\n[done] Complex analytics written to {outdir_path}/")
    spark.stop()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        required=True,
        help="Path to ingested Parquet (e.g. data/processed/arxiv_full)",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory for analysis CSVs + PNGs (e.g. reports/analysis_full)",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_complex_queries(args.parquet, args.outdir)


if __name__ == "__main__":
    main()
