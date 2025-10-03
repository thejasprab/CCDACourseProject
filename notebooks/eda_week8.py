# notebooks/eda_week8.py
"""
Comprehensive EDA for the arXiv metadata after ingestion.
Run:
  python notebooks/eda_week8.py --parquet data/processed/arxiv_parquet

This will:
  - Print schema and table-level summaries
  - Produce CSV summaries in reports/
  - Save several matplotlib charts in reports/*.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F, Window
from pyspark.sql import SparkSession


# ---------- helpers ----------

def ensure_dirs():
    Path("reports").mkdir(parents=True, exist_ok=True)

def save_df_as_csv(df_spark, path: str):
    df_spark.toPandas().to_csv(path, index=False)
    print(f"[saved] {path}")

def matplotlib_savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


# ---------- core EDA ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to Parquet output from src/ingestion.py")
    ap.add_argument("--topk", type=int, default=20, help="Top-K categories/authors to show")
    args = ap.parse_args()

    ensure_dirs()

    spark = SparkSession.builder.appName("arxiv_week8_eda").getOrCreate()
    df = spark.read.parquet(args.parquet)

    # Basic info
    print("\n=== SCHEMA ===")
    df.printSchema()

    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"\n=== SIZE ===\nrows={n_rows:,}  cols={n_cols}")

    # Completeness / nulls per column
    print("\n=== COMPLETENESS (non-null %) ===")
    completeness = []
    for c in df.columns:
        non_null = df.where(F.col(c).isNotNull()).count()
        completeness.append((c, non_null, non_null / n_rows * 100.0))
    comp_pd = pd.DataFrame(completeness, columns=["column", "non_null", "non_null_pct"]).sort_values("non_null_pct", ascending=False)
    comp_pd.to_csv("reports/completeness.csv", index=False)
    print(comp_pd.head(20))
    print("[saved] reports/completeness.csv")

    # Distinct counts for selected fields
    print("\n=== DISTINCT COUNTS (selected) ===")
    selected = ["arxiv_id", "primary_category", "year", "doi", "submitter"]
    distinct_rows = []
    for c in selected:
        if c in df.columns:
            distinct_rows.append((c, df.select(c).distinct().count()))
    distinct_pd = pd.DataFrame(distinct_rows, columns=["column", "distinct_count"]).sort_values("distinct_count", ascending=False)
    distinct_pd.to_csv("reports/distinct_selected.csv", index=False)
    print(distinct_pd)
    print("[saved] reports/distinct_selected.csv")

    # Basic descriptive stats for text lengths
    print("\n=== TEXT LENGTH STATS (title_len, abstract_len) ===")
    num_stats = (
        df.select("title_len", "abstract_len")
          .summary("count", "min", "25%", "50%", "75%", "max", "mean")
    )
    save_df_as_csv(num_stats, "reports/text_length_summary.csv")
    num_stats.show(truncate=False)

    # Top categories
    print("\n=== TOP CATEGORIES ===")
    topcats = (
        df.groupBy("primary_category")
          .count()
          .orderBy(F.desc("count"))
          .limit(args.topk)
    )
    save_df_as_csv(topcats, "reports/top_categories.csv")
    topcats_pd = topcats.toPandas()

    # Plot: Top categories (bar)
    if not topcats_pd.empty:
        plt.figure(figsize=(10, 5))
        topcats_pd = topcats_pd.sort_values("count", ascending=False)
        plt.bar(topcats_pd["primary_category"], topcats_pd["count"])
        plt.xticks(rotation=60, ha="right")
        plt.title(f"Top {len(topcats_pd)} Primary Categories")
        plt.xlabel("primary_category")
        plt.ylabel("count")
        matplotlib_savefig("reports/top_categories.png")

    # Papers per year (line)
    print("\n=== PAPERS PER YEAR ===")
    by_year = df.groupBy("year").count().orderBy("year")
    save_df_as_csv(by_year, "reports/by_year.csv")
    by_year_pd = by_year.toPandas()
    by_year_pd = by_year_pd.dropna()
    if not by_year_pd.empty:
        plt.figure(figsize=(9, 4))
        plt.plot(by_year_pd["year"], by_year_pd["count"])
        plt.title("Papers per Year")
        plt.xlabel("year")
        plt.ylabel("count")
        matplotlib_savefig("reports/papers_per_year.png")

    # Category x Year heatmap (top K categories)
    print("\n=== CATEGORY x YEAR HEATMAP (top categories) ===")
    top_cats_list = [r["primary_category"] for r in topcats.collect()]
    cat_year = (
        df.where(F.col("primary_category").isin(top_cats_list))
          .groupBy("primary_category", "year")
          .count()
    )
    cat_year_pd = cat_year.toPandas().pivot_table(index="primary_category", columns="year", values="count", fill_value=0)
    if not cat_year_pd.empty:
        plt.figure(figsize=(12, max(4, len(top_cats_list) * 0.35)))
        plt.imshow(cat_year_pd.values, aspect="auto")
        plt.yticks(range(len(cat_year_pd.index)), cat_year_pd.index)
        plt.xticks(range(len(cat_year_pd.columns)), cat_year_pd.columns, rotation=60, ha="right")
        plt.title("Counts by Primary Category (rows) and Year (cols)")
        plt.colorbar()
        matplotlib_savefig("reports/heatmap_category_year.png")
        cat_year_pd.to_csv("reports/category_year_matrix.csv")
        print("[saved] reports/category_year_matrix.csv")

    # Distribution of abstract length (hist)
    print("\n=== ABSTRACT LENGTH DISTRIBUTION ===")
    abs_len_pd = df.select("abstract_len").sample(False, 0.1, seed=42).toPandas()  # sample to keep plotting light
    if not abs_len_pd.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(abs_len_pd["abstract_len"].dropna(), bins=50)
        plt.title("Abstract Length Distribution (sampled)")
        plt.xlabel("abstract_len")
        plt.ylabel("frequency")
        matplotlib_savefig("reports/abstract_length_hist.png")

    # DOI availability by year (line)
    print("\n=== DOI AVAILABILITY BY YEAR ===")
    has_doi_by_year = (
        df.groupBy("year")
          .agg(F.avg(F.col("has_doi").cast("double")).alias("doi_rate"))
          .orderBy("year")
    )
    save_df_as_csv(has_doi_by_year, "reports/doi_rate_by_year.csv")
    doi_pd = has_doi_by_year.toPandas().dropna()
    if not doi_pd.empty:
        plt.figure(figsize=(9, 4))
        plt.plot(doi_pd["year"], (doi_pd["doi_rate"] * 100.0))
        plt.title("DOI Coverage by Year (%)")
        plt.xlabel("year")
        plt.ylabel("doi rate (%)")
        matplotlib_savefig("reports/doi_rate_by_year.png")

    # Top authors (based on exploded authors_list)
    if "authors_list" in df.columns:
        print("\n=== TOP AUTHORS (by paper count) ===")
        exploded = df.select(F.explode_outer("authors_list").alias("author"))
        top_authors = exploded.groupBy("author").count().orderBy(F.desc("count")).limit(args.topk)
        save_df_as_csv(top_authors, "reports/top_authors.csv")

        top_authors_pd = top_authors.toPandas()
        if not top_authors_pd.empty:
            plt.figure(figsize=(10, 5))
            top_authors_pd = top_authors_pd.sort_values("count", ascending=False)
            plt.bar(top_authors_pd["author"], top_authors_pd["count"])
            plt.xticks(rotation=60, ha="right")
            plt.title(f"Top {len(top_authors_pd)} Authors by Paper Count")
            plt.xlabel("author")
            plt.ylabel("count")
            matplotlib_savefig("reports/top_authors.png")

    # Versions (how many versions per paper)
    print("\n=== VERSION COUNT PER PAPER ===")
    if "versions" in df.columns:
        df_versions = df.withColumn("n_versions", F.size("versions"))
        vhist = (
            df_versions.groupBy("n_versions").count().orderBy("n_versions")
        )
        save_df_as_csv(vhist, "reports/version_count_hist.csv")

        vhist_pd = vhist.toPandas()
        if not vhist_pd.empty:
            plt.figure(figsize=(8, 4))
            plt.bar(vhist_pd["n_versions"], vhist_pd["count"])
            plt.title("Number of Versions per Paper")
            plt.xlabel("n_versions")
            plt.ylabel("count")
            matplotlib_savefig("reports/version_count_hist.png")

    # Category share cumulative (Pareto-style)
    print("\n=== CATEGORY PARETO (CUMULATIVE %) ===")
    cat_counts = df.groupBy("primary_category").count()
    cat_counts_pd = cat_counts.toPandas().sort_values("count", ascending=False)
    if not cat_counts_pd.empty:
        cat_counts_pd["cum_pct"] = (cat_counts_pd["count"].cumsum() / cat_counts_pd["count"].sum()) * 100.0
        cat_counts_pd.to_csv("reports/category_pareto.csv", index=False)
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(cat_counts_pd) + 1), cat_counts_pd["cum_pct"])
        plt.title("Cumulative Share of Papers by Category (Pareto)")
        plt.xlabel("rank of category")
        plt.ylabel("cumulative % of papers")
        matplotlib_savefig("reports/category_pareto.png")

    print("\n[done] EDA artifacts written to reports/")
    spark.stop()


if __name__ == "__main__":
    main()
