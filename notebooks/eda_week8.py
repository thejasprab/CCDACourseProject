# notebooks/eda_week8.py
"""
Comprehensive EDA for the arXiv metadata after ingestion.

Usage:
  python notebooks/eda_week8.py --parquet data/processed/arxiv_full
  # or for the small sample:
  python notebooks/eda_week8.py --parquet data/processed/arxiv_parquet

This script will:
  - Print schema and table-level summaries to stdout
  - Write compact CSV summaries to reports/<run_type>/
  - Save several matplotlib charts to reports/<run_type>/*.png

Notes:
  - <run_type> is "full" if the parquet path contains "full", otherwise "sample"
    (override with --outdir if you want a custom folder).
  - Plots avoid loading huge data to the driver by using aggregated Spark DF results.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
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
    # All saved tables are small (aggregations), safe to convert to pandas
    df_spark.toPandas().to_csv(path, index=False)
    print(f"[saved] {path}")

def matplotlib_savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


# ---------- core EDA ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to Parquet output from src/ingestion.py")
    ap.add_argument("--topk", type=int, default=20, help="Top-K categories/authors to show")
    ap.add_argument("--abslen-sample-frac", type=float, default=0.05,
                    help="Sampling fraction for abstract length histogram (0<frac<=1)")
    ap.add_argument("--outdir", default=None,
                    help="Optional subfolder name under reports/. If not set, auto-chooses 'full' or 'sample'.")
    args = ap.parse_args()

    outdir = pick_outdir(args.parquet, args.outdir)

    spark = SparkSession.builder.appName("arxiv_week8_eda").getOrCreate()
    df = spark.read.parquet(args.parquet)

    # Basic info
    print("\n=== SCHEMA ===")
    df.printSchema()

    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"\n=== SIZE ===\nrows={n_rows:,}  cols={n_cols}")

    
    print("\n=== COMPLETENESS (non-null %) ===")

    # compute row count first (single number)
    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"\n=== SIZE ===\nrows={n_rows:,}  cols={n_cols}")

    # Do small per-column jobs to avoid a single wide aggregation that can OOM
    rows = []
    for c in df.columns:
        # count non-nulls in a tiny job
        nn = df.select(F.count(F.when(F.col(c).isNotNull(), 1)).alias("nn")).collect()[0]["nn"]
        pct = float(nn) / n_rows * 100.0 if n_rows else 0.0
        rows.append((c, int(nn), pct))

    comp_pd = pd.DataFrame(rows, columns=["column", "non_null", "non_null_pct"])\
                .sort_values("non_null_pct", ascending=False)

    comp_csv = outdir / "completeness.csv"
    comp_pd.to_csv(comp_csv, index=False)
    print(comp_pd.head(20))
    print(f"[saved] {comp_csv}")

    # DISTINCT COUNTS (selected)
    print("\n=== DISTINCT COUNTS (selected) ===")
    selected = ["arxiv_id", "primary_category", "year", "doi", "submitter"]
    distinct_rows = []
    for c in selected:
        if c in df.columns:
            distinct_rows.append((c, df.select(c).distinct().count()))
    distinct_pd = pd.DataFrame(distinct_rows, columns=["column", "distinct_count"]).sort_values("distinct_count", ascending=False)
    distinct_csv = outdir / "distinct_selected.csv"
    distinct_pd.to_csv(distinct_csv, index=False)
    print(distinct_pd)
    print(f"[saved] {distinct_csv}")

    # TEXT LENGTH STATS
    print("\n=== TEXT LENGTH STATS (title_len, abstract_len) ===")
    num_stats = df.select("title_len", "abstract_len").summary("count", "min", "25%", "50%", "75%", "max", "mean")
    save_df_as_csv(num_stats, outdir / "text_length_summary.csv")
    num_stats.show(truncate=False)

    # TOP CATEGORIES (bar)
    print("\n=== TOP CATEGORIES ===")
    topcats = (
        df.groupBy("primary_category")
          .count()
          .orderBy(F.desc("count"))
          .limit(args.topk)
    )
    save_df_as_csv(topcats, outdir / "top_categories.csv")
    topcats_pd = topcats.toPandas()
    if not topcats_pd.empty:
        plt.figure(figsize=(10, 5))
        topcats_pd = topcats_pd.sort_values("count", ascending=False)
        plt.bar(topcats_pd["primary_category"], topcats_pd["count"])
        plt.xticks(rotation=60, ha="right")
        plt.title(f"Top {len(topcats_pd)} Primary Categories")
        plt.xlabel("primary_category")
        plt.ylabel("count")
        matplotlib_savefig(outdir / "top_categories.png")

    # PAPERS PER YEAR (line)
    print("\n=== PAPERS PER YEAR ===")
    by_year = df.groupBy("year").count().orderBy("year")
    save_df_as_csv(by_year, outdir / "by_year.csv")
    by_year_pd = by_year.toPandas().dropna()
    if not by_year_pd.empty:
        plt.figure(figsize=(9, 4))
        plt.plot(by_year_pd["year"], by_year_pd["count"])
        plt.title("Papers per Year")
        plt.xlabel("year")
        plt.ylabel("count")
        matplotlib_savefig(outdir / "papers_per_year.png")

    # CATEGORY x YEAR HEATMAP (for top-K categories only)
    print("\n=== CATEGORY x YEAR HEATMAP (top categories) ===")
    top_cats_list = topcats_pd["primary_category"].tolist() if not topcats_pd.empty else []
    if top_cats_list:
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
            matplotlib_savefig(outdir / "heatmap_category_year.png")
            cat_year_pd.to_csv(outdir / "category_year_matrix.csv")
            print(f"[saved] {outdir / 'category_year_matrix.csv'}")

    # ABSTRACT LENGTH DISTRIBUTION (hist; sampled)
    print("\n=== ABSTRACT LENGTH DISTRIBUTION ===")
    frac = args.abslen_sample_frac
    if frac > 0 and frac <= 1:
        abs_len_pd = df.select("abstract_len").sample(False, frac, seed=42).toPandas()
        if not abs_len_pd.empty:
            plt.figure(figsize=(8, 4))
            plt.hist(abs_len_pd["abstract_len"].dropna(), bins=50)
            plt.title(f"Abstract Length Distribution (sampled frac={frac})")
            plt.xlabel("abstract_len")
            plt.ylabel("frequency")
            matplotlib_savefig(outdir / "abstract_length_hist.png")

    # DOI AVAILABILITY BY YEAR (line)
    print("\n=== DOI AVAILABILITY BY YEAR ===")
    if "has_doi" in df.columns:
        has_doi_by_year = (
            df.groupBy("year")
              .agg(F.avg(F.col("has_doi").cast("double")).alias("doi_rate"))
              .orderBy("year")
        )
        save_df_as_csv(has_doi_by_year, outdir / "doi_rate_by_year.csv")
        doi_pd = has_doi_by_year.toPandas().dropna()
        if not doi_pd.empty:
            plt.figure(figsize=(9, 4))
            plt.plot(doi_pd["year"], (doi_pd["doi_rate"] * 100.0))
            plt.title("DOI Coverage by Year (%)")
            plt.xlabel("year")
            plt.ylabel("doi rate (%)")
            matplotlib_savefig(outdir / "doi_rate_by_year.png")

    # TOP AUTHORS (bar)
    if "authors_list" in df.columns:
        print("\n=== TOP AUTHORS (by paper count) ===")
        exploded = df.select(F.explode_outer("authors_list").alias("author"))
        top_authors = exploded.groupBy("author").count().orderBy(F.desc("count")).limit(args.topk)
        save_df_as_csv(top_authors, outdir / "top_authors.csv")

        top_authors_pd = top_authors.toPandas()
        if not top_authors_pd.empty:
            plt.figure(figsize=(10, 5))
            top_authors_pd = top_authors_pd.sort_values("count", ascending=False)
            plt.bar(top_authors_pd["author"], top_authors_pd["count"])
            plt.xticks(rotation=60, ha="right")
            plt.title(f"Top {len(top_authors_pd)} Authors by Paper Count")
            plt.xlabel("author")
            plt.ylabel("count")
            matplotlib_savefig(outdir / "top_authors.png")

    # VERSION COUNT PER PAPER (bar)
    print("\n=== VERSION COUNT PER PAPER ===")
    if "versions" in df.columns:
        df_versions = df.withColumn("n_versions", F.size("versions"))
        vhist = df_versions.groupBy("n_versions").count().orderBy("n_versions")
        save_df_as_csv(vhist, outdir / "version_count_hist.csv")

        vhist_pd = vhist.toPandas()
        if not vhist_pd.empty:
            plt.figure(figsize=(8, 4))
            plt.bar(vhist_pd["n_versions"], vhist_pd["count"])
            plt.title("Number of Versions per Paper")
            plt.xlabel("n_versions")
            plt.ylabel("count")
            matplotlib_savefig(outdir / "version_count_hist.png")

    # CATEGORY PARETO (cumulative %)
    print("\n=== CATEGORY PARETO (CUMULATIVE %) ===")
    cat_counts = df.groupBy("primary_category").count()
    cat_counts_pd = cat_counts.toPandas().sort_values("count", ascending=False)
    if not cat_counts_pd.empty:
        cat_counts_pd["cum_pct"] = (cat_counts_pd["count"].cumsum() / cat_counts_pd["count"].sum()) * 100.0
        cat_counts_pd.to_csv(outdir / "category_pareto.csv", index=False)
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(cat_counts_pd) + 1), cat_counts_pd["cum_pct"])
        plt.title("Cumulative Share of Papers by Category (Pareto)")
        plt.xlabel("rank of category")
        plt.ylabel("cumulative % of papers")
        matplotlib_savefig(outdir / "category_pareto.png")

    print(f"\n[done] EDA artifacts written to {outdir}/")
    spark.stop()


if __name__ == "__main__":
    main()
