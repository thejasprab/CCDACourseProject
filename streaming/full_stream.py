#!/usr/bin/env python3
"""
Full-dataset streaming job (weekly Kaggle snapshots).

Behavior
--------
- Watches a directory for new arxiv-YYYYMMDD.json(.jsonl) files (weekly snapshots).
- For each new file (micro-batch), applies the same transforms as the batch pipeline
  and writes per-drop reports (CSV + PNG) into:

    reports/streaming_full/YYYYMMDD/

- Designed to be run either continuously or once:

  Continuous mode (default):
      python streaming/full_stream.py

  Trigger-once mode (process all currently-available files then exit):
      python streaming/full_stream.py --once

Typical usage
-------------
1. Use `streaming/kaggle_downloader.py` (or equivalent) to download a new snapshot
   and place it under `data/stream/incoming/` as:

     arxiv-YYYYMMDD.json   # or .jsonl

2. Run this streaming job in the background, or periodically with `--once`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F, types as T

from engine.utils.spark_utils import get_spark
from engine.data.transformations import transform_all


# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------

INCOMING = "data/stream/incoming"
REPORTS_ROOT = Path("reports/streaming_full")
CHECKPOINT_DEFAULT = "data/stream/checkpoints_full"

# Match either:
#   arxiv-YYYYMMDD.json
#   arxiv-YYYYMMDD.jsonl
FILE_DATE_REGEX = re.compile(r".*arxiv-(\d{8})\.jsonl?$")

# Explicit JSON schema so the stream can start immediately without inference
JSON_SCHEMA = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("title", T.StringType()),
    T.StructField("abstract", T.StringType()),
    T.StructField("categories", T.StringType()),
    T.StructField("doi", T.StringType()),
    T.StructField("journal-ref", T.StringType()),
    T.StructField("comments", T.StringType()),
    T.StructField("submitter", T.StringType()),
    T.StructField("update_date", T.StringType()),
    T.StructField("submitted_date", T.StringType()),
    T.StructField("authors", T.StringType()),
    T.StructField("authors_parsed", T.ArrayType(T.ArrayType(T.StringType()))),
    T.StructField("versions", T.ArrayType(T.MapType(T.StringType(), T.StringType()))),
])


# -------------------------------------------------------------------
# Small I/O + plotting helpers
# -------------------------------------------------------------------

def _save_df_as_csv(df_spark, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_spark.toPandas().to_csv(path, index=False)
    print(f"[saved] {path}")


def _plot_line(x, y, title, xlabel, ylabel, outpng: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpng}")


def _plot_bar(labels, values, title, xlabel, ylabel, outpng: Path, rotate_xticks: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate_xticks:
        plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpng}")


# -------------------------------------------------------------------
# Per-batch handler
# -------------------------------------------------------------------

def _per_drop_reports(batch_df, batch_id: int) -> None:
    """
    foreachBatch handler for Structured Streaming.

    batch_df: Spark DataFrame containing only *new* rows for this micro-batch.
    We:
      - derive a `source_date` per row from input_file_name() (YYYYMMDD),
      - group by date,
      - apply `transform_all`,
      - and emit CSV + PNG reports under:

            reports/streaming_full/YYYYMMDD/
    """
    if batch_df.rdd.isEmpty():
        print(f"[batch {batch_id}] empty batch")
        return

    # Attach source filename + date stamp
    df_with_name = batch_df.withColumn("source_file", F.input_file_name())
    df_with_date = df_with_name.withColumn(
        "source_date",
        F.regexp_extract(
            F.col("source_file"),
            r"arxiv-(\d{8})\.jsonl?$",
            1,
        ),
    )

    dates = [r["source_date"] for r in df_with_date.select("source_date").distinct().collect()]
    if not dates:
        print(f"[batch {batch_id}] no matching arxiv-YYYYMMDD files in this batch")
        return

    for d in dates:
        if not d:
            continue

        print(f"[batch {batch_id}] generating reports for date={d}")
        sub = (
            df_with_date
            .filter(F.col("source_date") == F.lit(d))
            .drop("source_file", "source_date")
        )

        # Apply the shared Week-8-style transforms
        transformed = transform_all(sub)

        outdir = REPORTS_ROOT / d
        outdir.mkdir(parents=True, exist_ok=True)

        # 1) Papers per year
        by_year = transformed.groupBy("year").count().orderBy("year")
        _save_df_as_csv(by_year, outdir / "by_year.csv")
        py = by_year.toPandas().dropna()
        if not py.empty:
            _plot_line(
                py["year"],
                py["count"],
                "Papers per Year",
                "year",
                "count",
                outdir / "papers_per_year.png",
            )

        # 2) Top categories
        topcats = (
            transformed.groupBy("primary_category")
                       .count()
                       .orderBy(F.desc("count"))
                       .limit(30)
        )
        _save_df_as_csv(topcats, outdir / "top_categories.csv")
        pc = topcats.toPandas()
        if not pc.empty:
            _plot_bar(
                pc["primary_category"],
                pc["count"],
                "Top Primary Categories",
                "primary_category",
                "count",
                outdir / "top_categories.png",
                rotate_xticks=True,
            )

        # 3) DOI rate by year
        if "has_doi" in transformed.columns:
            doi_year = (
                transformed.groupBy("year")
                           .agg(F.avg(F.col("has_doi").cast("double")).alias("doi_rate"))
                           .orderBy("year")
            )
            _save_df_as_csv(doi_year, outdir / "doi_rate_by_year.csv")
            pd_doi = doi_year.toPandas().dropna()
            if not pd_doi.empty:
                _plot_line(
                    pd_doi["year"],
                    pd_doi["doi_rate"] * 100.0,
                    "DOI Coverage by Year (%)",
                    "year",
                    "doi rate (%)",
                    outdir / "doi_rate_by_year.png",
                )

        print(f"[batch {batch_id}] reports → {outdir}/")


# -------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=INCOMING,
        help="Directory to watch for full arXiv drops (default: data/stream/incoming)",
    )
    ap.add_argument(
        "--checkpoint",
        default=CHECKPOINT_DEFAULT,
        help="Checkpoint directory for Structured Streaming state",
    )
    ap.add_argument(
        "--trigger-seconds",
        type=int,
        default=60,
        help="Micro-batch frequency in seconds (ignored if --once is set).",
    )
    ap.add_argument(
        "--max-files-per-trigger",
        type=int,
        default=1,
        help="Maximum number of new files to pick up per micro-batch.",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help=(
            "Use Trigger.Once: process all available files, write reports, "
            "then exit when caught up."
        ),
    )
    args = ap.parse_args()

    # Ensure directories exist
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    Path(args.input).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    # Spark session (tuned via shared util)
    spark = get_spark("streaming_full_arxiv")
    spark.conf.set("spark.sql.streaming.schemaInference", "true")

    # Structured stream over incoming arxiv-YYYYMMDD JSON/JSONL files
    stream_df = (
        spark.readStream
             .format("json")
             .schema(JSON_SCHEMA)
             .option("multiLine", "false")
             .option("maxFilesPerTrigger", str(max(args.max_files_per_trigger, 1)))
             .option("pathGlobFilter", "arxiv-*.json*")
             .load(args.input)
    )

    writer = (
        stream_df.writeStream
                 .foreachBatch(_per_drop_reports)
                 .option("checkpointLocation", args.checkpoint)
    )

    if args.once:
        # Trigger.Once: process all currently available data and then stop.
        q = writer.trigger(once=True).start()
        print("[stream] Trigger.Once mode: will process all available files, then stop.")
    else:
        # Continuous micro-batches
        interval = max(args.trigger_seconds, 1)
        q = writer.trigger(processingTime=f"{interval} seconds").start()
        print(f"[stream] Watching {args.input} every {interval} second(s). Ctrl+C to stop.")

    try:
        q.awaitTermination()
    except KeyboardInterrupt:
        print("\n[stop] Stopping full stream...")
        q.stop()
        spark.stop()


if __name__ == "__main__":
    main()
