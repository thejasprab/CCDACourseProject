#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F, types as T  # noqa: E402

from engine.utils.spark_utils import get_spark
from engine.data.transformations import transform_all

INCOMING = "data/stream/incoming_sample"
REPORTS_ROOT = Path("reports/streaming_sample")
CHECKPOINT = "data/stream/checkpoints_sample"

FILE_DATE_REGEX = re.compile(r".*arxiv-sample-(\d{8})(\d{4})?\.jsonl$")

JSON_SCHEMA = T.StructType(
    [
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
        T.StructField(
            "versions", T.ArrayType(T.MapType(T.StringType(), T.StringType()))
        ),
    ]
)


def _save_df_as_csv(df_spark, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df_spark.toPandas().to_csv(path, index=False)
    print(f"[saved] {path}")


def _plot_line(x, y, title, xlabel, ylabel, outpng: Path):
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


def _plot_bar(labels, values, title, xlabel, ylabel, outpng: Path, rotate_xticks=False):
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


def _per_drop_reports(batch_df, batch_id: int):
    if batch_df.rdd.isEmpty():
        print(f"[batch {batch_id}] empty batch")
        return

    df_with_name = batch_df.withColumn("source_file", F.input_file_name())
    df_with_date = df_with_name.withColumn(
        "source_date",
        F.regexp_extract(
            F.col("source_file"), r"arxiv-sample-(\d{8})(\d{4})?\.jsonl$", 1
        ),
    )

    dates = [r["source_date"] for r in df_with_date.select("source_date").distinct().collect()]
    for d in dates:
        if not d:
            continue
        print(f"[batch {batch_id}] generating reports for date={d}")
        sub = df_with_date.filter(F.col("source_date") == F.lit(d)).drop(
            "source_file", "source_date"
        )

        transformed = transform_all(sub)

        outdir = REPORTS_ROOT / d
        outdir.mkdir(parents=True, exist_ok=True)

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

        topcats = (
            transformed.groupBy("primary_category")
            .count()
            .orderBy(F.desc("count"))
            .limit(15)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INCOMING)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--trigger-seconds", type=int, default=10)
    ap.add_argument("--max-files-per-trigger", type=int, default=1)
    args = ap.parse_args()

    spark = get_spark("streaming_sample")
    spark.conf.set("spark.sql.streaming.schemaInference", "true")

    stream_df = (
        spark.readStream.format("json")
        .schema(JSON_SCHEMA)
        .option("multiLine", "false")
        .option("maxFilesPerTrigger", str(max(args.max_files_per_trigger, 1)))
        .option("pathGlobFilter", "arxiv-sample-*.jsonl")
        .load(args.input)
    )

    q = (
        stream_df.writeStream.foreachBatch(_per_drop_reports)
        .option("checkpointLocation", args.checkpoint)
        .trigger(processingTime=f"{max(args.trigger_seconds,1)} seconds")
        .start()
    )

    print(
        f"[listen] Watching {args.input} every {args.trigger_seconds} second(s). Ctrl+C to stop."
    )
    try:
        q.awaitTermination()
    except KeyboardInterrupt:
        print("\n[stop] Stopping stream...")
        q.stop()


if __name__ == "__main__":
    main()
