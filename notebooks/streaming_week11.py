#!/usr/bin/env python3
"""
Week 11 (Full) Streaming-Oriented Runner (Daily Scheduler)

Behavior:
- Runs once per day (loop) OR just once with --once.
- On Sundays (local time), pulls the latest Kaggle arXiv snapshot via kagglehub,
  stages it to data/stream/incoming/arxiv-YYYYMMDD.json (YYYYMMDD = today's date),
  then processes that drop and writes CSV+PNG reports to reports/streaming_full/YYYYMMDD/.
- Maintains a small state file to avoid duplicate processing per date.

Run (daemon-ish):
  python notebooks/streaming_week11.py

Run a single check (useful for cron/CI/manual trigger):
  python notebooks/streaming_week11.py --once
"""
from __future__ import annotations
import os
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import kagglehub
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from src.utils import get_spark
from src.transformations import transform_all

# --- Locations ---
INCOMING = Path("data/stream/incoming")
REPORTS_ROOT = Path("reports/streaming_full")
STATE_DIR = Path("data/stream/state")
STATE_FILE = STATE_DIR / "last_full_run.txt"

# --- Helpers: time/date/state ---
def _today_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")  # local time

def _is_sunday() -> bool:
    # Monday=0 ... Sunday=6
    return datetime.now().weekday() == 6

def _load_last_date() -> str | None:
    if STATE_FILE.exists():
        return STATE_FILE.read_text().strip() or None
    return None

def _save_last_date(stamp: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(stamp, encoding="utf-8")

def _sleep_until_next_day():
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    secs = max(10, int((tomorrow - now).total_seconds()))
    print(f"[sleep] {secs}s until next daily check…")
    time.sleep(secs)

# --- Kaggle download + staging ---
def _download_kaggle_snapshot() -> Path:
    """
    Use kagglehub to fetch the Cornell arXiv snapshot; return the path to the JSON/JSONL file.
    """
    print("[kaggle] Downloading Cornell-University/arxiv …")
    base = Path(kagglehub.dataset_download("Cornell-University/arxiv"))
    # Prefer JSONL if present, else JSON
    candidates = list(base.rglob("arxiv-metadata-oai-snapshot.jsonl")) + \
                 list(base.rglob("arxiv-metadata-oai-snapshot.json"))
    if not candidates:
        raise FileNotFoundError("Could not find arxiv-metadata-oai-snapshot.json(.jsonl) in Kaggle download.")
    src = candidates[0]
    print(f"[kaggle] Found {src}")
    return src

def _stage_incoming(src: Path, stamp: str) -> Path:
    """
    Copy the Kaggle file into our incoming/ folder as arxiv-YYYYMMDD.json
    (we use .json extension regardless, the content is JSON Lines.)
    """
    INCOMING.mkdir(parents=True, exist_ok=True)
    dst = INCOMING / f"arxiv-{stamp}.json"
    # write to temp then atomic move
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    print(f"[stage] {dst}")
    return dst

# --- Small plotting helpers ---
def _save_df_as_csv(df_spark, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df_spark.toPandas().to_csv(path, index=False)
    print(f"[saved] {path}")

def _plot_line(x, y, title, xlabel, ylabel, outpng: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.tight_layout()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpng}")

def _plot_bar(labels, values, title, xlabel, ylabel, outpng: Path, rotate_xticks=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if rotate_xticks:
        plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outpng}")

# --- Processing per drop ---
def _process_drop(incoming_file: Path, stamp: str):
    """
    Batch process the just-downloaded file and write per-stamp reports.
    """
    print(f"[process] Generating reports for {stamp} from {incoming_file.name}")
    spark = get_spark("streaming_week11_full")

    # Kaggle file is typically JSONL (one object per line); read as non-multiline
    df_raw = (spark.read
                   .option("multiLine", "false")
                   .json(str(incoming_file)))

    df = transform_all(df_raw)
    outdir = REPORTS_ROOT / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    # Papers per year
    by_year = df.groupBy("year").count().orderBy("year")
    _save_df_as_csv(by_year, outdir / "by_year.csv")
    py = by_year.toPandas().dropna()
    if not py.empty:
        _plot_line(py["year"], py["count"], "Papers per Year", "year", "count", outdir / "papers_per_year.png")

    # Top categories (Top-30)
    topcats = (
        df.groupBy("primary_category")
          .count()
          .orderBy(F.desc("count"))
          .limit(30)
    )
    _save_df_as_csv(topcats, outdir / "top_categories.csv")
    pc = topcats.toPandas()
    if not pc.empty:
        _plot_bar(pc["primary_category"], pc["count"],
                  "Top Primary Categories", "primary_category", "count",
                  outdir / "top_categories.png", rotate_xticks=True)

    # DOI rate by year
    if "has_doi" in df.columns:
        doi_year = (df.groupBy("year")
                      .agg(F.avg(F.col("has_doi").cast("double")).alias("doi_rate"))
                      .orderBy("year"))
        _save_df_as_csv(doi_year, outdir / "doi_rate_by_year.csv")
        pd_doi = doi_year.toPandas().dropna()
        if not pd_doi.empty:
            _plot_line(pd_doi["year"], pd_doi["doi_rate"] * 100.0,
                       "DOI Coverage by Year (%)", "year", "doi rate (%)",
                       outdir / "doi_rate_by_year.png")

    spark.stop()
    print(f"[done] Reports → {outdir}/")

# --- One-shot Sunday check ---
def run_once():
    today = _today_stamp()
    last = _load_last_date()

    if not _is_sunday():
        print("[skip] Today is not Sunday; no weekly pull.")
        return

    if last == today:
        print(f"[skip] Already processed {today}.")
        return

    try:
        src = _download_kaggle_snapshot()
        dst = _stage_incoming(src, today)
        _process_drop(dst, today)
        _save_last_date(today)
    except Exception as e:
        print(f"[error] {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="Run a single check (good for cron/CI/tests) and exit.")
    args = ap.parse_args()

    INCOMING.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if args.once:
        run_once()
        return

    print("[loop] Daily scheduler started. Will attempt on Sundays.")
    while True:
        run_once()
        _sleep_until_next_day()

if __name__ == "__main__":
    main()
