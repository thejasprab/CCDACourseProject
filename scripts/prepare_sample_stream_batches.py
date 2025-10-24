#!/usr/bin/env python3
"""
Prepare weekly 'streaming' drops for the sample pipeline, emitted every N seconds.

- Reads:  data/sample/arxiv-sample.jsonl  (≈50k lines)
- Writes: data/stream/incoming_sample/arxiv-sample-YYYYMMDD.json
          5 files total; sizes [10k, 20k, 30k, 40k, 50k] (prefix/superset)
- File *names* advance one week at a time (YYYYMMDD), but files are
  *emitted* every --interval-seconds (default: 60s) to simulate streaming.

Run:
  python scripts/prepare_sample_stream_batches.py
Examples:
  # start at 2025-10-24 and drop a new weekly-dated file every minute
  python scripts/prepare_sample_stream_batches.py --start-date 2025-10-24 --interval-seconds 60

Options:
  --start-date YYYY-MM-DD   First weekly date in filenames (default: today, local)
  --interval-seconds N      Delay between drops in seconds (default: 60)
  --no-sleep                Emit all files immediately (no delay)
  --overwrite               Overwrite if a target already exists
"""

from __future__ import annotations
import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

SOURCE = Path("data/sample/arxiv-sample.jsonl")
OUTDIR = Path("data/stream/incoming_sample")
SIZES = [10_000, 20_000, 30_000, 40_000, 50_000]

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def write_prefix_atomic(src: Path, dst: Path, n_lines: int, overwrite: bool = False) -> int:
    """
    Write first n_lines from src (JSONL) to dst (.json) via a temp file,
    then atomically rename so Spark treats the arrival correctly.
    """
    if dst.exists() and not overwrite:
        print(f"[skip] exists: {dst}")
        return -1

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    count = 0
    with src.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= n_lines:
                break

    os.replace(tmp, dst)  # atomic move
    print(f"[write] {dst}  lines={count}")
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                    help='First weekly date (YYYY-MM-DD) used in filenames (default: today)')
    ap.add_argument("--interval-seconds", type=int, default=60,
                    help="Seconds to wait between drops (default: 60)")
    ap.add_argument("--no-sleep", action="store_true",
                    help="Emit all files immediately without waiting between drops")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite target files if they already exist")
    args = ap.parse_args()

    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}. Generate it with your dataset download script first.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    start_dt = parse_date(args.start_date)

    for i, size in enumerate(SIZES):
        drop_dt = start_dt + timedelta(weeks=i)
        stamp = drop_dt.strftime("%Y%m%d")  # e.g., 20251024
        target = OUTDIR / f"arxiv-sample-{stamp}.jsonl"  # per your requested naming

        _ = write_prefix_atomic(SOURCE, target, size, overwrite=args.overwrite)

        if i < len(SIZES) - 1 and not args.no_sleep:
            sleep_s = max(args.interval_seconds, 1)
            print(f"[wait] sleeping {sleep_s}s before next drop...")
            time.sleep(sleep_s)

    print(f"[done] Created {len(SIZES)} weekly-dated drops under {OUTDIR}/")

if __name__ == "__main__":
    main()
