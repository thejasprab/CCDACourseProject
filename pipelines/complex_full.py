#!/usr/bin/env python3
"""
Run complex queries on the FULL ingested dataset.

Default paths (can be overridden via CLI):
  parquet : data/processed/arxiv_full
  outdir  : reports/analysis_full
"""

from __future__ import annotations

import argparse

from engine.complex.complex_queries import run_complex_queries


DEFAULT_PARQUET = "data/processed/arxiv_full"
DEFAULT_OUTDIR = "reports/analysis_full"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET,
        help=f"Path to FULL parquet (default: {DEFAULT_PARQUET})",
    )
    ap.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help=f"Output dir for FULL complex analytics (default: {DEFAULT_OUTDIR})",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_complex_queries(args.parquet, args.outdir)


if __name__ == "__main__":
    main()
