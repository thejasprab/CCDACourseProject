"""
Download Cornell arXiv snapshot via KaggleHub.

Modes:
  - full   → data/raw/arxiv-metadata-oai-snapshot.json
  - sample → data/sample/arxiv-sample.jsonl (first N lines)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub


def is_likely_jsonl(path: Path, probe_lines: int = 5) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            for _ in range(probe_lines):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if not s.startswith("{"):
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False


def write_head_jsonl(src: Path, dst: Path, n: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= n:
                break
    return count


def download_full_raw() -> Path:
    print("[KaggleHub] Downloading Cornell-University/arxiv ...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    base = Path(path)
    candidates = list(base.rglob("arxiv-metadata-oai-snapshot.json")) + list(
        base.rglob("arxiv-metadata-oai-snapshot.jsonl")
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find arxiv-metadata-oai-snapshot.json(.jsonl) in Kaggle download."
        )
    src = candidates[0]
    print(f"[Found] {src}")
    proj_raw = Path("data/raw")
    proj_raw.mkdir(parents=True, exist_ok=True)
    target = proj_raw / "arxiv-metadata-oai-snapshot.json"
    if target.resolve() != src.resolve():
        print(f"[Copy] -> {target}")
        shutil.copyfile(src, target)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["full", "sample"],
        required=True,
        help="Download mode: full raw snapshot or only a JSONL sample.",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=30000,
        help="Lines to write into data/sample/arxiv-sample.jsonl (sample mode).",
    )
    args = ap.parse_args()

    target = download_full_raw()

    if args.mode == "sample":
        if args.sample_size <= 0:
            print("[sample] --sample-size <= 0, skipping sample generation.")
            return
        sample_dir = Path("data/sample")
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "arxiv-sample.jsonl"
        if is_likely_jsonl(target):
            n = write_head_jsonl(target, sample_path, args.sample_size)
            print(f"[Sample] Wrote first {n} JSONL lines to {sample_path}")
        else:
            n = write_head_jsonl(target, sample_path, args.sample_size)
            print(f"[Sample] (non-JSONL heuristic) Wrote {n} lines to {sample_path}")
    else:
        print("[full] Raw file ready at data/raw/arxiv-metadata-oai-snapshot.json")


if __name__ == "__main__":
    main()
