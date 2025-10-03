# scripts/download_arxiv.py
"""
Download the Kaggle Cornell arXiv metadata and (optionally) write a tiny JSONL sample
for Week-8 PRs / quick EDA in Codespaces.

Examples:
  python scripts/download_arxiv.py
  python scripts/download_arxiv.py --sample 50000
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import kagglehub

def is_likely_jsonl(path: Path, probe_lines: int = 5) -> bool:
    """
    Heuristic: JSONL typically has a complete JSON object per line.
    We'll read a few lines and check that each starts with '{' (after stripping).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i in range(probe_lines):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if not s.startswith("{"):
                    return False
        return True
    except Exception:
        return False

def write_head_jsonl(src: Path, dst: Path, n: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= n:
                break
    return count

def main():
    ap = argparse.ArgumentParser(description="Download Kaggle arXiv dataset and optionally create a JSONL sample.")
    ap.add_argument("--sample", type=int, default=30000, help="Number of lines to write into data/sample/arxiv-sample.jsonl (0 to skip).")
    args = ap.parse_args()

    print("[KaggleHub] Downloading Cornell-University/arxiv ...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print("Path to dataset files:", path)

    # Locate the metadata file in the downloaded folder
    base = Path(path)
    candidates = list(base.rglob("arxiv-metadata-oai-snapshot.json")) + \
                 list(base.rglob("arxiv-metadata-oai-snapshot.jsonl"))
    if not candidates:
        raise FileNotFoundError("Could not find arxiv-metadata-oai-snapshot.json(.jsonl) in Kaggle download.")

    src = candidates[0]
    print(f"[Found] {src}")

    # Copy raw file into project (but keep it out of Git; .gitignore has data/raw/)
    proj_raw = Path("data/raw")
    proj_raw.mkdir(parents=True, exist_ok=True)
    target = proj_raw / src.name
    if target.resolve() != src.resolve():
        print(f"[Copy] -> {target}")
        shutil.copyfile(src, target)

    # Create a small JSONL sample for Week-8 PRs
    if args.sample and args.sample > 0:
        sample_dir = Path("data/sample")
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "arxiv-sample.jsonl"

        if is_likely_jsonl(target):
            n = write_head_jsonl(target, sample_path, args.sample)
            print(f"[Sample] Wrote first {n} JSONL lines to {sample_path}")
        else:
            # If the downloaded file is pretty-printed JSON (rare), we’ll advise running with --multiline later.
            # Still write a small pseudo-sample by grabbing lines (not strict JSON); mainly for dev/testing of IO.
            n = write_head_jsonl(target, sample_path, args.sample)
            print(f"[Sample] (non-JSONL heuristic) Wrote {n} lines to {sample_path}")
            print("NOTE: If this isn't valid JSONL, use --multiline when running ingestion on the full file.")

    print("[Done] Ready for Week-8 ingestion.")

if __name__ == "__main__":
    main()
