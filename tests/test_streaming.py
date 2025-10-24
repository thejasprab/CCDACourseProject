# tests/test_streaming.py
import io
import os
import sys
import time
from pathlib import Path

import pytest


def _write_jsonl_sample(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            # minimal JSON per line
            f.write(f'{{"id":"{i}","title":"t{i}","abstract":"a{i}"}}\n')


def test_write_prefix_atomic_basic(tmp_path, monkeypatch):
    """
    Unit-test the low-level writer to ensure it writes exactly N non-empty lines,
    creates parent dirs, and performs an atomic replace.
    """
    # Import the module under test
    sys.path.insert(0, str(Path.cwd()))
    prep = __import__("scripts.prepare_sample_stream_batches".replace("/", "."), fromlist=["*"])

    src = tmp_path / "data" / "sample" / "arxiv-sample.jsonl"
    dst = tmp_path / "out" / "arxiv-sample-20250101.jsonl"
    _write_jsonl_sample(src, 10)

    written = prep.write_prefix_atomic(src, dst, n_lines=7, overwrite=False)
    assert written == 7
    assert dst.exists()

    # Check that only 7 non-empty lines are present
    with dst.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    assert len(lines) == 7

    # Re-run without overwrite; should skip and return -1
    written2 = prep.write_prefix_atomic(src, dst, n_lines=5, overwrite=False)
    assert written2 == -1

    # With overwrite=True it should replace contents
    written3 = prep.write_prefix_atomic(src, dst, n_lines=5, overwrite=True)
    assert written3 == 5
    with dst.open("r", encoding="utf-8") as f:
        lines2 = [ln for ln in f.readlines() if ln.strip()]
    assert len(lines2) == 5


def test_prepare_main_creates_expected_files(tmp_path, monkeypatch):
    """
    Integration-ish test of `main()`:
    - Point SOURCE and OUTDIR to temp dirs
    - Shrink SIZES to keep test snappy
    - Force --no-sleep so we don't actually wait
    - Assert filenames are weekly-stamped and end with .jsonl
    """
    sys.path.insert(0, str(Path.cwd()))
    prep = __import__("scripts.prepare_sample_stream_batches".replace("/", "."), fromlist=["*"])

    # Point module globals at temp fixture locations
    src = tmp_path / "data" / "sample" / "arxiv-sample.jsonl"
    outdir = tmp_path / "data" / "stream" / "incoming_sample"
    _write_jsonl_sample(src, 100)  # plenty for our tiny sizes

    monkeypatch.setattr(prep, "SOURCE", src, raising=True)
    monkeypatch.setattr(prep, "OUTDIR", outdir, raising=True)

    # Keep it quick: three files, increasing prefixes
    monkeypatch.setattr(prep, "SIZES", [3, 5, 7], raising=True)

    # Fake sleep to avoid delays even if code path hits it
    monkeypatch.setattr(time, "sleep", lambda s: None)

    # Run main with deterministic start date and no sleep
    argv = [
        "prepare_sample_stream_batches.py",
        "--start-date",
        "2025-10-24",
        "--interval-seconds",
        "1",
        "--no-sleep",
        "--overwrite",
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(sys, "argv", argv)

    # Execute
    prep.main()

    # Validate outputs
    files = sorted(outdir.glob("arxiv-sample-*.jsonl"))
    assert len(files) == 3

    # Expect weekly stepping in YYYYMMDD
    stamps = [p.stem.split("-")[-1] for p in files]
    assert stamps == ["20251024", "20251031", "20251107"]  # weekly increments from 2025-10-24

    # Confirm suffix is .jsonl (regression for the extension mismatch bug)
    assert all(p.suffix == ".jsonl" for p in files)

    # Confirm line counts match the requested prefixes
    counts = []
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            counts.append(sum(1 for ln in f if ln.strip()))
    assert counts == [3, 5, 7]


def test_streaming_filename_regex_captures_8_or_12_digits(tmp_path):
    """
    Check the regex used to extract date stamps from filenames accepts either:
    - YYYYMMDD.jsonl
    - YYYYMMDDHHMM.jsonl (12-digit stamp)
    and always returns the first 8 digits for the folder.
    """
    # Import the streaming module without requiring Spark constructs
    sys.path.insert(0, str(Path.cwd()))
    stream_mod = __import__("notebooks.streaming_sample_week11".replace("/", "."), fromlist=["*"])

    # Sanity: regex exists and looks for .jsonl files
    pattern = stream_mod.FILE_DATE_REGEX
    assert pattern.pattern.endswith(r"\.jsonl$")

    paths = [
        "data/stream/incoming_sample/arxiv-sample-20251024.jsonl",
        "data/stream/incoming_sample/arxiv-sample-202510240930.jsonl",
        "/abs/path/arxiv-sample-20240101.jsonl",
        "/abs/path/arxiv-sample-202401012359.jsonl",
    ]
    expected = ["20251024", "20251024", "20240101", "20240101"]
    for p, exp in zip(paths, expected):
        m = pattern.match(p)
        assert m, f"Pattern failed to match {p}"
        assert m.group(1) == exp


def test_streaming_schema_fields_present():
    """
    Light-touch schema test (skips if pyspark isn't installed).
    Ensures expected columns exist so streaming starts without inference.
    """
    pyspark = pytest.importorskip("pyspark")  # skip gracefully if not available
    sys.path.insert(0, str(Path.cwd()))
    stream_mod = __import__("notebooks.streaming_sample_week11".replace("/", "."), fromlist=["*"])

    schema = stream_mod.JSON_SCHEMA
    field_names = {f.name for f in schema.fields}

    # A few key fields to ensure compatibility with the sample dataset/transforms
    for col in [
        "id",
        "title",
        "abstract",
        "categories",
        "doi",
        "journal-ref",
        "authors",
        "authors_parsed",
        "versions",
        "submitted_date",
        "update_date",
    ]:
        assert col in field_names
