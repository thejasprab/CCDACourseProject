from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd


def _project_root() -> Path:
    """
    Return the project root directory (sparxiv).

    This file lives at: <project_root>/app/services/streaming_service.py
    So project_root is two levels up from app/services.
    """
    return Path(__file__).resolve().parents[2]


def _incoming_root() -> Path:
    """
    Directory where streaming input snapshots are staged, like:
      <project_root>/data/stream/incoming
    """
    return _project_root() / "data" / "stream" / "incoming"


def _reports_root(mode: str = "full") -> Path:
    """
    Root directory for streaming reports.

    Sample mode: <project_root>/reports/streaming_sample
    Full mode  : <project_root>/reports/streaming_full
    """
    base = _project_root() / "reports"
    return base / ("streaming_sample" if mode == "sample" else "streaming_full")


def list_streaming_incoming() -> List[str]:
    """
    List all incoming arxiv-YYYYMMDD.json files as full paths.
    """
    root = _incoming_root()
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("arxiv-*.json"))


def list_streaming_stamps(mode: str = "full") -> List[str]:
    """
    List all available date stamps (YYYYMMDD) for the given mode,
    based on the directory names under the streaming reports root.
    """
    root = _reports_root(mode)
    if not root.exists():
        return []
    stamps: List[str] = []
    for p in root.iterdir():
        if p.is_dir():
            stamps.append(p.name)
    return sorted(stamps)


def list_streaming_reports(mode: str, stamp: str) -> List[str]:
    """
    List all CSV reports for a given mode and date stamp (YYYYMMDD).
    Looks under reports/streaming_<mode>/<stamp>/.
    """
    root = _reports_root(mode) / stamp
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("*.csv"))


def list_streaming_figures(mode: str, stamp: str) -> List[str]:
    """
    List all PNG figures for a given mode and date stamp (YYYYMMDD).
    Returns filenames only (not full paths); server will reconstruct.
    """
    root = _reports_root(mode) / stamp
    if not root.exists():
        return []
    return sorted(p.name for p in root.glob("*.png"))


def load_streaming_report(path: str) -> Dict:
    """
    Load a streaming CSV report into a simple structure for the template.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)
    return {
        "path": str(p),
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }
