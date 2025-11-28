from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd


def _project_root() -> Path:
    """
    Return the project root directory (sparxiv).

    This file lives at: <project_root>/app/services/standard_service.py
    So project_root is two levels up from app/services.
    """
    return Path(__file__).resolve().parents[2]


def _standard_root(mode: str = "sample") -> Path:
    """
    Root directory for standard query outputs.

    Sample mode: <project_root>/reports/standard_queries_sample
    Full mode  : <project_root>/reports/standard_queries_full
    """
    base = _project_root() / "reports"
    return base / ("standard_queries_sample" if mode == "sample" else "standard_queries_full")


def list_standard_reports(mode: str = "sample") -> List[str]:
    """
    List all standard query CSV report paths for the given mode.
    """
    root = _standard_root(mode)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("*.csv"))


def list_standard_figures(mode: str = "sample") -> List[str]:
    """
    List PNG figures for the standard queries for the given mode.

    We return only the filenames. The server route reconstructs the full path.
    """
    root = _standard_root(mode)
    if not root.exists():
        return []
    return sorted(p.name for p in root.glob("*.png"))


def load_standard_report(path: str) -> Dict:
    """
    Load a standard query CSV into a simple structure for the template.
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
