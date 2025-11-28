from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd


def _project_root() -> Path:
    """
    Return the project root directory (sparxiv).

    This file lives at: <project_root>/app/services/complex_service.py
    So project_root is two levels up from app/services.
    """
    return Path(__file__).resolve().parents[2]


def _complex_root(mode: str = "sample") -> Path:
    """
    Internal helper to get the root directory for complex analytics outputs.

    Sample mode: <project_root>/reports/analysis_sample
    Full mode  : <project_root>/reports/analysis_full
    """
    base = _project_root() / "reports"
    return base / ("analysis_sample" if mode == "sample" else "analysis_full")


def list_complex_reports(mode: str = "sample") -> List[str]:
    root = _complex_root(mode)
    if not root.exists():
        return []
    # Return full paths for CSVs, used directly by load_complex_report.
    return sorted(str(p) for p in root.glob("*.csv"))


def list_complex_figures(mode: str = "sample") -> List[str]:
    """
    List PNG figures for the given mode.

    For figures we only return the filename (not the full path).
    The server route will reconstruct the full path based on mode.
    """
    root = _complex_root(mode)
    if not root.exists():
        return []
    return sorted(p.name for p in root.glob("*.png"))


def load_complex_report(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)
    return {
        "path": str(p),
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }
