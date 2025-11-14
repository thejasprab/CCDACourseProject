from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd


def list_complex_reports(mode: str = "sample") -> List[str]:
    root = Path("reports/analysis_sample" if mode == "sample" else "reports/analysis_full")
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("*.csv"))


def load_complex_report(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)
    return {"path": str(p), "columns": list(df.columns), "rows": df.to_dict(orient="records")}
