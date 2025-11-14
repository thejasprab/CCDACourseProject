from __future__ import annotations

from typing import List, Dict, Any

from app.config import settings
from app.services.spark_session import get_spark_session
from engine.search.search_engine import SearchEngine

_ENGINE_CACHE: dict[str, SearchEngine] = {}


def _get_engine(mode: str | None = None) -> SearchEngine:
    m = mode or settings.default_mode
    if m not in _ENGINE_CACHE:
        _ENGINE_CACHE[m] = SearchEngine(mode=m, spark=get_spark_session())
    return _ENGINE_CACHE[m]


def search_papers(
    title: str = "",
    abstract: str = "",
    k: int = 10,
    mode: str | None = None,
) -> List[Dict[str, Any]]:
    engine = _get_engine(mode)
    return engine.search(title=title, abstract=abstract, k=k)
