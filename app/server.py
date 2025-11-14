# app/server.py
from __future__ import annotations

from typing import Dict, Any, List

from flask import (
    Blueprint,
    Flask,
    render_template,
    request,
    redirect,
    url_for,
)

from app.config import settings
from app.services import search_service, filters_service, complex_service

bp = Blueprint("main", __name__)

# Simple labels for the two modes
DATASETS: Dict[str, str] = {
    "sample": "Sample (≈50k papers)",
    "full": "Full dataset",
}


def _normalize_mode(raw: str | None) -> str:
    if raw in DATASETS:
        return raw  # type: ignore[return-value]
    return settings.default_mode if settings.default_mode in DATASETS else "sample"


@bp.route("/", methods=["GET", "POST"])
def index():
    """
    Home page: free-text similarity search over arXiv papers.
    Uses the existing TF-IDF + cosine SearchEngine via search_service.search_papers.
    """
    mode = _normalize_mode(request.values.get("mode"))
    k_raw = request.values.get("k", "10")
    title = ""
    abstract = ""
    error: str | None = None
    results: List[Dict[str, Any]] = []

    # Precompute dataset labels for dropdown / radio buttons
    dataset_choices = DATASETS

    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        abstract = (request.form.get("abstract") or "").strip()
        k_raw = request.form.get("k", k_raw)

        if not title and not abstract:
            error = "Please enter at least a title or an abstract."
        else:
            try:
                k = int(k_raw)
            except ValueError:
                k = 10
            k = max(1, min(k, 50))

            results = search_service.search_papers(
                title=title,
                abstract=abstract,
                k=k,
                mode=mode,
            )

            if not results:
                error = "No results found. Try relaxing your query or switching dataset."

    # Optional: list popular categories as hints (even though we don't filter by them yet)
    try:
        popular_categories = filters_service.list_primary_categories(mode=mode, top_k=30)
    except Exception:
        popular_categories = []

    return render_template(
        "index.html",
        mode=mode,
        dataset_choices=dataset_choices,
        k=k_raw,
        title_query=title,
        abstract_query=abstract,
        results=results,
        error=error,
        popular_categories=popular_categories,
    )


@bp.route("/complex")
def complex_reports():
    """
    Page to browse CSV outputs produced by engine.complex.complex_queries
    via pipelines/complex_sample.py and pipelines/complex_full.py.
    """
    mode = _normalize_mode(request.args.get("mode"))
    dataset_choices = DATASETS

    reports = complex_service.list_complex_reports(mode=mode)
    selected_path = request.args.get("path")

    table: Dict[str, Any] | None = None
    selected_error: str | None = None

    if selected_path:
        try:
            table = complex_service.load_complex_report(selected_path)
        except FileNotFoundError:
            selected_error = f"Report not found on disk: {selected_path}"
        except Exception as exc:  # pragma: no cover
            selected_error = f"Failed to load report: {exc}"

    # For nicer labels in the dropdown/table
    named_reports = [
        {
            "path": p,
            "name": p.split("/")[-1],
        }
        for p in reports
    ]

    return render_template(
        "complex.html",
        mode=mode,
        dataset_choices=dataset_choices,
        reports=named_reports,
        table=table,
        selected_path=selected_path,
        selected_error=selected_error,
    )


def register_routes(app: Flask) -> None:
    """
    Called from app/__init__.py to attach routes to the Flask app.
    """
    app.register_blueprint(bp)


# Optional local debug entrypoint:
if __name__ == "__main__":  # pragma: no cover
    from app import create_app

    flask_app = create_app()
    # Make sure there's a secret key if you later add flash() or sessions.
    flask_app.config.setdefault("SECRET_KEY", "dev-secret-key-change-me")
    flask_app.run(debug=True, host="0.0.0.0", port=5000)
