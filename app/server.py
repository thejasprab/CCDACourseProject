# app/server.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from flask import (
    Blueprint,
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    abort,
)

from app.config import settings
from app.services import (
    search_service,
    filters_service,
    complex_service,
    standard_service,
    streaming_service,
)

bp = Blueprint("main", __name__)

# Simple labels for the two modes
DATASETS: Dict[str, str] = {
    "sample": "Sample (≈50k papers)",
    "full": "Full dataset (≈3M papers)",
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

    # Optional: list popular categories as hints (even though we do not filter by them yet)
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
    figures = complex_service.list_complex_figures(mode=mode)

    selected_path = request.args.get("path")
    selected_figure = request.args.get("figure")

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

    named_figures = [
        {
            "path": f,
            "name": f,
        }
        for f in figures
    ]

    return render_template(
        "complex.html",
        mode=mode,
        dataset_choices=dataset_choices,
        reports=named_reports,
        figures=named_figures,
        table=table,
        selected_path=selected_path,
        selected_figure=selected_figure,
        selected_error=selected_error,
    )


@bp.route("/complex/figure")
def complex_figure():
    """
    Serve a selected PNG figure from the complex analytics reports.
    """
    mode = _normalize_mode(request.args.get("mode"))
    filename = request.args.get("path")

    if not filename:
        abort(400)

    figures = complex_service.list_complex_figures(mode=mode)
    if filename not in figures:
        abort(404)

    # Use the same root logic as complex_service to avoid path drift
    root = complex_service._complex_root(mode)  # type: ignore[attr-defined]
    full_path = root / filename

    if not full_path.exists():
        abort(404)

    return send_file(full_path, mimetype="image/png")


@bp.route("/standard")
def standard_reports():
    """
    Page to browse CSV outputs produced by standard query pipelines.

    These live under:
      reports/standard_queries_sample
      reports/standard_queries_full
    """
    mode = _normalize_mode(request.args.get("mode"))
    dataset_choices = DATASETS

    reports = standard_service.list_standard_reports(mode=mode)
    figures = standard_service.list_standard_figures(mode=mode)

    selected_path = request.args.get("path")
    selected_figure = request.args.get("figure")

    table: Dict[str, Any] | None = None
    selected_error: str | None = None

    if selected_path:
        try:
            table = standard_service.load_standard_report(selected_path)
        except FileNotFoundError:
            selected_error = f"Report not found on disk: {selected_path}"
        except Exception as exc:  # pragma: no cover
            selected_error = f"Failed to load report: {exc}"

    named_reports = [
        {
            "path": p,
            "name": p.split("/")[-1],
        }
        for p in reports
    ]

    named_figures = [
        {
            "path": f,
            "name": f,
        }
        for f in figures
    ]

    return render_template(
        "standard.html",
        mode=mode,
        dataset_choices=dataset_choices,
        reports=named_reports,
        figures=named_figures,
        table=table,
        selected_path=selected_path,
        selected_figure=selected_figure,
        selected_error=selected_error,
    )


@bp.route("/standard/figure")
def standard_figure():
    """
    Serve a selected PNG figure from the standard query reports.
    """
    mode = _normalize_mode(request.args.get("mode"))
    filename = request.args.get("path")

    if not filename:
        abort(400)

    figures = standard_service.list_standard_figures(mode=mode)
    if filename not in figures:
        abort(404)

    root = standard_service._standard_root(mode)  # type: ignore[attr-defined]
    full_path = root / filename

    if not full_path.exists():
        abort(404)

    return send_file(full_path, mimetype="image/png")


@bp.route("/streaming")
def streaming_reports():
    """
    Page to browse streaming snapshots and their per-date reports.

    Incoming snapshots live under:
      data/stream/incoming/arxiv-YYYYMMDD.json

    Reports live under:
      reports/streaming_sample/YYYYMMDD/ (sample mode)
      reports/streaming_full/YYYYMMDD/   (full mode)
    """
    mode = _normalize_mode(request.args.get("mode"))
    dataset_choices = DATASETS

    # Stamps come from streaming report directories (sample/full)
    stamps = streaming_service.list_streaming_stamps(mode=mode)

    # Raw incoming files for context
    incoming_paths = streaming_service.list_streaming_incoming()
    incoming_named: List[Dict[str, Any]] = []
    for p in incoming_paths:
        name = p.split("/")[-1]
        stamp = None
        if name.startswith("arxiv-") and name.endswith(".json"):
            stamp = name[len("arxiv-"):-len(".json")]
        incoming_named.append(
            {
                "path": p,
                "name": name,
                "stamp": stamp,
            }
        )

    selected_stamp = request.args.get("stamp")
    selected_csv = request.args.get("csv")
    selected_figure = request.args.get("figure")

    csv_reports: List[str] = []
    figures: List[str] = []
    table: Dict[str, Any] | None = None
    selected_error: str | None = None

    if selected_stamp:
        csv_reports = streaming_service.list_streaming_reports(mode, selected_stamp)
        figures = streaming_service.list_streaming_figures(mode, selected_stamp)

        if selected_csv:
            try:
                table = streaming_service.load_streaming_report(selected_csv)
            except FileNotFoundError:
                selected_error = f"Report not found on disk: {selected_csv}"
            except Exception as exc:  # pragma: no cover
                selected_error = f"Failed to load report: {exc}"

    named_csvs = [
        {
            "path": p,
            "name": p.split("/")[-1],
        }
        for p in csv_reports
    ]

    named_figures = [
        {
            "path": f,
            "name": f,
        }
        for f in figures
    ]

    selected_incoming = None
    if selected_stamp:
        selected_incoming = next(
            (i for i in incoming_named if i.get("stamp") == selected_stamp),
            None,
        )

    return render_template(
        "streaming.html",
        mode=mode,
        dataset_choices=dataset_choices,
        stamps=stamps,
        incoming=incoming_named,
        selected_stamp=selected_stamp,
        selected_incoming=selected_incoming,
        csv_reports=named_csvs,
        figures=named_figures,
        selected_csv=selected_csv,
        selected_figure=selected_figure,
        table=table,
        selected_error=selected_error,
    )


@bp.route("/streaming/figure")
def streaming_figure():
    """
    Serve a selected PNG figure from a streaming report directory:
      reports/streaming_<mode>/<stamp>/<filename>.png
    """
    mode = _normalize_mode(request.args.get("mode"))
    stamp = request.args.get("stamp")
    filename = request.args.get("path")

    if not stamp or not filename:
        abort(400)

    figures = streaming_service.list_streaming_figures(mode, stamp)
    if filename not in figures:
        abort(404)

    root = streaming_service._reports_root(mode)  # type: ignore[attr-defined]
    full_path = root / stamp / filename

    if not full_path.exists():
        abort(404)

    return send_file(full_path, mimetype="image/png")


def register_routes(app: Flask) -> None:
    """
    Called from app/__init__.py to attach routes to the Flask app.
    """
    app.register_blueprint(bp)


# Optional local debug entrypoint:
if __name__ == "__main__":  # pragma: no cover
    from app import create_app

    flask_app = create_app()
    # Make sure there is a secret key if you later add flash() or sessions.
    flask_app.config.setdefault("SECRET_KEY", "dev-secret-key-change-me")
    flask_app.run(debug=True, host="0.0.0.0", port=5000)
