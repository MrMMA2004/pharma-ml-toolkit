"""Multi-format output of results — HTML reports and exportable tables.

"Various visual formats" in one place:

* :class:`HTMLReport` — a self-contained, dependency-free ``.html``
  artifact: metric cards, styled tables and **embedded** figures
  (base64 png/svg), openable in any browser, e-mailable, and safe to
  archive (no external assets);
* :func:`export_table` — the same DataFrame as ``csv``, ``tsv``,
  ``markdown``, ``json`` or ``html`` depending on the extension (the
  legacy scripts only ever wrote csv/xlsx);
* :func:`metrics_to_dataframe` — any report object
  (:class:`~pharmaml.ml.evaluation.ClassificationReport`,
  :class:`~pharmaml.ml.evaluation.RegressionReport`,
  :class:`~pharmaml.pharma.pk.PKSummary`, plain dict) as a tidy
  metric/value table;
* :func:`save_visual_bundle` — one call that renders figures + tables +
  metrics into a directory and the companion ``report.html``; this is
  what the CLI ``--plots``/``data entry`` options use.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.io import ensure_directory
from pharmaml.viz._compat import close_figure, figure_to_base64, save_figure

__all__ = [
    "TABLE_FORMATS",
    "HTMLReport",
    "export_table",
    "metrics_to_dataframe",
    "save_visual_bundle",
]

_LOG = get_logger(__name__)

#: Extensions :func:`export_table` understands.
TABLE_FORMATS: tuple[str, ...] = ("csv", "tsv", "md", "json", "html")

_REPORT_CSS = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
       margin: 0 auto; max-width: 960px; padding: 2rem 1.25rem 3rem;
       color: #1F2937; background: #FFFFFF; line-height: 1.55; }
h1 { font-size: 1.6rem; border-bottom: 2px solid #0F766E; padding-bottom: .4rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; color: #0F766E; }
p.subtitle { color: #6B7280; margin-top: -0.6rem; }
.metrics { display: flex; flex-wrap: wrap; gap: .6rem; margin: 1rem 0; }
.metric { border: 1px solid #E5E7EB; border-radius: .5rem; padding: .5rem .9rem;
          background: #F8FAFC; min-width: 7rem; }
.metric b { display: block; font-size: 1.05rem; color: #0F766E; }
.metric span { font-size: .75rem; color: #6B7280; text-transform: uppercase;
               letter-spacing: .04em; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: .5rem 0 1rem; }
th, td { border: 1px solid #E5E7EB; padding: .32rem .55rem; text-align: left; }
th { background: #F1F5F9; }
tbody tr:nth-child(even) { background: #F8FAFC; }
img { max-width: 100%; height: auto; border: 1px solid #E5E7EB; border-radius: .4rem;
      margin: .35rem 0 1.1rem; }
figcaption { font-size: .8rem; color: #6B7280; margin-top: -0.8rem; margin-bottom: 1rem; }
footer { margin-top: 2.5rem; font-size: .75rem; color: #9CA3AF;
         border-top: 1px solid #E5E7EB; padding-top: .6rem; }
@media (prefers-color-scheme: dark) {
  body { background: #0B1220; color: #E5E7EB; }
  h1 { border-color: #14B8A6; } h2 { color: #14B8A6; }
  .metric { background: #101B2D; border-color: #1F2A44; }
  .metric b { color: #14B8A6; }
  th { background: #101B2D; } td, th { border-color: #1F2A44; }
  tbody tr:nth-child(even) { background: #0E1728; }
  img { border-color: #1F2A44; }
}
"""


def _escape(value: Any) -> str:
    """HTML-escape a value, formatting floats compactly."""
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—" if np.isnan(value) else ("+∞" if value > 0 else "−∞")
        return f"{value:.6g}"
    return html.escape(str(value))


def _frame_to_html(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a DataFrame as a styled HTML table (no external CSS)."""
    shown = frame.head(max_rows) if max_rows else frame
    header = "".join(f"<th>{_escape(col)}</th>" for col in shown.columns)
    rows = []
    for _, row in shown.iterrows():
        cells = "".join(f"<td>{_escape(value)}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    extra = (
        f"\n<p class='subtitle'>… {len(frame) - len(shown)} more row(s) truncated</p>"
        if max_rows and len(frame) > max_rows
        else ""
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>{extra}"


def _frame_to_markdown(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    """Markdown table writer (avoids the ``tabulate`` dependency)."""
    shown = frame.head(max_rows) if max_rows else frame

    def cell(value: Any) -> str:
        text = "—" if (isinstance(value, float) and np.isnan(value)) else (
            f"{value:.6g}" if isinstance(value, float) else str(value)
        )
        return text.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(col) for col in shown.columns) + " |"
    rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(cell(value) for value in row.tolist()) + " |"
        for _, row in shown.iterrows()
    )
    return "\n".join([header, rule, body])


def metrics_to_dataframe(report: Any) -> pd.DataFrame:
    """Any report object as a tidy ``metric | value`` DataFrame.

    Accepts :class:`~pharmaml.ml.evaluation.ClassificationReport`,
    :class:`~pharmaml.ml.evaluation.RegressionReport`,
    :class:`~pharmaml.pharma.pk.PKSummary`, or a plain mapping.
    """
    if hasattr(report, "as_dict"):
        payload = report.as_dict()
    elif isinstance(report, dict):
        payload = report
    else:
        raise DataValidationError(
            "metrics_to_dataframe accepts report objects with as_dict() or a plain dict"
        )
    return pd.DataFrame(
        {"metric": list(payload.keys()), "value": list(payload.values())}
    )


def export_table(frame: pd.DataFrame, path: str | Path, max_rows: int | None = None) -> Path:
    """Write a DataFrame in a *visual* format chosen by the extension.

    Supported: ``.csv``, ``.tsv``, ``.md``/``.markdown`` (GitHub-ready
    table), ``.json`` (records orientation), ``.html`` (styled table
    that renders standalone in a browser).

    Returns
    -------
    Path
        The resolved output path (parent directories created).
    """
    file_path = Path(path)
    fmt = file_path.suffix.lower().lstrip(".")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        frame.to_csv(file_path, index=False)
    elif fmt == "tsv":
        frame.to_csv(file_path, index=False, sep="\t")
    elif fmt in {"md", "markdown"}:
        file_path = file_path.with_suffix(".md")
        file_path.write_text(_frame_to_markdown(frame, max_rows) + "\n", encoding="utf-8")
    elif fmt == "json":
        # ``default=`` is not accepted by DataFrame.to_json — convert
        # non-serialisable cells (numpy scalars, NaN-safe) up front.
        serialisable = frame.copy()
        for column in serialisable.columns:
            if not pd.api.types.is_numeric_dtype(serialisable[column]):
                serialisable[column] = serialisable[column].astype(str)
        serialisable.to_json(file_path, orient="records", indent=2)
    elif fmt == "html":
        document = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>table export</title><style>" + _REPORT_CSS + "</style></head><body>"
            + _frame_to_html(frame, max_rows) + "</body></html>"
        )
        file_path.write_text(document, encoding="utf-8")
    else:
        raise DataValidationError(
            f"Unsupported table format {file_path.suffix!r}; "
            f"supported: {', '.join(TABLE_FORMATS)}"
        )
    _LOG.info("Exported %d row(s) -> %s", len(frame), file_path)
    return file_path


class HTMLReport:
    """Builder for a self-contained HTML report.

    Usage::

        report = HTMLReport("QSAR run", subtitle="random_forest · 3 classes")
        report.add_metrics(classification_report)          # metric cards
        report.add_table(leaderboard, "Model leaderboard") # styled table
        report.add_figure(figure, "Confusion matrix")      # embedded png/svg
        report.save("reports/report.html")                 # single artifact

    The output file references **zero external resources** — figures are
    base64-embedded — so it renders identically offline and survives
    being moved or e-mailed.
    """

    def __init__(self, title: str, subtitle: str | None = None) -> None:
        self.title = str(title)
        self.subtitle = subtitle
        self._blocks: list[str] = []
        self._figure_count = 0

    # -- content builders -------------------------------------------------
    def add_metrics(self, metrics: Any, heading: str | None = None) -> "HTMLReport":
        """Metric cards from a report object or a ``{name: value}`` dict."""
        if hasattr(metrics, "as_dict"):
            payload = metrics.as_dict()
        elif isinstance(metrics, dict):
            payload = metrics
        else:
            raise DataValidationError("add_metrics accepts report objects or dicts")

        cards = "".join(
            f"<div class='metric'><span>{_escape(key)}</span>"
            f"<b>{_escape(value)}</b></div>"
            for key, value in payload.items()
        )
        self._blocks.append(
            (f"<h2>{_escape(heading)}</h2>" if heading else "")
            + f"<div class='metrics'>{cards}</div>"
        )
        return self

    def add_table(
        self,
        frame: pd.DataFrame,
        heading: str | None = None,
        max_rows: int | None = 40,
    ) -> "HTMLReport":
        """Styled HTML table (long tables are truncated at ``max_rows``)."""
        parts = [f"<h2>{_escape(heading)}</h2>"] if heading else []
        parts.append(_frame_to_html(frame, max_rows=max_rows))
        self._blocks.append("".join(parts))
        return self

    def add_text(self, text: str, heading: str | None = None) -> "HTMLReport":
        """A paragraph of escaped text (notes, provenance, next steps)."""
        parts = [f"<h2>{_escape(heading)}</h2>"] if heading else []
        parts.append(f"<p>{_escape(text)}</p>")
        self._blocks.append("".join(parts))
        return self

    def add_figure(self, figure: Any, caption: str | None = None, fmt: str = "png") -> "HTMLReport":
        """Embed a matplotlib figure inline (base64; not saved to disk)."""
        if fmt not in {"png", "svg"}:
            raise DataValidationError(
                "HTML embedding supports 'png' or 'svg' figures "
                f"(vector for pdf is not embeddable in browsers), got {fmt!r}"
            )
        data_uri = figure_to_base64(figure, fmt=fmt)
        self._figure_count += 1
        alt = _escape(caption or f"figure {self._figure_count}")
        block = f"<img src='{data_uri}' alt='{alt}'/>"
        if caption:
            block += f"<figcaption>{_escape(caption)}</figcaption>"
        self._blocks.append(block)
        return self

    # -- rendering --------------------------------------------------------
    def render(self) -> str:
        """Return the full HTML document as a string."""
        subtitle = (
            f"<p class='subtitle'>{_escape(self.subtitle)}</p>" if self.subtitle else ""
        )
        body = "\n".join(self._blocks)
        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
            f"<title>{_escape(self.title)}</title>\n<style>{_REPORT_CSS}</style>\n"
            "</head>\n<body>\n"
            f"<h1>{_escape(self.title)}</h1>\n{subtitle}\n{body}\n"
            "<footer>PharmaML report — self-contained HTML (figures embedded, "
            "no external assets)</footer>\n</body>\n</html>\n"
        )

    def save(self, path: str | Path) -> Path:
        """Write the report; parent directories are created."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.render(), encoding="utf-8")
        _LOG.info("HTML report (%d figure(s)) -> %s", self._figure_count, file_path)
        return file_path


def save_visual_bundle(
    figures: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
    metrics: Any = None,
    out_dir: str | Path = "reports",
    title: str = "PharmaML results",
    subtitle: str | None = None,
    formats: tuple[str, ...] = ("png",),
    close: bool = True,
) -> list[Path]:
    """Render figures + tables + metrics into one directory + HTML report.

    The bundle layout::

        <out_dir>/report.html        self-contained visual report
        <out_dir>/<figure>.png       every figure as a file
        <out_dir>/<table>.csv        every table as machine-readable csv
        <out_dir>/<table>.md         ... and human-readable markdown

    Parameters
    ----------
    figures:
        ``{filename_stem: matplotlib Figure}``.
    tables:
        ``{filename_stem: DataFrame}`` — each is exported as csv *and*
        markdown (plus embedded in the HTML report).
    metrics:
        Report object or dict rendered as metric cards at the top.
    formats:
        Figure file formats (``png``/``svg``/``pdf``); the HTML report
        always embeds png.
    close:
        Close the figures after rendering (set False to keep drawing).

    Returns
    -------
    list[Path]
        Every artifact written, in a stable order (report first).
    """
    out = ensure_directory(out_dir)
    written: list[Path] = []

    report = HTMLReport(title, subtitle=subtitle)
    if metrics is not None:
        report.add_metrics(metrics, heading="Metrics")

    for name, frame in (tables or {}).items():
        csv_path = export_table(frame, out / f"{name}.csv")
        md_path = export_table(frame, out / f"{name}.md")
        written.extend([csv_path, md_path])
        report.add_table(frame, heading=name.replace("_", " ").title())

    for name, figure in figures.items():
        for fmt in formats:
            written.append(save_figure(figure, out / f"{name}.{fmt}"))
        report.add_figure(figure, caption=name.replace("_", " ").title())

    written.insert(0, report.save(out / "report.html"))
    if close:
        for figure in figures.values():
            close_figure(figure)
    _LOG.info("Visual bundle: %d artifact(s) -> %s", len(written), out)
    return written
