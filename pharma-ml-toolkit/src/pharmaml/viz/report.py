"""Multi-format output of results — HTML/PDF/Word reports and exportable tables.

"Various visual formats" in one place:

* :class:`HTMLReport` — a self-contained, dependency-free ``.html``
  artifact: metric cards, styled tables and **embedded** figures
  (base64 png/svg), openable in any browser, e-mailable, and safe to
  archive (no external assets);
* :func:`save_pdf_report` — a multi-page **PDF document** (title +
  metrics, one page per table, one page per figure) via fpdf — the
  legacy scripts' *only* report format, now first-class (non-latin-1
  punctuation is transliterated so the built-in fonts never choke);
* :func:`export_table` — the same DataFrame as ``csv``, ``tsv``,
  ``markdown``, ``json`` or ``html`` depending on the extension (the
  legacy scripts only ever wrote csv/xlsx);
* :func:`export_docx` — any table as a real Word ``.docx`` document
  with a grid-styled header row (the legacy ``approved.py`` wrote a
  bare list of approved-drug names to .docx);
* :func:`metrics_to_dataframe` — any report object
  (:class:`~pharmaml.ml.evaluation.ClassificationReport`,
  :class:`~pharmaml.ml.evaluation.RegressionReport`,
  :class:`~pharmaml.pharma.pk.PKSummary`, plain dict) as a tidy
  metric/value table;
* :func:`save_visual_bundle` — one call that renders figures + tables +
  metrics into a directory and the companion ``report.html`` (plus an
  optional ``report.pdf`` with ``pdf_report=True``); this is what the
  CLI ``--plots``/``data entry`` options use.

Examples
--------
>>> import pandas as pd
>>> from pharmaml.viz.report import export_table, metrics_to_dataframe
>>> frame = metrics_to_dataframe({"roc_auc": 0.9, "n": 4})
>>> (frame.columns.tolist(), frame["metric"].tolist(), frame["value"].tolist())
(['metric', 'value'], ['roc_auc', 'n'], [0.9, 4.0])
>>> board = pd.DataFrame({"model": ["rf", "svm"], "roc_auc_ovo_mean": [0.91, 0.87]})
>>> out = "/tmp/pharmaml_doctest_leaderboard.md"
>>> export_table(board, out).as_posix()
'/tmp/pharmaml_doctest_leaderboard.md'
>>> print(open(out).read())
| model | roc_auc_ovo_mean |
| --- | --- |
| rf | 0.91 |
| svm | 0.87 |
<BLANKLINE>
>>> out_csv = "/tmp/pharmaml_doctest_leaderboard.csv"
>>> export_table(board, out_csv).as_posix()
'/tmp/pharmaml_doctest_leaderboard.csv'
>>> pd.read_csv(out_csv).values.tolist()
[['rf', 0.91], ['svm', 0.87]]
>>> from pathlib import Path
>>> Path(out).unlink(); Path(out_csv).unlink()

The document writers (fpdf + python-docx, both in the ``reports``
extra) upgrade the legacy scripts' *only* output formats:
>>> from pharmaml.viz.report import export_docx, save_pdf_report
>>> out_docx = export_docx(board, "/tmp/pharmaml_doctest_board.docx")
>>> out_docx.suffix
'.docx'
>>> out_pdf = save_pdf_report({}, tables={"leaderboard": board},
...                           out_path="/tmp/pharmaml_doctest_report.pdf",
...                           title="Doctest report")
>>> out_pdf.read_bytes()[:5]                      # a real PDF document
b'%PDF-'
>>> Path(out_docx).unlink(); Path(out_pdf).unlink()

>>> from pharmaml.viz.report import HTMLReport
>>> report = (HTMLReport("QSAR run", subtitle="random_forest | 3 classes")
...           .add_metrics({"roc_auc": 0.91, "n_samples": 4})
...           .add_table(board, "Model leaderboard")
...           .add_text("Cross-validated on 5 folds.", heading="Notes"))
>>> html = report.render()
>>> "<h1>QSAR run</h1>" in html and "<td>rf</td>" in html
True
>>> html.startswith("<!DOCTYPE html>")
True
>>> saved = report.save("/tmp/pharmaml_doctest_report.html")
>>> saved.as_posix()
'/tmp/pharmaml_doctest_report.html'
>>> Path("/tmp/pharmaml_doctest_report.html").unlink()
>>> HTMLReport("t").add_metrics(42)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: add_metrics accepts report objects or dicts
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger
from pharmaml.utils.io import ensure_directory
from pharmaml.viz._compat import close_figure, figure_to_base64, save_figure

__all__ = [
    "TABLE_FORMATS",
    "HTMLReport",
    "export_docx",
    "export_table",
    "metrics_to_dataframe",
    "save_pdf_report",
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
    # ``is not None`` (not truthiness): max_rows=0 is a valid "show nothing"
    shown = frame.head(max_rows) if max_rows is not None else frame
    header = "".join(f"<th>{_escape(col)}</th>" for col in shown.columns)
    rows = []
    for _, row in shown.iterrows():
        cells = "".join(f"<td>{_escape(value)}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    extra = (
        f"\n<p class='subtitle'>… {len(frame) - len(shown)} more row(s) truncated</p>"
        if max_rows is not None and len(frame) > len(shown)
        else ""
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>{extra}"


def _frame_to_markdown(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    """Markdown table writer (avoids the ``tabulate`` dependency)."""
    shown = frame.head(max_rows) if max_rows is not None else frame

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
    return f"{header}\n{rule}\n{body}"


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


#: Non-latin-1 punctuation that appears in PharmaML outputs (em/en
#: dashes, comparison arrows, typographic quotes, ...) transliterated to
#: the ASCII the PDF built-in fonts actually encode.  Everything else
#: outside latin-1 degrades to ``?`` via ``errors="replace"`` -- the
#: report never crashes on a metric name.
_NON_LATIN1_MAP: dict[str, str] = {
    "\u2014": "-",   # em dash
    "\u2013": "-",   # en dash
    "\u2212": "-",   # minus sign
    "\u2192": "->",  # rightwards arrow
    "\u2190": "<-",  # leftwards arrow
    "\u2265": ">=",  # greater-or-equal
    "\u2264": "<=",  # less-or-equal
    "\u2248": "~",   # almost equal
    "\u221e": "inf",  # infinity
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",  # ellipsis
}


def _latin1_safe(text: str) -> str:
    """Transliterate text down to what Helvetica can encode (latin-1)."""
    for src, dst in _NON_LATIN1_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _plain_value(value: Any) -> str:
    """Format a cell for the document writers (compact floats, ascii NaN)."""
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if not np.isfinite(value):
            return "+inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)


#: Rows shown per table page in :func:`save_pdf_report`.  38 rows of
#: 6 mm + heading + header + note stay under the A4 text height
#: (297 - 10 - 15 mm), so the "one page per table" layout holds;
#: longer tables are truncated with a note (auto page break would
#: otherwise split them silently).
_PDF_TABLE_MAX_ROWS = 38


def _pdf_write(pdf: Any, text: str, size: int, style: str = "", h: float = 6.0) -> None:
    """Write one wrapping text block, pinned to the left margin.

    fpdf2's ``multi_cell`` leaves the x cursor at the RIGHT margin after
    a block (``XPos.RIGHT`` default), so a second ``multi_cell(0, ...)``
    would find zero width and raise "Not enough horizontal space";
    fpdf1 reset x to the left margin.  ``set_x(l_margin)`` before every
    block is the behaviour both versions share.
    """
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(0, h, text)


def save_pdf_report(
    figures: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
    metrics: Any = None,
    out_path: str | Path = "report.pdf",
    title: str = "PharmaML results",
    subtitle: str | None = None,
    close: bool = True,
) -> Path:
    """Write a multi-page PDF document report (metrics, tables, figures).

    Upgrades the legacy scripts' *only* report format: every model
    script in the original archive ended by assembling an fpdf PDF with
    the classification report and the confusion/ROC images, one page
    each.  Here the same document is one typed call -- a title page
    with the metrics, one page per table (header row + rows, truncated
    at 38 rows with a note), and one page per figure (rendered to png
    through the headless Agg backend, so no display is needed).

    Requires fpdf (``pip install 'pharmaml[reports]'``); non-latin-1
    punctuation is transliterated (``em dash`` -> ``-``) because the
    built-in Helvetica only encodes latin-1 -- the report never crashes
    on a unicode metric name.

    Parameters
    ----------
    figures:
        ``{page_heading: matplotlib Figure}`` -- one PDF page each.
    tables:
        ``{page_heading: DataFrame}`` -- one PDF page each.
    metrics:
        Report object or dict, rendered as ``metric: value`` lines on
        the title page (via :func:`metrics_to_dataframe`).
    out_path:
        Destination ``.pdf`` path (parents created).
    title, subtitle:
        Title-page headings.
    close:
        Close the figures after writing (pass ``False`` to keep them).

    Returns
    -------
    Path
        The resolved output path.

    Raises
    ------
    DependencyError
        If fpdf is not installed.
    """
    for name in figures:
        _validate_bundle_name(name)
    for name in tables or {}:
        _validate_bundle_name(name)
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - environment guard
        raise DependencyError(
            "fpdf is not installed. Install with: pip install fpdf "
            "(or 'pharmaml[reports]')"
        ) from exc

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page: heading, optional subtitle, metrics as name: value.
    pdf.add_page()
    _pdf_write(pdf, _latin1_safe(title), 16, style="B", h=10)
    if subtitle:
        _pdf_write(pdf, _latin1_safe(subtitle), 11, style="I", h=7)
    if metrics is not None:
        pdf.ln(4)
        _pdf_write(pdf, "Metrics", 12, style="B", h=8)
        for row in metrics_to_dataframe(metrics).itertuples(index=False):
            _pdf_write(pdf, _latin1_safe(f"{row[0]}: {_plain_value(row[1])}"), 10, h=6)

    # One page per table: bold header row, then the body, 60 rows max.
    for name, frame in (tables or {}).items():
        pdf.add_page()
        _pdf_write(pdf, _latin1_safe(name.replace("_", " ").title()), 13, style="B", h=9)
        _pdf_write(
            pdf, _latin1_safe(" | ".join(str(col) for col in frame.columns)), 9, style="B"
        )
        shown = frame.head(_PDF_TABLE_MAX_ROWS)
        for row in shown.itertuples(index=False):
            line = " | ".join(_plain_value(value) for value in row)
            _pdf_write(pdf, _latin1_safe(line), 9, h=6)
        if len(frame) > _PDF_TABLE_MAX_ROWS:
            _pdf_write(pdf, f"... ({len(frame) - _PDF_TABLE_MAX_ROWS} more rows)", 9, h=6)

    # One page per figure: render to a temporary png (Agg, headless-safe)
    # and embed full-width -- fpdf reads the file at image() time, so the
    # temp directory only needs to live for the loop.
    with tempfile.TemporaryDirectory() as tmp:
        for name, figure in figures.items():
            png_path = Path(tmp) / f"{name}.png"
            save_figure(figure, png_path)
            pdf.add_page()
            _pdf_write(pdf, _latin1_safe(name.replace("_", " ").title()), 13, style="B", h=9)
            pdf.image(str(png_path), x=15, w=180)

    pdf.output(str(dest))
    if close:
        for figure in figures.values():
            close_figure(figure)
    _LOG.info("PDF report: %d page(s) -> %s", pdf.page, dest)
    return dest


def export_docx(
    frame: pd.DataFrame,
    path: str | Path,
    max_rows: int | None = None,
) -> Path:
    """Write a DataFrame as a Word ``.docx`` document with a grid table.

    Upgrades the legacy ``approved.py``, which wrote a bare list of
    approved-drug names to a .docx: any table now exports with a bold
    header row and a bordered grid, cell-formatted like every other
    PharmaML export (compact floats, ``NaN``/``inf`` never leak as raw
    ``nan``).

    Requires python-docx (``pip install 'pharmaml[reports]'``).

    Parameters
    ----------
    frame:
        The table to write.
    path:
        Destination ``.docx`` path (parent directories created).
    max_rows:
        Row cap (``None`` writes every row).

    Returns
    -------
    Path
        The resolved output path.

    Raises
    ------
    DependencyError
        If python-docx is not installed.
    """
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - environment guard
        raise DependencyError(
            "python-docx is not installed. Install with: pip install python-docx "
            "(or 'pharmaml[reports]')"
        ) from exc

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shown = frame.head(max_rows) if max_rows is not None else frame

    document = Document()
    table = document.add_table(rows=1, cols=len(shown.columns))
    table.style = "Table Grid"
    for cell, column in zip(table.rows[0].cells, shown.columns):
        cell.text = str(column)
        cell.paragraphs[0].runs[0].bold = True
    for row in shown.itertuples(index=False):
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = _plain_value(value)
    document.save(str(dest))
    _LOG.info("Wrote %d row(s) -> %s", len(shown), dest)
    return dest


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
    def add_metrics(self, metrics: Any, heading: str | None = None) -> HTMLReport:
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
    ) -> HTMLReport:
        """Styled HTML table (long tables are truncated at ``max_rows``)."""
        parts = [f"<h2>{_escape(heading)}</h2>"] if heading else []
        parts.append(_frame_to_html(frame, max_rows=max_rows))
        self._blocks.append("".join(parts))
        return self

    def add_text(self, text: str, heading: str | None = None) -> HTMLReport:
        """A paragraph of escaped text (notes, provenance, next steps)."""
        parts = [f"<h2>{_escape(heading)}</h2>"] if heading else []
        parts.append(f"<p>{_escape(text)}</p>")
        self._blocks.append("".join(parts))
        return self

    def add_figure(self, figure: Any, caption: str | None = None, fmt: str = "png") -> HTMLReport:
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


def _validate_bundle_name(name: str) -> None:
    """Reject stem keys that would escape the bundle directory.

    ``../escape`` or ``sub/fig`` flow unvalidated into ``out / f"{name}.{fmt}"``
    and would write OUTSIDE the bundle dir; only plain file names are safe.
    """
    if Path(name).name != name:
        raise DataValidationError(
            f"bundle artifact name {name!r} must be a plain file name "
            "(no directories or '..' components)"
        )


def save_visual_bundle(
    figures: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
    metrics: Any = None,
    out_dir: str | Path = "reports",
    title: str = "PharmaML results",
    subtitle: str | None = None,
    formats: tuple[str, ...] = ("png",),
    close: bool = True,
    pdf_report: bool = False,
) -> list[Path]:
    """Render figures + tables + metrics into one directory + HTML report.

    The bundle layout::

        <out_dir>/report.html        self-contained visual report
        <out_dir>/report.pdf         multi-page PDF document (pdf_report=True)
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
    pdf_report:
        Also write ``report.pdf`` — the multi-page document (title +
        metrics, one page per table, one page per figure) via
        :func:`save_pdf_report`.  Requires fpdf.

    Returns
    -------
    list[Path]
        Every artifact written, in a stable order (report.html first,
        report.pdf second when requested).
    """
    # Validate stem keys BEFORE creating anything: a "../x" key must fail
    # typed instead of writing outside out_dir.
    for name in figures:
        _validate_bundle_name(name)
    for name in tables or {}:
        _validate_bundle_name(name)

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

    pdf_path: Path | None = None
    if pdf_report:
        # After the figure files + HTML embedding, from the same (still
        # open) figure objects; the bundle closes them below itself.
        pdf_path = save_pdf_report(
            figures,
            tables=tables,
            metrics=metrics,
            out_path=out / "report.pdf",
            title=title,
            subtitle=subtitle,
            close=False,
        )

    written.insert(0, report.save(out / "report.html"))
    if pdf_path is not None:
        written.insert(1, pdf_path)
    if close:
        for figure in figures.values():
            close_figure(figure)
    _LOG.info("Visual bundle: %d artifact(s) -> %s", len(written), out)
    return written
