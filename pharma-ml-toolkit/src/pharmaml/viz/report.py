"""Multi-format output of results — HTML/PDF/Word reports and exportable tables.

"Various visual formats" in one place:

* :class:`HTMLReport` — a self-contained, dependency-free ``.html``
  artifact: metric cards, styled tables and **embedded** figures
  (base64 png/svg), openable in any browser, e-mailable, and safe to
  archive (no external assets);
* :func:`save_pdf_report` — a multi-page **PDF document** (title +
  metrics, one page per table, one page per figure) via fpdf — the
  legacy scripts' *only* report format, now first-class (non-latin-1
  punctuation is transliterated so the built-in fonts never choke),
  with a ``theme`` knob: the classic white paper or a ``dark``
  dark-ink variant that matches the app the report left;
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
>>> "<nav class='contents'" in html and "<li><a href='#pg-1' title='Jump to this page in the report'>Model leaderboard</a></li>" in html
True
>>> "<a class='back-contents' href='#contents' title='Back to the contents block at the top of the report'>\u2191 Contents</a>" in html
True
>>> html.startswith("<!DOCTYPE html>")
True
>>> HTMLReport("no pages").add_text("only prose").render().count("<nav class='contents'")
0
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

import base64
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
    "PDF_THEMES",
    "REPORT_THEMES",
    "TABLE_FORMATS",
    "HTMLReport",
    "export_docx",
    "export_table",
    "metrics_to_dataframe",
    "save_pdf_report",
    "save_visual_bundle",
]

_LOG = get_logger(__name__)

#: Extensions :func:`export_table` understands -- the round-201
#: Excel absorption added ``xlsx`` (openpyxl engine) beside the five
#: visual formats, so every playground table card can hand its result
#: to a spreadsheet with the same one-call story.
TABLE_FORMATS: tuple[str, ...] = ("csv", "tsv", "md", "json", "html", "xlsx")

_REPORT_CSS = """
/* Theme engine (100x-126): 'auto' follows the OS (the original
   prefers-color-scheme behaviour, palette preserved), 'light' and
   'dark' pin one palette.  The baked attribute is the INITIAL state;
   the in-report toggle button overrides it at runtime and remembers
   the reader's choice per browser (localStorage).  Muted greys stay
   fixed across themes (the shipped auto-dark never varied them). */
:root { --bg: #FFFFFF; --fg: #1F2937; --muted: #6B7280; --faint: #9CA3AF;
         --accent: #0F766E; --border: #E5E7EB; --panel: #F8FAFC;
         --panel-2: #F1F5F9; --row: #F8FAFC;
         color-scheme: light dark; }
html[data-theme='light'] { color-scheme: light; }
html[data-theme='dark'] { --bg: #0B1220; --fg: #E5E7EB; --accent: #14B8A6;
                          --border: #1F2A44; --panel: #101B2D; --panel-2: #101B2D;
                          --row: #0E1728; color-scheme: dark; }
@media (prefers-color-scheme: dark) {
  html[data-theme='auto'] { --bg: #0B1220; --fg: #E5E7EB; --accent: #14B8A6;
                            --border: #1F2A44; --panel: #101B2D; --panel-2: #101B2D;
                            --row: #0E1728; color-scheme: dark; }
}
body { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
       margin: 0 auto; max-width: 960px; padding: 2rem 1.25rem 3rem;
       color: var(--fg); background: var(--bg); line-height: 1.55; }
h1 { font-size: 1.6rem; border-bottom: 2px solid var(--accent); padding-bottom: .4rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; color: var(--accent); }
p.subtitle { color: var(--muted); margin-top: -0.6rem; }
.metrics { display: flex; flex-wrap: wrap; gap: .6rem; margin: 1rem 0; }
.metric { border: 1px solid var(--border); border-radius: .5rem; padding: .5rem .9rem;
          background: var(--panel); min-width: 7rem; }
.metric b { display: block; font-size: 1.05rem; color: var(--accent); }
.metric span { font-size: .75rem; color: var(--muted); text-transform: uppercase;
               letter-spacing: .04em; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: .5rem 0 1rem; }
th, td { border: 1px solid var(--border); padding: .32rem .55rem; text-align: left; }
th { background: var(--panel-2); }
tbody tr:nth-child(even) { background: var(--row); }
img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: .4rem;
      margin: .35rem 0 1.1rem; }
figcaption { font-size: .8rem; color: var(--muted); margin-top: -0.8rem; margin-bottom: 1rem; }
footer { margin-top: 2.5rem; font-size: .75rem; color: var(--faint);
         border-top: 1px solid var(--border); padding-top: .6rem; }
/* (100x-136) the Contents block -- the HTML twin of the PDF title
   page's mini table-of-contents: a bordered panel right under the
   subtitle listing every table/figure page name in walking order. */
nav.contents { border: 1px solid var(--border); border-radius: .5rem; background: var(--panel);
               padding: .75rem 1rem .85rem; margin: 1.1rem 0 0; }
nav.contents h2 { margin-top: 0; font-size: .95rem; }
ul.contents-list { margin: .2rem 0 0; padding-left: 1.15rem; font-size: .85rem; }
ul.contents-list li { padding: .1rem 0; }
ul.contents-list li::marker { color: var(--accent); }
/* (100x-137) the Contents lines are LINKS -- every table/figure page
   is an anchor target (the id its wrapper section carries), so a
   click jumps to the page; the PDF twin's ink is frozen, its lines
   stay plain text.  The link inherits the list's ink until the
   reader points at it (hover/focus), and the jumped-to page flashes
   a short accent ring (:target) so the arrival is never lost.
   (100x-140) every line also carries a title tooltip -- the POINTER
   confession the return link gained in 100x-139, applied
   symmetrically to the forward leg: a hovering pointer gets told
   where the jump goes, the same thing the aria world always knew. */
ul.contents-list a { color: inherit; text-decoration: none; }
ul.contents-list a:hover, ul.contents-list a:focus-visible {
  color: var(--accent); text-decoration: underline; }
/* (100x-138) the RETURN leg -- every page wrapper ends in a small
   "\u2191 Contents" link back to the header's block, so a reader deep
   in a long report is one click from the top (the Contents lines
   jump one way; this is the other).  Muted until hovered/focused,
   the same affordance discipline as the Contents lines, and the
   nav itself flashes the same accent ring when targeted so the
   return landing is never lost either.
   (100x-139) the link carries a title tooltip -- the pointer world
   gets the same confession the aria world always had. */
a.back-contents { display: block; text-align: right; font-size: .75rem;
                  color: var(--muted); text-decoration: none;
                  margin: .15rem 0 .8rem; }
a.back-contents:hover, a.back-contents:focus-visible {
  color: var(--accent); text-decoration: underline; }
section.page, figure.page, nav.contents { scroll-margin-top: 1.25rem; }
section.page:target, figure.page:target, nav.contents:target {
  outline: 2px solid var(--accent); outline-offset: .4rem;
  border-radius: .5rem; }
figure.page { margin: 0; } /* the wrapper resets the UA's 40px inset */
.theme-toggle { position: fixed; top: .8rem; right: 1rem; font: inherit; font-size: .78rem;
  border: 1px solid var(--border); background: var(--panel); color: var(--fg);
  border-radius: .4rem; padding: .28rem .65rem; cursor: pointer; opacity: .82;
  letter-spacing: .02em; }
.theme-toggle:hover, .theme-toggle:focus-visible { opacity: 1; border-color: var(--accent); }
"""

#: Themes :class:`HTMLReport` understands (100x-126).
REPORT_THEMES: tuple[str, ...] = ("auto", "light", "dark")

#: Themes :func:`save_pdf_report` understands (100x-129).  No ``auto``
#: here, on purpose: a PDF's ink is frozen at WRITE time -- there is
#: no reading-time toggle like the HTML report's -- so the caller
#: resolves any OS preference before the call (the playground maps its
#: theme radios through ``_resolve_auto_theme``).
PDF_THEMES: tuple[str, ...] = ("light", "dark")

#: The dark-ink variant's palette (100x-129) -- deliberately the
#: playground's own dark theme, so a PDF saved while the app is dark
#: matches the screen it left: page = the app's background, ink = the
#: app's text, accent = the app's highlight.  Figures stay on a white
#: plate (matplotlib renders them on white) -- the same bright-plate
#: look the dark HTML report gives its embedded figures.
_PDF_DARK: dict[str, tuple[int, int, int]] = {
    "page": (21, 26, 24),  # #151a18 -- the playground dark BG
    "ink": (231, 236, 233),  # #e7ece9 -- the playground dark INK
    "muted": (157, 171, 164),  # #9daba4 -- the playground dark MUTED
    "accent": (22, 169, 138),  # #16a98a -- the playground dark ACCENT
    "plate": (255, 255, 255),
}

#: Label the toggle button carries per theme (server-side fallback;
#: the inline script relabels on load and on every cycle).
_THEME_LABELS: dict[str, str] = {"auto": "\u25d0 auto", "light": "\u2600 light", "dark": "\u263d dark"}

#: (100x-139) The RETURN link every anchored page ends in -- ONE
#: string, minted once, so the table and figure wrappers can never
#: drift apart (the same lockstep discipline the page ids follow).  The
#: ``title`` tooltip confesses the jump target to the POINTER world --
#: the aria world already knows (the link's own text names the
#: destination); a pointer hovering the small muted link had no clue.
_BACK_CONTENTS_LINK = (
    "<a class='back-contents' href='#contents' "
    "title='Back to the contents block at the top of the report'>"
    "\u2191 Contents</a>"
)

#: Inline engine of the theme toggle -- zero external assets (the
#: self-contained contract holds: no network, no companion files).
_THEME_SCRIPT = """(function () {
  var root = document.documentElement;
  var button = document.getElementById('theme-toggle');
  var ORDER = ['auto', 'light', 'dark'];
  function apply(mode) {
    root.setAttribute('data-theme', mode);
    if (button) {
      button.textContent = ({auto: '\u25d0 auto', light: '\u2600 light',
                             dark: '\u263d dark'})[mode] || mode;
    }
  }
  var saved = null;
  try { saved = window.localStorage.getItem('pharmaml-report-theme'); } catch (e) {}
  if (saved === 'auto' || saved === 'light' || saved === 'dark') { apply(saved); }
  if (button) {
    button.addEventListener('click', function () {
      var current = root.getAttribute('data-theme') || 'auto';
      apply(ORDER[(ORDER.indexOf(current) + 1) % ORDER.length]);
      try {
        window.localStorage.setItem('pharmaml-report-theme',
                                   root.getAttribute('data-theme'));
      } catch (e) {}
    });
  }
})();"""


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
    that renders standalone in a browser) and ``.xlsx`` (Excel
    workbook via the openpyxl engine, round 201 -- the spreadsheet
    twin of the CSV arm, one sheet, no index column).

    Returns
    -------
    Path
        The resolved output path (parent directories created).

    Examples
    --------
    >>> from pharmaml.viz import TABLE_FORMATS
    >>> sorted(TABLE_FORMATS)
    ['csv', 'html', 'json', 'md', 'tsv', 'xlsx']
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
    elif fmt == "xlsx":
        # (round 201) the Excel arm -- openpyxl writes the workbook
        # (``engine=`` pinned so a headless xlrd never gets a chance);
        # the full frame lands in ONE sheet with no index column,
        # mirroring the CSV arm's layout exactly (``read_table`` reads
        # it back with the same columns).
        frame.to_excel(file_path, index=False, sheet_name="table", engine="openpyxl")
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


def _pdf_write(
    pdf: Any,
    text: str,
    size: int,
    style: str = "",
    h: float = 6.0,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Write one wrapping text block, pinned to the left margin.

    fpdf2's ``multi_cell`` leaves the x cursor at the RIGHT margin after
    a block (``XPos.RIGHT`` default), so a second ``multi_cell(0, ...)``
    would find zero width and raise "Not enough horizontal space";
    fpdf1 reset x to the left margin.  ``set_x(l_margin)`` before every
    block is the behaviour both versions share.

    ``color`` (100x-129) is the dark-ink variant's knob: ``None`` (the
    light paper default) never touches the text colour, so the classic
    output stays byte-identical; a tuple is applied before the block,
    and because fpdf colours STICK across writes, every dark-mode call
    passes its colour explicitly -- nothing drifts back to a stale ink.
    """
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", style, size)
    if color is not None:
        pdf.set_text_color(*color)
    pdf.multi_cell(0, h, text)


def save_pdf_report(
    figures: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
    metrics: Any = None,
    out_path: str | Path = "report.pdf",
    title: str = "PharmaML results",
    subtitle: str | None = None,
    close: bool = True,
    theme: str = "light",
) -> Path:
    """Write a multi-page PDF document report (metrics, tables, figures).

    Upgrades the legacy scripts' *only* report format: every model
    script in the original archive ended by assembling an fpdf PDF with
    the classification report and the confusion/ROC images, one page
    each.  Here the same document is one typed call -- a title page
    with the metrics, one page per table (header row + rows, truncated
    at 38 rows with a note), and one page per figure (rendered to png
    through the headless Agg backend, so no display is needed).

    (100x-135) The title page carries a CONTENTS section -- a mini
    table-of-contents listing every page name in the walking order
    (the table pages first, then the figure pages, exactly the
    sequence the document writes them), each line the SAME string the
    page's own heading prints.  The playground's count-prefixed names
    ("figure N · <caption>" / "table N · <caption>", 100x-134) make
    the list read like the picklist's numbered rows; direct callers'
    names read title-cased, the way their page headings always have.
    A report with no tables and no figures stays a pure title page --
    a contents list of nothing is noise, so it is skipped entirely.

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
    theme:
        (100x-129) Ink variant: ``"light"`` (default -- the classic
        white paper, byte-identical to every prior release) or
        ``"dark"`` -- near-black pages carrying the light ink, the
        playground's own dark palette, so the PDF matches the app it
        was saved from.  No ``"auto"`` here: a PDF's ink is frozen at
        write time, so the caller resolves any OS preference first.
        Figures stay readable in both variants -- in the dark one they
        sit on a white plate, the same bright-plate look the dark HTML
        report gives its embedded figures.

    Returns
    -------
    Path
        The resolved output path.

    Raises
    ------
    DependencyError
        If fpdf is not installed.
    ValueError
        If ``theme`` is not one of :data:`PDF_THEMES`.
    """
    if theme not in PDF_THEMES:
        raise ValueError(
            f"theme must be one of {PDF_THEMES}, got {theme!r} -- a PDF's ink "
            "is frozen at write time, so resolve any OS preference at the "
            "caller (the playground maps its theme radios first)"
        )
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

    dark = theme == "dark"
    # Light never touches a colour (byte-identical classic output);
    # dark carries the playground's own palette -- page fill, body ink,
    # muted notes and accent headings.
    ink: tuple[int, int, int] | None = _PDF_DARK["ink"] if dark else None
    muted: tuple[int, int, int] | None = _PDF_DARK["muted"] if dark else None
    accent: tuple[int, int, int] | None = _PDF_DARK["accent"] if dark else None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    if dark:
        # fpdf2 applies the page background to the current page AND
        # every later one -- including pages the auto page break
        # creates mid-table, so a long dark table never bleeds white.
        pdf.set_page_background(_PDF_DARK["page"])

    # Title page: heading, optional subtitle, the CONTENTS section
    # (100x-135), then the metrics as name: value.
    pdf.add_page()
    _pdf_write(pdf, _latin1_safe(title), 16, style="B", h=10, color=accent)
    if subtitle:
        _pdf_write(pdf, _latin1_safe(subtitle), 11, style="I", h=7, color=muted)

    def _page_heading(name: str) -> str:
        """The string a page's own heading prints (TOC lines match it)."""
        return _latin1_safe(name.replace("_", " ").title())

    # (100x-135) the mini table-of-contents: every page name, tables
    # first, in the walking order -- the SAME string the page's own
    # heading prints (the shared helper above), so the list can never
    # drift from what the pages actually say.  A long list simply
    # flows -- auto page break is on, so thirty pages of contents
    # stay honest (the list is never truncated or capped).
    if tables or figures:
        pdf.ln(4)
        _pdf_write(pdf, "Contents", 12, style="B", h=8, color=accent)
        # BOTH key lists, walked separately -- a name shared by a table
        # and a figure is TWO pages, and the list says so twice (a dict
        # merge would silently drop one of them).
        for name in [*(tables or {}), *figures]:
            _pdf_write(pdf, _page_heading(name), 10, h=6, color=ink)
    if metrics is not None:
        pdf.ln(4)
        _pdf_write(pdf, "Metrics", 12, style="B", h=8, color=accent)
        for row in metrics_to_dataframe(metrics).itertuples(index=False):
            _pdf_write(
                pdf, _latin1_safe(f"{row[0]}: {_plain_value(row[1])}"), 10, h=6, color=ink
            )

    # One page per table: bold header row, then the body, 60 rows max.
    for name, frame in (tables or {}).items():
        pdf.add_page()
        _pdf_write(pdf, _page_heading(name), 13, style="B", h=9, color=accent)
        _pdf_write(
            pdf,
            _latin1_safe(" | ".join(str(col) for col in frame.columns)),
            9,
            style="B",
            color=muted,
        )
        shown = frame.head(_PDF_TABLE_MAX_ROWS)
        for row in shown.itertuples(index=False):
            line = " | ".join(_plain_value(value) for value in row)
            _pdf_write(pdf, _latin1_safe(line), 9, h=6, color=ink)
        if len(frame) > _PDF_TABLE_MAX_ROWS:
            _pdf_write(
                pdf, f"... ({len(frame) - _PDF_TABLE_MAX_ROWS} more rows)", 9, h=6, color=muted
            )

    # One page per figure: render to a temporary png (Agg, headless-safe)
    # and embed full-width -- fpdf reads the file at image() time, so the
    # temp directory only needs to live for the loop.
    with tempfile.TemporaryDirectory() as tmp:
        for name, figure in figures.items():
            png_path = Path(tmp) / f"{name}.png"
            save_figure(figure, png_path)
            pdf.add_page()
            _pdf_write(pdf, _page_heading(name), 13, style="B", h=9, color=accent)
            if dark:
                # White plate under the figure: matplotlib renders on
                # white, and a raw white png on a near-black page reads
                # as a hole -- the plate frames it as a deliberate
                # bright window (the dark HTML report's own look).
                # PIL is matplotlib's hard dependency, but a missing
                # PIL only drops the plate, never the figure.
                plate_h: float | None = None
                try:
                    from PIL import Image

                    with Image.open(png_path) as raster:
                        px_w, px_h = raster.size
                    if px_w:
                        plate_h = 180.0 * float(px_h) / float(px_w)
                except Exception:  # noqa: BLE001 - the plate is cosmetic
                    plate_h = None
                if plate_h is not None:
                    pdf.set_fill_color(*_PDF_DARK["plate"])
                    pdf.rect(12, pdf.get_y() + 1.5, 186, plate_h + 3.0, "F")
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

    (100x-126) Themes: ``theme`` pins the INITIAL palette — ``"auto"``
    follows the reader's OS (the original ``prefers-color-scheme``
    behaviour), ``"light"`` / ``"dark"`` force one — and every report
    carries a small fixed toggle button (top-right) that cycles
    auto → light → dark at reading time, remembered per browser via
    ``localStorage``.  The script is inline, so the self-contained
    contract holds.

    (100x-136) The header carries a CONTENTS block — the HTML twin of
    the PDF title page's mini table-of-contents: every table and
    figure page name, collected in the walking order the file writes
    them, each line the SAME string the page itself prints (the
    table's ``<h2>`` heading / the figure's ``<figcaption>`` caption;
    a heading-less block lists its positional fallback ``table N`` /
    ``figure N``, the identity its ``alt`` already carries).  A report
    with no tables and no figures skips the block entirely — a
    contents list of nothing is noise, the PDF twin's rule.

    (100x-137) Every Contents line is a CLICKABLE anchor: each
    table/figure page rides inside its own wrapper (``<section
    class='page' id='pg-N'>`` for tables, ``<figure class='page'
    id='pg-N'>`` for figures — the figcaption's legitimate parent at
    last), and the line links to that id, so a click jumps the reader
    straight to the page.  The fragment links keep the
    self-contained contract (no network, no companion files); the
    jumped-to page flashes an accent ring via ``:target``.  The PDF
    twin has no links — its ink is frozen at write time — so its
    Contents lines stay plain text.

    (100x-138) The RETURN leg: every page wrapper ends in a small
    ``↑ Contents`` link back to the header's block (the nav carries
    ``id='contents'`` and its own ``tabindex='-1'`` so the jump lands
    focus there and flashes the same accent ring) — the anchor jumps
    one way, this carries the reader back, so a report thirty pages
    deep is still one click from the index.  The link is minted at
    add time with the wrapper: a page wrapper exists exactly when at
    least one Contents line does, so the back-link's target can
    never be missing.
    """

    def __init__(
        self,
        title: str,
        subtitle: str | None = None,
        theme: str = "auto",
    ) -> None:
        if theme not in REPORT_THEMES:
            raise DataValidationError(
                f"HTMLReport theme must be one of {REPORT_THEMES}, got {theme!r}"
            )
        self.title = str(title)
        self.subtitle = subtitle
        self.theme = theme
        self._blocks: list[str] = []
        self._figure_count = 0
        self._table_count = 0
        # (100x-136) the page names for the header's Contents block,
        # collected in call order — exactly the sequence the document
        # writes them (metrics/text sections are title-page material
        # and are not listed, the PDF twin's discipline).
        self._page_names: list[str] = []
        # (100x-137) the anchor ids riding in lockstep with the names
        # — one ``pg-N`` per table/figure page, minted at add time so
        # the block wrapper and the Contents line can never disagree.
        self._page_ids: list[str] = []

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
        # (100x-136) the Contents line this page contributes — the
        # heading the page prints, or the positional fallback when the
        # caller passed none (the identity is honest either way).
        self._table_count += 1
        # (100x-137) the page rides inside an anchored wrapper so its
        # Contents line can link to it (a table block is a document
        # section; tabindex lets the fragment jump land focus there).
        # (100x-136) the page names for the header's Contents block,
        # minted here, at add time — the block's lines and the wrapper's
        # id can never disagree.
        # (100x-138) the wrapper ends in the RETURN link -- back to the
        # header's Contents block (minted once as _BACK_CONTENTS_LINK,
        # so the wrapper's existence and the link's target stay in
        # lockstep).
        page_id = f"pg-{len(self._page_names) + 1}"
        self._blocks.append(
            f"<section class='page' id='{page_id}' tabindex='-1'>"
            + "".join(parts)
            + _BACK_CONTENTS_LINK
            + "</section>"
        )
        self._page_ids.append(page_id)
        self._page_names.append(heading or f"table {self._table_count}")
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
        # (100x-137) the page rides inside an anchored <figure> — the
        # figcaption's legitimate parent, and the Contents line's
        # link target (tabindex lets the fragment jump land focus).
        # (100x-138) the wrapper ends in the RETURN link, after the
        # caption — back to the header's Contents block (the SAME
        # _BACK_CONTENTS_LINK the table wrapper ends in — one string,
        # two wrappers, zero drift).
        page_id = f"pg-{len(self._page_names) + 1}"
        block = f"<figure class='page' id='{page_id}' tabindex='-1'>"
        block += f"<img src='{data_uri}' alt='{alt}'/>"
        if caption:
            block += f"<figcaption>{_escape(caption)}</figcaption>"
        block += _BACK_CONTENTS_LINK
        block += "</figure>"
        self._blocks.append(block)
        # (100x-136) the Contents line this page contributes — the
        # caption the page prints, or the SAME positional fallback the
        # alt already carries.
        self._page_ids.append(page_id)
        self._page_names.append(caption or f"figure {self._figure_count}")
        return self

    def add_image(self, path: str | Path, caption: str | None = None,
                  max_bytes: int = 8 << 20) -> HTMLReport:
        """Embed an existing PNG file inline (base64; not re-rendered).

        The §17D run-report path: run records reference exported
        figures by FILE, so the report embeds the file's own bytes —
        a re-render would need the workflow's live state the record
        deliberately does not carry.  Only PNG is accepted (the one
        format the workflow exports are guaranteed to be); a missing
        or oversized file raises with the reason rather than
        silently degrading to a text line.
        """
        p = Path(path)
        try:
            raw = p.read_bytes()
        except OSError as exc:
            raise DataValidationError(
                f"figure file unreadable: {p} ({exc.strerror or exc})"
            ) from exc
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise DataValidationError(
                f"not a PNG file: {p} (embedding supports PNG exports only)"
            )
        if len(raw) > max_bytes:
            raise DataValidationError(
                f"figure too large to embed: {p} "
                f"({len(raw) >> 20} MiB > {max_bytes >> 20} MiB cap)"
            )
        alt = _escape(caption or p.name)
        # the SAME anchored <figure> page add_figure builds — one
        # wrapper shape, two sources (live object / file bytes)
        self._figure_count += 1
        page_id = f"pg-{len(self._page_names) + 1}"
        data_uri = "data:image/png;base64," + base64.b64encode(raw).decode()
        block = f"<figure class='page' id='{page_id}' tabindex='-1'>"
        block += f"<img src='{data_uri}' alt='{alt}'/>"
        if caption:
            block += f"<figcaption>{_escape(caption)}</figcaption>"
        block += _BACK_CONTENTS_LINK
        block += "</figure>"
        self._blocks.append(block)
        self._page_ids.append(page_id)
        self._page_names.append(caption or p.name)
        return self

    # -- rendering --------------------------------------------------------
    def render(self) -> str:
        """Return the full HTML document as a string."""
        subtitle = (
            f"<p class='subtitle'>{_escape(self.subtitle)}</p>" if self.subtitle else ""
        )
        # (100x-136) the Contents block — the PDF title page's mini
        # table-of-contents mirrored in HTML: every table/figure page
        # name in the walking order, each line the SAME string the
        # page prints (no drift is possible — the names ARE the
        # strings the blocks rendered).  Skipped when the report has
        # no table/figure pages at all (the noise rule).
        # (100x-137) every line is a LINK to its page's anchor — the
        # ids were minted at add time in lockstep with the names, so
        # href and target can never disagree.  (100x-140) every line
        # carries the POINTER confession too — the same title tooltip
        # discipline the return link gained in 100x-139, applied
        # symmetrically to the forward leg (the aria world always
        # knew the destination; a hovering pointer gets told now) —
        # minted at this ONE place, so every line ships it in
        # lockstep with the href it explains.
        contents = ""
        if self._page_names:
            lines = "".join(
                f"<li><a href='#{page_id}' title='Jump to this page in the report'>"
                f"{_escape(name)}</a></li>"
                for page_id, name in zip(self._page_ids, self._page_names)
            )
            # (100x-138) the nav is the RETURN target now — its own id
            # and tabindex let the pages' "↑ Contents" links land the
            # jump (with focus) and flash the same accent ring.
            contents = (
                "<nav class='contents' id='contents' tabindex='-1' "
                "aria-label='Contents'>\n"
                "<h2>Contents</h2>\n"
                f"<ul class='contents-list'>\n{lines}\n</ul>\n"
                "</nav>\n"
            )
        body = "\n".join(self._blocks)
        # the toggle's server-side label matches the baked theme (the
        # script relabels on load and whenever the reader cycles); its
        # title attribute explains the cycle order and the persistence.
        toggle = (
            f"<button id='theme-toggle' class='theme-toggle' type='button' "
            f"aria-label='Toggle colour theme' "
            f"title=\"Cycle theme: auto \u2192 light \u2192 dark (remembered by "
            f"this browser)\">{_THEME_LABELS[self.theme]}</button>\n"
        )
        return (
            "<!DOCTYPE html>\n"
            f"<html lang='en' data-theme='{self.theme}'>\n<head>\n"
            "<meta charset='utf-8'>\n"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
            f"<title>{_escape(self.title)}</title>\n<style>{_REPORT_CSS}</style>\n"
            "</head>\n<body>\n"
            f"{toggle}<h1>{_escape(self.title)}</h1>\n{subtitle}\n{contents}{body}\n"
            "<footer>PharmaML report — self-contained HTML (figures embedded, "
            "no external assets)</footer>\n"
            f"<script>{_THEME_SCRIPT}</script>\n</body>\n</html>\n"
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
