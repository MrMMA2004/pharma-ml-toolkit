"""Biology Workbench: real screens, real biomarkers (audit row 4).

The Studio's increment-4 page (the playground Cell Lines tab). Four of
the tab's engines, each wired to the SAME public backend APIs the
cards call -- nothing on this page is fabricated:

- **Screen audit** -- ``pharmaml.biology.load_response_table`` →
  ``audit_response_table`` (missing metric values, duplicate pairs) →
  ``replicate_variability`` (the duplicate pair's spread) →
  ``collapse_replicates`` (the pivot precondition) →
  ``summarize_compounds`` (the per-compound panel summary) and
  ``response_matrix`` on the collapsed screen, held for the other
  tabs. The bundled GDSC-style screen (185 rows, 23 lines, 8
  compounds, metric ``LN_IC50``) loads on the first click.
- **Sensitivity matrix** -- the collapsed screen pivoted to the
  lines × compounds matrix, plus ``call_sensitive_resistant`` on the
  chosen compound's column (quartile calls: lower ``LN_IC50`` means
  more sensitive, exactly the card's convention).
- **Variable genes** -- ``most_variable_genes`` (the MAD ranking)
  with the per-gene MAD recomputed honestly from the frame, exactly
  the playground card's arithmetic.
- **Biomarker discovery** -- ``normalize_expression`` →
  ``impute_expression`` → ``sensitivity_associations`` (Spearman or
  Pearson against the chosen compound's response column, BH-corrected)
  → ``top_biomarkers``, with ``direction_of_association`` naming the
  top hit's story.

All four run in a :class:`~ui.workers.job_worker.JobWorker`. They are
single-shot engine calls without checkpoint loops, so the progress
channel reports honestly but the cooperative Stop only lands for
checkpoint-wired jobs -- the same disclosure the PK/PD page carries.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView, PandasTableModel
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[2]

#: The bundled GDSC-style response screen (185 rows, 23 lines, 8 compounds).
_SCREEN_PATH = _TK / "examples" / "data" / "cellline_screen.csv"

#: The bundled expression matrix (23 lines x 300 genes, cell_line index).
_EXPRESSION_PATH = _TK / "examples" / "data" / "cellline_expression.csv"

#: Correlation methods the associations engine exposes.
_METHODS = ("spearman", "pearson")


def _parse_float(
    text: str, name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    """Parse one float with an honest, named error."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not a number ({text!r})") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value:g})")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum} (got {value:g})")
    return value


def _parse_int(text: str, name: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not an integer ({text!r})")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} (got {value})")
    return value


@dataclass
class ScreenAuditOutcome:
    """What the screen-audit job hands back (all real numbers)."""

    audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    replicates: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows: int = 0
    n_pairs: int = 0
    n_cell_lines: int = 0
    n_compounds: int = 0
    metric: str = ""
    seconds: float = 0.0


@dataclass
class MatrixOutcome:
    """What the sensitivity-matrix job hands back."""

    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    line_names: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    compound: str = ""
    metric: str = ""
    seconds: float = 0.0


@dataclass
class MadGenesOutcome:
    """What the MAD-ranking job hands back (gene, MAD -- ranked)."""

    ranked: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_genes_total: int = 0
    seconds: float = 0.0


@dataclass
class BiomarkerOutcome:
    """What the biomarker job hands back."""

    top: pd.DataFrame = field(default_factory=pd.DataFrame)
    norm: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_genes: int = 0
    n_significant: int = 0
    strongest_rho: float = 0.0
    strongest_gene: str = ""
    direction: str = ""
    compound: str = ""
    method: str = ""
    seconds: float = 0.0


class BiologyWorkbenchPage(QWidget):
    """Audit a response screen; pivot it; rank genes; find biomarkers."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)

        # cross-tab state (written only from the GUI thread, in the
        # finished handlers; the jobs capture their inputs at submit)
        self._matrix: pd.DataFrame | None = None  # the collapsed screen, pivoted
        self._screen_metric: str = ""
        self._expr_norm: pd.DataFrame | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_audit_tab(), "Screen audit")
        self._tabs.addTab(self._build_matrix_tab(), "Sensitivity matrix")
        self._tabs.addTab(self._build_mad_tab(), "Variable genes (MAD)")
        self._tabs.addTab(self._build_biomarker_tab(), "Biomarker discovery")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Biology panel")
        splitter.addWidget(self._plot)
        splitter.setSizes([600, 500])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No biology run yet — load the bundled screen first.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

    # -- tab construction -----------------------------------------------------
    def _build_audit_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._screen_combo = QComboBox()
        self._screen_combo.addItem("cellline_screen.csv (bundled)")
        form.addRow("response screen:", self._screen_combo)
        self._audit_btn = QPushButton("Load & audit screen")
        self._audit_btn.setMinimumHeight(34)
        self._audit_btn.clicked.connect(self._start_audit)
        form.addRow("", self._audit_btn)
        hint = QLabel(
            "Loads through load_response_table, audits it (missing metric "
            "values, duplicate pairs), collapses the replicates and pivots "
            "the matrix the other tabs use."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_matrix_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._matrix_compound = QComboBox()
        form.addRow("compound:", self._matrix_compound)
        self._sens_q = QLineEdit("0.25")
        form.addRow("sensitive quantile:", self._sens_q)
        self._res_q = QLineEdit("0.75")
        form.addRow("resistant quantile:", self._res_q)
        self._matrix_btn = QPushButton("Pivot matrix & call extremes")
        self._matrix_btn.setMinimumHeight(34)
        self._matrix_btn.clicked.connect(self._start_matrix)
        form.addRow("", self._matrix_btn)
        hint = QLabel(
            "Lines x compounds matrix of the collapsed screen; the chosen "
            "compound's column gets the sensitive/intermediate/resistant "
            "quartile calls (lower metric = more sensitive)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_mad_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._mad_n = QLineEdit("10")
        form.addRow("top N genes:", self._mad_n)
        self._mad_btn = QPushButton("Rank by MAD")
        self._mad_btn.setMinimumHeight(34)
        self._mad_btn.clicked.connect(self._start_mad)
        form.addRow("", self._mad_btn)
        hint = QLabel(
            "most_variable_genes on the bundled expression matrix (23 lines "
            "x 300 genes): the N genes with the highest median absolute "
            "deviation, MAD recomputed from the frame."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_biomarker_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._marker_compound = QComboBox()
        form.addRow("compound:", self._marker_compound)
        self._method_combo = QComboBox()
        self._method_combo.addItems(_METHODS)
        form.addRow("correlation:", self._method_combo)
        self._topn_entry = QLineEdit("8")
        form.addRow("top N biomarkers:", self._topn_entry)
        self._marker_btn = QPushButton("Correlate & rank")
        self._marker_btn.setMinimumHeight(34)
        self._marker_btn.clicked.connect(self._start_markers)
        form.addRow("", self._marker_btn)
        hint = QLabel(
            "normalize_expression -> impute_expression -> "
            "sensitivity_associations (BH-corrected) -> top_biomarkers "
            "against the chosen compound's response column."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- job plumbing -----------------------------------------------------------
    def _on_started(self, name: str) -> None:
        for btn in (self._audit_btn, self._matrix_btn, self._mad_btn, self._marker_btn):
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in (self._audit_btn, self._matrix_btn, self._mad_btn, self._marker_btn):
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return True
        return False

    # -- screen audit --------------------------------------------------------------
    def _start_audit(self) -> None:
        if self._busy_check():
            return
        screen_path = _SCREEN_PATH

        def job(progress) -> ScreenAuditOutcome:
            from pharmaml.biology import (
                audit_response_table,
                collapse_replicates,
                load_response_table,
                replicate_variability,
                response_matrix,
                summarize_compounds,
            )

            started = _time.monotonic()
            progress(0.15, "loading the response table")
            screen = load_response_table(screen_path)
            n_rows = len(screen.frame)
            metric = screen.metrics[0]
            progress(0.4, "auditing replicates and metric values")
            audit = audit_response_table(screen)
            replicates = replicate_variability(screen)
            progress(0.6, "collapsing replicates")
            collapsed = collapse_replicates(screen, add_cv=True)
            summary = summarize_compounds(collapsed, metric)
            progress(0.8, "pivoting the collapsed screen")
            matrix = response_matrix(collapsed, metric)
            progress(1.0, "done")
            return ScreenAuditOutcome(
                audit=audit,
                replicates=replicates,
                summary=summary,
                matrix=matrix,
                n_rows=n_rows,
                n_pairs=int(collapsed.n_pairs),
                n_cell_lines=len(collapsed.cell_lines),
                n_compounds=len(collapsed.compounds),
                metric=metric,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "screen audit",
            job,
            on_finished=self._audit_done,
            on_failed=self._failed,
        )

    def _audit_done(self, outcome: object) -> None:
        assert isinstance(outcome, ScreenAuditOutcome)
        # cross-tab state, written on the GUI thread only
        self._matrix = outcome.matrix
        self._screen_metric = outcome.metric
        compounds = list(outcome.matrix.columns)
        for combo in (self._matrix_compound, self._marker_compound):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(compounds)
            combo.blockSignals(False)

        self._table.bind_frame(outcome.summary.reset_index())
        medians = outcome.summary["median"].to_numpy(dtype=float)
        self._plot.clear()
        self._plot.bars(medians, labels=list(outcome.summary.index))
        self._plot.set_labels(
            x="compound",
            y=f"median {outcome.metric} (lower = more sensitive)",
            title=(
                f"Per-compound panel summary · {outcome.n_cell_lines} lines · "
                f"{outcome.n_compounds} compounds"
            ),
        )
        issues = " · ".join(
            f"{row['issue']}: {row['detail']} ({int(row['count'])})"
            for _, row in outcome.audit.iterrows()
        )
        parts = [
            (
                f"Screen — {outcome.n_rows} rows -> {outcome.n_pairs} unique pairs · "
                f"{outcome.n_cell_lines} lines x {outcome.n_compounds} compounds · "
                f"metric {outcome.metric}"
            )
        ]
        if issues:
            parts.append(f"audit: {issues}")
        if len(outcome.replicates):
            parts.append(f"{len(outcome.replicates)} replicate pair(s) collapsed, CV recorded")
        parts.append(f"matrix {outcome.matrix.shape[0]}x{outcome.matrix.shape[1]} pivoted")
        parts.append(f"{outcome.seconds:.1f}s")
        self._report.setText(" · ".join(parts))

    # -- sensitivity matrix ------------------------------------------------------------
    def _start_matrix(self) -> None:
        if self._matrix is None or self._screen_metric == "":
            self._report.setText(
                "Load & audit a screen first — the matrix pivots its collapsed table."
            )
            return
        compound = self._matrix_compound.currentText()
        if not compound:
            self._report.setText("Pick a compound (the screen's compounds load with it).")
            return
        try:
            sens_q = _parse_float(self._sens_q.text(), "sensitive quantile", minimum=0.0, maximum=1.0)
            res_q = _parse_float(self._res_q.text(), "resistant quantile", minimum=0.0, maximum=1.0)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if sens_q >= res_q:
            self._report.setText(
                f"Input — the sensitive quantile must be below the resistant one "
                f"({sens_q:g} >= {res_q:g})"
            )
            return
        if self._busy_check():
            return
        metric = self._screen_metric
        matrix = self._matrix  # captured at submit; the GUI thread owns it

        def job(progress) -> MatrixOutcome:
            from pharmaml.biology import call_sensitive_resistant

            started = _time.monotonic()
            progress(0.3, "reading the collapsed matrix")
            values = matrix[compound]
            progress(0.6, "quartile calls")
            calls = call_sensitive_resistant(
                values, sensitive_quantile=sens_q, resistant_quantile=res_q
            )
            counts = {str(k): int(v) for k, v in calls.value_counts().items()}
            progress(1.0, "done")
            return MatrixOutcome(
                matrix=matrix,
                values=values.to_numpy(dtype=float),
                line_names=[str(i) for i in values.index],
                counts=counts,
                compound=compound,
                metric=metric,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"sensitivity matrix ({compound})",
            job,
            on_finished=self._matrix_done,
            on_failed=self._failed,
        )

    def _matrix_done(self, outcome: object) -> None:
        assert isinstance(outcome, MatrixOutcome)
        self._table.bind_frame(outcome.matrix.reset_index().rename(
            columns={outcome.matrix.index.name or "index": "cell_line"}
        ))
        self._plot.clear()
        self._plot.bars(outcome.values, labels=outcome.line_names)
        self._plot.set_labels(
            x="cell line",
            y=f"{outcome.metric} (lower = more sensitive)",
            title=f"{outcome.compound} across {len(outcome.line_names)} lines",
        )
        calls = " / ".join(
            f"{outcome.counts.get(kind, 0)} {kind}"
            for kind in ("sensitive", "intermediate", "resistant")
        )
        self._report.setText(
            f"Matrix — {outcome.matrix.shape[0]} lines x {outcome.matrix.shape[1]} "
            f"compounds · {outcome.compound}: {calls} · "
            f"{outcome.seconds:.1f}s"
        )

    # -- variable genes (MAD) ------------------------------------------------------------
    def _start_mad(self) -> None:
        try:
            top_n = _parse_int(self._mad_n.text(), "top N genes", minimum=1, maximum=100000)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return
        expression_path = _EXPRESSION_PATH

        def job(progress) -> MadGenesOutcome:
            from pharmaml.biology import most_variable_genes

            started = _time.monotonic()
            progress(0.25, "reading the expression matrix")
            frame = pd.read_csv(expression_path, index_col=0)
            progress(0.6, "MAD ranking")
            genes = most_variable_genes(frame, n=top_n)
            # MAD per kept gene, ranked -- recomputed honestly from the
            # frame (the library returns names only), the card's arithmetic.
            mads = (frame[genes] - frame[genes].median()).abs().median()
            ranked = pd.DataFrame({"gene": genes}).set_index("gene")
            ranked["MAD"] = mads
            progress(1.0, "done")
            return MadGenesOutcome(
                ranked=ranked,
                n_genes_total=int(frame.shape[1]),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "MAD gene ranking",
            job,
            on_finished=self._mad_done,
            on_failed=self._failed,
        )

    def _mad_done(self, outcome: object) -> None:
        assert isinstance(outcome, MadGenesOutcome)
        self._table.bind_frame(outcome.ranked.reset_index())
        self._plot.clear()
        self._plot.bars(
            outcome.ranked["MAD"].to_numpy(dtype=float),
            labels=[str(g) for g in outcome.ranked.index],
        )
        self._plot.set_labels(
            x="gene",
            y="median absolute deviation",
            title=f"Top {len(outcome.ranked)} of {outcome.n_genes_total:,} genes by MAD",
        )
        self._report.setText(
            f"Variable genes — top {len(outcome.ranked)} of "
            f"{outcome.n_genes_total:,} genes by MAD · most variable: "
            f"{outcome.ranked.index[0]} (MAD {float(outcome.ranked['MAD'].iloc[0]):.4g}) · "
            f"{outcome.seconds:.1f}s"
        )

    # -- biomarker discovery ------------------------------------------------------------
    def _start_markers(self) -> None:
        if self._matrix is None or self._screen_metric == "":
            self._report.setText(
                "Load & audit a screen first — biomarker discovery correlates "
                "its response matrix."
            )
            return
        compound = self._marker_compound.currentText()
        if not compound:
            self._report.setText("Pick a compound (the screen's compounds load with it).")
            return
        try:
            top_n = _parse_int(self._topn_entry.text(), "top N biomarkers", minimum=1, maximum=500)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return
        method = self._method_combo.currentText()
        matrix = self._matrix
        expression_path = _EXPRESSION_PATH
        norm_cache = self._expr_norm

        def job(progress) -> BiomarkerOutcome:
            from pharmaml.biology import (
                direction_of_association,
                impute_expression,
                normalize_expression,
                sensitivity_associations,
                top_biomarkers,
            )

            started = _time.monotonic()
            progress(0.2, "normalizing the expression matrix")
            norm = norm_cache
            if norm is None:
                frame = pd.read_csv(expression_path, index_col=0)
                norm = impute_expression(normalize_expression(frame))
            progress(0.5, f"{method} associations (BH-corrected)")
            associations = sensitivity_associations(norm, matrix[compound], method=method)
            progress(0.8, "ranking")
            top = top_biomarkers(associations, n=top_n)
            n_significant = int(associations["significant"].sum())
            strongest = float(associations["rho"].abs().max())
            strongest_row = associations.loc[associations["rho"].abs().idxmax()]
            direction = direction_of_association(
                float(strongest_row["rho"]), lower_is_sensitive=True
            )
            progress(1.0, "done")
            return BiomarkerOutcome(
                top=top,
                norm=norm,
                n_genes=len(associations),
                n_significant=n_significant,
                strongest_rho=strongest,
                strongest_gene=str(strongest_row["gene"]),
                direction=direction,
                compound=compound,
                method=method,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"biomarkers for {compound}",
            job,
            on_finished=self._markers_done,
            on_failed=self._failed,
        )

    def _markers_done(self, outcome: object) -> None:
        assert isinstance(outcome, BiomarkerOutcome)
        self._expr_norm = outcome.norm  # the GUI thread owns the cache
        shown = outcome.top.reset_index(drop=True)
        self._table.bind_frame(shown)
        self._plot.clear()
        self._plot.bars(
            outcome.top["rho"].abs().to_numpy(dtype=float),
            labels=[str(g) for g in outcome.top["gene"]],
        )
        self._plot.set_labels(
            x="gene",
            y=f"|rho| ({outcome.method})",
            title=f"Top {len(outcome.top)} biomarkers · {outcome.compound}",
        )
        self._report.setText(
            f"Biomarkers — {outcome.n_genes:,} genes correlated · "
            f"{outcome.n_significant} significant at FDR < 0.05 · strongest "
            f"|rho| {outcome.strongest_rho:.3f} ({outcome.strongest_gene}, "
            f"{outcome.direction}) — {outcome.compound} · {outcome.seconds:.1f}s"
        )

    # -- failures / theming -----------------------------------------------------------
    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "job failed"
        self._report.setText(f"Job failed — {first}")

    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    # -- test/inspection surface -----------------------------------------------------
    def model_frame(self) -> pd.DataFrame | None:
        """The frame currently bound to the result table (or None)."""
        model = self._table.model()
        return model.frame if isinstance(model, PandasTableModel) else None

    def report_text(self) -> str:
        return self._report.text()
