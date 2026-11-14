"""Best-of ML Workbench: the catalogue as data, the coverage audit, the five
native capability surfaces (16.8/16.12/16.14).

The Studio's increment-10 page (the playground Best-of ML tab, audit row
11 — ``lukasmasuch/best-of-ml-python`` absorbed).  Eight of the tab's
clusters, each wired to the SAME public backend APIs the cards call —
nothing on this page is fabricated:

- **Catalogue** -- ``pharmaml.ml.catalog`` (card 1): the 924-project
  collection as queryable data — substring search over names and
  descriptions, one category's projects in best-of score order, the top
  10 overall — with the top-10 star counts as the bar chart.
- **Coverage audit** -- ``coverage_table`` (card 2): the honest map of
  all 35 categories — 12 native / 6 partial / 4 external / 13
  out-of-scope on the shipped snapshot — as the verdict table plus the
  verdict-count bars.
- **Differential privacy** -- ``pharmaml.ml.privacy.dp_statistics`` +
  ``dp_sgd_noise_scale`` (card 3): a column's count/sum/mean/histogram
  under one epsilon budget (sequential composition, the histogram
  charged once via parallel composition) plus the RDP accountant's
  noise price for a DP-SGD plan; true vs DP bin counts drawn as
  grouped value-space bars.
- **Recommender systems** -- ``pharmaml.ml.recommend`` (card 4):
  user/item-based collaborative filtering over the seeded 24x20
  latent-factor matrix, judged by leave-one-out hit-rate against the
  popularity baseline (the popularity arm can and does win — the report
  says so rather than burying it), plus user 0's ranked recommendations.
- **Vector search (LSH)** -- ``pharmaml.ml.annsearch.ann_benchmark``
  (card 5): random-hyperplane LSH with multiprobe benchmarked against
  brute force — the recall@5 / candidate-compression trade-off curve as
  the plane count grows.
- **Adversarial robustness** -- ``pharmaml.ml.adversarial.adversarial_audit``
  (card 6): a zeroth-order sign-pattern family attacks the trained
  logistic model inside an L-infinity ball scaled by each feature's
  spread — the broken-rate curve and the median adversarial margin.
- **Time-series analysis** -- ``pharmaml.ml.timeseries`` (card 7): ACF
  (drawn as the lag bars) / PACF, classical additive decomposition
  with strengths, CUSUM changepoints and the window feature table.
- **Association rules** -- ``pharmaml.ml.apriori.apriori_pipeline``
  (card 8): pasted transactions through frequent itemsets and rules
  with support, confidence and lift — the lift per rule as the bars.

The DP / LSH / adversarial / time-series cards read the Data Explorer's
loaded frame when one exists (with the same guards the playground
applies: enough rows, finite values, two full periods) and otherwise
fall back to the tab's own seeded synthetic demos — mirrored exactly.

All eight run in a :class:`~ui.workers.job_worker.JobWorker`.  They are
single-shot engine calls without checkpoint loops, so the progress
channel reports honestly but the cooperative Stop only lands for
checkpoint-wired jobs -- the same disclosure the Foundations page
carries.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
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
from ..widgets.responsive import configure_splitter, wrap_in_scroll
from ..workers.job_worker import JobRunner

#: the association card's default pasted transactions (mirrored verbatim).
_ASSOC_DEFAULT = (
    "aspirin, metformin\n"
    "aspirin, statin\n"
    "aspirin, metformin, statin\n"
    "metformin\n"
    "aspirin, metformin"
)


def _parse_float(text: str, name: str, *, minimum: float | None = None) -> float:
    """Parse one float with an honest, named error."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not a number ({text!r})") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value:g})") from None
    return value


def _parse_int(text: str, name: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not an integer ({text!r})") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} (got {value})") from None
    return value


@dataclass
class CatalogOutcome:
    """What a catalogue job hands back (hits + the summary)."""

    hits: list[Any] = field(default_factory=list)
    summary: Any = None
    mode: str = ""  # "search 'x'" / "top 10" / "category 'x'"
    seconds: float = 0.0


@dataclass
class CoverageOutcome:
    """What the coverage job hands back (the 35 verdicts)."""

    table: list[Any] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class DPOutcome:
    """What the DP job hands back (the report + the accountant plan)."""

    report: Any = None
    plan: Any = None
    source: str = ""
    seconds: float = 0.0


@dataclass
class RecommenderOutcome:
    """What the recommender job hands back (evaluation + user-0 recs)."""

    matrix_shape: tuple[int, int] = (0, 0)
    observed: int = 0
    evaluation: Any = None
    recs: Any = None
    kind: str = "item"
    seconds: float = 0.0


@dataclass
class LSHOutcome:
    """What the LSH job hands back (the sweep reports)."""

    reports: list[Any] = field(default_factory=list)
    source: str = ""
    shape: tuple[int, int] = (0, 0)
    probes: int = 2
    seconds: float = 0.0


@dataclass
class AdversarialOutcome:
    """What the adversarial job hands back (the audit)."""

    audit: Any = None
    task_name: str = ""
    label: str = ""
    seconds: float = 0.0


@dataclass
class TimeseriesOutcome:
    """What the time-series job hands back (all five analyses)."""

    acf: Any = None
    pacf: Any = None
    decomposition: Any = None
    changes: Any = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = ""
    series_size: int = 0
    period: int = 24
    seconds: float = 0.0


@dataclass
class AprioriOutcome:
    """What the apriori job hands back (the pipeline result)."""

    result: Any = None
    seconds: float = 0.0


class BestOfMLWorkbenchPage(QWidget):
    """Browse the catalogue; audit coverage; run the five native surfaces + apriori."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
        data_explorer: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._data_explorer = data_explorer
        # review m-19: refresh the catalog pickers on every explorer load
        if self._data_explorer is not None and hasattr(self._data_explorer, "frameLoaded"):
            self._data_explorer.frameLoaded.connect(self._on_explorer_frame)


        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(wrap_in_scroll(self._build_catalog_tab()), "Catalogue")
        self._tabs.addTab(wrap_in_scroll(self._build_coverage_tab()), "Coverage audit")
        self._tabs.addTab(wrap_in_scroll(self._build_dp_tab()), "Differential privacy")
        self._tabs.addTab(wrap_in_scroll(self._build_recommender_tab()), "Recommender")
        self._tabs.addTab(wrap_in_scroll(self._build_lsh_tab()), "Vector search (LSH)")
        self._tabs.addTab(wrap_in_scroll(self._build_adversarial_tab()), "Adversarial")
        self._tabs.addTab(wrap_in_scroll(self._build_timeseries_tab()), "Time series")
        self._tabs.addTab(wrap_in_scroll(self._build_assoc_tab()), "Association rules")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Best-of ML")
        self._plot.set_empty_hint("Run a Best-of-ML comparison to see the family ranking")

        splitter.addWidget(self._plot)
        splitter.setSizes([560, 540])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No Best-of ML run yet.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

    # -- tab construction ----------------------------------------------------
    def _category_titles(self) -> list[str]:
        try:
            from pharmaml.ml.catalog import catalog_categories

            return ["all categories"] + [c.title for c in catalog_categories()]
        except Exception:  # noqa: BLE001 -- degraded build, not a crash
            return ["all categories"]

    def _build_catalog_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._cat_search = QLineEdit("differential privacy")
        form.addRow("Search:", self._cat_search)
        self._cat_category = QComboBox()
        self._cat_category.addItems(self._category_titles())
        form.addRow("Category:", self._cat_category)
        self._cat_search_btn = QPushButton("Search")
        self._cat_search_btn.setMinimumHeight(34)
        self._cat_search_btn.clicked.connect(lambda: self._start_catalog("search"))
        form.addRow("", self._cat_search_btn)
        self._cat_top_btn = QPushButton("Top 10")
        self._cat_top_btn.setMinimumHeight(34)
        self._cat_top_btn.clicked.connect(lambda: self._start_catalog("top"))
        form.addRow("", self._cat_top_btn)
        hint = QLabel(
            "The best-of-ml-python collection as data: 924 projects, 35 sections, "
            "snapshot 2025-10-30 (CC-BY-SA-4.0, embedded with attribution). "
            "Search keeps the catalogue's score order; the top-10 star counts "
            "draw as the bars."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_coverage_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._coverage_btn = QPushButton("Audit the toolkit")
        self._coverage_btn.setMinimumHeight(34)
        self._coverage_btn.clicked.connect(self._start_coverage)
        form.addRow("", self._coverage_btn)
        hint = QLabel(
            "The honest map: for each of the 34 catalogue categories (plus "
            "Others) — native, partial, external or out-of-scope, with the "
            "verdict note. The five native additions this round: privacy, "
            "recommend, annsearch, adversarial, timeseries."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_dp_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._dp_eps = QLineEdit("1.0")
        form.addRow("ε (budget):", self._dp_eps)
        self._dp_bins = QLineEdit("8")
        self._dp_bins.setToolTip("Histogram bins for the decision-profile comparison")
        form.addRow("Bins:", self._dp_bins)
        self._dp_column = QComboBox()
        self._dp_column.addItem("synthetic N(50, 5²) column (seed 42)")
        form.addRow("column:", self._dp_column)
        self._dp_btn = QPushButton("Privatise the column")
        self._dp_btn.setMinimumHeight(34)
        self._dp_btn.clicked.connect(self._start_dp)
        form.addRow("", self._dp_btn)
        hint = QLabel(
            "Count/sum/mean/histogram under one epsilon budget — sequential "
            "composition splits it across the queries, the histogram's bins "
            "count disjoint rows so they share ONE slice (parallel "
            "composition); plus the RDP accountant's noise price for a "
            "DP-SGD plan. The column picker offers the Data Explorer's "
            "numeric columns when a table is loaded."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_recommender_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._cf_kind = QComboBox()
        self._cf_kind.addItems(("item", "user"))
        form.addRow("method:", self._cf_kind)
        self._cf_k = QLineEdit("8")
        form.addRow("neighbours k:", self._cf_k)
        self._cf_n = QLineEdit("5")
        form.addRow("top N:", self._cf_n)
        self._cf_btn = QPushButton("Run the evaluation")
        self._cf_btn.setMinimumHeight(34)
        self._cf_btn.clicked.connect(self._start_recommender)
        form.addRow("", self._cf_btn)
        hint = QLabel(
            "Collaborative filtering over the seeded 24x20 latent-factor "
            "matrix (seed 42), judged by leave-one-out hit-rate against the "
            "popularity baseline — the honest evaluation the category's "
            "benchmarks run. The popularity arm CAN win; the report says so."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_lsh_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._lsh_probes = QComboBox()
        self._lsh_probes.addItems(("1", "2"))
        self._lsh_probes.setCurrentText("2")
        form.addRow("probes:", self._lsh_probes)
        self._lsh_btn = QPushButton("Run the sweep")
        self._lsh_btn.setMinimumHeight(34)
        self._lsh_btn.clicked.connect(self._start_lsh)
        form.addRow("", self._lsh_btn)
        hint = QLabel(
            "Random-hyperplane LSH with multiprobe vs brute force — "
            "recall@5 against candidate-set compression as the plane count "
            "grows (2/4/6/8/10 at k=5). Vectors come from the Data "
            "Explorer's numeric block (or the seeded three-blob synthetic)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_adversarial_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._adv_btn = QPushButton("Train & attack")
        self._adv_btn.setMinimumHeight(34)
        self._adv_btn.clicked.connect(self._start_adversarial)
        form.addRow("", self._adv_btn)
        hint = QLabel(
            "A zeroth-order sign-pattern family attacks the trained model's "
            "calls inside an L∞ ball scaled by each feature's spread "
            "(5-50% of measurement noise) — the broken-rate curve and the "
            "median adversarial margin. Trains logistic regression (or "
            "ridge) on the Data Explorer's table (or the seeded two-blob "
            "demo)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_timeseries_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._ts_column = QComboBox()
        self._ts_column.addItem("synthetic period-24 signal (seed 42)")
        form.addRow("column:", self._ts_column)
        self._ts_period = QLineEdit("24")
        form.addRow("period:", self._ts_period)
        self._ts_btn = QPushButton("Analyse the series")
        self._ts_btn.setMinimumHeight(34)
        self._ts_btn.clicked.connect(self._start_timeseries)
        form.addRow("", self._ts_btn)
        hint = QLabel(
            "ACF (the lag bars) / PACF — which lag carries the memory; "
            "classical additive decomposition with strengths; "
            "Kolmogorov-judged CUSUM changepoints; the tsfresh-lite window "
            "feature table. The picker offers the Data Explorer's numeric "
            "columns (needs two full periods)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_assoc_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._assoc_text = QPlainTextEdit()
        self._assoc_text.setPlainText(_ASSOC_DEFAULT)
        self._assoc_text.setPlaceholderText("one transaction per line (comma-separated items)")
        form.addRow("transactions:", self._assoc_text)
        self._assoc_support = QLineEdit("0.30")
        form.addRow("min support:", self._assoc_support)
        self._assoc_confidence = QLineEdit("0.40")
        form.addRow("min confidence:", self._assoc_confidence)
        self._assoc_btn = QPushButton("Mine rules")
        self._assoc_btn.setMinimumHeight(34)
        self._assoc_btn.clicked.connect(self._start_assoc)
        form.addRow("", self._assoc_btn)
        hint = QLabel(
            "The apriori pipeline (the Orange3 basket analysis): frequent "
            "itemsets above the support threshold, then rules above the "
            "confidence threshold ranked by lift."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- job plumbing -----------------------------------------------------------
    def _buttons(self) -> list[QPushButton]:
        return [
            self._cat_search_btn,
            self._cat_top_btn,
            self._coverage_btn,
            self._dp_btn,
            self._cf_btn,
            self._lsh_btn,
            self._adv_btn,
            self._ts_btn,
            self._assoc_btn,
        ]

    def _on_started(self, name: str) -> None:
        for btn in self._buttons():
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in self._buttons():
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return True
        return False

    def _failed(self, message: str) -> None:
        self._report.setText(message.splitlines()[0] if message else "The job failed.")

    def _explorer_frame(self) -> pd.DataFrame | None:
        if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
            frame = self._data_explorer.frame()
            if frame is not None and not frame.empty:
                return frame
        return None

    def _on_explorer_frame(self, frame: object) -> None:
        """frameLoaded slot (review m-19): keep pickers true to the live table."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return
        self._refresh_column_pickers(
            [str(c) for c in frame.select_dtypes(include=[np.number]).columns]
        )

    def _refresh_column_pickers(self, numeric: list[str]) -> None:
        """(Re)fill the DP / time-series pickers with the explorer's numeric columns."""
        if not numeric:
            return
        self._dp_column.clear()
        self._dp_column.addItem("synthetic N(50, 5²) column (seed 42)")
        self._dp_column.addItems(numeric)
        self._ts_column.clear()
        self._ts_column.addItem("synthetic period-24 signal (seed 42)")
        self._ts_column.addItems(numeric)

    # -- catalogue ------------------------------------------------------------
    def _start_catalog(self, mode: str) -> None:
        category = self._cat_category.currentText()
        if mode == "search":
            text = self._cat_search.text().strip()
            if not text:
                self._report.setText("Type a search term first.")
                return
        if self._busy_check():
            return
        if mode in ("search", "category"):
            frame = self._explorer_frame()
            if frame is not None:
                self._refresh_column_pickers(
                    [str(c) for c in frame.select_dtypes(include=[np.number]).columns]
                )

        def job(progress) -> CatalogOutcome:
            from pharmaml.ml.catalog import catalog_summary, catalog_top, search_catalog

            started = _time.monotonic()
            progress(0.4, "querying the embedded catalogue")
            if mode == "search":
                hits = search_catalog(
                    text, category=None if category == "all categories" else category, limit=200
                )
                label = f"search {text!r}"
            else:
                hits = catalog_top(
                    n=10, category=None if category == "all categories" else category
                )
                label = "top 10 by best-of score"
            progress(1.0, "done")
            return CatalogOutcome(
                hits=list(hits),
                summary=catalog_summary(),
                mode=label,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "catalogue query",
            job,
            on_finished=self._catalog_done,
            on_failed=self._failed,
        )

    def _catalog_done(self, outcome: object) -> None:
        assert isinstance(outcome, CatalogOutcome)
        summary = outcome.summary
        rows = [
            (
                project.name,
                project.category_title,
                int(project.stars),
                project.license or "—",
                project.description,
            )
            for project in outcome.hits
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["project", "category", "stars", "license", "description"]),
            note=outcome.mode,
        )
        self._plot.clear()
        if rows:
            top = rows[:10]
            self._plot.bars(
                np.asarray([float(row[2]) for row in top], dtype=float),
                labels=[row[0] for row in top],
                index=0,
            )
            self._plot.set_labels(x="project", y="stars", title="The strongest hits (star counts)")
        self._report.setText(
            f"{len(outcome.hits)} project(s) · {outcome.mode}"
            + (f" in {self._cat_category.currentText()}" if self._cat_category.currentText() != "all categories" else "")
            + f" · snapshot {summary.n_projects} projects / {summary.n_categories} sections / "
            f"{summary.total_stars:,} stars ({outcome.seconds:.1f}s)"
        )

    # -- coverage audit ------------------------------------------------------------
    def _start_coverage(self) -> None:
        if self._busy_check():
            return

        def job(progress) -> CoverageOutcome:
            from pharmaml.ml.catalog import coverage_table

            started = _time.monotonic()
            progress(0.5, "the 35 verdicts")
            table = list(coverage_table())
            counts: dict[str, int] = {}
            for entry in table:
                counts[entry.status] = counts.get(entry.status, 0) + 1
            progress(1.0, "done")
            return CoverageOutcome(table=table, counts=counts, seconds=_time.monotonic() - started)

        self._runner.submit(
            "coverage audit",
            job,
            on_finished=self._coverage_done,
            on_failed=self._failed,
        )

    def _coverage_done(self, outcome: object) -> None:
        assert isinstance(outcome, CoverageOutcome)
        counts = outcome.counts
        rows = [
            (entry.title, entry.status, entry.note)
            for entry in outcome.table
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["category", "status", "verdict"]),
            note="the honest map",
        )
        self._plot.clear()
        order = ("native", "partial", "external", "out-of-scope")
        self._plot.bars(
            np.asarray([float(counts.get(name, 0)) for name in order], dtype=float),
            labels=list(order),
            index=0,
        )
        self._plot.set_labels(x="verdict", y="categories", title="The coverage census")
        self._report.setText(
            f"{counts.get('native', 0)} native · {counts.get('partial', 0)} partial · "
            f"{counts.get('external', 0)} external · {counts.get('out-of-scope', 0)} "
            "out of scope — the five native additions this round: privacy, "
            "recommend, annsearch, adversarial, timeseries"
        )

    # -- differential privacy ------------------------------------------------------------
    def _dp_values(self) -> tuple[np.ndarray, str]:
        """A numeric column from the explorer, or the seeded synthetic fallback."""
        frame = self._explorer_frame()
        if frame is not None:
            column = self._dp_column.currentText()
            if column and column in [str(c) for c in frame.columns]:
                values = frame[column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size >= 8:
                    return values, f"column {column!r} of the loaded table"
        rng = np.random.default_rng(42)
        return rng.normal(50.0, 5.0, 400), "synthetic N(50, 5²) column (seed 42)"

    def _start_dp(self) -> None:
        try:
            epsilon = _parse_float(self._dp_eps.text(), "ε", minimum=0.01)
            n_bins = _parse_int(self._dp_bins.text(), "bins", minimum=0, maximum=20)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if epsilon > 10.0:
            self._report.setText("Input — ε must be <= 10")
            return
        values, source = self._dp_values()
        if self._busy_check():
            return

        def job(progress) -> DPOutcome:
            from pharmaml.ml.privacy import dp_sgd_noise_scale, dp_statistics

            started = _time.monotonic()
            progress(0.4, "the epsilon budget split")
            report = dp_statistics(values, epsilon=epsilon, n_bins=n_bins, seed=42)
            progress(0.8, "the RDP accountant")
            plan = dp_sgd_noise_scale(epsilon=float(self._dp_epsilon.value()) if getattr(self, "_dp_epsilon", None) else 4.0, delta=1e-5, steps=int(self._dp_steps.value()) if getattr(self, "_dp_steps", None) else 1500)
            progress(1.0, "done")
            return DPOutcome(report=report, plan=plan, source=source, seconds=_time.monotonic() - started)

        self._runner.submit(
            "DP statistics",
            job,
            on_finished=self._dp_done,
            on_failed=self._failed,
        )

    def _dp_done(self, outcome: object) -> None:
        assert isinstance(outcome, DPOutcome)
        report = outcome.report
        plan = outcome.plan
        rows: list[tuple[Any, ...]] = [
            ("count", float(report.true_count), float(report.dp_count), report.dp_count - report.true_count),
            ("sum", float(report.true_sum), float(report.dp_sum), report.dp_sum - report.true_sum),
            ("mean", float(report.true_mean), float(report.dp_mean), report.dp_mean - report.true_mean),
        ]
        if report.histogram:
            worst = max(report.histogram, key=lambda b: abs(b.dp_count - b.true_count))
            rows.append(
                (
                    f"bin [{worst.lower:.1f}, {worst.upper:.1f})",
                    float(worst.true_count),
                    float(worst.dp_count),
                    float(worst.dp_count - worst.true_count),
                )
            )
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["statistic", "true", "dp", "error"]),
            note="one epsilon budget",
        )
        self._plot.clear()
        if report.histogram:
            centers = np.asarray(
                [(b.lower + b.upper) / 2.0 for b in report.histogram], dtype=float
            )
            width = float(np.median(np.diff(centers))) * 0.4 if len(centers) > 1 else 0.5
            self._plot.bars(
                np.asarray([float(b.true_count) for b in report.histogram], dtype=float),
                x=centers - width,
                width=width * 1.6,
                index=0,
            )
            self._plot.bars(
                np.asarray([float(b.dp_count) for b in report.histogram], dtype=float),
                x=centers + width,
                width=width * 1.6,
                index=1,
            )
            self._plot.set_labels(x="bin centre", y="count", title="true (left) vs DP (right) bin counts")
        self._report.setText(
            f"{outcome.source} · n={report.n_values} · bounds "
            f"[{report.bounds[0]:.2f}, {report.bounds[1]:.2f}]"
            + (" (data-derived — production bounds must be public)" if report.bounds_from_data else "")
            + f" · dp mean {report.dp_mean:.4f} vs true {report.true_mean:.4f}\n"
            f"RDP accountant: eps={float(getattr(self, '_dp_epsilon', None) and self._dp_epsilon.value() or 4.0)}, delta=1e-5, 1500 steps needs sigma={plan.sigma:.1f} "
            "(the conservative plain-RDP ceiling; Opacus' amplified accountant needs ~1)"
        )

    # -- recommender ------------------------------------------------------------
    def _start_recommender(self) -> None:
        kind = self._cf_kind.currentText()
        try:
            k = _parse_int(self._cf_k.text(), "neighbours k", minimum=1, maximum=40)
            top_n = _parse_int(self._cf_n.text(), "top N", minimum=1, maximum=10)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return

        def job(progress) -> RecommenderOutcome:
            from pharmaml.ml.recommend import (
                evaluate_recommender,
                recommend_itembased,
                recommend_userbased,
                synthetic_ratings,
            )

            started = _time.monotonic()
            progress(0.3, "the seeded rating matrix")
            matrix = synthetic_ratings(seed=42)
            progress(0.6, "leave-one-out evaluation")
            evaluation = evaluate_recommender(matrix, kind=kind, k=k, top_n=top_n, seed=42)
            recommend = recommend_itembased if kind == "item" else recommend_userbased
            recs = recommend(matrix, user=0, k=k, top_n=min(3, top_n))
            progress(1.0, "done")
            observed = int((~np.isnan(matrix)).sum())
            return RecommenderOutcome(
                matrix_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
                observed=observed,
                evaluation=evaluation,
                recs=recs,
                kind=kind,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "recommender evaluation",
            job,
            on_finished=self._recommender_done,
            on_failed=self._failed,
        )

    def _recommender_done(self, outcome: object) -> None:
        assert isinstance(outcome, RecommenderOutcome)
        evaluation = outcome.evaluation
        recs = outcome.recs
        rows = [
            (int(rec.item), float(rec.predicted_rating), float(rec.neighbour_weight))
            for rec in recs.recommendations
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["item", "predicted", "weight"]),
            note="user 0's ranked recommendations",
        )
        self._plot.clear()
        self._plot.bars(
            np.asarray([float(evaluation.hits_cf), float(evaluation.hits_popularity)], dtype=float),
            labels=[f"CF ({outcome.kind})", "popularity"],
            index=0,
            width=0.5,
        )
        self._plot.set_labels(
            x="recommender",
            y=f"hits @ {evaluation.top_n}",
            title=f"leave-one-out hit-rate over {evaluation.n_eval_users} users",
        )
        verdict = (
            "the neighbourhood beat popularity"
            if evaluation.lift > 1.0
            else "popularity won — the report says so rather than burying it"
        )
        self._report.setText(
            f"{outcome.kind}-based CF (k={evaluation.k}, top {evaluation.top_n}) on "
            f"{outcome.matrix_shape[0]} users x {outcome.matrix_shape[1]} items "
            f"({outcome.observed} ratings): hit-rate {evaluation.hits_cf}/"
            f"{evaluation.n_eval_users} vs popularity {evaluation.hits_popularity}/"
            f"{evaluation.n_eval_users} (lift {evaluation.lift:.2f}) — {verdict}"
        )

    # -- LSH ------------------------------------------------------------
    def _lsh_vectors(self) -> tuple[np.ndarray, str]:
        """The numeric matrix for the ANN card, or the synthetic fallback."""
        frame = self._explorer_frame()
        if frame is not None:
            numeric = [str(c) for c in frame.columns if frame[c].dtype.kind in "biufc"]
            if len(numeric) >= 2:
                matrix = frame[numeric].to_numpy(dtype=float)
                matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
                if matrix.shape[0] >= 60:
                    return matrix, "the numeric block of the loaded table"
        rng = np.random.default_rng(42)
        return (
            np.vstack(
                [
                    rng.normal(size=(120, 8)) + 2.0,
                    rng.normal(size=(120, 8)) - 2.0,
                    rng.normal(size=(120, 8)) * 0.5,
                ]
            ),
            "synthetic three-blob collection (seed 42)",
        )

    def _start_lsh(self) -> None:
        probes = int(self._lsh_probes.currentText())
        vectors, source = self._lsh_vectors()
        if self._busy_check():
            return

        def job(progress) -> LSHOutcome:
            from pharmaml.ml.annsearch import ann_benchmark

            started = _time.monotonic()
            reports = []
            for step, planes in enumerate((2, 4, 6, 8, 10)):
                progress(0.1 + 0.17 * step, f"planes {planes}")
                reports.append(
                    ann_benchmark(vectors, n_planes=planes, probes=probes, k=5, seed=42)
                )
            progress(1.0, "done")
            return LSHOutcome(
                reports=reports,
                source=source,
                shape=(int(vectors.shape[0]), int(vectors.shape[1])),
                probes=probes,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "LSH benchmark",
            job,
            on_finished=self._lsh_done,
            on_failed=self._failed,
        )

    def _lsh_done(self, outcome: object) -> None:
        assert isinstance(outcome, LSHOutcome)
        rows = [
            (
                int(report.n_planes),
                float(report.recall_at_k),
                float(report.candidate_ratio),
                float(report.compression),
            )
            for report in outcome.reports
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["planes", "recall@5", "touched", "cheaper"]),
            note="vs brute force",
        )
        self._plot.clear()
        planes = np.asarray([row[0] for row in rows], dtype=float)
        recall = np.asarray([row[1] for row in rows], dtype=float)
        touched = np.asarray([row[2] for row in rows], dtype=float)
        self._plot.line(planes, recall, label="recall@5", index=0)
        self._plot.line(planes, touched, label="candidates touched", index=1)
        self._plot.set_labels(x="hyperplanes", y="fraction", title="the recall/compression trade-off")
        empty = sum(int(report.empty_queries) for report in outcome.reports)
        self._report.setText(
            f"{outcome.source} ({outcome.shape[0]} vectors x {outcome.shape[1]} dims, "
            f"probes={outcome.probes}): more planes = finer cells = cheaper but lower "
            f"recall; multiprobe buys it back — {empty} empty queries across the sweep"
        )

    # -- adversarial ------------------------------------------------------------
    def _adversarial_xy(self) -> tuple[np.ndarray, np.ndarray, str, str] | None:
        """Features + target from the explorer (or the two-blob fallback), task name."""
        frame = self._explorer_frame()
        label = "the loaded table"
        if frame is None:
            rng = np.random.default_rng(42)
            n_each = 80
            x1 = np.concatenate([rng.normal(0.0, 0.8, n_each), rng.normal(2.5, 0.8, n_each)])
            x2 = np.concatenate([rng.normal(0.0, 0.8, n_each), rng.normal(-1.5, 0.8, n_each)])
            x3 = rng.normal(0.0, 1.0, 2 * n_each)
            cls = np.concatenate([np.zeros(n_each), np.ones(n_each)])
            frame = pd.DataFrame({"x1": x1, "x2": x2, "noise": x3, "class": cls})
            label = "synthetic two-blob demo (seed 42)"
        columns = [str(c) for c in frame.columns]
        target = columns[-1]
        features = [c for c in columns if c != target and frame[c].dtype.kind in "biufc"]
        if not features:
            return None
        x = frame[features].to_numpy(dtype=float)
        raw_y = frame[target]
        if raw_y.dtype.kind in "biufc":
            y = raw_y.to_numpy(dtype=float)
            classes = set(np.unique(y[np.isfinite(y)]).tolist())
            task = "classification" if classes == {0.0, 1.0} else "regression"
        else:
            y = pd.factorize(raw_y)[0].astype(float)
            task = "classification"
        return x, y, task, label

    def _start_adversarial(self) -> None:
        resolved = self._adversarial_xy()
        if resolved is None:
            self._report.setText("No numeric feature columns — load a numeric table first.")
            return
        x, y, task, label = resolved
        if self._busy_check():
            return

        def job(progress) -> AdversarialOutcome:
            from pharmaml.ml.adversarial import adversarial_audit

            started = _time.monotonic()
            progress(0.35, "training the model")
            if task == "classification":
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(x, y.astype(int))
                task_name = "logistic regression"
            else:
                from sklearn.linear_model import Ridge

                model = Ridge(random_state=42)
                model.fit(x, y)
                task_name = "ridge regression"
            progress(0.6, "the zeroth-order attack family")
            scales = x.std(axis=0)
            scales = np.where(scales > 1e-12, scales, 1.0)
            audit = adversarial_audit(
                model.predict,
                x,
                y,
                radii=(0.05, 0.1, 0.25, 0.5),
                task=task,
                tolerance=float(np.std(y)) if task == "regression" else 1.0,
                feature_scales=scales,
                seed=42,
            )
            progress(1.0, "done")
            return AdversarialOutcome(
                audit=audit, task_name=task_name, label=label, seconds=_time.monotonic() - started
            )

        self._runner.submit(
            "adversarial audit",
            job,
            on_finished=self._adversarial_done,
            on_failed=self._failed,
        )

    def _adversarial_done(self, outcome: object) -> None:
        assert isinstance(outcome, AdversarialOutcome)
        audit = outcome.audit
        rows = [
            (float(result.radius), int(result.n_broken), float(result.broken_rate))
            for result in audit.radii_results
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["radius", "broken", "rate"]),
            note="L∞ ball, per-feature spread",
        )
        self._plot.clear()
        radii = np.asarray([row[0] for row in rows], dtype=float)
        rates = np.asarray([row[2] for row in rows], dtype=float)
        self._plot.line(radii, rates, index=0)
        self._plot.scatter(radii, rates, label="broken rate", index=1)
        self._plot.set_labels(x="radius (sd)", y="broken rate", title="the broken-rate curve")
        margin = (
            f"{audit.median_margin:.3f} sd"
            if np.isfinite(audit.median_margin)
            else "beyond the audited range"
        )
        self._report.setText(
            f"{outcome.task_name} on {outcome.label} ({audit.n_samples} rows x "
            f"{audit.n_features} features, {audit.task}): clean accuracy "
            f"{audit.clean_accuracy:.3f}; median adversarial margin {margin} — "
            "the zeroth-order family's lower bound on what a gradient-guided "
            "attacker could do"
        )

    # -- time series ------------------------------------------------------------
    def _ts_values(self) -> tuple[np.ndarray, str]:
        """A numeric column as the series, or the seeded synthetic fallback."""
        frame = self._explorer_frame()
        if frame is not None:
            column = self._ts_column.currentText()
            if column and column in [str(c) for c in frame.columns]:
                values = frame[column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size >= 48:
                    return values, f"column {column!r} of the loaded table"
        from pharmaml.ml.sequences import synthetic_series

        return synthetic_series("sine", 240, seed=42), "synthetic period-24 signal (seed 42)"

    def _start_timeseries(self) -> None:
        try:
            period = _parse_int(self._ts_period.text(), "period", minimum=2, maximum=168)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        series, source = self._ts_values()
        if series.size < 2 * period:
            self._report.setText(
                f"the series ({series.size} points) is shorter than two full periods "
                f"({2 * period}) — lower the period or load a longer column"
            )
            return
        if self._busy_check():
            return

        def job(progress) -> TimeseriesOutcome:
            from pharmaml.ml.timeseries import (
                autocorrelation,
                detect_changepoints,
                partial_autocorrelation,
                seasonal_decomposition,
                window_features,
            )

            started = _time.monotonic()
            progress(0.25, "ACF / PACF")
            max_lag = min(30, series.size - 2)
            acf = autocorrelation(series, max_lag=max_lag)
            pacf = partial_autocorrelation(series, max_lag=max_lag)
            progress(0.5, "the additive decomposition")
            decomposition = seasonal_decomposition(series, period=period)
            progress(0.75, "CUSUM changepoints + windows")
            changes = detect_changepoints(series)
            table = window_features(series, window=period)
            progress(1.0, "done")
            return TimeseriesOutcome(
                acf=acf,
                pacf=pacf,
                decomposition=decomposition,
                changes=changes,
                table=table,
                source=source,
                series_size=int(series.size),
                period=period,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "time-series analysis",
            job,
            on_finished=self._timeseries_done,
            on_failed=self._failed,
        )

    def _timeseries_done(self, outcome: object) -> None:
        assert isinstance(outcome, TimeseriesOutcome)
        acf = outcome.acf
        pacf = outcome.pacf
        decomposition = outcome.decomposition
        changes = outcome.changes
        table = outcome.table
        rows = [
            ("ACF peak lag", str(acf.peak_lag)),
            ("ACF significant lags", f"{len(acf.significant_lags)} of {acf.values.size - 1}"),
            ("PACF lag-1", f"{pacf.values[1]:.3f}"),
            ("seasonal strength", f"{decomposition.seasonal_strength:.3f}"),
            ("trend strength", f"{decomposition.trend_strength:.3f}"),
            ("changepoints", str(len(changes.points))),
        ]
        if changes.points:
            first = changes.points[0]
            rows.append(
                (
                    "first change",
                    (
                        f"index {first.index} ({first.direction}, statistic "
                        f"{first.statistic:.2f} vs {changes.critical_value:.3f})"
                    ),
                )
            )
        rows.append(("window rows", f"{len(table)} x {len(table.columns)}"))
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["metric", "value"]),
            note="the analysis half",
        )
        self._plot.clear()
        lags = np.arange(acf.values.size, dtype=float)
        self._plot.bars(acf.values.astype(float), x=lags, width=0.6, index=0)
        self._plot.set_labels(x="lag", y="ACF", title="the autocorrelation function")
        slope = float(table["slope"].mean()) if len(table) else float("nan")
        self._report.setText(
            f"{outcome.source} ({outcome.series_size} points, period {outcome.period}): "
            f"the seasonal profile explains {100 * decomposition.seasonal_strength:.0f}% "
            f"of the detrended variance; mean window slope {slope:+.4f}/step"
        )

    # -- association rules ------------------------------------------------------------
    def _start_assoc(self) -> None:
        transactions: list[list[str]] = []
        for line in self._assoc_text.toPlainText().splitlines():
            items = [item.strip() for item in line.split(",") if item.strip()]
            if items:
                transactions.append(items)
        if len(transactions) < 2:
            self._report.setText("Paste at least two transaction lines.")
            return
        try:
            min_support = _parse_float(self._assoc_support.text(), "min support", minimum=0.01)
            min_confidence = _parse_float(
                self._assoc_confidence.text(), "min confidence", minimum=0.01
            )
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if min_support > 1.0 or min_confidence > 1.0:
            self._report.setText("Input — support and confidence must be <= 1")
            return
        if self._busy_check():
            return

        def job(progress) -> AprioriOutcome:
            from pharmaml.ml.apriori import apriori_pipeline

            started = _time.monotonic()
            progress(0.5, "frequent itemsets")
            result = apriori_pipeline(
                transactions, min_support=min_support, min_confidence=min_confidence
            )
            progress(1.0, "done")
            return AprioriOutcome(result=result, seconds=_time.monotonic() - started)

        self._runner.submit(
            "association rules",
            job,
            on_finished=self._assoc_done,
            on_failed=self._failed,
        )

    def _assoc_done(self, outcome: object) -> None:
        assert isinstance(outcome, AprioriOutcome)
        result = outcome.result
        rules = result.rules
        self._table.bind_frame(rules, note="rules ranked by lift")
        self._plot.clear()
        if len(rules):
            labels = [
                f"{', '.join(str(i) for i in row['antecedent'])} → "
                f"{', '.join(str(i) for i in row['consequent'])}"
                for _index, row in rules.iterrows()
            ]
            self._plot.bars(
                rules["lift"].to_numpy(dtype=float),
                labels=labels,
                index=0,
                width=0.6,
            )
            self._plot.set_labels(x="rule", y="lift", title="association rules by lift")
        lines = [
            f"{', '.join(str(item) for item in row['antecedent'])} -> "
            f"{', '.join(str(item) for item in row['consequent'])}  supp "
            f"{row['support']:.2f} · conf {row['confidence']:.2f} · lift {row['lift']:.2f}"
            for _index, row in rules.head(8).iterrows()
        ]
        body = "\n".join(lines) if len(lines) else "no rules above the thresholds — lower support/confidence"
        self._report.setText(
            f"{result.n_transactions} transactions · {result.n_items} items · "
            f"{len(result.frequent)} itemsets · {len(rules)} rules\n{body}"
        )

    # -- theme ------------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    # -- test accessors ------------------------------------------------------------
    def report_text(self) -> str:
        return self._report.text()

    def model_frame(self) -> pd.DataFrame | None:
        """The frame currently bound to the result table (or None)."""
        model = self._table.model()
        return model.frame if isinstance(model, PandasTableModel) else None
