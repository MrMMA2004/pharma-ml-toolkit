"""Foundations Workbench: real statistics, real profiling (16.8/16.12/16.14).

The Studio's increment-9 page (the playground Foundations tab, audit
row 10).  Four of the tab's clusters, each wired to the SAME public
backend APIs the cards call -- nothing on this page is fabricated:

- **Compare groups** -- ``pharmaml.ml.stats.compare_groups`` (the DSBC
  statistics card): descriptives, five-number summaries, both
  normality verdicts, the decision branch reported (``both normal →
  Welch t`` / ``non-normal → Mann-Whitney``), the per-group confidence
  intervals and the effect size, with both group histograms drawn on a
  shared axis.  The synthetic demo mirrors the card exactly (rng 42,
  N(1.2, 0.4, 60) vs N(1.0, 0.45, 50)); the columns mode reads the
  Data Explorer's loaded frame.
- **Distribution sampler** -- ``pharmaml.ml.stats.sample_distribution``
  (the probability lecture's card): seeded draws from the course's
  five distributions with the theoretical density/pmf drawn over the
  sample histogram in the SAME value space, and the sample-vs-theory
  moments in the report.
- **Inferential tests** -- ``pharmaml.ml.inferential`` (the
  pingouin-style card): pasted ``label: values`` groups through Welch
  ANOVA plus BOTH pairwise post-hoc tables (Tukey HSD and
  Games-Howell) -- the Tukey table lands in the data table, the
  Games-Howell text in the report.
- **Table profiler** -- ``pharmaml.profiling.profile_dataset`` (the
  table-prep cluster the audit row names): a pasted CSV profiled with
  the type census, the per-column table (type / missing / distinct /
  top value / key stat) and the alert rules.

All four run in a :class:`~ui.workers.job_worker.JobWorker`.  They are
single-shot engine calls without checkpoint loops, so the progress
channel reports honestly but the cooperative Stop only lands for
checkpoint-wired jobs -- the same disclosure the PK/PD page carries.
"""

from __future__ import annotations

import io as _io
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
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

#: the inferential card's default pasted groups (mirrored verbatim).
_ISTAT_DEFAULT = (
    "control: 2.1, 2.4, 2.2, 2.5, 2.3\n"
    "low: 2.8, 3.1, 2.9, 3.0\n"
    "high: 3.6, 3.9, 3.4, 3.8, 3.7"
)

#: the profiler tab's default pasted CSV -- a compact honest demo with
#: one missing cell and a unique-valued column the alert rules flag.
_PROFILER_DEFAULT = (
    "dose,route,response,note\n"
    "10,IV,1.0,a\n"
    "12,ORAL,1.1,b\n"
    "9,IV,0.9,a\n"
    "11,ORAL,,b\n"
    "10,IV,1.05,c\n"
    "12,ORAL,1.2,a\n"
)

_DISTRIBUTIONS = ("normal", "uniform", "binomial", "poisson", "exponential")


def _parse_float(text: str, name: str, *, minimum: float | None = None) -> float:
    """Parse one float with an honest, named error."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not a number ({text!r})") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value:g})")
    return value


def _parse_int(text: str, name: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not an integer ({text!r})") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} (got {value})")
    return value


@dataclass
class StatsOutcome:
    """What the compare-groups job hands back (the real report object)."""

    report: Any
    group_a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    group_b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    label: str = ""
    seconds: float = 0.0


@dataclass
class SampleOutcome:
    """What the sampler job hands back (draws + theory points)."""

    sample: Any
    theory_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    theory_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    theory_label: str = ""
    discrete: bool = False
    seconds: float = 0.0


@dataclass
class InferentialOutcome:
    """What the inferential job hands back (all three real tables)."""

    anova: pd.DataFrame = field(default_factory=pd.DataFrame)
    tukey: pd.DataFrame = field(default_factory=pd.DataFrame)
    howell: pd.DataFrame = field(default_factory=pd.DataFrame)
    f_stat: float = 0.0
    p_value: float = 0.0
    n_groups: int = 0
    seconds: float = 0.0


@dataclass
class ProfileOutcome:
    """What the profiler job hands back (the DatasetProfile object)."""

    profile: Any
    columns_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    seconds: float = 0.0


class FoundationsWorkbenchPage(QWidget):
    """Compare groups; sample distributions; run post-hoc tests; profile tables."""

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

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_stats_tab(), "Compare groups")
        self._tabs.addTab(self._build_sampler_tab(), "Distribution sampler")
        self._tabs.addTab(self._build_inferential_tab(), "Inferential tests")
        self._tabs.addTab(self._build_profiler_tab(), "Table profiler")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Foundations")
        self._plot.set_empty_hint("Pick a sampler and press Draw samples to explore the distribution")

        splitter.addWidget(self._plot)
        splitter.setSizes([560, 540])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No Foundations run yet.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

    # -- tab construction ----------------------------------------------------
    def _build_stats_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._stat_mode = QComboBox()
        self._stat_mode.setToolTip("The statistic computed on the picked columns (mean, median, spread, tests…)")
        self._stat_mode.addItems(("synthetic demo (two groups)", "columns from the Data Explorer"))
        form.addRow("groups:", self._stat_mode)
        self._stat_col_a = QComboBox()
        form.addRow("column A:", self._stat_col_a)
        self._stat_col_b = QComboBox()
        form.addRow("column B:", self._stat_col_b)
        self._stat_alpha = QLineEdit("0.05")
        form.addRow("α (significance):", self._stat_alpha)
        self._stat_level = QLineEdit("0.95")
        form.addRow("CI level:", self._stat_level)
        self._stat_btn = QPushButton("Compare groups")
        self._stat_btn.setMinimumHeight(34)
        self._stat_btn.clicked.connect(self._start_stats)
        form.addRow("", self._stat_btn)
        hint = QLabel(
            "The demo mirrors the card exactly (rng 42, N(1.2, 0.4) n=60 vs "
            "N(1.0, 0.45) n=50); the columns mode reads the Data Explorer's "
            "loaded table. Both histograms share one axis."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_sampler_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._dist_kind = QComboBox()
        self._dist_kind.setToolTip("The distribution family the sampler draws from (normal, lognormal, beta, binomial…)")
        self._dist_kind.addItems(_DISTRIBUTIONS)
        form.addRow("distribution:", self._dist_kind)
        self._dist_n = QLineEdit("500")
        self._dist_n.setToolTip("Number of samples per draw")
        form.addRow("sample size:", self._dist_n)
        self._dist_loc = QLineEdit("0")
        self._dist_loc.setToolTip("Location parameter (mean shift / lower bound, family-dependent)")
        form.addRow("loc / μ:", self._dist_loc)
        self._dist_scale = QLineEdit("1")
        self._dist_scale.setToolTip("Scale parameter (spread, family-dependent)")
        form.addRow("scale / σ:", self._dist_scale)
        self._dist_p = QLineEdit("0.3")
        self._dist_p.setToolTip("Success probability for the binomial family")
        form.addRow("p (binomial):", self._dist_p)
        self._dist_trials = QLineEdit("10")
        form.addRow("trials (binomial):", self._dist_trials)
        self._dist_lam = QLineEdit("3")
        form.addRow("λ (poisson):", self._dist_lam)
        self._dist_btn = QPushButton("Draw sample")
        self._dist_btn.setMinimumHeight(34)
        self._dist_btn.clicked.connect(self._start_sampler)
        form.addRow("", self._dist_btn)
        return tab

    def _build_inferential_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._istat_text = QPlainTextEdit()
        self._istat_text.setPlainText(_ISTAT_DEFAULT)
        self._istat_text.setPlaceholderText("label: v1, v2, …  (one group per line)")
        form.addRow("groups:", self._istat_text)
        self._istat_btn = QPushButton("Run the tests")
        self._istat_btn.setMinimumHeight(34)
        self._istat_btn.clicked.connect(self._start_inferential)
        form.addRow("", self._istat_btn)
        hint = QLabel(
            "Welch ANOVA across the groups (robust to unequal variances); "
            "Tukey HSD lands in the table, Games-Howell in the report."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_profiler_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._profiler_text = QPlainTextEdit()
        self._profiler_text.setPlainText(_PROFILER_DEFAULT)
        self._profiler_text.setPlaceholderText("Paste a CSV (header row + data rows)")
        form.addRow("CSV:", self._profiler_text)
        self._profiler_btn = QPushButton("Profile table")
        self._profiler_btn.setMinimumHeight(34)
        self._profiler_btn.clicked.connect(self._start_profiler)
        form.addRow("", self._profiler_btn)
        hint = QLabel(
            "The fg-data-profiling decision rules as readable pandas: the type "
            "census, per-column profiles and the alert rules (missing / unique / "
            "constant / imbalanced …)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- job plumbing -----------------------------------------------------------
    def _on_started(self, name: str) -> None:
        for btn in (self._stat_btn, self._dist_btn, self._istat_btn, self._profiler_btn):
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in (self._stat_btn, self._dist_btn, self._istat_btn, self._profiler_btn):
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            # v1.2.204 m-5: name the running job so the guard is
            # actionable ("which job?") instead of anonymous.
            name = getattr(self._runner, "job_name", "") or "another operation"
            self._report.setText(
                f"Busy: '{name}' is running — Stop it (or check Job Center) first."
            )
            return True
        return False

    def _failed(self, message: str) -> None:
        self._report.setText(message.splitlines()[0] if message else "The job failed.")

    # -- compare groups ----------------------------------------------------
    def _start_stats(self) -> None:
        mode = self._stat_mode.currentText()
        group_a: np.ndarray
        group_b: np.ndarray
        label: str
        if mode.startswith("columns"):
            frame: pd.DataFrame | None = None
            if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
                frame = self._data_explorer.frame()
            if frame is None or frame.empty:
                self._report.setText(
                    "The Data Explorer has no table loaded yet (or switch to the demo)."
                )
                return
            numeric = [str(column) for column in frame.select_dtypes(include=[np.number]).columns]
            self._stat_col_a.clear()
            self._stat_col_b.clear()
            self._stat_col_a.addItems(numeric)
            self._stat_col_b.addItems(numeric)
            if len(numeric) >= 2:
                # A stays on the first column, B takes the second -- the
                # two-group comparison starts runnable instead of tripping
                # its own same-column guard
                self._stat_col_b.setCurrentIndex(1)
            column_a = self._stat_col_a.currentText()
            column_b = self._stat_col_b.currentText()
            if not numeric:
                self._report.setText("The loaded table has no numeric columns.")
                return
            if column_a == column_b:
                self._report.setText("Pick two DIFFERENT columns for the two groups.")
                return
            group_a = frame[column_a].to_numpy(dtype=float)
            group_b = frame[column_b].to_numpy(dtype=float)
            label = f"{column_a} vs {column_b}"
        else:
            # the card's own synthetic demo, generated INSIDE the job start
            # exactly like the playground does (rng 42) -- mirrored semantics
            rng = np.random.default_rng(42)
            group_a = rng.normal(1.2, 0.4, 60)
            group_b = rng.normal(1.0, 0.45, 50)
            label = "synthetic demo: N(1.2, 0.4) n=60 vs N(1.0, 0.45) n=50"
        try:
            alpha = _parse_float(self._stat_alpha.text(), "α", minimum=1e-6)
            level = _parse_float(self._stat_level.text(), "CI level", minimum=0.5)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if alpha > 0.5:
            self._report.setText("Input — α must be <= 0.5")
            return
        if level > 0.999:
            self._report.setText("Input — CI level must be <= 0.999")
            return
        if self._busy_check():
            return

        def job(progress) -> StatsOutcome:
            from pharmaml.ml.stats import compare_groups

            started = _time.monotonic()
            progress(0.4, "descriptives, normality, the test")
            report = compare_groups(group_a, group_b, alpha=alpha, level=level, seed=42)
            progress(1.0, "done")
            return StatsOutcome(
                report=report,
                group_a=np.asarray(group_a, dtype=float),
                group_b=np.asarray(group_b, dtype=float),
                label=label,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "group comparison",
            job,
            on_finished=self._stats_done,
            on_failed=self._failed,
        )

    def _stats_done(self, outcome: object) -> None:
        assert isinstance(outcome, StatsOutcome)
        report = outcome.report
        lines = [
            f"groups: {outcome.label}",
            "",
            f"A · {report.group_a.line()}",
            f"    {report.group_a.five_number()}",
            f"B · {report.group_b.line()}",
            f"    {report.group_b.five_number()}",
            "",
            f"normality A: {report.normality_a.line()}",
            f"normality B: {report.normality_b.line()}",
            "",
            f"branch: {report.branch}",
            f"test: {report.result.line()}",
            f"CI A: {report.interval_a.line()}",
            f"CI B: {report.interval_b.line()}",
            "",
            report.headline(),
        ]
        self._report.setText("\n".join(lines))
        self._table.bind_frame(
            pd.DataFrame(
                {
                    "group": ["A", "B"],
                    "n": [int(report.group_a.n), int(report.group_b.n)],
                    "mean": [report.group_a.mean, report.group_b.mean],
                    "std": [report.group_a.std, report.group_b.std],
                    "median": [report.group_a.median, report.group_b.median],
                }
            ),
            note="descriptives",
        )
        self._plot.clear()
        low = float(min(outcome.group_a.min(), outcome.group_b.min()))
        high = float(max(outcome.group_a.max(), outcome.group_b.max()))
        self._plot.histogram(outcome.group_a, bins=30, range=(low, high), index=0)
        self._plot.histogram(outcome.group_b, bins=30, range=(low, high), index=1)
        self._plot.set_labels(x="value", y="count", title=f"Two group histograms · {outcome.label}")

    # -- distribution sampler ----------------------------------------------------
    def _start_sampler(self) -> None:
        kind = self._dist_kind.currentText()
        try:
            n = _parse_int(self._dist_n.text(), "sample size", minimum=10, maximum=100000)
            loc = _parse_float(self._dist_loc.text(), "loc", minimum=-1e6)
            scale = _parse_float(self._dist_scale.text(), "scale", minimum=1e-6)
            p = _parse_float(self._dist_p.text(), "p", minimum=1e-6)
            lam = _parse_float(self._dist_lam.text(), "λ", minimum=1e-6)
            trials = _parse_int(self._dist_trials.text(), "trials", minimum=1, maximum=100000)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return

        def job(progress) -> SampleOutcome:
            from scipy import stats as scipy_stats

            from pharmaml.ml.stats import sample_distribution

            started = _time.monotonic()
            progress(0.35, "seeded draws")
            sample = sample_distribution(
                kind, n=n, loc=loc, scale=scale, p=p, lam=lam, trials=trials, seed=42
            )
            progress(0.8, "theory curve")
            discrete = kind in ("binomial", "poisson")
            if discrete:
                grid = np.arange(
                    int(sample.values.min()), int(sample.values.max()) + 1, dtype=float
                )
                pmf = (
                    scipy_stats.binom.pmf(grid, trials, p)
                    if kind == "binomial"
                    else scipy_stats.poisson.pmf(grid, lam)
                )
                theory_x, theory_y, theory_label = grid, pmf, "theory pmf"
            else:
                xs = np.linspace(
                    float(sample.values.min()), float(sample.values.max()), 400
                )
                if kind == "normal":
                    pdf = scipy_stats.norm.pdf(xs, loc=loc, scale=scale)
                    theory_label = f"N({loc:g}, {scale:g}²)"
                elif kind == "uniform":
                    pdf = scipy_stats.uniform.pdf(xs, loc=loc, scale=scale)
                    theory_label = f"U[{loc:g}, {loc + scale:g}]"
                else:
                    pdf = scipy_stats.expon.pdf(xs, scale=scale)
                    theory_label = f"Exp(scale {scale:g})"
                theory_x, theory_y = xs, pdf
            progress(1.0, "done")
            return SampleOutcome(
                sample=sample,
                theory_x=theory_x,
                theory_y=np.asarray(theory_y, dtype=float),
                theory_label=theory_label,
                discrete=discrete,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"distribution sampler ({kind})",
            job,
            on_finished=self._sampler_done,
            on_failed=self._failed,
        )

    def _sampler_done(self, outcome: object) -> None:
        assert isinstance(outcome, SampleOutcome)
        sample = outcome.sample
        self._report.setText(
            f"{sample.line()} · theory drawn: {outcome.theory_label} "
            f"({outcome.seconds:.1f}s)"
        )
        self._table.bind_frame(
            pd.DataFrame(
                {
                    "kind": [sample.kind],
                    "n": [sample.n],
                    "sample_mean": [float(sample.values.mean())],
                    "theory_mean": [sample.theoretical_mean],
                    "sample_std": [float(sample.values.std())],
                    "theory_std": [sample.theoretical_std],
                }
            ),
            note="moments",
        )
        self._plot.clear()
        if outcome.discrete:
            edges = np.arange(
                float(sample.values.min()) - 0.5, float(sample.values.max()) + 1.5, 1.0
            )
            self._plot.histogram(sample.values, bins=edges, index=0)
            self._plot.line(
                outcome.theory_x,
                outcome.theory_y * sample.n,
                label=outcome.theory_label,
                width=2,
                index=1,
            )
            self._plot.set_labels(
                x="value", y="count", title=f"{sample.kind} · n = {sample.n} (pmf × n)"
            )
        else:
            self._plot.histogram(sample.values, bins=40, index=0)
            self._plot.line(
                outcome.theory_x,
                outcome.theory_y,
                label=outcome.theory_label,
                width=2,
                index=1,
            )
            self._plot.set_labels(
                x="value", y="density / frequency", title=f"{sample.kind} · n = {sample.n}"
            )

    # -- inferential tests ----------------------------------------------------
    def _start_inferential(self) -> None:
        raw = self._istat_text.toPlainText()
        groups: dict[str, list[float]] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, _, values = line.partition(":")
            try:
                numbers = [float(part) for part in values.split(",") if part.strip()]
            except ValueError:
                numbers = []
            if numbers:
                groups[name.strip() or f"group {len(groups) + 1}"] = numbers
        if len(groups) < 2:
            self._report.setText("Paste at least two 'label: values' lines.")
            return
        ordered = list(groups.values())
        if self._busy_check():
            return

        def job(progress) -> InferentialOutcome:
            from pharmaml.ml.inferential import games_howell, tukey_hsd, welch_anova

            started = _time.monotonic()
            progress(0.3, "Welch ANOVA")
            anova = welch_anova(ordered)
            progress(0.6, "Tukey HSD")
            tukey = tukey_hsd(ordered)
            progress(0.9, "Games-Howell")
            howell = games_howell(ordered)
            progress(1.0, "done")
            row = anova.iloc[0]
            return InferentialOutcome(
                anova=anova,
                tukey=tukey,
                howell=howell,
                f_stat=float(row["F"]),
                p_value=float(row["p-unc"]),
                n_groups=len(ordered),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "inferential statistics",
            job,
            on_finished=self._inferential_done,
            on_failed=self._failed,
        )

    def _inferential_done(self, outcome: object) -> None:
        assert isinstance(outcome, InferentialOutcome)
        self._table.bind_frame(outcome.tukey, note="Tukey HSD (pairwise)")
        self._plot.clear()
        f_stat = outcome.f_stat
        p_value = outcome.p_value
        self._plot.bars(
            np.asarray([f_stat], dtype=float),
            labels=[f"Welch F ({outcome.n_groups} groups)"],
            index=0,
        )
        self._plot.set_labels(x="", y="F statistic", title="Welch ANOVA")
        self._report.setText(
            f"Welch F = {f_stat:.3f} · p = {p_value:.4g} · {outcome.n_groups} groups\n"
            f"Tukey HSD (pairwise) — in the table below.\n\n"
            f"Games-Howell (pairwise):\n{outcome.howell.to_string(index=False)}"
        )

    # -- table profiler ----------------------------------------------------
    def _start_profiler(self) -> None:
        raw = self._profiler_text.toPlainText().strip()
        if not raw:
            self._report.setText("Paste a CSV first (header row + data rows).")
            return
        if self._busy_check():
            return

        def job(progress) -> ProfileOutcome:
            from pharmaml.profiling import profile_dataset

            started = _time.monotonic()
            progress(0.35, "parsing the pasted CSV")
            frame = pd.read_csv(_io.StringIO(raw))
            progress(0.7, "type inference + alerts")
            profile = profile_dataset(frame)
            progress(1.0, "done")
            return ProfileOutcome(
                profile=profile,
                columns_frame=profile.as_frame(),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "table profile",
            job,
            on_finished=self._profiler_done,
            on_failed=self._failed,
        )

    def _profiler_done(self, outcome: object) -> None:
        assert isinstance(outcome, ProfileOutcome)
        profile = outcome.profile
        self._table.bind_frame(outcome.columns_frame, note="per-column profile")
        self._plot.clear()
        census_items = sorted(profile.types.items())
        self._plot.bars(
            np.asarray([float(count) for _kind, count in census_items], dtype=float),
            labels=[f"{kind}" for kind, _count in census_items],
            index=0,
        )
        self._plot.set_labels(x="column type", y="columns", title="The type census")
        alerts = "\n".join(
            f"  {alert.column}: {alert.kind} — {alert.message}" for alert in profile.alerts
        )
        self._report.setText(
            f"{profile.headline()}\n\nAlerts ({len(profile.alerts)}):\n"
            f"{alerts if alerts else '  none'}"
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
