"""PK/PD Workbench: real regimens, real NCA (16.8/16.12/16.14).

The Studio's increment-3 page (the playground PK/PD tab, audit row 3).
Three of the tab's engines, each wired to the SAME public backend APIs
the cards call -- nothing on this page is fabricated:

- **Regimen simulator** -- ``pharmaml.pkpd.compartments.TwoCompartment``
  (the matrix-exponential linCmt solution) with explicit repeated
  ``DosingEvent`` records: the central concentration curve with dose
  markers, Cmax/Tmax and the trapezoid AUC over the simulated window
  (the advanced card's "Regimen" engine).
- **Non-compartmental analysis** -- the pasted concentration-time series
  through ``validate_concentration_series`` → ``nca_summary`` (the NCA
  card's trapezoid metrics) plus the extended ``fit_terminal_phase`` /
  ``auc_linear_log`` (linear-up/log-down AUC, terminal λz window), with
  Vd and CL when an IV dose is given. The observed profile is drawn
  with its AUC shaded.
- **Dosing events** -- the NONMEM/mrgsolve event grammar
  (``ev`` / ``addl_until`` / ``realize_addl`` / ``ev_days``): one dose
  record plus additional doses until a stop time, and weekday
  schedules, realized as the full dose-event table.

All three run in a :class:`~ui.workers.job_worker.JobWorker`. They are
single-shot engine calls without checkpoint loops, so the progress
channel reports honestly but the cooperative Stop only lands for
checkpoint-wired jobs -- the same disclosure the standardize card
carries.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

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

#: The NCA card's realistic default series: a one-compartment IV bolus,
#: t1/2 ~ 4 h, sampled to 24 h (mirrored so the Studio starts runnable).
_NCA_DEFAULT_TIMES = "0, 0.5, 1, 2, 4, 6, 8, 12, 16, 24"
_NCA_DEFAULT_CONCS = "12.0, 10.2, 8.7, 6.4, 4.6, 3.3, 2.4, 1.2, 0.63, 0.17"

#: Regimen defaults (the advanced card's own starting values).
_REGIMEN_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("dose_mg", "dose (mg)", "100"),
    ("tau_h", "interval (h)", "24"),
    ("n_doses", "doses", "3"),
    ("ka", "ka (/h)", "1.0"),
    ("cl", "CL (L/h)", "5"),
    ("v2", "central V (L)", "50"),
    ("q", "Q (L/h)", "10"),
    ("v3", "peripheral V (L)", "100"),
    ("duration_h", "duration (h)", "72"),
)

#: Positivity rule copied from the card: these keys must be > 0 (CL may
# be 0 -- the parse only enforces >= 0, exactly like the playground).
_REGIMEN_POSITIVE = ("dose_mg", "tau_h", "ka", "v2", "q", "v3", "duration_h")


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


def _parse_series(text: str, name: str) -> list[float]:
    """Parse a comma/whitespace number list (the card's paste format)."""
    parts = str(text).replace(",", " ").split()
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            raise ValueError(f"{name}: not a number ({part!r})") from None
    return values


@dataclass
class RegimenOutcome:
    """What the regimen job hands back (all real numbers)."""

    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    concentration: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    route: str = "oral"
    duration_h: float = 0.0
    c_max: float = 0.0
    t_max: float = 0.0
    auc: float = 0.0
    seconds: float = 0.0


@dataclass
class NCAOutcome:
    """What the NCA job hands back (trapezoid + linear-up/log-down)."""

    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    concs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_points: int = 0
    auc_trap: float = 0.0
    auc_lld: float = 0.0
    half_life: float = 0.0
    window: str = ""
    r2_adj: float = 0.0
    has_dose: bool = False
    vd: float = 0.0
    clearance: float = 0.0
    seconds: float = 0.0


@dataclass
class EventsOutcome:
    """What the dosing-events job hands back (the realized table)."""

    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_events: int = 0
    n_addl: int = 0
    total_mg: float = 0.0
    weekly_times: list[float] = field(default_factory=list)
    seconds: float = 0.0


class PKPDWorkbenchPage(QWidget):
    """Simulate an exact regimen; run NCA; realize dosing events."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_regimen_tab(), "Regimen simulator")
        self._tabs.addTab(self._build_nca_tab(), "Non-compartmental analysis")
        self._tabs.addTab(self._build_events_tab(), "Dosing events")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="PK profile")
        splitter.addWidget(self._plot)
        splitter.setSizes([600, 500])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No PK/PD run yet.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

    # -- tab construction ----------------------------------------------------
    def _build_regimen_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._regimen_entries: dict[str, QLineEdit] = {}
        for key, label, default in _REGIMEN_FIELDS:
            entry = QLineEdit(default)
            self._regimen_entries[key] = entry
            form.addRow(f"{label}:", entry)
        self._route_combo = QComboBox()
        self._route_combo.addItems(("oral", "iv"))
        form.addRow("route:", self._route_combo)
        self._sim_btn = QPushButton("Simulate regimen")
        self._sim_btn.setMinimumHeight(34)
        self._sim_btn.clicked.connect(self._start_regimen)
        form.addRow("", self._sim_btn)
        return tab

    def _build_nca_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._nca_times = QLineEdit(_NCA_DEFAULT_TIMES)
        form.addRow("times (h):", self._nca_times)
        self._nca_concs = QLineEdit(_NCA_DEFAULT_CONCS)
        form.addRow("concentrations:", self._nca_concs)
        self._nca_dose = QLineEdit()
        self._nca_dose.setPlaceholderText("optional")
        form.addRow("IV dose (mg):", self._nca_dose)
        self._nca_btn = QPushButton("Analyse series")
        self._nca_btn.setMinimumHeight(34)
        self._nca_btn.clicked.connect(self._start_nca)
        form.addRow("", self._nca_btn)
        hint = QLabel(
            "Matched comma lists. Sorted times, non-negative concentrations; "
            "Vd and CL appear only when an IV dose is given."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_events_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._ev_dose = QLineEdit("500")
        form.addRow("dose (mg):", self._ev_dose)
        self._ev_ii = QLineEdit("12")
        form.addRow("interval (h):", self._ev_ii)
        self._ev_until = QLineEdit("72")
        form.addRow("until (h):", self._ev_until)
        self._ev_days = QLineEdit("m, w, f")
        form.addRow("weekday codes:", self._ev_days)
        self._ev_btn = QPushButton("Realize regimen")
        self._ev_btn.setMinimumHeight(34)
        self._ev_btn.clicked.connect(self._start_events)
        form.addRow("", self._ev_btn)
        return tab

    # -- job plumbing -----------------------------------------------------------
    def _on_started(self, name: str) -> None:
        for btn in (self._sim_btn, self._nca_btn, self._ev_btn):
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in (self._sim_btn, self._nca_btn, self._ev_btn):
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return True
        return False

    # -- regimen ----------------------------------------------------------------
    def _start_regimen(self) -> None:
        try:
            values = {
                key: _parse_float(entry.text(), key, minimum=0.0)
                for key, entry in self._regimen_entries.items()
            }
            n_doses = _parse_int(
                self._regimen_entries["n_doses"].text(), "n_doses", minimum=1, maximum=100
            )
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        for key in _REGIMEN_POSITIVE:
            if values[key] <= 0:
                self._report.setText(f"Input — {key} must be > 0")
                return
        if self._busy_check():
            return
        route = self._route_combo.currentText()

        def job(progress) -> RegimenOutcome:
            from pharmaml.pkpd.compartments import TwoCompartment
            from pharmaml.pkpd.ode import DosingEvent

            started = _time.monotonic()
            progress(0.05, "building dosing events")
            model = TwoCompartment(
                ka=values["ka"], cl=values["cl"], v2=values["v2"],
                q=values["q"], v3=values["v3"],
            )
            times = np.linspace(0.0, values["duration_h"], 145)
            compartment = 0 if route == "oral" else 1
            amount = values["dose_mg"] * (model.f if route == "oral" else 1.0)
            events = [
                DosingEvent(time=i * values["tau_h"], amount=amount, compartment=compartment)
                for i in range(n_doses)
                if i * values["tau_h"] <= values["duration_h"]
            ]
            progress(0.45, "matrix-exponential solve")
            amounts = model.amounts(times, values["dose_mg"], route=route, events=events)
            concentration = amounts[:, 1] / model.v2
            progress(0.85, "summarising")
            auc = float(
                np.sum((concentration[:-1] + concentration[1:]) * 0.5 * np.diff(times))
            )
            frame = pd.DataFrame(
                {
                    "time_h": [e.time for e in events],
                    "amount_mg": [e.amount for e in events],
                    "cmt": [e.compartment for e in events],
                    "evid": [1] * len(events),
                }
            )
            progress(1.0, "done")
            return RegimenOutcome(
                times=times,
                concentration=concentration,
                events=frame,
                route=route,
                duration_h=values["duration_h"],
                c_max=float(concentration.max()),
                t_max=float(times[int(concentration.argmax())]),
                auc=auc,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"regimen simulation ({route})",
            job,
            on_finished=self._regimen_done,
            on_failed=self._failed,
        )

    def _regimen_done(self, outcome: object) -> None:
        assert isinstance(outcome, RegimenOutcome)
        self._table.bind_frame(outcome.events)
        self._plot.clear()
        self._plot.line(outcome.times, outcome.concentration, label="central concentration")
        self._plot.vlines(outcome.events["time_h"].tolist())
        self._plot.set_labels(
            x="time (h)",
            y="central concentration (mg/L)",
            title=(
                f"Exact two-compartment regimen · {len(outcome.events)} "
                f"{outcome.route} dose(s)"
            ),
        )
        self._report.setText(
            f"Regimen — {len(outcome.events)} {outcome.route} dose(s) · "
            f"Cmax {outcome.c_max:.5g} mg/L at {outcome.t_max:.3g} h · "
            f"AUC(0–{outcome.duration_h:g} h) {outcome.auc:.5g} mg·h/L · "
            f"matrix-exponential linCmt solution ({outcome.seconds:.1f}s)"
        )

    # -- NCA --------------------------------------------------------------------
    def _start_nca(self) -> None:
        try:
            times = _parse_series(self._nca_times.text(), "times")
            concs = _parse_series(self._nca_concs.text(), "concentrations")
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if len(times) != len(concs):
            self._report.setText(
                f"Input — times and concentrations must list the same number of "
                f"points ({len(times)} vs {len(concs)})"
            )
            return
        if len(times) < 3:
            self._report.setText("Input — NCA needs at least 3 points.")
            return
        dose_raw = self._nca_dose.text().strip()
        dose: float | None = None
        if dose_raw:
            try:
                dose = _parse_float(dose_raw, "IV dose", minimum=0.0)
            except ValueError as exc:
                self._report.setText(f"Input — {exc}")
                return
            if dose <= 0:
                self._report.setText("Input — IV dose must be > 0 (or leave it empty).")
                return
        if self._busy_check():
            return

        def job(progress) -> NCAOutcome:
            from pharmaml.pharma.pk import (
                nca_summary,
                validate_concentration_series,
                vd_from_dose_auc,
            )
            from pharmaml.pkpd.nca_ext import auc_linear_log, fit_terminal_phase

            started = _time.monotonic()
            progress(0.15, "validating series")
            t, c = validate_concentration_series(times, concs)
            summary = nca_summary(t, c)
            progress(0.5, "terminal phase")
            fit = fit_terminal_phase(t, c)
            auc_lld = auc_linear_log(t, c)
            vd = vd_from_dose_auc(dose, summary.auc) if dose is not None else 0.0
            clearance = (
                (dose / summary.auc) if (dose is not None and summary.auc) else float("nan")
            )
            rows: list[tuple[str, float, str]] = [
                ("AUC (trapezoid)", summary.auc, "mg·h/L"),
                ("AUC (linear-up/log-down)", auc_lld, "mg·h/L"),
                ("AUMC (trapezoid)", summary.aucm, "mg·h²/L"),
                ("MRT", summary.mrt, "h"),
                ("Cmax", summary.c_max, "mg/L"),
                ("Tmax", summary.t_max, "h"),
                ("t½ terminal (trapezoid)", summary.half_life_terminal, "h"),
                ("λz", fit.lambda_z, "1/h"),
                ("t½ (λz)", fit.half_life, "h"),
                ("terminal R² (adj)", fit.r2_adj, "—"),
                ("span", fit.span_half_lives, "half-lives"),
            ]
            if dose is not None:
                rows.append(("Vd (dose/AUC)", vd, "L"))
                rows.append(("CL (dose/AUC)", clearance, "L/h"))
            progress(0.9, "done")
            return NCAOutcome(
                metrics=pd.DataFrame(rows, columns=["metric", "value", "unit"]),
                times=t,
                concs=c,
                n_points=len(t),
                auc_trap=float(summary.auc),
                auc_lld=float(auc_lld),
                half_life=float(fit.half_life),
                window=f"rows {fit.start_index}–{fit.end_index} ({fit.n_points} points)",
                r2_adj=float(fit.r2_adj),
                has_dose=dose is not None,
                vd=float(vd),
                clearance=float(clearance),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "non-compartmental analysis",
            job,
            on_finished=self._nca_done,
            on_failed=self._failed,
        )

    def _nca_done(self, outcome: object) -> None:
        assert isinstance(outcome, NCAOutcome)
        self._table.bind_frame(outcome.metrics)
        self._plot.clear()
        self._plot.line(outcome.times, outcome.concs, fill=True, label="observed (AUC shaded)")
        self._plot.scatter(outcome.times, outcome.concs, label="samples")
        self._plot.set_labels(
            x="time (h)",
            y="concentration (mg/L)",
            title=f"Observed profile · {outcome.n_points} points",
        )
        parts = [
            f"{outcome.n_points} points",
            f"AUC {outcome.auc_trap:.4g} trap / {outcome.auc_lld:.4g} lin-up/log-down mg·h/L",
            f"t½(λz) {outcome.half_life:.3g} h",
            f"window {outcome.window} · adj R² {outcome.r2_adj:.3f}",
        ]
        if outcome.has_dose:
            parts.append(f"Vd {outcome.vd:.4g} L · CL {outcome.clearance:.4g} L/h")
        parts.append(f"{outcome.seconds:.1f}s")
        self._report.setText("NCA — " + " · ".join(parts))

    # -- dosing events -------------------------------------------------------------
    def _start_events(self) -> None:
        try:
            dose = _parse_float(self._ev_dose.text(), "dose", minimum=0.0)
            interval = _parse_float(self._ev_ii.text(), "interval")
            until = _parse_float(self._ev_until.text(), "until", minimum=0.0)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if interval <= 0:
            self._report.setText("Input — interval must be > 0")
            return
        day_codes = [c.strip().lower() for c in self._ev_days.text().split(",") if c.strip()]
        if self._busy_check():
            return

        def job(progress) -> EventsOutcome:
            from pharmaml.pkpd.events import addl_until, ev, ev_days, realize_addl

            started = _time.monotonic()
            progress(0.2, "counting additional doses")
            n_addl = addl_until(0.0, interval, until)
            events = realize_addl(ev(dose, ii=interval, addl=n_addl))
            weekly = ev_days(dose, day_codes) if day_codes else []
            progress(0.7, "realizing the event table")
            rows = [
                {
                    "time_h": e.time,
                    "amount_mg": e.amt,
                    "cmt": e.cmt,
                    "evid": e.evid,
                    "addl": e.addl,
                    "from_addl": e.from_addl,
                    "source": "addl regimen",
                }
                for e in events
            ]
            for e in weekly:
                rows.append(
                    {
                        "time_h": e.time,
                        "amount_mg": e.amt,
                        "cmt": e.cmt,
                        "evid": e.evid,
                        "addl": e.addl,
                        "from_addl": e.from_addl,
                        "source": "weekday schedule",
                    }
                )
            total = float(sum(e.amt for e in events))
            progress(1.0, "done")
            return EventsOutcome(
                frame=pd.DataFrame(rows),
                n_events=len(events),
                n_addl=n_addl,
                total_mg=total,
                weekly_times=[e.time for e in weekly],
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "dosing events",
            job,
            on_finished=self._events_done,
            on_failed=self._failed,
        )

    def _events_done(self, outcome: object) -> None:
        assert isinstance(outcome, EventsOutcome)
        self._table.bind_frame(outcome.frame)
        self._plot.clear()
        weekly = outcome.weekly_times
        if weekly:
            marks = [0.0] + [t for t in weekly if t > 0]
            self._plot.vlines(marks)
        self._plot.set_labels(
            x="time (h)",
            y="—",
            title=(
                f"Dose-event table · {outcome.n_events} addl doses + "
                f"{len(weekly)} weekday doses"
            ),
        )
        parts = [
            (
                f"{outcome.n_events} doses · total {outcome.total_mg:g} mg · "
                f"+{outcome.n_addl} additional"
            )
        ]
        if weekly:
            times = ", ".join(f"{t:g}h" for t in weekly)
            parts.append(f"weekday schedule at {times}")
        parts.append(f"{outcome.seconds:.1f}s")
        self._report.setText("Events — " + " · ".join(parts))

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
