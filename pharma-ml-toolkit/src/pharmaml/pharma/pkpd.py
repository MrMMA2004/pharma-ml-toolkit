"""Compartmental PK/PD simulation: ODE engine, event tables, terminal-phase
NCA and population NLME fitting.

This module absorbs the *algorithmic semantics* of four simulation
frameworks into the toolkit's lightweight numpy/pandas idiom:

* **rxode2 / nlmixr2 event tables** (nlmixrdevelopment and nlmixr2 cores)
  -- dosing records ``(time, amount, compartment, infusion duration)``
  rendered as the NM-TRAN columns ``ID/TIME/EVID/AMT/CMT/RATE``, plus
  multiple-dose regimens and the solved-model (``linCmt``) concept: the
  one-compartment oral model ships both an ODE and the closed-form
  Bateman superposition used by the population fitter.
* **mrgsolve / RxODE ODE solving** -- a classical RK4 integrator that
  walks a piecewise event grid (dose times and infusion stops are exact
  breakpoints), so boluses jump states and infusions enter as
  piecewise-constant rate vectors, exactly the semantics the R engines
  give their compiled models.
* **pkpdutils NCA** (König) -- the Phoenix/PKNCA terminal-phase rule
  (``BEST_FIT``): candidate windows after the peak, ``>= 3`` points,
  negative slope, maximal *adjusted* R² with a more-points tie
  tolerance of ``1e-4``; then ``AUCinf`` observed/predicted, ``AUMC``
  extrapolation, infusion-corrected MRT, effective half-life and the
  CDISC PARAMCD crosswalk for the reported parameters.
* **nlmixr2 population estimation** -- the FOCEI objective as an exact
  Laplace approximation at each subject's conditional mode (MAP eta),
  an outer Nelder-Mead over log-scaled thetas/omegas/sigmas, and the
  classic diagnostics: ``objf/AIC/BIC``, BSV CV %, SD-scale eta
  shrinkage and the ``posthoc`` MAP refit.

Honest boundary: this is a miniature, not a vendored engine -- the C
compilation, sensitivity equations, SAEM/IMP samplers and CWRES/NPDE
tables of the originals are out of scope; everything here runs in
plain numpy on analytic or RK4-solved one-/two-compartment models.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "CDISC_PK_CODES",
    "DoseEvent",
    "EventTable",
    "PopulationFit",
    "TerminalFit",
    "bateman_superposition",
    "compartment_rhs",
    "fit_lambda_z",
    "fit_population",
    "focei_objective",
    "formulation_events",
    "formulation_fraction",
    "macro_constants",
    "map_etas",
    "nca_profile",
    "residual_sd",
    "sensitivity_analysis",
    "simulate_compartment",
    "simulate_population",
    "solve_ode",
    "steady_state_iterative",
    "steady_state_one_compartment",
    "superposition_profile",
]

# --------------------------------------------------------------------- #
# Event tables (rxode2 / nlmixr2 ``et()`` semantics)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class DoseEvent:
    """One dosing record -- the ``et()`` entry of rxode2/nlmixr2.

    ``duration=None`` is a bolus (EVID 1, RATE 0); a finite ``duration``
    is a zero-order infusion (EVID 1 with ``RATE = AMT / duration`` in
    the NM-TRAN rendering, the NONMEM ``-2'' style rate records of the
    R engines).  ``evid`` carries mrgsolve's extended record types
    where the engine honours them: 1 (dose), 3 (reset every state to
    zero), 8 (replace: ``y[cmt] = amount``); EVID 2/4/5/6 (compartment
    on/off, reset+dose, multiply) stay documented-but-not-carried --
    the honest boundary of this miniature.
    """

    time: float
    amount: float
    cmt: int = 0
    duration: float | None = None
    evid: int = 1

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise DataValidationError("dose amount must be non-negative")
        if self.duration is not None and self.duration <= 0:
            raise DataValidationError("infusion duration must be positive")
        if self.evid not in (1, 3, 8):
            raise DataValidationError("supported evid values: 1 (dose), 3 (reset), 8 (replace)")

    @property
    def rate(self) -> float | None:
        """Zero-order input rate (amount per time unit), ``None`` for boluses."""
        if self.duration is None:
            return None
        return self.amount / self.duration

    @property
    def end_time(self) -> float | None:
        """Stop time of the infusion (``None`` for boluses)."""
        if self.duration is None:
            return None
        return self.time + self.duration


class EventTable:
    """A dosing + observation schedule (rxode2's ``eventTable``).

    The table is the single source of truth handed to
    :func:`solve_ode`: bolus events jump the state vector at their
    time, infusion events contribute a piecewise-constant rate vector,
    and :meth:`to_frame` renders the NM-TRAN column layout nlmixr2
    expects (``ID/TIME/EVID/AMT/CMT/RATE``) so a schedule round-trips
    through a DataFrame exactly like an nlmixr2 dataset.

    Examples
    --------
    >>> table = EventTable()
    >>> _ = table.add_dosing(0.0, 250.0)
    >>> _ = table.regimen(dose=100.0, interval=8.0, n=5, start=24.0, cmt=1)
    >>> len(table)
    6
    >>> frame = table.to_frame()
    >>> list(frame.columns)
    ['ID', 'TIME', 'EVID', 'AMT', 'CMT', 'RATE']
    >>> frame["EVID"].tolist()
    [1, 1, 1, 1, 1, 1]
    >>> float(frame["RATE"].iloc[0])
    0.0
    >>> _ = table.add_dosing(60.0, 60.0, duration=2.0)
    >>> float(table.to_frame()["RATE"].iloc[-1])
    30.0
    """

    def __init__(self, events: list[DoseEvent] | None = None) -> None:
        self._events: list[DoseEvent] = list(events or [])

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"EventTable({len(self._events)} events)"

    def add_dosing(
        self,
        time: float,
        amount: float,
        cmt: int = 0,
        duration: float | None = None,
        ii: float | None = None,
        addl: int = 0,
        until: float | None = None,
        total: int | None = None,
    ) -> EventTable:
        """Append one dose (bolus unless ``duration`` is given).

        The mrgsolve/RxODE repetition conveniences: ``addl`` schedules
        ``addl`` additional doses every ``ii`` time units; ``until``
        derives ``addl = ceil((until - time)/ii) - 1`` (doses strictly
        *before* ``until``); ``total`` is the whole dose count (so
        ``addl = total - 1``).  ``ii`` is required whenever any of the
        three is given.
        """
        if total is not None:
            addl = max(int(total) - 1, 0)
        if until is not None:
            if ii is None or ii <= 0:
                raise DataValidationError("until needs a positive inter-dose interval ii")
            addl = max(math.ceil((until - time) / ii) - 1, 0)
        if addl and (ii is None or ii <= 0):
            raise DataValidationError("addl needs a positive inter-dose interval ii")
        self._events.append(
            DoseEvent(time=float(time), amount=float(amount), cmt=cmt, duration=duration)
        )
        for k in range(1, int(addl) + 1):
            self._events.append(
                DoseEvent(
                    time=float(time) + k * float(ii or 0.0),
                    amount=float(amount),
                    cmt=cmt,
                    duration=duration,
                )
            )
        return self

    def regimen(
        self,
        dose: float,
        interval: float,
        n: int,
        start: float = 0.0,
        cmt: int = 0,
        duration: float | None = None,
    ) -> EventTable:
        """``n`` equal doses every ``interval`` time units from ``start``.

        The ``Dosing.regimen`` / ``et()`` idiom: a steady regimen is
        three numbers, and the FIRST dose lands at ``start`` (not at
        ``start + interval``), matching both rxode2 and mrgsolve.
        """
        if n < 1:
            raise DataValidationError("regimen needs at least one dose")
        if interval <= 0:
            raise DataValidationError("dosing interval must be positive")
        for k in range(n):
            self.add_dosing(start + k * interval, dose, cmt=cmt, duration=duration)
        return self

    def boluses(self) -> list[DoseEvent]:
        """All bolus events sorted by time."""
        return sorted((e for e in self._events if e.duration is None), key=lambda e: e.time)

    def infusions(self) -> list[DoseEvent]:
        """All infusion events sorted by time."""
        return sorted((e for e in self._events if e.duration is not None), key=lambda e: e.time)

    def to_frame(self) -> pd.DataFrame:
        """Render as the NM-TRAN event table (nlmixr2's data contract).

        ``EVID=1`` marks dose records; ``RATE`` is the infusion rate in
        amount/time (0 for boluses), the convention NONMEM's ``RATE``
        column uses for zero-order input.
        """
        rows = []
        for event in sorted(self._events, key=lambda e: e.time):
            rows.append(
                {
                    "ID": 1,
                    "TIME": event.time,
                    "EVID": event.evid,
                    "AMT": event.amount,
                    "CMT": event.cmt + 1,
                    "RATE": event.rate or 0.0,
                }
            )
        return pd.DataFrame(rows, columns=["ID", "TIME", "EVID", "AMT", "CMT", "RATE"])


# --------------------------------------------------------------------- #
# ODE engine (mrgsolve / RxODE solving semantics)
# --------------------------------------------------------------------- #


def solve_ode(
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_eval: np.ndarray,
    events: EventTable | None = None,
    max_step: float = 0.01,
) -> np.ndarray:
    """Classical RK4 over a piecewise event grid; returns states at ``t_eval``.

    The grid is the sorted union of the evaluation times, the bolus
    times and the infusion stop times -- every event is a *breakpoint*,
    so a bolus is applied exactly at its timestamp and an infusion
    contributes a constant rate vector throughout its window (the
    mrgsolve/RxODE treatment of compiled dosing events).  Between
    breakpoints the interval is chopped into substeps of at most
    ``max_step`` and integrated with the classical fourth-order
    Runge-Kutta scheme.

    ``rhs(t, y, rate)`` receives the current time, the state vector and
    a rate vector of the same length holding the active zero-order
    input per compartment (all zeros between infusions).

    Examples
    --------
    >>> rhs = lambda t, y, rate: -0.25 * y + rate      # doctest: +SKIP
    >>> ys = solve_ode(rhs, np.array([10.0]), np.linspace(0, 4, 5))  # doctest: +SKIP
    >>> ys  # doctest: +SKIP
    array([[10.        ],
           [ 3.67879441], ...])

    The real checks live on the compartment models below; a quick one
    here: a pure exponential integrated over four half-lives.

    >>> def decay(_t, y, _rate):
    ...     return -0.25 * y
    >>> times = np.array([0.0, 4.0])
    >>> ys = solve_ode(decay, np.array([8.0]), times, max_step=0.01)
    >>> abs(float(ys[-1, 0]) - 8.0 * math.exp(-1.0)) < 1e-11   # RK4 accuracy
    True
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.size < 2:
        raise DataValidationError("t_eval needs at least two time points")
    if np.any(np.diff(t_eval) <= 0):
        raise DataValidationError("t_eval must be strictly increasing")
    y = np.asarray(y0, dtype=float).copy()
    n = y.size

    events = events or EventTable()
    rate = np.zeros(n)
    # breakpoints: output grid ∪ bolus times ∪ infusion starts/stops
    marks = {float(t) for t in t_eval}
    for event in events.boluses():
        marks.add(float(event.time))
    for event in events.infusions():
        marks.add(float(event.time))
        marks.add(float(event.end_time or event.time))

    def apply_discrete(event: DoseEvent) -> None:
        """EVID 1 bolus / 3 reset / 8 replace -- mrgsolve's implement()."""
        nonlocal y
        if event.evid == 3:
            y = np.zeros(n)
        elif event.evid == 8:
            y[event.cmt] = event.amount
        else:
            y[event.cmt] += event.amount
    grid = np.array(sorted(marks), dtype=float)
    grid = grid[(grid >= t_eval[0] - 1e-12) & (grid <= t_eval[-1] + 1e-12)]

    from itertools import pairwise

    out = np.full((t_eval.size, n), np.nan)
    # a discrete event landing exactly on the first output time applies first
    for event in events.boluses():
        if abs(event.time - t_eval[0]) <= 1e-12:
            apply_discrete(event)
    out[0] = y

    eval_positions = {int(i) for i in range(t_eval.size)}
    f = rhs

    for seg_start, seg_end in pairwise(grid):
        # infusion rates active on [seg_start, seg_end)
        rate[:] = 0.0
        for event in events.infusions():
            if event.time <= seg_start + 1e-12 and (event.end_time or 0.0) > seg_end - 1e-12:
                rate[event.cmt] += event.rate or 0.0
        span = seg_end - seg_start
        n_sub = max(1, math.ceil(span / max_step))
        dt = span / n_sub
        t = seg_start
        for _ in range(n_sub):
            k1 = f(t, y, rate)
            k2 = f(t + dt / 2, y + dt / 2 * k1, rate)
            k3 = f(t + dt / 2, y + dt / 2 * k2, rate)
            k4 = f(t + dt, y + dt * k3, rate)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        # discrete events landing exactly at seg_end
        for event in events.boluses():
            if abs(event.time - seg_end) <= 1e-12:
                apply_discrete(event)
        where = np.nonzero(np.isclose(t_eval, seg_end, rtol=0, atol=1e-12))[0]
        for idx in where:
            if idx in eval_positions:
                out[idx] = y
    return out


# --------------------------------------------------------------------- #
# Compartment models
# --------------------------------------------------------------------- #

_MODELS: dict[str, dict[str, Any]] = {
    "iv_1c": {"states": ("central",), "params": ("ke", "v")},
    "oral_1c": {"states": ("gut", "central"), "params": ("ka", "ke", "v", "f")},
    "iv_2c": {"states": ("central", "peripheral"), "params": ("k10", "k12", "k21", "v1")},
    "oral_2c": {"states": ("gut", "central", "peripheral"), "params": ("ka", "k10", "k12", "k21", "v1", "f")},
    "iv_3c": {"states": ("central", "peripheral", "deep"), "params": ("k10", "k12", "k21", "k13", "k31", "v1")},
    "oral_3c": {
        "states": ("gut", "central", "peripheral", "deep"),
        "params": ("ka", "k10", "k12", "k21", "k13", "k31", "v1", "f"),
    },
}


def compartment_rhs(
    model: str, params: dict[str, float]
) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
    """Build the ``rhs(t, y, rate)`` for a standard compartment model.

    ``iv_1c`` / ``oral_1c`` / ``iv_2c`` / ``oral_2c`` / ``iv_3c`` /
    ``oral_3c`` with first-order (micro-constant) transfers ``k12``/
    ``k21`` (and ``k13``/``k31`` for the deep compartment) and
    elimination ``k10`` from the central compartment; ``f`` is oral
    bioavailability applied to the gut dose (infusions and IV boluses
    ignore it, as in the engines' event handling).  The three-
    compartment pair is mrgsolve's ``advan 11/12`` territory.

    Examples
    --------
    >>> rhs = compartment_rhs("oral_1c", {"ka": 1.2, "ke": 0.2, "v": 30.0, "f": 0.9})
    >>> deriv = rhs(0.0, np.array([100.0, 0.0]), np.zeros(2))
    >>> [round(float(d), 4) for d in deriv]
    [-120.0, 108.0]
    """
    if model not in _MODELS:
        raise DataValidationError(
            f"unknown model {model!r}; choose one of {sorted(_MODELS)}"
        )
    spec = _MODELS[model]
    missing = [name for name in spec["params"] if name not in params]
    if missing:
        raise DataValidationError(f"model {model!r} misses parameters {missing}")

    ka = float(params.get("ka", 0.0))
    f = float(params.get("f", 1.0))
    k10 = float(params.get("ke", params.get("k10", 0.0)))
    k12 = float(params.get("k12", 0.0))
    k21 = float(params.get("k21", 0.0))
    k13 = float(params.get("k13", 0.0))
    k31 = float(params.get("k31", 0.0))

    if model == "iv_1c":

        def rhs(_t: float, y: np.ndarray, rate: np.ndarray) -> np.ndarray:
            return np.array([-k10 * y[0] + rate[0]])

    elif model == "oral_1c":

        def rhs(_t: float, y: np.ndarray, rate: np.ndarray) -> np.ndarray:
            return np.array(
                [-ka * y[0] + rate[0], ka * f * y[0] - k10 * y[1] + rate[1]]
            )

    elif model == "iv_2c":

        def rhs(_t: float, y: np.ndarray, rate: np.ndarray) -> np.ndarray:
            return np.array(
                [-(k10 + k12) * y[0] + k21 * y[1] + rate[0], k12 * y[0] - k21 * y[1]]
            )

    elif model == "oral_2c":

        def rhs(_t: float, y: np.ndarray, rate: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    -ka * y[0] + rate[0],
                    ka * f * y[0] - (k10 + k12) * y[1] + k21 * y[2] + rate[1],
                    k12 * y[1] - k21 * y[2],
                ]
            )

    elif model == "iv_3c":

        def rhs(_t: float, y: np.ndarray, rate: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    -(k10 + k12 + k13) * y[0] + k21 * y[1] + k31 * y[2] + rate[0],
                    k12 * y[0] - k21 * y[1],
                    k13 * y[0] - k31 * y[2],
                ]
            )

    else:  # oral_3c

        def rhs(_t: float, y: np.ndarray, rate: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    -ka * y[0] + rate[0],
                    ka * f * y[0] - (k10 + k12 + k13) * y[1] + k21 * y[2] + k31 * y[3] + rate[1],
                    k12 * y[1] - k21 * y[2],
                    k13 * y[1] - k31 * y[3],
                ]
            )

    return rhs


def simulate_compartment(
    model: str,
    params: dict[str, float],
    events: EventTable,
    t_eval: np.ndarray,
    max_step: float = 0.01,
) -> pd.DataFrame:
    """Simulate one compartment model on an event table; returns a frame.

    The output carries one column per state plus ``conc`` -- the
    central-compartment concentration ``central / V`` (or
    ``central / v1``), the quantity every PK observation models.  IV
    models dose compartment 0; oral models expect doses in the gut
    (compartment 0) with ``f`` scaling the absorbed amount.

    Examples
    --------
    >>> events = EventTable()
    >>> _ = events.add_dosing(0.0, 500.0)       # IV bolus, 500 mg
    >>> times = np.linspace(0.0, 24.0, 13)
    >>> frame = simulate_compartment("iv_1c", {"ke": 0.173, "v": 40.0}, events, times)
    >>> float(frame["conc"].iloc[0].round(3))
    12.5
    >>> round(float(frame["conc"].iloc[4] / frame["conc"].iloc[0]), 4)   # t=8h: e^-1.384
    0.2506
    """
    spec = _MODELS[model]  # type: ignore[index]
    states = tuple(spec["states"])  # type: ignore[index]
    rhs = compartment_rhs(model, params)
    n = len(states)
    y0 = np.zeros(n)
    events = EventTable(
        [e for e in (*events.boluses(), *events.infusions()) if e.time <= t_eval[-1] + 1e-12]
    )
    ys = solve_ode(rhs, y0, np.asarray(t_eval, dtype=float), events, max_step=max_step)
    frame = pd.DataFrame({"time": np.asarray(t_eval, dtype=float)})
    for i, name in enumerate(states):
        frame[name] = ys[:, i]
    volume = float(params.get("v", params.get("v1", 1.0)))
    frame["conc"] = frame["central"] / volume
    return frame


def bateman_superposition(
    times: np.ndarray | float,
    ka: float,
    ke: float,
    v: float,
    f: float,
    dose_times: np.ndarray,
    doses: np.ndarray,
) -> np.ndarray:
    """Closed-form one-compartment oral concentrations (``linCmt``'s sibling).

    The solved-model route nlmixr2 offers as ``linCmt()``: for each
    dose the Bateman term ``F·D/V · ka/(ka−ke) · (e^(−ke·(t−td)) −
    e^(−ka·(t−td)))`` contributes for ``t >= td``; superposing the terms
    gives multiple-dose profiles without touching an ODE solver.  The
    population fitter leans on this for speed; the flip-flop limit
    (``ka == ke``) falls back to ``F·D/V · ka·t·e^(−ka·t)``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0.0, 48.0, 5)
    >>> c = bateman_superposition(t, 1.2, 0.15, 30.0, 1.0, np.array([0.0]), np.array([250.0]))
    >>> bool(c[0] == 0.0)
    True
    >>> # AUC of the closed form equals F*D/CL exactly:
    >>> from pharmaml.pharma.pk import auc_trapezoid
    >>> grid = np.linspace(0.0, 200.0, 4001)
    >>> cg = bateman_superposition(grid, 1.2, 0.15, 30.0, 1.0, np.array([0.0]), np.array([250.0]))
    >>> round(auc_trapezoid(grid, cg), 1)   # truth 250/(0.15*30) = 55.6
    55.6
    """
    times = np.asarray(times, dtype=float)
    dose_times = np.atleast_1d(np.asarray(dose_times, dtype=float))
    doses = np.atleast_1d(np.asarray(doses, dtype=float))
    flat = np.atleast_1d(times)
    out_flat = np.zeros(flat.shape)
    for td, dose in zip(dose_times, doses, strict=True):
        elapsed = flat - td
        active = elapsed >= 0
        if abs(ka - ke) < 1e-12:
            term = np.where(active, dose * f / v * ka * np.maximum(elapsed, 0.0) * np.exp(-ka * elapsed), 0.0)
        else:
            coef = dose * f / v * ka / (ka - ke)
            term = np.where(
                active,
                coef * (np.exp(-ke * elapsed) - np.exp(-ka * elapsed)),
                0.0,
            )
        out_flat += term
    if times.ndim == 0:
        return np.array(float(out_flat[0]))  # type: ignore[return-value]
    return out_flat  # type: ignore[return-value]


# --------------------------------------------------------------------- #
# Terminal phase (pkpdutils' Phoenix-rule lambda-z)
# --------------------------------------------------------------------- #


@dataclass
class TerminalFit:
    """Result of the log-linear terminal-phase regression."""

    slope: float
    intercept: float
    r2: float
    r2_adj: float
    n_points: int
    t_first: float
    t_last: float
    start_index: int

    @property
    def lambda_z(self) -> float:
        """Terminal disposition rate ``lambda_z = -slope``."""
        return -self.slope

    @property
    def half_life(self) -> float:
        """Terminal half-life ``ln2 / lambda_z``."""
        return math.log(2) / self.lambda_z


def fit_lambda_z(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    min_points: int = 3,
    exclude_cmax: bool = True,
    tie_tolerance: float = 1e-4,
) -> TerminalFit:
    """Phoenix ``BEST_FIT`` lambda-z: maximal adjusted R² window after the peak.

    pkpdutils' transcribed rule (itself Phoenix WinNonlin's): candidate
    windows run from a start index to the last *regressable* (positive,
    finite) concentration; eligible windows need ``>= min_points``
    points after ``tmax`` (``exclude_cmax`` drops the peak itself), a
    negative log-slope, and the winner is the largest **adjusted**
    R² ``1 − (1−R²)(n−1)/(n−2)`` -- with ties inside
    ``tie_tolerance`` resolved toward the window with more points,
    then the earliest start (exactly pkpdutils' selection order).

    Examples
    --------
    >>> import numpy as np
    >>> # biexponential 30*e^{-0.8t} + 3*e^{-0.1t}: lambda_z must find 0.1
    >>> t = np.arange(0.0, 25.0, 2.5)
    >>> c = 30.0 * np.exp(-0.8 * t) + 3.0 * np.exp(-0.1 * t)
    >>> fit = fit_lambda_z(t, c)
    >>> round(fit.lambda_z, 3)
    0.101
    >>> fit.n_points >= 3
    True
    >>> round(fit.r2_adj, 4) >= 0.99
    True
    """
    times = np.asarray(times, dtype=float)
    conc = np.asarray(conc, dtype=float)
    if times.size != conc.size or times.size < min_points:
        raise DataValidationError("need at least as many paired points as min_points")
    regressable = np.isfinite(conc) & (conc > 0) & np.isfinite(times)
    if regressable.sum() < min_points:
        raise DataValidationError("fewer than min_points positive concentrations")
    tmax_index = int(np.argmax(np.where(np.isfinite(conc), conc, -np.inf)))
    idx = np.nonzero(regressable)[0]
    if exclude_cmax:
        idx = idx[idx > tmax_index]
    if idx.size < min_points:
        raise DataValidationError("not enough points after tmax for a terminal window")

    ln_c = np.log(np.where(conc > 0, conc, np.nan))
    candidates: list[tuple[int, float, float, float, float, int]] = []
    for start in idx[: idx.size - min_points + 1]:
        window = np.arange(start, idx[-1] + 1)
        window = window[regressable[window]]
        n = int(window.size)
        if n < min_points:
            continue
        x = times[window]
        y = ln_c[window]
        slope, intercept = np.polyfit(x, y, 1)
        if slope >= 0:
            continue
        ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            continue
        r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
        candidates.append((int(start), float(slope), float(intercept), float(r2), float(r2_adj), n))
    if not candidates:
        raise DataValidationError("no window with negative slope and >= min_points points")

    best_r2_adj = max(c[4] for c in candidates)
    eligible = [c for c in candidates if c[4] >= best_r2_adj - tie_tolerance]
    # pkpdutils' tie rule: more points win; ties broken by earliest start
    chosen = max(eligible, key=lambda c: (c[5], -c[0]))
    start, slope, intercept, r2, r2_adj, n = chosen
    window = np.arange(start, idx[-1] + 1)
    window = window[regressable[window]]
    return TerminalFit(
        slope=slope,
        intercept=intercept,
        r2=r2,
        r2_adj=r2_adj,
        n_points=n,
        t_first=float(times[window[0]]),
        t_last=float(times[window[-1]]),
        start_index=start,
    )


# --------------------------------------------------------------------- #
# Full single-dose NCA profile (pkpdutils nca.compute_parameters)
# --------------------------------------------------------------------- #

CDISC_PK_CODES: dict[tuple[str, str], tuple[str, str]] = {
    # (variable, route-prefix or "") -> (PARAMCD, PPTEST name); the
    # CDISC C85839 subset covering this toolkit's NCA output
    ("cmax", ""): ("CMAX", "Cmax"),
    ("tmax", ""): ("TMAX", "Time of CMAX"),
    ("cmin", ""): ("CMIN", "Cmin"),
    ("clast", ""): ("CLST", "Last Nonzero Conc"),
    ("tlast", ""): ("TLST", "Time of Last Nonzero Conc"),
    ("tlag", ""): ("TLAG", "Time Until First Nonzero Conc"),
    ("c0", ""): ("C0", "Initial Conc"),
    ("auc_last", ""): ("AUCLST", "AUC to Last Nonzero Conc"),
    ("auc_all", ""): ("AUCALL", "AUC All"),
    ("auc_inf_obs", ""): ("AUCIFO", "AUC Infinity Obs"),
    ("auc_inf_pred", ""): ("AUCIFP", "AUC Infinity Pred"),
    ("auc_extrap_fraction", ""): ("AUCPEO", "AUC %Extrapolation Obs"),
    ("aumc_last", ""): ("AUMCLST", "AUMC to Last Nonzero Conc"),
    ("aumc_inf", ""): ("AUMCIFO", "AUMC Infinity Obs"),
    ("aumc_extrap_fraction", ""): ("AUMCPEO", "AUMC % Extrapolation Obs"),
    ("lambda_z", ""): ("LAMZ", "Lambda z"),
    ("lambda_z_r2", ""): ("R2", "R Squared"),
    ("lambda_z_r2_adj", ""): ("R2ADJ", "R Squared Adjusted"),
    ("lambda_z_n_points", ""): ("LAMZNPT", "Number of Points for Lambda z"),
    ("lambda_z_span", ""): ("LAMZSPN", "Lambda z Span"),
    ("thalf", ""): ("LAMZHL", "Half-Life Lambda z"),
    ("thalf_eff", ""): ("EFFHL", "Effective Half-Life"),
    ("mrt", "oral"): ("MRTEVIFO", "MRT Extravasc Infinity Obs"),
    ("mrt", "iv_bolus"): ("MRTIBIFO", "MRT IV Bolus Infinity Obs"),
    ("mrt", "iv_infusion"): ("MRTICIFO", "MRT IV Cont Inf Infinity Obs"),
    ("cl", "iv"): ("CLO", "Total CL Obs"),
    ("cl_f", "oral"): ("CLFO", "Total CL Obs by F"),
    ("vz", "iv"): ("VZO", "Vz Obs"),
    ("vz_f", "oral"): ("VZFO", "Vz Obs by F"),
    ("vss", "iv"): ("VSSO", "Vol Dist Steady State Obs"),
    ("f_abs", ""): ("FABS", "Absolute Bioavailability"),
    ("f_rel", ""): ("FREL", "Relative Bioavailability"),
    ("accumulation_ratio", ""): ("AILAMZ", "Accumulation Index using Lambda z"),
}


def _segment_areas(times: np.ndarray, conc: np.ndarray, method: str) -> tuple[float, float]:
    """AUC + AUMC by the linear, log or linear-up/log-down trapezoid.

    pkpdutils' three rules: linear ``Δt(c1+c2)/2`` (moment
    ``Δt(t1c1+t2c2)/2``); log ``Δt(c1−c2)/ln(c1/c2)`` with moment
    ``Δt(t1c1−t2c2)/L + Δt²(c1−c2)/L²`` (exact for a mono-exponential);
    ``linear_log`` (the default, Phoenix's "linear up / log down") uses
    the log rule only on falling segments with positive endpoints.
    """
    dt = np.diff(times)
    c1, c2 = conc[:-1], conc[1:]
    linear_area = dt * (c1 + c2) / 2.0
    linear_moment = dt * (times[:-1] * c1 + times[1:] * c2) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(c1 / np.where(c2 == 0, np.nan, c2))
        log_ok = (c1 > 0) & (c2 > 0) & (c1 != c2)
        log_area = np.where(log_ok, dt * (c1 - c2) / np.where(ratio == 0, np.nan, ratio), 0.0)
        log_moment = np.where(
            log_ok,
            dt * (times[:-1] * c1 - times[1:] * c2) / np.where(ratio == 0, np.nan, ratio)
            + dt**2 * (c1 - c2) / np.where(ratio == 0, np.nan, ratio) ** 2,
            0.0,
        )
    if method == "linear":
        use_log = np.zeros(dt.size, dtype=bool)
    elif method == "log":
        use_log = log_ok
    elif method == "linear_log":
        use_log = log_ok & (c2 < c1)
    else:
        raise DataValidationError("method must be 'linear', 'log' or 'linear_log'")
    area = np.where(use_log, log_area, linear_area)
    moment = np.where(use_log, log_moment, linear_moment)
    return float(np.nansum(area)), float(np.nansum(moment))


def nca_profile(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    dose: float | None = None,
    route: str = "oral",
    infusion_duration: float | None = None,
    method: str = "linear_log",
) -> dict[str, float]:
    """The pkpdutils single-dose NCA parameter set, one dictionary.

    Computes: ``cmax/tmax/clast/tlast/tlag``, the IV-bolus
    log-back-extrapolated ``c0`` (or the inserted zero for
    oral/infusion starts), ``auc_last``/``auc_all``/``auc_inf_obs``/
    ``auc_inf_pred``, the ``aumc`` family, infusion-corrected ``mrt``,
    ``thalf`` (from :func:`fit_lambda_z`) and the effective half-life
    ``ln2·MRT``; with a ``dose``: clearance/volumes as ``cl/vz/vss``
    (IV) or ``cl_f/vz_f`` (oral); plus the acceptance diagnostics
    ``lambda_z_span`` and the AUC extrapolation fraction (flagged
    against the 20 % rule inside the docstring's spirit -- returned,
    not asserted).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    >>> c = 2.0 * np.exp(-0.25 * t) + 0.8 * np.exp(-0.35 * t) * (t > 0)
    >>> prof = nca_profile(t, c, dose=100.0, route="iv_bolus")
    >>> round(prof["lambda_z"], 3)     # mixture of 0.25/0.35 seen as ~0.26
    0.26
    >>> round(prof["thalf"], 2)
    2.67
    >>> round(prof["cl"], 1)           # ~ 100 / 10 (AUC ≈ 9.9-10)
    9.9
    >>> prof["auc_extrap_fraction"] < 0.25
    True
    >>> oral_t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0])
    >>> oral_c = np.array([0.0, 0.9, 1.8, 2.4, 2.0, 1.2, 0.7, 0.22, 0.07])
    >>> op = nca_profile(oral_t, oral_c, dose=100.0, route="oral")
    >>> round(op["tlag"], 2)          # BLQ t=0 sample → lag 0 (pkpdutils rule)
    0.0
    >>> op["cl_f"] > 0 and op["vz_f"] > 0
    True
    """
    times = np.asarray(times, dtype=float)
    conc = np.asarray(conc, dtype=float)
    if times.size != conc.size or times.size < 3:
        raise DataValidationError("need >= 3 paired concentration-time points")
    if np.any(np.diff(times) <= 0):
        raise DataValidationError("times must be strictly increasing")
    route = route.lower()
    if route not in ("oral", "iv_bolus", "iv_infusion"):
        raise DataValidationError("route must be 'oral', 'iv_bolus' or 'iv_infusion'")

    finite = np.isfinite(conc)
    t_v, c_v = times[finite], conc[finite]
    cmax = float(np.max(c_v))
    tmax = float(t_v[int(np.argmax(c_v))])
    positive = c_v > 0
    t_last = float(t_v[positive][-1]) if positive.any() else float("nan")
    c_last = float(c_v[positive][-1]) if positive.any() else float("nan")

    # lag time (oral): last zero sample before first positive
    tlag = float("nan")
    if route == "oral":
        first_pos = int(np.nonzero(positive)[0][0]) if positive.any() else -1
        if first_pos > 0:
            tlag = float(t_v[first_pos - 1])
        elif first_pos == 0:
            tlag = 0.0

    # C0 for IV bolus: log-linear back-extrapolation from the first two points
    c0 = float("nan")
    work_t, work_c = t_v.copy(), c_v.copy()
    if route == "iv_bolus" and t_v.size >= 2 and c_v[0] > 0:
        if c_v[1] < c_v[0] and t_v[1] > t_v[0]:
            slope = math.log(c_v[1] / c_v[0]) / (t_v[1] - t_v[0])
            c0 = c_v[0] * math.exp(-slope * t_v[0])
        else:
            c0 = float(c_v[0])
        if t_v[0] > 0 and c0 > 0:
            work_t = np.insert(work_t, 0, 0.0)
            work_c = np.insert(work_c, 0, c0)
    elif route in ("oral", "iv_infusion") and t_v.size and t_v[0] > 0:
        work_t = np.insert(work_t, 0, 0.0)
        work_c = np.insert(work_c, 0, 0.0)

    last_idx = int(np.nonzero(work_c > 0)[0][-1]) if (work_c > 0).any() else work_c.size - 1
    auc_last, aumc_last = _segment_areas(work_t[: last_idx + 1], work_c[: last_idx + 1], method)
    auc_all, aumc_all = _segment_areas(work_t, work_c, method)

    profile: dict[str, float] = {
        "cmax": cmax,
        "tmax": tmax,
        "clast": c_last,
        "tlast": t_last,
        "tlag": tlag,
        "c0": c0,
        "auc_last": float(auc_last),
        "auc_all": float(auc_all),
        "aumc_last": float(aumc_last),
        "aumc_all": float(aumc_all),
    }

    try:
        fit = fit_lambda_z(t_v, c_v)
    except DataValidationError:
        fit = None
    if fit is not None:
        lz = fit.lambda_z
        clast_pred = math.exp(fit.intercept - lz * t_last)
        auc_inf_obs = auc_last + c_last / lz
        auc_inf_pred = auc_last + clast_pred / lz
        aumc_inf = aumc_last + c_last * t_last / lz + c_last / lz**2
        mrt = aumc_inf / auc_inf_obs
        if route == "iv_infusion" and infusion_duration:
            mrt -= infusion_duration / 2.0
        profile.update(
            {
                "lambda_z": float(lz),
                "lambda_z_intercept": float(fit.intercept),
                "lambda_z_r2": float(fit.r2),
                "lambda_z_r2_adj": float(fit.r2_adj),
                "lambda_z_n_points": float(fit.n_points),
                "lambda_z_t_first": float(fit.t_first),
                "lambda_z_t_last": float(fit.t_last),
                "lambda_z_span": float((fit.t_last - fit.t_first) / fit.half_life),
                "clast_pred": float(clast_pred),
                "thalf": float(fit.half_life),
                "thalf_eff": float(math.log(2) * mrt),
                "auc_inf_obs": float(auc_inf_obs),
                "auc_inf_pred": float(auc_inf_pred),
                "auc_extrap_fraction": float((auc_inf_obs - auc_last) / auc_inf_obs),
                "auc_extrap_fraction_pred": float((auc_inf_pred - auc_last) / auc_inf_pred),
                "aumc_inf": float(aumc_inf),
                "aumc_extrap_fraction": float((aumc_inf - aumc_last) / aumc_inf),
                "mrt": float(mrt),
                "mrt_last": float(aumc_last / auc_last) if auc_last > 0 else float("nan"),
            }
        )
        if dose is not None and dose > 0:
            if route == "oral":
                profile["cl_f"] = float(dose / auc_inf_obs)
                profile["vz_f"] = float(dose / (auc_inf_obs * lz))
                profile["auc_inf_dn"] = float(auc_inf_obs / dose)
                profile["cmax_dn"] = float(cmax / dose)
            else:
                profile["cl"] = float(dose / auc_inf_obs)
                profile["vz"] = float(dose / (auc_inf_obs * lz))
                profile["auc_inf_dn"] = float(auc_inf_obs / dose)
                profile["cmax_dn"] = float(cmax / dose)
                if "mrt" in profile:
                    profile["vss"] = float(profile["cl"] * profile["mrt"])
    return profile


def superposition_profile(
    times: np.ndarray,
    conc: np.ndarray,
    dose_times: np.ndarray,
    amounts: np.ndarray,
    lambda_z: float,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Linear superposition of a single-dose curve onto a regimen.

    pkpdutils' ``superposition``: every dose contributes the observed
    single-dose curve shifted to its dose time and scaled by
    ``amount / amount_single``; past the last observation the curve
    continues as ``clast·e^(−lambda_z·(t−tlast))`` (so a finite
    terminal slope is required).  This is how steady state is previewed
    from one profile without a simulation.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0.0, 12.0, 13)
    >>> c = 8.0 * np.exp(-0.15 * t)
    >>> doses_t = np.arange(0.0, 60.0, 12.0)
    >>> amounts = np.full(5, 1.0)
    >>> ss = superposition_profile(t, c, doses_t, amounts, lambda_z=0.15,
    ...                            grid=np.linspace(0.0, 60.0, 601))
    >>> trough = float(ss.set_index("time")["conc"].loc[59.5])
    >>> exact = 8.0 * math.exp(-1.725) * (1.0 - math.exp(-9.0)) / (1.0 - math.exp(-0.15 * 12))
    >>> abs(trough - exact) < 1e-2   # geometric superposition of the tail
    True
    """
    times = np.asarray(times, dtype=float)
    conc = np.asarray(conc, dtype=float)
    dose_times = np.atleast_1d(np.asarray(dose_times, dtype=float))
    amounts = np.atleast_1d(np.asarray(amounts, dtype=float))
    if amounts.size == 0:
        raise DataValidationError("need at least one dose")
    amount_single = float(amounts[0])
    if grid is None:
        grid = np.arange(dose_times[0], dose_times[-1] + 12.0 / lambda_z, 0.25)
    grid = np.asarray(grid, dtype=float)
    tail = conc[-1] * np.exp(-lambda_z * (grid - times[-1]))
    single = np.interp(grid, times, conc, left=0.0, right=0.0)
    single = np.where(grid > times[-1], tail, single)
    total = np.zeros_like(grid)
    for td, amount in zip(dose_times, amounts, strict=True):
        shift = grid - td
        term = np.where(shift >= 0, np.interp(
            np.maximum(shift, 0.0), grid, single) * (amount / amount_single), 0.0)
        total += term
    return pd.DataFrame({"time": grid, "conc": total})


# --------------------------------------------------------------------- #
# Steady state (mrgsolve / RxODE ``ss`` semantics)
# --------------------------------------------------------------------- #


def steady_state_one_compartment(
    times: np.ndarray,
    ka: float,
    ke: float,
    v: float,
    f: float,
    dose: float,
    ii: float,
) -> np.ndarray:
    """Analytic steady-state profile of periodic oral dosing (``ss=1``).

    The geometric-series route mrgsolve dispatches to closed-form
    models: each exponential of the single-dose Bateman term is scaled
    by its washout factor ``1/(1 − e^(−λ·ii))``, so the steady-state
    trough-to-peak cycle over one interval needs no iteration at all::

        C_ss(t) = F·D·ka/(V(ka−ke)) · [e^(−ke·t)/(1−e^(−ke·ii))
                                     − e^(−ka·t)/(1−e^(−ka·ii))]

    with ``t`` taken modulo the inter-dose interval ``ii``.

    Examples
    --------
    >>> import numpy as np
    >>> ii = 12.0
    >>> t = np.linspace(0.0, ii, 121)
    >>> css = steady_state_one_compartment(t, 1.2, 0.15, 30.0, 1.0, 250.0, ii)
    >>> # analytic ss Cmax = single-dose Cmax scaled by the ke factor:
    >>> single = bateman_superposition(t, 1.2, 0.15, 30.0, 1.0, np.array([0.0]), np.array([250.0]))
    >>> peak = int(np.argmax(single))
    >>> ratio = float(css[peak] / single[peak])
    >>> expected = (1.0 - math.exp(-0.15 * ii)) ** -1 * 0.6   # between the two factors
    >>> ratio < 2.1 and ratio > 1.0            # accumulation in [1/(1-e^-ka·ii), 1/(1-e^-ke·ii)]
    True
    """
    times = np.asarray(times, dtype=float)
    t_mod = np.mod(times, ii)
    scale_ke = 1.0 / (1.0 - math.exp(-ke * ii))
    scale_ka = 1.0 / (1.0 - math.exp(-ka * ii))
    coef = dose * f * ka / (v * (ka - ke))
    return coef * (np.exp(-ke * t_mod) * scale_ke - np.exp(-ka * t_mod) * scale_ka)


def steady_state_iterative(
    model: str,
    params: dict[str, float],
    dose: float,
    ii: float,
    cmt: int = 0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_doses: int = 500,
) -> np.ndarray:
    """Iterative steady state for ODE models (mrgsolve's ``ss=1`` loop).

    Repeatedly dose and advance exactly one inter-dose window until
    *every* end-of-interval state repeats within ``|Δy| < rtol·|y| +
    atol`` -- the mrgsolve/RxODE convergence criterion verbatim (their
    ``ss_n`` is this function's ``max_doses``; a non-convergence
    raises rather than warning).  Bolus dosing only; returns the state
    vector at the end of the steady dosing interval (the trough).

    Examples
    --------
    >>> import numpy as np
    >>> # 1-compartment IV: ss bolus = D/(1-e^{-ke·ii}) exactly
    >>> y_ss = steady_state_iterative("iv_1c", {"ke": 0.2, "v": 25.0}, 100.0, 8.0)
    >>> round(float(y_ss[0]), 4) == round(100.0 / (1.0 - math.exp(-1.6)), 4)
    True
    """
    rhs = compartment_rhs(model, params)
    spec = _MODELS[model]
    n = len(spec["states"])
    y = np.zeros(n)
    grid = np.array([0.0, ii])
    y_end_prev: np.ndarray | None = None
    for _ in range(max_doses):
        y[cmt] += dose  # the ss record: one bolus, then advance one ii
        y_new = solve_ode(rhs, y, grid, events=None, max_step=0.01)[-1]
        if y_end_prev is not None and np.all(
            np.abs(y_new - y_end_prev) <= rtol * np.abs(y_new) + atol
        ):
            y_return = y_new.copy()
            y_return[cmt] += dose  # mrgsolve reports the state AT the ss record
            return y_return
        y_end_prev = y_new.copy()  # y itself is mutated by the next dose
        y = y_new
    raise DataValidationError(
        f"steady state not reached within {max_doses} doses (mrgsolve's ss_n limit)"
    )


# --------------------------------------------------------------------- #
# Macro constants (mrgsolve advan trans / RxODE rxDerived territory)
# --------------------------------------------------------------------- #


def macro_constants(
    k10: float, k12: float, k21: float, k13: float = 0.0, k31: float = 0.0
) -> dict[str, float]:
    """Hybrid (macro) constants from micro constants -- ``rxDerived``-lite.

    Builds the n-compartment transition matrix, takes its eigenvalues
    ``λ`` (sorted descending: α > β > γ), and reports the disposition
    exponentials' coefficients ``A/B/C`` for an IV bolus of dose 1
    (``C(t) = Σ c_i e^{−λ_i t}`` scaled by the central volume), the
    half-lives ``t½α/β/γ`` and ``vss = V1·(1 + k12/k21 + k13/k31)``.
    """
    if k13 and k31:
        matrix = np.array(
            [
                [-(k10 + k12 + k13), k21, k31],
                [k12, -k21, 0.0],
                [k13, 0.0, -k31],
            ]
        )
    else:
        matrix = np.array([[-(k10 + k12), k21], [k12, -k21]])
    eigenvalues = sorted(np.linalg.eigvals(matrix).real, reverse=True)
    lambdas = [-float(value) for value in eigenvalues]
    result: dict[str, float] = {}
    names = ["alpha", "beta", "gamma"]
    half_names = ["thalf_alpha", "thalf_beta", "thalf_gamma"]
    for name, half_name, lam in zip(names, half_names, lambdas, strict=False):
        result[name] = lam
        result[half_name] = math.log(2) / lam if lam > 0 else float("inf")
    # coefficients C(t) = sum_i coeff_i e^{-lam_i t}: solve from as many
    # time samples of the matrix exponential as there are exponentials
    from scipy.linalg import expm

    times = np.array([0.02, 0.5, 5.0, 25.0, 120.0])[: len(lambdas)]
    design = np.stack([np.exp(-lam * times) for lam in lambdas]).T
    response = np.array(
        [float((expm(matrix * t) @ np.eye(matrix.shape[0])[:, 0])[0]) for t in times]
    )
    coefficients = np.linalg.solve(design, response)
    for name, value in zip(["A", "B", "C"], coefficients, strict=False):
        result[name] = float(value)
    result["vss_ratio"] = 1.0 + k12 / k21 + (k13 / k31 if k13 and k31 else 0.0)
    return result


# --------------------------------------------------------------------- #
# Formulation release models (PK-Sim's formulation library)
# --------------------------------------------------------------------- #


def formulation_fraction(
    t: np.ndarray | float,
    kind: str,
    half_life: float | None = None,
    end_time: float | None = None,
    t50: float | None = None,
    shape: float | None = None,
    lag_time: float = 0.0,
    t80: float | None = None,
) -> np.ndarray:
    """Fraction (dose) released for PK-Sim's formulation library.

    ``dissolved`` releases everything at once; ``first_order`` follows
    ``1 − e^(−ln2·t/t½)``; ``zero_order`` is ``min(t/T_end, 1)``;
    ``weibull`` releases ``0`` until the lag then
    ``1 − e^(−(t−lag)^β/τ)`` with ``τ = t50^β/ln2`` (so the fraction is
    exactly 0.5 at ``t50`` after the lag); ``lint80`` is the piecewise
    Langenbucher/Lint80 shape PK-Sim ships for tablets (80% linear to
    ``t80``, a parabola to 1.5·t80, then complete).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0.0, 1.0, 2.0, 4.0])
    >>> formulation_fraction(t, "zero_order", end_time=2.0).round(2).tolist()
    [0.0, 0.5, 1.0, 1.0]
    >>> round(float(formulation_fraction(2.0, "weibull", t50=2.0, shape=1.5)), 4)
    0.5
    >>> round(float(formulation_fraction(0.5, "first_order", half_life=1.0)), 3)
    0.293
    """
    times = np.asarray(t, dtype=float)
    times = np.maximum(times - lag_time, 0.0) * (times > lag_time)
    if kind == "dissolved":
        fraction = np.ones_like(times)
    elif kind == "first_order":
        if not half_life or half_life <= 0:
            raise DataValidationError("first_order needs a positive half_life")
        fraction = 1.0 - np.exp(-math.log(2.0) * times / half_life)
    elif kind == "zero_order":
        if not end_time or end_time <= 0:
            raise DataValidationError("zero_order needs a positive end_time")
        fraction = np.minimum(times / end_time, 1.0)
    elif kind == "weibull":
        if not t50 or t50 <= 0 or not shape or shape <= 0:
            raise DataValidationError("weibull needs positive t50 and shape")
        tau = t50**shape / math.log(2.0)
        fraction = 1.0 - np.exp(-times**shape / tau)
    elif kind == "lint80":
        if not t80 or t80 <= 0:
            raise DataValidationError("lint80 needs a positive t80")
        fraction = np.where(
            times <= t80,
            0.8 * times / t80,
            np.where(
                times < 1.5 * t80,
                1.0 - 0.8 * ((times - 1.5 * t80) / t80) ** 2,
                1.0,
            ),
        )
    else:
        raise DataValidationError(
            "kind must be dissolved, first_order, zero_order, weibull or lint80"
        )
    fraction = np.clip(fraction, 0.0, 1.0)
    if np.ndim(t) == 0:
        return np.array(float(fraction))  # type: ignore[return-value]
    return fraction  # type: ignore[return-value]


def formulation_events(
    dose: float,
    kind: str,
    cmt: int = 0,
    step: float = 0.05,
    start: float = 0.0,
    **formulation_params: float | None,
) -> EventTable:
    """Render a formulation's release curve as a zero-order infusion series.

    PK-Sim applies the *fraction (dose)* table as the gut input; this
    engine discretises the same curve: the release between consecutive
    grid points becomes one short infusion of ``dose·ΔF`` lasting
    ``Δt`` (the table formulation's semantics, applied to every
    formulation type).

    Examples
    --------
    >>> table = formulation_events(100.0, "zero_order", end_time=1.0, step=0.25)
    >>> frame = table.to_frame()
    >>> len(frame)
    4
    >>> frame["AMT"].round(2).tolist()
    [25.0, 25.0, 25.0, 25.0]
    >>> float(frame["RATE"].iloc[0])
    100.0
    """
    resolved: dict[str, float] = {
        key: float(value) for key, value in formulation_params.items() if value is not None
    }
    horizon = float(resolved.get("end_time") or 0.0)
    if kind in ("weibull",):
        horizon = 5.0 * float(resolved.get("t50") or 1.0)
    elif kind == "first_order":
        horizon = 6.0 * float(resolved.get("half_life") or 1.0)
    elif kind == "lint80":
        horizon = 1.6 * float(resolved.get("t80") or 1.0)
    elif kind == "dissolved":
        horizon = step
    if horizon <= 0:
        raise DataValidationError("the formulation needs its characteristic time")
    grid = np.arange(start, start + horizon + step / 2, step)
    fractions = formulation_fraction(grid - start, kind, **resolved)
    table = EventTable()
    for k in range(1, grid.size):
        delta = float(fractions[k] - fractions[k - 1])
        if delta > 1e-12:
            table.add_dosing(
                float(grid[k - 1]), dose * delta, cmt=cmt, duration=float(step)
            )
    if not len(table):
        table.add_dosing(start, dose, cmt=cmt, duration=max(step, horizon))
    return table


# --------------------------------------------------------------------- #
# Sensitivity analysis (OSPSuite-R's runSensitivityAnalysis)
# --------------------------------------------------------------------- #


def sensitivity_analysis(
    simulate: Callable[[dict[str, float]], dict[str, float]],
    parameters: dict[str, float],
    variation_range: float = 0.1,
    number_of_steps: int = 2,
) -> pd.DataFrame:
    """Parameter elasticities by the OSPSuite two-sided perturbation grid.

    Every parameter is evaluated at ``p·(1 ± k·range/steps)`` for
    ``k = 1..number_of_steps`` (the default 0.1/2 grid → 90%, 95%,
    105%, 110% of reference, exactly ``ospsuite-env.R``'s defaults);
    the reported value is the log-log elasticity
    ``Δln(PK)/Δln(p)`` per output of ``simulate`` -- the
    ``PKParameterSensitivity{parameterName, pkParameterName, value}``
    contract.  ``simulate`` receives a parameter dict and returns a
    dict of PK outputs.

    Examples
    --------
    >>> def one_comp(params):
    ...     return {"auc": 100.0 / params["ke"]}
    >>> sa = sensitivity_analysis(one_comp, {"ke": 0.2})
    >>> round(float(sa.loc[sa["parameter"] == "ke", "value"].iloc[0]), 3)
    -1.0
    """
    if not parameters:
        raise DataValidationError("need at least one parameter")
    if variation_range <= 0 or number_of_steps < 1:
        raise DataValidationError("variation_range and number_of_steps must be positive")
    reference = simulate(dict(parameters))
    rows = []
    for name, value in parameters.items():
        for k in range(1, number_of_steps + 1):
            for sign in (-1.0, 1.0):
                factor = 1.0 + sign * k * variation_range / number_of_steps
                perturbed = dict(parameters)
                perturbed[name] = value * factor
                outputs = simulate(perturbed)
                for output, ref_value in reference.items():
                    new_value = outputs[output]
                    if new_value > 0 and ref_value > 0 and factor != 1.0:
                        elasticity = (
                            math.log(new_value / ref_value) / math.log(factor)
                        )
                        rows.append(
                            {
                                "parameter": name,
                                "pk_parameter": output,
                                "variation": factor,
                                "value": float(elasticity),
                            }
                        )
    if not rows:
        raise DataValidationError("no positive outputs to sensitise")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Population NLME (nlmixr2 FOCEI semantics in miniature)
# --------------------------------------------------------------------- #

_ERROR_MODELS = ("add", "prop", "combined1", "combined2")


def residual_sd(
    pred: np.ndarray | float,
    add: float = 0.0,
    prop: float = 0.0,
    model: str = "combined2",
) -> np.ndarray:
    """Residual error SD under nlmixr2's error models.

    ``add(a)`` → ``sd = a``; ``prop(b)`` → ``sd = b·f``;
    ``combined1`` → ``sd = a + b·f`` (the linear parameterisation
    rxode2 calls ``addProp=1``); ``combined2`` → ``sd = sqrt(a² +
    (b·f)²)`` (``addProp=2``, the quadratic form).  The ``i`` of FOCEi
    lives here: the SD is evaluated at the *individual* prediction, so
    the residual variance interacts with the etas.

    Examples
    --------
    >>> float(residual_sd(10.0, add=0.3, prop=0.1, model="combined2"))
    1.044030650891055
    >>> float(residual_sd(10.0, add=0.3, prop=0.1, model="combined1"))
    1.3
    >>> float(residual_sd(10.0, add=0.3))
    0.3
    """
    if model not in _ERROR_MODELS:
        raise DataValidationError(f"model must be one of {_ERROR_MODELS}")
    pred_arr = np.asarray(pred, dtype=float)
    if model == "add":
        sd = np.full(pred_arr.shape, float(add))
    elif model == "prop":
        sd = float(prop) * pred_arr
    elif model == "combined1":
        sd = float(add) + float(prop) * pred_arr
    else:
        sd = np.sqrt(float(add) ** 2 + (float(prop) * pred_arr) ** 2)
    if np.any(sd <= 0):
        raise DataValidationError("residual SD must stay positive")
    if np.ndim(pred) == 0:
        return np.array(float(sd))  # type: ignore[return-value]
    return sd  # type: ignore[return-value]


def _bateman_subject(
    times: np.ndarray,
    theta: dict[str, float],
    dose_times: np.ndarray,
    doses: np.ndarray,
) -> np.ndarray:
    ka = math.exp(theta["log_ka"])
    ke = math.exp(theta["log_ke"])
    v = math.exp(theta["log_v"])
    return bateman_superposition(times, ka, ke, v, 1.0, dose_times, doses)


def simulate_population(
    theta: dict[str, float],
    omega: np.ndarray,
    sigma: dict[str, float],
    n_subjects: int = 12,
    dose: float = 320.0,
    sample_times: np.ndarray | None = None,
    seed: int | None = None,
    error_model: str = "combined2",
) -> pd.DataFrame:
    """Simulate a one-compartment oral population study (``rxSolve`` style).

    Log-normal IIV exactly as the ``ini()`` block declares it
    (``eta ~ 0.1`` is a *variance*): ``ka = exp(log ka + eta_ka)`` etc.,
    etas drawn as ``N(0, Omega)``; observations add residual noise via
    the chosen error model.  The returned frame is the nlmixr2 data
    contract -- ``id, time, dv, amt, evid, cmt`` plus the true
    individual parameters (``ka_i, cl_i, v_i``) and ``ipred`` for
    judging a later fit.

    Examples
    --------
    >>> data = simulate_population(
    ...     {"log_ka": 0.2, "log_ke": -3.0, "log_v": 3.4},
    ...     omega=np.diag([0.09, 0.04, 0.02]),
    ...     sigma={"add": 0.25, "prop": 0.07},
    ...     n_subjects=6, seed=11,
    ... )
    >>> data["id"].nunique()
    6
    >>> set(data.columns) >= {"id", "time", "dv", "amt", "evid", "cmt", "ipred"}
    True
    >>> bool((data["dv"] > 0).all())
    True
    """
    if sample_times is None:
        sample_times = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0])
    omega = np.asarray(omega, dtype=float)
    rng = np.random.default_rng(seed)
    etas = rng.multivariate_normal(np.zeros(omega.shape[0]), omega, size=n_subjects)
    rows = []
    for i in range(n_subjects):
        ka = math.exp(theta["log_ka"] + etas[i, 0])
        ke = math.exp(theta["log_ke"] + etas[i, 1])
        v = math.exp(theta["log_v"] + etas[i, 2])
        ipred = bateman_superposition(sample_times, ka, ke, v, 1.0, np.array([0.0]), np.array([dose]))
        sd = residual_sd(ipred, sigma.get("add", 0.0), sigma.get("prop", 0.0), model=error_model)
        dv = rng.normal(ipred, sd)
        dv = np.maximum(dv, 1e-6)
        for j, t in enumerate(sample_times):
            rows.append(
                {
                    "id": i + 1,
                    "time": float(t),
                    "dv": float(dv[j]),
                    "amt": float(dose) if t == 0.0 else 0.0,
                    "evid": 1 if t == 0.0 else 0,
                    "cmt": 1,
                    "ipred": float(ipred[j]),
                    "ka_i": ka,
                    "cl_i": ke * v,
                    "v_i": v,
                    "eta_ka": float(etas[i, 0]),
                    "eta_ke": float(etas[i, 1]),
                    "eta_v": float(etas[i, 2]),
                }
            )
    return pd.DataFrame(rows)


def _subject_blocks(dataset: pd.DataFrame) -> list[dict[str, np.ndarray]]:
    """Split an nlmixr2-style dataset into per-subject dose/observation blocks."""
    required = {"id", "time", "dv", "amt", "evid"}
    missing = required - set(dataset.columns)
    if missing:
        raise DataValidationError(f"dataset misses columns {sorted(missing)}")
    blocks = []
    for subject, group in dataset.groupby("id", sort=True):
        doses = group[group["evid"] > 0]
        obs = group[(group["evid"] == 0) & np.isfinite(group["dv"].astype(float))]
        if obs.empty:
            continue
        blocks.append(
            {
                "id": subject,
                "time": obs["time"].to_numpy(dtype=float),
                "dv": obs["dv"].to_numpy(dtype=float),
                "dose_times": doses["time"].to_numpy(dtype=float),
                "doses": doses["amt"].to_numpy(dtype=float),
            }
        )
    return blocks


def map_etas(
    block: dict[str, np.ndarray],
    theta: dict[str, float],
    omega: np.ndarray,
    sigma: dict[str, float],
    error_model: str = "combined2",
) -> np.ndarray:
    """MAP etas for one subject -- nlmixr2's ``posthoc`` inner problem.

    Maximises the individual posterior ``p(eta | y, theta)`` by
    minimising ``0.5·Σ((dv−f)/sd)² + Σ log sd + 0.5·eta'Ω⁻¹eta`` with
    L-BFGS-B (the conditional mode of the FOCEI inner problem; with a
    Gaussian residual model this is exact Bayesian feedback).

    Examples
    --------
    >>> data = simulate_population(
    ...     {"log_ka": 0.2, "log_ke": -3.0, "log_v": 3.4},
    ...     omega=np.diag([0.09, 0.04, 0.02]),
    ...     sigma={"add": 0.05, "prop": 0.01}, n_subjects=1, seed=3,
    ... )
    >>> block = _subject_blocks(data)[0]
    >>> eta = map_etas(block, {"log_ka": 0.2, "log_ke": -3.0, "log_v": 3.4},
    ...                np.diag([0.09, 0.04, 0.02]), {"add": 0.05, "prop": 0.01})
    >>> eta.shape
    (3,)
    >>> bool(np.all(np.abs(eta) < 2.0))     # near the true (seeded) etas
    True
    """
    omega = np.asarray(omega, dtype=float)
    omega_inv = np.linalg.inv(omega)
    add, prop = float(sigma.get("add", 0.0)), float(sigma.get("prop", 0.0))

    def negative_log_posterior(eta: np.ndarray) -> float:
        params = {name: float(value) + float(e) for name, value, e in zip(
            ("log_ka", "log_ke", "log_v"),
            (theta["log_ka"], theta["log_ke"], theta["log_v"]),
            eta, strict=True)}
        pred = _bateman_subject(block["time"], params, block["dose_times"], block["doses"])
        sd = residual_sd(pred, add, prop, model=error_model)
        resid = (block["dv"] - pred) / sd
        return float(0.5 * resid @ resid + np.sum(np.log(sd)) + 0.5 * eta @ omega_inv @ eta)

    from scipy.optimize import minimize

    result = minimize(
        negative_log_posterior,
        np.zeros(omega.shape[0]),
        method="L-BFGS-B",
        options={"maxiter": 120, "ftol": 1e-10},
    )
    return np.asarray(result.x, dtype=float)


def _joint_nll(
    eta: np.ndarray,
    block: dict[str, np.ndarray],
    theta: dict[str, float],
    omega_inv: np.ndarray,
    add: float,
    prop: float,
    error_model: str,
) -> float:
    params = {name: float(value) + float(e) for name, value, e in zip(
        ("log_ka", "log_ke", "log_v"),
        (theta["log_ka"], theta["log_ke"], theta["log_v"]),
        eta, strict=True)}
    pred = _bateman_subject(block["time"], params, block["dose_times"], block["doses"])
    sd = residual_sd(pred, add, prop, model=error_model)
    resid = (block["dv"] - pred) / sd
    return float(0.5 * resid @ resid + np.sum(np.log(sd)) + 0.5 * eta @ omega_inv @ eta)


def _eta_hessian(
    eta: np.ndarray,
    block: dict[str, np.ndarray],
    theta: dict[str, float],
    omega_inv: np.ndarray,
    add: float,
    prop: float,
    error_model: str,
    step: float = 1e-3,
) -> np.ndarray:
    """Central-difference Hessian of the joint negative log-likelihood."""
    k = eta.size
    hessian = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = step
            ej[j] = step
            f_pp = _joint_nll(eta + ei + ej, block, theta, omega_inv, add, prop, error_model)
            f_pm = _joint_nll(eta + ei - ej, block, theta, omega_inv, add, prop, error_model)
            f_mp = _joint_nll(eta - ei + ej, block, theta, omega_inv, add, prop, error_model)
            f_mm = _joint_nll(eta - ei - ej, block, theta, omega_inv, add, prop, error_model)
            hessian[i, j] = hessian[j, i] = (f_pp - f_pm - f_mp + f_mm) / (4 * step * step)
    # guard: keep the Laplace |H| finite by flooring tiny eigenvalues
    eigvals = np.linalg.eigvalsh((hessian + hessian.T) / 2)
    if np.any(eigvals <= 1e-8):
        hessian = hessian + np.eye(k) * (1e-8 - float(eigvals.min()) + 1e-8)
    return hessian


def focei_objective(
    dataset: pd.DataFrame,
    theta: dict[str, float],
    omega: np.ndarray,
    sigma: dict[str, float],
    error_model: str = "combined2",
    etas: dict[object, np.ndarray] | None = None,
) -> tuple[float, dict[object, np.ndarray]]:
    """FOCEI objective value: the Laplace −2LL at MAP etas, full constants.

    Per subject: ``objf_i = Σ_j r_j²/sd_j² + 2Σ_j log sd_j + n_i·log
    2π + eta'Ω⁻¹eta + log|Ω| + log|H_i|`` with ``H_i`` the Hessian of
    the joint negative log-likelihood at the conditional mode -- the
    textbook FOCEI (NONMEM's ``f``, the ``i`` being the SD evaluated at
    the individual prediction, i.e. eta-residual interaction).  Etas
    passed in ``etas`` warm-start the inner optimisations (and are
    updated in place for the next outer iteration).

    Examples
    --------
    >>> data = simulate_population(
    ...     {"log_ka": 0.2, "log_ke": -3.0, "log_v": 3.4},
    ...     omega=np.diag([0.09, 0.04, 0.02]),
    ...     sigma={"add": 0.15, "prop": 0.05}, n_subjects=8, seed=5,
    ... )
    >>> truth = {"log_ka": 0.2, "log_ke": -3.0, "log_v": 3.4}
    >>> om = np.diag([0.09, 0.04, 0.02])
    >>> objf, _ = focei_objective(data, truth, om, {"add": 0.15, "prop": 0.05})
    >>> 8 * 11 - 60 < objf < 8 * 11 + 140      # ≈ nobs ± chi2 slack
    True
    """
    omega = np.asarray(omega, dtype=float)
    omega_inv = np.linalg.inv(omega)
    logdet_omega = float(np.linalg.slogdet(omega)[1])
    add, prop = float(sigma.get("add", 0.0)), float(sigma.get("prop", 0.0))
    blocks = _subject_blocks(dataset)
    if etas is None:
        etas = {}
    total = 0.0
    for block in blocks:
        key = block["id"]
        start = etas.get(key)
        if start is None:
            eta = map_etas(block, theta, omega, sigma, error_model=error_model)
        else:
            from scipy.optimize import minimize

            def neg(eta: np.ndarray, _block: dict[str, np.ndarray] = block) -> float:
                return _joint_nll(eta, _block, theta, omega_inv, add, prop, error_model)

            result = minimize(
                neg, np.asarray(start, dtype=float), method="L-BFGS-B",
                options={"maxiter": 40, "ftol": 1e-9},
            )
            eta = np.asarray(result.x, dtype=float)
        etas[key] = eta
        hessian = _eta_hessian(eta, block, theta, omega_inv, add, prop, error_model)
        n_i = block["dv"].size
        joint = _joint_nll(eta, block, theta, omega_inv, add, prop, error_model)
        objf_i = 2 * joint + float(np.linalg.slogdet(hessian)[1]) + logdet_omega
        # 2*joint already contains n_i*log(sqrt(2pi)) terms per obs:
        # 0.5*r² + log sd; the residual normal density contributes
        # n_i*log(2π)/2 which we add once below to reach −2LL constants.
        objf_i += n_i * math.log(2 * math.pi)
        total += objf_i
    return float(total), etas


@dataclass
class PopulationFit:
    """An nlmixr2-style fit report (``$parFixed`` + diagnostics)."""

    theta: dict[str, float]
    omega: np.ndarray
    sigma: dict[str, float]
    objf: float
    aic: float
    bic: float
    n_obs: int
    n_subjects: int
    etas: pd.DataFrame
    shrinkage: dict[str, float]
    bsv_cv: dict[str, float]
    predictions: pd.DataFrame
    history: list[float] = field(default_factory=list)

    @property
    def shrinkage_sd_percent(self) -> dict[str, float]:
        """SD-scale eta shrinkage in percent (the ``Shrink(SD)%`` column)."""
        return {name: 100.0 * value for name, value in self.shrinkage.items()}

    def summary_frame(self) -> pd.DataFrame:
        """The ``parFixedDf`` layout: parameter, estimate, BSV CV%, shrinkage %."""
        rows = []
        for name, value in self.theta.items():
            rows.append({"parameter": name, "estimate": math.exp(value), "type": "theta"})
        for name, value in self.bsv_cv.items():
            rows.append({"parameter": f"eta_{name}", "estimate": value, "type": "bsv_cv_percent"})
        for name, value in self.shrinkage_sd_percent.items():
            rows.append({"parameter": f"eta_{name}", "estimate": value, "type": "shrink_sd_percent"})
        return pd.DataFrame(rows)


def fit_population(
    dataset: pd.DataFrame,
    theta_init: dict[str, float],
    omega_init: np.ndarray,
    sigma_init: dict[str, float],
    error_model: str = "combined2",
    maxiter: int = 240,
) -> PopulationFit:
    """Fit the one-compartment oral population model by FOCEI.

    The outer problem is Nelder-Mead over ``[log θ, log diag(Ω), log
    σ]`` (positive parameters log-transformed, exactly nlmixr2's
    internal unconstrained scale); every objective evaluation runs the
    inner MAP problems warm-started from the previous etas.  Returned
    diagnostics: ``objf`` (−2LL, full constants), ``AIC = objf + 2p``,
    ``BIC = objf + p·ln(nobs)``, SD-scale eta shrinkage
    ``1 − sd(eta_i)/sqrt(ω²)`` and BSV CV % ``sqrt(exp(ω²) − 1)``.

    Examples
    --------
    >>> data = simulate_population(
    ...     {"log_ka": 0.25, "log_ke": -3.1, "log_v": 3.4},
    ...     omega=np.diag([0.09, 0.04, 0.02]),
    ...     sigma={"add": 0.2, "prop": 0.06}, n_subjects=10, seed=21,
    ... )
    >>> fit = fit_population(
    ...     data,
    ...     theta_init={"log_ka": 0.0, "log_ke": -2.8, "log_v": 3.2},
    ...     omega_init=np.diag([0.05, 0.02, 0.01]),
    ...     sigma_init={"add": 0.3, "prop": 0.1},
    ...     maxiter=120,
    ... )
    >>> 0.0 < math.exp(fit.theta["log_ke"]) < 0.1      # ke recovers ~0.045
    True
    >>> abs(math.exp(fit.theta["log_v"]) - 30.0) < 12  # v recovers ~30 L
    True
    >>> fit.objf < 400
    True
    >>> set(fit.shrinkage) == {"ka", "ke", "v"}
    True
    """
    theta_init = dict(theta_init)
    omega_init = np.asarray(omega_init, dtype=float).copy()
    blocks = _subject_blocks(dataset)
    n_obs = sum(block["dv"].size for block in blocks)
    n_subjects = len(blocks)
    eta_names = ("ka", "ke", "v")
    theta_names = ("log_ka", "log_ke", "log_v")

    def unpack(x: np.ndarray) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
        theta = {name: float(value) for name, value in zip(theta_names, x[:3], strict=True)}
        omega = np.diag(np.exp(x[3:6]))
        sigma = {"add": math.exp(float(x[6])), "prop": math.exp(float(x[7]))}
        return theta, omega, sigma

    from scipy.optimize import minimize

    state: dict[object, np.ndarray] = {}
    history: list[float] = []

    def objective(x: np.ndarray) -> float:
        theta, omega, sigma = unpack(x)
        try:
            objf, state_out = focei_objective(
                dataset, theta, omega, sigma, error_model=error_model, etas=state
            )
            state.update(state_out)
        except (np.linalg.LinAlgError, DataValidationError, ValueError):
            return 1e12
        history.append(objf)
        return objf

    x0 = np.concatenate(
        [
            [theta_init[name] for name in theta_names],
            np.log(np.diag(omega_init)),
            [
                math.log(max(float(sigma_init.get("add", 0.05)), 1e-4)),
                math.log(max(float(sigma_init.get("prop", 0.05)), 1e-4)),
            ],
        ]
    )
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4})
    theta, omega, sigma = unpack(np.asarray(result.x, dtype=float))
    objf, etas = focei_objective(dataset, theta, omega, sigma, error_model=error_model)

    rows = []
    pred_rows = []
    for block in blocks:
        eta = etas[block["id"]]
        rows.append({"id": block["id"], **{f"eta_{name}": float(e) for name, e in zip(eta_names, eta, strict=True)}})
        params = {name: float(value) + float(e) for name, value, e in zip(
            theta_names, (theta["log_ka"], theta["log_ke"], theta["log_v"]), eta, strict=True)}
        ipred = _bateman_subject(block["time"], params, block["dose_times"], block["doses"])
        pop_params = dict(theta)
        pred = _bateman_subject(block["time"], pop_params, block["dose_times"], block["doses"])
        for t, dv, ip, pr in zip(block["time"], block["dv"], ipred, pred, strict=True):
            pred_rows.append(
                {"id": block["id"], "time": float(t), "dv": float(dv),
                 "ipred": float(ip), "pred": float(pr)}
            )
    eta_frame = pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()
    shrinkage: dict[str, float] = {}
    bsv_cv: dict[str, float] = {}
    for i, name in enumerate(eta_names):
        omega_sd = math.sqrt(float(omega[i, i]))
        if len(eta_frame) >= 2 and omega_sd > 0:
            sd_eta = float(eta_frame[f"eta_{name}"].std(ddof=1))
            shrinkage[name] = max(0.0, 1.0 - sd_eta / omega_sd)
        else:
            shrinkage[name] = float("nan")
        bsv_cv[name] = math.sqrt(math.expm1(float(omega[i, i]))) * 100.0

    n_params = 3 + 3 + 2
    return PopulationFit(
        theta=theta,
        omega=omega,
        sigma=sigma,
        objf=float(objf),
        aic=float(objf + 2 * n_params),
        bic=float(objf + n_params * math.log(n_obs)),
        n_obs=n_obs,
        n_subjects=n_subjects,
        etas=eta_frame,
        shrinkage=shrinkage,
        bsv_cv=bsv_cv,
        predictions=pd.DataFrame(pred_rows),
        history=history,
    )
