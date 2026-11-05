"""Dosing-event semantics: EVID records, addl/until expansion, infusions, steady state.

This module ports the *data semantics* of the rxode2/mrgsolve simulation
stack to the toolkit: the EVID event-identifier codes shared with NONMEM,
the dosing-record dataclass (AMT/CMT/RATE/II/ADDL/SS/UNTIL), the
``addl``/``until`` expansion rules with both engines' boundary
conventions, the F/ALAG/R/D bioavailability ledger for infusion records,
and mrgsolve's ``steady_bolus`` fixed-point iteration for periodic steady
state.  Three engine-adjacent data tables round it out -- the published
solver defaults of rxode2/mrgsolve/OSPSuite's CVODES, the 22 OSPSuite
PK-analysis parameter codes, and the PK-Sim GFR-maturation constants --
together with the OSPSuite-R PK-sensitivity protocol and the Du Bois
body-surface-area formula.

Boundary
--------
rxode2 (R), mrgsolve (R) and OSPSuite's CVODES engine are **not**
vendored or executed here; only their semantics and published constants
are encoded, in pure Python/numpy.  The toolkit's own ODE integration is
fixed-step classical RK4 (the same scheme as
:mod:`pharmaml.pharma.pkpd`'s ``rk4_step``), not an adaptive
LSODA/CVODES solver, and ``SOLVER_DEFAULTS`` is data for parity
comparison, never a promise that those engines run.

References
----------
.. [1] Fidler, M., et al. "rxode2: Facilities for Simulating from
       ODE-Based Models" (R package, GPL).
.. [2] Metrum Research Group. "mrgsolve: Simulate from ODE-Based
       Models" (R package, GPL).
.. [3] Open Systems Pharmacology. "PK-Sim / OSPSuite PKAnalysis
       parameter codes, sensitivity-analysis protocol and PKSimDB
       ontogeny constants" (open-source suite).
.. [4] Du Bois, D. & Du Bois, E. F. "A formula to estimate the
       approximate surface area if height and weight be known."
       Arch. Intern. Med. 17 (1916): 863-871.

Examples
--------
>>> from pharmaml.pharma.events import EVID, EventRecord, expand_events
>>> int(EVID.DOSE), int(EVID.INFUSION_OFF)
(1, 9)
>>> [r.time for r in expand_events([EventRecord(time=0.0, amt=100.0, ii=24.0, addl=2)])]
[0.0, 24.0, 48.0]
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Protocol

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_positive

__all__ = [
    # constants (SCREAMING_CASE first, isort-style), then the class, then functions
    "EVID",
    "GFR_MATURATION",
    "OSP_PK_PARAMETERS",
    "SOLVER_DEFAULTS",
    "EventRecord",
    "dubois_bsa",
    "expand_events",
    "gfr_maturation",
    "infusion_windows",
    "pk_sensitivity",
    "steady_state_amounts",
]


class EVID(IntEnum):
    """Event-identifier (EVID) codes for dosing/observation data records.

    The plain 0-4 codes are the classic NONMEM set shared by mrgsolve and
    rxode2; 8 and 9 are the mrgsolve/rxode2 extension codes for infusion
    bookkeeping.  rxode2's own table additionally defines 5 (reset +
    infusion-on), 6 (infusion-on superimposed on the running infusion) and
    7 (infusion-off); this enum keeps 5 (as ``INFUSION_ON``) and 8/9 and
    deliberately omits 6/7 -- call :func:`EVID` with those values and it
    refuses them.  rxode2 also parses a *classic compound* EVID that packs
    the compartment, an infusion flag and a steady-state flag into one
    integer (``100*cmt + 10000*infusion_flag + ss_flag``); this module
    keeps the plain codes only and carries compartment / rate / ss as
    separate :class:`EventRecord` fields instead.

    Examples
    --------
    >>> [int(code) for code in EVID]
    [0, 1, 2, 3, 4, 5, 8, 9]
    >>> EVID.DOSE == 1 and EVID.REPLACE == 8
    True
    >>> EVID(6)  # rxode2's superposition-infusion code is NOT vendored
    Traceback (most recent call last):
    ...
    ValueError: 6 is not a valid EVID
    """

    OBSERVATION = 0
    DOSE = 1
    OTHER = 2
    RESET = 3
    DOSE_RESET = 4
    INFUSION_ON = 5
    REPLACE = 8
    INFUSION_OFF = 9


@dataclass(frozen=True)
class EventRecord:
    """One row of a NONMEM-style dosing/observation data set.

    Field semantics follow the rxode2/mrgsolve event vocabulary:

    - ``time``: record time (same units as ``ii``/``rate``).
    - ``amt``: dose amount (0 for observations).
    - ``evid``: event identifier; a plain int so rxode2's classic compound
      encodings stay representable (see :class:`EVID` for the plain codes).
    - ``cmt``: compartment number (1-based, as in NONMEM data sets).
    - ``rate``: infusion rate; 0 = bolus, > 0 = data-fixed rate,
      -1 = rate modeled in the ODE system, -2 = duration modeled instead.
    - ``ii``: dosing interval for repeated records.
    - ``addl``: number of additional doses at ``time + k*ii``.
    - ``ss``: 0 = off, 1 = steady state (replace), 2 = steady state with
      superposition onto the incoming state (mrgsolve's ss=2).
    - ``until``: expand ``addl`` until this time (the rxode2/mrgsolve
      convenience bound; see :func:`expand_events` for the boundary rule).

    Examples
    --------
    >>> EventRecord(time=0.0, amt=100.0)
    EventRecord(time=0.0, amt=100.0, evid=1, cmt=1, rate=0.0, ii=0.0, addl=0, ss=0, until=None)
    >>> EventRecord(time=0.0, amt=100.0, ii=24.0, addl=2).addl
    2
    >>> EventRecord(time=0.0, amt=100.0, addl=2, ii=0.0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: ii must be > 0 when addl > 0 or ss > 0 (repeated dosing needs an interval), got ii=0.0
    >>> EventRecord(time=0.0, amt=100.0, ss=3, ii=24.0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: ss must be 0 (off), 1 (steady state) or 2 (steady state with superposition), got 3
    >>> EventRecord(time=0.0, amt=100.0, rate=-3.0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: rate must be >= 0 (data-fixed infusion rate) or one of -1/-2 (modeled rate/duration), got -3.0
    """

    time: float
    amt: float = 0.0
    evid: int = 1
    cmt: int = 1
    rate: float = 0.0
    ii: float = 0.0
    addl: int = 0
    ss: int = 0
    until: float | None = None

    def __post_init__(self) -> None:
        for field in ("time", "amt", "ii", "rate"):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise DataValidationError(
                    f"{field} must be a finite number (NaN/inf rejected), got {value!r}"
                )
        if self.until is not None and not math.isfinite(self.until):
            raise DataValidationError(
                f"until must be a finite number (NaN/inf rejected), got {self.until!r}"
            )
        if self.ss not in (0, 1, 2):
            raise DataValidationError(
                "ss must be 0 (off), 1 (steady state) or 2 (steady state with "
                f"superposition), got {self.ss!r}"
            )
        if not (self.rate >= 0.0 or self.rate in (-1.0, -2.0)):
            raise DataValidationError(
                "rate must be >= 0 (data-fixed infusion rate) or one of -1/-2 "
                f"(modeled rate/duration), got {self.rate!r}"
            )
        if (self.addl > 0 or self.ss > 0) and self.ii <= 0.0:
            raise DataValidationError(
                "ii must be > 0 when addl > 0 or ss > 0 (repeated dosing needs "
                f"an interval), got ii={self.ii!r}"
            )


def _addl_from_until(time: float, ii: float, until: float, inclusive: bool) -> int:
    """Number of additional doses implied by an ``until`` bound.

    Inclusive (rxode2): the last additional dose may land exactly ON
    ``until`` -- ``addl = ceil((until - time - ii) / ii)``, plus one when
    the ratio is an exact integer.

    Exclusive (mrgsolve): the last additional dose falls strictly BEFORE
    ``until`` -- ``addl = ceil((until - time) / ii) - 1``.

    Both results are clamped at zero (an ``until`` before the next dose
    simply yields no additional rows; the seed record itself is always
    kept).
    """
    if inclusive:
        ratio = (until - time - ii) / ii
        n_extra = math.ceil(ratio)
        if ratio == math.floor(ratio):
            n_extra += 1
    else:
        n_extra = math.ceil((until - time) / ii) - 1
    return max(0, n_extra)


def expand_events(
    records: Sequence[EventRecord],
    until_inclusive: bool = True,
) -> list[EventRecord]:
    """Materialize ``addl``/``until`` rows into an explicit event list.

    Each record with ``addl > 0`` (or an ``until`` bound) is expanded into
    the seed row plus one row per additional dose at ``time + k*ii``.
    When ``until`` is set, the additional-dose count follows the ``until``
    rule and the explicit ``addl`` field is ignored:

    - ``until_inclusive=True`` (rxode2): the last dose lands ON ``until``
      (``addl = ceil((until - time - ii)/ii)``, +1 when exact);
    - ``until_inclusive=False`` (mrgsolve): the last dose lands strictly
      BEFORE ``until`` (``addl = ceil((until - time)/ii) - 1``).

    The output is fully materialized: every row carries ``addl = 0`` and
    ``until = None`` (all other fields are copied verbatim, ``time``
    shifted), so re-running :func:`expand_events` on its own output is a
    no-op.  Rows are returned in a stable time sort -- ties keep their
    input order.

    Parameters
    ----------
    records:
        Event records (each validated by :class:`EventRecord`).
    until_inclusive:
        Selects the ``until`` boundary convention (see above).

    Returns
    -------
    list[EventRecord]
        The expanded, time-sorted event list.

    Raises
    ------
    DataValidationError
        If a record is not an :class:`EventRecord`, or carries ``until``
        without a positive ``ii``.

    Examples
    --------
    >>> from pharmaml.pharma.events import EventRecord, expand_events
    >>> seed = EventRecord(time=0.0, amt=10.0, ii=28.0, until=112.0)
    >>> [r.time for r in expand_events([seed])]  # rxode2: last dose ON until
    [0.0, 28.0, 56.0, 84.0, 112.0]
    >>> [r.time for r in expand_events([seed], until_inclusive=False)]  # mrgsolve
    [0.0, 28.0, 56.0, 84.0]
    >>> rows = expand_events([EventRecord(time=0.0, amt=100.0, ii=24.0, addl=2)])
    >>> [r.time for r in rows]
    [0.0, 24.0, 48.0]
    >>> rows == expand_events(rows)  # materialized output is idempotent
    True
    >>> expand_events([EventRecord(time=0.0, amt=10.0, ii=0.0, until=112.0)])
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: until requires ii > 0 (the expansion interval)
    """
    expanded: list[EventRecord] = []
    for record in records:
        if not isinstance(record, EventRecord):
            raise DataValidationError(
                "records must be EventRecord instances, got "
                f"{type(record).__name__!r}"
            )
        if record.until is not None:
            if record.ii <= 0.0:
                raise DataValidationError(
                    "until requires ii > 0 (the expansion interval)"
                )
            n_extra = _addl_from_until(
                record.time, record.ii, record.until, until_inclusive
            )
        else:
            n_extra = int(record.addl)
        expanded.append(replace(record, addl=0, until=None))
        for k in range(1, n_extra + 1):
            expanded.append(replace(record, time=record.time + k * record.ii, addl=0, until=None))
    expanded.sort(key=lambda record: record.time)
    return expanded


def infusion_windows(
    records: Sequence[EventRecord],
    f: float = 1.0,
) -> list[tuple[float, float, int, float]]:
    """Zero-order infusion windows ``(start, end, cmt, rate)`` for rate > 0 records.

    For every record with a data-fixed infusion rate (``rate > 0``) the
    delivered amount is ``amt * f`` at exactly ``rate``, so the window
    lasts ``duration = amt * f / rate`` (mrgsolve's ``dur() = Fn * Amt /
    Rate``).  Records with ``rate == 0`` (boluses), observations and the
    modeled-rate/duration codes (-1/-2) produce no windows -- those need
    the ODE model in the loop.

    The full F/ALAG/R/D ledger mrgsolve tracks per dosing record, of
    which this function computes the data-fixed-rate branch:

    - ``F`` (bioavailability) scales the *delivered* amount: only
      ``F * amt`` reaches the compartment;
    - data-fixed rate (``rate > 0``): the rate is honoured exactly, so a
      smaller F *extends/shortens the duration* -- ``dur = F*amt/rate``;
    - modeled duration (``rate = -2``, duration D from the model): the
      effective rate becomes ``F*amt/D``;
    - modeled rate (``rate = -1``, rate R from the model): F shortens the
      duration to ``F*amt/R``;
    - ``ALAG`` (absorption lag): the dose spawns a lagged copy of the
      record at ``time + ALAG`` in the dosing compartment.

    Parameters
    ----------
    records:
        Event records (rate <= 0 rows are skipped).
    f:
        Bioavailability fraction applied to the delivered amount
        (0 < f; f = 1 for full availability).

    Returns
    -------
    list of tuple[float, float, int, float]
        ``(start, start + duration, cmt, rate)`` per infusion record.

    Examples
    --------
    >>> from pharmaml.pharma.events import EventRecord, infusion_windows
    >>> infusion_windows([EventRecord(time=0.0, amt=100.0, rate=25.0)])
    [(0.0, 4.0, 1, 25.0)]
    >>> infusion_windows([EventRecord(time=0.0, amt=100.0, rate=25.0)], f=0.5)
    [(0.0, 2.0, 1, 25.0)]
    >>> infusion_windows([EventRecord(time=0.0, amt=100.0, rate=0.0)])  # bolus
    []
    """
    validate_positive(f, "f")
    windows: list[tuple[float, float, int, float]] = []
    for record in records:
        if record.rate > 0.0:
            duration = record.amt * f / record.rate
            windows.append(
                (float(record.time), float(record.time + duration), int(record.cmt), float(record.rate))
            )
    return windows


class _ODEModel(Protocol):
    """Structural type: any ODE model exposing ``ode(state, time)``.

    :class:`pharmaml.pharma.pkpd.CompartmentModel` is the canonical
    implementation; the ``solve_ivp`` argument order ``ode(time, state)``
    and the time-free ``ode(state)`` form are detected and adapted at
    runtime (see :func:`steady_state_amounts`).
    """

    def ode(self, y: np.ndarray, t: float) -> np.ndarray: ...


def _wrap_state_first(
    ode_raw: Callable[..., np.ndarray],
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Adapter for ``ode(state, time)`` models (the pkpd / odeint order)."""

    def ode(state: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(ode_raw(state, t), dtype=float)

    return ode


def _wrap_time_first(
    ode_raw: Callable[..., np.ndarray],
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Adapter for ``ode(time, state)`` models (the solve_ivp order)."""

    def ode(state: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(ode_raw(t, state), dtype=float)

    return ode


def _wrap_time_free(
    ode_raw: Callable[..., np.ndarray],
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Adapter for time-invariant ``ode(state)`` models."""

    def ode(state: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(ode_raw(state), dtype=float)

    return ode


def _model_ode(model: _ODEModel) -> tuple[Callable[[np.ndarray, float], np.ndarray], int]:
    """Normalize ``model.ode`` to ``ode(state, time)`` and find the state size.

    Compartment models in the wild follow three conventions -- scipy's
    ``odeint`` order ``ode(y, t)``, ``solve_ivp`` order ``ode(t, y)`` and
    the time-free ``ode(y)`` of linear compartment systems.  Each adapter
    is probed with state vectors of length 1-4 and accepted only when it
    returns a derivative vector of the *same* length, which pins both the
    argument order and the number of compartments.
    """
    ode_attr = getattr(model, "ode", None)
    if not callable(ode_attr):
        raise DataValidationError(
            "model must expose a callable .ode(state, time) method "
            "(pharmaml.pharma.pkpd.CompartmentModel is the canonical input)"
        )
    ode_raw: Callable[..., np.ndarray] = ode_attr
    for wrapper in (_wrap_state_first, _wrap_time_first, _wrap_time_free):
        for n_state in (1, 2, 3, 4):
            try:
                derivative = wrapper(ode_raw)(np.zeros(n_state), 0.0)
            except (TypeError, ValueError, IndexError):
                continue
            if derivative.shape == (n_state,):
                return wrapper(ode_raw), n_state
    raise DataValidationError(
        "could not call model.ode with a state vector of length 1-4 in any "
        "argument order (ode(state, time) / ode(time, state) / ode(state)); "
        "the model must return a derivative vector matching the state length"
    )


def _rk4_advance(
    ode: Callable[[np.ndarray, float], np.ndarray],
    state: np.ndarray,
    t0: float,
    duration: float,
    steps: int,
) -> np.ndarray:
    """Classical fourth-order Runge-Kutta over ``steps`` equal substeps.

    Mirrors the scheme of ``pharmaml.pharma.pkpd.rk4_step``; kept local so
    the events layer carries no import-order dependency on the simulation
    module (identical classical-RK4 mathematics).
    """
    h = duration / steps
    y = np.array(state, dtype=float)
    for i in range(steps):
        t = t0 + i * h
        k1 = ode(y, t)
        k2 = ode(y + 0.5 * h * k1, t + 0.5 * h)
        k3 = ode(y + 0.5 * h * k2, t + 0.5 * h)
        k4 = ode(y + h * k3, t + h)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


_SS_SUBSTEPS = 200  # internal RK4 substeps per dosing interval (ii / 200)


def steady_state_amounts(
    model: _ODEModel,
    dose: float,
    ii: float,
    mode: str = "replace",
    ss_rtol: float = 1e-8,
    ss_atol: float = 1e-8,
    ss_n: int = 500,
    initial: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Periodic steady-state amounts for repeated bolus dosing (mrgsolve steady_bolus).

    Implements mrgsolve's ``steady_bolus`` fixed-point iteration on any
    ODE compartment model (the canonical input is
    :class:`pharmaml.pharma.pkpd.CompartmentModel`; any object with a
    callable ``.ode`` works -- ``ode(state, time)``, the ``solve_ivp``
    order ``ode(time, state)`` and the time-free ``ode(state)`` are all
    detected).  Starting from the zero state, one iteration doses
    ``dose`` into **compartment 0** (the depot for oral ``ka > 0``
    models -- for one-compartment/IV models compartment 0 is the central
    state) and advances one dosing interval ``ii`` with classical RK4 at
    a fine internal step (``ii / 200``, mirroring
    ``pharmaml.pharma.pkpd.rk4_step``'s scheme).  Convergence uses
    mrgsolve's mixed tolerance: all ``|y_new - y| < ss_rtol*|y| +
    ss_atol``; the returned state is the pre-dose (trough) steady state.

    Modes (the NONMEM ``ss`` flag):

    - ``mode="replace"`` (ss = 1): the steady state *replaces* whatever
      state the system started in -- the iteration starts from zero;
    - ``mode="add"`` (ss = 2, superposition): the periodic steady state
      is solved from zero and the incoming state ``initial`` is added on
      top (``initial = None`` adds nothing).

    If ``ss_n`` iterations pass without meeting the tolerance, the last
    iterate is returned and a ``RuntimeWarning`` is emitted (mrgsolve
    likewise warns rather than fails).

    Parameters
    ----------
    model:
        ODE compartment model with a callable ``.ode``.
    dose:
        Bolus dose amount (> 0) applied every interval.
    ii:
        Dosing interval (> 0).
    mode:
        ``"replace"`` (ss=1) or ``"add"`` (ss=2 superposition).
    ss_rtol, ss_atol:
        Relative / absolute convergence tolerances.
    ss_n:
        Maximum number of dose-and-advance iterations.
    initial:
        Incoming state for ``mode="add"`` (length must equal the model's
        compartment count); unused with ``"replace"``.

    Returns
    -------
    numpy.ndarray
        The steady-state amount vector just before the next dose.

    Examples
    --------
    >>> import math
    >>> from pharmaml.pharma.events import steady_state_amounts
    >>> try:  # canonical input (sibling ODE-simulation module)
    ...     from pharmaml.pharma.pkpd import CompartmentModel
    ...     model = CompartmentModel(cl=1.0, vd=10.0, ka=10.0)
    ... except ImportError:  # stand-in with identical depot -> central ODEs
    ...     class _OralTwoCmt:
    ...         def __init__(self, cl, vd, ka):
    ...             self.cl, self.vd, self.ka = cl, vd, ka
    ...         def ode(self, y, t):
    ...             ka, ke = self.ka, self.cl / self.vd
    ...             return np.array([-ka * y[0], ka * y[0] - ke * y[1]])
    ...     model = _OralTwoCmt(cl=1.0, vd=10.0, ka=10.0)
    >>> y_ss = steady_state_amounts(model, dose=10.0, ii=1.0)
    >>> analytic = 10.0 * math.exp(-0.1) / (1.0 - math.exp(-0.1))  # ke = cl/vd
    >>> abs(float(y_ss[1]) - analytic) / analytic < 0.02  # ~1% absorption lag at ka=10
    True
    >>> exact = 100.0 * (math.exp(-0.1) - math.exp(-10.0)) / (
    ...     9.9 * (1.0 - math.exp(-10.0)) * (1.0 - math.exp(-0.1)))
    >>> abs(float(y_ss[1]) - exact) / exact < 1e-4  # matches the closed form
    True
    >>> y_add = steady_state_amounts(model, dose=10.0, ii=1.0, mode="add", initial=[5.0, 20.0])
    >>> round(float(y_add[1]) - float(y_ss[1]), 6)  # ss=2: incoming state superposed
    20.0
    """
    validate_positive(dose, "dose")
    validate_positive(ii, "ii")
    validate_positive(ss_rtol, "ss_rtol")
    validate_positive(ss_atol, "ss_atol")
    if mode not in ("replace", "add"):
        raise DataValidationError(
            f"mode must be 'replace' (ss=1) or 'add' (ss=2 superposition), got {mode!r}"
        )
    if ss_n < 1:
        raise DataValidationError(f"ss_n must be >= 1, got {ss_n!r}")
    if initial is not None and mode != "add":
        raise DataValidationError(
            "initial is only used with mode='add' (the ss=2 superposition)"
        )
    ode, n_state = _model_ode(model)

    dose_vector = np.zeros(n_state)
    dose_vector[0] = dose  # compartment 0: the depot for oral (ka > 0) models
    state = np.zeros(n_state)
    for _ in range(ss_n):
        advanced = _rk4_advance(ode, state + dose_vector, 0.0, ii, _SS_SUBSTEPS)
        converged = bool(
            np.all(np.abs(advanced - state) < ss_rtol * np.abs(state) + ss_atol)
        )
        state = advanced
        if converged:
            break
    else:
        warnings.warn(
            f"steady_state_amounts did not converge within ss_n={ss_n} iterations; "
            "returning the last iterate",
            RuntimeWarning,
            stacklevel=2,
        )
    if mode == "add":
        incoming = (
            np.zeros(n_state) if initial is None else np.asarray(initial, dtype=float)
        )
        if incoming.shape != (n_state,):
            raise DataValidationError(
                f"initial must have length {n_state} (the model's compartment "
                f"count), got shape {incoming.shape}"
            )
        if not np.isfinite(incoming).all():
            raise DataValidationError("initial must contain finite numbers (NaN/inf rejected)")
        state = state + incoming
    return state


GFR_MATURATION: dict[str, float | dict[str, float]] = {
    # PKSimDB's standard GFR-maturation building block (exact constants).
    "TM50_WEEKS": 44.4,  # post-menstrual age at half-maturation (weeks)
    "TM50_SD": 1.04,
    "HILL": 15.0,  # maturation Hill coefficient
    "HILL_SD": 0.257,
    "F_GFR_PREMAT": 0.258,  # pre-maturation fraction of the specific GFR
    "GFR_MAT_L_MIN": 0.11704,  # matured GFR of the standard kidney (L/min)
    "GFR_MAT_SD": 0.026708,
    "V_KID_STD_L": 0.44,  # standard kidney volume (L)
    "GFR_SPECIES_L_MIN_PER_L": {
        "human": 0.254,
        "mouse": 0.824,
        "monkey": 0.58,
        "rabbit": 0.6,
        "rat": 0.57,
        "dog": 0.254,
    },
    "GFR_AGING_HILL": 1.5,  # Hill coefficient of the post-maturation aging decline
}


def _gfr_scalar(key: str) -> float:
    """Fetch a scalar constant from :data:`GFR_MATURATION` (tables rejected)."""
    value = GFR_MATURATION[key]
    if isinstance(value, dict):
        raise DataValidationError(f"{key} is a species table, not a scalar constant")
    return float(value)


def gfr_maturation(pma_weeks: float, weight_kidney_g: float | None = None) -> float:
    """Glomerular filtration rate (L/min) from post-menstrual age (PK-Sim model).

    The PK-Sim GFR-maturation ontogeny: a Hill sigmoid in post-menstrual
    age (PMA) scales the specific GFR (L/min per L kidney) between the
    pre-maturation floor ``F_GFR_PREMAT`` and the matured standard-kidney
    GFR ``GFR_MAT_L_MIN / V_KID_STD_L``::

        hill   = pma^HILL / (TM50^HILL + pma^HILL)
        spec   = (hill * (1 - F_GFR_PREMAT) + F_GFR_PREMAT) * GFR_MAT / V_KID_STD

    When a kidney weight is supplied the specific GFR is scaled by
    ``weight_kidney_g / 1000`` (grams -> L, kidney density ~1) to give an
    absolute L/min value.

    Parameters
    ----------
    pma_weeks:
        Post-menstrual age in weeks (> 0).
    weight_kidney_g:
        Optional kidney weight in grams (> 0); without it the *specific*
        GFR (per L kidney) is returned.

    Returns
    -------
    float
        GFR in L/min (specific, per L kidney, when no weight is given).

    Examples
    --------
    >>> from pharmaml.pharma.events import gfr_maturation
    >>> round(gfr_maturation(1000.0), 4)  # mature: (0.742 + 0.258) * 0.11704 / 0.44
    0.266
    >>> round(gfr_maturation(0.01), 4)  # preterm floor: 0.258 * 0.11704 / 0.44
    0.0686
    >>> round(gfr_maturation(40.0), 4)  # mid-maturation
    0.1027
    >>> round(gfr_maturation(1000.0, weight_kidney_g=440.0), 5)  # 0.44 L kidney
    0.11704
    """
    validate_positive(pma_weeks, "pma_weeks")
    if weight_kidney_g is not None:
        validate_positive(weight_kidney_g, "weight_kidney_g")
    hill = _gfr_scalar("HILL")
    tm50 = _gfr_scalar("TM50_WEEKS")
    f_premat = _gfr_scalar("F_GFR_PREMAT")
    gfr_mat = _gfr_scalar("GFR_MAT_L_MIN")
    v_kid = _gfr_scalar("V_KID_STD_L")
    # Overflow-stable sigmoid: 1 / (1 + (TM50/PMA)^HILL) == PMA^HILL/(TM50^HILL+PMA^HILL)
    hill_term = 1.0 / (1.0 + (tm50 / pma_weeks) ** hill)
    specific = (hill_term * (1.0 - f_premat) + f_premat) * gfr_mat / v_kid
    if weight_kidney_g is not None:
        specific *= weight_kidney_g / 1000.0
    return float(specific)


SOLVER_DEFAULTS: dict[str, dict[str, float | int | str]] = {
    # The three engines' published default tolerances, recorded as DATA for
    # parity comparison.  The toolkit itself integrates with fixed-step
    # classical RK4 (pharmaml.pharma.pkpd), not these adaptive engines.
    "rxode2": {"atol": 1e-8, "rtol": 1e-6, "maxsteps": 70000, "method": "liblsoda"},
    "mrgsolve": {"atol": 1e-8, "rtol": 1e-8, "maxsteps": 20000, "hmax": 0.0},
    "cvodes_osp": {"atol": 1e-10, "rtol": 1e-5, "hmax": 60.0, "mxstep": 100000},
}


OSP_PK_PARAMETERS: dict[str, int] = {
    # OSPSuite PKAnalysis parameter codes (22).  AUCM_* is the first
    # moment (AUMC); the _norm variants are dose-normalized; Tthreshold
    # additionally needs a user threshold concentration.
    "C_max": 1,
    "C_max_norm": 2,
    "C_min": 3,
    "C_min_norm": 4,
    "t_max": 5,
    "t_min": 6,
    "C_trough": 7,
    "C_trough_norm": 8,
    "AUC_tEnd": 9,
    "AUC_tEnd_norm": 10,
    "AUCM_tEnd": 11,
    "AUC_inf": 12,
    "AUC_inf_norm": 13,
    "AUC_tEnd_inf": 14,
    "AUC_tEnd_inf_norm": 15,
    "CL": 16,
    "MRT": 17,
    "FractionAucEndToInf": 18,
    "Thalf": 19,
    "Vss": 20,
    "Vd": 21,
    "Tthreshold": 22,
}


def pk_sensitivity(
    parameters: dict[str, float],
    pk_function: Callable[..., float],
    variation_range: float = 0.1,
    number_of_steps: int = 2,
) -> dict[str, float]:
    """PK parameter elasticities by the OSPSuite-R sensitivity protocol.

    Each parameter is perturbed to ``2 * number_of_steps`` points
    symmetric around its reference value -- factors ``1 +/- j *
    variation_range / number_of_steps`` for ``j = 1..number_of_steps``
    (the default 10% range at 2 steps gives 0.90/0.95/1.05/1.10 of the
    reference) -- and ``pk_function`` is evaluated with the full
    parameter mapping, one parameter replaced at a time.  The reported
    sensitivity is the log-log least-squares slope of ``ln(PK)`` versus
    ``ln(parameter)`` (a power-law elasticity: PK ~ parameter^slope),
    guarded by a ``1e-20`` epsilon so zero/negative evaluations map to
    the log of the guard instead of raising.

    Parameters
    ----------
    parameters:
        Reference parameter values, ``{name: value}`` (non-empty).
    pk_function:
        Callable accepting the parameters as keyword arguments and
        returning a scalar PK statistic.
    variation_range:
        Total perturbation range (0 < range < 1); 0.1 = +/- 10%.
    number_of_steps:
        Steps per side (>= 1); 2 steps -> 4 evaluation points.

    Returns
    -------
    dict[str, float]
        ``{parameter_name: elasticity}`` in input order.

    Raises
    ------
    DataValidationError
        If ``parameters`` is empty, ``variation_range`` is outside (0, 1)
        or ``number_of_steps`` < 1.

    Examples
    --------
    >>> from pharmaml.pharma.events import pk_sensitivity
    >>> sens = pk_sensitivity({"cl": 2.0, "vd": 50.0}, lambda cl, vd: 1.0 / cl)
    >>> {name: round(value, 6) for name, value in sens.items()}
    {'cl': -1.0, 'vd': 0.0}
    >>> sens = pk_sensitivity({"cl": 2.0, "vd": 50.0}, lambda cl, vd: cl / vd)
    >>> {name: round(value, 6) for name, value in sens.items()}
    {'cl': 1.0, 'vd': -1.0}
    """
    if not parameters:
        raise DataValidationError("parameters must be a non-empty {name: reference_value} mapping")
    if not 0.0 < variation_range < 1.0:
        raise DataValidationError(
            f"variation_range must lie in (0, 1), got {variation_range!r}"
        )
    if number_of_steps < 1:
        raise DataValidationError(f"number_of_steps must be >= 1, got {number_of_steps!r}")
    epsilon = 1e-20
    elasticities: dict[str, float] = {}
    for name, reference in parameters.items():
        log_p: list[float] = []
        log_pk: list[float] = []
        for step in range(1, number_of_steps + 1):
            for sign in (-1.0, 1.0):
                value = reference * (1.0 + sign * step * variation_range / number_of_steps)
                call_kwargs = dict(parameters)
                call_kwargs[name] = value
                output = float(pk_function(**call_kwargs))
                log_p.append(math.log(max(value, epsilon)))
                log_pk.append(math.log(max(output, epsilon)))
        mean_p = sum(log_p) / len(log_p)
        mean_pk = sum(log_pk) / len(log_pk)
        sxx = sum((x - mean_p) ** 2 for x in log_p)
        sxy = sum((x - mean_p) * (y - mean_pk) for x, y in zip(log_p, log_pk))
        elasticities[name] = sxy / sxx if sxx > 0.0 else 0.0
    return elasticities


def dubois_bsa(weight_kg: float, height_cm: float) -> float:
    """Body surface area (m2) by the Du Bois 1916 formula.

    ``BSA = 0.007184 * W^0.425 * H^0.725`` with W in kg and H in cm --
    the original 1916 archetype of the height-weight BSA regressions.
    Cross-reference :func:`pharmaml.pharma.dosage.bsa_mosteller` for the
    geometric-mean formula used by the dosage module (note its
    height-first argument order).

    Parameters
    ----------
    weight_kg:
        Body weight in kilograms (> 0).
    height_cm:
        Height in centimeters (> 0).

    Returns
    -------
    float
        Body surface area in m2.

    Examples
    --------
    >>> from pharmaml.pharma.events import dubois_bsa
    >>> round(dubois_bsa(70.0, 175.0), 3)  # Du Bois 1916, reference adult
    1.848
    >>> from pharmaml.pharma.dosage import bsa_mosteller
    >>> round(bsa_mosteller(175.0, 70.0), 3)  # Mosteller cross-check
    1.845
    """
    validate_positive(weight_kg, "weight_kg")
    validate_positive(height_cm, "height_cm")
    return float(0.007184 * weight_kg**0.425 * height_cm**0.725)
