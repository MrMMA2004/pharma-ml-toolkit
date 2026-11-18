"""ODE engine with pharmacokinetic dosing events (RxODE/pkpdutils semantics).

This module is the toolkit's native answer to the ODE-simulation layer
that ``RxODE``, ``mrgsolve`` and the classic ``pkpdutils`` expose: a
right-hand-side ``f(t, y)`` plus a schedule of :class:`DosingEvent`
objects (boluses and zero-order infusions into numbered compartments),
integrated with a hand-rolled adaptive **Dormand–Prince RK5(4)** step
controller — the same tableau ``scipy.integrate.solve_ivp`` uses for
``method="RK45"`` — or a fixed-step classical RK4 for teaching and
testing.  No compiled solver, no C++ codegen: the engine is ~200 lines
of numpy that any pharmacometric model can sit on top of.

Conventions
-----------
* ``rhs(t, y)`` must return the derivative as a sequence of floats the
  same length as ``y0``; times are hours;
* :class:`DosingEvent` with ``duration=0`` is an IV-bolus-style impulse
  (``y[compartment] += amount`` exactly at ``event.time``);
* ``duration > 0`` is a zero-order infusion: the engine adds the
  constant input rate ``amount / duration`` to the compartment between
  ``event.time`` and ``event.time + duration`` (the integration is
  segmented at every event boundary so the rate is piecewise-constant);
* output times are *forced step boundaries* — the reported states are
  the integrator's own states at those times, never interpolations.

Examples
--------
>>> import numpy as np
>>> from pharmaml.pkpd.ode import DosingEvent, solve_ode
>>> rhs = lambda t, y: -0.5 * y                      # e^{-t/2} decay
>>> sol = solve_ode(rhs, times=[0.0, 1.0, 2.0], y0=[2.0])
>>> np.round(sol.states[:, 0], 6)                     # 2 e^{-t/2} (tolerance)
array([2.      , 1.213062, 0.735759])

A bolus at t=2 adds its amount to compartment 0 exactly:

>>> sol = solve_ode(rhs, times=[2.0, 3.0], y0=[0.0],
...                events=[DosingEvent(time=2.0, amount=5.0)])
>>> np.round(sol.states[:, 0], 6)                     # 5 e^{-t/2} after the jump
array([5.      , 3.032654])

A 6-unit infusion over 3 hours (rate 2/h) against first-order
elimination — the analytic answer at the end of the infusion is
``(R/k) (1 - e^{-k T}) = 4 (1 - e^{-1.5})``:

>>> sol = solve_ode(rhs, times=[3.0], y0=[0.0],
...                events=[DosingEvent(time=0.0, amount=6.0, duration=3.0)])
>>> round(float(sol.states[0, 0]), 4)
3.1075

Fixed-step classical RK4 with a small step reproduces the exponential
to a ~1e-8 tolerance on this smooth problem:

>>> sol = solve_ode(rhs, times=[0.0, 1.0], y0=[1.0], method="rk4", max_step=0.01)
>>> round(float(sol.states[1, 0]), 8)                 # e^{-0.5}
0.60653066

Bad input is rejected loudly:

>>> solve_ode(rhs, times=[1.0, 0.0], y0=[1.0])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: times must be strictly increasing
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError

__all__ = ["DosingEvent", "ODESolution", "solve_ode"]

# Dormand–Prince RK5(4) tableau (the "RK45" of scipy/ode45 fame).
_DP_C = np.array([0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0])
_DP_A = [
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
]
_DP_B5 = np.array(
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0]
)
_DP_B4 = np.array(
    [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ]
)


@dataclass(frozen=True, slots=True)
class DosingEvent:
    """One administration: bolus (``duration=0``) or zero-order infusion.

    ``amount`` is the dose in mass units, ``compartment`` the state
    index it enters (0-based, matching the ``y0`` vector's layout).
    """

    time: float
    amount: float
    compartment: int = 0
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise DataValidationError(f"amount must be >= 0, got {self.amount}")
        if self.duration < 0:
            raise DataValidationError(f"duration must be >= 0, got {self.duration}")
        if self.compartment < 0:
            raise DataValidationError(
                f"compartment index must be >= 0, got {self.compartment}"
            )


@dataclass(slots=True)
class ODESolution:
    """Solver output: states on the requested grid plus step statistics."""

    times: np.ndarray
    states: np.ndarray
    n_steps: int = 0
    n_rejected: int = 0
    method: str = "rk45"

    def state(self, index: int) -> np.ndarray:
        """Return the state vector row ``index`` (or the final row if -1)."""
        return self.states[index]


def _validate_arguments(
    times: np.ndarray | list[float],
    y0: np.ndarray | list[float],
    events: tuple[DosingEvent, ...] | list[DosingEvent],
    method: str,
) -> tuple[np.ndarray, np.ndarray, tuple[DosingEvent, ...]]:
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size < 1:
        raise DataValidationError("times must contain at least one value")
    if not np.all(np.isfinite(t)):
        raise DataValidationError("times must be finite")
    if t.size > 1 and not np.all(np.diff(t) > 0):
        raise DataValidationError("times must be strictly increasing")
    y = np.asarray(y0, dtype=float).reshape(-1)
    if y.size < 1 or not np.all(np.isfinite(y)):
        raise DataValidationError("y0 must be a non-empty finite vector")
    if method not in {"rk45", "rk4"}:
        raise DataValidationError(f"method must be 'rk45' or 'rk4', got {method!r}")
    ev = tuple(sorted(events, key=lambda e: (e.time, e.duration)))
    for e in ev:
        if e.compartment >= y.size:
            raise DataValidationError(
                f"event compartment {e.compartment} outside the state vector "
                f"(size {y.size})"
            )
    return t, y, ev


def _segment_boundaries(
    t_begin: float,
    t_end: float,
    times: np.ndarray,
    events: tuple[DosingEvent, ...],
) -> np.ndarray:
    """All times the integration must hit: events, infusion ends, outputs."""
    marks = {t_begin, float(times[-1])}
    for e in events:
        marks.add(float(e.time))
        if e.duration > 0:
            marks.add(float(e.time + e.duration))
    for value in times:
        marks.add(float(value))
    return np.array(sorted(m for m in marks if t_begin <= m <= t_end))


def _infusion_rates(
    events: tuple[DosingEvent, ...], n_states: int, at: float
) -> np.ndarray:
    """Zero-order input rates active at time ``at`` (amount/duration)."""
    rates = np.zeros(n_states)
    for e in events:
        if e.duration > 0 and e.time <= at < e.time + e.duration:
            rates[e.compartment] += e.amount / e.duration
    return rates


def _dp54_step(f, t: float, y: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """One Dormand–Prince 5(4) step: (y_new, error_estimate)."""
    k = np.empty((7, y.size))
    k[0] = f(t, y)
    for stage in range(1, 7):
        acc = np.zeros_like(y)
        row = _DP_A[stage]
        for j, weight in enumerate(row):
            if weight:
                acc += weight * k[j]
        k[stage] = f(t + _DP_C[stage], y + h * acc)
    y_new = y + h * np.tensordot(_DP_B5, k, axes=(0, 0))
    error = h * np.tensordot(_DP_B5 - _DP_B4, k, axes=(0, 0))
    return y_new, error


def _integrate_segment(
    f,
    a: float,
    b: float,
    y: np.ndarray,
    boundaries: np.ndarray,
    out_times: np.ndarray,
    out_row: int,
    states: np.ndarray,
    method: str,
    rtol: float,
    atol: float,
    max_step: float | None,
    stats: dict[str, int],
) -> tuple[np.ndarray, int]:
    """Integrate [a, b]; record states whenever an output time is passed.

    Returns the state at ``b`` and the next unfilled output row index.
    Output times act as forced step boundaries so no interpolation is
    ever needed.
    """
    t = a
    # All boundaries inside (a, b] split the segment into sub-segments.
    inner = boundaries[(boundaries > a) & (boundaries <= b)]
    for edge in inner:
        sub_end = float(edge)
        if method == "rk4":
            span = sub_end - t
            n_sub = max(1, math.ceil(span / max_step)) if max_step else 100
            h = span / n_sub
            for _ in range(n_sub):
                k1 = f(t, y)
                k2 = f(t + h / 2.0, y + h / 2.0 * k1)
                k3 = f(t + h / 2.0, y + h / 2.0 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t += h
                stats["n_steps"] += 1
            t = sub_end
        else:
            h = 0.5 * (sub_end - t) if sub_end > t else 0.0
            if max_step is not None:
                h = min(h, max_step)
            while t < sub_end - 1e-14:
                if max_step is not None:
                    h = min(h, max_step)
                h = min(h, sub_end - t)
                y_new, err = _dp54_step(f, t, y, h)
                scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
                norm = float(np.sqrt(np.mean((err / scale) ** 2)))
                if norm <= 1.0:
                    t += h
                    y = y_new
                    stats["n_steps"] += 1
                    growth = 5.0 if norm == 0.0 else min(5.0, 0.9 * norm ** (-0.2))
                    h *= max(0.2, growth)
                else:
                    stats["n_rejected"] += 1
                    h *= max(0.2, 0.9 * norm ** (-0.25))
                if not np.all(np.isfinite(y)):
                    raise DataValidationError(
                        "the ODE state became non-finite — check the rhs "
                        "and the parameter values"
                    )
            t = sub_end
        # Record output times that this sub-segment has reached.
        while out_row < out_times.size and out_times[out_row] <= t + 1e-12:
            states[out_row] = y
            out_row += 1
    return y, out_row


def solve_ode(
    rhs,
    times: np.ndarray | list[float],
    y0: np.ndarray | list[float],
    events: tuple[DosingEvent, ...] | list[DosingEvent] = (),
    *,
    method: str = "rk45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> ODESolution:
    """Integrate ``dy/dt = rhs(t, y)`` with dosing events, on ``times``.

    Parameters
    ----------
    rhs:
        Callable ``rhs(t, y) -> dydt`` (numpy arrays in/out; any
        returning of lists is coerced).
    times:
        Strictly increasing output grid (hours).
    y0:
        Initial state at the integration start (the earliest of the
        first output time and the first event time).
    events:
        :class:`DosingEvent` schedule — boluses (impulses) and
        zero-order infusions, any compartment.
    method:
        ``'rk45'`` (adaptive Dormand–Prince 5(4), the default) or
        ``'rk4'`` (fixed-step classical RK4, ``max_step`` = the step;
        the segment is subdivided to honour it).
    rtol, atol:
        Local error tolerances for the adaptive controller.
    max_step:
        Upper bound on the accepted step.

    Returns
    -------
    ODESolution
        ``.times`` (the requested grid), ``.states`` (one row per
        requested time), plus step statistics.
    """
    t_grid, y_start, ev = _validate_arguments(times, y0, events, method)
    if rtol <= 0 or atol <= 0:
        raise DataValidationError("rtol and atol must be positive")

    t_begin = float(t_grid[0])
    for e in ev:
        if e.time > float(t_grid[-1]) + 1e-12:
            raise DataValidationError(
                f"event at t={e.time} lies after the last output time {t_grid[-1]}"
            )
        if e.time < t_begin:
            t_begin = float(e.time)
    t_end = float(t_grid[-1])

    n = y_start.size
    states = np.full((t_grid.size, n), np.nan)
    stats = {"n_steps": 0, "n_rejected": 0}

    y = y_start.astype(float).copy()
    boundaries = _segment_boundaries(t_begin, t_end, t_grid, ev)
    out_row = 0
    seg_starts = boundaries[:-1] if boundaries.size > 1 else np.array([t_begin])
    seg_ends = boundaries[1:] if boundaries.size > 1 else np.array([t_end])

    for a, b in zip(seg_starts, seg_ends):
        a = float(a)
        b = float(b)
        # Apply bolus events exactly at the segment start.
        for e in ev:
            if e.duration == 0.0 and abs(e.time - a) < 1e-12:
                y[e.compartment] += e.amount
        # Record output times that coincide with the segment start.
        while out_row < t_grid.size and t_grid[out_row] <= a + 1e-12:
            states[out_row] = y
            out_row += 1
        if b <= a + 1e-14:
            continue
        rates = np.asarray(_infusion_rates(ev, n, 0.5 * (a + b)))

        def f(t: float, yy: np.ndarray, _rates=rates) -> np.ndarray:
            return np.asarray(rhs(t, yy), dtype=float) + _rates

        y, out_row = _integrate_segment(
            f, a, b, y, boundaries, t_grid, out_row, states, method,
            rtol, atol, max_step, stats,
        )

    # A bolus scheduled exactly at the final output time has no segment
    # of its own to fire in (segment starts are always < t_end unless
    # the whole window degenerates to one point) — apply it here, once.
    t_end_is_start = bool(seg_starts.size) and abs(float(seg_starts[-1]) - t_end) < 1e-12
    if not t_end_is_start:
        for e in ev:
            if e.duration == 0.0 and abs(e.time - t_end) < 1e-12:
                y[e.compartment] += e.amount

    while out_row < t_grid.size:
        states[out_row] = y
        out_row += 1

    return ODESolution(
        times=t_grid.copy(),
        states=states,
        n_steps=stats["n_steps"],
        n_rejected=stats["n_rejected"],
        method=method,
    )
