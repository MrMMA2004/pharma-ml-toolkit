r"""The event-table grammar and the event-driven ODE driver.

Round 176 absorbed the *event half* of the two ODE platforms (the
analytic half lives in :mod:`pharmaml.pharma.lincmt`):

``metrumresearchgroup/mrgsolve``
    (``src/datarecord.cpp`` / ``R/class_ev.R``).  The full EVID
    semantics transfer exactly: 0 observation, 1 dose (bolus or
    infusion), 2 other (``cmt > 0`` turns a compartment on, ``cmt < 0``
    turns it off and zeroes it), 3 reset (all states zero, infusions
    stopped), 4 reset + dose, 8 replace the compartment amount.
    Infusions are constant rates accumulated into the RHS
    (``ydot = (f(y) + R0) * On`` — the mrgsolve ``rate_add``/
    ``rate_rm`` convention), with the off-event scheduled at
    ``time + F*amt/rate``.  ``addl`` realizes as ``time + ii*k``
    (``schedule()``); ``until`` converts to ``addl = ceil((until -
    time)/ii) - 1``; ``total`` to ``addl = total - 1``.

``nlmixr2/rxode2``
    (``src/par_solve.cpp``).  The steady-state finder: iterate doses
    until the trough-to-trough change satisfies ``|dy| < ssRtol*|y| +
    ssAtol`` on every tracked compartment (``ssBolusDone``), plus the
    Chiou two-point elimination-rate heuristic for constant infusions
    and the ``minSS``/``maxSS`` bounds.  The LSODA solver itself is the
    not-vendored part; a fixed-step RK4 advances between events.

Examples
--------
>>> import numpy as np
>>> from pharmaml.pharma.eventtable import EventTable, simulate_events
>>> rhs = lambda t, y: -0.1 * y                      # 1-cmt, kel = 0.1
>>> events = EventTable().dose(time=0.0, amt=100.0)
>>> t, y = simulate_events(rhs, y0=np.array([0.0]), events=events,
...                        t_eval=np.array([2.5]))
>>> round(float(y[0, 0]), 4)                         # 100*exp(-0.25)
77.8801

An infusion, turned off automatically at ``amt/rate``:

>>> events = EventTable().dose(time=0.0, amt=50.0, rate=10.0)
>>> t, y = simulate_events(rhs, y0=np.array([0.0]), events=events,
...                        t_eval=np.array([3.0, 10.0]))
>>> round(float(y[0, 0]), 4)   # 10/0.1*(1-exp(-0.3))
25.9182

Steady state of a q24h bolus regimen, found by the trough test:

>>> from pharmaml.pharma.eventtable import find_steady_state
>>> def trough(y): return float(y[0])
>>> ss = find_steady_state(rhs, np.array([0.0]), dose=100.0, ii=24.0,
...                        tracked=[0], min_doses=3, max_doses=60)
>>> round(ss["trough"], 4)      # 100*exp(-2.4)/(1-exp(-2.4))
9.9769
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "EventTable",
    "chiou_elimination_rate",
    "dose_grid",
    "expand_events",
    "find_steady_state",
    "simulate_events",
]

_COLUMNS = [
    "time",
    "amt",
    "evid",
    "cmt",
    "rate",
    "dur",
    "ii",
    "addl",
    "ss",
    "f",
    "tlag",
]


@dataclass(slots=True)
class _Record:
    """One raw event row before grammar expansion."""

    time: float = 0.0
    amt: float = 0.0
    evid: int = 1
    cmt: int = 0
    rate: float = 0.0
    dur: float = 0.0
    ii: float = 0.0
    addl: int = 0
    ss: bool = False
    f: float = 1.0
    tlag: float = 0.0


class EventTable:
    """Builder for the NONMEM-style event table.

    Columns follow the mrgsolve/rxode2 grammar: ``time, amt, evid, cmt,
    rate, dur, ii, addl, ss, f, tlag``.  Compartments are 0-based (cmt
    0 = central by the mrgsolve 1-based convention shifted down, so
    ``cmt=-1`` means "compartment 0 off" for evid 2).
    """

    def __init__(self) -> None:
        self._records: list[_Record] = []

    # -- fluent builders ---------------------------------------------------
    def dose(
        self,
        time: float = 0.0,
        amt: float = 0.0,
        cmt: int = 0,
        rate: float = 0.0,
        dur: float = 0.0,
        ii: float = 0.0,
        addl: int = 0,
        until: float | None = None,
        total: int | None = None,
        ss: bool = False,
        f: float = 1.0,
        tlag: float = 0.0,
    ) -> EventTable:
        """Add a dosing record (evid 1).

        ``rate`` > 0 makes an infusion of ``amt`` at that rate;
        ``dur`` > 0 (with rate 0) sets rate = amt/dur.  ``addl``
        additional doses follow every ``ii``; ``until`` (mrgsolve) or
        ``total`` convert to ``addl``.  ``ss`` marks a regimen already
        at steady state (honoured by :func:`find_steady_state` and the
        analytic engine, ignored by the numeric driver).
        """
        if until is not None and ii <= 0:
            raise DataValidationError("until needs ii > 0")
        if until is not None:
            addl = math.ceil((until - time) / ii) - 1 if until > time else 0
        if total is not None:
            addl = total - 1
        if rate > 0 and dur > 0:
            raise DataValidationError("rate and dur are mutually exclusive")
        if (addl > 0 or ss) and ii <= 0:
            raise DataValidationError("addl/ss records need ii > 0")
        self._records.append(
            _Record(float(time), float(amt), 1, int(cmt), float(rate), float(dur),
                    float(ii), int(addl), bool(ss), float(f), float(tlag))
        )
        return self

    def observation(self, time: float) -> EventTable:
        """Add an evid 0 observation record (kept for grammar fidelity)."""
        self._records.append(_Record(time=float(time), amt=0.0, evid=0))
        return self

    def reset(self, time: float) -> EventTable:
        """Add an evid 3 reset record (states zeroed, infusions stopped)."""
        self._records.append(_Record(time=float(time), amt=0.0, evid=3))
        return self

    def reset_dose(self, time: float, amt: float, cmt: int = 0) -> EventTable:
        """Add an evid 4 reset + dose record."""
        self._records.append(_Record(time=float(time), amt=float(amt), evid=4, cmt=int(cmt)))
        return self

    def replace(self, time: float, amt: float, cmt: int = 0) -> EventTable:
        """Add an evid 8 record: set the compartment amount to ``amt``."""
        self._records.append(_Record(time=float(time), amt=float(amt), evid=8, cmt=int(cmt)))
        return self

    def other(self, time: float, cmt: int) -> EventTable:
        """Add an evid 2 record: ``cmt > 0`` on, ``cmt < 0`` off + zero."""
        if cmt == 0:
            raise DataValidationError("evid 2 needs a non-zero cmt")
        self._records.append(_Record(time=float(time), amt=0.0, evid=2, cmt=int(cmt)))
        return self

    # -- conversion ---------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """The event table as a DataFrame (addl not yet expanded)."""
        rows = [vars(record) if hasattr(record, "__dict__") else _as_dict(record)
                for record in self._records]
        frame = pd.DataFrame(rows, columns=_COLUMNS)
        return frame.astype({"time": float, "amt": float, "evid": int, "cmt": int,
                             "rate": float, "dur": float, "ii": float, "addl": int,
                             "ss": bool, "f": float, "tlag": float})

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> EventTable:
        """Read an event table from a DataFrame with the grammar columns."""
        table = cls()
        for row in frame.to_dict("records"):
            table._records.append(
                _Record(
                    time=float(row.get("time", 0.0)),
                    amt=float(row.get("amt", 0.0)),
                    evid=int(row.get("evid", 1)),
                    cmt=int(row.get("cmt", 0)),
                    rate=float(row.get("rate", 0.0)),
                    dur=float(row.get("dur", 0.0)),
                    ii=float(row.get("ii", 0.0)),
                    addl=int(row.get("addl", 0)),
                    ss=bool(row.get("ss", False)),
                    f=float(row.get("f", 1.0)),
                    tlag=float(row.get("tlag", 0.0)),
                )
            )
        return table

    def __len__(self) -> int:
        return len(self._records)


def _as_dict(record: _Record) -> dict[str, object]:
    return {
        "time": record.time,
        "amt": record.amt,
        "evid": record.evid,
        "cmt": record.cmt,
        "rate": record.rate,
        "dur": record.dur,
        "ii": record.ii,
        "addl": record.addl,
        "ss": record.ss,
        "f": record.f,
        "tlag": record.tlag,
    }


def expand_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Realize ``addl`` repeats (mrgsolve ``schedule``/``realize_addl``).

    Children at ``time + ii*k`` (k = 1..addl) inherit the record; the
    parent's ``addl``/``ss`` bookkeeping follows ``realize_addl``: child
    records lose ``ss``, the parent keeps ``ii`` only when ``ss``.

    Examples
    --------
    >>> table = EventTable().dose(time=0.0, amt=100.0, ii=24.0, addl=2)
    >>> expanded = expand_events(table.to_frame())
    >>> list(expanded["time"])
    [0.0, 24.0, 48.0]
    >>> len(expanded)
    3
    """
    required = {"time", "amt", "evid"}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"event frame missing columns: {sorted(missing)}")
    out = []
    for row in frame.to_dict("records"):
        addl = int(row.get("addl", 0) or 0)
        ii = float(row.get("ii", 0.0) or 0.0)
        child = dict(row)
        child["addl"] = 0
        child["ss"] = False
        # evid 4 children are plain doses (rxode2 addlDropSs rule)
        if addl > 0 and int(child.get("evid", 1)) == 4:
            child["evid"] = 1
        if not row.get("ss", False):
            child["ii"] = 0.0
        out.append(child)
        for k in range(1, addl + 1):
            repeat = dict(child)
            repeat["time"] = float(row["time"]) + ii * k
            out.append(repeat)
    columns = [c for c in _COLUMNS if c in frame.columns]
    result = pd.DataFrame(out, columns=columns + [c for c in frame.columns if c not in columns])
    return result.sort_values("time", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# the numeric event driver
# ---------------------------------------------------------------------------


def _rk4_span(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y: np.ndarray,
    t0: float,
    t1: float,
    rates: np.ndarray,
    on: np.ndarray,
    substeps: int,
) -> np.ndarray:
    """Advance y from t0 to t1 with fixed-step RK4 (mrgsolve RHS convention)."""

    def full(t: float, state: np.ndarray) -> np.ndarray:
        return (rhs(t, state) + rates) * on

    h = (t1 - t0) / substeps
    t = t0
    for _ in range(substeps):
        k1 = full(t, y)
        k2 = full(t + h / 2, y + h / 2 * k1)
        k3 = full(t + h / 2, y + h / 2 * k2)
        k4 = full(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def simulate_events(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    events: EventTable | pd.DataFrame,
    t_eval: np.ndarray,
    substeps: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an ODE system under the event grammar (numeric RK4).

    ``rhs(t, y)`` returns the base derivatives; active infusion rates
    are added per compartment (``ydot = (rhs + R0) * On``) exactly as
    mrgsolve's ``call_derivs`` does.  Returns ``(t_eval, Y)`` with one
    state row per requested time.

    Examples
    --------
    >>> import numpy as np
    >>> rhs = lambda t, y: -0.1 * y
    >>> events = EventTable().dose(time=0.0, amt=100.0)
    >>> t, y = simulate_events(rhs, np.array([0.0]), events,
    ...                        t_eval=np.array([2.5]))
    >>> round(float(y[0, 0]), 4)
    77.8801
    """
    if isinstance(events, EventTable):
        frame = events.to_frame()
    else:
        frame = events
    frame = expand_events(frame)
    frame = frame[frame["evid"] != 0].reset_index(drop=True)  # observations carry no action

    y = np.asarray(y0, dtype=float).copy()
    n_cmt = y.size
    rates = np.zeros(n_cmt)
    on = np.ones(n_cmt)
    t_now: float | None = None  # state time; None = snap to first timeline point

    eval_sorted = np.sort(np.asarray(t_eval, dtype=float))
    outputs: list[np.ndarray] = []
    cursor = 0

    event_iter = frame.to_dict("records")
    pending_off: list[tuple[float, int, float]] = []  # (time, cmt, rate)

    # process timeline: event rows sorted by (time, position)
    rows = sorted(event_iter, key=lambda r: float(r["time"]) + float(r.get("tlag", 0.0) or 0.0))

    def advance_to(target: float) -> None:
        nonlocal y, t_now
        if t_now is None:
            t_now = target  # y0 is the state at the first timeline point
            return
        if target <= t_now + 1e-12:
            return
        y = _rk4_span(rhs, y, t_now, target, rates, on, substeps)
        t_now = target

    for row in rows:
        act_time = float(row["time"]) + float(row.get("tlag", 0.0) or 0.0)
        # flush infusion-off events due before this record
        while pending_off:
            off_time, off_cmt, off_rate = min(pending_off)
            if off_time > act_time + 1e-12:
                break
            pending_off.remove((off_time, off_cmt, off_rate))
            while cursor < eval_sorted.size and eval_sorted[cursor] < off_time - 1e-12:
                advance_to(float(eval_sorted[cursor]))
                outputs.append(y.copy())
                cursor += 1
            advance_to(off_time)
            rates[off_cmt] = max(0.0, rates[off_cmt] - off_rate)
        # emit requested times strictly before the event
        while cursor < eval_sorted.size and eval_sorted[cursor] < act_time - 1e-12:
            advance_to(float(eval_sorted[cursor]))
            outputs.append(y.copy())
            cursor += 1
        advance_to(act_time)
        evid = int(row.get("evid", 1))
        cmt = int(row.get("cmt", 0))
        amt = float(row.get("amt", 0.0))
        rate = float(row.get("rate", 0.0))
        dur = float(row.get("dur", 0.0))
        f = float(row.get("f", 1.0))
        if rate > 0 and dur > 0:
            raise DataValidationError("rate and dur are mutually exclusive")
        if evid == 1 or evid == 4:
            if evid == 4:  # reset + dose
                y[:] = 0.0
                rates[:] = 0.0
                on[:] = 1.0
            if rate > 0:
                # infusion: rate on now, off at time + F*amt/rate
                eff_rate = rate
                rates[cmt] += eff_rate
                duration = f * amt / rate if amt > 0 else math.inf
                if math.isfinite(duration):
                    pending_off.append((act_time + duration, cmt, eff_rate))
            elif dur > 0:
                eff_rate = amt / dur
                rates[cmt] += eff_rate
                pending_off.append((act_time + f * dur, cmt, eff_rate))
            else:
                y[cmt] += amt * f
        elif evid == 3:
            y[:] = 0.0
            rates[:] = 0.0
            on[:] = 1.0
            pending_off.clear()
        elif evid == 8:
            y[cmt] = amt
        elif evid == 2:
            if cmt > 0:
                on[cmt - 1] = 1.0
            else:
                idx = -cmt - 1
                on[idx] = 0.0
                y[idx] = 0.0
                rates[idx] = 0.0
        elif evid == 0:
            pass

    # trailing: infusion-offs and remaining eval times
    while cursor < eval_sorted.size:
        next_eval = float(eval_sorted[cursor])
        if pending_off:
            off_time, off_cmt, off_rate = min(pending_off)
        else:
            off_time = math.inf
            off_cmt, off_rate = 0, 0.0
        if off_time < next_eval - 1e-12:
            pending_off.remove((off_time, off_cmt, off_rate))
            advance_to(off_time)
            rates[off_cmt] = max(0.0, rates[off_cmt] - off_rate)
        else:
            advance_to(next_eval)
            outputs.append(y.copy())
            cursor += 1

    if len(outputs) != eval_sorted.size:
        raise DataValidationError("internal error: output/eval mismatch")
    return eval_sorted, np.vstack(outputs) if outputs else np.empty((0, n_cmt))


# ---------------------------------------------------------------------------
# steady state
# ---------------------------------------------------------------------------


def find_steady_state(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    dose: float,
    ii: float,
    tracked: Sequence[int] | None = None,
    cmt: int = 0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    min_doses: int = 3,
    max_doses: int = 500,
    substeps: int = 240,
) -> dict[str, Any]:
    """Iterate a bolus regimen to steady state (rxode2 ``ssBolusDone``).

    Convergence when the trough-to-trough change satisfies
    ``|dy| < ssRtol*|y| + ssAtol`` on every tracked compartment, after at
    least ``min_doses`` doses (the rxode2 ``minSS`` floor).  Returns the
    pre-dose (trough) state summary.

    Examples
    --------
    >>> import numpy as np
    >>> rhs = lambda t, y: -0.1 * y
    >>> ss = find_steady_state(rhs, np.array([0.0]), dose=100.0, ii=24.0)
    >>> round(float(ss["trough"]), 4)
    9.9769
    >>> ss["doses"] > 0
    True
    """
    if ii <= 0:
        raise DataValidationError("ii must be positive")
    y = np.asarray(y0, dtype=float).copy()
    tracked_idx = list(range(y.size)) if tracked is None else [int(i) for i in tracked]
    last_trough = None
    n = 0
    while n < max_doses:
        y[cmt] += dose
        y = _rk4_span(rhs, y, 0.0, ii, np.zeros(y.size), np.ones(y.size), substeps)
        n += 1
        trough = y.copy()
        if last_trough is not None and n >= min_doses:
            ok = all(
                abs(trough[i] - last_trough[i]) <= rtol * abs(trough[i]) + atol
                for i in tracked_idx
            )
            if ok:
                return {
                    "trough": float(trough[tracked_idx[0]]),
                    "doses": n,
                    "state": trough,
                }
        last_trough = trough
    raise DataValidationError(
        f"steady state not reached within {max_doses} doses (maxSS exceeded)"
    )


def chiou_elimination_rate(
    rate: float, a1: float, t1: float, a2: float, t2: float
) -> float:
    """The Chiou two-point elimination rate for constant infusions.

    ``k = R/(a1+a2) + (a1-a2)/((a1+a2)*(t2-t1))`` — the heuristic
    mrgsolve (``steady_zero``) and rxode2 (``solveSSinfIterate``) use to
    size the run length to reach infusion steady state.

    Examples
    --------
    >>> round(chiou_elimination_rate(10.0, 40.0, 0.0, 60.0, 10.0), 6)
    0.08
    """
    if t2 <= t1:
        raise DataValidationError("t2 must exceed t1")
    if a1 + a2 <= 0:
        raise DataValidationError("a1 + a2 must be positive")
    return rate / (a1 + a2) + (a1 - a2) / ((a1 + a2) * (t2 - t1))


def dose_grid(
    start: float = 0.0,
    end: float = 24.0,
    delta: float = 1.0,
    add: Sequence[float] = (),
    offset: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Render an output time grid (mrgsolve ``tgrid`` ``render_time``).

    ``seq(start, end, delta)`` unioned with the extra ``add`` times,
    then shifted by ``offset`` and scaled — the mrgsolve output-grid
    convention.

    Examples
    --------
    >>> dose_grid(0.0, 6.0, 2.0, add=[5.0])
    array([0., 2., 4., 5., 6.])
    """
    if delta <= 0:
        raise DataValidationError("delta must be positive")
    grid = np.arange(start, end + 1e-9, delta)
    if add:
        grid = np.union1d(grid, np.asarray(add, dtype=float))
    return (grid + offset) * scale
