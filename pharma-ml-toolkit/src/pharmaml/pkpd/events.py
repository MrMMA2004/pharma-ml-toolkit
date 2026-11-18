"""The mrgsolve / rxode2 event grammar over the toolkit's ODE engine.

Round 176's absorption of **mrgsolve** (metrumresearchgroup) and
**rxode2** (nlmixr2).  Both repositories are R packages with C++ cores;
the solver cores are already covered by
:mod:`pharmaml.pkpd.ode` (the RxODE-semantics engine absorbed in
v1.2.102).  What this round adds is the part pharmacometricians
actually touch daily — the *event grammar*:

* :class:`EventRow` / :func:`ev` — the mrgsolve ``ev()`` constructor
  semantics: ``tinf → rate = amt/tinf``, ``total → addl = total−1``,
  the ``until`` → ``addl`` derivation (both engines' rules — they
  differ by one dose at the boundary, pinned side by side in the
  :func:`addl_until` doctest), 1-based compartments, and the loud
  refusals (``evid=0``, modeled rate/duration ``-1``/``-2``, ``addl``
  without ``ii``);
* :func:`realize_addl` — the addl+ii expansion with mrgsolve's
  truncation at the last record time and the evid 4 → 1 demotion;
* :func:`ev_seq`, :func:`ev_days` — regimen composition: the
  advance-to-end-of-interval rule (``time + ii·addl + ii``), ``wait``
  and ``ii`` spacers, and the weekday schedule (m=0, t=24, w=48 …
  s=144 over a 168-hour week);
* :func:`simulate_regimen` — the grammar *executed*: evid 0/1/2/3/4/8
  (observations, doses, on/off records with infusion cancellation,
  resets, reset+dose, absolute replacement), lag times (ALAG),
  bioavailability ``f`` scaling the delivered amount — and with fixed
  rates shrinking the infusion *duration* (mrgsolve's
  ``dur() = Fn·Amt/Rate``) — plus the steady-state machinery
  (``ss=1`` replaces the state with the regimen's cycle fixed point,
  ``ss=2`` superposes it, exactly rxode2's
  ``Y_after = Y_before + Y_ss_new``) and the four-way **recsort**
  tie-breaking of doses against observations at equal times;
* numeric contracts straight from the engines' own test suites:
  ``amt=3000, rate=30 → tmax == 100`` exactly
  (mrgsolve ``test-infusion.R``), ``CL=1.3, rate=25, ss → CP ==
  25/1.3`` and the bolus steady-state peak ``c0/(1 − e^{−ke·ii})``
  (rxode2 ``test-steady-state.R``), and ``F=0.5`` halving the
  infusion duration (mrgsolve ``test-D-R-F.R``: tmax 10 → 5).

Honest boundaries: mrgsolve's C++ ADVAN polyexponential engine and
recsort sentinel positions (±10⁹), rxode2's internal
``cmt100·100000 + rateI·10000 + …`` evid encoding and its
``low/high`` window draws (the randomised sampling design layer),
NONMEM DAT parsing, and both engines' variance hierarchies
(``thetaMat``/``nStud`` — the toolkit's NLME lives in
:mod:`pharmaml.pkpd.population`).

Examples
--------
A one-compartment IV infusion — mrgsolve's own acceptance test: with
``amt=3000`` at ``rate=30`` the peak lands exactly at the end of the
infusion:

>>> import numpy as np
>>> from pharmaml.pkpd.events import ev, simulate_regimen
>>> ke = 0.03
>>> rhs = lambda t, y: -ke * y
>>> grid = np.arange(0.0, 121.0, 1.0)
>>> sol = simulate_regimen(rhs, [0.0], grid, [ev(3000.0, rate=30.0)])
>>> float(grid[int(np.argmax(sol.concentration(0, volume=100.0)))])
100.0

Steady state — rxode2's ``test-steady-state.R`` identities.  A
continuous infusion (``amt=0, rate=25, ss=1``) in a one-compartment
model with CL = 1.3 (ke = 0.065 at V = 20) sits at ``rate/CL``; a
bolus regimen's steady-state peak is ``c0/(1 − e^{−ke·ii})``:

>>> sol = simulate_regimen(lambda t, y: -0.065 * y, [0.0], [0.0, 6.0, 12.0],
...     [ev(0.0, rate=25.0, ss=1, ii=12.0)])
>>> [round(float(c), 4) for c in sol.concentration(0, volume=20.0)]
[19.2308, 19.2308, 19.2308]
>>> ke2, dose, vol, ii = 0.05, 3.0, 20.0, 12.0
>>> rhs2 = lambda t, y: -ke2 * y
>>> sol = simulate_regimen(rhs2, [0.0], [0.0],
...     [ev(dose, ss=1, ii=ii)])
>>> round(float(sol.concentration(0, volume=vol)[0]), 4)
0.3325
>>> round(float((dose / vol) / (1.0 - np.exp(-ke2 * ii))), 4)   # the closed form
0.3325

Bioavailability halves the infusion duration, not the rate
(mrgsolve's ``dur() = Fn·Amt/Rate``):

>>> peak_f1 = simulate_regimen(rhs, [0.0], np.arange(0.0, 25.0, 0.5),
...     [ev(100.0, rate=10.0)])
>>> peak_f05 = simulate_regimen(rhs, [0.0], np.arange(0.0, 25.0, 0.5),
...     [ev(100.0, rate=10.0, f=0.5)])
>>> grid = np.arange(0.0, 25.0, 0.5)
>>> int(grid[int(np.argmax(peak_f1.states[:, 0]))])
10
>>> int(grid[int(np.argmax(peak_f05.states[:, 0]))])
5

The two ``until`` conventions disagree by exactly one boundary dose —
mrgsolve's doses stop strictly *before* ``until``, rxode2's include
it:

>>> from pharmaml.pkpd.events import addl_until
>>> addl_until(0.0, 24.0, 120.0, convention="mrgsolve")
4
>>> addl_until(0.0, 24.0, 120.0, convention="rxode2")
5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.pkpd.ode import DosingEvent, solve_ode

__all__ = [
    "EventRow",
    "RegimenSolution",
    "addl_until",
    "ev",
    "ev_days",
    "ev_seq",
    "realize_addl",
    "simulate_regimen",
]

# mrgsolve's weekday offsets over the 168-hour dosing week
# (R/events.R, ev_days): Monday 0 through Sunday 144.
_DAY_OFFSETS = {
    "m": 0.0,
    "t": 24.0,
    "w": 48.0,
    "th": 72.0,
    "f": 96.0,
    "sa": 120.0,
    "s": 144.0,
}

_VALID_EVID = (0, 1, 2, 3, 4, 8)


@dataclass(frozen=True, slots=True)
class EventRow:
    """One record of the mrgsolve/rxode2 event grammar.

    ``cmt`` is 1-based (the mrgsolve convention); ``rate > 0`` makes
    the row a zero-order infusion of duration ``f·amt/rate``; ``ss``
    is 0 (no steady state), 1 (advance to the regimen's steady state)
    or 2 (superpose the steady state on the current state); ``lag``
    shifts the application time (ALAG) without moving the record's
    own time.
    """

    time: float
    amt: float = 0.0
    rate: float = 0.0
    ii: float = 0.0
    addl: int = 0
    evid: int = 1
    cmt: int = 1
    ss: int = 0
    f: float = 1.0
    lag: float = 0.0
    from_addl: bool = False

    def __post_init__(self) -> None:
        if self.evid not in _VALID_EVID:
            raise DataValidationError(
                f"evid must be one of {_VALID_EVID}, got {self.evid}"
            )
        if self.amt < 0:
            raise DataValidationError(f"amt must be >= 0, got {self.amt}")
        if self.rate < 0:
            raise DataValidationError(
                f"rate must be >= 0 (modeled rate -1 / duration -2 are "
                f"mrgsolve/rxode2 model-coupled features this engine does "
                f"not take), got {self.rate}"
            )
        if self.ii < 0:
            raise DataValidationError(f"ii must be >= 0, got {self.ii}")
        if self.addl < 0:
            raise DataValidationError(f"addl must be >= 0, got {self.addl}")
        if self.evid in (1, 4, 8) and self.cmt < 1:
            raise DataValidationError(
                f"cmt is 1-based and must be >= 1 for dosing records, got {self.cmt}"
            )
        if self.evid == 2 and self.cmt == 0:
            raise DataValidationError(
                "evid=2 records need a nonzero cmt (positive = on, negative = off)"
            )
        if self.ss not in (0, 1, 2):
            raise DataValidationError(f"ss must be 0, 1 or 2, got {self.ss}")
        if not 0 < self.f <= 1:
            raise DataValidationError(f"f must be in (0, 1], got {self.f}")
        if self.lag < 0:
            raise DataValidationError(f"lag must be >= 0, got {self.lag}")
        if (self.addl or self.ss) and self.ii <= 0:
            raise DataValidationError(
                "addl > 0 or ss > 0 requires ii > 0 (mrgsolve's rule)"
            )


def ev(
    amount: float,
    *,
    time: float = 0.0,
    cmt: int = 1,
    rate: float = 0.0,
    tinf: float | None = None,
    ii: float = 0.0,
    addl: int = 0,
    total: int | None = None,
    until: float | None = None,
    evid: int = 1,
    f: float = 1.0,
    lag: float = 0.0,
    ss: int = 0,
) -> EventRow:
    """Construct one dosing record — mrgsolve's ``ev()`` semantics.

    The derivation rules are mrgsolve's exactly
    (``R/class_ev.R::finalize_ev_data``):

    * ``tinf`` ( infusion duration) converts to ``rate = amt/tinf``;
    * ``total`` (number of doses) converts to ``addl = total − 1``;
    * ``until`` converts through :func:`addl_until` (the mrgsolve
      convention here — doses strictly before ``until``);
    * the pairs (rate, tinf) and (addl, total, until) are mutually
      exclusive and refused loudly.

    ``evid=0`` (an observation record) is refused — observations come
    from the output grid, not the dosing table.
    """
    if evid == 0:
        raise DataValidationError(
            "evid cannot be 0 (observation) — the output grid carries "
            "observations"
        )
    if tinf is not None:
        if rate > 0:
            raise DataValidationError("rate and tinf are mutually exclusive")
        if tinf <= 0:
            raise DataValidationError(f"tinf must be > 0, got {tinf}")
        rate = amount / tinf
    if total is not None and until is not None:
        raise DataValidationError("total and until are mutually exclusive")
    if (addl or total is not None or until is not None) and sum(
        1 for flag in (addl != 0, total is not None, until is not None) if flag
    ) > 1:
        raise DataValidationError(
            "addl, total and until are mutually exclusive"
        )
    if total is not None:
        if total < 1:
            raise DataValidationError(f"total must be >= 1, got {total}")
        addl = total - 1
    if until is not None:
        if ii <= 0:
            raise DataValidationError("until requires ii > 0")
        addl = addl_until(time, ii, until, convention="mrgsolve")
    return EventRow(
        time=time,
        amt=amount,
        rate=rate,
        ii=ii,
        addl=addl,
        evid=evid,
        cmt=cmt,
        ss=ss,
        f=f,
        lag=lag,
    )


def addl_until(
    time: float, ii: float, until: float, convention: str = "mrgsolve"
) -> int:
    """Expand ``until`` into an ``addl`` count — the two engines' rules.

    mrgsolve (``R/class_ev.R``): ``addl = ceiling((until − time)/ii) − 1``
    — the scheduled doses stop strictly *before* ``until``.

    rxode2 (``R/etNew.R::.etDoseUntil``): ``addl = ceil((until − time −
    ii)/ii)``, plus one when the remainder divides exactly — the dose
    landing exactly on ``until`` *is* scheduled.  For (time 0, ii 24,
    until 120): mrgsolve doses at 0, 24, 48, 72, 96 (addl 4), rxode2
    adds the 120 boundary dose (addl 5).
    """
    if ii <= 0:
        raise DataValidationError(f"ii must be > 0, got {ii}")
    if until <= time:
        raise DataValidationError(
            f"until must be after time, got until={until} <= time={time}"
        )
    if convention == "mrgsolve":
        return math.ceil((until - time) / ii) - 1
    if convention == "rxode2":
        span = until - time - ii
        base = math.ceil(span / ii) if span > 0 else 0
        extra = 1 if math.isclose(span / ii, round(span / ii)) and span > 0 else 0
        return int(base + extra)
    raise DataValidationError(
        f"convention must be 'mrgsolve' or 'rxode2', got {convention!r}"
    )


def realize_addl(
    events: list[EventRow] | EventRow, max_time: float | None = None
) -> list[EventRow]:
    """Expand addl+ii into explicit records — mrgsolve's expansion.

    Doses appear at ``time + k·ii`` for ``k = 1..addl``, truncated at
    ``max_time`` when given (mrgsolve truncates at the last record
    time for the ID inside a data set; a bare ``ev()`` object has no
    later records, so the default expands every copy).  The copies
    demote evid 4 to evid 1 (a reset only fires once) and carry
    ``from_addl=True`` so the recsort machinery can order them;
    every realized row has ii and addl zeroed.
    """
    rows = [events] if isinstance(events, EventRow) else list(events)
    out: list[EventRow] = []
    for row in rows:
        # The base record keeps its ii when it carries a steady-state
        # flag (the cycle length); addl copies become plain doses
        # (rxode2's addlDropSs default) and evid 4 demotes to 1.
        base = replace(
            row,
            ii=row.ii if row.ss else 0.0,
            addl=0,
            from_addl=False,
        )
        out.append(base)
        if not row.addl:
            continue
        if row.ii <= 0:
            raise DataValidationError(
                f"addl={row.addl} on the record at t={row.time} requires ii > 0"
            )
        for k in range(1, row.addl + 1):
            dose_time = row.time + row.ii * float(k)
            if max_time is not None and dose_time > max_time:
                break
            out.append(
                replace(
                    base,
                    time=dose_time,
                    ii=0.0,
                    evid=1 if row.evid == 4 else row.evid,
                    ss=0,
                    from_addl=True,
                )
            )
    return out


def ev_seq(
    *items: EventRow | list[EventRow], wait: float | None = None, ii: float | None = None
) -> list[EventRow]:
    """Sequence regimens — mrgsolve's ``ev_seq`` advance rules.

    The next event starts at ``last time + ii·addl + ii`` (the end of
    the previous dosing interval; a single bolus with no ii stays put).
    A ``wait`` spacer offsets from that interval end; an ``ii`` spacer
    offsets from the *last dose time* — with trailing addl doses those
    differ:

    >>> from pharmaml.pkpd.events import ev, ev_seq
    >>> rows = ev_seq(ev(100.0, ii=24.0, addl=2), ev(50.0), wait=10.0)
    >>> [round(r.time, 4) for r in rows]
    [0.0, 24.0, 48.0, 82.0]
    >>> rows = ev_seq(ev(100.0, ii=24.0, addl=2), ev(50.0), ii=10.0)
    >>> [round(r.time, 4) for r in rows]
    [0.0, 24.0, 48.0, 58.0]
    """
    if wait is not None and ii is not None:
        raise DataValidationError("wait and ii spacers are mutually exclusive")
    flat_items: list[list[EventRow]] = []
    for item in items:
        flat_items.append([item] if isinstance(item, EventRow) else list(item))
    if not flat_items:
        return []
    shifted: list[EventRow] = []
    offset = 0.0
    prev_last: EventRow | None = None
    for rows in flat_items:
        if prev_last is not None:
            interval_end = prev_last.time + prev_last.ii * float(prev_last.addl) + prev_last.ii
            last_dose = prev_last.time + prev_last.ii * float(prev_last.addl)
            if wait is not None:
                offset += interval_end + wait
            elif ii is not None:
                offset += last_dose + ii
            else:
                offset += interval_end
        shifted.extend(replace(r, time=r.time + offset) for r in rows)
        prev_last = rows[-1]
    return realize_addl(shifted)


def ev_days(
    amount: float,
    days: list[str],
    *,
    ii: float = 168.0,
    cmt: int = 1,
    rate: float = 0.0,
) -> list[EventRow]:
    """Weekday dosing — mrgsolve's ``ev_days``.

    Doses land on the listed weekdays of a 168-hour week (m=0, t=24,
    w=48, th=72, f=96, sa=120, s=144):

    >>> from pharmaml.pkpd.events import ev_days
    >>> [r.time for r in ev_days(100.0, ["m", "w", "f"])]
    [0.0, 48.0, 96.0]
    """
    if not days:
        raise DataValidationError("days must not be empty")
    unknown = [d for d in days if d.lower() not in _DAY_OFFSETS]
    if unknown:
        known = ", ".join(sorted(_DAY_OFFSETS))
        raise DataValidationError(
            f"unknown weekday codes {unknown} — the codes are: {known}"
        )
    return [
        ev(amount, time=_DAY_OFFSETS[d.lower()], cmt=cmt, rate=rate, ii=ii)
        for d in days
    ]


@dataclass(slots=True)
class RegimenSolution:
    """Simulation output of :func:`simulate_regimen`.

    ``times``/``states`` mirror :class:`pharmaml.pkpd.ode.ODESolution`
    (observations only — the mrgsolve ``obsonly`` view);
    ``concentration`` converts one compartment's amounts through a
    volume; ``frame`` is the full record table (observations evid 0
    and realized doses, recsort-ordered) with mrgsolve's column
    spelling.
    """

    times: np.ndarray
    states: np.ndarray
    frame: pd.DataFrame = field(repr=False)
    n_steps: int = 0
    n_rejected: int = 0
    method: str = "rk45"

    def concentration(self, compartment: int = 0, *, volume: float = 1.0) -> np.ndarray:
        """Amounts in one 0-based compartment divided by a volume."""
        if volume <= 0:
            raise DataValidationError(f"volume must be > 0, got {volume}")
        return self.states[:, compartment] / volume


def _application_time(row: EventRow) -> float:
    """When the dose physically fires — the record time plus ALAG."""
    return row.time + row.lag


def _cycle_fixed_point(
    rhs,
    y0: np.ndarray,
    row: EventRow,
    *,
    ss_iterations: int,
    ss_rtol: float,
    ss_atol: float,
    method: str,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Iterate one dosing cycle to its fixed point (mrgsolve's ss loop).

    Converges when every compartment changes by less than
    ``ss_rtol·|y| + ss_atol`` between cycles — mrgsolve's rule — capped
    at ``ss_iterations`` (mrgsolve's default 500).
    """
    n = y0.size
    start = _application_time(row)
    if row.rate > 0 and row.amt == 0:
        # Zero-order continuous infusion: the "cycle" is ii hours of the
        # constant rate with no bolus.
        events: tuple[DosingEvent, ...] = (
            DosingEvent(
                time=start,
                amount=row.rate * row.ii,
                compartment=row.cmt - 1,
                duration=row.ii,
            ),
        )
    else:
        events = _row_events(row)
    state = np.zeros(n, dtype=float)
    for _ in range(ss_iterations):
        sol = solve_ode(
            rhs,
            [start, start + row.ii],
            state,
            events,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        nxt = sol.states[-1]
        if np.all(np.abs(nxt - state) <= ss_rtol * np.abs(nxt) + ss_atol):
            return state
        state = nxt
    return state


def _row_events(row: EventRow) -> tuple[DosingEvent, ...]:
    """Translate one realized row into engine :class:`DosingEvent`\\ s."""
    comp = row.cmt - 1
    start = _application_time(row)
    amount = row.amt * row.f
    if row.rate > 0 and row.amt > 0:
        # mrgsolve's dur() = Fn * Amt / Rate — F shrinks the duration.
        return (
            DosingEvent(
                time=start, amount=amount, compartment=comp,
                duration=amount / row.rate,
            ),
        )
    if row.rate > 0 and row.amt == 0:
        # Continuous infusion: run the rate over a nominal ii window.
        return (
            DosingEvent(
                time=start, amount=row.rate * row.ii, compartment=comp,
                duration=row.ii,
            ),
        )
    return (DosingEvent(time=start, amount=amount, compartment=comp),)


def simulate_regimen(
    rhs,
    y0: np.ndarray | list[float],
    times: np.ndarray | list[float],
    events: list[EventRow] | EventRow,
    *,
    recsort: int = 1,
    method: str = "rk45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    ss_iterations: int = 500,
    ss_rtol: float = 1e-8,
    ss_atol: float = 1e-10,
) -> RegimenSolution:
    """Execute the event grammar on the ODE engine.

    evid semantics (mrgsolve ``src/datarecord.cpp``): 1 doses (bolus,
    or infusion when ``rate > 0`` — the duration is ``f·amt/rate`` so
    bioavailability shrinks the duration at a fixed rate); 2 with a
    *negative* cmt zeroes that compartment and cancels its pending
    infusions (the "off" record); 3 resets every compartment to ``y0``
    and cancels infusions; 4 resets then doses (the reset fires once —
    addl copies are demoted to evid 1); 8 *replaces* the compartment's
    amount with ``amt`` (absolute, not additive).  ``lag`` shifts the
    application time; ``ss=1`` jumps the state to the regimen's cycle
    fixed point, ``ss=2`` adds it on top (rxode2 superposition).

    ``recsort`` is mrgsolve's four-way tie-break for doses and
    observations sharing a timestamp: 1 (default) — addl doses before,
    explicit doses after observations; 2 — both after; 3 — both
    before; 4 — addl after, explicit before.  "Before" means the
    observation at that time sees the post-dose state.

    Output times must be strictly increasing; every event application
    time (record time ± lag) must lie within the window.
    """
    if recsort not in (1, 2, 3, 4):
        raise DataValidationError(
            f"recsort must be one of 1, 2, 3, 4, got {recsort}"
        )
    t_grid = np.asarray(times, dtype=float).reshape(-1)
    if t_grid.size < 1:
        raise DataValidationError("times must not be empty")
    if np.any(np.diff(t_grid) <= 0):
        raise DataValidationError("times must be strictly increasing")
    y_start = np.asarray(y0, dtype=float).reshape(-1)
    rows = [events] if isinstance(events, EventRow) else list(events)
    if not rows:
        raise DataValidationError("events must not be empty")
    realized = realize_addl(rows, max_time=float(t_grid[-1]))
    t_end = float(t_grid[-1])

    put_ev_first = recsort in (3, 4)
    addl_ev_first = recsort in (1, 3)

    # --- segment the timeline at state-jump records -------------------
    # Breaks at every record that discontinuously re-shapes the state:
    # resets (evid 3/4), replacements (8), on/off records (2) and the
    # steady-state jumps (ss rows get their own segment start).  The
    # window opens at the earlier of the first observation and the
    # first dose application, so doses before the first observation
    # still integrate.
    breaks: list[float] = []
    for row in realized:
        if row.evid in (2, 3, 4, 8) or row.ss:
            breaks.append(row.time)
    window_start = min(
        float(t_grid[0]),
        min((_application_time(r) for r in realized), default=float(t_grid[0])),
    )
    boundaries = sorted(set([window_start] + breaks + [t_end]))

    states = np.full((t_grid.size, y_start.size), np.nan)
    stats_steps = 0
    stats_rejected = 0
    y = y_start.astype(float).copy()
    # Non-ss bolus mass the engine folded into each recorded observation
    # (segment starts record post-dose); the recsort tie-break below
    # rebalances it.
    applied = np.zeros(t_grid.size, dtype=float)

    for b_idx, b_start in enumerate(boundaries):
        b_end = boundaries[b_idx + 1] if b_idx + 1 < len(boundaries) else t_end
        seg_obs = t_grid[(t_grid > b_start) & (t_grid < b_end)]
        seg_times = np.unique(
            np.concatenate(([b_start], seg_obs, [b_end], t_grid[t_grid == b_start]))
        )
        seg_rows = [
            r
            for r in realized
            if b_start <= r.time < b_end
            or (
                r.time == b_end
                and b_end == t_end
                and (not r.ss or b_start == b_end)
            )
        ]
        # Steady-state rows re-shape the state before their segment runs
        # (they always get their own segment start via `breaks`).
        seg_events: list[DosingEvent] = []
        for row in seg_rows:
            if row.ss:
                fp = _cycle_fixed_point(
                    rhs, y, row,
                    ss_iterations=ss_iterations, ss_rtol=ss_rtol,
                    ss_atol=ss_atol, method=method, rtol=rtol, atol=atol,
                )
                if row.ss == 1:
                    y = fp.copy()
                else:  # ss == 2: superpose on the current state
                    y = y + fp
                if row.rate > 0 and row.amt == 0:
                    # Continuous infusion carries past the window edge.
                    remaining = max(t_end - _application_time(row), 0.0)
                    if remaining > 0:
                        seg_events.append(
                            DosingEvent(
                                time=_application_time(row),
                                amount=row.rate * remaining,
                                compartment=row.cmt - 1,
                                duration=remaining,
                            )
                        )
                else:
                    seg_events.extend(_row_events(replace(row, ss=0)))
                continue
            if row.evid in (2, 3, 8):
                continue  # state records, not doses
            seg_events.extend(_row_events(row))
        # evid 8 replaces; evid 2 (negative cmt) zeroes one compartment;
        # evid 3/4 reset to the initial state.
        for row in seg_rows:
            if row.evid == 8:
                y = y.copy()
                y[row.cmt - 1] = row.amt * row.f
            elif row.evid == 2 and row.cmt < 0:
                y = y.copy()
                y[-row.cmt - 1] = 0.0
            elif row.evid in (3, 4):
                y = y_start.astype(float).copy()
        if b_start > t_end + 1e-12:
            continue
        # Non-ss boluses sitting exactly at the segment start are applied
        # by the engine before the first output row is recorded.
        start_bolus = sum(
            row.amt * row.f
            for row in seg_rows
            if row.evid in (1, 4)
            and row.rate == 0
            and not row.ss
            and abs(_application_time(row) - b_start) < 1e-12
        )
        sol = solve_ode(
            rhs, seg_times, y, tuple(seg_events),
            method=method, rtol=rtol, atol=atol,
        )
        stats_steps += sol.n_steps
        stats_rejected += sol.n_rejected
        end_state = sol.states[-1].copy()
        # Record this segment's observations.
        for i, t in enumerate(t_grid):
            if (
                np.isnan(states[i, 0])
                and t >= b_start - 1e-12
                and t <= b_end + 1e-12
                and (t < b_end - 1e-12 or b_end == t_end)
            ):
                j = int(np.searchsorted(seg_times, t))
                j = min(j, seg_times.size - 1)
                if abs(float(seg_times[j]) - t) < 1e-12:
                    states[i] = sol.states[j]
                    if abs(t - b_start) < 1e-12:
                        applied[i] += start_bolus
        y = end_state

    # --- recsort tie-break ----------------------------------------------
    # Steady-state rows always record post-dose at their jump (both
    # engines' ss-output convention).  Plain records follow recsort:
    # the engine's recordings carry `applied` (segment-start boluses);
    # the target is the pre-dose state plus exactly the mass of the
    # first-class rows (put_ev_first / addl_ev_first — "dose before
    # observation").
    first_mass = np.zeros(t_grid.size, dtype=float)
    for row in realized:
        if row.evid not in (1, 4) or row.rate > 0 or row.ss:
            continue
        first = addl_ev_first if row.from_addl else put_ev_first
        if not first:
            continue
        app = _application_time(row)
        for i in np.where(np.abs(t_grid - app) < 1e-9)[0]:
            first_mass[i] += row.amt * row.f
    for i in range(t_grid.size):
        if first_mass[i] or applied[i]:
            states[i] += first_mass[i] - applied[i]

    # --- the record table (mrgsolve's output columns) ------------------
    # Time-ordered with the recsort tie-break: at a shared timestamp the
    # first-class records (addl doses under addl_ev_first, explicit
    # doses under put_ev_first) precede the observation, the others
    # follow it.
    def _record_group(row: EventRow | None) -> int:
        if row is None:  # an observation record
            return 1
        if row.evid == 0:
            return 1
        first = addl_ev_first if row.from_addl else put_ev_first
        return 0 if first else 2

    entries: list[tuple[float, int, int, dict[str, float | int]]] = []
    for pos, t in enumerate(t_grid):
        entries.append(
            (
                float(t),
                _record_group(None),
                pos,
                {"time": float(t), "evid": 0, "amt": 0.0, "cmt": 0, "ii": 0.0,
                 "addl": 0, "rate": 0.0, "ss": 0},
            )
        )
    for pos, row in enumerate(realized):
        entries.append(
            (
                float(row.time),
                _record_group(row),
                pos,
                {
                    "time": float(row.time),
                    "evid": row.evid,
                    "amt": float(row.amt),
                    "cmt": row.cmt,
                    "ii": float(row.ii),
                    "addl": int(row.addl),
                    "rate": float(row.rate),
                    "ss": int(row.ss),
                },
            )
        )
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    frame = pd.DataFrame.from_records([e[3] for e in entries])

    filled = ~np.isnan(states[:, 0]) if states.shape[1] else None
    if filled is not None and not bool(np.all(filled)):
        missing = [float(t) for t, ok in zip(t_grid, filled) if not ok]
        raise DataValidationError(
            f"output times outside the simulated window: {missing[:3]}"
        )

    return RegimenSolution(
        times=t_grid.copy(),
        states=states,
        frame=frame,
        n_steps=stats_steps,
        n_rejected=stats_rejected,
        method=method,
    )
