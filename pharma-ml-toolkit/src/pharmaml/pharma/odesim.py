"""ODE-based PK/PD simulation — the mrgsolve and RxODE concepts on numpy.

Absorbed in round 100x-176 (the PK/PD simulation round) from two
repositories reviewed file by file:

* **mrgsolve** (metrumresearchgroup/mrgsolve, v2.0.x) — the model-spec
  concept: a small text DSL declares compartments, parameters and
  ``dxdt_<CMT>`` right-hand sides; ``ev()``/``data_set`` records carry
  the dosing semantics (bolus vs. infusion via ``rate`` — an infusion
  runs for ``amt/rate`` with paired on/off records, overlapping
  infusions sum their rates — plus ``ii``/``addl`` additional doses and
  ``evid`` 3 resets); ``$OMEGA`` random effects are drawn once per
  subject as multivariate normal via **eigendecomposition with negative
  eigenvalues clamped to zero** (``src/mrgsolve.cpp::MVGAUSS`` — never
  Cholesky, robust to semi-definite Ω).
* **RxODE** (nlmixrdevelopment/RxODE, v1.1.6) — the ``d/dt(state) = …``
  syntax with ``state(0) =`` initial conditions and plain-assignment
  outputs (``C2 = centr/V2``); ``et()``/``rxSolve`` event tables
  (``et(0, 48, by=0.5)`` grids, ``until``/``addl`` expansion);
  **``linCmt()``** analytic one-/two-compartment solutions with the
  exact closed forms from ``src/lincmt.c`` (``oneCmtKa``,
  ``twoCmtBolus`` — including the hybrid rate constants
  ``λ1,2 = ½[(k10+k12+k21) ± √((k10+k12+k21)² − 4·k21·k10)]``);
  ``transit(n, mtt)`` compartments (the gamma-PDF input of Savic
  2007); and the matrix exponential behind its ``indLin`` solver
  (scaling-and-squaring Padé — the Al-Mohy & Higham 2009 family, as
  ported in ``src/matexp_HM98.c``).

Honest boundaries (what is *not* vendored): LSODA, liblsoda and dop853
stay in their C/Fortran homes — this module integrates with its own
fixed-step **RK4** (plus Euler for the teaching comparison), the event
mesh clamping every step at dose/infusion boundaries so no
discontinuity is ever stepped over; there is no C++ code generation —
the DSL compiles to one Python lambda per model; steady-state
advancing (``ss=1`` superposition iteration) and per-observation EPS
residuals are left to the caller on the solved frames (ETAs per
subject *are* implemented, mrgsolve-style).

Times are hours, amounts share the dose unit (mg by convention) and
volumes are litres — the dimensionless-DSL choice both projects make
for their numeric cores.

Examples
--------
>>> mod = ODEModel(r'''
... # the RxODE flagship demo, transcribed from man/RxODE.Rd
... C2 = centr / V2
... C3 = peri / V3
... d/dt(depot) = -KA * depot
... d/dt(centr) = KA * depot - CL * C2 - Q * C2 + Q * C3
... d/dt(peri) = Q * C2 - Q * C3
... d/dt(eff) = Kin - Kout * (1 - C2 / (EC50 + C2)) * eff
... eff(0) = 1
... KA = 0.291; CL = 18.6; V2 = 40.2; Q = 10.5
... V3 = 297.0; Kin = 1.0; Kout = 1.0; EC50 = 200.0
... ''')
>>> sched = et(amt=10000, cmt='depot') + et(0, 48, by=0.5)
>>> res = solve_ode(mod, sched, h=0.05)
>>> res.frame.shape
(97, 7)
>>> float(res.frame['centr'].max()) > 0
True
>>> bool((res.frame['eff'] > 0).all())
True
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

_trapezoid: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
# numpy 1.x spells it trapz, 2.x renamed it trapezoid -- both supported

__all__ = [
    "Event",
    "EventSchedule",
    "ODEModel",
    "ODEResult",
    "draw_mvnorm",
    "et",
    "lin_cmt_1",
    "lin_cmt_2",
    "matrix_exponential",
    "simulate_subjects",
    "solve_ode",
    "transit_input",
]

# ---------------------------------------------------------------------------
# the numeric namespace every DSL expression is evaluated in
# ---------------------------------------------------------------------------

_lgamma = np.vectorize(math.lgamma, otypes=[float])

#: functions and constants available inside ODE right-hand sides — the
#: RxODE syntax-functions table (``vignettes/syntax-functions.csv``)
#: restricted to the vectorizable numpy core.
_SAFE_NAMESPACE: dict[str, object] = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "ln": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "fabs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "min": np.minimum,
    "max": np.maximum,
    "lgamma": _lgamma,
    "gamma": lambda x: np.exp(_lgamma(np.asarray(x, dtype=float))),
    "logit": lambda x: np.log(x / (1 - x)),
    "expit": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "pi": math.pi,
    "ln2": math.log(2.0),
    "inf": math.inf,
}


# ---------------------------------------------------------------------------
# matrix exponential (RxODE's expm: scaling-and-squaring Padé)
# ---------------------------------------------------------------------------

_PADE6 = np.array(
    [1.0, 0.5, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0]
)


def matrix_exponential(a: np.ndarray, t: float = 1.0) -> np.ndarray:
    """``expm(t·A)`` by Padé(6,6) with scaling-and-squaring.

    This is the RxODE ``indLinMatExpType`` algorithm family
    (``src/matexp_HM98.c`` ports R's ``expm``; EXPOKIT's DGPADM is
    degree-6 irreducible Padé with ``ns = max(0, floor(log2‖tH‖)+2))`` —
    the same recipe): scale ``A`` by a power of two until its infinity
    norm is ``≤ 0.5``, evaluate the Padé fraction
    ``N(b)·N(−b)⁻¹``, then square the result back.

    Parameters
    ----------
    a:
        Square matrix.
    t:
        Time multiplier (``t·A`` is what gets exponentiated).

    Examples
    --------
    >>> a = np.diag([-1.0, -2.0])
    >>> np.round(matrix_exponential(a, 1.0), 8)
    array([[0.36787944, 0.        ],
           [0.        , 0.13533528]])
    >>> import math
    >>> bool(abs(matrix_exponential(a, 2.0)[0, 0] - math.exp(-2)) < 1e-12)
    True
    >>> rot = matrix_exponential(np.array([[0.0, -math.pi], [math.pi, 0.0]]))
    >>> np.allclose(rot, np.array([[math.cos(math.pi), -math.sin(math.pi)],
    ...                             [math.sin(math.pi), math.cos(math.pi)]]))
    True
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix_exponential needs a square matrix, got {a.shape}")
    if t == 0.0:
        return np.eye(a.shape[0])
    scaled = a * float(t)
    norm = np.abs(scaled).sum(axis=1).max()
    s = math.ceil(math.log2(norm / 0.5)) if norm > 0.5 else 0
    b = scaled / (2.0**s)
    b2 = b @ b
    b3 = b2 @ b
    b4 = b2 @ b2
    b5 = b4 @ b
    b6 = b4 @ b2
    eye = np.eye(a.shape[0])
    upper = (
        _PADE6[0] * eye + _PADE6[1] * b + _PADE6[2] * b2 + _PADE6[3] * b3
        + _PADE6[4] * b4 + _PADE6[5] * b5 + _PADE6[6] * b6
    )
    lower = (
        _PADE6[0] * eye - _PADE6[1] * b + _PADE6[2] * b2 - _PADE6[3] * b3
        + _PADE6[4] * b4 - _PADE6[5] * b5 + _PADE6[6] * b6
    )
    result = np.linalg.solve(lower, upper)
    for _ in range(s):
        result = result @ result
    return result


# ---------------------------------------------------------------------------
# events — the et()/ev() semantics of RxODE and mrgsolve
# ---------------------------------------------------------------------------


class Event:
    """One dosing record — the ``et()``/``ev()`` row of RxODE/mrgsolve.

    ``rate > 0`` makes the record an **infusion** running for
    ``amt / rate`` hours (the mrgsolve ``datarecord.cpp`` duration
    rule, ``dur() = Fn·Amt/Rate``); ``rate == 0`` is a bolus.  ``addl``
    schedules identical extra doses every ``ii`` hours after this one;
    ``evid`` 3 is the RxODE reset record (states and active infusions
    are zeroed before the dose lands).
    """

    __slots__ = ("addl", "amt", "cmt", "evid", "ii", "rate", "time")

    def __init__(
        self,
        time: float = 0.0,
        amt: float = 0.0,
        rate: float = 0.0,
        cmt: int | str = 1,
        ii: float = 0.0,
        addl: int = 0,
        evid: int = 1,
    ) -> None:
        self.time = float(time)
        self.amt = float(amt)
        self.rate = float(rate)
        self.cmt = cmt
        self.ii = float(ii)
        self.addl = int(addl)
        self.evid = int(evid)

    def __repr__(self) -> str:  # pragma: no cover - display sugar
        return (
            f"Event(time={self.time:g}, amt={self.amt:g}, rate={self.rate:g}, "
            f"cmt={self.cmt!r}, ii={self.ii:g}, addl={self.addl}, evid={self.evid})"
        )


class EventSchedule:
    """A dosing + observation plan — RxODE's ``et()`` / mrgsolve's ``ev()``.

    Schedules combine with ``+`` (both projects' record concatenation,
    re-sorted by time) and carry a separate sampling grid the solver
    augments with every dosing discontinuity.

    Examples
    --------
    >>> qd = et(amt=10000, ii=24, addl=4)          # 5 daily doses
    >>> len(qd.doses)
    5
    >>> [d.time for d in qd.doses]
    [0.0, 24.0, 48.0, 72.0, 96.0]
    >>> inf = et(amt=3000, rate=30)                # 100-hour infusion
    >>> inf.doses[0].time, round(inf.off_times[0], 6)
    (0.0, 100.0)
    >>> (qd + inf).doses[0].amt
    10000.0
    """

    def __init__(
        self,
        doses: list[Event] | None = None,
        sampling: list[float] | None = None,
    ) -> None:
        self.doses: list[Event] = sorted(doses or [], key=lambda e: (e.time, e.cmt))
        self.sampling: list[float] = sorted(set(sampling or []))

    # -- constructors ----------------------------------------------------
    @staticmethod
    def dose(
        amt: float,
        *,
        time: float = 0.0,
        cmt: int | str = 1,
        rate: float = 0.0,
        ii: float = 0.0,
        addl: int = 0,
        until: float | None = None,
        evid: int = 1,
    ) -> EventSchedule:
        """One dosing record (``et(amt=…)`` / ``ev(amt=…)``).

        ``until`` converts to additional doses the mrgsolve way:
        ``addl = ceil((until − time)/ii) − 1`` (doses strictly before
        ``until``).
        """
        if until is not None:
            if ii <= 0:
                raise ValueError("until= requires ii= (the inter-dose interval)")
            addl = math.ceil((until - time) / ii) - 1
        addl = max(addl, 0)
        if addl and ii <= 0:
            raise ValueError("addl= requires ii= (the inter-dose interval)")
        doses = [
            Event(time=time + k * ii, amt=amt, rate=rate, cmt=cmt, ii=ii, evid=evid)
            for k in range(addl + 1)
        ]
        return EventSchedule(doses=doses)

    @staticmethod
    def obs(times: list[float] | tuple[float, ...] | np.ndarray) -> EventSchedule:
        """Sampling times only (RxODE's numeric ``et(0:24)``)."""
        return EventSchedule(sampling=[float(x) for x in np.asarray(times).ravel()])

    @staticmethod
    def grid(start: float, end: float, delta: float) -> EventSchedule:
        """The mrgsolve ``tgrid(start, end, delta)`` output grid."""
        if delta <= 0 or end < start:
            raise ValueError("grid needs end >= start and delta > 0")
        n = round((end - start) / delta)
        return EventSchedule(sampling=[start + k * delta for k in range(n + 1)])

    # -- combination ------------------------------------------------------
    def __add__(self, other: EventSchedule) -> EventSchedule:
        if not isinstance(other, EventSchedule):
            return NotImplemented
        return EventSchedule(
            doses=self.doses + other.doses,
            sampling=self.sampling + other.sampling,
        )

    # -- derived views ----------------------------------------------------
    @property
    def off_times(self) -> list[float]:
        """Infusion end times (``time + amt/rate``) — mrgsolve's evid-9."""
        return [d.time + d.amt / d.rate for d in self.doses if d.rate > 0]

    def times(self) -> list[float]:
        """Every discontinuity + sampling time, sorted and deduplicated."""
        all_times = set(self.sampling) | {d.time for d in self.doses}
        all_times.update(self.off_times)
        return sorted(t for t in all_times if t >= 0)

    def output_grid(self, default_end: float = 24.0, default_delta: float = 1.0) -> list[float]:
        """Observation times, defaulting to a 0–24 h grid (mrgsolve)."""
        if self.sampling:
            return self.sampling
        n = round(default_end / default_delta)
        return [k * default_delta for k in range(n + 1)]

    def __repr__(self) -> str:  # pragma: no cover - display sugar
        return f"EventSchedule(doses={len(self.doses)}, sampling={len(self.sampling)})"


def et(*args: float, by: float | None = None, length_out: int | None = None,
       **kwargs: object) -> EventSchedule:
    """Build one schedule piece — RxODE's ``et()`` dual semantics.

    Numeric positional arguments are sampling times: ``et(0, 24, 0.5)``
    samples at exactly those three values, while the keyword grid
    forms build ``seq``-style spans — ``et(0, 48, by=0.5)`` steps by
    0.5 and ``et(0, 48, length_out=97)`` interpolates — exactly like
    RxODE.  Dose keywords build a record (``et(amt=10000,
    cmt='depot', ii=24, addl=4)`` — see :meth:`EventSchedule.dose`).
    Pieces combine with ``+``.

    Examples
    --------
    >>> sched = et(0, 48, by=0.5) + et(amt=500, cmt=2)
    >>> len(sched.sampling)
    97
    >>> sched.doses[0].amt
    500.0
    >>> et(0, 12, length_out=5).sampling
    [0.0, 3.0, 6.0, 9.0, 12.0]
    """
    if (by is not None or length_out is not None) and kwargs:
        raise ValueError("grid keywords (by=/length_out=) cannot mix with dose keywords")
    if by is not None or length_out is not None:
        if len(args) == 1:
            start, end = 0.0, float(args[0])
        elif len(args) == 2:
            start, end = float(args[0]), float(args[1])
        else:
            raise ValueError("grid form takes one or two numeric bounds")
        if by is not None:
            if by <= 0:
                raise ValueError("by= must be positive")
            n = math.floor((end - start) / by + 1e-9)
            sampling = [round(start + k * by, 10) for k in range(n + 1)]
        else:
            if length_out is None or length_out < 2:
                raise ValueError("length_out= must be >= 2")
            sampling = [round(start + (end - start) * k / (length_out - 1), 10)
                        for k in range(length_out)]
        return EventSchedule(sampling=sampling)
    numeric = [float(x) for x in np.asarray(args, dtype=float).ravel()] if args else []
    if numeric and kwargs:
        raise ValueError("et() takes sampling times or dose keywords, not both")
    if numeric:
        return EventSchedule(sampling=numeric)
    dose_kwargs = dict(kwargs)
    amt = dose_kwargs.pop("amt", None)
    if amt is None:
        return EventSchedule()
    return EventSchedule.dose(float(amt), **dose_kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the model DSL — RxODE syntax, mrgsolve blocks flattened to lines
# ---------------------------------------------------------------------------

_DD_LINE = re.compile(r"^d/dt\s*\(\s*([A-Za-z_]\w*)\s*\)\s*=\s*(.+?)\s*$")
_INIT_LINE = re.compile(r"^([A-Za-z_]\w*)\s*\(\s*0\s*\)\s*=\s*(.+?)\s*$")
_CMT_LINE = re.compile(r"^cmt\s*\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)$")
_ASSIGN_LINE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
_IDENT = re.compile(r"[A-Za-z_]\w*")


class ODEModel:
    """A tiny ODE model spec — RxODE's ``RxODE({...})`` string syntax.

    The grammar (one statement per line, ``;``-separated statements on
    a line are fine, ``#``/``//`` comments anywhere):

    .. code-block:: text

        cmt(depot, centr, peri)      # declare compartments (init 0)
        depot(0) = 1                 # initial condition
        KA = 0.291                   # parameter (constant expression)
        C2 = centr / V2              # output (references a state or t)
        d/dt(depot) = -KA * depot    # the ODE itself

    A plain assignment whose right-hand side resolves to a constant
    over the declared parameters is a *parameter*; one that references
    a state, ``t`` or an unknown name (an ETA) becomes an *output*
    evaluated at every observation — exactly how RxODE separates LHS
    variables from ``theta``.  Outputs feed the ODEs they are used in
    (dependency-ordered at compile time).  The right-hand sides
    compile into a single Python lambda over the state vector, so a
    4,800-step simulation costs one ``eval`` in total, not one per
    step.

    Parameters
    ----------
    source:
        The model code.
    name:
        Optional label carried on the model (the mrgsolve ``$PROB``).

    Examples
    --------
    >>> mod = ODEModel('d/dt(a) = -k * a; a(0) = 10; k = 0.3')
    >>> mod.states
    ['a']
    >>> mod.params
    {'k': 0.3}
    """

    def __init__(self, source: str, *, name: str = "model") -> None:
        self.name = name
        self.states: list[str] = []
        self._state_set: set[str] = set()
        self.params: dict[str, float] = {}
        self.inits: dict[str, float] = {}
        self._odes: dict[str, str] = {}
        self.outputs: list[tuple[str, str]] = []
        self._parse(source)
        if not self._odes:
            raise ValueError(
                "the model has no d/dt(...) lines — mrgsolve calls that "
                "DXDTZERO, this engine refuses it"
            )
        unknown = [s for s in self._odes if s not in self._state_set]
        if unknown:
            raise ValueError(f"ODE for undeclared state {unknown[0]!r}")
        self._check_references()

    # -- parsing -----------------------------------------------------------
    def _parse(self, source: str) -> None:
        assignments: list[tuple[str, str]] = []
        for raw in source.splitlines():
            for marker in ("#", "//"):
                cut = raw.find(marker)
                if cut >= 0:
                    raw = raw[:cut]
            for fragment in raw.split(";"):
                line = fragment.strip()
                if not line:
                    continue
                m = _CMT_LINE.match(line)
                if m:
                    for token in m.group(1).split(","):
                        state = token.strip()
                        if state not in self._state_set:
                            self._state_set.add(state)
                            self.states.append(state)
                    continue
                m = _DD_LINE.match(line)
                if m:
                    state, rhs = m.group(1), m.group(2)
                    if state not in self._state_set:
                        self._state_set.add(state)
                        self.states.append(state)
                    self._odes[state] = rhs
                    continue
                m = _INIT_LINE.match(line)
                if m:
                    state, rhs = m.group(1), m.group(2)
                    if state not in self._state_set:
                        self._state_set.add(state)
                        self.states.append(state)
                    assignments.append((state + "(0)", rhs))
                    continue
                m = _ASSIGN_LINE.match(line)
                if m:
                    lhs, rhs = m.group(1), m.group(2)
                    if lhs in self._state_set:
                        raise ValueError(
                            f"{lhs!r} is a compartment — set its start via "
                            f"{lhs}(0) = …"
                        )
                    assignments.append((lhs, rhs))
                    continue
                raise ValueError(f"unparsable model line: {line!r}")
        # classify assignments: constants (params) vs outputs, by fixpoint
        pending = assignments
        while True:
            progressed = False
            rest: list[tuple[str, str]] = []
            for lhs, rhs in pending:
                if lhs.endswith("(0)"):
                    try:
                        self.inits[lhs[:-3]] = float(
                            eval(rhs, {"__builtins__": {}}, dict(self.params))
                        )
                    except (NameError, TypeError):
                        rest.append((lhs, rhs))
                    continue
                if any(state in _IDENT.findall(rhs) for state in self._state_set) or re.search(
                    r"\bt\b", rhs
                ):
                    self.outputs.append((lhs, rhs))
                    continue
                try:
                    self.params[lhs] = float(
                        eval(rhs, {"__builtins__": {}}, dict(self.params))
                    )
                    progressed = True
                except (NameError, TypeError):
                    rest.append((lhs, rhs))
            pending = rest
            if not pending or not progressed:
                break
        for lhs, rhs in pending:
            if lhs.endswith("(0)"):
                raise ValueError(
                    f"initial condition {lhs!r} is not a constant over the "
                    f"declared parameters ({rhs!r})"
                )
            self.outputs.append((lhs, rhs))

    def _check_references(self) -> None:
        """Every identifier must be a state, param, output or builtin."""
        known = (
            self._state_set
            | set(self.params)
            | {name for name, _ in self.outputs}
            | set(_SAFE_NAMESPACE)
            | {"t"}
        )

        def _is_eta(name: str) -> bool:
            # mrgsolve's ETA(n)/EPS(n) convention lands as ETA1 … at
            # compile time (simulate_subjects injects the draws)
            return re.fullmatch(r"ET[AP]\d+", name) is not None

        for state, rhs in self._odes.items():
            for ident in set(_IDENT.findall(rhs)):
                if ident not in known and not _is_eta(ident):
                    raise ValueError(
                        f"right-hand side of d/dt({state}) references unknown "
                        f"name {ident!r} (not a parameter, state or output)"
                    )
        for name, rhs in self.outputs:
            for ident in set(_IDENT.findall(rhs)):
                if ident not in known and not _is_eta(ident):
                    raise ValueError(
                        f"output {name!r} references unknown name {ident!r}"
                    )

    # -- compilation ---------------------------------------------------------
    def _ordered_outputs(self) -> list[tuple[str, str]]:
        """Outputs topologically sorted so dependents follow dependencies."""
        names = {name for name, _ in self.outputs}
        placed: list[tuple[str, str]] = []
        remaining = list(self.outputs)
        while remaining:
            progressed = False
            rest: list[tuple[str, str]] = []
            for name, rhs in remaining:
                refs = set(_IDENT.findall(rhs))
                if refs & ({placed_name for placed_name, _ in placed} - {name}) and (
                    refs & names - {name} - {p for p, _ in placed}
                ):
                    rest.append((name, rhs))
                    continue
                placed.append((name, rhs))
                progressed = True
            remaining = rest
            if remaining and not progressed:
                cycle = ", ".join(name for name, _ in remaining)
                raise ValueError(f"output definitions form a cycle: {cycle}")
        return placed

    def rhs_function(self, params: dict[str, float] | None = None) -> object:
        """Compile the right-hand side into ``f(t, y, rates) -> dydt``.

        ``params`` may override or extend the parsed parameters — the
        hook :func:`simulate_subjects` uses to inject ``ETA1 …`` draws.
        Outputs the ODEs depend on are evaluated first through a
        nested-lambda application (the dependency-ordered prefix).
        """
        merged: dict[str, float] = dict(self.params)
        if params:
            merged.update(params)
        ordered = self._ordered_outputs()
        scope = dict(_SAFE_NAMESPACE)
        scope.update(merged)
        ode_list = ", ".join(self._odes[s] for s in self.states)
        if ordered:
            out_names = ", ".join(name for name, _ in ordered)
            out_exprs = ", ".join(expr for _, expr in ordered)
            body = f"(lambda {out_names}: [{ode_list}])({out_exprs})"
        else:
            body = f"[{ode_list}]"
        src = f"lambda t, {', '.join(self.states)}: {body}"
        fn = eval(src, scope)

        def rhs(t: float, y: np.ndarray, rates: np.ndarray) -> np.ndarray:
            deriv = np.asarray(fn(t, *y), dtype=float)
            if rates.any():
                deriv = deriv + rates
            return deriv

        return rhs

    def output_function(self, params: dict[str, float] | None = None) -> object:
        """Compile the output assignments (``None`` when there are none)."""
        ordered = self._ordered_outputs()
        if not ordered:
            return None
        merged: dict[str, float] = dict(self.params)
        if params:
            merged.update(params)
        scope = dict(_SAFE_NAMESPACE)
        scope.update(merged)
        # evaluate layer by layer: each output may use the previous ones
        layers: list[list[tuple[str, str]]] = []
        pending = list(ordered)
        while pending:
            layer: list[tuple[str, str]] = []
            rest: list[tuple[str, str]] = []
            done = {n for l in layers for n, _ in l} | {n for n, _ in layer}
            for name, rhs in pending:
                refs = set(_IDENT.findall(rhs))
                if refs & ({n for n, _ in pending} - {name} - done):
                    rest.append((name, rhs))
                else:
                    layer.append((name, rhs))
            if layer:
                layers.append(layer)
            pending = rest
            if pending and not layer:
                cycle = ", ".join(n for n, _ in pending)
                raise ValueError(f"output definitions form a cycle: {cycle}")
        all_names = [name for layer in layers for name, _ in layer]
        eval_globals: dict[str, object] = {"__builtins__": {}}
        eval_globals.update(_SAFE_NAMESPACE)

        def outputs(t: float, y: np.ndarray) -> list[float]:
            env: dict[str, float] = dict(merged)
            for name_layer in layers:
                for name, rhs in name_layer:
                    local = dict(env)
                    local["t"] = t
                    local.update(dict(zip(self.states, (float(v) for v in y))))
                    env[name] = float(eval(rhs, eval_globals, local))
            return [env[name] for name in all_names]

        outputs._names = all_names  # type: ignore[attr-defined]
        return outputs

    def initial_vector(self, params: dict[str, float] | None = None) -> np.ndarray:
        """The state vector at t=0 (declared inits, else zero)."""
        y0 = np.zeros(len(self.states))
        for state, value in self.inits.items():
            y0[self.states.index(state)] = value
        return y0

    def compartment_index(self, cmt: int | str) -> int:
        """Resolve a compartment reference (name or 1-based index)."""
        if isinstance(cmt, int):
            if not 1 <= cmt <= len(self.states):
                raise ValueError(
                    f"cmt={cmt} out of range (model has {len(self.states)} states)"
                )
            return cmt - 1
        try:
            return self.states.index(cmt)
        except ValueError as exc:
            raise ValueError(
                f"unknown compartment {cmt!r} — model states: {self.states}"
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover - display sugar
        return (
            f"ODEModel(name={self.name!r}, states={self.states}, "
            f"params={len(self.params)}, outputs={len(self.outputs)})"
        )


class ODEResult:
    """A solved simulation — ``rxSolve``'s data frame, numpy-shaped."""

    def __init__(
        self,
        times: np.ndarray,
        states: np.ndarray,
        state_names: list[str],
        outputs: np.ndarray,
        output_names: list[str],
    ) -> None:
        self.times = times
        self.states = states
        self.state_names = state_names
        self.outputs = outputs
        self.output_names = output_names

    @property
    def frame(self) -> pd.DataFrame:
        """``time | states | outputs`` — the RxODE return table."""
        data: dict[str, object] = {"time": self.times}
        for j, name in enumerate(self.state_names):
            data[name] = self.states[:, j]
        for j, name in enumerate(self.output_names):
            data[name] = self.outputs[:, j]
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return int(self.times.size)


# ---------------------------------------------------------------------------
# the solver — fixed-step RK4 over an event-clamped mesh
# ---------------------------------------------------------------------------


def _euler_step(f: object, t: float, y: np.ndarray, h: float, rates: np.ndarray) -> np.ndarray:
    return y + h * f(t, y, rates)  # type: ignore[operator]


def _rk4_step(f: object, t: float, y: np.ndarray, h: float, rates: np.ndarray) -> np.ndarray:
    k1 = f(t, y, rates)  # type: ignore[operator]
    k2 = f(t + h / 2.0, y + h / 2.0 * k1, rates)  # type: ignore[operator]
    k3 = f(t + h / 2.0, y + h / 2.0 * k2, rates)  # type: ignore[operator]
    k4 = f(t + h, y + h * k3, rates)  # type: ignore[operator]
    return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_ode(
    model: ODEModel,
    schedule: EventSchedule | None = None,
    *,
    times: list[float] | np.ndarray | None = None,
    params: dict[str, float] | None = None,
    method: str = "rk4",
    h: float = 0.02,
) -> ODEResult:
    """Integrate a model over an event schedule — the ``rxSolve`` step.

    The event mesh (every sampling time, dose time and infusion-off
    time) is integrated with fixed steps of at most ``h``, the final
    sub-step clamped so no discontinuity is ever stepped over; boluses
    jump the state at their mesh point, infusions add their rate to
    the dosed compartment's derivative between the paired on/off times
    (overlapping infusions sum — the mrgsolve ``rate_add`` rule), and
    ``evid`` 3 records zero the state and active rates first.

    Parameters
    ----------
    model:
        The :class:`ODEModel` to integrate.
    schedule:
        Dosing + sampling plan (``None`` for pure observation).
    times:
        Explicit output grid (merged into the schedule).
    params:
        Parameter overrides (ETAs land here).
    method:
        ``'rk4'`` (default) or ``'euler'`` — the teaching comparison.
    h:
        Maximum integration step (hours).

    Examples
    --------
    >>> import numpy as np
    >>> mod = ODEModel('d/dt(a) = -k * a; a(0) = 8; k = 0.25')
    >>> res = solve_ode(mod, times=np.linspace(0, 8, 9), h=0.05)
    >>> np.allclose(res.states[:, 0], 8 * np.exp(-0.25 * res.times), atol=1e-5)
    True
    >>> bolus = et(amt=5, cmt='a') + et(0, 4, 1)
    >>> res2 = solve_ode(mod, bolus, h=0.05)
    >>> round(float(res2.states[0, 0]), 6)
    13.0
    """
    if method not in ("rk4", "euler"):
        raise ValueError(f"method must be 'rk4' or 'euler', got {method!r}")
    if h <= 0:
        raise ValueError("h must be positive")
    stepper = _rk4_step if method == "rk4" else _euler_step
    schedule = schedule or EventSchedule()
    if times is not None:
        schedule = schedule + EventSchedule.obs(list(np.asarray(times).ravel()))
    output_times = schedule.output_grid()
    mesh = schedule.times()
    if not mesh or mesh[0] > 0:
        mesh = [0.0] + [t for t in mesh if t > 0]

    rhs = model.rhs_function(params)
    out_fn = model.output_function(params)
    y = model.initial_vector(params)

    rates = np.zeros(len(model.states))
    infusion_off: dict[float, list[tuple[int, float]]] = {}
    for d in schedule.doses:
        if d.rate > 0:
            off = round(d.time + d.amt / d.rate, 9)
            infusion_off.setdefault(off, []).append(
                (model.compartment_index(d.cmt), d.rate)
            )
    doses_at: dict[float, list[Event]] = {}
    for d in schedule.doses:
        doses_at.setdefault(d.time, []).append(d)

    def apply_events(t: float) -> None:
        for off_list in infusion_off.get(round(t, 9), []):
            rates[off_list[0]] -= off_list[1]
        for event in doses_at.get(t, []):
            cmt_idx = model.compartment_index(event.cmt)
            if event.evid == 3:
                y[:] = 0.0
                rates[:] = 0.0
            if event.rate > 0:
                rates[cmt_idx] += event.rate
            elif event.evid in (1, 4):
                y[cmt_idx] += event.amt

    wanted = set(output_times)
    recorded: list[tuple[float, np.ndarray, list[float]]] = []

    t_now = mesh[0]
    apply_events(t_now)
    if 0.0 in wanted:
        recorded.append((0.0, y.copy(), _eval_outputs(out_fn, 0.0, y)))

    for t_next in mesh[1:]:
        while t_next - t_now > 1e-12:
            step = min(h, t_next - t_now)
            y = stepper(rhs, t_now, y, step, rates)
            t_now += step
        t_now = t_next
        apply_events(t_next)
        if t_next in wanted:
            recorded.append((t_next, y.copy(), _eval_outputs(out_fn, t_next, y)))

    out_times = np.array([r[0] for r in recorded], dtype=float)
    out_states = (
        np.vstack([r[1] for r in recorded])
        if recorded
        else np.zeros((0, len(model.states)))
    )
    if recorded and out_fn is not None and recorded[0][2]:
        out_values = np.array([r[2] for r in recorded], dtype=float)
    else:
        out_values = np.zeros((len(recorded), 0))
    out_names = [name for name, _ in model.outputs]
    return ODEResult(out_times, out_states, list(model.states), out_values, out_names)


def _eval_outputs(out_fn: object, t: float, y: np.ndarray) -> list[float]:
    if out_fn is None:
        return []
    return out_fn(t, y)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# analytic linear compartments — RxODE's linCmt()
# ---------------------------------------------------------------------------


def _depot_target(cmt: int | str, has_depot: bool) -> bool:
    """Does this record dose the depot (rather than central)?"""
    if not has_depot:
        return False
    if isinstance(cmt, int):
        return cmt == 1
    return str(cmt).strip().lower() in ("depot", "gut", "ev", "1", "cmt1")


def lin_cmt_1(
    schedule: EventSchedule,
    times: list[float] | np.ndarray,
    cl: float,
    v: float,
    ka: float | None = None,
    bioav: float = 1.0,
) -> np.ndarray:
    """Analytic one-compartment concentrations — RxODE ``oneCmt*``.

    Intravenous (``ka=None``) or first-order absorption, bolus or
    infusion, with exact superposition over every record of the
    schedule — the closed forms of ``src/lincmt.c``: bolus decay
    ``A1 = A1last·e^(−kel·t)``; absorption
    ``A_depot = A_d,last·e^(−ka·t)`` and
    ``A_central = A_d,last·ka/(ka−kel)·(e^(−kel·t)−e^(−ka·t)) +
    A_c,last·e^(−kel·t)``; an infusion of rate ``r``
    contributes ``(r/kel)·(1−e^(−kel·t))`` while it runs and its
    terminal value decaying afterwards.  Infusions enter the central
    compartment; boluses may target the depot when ``ka`` is given.

    Examples
    --------
    >>> import numpy as np, math
    >>> ts = np.linspace(0, 48, 481)
    >>> conc = lin_cmt_1(et(amt=100), ts, cl=5, v=50)
    >>> bool(abs(conc[0] - 2.0) < 1e-12)
    True
    >>> bool(abs(conc[-1] - 2.0 * math.exp(-4.8)) < 1e-12)
    True
    >>> from pharmaml.pharma.pk import auc_trapezoid
    >>> bool(abs(auc_trapezoid(ts, conc) + conc[-1] / 0.1 - 100 / 5) < 0.01)
    True
    >>> c_oral = lin_cmt_1(et(amt=100), ts, cl=5, v=50, ka=1.0)
    >>> bool(abs(ts[int(np.argmax(c_oral))] - math.log(10) / 0.9) < 0.1)
    True
    """
    kel = float(cl) / float(v)
    if ka is not None and abs(float(ka) - kel) < 1e-12:
        raise ValueError("ka == kel makes the closed form singular (flip-flop case)")
    ka_v = float(ka) if ka is not None else None
    ts = np.asarray(times, dtype=float)
    schedule = schedule + EventSchedule.obs(list(ts))
    mesh = schedule.times() or [0.0]
    doses_at: dict[float, list[Event]] = {}
    for d in schedule.doses:
        doses_at.setdefault(d.time, []).append(d)

    a_d = 0.0  # depot amount
    a_c = 0.0  # central amount
    out = np.full(ts.size, np.nan)
    wanted = {round(float(t), 9): i for i, t in enumerate(ts)}

    def record(t: float) -> None:
        slot = wanted.get(round(float(t), 9))
        if slot is not None and np.isnan(out[slot]):
            out[slot] = a_c / float(v)

    def apply_events(t: float) -> None:
        nonlocal a_d, a_c
        for event in doses_at.get(t, []):
            if event.evid == 3:
                a_d = a_c = 0.0
            if event.rate == 0 and event.evid in (1, 4):
                if _depot_target(event.cmt, ka_v is not None):
                    a_d += bioav * event.amt
                else:
                    a_c += bioav * event.amt

    t_now = mesh[0]
    apply_events(t_now)
    record(t_now)
    for t_next in mesh[1:]:
        dt = t_next - t_now
        if dt > 1e-15:
            if ka_v is None:
                a_c *= math.exp(-kel * dt)
            else:
                a_c = (
                    a_d * ka_v / (ka_v - kel)
                    * (math.exp(-kel * dt) - math.exp(-ka_v * dt))
                    + a_c * math.exp(-kel * dt)
                )
                a_d *= math.exp(-ka_v * dt)
        t_now = t_next
        apply_events(t_next)
        record(t_next)
    return out + _infusion_response(ts, schedule, kel) / float(v)


def _infusion_response(ts: np.ndarray, schedule: EventSchedule, kel: float) -> np.ndarray:
    """Superposed amount response of every infusion record (1-cmt)."""
    contribution = np.zeros_like(ts)
    for d in schedule.doses:
        if d.rate <= 0:
            continue
        t_on, t_off = d.time, d.time + d.amt / d.rate
        after = ts > t_on
        tau = np.minimum(ts[after], t_off) - t_on
        amount = d.rate / kel * (1.0 - np.exp(-kel * tau))
        past = ts > t_off
        if past.any():
            tail = d.rate / kel * (1.0 - math.exp(-kel * (t_off - t_on)))
            amount[past[after]] = tail  # replaced below with decayed tail
            amount[past[after]] = tail * np.exp(-kel * (ts[past] - t_off))
        contribution[after] += amount
    return contribution


def lin_cmt_2(
    schedule: EventSchedule,
    times: list[float] | np.ndarray,
    cl: float,
    v2: float,
    q: float,
    v3: float,
    ka: float | None = None,
    bioav: float = 1.0,
) -> np.ndarray:
    """Analytic two-compartment concentrations — the ``twoCmt*`` forms.

    Implemented exactly as RxODE's matrix-exponential route: the
    micro-constants ``kel = CL/V2``, ``k12 = Q/V2``, ``k21 = Q/V3``
    build the hybrid eigenvalues
    ``λ1,2 = ½[(kel+k12+k21) ± √((kel+k12+k21)² − 4·k21·kel)]``
    (``src/lincmt.c`` line 1223), and every mesh segment advances the
    amount vector through :func:`matrix_exponential` —
    ``y ← expm(A·Δt)·y + A⁻¹(expm(A·Δt) − I)·r`` for the active
    infusion rates ``r`` — reproducing the polyexponential closed
    forms with superposition for free.  ``cmt`` 1 / ``'depot'`` doses
    the absorption compartment when ``ka`` is given, else the central
    compartment; ``cmt`` 2 / ``'centr'`` always doses central.

    Examples
    --------
    >>> import numpy as np, math
    >>> ts = np.linspace(0, 96, 481)
    >>> sched = et(amt=100, cmt='depot') + et(0, 96, by=0.2)
    >>> conc = lin_cmt_2(sched, ts, cl=5, v2=50, q=10, v3=100, ka=0.8)
    >>> from pharmaml.pharma.pk import auc_trapezoid
    >>> kel, k12, k21 = 5 / 50, 10 / 50, 10 / 100
    >>> lam1 = ((kel + k12 + k21) - math.sqrt((kel + k12 + k21) ** 2 - 4 * k21 * kel)) / 2
    >>> auc = auc_trapezoid(ts, conc) + conc[-1] / lam1
    >>> bool(abs(auc - 100 / 5) < 0.3)            # AUC(inf) = F·D/CL
    True
    >>> slope = np.polyfit(ts[-160:], np.log(conc[-160:]), 1)[0]
    >>> bool(abs(-slope - lam1) < 0.02)            # terminal slope = -lambda1
    True
    """
    kel = float(cl) / float(v2)
    k12 = float(q) / float(v2)
    k21 = float(q) / float(v3)
    n_states = 3 if ka is not None else 2
    a = np.zeros((n_states, n_states))
    if ka is not None:
        a[0, 0] = -float(ka)
        a[1, 0] = float(ka)
        a[1, 1] = -(kel + k12)
        a[1, 2] = k21
        a[2, 1] = k12
        a[2, 2] = -k21
    else:
        a[0, 0] = -(kel + k12)
        a[0, 1] = k21
        a[1, 0] = k12
        a[1, 1] = -k21

    ts = np.asarray(times, dtype=float)
    schedule = schedule + EventSchedule.obs(list(ts))
    mesh = schedule.times() or [0.0]
    doses_at: dict[float, list[Event]] = {}
    for d in schedule.doses:
        doses_at.setdefault(d.time, []).append(d)

    y = np.zeros(n_states)
    infusion_windows: list[tuple[float, float, int, float]] = []
    for d in schedule.doses:
        if d.rate > 0:
            cmt_idx = 0 if _depot_target(d.cmt, ka is not None) else (
                n_states - 2 if ka is not None else 0
            )
            infusion_windows.append(
                (d.time, d.time + d.amt / d.rate, cmt_idx, d.rate)
            )

    wanted = {round(float(t), 9): i for i, t in enumerate(ts)}
    out = np.full(ts.size, np.nan)

    def record(t: float) -> None:
        slot = wanted.get(round(float(t), 9))
        if slot is not None and np.isnan(out[slot]):
            out[slot] = y[1 if ka is not None else 0] / float(v2)

    def active_rates(t: float) -> np.ndarray:
        r = np.zeros(n_states)
        for t_on, t_off, cmt_idx, rate in infusion_windows:
            if t_on - 1e-12 <= t < t_off - 1e-12:
                r[cmt_idx] += rate
        return r

    def apply_events(t: float) -> None:
        for event in doses_at.get(t, []):
            if event.evid == 3:
                y[:] = 0.0
                continue
            if event.rate == 0 and event.evid in (1, 4):
                if _depot_target(event.cmt, ka is not None):
                    y[0] += bioav * event.amt
                else:
                    y[n_states - 2 if ka is not None else 0] += bioav * event.amt

    eye = np.eye(n_states)
    t_now = mesh[0]
    apply_events(t_now)
    record(t_now)
    for t_next in mesh[1:]:
        dt = t_next - t_now
        if dt > 1e-15:
            r = active_rates((t_now + t_next) / 2.0)
            phi = matrix_exponential(a, dt)
            y = phi @ y
            if r.any():
                y = y + np.linalg.solve(a, (phi - eye) @ r)
        t_now = t_next
        apply_events(t_next)
        record(t_next)
    return out


# ---------------------------------------------------------------------------
# transit compartments and random effects
# ---------------------------------------------------------------------------


def transit_input(
    t: float | np.ndarray,
    n: float,
    mtt: float,
    bioav: float = 1.0,
    dose: float = 1.0,
) -> np.ndarray:
    """The analytic transit-compartment input — RxODE's ``transit()``.

    ``ktr = (n+1)/mtt`` (Savic 2007: a chain of ``n`` transit
    compartments with mean transit time ``mtt`` has an Erlang/gamma
    impulse response), and the input *rate* into the central
    compartment after a dose ``podo`` is

    .. code-block:: text

        bioav · podo · ktr · (ktr·t)^n · e^(−ktr·t) / Γ(n+1)

    — literally ``exp(log(bio*podo)+log(ktr)+n*log(ktr*t)-ktr*t-
    lgammafn(n+1))`` from RxODE's ``transit-compartments.Rmd``.

    Examples
    --------
    >>> import numpy as np
    >>> ts = np.linspace(0, 5, 5001)
    >>> rate = transit_input(ts, n=3, mtt=0.5, dose=20.0)
    >>> bool(abs(_trapezoid(rate, ts) - 20.0) < 1e-3)   # all the dose lands
    True
    >>> float(rate[0])
    0.0
    """
    if mtt <= 0 or n < 0:
        raise ValueError("transit needs mtt > 0 and n >= 0")
    t_arr = np.asarray(t, dtype=float)
    ktr = (float(n) + 1.0) / float(mtt)
    positive = t_arr > 0
    log_rate = np.full(t_arr.shape, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rate[positive] = (
            math.log(bioav * dose)
            + math.log(ktr)
            + float(n) * np.log(ktr * t_arr[positive])
            - ktr * t_arr[positive]
            - _lgamma(float(n) + 1.0)
        )
    return np.exp(log_rate)


def draw_mvnorm(
    cov: np.ndarray,
    size: int,
    seed: int | None = None,
) -> np.ndarray:
    """Draw ``size`` rows from MVN(0, cov) — mrgsolve's ``MVGAUSS``.

    Eigendecomposition with negative eigenvalues clamped to zero
    (``src/mrgsolve.cpp::MVGAUSS`` — robust to semi-definite ``Ω``
    where a Cholesky factorisation would fail): the draws are
    ``V·diag(√λ⁺)·z`` with ``z`` standard normal.

    Examples
    --------
    >>> import numpy as np
    >>> draws = draw_mvnorm(np.diag([4.0, 9.0]), 20000, seed=11)
    >>> draws.shape
    (20000, 2)
    >>> again = draw_mvnorm(np.diag([4.0, 9.0]), 20000, seed=11)
    >>> bool(np.array_equal(draws, again))
    True
    >>> v = draws.var(axis=0)
    >>> bool(abs(v[0] - 4) < 0.4 and abs(v[1] - 9) < 0.8)
    True
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got {cov.shape}")
    if size < 0:
        raise ValueError("size must be >= 0")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    clamped = np.clip(eigenvalues, 0.0, None)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((size, cov.shape[0]))
    return z @ (eigenvectors * np.sqrt(clamped)).T


def simulate_subjects(
    model: ODEModel,
    schedule: EventSchedule,
    omega: np.ndarray,
    n_sub: int,
    *,
    seed: int | None = None,
    method: str = "rk4",
    h: float = 0.02,
) -> pd.DataFrame:
    """Simulate ``n_sub`` subjects — ETAs per subject, mrgsolve style.

    ``omega`` (the ``$OMEGA`` matrix) yields one MVN(0, Ω) draw per
    subject (:func:`draw_mvnorm` — the eigendecomposition sampler);
    the draws are exposed to the model as parameters ``ETA1 … ETAk``
    so right-hand sides can write ``CL = TVCL * exp(ETA1)`` exactly
    like mrgsolve's ``CLi = exp(log(CL) + ECL)``.  Per-observation
    EPSILONs (``$SIGMA``) are deliberately not drawn — apply residual
    error to the returned frame, where it belongs.

    Examples
    --------
    >>> mod = ODEModel(r'''
    ... d/dt(cent) = -(CL / V) * cent
    ... CL = TVCL * exp(ETA1)
    ... TVCL = 5; V = 50
    ... ''')
    >>> sched = et(amt=100, cmt='cent') + et(0, 12, 2)
    >>> frame = simulate_subjects(mod, sched, np.diag([0.09]), 4, seed=5)
    >>> list(frame.columns[:4])
    ['subject', 'time', 'cent', 'CL']
    >>> frame['subject'].nunique()
    4
    """
    omega = np.asarray(omega, dtype=float)
    etas = draw_mvnorm(omega, n_sub, seed=seed)
    frames = []
    for i in range(n_sub):
        params = {f"ETA{k + 1}": float(etas[i, k]) for k in range(omega.shape[0])}
        result = solve_ode(model, schedule, params=params, method=method, h=h)
        subject = result.frame.copy()
        subject.insert(0, "subject", i + 1)
        for name, value in params.items():
            subject[name] = value
        frames.append(subject)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
