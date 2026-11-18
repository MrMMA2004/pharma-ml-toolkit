"""Exact linear compartment models via the matrix exponential (linCmt).

``rxode2``/``nlmixr2`` ship a compiled ``linCmt()`` solver for linear
PK systems; this module is its numpy-native twin.  A linear
compartment system ``dA/dt = K·A + b`` (``K`` the transfer/elimination
rate matrix, ``b`` a constant input) has the *exact* solution
``A(t) = exp(K t) A(0)`` — and with the classic augmented-matrix trick
``z = [A; 1]``, ``Z = [[K, b], [0, 0]]`` the constant input is folded
into the same exponential.  Bolus doses are impulses between segments,
so the whole multiple-dosing history is a chain of small matrix
exponentials evaluated at event boundaries: no integration error at
all, only round-off.

The :func:`_expm` kernel is a scaling-and-squaring Taylor series —
``scipy.linalg.expm``'s strategy in ~20 lines, no SciPy needed.

Also here: :func:`bateman` (the one-compartment oral curve with the
equal-rate limit ``ka = ke``) and :func:`superposition` (the principle
of superposition for regular multiple dosing, the exact steady-state
machinery behind accumulation).

Examples
--------
>>> import numpy as np
>>> from pharmaml.pkpd.compartments import (
...     TwoCompartment, bateman, superposition)

A two-compartment oral model (the pharmacometrics-portfolio /
rxode2 structure: depot → central ↔ peripheral, CL out of central):

>>> model = TwoCompartment(ka=1.2, cl=5.0, v2=50.0, q=3.0, v3=100.0)
>>> times = np.linspace(0.0, 24.0, 13)
>>> conc = model.concentrations(times, dose=100.0)
>>> round(float(conc.max()), 4)          # Cmax on the 2-hour grid
1.4693
>>> round(float(times[int(conc.argmax())]), 1)
2.0

With elimination switched off the administered amount is conserved
across the three compartments (an internal exactness check):

>>> closed = TwoCompartment(ka=1.2, cl=0.0, v2=50.0, q=3.0, v3=100.0)
>>> amounts = closed.amounts(times, dose=100.0)
>>> float(np.abs(amounts.sum(axis=1) - 100.0).max()) < 1e-9   # mass balance
True

The matrix-exponential route agrees with the toolkit's own RK45 ODE
engine to solver tolerance (a cross-engine validation):

>>> from pharmaml.pkpd.ode import DosingEvent, solve_ode
>>> rhs = model.rhs()
>>> ode = solve_ode(rhs, times, [0.0, 0.0, 0.0],
...                events=[DosingEvent(0.0, 100.0, compartment=0)])
>>> round(float(np.max(np.abs(ode.states[:, 1] / model.v2 - conc))), 8)
1.2e-07

The Bateman function with the flip-flop-safe equal-rate limit
(``ka = ke`` → ``C = D ka t e^{-ke t}``, the ``|ka-ke| < 1e-9`` guard
pkpdutils uses):

>>> round(bateman(2.0, 0.5, 0.5, 1.0), 9)          # equal-rate branch
0.60653066
>>> round(bateman(2.0, 0.5 + 1e-12, 0.5, 1.0), 9)  # continuity at the seam
0.60653066

Superposition of a monoexponential single-dose curve — after enough
doses the peak converges to ``D / (1 - e^{-ke tau})``:

>>> single = lambda t: np.where(np.asarray(t) >= 0, np.exp(-0.5 * np.asarray(t)), 0.0)
>>> peaks = superposition(np.array([38.0]), single, tau=2.0, n_doses=20)
>>> round(float(peaks[0]), 6)                      # 1/(1-e^{-1})
1.581977
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_positive

__all__ = [
    "TwoCompartment",
    "bateman",
    "superposition",
]


def _expm(matrix: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring Taylor series.

    The infinity norm is pushed below 1/2 by squaring-free scaling, a
    plain Taylor sum converges to machine precision there, and ``s``
    squarings restore the original scale — the same strategy
    ``scipy.linalg.expm`` formalises with Padé approximants.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.pkpd.compartments import _expm
    >>> quarter = _expm(np.array([[0.0, np.pi / 2], [-np.pi / 2, 0.0]]))
    >>> np.round(quarter, 9)        # rotation by 90 degrees
    array([[ 0.,  1.],
           [-1.,  0.]])
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise DataValidationError("_expm needs a square matrix")
    norm = float(np.abs(a).sum(axis=1).max()) if a.size else 0.0
    squarings = max(0, int(np.ceil(np.log2(max(norm, 0.5) / 0.5))))
    scaled = a / (2.0**squarings)
    result = np.eye(a.shape[0])
    term = np.eye(a.shape[0])
    for order in range(1, 40):
        term = term @ scaled / order
        result = result + term
        if float(np.abs(term).max()) < 1e-18:
            break
    for _ in range(squarings):
        result = result @ result
    return result


class _LinearSystem:
    """Exact solver for ``dA/dt = K A + b`` with dosing events.

    This is the numpy-native ``linCmt``: state transitions between
    event boundaries are single matrix exponentials of the augmented
    matrix ``[[K, b], [0, 0]]`` acting on ``[A; 1]`` — boluses are
    impulses applied at the boundaries, infusions contribute a constant
    ``b`` inside their window.
    """

    def __init__(self, rate_matrix: np.ndarray) -> None:
        matrix = np.asarray(rate_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise DataValidationError("rate_matrix must be square")
        if not np.all(np.isfinite(matrix)):
            raise DataValidationError("rate_matrix must be finite")
        self._k = matrix
        self._n = matrix.shape[0]

    def _solve_window(
        self, a: float, b: float, y: np.ndarray, events: tuple
    ) -> np.ndarray:
        """Exact state at ``b`` starting from ``y`` at ``a``."""
        rates = np.zeros(self._n)
        for e in events:
            if e.duration > 0 and e.time <= a < e.time + e.duration:
                rates[e.compartment] += e.amount / e.duration
        if not np.any(rates):
            return _expm(self._k * (b - a)) @ y
        augmented = np.zeros((self._n + 1, self._n + 1))
        augmented[: self._n, : self._n] = self._k
        augmented[: self._n, self._n] = rates
        z = np.concatenate([y, [1.0]])
        z = _expm(augmented * (b - a)) @ z
        return z[: self._n]

    def solve(
        self,
        times: np.ndarray | list[float],
        events: tuple | list,
        y0: np.ndarray | list[float],
    ) -> np.ndarray:
        """Amounts at each requested time (rows = times, cols = states)."""
        t = np.asarray(times, dtype=float).reshape(-1)
        if t.size == 0 or not np.all(np.diff(t) > 0) and t.size > 1:
            raise DataValidationError("times must be non-empty and increasing")
        y = np.asarray(y0, dtype=float).reshape(-1).astype(float).copy()
        if y.size != self._n:
            raise DataValidationError(
                f"y0 has {y.size} states, the system has {self._n}"
            )
        ev = tuple(sorted(events, key=lambda e: (e.time, e.duration)))
        for e in ev:
            if e.compartment >= self._n:
                raise DataValidationError(
                    f"event compartment {e.compartment} outside the system"
                )

        marks = [float(t[0])]
        if ev and ev[0].time < marks[0]:
            marks[0] = float(ev[0].time)
        t_end = float(t[-1])
        for e in ev:
            if e.time > t_end + 1e-12:
                raise DataValidationError(
                    f"event at t={e.time} lies after the last output time {t_end}"
                )
            if e.time >= marks[0]:
                marks.append(float(e.time))
            if e.duration > 0 and e.time + e.duration <= t_end:
                marks.append(float(e.time + e.duration))
        marks.extend(float(v) for v in t)
        marks = sorted(set(marks))

        out = np.full((t.size, self._n), np.nan)
        row = 0
        for i, edge in enumerate(marks):
            if i > 0:
                y = self._solve_window(marks[i - 1], edge, y, ev)
            for e in ev:
                if (
                    e.duration == 0.0
                    and abs(e.time - edge) < 1e-12
                    and (i == 0 or abs(e.time - marks[i - 1]) > 1e-12)
                ):
                    y[e.compartment] += e.amount
            while row < t.size and t[row] <= edge + 1e-12:
                out[row] = y
                row += 1
        while row < t.size:
            out[row] = y
            row += 1
        return out


@dataclass(frozen=True, slots=True)
class TwoCompartment:
    """Two-compartment model with first-order oral absorption.

    The rxode2 / pharmacometrics-portfolio structure::

        d/dt(depot)     = -ka * depot
        d/dt(central)   = ka * depot - (cl/v2) * central
                          - (q/v2) * central + (q/v3) * peripheral
        d/dt(peripheral)= (q/v2) * central - (q/v3) * peripheral

    ``concentrations`` returns central-compartment plasma amounts
    divided by ``v2``; oral doses land in the depot (scaled by ``f``),
    IV doses directly in the central compartment.
    """

    ka: float
    cl: float
    v2: float
    q: float
    v3: float
    f: float = 1.0

    def __post_init__(self) -> None:
        for name in ("ka", "v2", "q", "v3"):
            validate_positive(getattr(self, name), name)
        if self.cl < 0:
            raise DataValidationError(f"cl must be >= 0, got {self.cl}")
        if not 0 < self.f <= 1:
            raise DataValidationError(f"f must be in (0, 1], got {self.f}")

    def rate_matrix(self) -> np.ndarray:
        """The 3x3 transfer matrix K (columns sum to <= 0)."""
        ka, cl, v2, q, v3 = self.ka, self.cl, self.v2, self.q, self.v3
        return np.array(
            [
                [-ka, 0.0, 0.0],
                [ka, -(cl / v2) - (q / v2), q / v3],
                [0.0, q / v2, -q / v3],
            ]
        )

    def rhs(self):
        """Right-hand side for :func:`pharmaml.pkpd.ode.solve_ode`."""

        def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
            depot, central, peripheral = y
            return np.array(
                [
                    -self.ka * depot,
                    self.ka * depot
                    - (self.cl / self.v2) * central
                    - (self.q / self.v2) * central
                    + (self.q / self.v3) * peripheral,
                    (self.q / self.v2) * central - (self.q / self.v3) * peripheral,
                ]
            )

        return _rhs

    def amounts(
        self,
        times: np.ndarray | list[float],
        dose: float,
        *,
        route: str = "oral",
        events: tuple | list = (),
    ) -> np.ndarray:
        """Exact amounts (rows = times, columns = depot/central/peripheral)."""
        if route not in {"oral", "iv"}:
            raise DataValidationError(f"route must be 'oral' or 'iv', got {route!r}")
        validate_positive(dose, "dose")
        system = _LinearSystem(self.rate_matrix())
        if events:
            return system.solve(times, events, [0.0, 0.0, 0.0])
        from pharmaml.pkpd.ode import DosingEvent

        compartment = 0 if route == "oral" else 1
        amount = dose * self.f if route == "oral" else dose
        event = DosingEvent(
            time=float(np.asarray(times).reshape(-1)[0]),
            amount=amount,
            compartment=compartment,
        )
        return system.solve(times, (event,), [0.0, 0.0, 0.0])

    def concentrations(
        self,
        times: np.ndarray | list[float],
        dose: float,
        *,
        route: str = "oral",
    ) -> np.ndarray:
        """Plasma concentrations (central amount / V2) over ``times``."""
        amounts = self.amounts(times, dose, route=route)
        return amounts[:, 1] / self.v2


def bateman(
    dose_per_volume: float, ka: float, ke: float, t: float | np.ndarray
) -> float | np.ndarray:
    """One-compartment oral (Bateman) curve with the equal-rate limit.

    ``C(t) = (D/V) · ka/(ka-ke) · (e^{-ke t} - e^{-ka t})``; when
    ``|ka - ke| < 1e-9`` the closed form degenerates and the limit
    ``(D/V) · ka · t · e^{-ke t}`` is used instead (pkpdutils'
    ``Bateman`` guard against the flip-flop seam).
    """
    times = np.asarray(t, dtype=float)
    scalar = times.ndim == 0
    times = np.atleast_1d(times)
    if np.any(times < 0):
        raise DataValidationError("time t must be non-negative")
    if abs(ka - ke) < 1e-9:
        values = dose_per_volume * ka * times * np.exp(-ke * times)
    else:
        values = (
            dose_per_volume
            * (ka / (ka - ke))
            * (np.exp(-ke * times) - np.exp(-ka * times))
        )
    return float(values[0]) if scalar else values


def superposition(
    times: np.ndarray | list[float],
    single_dose_response,
    tau: float,
    n_doses: int,
) -> np.ndarray:
    """Principle of superposition for regular multiple dosing.

    Sums ``n_doses`` copies of the single-dose response shifted by
    ``k·tau`` (any contribution from a not-yet-administered dose —
    negative time — is zero, i.e. the response must be causal).

    Parameters
    ----------
    times:
        Output grid.
    single_dose_response:
        Callable mapping an array of times-since-dose to the response
        of ONE dose (must return 0 for negative arguments).
    tau:
        Dosing interval.
    n_doses:
        Number of administered doses.
    """
    tau = validate_positive(tau, "tau")
    if n_doses < 1:
        raise DataValidationError(f"n_doses must be >= 1, got {n_doses}")
    t = np.asarray(times, dtype=float).reshape(-1)
    total = np.zeros_like(t, dtype=float)
    for k in range(n_doses):
        total = total + np.asarray(
            single_dose_response(t - k * tau), dtype=float
        ).reshape(-1)
    return total
