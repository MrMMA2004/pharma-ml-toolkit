"""Compartmental PK by numerical ODE integration: the RK4 engine.

``scipy/scipy`` was reviewed file by file for this round (the
``scipy/integrate/`` tree: ``_ivp/`` -- ``solve_ivp``, ``RK45``,
``RK23``, the dense-output machinery -- ``_ode.py``, ``_quadpy``).
The adaptive multistep solvers and QUADPACK's quadrature stay with
scipy; what lands is the one ODE engine every pharmacokinetics
course actually teaches, run exactly and deterministically:

* :func:`simulate_compartment` -- the classical fixed-step
  **Runge-Kutta 4** integrator driving the standard compartment
  ODEs: one-compartment IV bolus (``dC/dt = -ke C``), one-compartment
  oral (gut → central, first-order absorption), and the
  two-compartment models (central ↔ peripheral, micro constants
  ``k10/k12/k21``).  The state vector is integrated on a dense
  internal grid, then linearly interpolated onto the requested
  sample times.

The honest scope: scipy's ``solve_ivp`` adapts its step to a
tolerance and carries dense output of matching order -- this
engine's step is fixed (``step=`` or auto-selected from the sample
span) and the interpolation onto sample times is linear.  The
payoff is the teaching contract: every number the module returns
can be reproduced with a pencil and the four RK4 slopes.

Two verifications run in the doctests themselves, because they are
the exam questions: the one-compartment IV bolus trajectory must
reproduce the closed form ``C0 e^{-ke t}`` (:func:`pharmaml.pharma.pk.
iv_bolus_concentration`), and every IV simulation's AUC must
satisfy the mass-balance identity ``AUC = Dose / CL`` to within the
integration error.

Examples
--------
>>> import numpy as np
>>> from pharmaml.pharma.compartment import simulate_compartment
>>> from pharmaml.pharma.pk import iv_bolus_concentration
>>> times = np.linspace(0.0, 24.0, 49)
>>> sim = simulate_compartment(times, dose=250.0, model="one",
...                            route="iv_bolus", ke=0.25, vd=40.0)
>>> closed = iv_bolus_concentration(250.0 / 40.0, 0.25, times)
>>> float(np.max(np.abs(sim.concentrations - closed))) < 1e-4
True
>>> round(float(sim.cmax), 4)
6.25
>>> oral = simulate_compartment(times, dose=250.0, model="one",
...                             route="oral", ke=0.25, vd=40.0, ka=1.2)
>>> float(abs(oral.tmax - np.log(1.2 / 0.25) / (1.2 - 0.25))) < 0.6
True
>>> two_t = np.linspace(0.0, 96.0, 193)   # 14 terminal half-lives
>>> two = simulate_compartment(two_t, dose=250.0, model="two",
...                            route="iv_bolus", k10=0.25, k12=0.15,
...                            k21=0.20, v1=40.0)
>>> round(float(two.concentrations[0]), 4)      # C(0) = Dose / V1
6.25
>>> abs(float(two.auc) - 250.0 / (0.25 * 40.0)) / (250.0 / 10.0) < 0.01
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import DataValidationError
from .pk import auc_trapezoid

__all__ = ["CompartmentResult", "simulate_compartment"]

#: The models :func:`simulate_compartment` drives.
COMPARTMENT_MODELS = ("one", "two")

#: The dosing routes (first-order absorption distinguishes ``oral``).
COMPARTMENT_ROUTES = ("iv_bolus", "oral")


@dataclass(frozen=True)
class CompartmentResult:
    """One compartmental simulation: the states, the curve, the deriveds.

    ``concentrations`` is the central (compartment 1) concentration
    at each requested time; ``amounts`` carries every compartment's
    amount trajectory over the internal grid end-points (the gut
    depot for oral dosing, the peripheral compartment for the
    two-compartment models).
    """

    times: np.ndarray
    concentrations: np.ndarray
    amounts: dict[str, np.ndarray]
    model: str
    route: str
    step: float
    dose: float

    @property
    def cmax(self) -> float:
        """Peak central concentration over the requested window."""
        return float(np.max(self.concentrations))

    @property
    def tmax(self) -> float:
        """Time of peak concentration over the requested window."""
        return float(self.times[int(np.argmax(self.concentrations))])

    @property
    def auc(self) -> float:
        """AUC over the requested window (linear trapezoid)."""
        return float(auc_trapezoid(self.times, self.concentrations))

    def __str__(self) -> str:
        return (
            f"CompartmentResult(model={self.model}, route={self.route}, "
            f"cmax={self.cmax:.3f}, tmax={self.tmax:.2f}, "
            f"auc={self.auc:.2f})"
        )


def _rk4_step(
    derivatives, state: np.ndarray, t: float, h: float
) -> np.ndarray:
    """One classical Runge-Kutta 4 step: the four slopes, weighted 1-2-2-1."""
    k1 = derivatives(t, state)
    k2 = derivatives(t + h / 2.0, state + (h / 2.0) * k1)
    k3 = derivatives(t + h / 2.0, state + (h / 2.0) * k2)
    k4 = derivatives(t + h, state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_compartment(
    times: np.ndarray,
    dose: float,
    *,
    model: str = "one",
    route: str = "iv_bolus",
    ka: float = 1.2,
    ke: float = 0.25,
    vd: float = 40.0,
    k10: float = 0.25,
    k12: float = 0.15,
    k21: float = 0.2,
    v1: float = 40.0,
    f: float = 1.0,
    step: float | None = None,
) -> CompartmentResult:
    """Simulate a compartmental PK model with fixed-step RK4.

    Parameters
    ----------
    times:
        Requested sample times (non-negative, increasing); the
        concentration curve is reported exactly there.
    dose:
        Administered amount (same units as ``amounts``; with ``vd``
        in L the concentrations are amount/L).
    model:
        ``'one'`` (``ke``/``vd``) or ``'two'`` (``k10``/``k12``/``k21``/``v1``).
    route:
        ``'iv_bolus'`` (instantaneous into the central compartment)
        or ``'oral'`` (a gut depot absorbed first-order at ``ka``,
        fraction ``f`` reaching the central compartment).
    ka, ke, vd:
        One-compartment parameters: absorption and elimination rate
        constants (1/time) and volume of distribution.
    k10, k12, k21, v1:
        Two-compartment micro constants: elimination from the
        central compartment, bidirectional central↔peripheral
        transfer, and the central volume.
    f:
        Bioavailable fraction for oral dosing (0 < f ≤ 1).
    step:
        The RK4 integration step; ``None`` picks one from the sample
        span (a 4000-point internal grid, capped so it never exceeds
        the coarsest sample gap).

    Returns
    -------
    CompartmentResult
        The concentration curve at ``times``, the per-compartment
        amounts, and the derived ``cmax``/``tmax``/``auc``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.pharma.compartment import simulate_compartment
    >>> times = np.linspace(0.0, 12.0, 25)
    >>> sim = simulate_compartment(times, 100.0, model="one",
    ...                            route="iv_bolus", ke=0.4, vd=25.0)
    >>> round(float(sim.concentrations[0]), 4)
    4.0
    >>> round(float(sim.concentrations[-1]), 6)
    0.032919
    """
    if model not in COMPARTMENT_MODELS:
        raise DataValidationError(
            f"model must be one of {COMPARTMENT_MODELS}, got {model!r}"
        )
    if route not in COMPARTMENT_ROUTES:
        raise DataValidationError(
            f"route must be one of {COMPARTMENT_ROUTES}, got {route!r}"
        )
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size < 2 or np.any(np.diff(t) <= 0.0) or t[0] < 0.0:
        raise DataValidationError(
            "times must be >= 2 increasing non-negative samples"
        )
    if not (np.isfinite(dose) and dose > 0.0):
        raise DataValidationError(f"dose must be positive, got {dose}")
    for name, value in (("ke", ke), ("vd", vd), ("ka", ka), ("k10", k10),
                        ("k12", k12), ("k21", k21), ("v1", v1)):
        if not (np.isfinite(value) and value > 0.0):
            raise DataValidationError(f"{name} must be positive, got {value}")
    if not (0.0 < f <= 1.0):
        raise DataValidationError(f"f must satisfy 0 < f <= 1, got {f}")

    t_max = float(t[-1])
    if step is None:
        span = max(t_max, 1.0)
        step = min(span / 4000.0, float(np.max(np.diff(t))) / 4.0)
        step = max(step, 1e-6)
    elif not (np.isfinite(step) and step > 0.0):
        raise DataValidationError(f"step must be positive, got {step}")
    grid = np.arange(0.0, t_max + step / 2.0, step)

    names: tuple[str, ...]
    if model == "one":
        volume = vd

        def derivatives(_t: float, state: np.ndarray) -> np.ndarray:
            if route == "oral":
                gut, central = state
                return np.array([-ka * gut, ka * gut - ke * central])
            return np.array([-ke * state[0]])

        if route == "oral":
            state0 = np.array([f * dose, 0.0])
            names = ("gut", "central")
        else:
            state0 = np.array([dose])
            names = ("central",)
    else:
        volume = v1

        def derivatives(_t: float, state: np.ndarray) -> np.ndarray:
            if route == "oral":
                gut, central, peripheral = state
                return np.array([
                    -ka * gut,
                    ka * gut - (k10 + k12) * central + k21 * peripheral,
                    k12 * central - k21 * peripheral,
                ])
            central, peripheral = state
            return np.array([
                -(k10 + k12) * central + k21 * peripheral,
                k12 * central - k21 * peripheral,
            ])

        if route == "oral":
            state0 = np.array([f * dose, 0.0, 0.0])
            names = ("gut", "central", "peripheral")
        else:
            state0 = np.array([dose, 0.0])
            names = ("central", "peripheral")

    # The RK4 march over the internal grid.
    trajectory = np.empty((grid.size, state0.size), dtype=float)
    state = state0.astype(float)
    trajectory[0] = state
    for i in range(1, grid.size):
        state = _rk4_step(derivatives, state, float(grid[i - 1]), step)
        trajectory[i] = state
    central_index = names.index("central")
    central_amounts = trajectory[:, central_index]
    concentrations_grid = central_amounts / volume
    concentrations = np.interp(t, grid, concentrations_grid)
    amounts = {
        name: np.interp(t, grid, trajectory[:, j])
        for j, name in enumerate(names)
    }
    return CompartmentResult(
        times=t,
        concentrations=concentrations,
        amounts=amounts,
        model=model,
        route=route,
        step=float(step),
        dose=float(dose),
    )
