r"""Analytical (closed-form) compartmental PK — the linCmt engine.

Round 176 absorbed the two ODE platforms' *analytic* half here, while the
compiled solvers stay out (honest boundary):

``nlmixr2/rxode2``
    (the R ODE solver).  The ``linCmt()`` closed forms were rebuilt from
    ``build/buildLinCmt.R`` and ``R/linCmt.R``: the two-compartment
    eigen-solution (``alpha``/``beta`` via ``0.5*(k10+k12+k21 ±
    sqrt((...)²-4·k10·k21))``), the eight hybrid coefficients
    (:func:`sol_comp2` reproduces the package's own golden test values),
    the oral first-order term ``(E0-expa)/(ka-L0)``, the infusion term
    ``(1-E)/L``, the trans-1/trans-11 parameterization conversions, and
    the degenerate ``ka == lambda`` guard (rxode2 delegates the limit to
    an external package; this module follows mrgsolve's behaviour and
    raises).  The LSODA/AutoSwitch machinery is the not-vendored part.

``metrumresearchgroup/mrgsolve``
    (the C++ ODE platform).  ``src/odeproblem.cpp``'s PolyExp
    superposition theory is the second half: exact repeated-dose
    accumulation as a geometric residual series and steady state as its
    ``1/(1-exp(-lambda*tau))`` limit — no 500-dose burn-in simulation.
    The periodic steady-state infusion piece below is re-derived from
    first principles and verified against explicit cycle summation
    (mrgsolve's own C++ comment block for the ``dx<=xinf`` branch does
    not survive transcription; the derivation here is the one that
    reproduces the explicit sum to 1e-10).

The engine is exact (linear systems superpose), so every dose — bolus,
infusion, first-order absorption, ``addl`` repeats — is summed in
closed form, and :func:`steady_state_curve` evaluates the periodic
limit directly.

Examples
--------
>>> from pharmaml.pharma.lincmt import LinCmtParams, Dose, sol_comp2
>>> import numpy as np
>>> comp = sol_comp2(k10=0.1, k12=3.0, k21=1.0)
>>> round(float(comp["alpha"]), 8), round(float(comp["beta"]), 8)
(4.07546291, 0.02453709)
>>> [round(float(v), 4) for v in comp["c_central"]]
[0.7592, -0.7406, 0.2408, 0.7406]

Two-compartment IV bolus, amount remaining in the central compartment:

>>> params = LinCmtParams(kel=0.1, k12=3.0, k21=1.0)
>>> doses = [Dose(time=0.0, amount=100.0)]
>>> from pharmaml.pharma.lincmt import lin_cmt_profile
>>> prof = lin_cmt_profile(params, doses, t=np.array([2.5]))
>>> round(float(prof["central"][0]), 6)
22.650114

First-order absorption (one compartment, the textbook bateman):

>>> params = LinCmtParams(kel=0.1, ka=1.2, v1=10.0)
>>> doses = [Dose(time=0.0, amount=100.0, cmt="depot")]
>>> from pharmaml.pharma.lincmt import lin_cmt_concentration
>>> conc = lin_cmt_concentration(params, doses, t=np.array([5.0]))
>>> round(float(conc[0]), 4)   # 100*1.2/(10*1.1)*(exp(-0.5)-exp(-6))
6.5897

Steady state of an intermittent infusion, in closed form:

>>> from pharmaml.pharma.lincmt import steady_state_curve
>>> params = LinCmtParams(kel=0.1, v1=1.0)
>>> dose = Dose(amount=6.0, rate=1.0, ii=24.0, ss=True)
>>> curve = steady_state_curve(params, dose, t=np.array([0.0, 3.0, 12.0]))
>>> [round(float(v), 6) for v in curve]
[0.820218, 3.19945, 2.723219]

Exact AUC to infinity (no trapezoid needed):

>>> from pharmaml.pharma.lincmt import lin_cmt_auc_inf
>>> params = LinCmtParams(kel=0.1, k12=3.0, k21=1.0)
>>> round(lin_cmt_auc_inf(params, [Dose(amount=100.0)]), 6)   # = D/CL
1000.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError

__all__ = [
    "Dose",
    "LinCmtParams",
    "hybrid_rate_constants",
    "lin_cmt_auc_inf",
    "lin_cmt_concentration",
    "lin_cmt_profile",
    "sol_comp2",
    "steady_state_auc_interval",
    "steady_state_curve",
]

# mrgsolve's degeneracy guard (src/odeproblem.cpp: |ka-k| < 2.22e-14 errors).
_DEGENERACY_TOL = 1e-10


def hybrid_rate_constants(k10: float, k12: float, k21: float) -> tuple[float, float]:
    """Alpha/beta hybrid rate constants of a two-compartment model.

    ``alpha = 0.5*(k10+k12+k21 + sqrt((k10+k12+k21)^2 - 4*k10*k21))`` and
    ``beta`` with the minus sign — rxode2 ``build/buildLinCmt.R``.

    Examples
    --------
    >>> hybrid_rate_constants(0.1, 3.0, 1.0)
    (4.075462910052909, 0.02453708994709025)
    """
    total = k10 + k12 + k21
    disc = total * total - 4.0 * k10 * k21
    if disc < 0:
        raise DataValidationError("k10*k21 exceeds the discriminant — invalid rate constants")
    root = math.sqrt(disc)
    return 0.5 * (total + root), 0.5 * (total - root)


def sol_comp2(k10: float, k12: float, k21: float) -> dict[str, Any]:
    """Two-compartment eigen decomposition with the four coefficient sets.

    Returns ``alpha``/``beta`` and the coefficient vectors of the closed
    form ``A_cmt(t) = sum_lam coeff * exp(-lam*t)``:

    * ``c_central`` — central amount from a central bolus
      ``[A_c(alpha), A_p(alpha), A_c(beta), A_p(beta)]``
    * ``c_peripheral`` — central + peripheral amounts from a peripheral bolus

    Matches the rxode2 golden test ``.solComp2(k10=0.1, k12=3, k21=1)``:
    ``L = (4.07546..., 0.02454...)``, ``C1 = [0.7592 -0.7406; 0.2408 0.7406]``.

    Examples
    --------
    >>> comp = sol_comp2(0.1, 3.0, 1.0)
    >>> round(float(comp["alpha"]), 6), round(float(comp["beta"]), 6)
    (4.075463, 0.024537)
    """
    alpha, beta = hybrid_rate_constants(k10, k12, k21)
    inv_d0 = 1.0 / (beta - alpha)
    inv_d1 = 1.0 / (alpha - beta)
    c_central = np.array(
        [
            (k21 - alpha) * inv_d0,  # central <- central, alpha term
            k12 * inv_d0,  # peripheral <- central, alpha term
            (k21 - beta) * inv_d1,  # central <- central, beta term
            k12 * inv_d1,  # peripheral <- central, beta term
        ]
    )
    c_peripheral = np.array(
        [
            k21 * inv_d0,  # central <- peripheral, alpha term
            (k10 + k12 - alpha) * inv_d0,  # peripheral <- peripheral, alpha
            k21 * inv_d1,  # central <- peripheral, beta term
            (k10 + k12 - beta) * inv_d1,  # peripheral <- peripheral, beta
        ]
    )
    return {
        "alpha": np.float64(alpha),
        "beta": np.float64(beta),
        "c_central": c_central,
        "c_peripheral": c_peripheral,
    }


@dataclass(frozen=True, slots=True)
class LinCmtParams:
    """Micro-constant parameterization of a 1- or 2-compartment model.

    ``kel`` is the elimination rate constant (k10), ``k12``/``k21`` the
    inter-compartmental ones (0 for one compartment), ``ka`` the
    first-order absorption rate (0 = no depot).  Construct from the
    clearance style (rxode2 "trans 1") with :meth:`from_clearance` or
    from hybrid ``alpha``/``beta``/A/B coefficients ("trans 10/11") with
    :meth:`from_macro`.
    """

    kel: float
    k12: float = 0.0
    k21: float = 0.0
    ka: float = 0.0
    v1: float = 1.0
    f: float = 1.0
    tlag: float = 0.0

    def __post_init__(self) -> None:
        if self.kel <= 0:
            raise DataValidationError("kel must be positive")
        if (self.k12 <= 0 or self.k21 <= 0) and not (
            self.k12 == 0 and self.k21 == 0
        ):
            raise DataValidationError("k12/k21 must be both zero (1-cmt) or both positive")
        if self.v1 <= 0:
            raise DataValidationError("v1 must be positive")
        if not 0 < self.f <= 1:
            raise DataValidationError("f must be in (0, 1]")
        if self.ka < 0 or self.tlag < 0:
            raise DataValidationError("ka and tlag must be non-negative")

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_clearance(
        cls,
        cl: float,
        v1: float,
        q: float = 0.0,
        v2: float = 0.0,
        ka: float = 0.0,
        f: float = 1.0,
        tlag: float = 0.0,
    ) -> LinCmtParams:
        """Clearance style (rxode2 trans 1): ``k10=CL/V1, k12=Q/V1, k21=Q/V2``.

        Examples
        --------
        >>> p = LinCmtParams.from_clearance(cl=5.0, v1=100.0, q=15.0, v2=150.0)
        >>> (round(p.kel, 4), round(p.k12, 4), round(p.k21, 4))
        (0.05, 0.15, 0.1)
        """
        if (q == 0) != (v2 == 0):
            raise DataValidationError("q and v2 must be given together (or both zero)")
        return cls(
            kel=cl / v1,
            k12=q / v1 if q else 0.0,
            k21=q / v2 if q else 0.0,
            ka=ka,
            v1=v1,
            f=f,
            tlag=tlag,
        )

    @classmethod
    def from_macro(
        cls,
        alpha: float,
        beta: float,
        a: float,
        b: float,
        ka: float = 0.0,
        f: float = 1.0,
        tlag: float = 0.0,
    ) -> LinCmtParams:
        """Hybrid A/B style (rxode2 trans 10/11 inverse).

        ``C_bolus(t) = D*(A*exp(-alpha*t) + B*exp(-beta*t))`` (amounts,
        V=1).  ``k21 = (A*beta + B*alpha)/(A+B)``, ``k10 = alpha*beta/k21``,
        ``k12 = alpha+beta-k21-k10``.

        Examples
        --------
        >>> p = LinCmtParams.from_macro(4.075462910052909, 0.02453708994709025, 0.7592, 0.2408)
        >>> (round(p.kel, 4), round(p.k12, 4), round(p.k21, 4))
        (0.1, 3.0, 1.0)
        """
        total = a + b
        if total <= 0:
            raise DataValidationError("A + B must be positive")
        k21 = (a * beta + b * alpha) / total
        k10 = alpha * beta / k21
        k12 = alpha + beta - k21 - k10
        return cls(kel=k10, k12=k12, k21=k21, ka=ka, v1=1.0 / total, f=f, tlag=tlag)

    # -- derived quantities ----------------------------------------------
    @property
    def two_compartments(self) -> bool:
        """Whether a peripheral compartment is present."""
        return self.k12 > 0

    def hybrid(self) -> tuple[float, ...]:
        """Hybrid rate constants — ``(kel,)`` or ``(alpha, beta)``."""
        if self.two_compartments:
            return hybrid_rate_constants(self.kel, self.k12, self.k21)
        return (self.kel,)

    def to_macro(self) -> dict[str, float]:
        """A/B/alpha/beta coefficients of the IV bolus concentration curve.

        Examples
        --------
        >>> p = LinCmtParams(kel=0.1, k12=3.0, k21=1.0)
        >>> m = p.to_macro()
        >>> round(m["a"], 4), round(m["b"], 4)
        (0.7592, 0.2408)
        """
        if not self.two_compartments:
            return {"alpha": self.kel, "beta": self.kel, "a": 1.0, "b": 0.0}
        comp = sol_comp2(self.kel, self.k12, self.k21)
        a = float(comp["c_central"][0])
        b = float(comp["c_central"][2])
        return {"alpha": float(comp["alpha"]), "beta": float(comp["beta"]), "a": a, "b": b}

    def to_clearance(self) -> dict[str, float]:
        """Back to the clearance style: CL, V1, Q, V2 (+ CL2 alias)."""
        if not self.two_compartments:
            return {"cl": self.kel * self.v1, "v1": self.v1, "q": 0.0, "v2": 0.0}
        return {
            "cl": self.kel * self.v1,
            "v1": self.v1,
            "q": self.k12 * self.v1,
            "v2": self.k12 * self.v1 / self.k21 if self.k21 else 0.0,
        }

    def check_ka(self) -> None:
        """Raise on the flip-flop degeneracy (mrgsolve's guard)."""
        if self.ka <= 0:
            return
        for lam in self.hybrid():
            if abs(self.ka - lam) < _DEGENERACY_TOL:
                raise DataValidationError(
                    "ka equals a hybrid rate constant (flip-flop degeneracy) — "
                    "mrgsolve refuses this case; perturb ka"
                )


@dataclass(slots=True)
class Dose:
    """One dosing record (the analytic engine's event grammar).

    ``cmt`` is ``"central"``, ``"peripheral"`` or ``"depot"``.  A bolus
    has ``rate=0``; an infusion sets ``rate`` (total ``amount`` infused)
    or ``duration`` (rate = amount/duration).  ``ii``/``addl`` repeat the
    record ``addl`` times every ``ii``; ``ss=True`` means the regimen has
    been running forever and the closed-form periodic limit is used.
    """

    time: float = 0.0
    amount: float = 0.0
    cmt: str = "central"
    rate: float = 0.0
    duration: float = 0.0
    ii: float = 0.0
    addl: int = 0
    ss: bool = False

    def __post_init__(self) -> None:
        if self.cmt not in {"central", "peripheral", "depot"}:
            raise DataValidationError(f"unknown compartment {self.cmt!r}")
        if self.amount < 0 or self.rate < 0 or self.duration < 0:
            raise DataValidationError("dose amount/rate/duration must be non-negative")
        if self.ii < 0 or self.addl < 0:
            raise DataValidationError("ii/addl must be non-negative")
        if self.ss and (self.ii <= 0 or self.amount <= 0):
            raise DataValidationError("steady-state records need ii > 0 and a dose")
        if self.addl > 0 and self.ii <= 0:
            raise DataValidationError("addl repeats need ii > 0")


# ---------------------------------------------------------------------------
# coefficient machinery
# ---------------------------------------------------------------------------


def _coefficient_sets(params: LinCmtParams) -> dict[str, dict[float, float]]:
    """Per-source {eigen: coefficient} sets for the central amount output.

    ``central``/``peripheral`` sources get the plain eigen coefficients;
    the ``depot`` source couples into central with rate ``ka`` so its
    impulse response is the bateman combination (checked against the
    ``ka == lambda`` guard).
    """
    params.check_ka()
    lams = params.hybrid()
    if params.two_compartments:
        comp = sol_comp2(params.kel, params.k12, params.k21)
        central = {lams[0]: float(comp["c_central"][0]), lams[1]: float(comp["c_central"][2])}
        peripheral = {lams[0]: float(comp["c_peripheral"][0]), lams[1]: float(comp["c_peripheral"][2])}
    else:
        central = {params.kel: 1.0}
        peripheral = {}
    sets: dict[str, dict[float, float]] = {"central": central, "peripheral": peripheral}
    if params.ka > 0:
        depot = {}
        for lam, coeff in central.items():
            depot[lam] = params.ka * coeff / (params.ka - lam)
        depot[params.ka] = -sum(depot.values())  # the -ka*exp(-ka t) tail
        sets["depot"] = depot
    return sets


def _as_array(t: float | np.ndarray) -> np.ndarray:
    return np.asarray(t, dtype=float)


def _expanded_doses(doses: list[Dose]) -> list[Dose]:
    """Realize addl repeats (mrgsolve ``schedule()``: ``time + ii*k``)."""
    out: list[Dose] = []
    for dose in doses:
        out.append(dose)
        for k in range(1, dose.addl + 1):
            out.append(
                Dose(
                    time=dose.time + dose.ii * k,
                    amount=dose.amount,
                    cmt=dose.cmt,
                    rate=dose.rate,
                    duration=dose.duration,
                )
            )
    return out


def _infusion_span(dose: Dose) -> tuple[float, float | None]:
    """(rate, duration) of a dose — bolus returns (0, None)."""
    if dose.rate > 0:
        if dose.amount > 0:
            return dose.rate, dose.amount / dose.rate
        return dose.rate, None  # constant-rate, never turned off
    if dose.duration > 0:
        return dose.amount / dose.duration, dose.duration
    return 0.0, None


def _bolus_response(
    coeffs: dict[float, float], u: np.ndarray, amount: float
) -> np.ndarray:
    """``amount * sum coeff * exp(-lam*u)`` with u<0 masked to zero."""
    out = np.zeros_like(u)
    positive = u >= 0
    for lam, coeff in coeffs.items():
        out = out + coeff * np.exp(-lam * np.where(positive, u, 0.0)) * positive
    return amount * out


def _infusion_response(
    coeffs: dict[float, float], u: np.ndarray, rate: float, duration: float | None
) -> np.ndarray:
    """Response to rate ``r`` on ``[0, duration]`` evaluated at ``u``.

    Single closed expression (verified against solve_ivp): the step-on
    term minus the step-off term, each clipped at zero.
    """
    out = np.zeros_like(u)
    on = np.clip(u, 0, None)
    if duration is None:
        off = np.zeros_like(u)
    else:
        off = np.clip(u - duration, 0, None)
    for lam, coeff in coeffs.items():
        growth = (1.0 - np.exp(-lam * on)) - (1.0 - np.exp(-lam * off))
        out = out + coeff * growth / lam
    return rate * out


# ---------------------------------------------------------------------------
# public engine
# ---------------------------------------------------------------------------


def lin_cmt_profile(
    params: LinCmtParams,
    doses: list[Dose],
    t: float | np.ndarray,
) -> dict[str, np.ndarray]:
    """Amounts per compartment of the analytic solution at times ``t``.

    Every dose (bolus, infusion, depot with lag/F, addl repeats) is
    superposed in closed form — exact for the linear system.

    Examples
    --------
    >>> import numpy as np
    >>> params = LinCmtParams(kel=0.1, k12=3.0, k21=1.0)
    >>> doses = [Dose(time=0.0, amount=100.0)]
    >>> prof = lin_cmt_profile(params, doses, t=np.array([2.5]))
    >>> round(float(prof["central"][0]), 6)
    22.650114
    >>> round(float(prof["peripheral"][0]), 6)
    69.648024
    """
    t_arr = _as_array(t)
    central = np.zeros_like(t_arr)
    peripheral = np.zeros_like(t_arr)
    depot = np.zeros_like(t_arr)
    sets = _coefficient_sets(params)
    central_sets = _central_output_sets(params, sets)

    for dose in _expanded_doses(list(doses)):
        start = dose.time + params.tlag if dose.cmt == "depot" else dose.time
        u = t_arr - start
        amount = dose.amount * params.f if dose.cmt == "depot" else dose.amount
        rate, duration = _infusion_span(dose)
        if dose.cmt == "depot":
            rate_in = rate * params.f
            if rate_in > 0:
                # infusion filling the depot, transferring with ka.  A finite
                # window equals endless-from-start minus endless-from-off
                # (linearity) — the difference is exactly the post-infusion
                # depot decay, verified against segmented solve_ivp.
                endless = _depot_infusion_response(sets["central"], u, rate_in, params.ka)
                depot_fill = rate_in * (1 - np.exp(-params.ka * np.clip(u, 0, None))) / params.ka
                if duration is not None:
                    shifted = _depot_infusion_response(
                        sets["central"], u - duration, rate_in, params.ka
                    )
                    endless = endless - shifted
                    depot_fill = depot_fill - rate_in * (
                        1 - np.exp(-params.ka * np.clip(u - duration, 0, None))
                    ) / params.ka
                    if params.two_compartments:
                        periph_sets = _peripheral_output_sets(params, sets)
                        central_from_p = _depot_infusion_response(
                            periph_sets["central"], u, rate_in, params.ka
                        ) - _depot_infusion_response(
                            periph_sets["central"], u - duration, rate_in, params.ka
                        )
                        peripheral = peripheral + central_from_p
                depot = depot + depot_fill
                central = central + endless
            else:
                depot = depot + _bolus_response({params.ka: 1.0}, u, amount)
                central = central + _bolus_response(sets["depot"], u, amount)
                if params.two_compartments:
                    peripheral = peripheral + _depot_peripheral_response(
                        params, sets, u, amount
                    )
        else:
            coeffs = central_sets[dose.cmt]
            if rate > 0:
                central = central + _infusion_response(coeffs, u, rate, duration)
                if params.two_compartments:
                    peripheral = peripheral + _infusion_response(
                        _peripheral_output_sets(params, sets)[dose.cmt], u, rate, duration
                    )
            else:
                central = central + _bolus_response(coeffs, u, amount)
                if params.two_compartments:
                    peripheral = peripheral + _bolus_response(
                        _peripheral_output_sets(params, sets)[dose.cmt], u, amount
                    )
    return {"depot": depot, "central": central, "peripheral": peripheral}


def _central_output_sets(
    params: LinCmtParams, sets: dict[str, dict[float, float]]
) -> dict[str, dict[float, float]]:
    return {cmt: s for cmt, s in sets.items() if cmt != "peripheral" or params.two_compartments}


def _peripheral_output_sets(
    params: LinCmtParams, sets: dict[str, dict[float, float]]
) -> dict[str, dict[float, float]]:
    """Peripheral-amount coefficient sets (only meaningful with 2 cmt)."""
    if not params.two_compartments:
        return {"central": {}, "peripheral": {}, "depot": {}}
    comp = sol_comp2(params.kel, params.k12, params.k21)
    lams = params.hybrid()
    from_central = {lams[0]: float(comp["c_central"][1]), lams[1]: float(comp["c_central"][3])}
    from_peripheral = {
        lams[0]: float(comp["c_peripheral"][1]),
        lams[1]: float(comp["c_peripheral"][3]),
    }
    if params.ka > 0:
        from_depot = {lam: params.ka * c / (params.ka - lam) for lam, c in from_central.items()}
        from_depot[params.ka] = -sum(from_depot.values())
        return {
            "central": from_central,
            "peripheral": from_peripheral,
            "depot": from_depot,
        }
    return {"central": from_central, "peripheral": from_peripheral}


def _depot_infusion_response(
    central_coeffs: dict[float, float], u: np.ndarray, rate: float, ka: float
) -> np.ndarray:
    """Central response to a constant rate INTO the depot.

    ``sum_lam a * [(1-e^-lu)/l - (e^-lu - e^-kau)/(ka-l)]`` — the
    convolution of the filling depot with the central propagation.
    """
    positive = np.clip(u, 0, None)
    out = np.zeros_like(u)
    for lam, coeff in central_coeffs.items():
        step = (1.0 - np.exp(-lam * positive)) / lam
        bateman = _ratio_exp(lam, ka, positive)
        out = out + coeff * (step - bateman)
    return rate * out


def _depot_peripheral_response(
    params: LinCmtParams,
    sets: dict[str, dict[float, float]],
    u: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Peripheral amount after a depot bolus (2-cmt only)."""
    from_depot = _peripheral_output_sets(params, sets)["depot"]
    return _bolus_response(from_depot, u, amount)


def _ratio_exp(lam: float, ka: float, u: np.ndarray) -> np.ndarray:
    """``(exp(-lam*u) - exp(-ka*u))/(ka-lam)`` with the degenerate limit."""
    if abs(ka - lam) < _DEGENERACY_TOL:
        return u * np.exp(-lam * u)
    return (np.exp(-lam * u) - np.exp(-ka * u)) / (ka - lam)


def lin_cmt_concentration(
    params: LinCmtParams,
    doses: list[Dose],
    t: float | np.ndarray,
) -> np.ndarray:
    """Central-compartment concentration (amount / ``v1``)."""

    profile = lin_cmt_profile(params, doses, t)
    return profile["central"] / params.v1


def lin_cmt_auc_inf(params: LinCmtParams, doses: list[Dose]) -> float:
    """Exact AUC(inf) of the central concentration after all doses.

    By linearity each dose contributes ``amount * sum(coeff/lam)`` over
    its source's coefficient set — the dose-normalised AUC is exactly
    ``1/CL`` for central doses.

    Examples
    --------
    >>> params = LinCmtParams(kel=0.1, k12=3.0, k21=1.0)
    >>> round(lin_cmt_auc_inf(params, [Dose(amount=100.0)]), 6)
    1000.0
    >>> params = LinCmtParams(kel=0.1, ka=1.2, v1=10.0)
    >>> round(lin_cmt_auc_inf(params, [Dose(amount=100.0, cmt='depot')]), 4)
    100.0
    """
    sets = _coefficient_sets(params)
    total = 0.0
    for dose in _expanded_doses(list(doses)):
        if dose.ss:
            raise DataValidationError(
                "AUC(inf) is infinite for steady-state regimens — "
                "use steady_state_auc_interval instead"
            )
        amount = dose.amount * params.f if dose.cmt == "depot" else dose.amount
        rate, duration = _infusion_span(dose)
        if dose.cmt == "depot" and rate > 0:
            rate = rate * params.f
        coeffs = sets[dose.cmt]
        integral_per_unit = sum(c / lam for lam, c in coeffs.items())
        if rate > 0:
            infused = rate * (duration if duration is not None else math.inf)
            total += infused * integral_per_unit
        else:
            total += amount * integral_per_unit
    return total / params.v1


# ---------------------------------------------------------------------------
# steady state (the PolyExp geometric-series limit)
# ---------------------------------------------------------------------------


def steady_state_curve(
    params: LinCmtParams,
    dose: Dose,
    t: float | np.ndarray,
) -> np.ndarray:
    """Concentration at steady state of one periodic regimen.

    ``t`` is absolute time; the phase within the cycle is
    ``(t - dose.time) mod ii``.  Bolus regimens use the geometric-series
    limit ``a*exp(-lam*x)/(1-exp(-lam*tau))``; intermittent infusions
    use the periodic closed form (re-derived, verified against explicit
    cycle summation to 1e-10).

    Examples
    --------
    >>> import numpy as np
    >>> params = LinCmtParams(kel=0.1, v1=1.0)
    >>> dose = Dose(amount=6.0, rate=1.0, ii=24.0, ss=True)
    >>> curve = steady_state_curve(params, dose, t=np.array([0.0, 3.0, 12.0]))
    >>> [round(float(v), 6) for v in curve]
    [0.820218, 3.19945, 2.723219]
    """
    if not dose.ss:
        raise DataValidationError("steady_state_curve needs a ss=True dose")
    params.check_ka()
    t_arr = _as_array(t)
    x = np.mod(t_arr - dose.time, dose.ii)
    sets = _coefficient_sets(params)
    tau = dose.ii
    rate, duration = _infusion_span(dose)

    if dose.cmt == "depot":
        if rate > 0:
            raise DataValidationError(
                "steady-state depot infusions are outside the analytic scope"
            )
        amount = dose.amount * params.f
        out = np.zeros_like(t_arr)
        for lam, coeff in sets["central"].items():
            on = _periodic_step(lam, tau, x)
            off = _periodic_step(params.ka, tau, x)
            out = out + coeff * (on - off) / (params.ka - lam)
        return params.ka * amount * out / params.v1

    coeffs = sets[dose.cmt]
    amount = dose.amount
    if rate > 0:
        # intermittent (or endless) infusion at steady state
        dur = duration if duration is not None else math.inf
        out = np.zeros_like(t_arr)
        for lam, coeff in coeffs.items():
            if math.isinf(dur):
                out = out + coeff / lam  # Css = R * sum a/lam
                continue
            during = x <= dur
            current = np.where(
                during, 1 - np.exp(-lam * x), (1 - np.exp(-lam * dur)) * np.exp(-lam * (x - dur))
            )
            tail = (1 - np.exp(-lam * dur)) * np.exp(-lam * (x + tau - dur))
            out = out + coeff * (current + tail / (1 - np.exp(-lam * tau))) / lam
        return rate * out / params.v1

    out = np.zeros_like(t_arr)
    for lam, coeff in coeffs.items():
        out = out + coeff * np.exp(-lam * x) / (1 - np.exp(-lam * tau))
    return amount * out / params.v1


def _periodic_step(lam: float, tau: float, x: np.ndarray) -> np.ndarray:
    """``exp(-lam*x)/(1-exp(-lam*tau))`` — the periodic impulse train sum."""
    return np.exp(-lam * x) / (1 - np.exp(-lam * tau))


def steady_state_auc_interval(params: LinCmtParams, dose: Dose) -> float:
    """AUC over one dosing interval at steady state.

    By periodicity this equals the single-dose AUC(inf) — the classic
    superposition identity, exact here by linearity.

    Examples
    --------
    >>> params = LinCmtParams(kel=0.1, v1=2.0)
    >>> dose = Dose(amount=100.0, ii=24.0, ss=True)
    >>> round(steady_state_auc_interval(params, dose), 4)
    500.0
    """
    single = Dose(time=dose.time, amount=dose.amount, cmt=dose.cmt, rate=dose.rate,
                  duration=dose.duration)
    return lin_cmt_auc_inf(params, [single])
