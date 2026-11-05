"""Population NLME estimation: FOCEi-lite with nlmixr2 conventions.

A lightweight, pure-numpy(+scipy-optimizer) nonlinear mixed-effects
engine in the spirit of ``nlmixr2``'s FOCEi:

* **inner problem** — per-subject empirical-Bayes (MAP) eta estimates
  by damped Gauss–Newton on the penalised residual
  ``[(y − f)/√R ; η/√ω]`` with the residual variance ``R`` refreshed
  between rounds (the proportional-error *interaction*);
* **outer problem** — Nelder–Mead (``scipy.optimize.minimize``, a
  declared dependency of the toolkit) over the packed vector
  ``[theta, log(omega_diag), log(sigma)]`` — every variance stays
  positive by construction;
* **objective** — the classical FOCEi form (Bauer's NONMEM tutorial /
  nlmixr2 convention): per subject
  ``e'V⁻¹e + ln|V| + η'Ω⁻¹η`` with ``V = R + GΩG'``
  (``G = ∂f/∂η`` at the conditional mode), plus ``N·ln|Ω|``; the
  reported ``objf`` omits the per-observation ``ln 2π`` constants so
  that ``logLik = −(objf + nobs·ln 2π)/2`` exactly as nlmixr2 prints
  it, and ``AIC/BIC`` use ``k = n_theta + n_eta``;
* **diagnostics** — the residual family ``PRED/IPRED/RES/IRES/IWRES/
  CPRED/CWRES`` (CWRES = the conditional linearisation back to η=0
  over ``V``) and the shrinkage table with nlmixr2's exact formulas
  (sd/var shrinkage, epsilon shrinkage ``100·(1 − sd(IWRES))``, the
  t-statistic and its p-value), plus BSV(CV%) = ``100·√(e^ω − 1)``.

What is *not* here (honest boundary): no C++ sensitivity ODEs (the
eta-gradients are finite differences), no full Ω blocks (diagonal
only), no SAEM/AGQ/importance-sampling alternatives, no censoring.

Examples
--------
A one-compartment oral model, 8 synthetic subjects, IIV on ka and CL:

>>> import numpy as np
>>> from pharmaml.pkpd.population import (
...     SubjectObservations, bsv_cv_percent, fit_nlme, simulate_population)

>>> def predict(theta, eta, times):
...     ka = np.exp(theta[0] + eta[0])
...     cl = np.exp(theta[1] + eta[1])
...     v = 20.0
...     ke = cl / v
...     gap = ka - ke
...     if abs(gap) < 1e-9:
...         return 100.0 * ka * times * np.exp(-ke * times) / v
...     return 100.0 * ka / (v * gap) * (np.exp(-ke * times)
...                                     - np.exp(-ka * times))

>>> times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
>>> profiles = simulate_population(
...     predict, theta=np.log([1.0, 3.0]), omega_diag=[0.09, 0.04],
...     times=times, n_subjects=8, seed=7, sigma_add=0.03, sigma_prop=0.05)
>>> subjects = [SubjectObservations(times, values) for values in profiles]

>>> fit = fit_nlme(predict, subjects,
...                theta0=np.log([0.8, 2.6]), omega0=[0.06, 0.03],
...                error_model="combined", sigma0=(0.05, 0.06))

The fixed effects find the ML optimum — with eight subjects it sits
within the study's sampling variability of the truths (ka 1.0, CL 3.0);
the objective / likelihood / information criteria follow nlmixr2's
printing conventions (the AIC may be negative — a continuous
likelihood makes that meaningless-good, not bad):

>>> [round(float(np.exp(t)), 3) for t in fit.theta]
[0.823, 3.277]
>>> fit.n_obs, fit.n_subjects
(48, 8)
>>> (round(fit.log_lik, 2), round(fit.aic, 2), round(fit.bic, 2))
(16.27, -20.55, -9.32)
>>> fit.bic > fit.aic          # BIC pays k*ln(48) instead of 2k
True

Shrinkage on this six-sample-per-subject design is moderate (the
nlmixr2 sd formula ``100·(1 − sd(η̂)/√ω)``):

>>> {name: round(vals["sd_shrink_percent"], 1)
...      for name, vals in fit.shrinkage.items() if name.startswith("eta")}
{'eta_1': 33.4, 'eta_2': 34.4}

The conditional weighted residuals centre on zero as they should:

>>> round(float(np.mean(fit.residuals["CWRES"])), 2)
0.17

BSV(CV%) for an IIV variance of 0.09 (log-normal parameters):

>>> round(bsv_cv_percent(0.09), 2)
30.69
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_positive

__all__ = [
    "NLMEFit",
    "SubjectObservations",
    "bsv_cv_percent",
    "fit_nlme",
    "simulate_population",
]

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(slots=True)
class SubjectObservations:
    """One subject's observations: times, values, covariates."""

    times: np.ndarray
    values: np.ndarray
    covariates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float).reshape(-1)
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if self.times.size != self.values.size or self.times.size == 0:
            raise DataValidationError("times and values must match and be non-empty")
        if np.any(np.diff(self.times) <= 0):
            raise DataValidationError("times must be strictly increasing")


def bsv_cv_percent(omega_variance: float) -> float:
    """Between-subject variability as CV% for a log-normal parameter.

    ``100·√(exp(ω) − 1)`` — the nlmixr2 ``BSV(CV%)`` column for
    exponentially transformed parameters (ω = the IIV *variance* on
    the log scale).
    """
    omega_variance = validate_positive(omega_variance, "omega_variance")
    return 100.0 * math.sqrt(math.exp(omega_variance) - 1.0)


def _residual_variance(
    predictions: np.ndarray, sigma_add: float, sigma_prop: float
) -> np.ndarray:
    """R = σ_add² + (σ_prop·f)² — the interaction-aware residual variance."""
    return sigma_add**2 + (sigma_prop * predictions) ** 2


def _numeric_jacobian(
    function, x: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    """Forward-difference Jacobian of a vector function at ``x``."""
    base = np.asarray(function(x), dtype=float).reshape(-1)
    jac = np.empty((base.size, x.size))
    for j in range(x.size):
        bumped = x.copy()
        delta = step * max(1.0, abs(float(x[j])))
        bumped[j] += delta
        jac[:, j] = (np.asarray(function(bumped), dtype=float).reshape(-1) - base) / delta
    return jac


def _damped_gauss_newton(residuals, x0: np.ndarray, max_iter: int = 30) -> np.ndarray:
    """Minimise ``sum(residuals(x)^2)`` by Levenberg-damped Gauss–Newton."""
    x = np.asarray(x0, dtype=float).copy()
    damping = 1e-3
    best = float(np.asarray(residuals(x)) ** 2 @ np.asarray(residuals(x)))
    for _ in range(max_iter):
        r = np.asarray(residuals(x), dtype=float).reshape(-1)
        jac = _numeric_jacobian(residuals, x)
        normal = jac.T @ jac
        gradient = jac.T @ r
        scale = max(float(np.trace(normal)), 1e-12)
        accepted = False
        for _ in range(8):
            try:
                step = np.linalg.solve(
                    normal + damping * scale * np.eye(x.size), gradient
                )
            except np.linalg.LinAlgError:  # pragma: no cover - singular
                damping *= 10.0
                continue
            candidate = x - step
            cost = float(
                np.asarray(residuals(candidate)) ** 2
                @ np.asarray(residuals(candidate))
            )
            if np.isfinite(cost) and cost <= best:
                improvement = best - cost
                x, best = candidate, cost
                damping = max(damping / 3.0, 1e-9)
                accepted = True
                if improvement < 1e-10:
                    return x
                break
            damping *= 10.0
        if not accepted:
            return x
    return x


def _map_eta(
    predict_fn,
    subject: SubjectObservations,
    theta: np.ndarray,
    omega_diag: np.ndarray,
    sigma_add: float,
    sigma_prop: float,
    eta0: np.ndarray | None = None,
) -> np.ndarray:
    """Per-subject MAP (empirical-Bayes) eta by penalised least squares.

    The residual variance is refreshed between damping rounds so a
    proportional error model interacts with the individual prediction
    (the "i" of FOCEi).
    """
    n_eta = omega_diag.size
    eta = np.zeros(n_eta) if eta0 is None else np.asarray(eta0, dtype=float).copy()
    omega_sd = np.sqrt(np.maximum(omega_diag, 1e-12))

    def penalised(eta_vec: np.ndarray) -> np.ndarray:
        f = np.asarray(
            predict_fn(theta, eta_vec, subject.times), dtype=float
        ).reshape(-1)
        r = _residual_variance(f, sigma_add, sigma_prop)
        data = (subject.values - f) / np.sqrt(np.maximum(r, 1e-12))
        return np.concatenate([data, eta_vec / omega_sd])

    for _ in range(3):
        eta = _damped_gauss_newton(penalised, eta)
    return eta


def _sensitivity(
    predict_fn,
    theta: np.ndarray,
    eta: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """G = ∂f/∂η at the conditional mode (central differences)."""
    n_obs = np.asarray(times).size
    n_eta = eta.size
    grad = np.empty((n_obs, n_eta))
    for j in range(n_eta):
        delta = 1e-5 * max(1.0, abs(float(eta[j])))
        plus = eta.copy()
        minus = eta.copy()
        plus[j] += delta
        minus[j] -= delta
        f_plus = np.asarray(predict_fn(theta, plus, times), dtype=float).reshape(-1)
        f_minus = np.asarray(predict_fn(theta, minus, times), dtype=float).reshape(-1)
        grad[:, j] = (f_plus - f_minus) / (2.0 * delta)
    return grad


def _objf_and_modes(
    predict_fn,
    subjects: list[SubjectObservations],
    theta: np.ndarray,
    omega_diag: np.ndarray,
    sigma_add: float,
    sigma_prop: float,
    *,
    warm_start: list[np.ndarray] | None = None,
) -> tuple[float, list[np.ndarray]]:
    """The FOCEi objective plus the conditional modes it implies."""
    n_subjects = len(subjects)
    objf = n_subjects * float(np.sum(np.log(np.maximum(omega_diag, 1e-12))))
    etas: list[np.ndarray] = []
    for index, subject in enumerate(subjects):
        eta0 = None if warm_start is None else warm_start[index]
        eta = _map_eta(
            predict_fn, subject, theta, omega_diag, sigma_add, sigma_prop, eta0
        )
        etas.append(eta)
        f = np.asarray(
            predict_fn(theta, eta, subject.times), dtype=float
        ).reshape(-1)
        e = subject.values - f
        grad = _sensitivity(predict_fn, theta, eta, subject.times)
        v = _residual_variance(f, sigma_add, sigma_prop)
        for k in range(eta.size):
            v = v + omega_diag[k] * grad[:, k] ** 2
        objf += float(np.sum(e**2 / v + np.log(v)))
        objf += float(np.sum(eta**2 / np.maximum(omega_diag, 1e-12)))
    return objf, etas


def _error_sigma_size(error_model: str) -> int:
    if error_model == "additive":
        return 1
    if error_model in {"proportional", "combined"}:
        return 2
    raise DataValidationError(
        f"error_model must be 'additive', 'proportional' or 'combined', "
        f"got {error_model!r}"
    )


def fit_nlme(
    predict_fn,
    subjects: list[SubjectObservations],
    theta0: np.ndarray | list[float],
    omega0: np.ndarray | list[float],
    *,
    error_model: str = "combined",
    sigma0: tuple[float, ...] | list[float] = (0.1, 0.1),
    outer_maxiter: int = 400,
) -> NLMEFit:
    """Fit a nonlinear mixed-effects model by FOCEi.

    Parameters
    ----------
    predict_fn:
        ``predict_fn(theta, eta, times) -> predictions`` — the
        mu-referenced structural model (e.g. parameters as
        ``exp(theta + eta)``).
    subjects:
        The observations, one :class:`SubjectObservations` each.
    theta0:
        Initial fixed effects (unconstrained scale).
    omega0:
        Initial IIV *variances* (diagonal of Ω), one per eta.
    error_model:
        ``'additive'`` (one sigma: sd), ``'proportional'`` (one sigma:
        fractional sd), or ``'combined'`` (two: additive sd then
        proportional fraction).
    sigma0:
        Initial residual-error parameters for the chosen model.
    outer_maxiter:
        Nelder–Mead iteration cap.

    Returns
    -------
    NLMEFit
        Estimates, objective, residuals, shrinkage and (when the
        outer Hessian cooperates) standard errors + condition number.
    """
    if not subjects:
        raise DataValidationError("fit_nlme needs at least one subject")
    theta_start = np.asarray(theta0, dtype=float).reshape(-1).copy()
    omega_start = np.asarray(omega0, dtype=float).reshape(-1).copy()
    if np.any(omega_start <= 0):
        raise DataValidationError("omega0 entries must be positive variances")
    n_sigma = _error_sigma_size(error_model)
    sigma_start = np.asarray(sigma0, dtype=float).reshape(-1).copy()
    if sigma_start.size != n_sigma or np.any(sigma_start <= 0):
        raise DataValidationError(
            f"error_model {error_model!r} needs {n_sigma} positive sigma value(s)"
        )
    n_theta, n_eta = theta_start.size, omega_start.size

    # Packed optimisation vector: theta | log(omega) | log(sigma).
    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = x[:n_theta]
        omega = np.exp(x[n_theta : n_theta + n_eta])
        sigma = np.exp(x[n_theta + n_eta :])
        return theta, omega, sigma

    def pack(theta: np.ndarray, omega: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.concatenate([theta, np.log(omega), np.log(sigma)])

    def sigmas_of(sigma: np.ndarray) -> tuple[float, float]:
        if error_model == "additive":
            return float(sigma[0]), 0.0
        if error_model == "proportional":
            return 0.0, float(sigma[0])
        return float(sigma[0]), float(sigma[1])

    warm: list[np.ndarray] | None = None

    def objective(x: np.ndarray) -> float:
        nonlocal warm
        theta, omega, sigma = unpack(x)
        s_add, s_prop = sigmas_of(sigma)
        value, etas = _objf_and_modes(
            predict_fn, subjects, theta, omega, s_add, s_prop, warm_start=warm
        )
        warm = etas
        if not math.isfinite(value):
            return 1e12
        return value

    x0 = pack(theta_start, omega_start, sigma_start)
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": outer_maxiter, "xatol": 1e-4, "fatol": 1e-4},
    )
    theta_hat, omega_hat, sigma_hat = unpack(np.asarray(result.x, dtype=float))
    s_add, s_prop = sigmas_of(sigma_hat)
    objf, etas = _objf_and_modes(
        predict_fn, subjects, theta_hat, omega_hat, s_add, s_prop, warm_start=warm
    )
    n_obs = int(sum(s.values.size for s in subjects))
    log_lik = -(objf + n_obs * _LOG_2PI) / 2.0
    k_params = n_theta + n_eta + n_sigma
    aic = -2.0 * log_lik + 2.0 * k_params
    bic = -2.0 * log_lik + k_params * math.log(n_obs)

    # --- residual family (flattened across subjects) --------------------
    columns: dict[str, list[np.ndarray]] = {
        name: [] for name in ("PRED", "IPRED", "RES", "IRES", "IWRES", "CPRED", "CWRES")
    }
    for subject, eta in zip(subjects, etas):
        f_pop = np.asarray(
            predict_fn(theta_hat, np.zeros(n_eta), subject.times), dtype=float
        ).reshape(-1)
        f_ind = np.asarray(
            predict_fn(theta_hat, eta, subject.times), dtype=float
        ).reshape(-1)
        grad = _sensitivity(predict_fn, theta_hat, eta, subject.times)
        r = _residual_variance(f_ind, s_add, s_prop)
        v = r.copy()
        for k in range(n_eta):
            v = v + omega_hat[k] * grad[:, k] ** 2
        c_pred = f_ind - grad @ eta
        columns["PRED"].append(f_pop)
        columns["IPRED"].append(f_ind)
        columns["RES"].append(subject.values - f_pop)
        columns["IRES"].append(subject.values - f_ind)
        columns["IWRES"].append((subject.values - f_ind) / np.sqrt(r))
        columns["CPRED"].append(c_pred)
        columns["CWRES"].append((subject.values - c_pred) / np.sqrt(v))
    residuals = {name: np.concatenate(parts) for name, parts in columns.items()}

    # --- shrinkage (nlmixr2 exact formulas) ------------------------------
    eta_matrix = np.vstack(etas) if n_eta else np.zeros((len(subjects), 0))
    shrinkage: dict[str, dict[str, float]] = {}
    for k in range(n_eta):
        col = eta_matrix[:, k]
        omega_sd = math.sqrt(omega_hat[k])
        sd_eta = float(np.std(col, ddof=1)) if col.size > 1 else float("nan")
        mean_eta = float(np.mean(col))
        if sd_eta > 0 and col.size > 1:
            t_stat = mean_eta / (sd_eta / math.sqrt(col.size))
            try:
                from scipy import stats as _stats

                p_value = float(2.0 * _stats.t.sf(abs(t_stat), col.size - 1))
            except Exception:  # noqa: BLE001  # pragma: no cover
                p_value = float("nan")
        else:
            t_stat, p_value = float("nan"), float("nan")
        shrinkage[f"eta_{k + 1}"] = {
            "sd_shrink_percent": 100.0 * (1.0 - sd_eta / omega_sd),
            "var_shrink_percent": 100.0
            * (1.0 - (float(np.var(col, ddof=1)) if col.size > 1 else 0.0) / omega_hat[k]),
            "t_statistic": t_stat,
            "p_value": p_value,
        }
    iwres_sd = float(np.std(residuals["IWRES"], ddof=1))
    shrinkage["epsilon"] = {
        "sd_shrink_percent": 100.0 * (1.0 - iwres_sd),
        "var_shrink_percent": 100.0 * (1.0 - iwres_sd**2),
        "t_statistic": float("nan"),
        "p_value": float("nan"),
    }

    # --- standard errors via the outer numerical Hessian (theta only) ---
    standard_errors: np.ndarray | None = None
    condition_number: float | None = None
    if n_theta:
        frozen = pack(theta_hat, omega_hat, sigma_hat)

        def theta_objective(theta_vec: np.ndarray) -> float:
            x = frozen.copy()
            x[:n_theta] = theta_vec
            value, _ = _objf_and_modes(
                predict_fn,
                subjects,
                theta_vec,
                omega_hat,
                s_add,
                s_prop,
                warm_start=list(etas),
            )
            return value if math.isfinite(value) else 1e12

        steps = 1e-4 * np.maximum(np.abs(theta_hat), 1.0)
        hessian = np.empty((n_theta, n_theta))
        ok = True
        for i in range(n_theta):
            for j in range(i, n_theta):
                pp = theta_hat.copy(); pp[i] += steps[i]; pp[j] += steps[j]
                pm = theta_hat.copy(); pm[i] += steps[i]; pm[j] -= steps[j]
                mp = theta_hat.copy(); mp[i] -= steps[i]; mp[j] += steps[j]
                mm = theta_hat.copy(); mm[i] -= steps[i]; mm[j] -= steps[j]
                hessian[i, j] = hessian[j, i] = (
                    theta_objective(pp)
                    - theta_objective(pm)
                    - theta_objective(mp)
                    + theta_objective(mm)
                ) / (4.0 * steps[i] * steps[j])
        try:
            covariance = np.linalg.inv(hessian)
            if np.all(np.diagonal(covariance) > 0):
                standard_errors = np.sqrt(np.diagonal(covariance))
                eigenvalues = np.linalg.eigvalsh(covariance)
                positive = eigenvalues[eigenvalues > 0]
                if positive.size == eigenvalues.size:
                    condition_number = float(
                        math.sqrt(positive.max() / positive.min())
                    )
            else:
                ok = False
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            standard_errors = None

    return NLMEFit(
        theta=theta_hat,
        omega_diag=omega_hat,
        sigma=sigma_hat,
        error_model=error_model,
        eta=eta_matrix,
        objf=objf,
        n_obs=n_obs,
        n_subjects=len(subjects),
        log_lik=log_lik,
        aic=aic,
        bic=bic,
        residuals=residuals,
        shrinkage=shrinkage,
        standard_errors=standard_errors,
        condition_number=condition_number,
    )


@dataclass(slots=True)
class NLMEFit:
    """FOCEi-lite fit result (nlmixr2-style reporting conventions)."""

    theta: np.ndarray
    omega_diag: np.ndarray
    sigma: np.ndarray
    error_model: str
    eta: np.ndarray
    objf: float
    n_obs: int
    n_subjects: int
    log_lik: float
    aic: float
    bic: float
    residuals: dict[str, np.ndarray]
    shrinkage: dict[str, dict[str, float]]
    standard_errors: np.ndarray | None = None
    condition_number: float | None = None
    method: str = "focei-lite"

    def parameter_table(self) -> pd.DataFrame:
        """Fixed effects with SE/RSE (a mini ``$parFixed``)."""
        rows = []
        for i, value in enumerate(self.theta):
            se = (
                float(self.standard_errors[i])
                if self.standard_errors is not None
                else float("nan")
            )
            rows.append(
                {
                    "parameter": f"theta_{i + 1}",
                    "estimate": float(value),
                    "se": se,
                    "rse_percent": 100.0 * se / abs(float(value))
                    if se and math.isfinite(se)
                    else float("nan"),
                }
            )
        return pd.DataFrame(rows)


def simulate_population(
    predict_fn,
    theta: np.ndarray | list[float],
    omega_diag: np.ndarray | list[float],
    times: np.ndarray | list[float],
    n_subjects: int,
    *,
    seed: int = 0,
    sigma_add: float = 0.0,
    sigma_prop: float = 0.0,
) -> list[np.ndarray]:
    """Monte-Carlo population profiles (the VPC simulation engine).

    Draws ``eta ~ N(0, diag(omega))`` per subject, evaluates the
    structural model, and optionally adds residual noise
    (additive sd / proportional fraction).
    """
    rng = np.random.default_rng(seed)
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    omega_vec = np.asarray(omega_diag, dtype=float).reshape(-1)
    time_grid = np.asarray(times, dtype=float).reshape(-1)
    if n_subjects < 1:
        raise DataValidationError("n_subjects must be >= 1")
    if np.any(omega_vec < 0):
        raise DataValidationError("omega_diag entries must be >= 0")
    profiles: list[np.ndarray] = []
    for _ in range(n_subjects):
        eta = rng.normal(0.0, np.sqrt(omega_vec)) if omega_vec.size else np.zeros(0)
        f = np.asarray(predict_fn(theta_vec, eta, time_grid), dtype=float).reshape(-1)
        noise = rng.normal(0.0, 1.0, time_grid.size)
        scale = np.sqrt(_residual_variance(f, sigma_add, sigma_prop))
        profiles.append(np.maximum(f + noise * scale, 0.0))
    return profiles
