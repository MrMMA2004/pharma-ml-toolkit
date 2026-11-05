r"""Simulation diagnostics — sensitivity elasticities, population ribbons, residuals.

Round 176 absorbed ``Open-Systems-Pharmacology/OSPSuite-R``, the R
wrapper around the C# engine.  The wrapper plumbing is the
not-vendored part; the pure-math half transfers:

* **Sensitivity analysis** (``R/sensitivity-analysis.R`` +
  ``vignettes/sensitivity-analysis.Rmd``) — the parameter perturbation
  grid ``theta * (1 ± variationRange*i/numberOfSteps)`` (default 2
  steps, ±10%), the dimensionless elasticity ``S = (dPK/PK)/(dθ/θ)``
  ("-1 for AUC vs kidney volume means +10% volume → −10% AUC"), the
  ranking by magnitude and the cumulative ``totalSensitivityThreshold``
  (0.9) filter, in the 4-column ``sa.csv`` exchange format
  (QuantityPath / Parameter / PKParameter / Value).
* **Population aggregation** (``R/plot-with-ospsuite-plots.R``
  ``.aggregateSimulatedData``) — per-time-point quantile ribbons
  (default 5/50/95, linear interpolation), arithmetic ``mean ±
  nsd*SD`` and geometric ``gm ×/÷ gsd^nsd`` bands.
* **Obs↔sim residuals** (``R/data-combined.R`` /
  ``R/utilities-data-combined.R``) — simulated values interpolated onto
  observation times with **no extrapolation** (``approx(rule = 1)``),
  the three residual scales (linear / log / ratio) with the NaN policy
  for non-positive values.
* **Geometric-error propagation under offsets** (``R/data-combined.R``
  830-877) — the rare ``GSD' = GSD^(y/(y+offset))`` first-order
  adjustment and the ``LLOQ' = (LLOQ + offset)*scale`` rule.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.pharma.diagnostics import sensitivity_elasticity
>>> from pharmaml.pharma.lincmt import LinCmtParams, Dose, lin_cmt_concentration
>>> params = LinCmtParams.from_clearance(cl=5.0, v1=50.0)
>>> doses = [Dose(amount=100.0)]
>>> def simulate(overrides):
...     local = LinCmtParams.from_clearance(
...         cl=overrides.get("cl", 5.0), v1=overrides.get("v", 50.0))
...     t = np.linspace(0, 240, 241)
...     return pd.Series(lin_cmt_concentration(local, doses, t), index=t)
>>> def auc(series):
...     return float(_trapezoid(series.values, series.index))
>>> table = sensitivity_elasticity(simulate, auc, {"cl": 5.0, "v": 50.0})
>>> round(float(table.loc["cl", "sensitivity"]), 2)    # AUC = D/CL exactly
-1.0
>>> abs(float(table.loc["v", "sensitivity"])) < 0.01   # AUC independent of V
True

>>> from pharmaml.pharma.diagnostics import aggregate_population
>>> frame = pd.DataFrame({"time": [1.0, 1.0, 1.0, 1.0], "conc": [3.0, 4.0, 5.0, 4.0]})
>>> ribbon = aggregate_population(frame, group_cols=("time",), how="arithmetic")
>>> [round(float(v), 3) for v in ribbon.loc[1.0, ["yMin", "yValues", "yMax"]]]
[3.184, 4.0, 4.816]

>>> from pharmaml.pharma.diagnostics import geometric_sd_offset
>>> round(float(geometric_sd_offset(2.0, 10.0, offset=10.0)), 4)   # 2^(10/20)
1.4142
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

_trapezoid: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
# numpy 1.x spells it trapz, 2.x renamed it trapezoid -- both supported

from pharmaml.exceptions import DataValidationError

__all__ = [
    "aggregate_population",
    "geometric_sd_offset",
    "mixed_error_bounds",
    "obs_sim_residuals",
    "parse_lloq_column",
    "sensitivity_elasticity",
    "steady_state_snap",
]


# ---------------------------------------------------------------------------
# sensitivity analysis (the OSPSuite-R scheme)
# ---------------------------------------------------------------------------


def sensitivity_elasticity(
    simulate: Callable[[Mapping[str, float]], pd.Series],
    pk: Callable[[pd.Series], float],
    reference: Mapping[str, float],
    n_steps: int = 2,
    variation_range: float = 0.1,
    total_threshold: float = 0.9,
    pk_name: str = "PK",
) -> pd.DataFrame:
    """Dimensionless PK-parameter sensitivity elasticities (OSPSuite-R).

    Each parameter is varied on the grid ``theta*(1 ±
    variation_range*i/n_steps)`` for ``i = 1..n_steps`` (the default 2
    steps / 10% range gives 90%, 95%, 105%, 110%); the elasticity is
    the log-log OLS slope of the PK value against the parameter (the
    reference point included).  Rows are ranked by ``|sensitivity|``
    with a cumulative share column; ``in_threshold`` marks the
    parameters contributing to the first ``total_threshold`` (default
    90%) of total sensitivity — ``allPKParameterSensitivitiesFor``.

    Parameters
    ----------
    simulate
        ``overrides -> concentration-time series`` callable.
    pk
        ``series -> scalar PK value`` (AUC, Cmax, ...).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pharmaml.pharma.lincmt import LinCmtParams, Dose, lin_cmt_concentration
    >>> doses = [Dose(amount=100.0)]
    >>> def sim(ov):
    ...     p = LinCmtParams(kel=ov.get("kel", 0.1))
    ...     t = np.linspace(0, 240, 241)
    ...     return pd.Series(lin_cmt_concentration(p, doses, t), index=t)
    >>> table = sensitivity_elasticity(sim, lambda s: float(_trapezoid(s, s.index)),
    ...                                 {"kel": 0.1}, pk_name="AUC")
    >>> round(float(table.loc["kel", "sensitivity"]), 2)   # -1 up to trapezoid error
    -1.0
    """
    if n_steps < 1:
        raise DataValidationError("n_steps must be >= 1")
    if not 0 < variation_range < 1:
        raise DataValidationError("variation_range must be in (0, 1)")

    base_pk = pk(simulate(dict(reference)))
    if base_pk <= 0 or not math.isfinite(base_pk):
        raise DataValidationError("reference PK value must be positive")

    rows: list[dict[str, Any]] = []
    for name, value in reference.items():
        if value == 0:
            raise DataValidationError(f"parameter {name!r} is zero; elasticity undefined")
        fractions = [1.0]
        for i in range(1, n_steps + 1):
            frac = variation_range * i / n_steps
            fractions.extend([1.0 - frac, 1.0 + frac])
        log_theta: list[float] = []
        log_pk: list[float] = []
        for frac in fractions:
            overrides = dict(reference)
            overrides[name] = value * frac
            value_pk = pk(simulate(overrides))
            if value_pk <= 0 or not math.isfinite(value_pk):
                continue
            log_theta.append(math.log(frac))
            log_pk.append(math.log(value_pk))
        if len(log_theta) < 2:
            rows.append({"parameter": name, "pk_parameter": pk_name,
                         "sensitivity": float("nan"), "cumulative": float("nan"),
                         "in_threshold": False})
            continue
        x = np.array(log_theta)
        y = np.array(log_pk)
        slope = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
        rows.append({"parameter": name, "pk_parameter": pk_name, "sensitivity": slope,
                     "cumulative": float("nan"), "in_threshold": False})

    frame = pd.DataFrame(rows).sort_values(
        "sensitivity", key=abs, ascending=False
    ).reset_index(drop=True)
    total = frame["sensitivity"].abs().sum()
    if total > 0:
        frame["cumulative"] = frame["sensitivity"].abs().cumsum() / total
        frame["in_threshold"] = frame["cumulative"] <= total_threshold + 1e-12
    return frame.set_index("parameter")


# ---------------------------------------------------------------------------
# population aggregation (the .aggregateSimulatedData ribbons)
# ---------------------------------------------------------------------------


def aggregate_population(
    frame: pd.DataFrame,
    value_col: str = "conc",
    group_cols: Sequence[str] = ("time",),
    how: str = "quantiles",
    quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
    nsd: float = 1.0,
) -> pd.DataFrame:
    """Aggregate a population's simulated output into a ribbon per time.

    ``how='quantiles'`` gives ``yMin/yValues/yMax`` at the quantiles
    (linear interpolation, R's type-7 default); ``'arithmetic'`` the
    mean ± ``nsd``·SD; ``'geometric'`` the geometric mean ×/÷
    ``gsd^nsd``.  Grouping follows the OSPSuite ``by = .(group, name,
    xValues)`` convention — anything grouping the rows identifies an
    output curve.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"time": [1.0]*4, "conc": [1.0, 2.0, 4.0, 8.0]})
    >>> ribbon = aggregate_population(frame, how="quantiles", quantiles=(0.25, 0.5, 0.75))
    >>> [round(float(v), 3) for v in ribbon.loc[1.0, ["yMin", "yValues", "yMax"]]]
    [1.75, 3.0, 5.0]
    """
    if how not in {"quantiles", "arithmetic", "geometric"}:
        raise DataValidationError("how must be 'quantiles', 'arithmetic' or 'geometric'")
    if value_col not in frame.columns:
        raise DataValidationError(f"frame lacks column {value_col!r}")
    cols = list(group_cols)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise DataValidationError(f"frame lacks group columns {missing}")

    def _quantile(group: pd.DataFrame) -> pd.Series:
        values = group[value_col].dropna()
        low, mid, high = np.quantile(values, quantiles, method="linear")
        return pd.Series({"yMin": low, "yValues": mid, "yMax": high})

    def _arithmetic(group: pd.DataFrame) -> pd.Series:
        values = group[value_col].dropna()
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return pd.Series(
            {"yMin": mean - nsd * sd, "yValues": mean, "yMax": mean + nsd * sd}
        )

    def _geometric(group: pd.DataFrame) -> pd.Series:
        values = group[value_col].dropna()
        values = values[values > 0]
        if values.empty:
            return pd.Series({"yMin": np.nan, "yValues": np.nan, "yMax": np.nan})
        logs = np.log(values.values)
        gm = float(np.exp(logs.mean()))
        gsd = float(np.exp(logs.std(ddof=1))) if len(values) > 1 else 1.0
        return pd.Series(
            {
                "yMin": gm / gsd**nsd,
                "yValues": gm,
                "yMax": gm * gsd**nsd,
            }
        )

    reducer = {"quantiles": _quantile, "arithmetic": _arithmetic, "geometric": _geometric}[how]
    return frame.groupby(cols, sort=True).apply(reducer, include_groups=False)


# ---------------------------------------------------------------------------
# obs vs sim residuals (R/data-combined.R)
# ---------------------------------------------------------------------------


def obs_sim_residuals(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    scale: str = "log",
    time_col: str = "time",
    value_col: str = "conc",
) -> pd.DataFrame:
    """Residuals of observed vs interpolated simulated values.

    Simulated values are linearly interpolated onto the observation
    times with **no extrapolation** (outside the simulated window →
    NaN, R's ``approx(rule = 1)``).  Scales: ``'linear'``
    (pred − obs), ``'log'`` (``log(pred) − log(obs)``, NaN where either
    is non-positive), ``'ratio'`` (obs / pred, NaN where pred ≤ 0).

    Examples
    --------
    >>> import pandas as pd
    >>> obs = pd.DataFrame({"time": [0.5, 1.5], "conc": [9.0, 4.0]})
    >>> sim = pd.DataFrame({"time": [0.0, 1.0, 2.0], "conc": [10.0, 5.0, 2.5]})
    >>> res = obs_sim_residuals(obs, sim, scale="linear")
    >>> [round(float(v), 4) for v in res["residual"]]   # pred at 0.5 = 7.5
    [-1.5, -0.25]
    >>> res_log = obs_sim_residuals(obs, sim, scale="log")
    >>> round(float(res_log["residual"][0]), 4)      # log(7.5) - log(9)
    -0.1823
    """
    if scale not in {"linear", "log", "ratio"}:
        raise DataValidationError("scale must be 'linear', 'log' or 'ratio'")
    for frame, label in ((observed, "observed"), (simulated, "simulated")):
        for col in (time_col, value_col):
            if col not in frame.columns:
                raise DataValidationError(f"{label} frame lacks column {col!r}")
    sim_t = simulated[time_col].to_numpy(dtype=float)
    sim_v = simulated[value_col].to_numpy(dtype=float)
    order = np.argsort(sim_t)
    sim_t = sim_t[order]
    sim_v = sim_v[order]
    obs_t = observed[time_col].to_numpy(dtype=float)
    obs_v = observed[value_col].to_numpy(dtype=float)

    pred = np.interp(obs_t, sim_t, sim_v)
    outside = (obs_t < sim_t[0]) | (obs_t > sim_t[-1])
    pred = np.where(outside, np.nan, pred)

    if scale == "linear":
        residual = pred - obs_v
    elif scale == "log":
        ok = (pred > 0) & (obs_v > 0) & ~np.isnan(pred)
        residual = np.where(ok, np.log(pred) - np.log(obs_v), np.nan)
    else:
        ok = (pred > 0) & ~np.isnan(pred)
        residual = np.where(ok, obs_v / pred, np.nan)
    return pd.DataFrame(
        {"time": obs_t, "observed": obs_v, "simulated": pred, "residual": residual}
    )


# ---------------------------------------------------------------------------
# geometric error propagation under offsets
# ---------------------------------------------------------------------------


def geometric_sd_offset(
    gsd: float | np.ndarray,
    y: float | np.ndarray,
    offset: float,
) -> float | np.ndarray:
    """Geometric SD adjusted for a y-offset (OSPSuite-R NEWS #2046).

    ``newError = rawError^(rawValue/(rawValue + offset))`` — the
    first-order correction because "a y offset other than 0 changes
    the spread of the values on the log scale".  NaN where ``y`` or
    ``y + offset`` is non-positive.

    Examples
    --------
    >>> round(float(geometric_sd_offset(2.0, 10.0, offset=10.0)), 4)
    1.4142
    """
    gsd_arr = np.asarray(gsd, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    shifted = y_arr + offset
    valid = (y_arr > 0) & (shifted > 0)
    with np.errstate(invalid="ignore"):
        result = np.where(valid, gsd_arr ** (y_arr / np.where(shifted == 0, 1.0, shifted)), np.nan)
    if np.isscalar(gsd) and np.isscalar(y):
        return float(result)
    return result


def mixed_error_bounds(
    values: Sequence[float],
    errors: Sequence[float],
    kinds: Sequence[str],
) -> pd.DataFrame:
    """Convert mixed arithmetic/geometric errors to yMin/yMax bounds.

    Geometric rows (``kinds[i] == 'geometric'``) become
    ``y/gsd .. y*gsd``; arithmetic rows ``y ± sd`` — the OSPSuite-R
    ``.convertInconsistentErrorTypes`` rule.

    Examples
    --------
    >>> frame = mixed_error_bounds([10.0, 10.0], [2.0, 2.0], ["arithmetic", "geometric"])
    >>> [round(float(v), 3) for v in frame["yMin"]], [round(float(v), 2) for v in frame["yMax"]]
    ([8.0, 5.0], [12.0, 20.0])
    """
    if not (len(values) == len(errors) == len(kinds)):
        raise DataValidationError("values, errors and kinds must align")
    rows = []
    for value, error, kind in zip(values, errors, kinds):
        if kind == "geometric":
            rows.append({"y": value, "yMin": value / error, "yMax": value * error})
        elif kind == "arithmetic":
            rows.append({"y": value, "yMin": value - error, "yMax": value + error})
        else:
            raise DataValidationError(f"unknown error kind {kind!r}")
    return pd.DataFrame(rows)


_LLOQ_PATTERN = re.compile(r"^\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def parse_lloq_column(column: Sequence[str | float]) -> pd.DataFrame:
    """Parse the ``'<0.001'`` censored convention (OSPSuite-R importer).

    Values written as ``'< x'`` mark observations below the LLOQ: the
    parsed frame carries the numeric value, the lloq and a censored
    flag.

    Examples
    --------
    >>> frame = parse_lloq_column([1.2, "<0.001", 0.5])
    >>> list(frame["censored"])
    [False, True, False]
    >>> round(float(frame.loc[1, "lloq"]), 4)
    0.001
    """
    values: list[float] = []
    lloqs: list[float] = []
    censored: list[bool] = []
    for raw in column:
        if isinstance(raw, str):
            match = _LLOQ_PATTERN.match(raw)
            if not match:
                raise DataValidationError(f"unparseable censored value {raw!r}")
            limit = float(match.group(1))
            values.append(limit)
            lloqs.append(limit)
            censored.append(True)
        else:
            values.append(float(raw))
            lloqs.append(float("nan"))
            censored.append(False)
    return pd.DataFrame({"value": values, "lloq": lloqs, "censored": censored})


def steady_state_snap(
    values: np.ndarray | Sequence[float],
    threshold: float = 1e-15,
) -> np.ndarray:
    """Snap near-zero steady-state values to exact zero (getSteadyState).

    After simulating to the last application + 3*24*60 minutes,
    OSPSuite-R zeroes every state with ``|v| <= 1e-15``.

    Examples
    --------
    >>> steady_state_snap([1e-16, 0.5, -2e-16]).tolist()
    [0.0, 0.5, 0.0]
    """
    arr = np.asarray(values, dtype=float).copy()
    arr[np.abs(arr) <= threshold] = 0.0
    return arr
