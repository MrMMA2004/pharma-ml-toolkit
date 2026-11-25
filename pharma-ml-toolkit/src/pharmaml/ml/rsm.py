"""Response Surface Methodology (RSM) -- part17 §6 (the R1 lane).

The scientific grounding, stated once up front: Response Surface
Methodology (Box & Wilson, 1951) is a *designed-experiment* modelling
discipline -- you first plan a design (an experiment matrix whose
factor levels are chosen so the second-order model is estimable),
run the experiments, fit the full quadratic polynomial in coded
factors, inspect its ANOVA and diagnostics, visualise the surface,
and only then optimise (single response directly; several responses
through the Derringer-Suich desirability transformation).  The
mandate's own distinction table holds: RSM is NOT the generic
multi-response ML of part17 §5 (that lane predicts; this lane
*explains* a designed experiment), NOT multi-objective optimisation
(the desirability scale answers "how acceptable is each response"
before any Pareto reasoning), NOT drug-response matrix
factorisation (no cell-line panels here), and NOT an ordinary 3-D
visualisation (the surfaces are fitted models with confidence
geometry, not plotted raw data).

What lands:

* :func:`build_design` -- the §6.3 design generators: central
  composite (rotatable ``alpha = (2k)**(1/4)`` / faced ``alpha = 1``
  / custom), Box-Behnken, and the full 2^k factorial, each with
  factor-range validation, coded and actual level tables, center
  points, whole-design replicates, seeded within-block
  randomisation, the classic two-block CCD split, and the honest
  assumptions stated (a rotatable CCD's axial runs sit OUTSIDE the
  declared factor ranges -- disclosed, never hidden).
* :func:`build_terms` + :func:`model_matrix` -- the §6.4
  second-order term registry (intercept, k linear, k(k-1)/2
  interaction, k quadratic terms) and its coded model matrix.
* :func:`fit_response_surface` -- the honest fit: least squares
  (minimum-norm pinv under rank deficiency, the rank and the
  aliased-term set DISCLOSED), the coefficient table with standard
  errors / t / p / per-term partial SS, the ANOVA partition
  (regression / residual / pure error / lack-of-fit, the last two
  only where replicated design points exist -- a null elsewhere,
  never invented), hierarchy notes (a significant interaction or
  quadratic with an insignificant parent linear term is FLAGGED,
  never auto-removed), the full diagnostics block (R2, adjusted R2,
  predicted R2 over the exact leave-one-out PRESS, RMSE, leverage,
  internal studentised residuals, Cook's distance with stated
  thresholds), and fitted-vs-observed with residual rows.
* :class:`ResponseSurfaceBundle` -- the persistence carrier: the
  factor specs, the term registry, one coefficient vector per
  response, the statistics, the coded design matrix and observed
  responses (the optimiser's start set and the desirability
  auto-bounds), and the JSON-safe contract -- picklable through the
  session model store and the joblib artifact lane.
* :func:`predict_surface` -- point predictions in ACTUAL units with
  the mean confidence interval and the new-observation prediction
  interval, and the honest zone vocabulary per point (inside the
  factor ranges / the CCD's axial region / outside the design
  region -- extrapolation is flagged, never silently extrapolated).
* :func:`contour_grid` -- the §6.5 visualisation data: a two-factor
  grid with every other factor held at explicitly displayed fixed
  values, either a chosen response's prediction or the combined
  desirability under the request's goals.
* :func:`optimize_desirability` -- the §6.6 multi-response
  optimiser: minimise / maximise / target / in-range goals, the
  Derringer-Suich individual desirabilities with per-response
  weights and exponents, the combined geometric desirability,
  optional linear factor constraints (over coded factors, honoured
  by SLSQP), a deterministic multi-start search confined to the
  coded box (the stated factor ranges -- extrapolation cannot occur
  by construction, and the prediction lane flags it separately),
  the per-factor sensitivity sweep, the curated Pareto alternatives
  (per-response individual optima + weight-tilted optima -- a
  deterministic sample, honestly not an exhaustive front), and the
  feasible-region mask for the chosen factor pair.
* :func:`demo_rsm_dataset` -- the seeded SYNTHETIC experiment frame:
  a 17-run rotatable CCD over three factors (temperature / pH /
  substrate concentration) with two known noise-free quadratic
  responses, labelled synthetic wherever it travels.  The surfaces
  are KNOWN, so the tests verify exact coefficient recovery and
  the optima verify against the analytic stationary conditions.

The honest scope: no mixture/process-noise modelling, no
non-polynomial surface families, no Bayesian RSM; blocking is the
classic two-block CCD split only (Box-Behnken and full-factorial
blocking are refused with a typed error, never faked); lack-of-fit
needs genuinely replicated design points (no replicates, no pure
error, a null -- stated); and every optimisation number is a model
estimate over the fitted surface, never an experimental fact.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy
from numpy.linalg import matrix_rank
from scipy import stats as sps
from scipy.linalg import qr as _qr_pivot
from scipy.optimize import minimize

__all__ = [
    "ALPHA_TYPES",
    "DEMO_FACTORS",
    "DEMO_RESPONSES",
    "DESIGN_TYPES",
    "GOALS",
    "MAX_CONSTRAINTS",
    "MAX_DESIGN_RUNS",
    "MAX_FACTORS",
    "MAX_GRID_POINTS",
    "MAX_RESPONSES",
    "MIN_FACTORS",
    "MIN_GRID_POINTS",
    "FactorSpec",
    "ResponseSurfaceBundle",
    "build_design",
    "build_terms",
    "contour_grid",
    "demo_rsm_dataset",
    "environment_metadata",
    "fit_response_surface",
    "model_matrix",
    "optimize_desirability",
    "predict_surface",
    "rsm_registry_payload",
]

# ---------------------------------------------------------------------------
# bounds -- stated once, enforced everywhere, never silently clamped
# ---------------------------------------------------------------------------

MIN_FACTORS = 2
MAX_FACTORS = 10
MIN_GRID_POINTS = 5
MAX_GRID_POINTS = 60
MAX_DESIGN_RUNS = 512
MAX_RESPONSES = 8
MAX_CONSTRAINTS = 16

DESIGN_TYPES = ("ccd", "box_behnken", "full_factorial")
ALPHA_TYPES = ("rotatable", "faced", "custom")
GOALS = ("maximize", "minimize", "target", "range")
_MAX_SEED = 2147483647


# ---------------------------------------------------------------------------
# factors / coding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSpec:
    """One declared factor: name, range, optional unit.

    The coded transform is the classic ``x = (actual - center) /
    half_range`` so the design lives in ``[-1, +1]`` per factor.
    """

    name: str
    low: float
    high: float
    unit: str | None = None

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def half_range(self) -> float:
        return (self.high - self.low) / 2.0

    def code(self, actual: float) -> float:
        return (float(actual) - self.center) / self.half_range

    def decode(self, coded: float) -> float:
        return self.center + self.half_range * float(coded)

    def view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "unit": self.unit,
            "center": self.center,
            "half_range": self.half_range,
        }


def _validate_factors(factors: list[FactorSpec]) -> None:
    """The typed factor contract: 2..10, unique names, low < high."""
    if not (MIN_FACTORS <= len(factors) <= MAX_FACTORS):
        raise ValueError(
            f"an RSM design needs {MIN_FACTORS}..{MAX_FACTORS} factors "
            f"({len(factors)} given)"
        )
    names = [f.name for f in factors]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"factor names must be unique (duplicates: {dupes})")
    for f in factors:
        if not (np.isfinite(f.low) and np.isfinite(f.high)):
            raise ValueError(f"factor {f.name!r} needs finite low/high bounds")
        if f.low >= f.high:
            raise ValueError(
                f"factor {f.name!r} needs low < high ({f.low} >= {f.high})"
            )


def _ccd_alpha(n_factors: int, alpha_type: str, alpha: float | None) -> float:
    """The CCD axial distance: rotatable (2k)**(1/4), faced = 1, custom."""
    if alpha_type == "rotatable":
        return float((2.0**n_factors) ** 0.25)
    if alpha_type == "faced":
        return 1.0
    if alpha is None:
        raise ValueError("a custom alpha needs the alpha value itself")
    if not (0.5 < alpha <= 4.0):
        raise ValueError(f"the custom alpha must lie in (0.5, 4.0] ({alpha})")
    return float(alpha)


# ---------------------------------------------------------------------------
# §6.3 -- the design generators
# ---------------------------------------------------------------------------


def _factorial_rows(n_factors: int) -> list[np.ndarray]:
    """The 2^k corner rows in coded space (all +/-1 combinations)."""
    rows: list[np.ndarray] = []
    for code in range(2**n_factors):
        rows.append(
            np.array(
                [-1.0 if (code >> i) & 1 == 0 else 1.0 for i in range(n_factors)]
            )
        )
    return rows


def _box_behnken_rows(n_factors: int) -> list[np.ndarray]:
    """The Box-Behnken rows: each factor PAIR at its 2^2 corners with
    every other factor held at the center -- all levels inside [-1, 1]
    (the design never leaves the declared factor ranges)."""
    rows: list[np.ndarray] = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    row = np.zeros(n_factors)
                    row[i] = si
                    row[j] = sj
                    rows.append(row)
    return rows


def _design_rows(
    n_factors: int, design: str, alpha: float
) -> tuple[list[np.ndarray], list[str]]:
    """The un-replicated coded rows with their run types."""
    rows: list[np.ndarray] = []
    kinds: list[str] = []
    if design == "ccd":
        for row in _factorial_rows(n_factors):
            rows.append(row)
            kinds.append("factorial")
        for i in range(n_factors):
            for s in (-alpha, alpha):
                row = np.zeros(n_factors)
                row[i] = s
                rows.append(row)
                kinds.append("axial")
    elif design == "box_behnken":
        for row in _box_behnken_rows(n_factors):
            rows.append(row)
            kinds.append("pair")
    elif design == "full_factorial":
        for row in _factorial_rows(n_factors):
            rows.append(row)
            kinds.append("factorial")
    else:  # pragma: no cover - registry guard
        raise ValueError(
            f"unknown design {design!r} (choose from {list(DESIGN_TYPES)})"
        )
    return rows, kinds


def build_design(
    factors: list[FactorSpec],
    design: str,
    *,
    alpha_type: str = "rotatable",
    alpha: float | None = None,
    center_points: int = 3,
    replicates: int = 1,
    randomize: bool = True,
    seed: int = 42,
    blocks: bool = False,
) -> dict[str, Any]:
    """Generate one design of experiments (§6.3).

    Returns the run table (coded AND actual levels, run order, block,
    run type), the design metadata, the run-type census, and the honest
    assumptions.  Randomisation shuffles the run order WITHIN each
    block (the block order itself stays fixed -- the time order of
    blocks is an experimental-protocol decision, stated, not randomised).
    """
    _validate_factors(factors)
    if design not in DESIGN_TYPES:
        raise ValueError(
            f"unknown design {design!r} (choose from {list(DESIGN_TYPES)})"
        )
    if design == "box_behnken" and len(factors) < 3:
        raise ValueError(
            f"a Box-Behnken design needs at least 3 factors ({len(factors)} given)"
        )
    if design == "full_factorial" and len(factors) > 8:
        raise ValueError(
            f"the full factorial design is capped at 8 factors "
            f"({len(factors)} given -- 2^{len(factors)} corners would "
            "exceed the 512-run bound; choose ccd or box_behnken)"
        )
    if not (0 <= center_points <= 50):
        raise ValueError(f"center_points must lie in 0..50 ({center_points})")
    if not (1 <= replicates <= 20):
        raise ValueError(f"replicates must lie in 1..20 ({replicates})")
    if not (0 <= seed <= _MAX_SEED):
        raise ValueError(f"the seed must lie in 0..{_MAX_SEED} ({seed})")
    if blocks and design != "ccd":
        raise ValueError(
            "blocking is supported for the CCD two-block split only "
            "(the Box-Behnken / full-factorial block structures are "
            "honestly refused, never faked)"
        )
    if blocks and replicates > 1:
        raise ValueError(
            "blocking rides the un-replicated two-block split only "
            "(set replicates=1 or blocks=false)"
        )

    k = len(factors)
    alpha_value = _ccd_alpha(k, alpha_type, alpha) if design == "ccd" else None
    base_rows, base_kinds = _design_rows(k, design, alpha_value or 1.0)

    rows: list[np.ndarray] = []
    kinds: list[str] = []
    for _ in range(replicates):
        rows.extend(base_rows)
        kinds.extend(base_kinds)
    for _ in range(center_points):
        rows.append(np.zeros(k))
        kinds.append("center")

    n_runs = len(rows)
    if n_runs > MAX_DESIGN_RUNS:
        raise ValueError(
            f"the design would need {n_runs} runs (> {MAX_DESIGN_RUNS}) "
            "-- reduce factors, replicates, or center points"
        )

    # the two-block CCD split: the factorial block carries ceil(centers/2)
    # center points, the axial block carries the rest
    centers_first = (center_points + 1) // 2 if blocks else center_points
    block_of: list[int] = []
    center_seen = 0
    for kind in kinds:
        if blocks:
            if kind == "factorial":
                block_of.append(1)
            elif kind == "axial":
                block_of.append(2)
            else:
                center_seen += 1
                block_of.append(1 if center_seen <= centers_first else 2)
        else:
            block_of.append(1)

    order = list(range(n_runs))
    if randomize:
        rng = np.random.default_rng(seed)
        if blocks:
            for block in (1, 2):
                idx = [i for i in range(n_runs) if block_of[i] == block]
                shuffled = rng.permutation(idx).tolist()
                pos = 0
                for i in range(n_runs):
                    if block_of[i] == block:
                        order[i] = int(shuffled[pos])
                        pos += 1
        else:
            order = rng.permutation(order).tolist()

    runs: list[dict[str, Any]] = []
    for rank_pos, row_index in enumerate(order, start=1):
        coded = rows[row_index]
        actual = {
            f.name: round(f.decode(v), 6) for f, v in zip(factors, coded)
        }
        runs.append(
            {
                "run_id": f"r{row_index + 1:03d}",
                "run_order": rank_pos,
                "block": block_of[row_index],
                "run_type": kinds[row_index],
                "coded": {f.name: round(float(v), 6) for f, v in zip(factors, coded)},
                "actual": actual,
            }
        )

    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1

    assumptions: list[str] = []
    if (
        design == "ccd"
        and alpha_type == "rotatable"
        and alpha_value is not None
        and alpha_value > 1.0
    ):
        assumptions.append(
            f"the rotatable alpha ({alpha_value:.3f} = (2*{k})**(1/4)) places "
            "the axial runs OUTSIDE the declared factor ranges "
            f"(coded +/-{alpha_value:.3f}); the ranges themselves stay the "
            "box the optimiser is confined to -- the axial region is "
            "disclosed here so it is never silently extrapolated"
        )
    if center_points == 0:
        assumptions.append(
            "zero center points: in a 2-level design the quadratic "
            "terms alias with the intercept -- the fit reports the "
            "aliasing honestly instead of inventing curvature estimates"
        )
    if replicates == 1 and center_points < 2:
        assumptions.append(
            "no replicated design points: pure error and the "
            "lack-of-fit F test are NOT computable (the fit reports "
            "nulls, never invented values)"
        )
    assumptions.append(
        "the coded levels are exact transforms of the declared ranges "
        "(center = (low+high)/2, unit = (high-low)/2); the run table "
        "carries BOTH coded and actual levels so the experiment sheet "
        "never needs re-derivation"
    )

    return {
        "factors": [f.view() for f in factors],
        "design": {
            "type": design,
            "alpha": alpha_value,
            "alpha_type": alpha_type if design == "ccd" else None,
            "n_factors": k,
            "runs": n_runs,
            "center_points": center_points,
            "replicates": replicates,
            "randomized": randomize,
            "seed": seed,
            "blocks": blocks,
        },
        "runs": runs,
        "run_type_counts": counts,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# §6.4 -- the second-order term registry and model matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """One model term: its registry key, kind, and human label."""

    key: str
    kind: str  # intercept | linear | interaction | quadratic
    label: str


def build_terms(factors: list[FactorSpec]) -> list[Term]:
    """The second-order term registry: 1 + k linear + C(k,2) interaction
    + k quadratic terms, in a STABLE order (the bundle's coefficient
    vectors ride this order)."""
    k = len(factors)
    terms = [Term("intercept", "intercept", "Intercept")]
    for i, f in enumerate(factors, start=1):
        terms.append(Term(f"x{i}", "linear", f"{f.name} (linear)"))
    for i in range(1, k + 1):
        for j in range(i + 1, k + 1):
            terms.append(
                Term(
                    f"x{i}x{j}",
                    "interaction",
                    f"{factors[i - 1].name}*{factors[j - 1].name}",
                )
            )
    for i, f in enumerate(factors, start=1):
        terms.append(Term(f"x{i}_sq", "quadratic", f"{f.name}^2"))
    return terms


def model_matrix(coded: np.ndarray, terms: list[Term]) -> np.ndarray:
    """The coded model matrix: one column per registry term.

    ``coded`` is (n, k) in coded units; the result is (n, p) with the
    intercept column first, then linear, interaction, quadratic.
    """
    coded = np.atleast_2d(np.asarray(coded, dtype=float))
    cols: list[np.ndarray] = []
    for term in terms:
        if term.kind == "intercept":
            cols.append(np.ones(coded.shape[0]))
        elif term.kind == "linear":
            cols.append(coded[:, int(term.key[1:]) - 1])
        elif term.kind == "interaction":
            i, j = int(term.key[1]), int(term.key[3])
            cols.append(coded[:, i - 1] * coded[:, j - 1])
        elif term.kind == "quadratic":
            cols.append(coded[:, int(term.key[1]) - 1] ** 2)
        else:  # pragma: no cover - registry guard
            raise ValueError(f"unknown term kind {term.kind!r}")
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


def _alias_disclosure(X: np.ndarray, terms: list[Term]) -> dict[str, Any] | None:
    """The rank/aliasing disclosure: QR with column pivoting over X'X
    names the linearly dependent (aliased) term columns honestly."""
    p = X.shape[1]
    rank = int(matrix_rank(X))
    if rank == p:
        return None
    try:
        _q, _r, pivots = _qr_pivot(X.T @ X, pivoting=True, mode="economic")
    except ValueError:  # pragma: no cover - degenerate guard
        return {
            "rank": rank,
            "columns": p,
            "aliased": None,
            "note": (
                f"the model matrix is rank deficient (rank {rank} of {p} "
                "columns) -- the minimum-norm solution is reported and "
                "the aliased set could not be enumerated"
            ),
        }
    dependent = [int(i) for i in range(p) if i not in {int(x) for x in pivots[:rank]}]
    aliased_keys = [terms[i].key for i in dependent]
    return {
        "rank": rank,
        "columns": p,
        "aliased": aliased_keys,
        "note": (
            f"the model matrix is rank deficient (rank {rank} of {p} "
            "columns): the listed terms are linear combinations of the "
            "others (aliased) -- the fit reports the minimum-norm "
            "solution and the aliased terms' coefficients are NOT "
            "separately interpretable"
        ),
    }


def _anova_partition(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    terms: list[Term],
    xtx_inv: np.ndarray,
    s2: float,
) -> dict[str, Any]:
    """The ANOVA rows + the term-level partial SS (each term's
    ``beta_j**2 / c_jj`` -- the adjusted sum of squares, which for the
    orthogonal-coded design coincides with the sequential one)."""
    n = X.shape[0]
    rank = int(matrix_rank(X))
    df_res = n - rank
    df_reg = rank - 1
    fitted = X @ beta
    resid = y - fitted
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = sst - sse

    anova: dict[str, Any] = {
        "regression": {
            "ss": ssr,
            "df": df_reg,
            "ms": ssr / df_reg if df_reg > 0 else None,
            "f": (ssr / df_reg) / s2 if df_reg > 0 and s2 > 0 else None,
            "p": float(sps.f.sf((ssr / df_reg) / s2, df_reg, df_res))
            if df_reg > 0 and s2 > 0 and df_res > 0
            else None,
        },
        "residual": {"ss": sse, "df": df_res, "ms": s2},
        "total": {"ss": sst, "df": n - 1},
    }

    # replicated coded design points -> pure error / lack of fit
    uniq, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    sspe = 0.0
    df_pe = 0
    for g in range(len(uniq)):
        if counts[g] > 1:
            group_y = y[inverse == g]
            sspe += float(((group_y - group_y.mean()) ** 2).sum())
            df_pe += int(counts[g] - 1)
    if df_pe > 0:
        ss_lof = sse - sspe
        df_lof = df_res - df_pe
        ms_pe = sspe / df_pe
        if df_lof > 0 and ms_pe > 0.0:
            f_lof = (ss_lof / df_lof) / ms_pe
            p_lof = float(sps.f.sf(f_lof, df_lof, df_pe))
            lof_note = (
                "a significant lack-of-fit F says the second-order "
                "polynomial does not capture the response's behaviour "
                "over the design region -- investigate before optimising"
            )
        else:
            f_lof = None
            p_lof = None
            lof_note = (
                "the pure-error variance is exactly zero (noise-free "
                "replicates) so the lack-of-fit F ratio is UNDEFINED "
                "-- reported as null, never invented"
            )
        anova["pure_error"] = {"ss": sspe, "df": df_pe, "ms": ms_pe}
        anova["lack_of_fit"] = {
            "ss": ss_lof,
            "df": df_lof,
            "ms": ss_lof / df_lof if df_lof > 0 else None,
            "f": f_lof,
            "p": p_lof,
            "note": lof_note,
        }
    else:
        anova["pure_error"] = None
        anova["lack_of_fit"] = None
        anova["lack_of_fit_note"] = (
            "no replicated design points: pure error (and with it the "
            "lack-of-fit F) is NOT computable -- nulls, never invented"
        )

    rows: list[dict[str, Any]] = []
    for j, term in enumerate(terms):
        if term.kind == "intercept":
            continue
        c_jj = float(xtx_inv[j, j])
        ss = float(beta[j] ** 2 / c_jj) if c_jj > 0 else 0.0
        f_j = ss / s2 if s2 > 0 else None
        rows.append(
            {
                "term": term.key,
                "label": term.label,
                "ss": ss,
                "f": f_j,
                "p": float(sps.f.sf(f_j, 1, df_res))
                if f_j is not None and df_res > 0
                else None,
            }
        )
    anova["terms"] = rows
    return anova


def fit_response_surface(
    frame: pd.DataFrame,
    factors: list[FactorSpec],
    response_columns: list[str],
    *,
    design_meta: dict[str, Any] | None = None,
    confidence: float = 0.95,
    provenance_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the full second-order model for EVERY response column (§6.4).

    The frame carries the OBSERVED experiments in actual units (the
    design's ``actual`` levels plus the measured responses).  Rows with
    any missing factor/response value are dropped complete-case -- the
    dropped count is reported, never silent.  The fitted bundle rides
    the returned dict as ``bundle`` for the model-store / predict /
    contour / optimise lanes.
    """
    _validate_factors(factors)
    if not (1 <= len(response_columns) <= MAX_RESPONSES):
        raise ValueError(
            f"the RSM fit needs 1..{MAX_RESPONSES} response columns "
            f"({len(response_columns)} given)"
        )
    if not (0.5 < confidence < 1.0):
        raise ValueError(
            f"the confidence level must lie in (0.5, 1.0) ({confidence})"
        )
    dupes = [c for c in response_columns if response_columns.count(c) > 1]
    if dupes:
        raise ValueError(
            f"response columns must be unique ({sorted(set(dupes))})"
        )
    missing = [c for c in response_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"response columns missing from the frame: {missing}")
    overlap = [f.name for f in factors if f.name in response_columns]
    if overlap:
        raise ValueError(f"columns cannot be both factor and response: {overlap}")

    work = frame[[f.name for f in factors] + list(response_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    complete = work.dropna()
    dropped = int(len(work) - len(complete))
    if len(complete) < 3:
        raise ValueError(
            f"only {len(complete)} complete rows survive the missing-value "
            "drop -- an RSM fit needs at least 3"
        )

    coded = np.column_stack(
        [
            (complete[f.name].to_numpy(dtype=float) - f.center) / f.half_range
            for f in factors
        ]
    )
    terms = build_terms(factors)
    X = model_matrix(coded, terms)
    n, p = X.shape
    if n < p:
        raise ValueError(
            f"the design has fewer runs ({n}) than model terms ({p}) -- "
            "the second-order model is not estimable; use a bigger design"
        )
    rank = int(matrix_rank(X))
    aliasing = _alias_disclosure(X, terms)

    started = time.time()
    beta_all: dict[str, np.ndarray] = {}
    stats_all: dict[str, dict[str, Any]] = {}
    coef_tables: dict[str, list[dict[str, Any]]] = {}
    anova_all: dict[str, dict[str, Any]] = {}
    diag_tables: dict[str, list[dict[str, Any]]] = {}
    hierarchy_notes: list[str] = []

    t_crit = float(sps.t.ppf(1.0 - (1.0 - confidence) / 2.0, max(n - rank, 1)))
    xtx_inv = np.linalg.pinv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)

    for response in response_columns:
        y = complete[response].to_numpy(dtype=float)
        beta = np.linalg.pinv(X) @ y
        beta_all[response] = beta
        fitted = X @ beta
        resid = y - fitted
        sse = float(resid @ resid)
        sst = float(((y - y.mean()) ** 2).sum())
        df_res = n - rank
        s2 = sse / df_res if df_res > 0 else float("nan")
        rmse = float(np.sqrt(s2)) if df_res > 0 else float("nan")
        r2 = (1.0 - sse / sst) if sst > 0 else float("nan")
        adj_r2 = (
            (1.0 - (sse / df_res) / (sst / (n - 1)))
            if df_res > 0 and n > 1 and sst > 0
            else float("nan")
        )
        press = float(np.sum((resid / np.maximum(1.0 - h, 1e-12)) ** 2))
        pred_r2 = (1.0 - press / sst) if sst > 0 else float("nan")

        anova_all[response] = _anova_partition(
            X, y, beta, terms, xtx_inv, s2 if s2 > 0 else 0.0
        )

        # the coefficient table
        se_beta = (
            np.sqrt(np.maximum(s2 * np.diag(xtx_inv), 0.0))
            if s2 > 0
            else np.zeros(p)
        )
        table: list[dict[str, Any]] = []
        for j, term in enumerate(terms):
            t_stat = float(beta[j] / se_beta[j]) if se_beta[j] > 0 else None
            p_val = (
                float(2.0 * sps.t.sf(abs(t_stat), df_res))
                if t_stat is not None and df_res > 0
                else None
            )
            table.append(
                {
                    "term": term.key,
                    "label": term.label,
                    "kind": term.kind,
                    "estimate": float(beta[j]),
                    "std_error": float(se_beta[j]) if se_beta[j] > 0 else None,
                    "t": t_stat,
                    "p": p_val,
                    "ci_low": float(beta[j] - t_crit * se_beta[j])
                    if se_beta[j] > 0
                    else None,
                    "ci_high": float(beta[j] + t_crit * se_beta[j])
                    if se_beta[j] > 0
                    else None,
                }
            )
        coef_tables[response] = table

        # hierarchy: significant interaction/quadratic with insignificant
        # parent linear term -- flagged, never auto-removed
        lin_p = {
            term.key: row["p"]
            for term, row in zip(terms, table)
            if term.kind == "linear"
        }
        for term, row in zip(terms, table):
            if (
                term.kind == "interaction"
                and row["p"] is not None
                and row["p"] < 0.05
            ):
                i, j = int(term.key[1]), int(term.key[3])
                weak = [
                    f"x{idx}"
                    for idx in (i, j)
                    if (lin_p.get(f"x{idx}", 1.0) or 1.0) >= 0.05
                ]
                if weak:
                    hierarchy_notes.append(
                        f"response {response!r}: the interaction "
                        f"{term.key} ({term.label}) is significant "
                        f"(p={row['p']:.4g}) while its parent linear "
                        f"term(s) {weak} are not -- model hierarchy says "
                        "keep the parents anyway (flagged here, NEVER "
                        "auto-removed)"
                    )
            if (
                term.kind == "quadratic"
                and row["p"] is not None
                and row["p"] < 0.05
            ):
                i = int(term.key[1])
                if (lin_p.get(f"x{i}", 1.0) or 1.0) >= 0.05:
                    hierarchy_notes.append(
                        f"response {response!r}: the quadratic "
                        f"{term.key} ({term.label}) is significant "
                        f"(p={row['p']:.4g}) while its parent linear "
                        f"term x{i} is not -- keep the parent (flagged, "
                        "NEVER auto-removed)"
                    )

        # the diagnostics + fitted-vs-observed table.  A fit at machine
        # precision (rmse at the float noise floor, the noise-free demo
        # class) reports its scaled diagnostics as exact zeros -- the
        # studentized residuals and Cook's distances are undefined over
        # a zero residual variance, and reporting the raw ratios would
        # flag floating-point dust as outliers; the guard states it.
        machine_precision = (
            rmse is not None
            and not np.isnan(rmse)
            and rmse <= 1e-9 * (1.0 + abs(float(np.mean(y))))
        )
        if machine_precision:
            studentized = np.zeros(n)
            cooks = np.zeros(n)
            diagnostics_note = (
                "the fit is at machine precision (rmse at the float "
                "noise floor): the studentized residuals and Cook's "
                "distances are reported as exactly 0.0 -- undefined "
                "over zero residual variance, never the floating-point "
                "dust flagged as outliers"
            )
        else:
            studentized = (
                resid / np.sqrt(np.maximum(s2 * (1.0 - h), 1e-12))
                if s2 > 0
                else np.zeros(n)
            )
            cooks = (
                (resid**2 * h)
                / (max(rank, 1) * s2 * np.maximum((1.0 - h) ** 2, 1e-12))
                if s2 > 0
                else np.zeros(n)
            )
            diagnostics_note = None
        cooks_threshold = max(1.0, 4.0 / n)
        diag_rows: list[dict[str, Any]] = []
        outliers = 0
        influential = 0
        for i in range(n):
            out_flag = abs(studentized[i]) > 3.0
            inf_flag = float(cooks[i]) > cooks_threshold
            outliers += int(out_flag)
            influential += int(inf_flag)
            diag_rows.append(
                {
                    "run": i + 1,
                    "observed": float(y[i]),
                    "fitted": float(fitted[i]),
                    "residual": float(resid[i]),
                    "studentized": float(studentized[i]),
                    "leverage": float(h[i]),
                    "cooks_distance": float(cooks[i]),
                    "outlier_flag": bool(out_flag),
                    "influential_flag": bool(inf_flag),
                }
            )
        diag_tables[response] = diag_rows
        stats_all[response] = {
            "n_runs": n,
            "df_residual": df_res,
            "rank": rank,
            "r2": r2,
            "adj_r2": adj_r2,
            "pred_r2": pred_r2,
            "rmse": rmse,
            "s": rmse,
            "press": press,
            "confidence": confidence,
            "outliers_flagged": outliers,
            "influential_flagged": influential,
            "cooks_threshold": cooks_threshold,
            "thresholds_note": (
                "outlier flag: |internal studentised residual| > 3; "
                "influential flag: Cook's distance > max(1, 4/n) = "
                f"{cooks_threshold:.3f} -- both STATED thresholds, both "
                "reported as flags, never silent removals"
            ),
            "machine_precision_fit": bool(machine_precision),
            "diagnostics_note": diagnostics_note,
        }

    fit_seconds = time.time() - started
    bundle = ResponseSurfaceBundle(
        factors=factors,
        terms=terms,
        response_names=list(response_columns),
        betas=beta_all,
        design_meta=dict(design_meta or {}),
        confidence=confidence,
        provenance=dict(provenance_extra or {}),
        xtx_inv=xtx_inv,
        X_coded=coded,
        y_observed={
            r: complete[r].to_numpy(dtype=float) for r in response_columns
        },
        df_residual=n - rank,
        rank=rank,
    )

    return {
        "bundle": bundle,
        "factors": [f.view() for f in factors],
        "terms": [
            {"key": t.key, "kind": t.kind, "label": t.label} for t in terms
        ],
        "n_runs": n,
        "dropped_rows": dropped,
        "coefficients": coef_tables,
        "anova": anova_all,
        "diagnostics": diag_tables,
        "fit_statistics": stats_all,
        "hierarchy_notes": hierarchy_notes,
        "aliasing": aliasing,
        "confidence": confidence,
        "fit_seconds": fit_seconds,
    }


# ---------------------------------------------------------------------------
# the persistence carrier
# ---------------------------------------------------------------------------


@dataclass
class ResponseSurfaceBundle:
    """The §7 persistence carrier: everything predict / contour /
    optimise / export need, in one picklable object (the model store
    and the joblib artifact lane carry it unchanged)."""

    factors: list[FactorSpec]
    terms: list[Term]
    response_names: list[str]
    betas: dict[str, np.ndarray]
    design_meta: dict[str, Any]
    confidence: float = 0.95
    provenance: dict[str, Any] = field(default_factory=dict)
    xtx_inv: np.ndarray | None = None
    X_coded: np.ndarray | None = None
    y_observed: dict[str, np.ndarray] = field(default_factory=dict)
    df_residual: int = 0
    rank: int = 0

    @property
    def contract(self) -> dict[str, Any]:
        """The JSON-safe contract (the §7 'persist the complete training
        contract' rule): factors, the term registry order, per-response
        coefficient vectors, the fit metadata, the environment."""
        return {
            "kind": "rsm",
            "model_version": 1,
            "factors": [f.view() for f in self.factors],
            "terms": [
                {"key": t.key, "kind": t.kind, "label": t.label}
                for t in self.terms
            ],
            "responses": list(self.response_names),
            "coefficients": {
                r: [float(b) for b in self.betas[r]] for r in self.response_names
            },
            "design_meta": dict(self.design_meta),
            "confidence": self.confidence,
            "n_factors": len(self.factors),
            "df_residual": self.df_residual,
            "rank": self.rank,
            "environment": environment_metadata(),
            "provenance": dict(self.provenance),
        }

    def response_index(self, name: str) -> int:
        try:
            return self.response_names.index(name)
        except ValueError:
            raise ValueError(
                f"unknown response {name!r} (the model carries "
                f"{self.response_names})"
            ) from None


def environment_metadata() -> dict[str, str]:
    """The pinned environment block that rides every bundle."""
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------------------
# §6.5 prediction
# ---------------------------------------------------------------------------


def _zone_vocabulary(coded: np.ndarray, alpha: float | None) -> list[str]:
    """The honest per-point zone: inside the ranges / the CCD axial
    region / outside the design region."""
    zones: list[str] = []
    reach = alpha if (alpha is not None and alpha > 1.0) else 1.0
    for row in coded:
        m = float(np.max(np.abs(row)))
        if m <= 1.0 + 1e-9:
            zones.append("inside_ranges")
        elif m <= reach + 1e-9:
            zones.append("axial_region")
        else:
            zones.append("outside_design")
    return zones


def predict_surface(
    bundle: ResponseSurfaceBundle,
    points: pd.DataFrame,
    *,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Point predictions in ACTUAL units: per response the fitted value,
    the mean confidence interval and the new-observation prediction
    interval, plus the per-point zone vocabulary (extrapolation
    flagged, never silently extrapolated)."""
    conf = bundle.confidence if confidence is None else confidence
    if not (0.5 < conf < 1.0):
        raise ValueError(f"the confidence level must lie in (0.5, 1.0) ({conf})")
    missing = [f.name for f in bundle.factors if f.name not in points.columns]
    if missing:
        raise ValueError(f"point rows missing factor columns: {missing}")
    values = points[[f.name for f in bundle.factors]].apply(
        pd.to_numeric, errors="coerce"
    )
    if values.isna().to_numpy().any():
        raise ValueError("point rows carry non-numeric factor values")
    actual = values.to_numpy(dtype=float)
    coded = np.column_stack(
        [
            (actual[:, i] - f.center) / f.half_range
            for i, f in enumerate(bundle.factors)
        ]
    )
    X = model_matrix(coded, bundle.terms)
    df_res = max(bundle.df_residual, 1)
    t_crit = float(sps.t.ppf(1.0 - (1.0 - conf) / 2.0, df_res))
    alpha = bundle.design_meta.get("alpha")
    zones = _zone_vocabulary(coded, alpha if isinstance(alpha, float) else None)

    inv = bundle.xtx_inv
    if inv is None:  # pragma: no cover - bundle integrity guard
        raise ValueError("the bundle carries no fit (xtx_inv missing)")
    rows: list[dict[str, Any]] = []
    for i in range(coded.shape[0]):
        x0 = X[i]
        var_mean = float(x0 @ (inv @ x0))
        row: dict[str, Any] = {
            "run": i + 1,
            "zone": zones[i],
            "coded": {
                f.name: float(coded[i, j])
                for j, f in enumerate(bundle.factors)
            },
            "actual": {
                f.name: float(actual[i, j])
                for j, f in enumerate(bundle.factors)
            },
        }
        for r in bundle.response_names:
            yhat = float(x0 @ bundle.betas[r])
            s2 = _response_s2(bundle, r)
            half_ci = t_crit * float(np.sqrt(max(s2 * var_mean, 0.0)))
            half_pi = t_crit * float(
                np.sqrt(max(s2 * (1.0 + var_mean), 0.0))
            )
            row[r] = {
                "fitted": yhat,
                "ci_low": yhat - half_ci,
                "ci_high": yhat + half_ci,
                "pi_low": yhat - half_pi,
                "pi_high": yhat + half_pi,
            }
        rows.append(row)

    extrapolated = [i for i, z in enumerate(zones) if z != "inside_ranges"]
    return {
        "rows": rows,
        "responses": list(bundle.response_names),
        "confidence": conf,
        "df_residual": bundle.df_residual,
        "extrapolation": {
            "count": len(extrapolated),
            "runs": [i + 1 for i in extrapolated],
            "vocabulary": {
                "inside_ranges": "inside the declared factor ranges",
                "axial_region": (
                    "outside the ranges but inside the CCD's axial region "
                    f"(|coded| <= {alpha if isinstance(alpha, float) else 'alpha'}) "
                    "-- extrapolation beyond the corners, covered by the "
                    "axial runs themselves"
                ),
                "outside_design": (
                    "outside the design region entirely -- the number is "
                    "a model extrapolation, flagged, never presented as "
                    "an interpolation"
                ),
            },
        },
    }


def _response_s2(bundle: ResponseSurfaceBundle, response: str) -> float:
    """The residual variance per response, recomputed from the stored
    observed rows (the bundle keeps y_observed + X_coded for exactly
    this)."""
    X_coded = bundle.X_coded
    y = bundle.y_observed.get(response)
    if X_coded is None or y is None:  # pragma: no cover - integrity guard
        raise ValueError(
            f"the bundle carries no observed rows for {response!r}"
        )
    X = model_matrix(X_coded, bundle.terms)
    resid = y - X @ bundle.betas[response]
    sse = float(resid @ resid)
    df = max(bundle.df_residual, 1)
    return sse / df


# ---------------------------------------------------------------------------
# §6.5 the contour / surface grids
# ---------------------------------------------------------------------------


def contour_grid(
    bundle: ResponseSurfaceBundle,
    *,
    factor_x: str,
    factor_y: str,
    fixed_actual: dict[str, float] | None = None,
    grid_points: int = 25,
    mode: Literal["response", "desirability"] = "response",
    response: str | None = None,
    goals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The two-factor grid payload: x/y in coded and actual units, the
    z values (one response's prediction, or the combined desirability
    under the goals), every other factor held at explicitly displayed
    fixed values (default: the design center)."""
    if not (MIN_GRID_POINTS <= grid_points <= MAX_GRID_POINTS):
        raise ValueError(
            f"grid_points must lie in {MIN_GRID_POINTS}..{MAX_GRID_POINTS} "
            f"({grid_points})"
        )
    names = [f.name for f in bundle.factors]
    for needed in (factor_x, factor_y):
        if needed not in names:
            raise ValueError(
                f"unknown factor {needed!r} (the model carries {names})"
            )
    if factor_x == factor_y:
        raise ValueError("the contour pair needs two DIFFERENT factors")
    if mode == "response":
        if response is None:
            raise ValueError("the response mode needs the response name")
        bundle.response_index(response)
    else:
        if not goals:
            raise ValueError("the desirability mode needs the goals list")
    fixed_default: dict[str, float] = {}
    for f in bundle.factors:
        if f.name in (factor_x, factor_y):
            continue
        supplied = (fixed_actual or {}).get(f.name)
        if supplied is None:
            fixed_default[f.name] = f.center
        else:
            fixed_default[f.name] = float(supplied)
    unknown_fixed = [k for k in (fixed_actual or {}) if k not in names]
    if unknown_fixed:
        raise ValueError(f"fixed values for unknown factors: {unknown_fixed}")

    xi = names.index(factor_x)
    yi = names.index(factor_y)
    grid = np.linspace(-1.0, 1.0, grid_points)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    base = np.zeros((grid_points * grid_points, len(bundle.factors)))
    base[:, xi] = xx.ravel()
    base[:, yi] = yy.ravel()
    for name, value in fixed_default.items():
        j = names.index(name)
        base[:, j] = bundle.factors[j].code(value)

    X = model_matrix(base, bundle.terms)
    if mode == "response":
        if response is None:  # pragma: no cover - the guard above
            raise ValueError("the response mode needs the response name")
        z = (X @ bundle.betas[response]).reshape(grid_points, grid_points)
        z_payload = [[float(v) for v in row] for row in z]
        z_name = response
        z_note = (
            f"the fitted second-order prediction of {response!r} over the "
            "pair's coded box [-1, +1] (the declared ranges), every other "
            "factor at the displayed fixed values -- a MODEL surface, "
            "not measured data"
        )
    else:
        parsed, _ = _validate_goals(bundle, goals or [])
        d_grid = _combined_grid(bundle, parsed, base, grid_points)
        z_payload = d_grid
        z_name = "combined_desirability"
        z_note = (
            "the combined desirability D over the pair's coded box "
            "[-1, +1] under the request's goals -- the desirability "
            "OVERLAY (§6.5's multi-response contour overlay)"
        )

    fx = bundle.factors[xi]
    fy = bundle.factors[yi]
    return {
        "factor_x": {
            "name": fx.name,
            "unit": fx.unit,
            "coded": [float(v) for v in grid],
            "actual": [float(fx.decode(v)) for v in grid],
        },
        "factor_y": {
            "name": fy.name,
            "unit": fy.unit,
            "coded": [float(v) for v in grid],
            "actual": [float(fy.decode(v)) for v in grid],
        },
        "fixed": {
            name: {
                "actual": float(value),
                "coded": float(bundle.factors[names.index(name)].code(value)),
            }
            for name, value in fixed_default.items()
        },
        "grid_points": grid_points,
        "z_name": z_name,
        "z": z_payload,
        "note": z_note,
        "units_note": (
            "the z values ride the response's own units; the axes carry "
            "the factors' declared units -- the grid rows index factor_x, "
            "the columns index factor_y"
        ),
    }


def _combined_grid(
    bundle: ResponseSurfaceBundle,
    parsed: dict[str, dict[str, Any]],
    base: np.ndarray,
    grid_points: int,
) -> list[list[float]]:
    """The combined desirability over a grid of coded points."""
    X = model_matrix(base, bundle.terms)
    values = {r: X @ bundle.betas[r] for r in parsed}
    total_w = sum(spec["weight"] for spec in parsed.values())
    combined = np.ones(X.shape[0])
    for r, spec in parsed.items():
        d = np.array(
            [_individual_desirability(float(v), spec) for v in values[r]]
        )
        combined *= np.power(np.maximum(d, 1e-300), spec["weight"] / total_w)
    d_grid = combined.reshape(grid_points, grid_points)
    return [[float(v) for v in row] for row in d_grid]


# ---------------------------------------------------------------------------
# §6.6 the desirability machinery
# ---------------------------------------------------------------------------


def _validate_goals(
    bundle: ResponseSurfaceBundle,
    goals: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The goals contract: every response of the model exactly once,
    each with a valid goal + bounds; the auto-bounds (observed min/max)
    applied where the request omits them -- the bounds source stated
    per goal, never silent."""
    if not goals:
        raise ValueError("the optimiser needs at least one goal")
    seen: list[str] = []
    parsed: dict[str, dict[str, Any]] = {}
    for g in goals:
        response = g.get("response")
        if response not in bundle.response_names:
            raise ValueError(
                f"goal for unknown response {response!r} "
                f"(the model carries {bundle.response_names})"
            )
        if response in seen:
            raise ValueError(f"response {response!r} carries more than one goal")
        seen.append(response)
        goal = g.get("goal")
        if goal not in GOALS:
            raise ValueError(
                f"unknown goal {goal!r} for {response!r} "
                f"(choose from {list(GOALS)})"
            )
        weight = float(g.get("weight", 1.0))
        if not (0.0 < weight <= 10.0):
            raise ValueError(
                f"the weight for {response!r} must lie in (0, 10] ({weight})"
            )
        exponent = float(g.get("exponent", 1.0))
        if not (0.1 <= exponent <= 10.0):
            raise ValueError(
                f"the exponent for {response!r} must lie in [0.1, 10] "
                f"({exponent})"
            )
        obs = bundle.y_observed.get(response)
        obs_min = float(np.min(obs)) if obs is not None and len(obs) else None
        obs_max = float(np.max(obs)) if obs is not None and len(obs) else None
        spec: dict[str, Any] = {
            "goal": goal,
            "weight": weight,
            "exponent": exponent,
        }
        provided_low = g.get("low")
        provided_high = g.get("high")
        if goal in ("maximize", "minimize"):
            low = provided_low if provided_low is not None else obs_min
            high = provided_high if provided_high is not None else obs_max
            if low is None or high is None:
                raise ValueError(
                    f"the {goal} goal for {response!r} needs low and high "
                    "bounds (the frame carried no observations to fall "
                    "back on)"
                )
            if not low < high:
                raise ValueError(
                    f"the {goal} goal for {response!r} needs low < high "
                    f"({low} >= {high})"
                )
            spec["low"] = float(low)
            spec["high"] = float(high)
            spec["bounds_source"] = (
                "request"
                if provided_low is not None and provided_high is not None
                else "observed"
            )
        elif goal == "target":
            low = provided_low if provided_low is not None else obs_min
            high = provided_high if provided_high is not None else obs_max
            target = g.get("target")
            if target is None:
                raise ValueError(
                    f"the target goal for {response!r} needs the target value"
                )
            if low is None or high is None or not low < target < high:
                raise ValueError(
                    f"the target goal for {response!r} needs "
                    f"low < target < high ({low}, {target}, {high})"
                )
            spec["low"] = float(low)
            spec["target"] = float(target)
            spec["high"] = float(high)
            spec["bounds_source"] = (
                "request" if provided_low is not None else "observed"
            )
        else:  # range
            low = provided_low
            high = g.get("high")
            if low is None or high is None or not low < high:
                raise ValueError(
                    f"the range goal for {response!r} needs both bounds "
                    f"with low < high ({low}, {high})"
                )
            spec["low"] = float(low)
            spec["high"] = float(high)
            spec["bounds_source"] = "request"
        parsed[response] = spec
    missing = [r for r in bundle.response_names if r not in seen]
    if missing:
        raise ValueError(f"every model response needs a goal (missing: {missing})")
    return parsed, {"responses": list(bundle.response_names)}


def _individual_desirability(value: float, spec: dict[str, Any]) -> float:
    """One Derringer-Suich individual desirability, stated per goal."""
    goal = spec["goal"]
    exponent = spec["exponent"]
    if goal == "range":
        return 1.0 if spec["low"] <= value <= spec["high"] else 0.0
    low, high = spec["low"], spec["high"]
    if goal == "maximize":
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return ((value - low) / (high - low)) ** exponent
    if goal == "minimize":
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        return ((high - value) / (high - low)) ** exponent
    # target
    target = spec["target"]
    if value <= low or value >= high:
        return 0.0
    if value == target:
        return 1.0
    if value < target:
        return ((value - low) / (target - low)) ** exponent
    return ((high - value) / (high - target)) ** exponent


def _d_value(
    bundle: ResponseSurfaceBundle,
    coded: np.ndarray,
    parsed: dict[str, dict[str, Any]],
) -> float:
    """The combined desirability D at one coded point: the weighted
    geometric mean of the individual desirabilities (weights
    normalised by their sum -- the Derringer-Suich combination)."""
    X = model_matrix(np.atleast_2d(np.asarray(coded, dtype=float)), bundle.terms)
    total_w = sum(spec["weight"] for spec in parsed.values())
    combined = 1.0
    for response, spec in parsed.items():
        value = float((X @ bundle.betas[response])[0])
        d = _individual_desirability(value, spec)
        combined *= max(d, 1e-300) ** (spec["weight"] / total_w)
    return combined


def _parse_constraints(
    n_factors: int, constraints: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Linear factor constraints over CODED factors: {"coefficients":
    [a1..ak], "sense": "<="|">=", "bound": b} -- a_i * x <= / >= b."""
    if not constraints:
        return []
    if len(constraints) > MAX_CONSTRAINTS:
        raise ValueError(
            f"at most {MAX_CONSTRAINTS} linear constraints ({len(constraints)} given)"
        )
    parsed: list[dict[str, Any]] = []
    for c in constraints:
        coeffs = c.get("coefficients")
        if not isinstance(coeffs, list) or len(coeffs) != n_factors:
            raise ValueError(
                f"each constraint needs one coefficient per factor ({n_factors}): {coeffs!r}"
            )
        numeric = []
        for a in coeffs:
            if not isinstance(a, (int, float)) or not np.isfinite(a):
                raise ValueError(f"constraint coefficients must be finite ({a!r})")
            numeric.append(float(a))
        sense = c.get("sense", "<=")
        if sense not in ("<=", ">="):
            raise ValueError(f"constraint sense must be '<=' or '>=' ({sense!r})")
        bound = c.get("bound")
        if not isinstance(bound, (int, float)) or not np.isfinite(bound):
            raise ValueError(f"constraint bound must be finite ({bound!r})")
        parsed.append(
            {"coefficients": numeric, "sense": sense, "bound": float(bound)}
        )
    return parsed


def _constraint_ok(x: np.ndarray, constraints: list[dict[str, Any]]) -> bool:
    for c in constraints:
        value = float(np.dot(c["coefficients"], x))
        if c["sense"] == "<=" and value > c["bound"] + 1e-9:
            return False
        if c["sense"] == ">=" and value < c["bound"] - 1e-9:
            return False
    return True


def _optimiser_starts(bundle: ResponseSurfaceBundle) -> list[np.ndarray]:
    """The deterministic start set: the design's own coded rows
    (deduped), the center, and the 2^k corners capped at 64."""
    if bundle.X_coded is None:  # pragma: no cover - integrity guard
        raise ValueError("the bundle carries no design rows")
    k = len(bundle.factors)
    starts: list[np.ndarray] = [np.zeros(k)]
    seen_starts: set[tuple[float, ...]] = {tuple(np.zeros(k))}
    for row in bundle.X_coded:
        key = tuple(np.round(row, 9))
        if key not in seen_starts:
            seen_starts.add(key)
            starts.append(np.array(row))
    corner_cap = min(2**k, 64)
    for row in _factorial_rows(k)[:corner_cap]:
        key = tuple(np.round(row, 9))
        if key not in seen_starts:
            seen_starts.add(key)
            starts.append(row.copy())
    return starts


def _polish(
    objective,
    starts: list[np.ndarray],
    bounds: list[tuple[float, float]],
    method: str,
    scipy_constraints: list[dict[str, Any]],
    constraint_list: list[dict[str, Any]],
) -> np.ndarray | None:
    """Run the deterministic multi-start sweep, return the best feasible
    coded point (or None when nothing feasible survives)."""
    best_x: np.ndarray | None = None
    best_value = -np.inf
    for start in starts:
        try:
            res = minimize(
                objective,
                start,
                method=method,
                bounds=bounds,
                constraints=scipy_constraints if method == "SLSQP" else [],
                options={"maxiter": 200, "ftol": 1e-12},
            )
        except Exception:  # noqa: BLE001, S112 — a failed start is skipped, the sweep's own robustness
            continue
        candidate = np.clip(np.asarray(res.x, dtype=float), -1.0, 1.0)
        if constraint_list and not _constraint_ok(candidate, constraint_list):
            continue
        value = -float(objective(candidate))
        if value > best_value:
            best_value = value
            best_x = candidate
    return best_x


def optimize_desirability(
    bundle: ResponseSurfaceBundle,
    goals: list[dict[str, Any]],
    *,
    constraints: list[dict[str, Any]] | None = None,
    pair: tuple[str, str] | None = None,
    grid_points: int = 25,
) -> dict[str, Any]:
    """The §6.6 multi-response optimisation.

    Deterministic multi-start local search confined to the coded box
    [-1, +1]^k (the declared factor ranges -- extrapolation cannot
    occur by construction; the prediction lane flags it separately):
    starts at the design's own points, the center, and the corners,
    each polished with a bounded gradient method (L-BFGS-B free of
    constraints; SLSQP honouring them).  The answer carries the
    candidate optimum with per-response predictions and individual
    desirabilities, the per-factor sensitivity sweep (a +-0.05 coded
    step, stated), the curated Pareto alternatives (per-response
    individual optima + weight-tilted optima -- a deterministic
    sample, honestly NOT the exhaustive front), and the feasible-
    region mask for the requested factor pair.
    """
    parsed, _info = _validate_goals(bundle, goals)
    names = [f.name for f in bundle.factors]
    k = len(bundle.factors)
    constraint_list = _parse_constraints(k, constraints)
    total_weight = sum(spec["weight"] for spec in parsed.values())

    def objective(x: np.ndarray, spec_map: dict[str, dict[str, Any]] = parsed) -> float:
        return -np.log(max(_d_value(bundle, x, spec_map), 1e-300))

    bounds = [(-1.0, 1.0)] * k
    scipy_constraints = []
    for c in constraint_list:
        if c["sense"] == "<=":
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, c=c: c["bound"]
                    - float(np.dot(c["coefficients"], x)),
                }
            )
        else:
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, c=c: float(np.dot(c["coefficients"], x))
                    - c["bound"],
                }
            )
    method = "SLSQP" if scipy_constraints else "L-BFGS-B"
    starts = _optimiser_starts(bundle)
    sweep_starts = starts if len(starts) <= 96 else starts[:96]

    best_x = _polish(objective, starts, bounds, method, scipy_constraints, constraint_list)
    if best_x is None:
        raise ValueError(
            "no feasible start survived the constraints -- loosen them "
            "or widen the factor ranges"
        )
    x_opt = best_x
    d_opt = _d_value(bundle, x_opt, parsed)

    def response_block(x: np.ndarray) -> dict[str, Any]:
        X = model_matrix(np.atleast_2d(x), bundle.terms)
        block: dict[str, Any] = {}
        for response, spec in parsed.items():
            value = float((X @ bundle.betas[response])[0])
            bounds_view = {
                key: spec.get(key)
                for key in ("low", "target", "high")
                if spec.get(key) is not None
            }
            block[response] = {
                "goal": spec["goal"],
                "prediction": value,
                "individual_desirability": _individual_desirability(
                    value, spec
                ),
                "weight": spec["weight"],
                "exponent": spec["exponent"],
                "bounds": bounds_view,
                "bounds_source": spec.get("bounds_source", "request"),
            }
        return block

    # the sensitivity sweep: +-0.05 coded per factor (clipped to the box)
    sensitivity: list[dict[str, Any]] = []
    for i in range(k):
        entry: dict[str, Any] = {
            "factor": names[i],
            "step_coded": 0.05,
            "low": {},
            "high": {},
        }
        for direction, tag in ((-0.05, "low"), (0.05, "high")):
            probe = x_opt.copy()
            probe[i] = float(np.clip(probe[i] + direction, -1.0, 1.0))
            d_probe = _d_value(bundle, probe, parsed)
            entry[tag] = {
                "coded": float(probe[i]),
                "combined_desirability": d_probe,
                "delta": d_probe - d_opt,
            }
        sensitivity.append(entry)

    # the Pareto alternatives: per-response individual optima + the
    # weight-tilted optima (one response's weight x4), both over the
    # capped sweep start list (the deterministic sample)
    alternatives: list[dict[str, Any]] = []
    for response, spec in parsed.items():
        solo_spec = {response: dict(spec, weight=1.0)}

        def solo_objective(
            x: np.ndarray, spec_map: dict[str, dict[str, Any]] = solo_spec
        ) -> float:
            return -np.log(max(_d_value(bundle, x, spec_map), 1e-300))

        best_solo = _polish(
            solo_objective, sweep_starts, bounds, method, scipy_constraints, constraint_list
        )
        if best_solo is not None:
            alternatives.append(
                {
                    "kind": "individual",
                    "response": response,
                    "goal": spec["goal"],
                    "combined_desirability": _d_value(
                        bundle, best_solo, solo_spec
                    ),
                    "note": (
                        f"the single-response optimum for {response!r} "
                        "alone (the Pareto corner -- how far the combined "
                        "optimum trades this response away)"
                    ),
                    "x_coded": {names[j]: float(best_solo[j]) for j in range(k)},
                    "x_actual": {
                        names[j]: float(bundle.factors[j].decode(best_solo[j]))
                        for j in range(k)
                    },
                    "predictions": {
                        r: float(
                            (
                                model_matrix(np.atleast_2d(best_solo), bundle.terms)
                                @ bundle.betas[r]
                            )[0]
                        )
                        for r in bundle.response_names
                    },
                }
            )

    for response in parsed:
        tilted_spec = {
            r: dict(spec, weight=(spec["weight"] * 4.0 if r == response else spec["weight"]))
            for r, spec in parsed.items()
        }

        def tilted_objective(
            x: np.ndarray, spec_map: dict[str, dict[str, Any]] = tilted_spec
        ) -> float:
            return -np.log(max(_d_value(bundle, x, spec_map), 1e-300))

        best_tilted = _polish(
            tilted_objective, sweep_starts, bounds, method, scipy_constraints, constraint_list
        )
        if best_tilted is not None:
            alternatives.append(
                {
                    "kind": "weight_tilted",
                    "response": response,
                    "goal": parsed[response]["goal"],
                    "combined_desirability": _d_value(
                        bundle, best_tilted, tilted_spec
                    ),
                    "note": (
                        f"the combined optimum with {response!r}'s weight "
                        "x4 -- how the compromise shifts when this "
                        "response dominates"
                    ),
                    "x_coded": {
                        names[j]: float(best_tilted[j]) for j in range(k)
                    },
                    "x_actual": {
                        names[j]: float(bundle.factors[j].decode(best_tilted[j]))
                        for j in range(k)
                    },
                    "predictions": {
                        r: float(
                            (
                                model_matrix(np.atleast_2d(best_tilted), bundle.terms)
                                @ bundle.betas[r]
                            )[0]
                        )
                        for r in bundle.response_names
                    },
                }
            )

    feasible_grid: dict[str, Any] | None = None
    if pair is not None:
        fx, fy = pair
        if fx not in names or fy not in names or fx == fy:
            raise ValueError(
                f"the feasible-grid pair needs two different known factors ({pair})"
            )
        xi = names.index(fx)
        yi = names.index(fy)
        if not (MIN_GRID_POINTS <= grid_points <= MAX_GRID_POINTS):
            raise ValueError(
                f"grid_points must lie in {MIN_GRID_POINTS}..{MAX_GRID_POINTS} "
                f"({grid_points})"
            )
        grid = np.linspace(-1.0, 1.0, grid_points)
        feasible = np.zeros((grid_points, grid_points), dtype=bool)
        for a, xv in enumerate(grid):
            for b, yv in enumerate(grid):
                point = x_opt.copy()
                point[xi] = xv
                point[yi] = yv
                feasible[a, b] = (
                    _constraint_ok(point, constraint_list) if constraint_list else True
                )
        feasible_grid = {
            "factor_x": fx,
            "factor_y": fy,
            "grid_points": grid_points,
            "other_factors_at": {
                names[j]: float(x_opt[j])
                for j in range(k)
                if j not in (xi, yi)
            },
            "feasible": [
                [bool(feasible[a, b]) for b in range(grid_points)]
                for a in range(grid_points)
            ],
            "note": (
                "the feasible-region mask over the pair's coded box with "
                "every other factor held at the candidate optimum's coded "
                "values (TRUE = feasible under the linear constraints)"
            ),
        }

    return {
        "candidate": {
            "x_coded": {names[j]: float(x_opt[j]) for j in range(k)},
            "x_actual": {
                names[j]: float(bundle.factors[j].decode(x_opt[j]))
                for j in range(k)
            },
            "combined_desirability": d_opt,
            "responses": response_block(x_opt),
        },
        "weights": {r: parsed[r]["weight"] for r in parsed},
        "total_weight": total_weight,
        "constraints": [
            {
                "coefficients": c["coefficients"],
                "sense": c["sense"],
                "bound": c["bound"],
            }
            for c in constraint_list
        ],
        "sensitivity": sensitivity,
        "sensitivity_note": (
            "each factor stepped +-0.05 in CODED units (clipped to the "
            "box) around the candidate -- the combined desirability's "
            "delta reads as the optimisation's soft direction per factor"
        ),
        "pareto_alternatives": alternatives,
        "pareto_note": (
            "the curated deterministic alternatives (per-response "
            "individual optima + weight-tilted combined optima) -- a "
            "SAMPLE of the trade-off surface, honestly NOT the "
            "exhaustive Pareto front; when the individual optima sit "
            "far apart the responses CONFLICT and the single combined "
            "optimum is a compromise, not a victory"
        ),
        "feasible_grid": feasible_grid,
        "region_note": (
            "the search is confined to the coded box [-1, +1]^k -- the "
            "declared factor ranges -- so the optimum CANNOT extrapolate "
            "outside the experiment by construction (the prediction "
            "lane flags extrapolated points separately with its zone "
            "vocabulary)"
        ),
        "bounds_source_note": (
            "desirability bounds the request did not set fall back to "
            "the OBSERVED design min/max per response -- stated per goal "
            "in bounds_source, never silent"
        ),
    }


# ---------------------------------------------------------------------------
# the synthetic demo
# ---------------------------------------------------------------------------

DEMO_FACTORS = [
    FactorSpec("temperature_c", 50.0, 90.0, unit="degC"),
    FactorSpec("ph", 5.0, 9.0, unit="pH"),
    FactorSpec("substrate_mM", 10.0, 50.0, unit="mM"),
]
DEMO_RESPONSES = ["yield_pct", "purity_pct"]

# the KNOWN synthetic surfaces (over coded x1, x2, x3), aligned to the
# term registry order (intercept, x1, x2, x3, x1x2, x1x3, x2x3, x1^2,
# x2^2, x3^2):
#   yield_pct  = 78 + 4x1 + 6x2 - 2x3 + 2.5x1x2 - 1x1x3 + 0.5x2x3
#                 - 4x1^2 - 3.5x2^2 - 1x3^2
#   purity_pct = 92 - 3x1 + 1.5x2 - 1x3 + 0.8x1x2 + 0.4x1x3 - 0.2x2x3
#                 - 2x1^2 - 1.5x2^2 - 0.5x3^2
_DEMO_COEFS = {
    "yield_pct": [
        78.0, 4.0, 6.0, -2.0, 2.5, -1.0, 0.5, -4.0, -3.5, -1.0,
    ],
    "purity_pct": [
        92.0, -3.0, 1.5, -1.0, 0.8, 0.4, -0.2, -2.0, -1.5, -0.5,
    ],
}


def demo_rsm_dataset() -> pd.DataFrame:
    """The seeded SYNTHETIC observed frame: a 17-run rotatable CCD over
    the three demo factors (2^3 factorial + 2*3 axial at alpha =
    1.682 + 3 centers, un-randomised so the frame is fully
    deterministic) with BOTH responses evaluated on the KNOWN surfaces,
    NO noise -- the exact-coefficient-recovery and exact-optimum
    verifications ride this.  Labelled synthetic wherever it travels."""
    factors = DEMO_FACTORS
    design = build_design(
        factors,
        "ccd",
        alpha_type="rotatable",
        center_points=3,
        replicates=1,
        randomize=False,
        seed=42,
    )
    terms = build_terms(factors)
    rows: list[dict[str, float]] = []
    for run in design["runs"]:
        coded = np.array(
            [run["coded"][f.name] for f in factors], dtype=float
        )
        X = model_matrix(np.atleast_2d(coded), terms)
        row: dict[str, float] = {
            f.name: float(f.decode(v)) for f, v in zip(factors, coded)
        }
        for response in DEMO_RESPONSES:
            row[response] = float((X @ np.array(_DEMO_COEFS[response]))[0])
        rows.append(row)
    return pd.DataFrame(rows)


def rsm_registry_payload() -> dict[str, Any]:
    """The vocabulary answer for the route's registry endpoint."""
    demo = demo_rsm_dataset()
    return {
        "designs": [
            {
                "key": "ccd",
                "label": "Central Composite Design",
                "alpha_types": list(ALPHA_TYPES),
                "min_factors": 2,
                "max_factors": MAX_FACTORS,
                "runs_formula": "2^k + 2k + centers (x replicates)",
                "blocking": True,
                "note": (
                    "the classic CCD: the 2^k factorial corners, 2k "
                    "axial runs at +/-alpha (rotatable alpha = (2k)^(1/4) "
                    "by default -- OUTSIDE the declared ranges, "
                    "disclosed), and replicated center points; the "
                    "two-block split (factorial + axial blocks) is "
                    "supported"
                ),
            },
            {
                "key": "box_behnken",
                "label": "Box-Behnken Design",
                "alpha_types": None,
                "min_factors": 3,
                "max_factors": MAX_FACTORS,
                "runs_formula": "4 * k(k-1)/2 + centers (x replicates)",
                "blocking": False,
                "note": (
                    "every factor PAIR at its 2^2 corners with all other "
                    "factors at the center -- all levels inside [-1, +1] "
                    "(the design never leaves the declared ranges); "
                    "blocking honestly NOT supported (a typed refusal)"
                ),
            },
            {
                "key": "full_factorial",
                "label": "Full 2^k Factorial",
                "alpha_types": None,
                "min_factors": 2,
                "max_factors": 8,
                "runs_formula": "2^k + centers (x replicates)",
                "blocking": False,
                "note": (
                    "all 2^k corners plus centers, capped at 8 factors "
                    "(2^9+ corners would exceed the 512-run bound); a "
                    "first-order screen -- the second-order model needs "
                    "the center/axial structure of the CCD for its "
                    "curvature terms"
                ),
            },
        ],
        "goals": list(GOALS),
        "model": {
            "order": "second-order polynomial in coded factors",
            "terms": "1 intercept + k linear + k(k-1)/2 interaction + k quadratic",
            "term_count_formula": "1 + 2k + k(k-1)/2",
        },
        "bounds": {
            "min_factors": MIN_FACTORS,
            "max_factors": MAX_FACTORS,
            "max_design_runs": MAX_DESIGN_RUNS,
            "max_responses": MAX_RESPONSES,
            "grid_points": [MIN_GRID_POINTS, MAX_GRID_POINTS],
            "confidence": "(0.5, 1.0)",
            "max_constraints": MAX_CONSTRAINTS,
            "seed": [0, _MAX_SEED],
        },
        "demo_dataset": {
            "name": "synthetic-rsm-demo",
            "synthetic": True,
            "factors": [f.view() for f in DEMO_FACTORS],
            "responses": list(DEMO_RESPONSES),
            "runs": len(demo),
            "design": "ccd rotatable (2^3 + 2*3 axial + 3 centers, un-randomised)",
            "noise": False,
            "note": (
                "the seeded SYNTHETIC demonstration experiment: the "
                "responses are KNOWN quadratic surfaces evaluated "
                "noise-free over the design -- every number it produces "
                "is a model estimate over generated data, never an "
                "experimental fact; the known coefficients make the "
                "exact-recovery verification possible"
            ),
        },
        "scope_statement": (
            "RSM is a designed-experiment modelling discipline: the "
            "design generators (CCD / Box-Behnken / full factorial), the "
            "second-order fit with its ANOVA and diagnostics, the "
            "contour / 3-D surface data, and the Derringer-Suich "
            "desirability optimisation -- NOT the generic multi-response "
            "ML (that lane predicts; this lane explains a designed "
            "experiment), NOT multi-objective optimisation, NOT "
            "drug-response matrix factorisation, and NOT an ordinary 3-D "
            "visualisation.  Blocking is the two-block CCD split only; "
            "lack-of-fit needs genuinely replicated design points "
            "(no replicates -> honest nulls); the optimiser is confined "
            "to the declared factor ranges so it cannot extrapolate by "
            "construction, and the prediction lane flags extrapolated "
            "points with its zone vocabulary."
        ),
    }
