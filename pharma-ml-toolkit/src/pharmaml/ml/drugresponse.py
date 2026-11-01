r"""Drug-response learning absorbed from three reviewed repositories.

``CSOgroup/CaDRReS`` was reviewed for this module at the model-file
level (the repository's own NumPy model class, its informative-weight
sibling and the scripts that build the input feature tables -- reviewed
for the data contract: cell line x gene feature matrices and drug
response tables with NaN where a pair was never screened).  The
published method (Suphavilai et al., 2018, bioRxiv) is a
*bilinear factorisation* with a sigmoid link, and every piece of it is
engine-free:

* the **gamma layer** -- a weight matrix ``W`` mapping a cell line's
  genomic features to ``K`` latent factors (linear: the gamma layer is
  the factorisation, which is exactly what makes the per-drug readout
  ``W w_d`` exact),
* **per-drug weight vectors** ``w_d`` over the latent layer with a
  sigmoid-linked predicted response score,
* full-batch gradient descent minimising squared error over the
  *observed* (cell line, drug) pairs only -- NaN entries simply carry
  no gradient, and
* the **informative-weight** idea (``cadrres-iw``): a per-feature
  prior variance that shrinks non-informative gene weights harder.
  :class:`InformativeRegression` distils it into a ridge regression
  with per-feature prior variances, so it can regularise any linear
  response problem, not just the gamma layer.

``Precily``'s deep drug-response portfolio was reviewed at the concept
level -- the contribution there is architectural (multi-omics deep
networks over expression + mutation + copy-number blocks) rather than
algorithmic, and the transferable piece is the *fusion discipline*:
z-score every block with statistics fitted and stored at train time,
then standardise block-wise so one wide block cannot dominate the
concatenated vector by width alone.  That lands as
:class:`MultiOmicsFusion`, the honest numpy-only stand-in for the
concatenate-and-scale front end of a multi-omics network.

``DeepResponse`` (variational-autoencoder drug response) was likewise
reviewed at the concept level: encoder → latent ``mu``/``logvar`` → a
reparameterised latent sample → decoder with an optional response
head, trained on the ELBO loss ``reconstruction + KL(q||p) +
response MSE``.  :class:`ResponseVAE` is that architecture with the
backward pass hand-derived and verified against finite differences
(:meth:`ResponseVAE.gradient_check`).

Around the models sit the pharmacology the screens actually measure:
the four-parameter log-logistic Hill curve
(:func:`dose_response_curve`, :func:`fit_hill_curve`) and the standard
log-dose sensitivity metric (:func:`auc_log_dose`), plus the
evaluation layer every response paper reports (:func:`response_metrics`
-- Pearson/Spearman/RMSE/MAE, reusing :mod:`pharmaml.ml.stats`' own
rank machinery -- and :func:`cross_validate_response`, grouped by cell
line so no cell line straddles a fold).

The honest scope, stated once: no GPU training, no pretrained
checkpoints, and no vendored preprocessing of the original data
releases (CaDRReS' essential-gene feature maps, the GDSC snapshots and
the R side of the enrichment world stay where they came from); the
sigmoid link and squared-error objective follow the published
description while every default hyperparameter here is chosen for
teaching-scale convergence, not tuned on any screen; and the Hill
fitting reuses the toolkit's own Levenberg-Marquardt engine
(:mod:`pharmaml.ml.fitting`) rather than growing a second optimiser.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.ml.drugresponse import (
...     DrugResponseModel, InformativeRegression, MultiOmicsFusion,
...     ResponseVAE, auc_log_dose, cross_validate_response,
...     dose_response_curve, fit_hill_curve, response_metrics,
... )

The four-parameter log-logistic Hill curve, its fit and the AUC:

>>> round(float(dose_response_curve(1.0, hill_coef=2.0, ic50=2.0)), 3)
0.8
>>> doses = np.geomspace(0.01, 100.0, 12)
>>> viability = dose_response_curve(doses, hill_coef=2.5, ic50=1.0,
...                                 top=0.9, bottom=0.1)
>>> fit = fit_hill_curve(doses, viability)
>>> round(fit.parameters["ic50"], 2)
1.0
>>> round(fit.parameters["hill_coef"], 2)
2.5
>>> round(fit.r_squared, 6)
1.0
>>> round(auc_log_dose([0.1, 1.0, 10.0], [1.0, 0.5, 0.0]), 3)
1.0

A CaDRReS-style screen: cell-line features in, (cell line x drug)
response table out, NaN where untested:

>>> rng = np.random.default_rng(42)
>>> features = pd.DataFrame(
...     rng.normal(size=(24, 6)),
...     index=[f"cell{i}" for i in range(24)],
...     columns=["G1", "G2", "G3", "G4", "G5", "G6"],
... )
>>> table = pd.DataFrame(
...     {"drugA": 1.0 / (1.0 + np.exp(-(3.0 * features["G1"] - 2.0 * features["G2"]))),
...      "drugB": 1.0 / (1.0 + np.exp(-(-2.0 * features["G3"])))},
...     index=features.index,
... )
>>> table.iloc[0, 0] = np.nan                  # one untested pair
>>> model = DrugResponseModel(n_factors=4, lr=0.5, epochs=600, seed=42)
>>> history = model.fit(features, table)
>>> history.pearson_r > 0.95                   # the bilinear net recovers the screen
True
>>> predictions = model.predict(["cell0", "cell1", "cell2"])
>>> predictions.shape
(3, 2)
>>> ranking = model.explain_drug("drugA")
>>> ranking["feature"].tolist()[:2]            # G1 / G2 drive drugA
['G1', 'G2']
>>> metrics = response_metrics(
...     table.loc[:, "drugA"], model.predict(features.index)["drugA"],
... )
>>> round(metrics["pearson"], 3) > 0.95
True
>>> report = cross_validate_response(
...     lambda: DrugResponseModel(n_factors=4, lr=0.5, epochs=600, seed=42),
...     features, table, n_folds=4, seed=42,
... )
>>> len(report.folds)
4
>>> report.summary["rmse"] < 0.1
True

The informative-weight ridge: informative features keep their weights,
non-informative ones are shrunk to silence:

>>> X = rng.normal(size=(60, 3))
>>> y = 3.0 * X[:, 0] + 0.5
>>> informative = InformativeRegression(lr=0.01, epochs=800, l2=0.1)
>>> losses = informative.fit(X, y, prior_variance=[100.0, 0.01, 0.01])
>>> round(float(informative.coef[0]), 1)
3.0
>>> bool(np.abs(informative.coef[1]) < 0.05)   # tight prior -> near-zero weight
True

Multi-omics fusion: per-block z-scores, stored and reused:

>>> expression = pd.DataFrame(
...     {"G1": [1.0, 3.0, 5.0, 7.0], "G2": [4.0, 3.0, 2.0, 1.0]},
...     index=["c1", "c2", "c3", "c4"],
... )
>>> mutation = pd.DataFrame(
...     {"TP53": [0, 1, 0, 1], "KRAS": [1, 1, 0, 0]},
...     index=["c1", "c2", "c3", "c4"],
... )
>>> fusion = MultiOmicsFusion()
>>> fused = fusion.fit({"expression": expression, "mutation": mutation}).transform(
...     {"expression": expression, "mutation": mutation},
... )
>>> fused.shape
(4, 4)
>>> fused.columns.tolist()
[('expression', 'G1'), ('expression', 'G2'), ('mutation', 'TP53'), ('mutation', 'KRAS')]
>>> round(float(fused[("expression", "G1")].mean()), 6)   # z-scored: centred
0.0
>>> fusion.block_report().set_index("block")["n_features"].to_dict()
{'expression': 2, 'mutation': 2}

A response VAE with the optional response head:

>>> Xv = rng.normal(size=(60, 8))
>>> yv = 2.0 * Xv[:, 0] - 1.0 * Xv[:, 3]
>>> vae = ResponseVAE(latent_dim=3, hidden=16, lr=0.02, epochs=250, seed=11)
>>> vhistory = vae.fit(Xv, y=yv, seed=3)
>>> vhistory.elbo[-1] > vhistory.elbo[0]       # the bound improves
True
>>> vae.gradient_check(Xv[:6], yv[:6]) < 1e-7  # backprop matches finite differences
True
>>> vae.encode(Xv).shape
(60, 3)
>>> round(float(np.corrcoef(vae.predict(Xv), yv)[0, 1]), 2) > 0.8
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.ml.fitting import levenberg_marquardt
from pharmaml.ml.stats import rank_correlations

__all__ = [
    "DrugResponseHistory",
    "DrugResponseModel",
    "HillFit",
    "InformativeRegression",
    "MultiOmicsFusion",
    "ResponseCVFold",
    "ResponseCVReport",
    "ResponseVAE",
    "VAEHistory",
    "auc_log_dose",
    "cross_validate_response",
    "dose_response_curve",
    "fit_hill_curve",
    "response_metrics",
]

_LOG = get_logger(__name__)

#: Random seed threaded through every weight init and split (the pin).
RANDOM_STATE = 42

#: Default Hill-coefficient bounds (the steepness sanity envelope).
_HILL_BOUNDS = (0.2, 8.0)


# ---------------------------------------------------------------------------
# Small NumPy helpers (the house shapes, checked once)
# ---------------------------------------------------------------------------


def _as_matrix(values: object, name: str) -> np.ndarray:
    """Coerce to a finite 2-D float matrix (1-D input becomes a column)."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise DataValidationError(f"{name} must be 2-D, got {matrix.ndim}-D")
    if matrix.shape[0] < 1:
        raise DataValidationError(f"{name} must have at least one row")
    if not np.all(np.isfinite(matrix)):
        raise DataValidationError(f"{name} must be finite (no NaN/inf)")
    return matrix


def _as_vector(values: object, name: str) -> np.ndarray:
    """Coerce to a finite flat float vector."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 1:
        raise DataValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} must be finite (no NaN/inf)")
    return array


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# The Hill curve and the log-dose AUC
# ---------------------------------------------------------------------------


def dose_response_curve(
    dose: float | Sequence[float] | np.ndarray,
    hill_coef: float,
    ic50: float,
    top: float = 1.0,
    bottom: float = 0.0,
) -> float | np.ndarray:
    """Four-parameter log-logistic Hill curve of a drug on one sample.

    ``y(d) = bottom + (top - bottom) / (1 + (d / IC50)^h)`` -- the
    standard screening parameterisation: ``top`` is the response at
    vanishing dose, ``bottom`` the response at saturation, ``ic50`` the
    inflection dose (response ``(top + bottom) / 2``) and ``hill_coef``
    the steepness (``h = 1`` hyperbolic, ``h > 1`` threshold-like).

    Parameters
    ----------
    dose:
        Dose value(s).  Zero is allowed (``y = top``); negative doses
        are not (the power term is undefined).
    hill_coef:
        The Hill coefficient, strictly positive.
    ic50:
        The mid-point dose, strictly positive.
    top:
        Response as dose → 0 (default 1.0, full viability).
    bottom:
        Response as dose → ∞ (default 0.0, fully inhibited).

    Returns
    -------
    float or numpy.ndarray
        The predicted response, scalar for scalar input.

    Raises
    ------
    DataValidationError
        On negative doses or non-positive ``hill_coef``/``ic50``.

    Examples
    --------
    >>> round(float(dose_response_curve(2.0, hill_coef=1.0, ic50=2.0)), 3)
    0.5
    >>> dose_response_curve([0.5, 2.0, 8.0], hill_coef=1.0, ic50=2.0).round(3).tolist()
    [0.8, 0.5, 0.2]
    >>> dose_response_curve(1.0, hill_coef=-1.0, ic50=2.0)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: hill_coef must be > 0, got -1.0
    """
    if hill_coef <= 0:
        raise DataValidationError(f"hill_coef must be > 0, got {hill_coef}")
    if ic50 <= 0:
        raise DataValidationError(f"ic50 must be > 0, got {ic50}")
    dose_array = np.asarray(dose, dtype=float)
    if np.any(dose_array < 0):
        raise DataValidationError("dose must be >= 0 (the power term is undefined below)")
    powered = (dose_array / ic50) ** hill_coef
    result = bottom + (top - bottom) / (1.0 + powered)
    if np.ndim(dose) == 0:
        return float(result)
    return result


def auc_log_dose(dose: Sequence[float] | np.ndarray, response: Sequence[float] | np.ndarray) -> float:
    """Area under the response curve over the ``log10`` dose axis.

    The standard drug-screening sensitivity metric: the trapezoid of
    ``response`` against ``log10(dose)`` after sorting by dose.  A
    flatter curve (insensitive sample) keeps a large area; a curve that
    dives early (sensitive sample) loses it.  Doses must be strictly
    positive (the log axis needs them) and responses finite.

    Parameters
    ----------
    dose:
        Dose values (any order; sorted internally).  Duplicated doses
        contribute zero-width segments, never duplicated area.
    response:
        The measured response at each dose, same length as ``dose``.

    Returns
    -------
    float
        The trapezoidal area on the ``log10`` dose grid.

    Raises
    ------
    DataValidationError
        On length mismatch, non-positive doses, or non-finite response.

    Examples
    --------
    >>> round(auc_log_dose([0.1, 1.0, 10.0], [1.0, 0.5, 0.0]), 3)
    1.0
    >>> round(auc_log_dose([10.0, 0.1, 1.0], [0.0, 1.0, 1.0]), 3)   # order-proof
    1.5
    """
    dose_array = _as_vector(dose, "dose")
    response_array = _as_vector(response, "response")
    if dose_array.size != response_array.size:
        raise DataValidationError(
            f"dose and response must have equal length, got "
            f"{(dose_array.size, response_array.size)!r}"
        )
    if np.any(dose_array <= 0):
        raise DataValidationError("dose must be strictly positive (the log10 axis)")
    order = np.argsort(dose_array, kind="stable")
    log_dose = np.log10(dose_array[order])
    sorted_response = response_array[order]
    # the trapezoid spelled out (numpy renamed trapz -> trapezoid in 2.0;
    # the version pin spans both, so the two-point rule is written by hand)
    widths = np.diff(log_dose)
    area = float(np.sum(0.5 * (sorted_response[1:] + sorted_response[:-1]) * widths))
    return area


@dataclass(frozen=True)
class HillFit:
    """One fitted Hill curve and its quality ledger.

    ``parameters`` carries ``top``, ``bottom``, ``ic50`` and
    ``hill_coef``; ``fitted_auc`` is the curve's log-dose area on the
    observed dose grid (the model-based counterpart of
    :func:`auc_log_dose`, immune to missing grid points).
    """

    parameters: dict[str, float]
    r_squared: float
    converged: bool
    n_points: int
    fitted_auc: float

    @property
    def ic50(self) -> float:
        """The mid-point dose of the fitted curve."""
        return self.parameters["ic50"]

    def predict(self, dose: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Evaluate the fitted curve at ``dose``."""
        return dose_response_curve(
            dose,
            hill_coef=self.parameters["hill_coef"],
            ic50=self.parameters["ic50"],
            top=self.parameters["top"],
            bottom=self.parameters["bottom"],
        )


def fit_hill_curve(
    dose: Sequence[float] | np.ndarray,
    response: Sequence[float] | np.ndarray,
    *,
    hill_bounds: tuple[float, float] = _HILL_BOUNDS,
    max_iterations: int = 400,
) -> HillFit:
    """Fit the four-parameter Hill curve to (dose, response) points.

    Least squares via the toolkit's own Levenberg-Marquardt engine
    (:func:`pharmaml.ml.fitting.levenberg_marquardt`, reused rather
    than re-derived): initial guesses read off the sorted data (``top``
    from the lowest dose, ``bottom`` from the highest, ``ic50`` by
    log-scale interpolation to the mid response, ``hill`` 1), with
    ``hill_bounds`` keeping degenerate steepness honest.

    Parameters
    ----------
    dose:
        Strictly positive dose values.
    response:
        The measured response at each dose (viability, % effect, ...),
        same length as ``dose``.
    hill_bounds:
        ``(low, high)`` bounds for the Hill coefficient.
    max_iterations:
        The LM iteration cap.

    Returns
    -------
    HillFit
        The fitted curve with ``r_squared``, convergence flag and the
        fitted curve's log-dose AUC.

    Raises
    ------
    DataValidationError
        On length mismatch, non-positive doses, non-finite data, or
        bad ``hill_bounds``.

    Examples
    --------
    >>> import numpy as np
    >>> doses = np.geomspace(0.01, 100.0, 12)
    >>> viability = dose_response_curve(doses, hill_coef=2.5, ic50=1.0,
    ...                                 top=0.9, bottom=0.1)
    >>> fit = fit_hill_curve(doses, viability)
    >>> round(fit.parameters["ic50"], 2), round(fit.parameters["hill_coef"], 2)
    (1.0, 2.5)
    >>> round(float(fit.predict(1.0)), 3)      # the mid-point response
    0.5
    """
    dose_array = _as_vector(dose, "dose")
    response_array = _as_vector(response, "response")
    if dose_array.size != response_array.size:
        raise DataValidationError(
            f"dose and response must have equal length, got "
            f"{(dose_array.size, response_array.size)!r}"
        )
    if np.any(dose_array <= 0):
        raise DataValidationError("dose must be strictly positive (the log-dose pivot)")
    if not (hill_bounds[0] < hill_bounds[1] and hill_bounds[0] > 0):
        raise DataValidationError(f"hill_bounds must satisfy 0 < low < high, got {hill_bounds!r}")

    order = np.argsort(dose_array, kind="stable")
    dose_sorted = dose_array[order]
    response_sorted = response_array[order]
    top_guess = float(response_sorted[0])
    bottom_guess = float(response_sorted[-1])
    log_dose = np.log10(dose_sorted)
    midpoint = 0.5 * (top_guess + bottom_guess)
    crossings = np.flatnonzero(np.diff(np.sign(response_sorted - midpoint)) != 0)
    if crossings.size:
        first = int(crossings[0])
        span = response_sorted[first + 1] - response_sorted[first]
        fraction = (midpoint - response_sorted[first]) / span if span != 0 else 0.0
        log_ic50 = float(log_dose[first] + fraction * (log_dose[first + 1] - log_dose[first]))
    else:
        log_ic50 = float(np.mean(log_dose))
    p0 = np.array([top_guess, bottom_guess, 10.0**log_ic50, 1.0])
    lower = np.array([-np.inf, -np.inf, 1e-12, hill_bounds[0]])
    upper = np.array([np.inf, np.inf, np.inf, hill_bounds[1]])

    def model(d: np.ndarray, top: float, bottom: float, ic50: float, hill: float) -> np.ndarray:
        return bottom + (top - bottom) / (1.0 + (d / ic50) ** hill)

    result = levenberg_marquardt(
        model,
        dose_array,
        response_array,
        p0,
        bounds=(lower, upper),
        max_iterations=max_iterations,
    )
    top, bottom, ic50, hill = (float(value) for value in result.parameters)
    fitted = model(dose_sorted, top, bottom, ic50, hill)
    ss_tot = float(np.sum((response_array - response_array.mean()) ** 2))
    r_squared = 1.0 - result.cost / ss_tot if ss_tot > 1e-24 else float("nan")
    report = HillFit(
        parameters={"top": top, "bottom": bottom, "ic50": ic50, "hill_coef": hill},
        r_squared=float(r_squared),
        converged=bool(result.converged),
        n_points=int(dose_array.size),
        fitted_auc=auc_log_dose(dose_sorted, fitted),
    )
    _LOG.debug(
        "hill fit: ic50=%.4g hill=%.2f R² %.4f (%s)",
        ic50,
        hill,
        r_squared,
        result.termination,
    )
    return report


# ---------------------------------------------------------------------------
# CaDRReS: the gamma layer, per-drug weights, observed-pair training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugResponseHistory:
    """The training curve of a :class:`DrugResponseModel` fit."""

    train_losses: tuple[float, ...]
    final_loss: float
    n_observed: int
    pearson_r: float
    n_epochs: int


class DrugResponseModel:
    """The CaDRReS drug-response model in plain NumPy, training included.

    Cell-line features are mapped by a weight matrix ``W`` (features
    plus a bias row → ``K`` latent *gamma* factors, linear -- the
    factorisation of the published model), each drug owns a weight
    vector over the latent layer (columns of ``w``, factors plus a
    bias row), and the predicted response score is the sigmoid of the
    bilinear form.  Training minimises squared error over the
    *observed* (cell line, drug) pairs only -- NaN table entries carry
    no gradient -- by full-batch gradient descent with L2 shrinkage on
    the non-bias rows.  Deterministic: seeded initialisation, no
    shuffling (full-batch GD is exactly reproducible, the toolkit's
    pin).

    Attributes
    ----------
    W : numpy.ndarray
        ``(n_features + 1, n_factors)`` -- the gamma layer; the last
        row is the gamma bias.
    w : numpy.ndarray
        ``(n_factors + 1, n_drugs)`` -- per-drug weight vectors over
        the latent layer; the last row holds the per-drug biases.
    feature_names, drug_names : list of str
        The training column labels, in order.
    cell_index : dict
        Cell-line name → training row, used by name-based lookups.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> features = pd.DataFrame(rng.normal(size=(24, 4)),
    ...                         index=[f"c{i}" for i in range(24)],
    ...                         columns=["G1", "G2", "G3", "G4"])
    >>> table = pd.DataFrame(
    ...     {"d1": 1.0 / (1.0 + np.exp(-2.0 * features["G1"])),
    ...      "d2": 1.0 / (1.0 + np.exp(3.0 * features["G3"]))},
    ...     index=features.index,
    ... )
    >>> model = DrugResponseModel(n_factors=3, lr=0.5, epochs=400, seed=42)
    >>> history = model.fit(features, table)
    >>> history.pearson_r > 0.9
    True
    >>> model.predict("c0").shape
    (1, 2)
    >>> model.latent_factors("c0").shape
    (3,)
    >>> model.explain_drug("d2")["feature"].tolist()[0]
    'G3'
    """

    def __init__(
        self,
        n_factors: int = 10,
        lr: float = 0.1,
        epochs: int = 200,
        l2: float = 1e-3,
        seed: int = RANDOM_STATE,
    ) -> None:
        if n_factors < 1:
            raise DataValidationError(f"n_factors must be >= 1, got {n_factors}")
        if lr <= 0:
            raise DataValidationError(f"lr must be > 0, got {lr}")
        if epochs < 1:
            raise DataValidationError(f"epochs must be >= 1, got {epochs}")
        if l2 < 0:
            raise DataValidationError(f"l2 must be >= 0, got {l2}")
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.W = np.zeros((0, 0))
        self.w = np.zeros((0, 0))
        self.feature_names: list[str] = []
        self.drug_names: list[str] = []
        self.cell_index: dict[str, int] = {}
        self._fitted = False

    # -- training --------------------------------------------------------- #

    def _loss_and_grads(
        self, design: np.ndarray, observed: np.ndarray, mask: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """One masked squared-error evaluation with its exact gradient.

        The backward pass is the chain rule of the bilinear model:
        ``dL/dlogit = (2/n_obs) · err · ŷ(1-ŷ)`` through the sigmoid,
        then ``dW = x̃ᵀ δγ``, ``dw = g̃ᵀ δlogit`` with the L2 term added
        on the non-bias rows only -- verified against finite
        differences by :meth:`gradient_check`.
        """
        n_obs = float(mask.sum())
        gamma = design @ self.W
        design_gamma = np.hstack([gamma, np.ones((gamma.shape[0], 1))])
        predicted = _sigmoid(design_gamma @ self.w)
        error = np.where(mask, predicted - observed, 0.0)
        reg = self.l2 * float(
            np.sum(self.W[: self.W.shape[0] - 1] ** 2)
            + np.sum(self.w[: self.w.shape[0] - 1] ** 2)
        )
        loss = float((error**2).sum() / n_obs) + reg
        d_logit = (2.0 / n_obs) * error * predicted * (1.0 - predicted)
        l2_rows_w = np.ones((self.w.shape[0], 1))
        l2_rows_w[-1, 0] = 0.0
        grad_w = design_gamma.T @ d_logit + 2.0 * self.l2 * l2_rows_w * self.w
        d_design_gamma = d_logit @ self.w.T
        l2_rows_W = np.ones((self.W.shape[0], 1))
        l2_rows_W[-1, 0] = 0.0
        grad_W = design.T @ d_design_gamma[:, : self.n_factors] + 2.0 * self.l2 * l2_rows_W * self.W
        return loss, {"W": grad_W, "w": grad_w}

    def fit(
        self,
        X_cell_lines: pd.DataFrame | np.ndarray,
        response_table: pd.DataFrame,
        progress: Callable[[int, float], None] | None = None,
    ) -> DrugResponseHistory:
        """Train on the observed pairs of a (cell line x drug) screen.

        Parameters
        ----------
        X_cell_lines:
            Cell-line features (rows = cell lines, columns = genomic
            features).  A DataFrame's index names the cell lines (the
            response table's rows must carry the same labels); an
            ndarray aligns positionally and gets generic names.
        response_table:
            Rows = cell lines, columns = drugs, values = normalised
            response the sigmoid can chase (NaN where untested -- those
            pairs carry no gradient).  Values outside (0, 1) warn once:
            the sigmoid output can never reach them.
        progress:
            Optional ``progress(epoch, loss)`` callback, invoked once
            per epoch.

        Returns
        -------
        DrugResponseHistory
            The per-epoch losses, the final loss, the observed-pair
            count and the in-sample Pearson r of the fitted scores.

        Raises
        ------
        DataValidationError
            On misaligned rows/columns, duplicate cell-line labels,
            non-finite features, an all-NaN screen, or unknown
            hyperparameters.
        """
        matrix = self._prepare_features(X_cell_lines)
        n_cells, n_features = matrix.shape
        if not isinstance(response_table, pd.DataFrame):
            raise DataValidationError(
                "response_table must be a pandas DataFrame (rows = cell lines, "
                "columns = drugs)"
            )
        table = response_table.apply(pd.to_numeric, errors="coerce")
        if isinstance(X_cell_lines, pd.DataFrame):
            duplicated = X_cell_lines.index[X_cell_lines.index.duplicated()]
            if not duplicated.empty:
                labels = sorted({str(label) for label in duplicated})
                raise DataValidationError(
                    f"X_cell_lines has duplicate cell-line label(s) {labels} -- the "
                    "feature/response row pairing would be ambiguous"
                )
            if table.index.duplicated().any():
                labels = sorted({str(label) for label in table.index[table.index.duplicated()]})
                raise DataValidationError(
                    f"response_table has duplicate cell-line label(s) {labels} -- "
                    "collapse the replicates first (biology.response.collapse_replicates)"
                )
            missing = [str(label) for label in X_cell_lines.index if label not in table.index]
            if missing:
                raise DataValidationError(
                    f"{len(missing)} training cell line(s) missing from the response "
                    f"table's rows, e.g. {missing[:3]}"
                )
            table = table.loc[list(X_cell_lines.index)]
        elif table.shape[0] != n_cells:
            raise DataValidationError(
                f"response_table has {table.shape[0]} rows for {n_cells} feature rows "
                "-- align the screen's cell lines with the feature matrix first"
            )
        observed = table.to_numpy(dtype=float)
        mask = np.isfinite(observed)
        if not mask.any():
            raise DataValidationError("the response table holds no observed value at all")
        out_of_range = int((np.abs(observed[mask] - 0.5) > 0.5).sum())
        if out_of_range:
            _LOG.warning(
                "%d observed response value(s) outside (0, 1) -- the sigmoid output "
                "can never reach them; normalise the table first",
                out_of_range,
            )
        silent_drugs = [str(drug) for drug in table.columns[~mask.any(axis=0)]]
        if silent_drugs:
            _LOG.warning(
                "%d drug column(s) have no observed value -- their weights stay at "
                "initialisation: %s",
                len(silent_drugs),
                ", ".join(silent_drugs[:5]),
            )
        self.drug_names = [str(drug) for drug in table.columns]

        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6.0 / (n_features + self.n_factors))
        self.W = rng.uniform(-limit, limit, size=(n_features + 1, self.n_factors))
        limit_d = np.sqrt(6.0 / (self.n_factors + len(self.drug_names)))
        self.w = rng.uniform(-limit_d, limit_d, size=(self.n_factors + 1, len(self.drug_names)))
        design = np.hstack([matrix, np.ones((n_cells, 1))])
        self._design_cache = design

        losses: list[float] = []
        for epoch in range(self.epochs):
            loss, grads = self._loss_and_grads(design, observed, mask)
            self.W -= self.lr * grads["W"]
            self.w -= self.lr * grads["w"]
            losses.append(loss)
            if progress is not None:
                progress(epoch, loss)
        self._fitted = True
        predicted = self._predict_matrix(design)
        pearson = float(np.corrcoef(predicted[mask], observed[mask])[0, 1])
        history = DrugResponseHistory(
            train_losses=tuple(losses),
            final_loss=losses[-1],
            n_observed=int(mask.sum()),
            pearson_r=pearson,
            n_epochs=int(self.epochs),
        )
        _LOG.info(
            "trained CaDRReS-style model: %d cell lines x %d drugs, %d observed "
            "pairs, final loss %.6g (pearson %.4f)",
            n_cells,
            len(self.drug_names),
            history.n_observed,
            history.final_loss,
            pearson,
        )
        return history

    def _prepare_features(self, X_cell_lines: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Validate the feature matrix and record its names/index."""
        if isinstance(X_cell_lines, pd.DataFrame):
            frame = X_cell_lines.apply(pd.to_numeric, errors="coerce")
            matrix = frame.to_numpy(dtype=float)
            if not np.all(np.isfinite(matrix)):
                raise DataValidationError(
                    "X_cell_lines holds non-finite values -- impute before fitting "
                    "(see pharmaml.biology.omics.impute_expression)"
                )
            self.feature_names = [str(column) for column in frame.columns]
            self.cell_index = {str(label): row for row, label in enumerate(frame.index)}
        else:
            matrix = _as_matrix(X_cell_lines, "X_cell_lines")
            self.feature_names = [f"feature_{i}" for i in range(matrix.shape[1])]
            self.cell_index = {f"cell_{i}": i for i in range(matrix.shape[0])}
        return matrix

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise DataValidationError(
                "the model has not been fitted yet -- call fit(X_cell_lines, "
                "response_table) first"
            )

    def _resolve_cells(
        self, cells: str | Sequence[str] | pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Turn names, a frame or a raw matrix into (design rows, labels)."""
        self._require_fitted()
        if isinstance(cells, str):
            if cells not in self.cell_index:
                raise DataValidationError(f"unknown cell line {cells!r}")
            rows = np.array([self.cell_index[cells]])
            training = self._training_matrix()
            return training[rows], [cells]
        if isinstance(cells, pd.DataFrame):
            frame = cells.apply(pd.to_numeric, errors="coerce")
            if [str(column) for column in frame.columns] != self.feature_names:
                raise DataValidationError(
                    "the frame's columns do not match the training features "
                    f"({list(frame.columns)[:3]}... vs {self.feature_names[:3]}...)"
                )
            matrix = frame.to_numpy(dtype=float)
            self._validate_new_matrix(matrix)
            return np.hstack([matrix, np.ones((matrix.shape[0], 1))]), [
                str(label) for label in frame.index
            ]
        if isinstance(cells, pd.Index):
            cells = np.asarray(cells)          # name lookup, same rules as arrays
        if isinstance(cells, (np.ndarray, list, tuple)):
            array = np.asarray(cells)
            if array.dtype.kind in ("U", "S", "O") and array.ndim <= 1:
                names = [str(name) for name in array.reshape(-1)]
                unknown = [name for name in names if name not in self.cell_index]
                if unknown:
                    raise DataValidationError(
                        f"unknown cell line(s) {unknown[:3]} -- predict accepts "
                        "training names or a feature matrix"
                    )
                training = self._training_matrix()
                rows = np.array([self.cell_index[name] for name in names])
                return training[rows], names
            matrix = _as_matrix(cells, "cells")
            self._validate_new_matrix(matrix)
            return np.hstack([matrix, np.ones((matrix.shape[0], 1))]), [
                f"cell_{i}" for i in range(matrix.shape[0])
            ]
        raise DataValidationError(
            "cells must be a cell-line name, a sequence of names, a feature frame "
            f"or a 2-D feature matrix, got {type(cells).__name__}"
        )

    def _training_matrix(self) -> np.ndarray:
        """The cached training design matrix (name lookups read from it)."""
        cache = getattr(self, "_design_cache", None)
        if cache is None:
            raise DataValidationError(
                "name-based prediction needs the training design matrix -- it was "
                "not cached; pass the feature frame/matrix instead"
            )
        return cache

    def _validate_new_matrix(self, matrix: np.ndarray) -> None:
        if matrix.shape[1] != len(self.feature_names):
            raise DataValidationError(
                f"the feature matrix has {matrix.shape[1]} columns, the model was "
                f"trained on {len(self.feature_names)}"
            )
        if not np.all(np.isfinite(matrix)):
            raise DataValidationError(
                "the feature matrix holds non-finite values -- impute before "
                "predicting (see pharmaml.biology.omics.impute_expression)"
            )

    def _predict_matrix(self, design: np.ndarray) -> np.ndarray:
        gamma = design @ self.W
        design_gamma = np.hstack([gamma, np.ones((gamma.shape[0], 1))])
        return _sigmoid(design_gamma @ self.w)

    # -- the CaDRReS readouts ---------------------------------------------- #

    def predict(
        self, cells: str | Sequence[str] | pd.DataFrame | np.ndarray
    ) -> pd.DataFrame:
        """Predicted response scores for cell lines (rows x drugs).

        ``cells`` may be training cell-line *names* (a single name or a
        sequence -- the training feature rows are looked up), a feature
        frame with the training columns (its index labels the output
        rows), or a raw 2-D feature matrix (generic ``cell_i`` labels).
        """
        design, labels = self._resolve_cells(cells)
        return pd.DataFrame(self._predict_matrix(design), index=labels, columns=self.drug_names)

    def latent_factors(
        self, cell: str | Sequence[str] | pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """The gamma layer's factors for one cell line (or several).

        A single name returns ``(n_factors,)``; names, frames and
        matrices return ``(n, n_factors)`` -- the same resolution rules
        as :meth:`predict`.
        """
        design, _labels = self._resolve_cells(cell)
        gamma = design @ self.W
        if isinstance(cell, str):
            return gamma.reshape(-1)
        return gamma

    def explain_drug(self, drug: str) -> pd.DataFrame:
        """Rank the input features behind one drug's response score.

        The drug's effective feature weights are the composed bilinear
        form ``W w_d`` (the gamma layer's feature weights contracted
        with the drug's factor weights -- the outer-product readout the
        linear gamma layer makes exact): the prediction is
        ``sigmoid(x · (W w_d) + const)``, so each weight *is* that
        feature's contribution to the score's logit.

        Returns
        -------
        pandas.DataFrame
            Columns ``feature`` and ``weight``, sorted by |weight|
            descending then feature name ascending (deterministic).

        Raises
        ------
        DataValidationError
            If the drug is unknown or the model is unfitted.
        """
        self._require_fitted()
        if drug not in self.drug_names:
            raise DataValidationError(
                f"unknown drug {drug!r}; the model was trained on {self.drug_names}"
            )
        column = self.drug_names.index(drug)
        weights = self.W[: len(self.feature_names)] @ self.w[: self.n_factors, column]
        report = pd.DataFrame(
            {"feature": self.feature_names, "weight": weights.astype(float)}
        )
        report = (
            report.assign(_magnitude=report["weight"].abs())
            .sort_values(["_magnitude", "feature"], ascending=[False, True])
            .drop(columns="_magnitude")
            .reset_index(drop=True)
        )
        return report

    def gradient_check(
        self,
        X_cell_lines: pd.DataFrame | np.ndarray,
        response_table: pd.DataFrame,
        epsilon: float = 1e-6,
    ) -> float:
        """Max |analytic − finite-difference| over every weight.

        The house precedent (ml/visiblenet's DAG, ml/sequences' LSTM)
        verifies hand-derived backprop to ~1e-8; this model's check is
        the same contract for the bilinear gamma layer.
        """
        matrix = self._prepare_features(X_cell_lines)
        table = response_table.apply(pd.to_numeric, errors="coerce")
        observed = table.to_numpy(dtype=float)
        mask = np.isfinite(observed)
        design = np.hstack([matrix, np.ones((matrix.shape[0], 1))])
        _loss, analytic = self._loss_and_grads(design, observed, mask)
        worst = 0.0
        for name in ("W", "w"):
            flat = getattr(self, name).reshape(-1)
            reference = analytic[name].reshape(-1)
            for index in range(flat.size):
                original = flat[index]
                flat[index] = original + epsilon
                plus, _grads = self._loss_and_grads(design, observed, mask)
                flat[index] = original - epsilon
                minus, _grads = self._loss_and_grads(design, observed, mask)
                flat[index] = original
                numeric = (plus - minus) / (2.0 * epsilon)
                worst = max(worst, abs(numeric - reference[index]))
        _LOG.debug("drug-response gradient_check: max deviation %.3e", worst)
        return float(worst)


# ---------------------------------------------------------------------------
# cadrres-iw distilled: ridge with per-feature informative priors
# ---------------------------------------------------------------------------


class InformativeRegression:
    """Ridge regression with per-feature prior variances (cadrres-iw).

    The informative-weight idea, distilled: the penalty on feature
    ``j``'s weight is ``l2 · β_j² / v_j`` where ``v_j`` is that
    feature's *prior variance* -- informative genes (large ``v_j``)
    keep their weights, non-informative ones (small ``v_j``) are
    shrunk toward silence.  With all-equal ``v`` this is plain ridge;
    the closed form ``(XᵀX + λD)β = Xᵀy`` exists, but training runs
    the same full-batch gradient descent the gamma layer trains with
    (zero initialisation, no random draws: bit-reproducible).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(5)
    >>> X = rng.normal(size=(80, 3))
    >>> y = 3.0 * X[:, 0] + 0.5
    >>> model = InformativeRegression(lr=0.01, epochs=800, l2=0.1)
    >>> losses = model.fit(X, y, prior_variance=[100.0, 0.01, 0.01])
    >>> round(float(model.coef[0]), 1)
    3.0
    >>> bool(np.abs(model.coef[1]) < 0.05)    # tight prior, silent weight
    True
    >>> round(float(model.predict(np.array([[1.0, 0.0, 0.0]]))[0]), 1)
    3.2
    """

    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 300,
        l2: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        if lr <= 0:
            raise DataValidationError(f"lr must be > 0, got {lr}")
        if epochs < 1:
            raise DataValidationError(f"epochs must be >= 1, got {epochs}")
        if l2 < 0:
            raise DataValidationError(f"l2 must be >= 0, got {l2}")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)
        self.coef = np.zeros(0)
        self.intercept = 0.0
        self.prior_variance: np.ndarray = np.ones(0)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]],
        y: np.ndarray | Sequence[float],
        prior_variance: np.ndarray | Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """Train by full-batch gradient descent on the informative loss.

        Parameters
        ----------
        X:
            Feature matrix (rows = samples, columns = features).  A
            DataFrame keeps its column order.
        y:
            The response values, one per row of ``X``.
        prior_variance:
            Per-feature prior variances (length = number of features,
            all strictly positive).  ``None`` uses all-ones -- plain
            ridge.  In the cadrres-iw spirit these come from whatever
            screen ranked the features (e.g. biomarker association
            strength), making the shrinkage *informative*.

        Returns
        -------
        tuple of float
            The per-epoch training losses (MSE + informative penalty).

        Raises
        ------
        DataValidationError
            On shape mismatch, non-finite data, or non-positive prior
            variances.
        """
        matrix = _as_matrix(X, "X")
        target = _as_vector(y, "y")
        if matrix.shape[0] != target.size:
            raise DataValidationError(
                f"X has {matrix.shape[0]} rows for {target.size} response values"
            )
        n_features = matrix.shape[1]
        if prior_variance is None:
            variances = np.ones(n_features)
        else:
            variances = np.asarray(prior_variance, dtype=float).reshape(-1)
            if variances.size != n_features:
                raise DataValidationError(
                    f"prior_variance must have one entry per feature "
                    f"({n_features}), got {variances.size}"
                )
            if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
                raise DataValidationError(
                    "prior_variance entries must be finite and > 0 (a zero prior "
                    "pins the weight at zero -- drop the feature instead)"
                )
        self.prior_variance = variances.copy()

        design = np.hstack([matrix, np.ones((matrix.shape[0], 1))]) if self.fit_intercept else matrix
        beta = np.zeros(design.shape[1])
        penalty_rows = np.ones(design.shape[1])
        penalty_rows[:n_features] = self.l2 / variances
        if not self.fit_intercept:
            penalty_rows[:] = self.l2 / variances
        n_rows = matrix.shape[0]
        losses: list[float] = []
        for _epoch in range(self.epochs):
            residual = design @ beta - target
            penalty = float(np.sum(penalty_rows * beta**2))
            losses.append(float(np.sum(residual**2) / n_rows) + penalty)
            gradient = (2.0 / n_rows) * (design.T @ residual) + 2.0 * penalty_rows * beta
            beta -= self.lr * gradient
        if self.fit_intercept:
            self.coef = beta[:n_features].copy()
            self.intercept = float(beta[n_features])
        else:
            self.coef = beta.copy()
            self.intercept = 0.0
        self._fitted = True
        _LOG.info(
            "informative regression fitted: %d features, %d rows, final loss %.6g "
            "(l2=%.3g, prior variance span %.3g..%.3g)",
            n_features,
            n_rows,
            losses[-1],
            self.l2,
            float(variances.min()),
            float(variances.max()),
        )
        return tuple(losses)

    def predict(self, X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]]) -> np.ndarray:
        """Predicted response for each row of ``X``.

        Raises
        ------
        DataValidationError
            If the model is unfitted or ``X``'s width disagrees with the
            training features.
        """
        if not self._fitted:
            raise DataValidationError("the model has not been fitted yet -- call fit(X, y) first")
        matrix = _as_matrix(X, "X")
        if matrix.shape[1] != self.coef.size:
            raise DataValidationError(
                f"X has {matrix.shape[1]} columns, the model was fitted on "
                f"{self.coef.size}"
            )
        return matrix @ self.coef + self.intercept


# ---------------------------------------------------------------------------
# Multi-omics fusion (the Precily transferable: z-score, scale, concat)
# ---------------------------------------------------------------------------


class MultiOmicsFusion:
    """Concatenate -omics blocks with per-block z-scores and block scaling.

    The fusion discipline every multi-omics network leans on: each
    block is z-scored column-wise with statistics fitted at
    :meth:`fit` time (mean/sd per column, stored; constant columns get
    ``z = 0`` like :func:`pharmaml.biology.omics.normalize_expression`),
    then -- with ``equal_block_variance=True`` -- the whole block is
    divided by ``sqrt(n_features)`` so every block carries the same
    total variance and a wide expression block cannot drown a narrow
    mutation block by width alone.  :meth:`transform` returns the
    concatenated matrix under two-level ``(block, feature)`` columns.

    Attributes
    ----------
    block_means_, block_stds_ : dict of pandas.Series
        Per-block column means / standard deviations (population, ddof=0),
        stored at fit time and reused verbatim at transform time.
    block_names_ : list of str
        The fit-time block keys, in the caller's order.

    Examples
    --------
    >>> import pandas as pd
    >>> expression = pd.DataFrame({"G1": [1.0, 3.0], "G2": [1.0, 3.0]},
    ...                           index=["c1", "c2"])
    >>> mutation = pd.DataFrame({"TP53": [0, 1]}, index=["c1", "c2"])
    >>> fusion = MultiOmicsFusion()
    >>> fused = fusion.fit({"expr": expression, "mut": mutation}).transform(
    ...     {"expr": expression, "mut": mutation},
    ... )
    >>> fused.shape
    (2, 3)
    >>> fused.loc["c1"].round(3).tolist()
    [-0.707, -0.707, -1.0]
    >>> fusion.block_report().set_index("block")["total_variance"].round(3).to_dict()
    {'expr': 1.0, 'mut': 1.0}
    """

    def __init__(self, equal_block_variance: bool = True) -> None:
        self.equal_block_variance = bool(equal_block_variance)
        self.block_means_: dict[str, pd.Series] = {}
        self.block_stds_: dict[str, pd.Series] = {}
        self.block_names_: list[str] = []
        self._fitted = False

    def fit(self, blocks: dict[str, pd.DataFrame]) -> MultiOmicsFusion:
        """Learn and store the per-block column statistics.

        Parameters
        ----------
        blocks:
            At least one named block (rows = samples, columns =
            features).  Every block must carry the *same* row labels;
            blocks are reordered to the first block's order when the
            labels agree as sets.

        Returns
        -------
        MultiOmicsFusion
            ``self``, for chaining into :meth:`transform`.

        Raises
        ------
        DataValidationError
            On an empty block set, a non-DataFrame block, mismatched
            row labels, or duplicated gene symbols within a block.
        """
        if not isinstance(blocks, dict) or not blocks:
            raise DataValidationError(
                "blocks must be a non-empty dict of {block name: DataFrame}"
            )
        reference_index: pd.Index | None = None
        reference_name = ""
        for name, block in blocks.items():
            if not isinstance(block, pd.DataFrame):
                raise DataValidationError(
                    f"block {name!r} must be a pandas DataFrame, got {type(block).__name__}"
                )
            if block.columns.duplicated().any():
                symbols = sorted({str(s) for s in block.columns[block.columns.duplicated()]})
                raise DataValidationError(
                    f"block {name!r} has duplicated gene symbol(s) {symbols}"
                )
            if reference_index is None:
                reference_index = block.index
                reference_name = str(name)
                continue
            if set(block.index) != set(reference_index):
                raise DataValidationError(
                    f"block {name!r} row labels do not match block {reference_name!r} "
                    "-- multi-omics fusion needs the same samples in every block"
                )
        self.block_names_ = [str(name) for name in blocks]
        self.block_means_ = {}
        self.block_stds_ = {}
        for name, block in blocks.items():
            aligned = block.loc[reference_index]  # type: ignore[index]
            values = aligned.apply(pd.to_numeric, errors="coerce")
            means = values.mean(axis=0)
            stds = values.std(axis=0, ddof=0)
            constant = [str(gene) for gene in stds.index[stds == 0]]
            if constant:
                shown = ", ".join(constant[:5])
                _LOG.warning(
                    "block %r: %d constant feature(s) -- z-score set to 0: %s",
                    str(name),
                    len(constant),
                    shown,
                )
            self.block_means_[str(name)] = means
            self.block_stds_[str(name)] = stds
        self._fitted = True
        _LOG.info(
            "multi-omics fusion fitted: %d block(s), %d column(s) total (%s)",
            len(self.block_names_),
            int(sum(len(self.block_means_[name]) for name in self.block_names_)),
            ", ".join(self.block_names_),
        )
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise DataValidationError(
                "the fusion has not been fitted yet -- call fit(blocks) first"
            )

    def transform(self, blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Standardise and concatenate blocks into one fused matrix.

        The fit-time means/sds are applied verbatim (new samples are
        fine; the stored statistics never refit).  Columns come out as
        two-level ``(block, feature)`` tuples.

        Raises
        ------
        DataValidationError
            On missing or unexpected block keys, or mismatched feature
            columns within a known block.
        """
        self._require_fitted()
        by_name = {str(name): block for name, block in blocks.items()}
        if set(by_name) != set(self.block_names_):
            missing = [name for name in self.block_names_ if name not in by_name]
            extra = [name for name in by_name if name not in set(self.block_names_)]
            raise DataValidationError(
                f"transform blocks must match the fitted blocks {self.block_names_} "
                f"-- missing {missing}, unexpected {extra}"
            )
        reference = by_name[self.block_names_[0]].index
        for name in self.block_names_[1:]:
            if not by_name[name].index.equals(reference):
                raise DataValidationError(
                    f"block {name!r} row labels differ from block "
                    f"{self.block_names_[0]!r} -- concatenation would silently "
                    "realign; pass blocks sharing one index"
                )
        pieces: list[pd.DataFrame] = []
        for name in self.block_names_:
            block = by_name[name]
            means = self.block_means_[name]
            stds = self.block_stds_[name]
            if [str(column) for column in block.columns] != [str(c) for c in means.index]:
                raise DataValidationError(
                    f"block {name!r} columns have changed since fit "
                    f"(got {list(block.columns)[:3]}... for {list(means.index)[:3]}...)"
                )
            values = block.apply(pd.to_numeric, errors="coerce")
            safe_stds = stds.where(stds != 0, 1.0)
            standardised = (values - means) / safe_stds
            standardised = standardised.where(values.notna(), np.nan)
            if self.equal_block_variance:
                standardised = standardised / np.sqrt(len(means))
            standardised.columns = pd.MultiIndex.from_tuples(
                [(name, str(column)) for column in standardised.columns],
                names=["block", "feature"],
            )
            pieces.append(standardised)
        fused = pd.concat(pieces, axis=1)
        _LOG.debug("transformed %d block(s) into a %r matrix", len(pieces), fused.shape)
        return fused

    def fit_transform(self, blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """``fit`` then ``transform`` in one call."""
        return self.fit(blocks).transform(blocks)

    def block_report(self) -> pd.DataFrame:
        """Per-block audit of the fitted fusion.

        One row per block: feature count, the stored mean/sd of the
        standardised values and the block's total column variance (the
        number ``equal_block_variance`` equalises to ~1).

        Raises
        ------
        DataValidationError
            If called before :meth:`fit`.
        """
        self._require_fitted()
        rows: list[dict[str, object]] = []
        for name in self.block_names_:
            stds = self.block_stds_[name]
            n_features = len(stds)
            if n_features == 0:
                rows.append(
                    {"block": name, "n_features": 0, "n_constant": 0, "total_variance": 0.0}
                )
                continue
            n_variable = int((stds > 0).sum())
            variance = n_variable / n_features if self.equal_block_variance else float(n_variable)
            rows.append(
                {
                    "block": name,
                    "n_features": n_features,
                    "n_constant": int((stds == 0).sum()),
                    "total_variance": variance,
                }
            )
        return pd.DataFrame(rows, columns=["block", "n_features", "n_constant", "total_variance"])


# ---------------------------------------------------------------------------
# DeepResponse distilled: the response VAE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VAEHistory:
    """The training ledger of a :class:`ResponseVAE` fit.

    ``elbo`` is the per-epoch *negative* loss (the bound itself: up
    means learning); the components carry the reconstruction MSE, the
    KL term and -- when a response vector was supplied -- the response
    MSE, all per sample.
    """

    elbo: tuple[float, ...]
    reconstruction: tuple[float, ...]
    kl_divergence: tuple[float, ...]
    response_loss: tuple[float, ...] | None
    n_epochs: int


def _xavier_pair(rng: np.random.Generator, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded Xavier-style init for one linear layer."""
    limit = float(np.sqrt(6.0 / (n_in + n_out))) if (n_in + n_out) > 0 else 0.5
    weight = rng.uniform(-limit, limit, size=(n_in, n_out))
    bias = np.zeros(n_out)
    return weight, bias


class ResponseVAE:
    """A variational autoencoder for omics profiles, response head optional.

    Encoder (tanh hidden) → latent ``mu`` / ``logvar`` → a
    reparameterised latent sample ``z = mu + eps·sigma`` → tanh-hidden
    decoder with a *linear* reconstruction head (real-valued, z-scored
    expression is the target), plus an optional linear response head on
    ``z``.  The loss is the ELBO in its negative form,
    ``reconstruction MSE + KL(q||p) + response_weight · response MSE``,
    minimised by full-batch gradient descent with the backward pass
    hand-derived through the reparameterisation trick and verified
    against finite differences (:meth:`gradient_check` -- the house
    precedent, ~1e-9 agreement).

    Attributes
    ----------
    elbo_history_ : tuple of float
        The per-epoch ELBO (negative loss) of the last ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(7)
    >>> X = rng.normal(size=(60, 8))
    >>> vae = ResponseVAE(latent_dim=3, hidden=16, lr=0.02, epochs=120, seed=11)
    >>> history = vae.fit(X, seed=3)
    >>> history.elbo[-1] > history.elbo[0]
    True
    >>> vae.gradient_check(X[:6]) < 1e-7      # backprop matches finite differences
    True
    >>> vae.encode(X).shape
    (60, 3)
    >>> vae.reconstruct(X).shape
    (60, 8)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: int = 32,
        lr: float = 0.01,
        epochs: int = 100,
        response_weight: float = 1.0,
        seed: int = RANDOM_STATE,
    ) -> None:
        if latent_dim < 1:
            raise DataValidationError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden < 1:
            raise DataValidationError(f"hidden must be >= 1, got {hidden}")
        if lr <= 0:
            raise DataValidationError(f"lr must be > 0, got {lr}")
        if epochs < 1:
            raise DataValidationError(f"epochs must be >= 1, got {epochs}")
        if response_weight < 0:
            raise DataValidationError(f"response_weight must be >= 0, got {response_weight}")
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.response_weight = float(response_weight)
        self.seed = int(seed)
        self.elbo_history_: tuple[float, ...] = ()
        self._fitted = False
        self._has_response = False

    def _init_weights(self, n_features: int, with_response: bool) -> None:
        rng = np.random.default_rng(self.seed)
        self.enc_w1, self.enc_b1 = _xavier_pair(rng, n_features, self.hidden)
        self.enc_w2, self.enc_b2 = _xavier_pair(rng, self.hidden, self.latent_dim)  # mu
        self.enc_w3, self.enc_b3 = _xavier_pair(rng, self.hidden, self.latent_dim)  # logvar
        self.dec_w1, self.dec_b1 = _xavier_pair(rng, self.latent_dim, self.hidden)
        self.dec_w2, self.dec_b2 = _xavier_pair(rng, self.hidden, n_features)
        if with_response:
            self.resp_w, self.resp_b = _xavier_pair(rng, self.latent_dim, 1)

    # -- forward / backward ------------------------------------------------ #

    def _loss_and_grads(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        eps: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
        """One ELBO evaluation with its exact gradient (fixed noise)."""
        n = X.shape[0]
        hidden = np.tanh(X @ self.enc_w1 + self.enc_b1)
        mu = hidden @ self.enc_w2 + self.enc_b2
        logvar = hidden @ self.enc_w3 + self.enc_b3
        sigma = np.exp(0.5 * logvar)
        z = mu + eps * sigma
        decoder_hidden = np.tanh(z @ self.dec_w1 + self.dec_b1)
        reconstruction = decoder_hidden @ self.dec_w2 + self.dec_b2
        response_hat = z @ self.resp_w + self.resp_b if y is not None else None

        recon_loss = float(np.sum((reconstruction - X) ** 2) / n)
        kl_loss = float(np.sum(np.exp(logvar) + mu**2 - 1.0 - logvar) / (2.0 * n))
        response_loss = 0.0
        if y is not None:
            response_loss = self.response_weight * float(
                np.sum((response_hat - y.reshape(-1, 1)) ** 2) / n
            )
        loss = recon_loss + kl_loss + response_loss
        parts = {"reconstruction": recon_loss, "kl": kl_loss, "response": response_loss}

        grads: dict[str, np.ndarray] = {}
        # decoder
        d_recon = 2.0 * (reconstruction - X) / n
        grads["dec_w2"] = decoder_hidden.T @ d_recon
        grads["dec_b2"] = d_recon.sum(axis=0)
        d_decoder_hidden = d_recon @ self.dec_w2.T
        d_decoder_pre = d_decoder_hidden * (1.0 - decoder_hidden**2)
        grads["dec_w1"] = z.T @ d_decoder_pre
        grads["dec_b1"] = d_decoder_pre.sum(axis=0)
        d_z = d_decoder_pre @ self.dec_w1.T
        # response head
        if y is not None:
            d_response = (2.0 * self.response_weight / n) * (response_hat - y.reshape(-1, 1))
            grads["resp_w"] = z.T @ d_response
            grads["resp_b"] = d_response.sum(axis=0)
            d_z = d_z + d_response @ self.resp_w.T
        # KL and the reparameterisation trick
        d_mu = d_z + mu / n
        d_logvar = d_z * eps * 0.5 * sigma + (np.exp(logvar) - 1.0) / (2.0 * n)
        grads["enc_w2"] = hidden.T @ d_mu
        grads["enc_b2"] = d_mu.sum(axis=0)
        grads["enc_w3"] = hidden.T @ d_logvar
        grads["enc_b3"] = d_logvar.sum(axis=0)
        d_hidden = d_mu @ self.enc_w2.T + d_logvar @ self.enc_w3.T
        d_encoder_pre = d_hidden * (1.0 - hidden**2)
        grads["enc_w1"] = X.T @ d_encoder_pre
        grads["enc_b1"] = d_encoder_pre.sum(axis=0)
        return loss, grads, parts

    def _parameters(self, with_response: bool) -> list[tuple[str, np.ndarray]]:
        names = ["enc_w1", "enc_b1", "enc_w2", "enc_b2", "enc_w3", "enc_b3",
                 "dec_w1", "dec_b1", "dec_w2", "dec_b2"]
        if with_response:
            names += ["resp_w", "resp_b"]
        return [(name, getattr(self, name)) for name in names]

    # -- the public surface ------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]],
        y: np.ndarray | Sequence[float] | None = None,
        seed: int | None = None,
    ) -> VAEHistory:
        """Train the autoencoder (and the response head when ``y`` is given).

        Parameters
        ----------
        X:
            Omics profiles (rows = samples).  Standardise first
            (:class:`MultiOmicsFusion`,
            :func:`pharmaml.biology.omics.normalize_expression`) -- the
            decoder's linear head expects comparable feature scales.
        y:
            Optional drug-response values, one per row of ``X``;
            trains the latent response head alongside the ELBO.
        seed:
            Seed for the per-epoch reparameterisation noise (default:
            the constructor's pin).

        Returns
        -------
        VAEHistory
            The per-epoch ELBO (negative loss) and its components.

        Raises
        ------
        DataValidationError
            On shape mismatch or non-finite data.
        """
        matrix = _as_matrix(X, "X")
        target: np.ndarray | None = None
        if y is not None:
            target = _as_vector(y, "y")
            if target.size != matrix.shape[0]:
                raise DataValidationError(
                    f"y has {target.size} values for {matrix.shape[0]} rows of X"
                )
        with_response = target is not None
        self._init_weights(matrix.shape[1], with_response)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        elbos: list[float] = []
        reconstructions: list[float] = []
        kls: list[float] = []
        responses: list[float] | None = [] if with_response else None
        for _epoch in range(self.epochs):
            eps = rng.standard_normal((matrix.shape[0], self.latent_dim))
            loss, grads, parts = self._loss_and_grads(matrix, target, eps)
            for name, gradient in grads.items():
                setattr(self, name, getattr(self, name) - self.lr * gradient)
            elbos.append(-loss)
            reconstructions.append(parts["reconstruction"])
            kls.append(parts["kl"])
            if responses is not None:
                responses.append(parts["response"])
        self._fitted = True
        self._has_response = with_response
        self.elbo_history_ = tuple(elbos)
        history = VAEHistory(
            elbo=tuple(elbos),
            reconstruction=tuple(reconstructions),
            kl_divergence=tuple(kls),
            response_loss=tuple(responses) if responses is not None else None,
            n_epochs=int(self.epochs),
        )
        _LOG.info(
            "response VAE fitted: %d x %d -> latent %d, elbo %.4g -> %.4g%s",
            matrix.shape[0],
            matrix.shape[1],
            self.latent_dim,
            elbos[0],
            elbos[-1],
            " (response head on)" if with_response else "",
        )
        return history

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise DataValidationError("the autoencoder has not been fitted yet -- call fit(X) first")

    def _as_input(self, X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]]) -> np.ndarray:
        return _as_matrix(X, "X")

    def encode(self, X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]]) -> np.ndarray:
        """The posterior mean ``mu`` of each row (the deterministic code).

        Raises
        ------
        DataValidationError
            If unfitted, the width disagrees with training, or values
            are non-finite.
        """
        self._require_fitted()
        matrix = self._as_input(X)
        hidden = np.tanh(matrix @ self.enc_w1 + self.enc_b1)
        return hidden @ self.enc_w2 + self.enc_b2

    def sample(
        self,
        X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]] | None = None,
        n: int | None = None,
        seed: int = RANDOM_STATE,
    ) -> np.ndarray:
        """Draw latent codes: posterior samples for ``X`` or prior draws.

        With ``X``: ``z = mu + eps·sigma`` per row (the reparameterised
        posterior draw).  With ``n``: ``n`` draws from the standard
        normal prior -- the generative side of the model.

        Raises
        ------
        DataValidationError
            If neither (or both) of ``X``/``n`` is given, or the model
            is unfitted.
        """
        self._require_fitted()
        rng = np.random.default_rng(seed)
        if X is None:
            if n is None:
                raise DataValidationError("pass exactly one of X (posterior) or n (prior)")
            return rng.standard_normal((int(n), self.latent_dim))
        if n is not None:
            raise DataValidationError("pass exactly one of X (posterior) or n (prior)")
        matrix = self._as_input(X)
        hidden = np.tanh(matrix @ self.enc_w1 + self.enc_b1)
        mu = hidden @ self.enc_w2 + self.enc_b2
        logvar = hidden @ self.enc_w3 + self.enc_b3
        return mu + rng.standard_normal(mu.shape) * np.exp(0.5 * logvar)

    def decode(self, z: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Reconstruct omics profiles from latent codes ``z``."""
        self._require_fitted()
        codes = _as_matrix(z, "z")
        if codes.shape[1] != self.latent_dim:
            raise DataValidationError(
                f"z has {codes.shape[1]} columns, the latent space is {self.latent_dim}"
            )
        hidden = np.tanh(codes @ self.dec_w1 + self.dec_b1)
        return hidden @ self.dec_w2 + self.dec_b2

    def reconstruct(self, X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]]) -> np.ndarray:
        """The deterministic reconstruction ``decode(encode(X))``."""
        return self.decode(self.encode(X))

    def predict(self, X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]]) -> np.ndarray:
        """The response head's prediction for each row of ``X``.

        Raises
        ------
        DataValidationError
            If the model was fitted without a response vector (``y``).
        """
        self._require_fitted()
        if not self._has_response:
            raise DataValidationError(
                "the autoencoder was fitted without a response vector -- pass y to "
                "fit(X, y) to train the response head"
            )
        return (self.encode(X) @ self.resp_w + self.resp_b).reshape(-1)

    def gradient_check(
        self,
        X: np.ndarray | pd.DataFrame | Sequence[Sequence[float]],
        y: np.ndarray | Sequence[float] | None = None,
        epsilon: float = 1e-6,
        seed: int = 0,
    ) -> float:
        """Max |analytic − finite-difference| over every weight.

        Uses one fixed reparameterisation noise draw (seeded) for both
        the analytic and the numeric side, so the stochastic path
        cancels and the check is exact.

        Raises
        ------
        DataValidationError
            If the model is unfitted or inputs are invalid.
        """
        self._require_fitted()
        matrix = self._as_input(X)
        target = _as_vector(y, "y") if y is not None else None
        if target is not None and target.size != matrix.shape[0]:
            raise DataValidationError("y must have one value per row of X")
        with_response = target is not None
        if with_response != self._has_response:
            raise DataValidationError(
                "gradient_check must mirror the fit: pass y if and only if the "
                "model was fitted with a response vector"
            )
        rng = np.random.default_rng(seed)
        eps = rng.standard_normal((matrix.shape[0], self.latent_dim))
        _loss, analytic, _parts = self._loss_and_grads(matrix, target, eps)

        def loss_of_current() -> float:
            return self._loss_and_grads(matrix, target, eps)[0]

        worst = 0.0
        for name, array in self._parameters(with_response):
            flat = array.reshape(-1)
            reference = analytic[name].reshape(-1)
            for index in range(flat.size):
                original = flat[index]
                flat[index] = original + epsilon
                plus = loss_of_current()
                flat[index] = original - epsilon
                minus = loss_of_current()
                flat[index] = original
                numeric = (plus - minus) / (2.0 * epsilon)
                worst = max(worst, abs(numeric - reference[index]))
        _LOG.debug("response VAE gradient_check: max deviation %.3e", worst)
        return float(worst)


# ---------------------------------------------------------------------------
# Evaluation: metrics and cell-line-grouped cross-validation
# ---------------------------------------------------------------------------


def response_metrics(
    y_true: np.ndarray | Sequence[float] | pd.Series,
    y_pred: np.ndarray | Sequence[float] | pd.Series,
) -> dict[str, float]:
    """The four numbers every response paper reports.

    Pearson and Spearman reuse :func:`pharmaml.ml.stats.rank_correlations`
    (the toolkit's own rank machinery -- Spearman is Pearson on average
    ranks, ties share the mean of their span); RMSE and MAE are the
    plain NumPy norms.  Pairs where either side is NaN are dropped
    before everything (the untested pairs of a screen), and the
    surviving count rides along as ``n``.

    Parameters
    ----------
    y_true:
        Observed response values (NaN allowed).
    y_pred:
        Predicted response values, same length (NaN allowed).

    Returns
    -------
    dict of str to float
        Keys ``pearson``, ``spearman``, ``rmse``, ``mae`` and ``n``;
        the correlations are NaN when fewer than two pairs survive or
        either side is constant.

    Raises
    ------
    DataValidationError
        On length mismatch.

    Examples
    --------
    >>> report = response_metrics([1.0, 2.0, 3.0, 4.0], [1.2, 1.8, 3.4, 3.6])
    >>> round(report["pearson"], 3)
    0.96
    >>> round(report["rmse"], 3)
    0.316
    >>> round(report["mae"], 3)
    0.3
    >>> short = response_metrics([1.0, float("nan"), 3.0], [1.0, 2.0, 2.0])
    >>> short["n"]
    2
    """
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size != prediction.size:
        raise DataValidationError(
            f"y_true and y_pred must have equal length, got {(truth.size, prediction.size)!r}"
        )
    mask = np.isfinite(truth) & np.isfinite(prediction)
    n = int(mask.sum())
    if n < 2:
        _LOG.warning("only %d joint finite pair(s) -- correlations undefined, set to NaN", n)
        return {"pearson": float("nan"), "spearman": float("nan"),
                "rmse": float("nan"), "mae": float("nan"), "n": n}
    observed_true = truth[mask]
    observed_pred = prediction[mask]
    ranks = rank_correlations(observed_true, observed_pred)
    error = observed_true - observed_pred
    return {
        "pearson": float(ranks.pearson),
        "spearman": float(ranks.spearman),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "n": n,
    }


@dataclass(frozen=True)
class ResponseCVFold:
    """One cell-line-grouped fold's ledger."""

    fold: int
    n_test_cells: int
    n_observed: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class ResponseCVReport:
    """The cross-validation ledger: per-fold rows plus the means."""

    folds: tuple[ResponseCVFold, ...]
    n_folds: int

    @property
    def summary(self) -> dict[str, float]:
        """The across-fold mean of every metric (NaN folds skipped)."""
        keys = ("pearson", "spearman", "rmse", "mae", "n")
        summary: dict[str, float] = {}
        for key in keys:
            values = [
                fold.metrics[key]
                for fold in self.folds
                if np.isfinite(fold.metrics.get(key, float("nan")))
            ]
            summary[key] = float(np.mean(values)) if values else float("nan")
        return summary

    def frame(self) -> pd.DataFrame:
        """The per-fold metrics as a DataFrame (one row per fold)."""
        rows = [
            {"fold": fold.fold, "n_test_cells": fold.n_test_cells, **fold.metrics}
            for fold in self.folds
        ]
        return pd.DataFrame(rows)


def cross_validate_response(
    model_factory: Callable[[], DrugResponseModel],
    X: pd.DataFrame | np.ndarray,
    table: pd.DataFrame,
    n_folds: int = 5,
    seed: int = RANDOM_STATE,
) -> ResponseCVReport:
    """Cell-line-grouped cross-validation of a response model.

    Folds split *cell lines*, never (cell line, drug) pairs -- the
    leak the grouped design exists to prevent: a model that memorised
    a cell line's profile must not meet it in validation.  Each fold
    trains a fresh model (from ``model_factory``) on the other cell
    lines' rows of ``X``/``table`` and scores its predictions on the
    held-out rows with :func:`response_metrics`.

    Parameters
    ----------
    model_factory:
        Zero-argument factory returning a fresh, unfitted
        :class:`DrugResponseModel` (or anything with the same
        ``fit(X, table)`` / ``predict(X)`` contract).
    X:
        Cell-line features (rows = cell lines).
    table:
        The (cell line x drug) response table, rows aligned with ``X``
        (a DataFrame ``X`` aligns by index; an ndarray positionally).
    n_folds:
        Number of cell-line groups (2..n_cells).
    seed:
        The shuffle seed for the group assignment.

    Returns
    -------
    ResponseCVReport
        Per-fold metrics plus the across-fold ``summary``.

    Raises
    ------
    DataValidationError
        On bad ``n_folds`` or misaligned ``X``/``table``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> features = pd.DataFrame(rng.normal(size=(20, 4)),
    ...                         index=[f"c{i}" for i in range(20)])
    >>> screen = pd.DataFrame(
    ...     {"d1": 1.0 / (1.0 + np.exp(-2.0 * features[0])),
    ...      "d2": 1.0 / (1.0 + np.exp(1.5 * features[1]))},
    ...     index=features.index,
    ... )
    >>> report = cross_validate_response(
    ...     lambda: DrugResponseModel(n_factors=3, lr=0.5, epochs=200, seed=42),
    ...     features, screen, n_folds=4, seed=42,
    ... )
    >>> len(report.folds)
    4
    >>> report.summary["n"] >= 8.0        # every fold scored observed pairs
    True
    """
    if isinstance(X, pd.DataFrame):
        cell_labels = [str(label) for label in X.index]
        matrix = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        matrix = _as_matrix(X, "X")
        cell_labels = [f"cell_{i}" for i in range(matrix.shape[0])]
    if not isinstance(table, pd.DataFrame):
        raise DataValidationError("table must be a pandas DataFrame (rows = cell lines)")
    aligned = table.apply(pd.to_numeric, errors="coerce")
    if isinstance(X, pd.DataFrame):
        missing = [label for label in cell_labels if label not in set(aligned.index)]
        if missing:
            raise DataValidationError(
                f"{len(missing)} cell line(s) of X missing from table's rows, "
                f"e.g. {missing[:3]}"
            )
        aligned = aligned.loc[cell_labels]
    elif aligned.shape[0] != matrix.shape[0]:
        raise DataValidationError(
            f"table has {aligned.shape[0]} rows for {matrix.shape[0]} feature rows"
        )
    n_cells = matrix.shape[0]
    if not 2 <= n_folds <= n_cells:
        raise DataValidationError(f"n_folds must be in [2, {n_cells}], got {n_folds}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_cells)
    groups = np.array_split(shuffled, n_folds)
    folds: list[ResponseCVFold] = []
    for fold_index, test_rows in enumerate(groups):
        train_rows = np.setdiff1d(shuffled, test_rows, assume_unique=True)
        model = model_factory()
        train_table = aligned.iloc[train_rows]
        test_table = aligned.iloc[test_rows]
        model.fit(matrix[train_rows], train_table)
        predicted = model.predict(matrix[test_rows])
        observed = test_table.to_numpy(dtype=float)
        prediction = predicted.to_numpy(dtype=float)
        mask = np.isfinite(observed)
        folds.append(
            ResponseCVFold(
                fold=fold_index,
                n_test_cells=int(test_rows.size),
                n_observed=int(mask.sum()),
                metrics=response_metrics(observed[mask], prediction[mask]),
            )
        )
    report = ResponseCVReport(folds=tuple(folds), n_folds=int(n_folds))
    _LOG.info(
        "cross-validated response model: %d fold(s) over %d cell lines, mean rmse %.4g",
        n_folds,
        n_cells,
        report.summary["rmse"],
    )
    return report
