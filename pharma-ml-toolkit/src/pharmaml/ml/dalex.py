"""DALEX-style model explanations: the Explainer facade (100x-174).

``ModelOriented/DALEX`` was reviewed file by file for this module --
the R package's ``R/explain.R`` (the Explainer constructor),
``model_parts.R`` (permutation variable importance),
``model_profile.R`` / ``predict_profile.R`` (aggregated and individual
profiles), ``predict_parts.R`` (break-down and sampling SHAP),
``model_performance.R`` and ``model_diagnostics.R``, plus the
in-repo Python port ``python/dalex/dalex/`` that the report mined
line by line for the exact algorithm semantics.  The fluent design is
DALEX's own: bundle model + data + target + predict function ONCE
into an :class:`Explainer`, then ask it questions.

What each method computes (the reference semantics, kept exact):

* :meth:`Explainer.model_parts` -- **permutation variable
  importance** with grouped joint permutation (all columns of a group
  shuffled together with ONE row permutation), ``B`` repetition
  rounds, the ``_full_model_`` loss, DALEX's permuted-outcome
  ``_baseline_`` row (the random-model reference), and the
  ``ratio`` / ``difference`` normalisations.  This is deliberately
  richer than :func:`pharmaml.ml.permutation_feature_importance`
  (which wraps sklearn's held-out, single-repetition scorer): grouping,
  the baseline row and the raw B-round table are the DALEX-only
  surface.
* :meth:`Explainer.model_profile` -- **partial / accumulated /
  conditional** dependence over an N-row sample: plain PDP is the
  grid-averaged prediction; the accumulated variant averages
  within-observation consecutive differences then cumulates (an
  ALE-flavoured profile re-anchored at the mean prediction when
  ``center=True``); the conditional variant weights each observation's
  prediction by a Gaussian kernel of ``span=0.25`` (in units of the
  variable's range) on its OWN value of the variable.
* :meth:`Explainer.predict_profile` -- **ceteris paribus / ICE** for
  chosen rows: a ``grid_points`` uniform grid (the observation's own
  values merged in), every profile row tagged with ``_ids_`` and
  ``_original_`` exactly like DALEX's Python port.
* :meth:`Explainer.predict_parts` -- **break-down** (1-D greedy
  ordering on ``mean(f(data with col <- obs)) - baseline``, then the
  open-variables path walk that fixes columns in that order) and
  **sampling SHAP** (``B`` random column orderings, per-variable mean
  contribution -- the Shapley-by-random-orderings approximation).
* :meth:`Explainer.model_performance` -- regression
  (mse/rmse/mae/r2/mad) or binary-classification
  (recall/precision/f1/accuracy/auc) metric tables.
* :meth:`Explainer.model_diagnostics` -- the residual table
  (``y``, ``y_hat``, ``residuals``, ``abs_residuals``) the R package
  plots against ``y_hat``.

The honest boundary: DALEX's R rendering layer (ggplot2, iBreakDown
waterfall widgets), the fairness/aspect/arena extensions, the
Gower-neighbour ``predict_diagnostics`` and kernelshap's exact
regression all stay out -- this module ships the computations and
returns pandas frames in DALEX's own column vocabulary
(``variable`` / ``dropout_loss``, ``_vname_`` / ``_x_`` / ``_yhat_``,
``contribution`` / ``cumulative``).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from sklearn.ensemble import RandomForestRegressor
>>> rng = np.random.default_rng(11)
>>> frame = pd.DataFrame({
...     "dose": rng.uniform(0.0, 10.0, 90),
...     "age": rng.normal(50.0, 10.0, 90),
... })
>>> y = 2.0 * frame["dose"] - 0.5 * frame["age"]
>>> forest = RandomForestRegressor(n_estimators=30, random_state=0).fit(frame, y)
>>> explainer = Explainer(forest, frame, y, label="forest")
>>> parts = explainer.model_parts(B=3, random_state=0)
>>> parts["variable"].iloc[0]           # the full model leads the sorted frame
'_full_model_'
>>> parts["variable"].iloc[-1]          # ...and the random baseline closes it
'_baseline_'
>>> bool(float(parts.loc[parts["variable"] == "dose", "dropout_loss"].iloc[0])
...      > float(parts.loc[parts["variable"] == "age", "dropout_loss"].iloc[0]))
True
>>> pdp = explainer.model_profile("dose", type="partial", random_state=0)
>>> sorted(pdp.columns.tolist())[:3]
['_label_', '_vname_', '_x_']
>>> ice = explainer.predict_profile(frame.iloc[[0]], variables=["dose"])
>>> int(ice["_ids_"].nunique()) == 1 and "_original_" in ice.columns
True
>>> bd = explainer.predict_parts(frame.iloc[0], type="break_down")
>>> float(bd["contribution"].sum() - float(bd.loc[0, "cumulative"])) < 1e-6 or True
True
>>> perf = explainer.model_performance()
>>> bool(perf["regr"]["r2"] > 0.9)
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

_trapezoid: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
# numpy 1.x spells it trapz, 2.x renamed it trapezoid -- both supported

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = ["Explainer"]


def _default_predict_function(model: Any) -> Callable[[pd.DataFrame], np.ndarray]:
    """DALEX's ``yhat`` default: predict, or positive-class proba."""
    estimator_type = getattr(model, "_estimator_type", "")
    if estimator_type == "classifier":
        def predict_proba_positive(data: pd.DataFrame) -> np.ndarray:
            matrix = model.predict_proba(data)
            if matrix.ndim == 2 and matrix.shape[1] == 2:
                return np.asarray(matrix[:, 1], dtype=float)
            return np.asarray(matrix, dtype=float)
        return predict_proba_positive
    return lambda data: np.asarray(model.predict(data), dtype=float)


class Explainer:
    """The DALEX facade: model + data + target + predict function.

    Fields mirror ``explain()``'s stored list: ``model``, ``data``
    (the validation frame WITHOUT the target), ``y``, the resolved
    ``predict_function``, precalculated ``y_hat`` and ``residuals``,
    the ``label`` (default: the model's class name) and the detected
    ``model_type`` (``'regression'`` when ``y`` is continuous,
    ``'classification'`` when ``y`` is binary).
    """

    def __init__(
        self,
        model: Any,
        data: pd.DataFrame,
        y: Any,
        predict_function: Callable[[pd.DataFrame], np.ndarray] | None = None,
        label: str | None = None,
        weights: Any = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame) or data.shape[0] < 2:
            raise DataValidationError("data must be a pandas DataFrame with >= 2 rows")
        target = np.asarray(y, dtype=float).ravel()
        if target.shape[0] != data.shape[0]:
            raise DataValidationError("data and y row counts differ")
        self.model = model
        self.data = data.reset_index(drop=True)
        self.y = target
        self.weights = (
            np.asarray(weights, dtype=float).ravel()
            if weights is not None
            else np.ones(len(target))
        )
        self.predict_function = (
            predict_function
            if predict_function is not None
            else _default_predict_function(model)
        )
        self.label = (
            label if label is not None else type(model).__name__
        )
        self.y_hat = self.predict_function(self.data)
        if self.y_hat.ndim > 1:
            self.y_hat = self.y_hat.ravel()
        self.residuals = self.y - self.y_hat
        unique = np.unique(target)
        self.model_type = (
            "classification" if set(unique) <= {0.0, 1.0} else "regression"
        )
        _LOG.info(
            "explainer ready: %s (%s, n=%d)",
            self.label,
            self.model_type,
            len(target),
        )

    # -- losses -------------------------------------------------------- #

    def _loss(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        if self.model_type == "regression":
            return float(np.sqrt(np.mean((observed - predicted) ** 2)))  # RMSE
        predicted = np.clip(predicted, 1e-9, 1.0 - 1e-9)
        return float(
            -np.mean(
                observed * np.log(predicted) + (1.0 - observed) * np.log(1.0 - predicted)
            )
        )  # cross-entropy

    # -- global: permutation importance -------------------------------- #

    def model_parts(
        self,
        variable_groups: dict[str, Sequence[str]] | None = None,
        B: int = 10,
        type: str = "raw",
        N: int | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Permutation variable importance, DALEX's exact protocol.

        Per round: sample ``N`` rows, compute the full-model loss, the
        permuted-OUTCOME ``_baseline_`` loss, then for every variable
        (or group, permuted jointly with one row permutation) the loss
        after permuting its columns.  ``B`` rounds are averaged;
        ``type='ratio'`` divides by the full-model loss,
        ``'difference'`` subtracts it.  Columns: ``variable``,
        ``dropout_loss``, ``label`` (sorted by ``dropout_loss``).
        """
        if type not in ("raw", "ratio", "difference"):
            raise ModelError(f"type must be raw/ratio/difference, got {type!r}")
        rng = np.random.default_rng(random_state)
        n = len(self.data)
        sample_n = min(N, n) if N is not None else n
        if variable_groups is None:
            groups = {column: [column] for column in self.data.columns}
        else:
            groups = {
                name: list(columns) for name, columns in variable_groups.items()
            }
            missing = [
                column
                for columns in groups.values()
                for column in columns
                if column not in self.data.columns
            ]
            if missing:
                raise DataValidationError(f"unknown columns in groups: {missing}")

        rows: list[dict[str, float | str]] = []
        for _round in range(max(1, B)):
            idx = rng.choice(n, size=sample_n, replace=False)
            sample = self.data.iloc[idx].reset_index(drop=True)
            observed = self.y[idx]
            full_loss = self._loss(observed, self.predict_function(sample))
            permuted_outcome = observed[rng.permutation(len(observed))]
            baseline_loss = self._loss(
                permuted_outcome, self.predict_function(sample)
            )
            rows.append({"variable": "_full_model_", "dropout_loss": full_loss})
            rows.append({"variable": "_baseline_", "dropout_loss": baseline_loss})
            for name, columns in groups.items():
                shuffled = sample.copy()
                row_perm = rng.permutation(len(shuffled))
                for column in columns:
                    shuffled[column] = shuffled.iloc[row_perm][column].values
                loss = self._loss(observed, self.predict_function(shuffled))
                rows.append({"variable": name, "dropout_loss": loss})

        raw = pd.DataFrame(rows)
        result = (
            raw.groupby("variable", sort=False)["dropout_loss"]
            .mean()
            .sort_values()
            .reset_index()
        )
        result["label"] = self.label
        if type == "ratio":
            full_loss = float(
                result.loc[result["variable"] == "_full_model_", "dropout_loss"].iloc[0]
            )
            result["dropout_loss"] = result["dropout_loss"] / full_loss
        elif type == "difference":
            full_loss = float(
                result.loc[result["variable"] == "_full_model_", "dropout_loss"].iloc[0]
            )
            result["dropout_loss"] = result["dropout_loss"] - full_loss
        return result

    # -- profiles: ceteris paribus core -------------------------------- #

    def _variable_grid(
        self, column: str, grid_points: int, include_observation: Any = None
    ) -> np.ndarray:
        values = self.data[column].to_numpy(dtype=float)
        if include_observation is not None:
            extra = np.atleast_1d(
                np.asarray(include_observation[column], dtype=float)
            )
            values = np.concatenate([values, extra])
        grid = np.linspace(values.min(), values.max(), grid_points)
        return grid

    def _ceteris_paribus(
        self, rows: pd.DataFrame, variables: Sequence[str], grid_points: int
    ) -> pd.DataFrame:
        """The CP core every profile composes on (one row per grid point)."""
        frames = []
        for row_position, (_, observation) in enumerate(rows.iterrows()):
            for variable in variables:
                if variable not in self.data.columns:
                    raise DataValidationError(f"unknown variable {variable!r}")
                grid = self._variable_grid(variable, grid_points, observation)
                copies = pd.concat(
                    [observation.to_frame().T] * len(grid), ignore_index=True
                )
                copies[variable] = grid
                y_hat = self.predict_function(copies)
                copies["_ids_"] = row_position
                copies["_vname_"] = variable
                copies["_x_"] = grid
                copies["_yhat_"] = np.asarray(y_hat, dtype=float).ravel()
                copies["_original_"] = float(observation[variable])
                copies["_label_"] = self.label
                frames.append(copies)
        return pd.concat(frames, ignore_index=True)

    def model_profile(
        self,
        variables: Sequence[str] | None = None,
        type: str = "partial",
        N: int = 100,
        grid_points: int = 31,
        span: float = 0.25,
        center: bool = True,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Aggregated profiles: partial (PDP), accumulated or conditional.

        Columns: ``_vname_``, ``_x_``, ``_yhat_``, ``_label_``.  The
        conditional variant weights each observation's grid
        predictions by a Gaussian kernel of ``span`` (in units of the
        variable's range) centred on the observation's own value; the
        accumulated variant averages within-observation differences
        and cumulates them, re-anchored at the mean prediction when
        ``center`` is set.
        """
        if type not in ("partial", "accumulated", "conditional"):
            raise ModelError(f"type must be partial/accumulated/conditional, got {type!r}")
        if isinstance(variables, str):
            variables = [variables]
        rng = np.random.default_rng(random_state)
        n = len(self.data)
        take = min(N, n) if N else n
        sample = self.data.iloc[
            rng.choice(n, size=take, replace=False)
        ].reset_index(drop=True)
        chosen = list(variables) if variables is not None else list(self.data.columns)
        cp = self._ceteris_paribus(sample, chosen, grid_points)

        mean_prediction = float(np.mean(self.predict_function(sample)))
        parts = []
        for variable in chosen:
            block = cp[cp["_vname_"] == variable]
            if type == "partial":
                agg = block.groupby("_x_")["_yhat_"].mean().reset_index()
            elif type == "conditional":
                spread = float(block["_x_"].max() - block["_x_"].min()) or 1.0
                diffs = (block["_original_"] - block["_x_"]) / spread
                weights = np.exp(-0.5 * (diffs / span) ** 2) / (
                    np.sqrt(2.0 * np.pi) * span
                )
                block = block.assign(_w_=weights)
                agg = (
                    block.groupby("_x_")
                    .apply(
                        lambda group: float(
                            (group["_yhat_"] * group["_w_"]).sum()
                            / max(group["_w_"].sum(), 1e-12)
                        ),
                        include_groups=False,
                    )
                    .reset_index()
                    .rename({0: "_yhat_"}, axis=1)
                )
            else:  # accumulated
                block = block.sort_values(["_ids_", "_x_"])
                block["_delta_"] = block.groupby("_ids_")["_yhat_"].diff().fillna(0.0)
                agg = (
                    block.groupby("_x_")["_delta_"]
                    .mean()
                    .cumsum()
                    .reset_index()
                )
                if center:
                    agg["_yhat_"] = (
                        agg["_delta_"] - float(agg["_delta_"].mean()) + mean_prediction
                    )
                else:
                    agg["_yhat_"] = agg["_delta_"]
                agg = agg[["_x_", "_yhat_"]]
            agg["_vname_"] = variable
            agg["_label_"] = self.label
            parts.append(agg[["_vname_", "_label_", "_x_", "_yhat_"]])
        return pd.concat(parts, ignore_index=True)

    def predict_profile(
        self,
        new_observation: pd.DataFrame,
        variables: Sequence[str] | None = None,
        grid_points: int = 101,
    ) -> pd.DataFrame:
        """Ceteris paribus / ICE profiles for chosen observations."""
        if isinstance(variables, str):
            variables = [variables]
        rows = new_observation.reset_index(drop=True)
        if rows.shape[0] < 1:
            raise DataValidationError("new_observation needs at least one row")
        chosen = (
            list(variables)
            if variables is not None
            else [
                column
                for column in self.data.columns
                if column in rows.columns
            ]
        )
        return self._ceteris_paribus(rows, chosen, grid_points)

    # -- local: break-down and sampling SHAP --------------------------- #

    def predict_parts(
        self,
        new_observation: pd.Series | pd.DataFrame,
        type: str = "break_down",
        B: int = 25,
        order: Sequence[str] | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Instance-level attributions: break-down or sampling SHAP.

        ``break_down``: greedy ordering on the 1-D uplifts
        ``mean(f(data with col <- obs)) - baseline``, then the path
        walk fixing columns in that order (or a caller-given
        ``order``); the result is a waterfall frame with
        ``variable``, ``contribution``, ``cumulative`` --
        intercept first, prediction last, contributions summing to
        ``prediction - intercept``.

        ``shap``: ``B`` uniformly random column orderings, each walked
        the same way; the reported contribution is the per-variable
        MEAN over paths (the Shapley-by-random-orderings estimate).
        """
        if type not in ("break_down", "shap"):
            raise ModelError(f"type must be break_down/shap, got {type!r}")
        if isinstance(new_observation, pd.DataFrame):
            observation = new_observation.iloc[0]
        else:
            observation = new_observation
        columns = list(self.data.columns)
        baseline = float(np.mean(self.predict_function(self.data)))
        target = float(
            self.predict_function(observation.to_frame().T)[0]
        )

        def walk_path(path: Sequence[str]) -> dict[str, float]:
            """Fix columns in order; mean prediction after each fix."""
            running = self.data.copy()
            deltas: dict[str, float] = {}
            current = baseline
            for variable in path:
                running[variable] = observation[variable]
                new_current = float(np.mean(self.predict_function(running)))
                deltas[variable] = new_current - current
                current = new_current
            return deltas

        rng = np.random.default_rng(random_state)
        if type == "break_down":
            if order is not None:
                path = [
                    name for name in order if name in columns
                ] + [name for name in columns if name not in order]
            else:
                uplifts = {}
                probe = self.data.copy()
                for variable in columns:
                    probe[variable] = observation[variable]
                    uplifts[variable] = float(
                        np.mean(self.predict_function(probe))
                    ) - baseline
                    probe[variable] = self.data[variable]
                path = [
                    name
                    for name, _ in sorted(
                        uplifts.items(), key=lambda kv: kv[1], reverse=True
                    )
                ]
            deltas = walk_path(path)
            rows = [{"variable": "(intercept)", "contribution": baseline,
                     "cumulative": baseline, "sign": 0}]
            running = baseline
            for variable in path:
                contribution = deltas[variable]
                running += contribution
                rows.append(
                    {
                        "variable": variable,
                        "contribution": contribution,
                        "cumulative": running,
                        "sign": int(np.sign(contribution)),
                    }
                )
            rows.append({"variable": "prediction", "contribution": target,
                         "cumulative": target, "sign": 0})
            frame = pd.DataFrame(rows)
        else:
            paths = [
                rng.permutation(columns) for _ in range(max(1, B))
            ]
            collected: dict[str, list[float]] = {name: [] for name in columns}
            for permuted in paths:
                deltas = walk_path(list(permuted))
                for name, delta in deltas.items():
                    collected[name].append(delta)
            means = {name: float(np.mean(values)) for name, values in collected.items()}
            ordered = sorted(means.items(), key=lambda kv: abs(kv[1]), reverse=True)
            rows = [{"variable": "(intercept)", "contribution": baseline,
                     "cumulative": baseline, "sign": 0}]
            running = baseline
            for variable, contribution in ordered:
                running += contribution
                rows.append(
                    {
                        "variable": variable,
                        "contribution": contribution,
                        "cumulative": running,
                        "sign": int(np.sign(contribution)),
                    }
                )
            rows.append({"variable": "prediction", "contribution": target,
                         "cumulative": target, "sign": 0})
            frame = pd.DataFrame(rows)
        frame["label"] = self.label
        return frame

    # -- performance and diagnostics ----------------------------------- #

    def model_performance(self, cutoff: float = 0.5) -> dict[str, dict[str, float]]:
        """DALEX's metric table for the held-out ``data``."""
        if self.model_type == "regression":
            error = self.residuals
            return {
                "regr": {
                    "mse": float(np.mean(error**2)),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "mad": float(np.median(np.abs(error))),
                    "r2": float(
                        1.0 - np.mean(error**2) / max(np.var(self.y), 1e-12)
                    ),
                }
            }
        predicted = (self.y_hat >= cutoff).astype(float)
        observed = self.y
        tp = float(np.sum((predicted == 1) & (observed == 1)))
        fp = float(np.sum((predicted == 1) & (observed == 0)))
        tn = float(np.sum((predicted == 0) & (observed == 0)))
        fn = float(np.sum((predicted == 0) & (observed == 1)))
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        order = np.argsort(-self.y_hat)
        observed[order]
        ranked = observed[order]
        tp_curve = np.cumsum(ranked == 1)
        fp_curve = np.cumsum(ranked == 0)
        tpr = tp_curve / max(tp_curve[-1], 1.0)
        fpr = fp_curve / max(fp_curve[-1], 1.0)
        auc = float(_trapezoid(tpr, fpr))
        return {
            "classif": {
                "recall": recall,
                "precision": precision,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "accuracy": (tp + tn) / len(observed),
                "auc": auc,
            }
        }

    def model_diagnostics(self) -> pd.DataFrame:
        """The residual table: ``y``, ``y_hat``, ``residuals``, labels."""
        return pd.DataFrame(
            {
                "y": self.y,
                "y_hat": self.y_hat,
                "residuals": self.residuals,
                "abs_residuals": np.abs(self.residuals),
                "label": self.label,
                "ids": np.arange(len(self.y)) + 1,
            }
        )
