"""OOF stacking (the look.ipynb upgrade): end-to-end, determinism, guards, progress."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pharmaml.ml.stacking as stacking_module
from pharmaml.exceptions import ModelError
from pharmaml.ml.models import build_model as _real_build_model
from pharmaml.ml.stacking import OOFStackingResult, stack_oof_regressors

# The shared synthetic split (module scope: every test class reuses it,
# the regressors themselves are re-fit inside each call anyway).
_X, _Y = make_regression(n_samples=80, n_features=10, noise=5.0, random_state=42)
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    _X, _Y, test_size=0.25, random_state=42
)  # 60 train / 20 test
BASE_MODELS = ["ridge", "random_forest", "extra_trees"]
META_MODELS = ["ridge", "huber_regression"]


class _ExplodingRegressor:
    """A registry-shaped estimator whose fit/predict always raise.

    Carries ``get_params``/``set_params`` so ``sklearn.base.clone``
    accepts it -- the failure is deferred to ``fit``, exactly like a
    real estimator that builds fine and blows up on bad data.
    """

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> _ExplodingRegressor:
        return self

    def fit(self, X: Any, y: Any) -> _ExplodingRegressor:
        raise RuntimeError("synthetic estimator failure")

    def predict(self, X: Any) -> np.ndarray:
        raise RuntimeError("synthetic estimator failure")


class TestStackOofRegressors:
    def test_end_to_end_synthetic(self) -> None:
        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=BASE_MODELS, meta_models=META_MODELS, n_splits=5,
        )
        assert isinstance(result, OOFStackingResult)
        # OOF matrix: one row per training sample, one column per base model
        assert result.oof_frame.shape == (60, 3)
        assert list(result.oof_frame.columns) == BASE_MODELS
        assert list(result.oof_frame.index) == list(range(60))
        # leaderboard: 2 meta rows + the residual row
        assert list(result.leaderboard.columns) == ["model", "oof_r2", "test_r2"]
        assert result.leaderboard.shape[0] == 3
        assert result.best_meta in META_MODELS
        assert np.isfinite(result.leaderboard.iloc[:2]["test_r2"]).all()
        residual_row = result.leaderboard.iloc[-1]
        assert residual_row["model"] == f"{result.best_meta}+residual"
        assert np.isfinite(residual_row["test_r2"])
        assert np.isfinite(result.best_test_r2)
        assert np.isfinite(result.residual_test_r2)
        assert result.target_transform == "none"
        assert result.n_splits == 5
        assert result.base_models == BASE_MODELS
        assert result.meta_models == META_MODELS
        assert len(result.y_test_trans) == len(Y_TEST)
        assert result.best_test_prediction.shape == Y_TEST.shape
        # the returned prediction is the residual-refined one: its R^2
        # reproduces residual_test_r2 exactly on the reported (raw) scale
        assert abs(r2_score(Y_TEST, result.best_test_prediction)
                   - result.residual_test_r2) < 1e-12

    def test_determinism_two_identical_calls(self) -> None:
        result_a = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge", "knn"], meta_models=META_MODELS, n_splits=5,
        )
        result_b = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge", "knn"], meta_models=META_MODELS, n_splits=5,
        )
        assert np.allclose(result_a.oof_frame.values, result_b.oof_frame.values)
        assert result_a.leaderboard["model"].tolist() == (
            result_b.leaderboard["model"].tolist()
        )
        assert result_a.best_meta == result_b.best_meta
        assert np.allclose(
            result_a.best_test_prediction, result_b.best_test_prediction
        )

    def test_quantile_target_transform(self) -> None:
        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge"], meta_models=["ridge"],
            target_transform="quantile",
        )
        assert result.target_transform == "quantile"
        assert len(result.y_test_trans) == len(Y_TEST)
        assert not np.allclose(result.y_test_trans, Y_TEST)  # genuinely transformed
        assert np.isfinite(result.leaderboard["test_r2"]).all()
        assert np.isfinite(result.best_test_r2)
        assert np.isfinite(result.residual_test_r2)
        # scores live on the TRANSFORMED scale: the returned prediction
        # reproduces residual_test_r2 against y_test_trans, not raw Y_TEST
        assert abs(
            r2_score(result.y_test_trans, result.best_test_prediction)
            - result.residual_test_r2
        ) < 1e-12

    def test_residual_false_has_no_residual_row(self) -> None:
        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge"], meta_models=META_MODELS, residual=False,
        )
        assert result.leaderboard.shape[0] == 2
        assert not result.leaderboard["model"].str.endswith("+residual").any()
        assert result.residual_test_r2 is None
        # best_test_prediction is the plain best-meta prediction: its R^2
        # matches the leaderboard's top-row score exactly
        assert abs(r2_score(Y_TEST, result.best_test_prediction)
                   - result.best_test_r2) < 1e-12

    def test_default_meta_models_run(self) -> None:
        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge"], n_splits=3, residual=False,
        )
        # the default list, filtered to available backends -- ridge is
        # always available, so the fallback can never leave it empty
        assert "ridge" in result.meta_models
        assert set(result.meta_models) <= {"ridge", "huber_regression", "lightgbm"}
        assert result.leaderboard.shape[0] == len(result.meta_models)

    def test_aliases_collapse_in_base_models(self) -> None:
        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge", "Ridge"], meta_models=["ridge"], residual=False,
        )
        assert result.base_models == ["ridge"]
        assert result.oof_frame.shape == (60, 1)

    def test_failing_meta_keeps_nan_row_and_sinks(self, monkeypatch) -> None:
        def fake_build_model(name: str, task: str, params: dict[str, Any] | None = None) -> Any:
            if name == "huber_regression":
                return _ExplodingRegressor()
            return _real_build_model(name, task)

        monkeypatch.setattr(stacking_module, "build_model", fake_build_model)
        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge"], meta_models=META_MODELS,
        )
        board = result.leaderboard.set_index("model")
        assert np.isnan(board.loc["huber_regression", "oof_r2"])
        assert np.isnan(board.loc["huber_regression", "test_r2"])
        # the NaN row sinks below every healthy row; the residual row is
        # appended after the sorted meta block
        assert board.index.tolist() == ["ridge", "huber_regression", "ridge+residual"]
        assert result.best_meta == "ridge"
        assert np.isfinite(result.best_test_r2)
        assert np.isfinite(result.residual_test_r2)

    def test_all_metas_failed_raises_model_error(self, monkeypatch) -> None:
        def fake_build_model(name: str, task: str, params: dict[str, Any] | None = None) -> Any:
            if name in {"ridge", "huber_regression"}:
                return _ExplodingRegressor()
            return _real_build_model(name, task)

        monkeypatch.setattr(stacking_module, "build_model", fake_build_model)
        with pytest.raises(ModelError, match="all meta-learners failed"):
            stack_oof_regressors(
                X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
                base_models=["knn"], meta_models=META_MODELS,
            )


class TestStackingGuards:
    def test_empty_base_models_raises(self) -> None:
        with pytest.raises(ModelError, match="base_models list is empty"):
            stack_oof_regressors(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, base_models=[])

    def test_unknown_base_model_raises(self) -> None:
        with pytest.raises(ModelError, match="Unknown model"):
            stack_oof_regressors(
                X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, base_models=["not_a_model"]
            )

    def test_unknown_meta_model_raises(self) -> None:
        with pytest.raises(ModelError, match="Unknown model"):
            stack_oof_regressors(
                X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
                base_models=["ridge"], meta_models=["not_a_model"],
            )

    def test_n_splits_below_two_raises(self) -> None:
        with pytest.raises(ModelError, match="n_splits must be >= 2"):
            stack_oof_regressors(
                X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, base_models=["ridge"], n_splits=1
            )

    def test_n_splits_exceeding_train_rows_raises(self) -> None:
        with pytest.raises(ModelError, match="exceeds the number of training rows"):
            stack_oof_regressors(
                X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, base_models=["ridge"], n_splits=100
            )

    def test_bad_target_transform_raises(self) -> None:
        with pytest.raises(ModelError, match="target_transform"):
            stack_oof_regressors(
                X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
                base_models=["ridge"], target_transform="zap",
            )

    def test_nan_target_raises(self) -> None:
        y_bad = Y_TRAIN.astype(float).copy()
        y_bad[3] = np.nan
        with pytest.raises(ModelError, match="NaN"):
            stack_oof_regressors(X_TRAIN, y_bad, X_TEST, Y_TEST, base_models=["ridge"])

    def test_row_count_mismatch_raises(self) -> None:
        with pytest.raises(ModelError, match="row"):
            stack_oof_regressors(
                X_TRAIN[:50], Y_TRAIN, X_TEST, Y_TEST, base_models=["ridge"]
            )

    def test_column_count_mismatch_raises(self) -> None:
        with pytest.raises(ModelError, match="column"):
            stack_oof_regressors(
                X_TRAIN, Y_TRAIN, X_TEST[:, :5], Y_TEST, base_models=["ridge"]
            )


class TestOnStepCallback:
    def test_payloads_cover_every_step(self) -> None:
        payloads: list[dict[str, Any]] = []
        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge", "knn"], meta_models=["ridge"],
            n_splits=5, on_step=payloads.append,
        )
        assert len(payloads) > 0
        assert result.best_meta == "ridge"  # the run completed
        assert {p["step"] for p in payloads} == {"oof", "base", "meta", "residual"}
        for payload in payloads:
            assert set(payload) >= {"step", "index", "total", "label"}
            assert isinstance(payload["index"], int)
            assert isinstance(payload["total"], int)
            assert 1 <= payload["index"] <= payload["total"]

    def test_raising_callback_does_not_break_run(self) -> None:
        def bad_callback(payload: dict[str, Any]) -> None:
            raise AssertionError("callbacks must never break the run")

        result = stack_oof_regressors(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
            base_models=["ridge"], meta_models=["ridge"], n_splits=3,
            on_step=bad_callback,
        )
        assert result.best_meta == "ridge"
        assert np.isfinite(result.best_test_r2)
        assert np.isfinite(result.residual_test_r2)
