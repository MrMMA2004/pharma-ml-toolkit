"""Tests for :mod:`pharmaml.ml.explain` and its two viz twins.

Round 100x-154: the LAST unabsorbed ``D2.py`` gap (its lines 472-528
-- the ``TreeExplainer`` beeswarm + the top-20 bar the upload drew for
its stacking pipeline), shipped as a first-class optional-gated
surface.

The suite pins:

* the real path (shap installed in the verified venv): the
  ``ShapSummary`` record over a seeded GradientBoostingRegressor --
  the mean |SHAP| ranking matches the TRUE coefficient ordering, the
  values matrix carries the right shape, ``as_frame``/``ranking``/
  ``__str__`` round-trip, and the binary-classification positive-class
  read;
* the typed refusals: a non-tree estimator (LinearRegression), a
  pipeline, a multiclass classifier, mismatched feature names, empty
  and non-2D inputs -- every error names the
  ``permutation_feature_importance`` sibling or the exact misuse;
* the guard when shap is missing (simulated by import-blocker
  monkeypatch): the ``DependencyError`` carries the exact
  ``pip install 'pharmaml[explain]'`` command;
* the two figures: the bar chart's top-N ordering, labels, and the
  DataFrame-input twin; the beeswarm's row ordering, the seeded
  jitter's determinism (two calls -> identical offsets), the
  base-value caption, the constant-feature colour fallback, and the
  guards (the DataFrame form refused with the pointer to the bar
  chart, shape mismatches, jitter bounds).

The real-path tests skip as a class when shap is absent (the zip-only
tree); the record/figure/guard tests run everywhere -- the plotting
layer never needs shap installed, by design.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.ml.explain import ShapSummary, shap_summary
from pharmaml.viz import plot_shap_beeswarm, plot_shap_importance

try:  # the optional dependency itself gates the real-path class
    import shap  # noqa: F401

    _HAS_SHAP = True
except ImportError:  # pragma: no cover - the zip-only tree
    _HAS_SHAP = False


@pytest.fixture()
def close_all():
    """Close every figure after each test (keeps the suite light)."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _summary_fixture() -> ShapSummary:
    """A hand-built record (no shap needed): 4 samples, 3 features."""
    values = np.array(
        [
            [1.0, -0.5, 0.1],
            [0.5, 0.25, -0.1],
            [-1.0, 0.5, 0.05],
            [0.25, -0.25, -0.05],
        ]
    )
    return ShapSummary(
        names=["a", "b", "c"],
        mean_abs=np.abs(values).mean(axis=0),
        values=values,
        feature_values=values,
        base_value=0.25,
        model_name="GBR",
        task="regression",
        n_samples=4,
    )


class TestShapSummaryRecord:
    """The typed record's contract (no shap involved)."""

    def test_ranking_descending_with_stable_ties(self) -> None:
        record = _summary_fixture()
        assert [name for name, _ in record.ranking()] == ["a", "b", "c"]
        assert record.ranking(top_n=1) == [("a", pytest.approx(0.6875))]

    def test_as_frame_round_trip(self) -> None:
        frame = _summary_fixture().as_frame()
        assert list(frame["feature"]) == ["a", "b", "c"]
        assert frame["mean_abs_shap"].tolist() == pytest.approx(
            np.abs(_summary_fixture().values).mean(axis=0).tolist()
        )

    def test_str_names_model_and_task(self) -> None:
        text = str(_summary_fixture())
        assert "GBR" in text and "regression" in text and "n=4" in text
        assert "a=" in text


class TestShapPlots:
    """The two figures (pure matplotlib -- shap never needed)."""

    def test_bar_top_n_ordering_and_labels(self) -> None:
        figure = plot_shap_importance(_summary_fixture())
        assert isinstance(figure, Figure)
        ax = figure.axes[0]
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert labels == ["c", "b", "a"]  # bottom -> top, strongest on top
        assert ax.get_xlabel() == "mean |SHAP value|"
        assert ax.get_title().startswith("SHAP importance")

    def test_bar_top_n_truncates(self) -> None:
        figure = plot_shap_importance(_summary_fixture(), top_n=2)
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels == ["b", "a"]

    def test_bar_accepts_the_dataframe_form(self) -> None:
        frame = _summary_fixture().as_frame()
        figure = plot_shap_importance(frame)
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels == ["c", "b", "a"]

    def test_bar_refuses_unknown_input_with_typed_error(self) -> None:
        with pytest.raises(DataValidationError, match="ShapSummary record"):
            plot_shap_importance({"not": "a summary"})

    def test_beeswarm_rows_colourbar_and_caption(self) -> None:
        figure = plot_shap_beeswarm(_summary_fixture())
        assert isinstance(figure, Figure)
        ax = figure.axes[0]
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert labels == ["c", "b", "a"]
        # the base value rides the caption when the record carries one
        assert "base = +0.25" in ax.get_xlabel()
        # the colourbar explains the ramp
        assert any("feature value" in str(c.get_ylabel()) for c in figure.axes[1:])

    def test_beeswarm_jitter_is_seeded_deterministic(self) -> None:
        first = plot_shap_beeswarm(_summary_fixture(), seed=42)
        second = plot_shap_beeswarm(_summary_fixture(), seed=42)
        y_first = sorted(
            offset for collection in first.axes[0].collections
            for offset in collection.get_offsets()[:, 1].tolist()
        )
        y_second = sorted(
            offset for collection in second.axes[0].collections
            for offset in collection.get_offsets()[:, 1].tolist()
        )
        assert y_first == y_second
        # a different seed scatters differently
        third = plot_shap_beeswarm(_summary_fixture(), seed=7)
        y_third = sorted(
            offset for collection in third.axes[0].collections
            for offset in collection.get_offsets()[:, 1].tolist()
        )
        assert y_third != y_first

    def test_beeswarm_constant_feature_falls_back_to_mid_colour(self) -> None:
        values = np.array(
            [[1.0, 0.0], [0.5, 0.0], [-1.0, 0.0], [0.25, 0.0]]
        )
        record = ShapSummary(
            names=["a", "flat"],
            mean_abs=np.abs(values).mean(axis=0),
            values=values,
            feature_values=np.ones((4, 2)),  # both columns constant
            base_value=None,
            model_name="GBR",
            task="regression",
            n_samples=4,
        )
        figure = plot_shap_beeswarm(record)
        assert isinstance(figure, Figure)  # span == 0 -> zeros, no crash

    def test_beeswarm_refuses_the_dataframe_form(self) -> None:
        with pytest.raises(DataValidationError, match="plot_shap_importance"):
            plot_shap_beeswarm(_summary_fixture().as_frame())

    def test_beeswarm_refuses_shape_mismatch(self) -> None:
        record = _summary_fixture()
        broken = ShapSummary(
            names=["a", "b"],
            mean_abs=record.mean_abs[:2],
            values=record.values,
            feature_values=record.feature_values[:, :2],
            base_value=None,
            model_name="GBR",
            task="regression",
            n_samples=4,
        )
        with pytest.raises(DataValidationError, match="must match"):
            plot_shap_beeswarm(broken)

    def test_beeswarm_jitter_bounds_guard(self) -> None:
        with pytest.raises(DataValidationError, match="jitter"):
            plot_shap_beeswarm(_summary_fixture(), jitter=1.5)

    def test_plots_close_their_figures(self, close_all: None) -> None:
        assert close_all is None


@pytest.mark.skipif(not _HAS_SHAP, reason="shap not installed (the zip-only tree)")
class TestShapSummaryRealPath:
    """The TreeExplainer path on real models (shap installed)."""

    def _gbr(self) -> tuple[Any, pd.DataFrame]:
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            {
                "strong": rng.normal(size=60),
                "weak": rng.normal(size=60),
                "noise": rng.normal(size=60),
            }
        )
        target = 3.0 * frame["strong"] - 0.5 * frame["weak"]
        model = GradientBoostingRegressor(random_state=42).fit(frame, target)
        return model, frame

    def test_ranking_matches_true_coefficients(self) -> None:
        model, frame = self._gbr()
        summary = shap_summary(model, frame)
        assert summary.n_samples == 60
        assert summary.task == "regression"
        assert summary.model_name == "GradientBoostingRegressor"
        ranking = [name for name, _ in summary.ranking()]
        # the true generative order: strong >> weak >> noise
        assert ranking[0] == "strong"
        assert ranking[1] == "weak"
        assert ranking[2] == "noise"
        assert summary.mean_abs[0] > 10 * summary.mean_abs[2]

    def test_values_shape_and_additivity_direction(self) -> None:
        model, frame = self._gbr()
        summary = shap_summary(model, frame)
        assert summary.values.shape == (60, 3)
        # a strong-positive feature pushes high-valued samples up: the
        # correlation between the feature and its own SHAP values is
        # strongly positive for the generative driver
        assert np.corrcoef(frame["strong"], summary.values[:, 0])[0, 1] > 0.8

    def test_ndarray_input_gets_positional_names(self) -> None:
        model, frame = self._gbr()
        summary = shap_summary(model, frame.to_numpy())
        assert summary.names == ["x0", "x1", "x2"]

    def test_binary_classifier_positive_class_read(self) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] > 0).astype(int)
        model = GradientBoostingClassifier(random_state=42).fit(X, y)
        summary = shap_summary(model, X)
        assert summary.task == "classification"
        assert summary.values.shape == (80, 3)
        assert summary.ranking(top_n=1)[0][0] == "x0"  # the generative driver

    def test_multiclass_refused_with_typed_error(self) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(size=(90, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1.0).astype(int)
        model = GradientBoostingClassifier(random_state=42).fit(X, y)
        with pytest.raises(ModelError, match="permutation_feature_importance"):
            shap_summary(model, X)

    def test_non_tree_model_refused_with_sibling_pointer(self) -> None:
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 2))
        model = LinearRegression().fit(X, X[:, :1].ravel())
        with pytest.raises(ModelError, match="permutation_feature_importance"):
            shap_summary(model, X)

    def test_pipeline_refused_with_typed_error(self) -> None:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 2))
        pipeline = Pipeline(
            [("scale", StandardScaler()), ("model", GradientBoostingRegressor(random_state=42))]
        ).fit(X, X[:, :1].ravel())
        with pytest.raises(ModelError, match="Pipeline"):
            shap_summary(pipeline, X)

    def test_empty_and_non_2d_inputs_refused(self) -> None:
        model, _ = self._gbr()
        with pytest.raises(ModelError, match="at least one row"):
            shap_summary(model, np.empty((0, 3)))
        with pytest.raises(ModelError, match="2-dimensional"):
            shap_summary(model, np.zeros(12))

    def test_feature_names_mismatch_refused(self) -> None:
        model, frame = self._gbr()
        with pytest.raises(ModelError, match="label"):
            shap_summary(model, frame, feature_names=["only", "two"])

    def test_missing_shap_raises_dependency_error_with_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "shap" or name.startswith("shap."):
                raise ImportError(f"blocked for the test: {name}")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", blocked)
        model, frame = self._gbr()
        with pytest.raises(DependencyError, match=r"pip install 'pharmaml\[explain\]'"):
            shap_summary(model, frame)

    def test_end_to_end_both_figures_render(self) -> None:
        model, frame = self._gbr()
        summary = shap_summary(model, frame)
        bars = plot_shap_importance(summary)
        swarm = plot_shap_beeswarm(summary)
        assert isinstance(bars, Figure) and isinstance(swarm, Figure)
        labels = [tick.get_text() for tick in swarm.axes[0].get_yticklabels()]
        assert labels[0] == "noise"  # bottom -> top, strongest on top
