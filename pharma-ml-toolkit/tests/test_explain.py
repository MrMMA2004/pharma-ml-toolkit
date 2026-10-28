"""Tests for :mod:`pharmaml.ml.explain` and its viz twins.

Round 100x-154: the LAST unabsorbed ``D2.py`` gap (its lines 472-528
-- the ``TreeExplainer`` beeswarm + the top-20 bar the upload drew for
its stacking pipeline), shipped as a first-class optional-gated
surface.

Round 100x-160 (round two): the SHAP repo's remaining surfaces --
``shap_interactions`` (the exact pairwise interaction cube, its guard
chain driven WITHOUT shap through a stub ``sys.modules`` entry),
``group_difference`` (pure-numpy cohort differences + the seeded
bootstrap CI), and the five new figures (waterfall, dependence with
the auto-interaction pick, heatmap, force, group difference).

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

import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure

from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.ml.explain import (
    ShapInteractions,
    ShapSummary,
    group_difference,
    shap_interactions,
    shap_summary,
)
from pharmaml.viz import (
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_group_difference,
    plot_shap_heatmap,
    plot_shap_importance,
    plot_shap_waterfall,
)

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


def _interactions_fixture() -> ShapInteractions:
    """A hand-built interaction cube (no shap needed): 3 samples, 3 features."""
    cube = np.zeros((3, 3, 3))
    cube[:, 0, 0] = 0.9     # a's main effect
    cube[:, 1, 1] = 0.3     # b's main effect
    cube[:, 2, 2] = 0.1     # c's main effect
    cube[:, 0, 1] = cube[:, 1, 0] = 0.6   # the a-b interaction
    cube[:, 0, 2] = cube[:, 2, 0] = 0.2
    cube[:, 1, 2] = cube[:, 2, 1] = 0.05
    return ShapInteractions(
        names=["a", "b", "c"],
        values=cube,
        feature_values=np.arange(9.0).reshape(3, 3),
        model_name="GBR",
        n_samples=3,
    )


def _difference_summary_fixture() -> ShapSummary:
    """A 4x2 record whose cohort difference is known exactly (2 vs 2)."""
    values = np.array(
        [
            [1.0, 0.25],
            [0.5, -0.25],
            [-0.5, 0.5],
            [-1.0, 0.0],
        ]
    )
    return ShapSummary(
        names=["a", "b"],
        mean_abs=np.abs(values).mean(axis=0),
        values=values,
        feature_values=values,
        base_value=None,
        model_name="GBR",
        task="regression",
        n_samples=4,
    )


def _local_fixture() -> ShapSummary:
    """A 2-row record for the local figures (waterfall / force)."""
    values = np.array([[1.25, -0.5, 0.25], [0.1, 0.2, -0.1]])
    return ShapSummary(
        names=["a", "b", "c"],
        mean_abs=np.abs(values).mean(axis=0),
        values=values,
        feature_values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        base_value=0.25,
        model_name="GBR",
        task="regression",
        n_samples=2,
    )


def _dependence_fixture() -> ShapSummary:
    """An 8-row record where b visibly drives a's SHAP value."""
    a = np.arange(8.0)
    b = np.tile([0.0, 1.0], 4)
    values = np.column_stack([b.copy(), np.zeros(8)])
    return ShapSummary(
        names=["a", "b"],
        mean_abs=np.abs(values).mean(axis=0),
        values=values,
        feature_values=np.column_stack([a, b]),
        base_value=None,
        model_name="GBR",
        task="regression",
        n_samples=8,
    )


def _stub_shap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cube: np.ndarray | None = None,
    cubes: Any = None,
) -> types.ModuleType:
    """Install a fake ``shap`` module in ``sys.modules``.

    The round-two guard tests drive ``shap_interactions`` WITHOUT the
    real package: the stub's ``TreeExplainer.shap_interaction_values``
    returns ``cube`` (or ``cubes`` verbatim -- a list or a 4-D array,
    the two multiclass forms), so every branch of the guard chain runs
    on machines where shap is absent too.
    """
    module = types.ModuleType("shap")

    class _StubExplainer:
        def __init__(self, model: object) -> None:
            self.model = model

        def shap_interaction_values(self, matrix: np.ndarray) -> Any:
            if cubes is not None:
                return cubes
            if cube is not None:
                return cube
            return np.zeros(
                (matrix.shape[0], matrix.shape[1], matrix.shape[1]), dtype=float
            )

    module.TreeExplainer = _StubExplainer
    monkeypatch.setitem(sys.modules, "shap", module)
    return module


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


class TestShapInteractions:
    """The interaction cube's record contract + guard chain (100x-160)."""

    def test_as_frame_row_count_is_upper_triangle(self) -> None:
        frame = _interactions_fixture().as_frame()
        # 3 features -> 3*4/2 = 6 upper-triangle rows, diagonal included
        assert len(frame) == 6
        assert list(frame.columns) == [
            "feature_a",
            "feature_b",
            "mean_abs_interaction",
        ]
        assert list(frame["feature_a"]) == ["a", "a", "a", "b", "b", "c"]

    def test_ranking_descending_over_upper_triangle(self) -> None:
        ranking = _interactions_fixture().ranking()
        assert [(a, b) for a, b, _ in ranking] == [
            ("a", "a"),
            ("a", "b"),
            ("b", "b"),
            ("a", "c"),
            ("c", "c"),
            ("b", "c"),
        ]
        assert _interactions_fixture().ranking(top_n=2) == [
            ("a", "a", pytest.approx(0.9)),
            ("a", "b", pytest.approx(0.6)),
        ]

    def test_str_names_model_and_pairs(self) -> None:
        text = str(_interactions_fixture())
        assert "GBR" in text and "n=3" in text and "a x a=0.9000" in text

    def test_dependency_error_when_shap_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "shap" or name.startswith("shap."):
                raise ImportError(f"blocked for the test: {name}")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(DependencyError, match=r"pip install 'pharmaml\[explain\]'"):
            shap_interactions(object(), np.zeros((3, 2)))

    def test_stub_success_path_shapes_the_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        _stub_shap(monkeypatch, cube=_interactions_fixture().values)
        result = shap_interactions(
            GradientBoostingRegressor(random_state=42), np.zeros((3, 3))
        )
        assert result.values.shape == (3, 3, 3)
        assert result.names == ["x0", "x1", "x2"]
        assert result.n_samples == 3
        assert result.model_name == "GradientBoostingRegressor"
        assert np.allclose(result.values, result.values.transpose(0, 2, 1))

    def test_stub_dataframe_input_keeps_column_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        _stub_shap(monkeypatch)
        frame = pd.DataFrame(np.zeros((3, 2)), columns=["pKa", "logP"])
        result = shap_interactions(GradientBoostingRegressor(random_state=42), frame)
        assert result.names == ["pKa", "logP"]
        assert result.values.shape == (3, 2, 2)

    def test_pipeline_refused_with_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        _stub_shap(monkeypatch)
        pipeline = Pipeline([("scale", StandardScaler())])
        with pytest.raises(ModelError, match="Pipeline"):
            shap_interactions(pipeline, np.zeros((4, 2)))

    def test_unfitted_estimator_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_shap(monkeypatch)
        with pytest.raises(ModelError, match="fitted estimator"):
            shap_interactions(object(), np.zeros((4, 2)))

    def test_multiclass_cube_refused_both_forms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(random_state=42)
        matrix = np.zeros((4, 3))
        # the trailing-class-axis form: (n, f, f, classes)
        _stub_shap(monkeypatch, cubes=np.zeros((4, 3, 3, 3)))
        with pytest.raises(ModelError, match="3 classes"):
            shap_interactions(model, matrix)
        # the legacy per-class list form: one cube per class
        _stub_shap(
            monkeypatch,
            cubes=[np.zeros((4, 3, 3)), np.zeros((4, 3, 3)), np.zeros((4, 3, 3))],
        )
        with pytest.raises(ModelError, match="3 classes"):
            shap_interactions(model, matrix)

    def test_shape_mismatch_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        _stub_shap(monkeypatch, cube=np.zeros((4, 3, 2)))  # not (n, f, f)
        with pytest.raises(ModelError, match="unexpected SHAP interaction values"):
            shap_interactions(GradientBoostingRegressor(random_state=42), np.zeros((4, 3)))

    def test_input_guards_with_stub(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        _stub_shap(monkeypatch)
        model = GradientBoostingRegressor(random_state=42)
        with pytest.raises(ModelError, match="2-dimensional"):
            shap_interactions(model, np.zeros(12))
        with pytest.raises(ModelError, match="at least one row"):
            shap_interactions(model, np.empty((0, 3)))
        with pytest.raises(ModelError, match="label"):
            shap_interactions(model, np.zeros((4, 3)), feature_names=["only", "two"])


class TestGroupDifference:
    """The pure-numpy cohort difference + bootstrap CI (100x-160)."""

    def test_exact_differences_on_crafted_cohorts(self) -> None:
        record = _difference_summary_fixture()
        mask = np.array([True, True, False, False])
        result = group_difference(record, mask, label="high potency")
        # group means [0.75, 0.0], rest means [-0.75, 0.25] -- exact
        assert result.differences.tolist() == pytest.approx([1.5, -0.25])
        assert (result.n_group, result.n_rest) == (2, 2)
        assert result.names == ["a", "b"]

    def test_bootstrap_ci_brackets_the_point_estimate(self) -> None:
        record = _difference_summary_fixture()
        result = group_difference(record, [True, True, False, False])
        assert (result.low <= result.differences).all()
        assert (result.differences <= result.high).all()

    def test_bootstrap_is_seeded_deterministic(self) -> None:
        # a larger record: the 2-vs-2 fixture's percentiles saturate at
        # the extremes, so seed sensitivity needs a real spread.
        rng = np.random.default_rng(0)
        values = rng.normal(size=(40, 3))
        record = ShapSummary(
            names=["a", "b", "c"],
            mean_abs=np.abs(values).mean(axis=0),
            values=values,
            feature_values=values,
            base_value=None,
            model_name="GBR",
            task="regression",
            n_samples=40,
        )
        mask = values[:, 0] > 0.0
        first = group_difference(record, mask)
        second = group_difference(record, mask)
        assert first.low.tolist() == second.low.tolist()
        assert first.high.tolist() == second.high.tolist()
        other = group_difference(record, mask, seed=7)
        assert other.low.tolist() != first.low.tolist()

    def test_confidence_bounds_widen_or_shrink(self) -> None:
        record = _difference_summary_fixture()
        mask = [True, True, False, False]
        narrow = group_difference(record, mask, confidence=50)
        wide = group_difference(record, mask, confidence=99)
        assert (narrow.high - narrow.low <= wide.high - wide.low).all()

    def test_ranking_as_frame_and_str(self) -> None:
        result = group_difference(
            _difference_summary_fixture(), [True, True, False, False]
        )
        assert result.ranking() == [("a", pytest.approx(1.5)), ("b", pytest.approx(-0.25))]
        assert result.ranking(top_n=1) == [("a", pytest.approx(1.5))]
        frame = result.as_frame()
        assert list(frame.columns) == ["feature", "difference", "low", "high"]
        assert list(frame["feature"]) == ["a", "b"]
        assert frame["difference"].tolist() == pytest.approx([1.5, -0.25])
        text = str(result)
        assert "a=+1.5000" in text

    def test_label_round_trip(self) -> None:
        result = group_difference(
            _difference_summary_fixture(),
            [True, True, False, False],
            label="high potency",
        )
        assert result.label == "high potency"
        assert "high potency" in str(result)

    def test_series_mask_accepted(self) -> None:
        mask = pd.Series([True, True, False, False])
        result = group_difference(_difference_summary_fixture(), mask)
        assert result.differences.tolist() == pytest.approx([1.5, -0.25])

    def test_wrong_length_mask_refused(self) -> None:
        with pytest.raises(ModelError, match="entries"):
            group_difference(_difference_summary_fixture(), [True, False])

    def test_all_true_and_all_false_masks_refused(self) -> None:
        with pytest.raises(ModelError, match="both cohorts need at least one row"):
            group_difference(_difference_summary_fixture(), [True] * 4)
        with pytest.raises(ModelError, match="both cohorts need at least one row"):
            group_difference(_difference_summary_fixture(), [False] * 4)

    def test_non_boolean_mask_refused(self) -> None:
        with pytest.raises(ModelError, match="boolean"):
            group_difference(_difference_summary_fixture(), [1, 0, 1, 0])

    def test_resample_and_confidence_guards(self) -> None:
        record = _difference_summary_fixture()
        mask = [True, True, False, False]
        with pytest.raises(ModelError, match="n_resamples"):
            group_difference(record, mask, n_resamples=0)
        with pytest.raises(ModelError, match="confidence"):
            group_difference(record, mask, confidence=101)
        with pytest.raises(ModelError, match="confidence"):
            group_difference(record, mask, confidence=0.0)

    def test_non_record_summary_refused(self) -> None:
        with pytest.raises(ModelError, match="ShapSummary record"):
            group_difference({"not": "a summary"}, [True, False])


class TestShapPlotsRoundTwo:
    """The five round-43 figures (pure matplotlib -- shap never needed)."""

    def test_waterfall_tick_labels_and_end_lines(self, close_all: None) -> None:
        figure = plot_shap_waterfall(_local_fixture())
        assert isinstance(figure, Figure)
        ax = figure.axes[0]
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        # largest |value| at the top; the sample's feature values ride
        # the labels (shap's show_data convention)
        assert labels == ["c = 3.00", "b = 2.00", "a = 1.00"]
        assert ax.get_xlabel() == "SHAP contribution (model output units)"
        ends = {
            round(float(line.get_xdata()[0]), 6)
            for line in ax.lines
            if len(line.get_xdata()) == 2
        }
        assert 0.25 in ends       # the base value
        assert 1.25 in ends       # the prediction f(x) = 0.25 + 1.0

    def test_waterfall_remainder_bar_when_max_display_shortens(
        self, close_all: None
    ) -> None:
        figure = plot_shap_waterfall(_local_fixture(), max_display=2)
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels == ["Σ remaining 1 feature", "b = 2.00", "a = 1.00"]

    def test_waterfall_refuses_summary_without_base_value(self) -> None:
        record = _local_fixture()
        broken = ShapSummary(
            names=record.names,
            mean_abs=record.mean_abs,
            values=record.values,
            feature_values=record.feature_values,
            base_value=None,
            model_name="GBR",
            task="regression",
            n_samples=2,
        )
        with pytest.raises(DataValidationError, match="carries none"):
            plot_shap_waterfall(broken)

    def test_waterfall_sample_out_of_range_refused(self) -> None:
        with pytest.raises(DataValidationError, match="out of range"):
            plot_shap_waterfall(_local_fixture(), sample=5)

    def test_waterfall_refuses_the_dataframe_form(self) -> None:
        with pytest.raises(DataValidationError, match="ShapSummary RECORD"):
            plot_shap_waterfall(_local_fixture().as_frame())

    def test_dependence_auto_picks_the_interacting_feature(
        self, close_all: None
    ) -> None:
        figure = plot_shap_dependence(_dependence_fixture(), "a")
        assert isinstance(figure, Figure)
        ax = figure.axes[0]
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("a", "SHAP value")
        # the colorbar names the auto-picked colour: b drives a's SHAP
        assert figure.axes[1].get_ylabel() == "b value"

    def test_dependence_explicit_color_by(self, close_all: None) -> None:
        figure = plot_shap_dependence(_dependence_fixture(), "a", color_by="b")
        assert figure.axes[1].get_ylabel() == "b value"

    def test_dependence_refuses_color_by_equal_to_feature(self) -> None:
        with pytest.raises(DataValidationError, match="differ from feature"):
            plot_shap_dependence(_dependence_fixture(), "a", color_by="a")

    def test_dependence_refuses_unknown_names(self) -> None:
        with pytest.raises(DataValidationError, match="not one of the summary's"):
            plot_shap_dependence(_dependence_fixture(), "zzz")
        with pytest.raises(DataValidationError, match="not one of the summary's"):
            plot_shap_dependence(_dependence_fixture(), "a", color_by="zzz")

    def test_dependence_plain_fallback_when_candidate_is_constant(
        self, close_all: None
    ) -> None:
        values = np.array([[1.0, 0.0], [0.5, 0.0], [-1.0, 0.0], [0.25, 0.0]])
        record = ShapSummary(
            names=["a", "flat"],
            mean_abs=np.abs(values).mean(axis=0),
            values=values,
            feature_values=np.ones((4, 2)),  # the only candidate is constant
            base_value=None,
            model_name="GBR",
            task="regression",
            n_samples=4,
        )
        figure = plot_shap_dependence(record, "a")
        # no honest colour exists -> plain scatter, honest title note,
        # and no colorbar axes
        assert len(figure.axes) == 1
        assert "no interacting colour found" in figure.axes[0].get_title()

    def test_dependence_explicit_none_stays_plain(self, close_all: None) -> None:
        figure = plot_shap_dependence(
            _dependence_fixture(), "a", color_by="none"
        )
        assert len(figure.axes) == 1
        assert "no interacting colour found" not in figure.axes[0].get_title()

    def test_dependence_jitter_bounds_guard(self) -> None:
        with pytest.raises(DataValidationError, match="jitter"):
            plot_shap_dependence(_dependence_fixture(), "a", jitter=1.5)

    def test_heatmap_feature_order_and_two_axes(self, close_all: None) -> None:
        figure = plot_shap_heatmap(_summary_fixture())
        assert isinstance(figure, Figure)
        heat = figure.axes[1]  # axes[0] is the mean-|SHAP| bar panel
        labels = [tick.get_text() for tick in heat.get_xticklabels()]
        assert labels == ["a", "b", "c"]  # strongest first, left to right
        # the multi-panel contract: the importance bar + the heatmap
        # (the colorbar axes rides after them)
        assert len(figure.axes) >= 2
        assert any(len(ax.images) == 1 for ax in figure.axes)
        assert any(len(ax.patches) > 0 for ax in figure.axes)

    def test_heatmap_max_display_truncates_columns(self, close_all: None) -> None:
        figure = plot_shap_heatmap(_summary_fixture(), max_display=2)
        labels = [
            tick.get_text() for tick in figure.axes[1].get_xticklabels()
        ]
        assert labels == ["a", "b"]

    def test_heatmap_single_sample_keeps_input_order(
        self, close_all: None
    ) -> None:
        values = np.array([[2.0, -0.5, 0.1]])
        record = ShapSummary(
            names=["a", "b", "c"],
            mean_abs=np.abs(values).mean(axis=0),
            values=values,
            feature_values=values,
            base_value=None,
            model_name="GBR",
            task="regression",
            n_samples=1,
        )
        figure = plot_shap_heatmap(record)
        rows = [tick.get_text() for tick in figure.axes[1].get_yticklabels()]
        assert rows == ["0"]  # linkage needs >= 2 rows: input order kept

    def test_heatmap_refuses_the_dataframe_form(self) -> None:
        with pytest.raises(DataValidationError, match="ShapSummary RECORD"):
            plot_shap_heatmap(_summary_fixture().as_frame())

    def test_force_end_annotations_and_row(self, close_all: None) -> None:
        figure = plot_shap_force(_local_fixture())
        assert isinstance(figure, Figure)
        ax = figure.axes[0]
        texts = [annotation.get_text() for annotation in ax.texts]
        assert texts == ["base = +0.25", "f(x) = +1.25"]
        assert ax.get_xlabel() == "model output"
        assert ax.get_yticklabels() == []  # a single row: no y ticks

    def test_force_segments_and_remainder(self, close_all: None) -> None:
        figure = plot_shap_force(_local_fixture(), max_display=2)
        ax = figure.axes[0]
        assert len(ax.patches) == 3  # two kept features + the summed rest
        assert [annotation.get_text() for annotation in ax.texts] == [
            "base = +0.25",
            "f(x) = +1.25",  # the prediction folds in the remainder too
        ]

    def test_force_guards(self) -> None:
        with pytest.raises(DataValidationError, match="carries none"):
            plot_shap_force(
                ShapSummary(
                    names=["a"],
                    mean_abs=np.array([1.0]),
                    values=np.zeros((2, 1)),
                    feature_values=np.zeros((2, 1)),
                    base_value=None,
                    model_name="GBR",
                    task="regression",
                    n_samples=2,
                )
            )
        with pytest.raises(DataValidationError, match="out of range"):
            plot_shap_force(_local_fixture(), sample=9)
        with pytest.raises(DataValidationError, match="top_n"):
            plot_shap_force(_local_fixture(), max_display=0)

    def test_group_difference_tick_order_and_xlabel(self, close_all: None) -> None:
        record = group_difference(
            _difference_summary_fixture(), [True, True, False, False], label="cohort"
        )
        figure = plot_shap_group_difference(record)
        assert isinstance(figure, Figure)
        ax = figure.axes[0]
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert labels == ["b", "a"]  # bottom -> top, strongest on top
        assert ax.get_xlabel() == "mean SHAP difference (group - rest)"
        # the label + cohort counts ride the title
        title = ax.get_title()
        assert "cohort" in title and "n=2" in title

    def test_group_difference_whiskers_present(self, close_all: None) -> None:
        record = group_difference(_difference_summary_fixture(), [True, True, False, False])
        figure = plot_shap_group_difference(record)
        assert any(
            isinstance(container, ErrorbarContainer)
            for container in figure.axes[0].containers
        )

    def test_group_difference_accepts_the_dataframe_form(
        self, close_all: None
    ) -> None:
        record = group_difference(_difference_summary_fixture(), [True, True, False, False])
        figure = plot_shap_group_difference(record.as_frame())
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels == ["b", "a"]

    def test_group_difference_refuses_unknown_input(self) -> None:
        with pytest.raises(DataValidationError, match="GroupDifference record"):
            plot_shap_group_difference({"not": "a record"})


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

    def test_interactions_cube_on_real_model(self) -> None:
        model, frame = self._gbr()
        result = shap_interactions(model, frame)
        assert result.values.shape == (60, 3, 3)
        assert result.n_samples == 60
        assert result.model_name == "GradientBoostingRegressor"
        assert result.names == ["strong", "weak", "noise"]
        # the pairwise cube is symmetric by construction
        assert np.allclose(result.values, result.values.transpose(0, 2, 1))
        # the generative driver's main effect dwarfs the noise pairs
        assert np.abs(result.values[:, 0, 0]).mean() > np.abs(
            result.values[:, 2, 2]
        ).mean()

    def test_interactions_end_to_end_figures_render(self) -> None:
        model, frame = self._gbr()
        summary = shap_summary(model, frame)
        heat = plot_shap_heatmap(summary)
        fall = plot_shap_waterfall(summary)
        push = plot_shap_force(summary)
        scatter = plot_shap_dependence(summary, "strong")
        cohort = group_difference(summary, frame["weak"].to_numpy() > 0.0)
        bars = plot_shap_group_difference(cohort)
        assert all(
            isinstance(figure, Figure)
            for figure in (heat, fall, push, scatter, bars)
        )
