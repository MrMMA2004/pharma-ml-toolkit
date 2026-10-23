"""Unit tests for applicability-domain analysis (the 2026-10 upload)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.applicability import (
    ApplicabilityDomainResult,
    BootstrapUncertainty,
    applicability_domain,
    bootstrap_uncertainty,
    knn_applicability_domain,
)

try:  # matplotlib gates only the figure test
    import matplotlib  # noqa: F401

    HAS_MPL = True
except ImportError:  # pragma: no cover - exercised on reports-less CI
    HAS_MPL = False


def _ridge_data(n: int = 60) -> tuple[pd.DataFrame, pd.Series, Ridge]:
    x = pd.DataFrame({"a": np.linspace(0.0, 10.0, n)})
    y = pd.Series(2.0 * x["a"] + 1.0)
    return x, y, Ridge().fit(x, y)


class TestKnnApplicabilityDomain:
    def test_interpolation_inside_extrapolation_outside(self) -> None:
        x, _y, _model = _ridge_data()
        result = knn_applicability_domain(x, x.iloc[::4], k=5)
        assert result.inside_ad.all()
        far = pd.DataFrame({"a": [1000.0]})
        out = knn_applicability_domain(x, far, k=5)
        assert not out.inside_ad[0]
        assert float(out.test_knn_mean[0]) > out.distance_threshold

    def test_training_distances_exclude_self_match(self) -> None:
        # a uniform grid: including the zero self-distance would deflate
        # every interior row's mean to 1.0 * spacing; excluding it the 5
        # nearest sit at ±d, ±d, ±2d, ±2d and one ±3d -> exactly 1.8 * d
        grid = np.linspace(0.0, 10.0, 101)
        x = pd.DataFrame({"a": grid})
        result = knn_applicability_domain(x, x.iloc[:1], k=5)
        expected = 1.8 * (10.0 / 100.0) / float(np.std(grid))
        assert result.train_knn_mean[50] == pytest.approx(expected, rel=1e-9)

    def test_threshold_is_percentile_of_train_distances(self) -> None:
        x, _y, _model = _ridge_data()
        result = knn_applicability_domain(x, x.iloc[:5], k=5, percentile=90.0)
        assert result.distance_threshold == pytest.approx(
            float(np.percentile(result.train_knn_mean, 90.0))
        )

    def test_numpy_arrays_accepted(self) -> None:
        train = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
        test = np.array([[0.5], [50.0]])
        result = knn_applicability_domain(train, test, k=4)
        assert result.inside_ad[0] and not result.inside_ad[1]
        assert result.feature_names == ["f0"]

    def test_feature_names_from_dataframe(self) -> None:
        x = pd.DataFrame({"molwt": np.arange(30.0), "tpsa": np.arange(30.0) * 2})
        result = knn_applicability_domain(x, x.iloc[:2], k=3)
        assert result.feature_names == ["molwt", "tpsa"]

    def test_bad_k(self) -> None:
        x, _y, _model = _ridge_data()
        with pytest.raises(ModelError, match="k must be >= 1"):
            knn_applicability_domain(x, x.iloc[:1], k=0)

    def test_k_needs_headroom_for_self_exclusion(self) -> None:
        x, _y, _model = _ridge_data(20)
        with pytest.raises(ModelError, match="self-distance"):
            knn_applicability_domain(x, x.iloc[:1], k=20)

    def test_bad_percentile(self) -> None:
        x, _y, _model = _ridge_data()
        for bad in (0.0, 100.5, -1.0):
            with pytest.raises(ModelError, match="percentile must be in"):
                knn_applicability_domain(x, x.iloc[:1], percentile=bad)

    def test_feature_count_mismatch(self) -> None:
        two_features = pd.DataFrame(
            {"a": np.linspace(0.0, 1.0, 20), "b": np.linspace(1.0, 2.0, 20)}
        )
        with pytest.raises(ModelError, match="descriptor spaces must match"):
            knn_applicability_domain(two_features, pd.DataFrame({"a": [0.5]}))

    def test_nan_descriptors_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="X_train contains 1 NaN"):
            knn_applicability_domain(
                pd.DataFrame({"a": [0.0, np.nan, 2.0]}),
                pd.DataFrame({"a": [1.0]}),
            )

    def test_inf_descriptors_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="infinite"):
            knn_applicability_domain(
                pd.DataFrame({"a": [0.0, np.inf, 2.0]}),
                pd.DataFrame({"a": [1.0]}),
            )

    def test_non_numeric_columns_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="non-numeric column"):
            knn_applicability_domain(
                pd.DataFrame({"a": [0.0, 1.0], "smiles": ["C", "CC"]}),
                pd.DataFrame({"a": [0.5]}),
            )

    def test_empty_matrix_rejected(self) -> None:
        with pytest.raises(DataValidationError, match="empty"):
            knn_applicability_domain(
                pd.DataFrame({"a": []}, dtype=float),
                pd.DataFrame({"a": [1.0]}),
            )


class TestBootstrapUncertainty:
    def test_deterministic_across_runs(self) -> None:
        x, y, model = _ridge_data()
        first = bootstrap_uncertainty(model, x, y, x.iloc[::2], n_models=6)
        second = bootstrap_uncertainty(model, x, y, x.iloc[::2], n_models=6)
        np.testing.assert_array_equal(first.pred_mean, second.pred_mean)
        np.testing.assert_array_equal(first.pred_std, second.pred_std)

    def test_seeded_resamples_differ(self) -> None:
        x, y, model = _ridge_data()
        first = bootstrap_uncertainty(model, x, y, x.iloc[::2], n_models=6,
                                      random_state=0)
        second = bootstrap_uncertainty(model, x, y, x.iloc[::2], n_models=6,
                                       random_state=99)
        assert not np.allclose(first.pred_mean, second.pred_mean)

    def test_predictions_kind_for_regressor(self) -> None:
        x, y, model = _ridge_data()
        boot = bootstrap_uncertainty(model, x, y, x.iloc[:4], n_models=4)
        assert boot.kind == "predictions"
        assert boot.pred_mean.shape == (4,)
        assert (boot.pred_std >= 0.0).all()

    def test_positive_probability_kind_for_binary_classifier(self) -> None:
        xc = pd.DataFrame({"a": [0.0, 0.1, 0.2, 0.3, 9.7, 9.8, 9.9, 10.0]})
        yc = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        clf = LogisticRegression().fit(xc, yc)
        boot = bootstrap_uncertainty(clf, xc, yc, xc, n_models=5)
        assert boot.kind == "positive-class probability"
        # probabilities live in [0, 1]
        assert (boot.pred_mean >= 0.0).all() and (boot.pred_mean <= 1.0).all()

    def test_pipeline_model_cloned_whole(self) -> None:
        # a prep + estimator pipeline refits end-to-end on each resample
        x = pd.DataFrame({"a": np.linspace(0.0, 8.0, 40)})
        y = pd.Series(3.0 * x["a"])
        pipe = Pipeline([("scale", None), ("reg", Ridge())]).fit(x, y)
        boot = bootstrap_uncertainty(pipe, x, y, x.iloc[:5], n_models=3)
        assert boot.kind == "predictions"
        assert boot.pred_mean.shape == (5,)

    def test_n_models_validation(self) -> None:
        x, y, model = _ridge_data()
        with pytest.raises(ModelError, match="n_models must be >= 1"):
            bootstrap_uncertainty(model, x, y, x.iloc[:1], n_models=0)

    def test_y_row_count_validation(self) -> None:
        x, y, model = _ridge_data()
        with pytest.raises(ModelError, match="X_train has 60 rows"):
            bootstrap_uncertainty(model, x, y.iloc[:10], x.iloc[:1], n_models=2)

    def test_multiclass_proba_rejected(self) -> None:
        x = pd.DataFrame({"a": [0.0, 5.0, 10.0] * 4})
        y = pd.Series([0, 1, 2] * 4)
        clf = LogisticRegression().fit(x, y)
        with pytest.raises(ModelError, match="exposes 3 classes"):
            bootstrap_uncertainty(clf, x, y, x.iloc[:1], n_models=2)

    def test_label_predictor_without_proba_rejected(self) -> None:
        from sklearn.svm import SVC

        xc = pd.DataFrame({"a": [0.0, 0.1, 9.9, 10.0]})
        yc = pd.Series(["no", "no", "yes", "yes"])
        clf = SVC().fit(xc, yc)  # probability=False: predict_proba missing
        with pytest.raises(ModelError, match="predicts class labels"):
            bootstrap_uncertainty(clf, xc, yc, xc.iloc[:1], n_models=2)

    def test_progress_callback_fired_per_fit(self) -> None:
        x, y, model = _ridge_data()
        seen: list[tuple[int, int]] = []
        bootstrap_uncertainty(model, x, y, x.iloc[:2], n_models=5,
                              progress=lambda done, total: seen.append((done, total)))
        assert seen == [(i, 5) for i in range(1, 6)]

    def test_broken_progress_reporter_swallowed(self) -> None:
        x, y, model = _ridge_data()

        def broken(done: int, total: int) -> None:
            raise RuntimeError("reporter died")

        boot = bootstrap_uncertainty(model, x, y, x.iloc[:2], n_models=3,
                                     progress=broken)
        assert boot.pred_mean.shape == (2,)  # the run survived

    def test_non_numeric_matrix_rejected(self) -> None:
        _x, _y, model = _ridge_data()
        with pytest.raises(DataValidationError, match="non-numeric column"):
            bootstrap_uncertainty(
                model,
                pd.DataFrame({"a": [0.0, 1.0], "s": ["C", "CC"]}),
                pd.Series([1.0, 2.0]),
                pd.DataFrame({"a": [0.5]}),
                n_models=2,
            )


class TestApplicabilityDomain:
    def test_full_workflow_reliable_flags(self) -> None:
        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y, x.iloc[::4], k=5, n_models=8)
        assert result.n_models == 8
        assert result.reliable is not None
        np.testing.assert_array_equal(
            result.reliable, result.inside_ad & result.low_uncertainty
        )
        assert result.uncertainty_kind == "predictions"
        assert result.fit_seconds >= 0.0

    def test_uncertainty_false_skips_bootstrap(self) -> None:
        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y, x.iloc[:3], uncertainty=False)
        assert result.n_models is None
        assert result.pred_mean is None
        assert result.reliable is None
        assert result.fit_seconds == 0.0

    def test_extrapolation_is_unreliable(self) -> None:
        x, y, model = _ridge_data()
        far = pd.DataFrame({"a": [500.0]})
        result = applicability_domain(model, x, y, far, k=5, n_models=4)
        assert not result.inside_ad[0]
        assert not result.reliable[0]

    def test_threshold_from_train_side_std(self) -> None:
        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y, x.iloc[:6], k=5, n_models=6)
        assert result.uncertainty_threshold == pytest.approx(
            float(np.percentile(result.train_pred_std, 95.0))
        )

    def test_to_frame_reliability_table(self) -> None:
        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y, x.iloc[:4], k=5, n_models=4)
        frame = result.to_frame(names=["C", "CC", "CCC", "CCCC"])
        assert list(frame.columns) == [
            "name", "predicted", "prediction_sd", "knn_distance",
            "inside_ad", "low_uncertainty", "reliable",
        ]
        assert frame["name"].tolist() == ["C", "CC", "CCC", "CCCC"]
        assert frame["inside_ad"].dtype == bool
        assert frame["reliable"].tolist() == result.reliable.tolist()

    def test_to_frame_generic_names(self) -> None:
        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y, x.iloc[:2], uncertainty=False)
        frame = result.to_frame()
        assert frame["name"].tolist() == ["compound_0", "compound_1"]
        # uncertainty columns honestly degrade to NA
        assert frame["reliable"].isna().all()
        assert frame["prediction_sd"].isna().all()

    def test_to_frame_short_name_list_padded(self) -> None:
        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y, x.iloc[:3], uncertainty=False)
        frame = result.to_frame(names=["only-one"])
        assert frame["name"].tolist() == ["only-one", "compound_1", "compound_2"]

    def test_as_dict_summary(self) -> None:
        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y, x.iloc[:5], k=5, n_models=4)
        summary = result.as_dict()
        assert summary["k"] == 5
        assert summary["n_test"] == 5
        assert summary["n_models"] == 4
        assert summary["inside_ad"] == int(result.inside_ad.sum())
        assert summary["reliable"] == int(result.reliable.sum())

    def test_str_representation(self) -> None:
        x, y, model = _ridge_data()
        text = str(applicability_domain(model, x, y, x.iloc[:2], k=5, n_models=2))
        assert "inside_ad=" in text and "reliable=" in text
        quick = str(knn_applicability_domain(x, x.iloc[:2], k=5))
        assert "inside_ad=" in quick and "reliable" not in quick

    def test_not_a_model_rejected(self) -> None:
        x, y, _model = _ridge_data()
        with pytest.raises(ModelError, match="fitted estimator with predict"):
            applicability_domain("nope", x, y, x.iloc[:1], n_models=2)

    def test_result_dataclass_defaults(self) -> None:
        # the distance half alone leaves the bootstrap half unset
        result = ApplicabilityDomainResult(
            k=3, percentile=95.0, distance_threshold=1.0,
            train_knn_mean=np.zeros(4), test_knn_mean=np.zeros(2),
            inside_ad=np.array([True, False]), feature_names=["a"],
        )
        assert result.n_models is None and result.reliable is None
        summary = result.as_dict()
        assert summary["low_uncertainty"] is None and summary["reliable"] is None

    def test_bootstrap_uncertainty_dataclass(self) -> None:
        boot = BootstrapUncertainty(
            pred_mean=np.zeros(2), pred_std=np.ones(2),
            train_pred_std=np.ones(5), n_models=3, kind="predictions",
        )
        assert boot.fit_seconds == 0.0


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
class TestApplicabilityDomainPlot:
    def test_two_panels_and_labels(self) -> None:
        from pharmaml.viz import close_figure
        from pharmaml.viz.ml_plots import plot_applicability_domain

        x, y, model = _ridge_data()
        result = applicability_domain(model, x, y,
                                      pd.DataFrame({"a": [0.5, 100.0]}),
                                      n_models=4)
        fig = plot_applicability_domain(result, y_label="predicted pIC50")
        try:
            assert len(fig.axes) == 2
            ax_a, ax_b = fig.axes
            assert ax_a.get_xlabel().startswith("mean kNN distance")
            assert ax_b.get_ylabel() == "predicted pIC50"
            legend = [t.get_text() for t in ax_b.get_legend().get_texts()]
            assert "inside domain" in legend and "outside domain" in legend
        finally:
            close_figure(fig)

    def test_knn_only_fallback_row_axis(self) -> None:
        from pharmaml.viz import close_figure
        from pharmaml.viz.ml_plots import plot_applicability_domain

        x, _y, _model = _ridge_data()
        result = knn_applicability_domain(x, pd.DataFrame({"a": [100.0]}), k=5)
        fig = plot_applicability_domain(result)
        try:
            assert fig.axes[1].get_ylabel() == "query row"
        finally:
            close_figure(fig)

    def test_wrong_result_rejected(self) -> None:
        from pharmaml.viz.ml_plots import plot_applicability_domain

        with pytest.raises(DataValidationError, match="ApplicabilityDomainResult"):
            plot_applicability_domain({"nope": True})
