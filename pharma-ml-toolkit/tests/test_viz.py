"""Tests for :mod:`pharmaml.viz` — every plot, every export format, the
HTML report and the one-click visual bundle.

All figures render through the headless Agg backend (pytest runs in a
plain Python process), so these tests double as the CI proof that
plotting works on servers without a display.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from pharmaml.data import load_dataset
from pharmaml.exceptions import DataValidationError, DependencyError, ModelError
from pharmaml.ml import (
    compare_models,
    evaluate_model,
    make_train_test_split,
    train_model,
)
from pharmaml.viz import (
    HTMLReport,
    export_docx,
    export_table,
    metrics_to_dataframe,
    plot_calibration_curve,
    plot_class_distribution,
    plot_concentration_profile,
    plot_confusion_matrix,
    plot_consensus_ranking,
    plot_cv_scores,
    plot_descriptor_distributions,
    plot_dose_accumulation,
    plot_energy_distribution,
    plot_feature_importance,
    plot_gp_uncertainty,
    plot_learning_curve,
    plot_model_comparison,
    plot_operator_trajectory,
    plot_pareto_front,
    plot_predicted_vs_actual,
    plot_receptor_atom_types,
    plot_residuals,
    plot_rl_introspection,
    plot_roc_curves,
    plot_similarity_heatmap,
    plot_table_comparison,
    plot_top_ligands,
    plot_training_history,
    save_figure,
    save_pdf_report,
    save_visual_bundle,
)


@pytest.fixture(scope="module")
def trained_classifier(classification_frame):
    """Fitted RF + held-out evaluation report on the shared fixture."""
    ds = load_dataset(
        _write_csv(classification_frame, "clf_fixture.csv"), target="Target"
    )
    X_train, X_test, y_train, y_test = make_train_test_split(
        ds.X, ds.y, test_size=0.25, random_state=42
    )
    result = train_model(
        "random_forest", "classification", X_train, y_train, tune=False
    )
    report = evaluate_model(result.model, X_test, y_test, "classification")
    return result, report, X_test, y_test


@pytest.fixture(scope="module")
def trained_regressor(regression_frame):
    ds = load_dataset(_write_csv(regression_frame, "reg_fixture.csv"), target="pIC50")
    X_train, X_test, y_train, y_test = make_train_test_split(
        ds.X, ds.y, test_size=0.25, random_state=42
    )
    result = train_model("random_forest", "regression", X_train, y_train, tune=False)
    report = evaluate_model(
        result.model, X_test, y_test, "regression", keep_predictions=True
    )
    return result, report, X_test, y_test


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    target = Path("/tmp") / name
    frame.to_csv(target, index=False)
    return target


@pytest.fixture()
def close_all():
    """Close every figure after each test (keeps the suite light)."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ---------------------------------------------------------------- ML plots
class TestConfusionMatrix:
    def test_from_report(self, trained_classifier, close_all):
        _, report, _, _ = trained_classifier
        fig = plot_confusion_matrix(report)
        assert fig.axes[0].get_xlabel() == "Predicted label"
        assert len(fig.axes[0].images) == 1

    def test_raw_matrix_with_labels(self, close_all):
        fig = plot_confusion_matrix([[7, 2], [1, 9]], labels=["neg", "pos"])
        assert fig is not None

    def test_normalized(self, trained_classifier, close_all):
        _, report, _, _ = trained_classifier
        fig = plot_confusion_matrix(report, normalize=True)
        assert "row-normalized" in fig.axes[0].get_title()

    def test_rejects_non_square(self, close_all):
        with pytest.raises(DataValidationError):
            plot_confusion_matrix(np.zeros((2, 3)))

    def test_rejects_label_count_mismatch(self, close_all):
        with pytest.raises(DataValidationError):
            plot_confusion_matrix([[1, 0], [0, 1]], labels=["a", "b", "c"])

    def test_float_dtype_matrix(self, close_all):
        """Regression: a float-dtype matrix used to hit the ':d' format
        code and raise ``ValueError: Unknown format code 'd'``."""
        fig = plot_confusion_matrix(np.array([[4.0, 1.0], [0.0, 5.0]]))
        annotations = [t.get_text() for t in fig.axes[0].texts]
        assert any("4" in text for text in annotations)

    def test_large_class_count_renders_as_a_plain_heatmap(self, close_all):
        """Regression (100x-61): an ID-column-as-target run once fed a
        420-class matrix here -- the figure tried a 464-inch canvas with
        176k per-cell text artists and OOM-killed the process.  Beyond
        the annotation limit the matrix must render as a plain heat-map
        with a capped figure size and zero text artists."""
        rng = np.random.default_rng(0)
        n = 120
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):  # a plausible sparse prediction pattern
            matrix[i, rng.integers(0, n)] += 1
        fig = plot_confusion_matrix(matrix, title="ID target")
        ax = fig.axes[0]
        assert len(ax.texts) == 0  # no per-cell annotations
        assert fig.get_size_inches()[0] <= 16.0  # capped, not 1.1 * n
        assert "120 classes, cells unannotated" in ax.get_title()
        assert len(ax.images) == 1  # the heat-map itself still renders


class TestRocCurves:
    def test_binary(self, trained_classifier, close_all):
        result, _, X_test, y_test = trained_classifier
        fig = plot_roc_curves(result.model, X_test, y_test)
        legend = fig.axes[0].get_legend().get_texts()
        assert any("AUC" in text.get_text() for text in legend)

    def test_multiclass_per_class_curves(self, close_all):
        from sklearn.datasets import make_classification as mk
        from sklearn.linear_model import LogisticRegression

        X, y = mk(n_samples=150, n_features=5, n_informative=4, n_redundant=0,
                  n_classes=3, n_clusters_per_class=1, random_state=3)
        model = LogisticRegression(max_iter=2000).fit(X, y)
        fig = plot_roc_curves(model, X, y)
        assert len(fig.axes[0].get_legend().get_texts()) == 4  # 3 classes + chance

    def test_model_without_proba_rejected(self, trained_classifier, close_all):
        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier().fit([[0], [1], [2], [3]], [0, 1, 0, 1])
        with pytest.raises(ModelError):
            plot_roc_curves(model, [[0], [1]], [0, 1])

    def test_stub_without_classes_rejected(self, close_all):
        """Regression (100x-41-B): a stub exposing predict_proba but not
        the fitted classes_ attribute used to die with a raw
        ``AttributeError`` — the ModelError guard checked only
        predict_proba."""

        class Stub:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]] * len(X))

        with pytest.raises(ModelError, match="classes_"):
            plot_roc_curves(Stub(), [[0.0], [1.0]], [0, 1])


class TestCalibrationCurve:
    def test_binary_with_brier(self, trained_classifier, close_all):
        result, _, X_test, y_test = trained_classifier
        fig = plot_calibration_curve(result.model, X_test, y_test)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("Brier" in label for label in labels)
        assert any("perfectly calibrated" in label for label in labels)

    def test_multiclass_one_curve_per_class(self, close_all):
        from sklearn.datasets import make_classification as mk
        from sklearn.linear_model import LogisticRegression

        X, y = mk(n_samples=150, n_features=5, n_informative=4, n_redundant=0,
                   n_classes=3, n_clusters_per_class=1, random_state=3)
        model = LogisticRegression(max_iter=2000).fit(X, y)
        fig = plot_calibration_curve(model, X, y)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert len(labels) == 4  # 3 one-vs-rest curves + the diagonal reference

    def test_model_without_proba_rejected(self, close_all):
        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier().fit([[0], [1], [2], [3]], [0, 1, 0, 1])
        with pytest.raises(ModelError):
            plot_calibration_curve(model, [[0], [1]], [0, 1])

    def test_rejects_single_bin(self, trained_classifier, close_all):
        result, _, X_test, y_test = trained_classifier
        with pytest.raises(DataValidationError, match="n_bins"):
            plot_calibration_curve(result.model, X_test, y_test, n_bins=1)

    def test_stub_without_classes_rejected(self, close_all):
        """Regression (100x-41-B): the calibration twin of the ROC guard —
        predict_proba present, classes_ missing, raw AttributeError."""

        class Stub:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]] * len(X))

        with pytest.raises(ModelError, match="classes_"):
            plot_calibration_curve(Stub(), [[0.0], [1.0]], [0, 1])


class TestLearningCurve:
    def test_curves_and_verdict(self, close_all):
        from sklearn.datasets import make_classification as mk
        from sklearn.ensemble import RandomForestClassifier

        X, y = mk(n_samples=120, n_features=6, n_informative=4, random_state=7)
        model = RandomForestClassifier(n_estimators=25, random_state=0)
        fig = plot_learning_curve(model, X, y, cv=3, scoring="accuracy",
                                  train_sizes=np.linspace(0.3, 1.0, 4))
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any(label.startswith("training score") for label in labels)
        assert any(label.startswith("CV score") for label in labels)
        title = fig.axes[0].get_title()
        assert "converging" in title or "overfitting" in title  # the verdict line
        assert len(fig.axes[0].collections) == 2  # ± std bands

    def test_rejects_single_fold(self, close_all):
        from sklearn.datasets import make_classification as mk

        X, y = mk(n_samples=40, n_features=4, random_state=1)
        with pytest.raises(DataValidationError, match="cv"):
            plot_learning_curve("not a model", X, y, cv=1)  # type: ignore[arg-type]


class TestTrainingHistory:
    HISTORY: ClassVar[list[dict[str, float | None]]] = [
        {"epoch": 1, "train_loss": 0.9, "val_loss": 1.0},
        {"epoch": 2, "train_loss": 0.6, "val_loss": 0.8},
        {"epoch": 3, "train_loss": 0.5, "val_loss": 0.85},
        {"epoch": 4, "train_loss": 0.45, "val_loss": 0.9},
    ]

    def test_two_curves_with_the_early_stop_verdict(self, close_all):
        fig = plot_training_history(self.HISTORY)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any(label.startswith("training loss (final") for label in labels)
        assert any(label.startswith("validation loss (best") for label in labels)
        assert "@ epoch 2" in labels[1]  # the restored-weights epoch
        title = ax.get_title(loc="left")
        assert "early stopped 2 epoch(s) after the best (epoch 2)" in title
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("epoch", "loss")
        assert len(ax.lines) == 3  # train + val + the best-epoch marker line

    def test_dataframe_history_and_still_improving_verdict(self, close_all):
        import pandas as pd

        frame = pd.DataFrame(self.HISTORY)
        # best val at the LAST epoch -> the patience window never ran out
        frame.loc[3, "val_loss"] = 0.4
        fig = plot_training_history(frame, title="Denoised history")
        title = fig.axes[0].get_title(loc="left")
        assert "still improving" in title
        assert title.startswith("Denoised history")

    def test_valless_history_plots_one_curve_and_says_so(self, close_all):
        history = [{"epoch": e, "train_loss": 1.0 / e} for e in range(1, 5)]
        fig = plot_training_history(history)
        ax = fig.axes[0]
        assert len(ax.get_legend().get_texts()) == 1
        assert "no validation split" in ax.get_title(loc="left")

    def test_none_val_losses_plot_as_gaps(self, close_all):
        history = [
            {"epoch": 1, "train_loss": 0.9, "val_loss": None},
            {"epoch": 2, "train_loss": 0.7, "val_loss": 0.8},
            {"epoch": 3, "train_loss": 0.5, "val_loss": None},
        ]
        fig = plot_training_history(history)
        assert len(fig.axes[0].get_legend().get_texts()) == 2  # gaps, not absent

    def test_empty_and_malformed_histories_raise_typed_errors(self, close_all):
        with pytest.raises(DataValidationError, match="history is empty"):
            plot_training_history([])
        with pytest.raises(DataValidationError, match="missing column"):
            plot_training_history([{"epoch": 1}])
        with pytest.raises(DataValidationError, match="non-numeric"):
            plot_training_history([{"epoch": 1, "train_loss": "lots"}])
        with pytest.raises(DataValidationError, match="list of per-epoch records"):
            plot_training_history(42)  # type: ignore[arg-type]


class TestRegressionPlots:
    def test_parity(self, trained_regressor, close_all):
        _, _report, _, _ = trained_regressor
        fig = plot_predicted_vs_actual(
            np.arange(10, dtype=float),
            np.arange(10, dtype=float) + np.linspace(-1, 1, 10),
            metrics={"RMSE": 0.5},
        )
        assert fig is not None

    def test_parity_shape_mismatch(self, close_all):
        with pytest.raises(DataValidationError):
            plot_predicted_vs_actual([1, 2, 3], [1, 2])

    def test_parity_empty_raises_typed(self, close_all):
        """Regression: empty y_true/y_pred used to crash with a raw
        zero-size ``ValueError`` — now the plot_cv_scores-style typed
        error instead."""
        with pytest.raises(DataValidationError, match="empty y_true/y_pred"):
            plot_predicted_vs_actual([], [])

    def test_residuals_empty_raises_typed(self, close_all):
        """Regression: ``plot_residuals([], [])`` used to render a silent
        empty figure — now the same typed empty contract as the parity
        plot."""
        with pytest.raises(DataValidationError, match="empty y_true/y_pred"):
            plot_residuals([], [])

    def test_residuals(self, close_all):
        rng = np.random.default_rng(0)
        y = rng.normal(size=60)
        fig = plot_residuals(y, y + rng.normal(scale=0.2, size=60))
        assert len(fig.axes) == 2

    def test_parity_from_regression_report(self, trained_regressor, close_all):
        _, report, _, _ = trained_regressor
        fig = plot_predicted_vs_actual(
            report.predictions, report.predictions, title="self-check"
        )
        assert "self-check" in fig.axes[0].get_title()


class TestGaussianProcessUncertaintyPlot:
    """100x-102 — plot_gp_uncertainty: the GP parity twin with ±z·σ
    error bars and the coverage annotation.
    """

    def test_basic_figure_and_labels(self, close_all):
        fig = plot_gp_uncertainty(
            [1.0, 2.0, 3.0], [1.1, 1.9, 3.2], [0.1, 0.2, 0.3]
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "actual"
        assert ax.get_ylabel() == "predicted (ŷ)"
        # one errorbar container + one scatter + the identity line
        assert len(ax.collections) >= 1
        assert len(ax.lines) >= 1

    def test_coverage_annotation_text(self, close_all):
        fig = plot_gp_uncertainty(
            [1.0, 2.0, 3.0], [1.1, 1.9, 3.2], [0.1, 0.2, 0.3]
        )
        # all three points straddle -> the box reports 100.0% coverage
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("coverage" in t and "100.0%" in t for t in texts)

    def test_sigma_shape_mismatch(self, close_all):
        with pytest.raises(DataValidationError, match="must match"):
            plot_gp_uncertainty([1.0, 2.0], [1.0], [0.1, 0.1])

    def test_negative_sigma_raises_typed(self, close_all):
        with pytest.raises(DataValidationError, match="non-negative"):
            plot_gp_uncertainty([1.0], [1.0], [-0.1])

    def test_empty_raises_typed(self, close_all):
        with pytest.raises(DataValidationError, match="empty y_true/y_pred"):
            plot_gp_uncertainty([], [], [])

    def test_non_positive_z_raises_typed(self, close_all):
        with pytest.raises(DataValidationError, match="z must be positive"):
            plot_gp_uncertainty([1.0], [1.0], [0.1], z=0.0)

    def test_end_to_end_from_trained_gp(self, regression_frame, close_all):
        from pharmaml.ml.evaluation import predict_with_uncertainty

        ds = load_dataset(
            _write_csv(regression_frame, "gp_fixture.csv"), target="pIC50"
        )
        X_train, X_test, y_train, y_test = make_train_test_split(
            ds.X, ds.y, test_size=0.25, random_state=42
        )
        result = train_model(
            "gaussian_process", "regression", X_train, y_train, tune=False
        )
        mean, sigma = predict_with_uncertainty(result.model, X_test)
        fig = plot_gp_uncertainty(y_test, mean, sigma, metrics={"R2": 0.9})
        assert fig is not None
        # metrics ride along in the annotation box
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("R2=0.9" in t for t in texts)


class TestLeaderboardAndDiagnostics:
    def test_model_comparison_bars(self, regression_frame, close_all):
        frame = regression_frame
        board = compare_models(
            frame.drop(columns=["pIC50"]), frame["pIC50"], "regression",
            models=["ridge", "random_forest"], cv=3,
        )
        fig = plot_model_comparison(board)
        assert len(fig.axes[0].patches) == 2

    def test_model_comparison_requires_model_column(self, close_all):
        with pytest.raises(DataValidationError):
            plot_model_comparison(pd.DataFrame({"a": [1]}))

    def test_model_comparison_metric_autodetect(self, close_all):
        board = pd.DataFrame({
            "model": ["a", "b"],
            "r2_mean": [0.5, 0.7],
            "r2_std": [0.05, 0.02],
        })
        fig = plot_model_comparison(board)
        assert "r2" in fig.axes[0].get_xlabel()

    def test_model_comparison_rejects_non_numeric_metric_cells(self, close_all):
        """Regression (100x-41-B): a string cell for the requested metric
        leaked a raw TypeError from sort_values (mixed '<' comparison) or
        a raw ValueError from to_numpy(dtype=float)."""
        board_mean = pd.DataFrame({
            "model": ["rf", "ridge"], "r2_mean": [0.5, "n/a"], "r2_std": [0.1, 0.2],
        })
        with pytest.raises(DataValidationError, match="r2_mean.*non-numeric"):
            plot_model_comparison(board_mean, metric="r2")
        board_std = pd.DataFrame({
            "model": ["rf", "ridge"], "r2_mean": [0.5, 0.4], "r2_std": ["??", 0.2],
        })
        with pytest.raises(DataValidationError, match="r2_std.*non-numeric"):
            plot_model_comparison(board_std, metric="r2")

    def test_model_comparison_nan_metric_cells_still_allowed(self, close_all):
        """NaN cells model failed CV runs (compare_models keeps those rows)
        — they must keep rendering; only non-numeric cells are rejected."""
        board = pd.DataFrame({
            "model": ["rf", "dead"], "r2_mean": [0.5, np.nan], "r2_std": [0.1, np.nan],
        })
        fig = plot_model_comparison(board, metric="r2")
        assert len(fig.axes[0].patches) == 2

    def test_model_comparison_rejects_duplicated_columns(self, close_all):
        """Regression (100x-41-B): a duplicated 'model' column made every
        y-tick read "model" (iterating a multi-column selection yields
        COLUMN names); duplicated metric columns break selection outright."""
        board = pd.concat(
            [
                pd.DataFrame({"model": ["rf", "ridge"], "r2_mean": [0.5, 0.4]}),
                pd.DataFrame({"model": ["lgbm", "svr"]}),
            ],
            axis=1,
        )
        with pytest.raises(DataValidationError, match="duplicate column names.*model"):
            plot_model_comparison(board, metric="r2")

    def test_class_distribution_mapping(self, close_all):
        fig = plot_class_distribution({"active": 30, "inactive": 70})
        assert len(fig.axes[0].patches) == 2

    def test_class_distribution_from_dataset(self, classification_frame, close_all):
        ds = load_dataset(_write_csv(classification_frame, "clf_cd.csv"), target="Target")
        fig = plot_class_distribution(ds)
        assert fig is not None

    def test_class_distribution_rejects_empty(self, close_all):
        with pytest.raises(DataValidationError):
            plot_class_distribution({})

    def test_feature_importance_from_model(self, trained_classifier, close_all):
        result, _, _, _ = trained_classifier
        fig = plot_feature_importance(result.model, feature_names=[f"f{i}" for i in range(6)])
        assert len(fig.axes[0].patches) == 6

    def test_feature_importance_from_mapping(self, close_all):
        fig = plot_feature_importance({"logp": 0.8, "mw": 0.4, "tpsa": 0.1})
        assert len(fig.axes[0].patches) == 3

    def test_feature_importance_top_n(self, close_all):
        fig = plot_feature_importance({f"f{i}": i / 10 for i in range(10)}, top_n=4)
        assert len(fig.axes[0].patches) == 4

    def test_feature_importance_top_n_contract(self, close_all):
        """top_n: None -> every feature; 0/-1/non-int -> typed error.

        Regression (100x-41-B): ``0``/``-1`` meant "empty figure"/"all
        but one" here while ``plot_top_ligands`` read both as "all rows",
        and ``None`` raised a RAW TypeError from ``min(top_n, len(values))``."""
        mapping = {f"f{i}": i / 10 for i in range(10)}
        assert len(plot_feature_importance(mapping, top_n=None).axes[0].patches) == 10
        for bad in (0, -1, 2.5):
            with pytest.raises(
                DataValidationError, match="top_n must be a positive integer or None"
            ):
                plot_feature_importance(mapping, top_n=bad)

    def test_feature_importance_ties_first_declared(self, close_all):
        """Regression: ``argsort(values)[::-1]`` flips tied groups, so ties
        selected the LAST-declared features; ties now resolve
        first-declared-wins (matching plot_top_ligands)."""
        mapping = {f"feat{i:02d}": 0.5 for i in range(10)}  # all values tied
        fig = plot_feature_importance(mapping, top_n=3)
        ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert set(ticks) == {"feat00", "feat01", "feat02"}

    def test_feature_importance_unsupported_model(self, close_all):
        with pytest.raises(DataValidationError):
            plot_feature_importance(object())

    def test_cv_scores_array(self, close_all):
        fig = plot_cv_scores([0.7, 0.8, 0.75])
        assert "mean" in fig.axes[0].get_legend().get_texts()[0].get_text()

    def test_cv_scores_from_training_result(self, trained_regressor, close_all):
        _result, _, _, _ = trained_regressor
        fig = plot_cv_scores(np.array([0.6, 0.7]))
        assert fig is not None

    def test_cv_scores_empty(self, close_all):
        with pytest.raises(DataValidationError):
            plot_cv_scores([])


# ---------------------------------------------------------------- pharma
class TestPharmaPlots:
    def test_concentration_profile(self, close_all):
        t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        c = 10.0 * np.exp(-0.25 * t)
        fig = plot_concentration_profile(t, c)
        assert len(fig.axes) == 2

    def test_concentration_profile_with_summary(self, close_all):
        from pharmaml.pharma.pk import nca_summary

        t = np.array([0.0, 1.0, 4.0, 8.0, 12.0])
        c = np.array([0.0, 9.8, 5.1, 2.2, 0.9])
        fig = plot_concentration_profile(t, c, summary=nca_summary(t, c))
        assert "t½" in fig.axes[1].get_title()

    def test_concentration_profile_non_positive_log_panel(self, close_all):
        t = np.array([0.0, 1.0, 2.0])
        c = np.array([0.0, 0.0, 0.0])
        fig = plot_concentration_profile(t, c)
        assert "no positive concentrations" in fig.axes[1].texts[0].get_text()

    def test_dose_accumulation(self, close_all):
        fig = plot_dose_accumulation(dose=500, tau=8, t_half=6, vd=40, n_doses=5)
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert any("Cmax,ss" in label for label in labels)

    def test_dose_accumulation_requires_exactly_one_rate(self, close_all):
        with pytest.raises(DataValidationError):
            plot_dose_accumulation(dose=100, tau=8, t_half=6, ke=0.1, vd=40)
        with pytest.raises(DataValidationError):
            plot_dose_accumulation(dose=100, tau=8, vd=40)

    def test_dose_accumulation_validates_regimen_up_front(self, close_all):
        """Regression: a typo'd regimen (negative dose, impossible
        bioavailability) used to be caught deep inside accumulation_ratio
        or plotted silently as negative concentrations."""
        with pytest.raises(DataValidationError, match="dose"):
            plot_dose_accumulation(dose=-500, tau=8, t_half=6, vd=40)
        with pytest.raises(DataValidationError, match="tau"):
            plot_dose_accumulation(dose=500, tau=0, t_half=6, vd=40)
        with pytest.raises(DataValidationError, match="vd"):
            plot_dose_accumulation(dose=500, tau=8, t_half=6, vd=-40)
        with pytest.raises(DataValidationError, match="bioavailability"):
            plot_dose_accumulation(dose=500, tau=8, t_half=6, vd=40, f=1.5)

    def test_dose_accumulation_extreme_half_life(self, close_all):
        """Regression: t_half >= ~1e17 underflows ``1 - exp(-ke*tau)`` to
        exactly zero and used to raise a raw ZeroDivisionError; t_half=1e16
        stays finite (Cmax,ss envelope ~9e16)."""
        with pytest.raises(DataValidationError, match="underflowed"):
            plot_dose_accumulation(dose=100.0, tau=1.0, t_half=1e17, vd=10.0)
        fig = plot_dose_accumulation(dose=100.0, tau=1.0, t_half=1e16, vd=10.0)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any(label.startswith("Cmax,ss") for label in labels)

    def test_concentration_profile_default_auc_uses_package_trapezoid(self, close_all):
        """The no-summary/no-auc path must go through pharmaml's own
        auc_trapezoid (numpy>=1.24 has no np.trapezoid)."""
        t = np.array([0.0, 2.0, 4.0])
        c = np.array([0.0, 6.0, 0.0])
        fig = plot_concentration_profile(t, c)
        legend_texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert any(text.startswith("AUC = 12") for text in legend_texts)  # triangle: ½·4·6


# ---------------------------------------------------------------- chem
class TestChemPlots:
    def test_descriptor_distributions(self, regression_frame, close_all):
        fig = plot_descriptor_distributions(
            regression_frame.drop(columns=["pIC50"]), max_columns=4
        )
        assert len([ax for ax in fig.axes if ax.get_visible()]) == 4

    def test_descriptor_distributions_column_subset(self, regression_frame, close_all):
        fig = plot_descriptor_distributions(
            regression_frame, columns=["descr_00", "descr_01", "pIC50"]
        )
        assert fig is not None

    def test_descriptor_distributions_unknown_column(self, regression_frame, close_all):
        with pytest.raises(DataValidationError):
            plot_descriptor_distributions(regression_frame, columns=["nope"])

    def test_descriptor_distributions_rejects_empty(self, close_all):
        with pytest.raises(DataValidationError):
            plot_descriptor_distributions(pd.DataFrame())

    def test_descriptor_distributions_rejects_non_positive_bins(self, close_all):
        """Regression (100x-41-B): bins <= 0 leaked matplotlib's raw
        ``ValueError: `bins` must be positive, when an integer``."""
        frame = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        for bad in (0, -1):
            with pytest.raises(DataValidationError, match="bins must be a positive integer"):
                plot_descriptor_distributions(frame, bins=bad)

    def test_descriptor_distributions_accepts_named_bin_strategy(self, close_all):
        """matplotlib bin specs (named strategies, edge sequences) must
        keep passing through the new typed guard."""
        frame = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        assert plot_descriptor_distributions(frame, bins="auto") is not None

    def test_descriptor_distributions_rejects_duplicated_columns(self, close_all):
        """Regression (100x-41-B): a duplicated column name turns
        frame[name] into a multi-column DataFrame whose is_numeric_dtype
        check is False, so the grid reported the MISLEADING "no numeric
        columns to plot" even when every duplicate was numeric."""
        dup_numeric = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["v", "v"])
        with pytest.raises(DataValidationError, match=r"duplicate column names: \['v'\]"):
            plot_descriptor_distributions(dup_numeric)
        dup_mixed = pd.concat(
            [pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"a": ["x", "y"]})], axis=1
        )
        with pytest.raises(DataValidationError, match=r"duplicate column names: \['a'\]"):
            plot_descriptor_distributions(dup_mixed)

    def test_similarity_heatmap_dataframe_labels(self, close_all):
        rng = np.random.default_rng(1)
        m = rng.random((4, 4))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        frame = pd.DataFrame(m, index=list("wxyz"), columns=list("wxyz"))
        fig = plot_similarity_heatmap(frame)
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == list("wxyz")

    def test_similarity_heatmap_annotates_small(self, close_all):
        fig = plot_similarity_heatmap(np.eye(3))
        assert fig.axes[0].texts  # cell annotations present

    def test_similarity_heatmap_non_square(self, close_all):
        with pytest.raises(DataValidationError):
            plot_similarity_heatmap(np.zeros((2, 3)))

    def test_similarity_heatmap_rejects_non_numeric_frame(self, close_all):
        """Regression (100x-41-B): a string (non-numeric) frame/array
        leaked numpy's raw ``could not convert string to float``."""
        with pytest.raises(DataValidationError, match="must be numeric"):
            plot_similarity_heatmap(pd.DataFrame([[
                "high", "low"], ["low", "high"
            ]], index=["e", "p"], columns=["e", "p"]))
        with pytest.raises(DataValidationError, match="must be numeric"):
            plot_similarity_heatmap([["high", "low"], ["low", "high"]])

    def test_table_comparison_two_panels(self, close_all):
        from pharmaml.chem import compare_tables

        a = pd.DataFrame({"Ligand": ["aspirin.pdbqt", "naproxen.pdbqt", "ibuprofen.pdbqt"]})
        b = pd.DataFrame({"Ligand": ["aspirin", "ketoprofen", "tolmetin"]})
        comparison = compare_tables(a, b, column="Ligand")
        fig = plot_table_comparison(
            comparison, labels=("docking_summary.csv", "library.csv")
        )
        assert len(fig.axes) == 2  # overlap bar + per-bucket counts
        # Bucket names carry the caller's table labels.
        ylabels = {t.get_text() for t in fig.axes[1].get_yticklabels()}
        assert "only in docking_summary.csv" in ylabels
        assert "only in library.csv" in ylabels
        # The top overlap bar exposes one legend entry per bucket.
        assert len(fig.axes[0].get_legend_handles_labels()[1]) == 3

    def test_table_comparison_saves_png(self, close_all, tmp_path):
        from pharmaml.chem import compare_tables

        a = pd.DataFrame({"Ligand": ["l1", "l2", "l3", "l4", "l5"]})
        b = pd.DataFrame({"Ligand": ["l1", "l2", "l3"]})
        comparison = compare_tables(a, b, column="Ligand")
        fig = plot_table_comparison(comparison)
        out = save_figure(fig, tmp_path / "reconciliation.png")
        assert out.exists() and out.stat().st_size > 1_000

    def test_table_comparison_rejects_wrong_object(self, close_all):
        with pytest.raises(DataValidationError, match="TableComparison"):
            plot_table_comparison({"common": ["x"]})

    def test_table_comparison_rejects_empty_reconciliation(self, close_all):
        from pharmaml.chem import compare_tables

        empty = compare_tables(
            pd.DataFrame({"Ligand": []}), pd.DataFrame({"Ligand": []}), column="Ligand"
        )
        with pytest.raises(DataValidationError, match="nothing to compare"):
            plot_table_comparison(empty)

    def test_table_comparison_rejects_wrong_labels_arity(self, close_all):
        """Regression (100x-41-B): a 1- or 3-item labels tuple died on
        the tuple unpacking with a raw ``ValueError``."""
        from pharmaml.chem import compare_tables

        comparison = compare_tables(
            pd.DataFrame({"Ligand": ["a", "b"]}),
            pd.DataFrame({"Ligand": ["a"]}),
            column="Ligand",
        )
        for bad in (("only_a",), ("a", "b", "c")):
            with pytest.raises(DataValidationError, match="labels must be a .* pair"):
                plot_table_comparison(comparison, labels=bad)


# ---------------------------------------------------------------- docking
class TestDockingPlots:
    def _summary(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            "folder": [f"ligand_{i:02d}" for i in range(10)],
            "best_lowest_energy": -6.0 - rng.random(10) * 3.0,
            "best_num_in_cluster": rng.integers(1, 40, size=10),
        })

    def test_energy_distribution(self, close_all):
        fig = plot_energy_distribution(self._summary())
        assert any("median" in t.get_text() for t in fig.axes[0].get_legend().get_texts())

    def test_top_ligands_sorted(self, close_all):
        summary = self._summary()
        fig = plot_top_ligands(summary, top_n=5)
        assert len(fig.axes[0].patches) == 5

    def test_top_ligands_annotate_outside_negative_bars(self, close_all):
        """Regression: value labels used to be drawn on top of the bar
        for the usual negative energies — they must sit outside the tip."""
        fig = plot_top_ligands(self._summary(), top_n=5)
        alignments = [text.get_ha() for text in fig.axes[0].texts]
        assert alignments and set(alignments) == {"right"}  # all energies < 0

    def _duplicated_index_summary(self) -> pd.DataFrame:
        """The classic append-a-re-run pattern: a concat whose index labels
        0..2 are duplicated."""
        scan = pd.DataFrame({
            "folder": ["01_aspirin", "02_ibuprofen", "03_naproxen", "06_theophylline"],
            "best_lowest_energy": [-7.1, -8.5, -11.5, -11.51],
        })
        combined = pd.concat([scan, scan.head(3)])
        assert combined.index.has_duplicates
        return combined

    def test_energy_distribution_duplicated_index_names_true_best(self, close_all):
        """Regression: with duplicated index labels the label-based row
        lookup expanded them, so the legend named the WRONG best ligand
        (02_ibuprofen instead of 06_theophylline)."""
        fig = plot_energy_distribution(self._duplicated_index_summary())
        best = next(
            t.get_text() for t in fig.axes[0].get_legend().get_texts()
            if t.get_text().startswith("best =")
        )
        assert best == "best = -11.5 (06_theophylline)"

    def test_top_ligands_duplicated_index(self, close_all):
        """Regression: the same duplicated-index frame used to crash
        plot_top_ligands with a raw ValueError (misaligned arrays)."""
        fig = plot_top_ligands(self._duplicated_index_summary(), top_n=3)
        ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        # true top-3 energies: -11.51 (theophylline) + the two -11.5 rows
        assert set(ticks) == {"03_naproxen", "06_theophylline"}
        assert ticks[-1] == "06_theophylline"  # strongest binder on top

    def test_top_ligands_tie_cut_is_first_declared(self, close_all):
        """Regression: all-tied energies — the top_n cut must keep the
        FIRST-declared ligands (kind="stable"), at any frame size (the
        default quicksort cut is an arbitrary middle block at n=512)."""
        expected = {f"L{i:04d}" for i in range(5)}
        for n in (8, 512):
            frame = pd.DataFrame({
                "folder": [f"L{i:04d}" for i in range(n)],
                "best_lowest_energy": [-5.0] * n,
            })
            fig = plot_top_ligands(frame, top_n=5)
            ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
            assert set(ticks) == expected
            assert ticks[-1] == "L0000"  # first-declared tie winner on top

    def test_top_ligands_top_n_contract(self, close_all):
        """top_n: None -> every ligand; 0/-1/non-int -> typed error.

        Regression (100x-41-B): plot_top_ligands read 0/-1 as "all rows"
        while plot_feature_importance read them as "empty"/"all but one";
        a string top_n raised a raw TypeError from the comparison."""
        summary = self._summary()
        assert len(plot_top_ligands(summary, top_n=None).axes[0].patches) == 10
        for bad in (0, -1, "3", 2.5):
            with pytest.raises(
                DataValidationError, match="top_n must be a positive integer or None"
            ):
                plot_top_ligands(summary, top_n=bad)

    def test_energy_distribution_rejects_non_positive_bins(self, close_all):
        """Regression (100x-41-B): bins <= 0 leaked matplotlib's raw
        ``ValueError: `bins` must be positive, when an integer``."""
        for bad in (0, -3):
            with pytest.raises(DataValidationError, match="bins must be a positive integer"):
                plot_energy_distribution(self._summary(), bins=bad)

    def test_energy_distribution_accepts_named_bin_strategy(self, close_all):
        assert plot_energy_distribution(self._summary(), bins="auto") is not None

    def test_explicit_energy_column_must_exist(self, close_all):
        """Regression (100x-41-B): an explicitly requested column missing
        from the frame was SILENTLY ignored, falling back to the default
        whenever top_lowest_energy happened to be present."""
        fallback = self._summary().rename(
            columns={"best_lowest_energy": "top_lowest_energy"}
        )
        with pytest.raises(
            DataValidationError, match="'binding'.*available.*top_lowest_energy"
        ):
            plot_energy_distribution(fallback, energy_column="binding")
        with pytest.raises(DataValidationError, match="'binding'"):
            plot_top_ligands(fallback, energy_column="binding")

    def test_default_energy_column_missing_message_unchanged(self, close_all):
        """None + no best/top column keeps the pre-existing message (the
        fallback contract itself is unchanged)."""
        with pytest.raises(DataValidationError, match="no energy column found"):
            plot_energy_distribution(pd.DataFrame({"folder": ["a"]}))

    def test_energy_fallback_column(self, close_all):
        summary = self._summary().rename(columns={"best_lowest_energy": "top_lowest_energy"})
        fig = plot_energy_distribution(summary)
        assert fig is not None

    def test_consensus_ranking_sorted_nan_dropped(self, close_all):
        """The figure twin of the explainable leaderboard: strongest
        consensus on top, NaN rows (incomplete inputs) cannot rank."""
        rescored = pd.DataFrame({
            "ligand": ["A", "B", "C", "D"],
            "consensus": [0.2, 0.9, 0.5, float("nan")],
        })
        fig = plot_consensus_ranking(rescored)
        ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert ticks == ["A", "C", "B"]  # bottom-to-top: strongest last
        assert len(fig.axes[0].patches) == 3  # the NaN row is dropped
        assert fig.axes[0].get_xlabel() == (
            "consensus (0–1, weighted mean of the _norm units) — higher is stronger"
        )

    def test_consensus_ranking_top_n_and_identity(self, close_all):
        rescored = pd.DataFrame({
            "folder": ["lig1", "lig2", "lig3", "lig4"],
            "consensus": [0.1, 0.4, 0.9, 0.6],
        })
        fig = plot_consensus_ranking(rescored, top_n=2)
        # identity defaults to the frame's FIRST column (folder), and the
        # cut keeps the two strongest
        ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert ticks == ["lig4", "lig3"]
        # an explicit identity column must exist, naming the available set
        with pytest.raises(
            DataValidationError, match="'name'.*available.*folder"
        ):
            plot_consensus_ranking(rescored, identity_column="name")

    def test_consensus_ranking_validation(self, close_all):
        rescored = pd.DataFrame({"ligand": ["A"], "consensus": [0.5]})
        # not a DataFrame
        with pytest.raises(DataValidationError, match="must be a pandas DataFrame"):
            plot_consensus_ranking([("A", 0.5)])
        # missing consensus column, naming the available columns
        with pytest.raises(
            DataValidationError, match="'consensus'.*available.*ligand"
        ):
            plot_consensus_ranking(pd.DataFrame({"ligand": ["A"]}))
        # all-NaN consensus: nothing can rank
        with pytest.raises(DataValidationError, match="no numeric consensus"):
            plot_consensus_ranking(
                pd.DataFrame({"ligand": ["A", "B"], "consensus": [None, None]})
            )
        # the shared top_n contract
        for bad in (0, -1, "3", 2.5):
            with pytest.raises(
                DataValidationError, match="top_n must be a positive integer or None"
            ):
                plot_consensus_ranking(rescored, top_n=bad)

    def test_missing_energy_column(self, close_all):
        with pytest.raises(DataValidationError):
            plot_energy_distribution(pd.DataFrame({"folder": ["a"]}))

    def test_requires_dataframe(self, close_all):
        with pytest.raises(DataValidationError):
            plot_energy_distribution("not a frame")

    def test_receptor_atom_types_from_dict(self, close_all):
        fig = plot_receptor_atom_types(
            {"A": 130, "C": 400, "HD": 190, "OA": 120, "NA": 30, "SA": 8}
        )
        ax = fig.axes[0]
        assert len(ax.patches) == 6
        # every bar carries its count annotation
        assert sum(1 for t in ax.texts if t.get_text().isdigit()) == 6
        # chemistry classes present in the legend
        legend_texts = " | ".join(t.get_text() for t in ax.get_legend().get_texts())
        assert "polar hydrogen" in legend_texts and "non-polar carbon" in legend_texts

    def test_receptor_atom_types_sorted_desc(self, close_all):
        fig = plot_receptor_atom_types({"C": 10, "HD": 200, "OA": 50})
        heights = [p.get_height() for p in fig.axes[0].patches]
        assert heights == sorted(heights, reverse=True)

    def test_receptor_atom_types_from_pdbqt_file(self, close_all, tmp_path):
        pdbqt = tmp_path / "receptor.pdbqt"
        pdbqt.write_text(
            "ATOM      1  C   LYS A   1     34.741  20.264 -10.844  1.00  0.00    +0.242 C\n"
            "ATOM      2  O   LYS A   1     33.945  20.813 -10.081  1.00  0.00    -0.273 OA\n"
            "ATOM      3 HD1  LYS A   1     35.900  20.900 -10.900  1.00  0.00    +0.100 HD\n",
            encoding="utf-8",
        )
        fig = plot_receptor_atom_types(pdbqt)
        assert len(fig.axes[0].patches) == 3

    def test_receptor_atom_types_ignores_non_alpha_tail(self, close_all, tmp_path):
        pdbqt = tmp_path / "weird.pdbqt"
        pdbqt.write_text(
            "ATOM      1  C   LIG A   1      0.000   0.000   0.000  1.00  0.00    +0.20 C\n"
            "REMARK   something 123.45\n",
            encoding="utf-8",
        )
        fig = plot_receptor_atom_types(pdbqt)
        assert len(fig.axes[0].patches) == 1

    def test_receptor_atom_types_empty_dict(self, close_all):
        with pytest.raises(DataValidationError):
            plot_receptor_atom_types({})

    def test_receptor_atom_types_missing_file(self, close_all, tmp_path):
        with pytest.raises(DataValidationError):
            plot_receptor_atom_types(tmp_path / "nope.pdbqt")

    def test_receptor_atom_types_accepts_report_like_object(self, close_all):
        class _FakeReport:
            atom_types: ClassVar[dict[str, int]] = {"C": 12, "OA": 3}
            output_pdbqt = "unused.pdbqt"

        fig = plot_receptor_atom_types(_FakeReport())
        assert len(fig.axes[0].patches) == 2

    def test_receptor_atom_types_rejects_junk_dict_values(self, close_all):
        """Regression (100x-41-B): a receptor dict with junk values
        leaked the raw ``int()`` ValueError/TypeError from the count
        comprehension; a report-like object with junk counts too."""
        with pytest.raises(DataValidationError, match="must be integers"):
            plot_receptor_atom_types({"HD": "junk", "A": 130})
        with pytest.raises(DataValidationError, match="must be integers"):
            plot_receptor_atom_types({"HD": None, "A": 130})

        class _JunkReport:
            atom_types: ClassVar[dict[str, object]] = {"C": "many"}
            output_pdbqt = "unused.pdbqt"

        with pytest.raises(DataValidationError, match="must be integers"):
            plot_receptor_atom_types(_JunkReport())


# ---------------------------------------------------------------- generative
@pytest.fixture(scope="module")
def adaptive_run():
    """A cheap adaptive GA run (~0.1 s): pop 8 x gen 3, topology mode."""
    from pharmaml.generative import DEFAULT_CORPUS, GeneticOptimizer, QedObjective

    return GeneticOptimizer(
        QedObjective(),
        population_size=8,
        generations=3,
        seed=42,
        mutation="topology",
        operator_selection="adaptive",
    ).optimize(corpus=DEFAULT_CORPUS[:10])


class TestGenerativePlots:
    def test_two_panels_and_legends(self, adaptive_run, close_all):
        fig = plot_operator_trajectory(adaptive_run)
        assert len(fig.axes) == 2  # score panel + the default mutation level
        score_legend = {
            text.get_text() for text in fig.axes[0].get_legend().get_texts()
        }
        assert score_legend == {"best score", "population mean"}
        level_legend = {
            text.get_text() for text in fig.axes[1].get_legend().get_texts()
        }
        assert level_legend == {"topology", "character"}

    def test_all_three_levels_render_three_plus_one_axes(self, adaptive_run, close_all):
        fig = plot_operator_trajectory(
            adaptive_run, levels=("level1", "mutation", "geometry")
        )
        assert len(fig.axes) == 4

    def test_uniform_run_and_unknown_level_raise_typed_errors(
        self, adaptive_run, close_all
    ):
        from pharmaml.generative import GeneticOptimizer, QedObjective

        uniform = GeneticOptimizer(
            QedObjective(), population_size=8, generations=3, seed=42
        ).optimize(corpus=["CCO", "c1ccccc1"])
        with pytest.raises(DataValidationError, match="operator_trajectory"):
            plot_operator_trajectory(uniform)
        with pytest.raises(DataValidationError, match="not in this run's trajectory"):
            plot_operator_trajectory(adaptive_run, levels=("nope",))


@pytest.fixture(scope="module")
def rl_run():
    """A cheap RL run (~2 s): QED over a tiny corpus, 5 iterations."""
    from pharmaml.generative import DEFAULT_CORPUS, QedObjective, run_rl_optimization

    return run_rl_optimization(
        QedObjective(),
        corpus=list(DEFAULT_CORPUS[:24]),
        prior_epochs=6,
        iterations=5,
        batch_size=16,
        seed=7,
    )


class TestRlIntrospectionPlot:
    def test_four_panels_and_legends(self, rl_run, close_all):
        fig = plot_rl_introspection(rl_run)
        assert len(fig.axes) == 4
        score_legend = {
            text.get_text() for text in fig.axes[0].get_legend().get_texts()
        }
        assert score_legend == {"best score", "batch mean"}
        logprob_legend = {
            text.get_text() for text in fig.axes[1].get_legend().get_texts()
        }
        assert logprob_legend == {"agent", "prior (frozen)"}
        assert fig.axes[3].get_xlabel() == "iteration"
        assert fig.axes[1].get_ylabel() == "mean sequence log-prob"

    def test_anchor_weight_annotated_in_drift_ylabel(self, rl_run, close_all):
        # the run reports metadata["anchor_weight"] -> the drift panel's
        # y-label carries it (the brake's strength, right where its
        # penalty term is plotted)
        fig = plot_rl_introspection(rl_run)
        assert f"anchor_weight={rl_run.metadata['anchor_weight']}" in (
            fig.axes[2].get_ylabel()
        )
        assert "drift" in fig.axes[2].get_ylabel()

    def test_drift_panel_starts_at_exact_zero(self, rl_run, close_all):
        # panel 3's first y value is the float-exact 0.0: the agent begins
        # as a bit-exact copy of the frozen prior
        fig = plot_rl_introspection(rl_run)
        line = fig.axes[2].lines[0]
        assert next(iter(line.get_ydata())) == 0.0

    def test_ga_history_and_empty_history_raise_typed_errors(
        self, adaptive_run, close_all
    ):
        from pharmaml.generative import GenerationResult

        # a GA history records scores but no prior-drift fields
        with pytest.raises(DataValidationError, match="diagnostics"):
            plot_rl_introspection(adaptive_run)
        empty = GenerationResult(smiles=["CCO"], scores=[0.5])
        with pytest.raises(DataValidationError, match="no history"):
            plot_rl_introspection(empty)


# ---------------------------------------------------------------- reports
class TestTableExport:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "model": ["rf", "ridge"],
            "score": [0.91, 0.84],
            "note": ["ok", "linear"],  # exercise the str() path
        })

    def test_all_formats(self, tmp_path):
        frame = self._frame()
        for extension in ("csv", "tsv", "md", "json", "html"):
            out = export_table(frame, tmp_path / f"t.{extension}")
            assert out.exists() and out.stat().st_size > 0, extension

    def test_csv_roundtrip(self, tmp_path):
        out = export_table(self._frame(), tmp_path / "t.csv")
        back = pd.read_csv(out)
        assert list(back["model"]) == ["rf", "ridge"]

    def test_markdown_table_shape(self, tmp_path):
        out = export_table(self._frame(), tmp_path / "t.md")
        lines = out.read_text().strip().splitlines()
        assert lines[0].startswith("| model")
        assert set(lines[1].replace("|", "").split()) == {"---"}

    def test_html_document(self, tmp_path):
        out = export_table(self._frame(), tmp_path / "t.html")
        text = out.read_text()
        assert text.startswith("<!DOCTYPE html>") and "<table>" in text

    def test_json_records(self, tmp_path):
        out = export_table(self._frame(), tmp_path / "t.json")
        payload = json.loads(out.read_text())
        assert payload[0]["model"] == "rf"

    def test_unknown_format(self, tmp_path):
        with pytest.raises(DataValidationError):
            export_table(self._frame(), tmp_path / "t.parquet")

    def test_max_rows_truncation(self, tmp_path):
        frame = pd.DataFrame({"a": range(50)})
        export_table(frame, tmp_path / "t.md", max_rows=5)
        body = (tmp_path / "t.md").read_text().splitlines()
        assert len([ln for ln in body if ln.startswith("|")]) == 7  # header+rule+5

    def test_max_rows_zero_is_not_falsy_shows_all(self, tmp_path):
        """Regression: ``if max_rows`` treated 0 as None and showed every
        row; ``max_rows=0`` is a valid show-nothing request."""
        frame = pd.DataFrame({"a": range(3)})
        export_table(frame, tmp_path / "t.html", max_rows=0)
        html_text = (tmp_path / "t.html").read_text()
        assert "<td>" not in html_text          # no data rows rendered
        assert "3 more row(s) truncated" in html_text


class TestHTMLReport:
    def test_full_report(self, trained_classifier, tmp_path):
        _result, report, _X_test, _y_test = trained_classifier
        page = (
            HTMLReport("Test report", subtitle="unit test")
            .add_metrics(report, heading="Metrics")
            .add_table(metrics_to_dataframe(report), heading="Metrics table")
            .add_text("Generated by pytest.", heading="Notes")
            .add_figure(plot_confusion_matrix(report), caption="Confusion")
        )
        out = page.save(tmp_path / "report.html")
        text = out.read_text()
        assert "<h1>Test report</h1>" in text
        assert "data:image/png;base64," in text
        assert "Confusion" in text
        assert text.count("<table>") == 1  # the add_table call
        assert text.count("<div class='metric'>") >= 8  # metric cards

    def test_add_metrics_rejects_junk(self):
        with pytest.raises(DataValidationError):
            HTMLReport("t").add_metrics(42)

    def test_svg_embed(self, trained_classifier, tmp_path):
        _, report, _, _ = trained_classifier
        page = HTMLReport("svg").add_figure(
            plot_confusion_matrix(report), fmt="svg"
        )
        text = page.render()
        assert "data:image/svg+xml;base64," in text

    def test_pdf_embed_rejected(self, trained_classifier):
        _, report, _, _ = trained_classifier
        with pytest.raises(DataValidationError):
            HTMLReport("t").add_figure(plot_confusion_matrix(report), fmt="pdf")

    def test_empty_report_renders(self):
        text = HTMLReport("empty").render()
        assert "<h1>empty</h1>" in text and "footer" in text


class TestHTMLReportContents:
    """(100x-136) The header's CONTENTS block -- the HTML twin of the
    PDF title page's mini table-of-contents (100x-135): every table
    and figure page name in the walking order the file writes them,
    each line the SAME string the page's own heading prints, skipped
    entirely when the report has no table/figure pages.

    (100x-137) Every line is a CLICKABLE anchor -- the page rides in
    an id-carrying wrapper (``<section class='page' id='pg-N'>`` for
    tables, ``<figure class='page' id='pg-N'>`` for figures) and the
    line links to it, so a click jumps the reader to the page."""

    def test_contents_lists_pages_in_walking_order(self, trained_classifier):
        _result, report, _X_test, _y_test = trained_classifier
        page = (
            HTMLReport("Contents report", subtitle="the twin")
            .add_metrics(report, heading="Metrics")
            .add_table(metrics_to_dataframe(report), heading="table 1 · board")
            .add_figure(plot_confusion_matrix(report), caption="figure 1 · cm")
            .add_figure(plot_confusion_matrix(report), caption="figure 2 · cm")
        )
        text = page.render()
        # the nav block with its semantic label + the heading
        # (100x-138) the nav is also the RETURN target -- id + tabindex
        # so the pages' "↑ Contents" links land the jump with focus
        assert (
            "<nav class='contents' id='contents' tabindex='-1' "
            "aria-label='Contents'>" in text
        )
        assert "<h2>Contents</h2>" in text
        # the lines in CALL order (the walking order the file writes
        # them), each the SAME string the page itself prints -- each
        # wrapped in the anchor that jumps to its page (100x-137)
        assert "<li><a href='#pg-1'>table 1 · board</a></li>" in text
        assert "<li><a href='#pg-2'>figure 1 · cm</a></li>" in text
        assert "<li><a href='#pg-3'>figure 2 · cm</a></li>" in text
        assert (
            text.index("<li><a href='#pg-1'>table 1 · board</a></li>")
            < text.index("<li><a href='#pg-2'>figure 1 · cm</a></li>")
            < text.index("<li><a href='#pg-3'>figure 2 · cm</a></li>")
        )
        # the page's own headings match their Contents lines (no drift)
        assert "<h2>table 1 · board</h2>" in text
        assert "<figcaption>figure 1 · cm</figcaption>" in text
        # the block sits in the HEADER: after the subtitle, before the
        # first content block (the metrics cards)
        assert text.index("aria-label='Contents'") < text.index("class='metrics'")

    def test_contents_lines_link_to_real_page_anchors(self, trained_classifier):
        """(100x-137) Every Contents line's href resolves to a REAL id
        the document carries -- the ids are minted at add time in
        lockstep with the names, so href and target can never drift.
        Tables ride in ``<section class='page'>`` wrappers, figures in
        ``<figure class='page'>`` (the figcaption's legitimate parent).
        """
        _result, report, _X_test, _y_test = trained_classifier
        text = (
            HTMLReport("anchors")
            .add_metrics(report, heading="Metrics")
            .add_table(metrics_to_dataframe(report), heading="board")
            .add_figure(plot_confusion_matrix(report), caption="cm")
            .render()
        )
        # the wrappers carry the ids, with tabindex so a fragment jump
        # lands focus on the page (the keyboard reader's arrival)
        assert "<section class='page' id='pg-1' tabindex='-1'>" in text
        assert "<figure class='page' id='pg-2' tabindex='-1'>" in text
        # every href in the Contents resolves -- no orphan links
        for page_id in ("pg-1", "pg-2"):
            assert f"href='#{page_id}'" in text
            assert f"id='{page_id}'" in text
        # the anchors are pure fragment links: the self-contained
        # contract holds (no http, no mailto, no companion files)
        assert "href='http" not in text
        assert "href='mailto" not in text

    def test_contents_anchor_css_rides_the_theme_variables(self):
        """(100x-137) The link styling, the jump landing ring and the
        wrapper geometry all ride the CSS variables, so both themes
        carry them; the figure wrapper resets the UA inset."""
        text = HTMLReport("styled").add_table(
            pd.DataFrame({"a": [1]}), heading="t"
        ).render()
        assert "ul.contents-list a { color: inherit;" in text
        assert "ul.contents-list a:hover, ul.contents-list a:focus-visible {" in text
        assert "color: var(--accent); text-decoration: underline; }" in text
        # (100x-138) the return leg's ring covers the nav too
        assert (
            "section.page:target, figure.page:target, nav.contents:target {" in text
        )
        assert "outline: 2px solid var(--accent);" in text
        assert "scroll-margin-top: 1.25rem;" in text
        assert "figure.page { margin: 0; }" in text

    def test_contents_skipped_without_table_or_figure_pages(self):
        text = (
            HTMLReport("prose only")
            .add_metrics({"a": 1})
            .add_text("just prose", heading="Notes")
            .render()
        )
        # no nav block, no Contents heading (the CSS comment's word
        # does not count -- the RENDERED document is what matters)
        assert "<nav class='contents'" not in text
        assert "<h2>Contents</h2>" not in text
        assert "<ul class='contents-list'>" not in text

    def test_contents_fallback_names_for_headingless_blocks(
        self, trained_classifier
    ):
        """A heading-less table / caption-less figure lists its
        positional fallback -- the SAME identity the figure's alt
        already carries; the anchor wrapper exists either way (the
        Contents line needs its jump target even without a heading)."""
        _result, report, _X_test, _y_test = trained_classifier
        text = (
            HTMLReport("fallbacks")
            .add_table(metrics_to_dataframe(report))
            .add_figure(plot_confusion_matrix(report))
            .render()
        )
        assert "<li><a href='#pg-1'>table 1</a></li>" in text
        assert "<li><a href='#pg-2'>figure 1</a></li>" in text
        # the figure's alt carries the same positional identity
        assert "alt='figure 1'" in text
        # heading-less pages still carry their anchor wrappers
        assert "<section class='page' id='pg-1' tabindex='-1'>" in text
        assert "<figure class='page' id='pg-2' tabindex='-1'>" in text

    def test_contents_styles_ride_the_theme_variables(self):
        text = HTMLReport("styled").add_table(
            pd.DataFrame({"a": [1]}), heading="t"
        ).render()
        # the panel rides the CSS variables so both themes carry it
        assert "nav.contents {" in text
        assert "background: var(--panel)" in text
        assert "ul.contents-list li::marker { color: var(--accent); }" in text

    def test_visual_bundle_report_gains_contents(self, tmp_path):
        """``save_visual_bundle`` composes through HTMLReport -- its
        report.html carries the Contents with the title-cased page
        names, automatically (no caller change)."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.figure import Figure

        figure = Figure()
        figure.add_subplot(111).plot([0, 1], [0, 1])
        board = pd.DataFrame({"model": ["rf"], "score": [0.9]})
        written = save_visual_bundle(
            {"confusion": figure},
            tables={"leaderboard": board},
            metrics={"a": 1},
            out_dir=tmp_path / "bundle",
            close=False,
        )
        assert written, "the bundle wrote nothing"
        text = (tmp_path / "bundle" / "report.html").read_text()
        assert "<nav class='contents'" in text
        assert "<li><a href='#pg-1'>Leaderboard</a></li>" in text
        assert "<li><a href='#pg-2'>Confusion</a></li>" in text
        assert text.index("<li><a href='#pg-1'>Leaderboard</a></li>") < text.index(
            "<li><a href='#pg-2'>Confusion</a></li>"
        )
        # (100x-137) the anchors resolve: the bundle's pages carry the
        # ids the Contents lines link to
        assert "id='pg-1'" in text and "id='pg-2'" in text


class TestHTMLReportTheme:
    """(100x-126) The report's theme engine: a ``theme`` constructor
    parameter bakes the initial ``data-theme``, every report carries a
    toggle button that cycles auto → light → dark at reading time
    (remembered per browser), and the CSS applies the dark palette via
    ``[data-theme='dark']`` (forced) or the ``prefers-color-scheme``
    media query (auto) — the self-contained contract holds."""

    def test_theme_validation(self):
        with pytest.raises(DataValidationError):
            HTMLReport("t", theme="blue")
        with pytest.raises(DataValidationError):
            HTMLReport("t", theme="AUTO")  # case-sensitive

    def test_baked_attribute_and_label_per_theme(self):
        labels = {"auto": "\u25d0 auto", "light": "\u2600 light", "dark": "\u263d dark"}
        for theme, label in labels.items():
            text = HTMLReport("t", theme=theme).render()
            assert f"<html lang='en' data-theme='{theme}'>" in text, theme
            assert f">{label}</button>" in text, theme

    def test_toggle_button_and_inline_engine(self):
        text = HTMLReport("t").render()
        assert "<button id='theme-toggle' class='theme-toggle'" in text
        assert "aria-label='Toggle colour theme'" in text
        # exactly ONE script, inline, and it persists the reader's pick
        assert text.count("<script>") == 1
        assert "localStorage" in text and "pharmaml-report-theme" in text
        assert text.index("<script>") < text.index("</body>")

    def test_css_engine_selectors(self):
        text = HTMLReport("t").render()
        # forced dark, the auto media query, and the explicit light pin
        assert "html[data-theme='dark']" in text
        assert "html[data-theme='auto']" in text
        assert "html[data-theme='light']" in text
        assert "@media (prefers-color-scheme: dark)" in text
        # the dark palette values ride BOTH the forced block and the
        # media block (auto keeps the original behaviour)
        assert text.count("--bg: #0B1220") == 2
        assert text.count("--accent: #14B8A6") == 2
        # the toggle rides the same variables as everything else
        assert ".theme-toggle { position: fixed;" in text
        assert "border-color: var(--accent)" in text

    def test_self_contained_contract_holds(self):
        # the theme engine adds ZERO external references: no http(s)
        # URLs anywhere, the only script is the inline engine
        text = HTMLReport("t").add_text("body").render()
        assert "http://" not in text and "https://" not in text
        assert "src='http" not in text and "<link" not in text

    def test_full_report_with_theme(self, trained_classifier, tmp_path):
        _result, report, _X_test, _y_test = trained_classifier
        page = HTMLReport("Themed", theme="dark").add_metrics(
            metrics_to_dataframe(report).to_dict(orient="list")
        )
        out = page.save(tmp_path / "themed.html")
        text = out.read_text()
        assert "data-theme='dark'" in text and "\u263d dark</button>" in text


class TestMetricsToDataFrame:
    def test_classification_report(self, trained_classifier):
        _, report, _, _ = trained_classifier
        frame = metrics_to_dataframe(report)
        assert set(frame.columns) == {"metric", "value"}
        assert "accuracy" in set(frame["metric"])

    def test_plain_dict(self):
        frame = metrics_to_dataframe({"a": 1})
        assert frame["value"].tolist() == [1]

    def test_rejects_junk(self):
        with pytest.raises(DataValidationError):
            metrics_to_dataframe("nope")


class TestVisualBundle:
    def test_bundle_artifacts(self, trained_classifier, tmp_path):
        _, report, _, _ = trained_classifier
        figures = {
            "confusion": plot_confusion_matrix(report),
            "roc": plot_confusion_curves_safe(trained_classifier),
        }
        tables = {"metrics": metrics_to_dataframe(report)}
        written = save_visual_bundle(
            figures=figures,
            tables=tables,
            metrics=report,
            out_dir=tmp_path / "bundle",
            title="Bundle test",
            formats=("png", "svg"),
        )
        names = {p.name for p in written}
        assert "report.html" in names
        assert {"confusion.png", "confusion.svg", "roc.png", "roc.svg"} <= names
        assert {"metrics.csv", "metrics.md"} <= names
        html = (tmp_path / "bundle" / "report.html").read_text()
        assert html.count("data:image/png;base64,") == 2

    def test_bundle_rejects_path_escape(self, trained_classifier, tmp_path):
        """Regression: a '../escape' stem key used to write OUTSIDE the
        bundle directory (``out / "../escape.png"``)."""
        _, report, _, _ = trained_classifier
        out_dir = tmp_path / "bundle"
        with pytest.raises(DataValidationError, match="plain file name"):
            save_visual_bundle(
                {"../escape": plot_confusion_matrix(report)}, out_dir=out_dir
            )
        with pytest.raises(DataValidationError, match="plain file name"):
            save_visual_bundle(
                {}, tables={"../escape": metrics_to_dataframe(report)},
                out_dir=out_dir,
            )
        # nothing escaped, and the bundle dir was never even created
        assert not (tmp_path / "escape.png").exists()
        assert not (tmp_path / "escape.csv").exists()
        assert not out_dir.exists()

    def test_bundle_closes_figures(self, trained_classifier, tmp_path):
        _, report, _, _ = trained_classifier
        fig = plot_confusion_matrix(report)
        save_visual_bundle(figures={"f": fig}, out_dir=tmp_path / "b2", close=True)
        import matplotlib.pyplot as plt

        assert fig not in [plt.figure(n) for n in plt.get_fignums()]

    def test_bundle_pdf_report(self, trained_classifier, tmp_path):
        """pdf_report=True adds the multi-page document, ordered second."""
        _, report, _, _ = trained_classifier
        written = save_visual_bundle(
            figures={"confusion": plot_confusion_matrix(report)},
            tables={"metrics": metrics_to_dataframe(report)},
            out_dir=tmp_path / "bpdf",
            pdf_report=True,
        )
        names = [p.name for p in written]
        assert names[0] == "report.html"
        assert names[1] == "report.pdf"
        assert (tmp_path / "bpdf" / "report.pdf").read_bytes()[:5] == b"%PDF-"

    def test_bundle_without_pdf_report_unchanged(self, trained_classifier, tmp_path):
        """Default False keeps the 1.2.23 artifact set (no report.pdf)."""
        _, report, _, _ = trained_classifier
        written = save_visual_bundle(
            figures={"confusion": plot_confusion_matrix(report)},
            out_dir=tmp_path / "bnopdf",
        )
        assert not any(p.name == "report.pdf" for p in written)


class TestSavePdfReport:
    """The multi-page PDF document -- the legacy scripts' ONLY report
    format (fpdf was declared in the reports extra since 1.0 but never
    wired to a function until now)."""

    @staticmethod
    def _reader(path):
        pypdf = pytest.importorskip("pypdf")
        return pypdf.PdfReader(str(path))

    def test_pages_metrics_table_and_figure(self, trained_classifier, tmp_path):
        _, report, _, _ = trained_classifier
        out = save_pdf_report(
            {"confusion": plot_confusion_matrix(report)},
            tables={"metrics": metrics_to_dataframe(report)},
            metrics=report,
            out_path=tmp_path / "report.pdf",
            title="PDF test",
        )
        assert out.read_bytes()[:5] == b"%PDF-"
        reader = self._reader(out)
        assert len(reader.pages) == 3  # title+metrics / table / figure
        page0 = reader.pages[0].extract_text() or ""
        assert "PDF test" in page0
        assert "accuracy" in page0  # a committed metric line
        assert len(reader.pages[2].images) == 1  # the embedded figure

    def test_unicode_title_transliterated_not_crashed(self, tmp_path):
        # Helvetica encodes latin-1 only; the em dash, >= and the arrow
        # must survive as ASCII instead of raising FPDFException.
        out = save_pdf_report(
            {},
            out_path=tmp_path / "uni.pdf",
            title="potency >= 5 — active -> shortlist",
        )
        reader = self._reader(out)
        assert len(reader.pages) == 1
        page0 = reader.pages[0].extract_text() or ""
        assert "potency >= 5" in page0 and "active -> shortlist" in page0

    def test_table_truncation_note(self, tmp_path):
        frame = pd.DataFrame({"a": range(80)})
        out = save_pdf_report({}, tables={"big": frame}, out_path=tmp_path / "big.pdf")
        reader = self._reader(out)
        assert len(reader.pages) == 2  # title / table -- the cap keeps ONE page
        page1 = reader.pages[1].extract_text() or ""
        assert "42 more rows" in page1
        assert "\n75\n" not in page1  # row 75 was truncated, not rendered

    def test_close_false_keeps_figure_open(self, trained_classifier, tmp_path):
        _, report, _, _ = trained_classifier
        fig = plot_confusion_matrix(report)
        save_pdf_report({"f": fig}, out_path=tmp_path / "open.pdf", close=False)
        import matplotlib.pyplot as plt

        assert fig in [plt.figure(n) for n in plt.get_fignums()]

    def test_path_escape_rejected(self, trained_classifier, tmp_path):
        _, report, _, _ = trained_classifier
        fig = plot_confusion_matrix(report)
        with pytest.raises(DataValidationError, match="plain file name"):
            save_pdf_report({"../escape": fig}, out_path=tmp_path / "x.pdf")
        with pytest.raises(DataValidationError, match="plain file name"):
            save_pdf_report({}, tables={"../escape": pd.DataFrame({"a": [1]})},
                            out_path=tmp_path / "x.pdf")
        assert not (tmp_path / "escape.png").exists()

    def test_fpdf_missing_raises(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "fpdf", None)
        with pytest.raises(DependencyError, match="fpdf"):
            save_pdf_report({}, out_path=tmp_path / "x.pdf")

    def test_title_page_contents_section(self, trained_classifier, tmp_path):
        """(100x-135) the title page's mini table-of-contents: every
        page name listed in the WALKING order (tables first, then
        figures), each line the SAME string the page's own heading
        prints -- the count-prefixed names read like the picklist's
        numbered rows.  Metrics still ride the title page below it."""
        _, report, _, _ = trained_classifier
        out = save_pdf_report(
            {"figure 2 · confusion": plot_confusion_matrix(report)},
            tables={"table 1 · metrics": metrics_to_dataframe(report)},
            metrics=report,
            out_path=tmp_path / "toc.pdf",
        )
        reader = self._reader(out)
        page0 = reader.pages[0].extract_text() or ""
        assert "Contents" in page0
        assert "Metrics" in page0  # the section below the contents list
        # the walking order: the table's line rides BEFORE the figure's
        assert page0.index("Table 1 · Metrics") < page0.index(
            "Figure 2 · Confusion"
        )
        # each contents line is the page's OWN heading (no drift)
        assert "Table 1 · Metrics" in (reader.pages[1].extract_text() or "")
        assert "Figure 2 · Confusion" in (reader.pages[2].extract_text() or "")

    def test_contents_skipped_when_nothing_to_list(self, tmp_path):
        """A report with no tables and no figures stays a pure title
        page -- a contents list of nothing is noise, skipped entirely."""
        out = save_pdf_report({}, out_path=tmp_path / "bare.pdf", title="Bare")
        page0 = self._reader(out).pages[0].extract_text() or ""
        assert "Bare" in page0
        assert "Contents" not in page0

    def test_contents_lists_a_shared_name_twice(self, tmp_path):
        """A table AND a figure sharing one name are TWO pages -- the
        contents list says so twice (a dict merge would silently drop
        one of them, and the list would lie about the page count)."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        try:
            frame = pd.DataFrame({"a": [1]})
            out = save_pdf_report(
                {"shared": fig},
                tables={"shared": frame},
                out_path=tmp_path / "twice.pdf",
                close=False,
            )
        finally:
            plt.close(fig)
        reader = self._reader(out)
        assert len(reader.pages) == 3  # title / table / figure
        page0 = reader.pages[0].extract_text() or ""
        assert page0.count("Shared") == 2  # listed twice, honestly
        assert "Shared" in (reader.pages[1].extract_text() or "")
        assert "Shared" in (reader.pages[2].extract_text() or "")


class TestExportDocx:
    """Word tables -- the legacy approved.py wrote a bare name list."""

    def test_roundtrip_table_with_header(self, tmp_path):
        frame = pd.DataFrame({"model": ["rf", "svm"], "score": [0.91, 0.87]})
        out = export_docx(frame, tmp_path / "board.docx")
        assert out.suffix == ".docx" and out.exists()
        docx = pytest.importorskip("docx")
        table = docx.Document(str(out)).tables[0]
        assert table.cell(0, 0).text == "model"
        assert table.cell(1, 0).text == "rf"
        assert table.cell(1, 1).text == "0.91"  # compact float formatting

    def test_max_rows_cap(self, tmp_path):
        frame = pd.DataFrame({"a": range(10)})
        out = export_docx(frame, tmp_path / "t.docx", max_rows=3)
        docx = pytest.importorskip("docx")
        assert len(docx.Document(str(out)).tables[0].rows) == 4  # header + 3

    def test_docx_missing_raises(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "docx", None)
        with pytest.raises(DependencyError, match="python-docx"):
            export_docx(pd.DataFrame({"a": [1]}), tmp_path / "t.docx")


def plot_confusion_curves_safe(trained_classifier):
    """ROC figure helper (fixture unpacking without pytest warnings)."""
    result, _, X_test, y_test = trained_classifier
    return plot_roc_curves(result.model, X_test, y_test)


class TestSaveFigure:
    def test_formats_and_dirs(self, tmp_path, trained_classifier):
        _, report, _, _ = trained_classifier
        fig = plot_confusion_matrix(report)
        for extension in ("png", "svg", "pdf"):
            out = save_figure(fig, tmp_path / "nested" / f"fig.{extension}")
            assert out.exists() and out.stat().st_size > 0

    def test_unknown_extension(self, tmp_path, trained_classifier):
        _, report, _, _ = trained_classifier
        with pytest.raises(DataValidationError):
            save_figure(plot_confusion_matrix(report), tmp_path / "f.gif")


class TestNotebookArtifacts:
    """The shipped notebooks must stay executed and error-free."""

    @staticmethod
    def _read(example_name: str):
        import nbformat

        path = Path(__file__).resolve().parents[1] / "examples" / example_name
        assert path.is_file(), f"examples/{example_name} missing from the ZIP"
        return nbformat.read(path, as_version=4)

    @staticmethod
    def _stats(nb):
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        errors = [
            o for c in code_cells for o in c.get("outputs", [])
            if o.output_type == "error"
        ]
        figures = sum(
            1 for c in code_cells for o in c.get("outputs", [])
            if o.get("data", {}).get("image/png")
        )
        executed = sum(1 for c in code_cells if c.get("execution_count"))
        return code_cells, errors, figures, executed

    def test_complete_ml_workflow_notebook(self):
        nb = self._read("complete_ml_workflow.ipynb")
        code_cells, errors, figures, executed = self._stats(nb)
        assert len(code_cells) >= 30
        assert executed == len(code_cells), "notebook has unexecuted code cells"
        assert not errors, f"notebook has {len(errors)} committed error output(s)"
        assert figures >= 14, "notebook lost its inline figures"

        sources = "\n".join(c.source for c in code_cells)
        # The feature-calculation section must stay shipped (§2).
        for symbol in (
            "compute_descriptors",
            "compute_fingerprints",
            "compute_maccs",
            "tanimoto_matrix",
            "plot_similarity_heatmap",
        ):
            assert symbol in sources, f"notebook lost its {symbol} section"
        desc_cell = next(
            (c for c in code_cells if "compute_descriptors" in c.source), None
        )
        assert desc_cell is not None
        desc_text = "".join(
            o.get("text", "") for o in desc_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "valid molecules" in desc_text, "descriptor cell lost its output"

        # §7/§8 classification recipes: binary (2 classes) + multiclass
        # with a selectable class count (N_CLASSES).  The demo tables load
        # from examples/data/ (qsar_classification.csv / qsar_multiclass.csv)
        # — never fabricated inline (see test_notebook_variants.py).
        for symbol in (
            'DATA_DIR / "qsar_classification.csv"',
            'DATA_DIR / "qsar_multiclass.csv"',
            "N_CLASSES",
        ):
            assert symbol in sources, f"notebook lost its {symbol} section"
        recipe_cell = next(
            (c for c in code_cells if "N_CLASSES = " in c.source), None
        )
        assert recipe_cell is not None, "notebook lost its selectable-K cell"
        recipe_text = "".join(
            o.get("text", "") for o in recipe_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "classes (4)" in recipe_text, "selectable-K cell lost its class census"

    def test_complete_docking_workflow_notebook(self):
        nb = self._read("complete_docking_workflow.ipynb")
        code_cells, errors, figures, executed = self._stats(nb)
        assert len(code_cells) >= 30
        assert executed == len(code_cells), "notebook has unexecuted code cells"
        assert not errors, f"notebook has {len(errors)} committed error output(s)"
        assert figures >= 2, "docking notebook lost its figures"

        sources = "\n".join(c.source for c in code_cells)
        for symbol in (
            # §1 CSV ligand import / §2 protein import & preparation
            "load_ligand_csv",
            "prepare_ligand_library",
            "distribute_receptor",
            "prepare_protein",
            # §3+ the original workflow
            "smiles_to_pdbqt",
            "compute_grid_box",
            "write_gpf",
            "run_vina",
            "batch_run_directories",
            "parse_dlg_summary",
            "check_docking_completion",
            "scan_docking_directory",
            "parse_vina_output",
            "ligand_efficiency_series",
            "consensus_rescore",
            "plot_energy_distribution",
            "plot_top_ligands",
            "save_visual_bundle",
        ):
            assert symbol in sources, f"docking notebook lost its {symbol} section"

        # The protein-preparation audit output must stay committed
        # (1262-atom lysozyme receptor, 78 waters removed).
        prep_cell = next(
            (c for c in code_cells if "protein = prepare_protein" in c.source), None
        )
        assert prep_cell is not None, "notebook lost its prepare_protein cell"
        prep_text = "".join(
            o.get("text", "") for o in prep_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "1262 atoms" in prep_text, "receptor audit output changed"
        assert "78 water" in prep_text, "water-removal accounting lost"

        # The ligand-library audit must stay committed too.
        load_cell = next(
            (c for c in code_cells if "library = load_ligand_csv" in c.source), None
        )
        assert load_cell is not None, "notebook lost its load_ligand_csv cell"
        load_text = "".join(
            o.get("text", "") for o in load_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "valid         : 8" in load_text, "ligand library audit changed"

        # §4's grid box must carry the fencepost-corrected sizes:
        # npts - 1 grid intervals, not npts (size 22.1 A, Vina 22.125 A).
        box_cell = next(
            (c for c in code_cells if "box = compute_grid_box" in c.source), None
        )
        assert box_cell is not None, "notebook lost its grid-box cell"
        box_text = "".join(
            o.get("text", "") for o in box_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "size   : (22.1, 22.1, 22.1) A" in box_text, (
            "grid-box size drifted from the fencepost-corrected 22.1 A"
        )
        assert "'size_x': 22.125" in box_text, "Vina grid size drifted from 22.125"

        # §6's completion audit runs on a scratch copy of the shipped
        # campaign (examples/data/docking_campaign/): 12 OK, 1 missing
        # log, 1 interrupted — exactly the planted audit cases.
        audit_cell = next(
            (c for c in code_cells
             if "audit = check_docking_completion" in c.source), None
        )
        assert audit_cell is not None, "notebook lost its completion-audit cell"
        audit_text = "".join(
            o.get("text", "") for o in audit_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "{'OK': 12, 'MISSING_LOG': 1, 'INCOMPLETE': 1}" in audit_text, (
            "campaign audit drifted from the planted 12/1/1 census"
        )

    def test_complete_pk_workflow_notebook(self):
        nb = self._read("complete_pk_workflow.ipynb")
        code_cells, errors, figures, executed = self._stats(nb)
        assert len(code_cells) >= 25
        assert executed == len(code_cells), "notebook has unexecuted code cells"
        assert not errors, f"notebook has {len(errors)} committed error output(s)"
        assert figures >= 2, "PK notebook lost its figures"

        sources = "\n".join(c.source for c in code_cells)
        for symbol in (
            # §1 data entry / validation
            "read_table",
            "require_columns",
            "validate_concentration_series",
            # §2–§7 the PK core
            "nca_summary",
            "auc_trapezoid",
            "aucm_trapezoid",
            "half_life",
            "elimination_rate_constant",
            "clearance",
            "volume_of_distribution",
            "oral_cmax",
            "iv_bolus_concentration",
            "iv_infusion_steady_state",
            "accumulation_ratio",
            "plot_dose_accumulation",
            # §8–§11 PD / dosage / patient / units
            "sigmoid_emax_effect",
            "ec50_for_target_effect",
            "maintenance_dose",
            "loading_dose",
            "creatinine_clearance",
            "renal_dose_adjustment",
            "tdm_trough_adjustment",
            "UnitConverter",
            "molar_mass",
            # §12 publish
            "plot_concentration_profile",
            "export_table",
            "save_visual_bundle",
        ):
            assert symbol in sources, f"PK notebook lost its {symbol} section"

        # The NCA numbers must stay committed (fixed-seed demo profile:
        # AUC 17.961 ng·h/mL, terminal half-life 1.623 h).
        nca_cell = next(
            (c for c in code_cells if "summary = nca_summary" in c.source), None
        )
        assert nca_cell is not None, "notebook lost its nca_summary cell"
        nca_text = "".join(
            o.get("text", "") for o in nca_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "AUC" in nca_text, "NCA cell lost its output"
        assert "17.961" in nca_text, "NCA AUC value changed"
        assert "1.623" in nca_text, "terminal half-life value changed"

        # §3 exposes the AUC math: linear vs log trapezoid on the same
        # shipped profile.  The log value is the post-round-20a
        # log-trapezoid rule (17.2536) — the committed output predating
        # the fix (17.2347) is long gone.
        auc_cell = next(
            (c for c in code_cells if "auc_log = auc_trapezoid" in c.source), None
        )
        assert auc_cell is not None, "notebook lost its AUC-math cell"
        auc_text = "".join(
            o.get("text", "") for o in auc_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "AUC linear         17.9607" in auc_text, "linear AUC pin drifted"
        assert "AUC log            17.2536" in auc_text, (
            "log AUC drifted from the log-trapezoid rule (17.2536)"
        )

        # The validation guard (§1) and the recap (§13) must stay shipped.
        assert any(
            c.cell_type == "markdown" and c.source.startswith("## 13 · Recap")
            for c in nb.cells
        ), "notebook lost its recap section"

    def test_complete_cellline_workflow_notebook(self):
        nb = self._read("complete_cellline_workflow.ipynb")
        code_cells, errors, figures, executed = self._stats(nb)
        assert len(code_cells) >= 15
        assert executed == len(code_cells), "notebook has unexecuted code cells"
        assert not errors, f"notebook has {len(errors)} committed error output(s)"
        assert figures >= 3, "cell-line notebook lost its figures"

        sources = "\n".join(c.source for c in code_cells)
        for symbol in (
            # §1–§5 response-screen core
            "load_response_table",
            "audit_response_table",
            "collapse_replicates",
            "ln_ic50_to_pic50",
            "response_matrix",
            "summarize_compounds",
            "call_sensitive_resistant",
            # §6–§7 omics + biomarkers
            "normalize_expression",
            "most_variable_genes",
            "sensitivity_associations",
            "top_biomarkers",
            "direction_of_association",
            # §8–§9 modelling + deep learning
            "train_model",
            "compare_models",
            "dnn",
            "dnn_residual",
            "plot_predicted_vs_actual",
            "plot_model_comparison",
            # §2b replicate variability (100x-71/72)
            "replicate_variability",
            "add_cv=True",
            "LN_IC50_replicate_cv_pct",
        ):
            assert symbol in sources, f"cell-line notebook lost its {symbol} section"

        # The replicate-CV verdict must stay committed: the demo pair
        # (same median 2.0, ten-fold different noise) is the section's
        # teaching output, and the shipped duplicate (MCF-7 x
        # crizotinib, byte-identical rows) collapses with CV 0.0.
        cv_cell = next(
            (c for c in code_cells if "replicate_variability(demo_screen)" in c.source),
            None,
        )
        assert cv_cell is not None, "notebook lost its replicate-CV cell"
        cv_text = "".join(
            o.get("text", "") for o in cv_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "70.710678" in cv_text, "demo pair A's CV changed"
        assert "7.071068" in cv_text, "demo pair B's CV changed"
        assert "MCF-7 crizotinib" in cv_text, "shipped duplicate pair vanished"
        assert "3.712163" in cv_text, "collapsed median drifted"
        cv_table = "".join(
            "".join(o.get("data", {}).get("text/plain", ""))
            for o in cv_cell.get("outputs", [])
            if o.get("output_type") == "execute_result"
        )
        assert "LN_IC50_replicate_cv_pct" in cv_table, "add_cv companion table lost"

        # The biomarker verdict must stay committed: the planted EGFR ->
        # erlotinib association is the notebook's headline result.
        verdict_cell = next(
            (c for c in code_cells if "EGFR vs erlotinib" in c.source), None
        )
        assert verdict_cell is not None, "notebook lost its biomarker verdict cell"
        verdict_text = "".join(
            o.get("text", "") for o in verdict_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "rho=+0.978" in verdict_text, "EGFR/erlotinib rho changed"
        assert "significant=True" in verdict_text, "planted biomarker no longer significant"

        # The §10 recap must stay shipped.
        assert any(
            c.cell_type == "markdown" and c.source.startswith("## 10 · Recap")
            for c in nb.cells
        ), "notebook lost its recap section"

    def test_complete_generative_workflow_notebook(self):
        nb = self._read("complete_generative_workflow.ipynb")
        code_cells, errors, figures, executed = self._stats(nb)
        assert len(code_cells) >= 18
        assert executed == len(code_cells), "notebook has unexecuted code cells"
        assert not errors, f"notebook has {len(errors)} committed error output(s)"
        assert figures >= 4, "generative notebook lost its figures"

        sources = "\n".join(c.source for c in code_cells)
        for symbol in (
            # §1 corpus + landscape
            "DEFAULT_CORPUS",
            "compute_descriptors",
            "plot_descriptor_distributions",
            # §2–§3 the objective library + the veto
            "QedObjective",
            "LogPBandObjective",
            "TpsaBandObjective",
            "MolecularWeightBandObjective",
            "SimilarityObjective",
            "SubstructureObjective",
            "NoveltyObjective",
            "MultiObjective",
            "score_with_breakdown",
            # §4–§6 the GA: de novo, seeded, scaffold-preserving
            "GeneticOptimizer",
            "seed_smiles",
            '"scaffold"',
            "MurckoScaffold",
            # §7 RL: prior + fine-tuning
            "SmilesRnn",
            "train_prior",
            "ReinventOptimizer",
            # §9–§10 head-to-head + publish
            "to_dataframe",
            "save_visual_bundle",
            # §6b adaptive operators — the learning-curve figure
            "plot_operator_trajectory",
            # §4b the crowding-distance shortlist (100x-70)
            "crowding_distance",
            "front.prune(8)",
            "title=\"Crowding-encoded front\"",
        ):
            assert symbol in sources, f"generative notebook lost its {symbol} section"

        # The corpus audit must stay committed: 244 SMILES, all unique.
        corpus_cell = next(
            (c for c in code_cells if "corpus size" in c.source), None
        )
        assert corpus_cell is not None, "notebook lost its corpus audit cell"
        corpus_text = "".join(
            o.get("text", "") for o in corpus_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "corpus size : 244 SMILES" in corpus_text, "corpus audit changed"
        assert "unique      : 244" in corpus_text, "corpus uniqueness lost"

        # The scaffold A/B verdict must stay committed: 12 seeds -> 7
        # frameworks, scaffold mode 100% retention vs uniform 50.5%.
        scaffold_cell = next(
            (c for c in code_cells if "seed_frameworks" in c.source), None
        )
        assert scaffold_cell is not None, "notebook lost its scaffold A/B cell"
        scaffold_text = "".join(
            o.get("text", "") for o in scaffold_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "7 distinct Murcko frameworks" in scaffold_text, (
            "seed framework census changed"
        )
        retention_text = "".join(
            "".join(o.get("data", {}).get("text/plain", ""))
            for o in scaffold_cell.get("outputs", [])
            if o.get("output_type") == "execute_result"
        )
        assert "scaffold    1.000" in retention_text, "scaffold retention changed"
        assert "uniform     0.505" in retention_text, "uniform retention changed"

        # The RL summary must stay committed.  Round 26 moved the RL to a
        # running-baseline advantage: on this seed the mean score keeps
        # improving to the last iteration, so the plateau early-stop stays
        # armed but never fires (early_stopped=False) and the run yields
        # 107 unique molecules with best score 0.601.
        rl_cell = next(
            (c for c in code_cells if "early stopped" in c.source), None
        )
        assert rl_cell is not None, "notebook lost its RL summary cell"
        rl_text = "".join(
            o.get("text", "") for o in rl_cell.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        assert "early stopped   : False" in rl_text, "RL early-stop flag changed"
        assert "107 unique molecule(s)" in rl_text, "RL unique-molecule count changed"
        assert "(score 0.601)" in rl_text, "RL best-score headline changed"

        # The §11 recap must stay shipped.
        assert any(
            c.cell_type == "markdown" and c.source.startswith("## 11 · Recap")
            for c in nb.cells
        ), "notebook lost its recap section"


@pytest.fixture()
def pareto_result():
    """The hand-derived 5-member front + 2 dominated molecules (no RDKit).

    Front crowdings over (qed, nov): M1/M2 +inf corners, M3 1.1, M4 0.9,
    M5 1.2 -- the pinned values TestParetoCrowding derives by hand.
    """
    from pharmaml.generative import GenerationResult

    return GenerationResult(
        smiles=["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
        scores=[0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5],
        breakdown=[{"qed": 0.9, "nov": 0.2},
                   {"qed": 0.4, "nov": 0.9},
                   {"qed": 0.85, "nov": 0.25},
                   {"qed": 0.5, "nov": 0.6},
                   {"qed": 0.6, "nov": 0.55},
                   {"qed": 0.45, "nov": 0.5},
                   {"qed": 0.8, "nov": 0.1}],
    )


class TestParetoFrontPlot:
    """plot_pareto_front's crowding encoding (the 100x-69 viz axis).

    Fail-first context: the front was a single marker-style line -- the
    crowding ranking (100x-67) had no visual surface, so a reader could
    not see WHICH front members were isolated trade-off anchors vs
    crowded middles. Now the line is the curve, interiors are sized by
    isolation, and the +inf corners are amber diamonds.
    """

    def test_legend_and_trace_structure(self, pareto_result, close_all):
        fig = plot_pareto_front(pareto_result, x="qed", y="nov")
        ax = fig.axes[0]
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend == [
            "dominated", "Pareto front (5)",
            "interior — size = crowding", "corner — +inf crowding",
        ]
        # the trade-off curve is connective tissue -- markers moved to the
        # crowding-encoded traces
        line = ax.lines[0]
        assert line.get_marker() == "None"
        labels = {c.get_label(): c for c in ax.collections}
        assert len(labels["dominated"].get_offsets()) == 2  # M6, M7
        assert len(labels["interior — size = crowding"].get_offsets()) == 3
        assert len(labels["corner — +inf crowding"].get_offsets()) == 2

    def test_interior_marker_sizes_follow_crowding(self, pareto_result, close_all):
        # interiors in x order: M4 (0.9), M5 (1.2), M3 (1.1) -> sizes
        # 34 + 140 x crowding: 160, 202, 188 -- the most isolated member
        # stands largest, the most crowded smallest
        fig = plot_pareto_front(pareto_result, x="qed", y="nov")
        interior = next(
            c for c in fig.axes[0].collections
            if c.get_label() == "interior — size = crowding"
        )
        assert interior.get_sizes().tolist() == [160.0, 202.0, 188.0]

    def test_corners_are_diamonds_in_amber(self, pareto_result, close_all):
        fig = plot_pareto_front(pareto_result, x="qed", y="nov")
        corner = next(
            c for c in fig.axes[0].collections
            if c.get_label() == "corner — +inf crowding"
        )
        # the diamond marker path has 4 outer vertices (vs the circle's
        # 25-point approximation)
        vertices = corner.get_paths()[0].vertices
        assert len(vertices) == 5  # 4 corners + closing point
        assert tuple(corner.get_facecolor()[0][:3]) != (0.0, 0.0, 0.0)

    def test_two_member_front_renders_corners_only(self, close_all):
        from pharmaml.generative import GenerationResult

        result = GenerationResult(
            smiles=["P", "Q"], scores=[0.9, 0.5],
            breakdown=[{"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}],
        )
        fig = plot_pareto_front(result, x="a", y="b")
        labels = {c.get_label() for c in fig.axes[0].collections}
        assert "interior — size = crowding" not in labels  # both are corners
        corner = next(
            c for c in fig.axes[0].collections
            if c.get_label() == "corner — +inf crowding"
        )
        assert len(corner.get_offsets()) == 2

    def test_title_and_axes_carry_the_objective_pair(
        self, pareto_result, close_all
    ) -> None:
        fig = plot_pareto_front(pareto_result, x="qed", y="nov", title="Trade")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "qed"
        assert ax.get_ylabel() == "nov"
        # the title is left-anchored (the house style for audit lines)
        assert ax.get_title(loc="left").startswith("Trade — 5 of 7 molecules")


class TestHTMLReportBackLinks:
    """Round 100x-138: the RETURN leg — every page wrapper ends in a
    small "↑ Contents" link back to the header's block (the nav carries
    its own id and tabindex, and flashes the same accent ring when the
    jump lands), so a reader deep in a long report is one click from the
    index.  The link is minted at add time with the wrapper: a page
    wrapper exists exactly when at least one Contents line does, so the
    back-link's target can never be missing.

    Round 100x-139: the link now carries a ``title`` tooltip — the
    pointer world gets the same confession the aria world always had
    (the link's own text names the destination; a hovering pointer had
    no clue).  One string, minted once (``_BACK_CONTENTS_LINK``), so
    the table and figure wrappers can never drift apart."""

    #: The full return-link string every wrapper ends in (100x-139).
    BACK_LINK = (
        "<a class='back-contents' href='#contents' "
        "title='Back to the contents block at the top of the report'>"
        "\u2191 Contents</a>"
    )

    def test_every_page_wrapper_ends_in_the_return_link(self, trained_classifier):
        _result, report, _X_test, _y_test = trained_classifier
        text = (
            HTMLReport("round trip")
            .add_metrics(report, heading="Metrics")
            .add_table(metrics_to_dataframe(report), heading="board")
            .add_figure(plot_confusion_matrix(report), caption="cm")
            .render()
        )
        # the nav is the RETURN TARGET: its own id + tabindex (a landing
        # aid, never a tab stop) so the jump lands focus there
        assert (
            "<nav class='contents' id='contents' tabindex='-1' "
            "aria-label='Contents'>" in text
        )
        # the table wrapper ends in the return link, after the table
        assert (
            "</table>" + self.BACK_LINK + "</section>" in text
        )
        # the figure wrapper ends in the return link, after the caption
        assert (
            "<figcaption>cm</figcaption>" + self.BACK_LINK + "</figure>" in text
        )
        # one return link per page wrapper, never more
        assert text.count(self.BACK_LINK) == 2
        # (100x-139) the tooltip confesses the jump target -- the exact
        # wording, on BOTH wrappers (the one-string discipline)
        assert text.count("title='Back to the contents block at the top of the report'") == 2
        # the return target is REAL (the same lockstep discipline the
        # Contents lines' hrefs carry) and the link is a pure fragment
        assert "id='contents'" in text
        assert "href='http" not in text

    def test_captionless_figures_carry_the_link_too(self, trained_classifier):
        _result, report, _X_test, _y_test = trained_classifier
        text = HTMLReport("no caption").add_figure(
            plot_confusion_matrix(report)
        ).render()
        # the img's own fallback alt rides, and the return link still
        # closes the wrapper (the Contents line needs its jump target
        # with or without a caption -- so does the return leg)
        assert "/>" + self.BACK_LINK + "</figure>" in text

    def test_the_return_link_styling_rides_the_theme_variables(self):
        text = HTMLReport("styled").add_table(
            pd.DataFrame({"a": [1]}), heading="t"
        ).render()
        # muted ink until hovered/focused (the Contents lines'
        # affordance discipline), right-aligned at the wrapper's foot
        assert (
            "a.back-contents { display: block; text-align: right; "
            "font-size: .75rem;" in text
        )
        assert "a.back-contents:hover, a.back-contents:focus-visible {" in text
        # the nav rides the SAME arrival ring + scroll margin the pages
        # do, so the return landing is never lost either
        assert (
            "section.page:target, figure.page:target, nav.contents:target {" in text
        )
        assert "section.page, figure.page, nav.contents { scroll-margin-top: 1.25rem; }" in text

    def test_no_back_links_without_the_contents_block(self):
        """The noise rule's other half: a prose-only report renders no nav
        and no back-link ELEMENTS (the static CSS rides every report,
        like every other rule — the RENDERED document is what counts)."""
        text = HTMLReport("prose only").add_text("just prose").render()
        assert "<nav class='contents'" not in text
        assert "href='#contents'" not in text
        assert "<section class='page'" not in text and "<figure class='page'" not in text

    def test_visual_bundle_gains_the_return_leg(self, tmp_path):
        """``save_visual_bundle`` composes through HTMLReport — its
        report.html gains the return links automatically (no caller
        change, the standing convention)."""
        from matplotlib.figure import Figure

        figure = Figure()
        figure.add_subplot(111).plot([0, 1], [0, 1])
        board = pd.DataFrame({"model": ["rf"], "score": [0.9]})
        written = save_visual_bundle(
            {"confusion": figure},
            tables={"leaderboard": board},
            metrics={"a": 1},
            out_dir=tmp_path / "bundle",
            close=False,
        )
        assert written, "the bundle wrote nothing"
        text = (tmp_path / "bundle" / "report.html").read_text()
        assert "<nav class='contents' id='contents' tabindex='-1'" in text
        assert text.count(self.BACK_LINK) == 2
        assert "id='pg-1'" in text and "id='pg-2'" in text
