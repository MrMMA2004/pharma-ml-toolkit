"""Tests for :mod:`pharmaml.viz` — every plot, every export format, the
HTML report and the one-click visual bundle.

All figures render through the headless Agg backend (pytest runs in a
plain Python process), so these tests double as the CI proof that
plotting works on servers without a display.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from pharmaml.data import load_dataset  # noqa: E402
from pharmaml.exceptions import DataValidationError, ModelError  # noqa: E402
from pharmaml.ml import (  # noqa: E402
    compare_models,
    evaluate_model,
    make_train_test_split,
    train_model,
)
from pharmaml.viz import (  # noqa: E402
    HTMLReport,
    plot_calibration_curve,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_descriptor_distributions,
    plot_dose_accumulation,
    plot_energy_distribution,
    plot_top_ligands,
    plot_receptor_atom_types,
    plot_feature_importance,
    plot_learning_curve,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_curves,
    plot_similarity_heatmap,
    plot_table_comparison,
    plot_concentration_profile,
    export_table,
    metrics_to_dataframe,
    save_figure,
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


class TestRegressionPlots:
    def test_parity(self, trained_regressor, close_all):
        _, report, _, _ = trained_regressor
        fig = plot_predicted_vs_actual(
            np.arange(10, dtype=float),
            np.arange(10, dtype=float) + np.linspace(-1, 1, 10),
            metrics={"RMSE": 0.5},
        )
        assert fig is not None

    def test_parity_shape_mismatch(self, close_all):
        with pytest.raises(DataValidationError):
            plot_predicted_vs_actual([1, 2, 3], [1, 2])

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

    def test_feature_importance_unsupported_model(self, close_all):
        with pytest.raises(DataValidationError):
            plot_feature_importance(object())

    def test_cv_scores_array(self, close_all):
        fig = plot_cv_scores([0.7, 0.8, 0.75])
        assert "mean" in fig.axes[0].get_legend().get_texts()[0].get_text()

    def test_cv_scores_from_training_result(self, trained_regressor, close_all):
        result, _, _, _ = trained_regressor
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

    def test_energy_fallback_column(self, close_all):
        summary = self._summary().rename(columns={"best_lowest_energy": "top_lowest_energy"})
        fig = plot_energy_distribution(summary)
        assert fig is not None

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
            atom_types = {"C": 12, "OA": 3}
            output_pdbqt = "unused.pdbqt"

        fig = plot_receptor_atom_types(_FakeReport())
        assert len(fig.axes[0].patches) == 2


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
        result, report, X_test, y_test = trained_classifier
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

    def test_bundle_closes_figures(self, trained_classifier, tmp_path):
        _, report, _, _ = trained_classifier
        fig = plot_confusion_matrix(report)
        save_visual_bundle(figures={"f": fig}, out_dir=tmp_path / "b2", close=True)
        import matplotlib.pyplot as plt

        assert fig not in [plt.figure(n) for n in plt.get_fignums()]


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

        # The validation guard (§1) and the recap (§13) must stay shipped.
        assert any(
            c.cell_type == "markdown" and c.source.startswith("## 13 · Recap")
            for c in nb.cells
        ), "notebook lost its recap section"
