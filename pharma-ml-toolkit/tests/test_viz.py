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
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_descriptor_distributions,
    plot_dose_accumulation,
    plot_energy_distribution,
    plot_top_ligands,
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_curves,
    plot_similarity_heatmap,
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
        assert len(code_cells) >= 18
        assert executed == len(code_cells), "notebook has unexecuted code cells"
        assert not errors, f"notebook has {len(errors)} committed error output(s)"
        assert figures >= 2, "docking notebook lost its figures"

        sources = "\n".join(c.source for c in code_cells)
        for symbol in (
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
