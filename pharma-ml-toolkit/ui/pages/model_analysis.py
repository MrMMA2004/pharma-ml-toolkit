"""Model Analysis: real training, real curves, off the GUI thread (16.8.6/16.14).

Pick a bundled QSAR dataset (or the table opened in Data Explorer), a
target column and a registry model; ``train_model`` runs in a
:class:`JobWorker` (QThreadPool) and the finished page plots REAL
predictions-vs-actual, residuals and -- only when the estimator
exposes them -- feature importances. Metrics come from the actual
held-out split; nothing on this page is fabricated.

Round 226 (the owner's studio-parity directive): the Part-3 §19
cross-validation interface migrated here from the legacy Tkinter
playground -- the **Compare models (CV)** board runs the WHOLE
available registry under ONE shared splitter (``compare_models_cv``)
with the fold count and the fold strategy chosen on the page
(auto / kfold / stratified / group / stratified_group / scaffold),
the leaderboard carries the formatted ``mean ± std (k-fold CV)``
score per row, and double-clicking a row opens the full §19 report
window: the fold-by-fold table with train-side mirrors, the per-metric
aggregates with their honest directions, the strategy/metric notes,
and the fold-by-fold PyQtGraph chart (fold scores, mean line, ± 1 std
band edges). The group-source detection is honest: scaffold CV needs
a SMILES-like column, grouped strategies need a group/cluster/series/
batch/plate/cohort column -- a doomed request is refused BEFORE any
job starts, never silently degraded.

Round 228 (the retirement round): the page keeps what a trained run
needs for everything downstream -- ``AnalysisOutcome`` now carries the
fitted estimator, the splits and the held-out metrics, exposed through
``last_run()`` for the ML Workbench page (ensembles, persistence,
applicability domain).  The **manual hyperparameters** table edits the
selected registry model's ``param_grid`` knobs by hand (blank keeps the
estimator's own default; the values ride ``train_model``'s
``model_params``), and four NATIVE diagnostic figures join the curves:
the **calibration curve** (predicted probability vs observed frequency
per bin -- ``sklearn.calibration`` under the job, drawn Qt-native), the
**learning curve** (train sizes vs CV score with the ± 1 std band
edges), and the **ROC / PR curves** for classifiers that expose
``predict_proba`` (one line per class, the macro scores in the title).
No matplotlib import anywhere on the page (§16.8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

if TYPE_CHECKING:
    from pharmaml.ml.cv_results import CVResult


def _parity_meta(y_test: Any, y_pred: Any) -> Any:
    """Per-point hover metadata for the regression parity plot (§21.5).

    The readout already shows actual and predicted (the plotted x/y);
    the added facts are the 0-based held-out row and the signed
    residual.  Out-of-range indexes answer ``{}`` -- hover metadata
    must never raise (the ``_molecule_meta`` contract, generative page).
    """
    test = np.asarray(y_test, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    def meta(i: int) -> dict[str, Any]:
        if not 0 <= i < test.size or i >= pred.size:
            return {}
        return {"row": i, "residual": float(pred[i] - test[i])}

    return meta


def _residual_meta(y_test: Any, y_pred: Any) -> Any:
    """Per-point hover metadata for the residual plot (§21.5).

    The residual plot shows predicted (x) and residual (y) -- both
    derivable from the parity pair -- so the fact this provider adds is
    the ACTUAL value behind the point, plus the same 0-based row the
    parity plot carries (one key, one meaning, across both panels).
    """
    test = np.asarray(y_test, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    def meta(i: int) -> dict[str, Any]:
        if not 0 <= i < test.size or i >= pred.size:
            return {}
        return {"row": i, "actual": float(test[i])}

    return meta


def _fold_meta(scores: Any, mean: float) -> Any:
    """Per-point hover metadata for the CV fold-score chart (§21.5).

    x/y show the fold and its score; the signed delta against the mean
    is the added fact (a fold above the mean reads positive).  Fold
    numbers are 1-based to match the chart's x axis.
    """
    values = np.asarray(scores, dtype=float)

    def meta(i: int) -> dict[str, Any]:
        if not 0 <= i < values.size:
            return {}
        return {"fold": i + 1, "delta vs mean": float(values[i] - mean)}

    return meta

_TK = Path(__file__).resolve().parents[2]
_DATASETS = {
    "qsar_regression.csv": "pIC50",
    "qsar_classification.csv": "activity",
    "walkthrough_classification.csv": "",
}

#: (round 226, §19.8 port) the fold strategies the Compare board accepts
#: -- ``auto`` resolves to stratified (classification) / kfold
#: (regression); the rest pass straight to ``compare_models_cv``.
_CV_STRATEGIES: tuple[str, ...] = (
    "auto",
    "kfold",
    "stratified",
    "group",
    "stratified_group",
    "scaffold",
)

#: SMILES-column candidates (the docking module's list, mirrored by the
#: playground's detector): name-based, case-insensitive.
_SMILES_COLUMN_CANDIDATES: tuple[str, ...] = (
    "smiles",
    "smile",
    "canonical_smiles",
    "smiles_nostereo",
    "smiles_no_stereo",
    "smiles_string",
    "canonical",
)

#: Column-name tokens that mark a GROUP source -- the replicate/cluster
#: labels the grouped CV strategies keep inside one side of every fold.
_GROUP_COLUMN_TOKENS: tuple[str, ...] = (
    "group",
    "cluster",
    "series",
    "batch",
    "plate",
    "cohort",
)


def detect_smiles_column(frame: pd.DataFrame) -> str | None:
    """Find the SMILES column of a loaded table, or ``None`` when absent.

    Name-based and case-insensitive over the same candidate list the
    docking module uses; the honest ``None`` (the engine's
    ``csv_standardize`` detector raises instead -- the GUI wants a
    quiet refusal it can show before any job starts).
    """
    lowered = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in _SMILES_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def detect_group_column(frame: pd.DataFrame) -> str | None:
    """Find a group-like column for the grouped CV strategies, or ``None``.

    A column whose (case-insensitive, ``_``/space-stripped) name
    contains a group token (``group`` / ``cluster`` / ``series`` /
    ``batch`` / ``plate`` / ``cohort``) and holds 2..n_rows/2 distinct
    values -- one label per replicate set, not a unique-per-row ID and
    not a constant.
    """
    n_rows = len(frame)
    for column in frame.columns:
        flat = str(column).strip().lower().replace(" ", "_")
        if any(token in flat for token in _GROUP_COLUMN_TOKENS):
            distinct = frame[column].nunique()
            if 2 <= distinct <= max(2, n_rows // 2):
                return str(column)
    return None


def numeric_feature_columns(
    frame: pd.DataFrame, target: str
) -> tuple[list[str], list[str]]:
    """Split a dataset's non-target columns into ``(numeric, dropped)``.

    The train flows' shared rule: numeric columns become features;
    everything else -- SMILES, IDs, free text -- is reported as dropped
    so the page can say WHY those columns never reached the models.
    """
    numeric = [
        column
        for column in frame.columns
        if column != target and pd.api.types.is_numeric_dtype(frame[column])
    ]
    dropped = [
        str(column)
        for column in frame.columns
        if column != target and column not in numeric
    ]
    return numeric, dropped


def detect_task(frame: pd.DataFrame, target: str) -> str:
    """Regression when the target is numeric with > 10 distinct values."""
    y = frame[target]
    numeric_y = pd.api.types.is_numeric_dtype(y)
    classes = y.astype(str).nunique()
    return "regression" if numeric_y and classes > 10 else "classification"


@dataclass
class AnalysisOutcome:
    """What the background job hands back (all real numbers).

    Round 228: the fitted ``estimator`` and the splits ride along so
    the ML Workbench (ensembles, persistence, applicability domain)
    and the diagnostic figures can reuse the SAME run instead of
    retraining -- the page's ``last_run()`` hands this to whoever
    needs it, estimators never crossing a thread boundary (the job
    finishes before the reference is read).
    """

    task: str = ""
    model: str = ""
    n_train: int = 0
    n_test: int = 0
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    metric_name: str = ""
    metric_value: float = float("nan")
    cv_mean: float = float("nan")
    importance_names: list[str] = field(default_factory=list)
    importance_values: np.ndarray | None = None
    seconds: float = 0.0
    estimator: object | None = None
    X_train: object | None = None
    y_train: object | None = None
    X_test: object | None = None
    feature_names: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class ModelAnalysisPage(QWidget):
    """Train a registry model on a real dataset and inspect the fit."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
        data_explorer: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._data_explorer = data_explorer
        self._frame: pd.DataFrame | None = None
        self._last_outcome: AnalysisOutcome | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self._dataset_combo = QComboBox()
        for name in _DATASETS:
            self._dataset_combo.addItem(name)
        self._dataset_combo.addItem("(table from Data Explorer)")
        self._dataset_combo.currentTextChanged.connect(self._load_dataset)
        self._target_combo = QComboBox()
        self._model_combo = QComboBox()
        try:
            from pharmaml.ml.models import MODEL_REGISTRY

            for key in sorted(MODEL_REGISTRY):
                self._model_combo.addItem(key)
            self._model_combo.setCurrentText("random_forest")
        except Exception:  # noqa: BLE001 - broken-install fallback to the default entry
            self._model_combo.addItem("random_forest")
        self._train_btn = QPushButton("Train")
        self._train_btn.setMinimumHeight(34)
        self._train_btn.clicked.connect(self._start_training)
        # -- the manual hyperparameters table (round 228) -------------- #
        self._params_table = QTableWidget(0, 2)
        self._params_table.setHorizontalHeaderLabels(["param", "value"])
        self._params_table.verticalHeader().setVisible(False)
        self._params_table.setMaximumHeight(150)
        self._params_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._params_table.setToolTip(
            "The selected registry model's param_grid knobs, editable by "
            "hand — each value cell starts EMPTY (the estimator's own "
            "default; blank keeps that default). Pinned keys are simply "
            "passed to train_model's model_params; ints, floats, None, "
            "true/false and strings parse honestly."
        )
        self._model_combo.currentTextChanged.connect(self._rebuild_param_rows)
        self._reset_params_btn = QPushButton("Reset to defaults")
        self._reset_params_btn.setToolTip(
            "Clear every value cell back to the estimator's own defaults "
            "(the blank state — nothing is pinned)."
        )
        self._reset_params_btn.clicked.connect(self._clear_param_values)
        # -- the §19 CV controls (round 226 port) ------------------------- #
        self._cv_folds = QSpinBox()
        self._cv_folds.setRange(2, 10)
        self._cv_folds.setValue(5)
        self._cv_folds.setToolTip(
            "Cross-validation folds for the Compare-models leaderboard's "
            "full §19 CV report (fold scores, mean ± std, per-model "
            "detail). 5 = the directive's default; 3 for expensive "
            "models or limited data."
        )
        self._cv_strategy = QComboBox()
        for token in _CV_STRATEGIES:
            self._cv_strategy.addItem(token)
        self._cv_strategy.setCurrentText("auto")
        self._cv_strategy.setToolTip(
            "How the Compare-models CV folds are drawn "
            "(pharmaml.ml.cv_results): auto -- stratified for "
            "classification, kfold for regression; group / "
            "stratified_group -- replicate rows never straddle folds "
            "(needs a group-like column; SMILES column groups by "
            "scaffold); scaffold -- Bemis-Murcko frameworks never span "
            "train/validation folds, the honest structural-"
            "generalisation estimate (expect LOWER scores than "
            "row-level CV -- that gap is the optimism of random folds "
            "made visible). Double-click a leaderboard row for the full "
            "§19 fold-by-fold report."
        )
        self._compare_btn = QPushButton("Compare models (CV)")
        self._compare_btn.setMinimumHeight(34)
        self._compare_btn.clicked.connect(self._start_compare)
        form.addRow("Dataset:", self._dataset_combo)
        form.addRow("Target column:", self._target_combo)
        form.addRow("Model:", self._model_combo)
        params_buttons = QHBoxLayout()
        params_buttons.addWidget(self._reset_params_btn)
        params_buttons.addStretch(1)
        form.addRow("Manual hyperparameters:", self._params_table)
        form.addRow("", params_buttons)
        form.addRow("cv folds:", self._cv_folds)
        form.addRow("cv strategy:", self._cv_strategy)
        buttons = QHBoxLayout()
        buttons.addWidget(self._train_btn)
        buttons.addWidget(self._compare_btn)
        self._uncertainty_btn = QPushButton("Uncertainty…")
        self._uncertainty_btn.clicked.connect(self._show_uncertainty)
        buttons.addWidget(self._uncertainty_btn)
        self._calibration_btn = QPushButton("Probability recalibration…")
        self._calibration_btn.clicked.connect(self._show_calibration)
        buttons.addWidget(self._calibration_btn)
        self._validation_btn = QPushButton("Nested validation…")
        self._validation_btn.clicked.connect(self._show_validation)
        buttons.addWidget(self._validation_btn)
        buttons.addStretch(1)
        form.addRow("", buttons)
        lay.addLayout(form)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._pred_plot = PlotPanel(title="Predicted vs actual (held-out split)")
        self._resid_plot = PlotPanel(title="Residuals vs predicted")
        splitter.addWidget(self._pred_plot)
        splitter.addWidget(self._resid_plot)
        splitter.setSizes([420, 420])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._importance_plot = PlotPanel(title="Feature importance (when the estimator exposes it)")
        self._importance_plot.setMinimumHeight(220)
        lay.addWidget(self._importance_plot)
        self._metric_label = QLabel("No training run yet.")
        self._metric_label.setObjectName("studioStatus")
        self._metric_label.setWordWrap(True)
        lay.addWidget(self._metric_label)

        # (R30-c) per-page chart export under the live palette — one
        # row per plot, all reporting into the metric line
        from ..widgets.chart_export import attach_chart_export

        _export_row = QHBoxLayout()
        attach_chart_export(
            self, self._pred_plot, "model-analysis-parity",
            layout=_export_row, status_label=self._metric_label,
        )
        attach_chart_export(
            self, self._resid_plot, "model-analysis-residuals",
            layout=_export_row, status_label=self._metric_label,
        )
        attach_chart_export(
            self, self._importance_plot, "model-analysis-importance",
            layout=_export_row, status_label=self._metric_label,
        )
        _export_row.addStretch(1)
        lay.addLayout(_export_row)

        # -- the native diagnostic figures (round 228) ------------------ #
        diag_buttons = QHBoxLayout()
        self._diag_buttons: list[QPushButton] = []
        for label, handler, tip in (
            (
                "Calibration curve",
                self._start_calibration,
                (
                    "The reliability diagram of the last trained CLASSIFIER: "
                    "predicted probability vs observed frequency per bin "
                    "(sklearn.calibration under the job, drawn Qt-native) — "
                    "the honest answer to 'can I trust p = 0.7?'."
                ),
            ),
            (
                "Learning curve",
                self._start_learning_curve,
                (
                    "Train-set size vs CV score (sklearn.learning_curve under "
                    "the job): the honest answer to 'should we assay 500 more "
                    "compounds?' — a curve still climbing means yes, a plateau "
                    "means no. The ± 1 std band edges ride the dashed lines."
                ),
            ),
            (
                "ROC curves",
                self._start_roc,
                (
                    "One-vs-rest ROC per class of the last trained classifier "
                    "(needs predict_proba); the macro AUC rides the title."
                ),
            ),
            (
                "PR curve",
                self._start_pr,
                (
                    "One-vs-rest precision-recall per class of the last trained "
                    "classifier (needs predict_proba); the macro average "
                    "precision rides the title."
                ),
            ),
        ):
            btn = QPushButton(label)
            btn.setMinimumHeight(30)
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            self._diag_buttons.append(btn)
            diag_buttons.addWidget(btn)
        diag_buttons.addStretch(1)
        lay.addLayout(diag_buttons)
        self._diag_plot = PlotPanel(title="Diagnostics (calibration / learning curve / ROC / PR)")
        self._diag_plot.setMinimumHeight(230)
        lay.addWidget(self._diag_plot)
        self._diag_status = QLabel(
            "The diagnostic figures need a trained run — Train a model first."
        )
        self._diag_status.setObjectName("studioStatus")
        self._diag_status.setWordWrap(True)
        lay.addWidget(self._diag_status)

        # -- the §19 CV leaderboard (round 226 port) ---------------------- #
        self._cv_results: dict[str, CVResult] = {}
        self._board_status = QLabel(
            "Compare models (CV) cross-validates the available registry "
            "under ONE shared splitter — double-click a result row for "
            "the fold-by-fold §19 report."
        )
        self._board_status.setObjectName("studioStatus")
        self._board_status.setWordWrap(True)
        lay.addWidget(self._board_status)
        self._board = QTableWidget()
        self._board.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._board.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._board.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._board.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self._board.setWordWrap(False)
        self._board.itemDoubleClicked.connect(self._open_cv_report)
        self._board.setMinimumHeight(140)
        self._board.setMaximumHeight(260)
        lay.addWidget(self._board)

        self._load_dataset(self._dataset_combo.currentText())
        self._rebuild_param_rows(self._model_combo.currentText())

    # -- data ---------------------------------------------------------------
    def _show_calibration(self) -> None:
        from .calibration_dialog import CalibrationDialog

        dialog = getattr(self, "_calibration_dialog", None)
        if dialog is None:
            dialog = CalibrationDialog(self._runner, self)
            self._calibration_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _show_validation(self) -> None:
        from .validation_dialog import ValidationDialog

        dialog = getattr(self, "_validation_dialog", None)
        if dialog is None:
            dialog = ValidationDialog(self._runner, self)
            self._validation_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _show_uncertainty(self) -> None:
        from .uncertainty_dialog import UncertaintyDialog

        # Keep one instance per page, including while a background job finishes.
        dialog = getattr(self, "_uncertainty_dialog", None)
        if dialog is None:
            dialog = UncertaintyDialog(self._runner, self)
            self._uncertainty_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _load_dataset(self, name: str) -> None:
        self._target_combo.clear()
        if name == "(table from Data Explorer)":
            frame = None
            if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
                frame = self._data_explorer.frame()
            if frame is None or frame.empty:
                self._frame = None
                self._metric_label.setText("Data Explorer has no table loaded yet.")
                return
            self._frame = frame
        else:
            path = _TK / "examples" / "data" / name
            if not path.exists():  # pragma: no cover - missing example
                self._frame = None
                return
            from pharmaml.utils.io import read_table

            self._frame = read_table(path)
        default = _DATASETS.get(name, "")
        for col in self._frame.columns:
            self._target_combo.addItem(str(col))
        if default and default in [str(c) for c in self._frame.columns]:
            self._target_combo.setCurrentText(default)

    # -- training ------------------------------------------------------------
    def _on_started(self, name: str) -> None:
        self._train_btn.setEnabled(False)
        self._compare_btn.setEnabled(False)
        for btn in self._diag_buttons:
            btn.setEnabled(False)
        self._metric_label.setText(f"Training {name}…")

    def _on_ended(self) -> None:
        self._train_btn.setEnabled(True)
        self._compare_btn.setEnabled(True)
        for btn in self._diag_buttons:
            btn.setEnabled(True)

    def last_run(self) -> AnalysisOutcome | None:
        """The last rendered training run (estimator + splits + metrics).

        Round 228's hand-off contract for the ML Workbench page — the
        ensembles, persistence and applicability-domain cards rebuild
        from THIS run instead of retraining.  ``None`` before the first
        Train (the callers' pre-flight says so honestly).
        """
        return self._last_outcome

    def apply_search_result(self, family: str, params: dict) -> str:
        """Point the page at a tuner's winning family + configuration.

        The ML Workbench's ``Apply best`` calls this: registry families
        switch the model combo and pin the winning values into the
        manual-hyperparameters table; families the registry does not
        carry (``dnn`` — the Deep Learning page owns its configuration)
        answer with the honest note instead of a silent no-op.  Returns
        the status line the caller shows.
        """
        available = [
            self._model_combo.itemText(index) for index in range(self._model_combo.count())
        ]
        if family in available:
            self._model_combo.setCurrentText(family)
            self._rebuild_param_rows(family)
            pinned = 0
            for row in range(self._params_table.rowCount()):
                key_item = self._params_table.item(row, 0)
                value_item = self._params_table.item(row, 1)
                if key_item is None or value_item is None:
                    continue
                name = str(key_item.text())
                if name in params:
                    value_item.setText(str(params[name]))
                    pinned += 1
            return f"applied: {family} with {pinned} pinned param(s)"
        return (
            f"the {family} family trains on the Deep Learning page — the "
            "winning configuration rides the search summary line"
        )

    # -- manual hyperparameters (round 228) -------------------------------- #
    def _rebuild_param_rows(self, model: str) -> None:
        """Re-fill the params table with the model's param_grid knobs."""
        self._params_table.setRowCount(0)
        try:
            from pharmaml.ml.models import MODEL_REGISTRY

            spec = MODEL_REGISTRY.get(model)
            grid = dict(getattr(spec, "param_grid", {}) or {})
        except Exception:  # noqa: BLE001 - broken-install fallback: empty table
            grid = {}
        self._params_table.setRowCount(len(grid))
        for row, (name, values) in enumerate(grid.items()):
            shown = ", ".join(str(value) for value in values)
            key_item = QTableWidgetItem(str(name))
            key_item.setToolTip(f"registry grid: {shown}")
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._params_table.setItem(row, 0, key_item)
            value_item = QTableWidgetItem("")
            value_item.setToolTip("blank keeps the estimator's own default")
            self._params_table.setItem(row, 1, value_item)

    def _clear_param_values(self) -> None:
        """Blank every value cell — nothing pinned, all defaults."""
        for row in range(self._params_table.rowCount()):
            item = self._params_table.item(row, 1)
            if item is not None:
                item.setText("")

    def _collect_model_params(self) -> dict[str, object]:
        """Parse the value cells into ``train_model``'s ``model_params``.

        Blank cells are skipped (the estimator's own default); ints,
        floats, ``None``, ``true``/``false`` and strings parse honestly
        — anything else rides as the raw text.
        """
        params: dict[str, object] = {}
        for row in range(self._params_table.rowCount()):
            key_item = self._params_table.item(row, 0)
            value_item = self._params_table.item(row, 1)
            if key_item is None or value_item is None:
                continue
            text = value_item.text().strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered == "none":
                parsed: object = None
            elif lowered == "true":
                parsed = True
            elif lowered == "false":
                parsed = False
            else:
                try:
                    parsed = int(text)
                except ValueError:
                    try:
                        parsed = float(text)
                    except ValueError:
                        parsed = text
            params[str(key_item.text())] = parsed
        return params

    # -- the native diagnostic figures (round 228) ------------------------ #
    def _require_last_run(self) -> AnalysisOutcome | None:
        run = self._last_outcome
        if run is None or run.estimator is None or run.X_test is None:
            self._diag_status.setText(
                "The diagnostic figures need a trained run — Train a model first."
            )
            return None
        return run

    def _require_proba_run(self, figure: str) -> AnalysisOutcome | None:
        run = self._require_last_run()
        if run is None:
            return None
        if not hasattr(run.estimator, "predict_proba"):
            self._diag_status.setText(
                f"The {figure} needs a classifier that exposes "
                "predict_proba — the last run is a regression model "
                "(or a proba-less family). Train a classifier first."
            )
            return None
        return run

    def _submit_diag(self, name: str, job, on_finished) -> None:
        if self._runner.busy:
            self._diag_status.setText("A job is already running — Stop it first.")
            return
        self._runner.submit(
            name,
            job,
            on_finished=on_finished,
            on_failed=self._diag_failed,
            on_cancelled=self._diag_cancelled,
            on_progress=self._progress.update_progress,
        )

    def _start_calibration(self) -> None:
        """The reliability diagram of the last trained classifier."""
        run = self._require_proba_run("calibration curve")
        if run is None:
            return
        bins = 10
        estimator = run.estimator
        X_test = run.X_test
        y_true = np.asarray(run.y_test).astype(str)
        classes = list(run.classes) or sorted(set(y_true))
        positive = classes[-1] if classes else ""

        def job(progress):
            from sklearn.calibration import calibration_curve

            progress(0.3, "scoring probabilities")
            proba = estimator.predict_proba(X_test)[:, -1]
            observed = (y_true == positive).astype(float)
            fraction_positive, mean_predicted = calibration_curve(
                observed, proba, n_bins=bins, strategy="uniform"
            )
            progress(0.9, "done")
            return (
                np.asarray(mean_predicted, dtype=float),
                np.asarray(fraction_positive, dtype=float),
                float(proba.mean()),
            )

        self._submit_diag(
            f"calibration curve ({run.model})",
            job,
            self._render_calibration,
        )

    def _render_calibration(self, payload) -> None:
        mean_predicted, fraction_positive, mean_proba = payload
        self._diag_plot.clear()
        span = np.array([0.0, 1.0])
        self._diag_plot.line(span, span, label="perfect", width=1, index=2, style="dash")
        self._diag_plot.line(
            mean_predicted, fraction_positive, label="observed", width=2, index=0
        )
        self._diag_plot.set_labels(
            x="mean predicted probability", y="observed frequency", title="Calibration (reliability)"
        )
        self._diag_plot.add_legend()
        self._diag_status.setText(
            f"Calibration over {len(mean_predicted)} uniform bins — mean "
            f"predicted probability {mean_proba:.3f}; the dashed diagonal "
            "is perfect reliability."
        )

    def _start_learning_curve(self) -> None:
        """Train sizes vs CV score, the 'assay more?' picture."""
        run = self._require_last_run()
        if run is None:
            return
        estimator = run.estimator
        X_train = run.X_train
        y_train = np.asarray(run.y_train)
        task = run.task
        folds = int(self._cv_folds.value())

        def job(progress):
            from sklearn.model_selection import learning_curve

            progress(0.1, "sizing the ladder")
            scoring = "r2" if task == "regression" else "accuracy"
            sizes, train_scores, val_scores = learning_curve(
                estimator,
                X_train,
                y_train,
                train_sizes=np.linspace(0.2, 1.0, 5),
                cv=min(folds, 5),
                scoring=scoring,
                n_jobs=1,
            )
            progress(0.9, "done")
            return (
                np.asarray(sizes, dtype=float),
                train_scores.mean(axis=1),
                train_scores.std(axis=1),
                val_scores.mean(axis=1),
                val_scores.std(axis=1),
                scoring,
            )

        self._submit_diag(
            f"learning curve ({run.model})",
            job,
            self._render_learning_curve,
        )

    def _render_learning_curve(self, payload) -> None:
        sizes, train_mean, _train_std, val_mean, val_std, scoring = payload
        self._diag_plot.clear()
        self._diag_plot.line(sizes, train_mean, label="train", width=2, index=0)
        self._diag_plot.line(sizes, val_mean, label="CV", width=2, index=1)
        for edge, name in ((val_mean - val_std, "CV − 1 std"), (val_mean + val_std, "CV + 1 std")):
            self._diag_plot.line(sizes, edge, label=name, width=1, index=2, style="dash")
        self._diag_plot.set_labels(
            x="train rows", y=scoring, title="Learning curve (still climbing = assay more)"
        )
        self._diag_plot.add_legend()
        self._diag_status.setText(
            f"Learning curve over {len(sizes)} sizes × {min(int(self._cv_folds.value()), 5)} folds — "
            f"CV {scoring} {val_mean[-1]:.3f} ± {val_std[-1]:.3f} at the full "
            f"{int(sizes[-1])} rows."
        )

    def _start_roc(self) -> None:
        """One-vs-rest ROC per class of the last trained classifier."""
        run = self._require_proba_run("ROC curves")
        if run is None:
            return
        estimator = run.estimator
        X_test = run.X_test
        y_true = np.asarray(run.y_test).astype(str)
        classes = list(run.classes) or sorted(set(y_true))

        def job(progress):
            from sklearn.metrics import roc_auc_score, roc_curve

            progress(0.2, "scoring probabilities")
            proba = estimator.predict_proba(X_test)
            columns = list(getattr(estimator, "classes_", [])) or classes
            columns = [str(value) for value in columns]
            curves = []
            aucs = []
            for index, cls in enumerate(columns):
                truth = (y_true == cls).astype(int)
                scores = proba[:, index]
                fpr, tpr, _ = roc_curve(truth, scores)
                auc = roc_auc_score(truth, scores)
                curves.append((str(cls), np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)))
                aucs.append(float(auc))
                progress(0.2 + 0.7 * (index + 1) / len(columns), f"roc {cls}")
            return curves, float(np.mean(aucs)), len(columns)

        self._submit_diag(f"ROC curves ({run.model})", job, self._render_roc)

    def _render_roc(self, payload) -> None:
        curves, macro_auc, n_classes = payload
        self._diag_plot.clear()
        span = np.array([0.0, 1.0])
        self._diag_plot.line(span, span, label="chance", width=1, index=2, style="dash")
        for index, (cls, fpr, tpr) in enumerate(curves):
            self._diag_plot.line(fpr, tpr, label=str(cls), width=2, index=index % 2)
        self._diag_plot.set_labels(
            x="false-positive rate", y="true-positive rate", title=f"ROC (macro AUC {macro_auc:.3f})"
        )
        self._diag_plot.add_legend()
        self._diag_status.setText(
            f"One-vs-rest ROC over {n_classes} class(es) — macro AUC "
            f"{macro_auc:.3f}; the dashed diagonal is chance."
        )

    def _start_pr(self) -> None:
        """One-vs-rest precision-recall per class of the last classifier."""
        run = self._require_proba_run("PR curve")
        if run is None:
            return
        estimator = run.estimator
        X_test = run.X_test
        y_true = np.asarray(run.y_test).astype(str)
        classes = list(run.classes) or sorted(set(y_true))

        def job(progress):
            from sklearn.metrics import average_precision_score, precision_recall_curve

            progress(0.2, "scoring probabilities")
            proba = estimator.predict_proba(X_test)
            columns = list(getattr(estimator, "classes_", [])) or classes
            columns = [str(value) for value in columns]
            curves = []
            aps = []
            for index, cls in enumerate(columns):
                truth = (y_true == cls).astype(int)
                scores = proba[:, index]
                precision, recall, _ = precision_recall_curve(truth, scores)
                ap = average_precision_score(truth, scores)
                curves.append(
                    (str(cls), np.asarray(recall, dtype=float), np.asarray(precision, dtype=float))
                )
                aps.append(float(ap))
                progress(0.2 + 0.7 * (index + 1) / len(columns), f"pr {cls}")
            return curves, float(np.mean(aps)), len(columns), float(len(y_true))

        self._submit_diag(f"PR curve ({run.model})", job, self._render_pr)

    def _render_pr(self, payload) -> None:
        curves, macro_ap, n_classes, n_rows = payload
        self._diag_plot.clear()
        for index, (cls, recall, precision) in enumerate(curves):
            self._diag_plot.line(recall, precision, label=str(cls), width=2, index=index % 2)
        self._diag_plot.set_labels(
            x="recall", y="precision", title=f"PR (macro AP {macro_ap:.3f})"
        )
        self._diag_plot.add_legend()
        self._diag_status.setText(
            f"One-vs-rest precision-recall over {n_classes} class(es) — "
            f"macro average precision {macro_ap:.3f} on {int(n_rows)} held-out rows."
        )

    def _diag_failed(self, message: str) -> None:
        self._diag_status.setText(
            f"Diagnostic failed — {message.splitlines()[0] if message else 'error'}"
        )

    def _diag_cancelled(self, seconds: float) -> None:
        self._diag_status.setText(
            f"Diagnostic cancelled after {seconds:.1f}s (cooperative stop)."
        )

    def _start_training(self) -> None:
        if self._frame is None:
            self._metric_label.setText("Load a dataset first.")
            return
        frame = self._frame
        target = self._target_combo.currentText()
        model = self._model_combo.currentText()
        if not target or target not in frame.columns:
            self._metric_label.setText("Pick a target column.")
            return
        if self._runner.busy:
            self._metric_label.setText("A job is already running — Stop it first.")
            return

        task = detect_task(frame, target)
        model_params = self._collect_model_params()
        y = frame[target]
        X = frame.drop(columns=[target])
        X = X.select_dtypes(include=[np.number])

        from sklearn.model_selection import train_test_split

        stratify = None if task == "regression" else y.astype(str).to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )

        def job(progress) -> AnalysisOutcome:
            from pharmaml.ml.training import train_model

            progress(0.1, "loading")
            result = train_model(
                model,
                task,
                X_train,
                y_train,
                tune=False,
                preprocess=True,
                n_jobs=1,
                model_params=model_params or None,
            )
            progress(0.7, "predicting")
            fitted = result.model
            y_pred = fitted.predict(X_test)
            # (round 228) the honest dtype split: regression keeps floats,
            # classification keeps the label array AS-IS -- the old blanket
            # float cast crashed every string-class run ('could not convert
            # string to float: weak'), a latent bug the regression-only
            # tests never met.
            y_test_arr = np.asarray(y_test)
            y_pred_arr = np.asarray(y_pred)
            outcome = AnalysisOutcome(
                task=task,
                model=model,
                n_train=len(X_train),
                n_test=len(X_test),
                y_test=(y_test_arr.astype(float) if task == "regression" else y_test_arr),
                y_pred=(y_pred_arr.astype(float) if task == "regression" else y_pred_arr),
                cv_mean=(
                    float(result.cv_mean)
                    if result.cv_mean is not None and result.cv_mean == result.cv_mean
                    else float("nan")
                ),
                seconds=float(result.fit_seconds),
                estimator=fitted,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                feature_names=[str(name) for name in X.columns],
            )
            progress(0.85, "scoring")
            if task == "regression":
                from sklearn.metrics import mean_absolute_error, r2_score

                # The same float arrays the outcome carries (regression
                # branch); locals keep the metrics arithmetic off the
                # ndarray-|None union the dataclass allows.
                y_true = y_test_arr.astype(float)
                y_hat = y_pred_arr.astype(float)
                outcome.metric_name = "R² (held-out)"
                outcome.metric_value = float(r2_score(y_true, y_hat))
                outcome.metrics = {
                    "r2": outcome.metric_value,
                    "rmse": float(np.sqrt(np.mean((y_true - y_hat) ** 2))),
                    "mae": float(mean_absolute_error(y_true, y_hat)),
                }
            else:
                from sklearn.metrics import (
                    accuracy_score,
                    balanced_accuracy_score,
                    matthews_corrcoef,
                    precision_score,
                    recall_score,
                )

                truth = np.asarray(y_test).astype(str)
                predicted = np.asarray(y_pred).astype(str)
                outcome.metric_name = "accuracy (held-out)"
                outcome.metric_value = float(accuracy_score(truth, predicted))
                outcome.classes = sorted(set(truth) | set(predicted))
                outcome.metrics = {
                    "accuracy": outcome.metric_value,
                    "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                    "precision_macro": float(precision_score(truth, predicted, average="macro", zero_division=0)),
                    "recall_macro": float(recall_score(truth, predicted, average="macro", zero_division=0)),
                    "mcc": float(matthews_corrcoef(truth, predicted)),
                }
                if hasattr(fitted, "predict_proba"):
                    from sklearn.metrics import roc_auc_score

                    proba = fitted.predict_proba(X_test)
                    columns = [str(value) for value in getattr(fitted, "classes_", [])] or outcome.classes
                    try:
                        outcome.metrics["roc_auc"] = float(
                            roc_auc_score(truth, proba, multi_class="ovr", labels=columns)
                        )
                    except ValueError:  # single-class held-out split: honest skip
                        pass
            estimator = fitted.steps[-1][1] if hasattr(fitted, "steps") else fitted
            names = list(X.columns)
            if hasattr(estimator, "feature_importances_"):
                outcome.importance_names = names
                outcome.importance_values = np.asarray(estimator.feature_importances_, dtype=float)
            elif hasattr(estimator, "coef_"):
                coef = np.asarray(estimator.coef_, dtype=float)
                if coef.ndim == 1:
                    outcome.importance_names = names
                    outcome.importance_values = np.abs(coef)
            progress(1.0, "done")
            return outcome

        self._runner.submit(
            f"{model} on {target}",
            job,
            on_finished=self._render,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    # -- rendering --------------------------------------------------------------
    def _render(self, outcome: AnalysisOutcome) -> None:
        self._last_outcome = outcome
        self._pred_plot.clear()
        self._resid_plot.clear()
        self._importance_plot.clear()
        o: AnalysisOutcome = outcome
        if o.y_test is None or o.y_pred is None:
            return
        if o.task == "regression":
            lo = float(min(o.y_test.min(), o.y_pred.min()))
            hi = float(max(o.y_test.max(), o.y_pred.max()))
            span = np.array([lo, hi])
            self._pred_plot.line(span, span, label="y = x", width=1, index=1, style="dash")
            self._pred_plot.scatter(
                o.y_test, o.y_pred, label="held-out samples", meta=_parity_meta(o.y_test, o.y_pred)
            )
            self._pred_plot.set_labels(x="actual", y="predicted")
            self._resid_plot.scatter(
                o.y_pred, o.y_test - o.y_pred, meta=_residual_meta(o.y_test, o.y_pred)
            )
            self._resid_plot.line(
                np.array([float(o.y_pred.min()), float(o.y_pred.max())]),
                np.zeros(2),
                width=1,
                index=1,
                style="dash",
            )
            self._resid_plot.set_labels(x="predicted", y="residual")
        else:
            classes = sorted(set(np.asarray(o.y_test).astype(str)) | set(np.asarray(o.y_pred).astype(str)))
            matrix = np.zeros((len(classes), len(classes)), dtype=float)
            for actual, predicted in zip(np.asarray(o.y_test).astype(str), np.asarray(o.y_pred).astype(str)):
                matrix[classes.index(actual), classes.index(predicted)] += 1
            self._pred_plot.bars(matrix.sum(axis=1), labels=classes)
            self._pred_plot.set_labels(x="class", y="count", title="Actual class counts")
            self._importance_plot.set_labels(x="feature", y="|coefficient| / importance")
        if o.importance_values is not None and len(o.importance_names):
            order = np.argsort(o.importance_values)[::-1][:12]
            self._importance_plot.bars(
                o.importance_values[order],
                labels=[o.importance_names[i] for i in order],
            )
            self._importance_plot.set_labels(x="feature", y="importance")
        cv = f" · CV mean {o.cv_mean:.3f}" if o.cv_mean == o.cv_mean else ""
        self._metric_label.setText(
            f"{o.model} ({o.task}) · train {o.n_train} / test {o.n_test} · "
            f"{o.metric_name} {o.metric_value:.3f}{cv} · fitted in {o.seconds:.1f}s"
        )
        self._diag_status.setText(
            f"Last run: {o.model} ({o.task}) — the calibration / learning-curve / "
            "ROC / PR figures below reuse THIS fit."
        )

    # -- the §19 compare board (round 226 port) ----------------------------- #
    def _start_compare(self) -> None:
        """Cross-validate the available registry under ONE shared splitter.

        The honest sourcing mirrors the playground's Compare card:
        ``scaffold`` derives fold groups from the SMILES column's
        Bemis–Murcko frameworks (refused before any job when the table
        carries none); ``group``/``stratified_group`` use a detected
        group-like column (same pre-flight refusal); ``auto`` lets the
        engine pick stratified/kfold from the task.
        """
        if self._frame is None:
            self._metric_label.setText("Load a dataset first.")
            return
        frame = self._frame
        target = self._target_combo.currentText()
        if not target or target not in frame.columns:
            self._metric_label.setText("Pick a target column.")
            return
        if self._runner.busy:
            self._metric_label.setText("A job is already running — Stop it first.")
            return

        task = detect_task(frame, target)
        folds = int(self._cv_folds.value())
        cv_strategy = self._cv_strategy.currentText()
        smiles: list[str] | None = None
        groups: list[object] | None = None
        if cv_strategy == "scaffold":
            smiles_column = detect_smiles_column(frame)
            if smiles_column is None:
                self._board_status.setText(
                    "Scaffold CV groups rows by their Bemis-Murcko framework "
                    "— the loaded dataset has no SMILES-like column. Switch "
                    "the cv strategy to auto/stratified or load a molecular "
                    "dataset."
                )
                return
            smiles = [str(value) for value in frame[smiles_column].tolist()]
        elif cv_strategy in ("group", "stratified_group"):
            group_column = detect_group_column(frame)
            if group_column is None:
                self._board_status.setText(
                    f"The {cv_strategy!r} fold strategy needs a group-like "
                    "column (group / cluster / series / batch / plate / "
                    "cohort) so replicate rows never straddle folds — the "
                    "dataset carries none. Scaffold CV offers the same "
                    "leakage honesty for molecular data (it derives the "
                    "groups from the SMILES column)."
                )
                return
            groups = frame[group_column].tolist()
        engine_strategy = None if cv_strategy == "auto" else cv_strategy

        def job(progress) -> tuple[list[CVResult], pd.DataFrame]:
            from pharmaml.ml import compare_models_cv

            numeric, _dropped = numeric_feature_columns(frame, target)

            def on_model(info: dict) -> bool:
                # one heartbeat per leaderboard row; the callback's
                # return value IS the cooperative-stop channel.
                return bool(
                    progress(
                        float(info["index"]) / max(1, int(info["total"])),
                        f"cv {info['model']}",
                    )
                )

            progress(0.02, "building folds")
            return compare_models_cv(
                frame[numeric],
                frame[target],
                task,
                strategy=engine_strategy,
                n_folds=folds,
                smiles=smiles,
                groups=groups,
                on_model=on_model,
            )

        self._runner.submit(
            f"{folds}-fold {cv_strategy} CV compare ({task})",
            job,
            on_finished=self._render_compare,
            on_failed=self._compare_failed,
            on_cancelled=self._compare_cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_compare(self, payload: tuple[list[CVResult], pd.DataFrame]) -> None:
        """Fill the leaderboard from ``(results, board)`` and keep the
        full CVResults for the double-click report window."""
        results, board = payload
        self._cv_results = {result.model: result for result in results}
        columns = [str(column) for column in board.columns]
        self._board.clear()
        self._board.setColumnCount(len(columns))
        self._board.setHorizontalHeaderLabels(columns)
        self._board.setRowCount(len(board))
        for row_index, (_, row) in enumerate(board.iterrows()):
            for col_index, column in enumerate(columns):
                value = row[column]
                if (isinstance(value, float) and math.isnan(value)) or value is None:
                    text = "nan"
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if column == "model":
                    item.setData(Qt.ItemDataRole.UserRole, str(value))
                self._board.setItem(row_index, col_index, item)
        strategy = str(board.iloc[0]["strategy"]) if len(board) else "auto"
        n_folds = int(board.iloc[0]["folds"]) if len(board) else 0
        self._board_status.setText(
            f"{n_folds}-fold {strategy} CV over the available registry — every "
            "row is 'mean ± std (k-fold CV)' under the SAME folds; double-click "
            "a row for the fold-by-fold §19 report (individual fold scores, "
            "train-side mirrors, notes)."
        )

    def _open_cv_report(self, item: QTableWidgetItem) -> None:
        """Double-click on the board opens the clicked model's §19 report."""
        row = item.row()
        model_item = self._board.item(row, 0)
        if model_item is None:
            return
        model_name = model_item.data(Qt.ItemDataRole.UserRole) or model_item.text()
        result = self._cv_results.get(str(model_name))
        if result is None:
            # another flow filled the board (training alone leaves no CV data)
            self._board_status.setText(
                "no §19 CV report on this row — run Compare models (CV) first; "
                "double-click detail is the compare board's own."
            )
            return
        dialog = CVReportDialog(result, parent=self)
        dialog.exec()

    def _compare_failed(self, message: str) -> None:
        self._board_status.setText(
            f"Compare failed — {message.splitlines()[0] if message else 'error'}"
        )

    def _compare_cancelled(self, seconds: float) -> None:
        self._board_status.setText(
            f"Compare cancelled after {seconds:.1f}s (cooperative stop)."
        )

    def _failed(self, message: str) -> None:
        self._metric_label.setText(f"Training failed — {message.splitlines()[0]}")

    def _cancelled(self, seconds: float) -> None:
        self._metric_label.setText(f"Training cancelled after {seconds:.1f}s (cooperative stop).")

    def apply_palette(self, palette: ChartPalette) -> None:
        for plot in (self._pred_plot, self._resid_plot, self._importance_plot):
            plot.apply_palette(palette)


class CVReportDialog(QDialog):
    """The §19 fold-by-fold report for ONE model (the double-click window).

    Mirrors the playground's ``_ml_cv_detail`` window, PyQtGraph-native:
    the §19.2 headline (``mean ± std (k-fold CV)``), the metadata line,
    the fold-by-fold table (validation scores + train-side mirrors +
    fold sizes), the per-metric aggregates with honest directions, the
    strategy/metric notes, and the fold chart (fold scores, the mean
    line, the ± 1 std band edges drawn as dashed lines -- PlotPanel has
    no fill-between, so the BAND is shown by its honest edges). CV
    numbers stay structurally separate from any independent-test
    numbers (§19.7's "clearly distinguish" rule).
    """

    def __init__(self, result: CVResult, parent: QWidget | None = None) -> None:
        from pharmaml.ml.cv_results import METRIC_LABELS

        super().__init__(parent)
        self.setWindowTitle(f"CV report — {result.model}")
        self.resize(900, 680)
        lay = QVBoxLayout(self)

        headline = QLabel(f"{result.model} — {result.summary()}")
        headline.setObjectName("studioStatus")
        headline.setWordWrap(True)
        lay.addWidget(headline)
        primary_label = METRIC_LABELS.get(result.primary, (result.primary, False))[0]
        meta = QLabel(
            f"task {result.task} · strategy {result.strategy} · "
            f"{result.n_folds} folds · seed {result.seed} · {primary_label} "
            f"(primary) · {result.duration_s:.1f} s wall · "
            f"pharmaml {result.toolkit_version}"
        )
        meta.setWordWrap(True)
        lay.addWidget(meta)

        # -- the fold-by-fold table --------------------------------------- #
        short_names = [
            (name, METRIC_LABELS.get(name, (name, False))[0]) for name in result.metrics
        ]
        fold_columns = ["fold", "n_train", "n_val"] + [
            f"val {label}" for _name, label in short_names
        ]
        fold_table = QTableWidget(len(result.folds), len(fold_columns))
        fold_table.setHorizontalHeaderLabels(fold_columns)
        fold_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        fold_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for row_index, record in enumerate(result.folds):
            cells: list[object] = [record.fold, record.n_train, record.n_val]
            for name, _label in short_names:
                score = record.scores.get(name)
                cells.append(round(score, 4) if score is not None else "nan")
            for col_index, cell in enumerate(cells):
                text = f"{cell:.4f}" if isinstance(cell, float) else str(cell)
                fold_table.setItem(row_index, col_index, QTableWidgetItem(text))
        lay.addWidget(fold_table, stretch=3)

        # -- the aggregates + train-side mirror --------------------------- #
        agg_columns = ["metric", "mean ± std (k-fold CV)", "train mean", "direction"]
        agg_table = QTableWidget(len(result.metrics), len(agg_columns))
        agg_table.setHorizontalHeaderLabels(agg_columns)
        agg_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        agg_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for row_index, (name, summary) in enumerate(result.metrics.items()):
            label, lower_is_better = METRIC_LABELS.get(name, (name, False))
            train_values: list[float] = [
                float(mirror)
                for mirror in (
                    record.train_scores[name]
                    for record in result.folds
                    if record.train_scores.get(name) is not None
                )
                if mirror is not None
            ]
            train_mean = (
                f"{sum(train_values) / len(train_values):.4f}" if train_values else "—"
            )
            direction = "lower is better" if lower_is_better else "higher is better"
            for col_index, cell_value in enumerate(
                (label, summary.summary(), train_mean, direction)
            ):
                agg_table.setItem(row_index, col_index, QTableWidgetItem(str(cell_value)))
        lay.addWidget(agg_table, stretch=3)

        # -- the notes ------------------------------------------------------ #
        if result.notes:
            notes = QTextEdit()
            notes.setReadOnly(True)
            notes.setPlainText("\n".join(f"• {note}" for note in result.notes))
            notes.setMaximumHeight(min(140, 24 * (len(result.notes) + 1)))
            lay.addWidget(notes)

        # -- the fold chart (PyQtGraph; no matplotlib backend roulette) ---- #
        try:
            summary = result.metrics[result.primary]
            label, lower_is_better = METRIC_LABELS.get(
                result.primary, (result.primary, False)
            )
            chart = PlotPanel(
                title=f"{result.model} — {result.summary()}"
                + (" (lower is better)" if lower_is_better else "")
            )
            chart.setMinimumHeight(180)
            folds = np.arange(1, result.n_folds + 1, dtype=float)
            chart.scatter(
                folds,
                np.asarray(summary.folds, dtype=float),
                label="fold score",
                meta=_fold_meta(summary.folds, float(summary.mean)),
            )
            chart.line(
                np.array([1.0, float(result.n_folds)]),
                np.array([summary.mean, summary.mean]),
                label="mean",
                width=1,
                index=1,
                style="dash",
            )
            for edge, name_label in (
                (summary.mean - summary.std, "− 1 std"),
                (summary.mean + summary.std, "+ 1 std"),
            ):
                chart.line(
                    np.array([1.0, float(result.n_folds)]),
                    np.array([edge, edge]),
                    label=name_label,
                    width=1,
                    index=2,
                    style="dash",
                )
            chart.set_labels(x="fold", y=label)
            chart.add_legend()
            lay.addWidget(chart, stretch=2)
        except Exception as exc:  # noqa: BLE001 — honest note, never a blocker
            chart_note = QLabel(
                f"fold chart unavailable ({exc}) — the tables above carry "
                "the same numbers."
            )
            chart_note.setWordWrap(True)
            lay.addWidget(chart_note)
