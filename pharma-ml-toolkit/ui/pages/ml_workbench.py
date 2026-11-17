"""ML Workbench: the Data & ML tab's remaining cards, Studio edition.

The round-228 port of the legacy Tkinter playground's Data & ML tab (the
owner's "everything must be in studio.py" directive, the retirement
round): the five card families the Model Analysis page does not already
carry, each wired to the SAME public backend APIs the cards call --

* **Hyperparameter search** -- the keras-tuner-style searchers
  (:class:`~pharmaml.ml.tuner.RandomSearch` / ``TPESearch`` /
  ``Hyperband``) over the bespoke trio (ann / dnn / hsvr) plus EVERY
  grid-bearing registry family, with the family's defaults scored under
  the same folds as the baseline the search has to beat, the held-out
  verdict on the refit winner, the trial leaderboard board, and "Apply
  best" handing the winning configuration to the Model Analysis page.
* **Leaderboards** -- the cross-validated feature-family benchmark
  (``benchmark_feature_families`` over the dataset's SMILES column) and
  the out-of-fold stacking leaderboard (``stack_oof_regressors`` with
  its native parity scatter).
* **Ensemble builder** -- voting / stacking ensembles of the ticked
  registry members (``build_voting`` / ``build_stacking``), scored on a
  held-out split, the parity scatter / class-count bars drawn
  Qt-native, the fitted ensemble kept for the persistence and AD cards.
* **Persistence** -- the provenance-carrying model bundle
  (``save_model`` + ``write_metadata_sidecar``), its reload with the
  version-honest summary line, and predict-on-CSV with the bundle's
  feature-contract audit (``validate_bundle_input``) refusing a
  mismatched table BEFORE any prediction runs.
* **Applicability domain** -- the uploaded kNN-distance +
  bootstrap-uncertainty workflow (``applicability_domain``) over the
  Model Analysis run, this page's ensemble, or a loaded bundle, with
  the reliability table and the inside/outside scatter drawn Qt-native.

The page reuses the Model Analysis page's dataset (its frame and target
combobox) through ``_training_context()``; a doomed request is refused
BEFORE any job starts (no dataset / no target / no SMILES column / no
run yet / regression-only card given a classification target / too few
rows), never silently degraded.  Every figure is NATIVE PyQtGraph
(§16.8 -- no matplotlib import anywhere on this page); every long job
runs in a :class:`~ui.workers.job_worker.JobWorker` and widgets are
never touched from the worker.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import configure_splitter, wrap_in_scroll
from ..workers.job_worker import JobRunner
from .model_analysis import detect_smiles_column, detect_task, numeric_feature_columns

#: Searcher tokens -- the CLI's ``ml tune --searcher`` registry,
#: mirrored verbatim from the playground's Tune card.
TUNE_SEARCHERS: tuple[str, ...] = ("random", "tpe", "hyperband")

#: Pruner tokens (100x-160): off = every trial runs every fold; median
#: = Optuna's MedianPruner wired into the fold loop.  Hyperband keeps
#: its own halving.
TUNE_PRUNERS: tuple[str, ...] = ("off", "median")

#: The resource slot each Hyperband-able family halves over -- ann's
#: sklearn ``max_iter`` grid, dnn's torch ``epochs`` grid.  hsvr and the
#: generated registry families declare none, so the card refuses a
#: hyperband run over them BEFORE any job starts.
TUNE_RESOURCES: dict[str, str] = {"ann": "max_iter", "dnn": "epochs"}

#: Bespoke (non-generated) hypermodel builder names, keyed by
#: ``(estimator, task)`` -- resolved inside the worker through the
#: tuner's own ``hypermodel_by_name``.
_BESPOKE_HYPERMODELS: dict[tuple[str, str], str] = {
    ("ann", "regression"): "build_ann_regressor",
    ("ann", "classification"): "build_ann_classifier",
    ("dnn", "regression"): "build_deep_regressor",
    ("dnn", "classification"): "build_deep_classifier",
    ("hsvr", "regression"): "build_hsvr",
}

#: Optional backends a Tune family needs before any search starts (the
#: dnn/torch guard pattern extended to the boosting families -- a
#: missing backend would otherwise surface only as the all-trials-
#: failed ModelError, a mid-job crash the pre-flight replaces).
_TUNE_REQUIRES: dict[str, str] = {
    "dnn": "torch",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}

#: The broken-install fallback: the grid-bearing registry families in
#: the registry's own declared order (the playground's own curated
#: list).  With pharmaml importable the constructor rebuilds this from
#: ``registry_hypermodels()`` itself.
_FALLBACK_TUNE_FAMILIES: tuple[str, ...] = (
    "logistic_regression",
    "ridge_classifier",
    "sgd_classifier",
    "lasso_logistic",
    "elastic_net_logistic",
    "lasso",
    "ridge",
    "elastic_net",
    "huber_regression",
    "pls",
    "knn",
    "gaussian_nb",
    "multinomial_nb",
    "bernoulli_nb",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "svm",
    "adaboost",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
)

#: The fallback's task-support map (``token -> supported tasks``).
_FALLBACK_REGISTRY_TASKS: dict[str, set[str]] = {
    "logistic_regression": {"classification"},
    "ridge_classifier": {"classification"},
    "sgd_classifier": {"classification"},
    "lasso_logistic": {"classification"},
    "elastic_net_logistic": {"classification"},
    "gaussian_nb": {"classification"},
    "multinomial_nb": {"classification"},
    "bernoulli_nb": {"classification"},
    "lasso": {"regression"},
    "ridge": {"regression"},
    "elastic_net": {"regression"},
    "huber_regression": {"regression"},
    "pls": {"regression"},
    "knn": {"regression", "classification"},
    "decision_tree": {"regression", "classification"},
    "random_forest": {"regression", "classification"},
    "extra_trees": {"regression", "classification"},
    "svm": {"regression", "classification"},
    "adaboost": {"regression", "classification"},
    "gradient_boosting": {"regression", "classification"},
    "xgboost": {"regression", "classification"},
    "lightgbm": {"regression", "classification"},
    "catboost": {"regression", "classification"},
}

#: Ensemble-builder card: registry members offered as checkboxes --
#: ``(registry token, label)`` (dual-task, dependency-free families so
#: every combination builds for both tasks).
ENSEMBLE_MEMBER_LABELS: tuple[tuple[str, str], ...] = (
    ("random_forest", "Random forest"),
    ("extra_trees", "Extra trees"),
    ("svm", "SVM"),
    ("knn", "KNN"),
    ("decision_tree", "Decision tree"),
    ("logistic_regression", "Logistic (cls)"),
    ("ridge", "Ridge (reg)"),
    ("gradient_boosting", "Gradient boosting"),
    ("ann", "ANN (MLP)"),
    ("gaussian_process", "Gaussian process"),
)

#: The default ticked trio -- the rar's own Voting.py / Stacking.py
#: members (RF + SVM + KNN), matching the registry ensembles' defaults.
ENSEMBLE_DEFAULT_MEMBERS: tuple[str, ...] = ("random_forest", "svm", "knn")

#: Applicability-domain card: where the model and its split come from.
AD_SOURCES: tuple[str, ...] = ("Model Analysis run", "Workbench ensemble", "loaded bundle")

#: The report keys the Save-bundle card records as the bundle's metrics
#: (whichever the report carries; the playground's own key list).
_PERSIST_METRIC_KEYS: tuple[str, ...] = (
    "r2",
    "rmse",
    "mae",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "mcc",
    "roc_auc",
)


def _config_text(params: dict[str, Any]) -> str:
    """Render one trial's hyperparameters as ``k=v`` tokens.

    The playground's ``tune_config_text`` discipline: floats through
    ``%g``, booleans as ``on``/``off``, keys sorted so the
    leaderboard's configuration column is stable across trials.
    """
    tokens: list[str] = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, bool):
            text = "on" if value else "off"
        elif isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        tokens.append(f"{key}={text}")
    return " ".join(tokens)


def _cell_text(value: Any, limit: int = 30) -> str:
    """Render one table cell for the preview/leaderboard boards."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4g}"
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _populate_table(table: QTableWidget, columns: list[str], rows: list[list[str]]) -> None:
    """Fill a read-only board with pre-formatted string cells."""
    table.clear()
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setRowCount(len(rows))
    for row_index, cells in enumerate(rows):
        for col_index, text in enumerate(cells):
            table.setItem(row_index, col_index, QTableWidgetItem(text))


def _report_metrics(report: Any) -> dict[str, float]:
    """The report's metric block for the bundle's metadata."""
    metrics: dict[str, float] = {}
    for key in _PERSIST_METRIC_KEYS:
        value = getattr(report, key, None)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _slug(name: str) -> str:
    """A filesystem-safe default bundle name."""
    return name.replace(" ", "_").replace("(", "").replace(")", "") or "model"


class MLWorkbenchPage(QWidget):
    """The playground's remaining Data & ML cards, ported to the Studio."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
        model_analysis: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._model_analysis = model_analysis
        # (part8 Phase F) the last semi-supervised result, kept for
        # export/inspection between runs
        self._semi_result: Any = None
        # (part8 Phase 10) the last applicability-domain result + the
        # model's query predictions behind the view selector
        self._ad_result: Any = None
        self._ad_query_pred: Any = None
        # (part8 §14.1, round 5) the live active-learning campaign: the
        # loop object + the last exported batch + its context key
        self._al_loop: Any = None
        self._al_key: tuple[str, int, int] | None = None
        self._al_last_batch: Any = None
        # review m-19: the statistics tab's grouping picker used to fill
        # once at construction and go stale after the next dataset load;
        # the Model Analysis page now emits frameLoaded and this page
        # re-lists the columns on every change.
        if self._model_analysis is not None and hasattr(self._model_analysis, "frameLoaded"):
            self._model_analysis.frameLoaded.connect(self._on_analysis_frame)

        # -- the cards' kept state ----------------------------------------
        self._tune_last: tuple[Any, str, str, str] | None = None
        self._stack_last: Any = None
        self._ens_last: tuple[Any, Any, str, list[str], str, Any, Any, Any, Any, str] | None = None
        self._bundle: tuple[Any, dict[str, Any], Path] | None = None
        self._predictions: pd.DataFrame | None = None
        self._ad_frame: pd.DataFrame | None = None
        self._active_status: QLabel | None = None
        self._job_buttons: list[QPushButton] = []

        # -- (round 229, Part-3 §20) the family benchmark's durable
        # checkpoint home: Stop keeps the finished families on disk and
        # the next Start resumes from the first unscored one.  An
        # unwritable home degrades to None (fresh runs, no resume)
        # instead of failing the page.
        self._bench_resume_note = ""
        try:
            from pharmaml.utils.checkpoints import default_directory

            self._bench_checkpoint_dir: Path | None = default_directory(
                "family-benchmark"
            )
        except Exception:  # noqa: BLE001 -- home may be unwritable
            self._bench_checkpoint_dir = None

        # -- the estimator list + the task-support map (the registry's
        # own declared grid when pharmaml is importable, the curated
        # fallback otherwise) --------------------------------------------
        families: tuple[str, ...] = _FALLBACK_TUNE_FAMILIES
        tasks: dict[str, set[str]] = {
            token: set(supported) for token, supported in _FALLBACK_REGISTRY_TASKS.items()
        }
        try:
            from pharmaml.ml.tuner import registry_hypermodels

            pairs = registry_hypermodels()
            if pairs:
                families = tuple(dict.fromkeys(token for token, _task in pairs))
                tasks = {}
                for token, task in pairs:
                    tasks.setdefault(token, set()).add(task)
        except Exception:  # noqa: BLE001, S110 - broken-install fallback to the curated list
            pass
        self._tune_estimators: tuple[str, ...] = ("ann", "hsvr", "dnn") + families
        self._registry_tasks = tasks

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        # §23.2 (round 235): every tab body rides the scroll policy --
        # the probe measured the tuner form at 479 px and Leaderboards
        # at 775 px minimum against the ~450 px the 860×560 minimum
        # window leaves, i.e. run buttons and results boards clipped
        # below the fold.  The wrapper is a no-op while content fits.
        self._tabs.addTab(wrap_in_scroll(self._build_tune_tab()), "Hyperparameter search")
        self._tabs.addTab(wrap_in_scroll(self._build_leaderboards_tab()), "Leaderboards")
        self._tabs.addTab(wrap_in_scroll(self._build_ensemble_tab()), "Ensemble builder")
        self._tabs.addTab(wrap_in_scroll(self._build_persistence_tab()), "Persistence")
        self._tabs.addTab(wrap_in_scroll(self._build_ad_tab()), "Applicability domain")
        self._tabs.addTab(
            wrap_in_scroll(self._build_semi_tab()), "Semi-supervised"
        )
        self._tabs.addTab(
            wrap_in_scroll(self._build_al_tab()), "Active learning"
        )
        self._tabs.addTab(
            wrap_in_scroll(self._build_statistical_tab()), "Statistical tools"
        )
        lay.addWidget(self._tabs)

        self._job_buttons = [
            self._tune_btn,
            self._tune_apply_btn,
            self._bench_btn,
            self._stack_btn,
            self._ens_btn,
            self._persist_save_btn,
            self._persist_open_btn,
            self._persist_predict_btn,
            self._persist_write_btn,
            self._persist_card_btn,
            self._ad_btn,
            self._stat_conformal_btn,
            self._stat_conformal_cls_btn,
            self._stat_calibration_btn,
            self._stat_decision_btn,
            self._stat_drift_btn,
            self._stat_compare_btn,
            self._stat_correct_btn,
            self._stat_power_btn,
            self._stat_coverage_btn,
            self._stat_subgroup_btn,
            self._stat_export_btn,
            self._al_btn,
            self._al_export_btn,
            self._al_import_btn,
        ]

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

    # -- shared plumbing ----------------------------------------------------
    def _training_context(self) -> tuple[pd.DataFrame | None, str, str]:
        """The Model Analysis page's ``(frame, target, task)``.

        The Studio's data source rule: the workbench reuses the training
        page's loaded frame and its target combobox (guarded -- the
        page may not be wired in, or carry no table yet).  Returns a
        ``(None, "", "")`` triple the callers refuse on honestly.
        """
        frame = getattr(self._model_analysis, "_frame", None)
        if frame is None:
            return None, "", ""
        combo = getattr(self._model_analysis, "_target_combo", None)
        target = str(combo.currentText()) if combo is not None else ""
        if not target or target not in [str(column) for column in frame.columns]:
            return frame, "", ""
        return frame, target, detect_task(frame, target)

    def _current_target(self) -> str | None:
        """The training page's current target combobox text, guarded."""
        combo = getattr(self._model_analysis, "_target_combo", None)
        return str(combo.currentText()) or None if combo is not None else None

    def _on_started(self, name: str) -> None:
        for btn in self._job_buttons:
            btn.setEnabled(False)
        status = self._active_status
        if status is not None:
            status.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in self._job_buttons:
            btn.setEnabled(True)

    def _submit(self, name: str, job: Any, on_finished: Any, status: QLabel) -> None:
        """Shared runner submit with the duplicate-submission guard.

        Failures render to the calling card's own status label;
        cancellations say so (the cooperative stop).
        """
        if self._runner.busy:
            status.setText(f"Busy: '{getattr(self._runner, 'job_name', '') or 'another operation'}' is running — Stop it (or check Job Center) first.")
            return
        self._active_status = status

        def failed(message: str) -> None:
            first = message.splitlines()[0] if message else "job failed"
            status.setText(f"Job failed — {first}")

        def cancelled(seconds: float) -> None:
            status.setText(f"Cancelled after {seconds:.1f}s (cooperative stop).")

        self._runner.submit(
            name,
            job,
            on_finished=on_finished,
            on_failed=failed,
            on_cancelled=cancelled,
            on_progress=self._progress.update_progress,
        )

    def _hypermodel_name(self, estimator: str, task: str) -> str | None:
        """The ``(estimator, task)`` -> builder-name map (None = mismatch)."""
        if estimator in ("ann", "hsvr", "dnn"):
            return _BESPOKE_HYPERMODELS.get((estimator, task))
        supported = self._registry_tasks.get(estimator)
        if not supported or task not in supported:
            return None
        suffix = "regressor" if task == "regression" else "classifier"
        return f"build_{estimator}_{suffix}"

    def _status_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("studioStatus")
        label.setWordWrap(True)
        return label

    def _make_board(self, minimum_height: int = 150) -> QTableWidget:
        board = QTableWidget()
        board.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        board.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        board.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        board.setWordWrap(False)
        board.verticalHeader().setVisible(False)
        board.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        board.setMinimumHeight(minimum_height)
        return board

    # -- tab 1: the tuner -----------------------------------------------------
    def _build_tune_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._tune_estimator = QComboBox()
        for token in self._tune_estimators:
            self._tune_estimator.addItem(token)
        self._tune_estimator.setCurrentText("ann")
        self._tune_estimator.setToolTip(
            "ann = the sklearn MLP hypermodel; dnn = the torch deep MLP "
            "(batchnorm searchable, needs torch); hsvr = the hierarchical SVR "
            "(regression only); every other grid-bearing registry family "
            "(random_forest, svm, knn, ridge, ...) is searched over its own "
            "declared param_grid. The same families the CLI's ml tune "
            "--estimator accepts."
        )
        form.addRow("estimator:", self._tune_estimator)
        self._tune_searcher = QComboBox()
        for token in TUNE_SEARCHERS:
            self._tune_searcher.addItem(token)
        self._tune_searcher.setToolTip(
            "random = RandomSearch, every trial at the full budget (the trials "
            "spinbox decides how many); tpe = TPESearch, Optuna's "
            "Tree-structured Parzen Estimator (the best gamma(n) trials seed "
            "one KDE mixture and the rest another); hyperband = successive "
            "halving over the family's resource slot -- ann or dnn only, the "
            "trials spinbox is ignored (the brackets decide)."
        )
        form.addRow("searcher:", self._tune_searcher)
        self._tune_trials = QSpinBox()
        self._tune_trials.setRange(4, 40)
        self._tune_trials.setValue(8)
        self._tune_trials.setToolTip(
            "RandomSearch/TPE only: how many full-budget trials to draw. "
            "Hyperband ignores this -- its brackets fix the trial count from "
            "eta and the resource ladder."
        )
        form.addRow("trials:", self._tune_trials)
        self._tune_eta = QSpinBox()
        self._tune_eta.setRange(2, 4)
        self._tune_eta.setValue(3)
        self._tune_eta.setToolTip(
            "Hyperband only: the halving factor eta -- every rung keeps the "
            "top ceil(n/eta) configs and multiplies their budget by eta. "
            "RandomSearch/TPE ignore this."
        )
        form.addRow("eta (hyperband):", self._tune_eta)
        self._tune_cv = QSpinBox()
        self._tune_cv.setRange(2, 5)
        self._tune_cv.setValue(3)
        self._tune_cv.setToolTip(
            "Cross-validation folds every trial (and the defaults baseline) "
            "is scored under -- the same shuffled folds, the apples-to-apples "
            "comparison the search has to beat."
        )
        form.addRow("cv folds:", self._tune_cv)
        self._tune_pruner = QComboBox()
        for token in TUNE_PRUNERS:
            self._tune_pruner.addItem(token)
        self._tune_pruner.setToolTip(
            "off = every trial runs every fold. median = Optuna's "
            "MedianPruner wired into the fold loop: a trial whose running "
            "fold mean sits below the median of the completed trials at the "
            "SAME fold is cut short (state 'pruned', partial fold scores "
            "kept). Hyperband ignores this -- its rungs do their own halving."
        )
        form.addRow("pruner:", self._tune_pruner)

        buttons = QHBoxLayout()
        self._tune_btn = QPushButton("Run search")
        self._tune_btn.setMinimumHeight(34)
        self._tune_btn.clicked.connect(self._start_tune_search)
        self._tune_btn.setToolTip(
            "Search the chosen family's space -- RandomSearch samples "
            "full-budget trials, Hyperband halves the resource rung by rung; "
            "trials rank by mean CV score on the board below; the summary "
            "line carries best vs defaults (the gain) and the held-out "
            "verdict. Deterministic end-to-end (seed 42)."
        )
        buttons.addWidget(self._tune_btn)
        self._tune_apply_btn = QPushButton("Apply best")
        self._tune_apply_btn.clicked.connect(self._apply_best)
        self._tune_apply_btn.setToolTip(
            "Carry the winning configuration forward: point the Model "
            "Analysis page's model combo at the family and pin the winning "
            "values into its manual-hyperparameters table (families the "
            "registry does not carry, e.g. dnn, answer with the honest note)."
        )
        buttons.addWidget(self._tune_apply_btn)
        buttons.addStretch(1)
        form.addRow("", buttons)

        self._tune_summary = self._status_label(
            "no search yet -- the tuner reuses the Model Analysis page's "
            "dataset and target."
        )
        form.addRow("", self._tune_summary)
        self._tune_board = self._make_board()
        self._tune_board.setToolTip(
            "The trial leaderboard: completed trials ranked by mean CV score, "
            "then failed/pruned trials (rank empty, pruned scores marked with "
            "a trailing dot -- the last fold mean the pruner acted on)."
        )
        form.addRow("", self._tune_board)
        return tab

    def _start_tune_search(self) -> None:
        """Run a hyperparameter search over the chosen family."""
        frame, target, task = self._training_context()
        if frame is None:
            self._tune_summary.setText(
                "Load a dataset on the Model Analysis page first -- the search "
                "trains on its frame and target."
            )
            return
        if not target:
            self._tune_summary.setText("Pick a target column on the Model Analysis page first.")
            return
        numeric, dropped = numeric_feature_columns(frame, target)
        if not numeric:
            self._tune_summary.setText(
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + " -- the hypermodels need numeric feature columns (compute "
                "descriptors first, the Chemistry tab's feature factory)."
            )
            return
        estimator = self._tune_estimator.currentText()
        hypermodel_name = self._hypermodel_name(estimator, task)
        if hypermodel_name is None:
            other = "regression" if task == "classification" else "classification"
            self._tune_summary.setText(
                f"The {estimator} family is {other}-only -- pick a dual-task "
                f"family (ann, dnn, knn, random_forest, ...) for a {task} target."
            )
            return
        required = _TUNE_REQUIRES.get(estimator)
        if required is not None and importlib.util.find_spec(required) is None:
            extra = "deep" if estimator == "dnn" else "boosting"
            self._tune_summary.setText(
                f"The {estimator} family needs the optional {required} backend "
                f"-- pip install 'pharmaml[{extra}]' and re-run the search."
            )
            return
        searcher = self._tune_searcher.currentText()
        pruner_choice = self._tune_pruner.currentText()
        resource = TUNE_RESOURCES.get(estimator)
        if searcher == "hyperband" and resource is None:
            self._tune_summary.setText(
                "Hyperband halves a resource slot -- ann or dnn only (their "
                f"max_iter / epochs grids); the {estimator} hypermodel declares "
                "no resource slot. Switch the searcher back to random/tpe."
            )
            return
        trials = int(self._tune_trials.value())
        eta = int(self._tune_eta.value())
        cv = int(self._tune_cv.value())

        def job(progress: Any) -> Any:
            from sklearn.model_selection import cross_val_score, train_test_split

            import pharmaml.ml.tuner as tuner_module
            from pharmaml.ml import evaluate_model
            from pharmaml.ml.tuner import (
                Hyperband,
                HyperParameters,
                MedianPruner,
                RandomSearch,
                TPESearch,
                make_cv_splitter,
                resource_bounds,
            )

            def on_trial(info: dict[str, Any]) -> None:
                note = ""
                if info.get("best_score") is not None:
                    note = f"best {info['best_score']:.3f}"
                if info.get("state") == "pruned":
                    note = (note + " · " if note else "") + "trial pruned"
                if info.get("error") is not None:
                    note = (note + " · " if note else "") + "trial failed"
                progress(info["trial"] / max(1, int(info.get("total") or 1)), note)

            hypermodel = tuner_module.hypermodel_by_name(hypermodel_name)
            objective = "r2" if task == "regression" else "accuracy"
            y = frame[target]
            stratify = None if task == "regression" else y.astype(str).to_numpy()
            x_train, x_test, y_train, y_test = train_test_split(
                frame[numeric], y, test_size=0.25, random_state=42, stratify=stratify
            )
            ladder: tuple[int, int] | None = None
            pruner = (
                MedianPruner()
                if pruner_choice == "median" and searcher != "hyperband"
                else None
            )
            if searcher == "hyperband":
                # the ladder IS the family's declared resource grid -- r_min
                # its smallest value, r_max its largest (the full budget a
                # plain Train runs at).
                r_min, r_max = resource_bounds(hypermodel, resource or "epochs")
                ladder = (r_min, r_max)
                searcher_obj: Any = Hyperband(
                    hypermodel,
                    objective=objective,
                    resource=resource or "epochs",
                    max_resource=r_max,
                    min_resource=r_min,
                    eta=eta,
                    cv=cv,
                    task=task,
                    seed=42,
                    on_trial=on_trial,
                )
            elif searcher == "tpe":
                searcher_obj = TPESearch(
                    hypermodel,
                    objective=objective,
                    max_trials=trials,
                    cv=cv,
                    task=task,
                    seed=42,
                    on_trial=on_trial,
                    pruner=pruner,
                )
            else:
                searcher_obj = RandomSearch(
                    hypermodel,
                    objective=objective,
                    max_trials=trials,
                    cv=cv,
                    task=task,
                    seed=42,
                    on_trial=on_trial,
                    pruner=pruner,
                )
            results = searcher_obj.search(x_train, y_train)
            # the baseline: the hypermodel's defaults under the tuner's own
            # fold protocol (the public splitter -- same folds, only the
            # configuration differs).
            defaults = hypermodel(HyperParameters())
            base = cross_val_score(
                defaults,
                x_train,
                y_train,
                scoring=objective,
                cv=make_cv_splitter(task, y_train, cv),
            )
            # the held-out verdict: the winner was refit on the train split
            # inside search(); score it like any trained model.
            report = evaluate_model(
                results.best_estimator, x_test, y_test, task, keep_predictions=False
            )
            return results, float(base.mean()), float(base.std()), report, objective, ladder

        def done(payload: Any) -> None:
            results, base_mean, base_std, report, objective, ladder = payload
            self._tune_last = (results, estimator, task, target)
            held_out = report.r2 if task == "regression" else report.accuracy
            held_label = "held-out R²" if task == "regression" else "held-out accuracy"
            gain = results.best_score - base_mean
            self._tune_summary.setText(
                f"best {results.best_score:.4f} ± {results.best_trial.score_std:.4f} "
                f"CV {objective} · defaults {base_mean:.4f} ± {base_std:.4f} → "
                f"gain {gain:+.4f} · {held_label} {held_out:.4f} · winner: "
                f"{_config_text(dict(results.best_params))}"
            )
            ranked = sorted(
                (t for t in results.trials if not t.failed and not t.pruned),
                key=lambda t: t.score,
                reverse=True,
            )
            failed_trials = [t for t in results.trials if t.failed]
            pruned_trials = [t for t in results.trials if t.pruned]
            columns = ["rank", "trial", "configuration", "score", "score_std"]
            if ladder is not None:
                columns.insert(3, "resource")
            rows: list[list[str]] = []
            for rank, trial in enumerate(ranked, start=1):
                config = _cell_text(_config_text(dict(trial.hyperparameters)), limit=60)
                cells = [str(rank), str(trial.trial_id), config]
                if ladder is not None:
                    cells.append("" if trial.resource is None else f"{trial.resource:g}")
                cells.append(f"{trial.score:.4f}")
                cells.append(f"{trial.score_std:.4f}")
                rows.append(cells)
            for trial in failed_trials:
                config = _cell_text(_config_text(dict(trial.hyperparameters)), limit=60)
                cells = ["", str(trial.trial_id), config]
                if ladder is not None:
                    cells.append("" if trial.resource is None else f"{trial.resource:g}")
                cells.append(f"{trial.score:.4f}")  # a failed trial's nan shows as nan
                cells.append("—")
                rows.append(cells)
            for trial in pruned_trials:
                config = _cell_text(_config_text(dict(trial.hyperparameters)), limit=60)
                cells = ["", str(trial.trial_id), config]
                if ladder is not None:
                    cells.append("" if trial.resource is None else f"{trial.resource:g}")
                # the honest partial score: the last running fold mean the
                # pruner acted on, marked so it never reads as a full score.
                partial = (
                    f"{float(trial.fold_scores[-1]):.4f}·" if trial.fold_scores else "—"
                )
                cells.append(partial)
                cells.append("—")
                rows.append(cells)
            _populate_table(self._tune_board, columns, rows)

        self._submit(f"tuning {estimator} ({searcher})", job, done, self._tune_summary)

    def _apply_best(self) -> None:
        """Hand the winning configuration to the Model Analysis page."""
        if self._tune_last is None:
            self._tune_summary.setText(
                "Run a search first -- Apply best carries the winning "
                "configuration forward."
            )
            return
        results, estimator, _task, _target = self._tune_last
        apply = getattr(self._model_analysis, "apply_search_result", None)
        if not callable(apply):
            self._tune_summary.setText(
                "The Model Analysis page is not wired in -- its model combo "
                "cannot be pointed at the winner (the winning configuration "
                "rides the search summary line)."
            )
            return
        params = results.best_params
        if not isinstance(params, dict):
            params = dict(params)
        self._tune_summary.setText(str(apply(estimator, dict(params))))

    # -- tab 2: leaderboards ---------------------------------------------------
    def _build_leaderboards_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)

        # -- the family benchmark card ------------------------------------
        bench_header = QLabel(
            "Family benchmark -- one fixed model, cross-validated ACROSS the "
            "toolkit's 2D feature families over the dataset's SMILES column."
        )
        bench_header.setWordWrap(True)
        lay.addWidget(bench_header)
        bench_form = QFormLayout()
        self._bench_model = QLineEdit("auto")
        self._bench_model.setToolTip(
            "Registry name of the benchmark model ('auto' = the library "
            "default: lightgbm when installed, else the fastest always-"
            "available twin). One model fixed so the board isolates the "
            "FEATURISATION effect."
        )
        bench_form.addRow("model:", self._bench_model)
        self._bench_folds = QSpinBox()
        self._bench_folds.setRange(2, 10)
        self._bench_folds.setValue(3)
        self._bench_folds.setToolTip(
            "Cross-validation folds per family. 3 = the directive's default; "
            "2 for a quick screen."
        )
        bench_form.addRow("folds:", self._bench_folds)
        # (part8 §5/Phase 6) manual n_jobs for the family benchmark --
        # previously pinned serial, now the same editable field the
        # Chemistry Lab uses.
        from ..widgets.njobs_field import NJobsField

        self._bench_njobs = NJobsField(default=-1)
        self._bench_njobs.setToolTip(
            "Worker count for benchmark_feature_families (joblib "
            "convention: -1 = all cores, 1 = serial). 0 is invalid."
        )
        bench_form.addRow("Threads (n_jobs):", self._bench_njobs)
        self._bench_btn = QPushButton("Benchmark feature families")
        self._bench_btn.setMinimumHeight(34)
        self._bench_btn.clicked.connect(self._start_benchmark)
        self._bench_btn.setToolTip(
            "Featurise the dataset's SMILES column with every 2D family "
            "(shape_3d and rism opt-in only -- the 3D families embed "
            "conformers per molecule) and cross-validate the fixed model on "
            "each: the board shows which featurisation carries the signal. "
            "A family that fails keeps its row with a NaN score and sinks. "
            "Progress is checkpointed after every family: Stop keeps the "
            "finished families on disk and the next Start resumes from the "
            "first unscored one (a changed dataset, model or fold count "
            "starts fresh)."
        )
        bench_form.addRow("", self._bench_btn)
        lay.addLayout(bench_form)
        self._bench_status = self._status_label(
            "the family benchmark featurises the dataset's SMILES column -- "
            "load a molecular dataset on the Model Analysis page."
        )
        lay.addWidget(self._bench_status)
        self._bench_board = self._make_board(minimum_height=170)
        self._bench_board.setToolTip(
            "One row per feature family: n_features, n_samples, the primary "
            "metric's mean ± std, fit time, folds -- sorted by the metric."
        )
        lay.addWidget(self._bench_board)

        # -- the OOF stacking card ------------------------------------------
        stack_header = QLabel(
            "OOF stacking -- out-of-fold base predictions, the blenders ranked "
            "on them (regression targets only)."
        )
        stack_header.setWordWrap(True)
        lay.addWidget(stack_header)
        stack_form = QFormLayout()
        self._stack_base = QLineEdit("ridge, random_forest, gradient_boosting, extra_trees")
        self._stack_base.setToolTip(
            "Comma-separated registry names of the level-0 learners (the "
            "stack needs something to blend, e.g. ridge, random_forest)."
        )
        stack_form.addRow("base models:", self._stack_base)
        self._stack_meta = QLineEdit("ridge, random_forest, gradient_boosting")
        self._stack_meta.setToolTip(
            "Comma-separated registry names of the level-1 blenders; EMPTY "
            "means the library default trio (filtered to the available "
            "registry entries)."
        )
        stack_form.addRow("meta models:", self._stack_meta)
        self._stack_folds = QSpinBox()
        self._stack_folds.setRange(2, 10)
        self._stack_folds.setValue(5)
        self._stack_folds.setToolTip(
            "OOF fold count: every base model predicts its own held-out fold, "
            "so the blender never scores a fold it was fitted on."
        )
        stack_form.addRow("OOF folds:", self._stack_folds)
        self._stack_test = QSpinBox()
        self._stack_test.setRange(10, 50)
        self._stack_test.setValue(20)
        self._stack_test.setToolTip(
            "Percent of the rows held out for the leaderboard's test R² "
            "(the notebook's own 80/20 holdout)."
        )
        stack_form.addRow("test holdout %:", self._stack_test)
        checks = QHBoxLayout()
        self._stack_quantile = QCheckBox("quantile transform")
        self._stack_quantile.setToolTip(
            "Fit a seeded QuantileTransformer on the TRAINING target and "
            "apply it to both targets -- every reported score is then on the "
            "TRANSFORMED scale (the notebook's own convention)."
        )
        checks.addWidget(self._stack_quantile)
        self._stack_residual = QCheckBox("residual corrector")
        self._stack_residual.setChecked(True)
        self._stack_residual.setToolTip(
            "Fit a Ridge corrector on the best blender's OOF residuals and "
            "append the '<best>+residual' leaderboard row; the parity "
            "scatter carries the refined values."
        )
        checks.addWidget(self._stack_residual)
        checks.addStretch(1)
        stack_form.addRow("", checks)
        self._stack_btn = QPushButton("Run OOF stacking")
        self._stack_btn.setMinimumHeight(34)
        self._stack_btn.clicked.connect(self._start_stack)
        self._stack_btn.setToolTip(
            "Build the out-of-fold prediction matrix of the base models, fit "
            "every meta-learner on it, and rank them by held-out test R² -- "
            "the look.ipynb leakage-fix story as the stacking leaderboard."
        )
        stack_form.addRow("", self._stack_btn)
        lay.addLayout(stack_form)
        self._stack_status = self._status_label(
            "OOF stacking ranks blenders on out-of-fold predictions -- "
            "regression targets only."
        )
        lay.addWidget(self._stack_status)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._stack_board = self._make_board(minimum_height=140)
        self._stack_board.setToolTip(
            "One row per meta-learner: OOF R² (the training-side rank) and "
            "test R² (the held-out verdict), the '+residual' row last."
        )
        splitter.addWidget(self._stack_board)
        self._stack_plot = PlotPanel(title="OOF stacking parity (best blender)")
        self._stack_plot.set_empty_hint("Run the OOF stacking benchmark to see the blender parity")

        splitter.addWidget(self._stack_plot)
        splitter.setSizes([420, 420])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)
        return tab

    def _start_benchmark(self) -> None:
        """Cross-validate one fixed model across the 2D feature families."""
        frame, target, task = self._training_context()
        if frame is None:
            self._bench_status.setText(
                "Load a dataset on the Model Analysis page first -- the "
                "benchmark featurises its SMILES column."
            )
            return
        if not target:
            self._bench_status.setText("Pick a target column on the Model Analysis page first.")
            return
        smiles_column = detect_smiles_column(frame)
        if smiles_column is None:
            self._bench_status.setText(
                "The family benchmark featurises SMILES -- the loaded dataset "
                "has no SMILES-like column (looking for smiles / "
                "canonical_smiles / ...). Load a molecular dataset (e.g. "
                "qsar_classification.csv) on the Model Analysis page."
            )
            return
        if target == smiles_column:
            self._bench_status.setText(
                "Pick a numeric target column -- the SMILES column is an "
                "identifier, not a label to model."
            )
            return
        model_name = self._bench_model.text().strip()
        folds = int(self._bench_folds.value())
        # part8 §5: pass the manual n_jobs through; invalid input runs
        # serial with a shell toast instead of raising mid-benchmark
        # (a status-label note would be overwritten by the run's own
        # progress line before anyone could read it).
        bench_njobs = self._bench_njobs.value()
        if bench_njobs is None:
            from ..widgets.toast_routing import shell_notify

            bench_njobs = 1
            shell_notify(
                self,
                "ML Workbench — n_jobs must be a nonzero integer (got "
                f"{self._bench_njobs.currentText().strip()!r}); "
                "benchmarking serially",
                "warning",
            )
        smiles = frame[smiles_column]
        y = frame[target]
        self._bench_resume_note = ""

        def job(progress: Any) -> Any:
            from pharmaml.ml import benchmark_feature_families

            # (round 229, Part-3 §20) surface the resume story before
            # the run: what the checkpoint on disk will reuse.  A plain
            # page attribute, filled by the worker thread and read by
            # the done() slot on the GUI thread afterwards.
            if self._bench_checkpoint_dir is not None:
                try:
                    from pharmaml.utils.checkpoints import CheckpointManager

                    info = CheckpointManager(
                        self._bench_checkpoint_dir, "family-benchmark"
                    ).latest()
                    if info is not None and int(info.total or 0) > 0:
                        self._bench_resume_note = (
                            f"checkpoint found: {int(info.step or 0)}/{int(info.total or 0)} "
                            "families already scored — resuming there"
                        )
                except Exception:  # noqa: BLE001 -- the note is cosmetic
                    self._bench_resume_note = ""

            def on_family(info: dict[str, Any]) -> None:
                progress(info["index"] / max(1, int(info["total"])), str(info["family"]))

            return benchmark_feature_families(
                smiles,
                y,
                task,
                model=None if model_name in ("", "auto") else model_name,
                cv=folds,
                n_jobs=bench_njobs,
                on_family=on_family,
                checkpoint_dir=self._bench_checkpoint_dir,
            )

        def done(board: Any) -> None:
            columns = [str(column) for column in board.columns]
            rows = [
                [_cell_text(value, limit=30) for value in record]
                for record in board.to_numpy().tolist()
            ]
            _populate_table(self._bench_board, columns, rows)
            metric_column = next(
                (column for column in board.columns if str(column).endswith("_mean")), None
            )
            n_nan = int(board[metric_column].isna().sum()) if metric_column is not None else 0
            note = (
                f"{folds}-fold family benchmark over the 2D families "
                "(shape_3d and rism opt-in only) — the score column is the "
                "task's objective (higher is better)"
            )
            if n_nan:
                note += f"; {n_nan} family(ies) failed and sank"
            if self._bench_resume_note:
                note = f"{self._bench_resume_note}; {note}"
            self._bench_status.setText(note)
            # part5 §11: the finished board becomes a saved evaluation —
            # the ML card's real evidence. Best-effort by contract.
            # (round 6, §14.2 Proposal A) the record carries the
            # dataset's workspace provenance — the current content
            # version + column signature at the time of the run — so a
            # later re-exported CSV never silently invalidates the
            # stored numbers.
            try:
                from pharmaml.studio import record_run
                from pharmaml.studio import workspace as _ws

                dataset_provenance: dict[str, Any] = {}
                source = getattr(self._model_analysis, "_external_path", None)
                if source is not None:
                    prior = next((
                        e for e in _ws.entries("dataset")
                        if e.path == str(source)
                    ), None)
                    if prior is not None:
                        dataset_provenance = {
                            "dataset_name": prior.name,
                            "dataset_version": prior.content_version,
                            "dataset_sha256": prior.sha256[:16],
                            "dataset_columns": prior.column_signature,
                        }

                best_row = None
                if metric_column is not None:
                    finite = board[board[metric_column].notna()]
                    if not finite.empty:
                        best_row = finite.loc[finite[metric_column].idxmax()]
                record_run(
                    "ml-evaluation",
                    f"family benchmark · {target} ({folds}-fold CV)",
                    {
                        "eval_kind": "cv",
                        "folds": folds,
                        "families": len(board),
                        "metric": (
                            str(metric_column)[: -len("_mean")] if metric_column else "score"
                        ),
                        "best_mean": (
                            float(best_row[metric_column]) if best_row is not None else None
                        ),
                        "best_family": (
                            str(best_row["family"]) if best_row is not None else None
                        ),
                        "target": target,
                        "task": task,
                        "n_failed": int(n_nan),
                        **dataset_provenance,
                    },
                )
            except Exception:  # noqa: BLE001, S110 -- the record must never fail the board
                pass

        self._submit(f"benchmarking feature families ({task})", job, done, self._bench_status)

    def _start_stack(self) -> None:
        """The out-of-fold stacking leaderboard + parity scatter."""
        frame, target, task = self._training_context()
        if frame is None:
            self._stack_status.setText(
                "Load a dataset on the Model Analysis page first -- the stack "
                "regresses its target."
            )
            return
        if not target:
            self._stack_status.setText("Pick a target column on the Model Analysis page first.")
            return
        numeric, dropped = numeric_feature_columns(frame, target)
        if not numeric:
            self._stack_status.setText(
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + " -- the models need numeric feature columns (compute "
                "descriptors first, the Chemistry tab's feature factory)."
            )
            return
        if task != "regression":
            self._stack_status.setText(
                f"The OOF stacking card is a regressor — the current task "
                f"resolves to {task}. Train a classification model from the "
                "Model Analysis page instead."
            )
            return
        base_models = [
            part.strip() for part in self._stack_base.text().split(",") if part.strip()
        ]
        if not base_models:
            self._stack_status.setText(
                "Name at least one base model (comma-separated registry names, "
                "e.g. ridge, random_forest) -- the stack needs something to blend."
            )
            return
        meta_models = [
            part.strip() for part in self._stack_meta.text().split(",") if part.strip()
        ] or None
        folds = int(self._stack_folds.value())
        test_percent = int(self._stack_test.value())
        quantile = bool(self._stack_quantile.isChecked())
        residual = bool(self._stack_residual.isChecked())
        n_base = len(base_models)
        # the cleared-meta fallback rides the library's default trio, so
        # the status line keeps its shape either way.
        n_meta = len(meta_models) if meta_models else 3

        # the split is built on the MAIN thread (the worker never touches
        # the frame), with the notebook's random_state=42.
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            frame[numeric], frame[target], test_size=test_percent / 100.0, random_state=42
        )

        def job(progress: Any) -> Any:
            from pharmaml.ml import stack_oof_regressors

            def on_step(info: dict[str, Any]) -> None:
                progress(info["index"] / max(1, int(info["total"])), str(info["label"]))

            return stack_oof_regressors(
                x_train,
                y_train,
                x_test,
                y_test,
                base_models=base_models,
                meta_models=meta_models,
                n_splits=folds,
                target_transform="quantile" if quantile else "none",
                residual=residual,
                random_state=42,
                on_step=on_step,
            )

        def done(result: Any) -> None:
            self._stack_last = result
            columns = [str(column) for column in result.leaderboard.columns]
            rows = [
                [_cell_text(value, limit=30) for value in record]
                for record in result.leaderboard.to_numpy().tolist()
            ]
            _populate_table(self._stack_board, columns, rows)
            note = (
                f"{folds}-fold OOF stacking · {n_base} base × {n_meta} meta · "
                f"best blender: {result.best_meta} (test R² {result.best_test_r2:.4f})"
            )
            if result.residual_test_r2 is not None:
                note += f" · +residual test R² {result.residual_test_r2:.4f}"
            note += (
                " · quantile transform on: R² is on the TRANSFORMED scale"
                if quantile
                else " · R² on the raw target scale"
            )
            self._stack_status.setText(note)
            # the notebook's parity scatter, drawn Qt-native (§16.8): the
            # best blender's test predictions against the (transformed)
            # truth, the y=x diagonal as the dashed reference.
            self._stack_plot.clear()
            truth = np.asarray(result.y_test_trans, dtype=float)
            predicted = np.asarray(result.best_test_prediction, dtype=float)
            self._stack_plot.scatter(truth, predicted, label="held-out")
            low = float(min(truth.min(), predicted.min()))
            high = float(max(truth.max(), predicted.max()))
            span = np.array([low, high])
            self._stack_plot.line(span, span, label="y = x", width=1, index=1, style="dash")
            title = f"OOF stacking — {result.best_meta}"
            if result.residual_test_r2 is not None:
                title += " + residual"
            self._stack_plot.set_labels(x="true (target scale)", y="predicted", title=title)

        label = f"OOF stacking ({n_base} base × {n_meta} meta)"
        self._submit(label, job, done, self._stack_status)

    # -- tab 3: ensemble builder -------------------------------------------------
    def _build_ensemble_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        members = QWidget()
        grid = QGridLayout(members)
        grid.setContentsMargins(0, 0, 0, 0)
        self._ens_members: dict[str, QCheckBox] = {}
        for index, (token, label) in enumerate(ENSEMBLE_MEMBER_LABELS):
            box = QCheckBox(label)
            box.setChecked(token in ENSEMBLE_DEFAULT_MEMBERS)
            box.setToolTip(
                f"Tick to include the {token} registry family in the ensemble "
                "(aliases resolve, duplicates collapse)."
            )
            self._ens_members[token] = box
            grid.addWidget(box, index // 2, index % 2)
        form.addRow("members:", members)
        self._ens_kind = QComboBox()
        for token in ("voting", "stacking"):
            self._ens_kind.addItem(token)
        self._ens_kind.setToolTip(
            "voting = the registry VotingClassifier/Regressor (soft = "
            "probability averaging, needs predict_proba on every member; "
            "hard = majority vote); stacking = StackingClassifier/Regressor "
            "with the meta model blending the members' predictions."
        )
        form.addRow("kind:", self._ens_kind)
        self._ens_voting = QComboBox()
        for token in ("soft", "hard"):
            self._ens_voting.addItem(token)
        self._ens_voting.setToolTip(
            "Voting kind only: soft averages the members' probabilities "
            "(needs predict_proba everywhere); hard counts majority votes "
            "(any classifier, any class count). Ignored for regression."
        )
        form.addRow("voting:", self._ens_voting)
        self._ens_meta = QComboBox()
        self._ens_meta.addItem("auto")
        try:
            from pharmaml.ml.models import MODEL_REGISTRY

            for key in sorted(MODEL_REGISTRY):
                self._ens_meta.addItem(key)
        except Exception:  # noqa: BLE001, S110 - broken-install fallback to 'auto'
            pass
        self._ens_meta.setToolTip(
            "Stacking kind only: the level-1 blender's registry name ('auto' "
            "= the library default: logistic_regression for classification, "
            "ridge for regression)."
        )
        form.addRow("meta model:", self._ens_meta)
        self._ens_btn = QPushButton("Train ensemble")
        self._ens_btn.setMinimumHeight(34)
        self._ens_btn.clicked.connect(self._start_ensemble)
        self._ens_btn.setToolTip(
            "Build and fit the ensemble of the ticked members on a held-out "
            "split (80/20, stratified for classification), score it like any "
            "trained model, and keep the fitted estimator for the persistence "
            "and applicability-domain cards."
        )
        form.addRow("", self._ens_btn)
        lay.addLayout(form)
        self._ens_summary = self._status_label(
            "no ensemble yet -- tick the members and hit Train ensemble."
        )
        lay.addWidget(self._ens_summary)
        self._ens_plot = PlotPanel(title="Ensemble (parity / class counts)")
        self._ens_plot.set_empty_hint("Run an ensemble to see parity and class counts")

        lay.addWidget(self._ens_plot, stretch=1)
        return tab

    def _start_ensemble(self) -> None:
        """Train a voting / stacking ensemble of the ticked members."""
        frame, target, task = self._training_context()
        if frame is None:
            self._ens_summary.setText(
                "Load a dataset on the Model Analysis page first -- the "
                "ensemble trains on its frame and target."
            )
            return
        if not target:
            self._ens_summary.setText("Pick a target column on the Model Analysis page first.")
            return
        numeric, dropped = numeric_feature_columns(frame, target)
        if not numeric:
            self._ens_summary.setText(
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + " -- the ensemble needs numeric feature columns (compute "
                "descriptors first, the Chemistry tab's feature factory)."
            )
            return
        members = [token for token, box in self._ens_members.items() if box.isChecked()]
        if not members:
            self._ens_summary.setText("Tick at least one base model for the ensemble.")
            return
        kind = self._ens_kind.currentText()
        voting_rule = self._ens_voting.currentText()
        meta_token = self._ens_meta.currentText()
        if kind == "voting" and task == "regression":
            voting_rule = "soft"  # ignored by VotingRegressor; keep the token valid

        # the split is built on the MAIN thread (the worker never touches
        # the frame), the tuner's own 80/20 with stratification.
        from sklearn.model_selection import train_test_split

        stratify = None if task == "regression" else frame[target].astype(str).to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(
            frame[numeric], frame[target], test_size=0.25, random_state=42, stratify=stratify
        )

        def job(progress: Any) -> Any:
            from pharmaml.ml.evaluation import evaluate_model
            from pharmaml.ml.models import build_stacking, build_voting

            progress(0.1, "building the ensemble")
            if kind == "voting":
                estimator = build_voting(task, members, voting=voting_rule)
                name = f"voting ({'+'.join(members)})"
            else:
                meta = None if meta_token == "auto" else meta_token
                estimator = build_stacking(task, members, meta_model=meta)
                name = f"stacking ({'+'.join(members)} → {meta_token})"
            progress(0.4, "fitting")
            estimator.fit(x_train, y_train)
            # the regression-only train-side pair -- the report carries the
            # overfit gap too (the D2.py audit's one-number diagnosis).
            train_side: dict[str, Any] = (
                {"X_train": x_train, "y_train": y_train} if task == "regression" else {}
            )
            report = evaluate_model(
                estimator,
                x_test,
                y_test,
                task,
                keep_predictions=(task == "regression"),
                **train_side,
            )
            progress(1.0, "done")
            return estimator, report, task, members, name, x_train, y_train, x_test, y_test

        def done(payload: Any) -> None:
            estimator, report, task_, members_, name, xtr, ytr, xte, yte = payload
            self._ens_last = (estimator, report, task_, members_, name, xtr, ytr, xte, yte, target)
            held = report.r2 if task_ == "regression" else report.accuracy
            held_label = "R²" if task_ == "regression" else "accuracy"
            text = f"{name} trained — held-out {held_label} {held:.4f}"
            gap = getattr(report, "r2_gap", None)
            if gap is not None:
                text += f" · train-side overfit gap {gap:+.4f} (a diagnostic, not a quality claim)"
            self._ens_summary.setText(text)
            self._ens_plot.clear()
            if task_ == "regression":
                y_true = np.asarray(yte, dtype=float)
                predicted = getattr(report, "predictions", None)
                if predicted is None:
                    predicted = estimator.predict(xte)
                y_pred = np.asarray(predicted, dtype=float)
                self._ens_plot.scatter(y_true, y_pred, label="held-out samples")
                low = float(min(y_true.min(), y_pred.min()))
                high = float(max(y_true.max(), y_pred.max()))
                span = np.array([low, high])
                self._ens_plot.line(span, span, label="y = x", width=1, index=1, style="dash")
                self._ens_plot.set_labels(x="actual", y="predicted", title=f"{name} — parity")
            else:
                labels, counts = np.unique(np.asarray(yte).astype(str), return_counts=True)
                self._ens_plot.bars(
                    counts.astype(float), labels=[str(value) for value in labels]
                )
                self._ens_plot.set_labels(
                    x="class", y="count", title="Held-out class counts"
                )

        self._submit(f"training {kind} ensemble", job, done, self._ens_summary)

    # -- tab 4: persistence --------------------------------------------------------
    def _build_persistence_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        buttons = QHBoxLayout()
        self._persist_save_btn = QPushButton("Save model bundle…")
        self._persist_save_btn.clicked.connect(self._save_bundle)
        self._persist_save_btn.setToolTip(
            "Write the fitted estimator (this page's ensemble, else the Model "
            "Analysis page's last run) as a joblib BUNDLE with its provenance "
            "metadata -- name, task, feature order, held-out metrics, target "
            "column, toolkit/sklearn versions -- plus the .json sidecar."
        )
        buttons.addWidget(self._persist_save_btn)
        self._persist_open_btn = QPushButton("Open model bundle…")
        self._persist_open_btn.clicked.connect(self._open_bundle)
        self._persist_open_btn.setToolTip(
            "Load a bundle saved by save_model: the estimator comes back with "
            "its metadata (name, task, trained-at, feature contract, the "
            "training stack's versions) -- ready for Predict on CSV and the "
            "applicability-domain card."
        )
        buttons.addWidget(self._persist_open_btn)
        self._persist_predict_btn = QPushButton("Predict on CSV…")
        self._persist_predict_btn.clicked.connect(self._predict_csv)
        self._persist_predict_btn.setToolTip(
            "Audit a picked CSV against the loaded bundle's feature contract "
            "(missing/extra/reordered columns named BEFORE any prediction "
            "runs), then predict: a 'prediction' column plus p_<class> "
            "columns when the classifier exposes predict_proba."
        )
        buttons.addWidget(self._persist_predict_btn)
        self._persist_write_btn = QPushButton("Save predictions…")
        self._persist_write_btn.clicked.connect(self._save_predictions)
        self._persist_write_btn.setToolTip(
            "Write the full prediction table (the preview below caps at 12 "
            "rows) as a CSV."
        )
        buttons.addWidget(self._persist_write_btn)
        self._persist_card_btn = QPushButton("Write model card…")
        self._persist_card_btn.clicked.connect(self._write_model_card)
        self._persist_card_btn.setToolTip(
            "Build and write the model card for the loaded bundle — "
            "markdown or HTML by the chosen suffix. The card carries the "
            "bundle's provenance: name, task, feature contract, training "
            "stack versions, and (when the current dataset matches) the "
            "per-feature min/max applicability box."
        )
        buttons.addWidget(self._persist_card_btn)
        self._persist_manifest_btn = QPushButton("Write reproducibility manifest…")
        self._persist_manifest_btn.clicked.connect(self._write_manifest)
        self._persist_manifest_btn.setToolTip(
            "Export the environment + workflow slice as a JSON sidecar "
            "beside the bundle: pharmaml/Python/pinned-stack versions, "
            "platform, the recorded runs' hyperparameters/seeds/n_jobs, "
            "and the workspace datasets' fingerprints + content versions "
            "(§14.2 Proposal B). Descriptive, not executable — nothing "
            "is reinstalled or re-run."
        )
        buttons.addWidget(self._persist_manifest_btn)
        form.addRow("", buttons)
        lay.addLayout(form)
        self._persist_status = self._status_label(
            "no bundle yet -- train a model (the Model Analysis page) or an "
            "ensemble here, then save it."
        )
        lay.addWidget(self._persist_status)
        preview = QLabel("Prediction preview (first 12 rows):")
        lay.addWidget(preview)
        self._predict_table = self._make_board(minimum_height=180)
        self._predict_table.setToolTip(
            "The prediction table's preview -- the full table rides Save "
            "predictions…, never this capped view."
        )
        lay.addWidget(self._predict_table, stretch=1)
        return tab

    def _persistence_source(self) -> tuple[str, Any, str, str | None, list[str], dict[str, float]] | None:
        """The first available fitted model: this page's ensemble, else the
        Model Analysis page's last training run (``None`` when neither)."""
        if self._ens_last is not None:
            estimator, report, task, _members, name, x_train, _yt, _xt, _ys, target = self._ens_last
            features = [str(column) for column in list(getattr(x_train, "columns", []))]
            return name, estimator, task, target, features, _report_metrics(report)
        run = getattr(self._model_analysis, "last_run", None)
        outcome = run() if callable(run) else None
        if outcome is not None and getattr(outcome, "estimator", None) is not None:
            name = str(getattr(outcome, "model", "model"))
            task = str(getattr(outcome, "task", "regression"))
            features = [str(column) for column in list(getattr(outcome, "feature_names", None) or [])]
            metrics = {
                str(key): float(value)
                for key, value in dict(getattr(outcome, "metrics", None) or {}).items()
                if isinstance(value, (int, float))
            }
            return name, outcome.estimator, task, self._current_target(), features, metrics
        return None

    def _save_bundle(self) -> None:
        """Ask a path, then write the bundle (the card's own refusal first)."""
        source = self._persistence_source()
        if source is None:
            self._persist_status.setText(
                "Train a model or an ensemble first — the bundle saves the "
                "fitted estimator and its metrics."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save the model bundle",
            f"{_slug(source[0])}.joblib",
            "Model bundle (*.joblib *.pkl);;All files (*)",
        )
        if not path:
            return
        self._save_bundle_to(path)

    def _save_bundle_to(self, path: str) -> None:
        """The writer behind the Save button (tests drive this, dialog-free)."""
        source = self._persistence_source()
        if source is None:
            self._persist_status.setText(
                "Train a model or an ensemble first — the bundle saves the "
                "fitted estimator and its metrics."
            )
            return
        name, model, task, target, features, metrics = source

        def job(progress: Any) -> Any:
            from pharmaml.ml.persistence import save_model, write_metadata_sidecar

            progress(0.5, "writing the bundle")
            written = save_model(
                model,
                path,
                name=name,
                task=task,
                features=list(features) if features else None,
                best_params={},
                metrics=metrics,
                target=target,
            )
            sidecar = write_metadata_sidecar(written)
            progress(1.0, "done")
            return written, sidecar

        def done(payload: Any) -> None:
            written, sidecar = payload
            self._persist_status.setText(f"bundle written: {written.name} (+ {sidecar.name})")
            # part6 §8 (Phase 4): a saved bundle enters the shared
            # workspace's model registry (best-effort side record).
            try:
                from pharmaml.studio import note as note_workspace

                note_workspace("bundle", written, note=f"{sidecar.name} alongside")
            except Exception:  # Never block the save's success path.
                logging.getLogger(__name__).warning("Could not record bundle in workspace", exc_info=True)

        self._submit("saving model bundle", job, done, self._persist_status)

    def _open_bundle(self) -> None:
        """Load a saved bundle and show its provenance summary."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a model bundle", "", "Model bundle (*.joblib *.pkl);;All files (*)"
        )
        if not path:
            return
        self._load_bundle_from(Path(path))

    def _load_bundle_from(self, path: Path) -> None:
        """The loader behind the Open button (tests drive this, dialog-free)."""
        def job(progress: Any) -> Any:
            from pharmaml.ml.persistence import load_model

            progress(0.5, "reading the bundle")
            return load_model(path)

        def done(payload: Any) -> None:
            model, metadata = payload
            self._bundle = (model, metadata, Path(path))
            self._predictions = None
            _populate_table(self._predict_table, [], [])
            features = metadata.get("features")
            feature_text = f"{len(features)} features" if features else "feature count contract"
            self._persist_status.setText(
                f"{metadata.get('name', '?')} · {metadata.get('task', '?')} · "
                f"trained {str(metadata.get('trained_at', '?'))[:19]} · "
                f"{feature_text} · pharmaml {metadata.get('pharmaml_version', '?')} "
                f"· sklearn {metadata.get('sklearn_version', '?')}"
            )

        self._submit(f"loading {Path(path).name}", job, done, self._persist_status)

    def _predict_csv(self) -> None:
        """Audit a picked CSV against the bundle's contract and predict."""
        if self._bundle is None:
            self._persist_status.setText(
                "Open a model bundle first — Predict scores a CSV against "
                "the loaded bundle's feature contract."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open the data to predict on (CSV or Excel)", "", "Tables (*.csv *.tsv *.xlsx);;All files (*)"
        )
        if not path:
            return
        self._predict_on(Path(path))

    def _predict_on(self, path: Path) -> None:
        """The predictor behind the Predict button (tests drive this)."""
        if self._bundle is None:
            self._persist_status.setText(
                "Open a model bundle first — Predict scores a CSV against "
                "the loaded bundle's feature contract."
            )
            return
        model, metadata, bundle_path = self._bundle
        task = str(metadata.get("task", "regression"))

        def job(progress: Any) -> Any:
            from pharmaml.ml.persistence import validate_bundle_input
            from pharmaml.utils.io import read_table

            progress(0.2, "reading the table")
            frame = read_table(path)
            # the training target rides along when the scored CSV is the
            # training table itself -- the bundle records its name, and the
            # audit drops it automatically (the ml check convention).
            target_name = metadata.get("target")
            if target_name and target_name in frame.columns:
                frame = frame.drop(columns=[target_name])
            audit = validate_bundle_input(bundle_path, frame)
            if not audit.ok:
                raise ValueError(
                    f"the CSV does not satisfy the bundle's feature contract:\n{audit}"
                )
            features = metadata.get("features")
            x = frame[list(features)] if features else frame.select_dtypes("number")
            progress(0.7, "predicting")
            prediction = model.predict(x)
            result = frame.copy()
            result["prediction"] = prediction
            if task == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)
                classes = list(getattr(model, "classes_", []))
                for index, cls in enumerate(classes):
                    result[f"p_{cls}"] = proba[:, index]
            progress(1.0, "done")
            return result

        def done(result: Any) -> None:
            self._predictions = result
            columns = [str(column) for column in result.columns]
            preview = result.head(12)
            rows = [
                [_cell_text(value, limit=26) for value in record]
                for record in preview.to_numpy().tolist()
            ]
            _populate_table(self._predict_table, columns, rows)
            self._persist_status.setText(
                f"{len(result)} rows predicted — Save predictions… writes the full table"
            )

        self._submit(f"predicting on {path.name}", job, done, self._persist_status)

    def _save_predictions(self) -> None:
        """Write the last prediction table as a CSV."""
        if self._predictions is None:
            self._persist_status.setText(
                "Predict on a CSV first — there is no prediction table to save yet."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the prediction table", "predictions.csv", "CSV table (*.csv)"
        )
        if not path:
            return
        self._save_predictions_to(Path(path))

    def _save_predictions_to(self, path: Path) -> None:
        """The writer behind the Save-predictions button (dialog-free)."""
        frame = self._predictions
        if frame is None:
            self._persist_status.setText(
                "Predict on a CSV first — there is no prediction table to save yet."
            )
            return

        def job(progress: Any) -> Any:
            progress(0.5, "writing the CSV")
            frame.to_csv(path, index=False)
            progress(1.0, "done")
            return path

        def done(written: Any) -> None:
            self._persist_status.setText(f"Predictions written to {written}")

        self._submit("saving predictions", job, done, self._persist_status)

    def _write_model_card(self) -> None:
        """Write the model card (markdown or HTML) for the loaded bundle."""
        if self._bundle is None:
            self._persist_status.setText(
                "Open a model bundle first — the card documents a saved "
                "bundle's provenance."
            )
            return
        _model, _metadata, bundle_path = self._bundle
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Write model card",
            str(bundle_path.with_suffix(".md")),
            "Markdown (*.md);;HTML (*.html)",
        )
        if not path:
            return
        frame, _target, _task = self._training_context()

        def job(progress: Any) -> Any:
            from pharmaml.ml.model_card import write_model_card

            progress(0.5, "assembling the card from the bundle")
            return write_model_card(
                bundle_path, path, training_frame=frame
            )

        def done(written: Any) -> None:
            self._persist_status.setText(f"Model card written to {written}")

        self._submit("writing the model card", job, done, self._persist_status)

    def _write_manifest(self) -> None:
        """Write the reproducibility manifest (§14.2 Proposal B).

        The manifest lands beside the loaded bundle (``<bundle>.manifest.json``)
        and captures the environment + the workspace's recorded slice:
        pinned-stack versions, per-run hyperparameters/seeds/n_jobs, and
        dataset fingerprints with Proposal A content versions.
        """
        if self._bundle is None:
            self._persist_status.setText(
                "Open a model bundle first — the manifest documents a "
                "saved bundle's environment."
            )
            return
        _model, _metadata, bundle_path = self._bundle
        target = bundle_path.with_suffix(".manifest.json")
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Write reproducibility manifest",
            str(target),
            "JSON (*.json)",
        )
        if not path:
            return

        def job(progress: Any) -> Any:
            from pharmaml.studio import (
                entries as workspace_entries,
                latest_runs,
                write_manifest,
            )

            progress(0.4, "collecting the workspace slice")
            runs = [r for f in (
                "ml-evaluation", "dl-training",
            ) for r in latest_runs(f, limit=5)]
            datasets = [
                e for e in workspace_entries("dataset") if e.sha256
            ][:8]
            progress(0.7, f"{len(runs)} run(s), {len(datasets)} dataset(s)")
            return write_manifest(path, runs=runs, datasets=datasets)

        def done(written: Any) -> None:
            self._persist_status.setText(f"Manifest written to {written}")

        self._submit("writing the reproducibility manifest", job, done, self._persist_status)

    # -- tab 5: applicability domain -------------------------------------------------
    def _build_ad_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self._ad_source = QComboBox()
        for token in AD_SOURCES:
            self._ad_source.addItem(token)
        self._ad_source.setToolTip(
            "Where the model and its split come from: the Model Analysis "
            "page's last training run (its fitted estimator + split), this "
            "page's ensemble (the Ensemble builder tab), or a bundle loaded "
            "on the Persistence tab (the frames rebuilt from the CURRENT "
            "dataset's numeric columns)."
        )
        form.addRow("source:", self._ad_source)
        self._ad_k = QSpinBox()
        self._ad_k.setRange(1, 20)
        self._ad_k.setValue(5)
        self._ad_k.setToolTip(
            "Neighbours of the distance domain: the mean kNN distance of "
            "each query row in the standardised descriptor space, compared "
            "against the training rows' percentile threshold."
        )
        form.addRow("k (neighbours):", self._ad_k)
        self._ad_pct = QDoubleSpinBox()
        self._ad_pct.setRange(50.0, 99.9)
        self._ad_pct.setDecimals(1)
        self._ad_pct.setSingleStep(1.0)
        self._ad_pct.setValue(95.0)
        self._ad_pct.setToolTip(
            "The percentile BOTH thresholds are cut at (the distance "
            "threshold from the training kNN distances, the uncertainty "
            "threshold from the training-side ensemble spread)."
        )
        form.addRow("threshold percentile:", self._ad_pct)
        self._ad_boot = QSpinBox()
        self._ad_boot.setRange(0, 100)
        self._ad_boot.setValue(30)
        self._ad_boot.setToolTip(
            "Bootstrap ensemble size for the uncertainty half (0 = the "
            "kNN-only pass, no bootstrap fits); each member refits on a "
            "resample and the spread becomes the per-row uncertainty."
        )
        form.addRow("bootstrap models:", self._ad_boot)
        # (part8 Phase 10) the four required analyses behind one selector:
        # training-distance distribution + threshold, query distance vs
        # predicted value (in/out categories), uncertainty distribution
        # (when the bootstrap half ran), and the combined reliability
        # view (AD x uncertainty, the backend's own decision rule).
        self._ad_view = QComboBox()
        for token in (
            "query distance (inside/outside)",
            "training distance distribution",
            "distance vs predicted value",
            "uncertainty distribution",
            "combined reliability",
        ):
            self._ad_view.addItem(token)
        self._ad_view.setToolTip(
            "Which Applicability-Domain analysis to display. The "
            "uncertainty and combined views need the bootstrap half "
            "(bootstrap models > 0)."
        )
        self._ad_view.currentTextChanged.connect(self._render_ad_view)
        form.addRow("view:", self._ad_view)
        self._ad_btn = QPushButton("Run AD analysis")
        self._ad_btn.setMinimumHeight(34)
        self._ad_btn.clicked.connect(self._start_ad)
        self._ad_btn.setToolTip(
            "The uploaded workflow in one click: standardise the descriptor "
            "space, measure every query row's mean kNN distance against the "
            "percentile threshold, bootstrap the model ensemble for the "
            "uncertainty spread, and fold both into the per-row reliability "
            "verdict (inside AD & low uncertainty)."
        )
        form.addRow("", self._ad_btn)
        lay.addLayout(form)
        self._ad_status = self._status_label(
            "the applicability-domain analysis reuses a trained run, this "
            "page's ensemble, or a loaded bundle."
        )
        lay.addWidget(self._ad_status)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._ad_table = self._make_board(minimum_height=170)
        self._ad_table.setToolTip(
            "The reliability table, one row per query compound: name, "
            "predicted, prediction_sd, knn_distance, inside_ad, "
            "low_uncertainty, reliable (the uncertainty columns degrade to "
            "NA in the kNN-only pass)."
        )
        splitter.addWidget(self._ad_table)
        # (part8 Phase 10) click a row -> the sample dossier (SMILES,
        # prediction, distances, thresholds, the reliability verdict).
        self._ad_table.cellClicked.connect(self._ad_inspect_row)
        self._ad_plot = PlotPanel(title="Applicability domain — inside vs outside")
        self._ad_plot.set_empty_hint("Train a model, then compute the applicability domain")

        splitter.addWidget(self._ad_plot)
        splitter.setSizes([520, 420])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)
        return tab

    def _start_ad(self) -> None:
        """The uploaded applicability-domain workflow, one click."""
        source = self._ad_source.currentText()
        k = int(self._ad_k.value())
        percentile = float(self._ad_pct.value())
        n_models = int(self._ad_boot.value())
        frame, target, _task = self._training_context()
        model: Any = None
        x_train: Any = None
        y_train: Any = None
        x_test: Any = None
        task = ""
        query_names: list[str] | None = None
        label = "model"

        if source == "Model Analysis run":
            run = getattr(self._model_analysis, "last_run", None)
            outcome = run() if callable(run) else None
            if outcome is None or getattr(outcome, "estimator", None) is None or getattr(
                outcome, "X_test", None
            ) is None:
                self._ad_status.setText(
                    "Train a model on the Model Analysis page first — the "
                    "analysis reuses its fitted estimator and split (or "
                    "switch the source below)."
                )
                return
            model = outcome.estimator
            x_train, y_train, x_test = outcome.X_train, outcome.y_train, outcome.X_test
            task = str(outcome.task)
            label = f"{outcome.model} ({task})"
        elif source == "Workbench ensemble":
            if self._ens_last is None:
                self._ad_status.setText(
                    "Train an ensemble on the Ensemble builder tab first — "
                    "the analysis reuses its split (or switch the source)."
                )
                return
            estimator, _report, task, _members, name, xtr, ytr, xte, _yte, _tgt = self._ens_last
            model, x_train, y_train, x_test = estimator, xtr, ytr, xte
            label = name
        else:  # loaded bundle
            if self._bundle is None:
                self._ad_status.setText(
                    "Open a model bundle on the Persistence tab first — the "
                    "bundle mode scores the loaded model over the current "
                    "dataset."
                )
                return
            if frame is None or not target:
                self._ad_status.setText(
                    "Load a dataset on the Model Analysis page first — the "
                    "bundle mode builds its train/test frames from the "
                    "current dataset's numeric columns."
                )
                return
            model, metadata, _bundle_path = self._bundle
            task = str(metadata.get("task", "regression"))
            label = str(metadata.get("name", "bundle"))
            numeric, dropped = numeric_feature_columns(frame, target)
            if not numeric:
                self._ad_status.setText(
                    "Every non-target column is non-numeric"
                    + (f" ({', '.join(dropped)})" if dropped else "")
                    + " — the bundle mode needs the dataset's numeric "
                    "descriptor columns."
                )
                return
            from sklearn.model_selection import train_test_split

            stratify = None if task == "regression" else frame[target].astype(str).to_numpy()
            x_train, x_test, y_train, _y_test = train_test_split(
                frame[numeric], frame[target], test_size=0.25, random_state=42,
                stratify=stratify,
            )

        # the query names: the original frame's SMILES (or first column)
        # looked up through the preserved split index -- the upload's own
        # test_df.iloc[:, 0] convention.
        if frame is not None and x_test is not None:
            smiles = detect_smiles_column(frame)
            if smiles is not None:
                try:
                    query_names = [str(value) for value in frame.loc[x_test.index, smiles]]
                except Exception:  # noqa: BLE001 - index drift: generic names
                    query_names = None

        # the k-guard belongs to EVERY too-few-rows case, before the job.
        n_train = 0 if x_train is None else len(x_train)
        if n_train <= k:
            self._ad_status.setText(
                f"k={k} needs at least {k + 1} training rows; only {n_train} "
                "available -- lower k or skip the analysis."
            )
            return

        def job(progress: Any) -> Any:
            from pharmaml.ml import applicability_domain

            def on_fit(done_count: int, total: int) -> None:
                progress(
                    done_count / max(1, int(total)),
                    f"bootstrap member {done_count}/{total}",
                )

            return applicability_domain(
                model,
                x_train,
                y_train,
                x_test,
                k=k,
                percentile=percentile,
                n_models=max(n_models, 1),
                uncertainty=n_models > 0,
                progress=on_fit,
            )

        def done(result: Any) -> None:
            table = result.to_frame(names=query_names)
            self._ad_frame = table
            # (part8 Phase 10) the views re-render from the SAME stored
            # result; predictions come from the model on the query rows
            # (the ensemble's own readout the backend scored against).
            self._ad_result = result
            try:
                self._ad_query_pred = np.asarray(
                    model.predict(x_test), dtype=float
                ).ravel()
            except Exception:  # noqa: BLE001 - classifier or absent split
                self._ad_query_pred = None
            columns = [str(column) for column in table.columns]
            shown = table.head(200)
            rows = [
                [_cell_text(value, limit=26) for value in record]
                for record in shown.to_numpy().tolist()
            ]
            _populate_table(self._ad_table, columns, rows)
            inside = int(result.inside_ad.sum())
            total_rows = int(result.inside_ad.size)
            if result.reliable is not None:
                reliable = int(result.reliable.sum())
                low = int(result.low_uncertainty.sum())
                text = (
                    f"distance threshold {result.distance_threshold:.3g} "
                    f"(p{percentile:g}, k={k}) — {inside}/{total_rows} inside AD · "
                    f"{result.n_models}-member ensemble in "
                    f"{result.fit_seconds:.1f} s · uncertainty threshold "
                    f"{result.uncertainty_threshold:.3g} — {low}/{total_rows} "
                    f"low-uncertainty · {reliable}/{total_rows} reliable"
                )
            else:
                text = (
                    f"distance threshold {result.distance_threshold:.3g} "
                    f"(p{percentile:g}, k={k}) — {inside}/{total_rows} inside AD "
                    "(kNN-only pass: bootstrap models = 0)"
                )
            if len(table) > 200:
                text += f" · table capped at the first 200 of {len(table)} rows"
            self._ad_status.setText(text)
            self._render_ad_view(self._ad_view.currentText())

        self._submit(f"applicability domain ({label})", job, done, self._ad_status)

    # -- (part8 Phase 10) the AD view renderer --------------------------------------
    def _render_ad_view(self, view: str) -> None:
        """Render the selected Applicability-Domain analysis (Phase 10).

        One stored :class:`~pharmaml.ml.applicability.ApplicabilityDomainResult`
        behind five readouts — the audit round's "complete, scientifically
        interpretable workflow" rather than the single kNN chart.  Each
        view re-derives from the backend result; nothing is recomputed
        and no status is invented (the combined view degrades honestly
        when the bootstrap half did not run).
        """
        result = getattr(self, "_ad_result", None)
        if result is None:
            return
        plot = self._ad_plot
        plot.clear()
        distances = np.asarray(result.test_knn_mean, dtype=float)
        mask = np.asarray(result.inside_ad, dtype=bool)
        threshold = float(result.distance_threshold)
        total = int(distances.size)

        if view == "training distance distribution":
            # (A) the descriptor-space density the model was fitted on,
            # with the percentile threshold that splits inside/outside.
            # (bars() carries no series label — the histogram renders as
            # per-bin points at the bin centers, same information, and
            # the threshold line rides its own series.)
            train_d = np.asarray(result.train_knn_mean, dtype=float)
            counts, edges = np.histogram(train_d, bins=40)
            centers = (edges[:-1] + edges[1:]) / 2.0
            nonzero = counts > 0
            plot.scatter(
                centers[nonzero], counts[nonzero].astype(float),
                label="training rows",
            )
            plot.scatter(
                np.array([threshold]), np.array([float(counts.max()) * 0.05]),
                label="threshold",
            )
            plot.line(
                np.array([threshold, threshold]),
                np.array([0.0, float(counts.max())]),
                label="threshold line", width=1, index=1, style="dash",
            )
            plot.set_labels(x="mean kNN distance (training, standardised)",
                            y="molecules", title="Training distance distribution")
            return

        if view == "distance vs predicted value":
            # (B) query compounds by distance vs the ensemble prediction,
            # distinct in/out categories, threshold line.
            pred = getattr(self, "_ad_query_pred", None)
            if pred is None or pred.size != total:
                self._ad_status.setText(
                    "distance-vs-prediction needs the model's query "
                    "predictions — retrain, then re-run the analysis."
                )
                return
            plot.scatter(distances[mask], pred[mask], label="inside AD")
            plot.scatter(distances[~mask], pred[~mask], label="outside AD")
            yspan = np.array([float(pred.min()), float(pred.max())])
            plot.line(np.array([threshold, threshold]), yspan,
                      label="threshold", width=1, index=1, style="dash")
            plot.set_labels(x="mean kNN distance (query)",
                            y="predicted value",
                            title="Query distance vs predicted value")
            return

        if view == "uncertainty distribution":
            # (C) the bootstrap spread per query row against the
            # uncertainty threshold (only when the half actually ran).
            std = getattr(result, "pred_std", None)
            if getattr(result, "n_models", 0) in (None, 0) or std is None:
                self._ad_status.setText(
                    "uncertainty view: the kNN-only pass carries no "
                    "bootstrap spread — set bootstrap models > 0 and re-run."
                )
                return
            std = np.asarray(std, dtype=float)
            u_threshold = float(getattr(result, "uncertainty_threshold", 0.0) or 0.0)
            positions = np.arange(1, std.size + 1, dtype=float)
            low = np.asarray(result.low_uncertainty, dtype=bool)
            plot.scatter(positions[low], std[low], label="low uncertainty")
            plot.scatter(positions[~low], std[~low], label="high uncertainty")
            plot.line(np.array([1.0, float(std.size)]),
                      np.array([u_threshold, u_threshold]),
                      label="threshold", width=1, index=1, style="dash")
            plot.set_labels(x="query row", y="prediction SD",
                            title="Prediction uncertainty")
            return

        if view == "combined reliability":
            # (D) the four-quadrant reliability story on the backend's
            # own decision rule (reliable = inside AD & low uncertainty).
            if getattr(result, "n_models", 0) in (None, 0) or result.reliable is None:
                self._ad_status.setText(
                    "combined view: needs the bootstrap half — set "
                    "bootstrap models > 0 and re-run."
                )
                return
            positions = np.arange(1, total + 1, dtype=float)
            reliable = np.asarray(result.reliable, dtype=bool)
            low = np.asarray(result.low_uncertainty, dtype=bool)
            plot.scatter(positions[reliable], distances[reliable],
                         label="reliable (in AD, low unc.)")
            both = (~reliable) & mask & (~low)
            plot.scatter(positions[both], distances[both],
                         label="in AD, high uncertainty")
            out_low = (~mask) & low & (~reliable)
            plot.scatter(positions[out_low], distances[out_low],
                         label="out of AD, low uncertainty")
            out_high = (~mask) & (~low)
            plot.scatter(positions[out_high], distances[out_high],
                         label="out of AD, high uncertainty")
            plot.line(np.array([1.0, float(total)]),
                      np.array([threshold, threshold]),
                      label="distance threshold", width=1, index=1, style="dash")
            plot.set_labels(x="query row", y="kNN distance",
                            title="Combined reliability (AD × uncertainty)")
            return

        # default: the historical inside/outside figure
        positions = np.arange(1, total + 1, dtype=float)
        plot.scatter(positions[mask], distances[mask], label="inside AD")
        plot.scatter(positions[~mask], distances[~mask], label="outside AD")
        span = np.array([1.0, float(max(total, 1))])
        plot.line(span, np.array([threshold, threshold]),
                  label="threshold", width=1, index=1, style="dash")
        plot.set_labels(x="held-out row", y="kNN distance",
                        title="Applicability domain — inside vs outside")

    # -- (part8 Phase 10) sample-level inspection ------------------------------------
    def _ad_inspect_row(self, row: int, column: int) -> None:
        """Click a table row -> the compound's full AD dossier."""
        result = getattr(self, "_ad_result", None)
        frame = getattr(self, "_ad_frame", None)
        if result is None or frame is None or not 0 <= row < len(frame):
            return
        record = frame.iloc[row]
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Applicability domain — sample inspection")
        lay = QVBoxLayout(dialog)
        fields = [
            ("name / SMILES", str(record.get("name", ""))),
            ("predicted", f"{float(record.get('predicted', float('nan'))):.4g}"),
            ("kNN distance", f"{float(record.get('knn_distance', float('nan'))):.4g}"),
            ("distance threshold",
             f"{float(result.distance_threshold):.4g} "
             f"(p{getattr(result, 'percentile', 95):g}, k={result.k})"),
            ("inside AD", "yes" if bool(record.get("inside_ad", False)) else "no"),
        ]
        std = record.get("prediction_sd", None)
        if std is not None and std == std:
            fields.append(("prediction SD", f"{float(std):.4g}"))
            fields.append(("uncertainty threshold",
                           f"{float(getattr(result, 'uncertainty_threshold', 0.0) or 0.0):.4g}"))
            fields.append(("low uncertainty",
                           "yes" if bool(record.get("low_uncertainty", False)) else "no"))
            reliable = record.get("reliable", None)
            if reliable is not None and reliable == reliable:
                fields.append(("reliable (backend rule)",
                               "yes" if bool(reliable) else "no"))
        for label, value in fields:
            line = QLabel(f"<b>{label}:</b> {value}")
            line.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            lay.addWidget(line)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.reject)
        lay.addWidget(buttons)
        dialog.exec()

    # -- tab: semi-supervised learning (part8 Phase F) ------------------------------
    # The three scikit-learn semi-supervised estimators (SelfTraining,
    # LabelPropagation, LabelSpreading) with the conventional -1 unlabeled
    # marker, a leakage-safe evaluation against genuine held-out labels,
    # a supervised baseline, and the pseudo-label inspection table — one
    # canonical backend (pharmaml.ml.semi_supervised), never a GUI
    # reimplementation.
    def _build_semi_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()

        self._semi_method = QComboBox()
        for token in ("self-training", "label propagation", "label spreading"):
            self._semi_method.addItem(token)
        self._semi_method.setToolTip(
            "SelfTrainingClassifier iteratively labels the most confident "
            "unlabeled rows and refits; LabelPropagation/LabelSpreading "
            "spread labels over a similarity graph (spreading clamps the "
            "original labels with alpha)."
        )
        form.addRow("method:", self._semi_method)

        self._semi_unlabeled = QComboBox()
        self._semi_unlabeled.setEditable(True)
        self._semi_unlabeled.addItem("-1")
        self._semi_unlabeled.addItem("unlabeled")
        self._semi_unlabeled.addItem("NA")
        self._semi_unlabeled.setToolTip(
            "How unlabeled rows are marked in the target column. -1 is the "
            "conventional sentinel; 'unlabeled'/'NA' are converted safely "
            "before training (a legitimate class is never converted)."
        )
        form.addRow("unlabeled marker:", self._semi_unlabeled)

        self._semi_threshold = QDoubleSpinBox()
        self._semi_threshold.setRange(0.5, 0.99)
        self._semi_threshold.setDecimals(2)
        self._semi_threshold.setSingleStep(0.01)
        self._semi_threshold.setValue(0.75)
        self._semi_threshold.setToolTip(
            "Self-training only: the confidence cut a pseudo-label must "
            "clear. Consider calibrating the base estimator when the "
            "threshold is high (uncalibrated probabilities can stall the "
            "labelling)."
        )
        form.addRow("confidence threshold:", self._semi_threshold)

        self._semi_max_iter = QSpinBox()
        self._semi_max_iter.setRange(1, 100)
        self._semi_max_iter.setValue(10)
        self._semi_max_iter.setToolTip(
            "Max self-training iterations / graph propagation sweeps."
        )
        form.addRow("max iterations:", self._semi_max_iter)

        self._semi_kernel = QComboBox()
        for token in ("knn", "rbf"):
            self._semi_kernel.addItem(token)
        self._semi_kernel.setToolTip(
            "Graph methods only: the similarity kernel (kNN or RBF)."
        )
        form.addRow("kernel (graph methods):", self._semi_kernel)

        self._semi_neighbors = QSpinBox()
        self._semi_neighbors.setRange(1, 50)
        self._semi_neighbors.setValue(7)
        self._semi_neighbors.setToolTip("Graph methods only: kNN neighbour count.")
        form.addRow("k (neighbours):", self._semi_neighbors)

        self._semi_btn = QPushButton("Train + evaluate")
        self._semi_btn.setMinimumHeight(34)
        self._semi_btn.clicked.connect(self._start_semi)
        self._semi_btn.setToolTip(
            "Runs on the Model Analysis page's loaded frame and target: "
            "train the semi-supervised model over labeled + unlabeled rows, "
            "then score the leakage-safe protocol (held-out genuine labels, "
            "supervised baseline on the same split)."
        )
        form.addRow("", self._semi_btn)
        lay.addLayout(form)

        self._semi_status = self._status_label(
            "semi-supervised learning needs the Model Analysis page's frame "
            "with a classification target where some rows are unlabeled."
        )
        lay.addWidget(self._semi_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._semi_table = self._make_board(minimum_height=170)
        self._semi_table.setToolTip(
            "Sample-level result: the final label of every row (pseudo "
            "labels flagged), its confidence, and the SMILES when the "
            "frame carries one."
        )
        splitter.addWidget(self._semi_table)
        self._semi_plot = PlotPanel(title="Semi-supervised — confidence by row")
        self._semi_plot.set_empty_hint(
            "Train a semi-supervised model — pseudo-label confidences land here"
        )
        splitter.addWidget(self._semi_plot)
        configure_splitter(splitter, stretch=(2, 3))
        lay.addWidget(splitter, stretch=1)
        return tab

    def _start_semi(self) -> None:
        """Form → validation → worker → backend → table + figure."""
        frame, target, _task = self._training_context()
        if frame is None:
            self._semi_status.setText(
                "Load a dataset on the Model Analysis page first (e.g. "
                "qsar_classification.csv)."
            )
            return
        if not target:
            self._semi_status.setText(
                "Pick a classification target on the Model Analysis page."
            )
            return

        marker_text = self._semi_unlabeled.currentText().strip()
        marker: object = -1
        if marker_text not in ("-1", ""):
            marker = marker_text
        method_map = {
            "self-training": "self_training",
            "label propagation": "label_propagation",
            "label spreading": "label_spreading",
        }
        method = method_map[str(self._semi_method.currentText())]
        threshold = float(self._semi_threshold.value())
        max_iter = int(self._semi_max_iter.value())
        kernel = str(self._semi_kernel.currentText())
        neighbors = int(self._semi_neighbors.value())

        y_raw = frame[target]
        smiles_col = detect_smiles_column(frame)
        smiles: list[str] | None = None
        if smiles_col is not None:
            try:
                smiles = [str(v) for v in frame[smiles_col]]
            except Exception:  # noqa: BLE001 - display-only nicety
                smiles = None

        def job(progress: Any) -> Any:
            from pharmaml.ml.semi_supervised import (
                evaluate_semi_supervised,
                train_semi_supervised,
                validate_unlabeled,
            )

            progress(0.1, "validating labels")
            labels, info = validate_unlabeled(y_raw, unlabeled_marker=marker)
            progress(0.35, f"{info['n_unlabeled']} unlabeled of {len(labels)} rows")
            x = frame.select_dtypes("number").drop(columns=[target], errors="ignore")
            result = train_semi_supervised(
                x, labels, method=method,
                threshold=threshold, max_iter=max_iter,
                kernel=kernel, n_neighbors=neighbors,
            )
            progress(0.7, "evaluating (leakage-safe split)")
            evaluation = evaluate_semi_supervised(
                x, labels, method=method,
                threshold=threshold, max_iter=max_iter,
                kernel=kernel, n_neighbors=neighbors,
            )
            return result, evaluation, info

        def done(payload: Any) -> None:
            result, evaluation, info = payload
            self._semi_result = result
            table = result.to_frame()
            if smiles is not None and len(smiles) == len(table):
                table.insert(0, "SMILES", smiles)
            columns = [str(c) for c in table.columns]
            shown = table.head(200)
            rows = [
                [_cell_text(v, limit=26) for v in record]
                for record in shown.to_numpy().tolist()
            ]
            _populate_table(self._semi_table, columns, rows)
            self._semi_status.setText(
                f"{info['n_labeled']} labeled + {info['n_unlabeled']} "
                f"unlabeled — {evaluation.summary()}"
                + (f" · table capped at the first 200 of {len(table)} rows"
                   if len(table) > 200 else "")
            )
            # confidence by row, pseudo-labeled rows distinct from genuine
            self._semi_plot.clear()
            conf = np.asarray(result.confidence, dtype=float)
            pseudo = np.asarray(result.pseudo_label_mask, dtype=bool)
            pos = np.arange(1, conf.size + 1, dtype=float)
            self._semi_plot.scatter(
                pos[pseudo], conf[pseudo], label="pseudo-labeled"
            )
            self._semi_plot.scatter(
                pos[~pseudo], conf[~pseudo], label="genuinely labeled"
            )
            self._semi_plot.set_labels(
                x="row", y="confidence",
                title=f"Semi-supervised ({method}) — confidence by row",
            )

        self._submit(f"semi-supervised ({method})", job, done, self._semi_status)

    # -- tab 5b: active learning (part8 §14.1, round 5) ---------------------
    # The backend loop (ActiveLearningLoop) shipped in round 3; this tab
    # is the §14.1 required workflow made visible: Labeled + Pool →
    # train → strategy → ranked candidates → rationale → export →
    # import labels → retrain.  Labels are NEVER fabricated: the only
    # path from pool to labeled is the imported-CSV fold-in.
    def _build_al_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        header = QLabel(
            "Active learning for assay prioritization: hold out part of the "
            "labeled data as a candidate pool, rank the next batch by a "
            "query strategy, export the candidates for the lab, then import "
            "the measured labels and retrain. Experimental labels are only "
            "ever imported — never invented."
        )
        header.setWordWrap(True)
        lay.addWidget(header)
        form = QFormLayout()

        self._al_pool_pct = QSpinBox()
        self._al_pool_pct.setRange(10, 90)
        self._al_pool_pct.setValue(30)
        self._al_pool_pct.setToolTip(
            "Percent of the labeled rows held out as the unlabeled pool "
            "(seeded split, stratified for classification targets). The "
            "rest trains the first model."
        )
        form.addRow("pool holdout %:", self._al_pool_pct)

        self._al_strategy = QComboBox()
        for token in (
            "uncertainty", "margin", "entropy", "committee",
            "diversity", "representative",
        ):
            self._al_strategy.addItem(token)
        self._al_strategy.setToolTip(
            "uncertainty: 1 - max P(y|x), least confident first. margin: "
            "P(top1) - P(top2), smallest decision margin first. entropy: "
            "largest predictive entropy. committee: bootstrap-ensemble "
            "disagreement (std of P). diversity: farthest-from-labeled "
            "first (MaxMin distance). representative: densest pool "
            "regions first (kNN density)."
        )
        form.addRow("query strategy:", self._al_strategy)

        self._al_batch = QSpinBox()
        self._al_batch.setRange(1, 200)
        self._al_batch.setValue(10)
        self._al_batch.setToolTip(
            "How many candidates one query round exports for the lab."
        )
        form.addRow("batch size:", self._al_batch)

        self._al_seed = QSpinBox()
        self._al_seed.setRange(0, 9999)
        self._al_seed.setValue(0)
        self._al_seed.setToolTip("Seeded split + strategy reproducibility.")
        form.addRow("random seed:", self._al_seed)

        go_row = QHBoxLayout()
        self._al_btn = QPushButton("Start / advance campaign")
        self._al_btn.setMinimumHeight(34)
        self._al_btn.clicked.connect(self._start_al_query)
        go_row.addWidget(self._al_btn)
        self._al_export_btn = QPushButton("Export candidates (CSV)…")
        self._al_export_btn.setEnabled(False)
        self._al_export_btn.clicked.connect(self._al_export_candidates)
        go_row.addWidget(self._al_export_btn)
        self._al_import_btn = QPushButton("Import experimental labels (CSV)…")
        self._al_import_btn.setEnabled(False)
        self._al_import_btn.clicked.connect(self._al_import_labels)
        go_row.addWidget(self._al_import_btn)
        go_row.addStretch(1)
        form.addRow(go_row)
        lay.addLayout(form)

        self._al_status = self._status_label(
            "Load a classification dataset on the Model Analysis page, then "
            "start a campaign."
        )
        lay.addWidget(self._al_status)
        self._al_table = QTableWidget()
        self._al_table.setMinimumHeight(220)
        lay.addWidget(self._al_table)
        return tab

    def _start_al_query(self) -> None:
        """One campaign round: build/advance the loop, rank, render."""
        frame, target, task = self._training_context()
        if frame is None:
            self._al_status.setText(
                "Load a dataset on the Model Analysis page first (e.g. "
                "qsar_classification.csv)."
            )
            return
        if not target:
            self._al_status.setText(
                "Pick a classification target on the Model Analysis page."
            )
            return
        if task != "classification":
            self._al_status.setText(
                "Active learning ranks class predictions — pick a "
                "classification target (regression targets are refused)."
            )
            return

        strategy = str(self._al_strategy.currentText())
        n_query = int(self._al_batch.value())
        seed = int(self._al_seed.value())
        pool_pct = int(self._al_pool_pct.value())
        context_key = (target, seed, pool_pct)

        def job(progress: Any) -> Any:
            from pharmaml.ml.active_learning import ActiveLearningLoop

            progress(0.15, "preparing frames")
            numeric = [
                c for c in frame.select_dtypes("number").columns
                if c != target
            ]
            if not numeric:
                raise ValueError(
                    "no numeric feature columns beside the target — "
                    "featurize first (Chemistry Lab)"
                )
            smiles_col = detect_smiles_column(frame)
            y = frame[target]
            labeled_mask = y.notna()
            rng = np.random.default_rng(seed)
            labeled_idx = np.flatnonzero(labeled_mask.to_numpy())
            # seeded stratified-ish holdout: shuffle within each class
            # so class balance survives the split
            classes = np.unique(y[labeled_mask].to_numpy())
            pool_part: list[int] = []
            for cls in classes:
                cls_idx = labeled_idx[y.to_numpy()[labeled_idx] == cls]
                rng.shuffle(cls_idx)
                take = max(1, int(round(len(cls_idx) * pool_pct / 100.0)))
                pool_part.extend(cls_idx[:take].tolist())
            pool_set = set(pool_part)
            train_idx = [i for i in labeled_idx.tolist() if i not in pool_set]

            shared = {
                "feature_columns": numeric,
                "target_column": target,
                "smiles_column": smiles_col,
                "random_state": seed,
            }
            loop = ActiveLearningLoop(
                frame.iloc[train_idx].reset_index(drop=True),
                frame.iloc[sorted(pool_set)].reset_index(drop=True),
                **shared,
            )
            progress(0.5, f"querying ({strategy})")
            batch = loop.query(n_query=n_query, strategy=strategy)
            progress(0.9, "done")
            return loop, batch, context_key

        def done(payload: Any) -> None:
            loop, batch, key = payload
            # a changed context (target/seed/pool) restarts the campaign
            # visibly; the same context advances the live loop
            if getattr(self, "_al_key", None) != key:
                self._al_key = key
            self._al_loop = loop
            self._al_last_batch = batch
            columns = [str(c) for c in batch.columns]
            rows = [
                [_cell_text(v, limit=40) for v in record]
                for record in batch.to_numpy().tolist()
            ]
            _populate_table(self._al_table, columns, rows)
            rounds = len(getattr(loop, "history", []))
            self._al_status.setText(
                f"Round {rounds}: {len(batch)} candidates by "
                f"'{strategy}' — pool {len(loop.pool)} · labeled "
                f"{len(loop.labeled)}. Export for the lab, then import "
                "the measured labels to retrain."
            )
            self._al_export_btn.setEnabled(True)
            self._al_import_btn.setEnabled(True)

        self._submit(
            f"active learning ({strategy})", job, done, self._al_status
        )

    def _al_export_candidates(self) -> None:
        """Export the current batch: SMILES, IDs, scores, rationale."""
        batch = getattr(self, "_al_last_batch", None)
        if batch is None or len(batch) == 0:
            self._al_status.setText("No candidates yet — run a query first.")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export assay candidates", "al_candidates.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            batch.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001 - user-facing report
            self._al_status.setText(f"Export failed — {exc}")
            return
        self._al_status.setText(
            f"Exported {len(batch)} candidates to {Path(path).name} — "
            "add an 'experimental_label' column with the measured values, "
            "then import it back."
        )

    def _al_import_labels(self) -> None:
        """Fold measured labels into the loop and retrain (§14.1)."""
        loop = getattr(self, "_al_loop", None)
        if loop is None:
            self._al_status.setText("No active campaign — start one first.")
            return
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import experimental labels", "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        def job(progress: Any) -> Any:
            progress(0.2, "reading labels")
            labels = pd.read_csv(path)
            progress(0.5, "folding into the labeled set")
            moved = loop.import_experimental_labels(labels)
            progress(0.8, "retraining")
            loop.retrain()
            return moved

        def done(moved: int) -> None:
            self._al_status.setText(
                f"Imported {moved} experimental labels — pool "
                f"{len(loop.pool)} · labeled {len(loop.labeled)}. The "
                "next query round uses the retrained model."
            )

        def failed(message: str) -> None:
            first = message.splitlines()[0] if message else "import failed"
            self._al_status.setText(f"Import failed — {first}")

        self._runner.submit(
            "active learning: import labels", job,
            on_finished=done, on_failed=failed,
            on_cancelled=lambda s: self._al_status.setText(
                f"Cancelled after {s:.1f}s (cooperative stop)."
            ),
            on_progress=self._progress.update_progress,
        )

    # -- tab 6: statistical tools ----------------------------------------------------
    # part6 Phase 3 batch 1: the four statistical backends (conformal
    # intervals, probability calibration, decision curves, drift
    # detection) get real GUI workflows — form → validation → worker →
    # the canonical backend → results + figure + export (§3.2/§3.3: one
    # canonical implementation, never a GUI reimplementation).
    def _build_statistical_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()

        self._stat_source = QComboBox()
        for token in ("Model Analysis run", "Workbench ensemble", "Loaded bundle"):
            self._stat_source.addItem(token)
        self._stat_source.setToolTip(
            "Where the fitted model and its held-out split come from — the "
            "same three sources the applicability-domain tab uses."
        )
        form.addRow("model source:", self._stat_source)

        self._stat_alpha = QDoubleSpinBox()
        self._stat_alpha.setRange(0.01, 0.4)
        self._stat_alpha.setDecimals(2)
        self._stat_alpha.setSingleStep(0.01)
        self._stat_alpha.setValue(0.10)
        self._stat_alpha.setToolTip(
            "Conformal miscoverage alpha: the intervals target 1-alpha "
            "empirical coverage (0.10 = 90% intervals)."
        )
        form.addRow("conformal alpha:", self._stat_alpha)

        self._stat_bins = QSpinBox()
        self._stat_bins.setRange(2, 50)
        self._stat_bins.setValue(10)
        self._stat_bins.setToolTip(
            "Equal-width probability bins for the reliability diagram."
        )
        form.addRow("calibration bins:", self._stat_bins)

        self._stat_cost = QDoubleSpinBox()
        self._stat_cost.setRange(0.05, 20.0)
        self._stat_cost.setDecimals(2)
        self._stat_cost.setSingleStep(0.05)
        self._stat_cost.setValue(1.0)
        self._stat_cost.setToolTip(
            "Decision-curve cost ratio: how many false positives one "
            "missed positive costs (1.0 = the classical curve)."
        )
        form.addRow("cost ratio (FP per FN):", self._stat_cost)

        run_row = QHBoxLayout()
        self._stat_conformal_btn = QPushButton("Conformal intervals")
        self._stat_conformal_btn.setMinimumHeight(30)
        self._stat_conformal_btn.clicked.connect(self._start_stat_conformal)
        self._stat_conformal_btn.setToolTip(
            "Split-conformal intervals for a REGRESSION model: calibrate "
            "a random half of the held-out rows, then evaluate on the other "
            "half. Independence from fitting and model selection is not verified."
        )
        run_row.addWidget(self._stat_conformal_btn)
        self._stat_conformal_cls_btn = QPushButton("Conformal label sets")
        self._stat_conformal_cls_btn.setMinimumHeight(30)
        self._stat_conformal_cls_btn.clicked.connect(self._start_stat_conformal_cls)
        self._stat_conformal_cls_btn.setToolTip(
            "Prediction sets for a CLASSIFICATION model: the marginal "
            "split-conformal scheme and the label-conditional scheme "
            "calibrated on one half of the held-out rows and evaluated on "
            "the other. Independence from fitting is not verified."
        )
        run_row.addWidget(self._stat_conformal_cls_btn)
        self._stat_calibration_btn = QPushButton("Calibration report")
        self._stat_calibration_btn.setMinimumHeight(30)
        self._stat_calibration_btn.clicked.connect(self._start_stat_calibration)
        self._stat_calibration_btn.setToolTip(
            "Reliability diagnostics for a CLASSIFICATION model: Brier "
            "score, ECE/MCE and the binned reliability diagram."
        )
        run_row.addWidget(self._stat_calibration_btn)
        self._stat_decision_btn = QPushButton("Decision curve")
        self._stat_decision_btn.setMinimumHeight(30)
        self._stat_decision_btn.clicked.connect(self._start_stat_decision)
        self._stat_decision_btn.setToolTip(
            "Net-benefit decision curve for a CLASSIFICATION model: the "
            "model's benefit against treat-all and treat-none over the "
            "threshold range."
        )
        run_row.addWidget(self._stat_decision_btn)
        form.addRow("model tools:", run_row)

        self._stat_drift_reference = QLineEdit()
        self._stat_drift_reference.setPlaceholderText(
            "reference (training) CSV — blank = the current dataset"
        )
        form.addRow("drift reference:", self._stat_drift_reference)
        self._stat_drift_current = QLineEdit()
        self._stat_drift_current.setPlaceholderText("current (new batch) CSV — required")
        form.addRow("drift current:", self._stat_drift_current)

        self._stat_drift_btn = QPushButton("Detect drift")
        self._stat_drift_btn.setMinimumHeight(30)
        self._stat_drift_btn.clicked.connect(self._start_stat_drift)
        self._stat_drift_btn.setToolTip(
            "Per-feature distribution-shift audit of the current batch "
            "against the reference: PSI (and two-sample KS when "
            "available), with stable / moderate / shifted verdicts."
        )
        form.addRow("", self._stat_drift_btn)

        # part6 Phase 3, batch 2: the ML statistics cluster's five rows
        self._build_statistics_batch2_form(form)

        lay.addLayout(form)
        self._stat_status = self._status_label(
            "the statistical tools reuse a trained run (conformal needs "
            "regression, calibration and decision curves need "
            "classification), or compare two CSVs for drift."
        )
        lay.addWidget(self._stat_status)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._stat_table = self._make_board(minimum_height=170)
        self._stat_table.setToolTip(
            "The tool's result table: conformal intervals per held-out "
            "row, reliability bins, decision-curve points, or the "
            "per-feature drift verdicts."
        )
        splitter.addWidget(self._stat_table)
        self._stat_plot = PlotPanel(title="Statistical tools")
        self._stat_plot.set_empty_hint("Pick a statistical tool and run it on your table")

        splitter.addWidget(self._stat_plot)
        splitter.setSizes([520, 420])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._stat_export_btn = QPushButton("Export last result (CSV)")
        self._stat_export_btn.setMinimumHeight(30)
        self._stat_export_btn.clicked.connect(self._export_stat_result)
        self._stat_export_btn.setToolTip(
            "Write the last tool's result table to a CSV of your choice."
        )
        lay.addWidget(self._stat_export_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self._stat_frame: pd.DataFrame | None = None
        # batch 2: the coverage diagnostics audit the last conformal
        # run's covered flag, so the tab keeps the result object.
        self._stat_conformal_last: Any = None
        self._stat_conformal_evaluation: Any = None
        self._refresh_stat_group_columns()
        return tab

    def _stat_model_and_split(
        self,
    ) -> tuple[Any, Any, Any, Any, str, str] | None:
        """Resolve the selected source into (model, X, y, task, label).

        Uses the SAME three sources as the applicability-domain tab so
        the two tools never disagree about what was trained.  Returns
        None after explaining the problem on the status label.
        """
        source = self._stat_source.currentText()
        frame, target, _task = self._training_context()

        if source == "Model Analysis run":
            run = getattr(self._model_analysis, "last_run", None)
            outcome = run() if callable(run) else None
            if outcome is None or getattr(outcome, "estimator", None) is None:
                self._stat_status.setText(
                    "Train a model on the Model Analysis page first — the "
                    "statistical tools reuse its fitted estimator and "
                    "held-out split (or switch the source)."
                )
                return None
            model = outcome.estimator
            x_test, y_test = outcome.X_test, outcome.y_test
            task = str(outcome.task)
            label = f"{outcome.model} ({task})"
        elif source == "Workbench ensemble":
            if self._ens_last is None:
                self._stat_status.setText(
                    "Train an ensemble on the Ensemble builder tab first — "
                    "the tools reuse its split (or switch the source)."
                )
                return None
            estimator, _report, task, _members, name, _xtr, _ytr, xte, yte, _tgt = self._ens_last
            model, x_test, y_test, label = estimator, xte, yte, name
        else:  # loaded bundle
            if self._bundle is None:
                self._stat_status.setText(
                    "Open a model bundle on the Persistence tab first — the "
                    "bundle mode scores the loaded model over the current "
                    "dataset's held-out share."
                )
                return None
            if frame is None or not target:
                self._stat_status.setText(
                    "Load a dataset on the Model Analysis page first — the "
                    "bundle mode splits the CURRENT dataset for the tools."
                )
                return None
            model, metadata, _bundle_path = self._bundle
            task = str(metadata.get("task", "regression"))
            label = str(metadata.get("name", "bundle"))
            numeric, dropped = numeric_feature_columns(frame, target)
            if not numeric:
                self._stat_status.setText(
                    "Every non-target column is non-numeric"
                    + (f" ({', '.join(dropped)})" if dropped else "")
                    + " — the bundle mode needs numeric descriptor columns."
                )
                return None
            from sklearn.model_selection import train_test_split

            stratify = None if task == "regression" else frame[target].astype(str).to_numpy()
            _x_train, x_test, _y_train, y_test = train_test_split(
                frame[numeric], frame[target], test_size=0.25, random_state=42,
                stratify=stratify,
            )
        return model, x_test, y_test, task, label

    def _start_stat_conformal(self) -> None:
        """Split-conformal intervals on the held-out split (regression)."""
        self._stat_frame = None
        self._stat_table.setRowCount(0)
        self._stat_plot.clear()
        self._stat_conformal_last = None
        self._stat_conformal_evaluation = None
        resolved = self._stat_model_and_split()
        if resolved is None:
            return
        model, x_test, y_test, task, label = resolved
        if task != "regression":
            self._stat_status.setText(
                "Conformal intervals are a regression tool — this source is "
                f"{task}. Train a regression model (or switch the source)."
            )
            return
        alpha = float(self._stat_alpha.value())
        from pharmaml.exceptions import DataValidationError
        from pharmaml.ml.heldout_partition import split_calibration_evaluation

        try:
            x_cal, x_eval, y_cal, y_eval = split_calibration_evaluation(x_test, y_test)
        except DataValidationError as exc:
            self._stat_status.setText(f"Cannot split held-out rows: {exc}")
            return

        def job(progress: Any) -> Any:
            from pharmaml.ml.conformal import calibrate_split_conformal

            progress(0.3, "calibrating the interval radius on held-out rows")
            calibrated = calibrate_split_conformal(model, x_cal, y_cal, alpha=alpha)
            progress(0.8, "scoring the held-out rows")
            return calibrated.predict_interval(x_eval, y_true=y_eval)

        def done(result: Any) -> None:
            table = result.to_dataframe()
            self._stat_frame = table
            self._stat_conformal_last = result  # the coverage workflow audits this
            self._stat_conformal_evaluation = (x_eval, np.asarray(y_eval).copy())
            columns = [str(column) for column in table.columns]
            shown = table.head(200)
            rows = [
                [_cell_text(value, limit=26) for value in record]
                for record in shown.to_numpy().tolist()
            ]
            _populate_table(self._stat_table, columns, rows)
            coverage = (
                f" · empirical coverage {result.coverage:.1%}"
                if result.coverage is not None
                else ""
            )
            self._stat_status.setText(
                f"radius {result.radius:.4g} for {1 - alpha:.0%} intervals "
                f"({result.n_test} evaluation rows; {len(x_cal)} calibration rows){coverage} · "
                f"source {label} · independence from fitting/selection not verified"
            )
            self._stat_plot.clear()
            positions = np.arange(1, result.n_test + 1, dtype=float)
            self._stat_plot.scatter(positions, np.asarray(result.prediction), label="prediction")
            lower = np.asarray(result.lower)
            upper = np.asarray(result.upper)
            self._stat_plot.line(positions, lower, label="lower", width=1, index=1, style="dash")
            self._stat_plot.line(positions, upper, label="upper", width=1, index=1, style="dash")
            self._stat_plot.set_labels(
                x="held-out row", y="predicted", title=f"Split-conformal {1 - alpha:.0%} intervals"
            )

        self._submit(f"conformal intervals ({label})", job, done, self._stat_status)

    def _start_stat_conformal_cls(self) -> None:
        """Conformal label sets on the held-out split (classification).

        part6 Phase 3, batch 4 — the classification mirror of the
        regression conformal card above: both calibration schemes from
        ``ml.conformal_classification`` run side by side over the same
        calibration rows and evaluate on disjoint held-out rows. This does
        not verify independence from fitting, selection or prior evaluation.
        """
        self._stat_frame = None
        self._stat_table.setRowCount(0)
        self._stat_plot.clear()
        resolved = self._stat_model_and_split()
        if resolved is None:
            return
        model, x_test, y_test, task, label = resolved
        if task == "regression":
            self._stat_status.setText(
                "Conformal label sets are a classification tool — this source "
                "is regression. Train a classifier (or switch the source)."
            )
            return
        alpha = float(self._stat_alpha.value())
        from pharmaml.exceptions import DataValidationError
        from pharmaml.ml.heldout_partition import split_calibration_evaluation

        try:
            x_cal, x_eval, y_cal, y_eval = split_calibration_evaluation(x_test, y_test)
        except DataValidationError as exc:
            self._stat_status.setText(f"Cannot split held-out rows: {exc}")
            return

        def job(progress: Any) -> Any:
            from pharmaml.ml.conformal_classification import (
                calibrate_label_conditional_classifier,
                calibrate_split_conformal_classifier,
            )

            progress(0.3, "calibrating the marginal threshold on held-out rows")
            split = calibrate_split_conformal_classifier(model, x_cal, y_cal, alpha=alpha)
            progress(0.6, "calibrating one threshold per class")
            conditional = calibrate_label_conditional_classifier(
                model, x_cal, y_cal, alpha=alpha
            )
            progress(0.85, "scoring the held-out rows under both schemes")
            return (
                split.predict_set(x_eval, y_true=y_eval),
                conditional.predict_set(x_eval, y_true=y_eval),
            )

        def done(result: Any) -> None:
            split, conditional = result
            classes = [str(c) for c in split.classes]
            membership = np.asarray(split.membership)
            cond_membership = np.asarray(conditional.membership)
            sizes = membership.sum(axis=1)
            cond_sizes = cond_membership.sum(axis=1)
            covered = np.asarray(split.covered)
            cond_covered = np.asarray(conditional.covered)
            # the verbatim per-row table: both schemes' sets as label lists
            table = pd.DataFrame(
                {
                    "true": [_cell_text(v, limit=18) for v in np.asarray(y_eval)],
                    "split_set": [
                        "{" + ", ".join(c for c, m in zip(classes, row) if m) + "}"
                        if any(row) else "∅" for row in membership
                    ],
                    "split_size": sizes,
                    "cond_set": [
                        "{" + ", ".join(c for c, m in zip(classes, row) if m) + "}"
                        if any(row) else "∅" for row in cond_membership
                    ],
                    "cond_size": cond_sizes,
                }
            )
            self._stat_frame = table
            shown = table.head(200)
            _populate_table(
                self._stat_table,
                [str(c) for c in shown.columns],
                [
                    [_cell_text(v, limit=30) for v in record]
                    for record in shown.to_numpy().tolist()
                ],
            )
            self._stat_status.setText(
                f"split: threshold {split.threshold:.3f}, coverage "
                f"{covered.mean():.1%}, mean set {sizes.mean():.2f} · "
                f"label-conditional: coverage {cond_covered.mean():.1%}, "
                f"mean set {cond_sizes.mean():.2f} · {len(table)} evaluation "
                f"rows; {len(x_cal)} calibration rows · source {label} · "
                "independence from fitting/selection not verified"
            )
            self._stat_plot.clear()
            positions = np.arange(1, len(sizes) + 1, dtype=float)
            order = np.argsort(sizes)
            self._stat_plot.scatter(
                positions, sizes[order].astype(float), label="split set size"
            )
            self._stat_plot.line(
                positions,
                cond_sizes[order].astype(float),
                label="label-conditional",
                width=1,
                index=1,
                style="dash",
            )
            self._stat_plot.set_labels(
                x="held-out row (sorted by split set size)",
                y="prediction-set size",
                title=f"Conformal label sets at alpha={alpha:g}",
            )

        self._submit(
            f"conformal label sets ({label})", job, done, self._stat_status
        )

    def _start_stat_calibration(self) -> None:
        """Reliability diagnostics (classification)."""
        resolved = self._stat_model_and_split()
        if resolved is None:
            return
        model, x_test, y_test, task, label = resolved
        if task != "classification":
            self._stat_status.setText(
                "Calibration reports are a classification tool — this source "
                f"is {task}. Train a classification model (or switch the source)."
            )
            return
        n_bins = int(self._stat_bins.value())

        def job(progress: Any) -> Any:
            from pharmaml.ml.calibration import calibration_report

            progress(0.5, "scoring the held-out rows")
            return calibration_report(model, x_test, y_test, n_bins=n_bins)

        def done(result: Any) -> None:
            rows = [
                [
                    f"{b.lower:.2f}–{b.upper:.2f}",
                    str(b.count),
                    f"{b.mean_predicted:.3f}",
                    f"{b.fraction_positive:.3f}",
                ]
                for b in result.bins
                if b.count
            ]
            _populate_table(
                self._stat_table,
                ["bin", "count", "mean predicted", "fraction positive"],
                rows,
            )
            self._stat_frame = pd.DataFrame(
                [
                    {
                        "bin_lower": b.lower,
                        "bin_upper": b.upper,
                        "count": b.count,
                        "mean_predicted": b.mean_predicted,
                        "fraction_positive": b.fraction_positive,
                    }
                    for b in result.bins
                ]
            )
            self._stat_status.setText(
                f"Brier {result.brier:.4f} · ECE {result.ece:.4f} · MCE "
                f"{result.mce:.4f} · base rate {result.base_rate:.3f} · "
                f"{result.n_samples} held-out rows · source {label}"
            )
            self._stat_plot.clear()
            self._stat_plot.scatter(
                np.asarray([b.mean_predicted for b in result.bins if b.count]),
                np.asarray([b.fraction_positive for b in result.bins if b.count]),
                label="observed",
            )
            span = np.array([0.0, 1.0])
            self._stat_plot.line(span, span, label="perfect", width=1, index=1, style="dash")
            self._stat_plot.set_labels(
                x="predicted probability", y="observed fraction",
                title=f"Reliability diagram ({n_bins} bins)",
            )

        self._submit(f"calibration report ({label})", job, done, self._stat_status)

    def _start_stat_decision(self) -> None:
        """Net-benefit decision curve (classification)."""
        resolved = self._stat_model_and_split()
        if resolved is None:
            return
        model, x_test, y_test, task, label = resolved
        if task != "classification":
            self._stat_status.setText(
                "Decision curves are a classification tool — this source is "
                f"{task}. Train a classification model (or switch the source)."
            )
            return
        cost_ratio = float(self._stat_cost.value())

        def job(progress: Any) -> Any:
            from pharmaml.ml.decision import decision_curve

            progress(0.5, "sweeping the threshold range")
            return decision_curve(model, x_test, y_test, cost_ratio=cost_ratio)

        def done(result: Any) -> None:
            points = result.points
            rows = [
                [
                    f"{p.threshold:.3f}",
                    f"{p.net_benefit:.4f}",
                    f"{p.standardized_net_benefit:.4f}",
                    f"{p.treat_all_net_benefit:.4f}",
                ]
                for p in points[:: max(1, len(points) // 100)]
            ]
            _populate_table(
                self._stat_table,
                ["threshold", "net benefit", "standardized", "treat-all"],
                rows,
            )
            self._stat_frame = pd.DataFrame(
                [
                    {
                        "threshold": p.threshold,
                        "net_benefit": p.net_benefit,
                        "standardized_net_benefit": p.standardized_net_benefit,
                        "treat_all_net_benefit": p.treat_all_net_benefit,
                    }
                    for p in points
                ]
            )
            best = max(points, key=lambda p: p.net_benefit)
            self._stat_status.setText(
                f"best net benefit {best.net_benefit:.4f} at threshold "
                f"{best.threshold:.3f} · base rate {result.base_rate:.3f} · "
                f"{result.n_samples} held-out rows · cost ratio {cost_ratio:g} · "
                f"source {label}"
            )
            self._stat_plot.clear()
            thresholds = np.asarray([p.threshold for p in points])
            self._stat_plot.line(
                thresholds, np.asarray([p.net_benefit for p in points]),
                label="model", width=2,
            )
            self._stat_plot.line(
                thresholds, np.asarray([p.treat_all_net_benefit for p in points]),
                label="treat all", width=1, index=1, style="dash",
            )
            self._stat_plot.line(
                thresholds, np.zeros_like(thresholds),
                label="treat none", width=1, index=1, style="dot",
            )
            self._stat_plot.set_labels(
                x="threshold probability", y="net benefit",
                title="Decision curve — model vs treat-all/treat-none",
            )

        self._submit(f"decision curve ({label})", job, done, self._stat_status)

    def _start_stat_drift(self) -> None:
        """Per-feature drift audit of a current batch vs a reference."""
        from pharmaml.utils.io import read_table

        current_path = self._stat_drift_current.text().strip()
        if not current_path:
            self._stat_status.setText(
                "Name the current (new batch) CSV first — drift compares it "
                "against the reference."
            )
            return
        reference_path = self._stat_drift_reference.text().strip()
        # review C-3.5: _training_context() reads the Model Analysis page's
        # QComboBox — GUI state. Snapshot it HERE on the GUI thread (the
        # file's own contract: "the worker never touches the frame"); the
        # worker used to call it from the job thread, which is undefined
        # behaviour in Qt.
        reference_frame = (
            None if reference_path else self._training_context()[0]
        )

        def job(progress: Any) -> Any:
            from pharmaml.ml.drift import detect_drift

            progress(0.2, "reading the two tables")
            reference = read_table(reference_path) if reference_path else reference_frame
            current = read_table(current_path)
            if reference is None:
                raise ValueError(
                    "no reference available — name a reference CSV or load "
                    "the current dataset on the Model Analysis page"
                )
            progress(0.6, "comparing feature distributions")
            return detect_drift(reference, current)

        def done(result: Any) -> None:
            rows = [
                [
                    f.name,
                    f"{f.psi:.4f}",
                    "—" if f.ks_statistic is None else f"{f.ks_statistic:.3f}",
                    "—" if f.ks_pvalue is None else f"{f.ks_pvalue:.3g}",
                    f"{f.reference_mean:.4g}",
                    f"{f.current_mean:.4g}",
                    f.verdict,
                ]
                for f in result.features
            ]
            _populate_table(
                self._stat_table,
                ["feature", "psi", "ks stat", "ks p", "ref mean", "cur mean", "verdict"],
                rows,
            )
            self._stat_frame = pd.DataFrame(
                [
                    {
                        "feature": f.name,
                        "psi": f.psi,
                        "ks_statistic": f.ks_statistic,
                        "ks_pvalue": f.ks_pvalue,
                        "reference_mean": f.reference_mean,
                        "current_mean": f.current_mean,
                        "verdict": f.verdict,
                    }
                    for f in result.features
                ]
            )
            shifted = sum(1 for f in result.features if f.verdict == "shifted")
            moderate = sum(1 for f in result.features if f.verdict == "moderate")
            self._stat_status.setText(
                f"{len(result.features)} features compared · {shifted} shifted · "
                f"{moderate} moderate · source {result.source} · "
                f"thresholds psi {result.warn_psi:g}/{result.shift_psi:g}"
            )
            self._stat_plot.clear()
            if result.features:
                names = [f.name for f in result.features]
                psis = np.asarray([f.psi for f in result.features])
                positions = np.arange(len(names), dtype=float)
                self._stat_plot.scatter(positions, psis, label="psi")
                for psi_level, name in ((result.warn_psi, "warn"), (result.shift_psi, "shift")):
                    self._stat_plot.line(
                        positions,
                        np.full_like(positions, psi_level),
                        label=name, width=1, index=1, style="dash",
                    )
                self._stat_plot.set_labels(
                    x="feature (see table)", y="psi",
                    title="Population stability index per feature",
                )

        self._submit("drift detection", job, done, self._stat_status)

    def _export_stat_result(self) -> None:
        """Write the last statistical tool's result table to CSV."""
        if self._stat_frame is None:
            self._stat_status.setText("Run a statistical tool first — there is no result table yet.")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export statistical result", "statistical-result.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._stat_frame.to_csv(path, index=False)
        except OSError as error:
            self._stat_status.setText(f"Export failed — {error}")
            return
        self._stat_status.setText(f"Result written to {path}")

    # -- part6 Phase 3, batch 2: the ML statistics cluster ---------------
    # Five more §3.2 workflows over canonical backends: the model
    # comparison test, multiplicity correction, the power picture,
    # coverage-by-group diagnostics and the subgroup audit.  The model
    # card (the cluster's sixth member) lives on the Persistence tab
    # where the bundles are.

    def _build_statistics_batch2_form(self, form: QFormLayout) -> None:
        """The batch-2 form rows (added under the batch-1 rows)."""
        self._stat_models = QLineEdit()
        self._stat_models.setText("ridge, random_forest, gradient_boosting")
        self._stat_models.setToolTip(
            "Comma-separated model keys for the comparison test — the "
            "panel runs under one shared splitter so the boards are "
            "comparable, then every model is tested against the leader."
        )
        form.addRow("comparison panel:", self._stat_models)
        self._stat_folds = QSpinBox()
        self._stat_folds.setRange(2, 10)
        self._stat_folds.setValue(5)
        self._stat_folds.setToolTip("Cross-validation folds shared by the whole panel.")
        form.addRow("folds:", self._stat_folds)
        self._stat_correction = QComboBox()
        for token in ("fdr_bh", "bonferroni", "holm"):
            self._stat_correction.addItem(token)
        self._stat_correction.setToolTip(
            "Multiplicity correction for the comparison family (also the "
            "method behind the standalone p-value corrector)."
        )
        form.addRow("correction:", self._stat_correction)

        self._stat_paste = QLineEdit()
        self._stat_paste.setPlaceholderText(
            "paste raw p-values, comma- or space-separated"
        )
        self._stat_paste.setToolTip(
            "The standalone multiplicity corrector: any family of raw "
            "p-values, adjusted with the correction above."
        )
        form.addRow("p-value family:", self._stat_paste)

        run_row = QHBoxLayout()
        self._stat_compare_btn = QPushButton("Comparison test")
        self._stat_compare_btn.setMinimumHeight(30)
        self._stat_compare_btn.clicked.connect(self._start_stat_comparison)
        self._stat_compare_btn.setToolTip(
            "Cross-validate the panel, then test every model against the "
            "leader with the family corrected: differences, adjusted "
            "p-values and the on-par/better/worse verdicts."
        )
        run_row.addWidget(self._stat_compare_btn)
        self._stat_correct_btn = QPushButton("Correct p-values")
        self._stat_correct_btn.setMinimumHeight(30)
        self._stat_correct_btn.clicked.connect(self._start_stat_multiplicity)
        self._stat_correct_btn.setToolTip(
            "Benjamini-Hochberg / Bonferroni / Holm adjustment of the "
            "pasted family — the q-values, same length and order."
        )
        run_row.addWidget(self._stat_correct_btn)
        self._stat_power_btn = QPushButton("Power report")
        self._stat_power_btn.setMinimumHeight(30)
        self._stat_power_btn.clicked.connect(self._start_stat_power)
        self._stat_power_btn.setToolTip(
            "The fold-count power picture: the minimum detectable effect "
            "at the target power, the observed effect's power when given, "
            "and the power curve across fold counts."
        )
        run_row.addWidget(self._stat_power_btn)
        form.addRow("panel tools:", run_row)

        self._stat_group_column = QComboBox()
        self._stat_group_column.setEditable(True)
        self._stat_group_column.setToolTip(
            "The subgroup/coverage grouping column — any dataset column "
            "(categorical or binned numeric). Refreshed when a dataset "
            "loads; free text also works."
        )
        form.addRow("group column:", self._stat_group_column)
        self._stat_mingroup = QSpinBox()
        self._stat_mingroup.setRange(2, 200)
        self._stat_mingroup.setValue(10)
        self._stat_mingroup.setToolTip(
            "Minimum rows per subgroup; smaller groups stay listed but "
            "are flagged insufficient (and excluded from the gap tests)."
        )
        form.addRow("min group size:", self._stat_mingroup)
        self._stat_effect = QDoubleSpinBox()
        self._stat_effect.setRange(0.0, 5.0)
        self._stat_effect.setDecimals(2)
        self._stat_effect.setSingleStep(0.1)
        self._stat_effect.setValue(0.0)
        self._stat_effect.setToolTip(
            "Optional observed standardized effect for the power report "
            "(0 = report only the minimum detectable effect and curve)."
        )
        form.addRow("observed effect:", self._stat_effect)

        run_row2 = QHBoxLayout()
        self._stat_coverage_btn = QPushButton("Coverage by group")
        self._stat_coverage_btn.setMinimumHeight(30)
        self._stat_coverage_btn.clicked.connect(self._start_stat_coverage)
        self._stat_coverage_btn.setToolTip(
            "Coverage diagnostics for the last conformal run: per-group "
            "empirical coverage with Wilson intervals, against the "
            "nominal level."
        )
        run_row2.addWidget(self._stat_coverage_btn)
        self._stat_subgroup_btn = QPushButton("Subgroup audit")
        self._stat_subgroup_btn.setMinimumHeight(30)
        self._stat_subgroup_btn.clicked.connect(self._start_stat_subgroups)
        self._stat_subgroup_btn.setToolTip(
            "Stratified performance per subgroup plus the multiplicity-"
            "corrected gap scan (every subgroup vs the rest)."
        )
        run_row2.addWidget(self._stat_subgroup_btn)
        form.addRow("group tools:", run_row2)

    def _on_analysis_frame(self, frame: object) -> None:
        """frameLoaded slot (review m-19): re-list the grouping choices."""
        self._refresh_stat_group_columns()

    def _refresh_stat_group_columns(self) -> None:
        """Re-list the grouping choices from the loaded dataset."""
        frame, _target, _task = self._training_context()
        if frame is None:
            return
        current = self._stat_group_column.currentText()
        self._stat_group_column.clear()
        self._stat_group_column.addItems([str(column) for column in frame.columns])
        if current:
            index = self._stat_group_column.findText(current)
            if index >= 0:
                self._stat_group_column.setCurrentIndex(index)
            else:
                self._stat_group_column.setCurrentText(current)

    def _start_stat_comparison(self) -> None:
        """The panel comparison test: CV + family-corrected tests."""
        frame, target, task = self._training_context()
        if frame is None or not target:
            self._stat_status.setText(
                "Load a dataset on the Model Analysis page first — the "
                "comparison test cross-validates the current dataset's panel."
            )
            return
        models = [
            token.strip()
            for token in self._stat_models.text().split(",")
            if token.strip()
        ]
        if len(models) < 2:
            self._stat_status.setText(
                "The comparison test needs at least two models in the panel "
                "— there is nothing to compare against the leader otherwise."
            )
            return
        n_folds = int(self._stat_folds.value())
        correction = self._stat_correction.currentText()
        smiles = detect_smiles_column(frame)
        numeric, dropped = numeric_feature_columns(frame, target)
        if not numeric:
            self._stat_status.setText(
                "Every non-target column is non-numeric"
                + (f" ({', '.join(dropped)})" if dropped else "")
                + " — the comparison test needs descriptor columns."
            )
            return
        label = f"{len(models)} models, {target}"

        def job(progress: Any) -> Any:
            from pharmaml.ml.comparison import model_comparison_test
            from pharmaml.ml.cv_results import compare_models_cv

            def on_model(info: dict[str, Any]) -> None:
                progress(
                    (info.get("index", 0) + 1) / max(1, int(info.get("total", 1))),
                    f"cross-validating {info.get('model', 'model')}",
                )

            results, _board = compare_models_cv(
                frame[numeric],
                frame[target],
                task,
                models=models,
                n_folds=n_folds,
                smiles=frame[smiles] if smiles else None,
                on_model=on_model,
            )
            progress(0.9, "testing the family against the leader")
            return model_comparison_test(results, alpha=0.05, correction=correction)

        def done(result: Any) -> None:
            rows = []
            for row in result.rows:
                rows.append(
                    [
                        row.model,
                        f"{row.mean:.4g}",
                        f"{row.difference:+.4g}",
                        "—" if row.adjusted_p is None else f"{row.adjusted_p:.4g}",
                        row.note,
                    ]
                )
            _populate_table(
                self._stat_table,
                ["model", f"mean {result.metric}", "vs leader", "adjusted p", "verdict"],
                rows,
            )
            self._stat_frame = pd.DataFrame(
                [
                    {
                        "model": r.model,
                        "mean": r.mean,
                        "leader_mean": r.leader_mean,
                        "difference": r.difference,
                        "adjusted_p": r.adjusted_p,
                        "note": r.note,
                    }
                    for r in result.rows
                ]
            )
            better = sum(1 for r in result.rows if "better" in r.note)
            self._stat_status.setText(
                f"leader {result.leader} ({result.leader_mean:.4g} {result.metric}) "
                f"· {result.n_folds} folds · {result.test} test · {result.correction} "
                f"correction · {better} model(s) beat it — the rest are on par "
                "within the family correction"
            )
            self._stat_plot.clear()
            names = [r.model for r in result.rows]
            means = np.asarray([r.mean for r in result.rows], dtype=float)
            self._stat_plot.bars(means, labels=names)
            self._stat_plot.set_labels(
                x="model", y=f"mean {result.metric}",
                title=f"Panel means — leader {result.leader}",
            )

        self._submit(f"comparison test ({label})", job, done, self._stat_status)

    def _start_stat_multiplicity(self) -> None:
        """Standalone p-value family correction."""
        raw = self._stat_paste.text().replace(",", " ").split()
        if not raw:
            self._stat_status.setText(
                "Paste the raw p-values first — comma- or space-separated."
            )
            return
        try:
            p_values = [float(token) for token in raw]
        except ValueError:
            self._stat_status.setText(
                "Those are not numbers — paste p-values like 0.03, 0.12, 0.001."
            )
            return
        method = self._stat_correction.currentText()

        def job(progress: Any) -> Any:
            from pharmaml.ml.multiplicity import correct_p_values

            progress(0.5, f"adjusting {len(p_values)} p-values ({method})")
            return correct_p_values(p_values, method=method)

        def done(adjusted: Any) -> None:
            rows = [
                [str(index + 1), f"{raw_value:.4g}", f"{adj_value:.4g}"]
                for index, (raw_value, adj_value) in enumerate(zip(p_values, adjusted))
            ]
            _populate_table(self._stat_table, ["#", "raw p", f"adjusted ({method})"], rows)
            self._stat_frame = pd.DataFrame(
                {
                    "raw_p": p_values,
                    "adjusted_p": [float(value) for value in adjusted],
                }
            )
            significant = sum(1 for value in adjusted if value < 0.05)
            self._stat_status.setText(
                f"{len(p_values)} p-values adjusted ({method}) — "
                f"{significant} remain significant at 0.05"
            )
            self._stat_plot.clear()
            positions = np.arange(1, len(p_values) + 1, dtype=float)
            self._stat_plot.scatter(positions, np.asarray(p_values), label="raw")
            self._stat_plot.scatter(
                positions, np.asarray(adjusted, dtype=float), label="adjusted"
            )
            span = np.array([1.0, float(max(len(p_values), 1))])
            self._stat_plot.line(
                span, np.array([0.05, 0.05]), label="0.05", width=1, index=1, style="dash",
            )
            self._stat_plot.set_labels(
                x="hypothesis #", y="p-value", title=f"{method} adjustment",
            )

        self._submit(f"p-value correction ({method})", job, done, self._stat_status)

    def _start_stat_power(self) -> None:
        """The fold-count power picture."""
        n_folds = int(self._stat_folds.value())
        effect = float(self._stat_effect.value())
        observed = effect if effect > 0.0 else None

        def job(progress: Any) -> Any:
            from pharmaml.ml.power import power_report

            progress(0.5, "solving the minimum detectable effect")
            return power_report(n_folds, observed_effect=observed)

        def done(result: Any) -> Any:
            mde = result.mde
            needed = None
            if observed is not None:
                from pharmaml.ml.power import folds_for_power

                needed = folds_for_power(observed)
            rows = [
                ["folds", str(result.n_folds)],
                ["alpha", f"{result.alpha:g}"],
                ["target power", f"{result.target_power:g}"],
                ["minimum detectable |d|", f"{mde:.4g}"],
            ]
            if result.observed_effect is not None:
                rows.append(["observed |d|", f"{result.observed_effect:.4g}"])
                rows.append(
                    ["observed power", f"{result.observed_power:.1%}"]
                    if result.observed_power is not None
                    else ["observed power", "—"]
                )
                rows.append(
                    [
                        "verdict",
                        "sufficient folds" if result.sufficient else
                        f"under-powered (needs ~{needed} folds)",
                    ]
                )
            _populate_table(self._stat_table, ["quantity", "value"], rows)
            self._stat_frame = pd.DataFrame(rows, columns=["quantity", "value"])
            headline = (
                f"MDE {mde:.3g} at {result.target_power:.0%} power over "
                f"{result.n_folds} folds"
            )
            if result.observed_power is not None:
                headline += (
                    f" · observed effect {result.observed_effect:.3g} runs at "
                    f"{result.observed_power:.1%} power"
                    + ("" if result.sufficient else f" — need ~{needed} folds")
                )
            self._stat_status.setText(headline)
            # the power curve across fold counts, from the same solver
            self._stat_plot.clear()
            from pharmaml.ml.power import paired_power

            fold_range = np.arange(2, 31, dtype=float)
            curve = np.asarray(
                [paired_power(observed if observed is not None else mde, int(n)) for n in fold_range]
            )
            self._stat_plot.line(fold_range, curve, label="power", width=2)
            self._stat_plot.line(
                fold_range, np.full_like(fold_range, result.target_power),
                label="target", width=1, index=1, style="dash",
            )
            effect_label = observed if observed is not None else mde
            self._stat_plot.set_labels(
                x="folds", y="power",
                title=f"Paired power at |d| = {effect_label:.3g}",
            )

        self._submit(f"power report ({n_folds} folds)", job, done, self._stat_status)

    def _start_stat_coverage(self) -> None:
        """Coverage diagnostics, grouped, for the last conformal run."""
        self._stat_frame = None
        self._stat_table.setRowCount(0)
        self._stat_plot.clear()
        intervals = self._stat_conformal_last
        if intervals is None:
            self._stat_status.setText(
                "Run Conformal intervals first — the coverage diagnostics "
                "audit ITS covered flag per group."
            )
            return
        covered = np.asarray(intervals.covered, dtype=bool)
        if covered.size == 0:
            self._stat_status.setText(
                "The last conformal run had no labels — its coverage is "
                "undefined. Rerun it with a labelled held-out split."
            )
            return
        frame, target, _task = self._training_context()
        column = self._stat_group_column.currentText().strip()
        if frame is None or not column or column not in frame.columns:
            self._stat_status.setText(
                "Name a grouping column from the loaded dataset first — "
                "coverage is audited per group."
            )
            return
        from pharmaml.exceptions import DataValidationError
        from pharmaml.ml.heldout_partition import aligned_group_values

        evaluation = self._stat_conformal_evaluation
        if evaluation is None:
            self._stat_status.setText("Evaluation row identity is unavailable — rerun Conformal intervals.")
            return
        x_eval, y_eval = evaluation
        try:
            groups = aligned_group_values(frame, x_eval, y_eval, target, column)
        except DataValidationError as exc:
            self._stat_status.setText(str(exc))
            return
        confidence = 1.0 - float(intervals.alpha)
        sizes = np.asarray(intervals.upper) - np.asarray(intervals.lower)

        def job(progress: Any) -> Any:
            from pharmaml.ml.coverage_diagnostics import coverage_by_group

            progress(0.5, "summarizing coverage per group")
            return coverage_by_group(covered, groups, sizes=sizes, confidence=confidence)

        def done(result: Any) -> None:
            rows = []
            for g in (result.overall, *result.groups):
                rows.append(
                    [
                        str(g.group),
                        str(g.n_test),
                        str(g.n_covered),
                        f"{g.coverage:.1%}",
                        f"[{g.lower:.1%}, {g.upper:.1%}]",
                    ]
                )
            _populate_table(
                self._stat_table,
                ["group", "n", "covered", "coverage", f"{result.confidence:.0%} Wilson CI"],
                rows,
            )
            self._stat_frame = pd.DataFrame(
                [
                    {
                        "group": str(g.group),
                        "n_test": g.n_test,
                        "n_covered": g.n_covered,
                        "coverage": g.coverage,
                        "lower": g.lower,
                        "upper": g.upper,
                        "mean_width": g.mean_size,
                    }
                    for g in (result.overall, *result.groups)
                ]
            )
            self._stat_status.setText(
                f"overall coverage {result.overall.coverage:.1%} "
                f"(nominal {confidence:.0%}) over {result.overall.n_test} rows · "
                f"{len(result.groups)} groups · current group labels not authenticated; "
                "independence and subgroup coverage are not guaranteed"
            )
            self._stat_plot.clear()
            names = [str(g.group) for g in result.groups] or ["overall"]
            coverages = (
                np.asarray([g.coverage for g in result.groups], dtype=float)
                if result.groups
                else np.asarray([result.overall.coverage])
            )
            self._stat_plot.bars(coverages, labels=names)
            self._stat_plot.line(
                np.arange(len(names), dtype=float),
                np.full(len(names), confidence),
                label="nominal", width=1, index=1, style="dash",
            )
            self._stat_plot.set_labels(
                x="group", y="coverage", title="Coverage by group",
            )

        self._submit(f"coverage diagnostics ({column})", job, done, self._stat_status)

    def _start_stat_subgroups(self) -> None:
        """Stratified performance + the corrected gap scan."""
        self._stat_frame = None
        self._stat_table.setRowCount(0)
        self._stat_plot.clear()
        resolved = self._stat_model_and_split()
        if resolved is None:
            return
        model, x_test, y_test, task, label = resolved
        frame, _target, _task2 = self._training_context()
        column = self._stat_group_column.currentText().strip()
        if frame is None or not column or column not in frame.columns:
            self._stat_status.setText(
                "Name a grouping column from the loaded dataset first — "
                "the subgroup audit stratifies the held-out rows by it."
            )
            return
        from pharmaml.exceptions import DataValidationError
        from pharmaml.ml.heldout_partition import aligned_group_values

        _frame, target, _task = self._training_context()
        try:
            groups = aligned_group_values(frame, x_test, y_test, target, column)
        except DataValidationError as exc:
            self._stat_status.setText(str(exc))
            return
        min_group_size = int(self._stat_mingroup.value())

        def job(progress: Any) -> Any:
            from pharmaml.ml.subgroups import subgroup_gap_scan, subgroup_report

            progress(0.4, "scoring the subgroups")
            report = subgroup_report(
                model, x_test, y_test, groups, task=task, min_group_size=min_group_size
            )
            progress(0.8, "scanning the gaps (family-corrected)")
            gaps = subgroup_gap_scan(
                model, x_test, y_test, groups, task=task, min_group_size=min_group_size
            )
            return report, gaps

        def done(result: Any) -> None:
            report, gaps = result
            rows = []
            for s in report.subgroups:
                metric_bits = [
                    f"{name}={value:.3g}" if value is not None else f"{name}=—"
                    for name, value in s.metrics.items()
                ]
                rows.append(
                    [
                        s.value,
                        str(s.n),
                        "yes" if s.sufficient else "NO",
                        " · ".join(metric_bits[:3]),
                    ]
                )
            _populate_table(
                self._stat_table,
                ["subgroup", "n", "sufficient", "held-out metrics"],
                rows,
            )
            self._stat_frame = pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "subgroup": s.value,
                                "n": s.n,
                                "sufficient": s.sufficient,
                                **s.metrics,
                            }
                            for s in report.subgroups
                        ]
                    ),
                    pd.DataFrame(
                        [
                            {
                                "subgroup": g.value,
                                "metric": g.metric,
                                "n": g.n,
                                "delta": g.delta,
                                "direction": g.direction,
                                "fdr_p": g.fdr_p_value,
                                "significant": g.fdr_significant,
                            }
                            for g in gaps
                        ]
                    ),
                ],
                ignore_index=True,
            )
            significant = sum(1 for g in gaps if g.fdr_significant)
            primary = (
                report.metric_names[0] if report.metric_names else "metric"
            )
            self._stat_status.setText(
                f"{len(report.subgroups)} subgroups over {primary} · "
                f"{len(gaps)} gaps tested (fdr_bh) — {significant} significant · "
                f"source {label} · column {column} · current group labels not authenticated"
            )
            self._stat_plot.clear()
            names = [s.value for s in report.subgroups]
            values = np.asarray(
                [
                    s.metrics.get(primary) if s.metrics.get(primary) is not None else np.nan
                    for s in report.subgroups
                ],
                dtype=float,
            )
            self._stat_plot.bars(values, labels=names)
            overall = report.overall.get(primary)
            if overall is not None:
                self._stat_plot.line(
                    np.arange(len(names), dtype=float),
                    np.full(len(names), overall),
                    label="overall", width=1, index=1, style="dash",
                )
            self._stat_plot.set_labels(
                x="subgroup", y=primary, title=f"Subgroup {primary} vs overall",
            )

        self._submit(f"subgroup audit ({column})", job, done, self._stat_status)

    # -- theming / test surface ---------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        """Re-colour every PlotPanel under the active theme."""
        for plot in (self._stack_plot, self._ens_plot, self._ad_plot, self._stat_plot):
            plot.apply_palette(palette)

    def report_text(self) -> str:
        """The tuner card's summary line (the page's primary status)."""
        return self._tune_summary.text()

    def bench_checkpoint_dir(self) -> Path | None:
        """The benchmark checkpoint home (the §11 evidence context reads it)."""
        return self._bench_checkpoint_dir
