"""Deep Learning Workbench: training runs, runboard, UMAP, curation (audit row 8).

The Studio's increment-7 page (the playground Deep Learning tab). The
audit row's four clusters -- the training run with its figures, the
runboard, the UMAP map and the fiftyone-style curation -- each wired to
the SAME public backend APIs the playground cards call, over a shared
dataset + target selection:

- **Train & evaluate** -- ``make_train_test_split`` (random, seeded)
  over the numeric-feature rule (every numeric column except the
  target), then the ``pharmaml.ml`` estimators: ``DeepMLP*`` /
  ``ResidualMLP*`` (torch) or ``NumpyMLP*`` (the toolkit's own numpy
  autograd engine when torch is absent), fitted with the per-epoch
  ``on_epoch`` callback, then ``evaluate_model`` on the held-out rows.
  Regression renders the parity scatter with the R²/RMSE/MAE triplet
  plus the overfit line (Pearson r, train R², the gap); classification
  renders the per-class F1 bars with the confusion table.  The fitted
  network + its test rows ride the page state for the other tabs.
- **Runboard** -- ``RunTrace`` over the LAST run's ``history_``:
  ``add_scalar`` per epoch, the raw + EMA-smoothed (``smooth_series``,
  bias-corrected, TensorBoard's own slider math) loss curves, and the
  weight histogram via ``numpy_layer_stack`` + ``histogram_event``
  (the numpy engine's layers snapshot -- a torch run refuses honestly,
  naming the engine rule).
- **UMAP map** -- ``umap_embed`` on the LAST run's test rows (or the
  dataset's numeric matrix): the fuzzy k-NN graph, the a/b curve from
  min_dist, spectral init, negative-sampling SGD -- with the
  Venna-Kaski ``trustworthiness`` score at k<=10 in the report.
- **Curation** -- ``SampleSet.from_frame`` + the ``DataView`` stage
  algebra (``filter`` / ``sort_by`` / ``limit`` / the identity reset),
  then the four audits: ``find_duplicates`` (identical-field groups),
  ``find_outliers`` (z-score > 4), ``uniqueness`` (1 - mean cosine
  similarity to the k nearest neighbours) and ``sort_by_similarity``
  (the projector's click: the k most cosine-similar rows).

Every heavy path runs in a :class:`~ui.workers.job_worker.JobWorker`;
they are single-shot engine calls without checkpoint loops, so the
cooperative Stop only lands for checkpoint-wired jobs -- the same
disclosure the PK/PD, Biology, Docking and Unsupervised pages carry.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[2]

#: The page's datasets: the QSAR regression table (260 rows x 21 cols,
#: target pIC50 over 20 descriptor features) and the QSAR classification
#: table (420 rows x 11 cols, target activity over 8 descriptor features).
_DATASETS = (
    ("qsar_regression.csv", "pIC50"),
    ("qsar_classification.csv", "activity"),
)
_DATA_EXPLORER_ENTRY = "(table from Data Explorer)"

_ARCHITECTURES: list[tuple[str, str]] = [
    ("deep", "DeepMLP (torch)"),
    ("residual", "ResidualMLP (torch)"),
    ("numpy", "NumpyMLP (numpy autograd)"),
]

_FILTER_OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "between", "contains")


def _maybe_number(text: str) -> Any:
    """A form value as int/float when it parses, else the raw string."""
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _parity_meta(y_test: Any, y_pred: Any) -> Any:
    """Per-point hover metadata for the deep-learning parity plot (§21.5).

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


def _embedding_meta(n_rows: int) -> Any:
    """Per-point hover metadata for the UMAP embedding scatter (§21.5).

    x/y are the embedding coordinates; the 0-based source row is the
    added fact -- the same index the row column of the bound table and
    the view algebra use, so a hover points straight back at the data.
    """

    def meta(i: int) -> dict[str, Any]:
        if not 0 <= i < n_rows:
            return {}
        return {"row": i}

    return meta


def _parse_hidden(text: str) -> tuple[int, ...]:
    """Comma-separated widths -> a tuple (empty text = the linear net)."""
    stripped = text.strip()
    if not stripped:
        return ()
    widths: list[int] = []
    for piece in stripped.split(","):
        piece = piece.strip()
        if not piece:
            continue
        width = int(piece)
        if width < 1:
            raise ValueError(f"layer widths must be >= 1, got {width}")
        widths.append(width)
    return tuple(widths)


def _torch_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


@dataclass
class TrainOutcome:
    """What the training job hands back (the network + its report)."""

    network: Any = None
    report: Any = None
    estimator_name: str = ""
    engine: str = ""
    task: str = ""
    target: str = ""
    n_features: int = 0
    n_train: int = 0
    n_test: int = 0
    epochs_asked: int = 0
    epochs_run: int = 0
    seconds: float = 0.0
    x_test: Any = None
    y_test: Any = None


@dataclass
class CurvesOutcome:
    """What the runboard curves job hands back."""

    epochs: np.ndarray | None = None
    raw_train: list[float] = field(default_factory=list)
    ema_train: list[float] = field(default_factory=list)
    raw_val: list[float] = field(default_factory=list)
    ema_val: list[float] = field(default_factory=list)
    beta: float = 0.6
    seconds: float = 0.0


@dataclass
class WeightsOutcome:
    """What the runboard weights job hands back."""

    events: list[Any] = field(default_factory=list)
    bias_stds: list[float] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class UmapOutcome:
    """What the UMAP job hands back."""

    record: Any = None
    score: float = float("nan")
    source: str = ""
    seconds: float = 0.0


@dataclass
class AuditOutcome:
    """What a curation audit job hands back."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""
    seconds: float = 0.0


class DeepLearningWorkbenchPage(QWidget):
    """The Deep Learning tab's four clusters over a shared dataset."""

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
        self._last_run: TrainOutcome | None = None
        self._view: Any = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # -- shared dataset + target selection ---------------------------------
        top = QFormLayout()
        self._dataset_combo = QComboBox()
        for name, _target in _DATASETS:
            self._dataset_combo.addItem(f"{name} (bundled)")
        self._dataset_combo.addItem(_DATA_EXPLORER_ENTRY)
        self._dataset_combo.currentTextChanged.connect(self._load_dataset)
        top.addRow("dataset:", self._dataset_combo)
        self._target_combo = QComboBox()
        top.addRow("target column:", self._target_combo)
        lay.addLayout(top)

        # -- the four engine tabs ------------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_train_tab(), "Train & evaluate")
        self._tabs.addTab(self._build_runboard_tab(), "Runboard")
        self._tabs.addTab(self._build_umap_tab(), "UMAP map")
        self._tabs.addTab(self._build_curation_tab(), "Curation")
        # part6 Phase 3, batch 3: the efficiency + GNN labs
        self._tabs.addTab(self._build_efficiency_tab(), "Efficiency lab")
        self._tabs.addTab(self._build_gnn_lab_tab(), "GNN lab")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Deep learning panel")
        splitter.addWidget(self._plot)
        splitter.setSizes([560, 540])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No deep-learning run yet — pick a dataset and train a network.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

        self._load_dataset(self._dataset_combo.currentText())

    # -- tab construction -----------------------------------------------------
    def _build_train_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._architecture = QComboBox()
        for _key, label in _ARCHITECTURES:
            self._architecture.addItem(label)
        form.addRow("architecture:", self._architecture)
        self._hidden_entry = QLineEdit("32,16")
        form.addRow("hidden layers:", self._hidden_entry)
        task_row = QWidget()
        task_lay = QHBoxLayout(task_row)
        task_lay.setContentsMargins(0, 0, 0, 0)
        self._task_combo = QComboBox()
        for task in ("auto", "regression", "classification"):
            self._task_combo.addItem(task)
        task_lay.addWidget(self._task_combo)
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(40)
        task_lay.addWidget(QLabel("epochs:"))
        task_lay.addWidget(self._epochs_spin)
        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(1, 100)
        self._patience_spin.setValue(8)
        task_lay.addWidget(QLabel("patience:"))
        task_lay.addWidget(self._patience_spin)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        task_lay.addWidget(QLabel("seed:"))
        task_lay.addWidget(self._seed_spin)
        form.addRow("task / budget:", task_row)
        self._train_btn = QPushButton("Train & evaluate")
        self._train_btn.setMinimumHeight(34)
        self._train_btn.clicked.connect(self._start_train)
        form.addRow("", self._train_btn)
        hint = QLabel(
            "make_train_test_split (random, seeded) over the numeric features, then "
            "DeepMLP*/ResidualMLP* (torch) or NumpyMLP* (the numpy autograd twin "
            "when torch is absent) with per-epoch callbacks and early stopping on a "
            "held-out validation fraction, then evaluate_model on the test rows: "
            "the parity scatter + the overfit line (regression) or the per-class "
            "F1 bars + the confusion table (classification)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_runboard_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._beta_spin = QDoubleSpinBox()
        self._beta_spin.setRange(0.0, 0.95)
        self._beta_spin.setSingleStep(0.05)
        self._beta_spin.setValue(0.6)
        self._beta_spin.setToolTip("0 = raw curves (TensorBoard's EMA slider semantics)")
        form.addRow("EMA beta (0 = raw):", self._beta_spin)
        self._curves_btn = QPushButton("Loss curves (EMA)")
        self._curves_btn.setMinimumHeight(34)
        self._curves_btn.clicked.connect(self._start_curves)
        form.addRow("", self._curves_btn)
        self._weights_btn = QPushButton("Weight histogram")
        self._weights_btn.setMinimumHeight(34)
        self._weights_btn.clicked.connect(self._start_weights)
        form.addRow("", self._weights_btn)
        hint = QLabel(
            "RunTrace over the LAST run's per-epoch history_: add_scalar per epoch, "
            "the raw + EMA-smoothed loss curves (smooth_series, the bias-corrected "
            "EMA TensorBoard's slider computes), then numpy_layer_stack + "
            "histogram_event: the numpy engine's weight buckets per layer -- a torch "
            "run refuses honestly (the stack needs the numpy engine's .linears)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_umap_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        knobs = QWidget()
        knobs_lay = QHBoxLayout(knobs)
        knobs_lay.setContentsMargins(0, 0, 0, 0)
        self._umap_neighbors = QSpinBox()
        self._umap_neighbors.setRange(5, 50)
        self._umap_neighbors.setValue(15)
        knobs_lay.addWidget(QLabel("neighbours:"))
        knobs_lay.addWidget(self._umap_neighbors)
        self._umap_mindist = QDoubleSpinBox()
        self._umap_mindist.setRange(0.0, 0.99)
        self._umap_mindist.setSingleStep(0.05)
        self._umap_mindist.setValue(0.1)
        knobs_lay.addWidget(QLabel("min_dist:"))
        knobs_lay.addWidget(self._umap_mindist)
        self._umap_metric = QComboBox()
        for metric in ("euclidean", "manhattan", "cosine"):
            self._umap_metric.addItem(metric)
        knobs_lay.addWidget(QLabel("metric:"))
        knobs_lay.addWidget(self._umap_metric)
        form.addRow("knobs:", knobs)
        self._embed_btn = QPushButton("Embed")
        self._embed_btn.setMinimumHeight(34)
        self._embed_btn.clicked.connect(self._start_umap)
        form.addRow("", self._embed_btn)
        hint = QLabel(
            "umap_embed on the LAST run's test rows (or the dataset's numeric "
            "matrix): the fuzzy k-NN graph, the a/b curve from min_dist, spectral "
            "init and negative-sampling SGD, n_epochs 200 -- the title carries the "
            "Venna-Kaski trustworthiness score (k<=10)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_curation_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        stage_row = QWidget()
        stage_lay = QHBoxLayout(stage_row)
        stage_lay.setContentsMargins(0, 0, 0, 0)
        self._cur_field = QComboBox()
        stage_lay.addWidget(self._cur_field)
        self._cur_op = QComboBox()
        for op in _FILTER_OPS:
            self._cur_op.addItem(op)
        stage_lay.addWidget(self._cur_op)
        self._cur_value = QLineEdit()
        self._cur_value.setPlaceholderText("value ('lo,hi' for between, comma list for in)")
        stage_lay.addWidget(self._cur_value)
        self._filter_btn = QPushButton("Filter")
        self._filter_btn.clicked.connect(self._apply_filter)
        stage_lay.addWidget(self._filter_btn)
        form.addRow("stage:", stage_row)
        chain_row = QWidget()
        chain_lay = QHBoxLayout(chain_row)
        chain_lay.setContentsMargins(0, 0, 0, 0)
        self._sort_btn = QPushButton("Sort by target desc")
        self._sort_btn.clicked.connect(self._apply_sort)
        chain_lay.addWidget(self._sort_btn)
        self._limit_btn = QPushButton("Limit 5")
        self._limit_btn.clicked.connect(self._apply_limit)
        chain_lay.addWidget(self._limit_btn)
        self._reset_btn = QPushButton("Reset view")
        self._reset_btn.clicked.connect(self._reset_view)
        chain_lay.addWidget(self._reset_btn)
        form.addRow("chain:", chain_row)
        audit_row = QWidget()
        audit_lay = QHBoxLayout(audit_row)
        audit_lay.setContentsMargins(0, 0, 0, 0)
        self._duplicates_btn = QPushButton("Duplicates")
        self._duplicates_btn.clicked.connect(self._start_duplicates)
        audit_lay.addWidget(self._duplicates_btn)
        self._outliers_btn = QPushButton("Outliers (z>4)")
        self._outliers_btn.clicked.connect(self._start_outliers)
        audit_lay.addWidget(self._outliers_btn)
        self._uniqueness_btn = QPushButton("Uniqueness")
        self._uniqueness_btn.clicked.connect(self._start_uniqueness)
        audit_lay.addWidget(self._uniqueness_btn)
        self._similarity_btn = QPushButton("Similarity")
        self._similarity_btn.clicked.connect(self._start_similarity)
        audit_lay.addWidget(self._similarity_btn)
        self._sim_row = QSpinBox()
        self._sim_row.setRange(0, 999999)
        self._sim_row.setValue(0)
        audit_lay.addWidget(QLabel("row:"))
        audit_lay.addWidget(self._sim_row)
        form.addRow("audits:", audit_row)
        hint = QLabel(
            "SampleSet.from_frame + the DataView stage algebra (filter / sort_by / "
            "limit / identity reset, every stage named in the report), then the four "
            "audits: find_duplicates (identical-field groups), find_outliers (z-score "
            "> 4), uniqueness (1 - mean cosine similarity to the k=5 nearest "
            "neighbours) and sort_by_similarity (row 0's k=5 most cosine-similar "
            "rows)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- data ------------------------------------------------------------------
    def _load_dataset(self, name: str) -> None:
        self._target_combo.clear()
        self._cur_field.clear()
        base = name.replace(" (bundled)", "")
        frame: pd.DataFrame | None
        if base == _DATA_EXPLORER_ENTRY:
            frame = None
            if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
                frame = self._data_explorer.frame()
            if frame is None or frame.empty:
                self._frame = None
                self._view = None
                self._report.setText("Data Explorer has no table loaded yet.")
                return
        else:
            path = _TK / "examples" / "data" / base
            if not path.exists():  # pragma: no cover - missing example
                self._frame = None
                return
            from pharmaml.utils.io import read_table

            frame = read_table(path)
        self._frame = frame
        for column in frame.columns:
            self._target_combo.addItem(str(column))
            self._cur_field.addItem(str(column))
        default_target = dict(_DATASETS).get(base, "")
        if default_target and default_target in [str(c) for c in frame.columns]:
            self._target_combo.setCurrentText(default_target)
        elif self._target_combo.count():
            self._target_combo.setCurrentIndex(self._target_combo.count() - 1)
        if self._cur_field.count():
            self._cur_field.setCurrentText(default_target or str(frame.columns[0]))
        self._view = None  # the curation view builds lazily from the frame
        self._report.setText(
            f"{base}: {frame.shape[0]} rows · {frame.shape[1]} columns · target {self._target_combo.currentText()!r}"
        )

    def _numeric_features(self, target: str) -> list[str]:
        assert self._frame is not None
        return [
            str(column)
            for column in self._frame.columns
            if column != target and pd.api.types.is_numeric_dtype(self._frame[column])
        ]

    def _infer_task(self, target: str) -> str:
        assert self._frame is not None
        column = self._frame[target]
        if not pd.api.types.is_numeric_dtype(column):
            return "classification"
        return "classification" if column.nunique() <= 10 else "regression"

    # -- tab construction (part6 Phase 3, batch 3: the efficiency + GNN labs) ----
    def _build_efficiency_tab(self) -> QWidget:
        """The efficiency lab: pruning, quantization and the ZeRO planner.

        part6 Phase 3, batch 3 — three §3.2 workflows over the canonical
        backends (``ml.pruning``, ``ml.quantization``, ``ml.harness``):
        a tape MLP gets magnitude-pruned along Zhu & Gupta's ramp, a
        weight block gets the three-method quantization report, and a
        model size gets the ZeRO Table-3 memory arithmetic across
        stages 0-3.
        """
        tab = QWidget()
        form = QFormLayout(tab)

        self._eff_params = QSpinBox()
        self._eff_params.setRange(2, 4096)
        self._eff_params.setValue(256)
        self._eff_params.setToolTip(
            "The tape MLP's hidden size: Linear(in, hidden) + ReLU + "
            "Linear(hidden, out), seeded — the pruning and quantization "
            "subject."
        )
        form.addRow("hidden size:", self._eff_params)
        self._eff_sparsity = QDoubleSpinBox()
        self._eff_sparsity.setRange(0.05, 0.95)
        self._eff_sparsity.setDecimals(2)
        self._eff_sparsity.setSingleStep(0.05)
        self._eff_sparsity.setValue(0.5)
        self._eff_sparsity.setToolTip(
            "Target sparsity for the one-shot prune (and the ramp's "
            "final level for the iterative loop)."
        )
        form.addRow("target sparsity:", self._eff_sparsity)
        self._eff_scope = QComboBox()
        for token in ("local", "global"):
            self._eff_scope.addItem(token)
        self._eff_scope.setToolTip(
            "local: each Linear layer pruned to the target on its own; "
            "global: one threshold across all layers (small layers "
            "can vanish entirely)."
        )
        form.addRow("prune scope:", self._eff_scope)
        self._eff_steps = QSpinBox()
        self._eff_steps.setRange(1, 10)
        self._eff_steps.setValue(4)
        self._eff_steps.setToolTip(
            "The gradual ramp's levels for the iterative loop (1 = one-shot)."
        )
        form.addRow("ramp steps:", self._eff_steps)

        run_row = QWidget()
        run_lay = QHBoxLayout(run_row)
        run_lay.setContentsMargins(0, 0, 0, 0)
        self._eff_prune_btn = QPushButton("Prune")
        self._eff_prune_btn.clicked.connect(self._start_eff_prune)
        self._eff_prune_btn.setToolTip(
            "Magnitude-prune the tape MLP one-shot, then walk the gradual "
            "sparsity ramp with a retrain callback between levels — the "
            "per-layer report, the schedule and the recorded losses."
        )
        run_lay.addWidget(self._eff_prune_btn)
        self._eff_quant_btn = QPushButton("Quantize")
        self._eff_quant_btn.clicked.connect(self._start_eff_quant)
        self._eff_quant_btn.setToolTip(
            "The three-method quantization report over the MLP's first "
            "Linear weight: NF4 (block 64 + double quant), int8 "
            "per-tensor and int8 per-channel — bits per weight, RMSE, "
            "relative error."
        )
        run_lay.addWidget(self._eff_quant_btn)
        self._eff_zero_btn = QPushButton("ZeRO planner")
        self._eff_zero_btn.clicked.connect(self._start_eff_zero)
        self._eff_zero_btn.setToolTip(
            "The ZeRO Table-3 memory arithmetic for this parameter count "
            "across stages 0-3 and the chosen world size — what each "
            "stage partitions, in bytes."
        )
        run_lay.addWidget(self._eff_zero_btn)
        form.addRow("run:", run_row)

        self._eff_world = QSpinBox()
        self._eff_world.setRange(1, 64)
        self._eff_world.setValue(4)
        self._eff_world.setToolTip("Simulated ranks for the ZeRO planner.")
        form.addRow("world size:", self._eff_world)
        self._eff_millions = QDoubleSpinBox()
        self._eff_millions.setRange(0.1, 1000.0)
        self._eff_millions.setDecimals(1)
        self._eff_millions.setSingleStep(1.0)
        self._eff_millions.setValue(1.0)
        self._eff_millions.setToolTip(
            "The ZeRO planner's model size in millions of parameters "
            "(fp32 training: params + grads + an 8-byte Adam state pair)."
        )
        form.addRow("ZeRO model (M params):", self._eff_millions)

        hint = QLabel(
            "the efficiency lab: ml.pruning's magnitude mask + Zhu-Gupta "
            "ramp (tape MLP), ml.quantization's NF4/int8 report over the "
            "same weight, and ml.harness's ZeRO memory arithmetic — "
            "results land in the shared table/plot below."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_gnn_lab_tab(self) -> QWidget:
        """The GNN lab: motif dataset -> classifier -> explanation.

        part6 Phase 3, batch 3 — the geometric-deep-learning entry
        point (``ml.geometric`` + ``ml.gnnxai``'s fidelity arithmetic
        inside): the GNNExplainer synthetic motif dataset, a GCN/SAGE/
        GAT classifier trained on it, and a per-class edge-score
        explanation with the fidelity pair and the probability walk."
        """
        tab = QWidget()
        form = QFormLayout(tab)

        self._gnn_motif = QComboBox()
        for token in ("house", "cycle"):
            self._gnn_motif.addItem(token)
        self._gnn_motif.setToolTip(
            "The GNNExplainer synthetic's motif: a 5-node house (square "
            "plus roof) or a 6-node cycle, attached to half the random "
            "base graphs."
        )
        form.addRow("motif:", self._gnn_motif)
        self._gnn_graphs = QSpinBox()
        self._gnn_graphs.setRange(20, 400)
        self._gnn_graphs.setValue(120)
        self._gnn_graphs.setSingleStep(20)
        self._gnn_graphs.setToolTip("How many synthetic graphs to generate.")
        form.addRow("graphs:", self._gnn_graphs)
        self._gnn_layer = QComboBox()
        for token in ("gcn", "sage", "gat"):
            self._gnn_layer.addItem(token)
        self._gnn_layer.setToolTip("The convolution kind behind both layers.")
        form.addRow("convolution:", self._gnn_layer)
        self._gnn_epochs = QSpinBox()
        self._gnn_epochs.setRange(20, 500)
        self._gnn_epochs.setValue(200)
        self._gnn_epochs.setSingleStep(50)
        self._gnn_epochs.setToolTip("Training epochs over the collated graphs.")
        form.addRow("epochs:", self._gnn_epochs)
        self._gnn_sparsity = QDoubleSpinBox()
        self._gnn_sparsity.setRange(0.1, 0.9)
        self._gnn_sparsity.setDecimals(2)
        self._gnn_sparsity.setSingleStep(0.05)
        self._gnn_sparsity.setValue(0.7)
        self._gnn_sparsity.setToolTip(
            "The explanation's edge-sparsity: the fraction of edges the "
            "hard mask removes (DIG's control_sparsity)."
        )
        form.addRow("explain sparsity:", self._gnn_sparsity)
        self._gnn_seed = QSpinBox()
        self._gnn_seed.setRange(0, 999999)
        self._gnn_seed.setValue(27)
        self._gnn_seed.setToolTip("Deterministic seed for the classifier.")
        form.addRow("seed:", self._gnn_seed)

        self._gnn_train_btn = QPushButton("Train & explain")
        self._gnn_train_btn.setMinimumHeight(34)
        self._gnn_train_btn.clicked.connect(self._start_gnn_lab)
        self._gnn_train_btn.setToolTip(
            "Generate the motif dataset, fit the graph classifier, then "
            "explain graph #0: per-edge gate scores, the hard mask, the "
            "probability walk (origin -> masked -> maskout -> zero) and "
            "the fidelity pair — plus how many of the true motif edges "
            "the mask kept."
        )
        form.addRow("", self._gnn_train_btn)
        hint = QLabel(
            "ml.geometric's motif_dataset + GNNClassifier + gnn_explain "
            "(the gnnxai fidelity arithmetic inside): train on the "
            "synthetic motif graphs, then read which edges the "
            "explanation blames for the label."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- efficiency lab jobs --------------------------------------------------------
    def _eff_network(self, hidden: int):
        """The seeded tape MLP every efficiency workflow shares."""
        from pharmaml.ml.autograd import Linear, ReLU, Sequential

        return Sequential(Linear(8, hidden, seed=7), ReLU(), Linear(hidden, 4, seed=11))

    def _start_eff_prune(self) -> None:
        """One-shot + gradual magnitude pruning over the tape MLP."""
        if self._busy_check():
            return
        hidden = int(self._eff_params.value())
        sparsity = float(self._eff_sparsity.value())
        scope = self._eff_scope.currentText()
        steps = int(self._eff_steps.value())

        def job(progress):
            from pharmaml.ml.pruning import (
                apply_magnitude_pruning,
                iterative_prune_and_retrain,
                sparsity_schedule,
            )

            started = _time.monotonic()
            progress(0.2, "one-shot magnitude prune")
            fresh = self._eff_network(hidden)
            report = apply_magnitude_pruning(fresh, sparsity, scope=scope)
            progress(0.6, "walking the gradual ramp")
            schedule = sparsity_schedule(0.0, sparsity, steps)
            losses: list[float] = []

            def retrain(module) -> float:
                # the honest minimal retrace: the masked forward's loss on
                # a fixed batch (deterministic, no training loop here —
                # the ramp's own arithmetic is the subject)
                from pharmaml.ml.autograd import Tensor

                rng = np.random.default_rng(3)
                batch = Tensor(rng.normal(size=(16, 8)), requires_grad=False)
                target = np.zeros((16, 4))
                out = module.forward(batch)
                value = float(np.mean((np.asarray(out.data) - target) ** 2))
                losses.append(value)
                return value

            net, ramp_report, _recorded = iterative_prune_and_retrain(
                fresh, retrain, target_sparsity=sparsity, steps=steps, scope=scope
            )
            progress(1.0, "done")
            return report, ramp_report, schedule, _time.monotonic() - started

        def render(result) -> None:
            report, ramp_report, schedule, seconds = result
            rows = []
            for layer in report.layers:
                rows.append(
                    {
                        "layer": layer.name,
                        "shape": "x".join(str(s) for s in layer.shape),
                        "n_weights": layer.n_weights,
                        "n_zeroed": layer.n_zeroed,
                        "sparsity": round(layer.sparsity, 4),
                    }
                )
            table = pd.DataFrame(rows)
            table["kind"] = "one-shot"
            ramp_rows = pd.DataFrame(
                [
                    {
                        "layer": layer.name,
                        "shape": "x".join(str(s) for s in layer.shape),
                        "n_weights": layer.n_weights,
                        "n_zeroed": layer.n_zeroed,
                        "sparsity": round(layer.sparsity, 4),
                        "kind": "ramp-final",
                    }
                    for layer in ramp_report.layers
                ]
            )
            full = pd.concat([table, ramp_rows], ignore_index=True)
            self._table.bind_frame(full)
            self._report.setText(
                f"one-shot {report.achieved_sparsity:.1%} ({scope}) · ramp "
                f"{' -> '.join(f'{level:.0%}' for level in schedule)} · "
                f"final {ramp_report.achieved_sparsity:.1%} · "
                f"{len(report.layers)} Linear layers · pruned in {seconds:.1f}s"
            )
            self._plot.clear()
            levels = np.arange(len(schedule), dtype=float)
            self._plot.line(
                levels, np.asarray(schedule, dtype=float), label="ramp", width=2
            )
            self._plot.line(
                levels,
                np.full(len(schedule), report.achieved_sparsity),
                label="one-shot", width=1, index=1, style="dash",
            )
            self._plot.set_labels(
                x="ramp step", y="sparsity",
                title=f"Magnitude pruning ({scope}) — target {sparsity:.0%}",
            )

        self._runner.submit(
            "efficiency: pruning",
            job,
            on_finished=render,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
        )

    def _start_eff_quant(self) -> None:
        """The three-method quantization report over the MLP's weight."""
        if self._busy_check():
            return
        hidden = int(self._eff_params.value())

        def job(progress):
            from pharmaml.ml.quantization import quantization_report

            started = _time.monotonic()
            progress(0.3, "extracting the first Linear weight")
            net = self._eff_network(hidden)
            weight = np.asarray(net.layers[0].weight.data)
            reports = {}
            for index, method in enumerate(("nf4", "int8", "int8_channel")):
                progress(0.3 + 0.2 * (index + 1), f"quantizing ({method})")
                reports[method] = quantization_report(weight, method=method)
            progress(1.0, "done")
            return reports, weight.shape, _time.monotonic() - started

        def render(result) -> None:
            reports, shape, seconds = result
            rows = []
            for method, r in reports.items():
                rows.append(
                    {
                        "method": method,
                        "shape": "x".join(str(s) for s in r.shape),
                        "n_weights": r.n_weights,
                        "bits_per_weight": round(r.bits_per_weight, 3),
                        "rmse": float(f"{r.rmse:.3g}"),
                        "relative_error": round(r.relative_error, 4),
                    }
                )
            self._table.bind_frame(pd.DataFrame(rows))
            self._report.setText(
                f"weight {shape[0]}x{shape[1]} · nf4 {reports['nf4'].bits_per_weight} "
                f"bpw (rmse {reports['nf4'].rmse:.3g}) vs int8_channel "
                f"{reports['int8_channel'].bits_per_weight} bpw (rmse "
                f"{reports['int8_channel'].rmse:.3g}) · reported in {seconds:.1f}s"
            )
            self._plot.clear()
            methods = list(reports)
            positions = np.arange(len(methods), dtype=float)
            self._plot.bars(
                np.asarray([reports[m].relative_error for m in methods]),
                labels=[f"{m}\n{reports[m].bits_per_weight}bpw" for m in methods],
            )
            self._plot.set_labels(
                x="method", y="relative error",
                title="Quantization error by method",
            )

        self._runner.submit(
            "efficiency: quantization",
            job,
            on_finished=render,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
        )

    def _start_eff_zero(self) -> None:
        """The ZeRO Table-3 memory plan across stages 0-3."""
        if self._busy_check():
            return
        millions = float(self._eff_millions.value())
        world = int(self._eff_world.value())
        n_params = int(millions * 1_000_000)

        def job(progress):
            from pharmaml.ml.harness import zero_memory_model

            started = _time.monotonic()
            progress(0.5, "the Table-3 arithmetic")
            stages = {stage: zero_memory_model(n_params, world_size=world, stage=stage) for stage in range(4)}
            progress(1.0, "done")
            return stages, _time.monotonic() - started

        def render(result) -> None:
            stages, seconds = result
            rows = []
            for stage, z in stages.items():
                rows.append(
                    {
                        "stage": stage,
                        "partitioned": {0: "nothing", 1: "optimizer", 2: "grads+optimizer", 3: "params+grads+optimizer"}[stage],
                        "params_MB": round(z.params_bytes / 1e6, 1),
                        "grads_MB": round(z.grads_bytes / 1e6, 1),
                        "optimizer_MB": round(z.optimizer_bytes / 1e6, 1),
                        "total_MB": round(
                            (z.params_bytes + z.grads_bytes + z.optimizer_bytes) / 1e6, 1
                        ),
                    }
                )
            self._table.bind_frame(pd.DataFrame(rows))
            zero_total = stages[0].params_bytes + stages[0].grads_bytes + stages[0].optimizer_bytes
            stage3 = stages[3]
            s3_total = stage3.params_bytes + stage3.grads_bytes + stage3.optimizer_bytes
            self._report.setText(
                f"{millions:g}M params · world size {world} · stage 0 {zero_total/1e6:.0f} MB "
                f"-> stage 3 {s3_total/1e6:.0f} MB per rank ({1 - s3_total/zero_total:.0%} saved) "
                f"· planned in {seconds:.1f}s"
            )
            self._plot.clear()
            self._plot.bars(
                np.asarray([stages[s].params_bytes / 1e6 for s in range(4)]), labels=[f"stage {s}" for s in range(4)]
            )
            self._plot.set_labels(
                x="ZeRO stage", y="params MB per rank",
                title=f"ZeRO partitioning — {millions:g}M params, {world} ranks",
            )

        self._runner.submit(
            "efficiency: ZeRO planner",
            job,
            on_finished=render,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
        )

    # -- GNN lab job ---------------------------------------------------------------
    def _start_gnn_lab(self) -> None:
        """Motif dataset -> graph classifier -> edge explanation."""
        if self._busy_check():
            return
        motif = self._gnn_motif.currentText()
        n_graphs = int(self._gnn_graphs.value())
        layer_kind = self._gnn_layer.currentText()
        epochs = int(self._gnn_epochs.value())
        explain_sparsity = float(self._gnn_sparsity.value())
        seed = int(self._gnn_seed.value())

        def job(progress):
            from pharmaml.ml.geometric import GNNClassifier, gnn_explain, motif_dataset

            started = _time.monotonic()
            progress(0.1, f"generating {n_graphs} {motif} graphs")
            graphs = motif_dataset(n_graphs=n_graphs, seed=0, motif=motif)
            progress(0.15, f"training the {layer_kind} classifier")
            model = GNNClassifier(
                graphs[0].x.shape[1], 16, 2, layer_kind=layer_kind, seed=seed
            )
            model.fit(graphs, epochs=epochs, lr=0.05)
            accuracy = model.score(graphs)
            progress(0.85, "explaining graph #0")
            target = graphs[0]
            explanation = gnn_explain(
                model,
                target.x,
                target.edge_index,
                y=target.y,
                epochs=min(100, epochs),
                sparsity=explain_sparsity,
                seed=seed,
            )
            # motif recovery: how many true motif edges survive the mask
            house = {frozenset(edge) for edge in target.motif_edges}
            hard = explanation.hard_mask[int(target.y)]
            kept_edges = [
                frozenset((int(a), int(b)))
                for a, b, keep in zip(
                    explanation.edge_index[0], explanation.edge_index[1], hard
                )
                if keep
            ]
            kept_house = sum(1 for e in kept_edges if e in house)
            progress(1.0, "done")
            return (
                explanation, accuracy, len(graphs), kept_house, len(house),
                len(kept_edges), target.y, list(target.motif_edges),
                _time.monotonic() - started,
            )

        def render(result) -> None:
            (
                explanation, accuracy, n, kept_house, n_house, kept, label,
                motif_edges, seconds,
            ) = result
            scores = explanation.edge_scores[int(label)]
            hard = explanation.hard_mask[int(label)]
            edges = list(zip(explanation.edge_index[0], explanation.edge_index[1]))
            truth = {frozenset(edge) for edge in motif_edges}
            top = np.argsort(scores)[::-1][:12]
            rows = []
            for index in top:
                a, b = int(edges[index][0]), int(edges[index][1])
                rows.append(
                    {
                        "edge": f"{a}-{b}",
                        "score": round(float(scores[index]), 4),
                        "kept": bool(hard[index]),
                        "in_true_motif": frozenset((a, b)) in truth,
                    }
                )
            self._table.bind_frame(pd.DataFrame(rows))
            probs = explanation.probs
            self._report.setText(
                f"train accuracy {accuracy:.0%} over {n} graphs · label {label} · "
                f"kept {kept} edges ({explanation.sparsity:.0%} removed) · "
                f"motif edges in mask {kept_house}/{n_house} · "
                f"fidelity {explanation.fidelity:+.3f} / {explanation.fidelity_inv:+.3f} · "
                f"p(origin) {probs['origin']:.2f} -> p(masked) {probs['masked']:.2f} · "
                f"{seconds:.1f}s"
            )
            self._plot.clear()
            order = np.argsort(scores)
            self._plot.scatter(
                np.arange(len(order), dtype=float), np.asarray(scores)[order], label="edge score"
            )
            self._plot.set_labels(
                x="edge (sorted)", y="gate score",
                title=f"Explanation edge scores — label {label}",
            )

        self._runner.submit(
            "GNN lab: motif train + explain",
            job,
            on_finished=render,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
        )

    # -- job plumbing -----------------------------------------------------------
    def _buttons(self) -> tuple[QPushButton, ...]:
        return (
            self._train_btn,
            self._curves_btn,
            self._weights_btn,
            self._embed_btn,
            self._duplicates_btn,
            self._outliers_btn,
            self._uniqueness_btn,
            self._similarity_btn,
            self._eff_prune_btn,
            self._eff_quant_btn,
            self._eff_zero_btn,
            self._gnn_train_btn,
        )

    def _on_started(self, name: str) -> None:
        for btn in self._buttons():
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in self._buttons():
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return True
        return False

    # -- train & evaluate -----------------------------------------------------------
    def _start_train(self) -> None:
        if self._busy_check():
            return
        if self._frame is None:
            self._report.setText("Load a dataset first.")
            return
        target = self._target_combo.currentText()
        if not target:
            self._report.setText("Pick the target column.")
            return
        numeric = self._numeric_features(target)
        if not numeric:
            self._report.setText(
                "The estimators need numeric feature columns — every other column is non-numeric."
            )
            return
        try:
            hidden = _parse_hidden(self._hidden_entry.text())
        except ValueError as exc:
            self._report.setText(f"Check the hidden layers — {exc}")
            return
        task = self._task_combo.currentText()
        if task == "auto":
            task = self._infer_task(target)
        family = _ARCHITECTURES[self._architecture.currentIndex()][0]
        epochs = self._epochs_spin.value()
        patience = self._patience_spin.value()
        seed = self._seed_spin.value()
        frame = self._frame

        def job(progress) -> TrainOutcome:
            from pharmaml.ml import evaluate_model
            from pharmaml.ml.splits import make_train_test_split

            started = _time.monotonic()
            x_train, x_test, y_train, y_test = make_train_test_split(
                frame[numeric], frame[target], strategy="random", random_state=seed
            )

            def on_epoch(info: dict[str, Any]) -> None:
                note = (
                    f"val_loss {info['val_loss']:.3g}"
                    if info.get("val_loss") is not None
                    else f"train_loss {info['train_loss']:.3g}"
                )
                progress(
                    info["epoch"] / max(info["epochs"], 1),
                    f"epoch {info['epoch']}/{info['epochs']} · {note}",
                )

            engine = "numpy" if family == "numpy" or not _torch_available() else "torch"
            kwargs: dict[str, Any] = {
                "hidden_layers": hidden,
                "activation": "relu",
                "epochs": epochs,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "patience": patience,
                "val_split": 0.15,
                "dropout": 0.1,
                "random_state": seed,
            }
            estimator_cls: type
            if engine == "numpy":
                from pharmaml.ml.numpy_neural import NumpyMLPClassifier, NumpyMLPRegressor

                estimator_cls = NumpyMLPRegressor if task == "regression" else NumpyMLPClassifier
                kwargs["optimizer"] = "adam"
            else:
                from pharmaml.ml.neural import (
                    DeepMLPClassifier,
                    DeepMLPRegressor,
                    ResidualMLPClassifier,
                    ResidualMLPRegressor,
                )

                if family == "residual":
                    estimator_cls = (
                        ResidualMLPRegressor if task == "regression" else ResidualMLPClassifier
                    )
                else:
                    estimator_cls = DeepMLPRegressor if task == "regression" else DeepMLPClassifier
            network = estimator_cls(**kwargs)
            network.fit(x_train, y_train, on_epoch=on_epoch)
            train_side: dict[str, Any] = (
                {"X_train": x_train, "y_train": y_train} if task == "regression" else {}
            )
            report = evaluate_model(
                network,
                x_test,
                y_test,
                task,
                keep_predictions=(task == "regression"),
                **train_side,
            )
            progress(1.0, "done")
            return TrainOutcome(
                network=network,
                report=report,
                estimator_name=type(network).__name__,
                engine=engine,
                task=task,
                target=target,
                n_features=len(numeric),
                n_train=len(x_train),
                n_test=len(x_test),
                epochs_asked=epochs,
                epochs_run=int(network.n_iter_),
                seconds=_time.monotonic() - started,
                x_test=x_test,
                y_test=y_test,
            )

        self._runner.submit(
            f"training {family} ({task}) on {target}",
            job,
            on_finished=self._render_train,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_train(self, outcome: TrainOutcome) -> None:
        # part5 §11: the run's REAL history (epoch/train_loss/val_loss per
        # epoch, from the trained network) becomes a saved run — the DL
        # card's evidence. Best-effort by contract, never fails the render.
        try:
            from pharmaml.studio import record_run

            record_run(
                "dl-training",
                f"{outcome.estimator_name} on {outcome.target}",
                {
                    "engine": outcome.engine,
                    "task": outcome.task,
                    "target": outcome.target,
                    "n_train": outcome.n_train,
                    "n_test": outcome.n_test,
                    "epochs_run": outcome.epochs_run,
                    "epochs_asked": outcome.epochs_asked,
                    "history": [dict(point) for point in outcome.network.history_],
                },
            )
        except Exception:  # noqa: BLE001, S110 -- the record must never fail the render
            pass
        self._plot.clear()
        self._last_run = outcome
        report = outcome.report
        early = (
            f"early stop at epoch {outcome.epochs_run}"
            if outcome.epochs_run < outcome.epochs_asked
            else f"ran all {outcome.epochs_asked} epochs"
        )
        if outcome.task == "regression":
            actual = np.asarray(outcome.y_test, dtype=float)
            predictions = np.asarray(report.predictions, dtype=float)
            self._plot.scatter(actual, predictions, label="held-out rows", meta=_parity_meta(actual, predictions))
            lo = float(min(actual.min(), predictions.min()))
            hi = float(max(actual.max(), predictions.max()))
            self._plot.line(
                np.array([lo, hi]),
                np.array([lo, hi]),
                label="parity (y = x)",
                width=2,
                index=1,
                style="dash",
            )
            self._plot.add_legend()
            self._plot.set_labels(
                x=f"actual {outcome.target}",
                y=f"predicted {outcome.target}",
                title=f"Parity · {outcome.estimator_name}",
            )
            self._table.bind_frame(
                pd.DataFrame(
                    {
                        f"actual {outcome.target}": actual,
                        f"predicted {outcome.target}": predictions,
                        "error": predictions - actual,
                    }
                )
            )
            overfit = " · ".join(
                f"{name} {value:.3f}"
                for name, value in (
                    ("Pearson r", report.pearson_r),
                    ("train R²", report.train_r2),
                    ("gap", report.r2_gap),
                )
                if value is not None
            )
            self._report.setText(
                f"{outcome.estimator_name} · {outcome.task} · R² {report.r2:.3f} · "
                f"RMSE {report.rmse:.3f} · MAE {report.mae:.3f}"
                + (f" · {overfit}" if overfit else "")
                + f" · {early} · {outcome.n_train} train / "
                f"{outcome.n_test} test rows over {outcome.n_features} features · "
                f"trained in {outcome.seconds:.1f}s"
            )
        else:
            labels = [str(label) for label in report.labels]
            f1s = [float(report.per_class[label]["f1"]) for label in labels]
            self._plot.bars(np.asarray(f1s, dtype=float), labels=labels)
            self._plot.set_labels(
                x="class", y="F1", title=f"Per-class F1 · {outcome.estimator_name}"
            )
            confusion = np.asarray(report.confusion)
            frame = pd.DataFrame(
                confusion,
                index=[f"actual {label}" for label in labels],
                columns=[f"pred {label}" for label in labels],
            )
            self._table.bind_frame(
                frame.reset_index().rename(columns={"index": "actual \\ predicted"})
            )
            supports = "/".join(
                f"{label} {int(report.per_class[label]['support'])}" for label in labels
            )
            self._report.setText(
                f"{outcome.estimator_name} · {outcome.task} · accuracy "
                f"{report.accuracy:.3f} · balanced {report.balanced_accuracy:.3f} · "
                f"F1 macro {report.f1_macro:.3f} · {len(labels)} classes "
                f"({supports} test rows) · {early} · {outcome.n_train} train / "
                f"{outcome.n_test} test rows over {outcome.n_features} features · "
                f"trained in {outcome.seconds:.1f}s"
            )

    # -- runboard: the loss curves -------------------------------------------------------
    def _start_curves(self) -> None:
        if self._busy_check():
            return
        if self._last_run is None or not self._last_run.network.history_:
            self._report.setText(
                "Train a network first — the board reads the LAST run's per-epoch history."
            )
            return
        beta = float(self._beta_spin.value())
        history = list(self._last_run.network.history_)

        def job(progress) -> CurvesOutcome:
            from pharmaml.ml.runboard import RunTrace, smooth_series

            started = _time.monotonic()
            progress(0.4, "replaying the epoch log into RunTrace")
            trace = RunTrace("last-run")
            for record in history:
                epoch = int(record.get("epoch", 0))
                trace.add_scalar("loss/train", float(record.get("train_loss", 0.0)), step=epoch)
                if record.get("val_loss") is not None:
                    trace.add_scalar("loss/val", float(record["val_loss"]), step=epoch)
            raw_train = [event.value for event in trace.scalars("loss/train")]
            raw_val = [event.value for event in trace.scalars("loss/val")]
            progress(0.8, "smoothing")
            out = CurvesOutcome(
                epochs=np.arange(1, len(raw_train) + 1, dtype=float),
                raw_train=raw_train,
                ema_train=smooth_series(raw_train, beta=beta),
                raw_val=raw_val,
                ema_val=smooth_series(raw_val, beta=beta) if raw_val else [],
                beta=beta,
                seconds=_time.monotonic() - started,
            )
            progress(1.0, "done")
            return out

        self._runner.submit(
            f"runboard curves (beta {beta:g})",
            job,
            on_finished=self._render_curves,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_curves(self, outcome: CurvesOutcome) -> None:
        self._plot.clear()
        if outcome.epochs is None:  # the dataclass default; the job always fills it
            self._report.setText("No epoch axis in the outcome — nothing to draw.")
            return
        epochs = outcome.epochs
        self._plot.line(
            epochs, np.asarray(outcome.raw_train), label="loss/train (raw)", width=1, index=0
        )
        self._plot.line(
            epochs,
            np.asarray(outcome.ema_train),
            label=f"loss/train (EMA {outcome.beta:g})",
            width=2,
            index=0,
        )
        if outcome.raw_val:
            self._plot.line(
                epochs, np.asarray(outcome.raw_val), label="loss/val (raw)", width=1, index=1
            )
            self._plot.line(
                epochs,
                np.asarray(outcome.ema_val),
                label=f"loss/val (EMA {outcome.beta:g})",
                width=2,
                index=1,
            )
        self._plot.add_legend()
        self._plot.set_labels(x="epoch", y="loss", title="Run board — the loss curves")
        history_frame = pd.DataFrame(
            {
                "epoch": epochs.astype(int),
                "train_loss": outcome.raw_train,
                "train_ema": outcome.ema_train,
                "val_loss": outcome.raw_val if outcome.raw_val else [float("nan")] * len(epochs),
                "val_ema": outcome.ema_val if outcome.ema_val else [float("nan")] * len(epochs),
            }
        )
        self._table.bind_frame(history_frame)
        first = outcome.raw_train[0] if outcome.raw_train else float("nan")
        last_ema = outcome.ema_train[-1] if outcome.ema_train else float("nan")
        val_note = f" · {len(outcome.raw_val)} validation points" if outcome.raw_val else ""
        self._report.setText(
            f"runboard · {len(outcome.raw_train)} epochs recorded · smoothing beta "
            f"{outcome.beta:g} (0 = raw) · train {first:.3f} → {last_ema:.3f} (EMA "
            f"last){val_note} · replayed in {outcome.seconds:.1f}s"
        )

    # -- runboard: the weight histogram ----------------------------------------------------
    def _start_weights(self) -> None:
        if self._busy_check():
            return
        if self._last_run is None:
            self._report.setText(
                "Train a network first — weight histograms read the LAST trained network."
            )
            return
        network = self._last_run.network

        def job(progress) -> WeightsOutcome:
            from pharmaml.ml.runboard import histogram_event
            from pharmaml.ml.xai import numpy_layer_stack

            started = _time.monotonic()
            progress(0.4, "snapshotting the layer stack")
            stack = numpy_layer_stack(network)
            events = []
            bias_stds: list[float] = []
            for weight, bias in stack.layers:
                values = np.asarray(weight, dtype=float).reshape(-1)
                events.append(histogram_event(values, step=0))
                if bias is not None:
                    bias_values = np.asarray(bias, dtype=float).reshape(-1)
                    bias_stds.append(float(histogram_event(bias_values, step=0).std))
                else:
                    bias_stds.append(float("nan"))
            progress(1.0, "done")
            return WeightsOutcome(
                events=events, bias_stds=bias_stds, seconds=_time.monotonic() - started
            )

        self._runner.submit(
            "runboard weight histogram",
            job,
            on_finished=self._render_weights,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_weights(self, outcome: WeightsOutcome) -> None:
        self._plot.clear()
        first = outcome.events[0]
        self._plot.bars(
            np.asarray(first.bucket_counts, dtype=float),
            labels=None,
            width=1.0,
        )
        self._plot.set_labels(
            x="weight bucket",
            y="count",
            title=f"weights/layer1 — {first.count} weights · mean {first.mean:.3g} ± {first.std:.3g}",
        )
        rows = []
        for index, event in enumerate(outcome.events, start=1):
            rows.append(
                {
                    "layer": f"layer{index}",
                    "weights": int(event.count),
                    "min": float(event.min),
                    "max": float(event.max),
                    "mean": float(event.mean),
                    "std": float(event.std),
                    "bias std": outcome.bias_stds[index - 1],
                }
            )
        self._table.bind_frame(pd.DataFrame(rows))
        parts = [
            f"weights/layer1 — {first.count} weights · min {first.min:.3g} · max {first.max:.3g}"
        ]
        parts.append(f"mean {first.mean:.3g} ± {first.std:.3g}")
        parts.append(f"{len(outcome.events)} layers snapshotted (the numpy engine's .linears)")
        parts.append(f"histogrammed in {outcome.seconds:.1f}s")
        self._report.setText(" · ".join(parts))

    # -- the UMAP map -------------------------------------------------------------------------
    def _start_umap(self) -> None:
        if self._busy_check():
            return
        matrix: np.ndarray | None = None
        source = ""
        candidate: np.ndarray | None = None
        if self._last_run is not None and self._last_run.x_test is not None:
            candidate = np.asarray(self._last_run.x_test, dtype=float)
        if candidate is not None and candidate.ndim == 2 and candidate.shape[0] >= 10:
            matrix = candidate
            source = "the last run's test rows"
        elif self._frame is not None:
            target = self._target_combo.currentText()
            numeric = [
                column
                for column in self._frame.columns
                if column != target and pd.api.types.is_numeric_dtype(self._frame[column])
            ]
            if len(numeric) >= 2 and len(self._frame) >= 10:
                matrix = self._frame[numeric].to_numpy(dtype=float)
                source = f"the dataset's {len(numeric)} numeric columns"
        if matrix is None:
            self._report.setText(
                "Not enough rows — UMAP needs a run's test rows or a dataset with at least 10 numeric rows."
            )
            return
        n_neighbors = min(self._umap_neighbors.value(), matrix.shape[0] - 1)
        min_dist = float(self._umap_mindist.value())
        metric = self._umap_metric.currentText()
        final_matrix = matrix

        def job(progress) -> UmapOutcome:
            from pharmaml.ml.umap import trustworthiness, umap_embed

            started = _time.monotonic()
            progress(0.3, "building the fuzzy k-NN graph")
            record = umap_embed(
                final_matrix,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                n_epochs=200,
                random_state=42,
            )
            progress(0.85, "scoring trustworthiness")
            score = trustworthiness(
                final_matrix,
                record.embedding_array,
                n_neighbors=min(10, final_matrix.shape[0] // 3 or 1),
            )
            progress(1.0, "done")
            return UmapOutcome(
                record=record,
                score=float(score),
                source=source,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"UMAP over {matrix.shape[0]} rows ({metric})",
            job,
            on_finished=self._render_umap,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_umap(self, outcome: UmapOutcome) -> None:
        self._plot.clear()
        record = outcome.record
        embedding = record.embedding_array
        self._plot.scatter(
            embedding[:, 0], embedding[:, 1], label="embedded rows", meta=_embedding_meta(int(embedding.shape[0]))
        )
        self._plot.add_legend()
        self._plot.set_labels(
            x="UMAP1",
            y="UMAP2",
            title=f"UMAP — trustworthiness {outcome.score:.2f} · a={record.a:.3f} b={record.b:.3f}",
        )
        self._table.bind_frame(record.as_frame().reset_index().rename(columns={"index": "row"}))
        self._report.setText(
            f"{record!s} · trustworthiness {outcome.score:.3f} (Venna-Kaski, k≤10) · "
            f"source: {outcome.source} · embedded in {outcome.seconds:.1f}s"
        )

    # -- curation: the view algebra (synchronous -- sub-millisecond stage chains) -------------
    def _ensure_view(self) -> Any:
        if self._frame is None:
            self._report.setText("Load a dataset first.")
            return None
        if self._view is None:
            from pharmaml.ml.curation import SampleSet

            self._view = SampleSet.from_frame(self._frame).view()
        return self._view

    def _render_view(self, view: Any) -> None:
        frame = view.to_frame()
        self._table.bind_frame(frame.reset_index().rename(columns={"index": "row"}))
        self._report.setText(f"view: {view.summary()} · {len(frame)} rows")

    def _apply_filter(self) -> None:
        view = self._ensure_view()
        if view is None:
            return
        from pharmaml.exceptions import PharmaMLError

        field = self._cur_field.currentText()
        op = self._cur_op.currentText()
        raw = self._cur_value.text().strip()
        try:
            if op == "in":
                value: Any = [_maybe_number(piece) for piece in raw.split(",") if piece.strip()]
                if not value:
                    self._report.setText("The 'in' stage needs a comma-separated value list.")
                    return
            elif op == "between":
                pieces = [piece.strip() for piece in raw.split(",")]
                if len(pieces) != 2:
                    self._report.setText("The 'between' stage needs 'lo,hi'.")
                    return
                value = (_maybe_number(pieces[0]), _maybe_number(pieces[1]))
            else:
                if not raw:
                    self._report.setText("The filter stage needs a value.")
                    return
                value = _maybe_number(raw)
            self._view = view.filter(field, op, value)
        except PharmaMLError as exc:
            self._report.setText(f"Stage refused — {str(exc).splitlines()[0]}")
            return
        except ValueError as exc:
            self._report.setText(f"Bad value — {exc}")
            return
        self._render_view(self._view)

    def _apply_sort(self) -> None:
        view = self._ensure_view()
        if view is None:
            return
        target = self._target_combo.currentText()
        self._view = view.sort_by(target, reverse=True)
        self._render_view(self._view)

    def _apply_limit(self) -> None:
        view = self._ensure_view()
        if view is None:
            return
        self._view = view.limit(5)
        self._render_view(self._view)

    def _reset_view(self) -> None:
        if self._frame is None:
            self._report.setText("Load a dataset first.")
            return
        self._view = None
        view = self._ensure_view()
        assert view is not None
        self._render_view(view)

    # -- curation: the audits (worker jobs -- the cosine matrices are the heavy side) -----------
    def _submit_audit(self, name: str, work, render) -> None:
        if self._busy_check():
            return
        if self._frame is None:
            self._report.setText("Load a dataset first.")
            return
        frame = self._frame

        def job(progress) -> AuditOutcome:
            started = _time.monotonic()
            progress(0.5, name)
            out = work(frame)
            progress(1.0, "done")
            out.seconds = _time.monotonic() - started
            return out

        self._runner.submit(
            name,
            job,
            on_finished=render,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _start_duplicates(self) -> None:
        def work(frame: pd.DataFrame) -> AuditOutcome:
            from pharmaml.ml.curation import SampleSet, find_duplicates

            report = find_duplicates(SampleSet.from_frame(frame).view())
            rows = [
                {
                    "group": index + 1,
                    "rows": len(group),
                    "identical": ", ".join(f"#{sid}" for sid in group),
                }
                for index, group in enumerate(report.groups)
            ]
            if not rows:
                rows = [
                    {"group": 0, "rows": 0, "identical": "no duplicate rows — every row is unique"}
                ]
            return AuditOutcome(table=pd.DataFrame(rows), summary=report.summary())

        self._submit_audit("find_duplicates", work, self._render_audit)

    def _start_outliers(self) -> None:
        def work(frame: pd.DataFrame) -> AuditOutcome:
            from pharmaml.ml.curation import SampleSet, find_outliers

            report = find_outliers(
                SampleSet.from_frame(frame).view(), method="zscore", threshold=4.0
            )
            rows = [
                {
                    "field": field,
                    "flagged": len(ids),
                    "sample ids": ", ".join(f"#{sid}" for sid in ids[:10]),
                }
                for field, ids in report.flagged.items()
            ]
            if not rows:
                rows = [
                    {
                        "field": "—",
                        "flagged": 0,
                        "sample ids": "no z-score outliers at threshold 4.0",
                    }
                ]
            return AuditOutcome(table=pd.DataFrame(rows), summary=report.summary())

        self._submit_audit("find_outliers (z-score > 4)", work, self._render_audit)

    def _start_uniqueness(self) -> None:
        def work(frame: pd.DataFrame) -> AuditOutcome:
            from pharmaml.ml.curation import SampleSet, uniqueness

            scores = uniqueness(SampleSet.from_frame(frame).view(), k=5)
            ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:20]
            table = pd.DataFrame(ranked, columns=["sample id", "uniqueness"])
            head = ", ".join(f"#{sid} {score:.3f}" for sid, score in ranked[:3])
            return AuditOutcome(
                table=table,
                summary=f"{len(scores)} samples scored (1 - mean cosine similarity to the k=5 nearest) · top: {head}",
            )

        self._submit_audit("uniqueness (k=5)", work, self._render_audit)

    def _start_similarity(self) -> None:
        row = self._sim_row.value()

        def work(frame: pd.DataFrame) -> AuditOutcome:
            from pharmaml.ml.curation import SampleSet, sort_by_similarity

            hits = sort_by_similarity(SampleSet.from_frame(frame).view(), row, k=5)
            table = pd.DataFrame(hits, columns=["sample id", "cosine similarity"])
            others = [(sid, score) for sid, score in hits if sid != row][:3]
            head = ", ".join(f"#{sid} {score:.3f}" for sid, score in others)
            return AuditOutcome(
                table=table,
                summary=f"sort_by_similarity(#{row}): {head}",
            )

        self._submit_audit(f"sort_by_similarity (row {row})", work, self._render_audit)

    def _render_audit(self, outcome: AuditOutcome) -> None:
        self._plot.clear()
        self._table.bind_frame(outcome.table)
        self._report.setText(f"{outcome.summary} · audited in {outcome.seconds:.1f}s")

    # -- shared finishers --------------------------------------------------------------
    def _failed(self, message: str) -> None:
        self._report.setText(f"Run failed — {message.splitlines()[0]}")

    def _cancelled(self, seconds: float) -> None:
        self._report.setText(f"Run cancelled after {seconds:.1f}s (cooperative stop).")

    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    # -- test accessors ------------------------------------------------------------
    def report_text(self) -> str:
        return self._report.text()
