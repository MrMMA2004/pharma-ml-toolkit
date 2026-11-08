"""Unsupervised Workbench: PCA, clustering, density, manifolds (audit row 7).

The Studio's increment-6 page (the playground Unsupervised tab). The
PDSH handbook's unsupervised roster -- 05.09 PCA, 05.10 manifolds,
05.11 k-means, 05.12 GMMs, 05.13 KDE, plus the agglomerative tree --
each wired to the SAME public backend APIs the playground cards call,
over a shared dataset + column selection:

- **PCA** -- ``pca_reduce`` on the checked columns (auto n_components
  = min(10, n, p)): the explained-variance ladder as bars with the
  cumulative dash line (the honest "how many components are enough?"
  figure), the loadings table (what PC1 actually means -- on the
  bundled descriptor library PC1 is dominated by NumHAcceptors), and
  the cumulative share at k printed in the report.
- **Clustering** -- ``kmeans_cluster`` (Lloyd / mini-batch, manual k
  or the silhouette scan), ``gmm_cluster`` (the BIC/AIC sweep or
  manual components, every covariance type, the soft responsibilities
  landing in the table) and ``hierarchical_cluster`` (every linkage,
  the merge heights in the report). The scatter shows the first two
  checked columns coloured by cluster membership.
- **Density (KDE)** -- ``kde_fit`` on one numeric column with the
  CV-chosen (or manual) bandwidth: the density curve over the
  histogram twin (scaled by n·bin-width, the matplotlib convention),
  the 200-point curve table, bandwidth + log-likelihood in the report.
- **Manifold** -- ``manifold_embedding`` for isomap / LLE (standard,
  modified, ltsa) / MDS / spectral / t-SNE: the 2-D embedding scatter
  and coordinate table, the method's own knobs echoed honestly.

Incomplete rows are dropped up-front with the count reported (the
engines' completeness rule made visible). Every heavy path runs in a
:class:`~ui.workers.job_worker.JobWorker`; they are single-shot engine
calls without checkpoint loops, so the cooperative Stop only lands for
checkpoint-wired jobs -- the same disclosure the PK/PD, Biology and
Docking pages carry.
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
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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

#: The page's datasets: the descriptor library (77 rows x 8 descriptor
#: columns -- the handbook's own "descriptor table" story) and the QSAR
#: regression table (260 rows x 21 numeric columns).
_DATASETS = ("library_with_descriptors.csv", "qsar_regression.csv")
_DATA_EXPLORER_ENTRY = "(table from Data Explorer)"

_CLUSTER_METHODS: list[tuple[str, str]] = [
    ("kmeans", "k-means (Lloyd)"),
    ("mini_batch", "mini-batch k-means"),
    ("gmm_sweep", "Gaussian mixture (BIC sweep)"),
    ("gmm_manual", "Gaussian mixture (manual k)"),
    ("ward", "hierarchical (ward)"),
    ("single", "hierarchical (single)"),
    ("average", "hierarchical (average)"),
    ("complete", "hierarchical (complete)"),
]


@dataclass
class PcaOutcome:
    """What the PCA job hands back (the engine record + the echo)."""

    record: Any = None
    columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_dropped: int = 0
    seconds: float = 0.0


@dataclass
class ClusterOutcome:
    """What the clustering job hands back (normalised across engines)."""

    engine: str = ""
    method_label: str = ""
    k: int = 0
    n_rows: int = 0
    n_dropped: int = 0
    sizes: list[int] = field(default_factory=list)
    labels: np.ndarray | None = None
    inertia: float = float("nan")
    silhouette: float | None = None
    chosen_by: str = ""
    converged: bool = False
    log_likelihood: float = float("nan")
    weights: list[float] = field(default_factory=list)
    merge_heights: list[float] = field(default_factory=list)
    centers: pd.DataFrame = field(default_factory=pd.DataFrame)
    probabilities: pd.DataFrame = field(default_factory=pd.DataFrame)
    scan: pd.DataFrame = field(default_factory=pd.DataFrame)
    scatter_x: np.ndarray | None = None
    scatter_y: np.ndarray | None = None
    scatter_names: tuple[str, str] = ("", "")
    seconds: float = 0.0


@dataclass
class KdeOutcome:
    """What the KDE job hands back."""

    record: Any = None
    values: np.ndarray | None = None
    n_dropped: int = 0
    seconds: float = 0.0


@dataclass
class EmbedOutcome:
    """What the manifold job hands back."""

    record: Any = None
    method: str = ""
    columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_dropped: int = 0
    seconds: float = 0.0


class UnsupervisedWorkbenchPage(QWidget):
    """The handbook's unsupervised roster over a shared dataset."""

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

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # -- shared dataset + column selection ---------------------------------
        top = QFormLayout()
        self._dataset_combo = QComboBox()
        for name in _DATASETS:
            self._dataset_combo.addItem(f"{name} (bundled)")
        self._dataset_combo.addItem(_DATA_EXPLORER_ENTRY)
        self._dataset_combo.currentTextChanged.connect(self._load_dataset)
        top.addRow("dataset:", self._dataset_combo)

        columns_row = QWidget()
        columns_lay = QHBoxLayout(columns_row)
        columns_lay.setContentsMargins(0, 0, 0, 0)
        self._column_list = QListWidget()
        self._column_list.setMaximumHeight(120)
        columns_lay.addWidget(self._column_list, stretch=1)
        toggle_col = QVBoxLayout()
        self._all_btn = QPushButton("All")
        self._all_btn.clicked.connect(lambda: self._set_all_columns(True))
        self._none_btn = QPushButton("None")
        self._none_btn.clicked.connect(lambda: self._set_all_columns(False))
        toggle_col.addWidget(self._all_btn)
        toggle_col.addWidget(self._none_btn)
        toggle_col.addStretch(1)
        columns_lay.addLayout(toggle_col)
        top.addRow("columns:", columns_row)
        lay.addLayout(top)

        # -- the four engine tabs ------------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_pca_tab(), "PCA")
        self._tabs.addTab(self._build_cluster_tab(), "Clustering")
        self._tabs.addTab(self._build_kde_tab(), "Density (KDE)")
        self._tabs.addTab(self._build_manifold_tab(), "Manifold")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Unsupervised panel")
        splitter.addWidget(self._plot)
        splitter.setSizes([560, 540])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No unsupervised run yet — pick a dataset and fit PCA.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

        self._load_dataset(self._dataset_combo.currentText())

    # -- tab construction -----------------------------------------------------
    def _build_pca_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._pca_components = QSpinBox()
        self._pca_components.setRange(0, 10)
        self._pca_components.setValue(0)
        self._pca_components.setToolTip("0 = auto: min(10, n_samples, n_features)")
        form.addRow("n components (0 = auto):", self._pca_components)
        self._pca_btn = QPushButton("Fit PCA")
        self._pca_btn.setMinimumHeight(34)
        self._pca_btn.clicked.connect(self._start_pca)
        form.addRow("", self._pca_btn)
        hint = QLabel(
            "pca_reduce on the checked columns (standardised -- descriptors "
            "live on wildly different scales): the explained-variance ladder "
            "with the cumulative line, the loadings table, and the dominant "
            "feature of PC1 in the report."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_cluster_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._cluster_method = QComboBox()
        for _key, label in _CLUSTER_METHODS:
            self._cluster_method.addItem(label)
        form.addRow("method:", self._cluster_method)
        self._cluster_k = QSpinBox()
        self._cluster_k.setRange(0, 20)
        self._cluster_k.setValue(3)
        self._cluster_k.setToolTip("0 = auto: silhouette scan (k-means) or the BIC sweep (GMM)")
        form.addRow("k / components (0 = auto):", self._cluster_k)
        self._gmm_covariance = QComboBox()
        for covariance in ("full", "tied", "diag", "spherical"):
            self._gmm_covariance.addItem(covariance)
        form.addRow("GMM covariance:", self._gmm_covariance)
        self._cluster_btn = QPushButton("Cluster")
        self._cluster_btn.setMinimumHeight(34)
        self._cluster_btn.clicked.connect(self._start_cluster)
        form.addRow("", self._cluster_btn)
        hint = QLabel(
            "kmeans_cluster (Lloyd / mini-batch, manual k or the silhouette "
            "scan), gmm_cluster (BIC sweep or manual, every covariance type, "
            "the soft responsibilities in the table) and hierarchical_cluster "
            "(every linkage, merge heights in the report). The scatter shows "
            "the first two checked columns coloured by cluster."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_kde_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._kde_column = QComboBox()
        form.addRow("column:", self._kde_column)
        self._kde_bandwidth = QLineEdit()
        self._kde_bandwidth.setPlaceholderText("auto (cross-validated)")
        form.addRow("bandwidth (blank = auto):", self._kde_bandwidth)
        self._kde_kernel = QComboBox()
        for kernel in ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"):
            self._kde_kernel.addItem(kernel)
        form.addRow("kernel:", self._kde_kernel)
        self._kde_btn = QPushButton("Fit KDE")
        self._kde_btn.setMinimumHeight(34)
        self._kde_btn.clicked.connect(self._start_kde)
        form.addRow("", self._kde_btn)
        hint = QLabel(
            "kde_fit on ONE numeric column: the CV-chosen (or manual) "
            "bandwidth, the density curve over the histogram twin (scaled "
            "by n·bin-width), the 200-point curve table, and the mean "
            "log-likelihood in the report."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_manifold_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._manifold_method = QComboBox()
        for method in ("isomap", "lle", "mds", "spectral", "tsne"):
            self._manifold_method.addItem(method)
        form.addRow("method:", self._manifold_method)
        self._manifold_neighbors = QSpinBox()
        self._manifold_neighbors.setRange(2, 100)
        self._manifold_neighbors.setValue(10)
        form.addRow("n neighbors (isomap/lle/spectral):", self._manifold_neighbors)
        self._lle_method = QComboBox()
        for variant in ("standard", "modified", "ltsa"):
            self._lle_method.addItem(variant)
        form.addRow("LLE variant:", self._lle_method)
        self._tsne_perplexity = QSpinBox()
        self._tsne_perplexity.setRange(2, 100)
        self._tsne_perplexity.setValue(30)
        form.addRow("t-SNE perplexity:", self._tsne_perplexity)
        self._manifold_btn = QPushButton("Embed (2-D)")
        self._manifold_btn.setMinimumHeight(34)
        self._manifold_btn.clicked.connect(self._start_manifold)
        form.addRow("", self._manifold_btn)
        hint = QLabel(
            "manifold_embedding (PDSH 05.10) on the checked columns: the "
            "2-D embedding scatter + coordinate table for isomap, LLE "
            "(standard/modified/ltsa), MDS, spectral and t-SNE -- the "
            "method's own knobs echoed in the report."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- data ------------------------------------------------------------------
    def _load_dataset(self, name: str) -> None:
        self._column_list.clear()
        self._kde_column.clear()
        base = name.replace(" (bundled)", "")
        frame: pd.DataFrame | None
        if base == _DATA_EXPLORER_ENTRY:
            frame = None
            if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
                frame = self._data_explorer.frame()
            if frame is None or frame.empty:
                self._frame = None
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
        numeric = [str(c) for c in frame.select_dtypes(include=[np.number]).columns]
        for column in numeric:
            item = QListWidgetItem(column)
            item.setCheckState(Qt.CheckState.Checked)
            self._column_list.addItem(item)
            self._kde_column.addItem(column)
        if self._kde_column.count():
            self._kde_column.setCurrentText("MolWt" if "MolWt" in numeric else numeric[0])
        self._report.setText(f"{base}: {frame.shape[0]} rows · {len(numeric)} numeric columns checked.")

    def _set_all_columns(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self._column_list.count()):
            self._column_list.item(i).setCheckState(state)

    def _checked_columns(self) -> list[str]:
        columns: list[str] = []
        for i in range(self._column_list.count()):
            item = self._column_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                columns.append(item.text())
        return columns

    # -- job plumbing -----------------------------------------------------------
    def _buttons(self) -> tuple[QPushButton, ...]:
        return (self._pca_btn, self._cluster_btn, self._kde_btn, self._manifold_btn)

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

    def _matrix_or_report(self) -> pd.DataFrame | None:
        """The checked-column matrix, complete rows only (drop count visible)."""
        if self._frame is None:
            self._report.setText("Load a dataset first.")
            return None
        columns = self._checked_columns()
        if not columns:
            self._report.setText("Check at least one numeric column.")
            return None
        matrix = self._frame[columns]
        return matrix

    # -- PCA -----------------------------------------------------------------------
    def _start_pca(self) -> None:
        if self._busy_check():
            return
        matrix = self._matrix_or_report()
        if matrix is None:
            return
        n_components = self._pca_components.value()
        requested = n_components if n_components > 0 else None
        columns = list(matrix.columns)

        def job(progress) -> PcaOutcome:
            from pharmaml.ml.unsupervised import pca_reduce

            started = _time.monotonic()
            progress(0.2, "dropping incomplete rows")
            complete = matrix.dropna()
            n_dropped = int(len(matrix) - len(complete))
            progress(0.5, "fitting PCA")
            record = pca_reduce(complete, requested)
            progress(1.0, "done")
            return PcaOutcome(
                record=record,
                columns=columns,
                n_rows=len(complete),
                n_dropped=n_dropped,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"PCA over {len(columns)} columns",
            job,
            on_finished=self._render_pca,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_pca(self, outcome: PcaOutcome) -> None:
        self._plot.clear()
        record = outcome.record
        ratios = np.asarray(record.explained_variance_ratio, dtype=float)
        labels = [f"PC{i + 1}" for i in range(len(ratios))]
        self._plot.bars(ratios, labels=labels)
        self._plot.line(
            np.arange(len(ratios), dtype=float),
            np.asarray(record.cumulative_variance, dtype=float),
            label="cumulative",
            width=2,
            index=1,
            style="dash",
        )
        self._plot.add_legend()
        self._plot.set_labels(x="component", y="explained variance", title="The variance ladder")
        self._table.bind_frame(record.loadings.reset_index().rename(columns={"index": "feature"}))
        pc1 = record.loadings["PC1"]
        dominant = str(pc1.abs().idxmax())
        dropped = f" · {outcome.n_dropped} incomplete rows dropped" if outcome.n_dropped else ""
        head = " · ".join(f"PC{i + 1} {ratio:.1%}" for i, ratio in enumerate(ratios[:3]))
        self._report.setText(
            f"PCA · {record.n_components} components over {record.n_features} features, "
            f"{outcome.n_rows} rows{dropped} · {head} · cumulative at "
            f"{record.n_components} = {record.total_variance_explained:.1%} · PC1 dominated "
            f"by {dominant} (|loading| {float(pc1.abs().max()):.3f}) · fitted in "
            f"{outcome.seconds:.1f}s"
        )

    # -- clustering -------------------------------------------------------------------
    def _start_cluster(self) -> None:
        if self._busy_check():
            return
        matrix = self._matrix_or_report()
        if matrix is None:
            return
        key = _CLUSTER_METHODS[self._cluster_method.currentIndex()][0]
        label = _CLUSTER_METHODS[self._cluster_method.currentIndex()][1]
        k_value = self._cluster_k.value()
        covariance = self._gmm_covariance.currentText()
        columns = list(matrix.columns)

        def job(progress) -> ClusterOutcome:
            from pharmaml.ml.unsupervised import (
                gmm_cluster,
                hierarchical_cluster,
                kmeans_cluster,
            )

            started = _time.monotonic()
            progress(0.15, "dropping incomplete rows")
            complete = matrix.dropna()
            n_dropped = int(len(matrix) - len(complete))
            out = ClusterOutcome(
                engine=key,
                method_label=label,
                n_rows=len(complete),
                n_dropped=n_dropped,
            )
            if len(columns) >= 2:
                out.scatter_x = complete[columns[0]].to_numpy(dtype=float)
                out.scatter_y = complete[columns[1]].to_numpy(dtype=float)
                out.scatter_names = (columns[0], columns[1])
            progress(0.45, "clustering")
            if key in {"kmeans", "mini_batch"}:
                km = kmeans_cluster(
                    complete, k=k_value if k_value > 0 else None, mini_batch=key == "mini_batch"
                )
                out.k = int(km.k)
                out.sizes = [int(count) for count in km.sizes]
                out.labels = np.asarray(km.labels, dtype=int)
                out.inertia = float(km.inertia)
                out.silhouette = km.silhouette
                out.centers = km.centers
                out.scan = km.scan
            elif key in {"gmm_sweep", "gmm_manual"}:
                mix = gmm_cluster(
                    complete,
                    n_components=k_value if key == "gmm_manual" and k_value > 0 else None,
                    covariance_type=covariance,
                )
                out.k = int(mix.n_components)
                out.sizes = [int(count) for count in pd.Series(mix.labels).value_counts().sort_index()]
                out.labels = np.asarray(mix.labels, dtype=int)
                out.chosen_by = mix.chosen_by
                out.converged = bool(mix.converged)
                out.log_likelihood = float(mix.log_likelihood)
                out.weights = [float(weight) for weight in mix.weights]
                out.centers = mix.means
                out.probabilities = mix.probabilities
                out.scan = mix.scan
            else:
                tree = hierarchical_cluster(complete, k=k_value if k_value > 0 else 3, linkage=key)
                out.k = int(tree.k)
                counts = pd.Series(tree.labels).value_counts().sort_index()
                out.sizes = [int(count) for count in counts]
                out.labels = np.asarray(tree.labels, dtype=int)
                out.inertia = float(tree.inertia)
                out.merge_heights = [float(height) for height in tree.merge_heights]
                out.centers = pd.DataFrame({"cluster": counts.index, "size": counts.to_numpy()})
            progress(1.0, "done")
            out.seconds = _time.monotonic() - started
            return out

        self._runner.submit(
            f"{label} over {len(columns)} columns",
            job,
            on_finished=self._render_cluster,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_cluster(self, outcome: ClusterOutcome) -> None:
        self._plot.clear()
        labels = outcome.labels
        xs = outcome.scatter_x
        ys = outcome.scatter_y
        if labels is not None and xs is not None and ys is not None:
            for cluster in sorted({int(label) for label in labels}):
                mask = labels == cluster
                self._plot.scatter(
                    xs[mask],
                    ys[mask],
                    label=f"cluster {cluster}",
                    index=cluster,
                )
            self._plot.add_legend()
            self._plot.set_labels(
                x=outcome.scatter_names[0],
                y=outcome.scatter_names[1],
                title=f"{outcome.method_label} membership",
            )
        sizes = np.asarray(outcome.sizes, dtype=float)
        self._plot.bars(sizes, labels=[f"c{i}" for i in range(len(sizes))], index=1)
        # GMM -> the soft responsibilities (the honest table, one row per
        # sample); k-means -> the centroids (indexed by cluster);
        # hierarchical -> the cluster sizes frame as built by the job.
        if outcome.engine in {"kmeans", "mini_batch"}:
            table = outcome.centers.reset_index().rename(columns={"index": "cluster"})
        elif outcome.engine in {"gmm_sweep", "gmm_manual"} and not outcome.probabilities.empty:
            table = outcome.probabilities.reset_index().rename(columns={"index": "sample"})
        else:
            table = outcome.centers
        self._table.bind_frame(table)
        dropped = f" · {outcome.n_dropped} incomplete rows dropped" if outcome.n_dropped else ""
        parts = [f"{outcome.method_label} · k={outcome.k} · {outcome.n_rows} rows{dropped}"]
        if outcome.engine in {"kmeans", "mini_batch"}:
            silhouette = "n/a" if outcome.silhouette is None else f"{outcome.silhouette:.3f}"
            parts.append(f"silhouette {silhouette} · inertia {outcome.inertia:.4g}")
            if not outcome.scan.empty:
                parts.append("k by silhouette scan")
        elif outcome.engine in {"gmm_sweep", "gmm_manual"}:
            parts.append(
                f"chosen by {outcome.chosen_by} · {'converged' if outcome.converged else 'NOT converged'} "
                f"· log-likelihood {outcome.log_likelihood:.2f}"
            )
            parts.append("weights " + "/".join(f"{weight:.2f}" for weight in outcome.weights))
        else:
            parts.append(f"inertia {outcome.inertia:.4g} · merge heights {len(outcome.merge_heights)}")
        parts.append("sizes " + "/".join(str(size) for size in outcome.sizes))
        parts.append(f"fitted in {outcome.seconds:.1f}s")
        self._report.setText(" · ".join(parts))

    # -- KDE ----------------------------------------------------------------------------
    def _start_kde(self) -> None:
        if self._busy_check():
            return
        if self._frame is None:
            self._report.setText("Load a dataset first.")
            return
        column = self._kde_column.currentText()
        if not column or column not in self._frame.columns:
            self._report.setText("Pick a numeric column.")
            return
        raw = self._frame[column]
        if not pd.api.types.is_numeric_dtype(raw):
            self._report.setText("The KDE column must be numeric.")
            return
        text = self._kde_bandwidth.text().strip()
        bandwidth: float | None
        try:
            bandwidth = float(text) if text else None
            if bandwidth is not None and bandwidth <= 0:
                self._report.setText("Bandwidth must be positive.")
                return
        except ValueError:
            self._report.setText("Bandwidth must be a number (or blank for auto).")
            return
        kernel = self._kde_kernel.currentText()

        def job(progress) -> KdeOutcome:
            from pharmaml.ml.unsupervised import kde_fit

            started = _time.monotonic()
            progress(0.25, "dropping incomplete rows")
            values = raw.dropna()
            n_dropped = int(len(raw) - len(values))
            progress(0.6, "fitting KDE")
            record = kde_fit(values, bandwidth=bandwidth, kernel=kernel)
            progress(1.0, "done")
            return KdeOutcome(
                record=record,
                values=np.asarray(values, dtype=float),
                n_dropped=n_dropped,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"KDE on {column}",
            job,
            on_finished=self._render_kde,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_kde(self, outcome: KdeOutcome) -> None:
        self._plot.clear()
        record = outcome.record
        grid = np.asarray(record.grid, dtype=float)
        density = np.asarray(record.density, dtype=float)
        values = outcome.values
        if values is not None and len(values):
            self._plot.histogram(values, bins=24, index=0)
        if values is not None and len(grid) > 1:
            width = float(np.median(np.diff(grid)))
            self._plot.line(
                grid,
                density * len(values) * width,
                label=f"density (bw {record.bandwidth:.3g})",
                width=2,
                index=1,
            )
            self._plot.add_legend()
        self._plot.set_labels(
            x=str(record.column),
            y="count / scaled density",
            title=f"KDE · bandwidth {record.bandwidth:.3g} ({record.bandwidth_source})",
        )
        self._table.bind_frame(record.as_frame())
        dropped = f" · {outcome.n_dropped} incomplete rows dropped" if outcome.n_dropped else ""
        self._report.setText(
            f"KDE · column {record.column!r} · n={record.n_samples}{dropped} · bandwidth "
            f"{record.bandwidth:.4g} ({record.bandwidth_source}, kernel {record.kernel}) · "
            f"mean log-likelihood {record.log_likelihood:.3f} · grid of {len(grid)} points · "
            f"fitted in {outcome.seconds:.1f}s"
        )

    # -- manifold --------------------------------------------------------------------------
    def _start_manifold(self) -> None:
        if self._busy_check():
            return
        matrix = self._matrix_or_report()
        if matrix is None:
            return
        method = self._manifold_method.currentText()
        n_neighbors = self._manifold_neighbors.value()
        lle_method = self._lle_method.currentText()
        perplexity = float(self._tsne_perplexity.value())
        columns = list(matrix.columns)

        def job(progress) -> EmbedOutcome:
            from pharmaml.ml.manifold import manifold_embedding

            started = _time.monotonic()
            progress(0.2, "dropping incomplete rows")
            complete = matrix.dropna()
            n_dropped = int(len(matrix) - len(complete))
            progress(0.5, f"embedding with {method}")
            record = manifold_embedding(
                complete,
                method,
                n_neighbors=n_neighbors,
                perplexity=perplexity,
                lle_method=lle_method,
            )
            progress(1.0, "done")
            return EmbedOutcome(
                record=record,
                method=method,
                columns=columns,
                n_rows=len(complete),
                n_dropped=n_dropped,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"{method} embedding over {len(columns)} columns",
            job,
            on_finished=self._render_manifold,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_manifold(self, outcome: EmbedOutcome) -> None:
        self._plot.clear()
        record = outcome.record
        coords = record.coords
        self._plot.scatter(
            coords.iloc[:, 0].to_numpy(dtype=float),
            coords.iloc[:, 1].to_numpy(dtype=float),
            label=outcome.method,
        )
        self._plot.add_legend()
        self._plot.set_labels(
            x=str(coords.columns[0]),
            y=str(coords.columns[1]),
            title=f"{outcome.method} embedding ({record.n_samples} samples)",
        )
        self._table.bind_frame(coords.reset_index().rename(columns={"index": "sample"}))
        dropped = f" · {outcome.n_dropped} incomplete rows dropped" if outcome.n_dropped else ""
        # EmbeddingRecord.__str__ already carries method / rows / features /
        # dimensionality / the method's own knobs -- echo it verbatim.
        self._report.setText(f"{record!s}{dropped} · embedded in {outcome.seconds:.1f}s")

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
