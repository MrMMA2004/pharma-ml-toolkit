"""Data Explorer: real datasets in, tables + distributions out (16.12/16.8).

Loads any CSV/TSV/Excel table through ``pharmaml.utils.io.read_table``
-- the exact reader the playground's cards use (same Excel guarantee,
v1.2.116+) -- renders it via the model/view table, and plots a real
histogram of the selected numeric column with honest per-column stats
(dtype, missing count, mean/range for numerics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView
from ..widgets.plot_widget import PlotPanel
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[2]


class DataExplorerPage(QWidget):
    """Open real tables, inspect them, plot a column's distribution."""

    def __init__(self, parent: QWidget | None = None, theme: ThemeManager | None = None,
                 runner: JobRunner | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._audit_generation = 0
        self._frame: pd.DataFrame | None = None
        self._source: str = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self._dataset_combo = QComboBox()
        self._dataset_combo.setMinimumWidth(280)
        self._dataset_combo.currentTextChanged.connect(self._load_builtin)
        self._open_btn = QPushButton("Open CSV / Excel…")
        self._open_btn.clicked.connect(self._open_dialog)
        self._copy_btn = QPushButton("Copy table")
        self._copy_btn.clicked.connect(self._copy)
        self._status = QLabel("No dataset loaded")
        self._status.setObjectName("studioStatus")
        top.addWidget(QLabel("Dataset:"))
        top.addWidget(self._dataset_combo, stretch=1)
        top.addWidget(self._open_btn)
        top.addWidget(self._copy_btn)
        top.addWidget(self._status)
        lay.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self._column_combo = QComboBox()
        self._column_combo.currentIndexChanged.connect(self._plot_column)
        right_lay.addWidget(QLabel("Column distribution:"))
        right_lay.addWidget(self._column_combo)
        self._plot = PlotPanel(title="Column histogram")
        right_lay.addWidget(self._plot, stretch=1)
        self._stats_label = QLabel("")
        self._stats_label.setObjectName("studioStatus")
        self._stats_label.setWordWrap(True)
        right_lay.addWidget(self._stats_label)
        right_lay.addWidget(QLabel("Data quality / exact split audit:"))
        audit_options = QHBoxLayout()
        self._audit_target = QComboBox()
        self._audit_key = QComboBox()
        self._audit_split = QComboBox()
        for title, combo in (("Class", self._audit_target), ("Key", self._audit_key),
                             ("Split", self._audit_split)):
            audit_options.addWidget(QLabel(title))
            audit_options.addWidget(combo)
        right_lay.addLayout(audit_options)
        self._audit_button = QPushButton("Run data audit")
        self._audit_button.clicked.connect(self._run_audit)
        right_lay.addWidget(self._audit_button)
        self._audit_output = QPlainTextEdit()
        self._audit_output.setReadOnly(True)
        self._audit_output.setPlaceholderText("Select optional columns, then run the audit.")
        right_lay.addWidget(self._audit_output)
        splitter.addWidget(right)
        splitter.setSizes([640, 420])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._refresh_builtin_list()

    # -- loading (real data only) -----------------------------------------------
    def _refresh_builtin_list(self) -> None:
        root = _TK / "examples" / "data"
        names = ["", *(p.name for p in sorted(root.glob("*.csv")) if p.is_file())]
        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        self._dataset_combo.addItems(names)
        self._dataset_combo.blockSignals(False)

    def _load_builtin(self, name: str) -> None:
        if not name:
            return
        path = _TK / "examples" / "data" / name
        if path.exists():
            self.load_path(path)

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV / Excel table",
            str(_TK / "examples" / "data"),
            "Tables (*.csv *.tsv *.xlsx *.xls);;All files (*)",
        )
        if path:
            self.load_path(Path(path))

    def load_path(self, path: Path) -> None:
        """Read through pharmaml's own reader (CSV/TSV/XLSX) and render."""
        from pharmaml.utils.io import read_table

        try:
            frame = read_table(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self._status.setText(f"Could not read {path.name}: {exc}")
            self._frame = None
            self._source = ""
            self._reset_audit([])
            self._table.bind_frame(None, note=f"Could not read {path.name}")
            return
        self._frame = frame
        self._reset_audit([str(c) for c in frame.columns])
        self._source = str(path)
        self._table.bind_frame(frame, note="No dataset loaded")
        self._status.setText(f"{path.name} — {self._table.statusHint}")
        self._column_combo.blockSignals(True)
        self._column_combo.clear()
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        self._column_combo.addItems([str(c) for c in numeric])
        self._column_combo.blockSignals(False)
        if numeric:
            self._plot_column(0)

    def _reset_audit(self, columns: list[str]) -> None:
        self._audit_generation += 1
        self._audit_enabled(True)
        for combo in (self._audit_target, self._audit_key, self._audit_split):
            combo.clear()
            combo.addItem("(none)", None)
            for column in columns:
                combo.addItem(column, column)
        self._audit_output.clear()

    def _run_audit(self) -> None:
        from pharmaml.ml.research_audit import audit_dataset

        if self._frame is None:
            self._audit_output.setPlainText("Load a dataset first.")
            return
        generation = self._audit_generation
        frame = self._frame
        target = self._audit_target.currentData()
        key = self._audit_key.currentData()
        partition = self._audit_split.currentData()
        self._audit_output.setPlainText("Running audit…")
        self._audit_enabled(False)
        worker = self._runner.submit(
            "Data quality audit",
            lambda _progress: audit_dataset(frame, target=target, key=key, partition=partition),
            on_finished=lambda report: self._audit_finished(report, generation),
            on_failed=lambda message: self._audit_failed(message, generation),
        )
        if worker is None:
            self._audit_enabled(True)
            self._audit_output.setPlainText("Another Studio job is running; try again when it finishes.")

    def _audit_enabled(self, enabled: bool) -> None:
        for widget in (self._audit_target, self._audit_key, self._audit_split, self._audit_button):
            widget.setEnabled(enabled)

    def _audit_failed(self, message: str, generation: int) -> None:
        if generation == self._audit_generation:
            self._audit_enabled(True)
            self._audit_output.setPlainText(f"Audit failed: {message}")

    def _audit_finished(self, report, generation: int) -> None:
        if generation != self._audit_generation:
            return
        self._audit_enabled(True)
        missing = ", ".join(f"{name}: {count}" for name, count in report.missing_by_column.items())
        lines = [f"Rows: {report.rows}; columns: {report.columns}",
                 f"Missing cells by column: {missing}",
                 f"Duplicate rows (extra occurrences): {report.duplicate_rows}"]
        if report.duplicate_keys is not None:
            lines.append(f"Duplicate keys (extra occurrences): {report.duplicate_keys}; "
                         f"missing or blank keys: {report.missing_keys}")
        if report.class_counts is not None:
            lines.append("Class counts: " + ", ".join(f"{k}: {v}" for k, v in report.class_counts))
        if report.shared_keys is not None:
            lines.extend([f"Rows by partition: {report.partition_rows}",
                          (f"Shared exact keys: {report.shared_keys}; "
                          f"rows with cross-partition overlap: {report.overlap_rows}")])
        else:
            lines.append("Split overlap: not audited.")
        lines.append(report.limitations)
        self._audit_output.setPlainText("\n".join(lines))

    # -- plotting -------------------------------------------------------------------
    def _plot_column(self, index: int) -> None:
        if self._frame is None or index < 0:
            return
        name = self._column_combo.currentText()
        if not name or name not in self._frame.columns:
            return
        series = self._frame[name].dropna()
        if series.empty:
            return
        values = series.to_numpy(dtype=float, copy=False)
        self._plot.clear()
        self._plot.histogram(values, bins=min(40, max(10, int(np.sqrt(len(values))))))
        self._plot.set_labels(x=str(name), y="count", title=f"Distribution of {name}")
        missing = int(self._frame[name].isna().sum())
        stats = (
            f"rows {len(series):,} · missing {missing:,} · "
            f"mean {np.mean(values):.4g} · min {np.min(values):.4g} · max {np.max(values):.4g}"
        )
        self._stats_label.setText(stats)

    def _copy(self) -> None:
        text = self._table.copy_selection()
        if text:
            self._status.setText("Table copied to clipboard (TSV)")

    # -- theming ---------------------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    def frame(self) -> pd.DataFrame | None:
        return self._frame

    def source(self) -> str:
        return self._source

    def active_column(self) -> str:
        """The column the explorer currently plots (the §11 target pick)."""
        return self._column_combo.currentText()
