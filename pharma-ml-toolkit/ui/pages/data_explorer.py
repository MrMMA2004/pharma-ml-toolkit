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
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..font_loader import set_mono
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
        self._plot.set_empty_hint("Open CSV / Excel… and pick a numeric column to see its distribution")

        right_lay.addWidget(self._plot, stretch=1)
        # (R30-c) the per-page chart export under the live palette
        from ..widgets.chart_export import attach_chart_export

        _export_row = QHBoxLayout()
        attach_chart_export(
            self, self._plot, "explorer-histogram",
            layout=_export_row, status_label=self._status,
        )
        _export_row.addStretch(1)
        right_lay.addLayout(_export_row)
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
        set_mono(self._audit_output)  # numerical audit log
        right_lay.addWidget(self._audit_output)

        # part6 Phase 3, batch 4: the exact split-overlap audit panel
        right_lay.addWidget(QLabel("Exact split-overlap audit (ml.split_audit):"))
        self._split_audit_column = QComboBox()
        self._split_audit_column.setToolTip(
            "The identifier column to audit across the partitions below "
            "(e.g. compound_id or SMILES). Refreshed when a dataset loads."
        )
        right_lay.addWidget(QLabel("id column:"))
        right_lay.addWidget(self._split_audit_column)
        self._split_audit_ranges = QPlainTextEdit()
        set_mono(self._split_audit_ranges)  # ranges are numeric literals
        self._split_audit_ranges.setPlaceholderText(
            "train: 0-251\ncalibration: 252-335\ntest: 336-419\n\n"
            "(one 'name: lo-hi' row-range per partition — indices are "
            "0-based, hi exclusive; overlapping ranges are exactly what "
            "this audit is for)"
        )
        self._split_audit_ranges.setMaximumHeight(88)
        right_lay.addWidget(QLabel("partitions (row ranges):"))
        right_lay.addWidget(self._split_audit_ranges)
        self._split_audit_btn = QPushButton("Audit split overlap")
        self._split_audit_btn.clicked.connect(self._run_split_audit)
        self._split_audit_btn.setToolTip(
            "Exact identifier overlap across the named partitions: shared "
            "keys, per-partition row counts, duplicates and the pairwise "
            "overlap matrix. No SMILES canonicalization or fuzzy matching "
            "— exact keys only."
        )
        right_lay.addWidget(self._split_audit_btn)
        self._split_audit_output = QPlainTextEdit()
        set_mono(self._split_audit_output)  # split ranges, numerical log
        self._split_audit_output.setReadOnly(True)
        self._split_audit_output.setPlaceholderText(
            "Name the id column and the row ranges, then audit."
        )
        right_lay.addWidget(self._split_audit_output)
        self._split_audit_frame: pd.DataFrame | None = None

        # part6 Phase 3, batch 4: the SAR matched-pairs panel (ml.sar)
        right_lay.addWidget(QLabel("SAR matched pairs & activity cliffs (ml.sar):"))
        sar_cols = QHBoxLayout()
        self._sar_smiles = QComboBox()
        self._sar_smiles.setToolTip("The SMILES column to fragment.")
        self._sar_activity = QComboBox()
        self._sar_activity.setToolTip(
            "The numeric activity/property column — cliffs are pairs whose "
            "delta clears the threshold (the default 1.0 = one pIC50 log "
            "unit)."
        )
        sar_cols.addWidget(QLabel("smiles:"))
        sar_cols.addWidget(self._sar_smiles, stretch=1)
        sar_cols.addWidget(QLabel("activity:"))
        sar_cols.addWidget(self._sar_activity, stretch=1)
        right_lay.addLayout(sar_cols)
        self._sar_min_delta = QDoubleSpinBox()
        self._sar_min_delta.setRange(0.01, 100.0)
        self._sar_min_delta.setDecimals(2)
        self._sar_min_delta.setSingleStep(0.25)
        self._sar_min_delta.setValue(1.0)
        self._sar_min_delta.setToolTip(
            "The cliff threshold on |activity delta| (1.0 = one log unit)."
        )
        right_lay.addWidget(QLabel("cliff threshold (|delta| >=):"))
        right_lay.addWidget(self._sar_min_delta)
        self._sar_btn = QPushButton("Find matched pairs & cliffs")
        self._sar_btn.clicked.connect(self._run_sar)
        self._sar_btn.setToolTip(
            "Single-cut fragment every molecule, group by shared context: "
            "all matched molecular pairs (one substituent swap, delta "
            "always >= 0) and the activity cliffs among them."
        )
        right_lay.addWidget(self._sar_btn)
        self._sar_output = QPlainTextEdit()
        set_mono(self._sar_output)  # SAR rule log
        self._sar_output.setReadOnly(True)
        self._sar_output.setPlaceholderText(
            "Pick a SMILES column and a numeric activity column, then find "
            "the matched pairs."
        )
        self._sar_output.setMaximumHeight(150)
        right_lay.addWidget(self._sar_output)
        self._sar_frame: pd.DataFrame | None = None
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
        # part6 §8 (Phase 4): a successful load enters the shared
        # workspace's dataset registry (best-effort side record).
        try:
            from pharmaml.studio import note as note_workspace

            note_workspace("dataset", path, note=f"{len(frame)} rows")
        except Exception:  # noqa: BLE001 - the registry must never block a load
            pass
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
        # the split-overlap panel's id column rides the same refresh
        current = self._split_audit_column.currentText()
        self._split_audit_column.clear()
        self._split_audit_column.addItems(columns)
        if current:
            index = self._split_audit_column.findText(current)
            if index >= 0:
                self._split_audit_column.setCurrentIndex(index)
        self._split_audit_output.clear()
        # the SAR panel's smiles/activity columns too
        smiles_current = self._sar_smiles.currentText()
        self._sar_smiles.clear()
        self._sar_smiles.addItems(columns)
        activity_current = self._sar_activity.currentText()
        self._sar_activity.clear()
        self._sar_activity.addItems(
            [c for c in columns if pd.api.types.is_numeric_dtype(self._frame[c])]
            if self._frame is not None else columns
        )
        for combo, text in (
            (self._sar_smiles, smiles_current),
            (self._sar_activity, activity_current),
        ):
            if text:
                index = combo.findText(text)
                if index >= 0:
                    combo.setCurrentIndex(index)
        self._sar_output.clear()

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

    # -- exact split-overlap audit (part6 Phase 3, batch 4) -----------------------
    def _parse_partition_ranges(self, text: str, n_rows: int) -> dict[str, list]:
        """Parse 'name: lo-hi' lines into {name: [keys]} over the frame."""
        partitions: dict[str, list] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"missing ':' in {line!r}")
            name, _, span = line.partition(":")
            name = name.strip()
            if not name:
                raise ValueError(f"empty partition name in {line!r}")
            if name in partitions:
                raise ValueError(f"duplicate partition name {name!r}")
            if "-" not in span:
                raise ValueError(f"missing '-' range in {line!r}")
            lo_text, _, hi_text = span.partition("-")
            try:
                lo, hi = int(lo_text.strip()), int(hi_text.strip())
            except ValueError as exc:
                raise ValueError(f"non-integer range in {line!r}") from exc
            if lo < 0 or hi <= lo or lo >= n_rows:
                raise ValueError(f"range {lo}-{hi} outside the table (0..{n_rows - 1})")
            hi = min(hi, n_rows)
            partitions[name] = list(range(lo, hi))
        if len(partitions) < 2:
            raise ValueError("name at least two partitions (e.g. train + test)")
        return partitions

    def _run_split_audit(self) -> None:
        from pharmaml.ml.split_audit import audit_split_overlap

        if self._frame is None:
            self._split_audit_output.setPlainText("Load a dataset first.")
            return
        column = self._split_audit_column.currentText()
        if not column or column not in self._frame.columns:
            self._split_audit_output.setPlainText("Pick the identifier column to audit.")
            return
        frame = self._frame
        try:
            partitions = self._parse_partition_ranges(
                self._split_audit_ranges.toPlainText(), len(frame)
            )
        except ValueError as exc:
            self._split_audit_output.setPlainText(f"Check the partitions — {exc}")
            return
        values = frame[column].tolist()
        keys = {name: [values[i] for i in rows] for name, rows in partitions.items()}
        self._split_audit_output.setPlainText("Auditing split overlap…")
        worker = self._runner.submit(
            "Exact split-overlap audit",
            lambda _progress: audit_split_overlap(keys),
            on_finished=lambda report: self._split_audit_finished(report),
            on_failed=lambda message: self._split_audit_output.setPlainText(
                f"Audit failed: {message}"
            ),
        )
        if worker is None:
            self._split_audit_output.setPlainText(
                "Another Studio job is running; try again when it finishes."
            )

    def _split_audit_finished(self, report) -> None:
        lines = [f"Partitions: " + ", ".join(
            f"{name}={rows:,}" for name, rows in zip(report.partition_names, report.partition_rows)
        )]
        lines.append(
            "Duplicates within a partition (extra occurrences): "
            + ", ".join(
                f"{name}={extra:,}"
                for name, extra in zip(report.partition_names, report.duplicate_rows)
            )
        )
        if report.has_overlap:
            lines.append(
                f"CROSS-PARTITION OVERLAP: {report.n_shared_keys:,} shared key(s), "
                f"{sum(report.overlap_rows):,} row(s) involved"
            )
            pairwise = report.pairwise_dataframe()
            lines.append("Pairwise overlap matrix:")
            lines.append(pairwise.to_string())
            lines.append("Shared keys:")
            lines.append(report.to_dataframe().head(50).to_string(index=False))
        else:
            lines.append("No cross-partition overlap — the splits are disjoint.")
        lines.append(
            "(exact-key audit: no SMILES canonicalization, no fuzzy matching)"
        )
        self._split_audit_output.setPlainText("\n".join(lines))
        pairwise = report.pairwise_dataframe()
        self._split_audit_frame = pairwise
        self._plot.clear()
        names = list(report.partition_names)
        counts = [float(pairwise.iloc[i].sum() - pairwise.iloc[i, i]) for i in range(len(names))]
        self._plot.bars(np.asarray(counts), labels=names)
        self._plot.set_labels(
            x="partition", y="shared keys", title="Exact split-overlap audit"
        )

    # -- SAR matched pairs & activity cliffs (part6 Phase 3, batch 4) --------------
    def _run_sar(self) -> None:
        from pharmaml.ml.sar import activity_cliffs, matched_pairs

        if self._frame is None:
            self._sar_output.setPlainText("Load a dataset first.")
            return
        smiles_col = self._sar_smiles.currentText()
        activity_col = self._sar_activity.currentText()
        if not smiles_col or smiles_col not in self._frame.columns:
            self._sar_output.setPlainText("Pick the SMILES column.")
            return
        if not activity_col or activity_col not in self._frame.columns:
            self._sar_output.setPlainText("Pick the numeric activity column.")
            return
        if not pd.api.types.is_numeric_dtype(self._frame[activity_col]):
            self._sar_output.setPlainText(
                f"{activity_col!r} is not numeric — the activity column must "
                "be numeric (cliffs are a delta on it)."
            )
            return
        frame = self._frame
        min_delta = float(self._sar_min_delta.value())
        self._sar_output.setPlainText("Fragmenting and matching…")
        worker = self._runner.submit(
            "SAR matched pairs",
            lambda _progress: (
                matched_pairs(frame, smiles_col, activity_col),
                activity_cliffs(frame, smiles_col, activity_col, min_delta=min_delta),
            ),
            on_finished=lambda result: self._sar_finished(result, min_delta),
            on_failed=lambda message: self._sar_output.setPlainText(
                f"Matched-pairs run failed: {message}"
            ),
        )
        if worker is None:
            self._sar_output.setPlainText(
                "Another Studio job is running; try again when it finishes."
            )

    def _sar_finished(self, result, min_delta: float) -> None:
        pairs, cliffs = result
        self._sar_frame = pairs
        lines = [cliffs.summary()]
        lines.append(
            f"Columns: {', '.join(str(c) for c in pairs.columns)}"
        )
        if pairs.empty:
            lines.append(
                "No matched pairs — no two molecules share a fragmentation "
                "context (try a larger or more homogeneous table)."
            )
        else:
            lines.append("Top 10 pairs by activity delta:")
            top = pairs.sort_values("delta", ascending=False).head(10)
            lines.append(top.to_string(index=False))
            lines.append(
                "(every pair is one substituent swap; delta >= 0 because the "
                "lower-activity side is listed first)"
            )
        self._sar_output.setPlainText("\n".join(lines))
        self._plot.clear()
        if not pairs.empty:
            deltas = np.asarray(pairs["delta"], dtype=float)
            counts, edges = np.histogram(deltas, bins=min(20, max(5, len(deltas) // 8)))
            centers = (edges[:-1] + edges[1:]) / 2.0
            self._plot.bars(counts.astype(float), labels=[
                f"{c:.2f}" for c in centers
            ])
            self._plot.set_labels(
                x="activity delta", y="pairs",
                title=f"Matched-pair deltas (cliffs at |delta| >= {min_delta:g})",
            )

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
