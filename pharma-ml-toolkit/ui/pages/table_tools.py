"""Table Tools — the pandas-reshaping surface (part8 r9, COVERAGE rows 26-33).

The seven documented STRETCH cards from the data-handbook audit, wired
as one page in the Datasets workspace: combine (concat), merge, groupby
summary, pivot, string cleaning, resample/rolling, and filter
expressions.  The page consumes the Data Explorer's shared frame
(``frameLoaded``) plus any second loaded frame for the two-table
operations, runs every reshaping through the :class:`JobRunner`
(never the GUI thread), previews the result through the same
preview-windowed table the Explorer uses, and can bind the output back
as the workspace's active frame (the report line records what ran).

Design notes:
* One card per operation, exactly the backend's contract — no
  simplified re-implementations in the GUI (part8 §1.7).
* The result preview never mutates the source frame; "Bind as active
  dataset" is the explicit user action, announced through
  :attr:`frameLoaded` so every downstream page stays honest.
* Errors surface the backend's exact message (validation names the
  missing column, the invalid rule, the offending rows) — never a
  generic "failed".
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..widgets.data_table import DataTableView
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner


class TableToolsPage(QWidget):
    """Reshape the loaded dataset: combine, merge, summarize, pivot…"""

    #: The reshaped frame when the user binds it (mirrors the
    #: Explorer's own announcement so downstream pages refresh).
    frameLoaded = Signal(object)

    def __init__(
        self,
        runner: JobRunner | None = None,
        data_explorer: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner or JobRunner(self)
        self._data_explorer = data_explorer
        self._frame: pd.DataFrame | None = None
        self._second_frame: pd.DataFrame | None = None
        self._second_source = ""
        self._result: pd.DataFrame | None = None
        self._last_report: Any = None

        if self._data_explorer is not None and hasattr(self._data_explorer, "frameLoaded"):
            self._data_explorer.frameLoaded.connect(self._on_explorer_frame)

        root = QVBoxLayout(self)

        # -- header row: source status + second-frame loader ---------------
        header = QHBoxLayout()
        self._source_label = QLabel("No dataset loaded (load one in Data Explorer)")
        self._source_label.setObjectName("hint")
        header.addWidget(self._source_label, 1)
        self._load_second = QPushButton("Load second table…")
        self._load_second.setToolTip(
            "Load a second CSV/Excel table for the Combine and Merge cards"
        )
        self._load_second.clicked.connect(self._pick_second)
        header.addWidget(self._load_second)
        self._second_label = QLabel("")
        self._second_label.setObjectName("hint")
        header.addWidget(self._second_label)
        root.addLayout(header)

        # -- the operation tabs --------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_combine_tab(), "Combine")
        self._tabs.addTab(self._build_merge_tab(), "Merge")
        self._tabs.addTab(self._build_groupby_tab(), "GroupBy")
        self._tabs.addTab(self._build_pivot_tab(), "Pivot")
        self._tabs.addTab(self._build_strings_tab(), "Clean text")
        self._tabs.addTab(self._build_timeseries_tab(), "Time series")
        self._tabs.addTab(self._build_filter_tab(), "Filter")

        # -- result panel -----------------------------------------------------
        self._table = DataTableView()
        self._table.bind_frame(None, note="Run an operation to preview its result")
        self._status = QTextEdit(readOnly=True)
        self._status.setObjectName("log")
        self._status.setFixedHeight(84)
        self._bind_btn = QPushButton("Bind as active dataset")
        self._bind_btn.setToolTip(
            "Make the previewed result the workspace's active frame "
            "(announced to every page listening to frameLoaded)"
        )
        self._bind_btn.setEnabled(False)
        self._bind_btn.clicked.connect(self._bind_result)

        result_box = QGroupBox("Result")
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self._table)
        result_layout.addWidget(self._status)
        bind_row = QHBoxLayout()
        bind_row.addStretch(1)
        bind_row.addWidget(self._bind_btn)
        result_layout.addLayout(bind_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._tabs)
        splitter.addWidget(result_box)
        # (r270) the §23.2 splitter policy — non-collapsible children +
        # hard minimum extents + growth flowing to the result side. The
        # page shipped in v1.2.213 with the bare QSplitter defaults; the
        # round-270 gate re-run caught it (the r269 ship never recorded
        # its chunk-4d evidence) and this is the source fix.
        configure_splitter(splitter, min_sizes=(320, 320), stretch=(3, 4))
        root.addWidget(splitter, 1)

        self._refresh_column_combos()

    # -- frame plumbing ---------------------------------------------------

    def _on_explorer_frame(self, frame: object) -> None:
        """Shared-frame slot: the Explorer's active dataset arrived."""
        self._frame = frame
        name = "Data Explorer dataset"
        source = getattr(self._data_explorer, "source", lambda: "")()
        if source:
            name = source.rsplit("/", 1)[-1]
        self._source_label.setText(
            f"{name} — {len(frame):,} rows × {frame.shape[1]} columns"
        )
        self._refresh_column_combos()

    def frame(self) -> pd.DataFrame | None:
        """The Explorer's (or bound) frame this page operates on."""
        if self._frame is not None:
            return self._frame
        if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
            return self._data_explorer.frame()
        return None

    def _pick_second(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load second table", "", "Tables (*.csv *.tsv *.xlsx)"
        )
        if not path:
            return
        if self._runner.busy:
            self._note("Busy — wait for the running job to finish first.")
            return

        from pathlib import Path

        target = Path(path)

        def job(progress: object) -> object:
            from pharmaml.utils.io import read_table

            return read_table(target)

        def done(frame: object) -> None:
            self._second_frame = frame
            self._second_source = target.name
            self._second_label.setText(
                f"second: {target.name} ({len(frame):,} rows)"
            )
            self._second_combo_refresh()
            self._note(f"Loaded second table {target.name}: {len(frame):,} rows.")

        def failed(message: str) -> None:
            self._note(f"Could not read {target.name}: {message.splitlines()[0]}")

        self._runner.submit(f"load second table ({target.name})", job, done, failed)

    # -- shared helpers -----------------------------------------------------

    def _note(self, text: str) -> None:
        self._status.setPlainText(text)

    def _require_frame(self) -> pd.DataFrame | None:
        frame = self.frame()
        if frame is None:
            self._note("No dataset loaded — load one in Data Explorer first.")
        return frame

    def _submit(self, name: str, job, done=None) -> None:
        """Run one reshape operation through the JobRunner.

        ``job`` receives no arguments and returns ``(frame, report)``;
        ``done`` (optional) receives the frame on the GUI thread after
        the common preview+report handling below.
        """

        def _done(payload: object) -> None:
            out, report = payload
            self._result = out
            self._last_report = report
            self._table.bind_frame(out, note="Result preview (not yet bound)")
            self._note(report.summary_line())
            self._bind_btn.setEnabled(out is not None and not out.empty)
            if done is not None:
                done(out)

        def _failed(message: str) -> None:
            self._result = None
            self._bind_btn.setEnabled(False)
            first = message.splitlines()[0] if message else "unknown error"
            self._note(f"{name} failed: {first}")

        if self._runner.busy:
            self._note(f"Busy: '{self._runner.job_name}' is running — stop it first.")
            return
        self._runner.submit(name, job, _done, _failed)

    def _columns(self) -> list[str]:
        frame = self.frame()
        return [str(c) for c in frame.columns] if frame is not None else []

    def _refresh_column_combos(self) -> None:
        columns = self._columns()
        for combo in self._all_combos():
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        # numeric-only pickers
        frame = self.frame()
        numeric = (
            [str(c) for c in frame.select_dtypes("number").columns]
            if frame is not None else []
        )
        for combo in (self._groupby_value, self._pivot_values, self._timeseries_value):
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItems(numeric)
            if current in numeric:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        # string-ish pickers prefer object/string columns
        texty = (
            [str(c) for c in frame.columns
             if not pd.api.types.is_numeric_dtype(frame[c])]
            if frame is not None else []
        )
        for combo in (self._strings_column,):
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItems(texty or columns)
            if current in (texty or columns):
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _all_combos(self) -> list[QComboBox]:
        return [
            self._merge_key, self._groupby_key, self._pivot_index,
            self._pivot_columns, self._timeseries_time,
        ]

    def _second_combo_refresh(self) -> None:
        if self._second_frame is None:
            return
        columns = [str(c) for c in self._second_frame.columns]
        self._merge_key.blockSignals(True)
        current = self._merge_key.currentText()
        self._merge_key.clear()
        self._merge_key.addItems([c for c in self._columns() if c in columns] or columns)
        if current in columns:
            self._merge_key.setCurrentText(current)
        self._merge_key.blockSignals(False)

    def _bind_result(self) -> None:
        if self._result is None or self._result.empty:
            return
        frame = self._result
        self._frame = frame
        self._source_label.setText(
            f"bound result — {len(frame):,} rows × {frame.shape[1]} columns"
        )
        self._refresh_column_combos()
        self.frameLoaded.emit(frame)
        try:
            from pharmaml.studio import note as note_workspace

            note_workspace(
                "dataset", "table-tools-result",
                note=f"{len(frame)} rows", columns=frame,
            )
        except Exception:  # noqa: BLE001,S110 - registry is best-effort
            pass
        self._note(
            "Bound as the active dataset — downstream pages refreshed. "
            "The original table remains in Data Explorer."
        )

    # -- tab builders ---------------------------------------------------------

    def _build_combine_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._combine_axis = QComboBox()
        self._combine_axis.addItem("Stack rows (append replicate tables)", 0)
        self._combine_axis.addItem("Side-by-side (equal row counts)", 1)
        form.addRow("Direction:", self._combine_axis)
        self._combine_keys = QLineEdit()
        self._combine_keys.setPlaceholderText("optional labels, comma-separated (lab_a, lab_b)")
        form.addRow("Frame labels:", self._combine_keys)
        run = QPushButton("Combine")
        run.clicked.connect(self._run_combine)
        form.addRow(run)
        hint = QLabel(
            "Stacks the active dataset and the second table into one frame; "
            "a source_frame column records each row's origin."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return tab

    def _build_merge_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._merge_key = QComboBox()
        form.addRow("Key column:", self._merge_key)
        self._merge_how = QComboBox()
        self._merge_how.addItem("Keep left rows (left)", "left")
        self._merge_how.addItem("Shared rows only (inner)", "inner")
        self._merge_how.addItem("Keep everything (outer)", "outer")
        form.addRow("Join type:", self._merge_how)
        run = QPushButton("Merge")
        run.clicked.connect(self._run_merge)
        form.addRow(run)
        hint = QLabel(
            "Merges the active dataset with the second table on one shared "
            "key column. Duplicate keys are reported (fan-out warning)."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return tab

    def _build_groupby_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._groupby_key = QComboBox()
        form.addRow("Group by:", self._groupby_key)
        self._groupby_value = QComboBox()
        self._groupby_value.setToolTip("Numeric column to summarize (all are summarized by default)")
        form.addRow("Value column:", self._groupby_value)
        self._groupby_aggs = QLineEdit("mean, std, min, max, count")
        self._groupby_aggs.setToolTip("any of: mean, median, std, min, max, count, sum, nunique")
        form.addRow("Statistics:", self._groupby_aggs)
        run = QPushButton("Summarize")
        run.clicked.connect(self._run_groupby)
        form.addRow(run)
        hint = QLabel(
            "Per-group aggregate statistics (split-apply-combine): one row per "
            "group, one column per statistic."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return tab

    def _build_pivot_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._pivot_index = QComboBox()
        form.addRow("Row column(s):", self._pivot_index)
        self._pivot_columns = QComboBox()
        form.addRow("Spread column:", self._pivot_columns)
        self._pivot_values = QComboBox()
        form.addRow("Value column:", self._pivot_values)
        self._pivot_agg = QComboBox()
        for agg in ("mean", "median", "min", "max", "count", "sum", "nunique"):
            self._pivot_agg.addItem(agg, agg)
        form.addRow("Aggregation:", self._pivot_agg)
        run = QPushButton("Pivot")
        run.clicked.connect(self._run_pivot)
        form.addRow(run)
        hint = QLabel(
            "Reshapes long → wide: one column per value of the spread column, "
            "cells aggregate the value column. Missing combinations stay empty."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return tab

    def _build_strings_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._strings_column = QComboBox()
        form.addRow("Text column:", self._strings_column)
        self._strings_strip = QCheckBox("Strip surrounding whitespace")
        self._strings_strip.setChecked(True)
        form.addRow(self._strings_strip)
        self._strings_collapse = QCheckBox("Collapse inner whitespace runs")
        self._strings_collapse.setChecked(True)
        form.addRow(self._strings_collapse)
        self._strings_case = QComboBox()
        self._strings_case.addItem("keep case", "keep")
        self._strings_case.addItem("lowercase", "lower")
        self._strings_case.addItem("UPPERCASE", "upper")
        form.addRow("Letter case:", self._strings_case)
        self._strings_drop = QCheckBox("Drop rows with missing values")
        form.addRow(self._strings_drop)
        run = QPushButton("Clean")
        run.clicked.connect(self._run_strings)
        form.addRow(run)
        hint = QLabel(
            "Vectorized cleaning of one text column (SMILES/categoricals). "
            "The source table is never modified — the cleaned copy is the result."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return tab

    def _build_timeseries_tab(self) -> QWidget:
        QWidget()  # the container is built via _finish_timeseries_tab
        inner = QTabWidget()
        # resample sub-tab
        resample_tab = QWidget()
        form = QFormLayout(resample_tab)
        self._timeseries_time = QComboBox()
        form.addRow("Time column:", self._timeseries_time)
        self._timeseries_value = QComboBox()
        self._timeseries_value.setToolTip("Numeric value column")
        form.addRow("Value column:", self._timeseries_value)
        self._resample_rule = QComboBox()
        return self._finish_timeseries_tab(inner, form, resample_tab)

    def _finish_timeseries_tab(self, inner: QTabWidget, form: QFormLayout, resample_tab: QWidget) -> QWidget:
        for rule in ("1h", "1D", "15min", "30min", "1W", "1ME"):
            self._resample_rule.addItem(rule, rule)
        self._resample_rule.setEditable(True)
        form.addRow("Frequency:", self._resample_rule)
        self._resample_agg = QComboBox()
        for agg in ("mean", "median", "min", "max", "count"):
            self._resample_agg.addItem(agg, agg)
        form.addRow("Aggregation:", self._resample_agg)
        run_resample = QPushButton("Resample")
        run_resample.clicked.connect(self._run_resample)
        form.addRow(run_resample)
        hint = QLabel(
            "Aggregates the time series onto a fixed frequency grid "
            "(empty windows are reported, never zero-filled)."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        # rolling sub-tab
        rolling_tab = QWidget()
        rform = QFormLayout(rolling_tab)
        self._rolling_window = QSpinBox()
        self._rolling_window.setRange(2, 10000)
        self._rolling_window.setValue(3)
        rform.addRow("Window (points):", self._rolling_window)
        self._rolling_agg = QComboBox()
        for agg in ("mean", "median", "std", "min", "max", "sum"):
            self._rolling_agg.addItem(agg, agg)
        rform.addRow("Statistic:", self._rolling_agg)
        self._rolling_min_periods = QSpinBox()
        self._rolling_min_periods.setRange(1, 10000)
        self._rolling_min_periods.setValue(1)
        rform.addRow("Min. periods:", self._rolling_min_periods)
        run_rolling = QPushButton("Rolling window")
        run_rolling.clicked.connect(self._run_rolling)
        rform.addRow(run_rolling)
        rhint = QLabel(
            "Moving-window statistics over the time-sorted table; the "
            "original columns are kept, one _rolling_ column is added."
        )
        rhint.setObjectName("hint")
        rhint.setWordWrap(True)
        rform.addRow(rhint)
        inner.addTab(resample_tab, "Resample")
        inner.addTab(rolling_tab, "Rolling")
        return inner

    def _build_filter_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._filter_expression = QLineEdit()
        self._filter_expression.setPlaceholderText(
            "e.g.  ic50 < 100 and compound in ['aspirin', 'ibuprofen']"
        )
        self._filter_expression.setToolTip(
            "pandas query syntax: column names, comparisons, and/or/not, in […]. "
            "Keyword-named columns (e.g. yield) are escaped automatically."
        )
        form.addRow("Expression:", self._filter_expression)
        run = QPushButton("Filter rows")
        run.clicked.connect(self._run_filter)
        run.setDefault(True)
        form.addRow(run)
        hint = QLabel(
            "Keeps the rows matching the expression (pandas query grammar — "
            "never arbitrary Python). Column names with spaces need backticks: "
            "`IC50 nM` < 100."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return tab

    # -- operation runners ------------------------------------------------------

    def _run_combine(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        if self._second_frame is None:
            self._note("Combine needs a second table — load one above first.")
            return
        axis = self._combine_axis.currentData()
        raw_keys = self._combine_keys.text().strip()
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()] or None
        left, right = frame, self._second_frame

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import combine_frames

            return combine_frames([left, right], axis=axis, keys=keys)

        self._submit("combine", job)

    def _run_merge(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        if self._second_frame is None:
            self._note("Merge needs a second table — load one above first.")
            return
        key = self._merge_key.currentText()
        how = self._merge_how.currentData()
        left, right = frame, self._second_frame

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import merge_frames

            return merge_frames(left, right, on=key, how=how)

        self._submit(f"merge on {key!r}", job)

    def _run_groupby(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        key = self._groupby_key.currentText()
        value = self._groupby_value.currentText()
        aggs = [
            a.strip() for a in self._groupby_aggs.text().split(",")
            if a.strip()
        ]
        source = frame

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import groupby_summary

            aggregations = {value: aggs} if value else None
            return groupby_summary(source, by=[key], aggregations=aggregations)

        self._submit(f"groupby {key}", job)

    def _run_pivot(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        index = self._pivot_index.currentText()
        spread = self._pivot_columns.currentText()
        values = self._pivot_values.currentText()
        agg = self._pivot_agg.currentData()
        source = frame

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import pivot_table

            return pivot_table(
                source, index=[index], columns=spread, values=values, aggfunc=agg,
            )

        self._submit(f"pivot {index}×{spread}", job)

    def _run_strings(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        column = self._strings_column.currentText()
        case = self._strings_case.currentData()
        source = frame
        strip = self._strings_strip.isChecked()
        collapse = self._strings_collapse.isChecked()
        drop_missing = self._strings_drop.isChecked()

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import clean_strings

            return clean_strings(
                source, column=column, strip=strip,
                lowercase=case == "lower", uppercase=case == "upper",
                collapse_whitespace=collapse, drop_missing=drop_missing,
            )

        self._submit(f"clean '{column}'", job)

    def _run_resample(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        time_column = self._timeseries_time.currentText()
        value = self._timeseries_value.currentText()
        rule = self._resample_rule.currentText().strip()
        agg = self._resample_agg.currentData()
        source = frame

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import resample_timeseries

            return resample_timeseries(
                source, time_column=time_column, value_columns=[value],
                rule=rule, aggregation=agg,
            )

        self._submit(f"resample {rule}", job)

    def _run_rolling(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        time_column = self._timeseries_time.currentText()
        value = self._timeseries_value.currentText()
        window = self._rolling_window.value()
        agg = self._rolling_agg.currentData()
        min_periods = self._rolling_min_periods.value()
        source = frame

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import rolling_window

            return rolling_window(
                source, time_column=time_column, value_columns=[value],
                window=window, aggregation=agg, min_periods=min_periods,
            )

        self._submit(f"rolling {window}-point {agg}", job)

    def _run_filter(self) -> None:
        frame = self._require_frame()
        if frame is None:
            return
        expression = self._filter_expression.text()
        source = frame

        def job(_progress: object) -> object:
            from pharmaml.utils.reshape import filter_frame

            return filter_frame(source, expression)

        self._submit("filter", job)
