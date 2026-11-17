"""Audit exact keys across separate files without type inference."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workers.job_worker import JobRunner


class FileSplitAuditDialog(QDialog):
    """Run the backend's literal-identifier audit on named partition files."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audit separate split files")
        self.resize(760, 650)
        self._runner = runner
        self._revision = 0
        self._report: dict | None = None
        layout = QVBoxLayout(self)
        intro = QLabel("Audit two or more separate CSV, TSV, TXT or Excel files. "
                       "Identifiers are read as text, including leading zeros and literal NA values.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self._partitions = QPlainTextEdit()
        self._partitions.setPlaceholderText("One NAME=PATH per line, for example:\ntrain=/path/train.csv\ntest=/path/test.csv")
        self._partitions.setMaximumHeight(110)
        self._partitions.textChanged.connect(self._invalidate)
        form.addRow("Partition files:", self._partitions)
        self._browse = QPushButton("Add file…")
        self._browse.clicked.connect(self._add_file)
        form.addRow("", self._browse)
        self._keys = QPlainTextEdit()
        self._keys.setPlaceholderText("One exact column name per line; multiple columns form a composite key")
        self._keys.setMaximumHeight(75)
        self._keys.textChanged.connect(self._invalidate)
        form.addRow("Key columns:", self._keys)
        self._delimiter = QComboBox()
        self._delimiter.addItem("Automatic (TSV: tab; otherwise comma)", None)
        for label in ("comma", "tab", "semicolon"):
            self._delimiter.addItem(label.capitalize(), label)
        self._delimiter.currentIndexChanged.connect(self._invalidate)
        form.addRow("Text delimiter:", self._delimiter)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self._run = QPushButton("Audit files")
        self._run.clicked.connect(self._start)
        self._save = QPushButton("Save summary…")
        self._save.setEnabled(False)
        self._save.clicked.connect(self._save_summary)
        buttons.addWidget(self._run)
        buttons.addWidget(self._save)
        layout.addLayout(buttons)
        self._status = QLabel("No audit yet. This job cannot be stopped or resumed.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        warning = QLabel("Exact matches only: a clean result does not establish independence or rule out "
                         "related compounds or other leakage. Files must stay unchanged while reading. "
                         "The summary omits raw identifiers; names and column labels can still be sensitive.")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self._controls = (self._partitions, self._browse, self._keys, self._delimiter, self._run)

    def _add_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a partition file", "",
                                             "Tables (*.csv *.tsv *.txt *.xlsx *.xlsm);;All files (*)")
        if not path:
            return
        name, accepted = QInputDialog.getText(self, "Partition name", "Name (letters, numbers, _ or -):")
        if accepted and name:
            self._partitions.appendPlainText(f"{name}={path}")

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        self._report = None
        if hasattr(self, "_result"):
            self._result.clear()
            self._save.setEnabled(False)
            self._status.setText("Inputs changed; run the audit again.")

    def _enable(self, enabled: bool) -> None:
        for control in self._controls:
            control.setEnabled(enabled)
        self._save.setEnabled(enabled and self._report is not None)

    def _start(self) -> None:
        from pharmaml.ml.file_split_audit import audit_partition_files

        partitions: dict[str, str] = {}
        for line in self._partitions.toPlainText().splitlines():
            if not line.strip():
                continue
            name, sep, path = line.partition("=")
            name, path = name.strip(), path.strip()
            if not sep or not name or not path or name in partitions:
                self._invalidate()
                self._status.setText("Use one unique NAME=PATH per line; names and paths must not be empty.")
                return
            partitions[name] = path
        keys = [line for line in self._keys.toPlainText().splitlines() if line.strip()]
        if len(partitions) < 2 or not keys:
            self._invalidate()
            self._status.setText("Provide at least two named files and one key column.")
            return
        delimiter = self._delimiter.currentData()
        revision = self._revision
        self._report = None
        self._result.clear()
        self._status.setText("Reading and auditing files; this job cannot be stopped or resumed.")
        self._enable(False)

        def run(_progress):
            report = audit_partition_files(partitions, keys, delimiter)
            return {
                "key_columns": keys,
                "shared_key_count": report.n_shared_keys,
                "partitions": [
                    {"name": name, "rows": rows, "duplicate_rows": duplicates, "overlap_rows": overlap}
                    for name, rows, duplicates, overlap in zip(
                        report.partition_names, report.partition_rows, report.duplicate_rows,
                        report.overlap_rows, strict=True)
                ],
                "pairwise_shared_unique_keys": {
                    name: {other: int(report.pairwise[i, j])
                           for j, other in enumerate(report.partition_names)}
                    for i, name in enumerate(report.partition_names)
                },
            }

        worker = self._runner.submit("Separate-file split audit", run,
                                     on_finished=lambda result: self._finished(result, revision),
                                     on_failed=lambda error: self._failed(error, revision))
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _finished(self, report: dict, revision: int) -> None:
        self._enable(True)
        if revision != self._revision:
            return
        self._report = report
        self._save.setEnabled(True)
        self._result.setPlainText(json.dumps(report, indent=2, ensure_ascii=False))
        count = report["shared_key_count"]
        self._status.setText(f"Audit complete: {count} shared exact key(s). Review limitations and counts.")

    def _failed(self, error: str, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Audit failed: {error.splitlines()[0]}")

    def _save_summary(self) -> None:
        if self._report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save summary (new JSON file)", "", "JSON (*.json)",
                                             options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        try:
            if Path(path).suffix.lower() != ".json":
                raise ValueError("Choose a .json file")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except (OSError, ValueError, TypeError) as exc:
            self._status.setText(f"Could not save summary; existing files are never replaced: {exc}")
            return
        self._status.setText("Summary saved. Review names and column labels before sharing.")
