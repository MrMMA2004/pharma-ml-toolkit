"""The §17D run-report dialog.

Opened from the Job Center's history: the researcher selects ONE
recorded run and gets the structured scientific report — the eight
§17D sections previewed as text, then saved as a self-contained HTML
file through the existing writer.  The dialog never adds content of
its own: the preview is exactly what ``report_sections`` composed.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from pharmaml.studio.reports import (
    _SECTION_ORDER,
    report_sections,
    save_run_report,
)


class RunReportDialog(QDialog):
    """Preview + save the §17D scientific report for one saved run."""

    def __init__(self, record: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scientific run report")
        self.resize(820, 560)
        self._record = record

        lay = QVBoxLayout(self)
        label = str(record.get("label", "(unlabelled)"))
        family = str(record.get("family", ""))
        head = QLabel(f"Report preview — {label}  [{family}]")
        head.setObjectName("studioStatus")
        head.setWordWrap(True)
        lay.addWidget(head)

        self._note = QLabel(
            "Structure: " + " → ".join(_SECTION_ORDER)
            + ". The report states facts only — no conclusions are "
              "written automatically.")
        self._note.setWordWrap(True)
        lay.addWidget(self._note)

        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        sections = report_sections(record)
        blocks = []
        for name in _SECTION_ORDER:
            blocks.append(f"== {name}")
            blocks.append(sections[name])
            blocks.append("")
        self._preview.setPlainText("\n".join(blocks))
        lay.addWidget(self._preview, stretch=1)

        row = QHBoxLayout()
        save_btn = QPushButton("Save self-contained HTML…")
        save_btn.clicked.connect(self._save)
        row.addWidget(save_btn)
        self._status = QLabel("")
        self._status.setObjectName("studioStatus")
        row.addWidget(self._status, stretch=1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def _save(self) -> None:
        default = f"run-report-{str(self._record.get('label', 'run'))}.html"
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save the run report", default, "HTML (*.html)")
        if not chosen:
            return
        try:
            path = save_run_report(self._record, chosen)
        except Exception as exc:  # noqa: BLE001 - honest inline error
            QMessageBox.critical(self, "Report failed",
                                 f"{type(exc).__name__}: {exc}")
            return
        self._status.setText(f"saved: {path}")
