"""Read-only guided Vina campaign input, engine, box and output review."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from ..workers.job_worker import JobRunner


class CampaignReviewWizard(QWizard):
    """Collect and review one Vina batch plan without ever launching or writing."""

    def __init__(self, runner: JobRunner, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vina campaign review")
        self.resize(760, 630)
        self._runner = runner
        self._revision = 0

        inputs = QWizardPage()
        inputs.setTitle("1. Choose inputs and output")
        inputs.setSubTitle("Existing prepared PDBQT inputs only; this wizard does not prepare files.")
        form = QFormLayout(inputs)
        self._config = self._path_row(inputs, form, "Vina config", "config", file=True)
        self._ligands = self._path_row(inputs, form, "Ligand directory", "ligands", file=False)
        self._output = self._path_row(inputs, form, "Output directory", "output", file=False)
        self.addPage(inputs)

        engine = QWizardPage()
        engine.setTitle("2. Engine and box")
        engine.setSubTitle("The six center/size values (Å) and receptor come from the config.")
        engine_form = QFormLayout(engine)
        self._engine = QLineEdit("vina")
        self._engine.textChanged.connect(self._invalidate)
        engine_form.addRow("Executable name or path:", self._engine)
        self._engine_dir = QLineEdit()
        self._engine_dir.setPlaceholderText("Optional engine directory")
        self._engine_dir.textChanged.connect(self._invalidate)
        engine_form.addRow("Engine directory:", self._engine_dir)
        hint = QLabel("Review resolves a launch prefix but does not execute the engine. "
                      "A found executable or Wine does not prove the engine works.")
        hint.setWordWrap(True)
        engine_form.addRow("", hint)
        self.addPage(engine)

        review = QWizardPage()
        review.setTitle("3. Review (no execution)")
        review.setSubTitle("Check inputs and review box, engine and existing output census.")
        layout = QVBoxLayout(review)
        self._check = QPushButton("Check and review")
        self._check.clicked.connect(self._start)
        layout.addWidget(self._check)
        self._status = QLabel("Not checked. Finish closes this wizard; it does not run docking.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        self.addPage(review)

    def _path_row(self, page: QWizardPage, form: QFormLayout, title: str,
                  field_name: str, *, file: bool) -> QLineEdit:
        row = QHBoxLayout()
        field = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse(field, file=file))
        field.textChanged.connect(self._invalidate)
        row.addWidget(field, 1)
        row.addWidget(browse)
        form.addRow(f"{title}:", row)
        page.registerField(field_name + "*", field)
        return field

    def _browse(self, field: QLineEdit, *, file: bool) -> None:
        if file:
            value, _ = QFileDialog.getOpenFileName(self, "Select Vina config", "", "All files (*)")
        else:
            value = QFileDialog.getExistingDirectory(self, "Select directory")
        if value:
            field.setText(value)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        # Signals can fire during construction before review widgets exist.
        if hasattr(self, "_result"):
            self._result.clear()
            self._status.setText("Inputs changed; review again. No docking has run.")

    def _start(self) -> None:
        from pharmaml.docking.campaign_preview import preview_vina_campaign

        config, ligands, output = (self._config.text().strip(), self._ligands.text().strip(),
                                   self._output.text().strip())
        engine, engine_dir = self._engine.text().strip(), self._engine_dir.text().strip() or None
        if not config or not ligands or not output or not engine:
            self._result.clear()
            self._status.setText("Provide config, ligand and output paths, and an engine name or path.")
            return
        revision = self._revision
        self._result.clear()
        self._status.setText("Checking inputs and engine; no docking is being executed.")
        self._check.setEnabled(False)
        worker = self._runner.submit(
            "Vina campaign review",
            lambda _progress: preview_vina_campaign(config, ligands, output,
                                                   vina_bin=engine, engine_dir=engine_dir),
            on_finished=lambda result: self._finished(result, revision),
            on_failed=lambda message: self._failed(message, revision),
        )
        if worker is None:
            self._check.setEnabled(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _finished(self, result: object, revision: int) -> None:
        self._check.setEnabled(True)
        if revision == self._revision:
            self._result.setPlainText(json.dumps(result, indent=2, default=str, allow_nan=False))
            self._status.setText("Review only: no docking executed. Read warnings before proceeding.")

    def _failed(self, message: str, revision: int) -> None:
        self._check.setEnabled(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Review failed: {message.splitlines()[0]}")
