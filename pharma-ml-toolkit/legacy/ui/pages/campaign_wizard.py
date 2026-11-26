"""Review-first Vina campaign workflow with explicit fresh-directory launch."""
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from ..workers.job_worker import JobRunner


class CampaignReviewWizard(QWizard):
    """Review inputs, then optionally launch a sequential, non-resumable Vina batch."""

    def __init__(self, runner: JobRunner, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vina campaign review and launch")
        self.resize(760, 630)
        self._runner = runner
        self._revision = 0
        self._launch_plan = None
        self._launch_running = False

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
        self._engine.setToolTip("Docking engine to drive (vina …); must be on PATH or resolved below")
        self._engine.textChanged.connect(self._invalidate)
        engine_form.addRow("Executable name or path:", self._engine)
        self._engine_dir = QLineEdit()
        self._engine_dir.setToolTip("Optional directory where the engine binary lives (empty = PATH)")
        self._engine_dir.setPlaceholderText("Optional engine directory")
        self._engine_dir.textChanged.connect(self._invalidate)
        engine_form.addRow("Engine directory:", self._engine_dir)
        hint = QLabel("Review resolves a launch prefix but does not execute the engine. "
                      "A found executable or Wine does not prove the engine works.")
        hint.setWordWrap(True)
        engine_form.addRow("", hint)
        self.addPage(engine)

        review = QWizardPage()
        review.setTitle("3. Review and optional launch")
        review.setSubTitle("Review first; launching requires separate confirmation and a new output directory.")
        layout = QVBoxLayout(review)
        self._check = QPushButton("Check and review")
        self._check.clicked.connect(self._start)
        layout.addWidget(self._check)
        self._launch = QPushButton("Launch reviewed campaign…")
        self._launch.setEnabled(False)
        self._launch.clicked.connect(self._confirm_launch)
        layout.addWidget(self._launch)
        self._stop = QPushButton("Stop after current ligand")
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._request_stop)
        layout.addWidget(self._stop)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)
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
        self._launch_plan = None
        if hasattr(self, "_launch"):
            self._launch.setEnabled(False)
        # Signals can fire during construction before review widgets exist.
        if hasattr(self, "_result"):
            self._result.clear()
            self._status.setText("Inputs changed; review again before launching.")

    def _start(self) -> None:
        from pharmaml.docking.campaign_launch import build_vina_launch_plan
        from pharmaml.docking.campaign_preview import preview_vina_campaign

        config, ligands, output = (self._config.text().strip(), self._ligands.text().strip(),
                                   self._output.text().strip())
        engine, engine_dir = self._engine.text().strip(), self._engine_dir.text().strip() or None
        if not config or not ligands or not output or not engine:
            self._result.clear()
            self._status.setText("Provide config, ligand and output paths, and an engine name or path.")
            return
        revision = self._revision
        self._launch_plan = None
        self._launch.setEnabled(False)
        self._result.clear()
        self._status.setText("Checking inputs and engine; no docking is being executed.")
        self._check.setEnabled(False)
        def review(_progress):
            preview = preview_vina_campaign(config, ligands, output, vina_bin=engine, engine_dir=engine_dir)
            try:
                plan = build_vina_launch_plan(config, ligands, output, vina_bin=engine, engine_dir=engine_dir)
                return {"preview": preview, "launch_plan": plan, "launch_unavailable": None}
            except Exception as exc:  # noqa: BLE001 - keep read-only review available with explicit reason
                return {"preview": preview, "launch_plan": None, "launch_unavailable": str(exc)}

        worker = self._runner.submit(
            "Vina campaign review", review,
            on_finished=lambda result: self._finished(result, revision),
            on_failed=lambda message: self._failed(message, revision),
        )
        if worker is None:
            self._check.setEnabled(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _finished(self, result: object, revision: int) -> None:
        self._check.setEnabled(True)
        if revision == self._revision:
            self._launch_plan = result["launch_plan"]
            self._launch.setEnabled(self._launch_plan is not None)
            self._result.setPlainText(json.dumps(result, indent=2, default=str, allow_nan=False))
            self._status.setText("Review only: no docking executed. Read warnings before proceeding.")

    def _failed(self, message: str, revision: int) -> None:
        self._check.setEnabled(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Review failed: {message.splitlines()[0]}")

    def _confirm_launch(self) -> None:
        if self._launch_plan is None or self._runner.busy:
            return
        output = self._launch_plan["preview"]["output_dir"]
        answer = QMessageBox.question(
            self, "Launch Vina campaign?",
            f"Run the selected external engine sequentially and create a NEW directory:\n{output}\n\n"
            "Input snapshots, poses, logs and a manifest will be retained there. Stop takes effect "
            "between ligands; there is NO resume. A failure or crash can leave partial files. "
            "Closing the wizard does not stop a running job. Run only a trusted engine "
            "with prepared inputs. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start_launch()

    def _start_launch(self) -> None:
        from pharmaml.docking.campaign_launch import launch_vina_campaign

        plan = self._launch_plan
        if plan is None:
            self._status.setText("Review a launchable plan first.")
            return
        revision = self._revision
        self._launch_plan = None
        self._launch_running = True
        self._check.setEnabled(False)
        self._launch.setEnabled(False)
        self._stop.setEnabled(True)
        self.page(0).setEnabled(False)
        self.page(1).setEnabled(False)
        self._progress.setValue(0)
        self._status.setText("Launching; input snapshots are being checked. No resume is available.")
        worker = self._runner.submit(
            "Vina campaign launch",
            lambda progress: launch_vina_campaign(
                plan, on_progress=lambda done, total: progress(done / total, f"Completed {done}/{total} ligands")),
            on_progress=self._launch_progress,
            on_finished=lambda result: self._launch_finished(result, revision),
            on_failed=lambda error: self._launch_failed(error, revision),
        )
        if worker is None:
            self._end_launch()
            self._status.setText("Another Studio job is running; review again after it finishes.")

    def _request_stop(self) -> None:
        if self._launch_running:
            self._runner.request_stop()
            self._stop.setEnabled(False)
            self._status.setText("Stop requested; waiting for the current ligand to finish.")

    def _launch_progress(self, percent: int, label: str) -> None:
        self._progress.setValue(percent)
        if self._stop.isEnabled():
            self._status.setText(f"{label}; stop takes effect between ligands.")

    def _end_launch(self) -> None:
        self._launch_running = False
        self._check.setEnabled(True)
        self._launch.setEnabled(False)
        self._stop.setEnabled(False)
        self.page(0).setEnabled(True)
        self.page(1).setEnabled(True)

    def _launch_finished(self, result: object, revision: int) -> None:
        self._end_launch()
        if revision == self._revision:
            self._result.setPlainText(json.dumps(result, indent=2, default=str, allow_nan=False))
            self._status.setText(f"Campaign {result['status']}; inspect output and manifest. No resume is available.")

    def _launch_failed(self, error: str, revision: int) -> None:
        self._end_launch()
        if revision == self._revision:
            self._status.setText(f"Launch failed: {error.splitlines()[0]} "
                                 "A partial output directory may remain; no resume is available.")
