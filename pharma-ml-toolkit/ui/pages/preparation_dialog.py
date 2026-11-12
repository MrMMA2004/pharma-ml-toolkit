"""Review-first receptor and ligand preparation; never executes docking."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..workers.job_worker import JobRunner


class PreparationDialog(QDialog):
    """Review input checks and explicitly prepare into a fresh directory."""

    def __init__(self, runner: JobRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prepare docking inputs")
        self.resize(790, 720)
        self._runner = runner
        self._revision = 0
        self._plan = None
        layout = QVBoxLayout(self)
        intro = QLabel("Review receptor cleaning and ligand inputs first. Preparation writes to a new "
                       "directory; it does not select a box or run docking.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self._controls: list[QWidget] = []
        self._receptor = self._file_row(form, "Receptor (PDB/PDBQT/mmCIF)")
        self._table = self._file_row(form, "Ligand table (SMILES)")
        self._output = QLineEdit()
        self._output.setPlaceholderText("New directory path; parent must already exist")
        self._output.textChanged.connect(self._invalidate)
        form.addRow("New output directory:", self._output)
        self._controls.append(self._output)
        self._smiles, self._name, self._chain = QLineEdit(), QLineEdit(), QLineEdit()
        self._smiles.setPlaceholderText("Optional; auto-detected when blank")
        self._name.setPlaceholderText("Optional; auto-detected when blank")
        self._chain.setPlaceholderText("Optional single PDB chain character; blank keeps all")
        for title, field in (("SMILES column:", self._smiles), ("Name column:", self._name),
                             ("Receptor chain:", self._chain)):
            field.textChanged.connect(self._invalidate)
            form.addRow(title, field)
            self._controls.append(field)
        self._waters = QCheckBox("Remove waters")
        self._waters.setChecked(True)
        self._hetatm = QCheckBox("Keep non-water HETATM (cofactors, ions, etc.)")
        for box in (self._waters, self._hetatm):
            box.stateChanged.connect(self._invalidate)
            form.addRow("", box)
            self._controls.append(box)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_147_483_647)
        self._seed.setValue(0)
        self._seed.valueChanged.connect(self._invalidate)
        form.addRow("Ligand embedding seed:", self._seed)
        self._controls.append(self._seed)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self._review = QPushButton("Review inputs")
        self._review.clicked.connect(self._start_review)
        self._prepare = QPushButton("Prepare reviewed inputs…")
        self._prepare.setEnabled(False)
        self._prepare.clicked.connect(self._confirm_prepare)
        buttons.addWidget(self._review)
        buttons.addWidget(self._prepare)
        layout.addLayout(buttons)
        self._status = QLabel("No review yet. This job cannot be stopped or resumed.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        layout.addWidget(self._result, 1)
        warning = QLabel("Review waters, cofactors, chains, protonation, tautomers and chemistry independently. "
                         "First model and first altloc are used. Failures can leave partial outputs; "
                         "closing this dialog does not stop preparation.")
        warning.setWordWrap(True)
        layout.addWidget(warning)

    def _file_row(self, form: QFormLayout, label: str) -> QLineEdit:
        row = QHBoxLayout()
        field = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse(field))
        field.textChanged.connect(self._invalidate)
        row.addWidget(field, 1)
        row.addWidget(browse)
        form.addRow(label + ":", row)
        self._controls.extend((field, browse))
        return field

    def _browse(self, field: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select preparation input", "", "All files (*)")
        if path:
            field.setText(path)

    def _invalidate(self, *_args: object) -> None:
        self._revision += 1
        self._plan = None
        if hasattr(self, "_result"):
            self._result.clear()
            self._prepare.setEnabled(False)
            self._status.setText("Inputs changed; review again before preparing.")

    def _enable(self, enabled: bool) -> None:
        for control in self._controls:
            control.setEnabled(enabled)
        self._review.setEnabled(enabled)
        self._prepare.setEnabled(enabled and self._plan is not None)

    def _start_review(self) -> None:
        from pharmaml.docking.campaign_preparation import build_preparation_plan

        receptor, table, output = self._receptor.text().strip(), self._table.text().strip(), self._output.text().strip()
        if not receptor or not table or not output:
            self._status.setText("Select receptor, ligand table and a new output path.")
            self._result.clear()
            self._plan = None
            self._prepare.setEnabled(False)
            return
        options = {"smiles_col": self._smiles.text().strip() or None, "name_col": self._name.text().strip() or None,
                   "chain": self._chain.text().strip() or None, "remove_waters": self._waters.isChecked(),
                   "keep_hetatm": self._hetatm.isChecked(), "seed": self._seed.value()}
        revision = self._revision
        self._plan = None
        self._result.clear()
        self._status.setText("Reviewing inputs; no files are being prepared.")
        self._enable(False)
        worker = self._runner.submit(
            "Docking input review",
            lambda _progress: build_preparation_plan(Path(receptor), Path(table), Path(output), **options),
            on_finished=lambda plan: self._review_done(plan, revision),
            on_failed=lambda error: self._failed(error, revision),
        )
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; try again later.")

    def _review_done(self, plan: object, revision: int) -> None:
        if revision == self._revision:
            self._plan = plan
            self._result.setPlainText(json.dumps(plan, indent=2, allow_nan=False))
            self._status.setText("Review complete; inspect choices and warnings before preparing.")
        self._enable(True)

    def _confirm_prepare(self) -> None:
        if self._plan is None or self._runner.busy:
            return
        answer = QMessageBox.question(
            self, "Prepare docking inputs?",
            f"Create a NEW directory and prepare receptor and ligands:\n{self._plan['output_dir']}\n\n"
            "Input snapshots and partial results will remain on failure. This does not dock, and cannot "
            "be stopped or resumed. Closing the dialog does not stop it. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._start_prepare()

    def _start_prepare(self) -> None:
        from pharmaml.docking.campaign_preparation import prepare_reviewed_inputs

        if self._plan is None:
            return
        plan, revision = self._plan, self._revision
        self._plan = None
        self._status.setText("Preparing receptor and ligands; this job cannot be stopped or resumed.")
        self._enable(False)
        worker = self._runner.submit("Prepare docking inputs", lambda _progress: prepare_reviewed_inputs(plan),
                                     on_finished=lambda result: self._prepared(result, revision),
                                     on_failed=lambda error: self._failed(error, revision))
        if worker is None:
            self._enable(True)
            self._status.setText("Another Studio job is running; review again later.")

    def _prepared(self, result: object, revision: int) -> None:
        self._enable(True)
        if revision == self._revision:
            self._result.setPlainText(json.dumps(result, indent=2, allow_nan=False))
            self._status.setText(f"Preparation {result['status']}. Inspect receptor, failures and ligand structures.")

    def _failed(self, error: str, revision: int) -> None:
        self._plan = None
        self._enable(True)
        if revision == self._revision:
            self._result.clear()
            self._status.setText(f"Preparation/review failed: {error.splitlines()[0]} "
                                 "A partial output directory may remain; review again.")
