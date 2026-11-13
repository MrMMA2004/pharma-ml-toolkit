"""The §17A workflow-templates tab (Utilities Workbench).

A template browser over :mod:`pharmaml.studio.pipelines`: the seven
part6 §17A scientific workflow templates, each shown with its owning
card, its stage chain (every stage names the existing backend it
calls), and — for the two parameter-light templates — an honest,
runnable form.  The run itself threads the engine's per-stage
outcomes into the progress panel and writes a JSON run record next to
the exports, exactly what the §17C comparison surface will consume.

Design notes
------------
* NO drag-and-drop (§17B's explicit instruction: a reliable sequential
  engine first).  The tab runs a template's fixed stage chain; the
  chain is displayed, not hand-wired on a canvas.
* Missing optional dependencies surface through the engine's
  ``DependencyError`` message (the exact pip command) in the log —
  never a silent empty result.
* The two runnable forms (PK/PD, MD) need no heavy optional
  dependency and complete in seconds; the data-dependent templates
  (QSAR, descriptors, response, generative, docking) show their stage
  chain and parameter documentation and run via the same button once
  the caller supplies paths — the form fields are validated before
  the run starts, with the validation error shown inline.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pharmaml.studio.pipelines import (
    TEMPLATES,
    build_template,
    template_names,
)


class WorkflowTemplatesTab(QWidget):
    """Browser + runner for the §17A workflow-template library."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_record: dict | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        pick = QHBoxLayout()
        pick.addWidget(QLabel("template:"))
        self._choice = QComboBox()
        for name in template_names():
            entry = TEMPLATES[name]
            self._choice.addItem(f"{name}  [{entry['card']}]", name)
        self._choice.currentIndexChanged.connect(self._refresh)
        pick.addWidget(self._choice, stretch=1)
        lay.addLayout(pick)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        self._stages = QTextEdit()
        self._stages.setReadOnly(True)
        lay.addWidget(self._stages, stretch=2)

        form = QFormLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText(
            "export folder (required — pick or type a path)")
        form.addRow("export folder:", self._out_dir)
        # the per-template input area: swapped by _refresh() so every
        # template carries its own card-integrated form (R26-c)
        self._form_host = QWidget()
        self._form_layout = QFormLayout(self._form_host)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        form.addRow(self._form_host)
        lay.addLayout(form)

        self._pick_btn = QPushButton("Pick export folder…")
        self._pick_btn.clicked.connect(self._pick_out_dir)
        self._run_btn = QPushButton("Run template")
        self._run_btn.clicked.connect(self._run)
        btns = QHBoxLayout()
        btns.addWidget(self._pick_btn)
        btns.addWidget(self._run_btn)
        self._remember = QPushButton("Remember export folder")
        self._remember.setToolTip(
            "§17G: persist this folder for THIS template — it comes back "
            "pre-filled the next time you pick the template.")
        self._remember.clicked.connect(self._remember_folder)
        btns.addWidget(self._remember)
        btns.addStretch(1)
        lay.addLayout(btns)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText(
            "run record: per-stage outcomes, timings, artifacts, errors")
        lay.addWidget(self._log, stretch=3)

        self._refresh()

    # -- browsing -----------------------------------------------------------

    def _current_name(self) -> str:
        return self._choice.currentData()

    def _refresh(self) -> None:
        name = self._current_name()
        entry = TEMPLATES[name]
        self._summary.setText(f"{entry['summary']}  (card: {entry['card']})")
        lines = ["stage chain (each stage calls an existing backend):"]
        # instantiate with safe defaults purely to DESCRIBE the chain
        # (the factories only build Stage objects — no science runs)
        try:
            pipe = self._describe_chain(name)
            for i, stage in enumerate(pipe.stages, 1):
                lines.append(f"{i}. {stage.name} — {stage.description}")
        except Exception as exc:  # noqa: BLE001 - honest display
            lines.append(f"(chain preview unavailable: {exc})")
        self._stages.setPlainText("\n".join(lines))
        self._build_form_for(name)
        # fill the target/compound combo live as the path is typed/picked
        for key in ("data_path", "screen_path"):
            field = self._inputs.get(key)
            if isinstance(field, QLineEdit):
                field.textChanged.connect(self._refresh_target_combo)
        # §17G: recall this template's remembered export folder
        from pharmaml.studio.workspace import last_export_folder

        remembered = last_export_folder(name)
        self._out_dir.setText(remembered)

    def _describe_chain(self, name: str):
        """Build the pipeline with placeholder args for display."""
        if name == "pk_pd_modeling":
            return build_template(name, out_dir=".", ka=1.2, ke=0.25,
                                  vd=40.0, dose=250.0)
        if name == "md_analysis":
            return build_template(name, out_dir=".", n_frames=80, seed=0)
        if name == "molecular_generation_screening":
            return build_template(name, out_dir=".", objective="qed",
                                  generations=3, seed=42)
        if name in ("qsar_model_development",):
            return build_template(name, out_dir=".", data_path="data.csv",
                                  target=None, models=None, cv=5)
        if name == "drug_response_prediction":
            return build_template(name, out_dir=".", screen_path="screen.csv",
                                  expression_path="expr.csv",
                                  compound=None, models=None, cv=5)
        if name == "molecular_descriptor_generation":
            return build_template(name, out_dir=".", data_path="mols.csv",
                                  smiles_col="smiles", kind="morgan")
        if name == "docking_preparation_analysis":
            return build_template(name, out_dir=".", ligands_path="lib.csv",
                                  receptor_pdbqt="ref.pdbqt",
                                  smiles_col="SMILES")
        raise ValueError(name)

    # -- per-template card-integrated forms (R26-c) ---------------------------

    def _clear_form(self) -> None:
        while (row := self._form_layout.takeAt(0)) is not None:
            if row.widget():
                row.widget().deleteLater()

    def _add_path_row(self, label: str, placeholder: str,
                      pick_caption: str) -> QLineEdit:
        """One labelled path field with its own inline Pick button."""
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        pick = QPushButton("…")
        pick.setFixedWidth(28)
        pick.clicked.connect(lambda: self._pick_file(edit, pick_caption))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=1)
        row.addWidget(pick)
        host = QWidget()
        host.setLayout(row)
        self._form_layout.addRow(label, host)
        return edit

    def _pick_file(self, edit: QLineEdit, caption: str) -> None:
        from PySide6.QtWidgets import QFileDialog

        chosen, _ = QFileDialog.getOpenFileName(self, caption)
        if chosen:
            edit.setText(chosen)

    def _build_form_for(self, name: str) -> None:
        """Swap the input form to the current template's fields."""
        self._clear_form()
        self._inputs: dict[str, object] = {}
        if name == "pk_pd_modeling":
            for key, label, top, decimals in (
                    ("ka", "ka (1/h)", 1000.0, 3),
                    ("ke", "ke (1/h)", 1000.0, 3),
                    ("vd", "Vd (L)", 100000.0, 1),
                    ("dose", "dose (mg)", 100000.0, 1)):
                box = QDoubleSpinBox()
                box.setRange(0.0, top)
                box.setDecimals(decimals)
                box.setValue({"ka": 1.2, "ke": 0.25,
                              "vd": 40.0, "dose": 250.0}[key])
                self._form_layout.addRow(label, box)
                self._inputs[key] = box
        elif name == "md_analysis":
            box = QSpinBox()
            box.setRange(10, 500)
            box.setValue(80)
            self._form_layout.addRow("MD frames:", box)
            self._inputs["n_frames"] = box
        elif name == "molecular_generation_screening":
            objective = QComboBox()
            objective.addItems(["qed", "sa", "logp_band", "mw_band",
                                "tpsa_band"])
            self._form_layout.addRow("objective:", objective)
            gens = QSpinBox()
            gens.setRange(1, 50)
            gens.setValue(3)
            self._form_layout.addRow("generations:", gens)
            self._inputs.update(objective=objective, generations=gens)
        elif name in ("qsar_model_development",
                      "molecular_descriptor_generation"):
            key = "data_path"
            field = self._add_path_row(
                "dataset CSV:", "molecules CSV (smiles + target columns)",
                "Pick the dataset CSV")
            self._inputs[key] = field
            if name == "qsar_model_development":
                target = QComboBox()
                target.setEditable(True)
                target.setToolTip(
                    "the prediction target column; leave as (auto) to let "
                    "the pipeline pick the last non-SMILES column")
                target.addItem("(auto)")
                self._form_layout.addRow("target column:", target)
                self._inputs["target"] = target
                self._inputs["_data_field"] = field
            smiles = QComboBox()
            smiles.setEditable(True)
            smiles.setToolTip("the SMILES column (default: smiles/SMILES)")
            smiles.addItem("smiles")
            smiles.addItem("SMILES")
            if name == "molecular_descriptor_generation":
                self._form_layout.addRow("SMILES column:", smiles)
                self._inputs["smiles_col"] = smiles
            else:
                self._inputs["smiles_col"] = smiles
        elif name == "drug_response_prediction":
            screen = self._add_path_row(
                "response screen CSV:",
                "drug-response screen (cell lines × compounds)",
                "Pick the response screen CSV")
            expr = self._add_path_row(
                "expression CSV:", "gene-expression matrix",
                "Pick the expression CSV")
            compound = QComboBox()
            compound.setEditable(True)
            compound.setToolTip(
                "restrict to one compound; (auto) takes the screen's "
                "first compound (stated in the run record)")
            compound.addItem("(auto)")
            self._form_layout.addRow("compound:", compound)
            cv = QSpinBox()
            cv.setRange(2, 10)
            cv.setValue(5)
            self._form_layout.addRow("CV folds:", cv)
            self._inputs.update(screen_path=screen, expression_path=expr,
                                compound=compound, cv=cv)
        elif name == "docking_preparation_analysis":
            ligands = self._add_path_row(
                "ligands CSV:", "SMILES ligand library CSV",
                "Pick the ligands CSV")
            receptor = self._add_path_row(
                "receptor PDBQT:", "prepared receptor (.pdbqt)",
                "Pick the receptor PDBQT")
            smiles = QComboBox()
            smiles.setEditable(True)
            smiles.setToolTip("the SMILES column (default: SMILES)")
            smiles.addItem("SMILES")
            smiles.addItem("smiles")
            self._form_layout.addRow("SMILES column:", smiles)
            self._inputs.update(ligands_path=ligands,
                                receptor_pdbqt=receptor, smiles_col=smiles)
        else:
            raise ValueError(name)

    def _csv_columns(self, path: str) -> list[str]:
        """Best-effort column peek (empty on any read failure)."""
        try:
            import csv

            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                return [str(c).strip() for c in next(csv.reader(fh))]
        except (OSError, StopIteration, UnicodeError):
            return []

    def _refresh_target_combo(self) -> None:
        """Fill the target/compound combos with the picked CSV's columns."""
        field = self._inputs.get("data_path") or self._inputs.get("screen_path")
        if not isinstance(field, QLineEdit):
            return
        combo = self._inputs.get("target") or self._inputs.get("compound")
        if not isinstance(combo, QComboBox):
            return
        path = field.text().strip()
        if not path:
            return
        keep = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(auto)")
        for column in self._csv_columns(path):
            if column and column.lower() not in ("smiles",):
                combo.addItem(column)
        index = combo.findText(keep)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    # -- running ------------------------------------------------------------

    def _pick_out_dir(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        chosen = QFileDialog.getExistingDirectory(self, "Pick export folder")
        if chosen:
            self._out_dir.setText(chosen)

    def _log_line(self, text: str) -> None:
        self._log.append(text)

    def _remember_folder(self) -> None:
        """§17G: persist the current folder for this template."""
        from pharmaml.studio.workspace import remember_export_folder

        folder = self._out_dir.text().strip()
        if not folder:
            self._log_line("error: nothing to remember — pick a folder first")
            return
        if remember_export_folder(folder, self._current_name()):
            self._log_line(f"remembered export folder for "
                           f"{self._current_name()}: {folder}")

    def _run(self) -> None:
        name = self._current_name()
        out_dir = self._out_dir.text().strip()
        if not out_dir:
            self._log_line("error: pick an export folder first")
            return

        def text(key: str) -> str:
            field = self._inputs.get(key)
            return field.text().strip() if isinstance(field, QLineEdit) else ""

        def combo(key: str, auto: str | None = None) -> str | None:
            field = self._inputs.get(key)
            if isinstance(field, QComboBox):
                value = field.currentText().strip()
                if value == "(auto)":
                    return auto
                return value
            return auto

        def required(key: str, what: str) -> str:
            value = text(key)
            if not value:
                self._log_line(f"error: {what} is required — pick the file")
                return ""
            return value

        try:
            if name == "pk_pd_modeling":
                pipe = build_template(
                    name, out_dir=out_dir,
                    ka=float(self._inputs["ka"].value()),
                    ke=float(self._inputs["ke"].value()),
                    vd=float(self._inputs["vd"].value()),
                    dose=float(self._inputs["dose"].value()))
            elif name == "md_analysis":
                pipe = build_template(name, out_dir=out_dir,
                                      n_frames=int(
                                          self._inputs["n_frames"].value()),
                                      seed=0)
            elif name == "molecular_generation_screening":
                pipe = build_template(
                    name, out_dir=out_dir,
                    objective=str(self._inputs["objective"].currentText()),
                    generations=int(
                        self._inputs["generations"].value()), seed=42)
            elif name == "qsar_model_development":
                data = required("data_path", "the dataset CSV")
                if not data:
                    return
                pipe = build_template(
                    name, out_dir=out_dir, data_path=data,
                    target=combo("target"), models=None,
                    cv=5)
            elif name == "molecular_descriptor_generation":
                data = required("data_path", "the dataset CSV")
                if not data:
                    return
                pipe = build_template(
                    name, out_dir=out_dir, data_path=data,
                    smiles_col=combo("smiles_col", "smiles"),
                    kind="morgan")
            elif name == "drug_response_prediction":
                screen = required("screen_path", "the response screen CSV")
                if not screen:
                    return
                expr = required("expression_path",
                                "the expression CSV")
                if not expr:
                    return
                pipe = build_template(
                    name, out_dir=out_dir, screen_path=screen,
                    expression_path=expr,
                    compound=combo("compound"),
                    models=None, cv=int(self._inputs["cv"].value()))
            elif name == "docking_preparation_analysis":
                ligands = required("ligands_path", "the ligands CSV")
                if not ligands:
                    return
                receptor = required("receptor_pdbqt",
                                    "the receptor PDBQT")
                if not receptor:
                    return
                pipe = build_template(
                    name, out_dir=out_dir, ligands_path=ligands,
                    receptor_pdbqt=receptor,
                    smiles_col=combo("smiles_col", "SMILES"))
            else:
                raise ValueError(name)
        except Exception as exc:  # noqa: BLE001 - honest inline error
            self._log_line(f"error: {type(exc).__name__}: {exc}")
            return

        self._log_line(f"running {name} …")
        try:
            result = pipe.run(on_stage=lambda o: self._log_line(
                f"  {'ok ' if o.ok else 'FAIL'} {o.stage} "
                f"({o.seconds:.2f}s) {', '.join(o.produced)}"
                + (f" — {o.error}" if o.error else "")))
        except Exception as exc:  # noqa: BLE001 - honest inline error
            self._log_line(f"error: {type(exc).__name__}: {exc}")
            return
        record = json.loads(result.to_json())
        record_path = Path(out_dir) / f"{name}.run.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2))
        self._last_record = record
        self._log_line(
            f"done in {result.seconds:.2f}s — run record: {record_path}")
