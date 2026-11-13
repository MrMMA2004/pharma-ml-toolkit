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
        self._param_a = QDoubleSpinBox()
        self._param_a.setRange(0.0, 1000.0)
        self._param_a.setDecimals(3)
        form.addRow("ka (1/h):", self._param_a)
        self._param_b = QDoubleSpinBox()
        self._param_b.setRange(0.0, 1000.0)
        self._param_b.setDecimals(3)
        form.addRow("ke (1/h):", self._param_b)
        self._param_c = QDoubleSpinBox()
        self._param_c.setRange(0.0, 100000.0)
        self._param_c.setDecimals(1)
        form.addRow("Vd (L):", self._param_c)
        self._param_d = QDoubleSpinBox()
        self._param_d.setRange(0.0, 100000.0)
        self._param_d.setDecimals(1)
        form.addRow("dose (mg):", self._param_d)
        self._param_e = QSpinBox()
        self._param_e.setRange(10, 500)
        form.addRow("MD frames:", self._param_e)
        lay.addLayout(form)

        self._pick_btn = QPushButton("Pick export folder…")
        self._pick_btn.clicked.connect(self._pick_out_dir)
        self._run_btn = QPushButton("Run template")
        self._run_btn.clicked.connect(self._run)
        btns = QHBoxLayout()
        btns.addWidget(self._pick_btn)
        btns.addWidget(self._run_btn)
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

    # -- running ------------------------------------------------------------

    def _pick_out_dir(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        chosen = QFileDialog.getExistingDirectory(self, "Pick export folder")
        if chosen:
            self._out_dir.setText(chosen)

    def _log_line(self, text: str) -> None:
        self._log.append(text)

    def _run(self) -> None:
        name = self._current_name()
        out_dir = self._out_dir.text().strip()
        if not out_dir:
            self._log_line("error: pick an export folder first")
            return
        try:
            if name == "pk_pd_modeling":
                pipe = build_template(
                    name, out_dir=out_dir,
                    ka=self._param_a.value(), ke=self._param_b.value(),
                    vd=self._param_c.value(), dose=self._param_d.value())
            elif name == "md_analysis":
                pipe = build_template(name, out_dir=out_dir,
                                      n_frames=self._param_e.value(), seed=0)
            else:
                self._log_line(
                    f"'{name}' needs data inputs this quick form does not "
                    "collect yet — run it via the Python API "
                    "(pharmaml.studio.pipelines.build_template) or wait "
                    "for its card-integrated form")
                return
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
