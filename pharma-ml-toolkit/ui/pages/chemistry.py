"""Chemistry Workbench: real molecules in, real features out (16.8/16.12/16.14).

The Studio's increment-2 page (playground Chemistry tab, audit row 2):
pick a SMILES-bearing table (a bundled dataset or the one opened in
Data Explorer), standardize the column through ``pharmaml.chem.csv_standardize.standardize_csv``
-- the playground's own batch front door -- or compute a multi-family
feature matrix through ``pharmaml.chem.features.compute_feature_frame``
(the feature factory, checkpoint-wired to the cooperative-stop channel).

Both jobs run in a :class:`~ui.workers.job_worker.JobWorker`; the
feature job carries the v1.2.119 pre-flight batch guard: a 5-molecule
probe fixes the exact per-molecule column count, then the dense-matrix
estimate (rows x columns x 8 B x 3 copies) is checked against half of
``MemAvailable`` -- the same arithmetic the playground's cards use --
and the batch refuses with its numbers BEFORE the heavy compute.
Nothing on this page is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView, PandasTableModel
from ..widgets.njobs_field import NJobsField
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[2]

#: Bundled datasets that actually carry a SMILES column (probed on disk).
_BUILTIN = [
    "ligand_library.csv",
    "library_with_descriptors.csv",
    "qsar_classification.csv",
    "screening_library.csv",
    "seed_molecules.csv",
    "walkthrough_compounds.csv",
]

#: Family presets -> the compute_feature_frame ``families`` tuples.
_FAMILY_PRESETS: dict[str, tuple[str, ...] | None] = {
    "descriptors (fast)": ("descriptors",),
    "descriptors + MACCS + E-state": ("descriptors", "maccs", "estate"),
    "ECFP (Morgan bits)": ("ecfp",),
    "All families (heavy)": None,  # resolved to FEATURE_FAMILIES at run time
}

# -- the v1.2.119 batch-guard arithmetic (the playground's own constants) ------

_GUARD_MEM_FRACTION = 0.5  # half of MemAvailable: ranking/render copies follow
_MATRIX_COPIES = 3  # vstack -> DataFrame -> ranking hold the matrix ~3x


def _meminfo_mb(field_name: str) -> float | None:
    """Read a ``/proc/meminfo`` field in MB; ``None`` off-Linux."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field_name + ":"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _feature_matrix_bytes(molecules: int, columns: int) -> int:
    """The dense-matrix footprint estimate: rows x columns x 8 B x copies."""
    return int(molecules) * max(int(columns), 1) * 8 * _MATRIX_COPIES


@dataclass
class StandardizeOutcome:
    """What the standardize job hands back (all real records)."""

    summary: dict = field(default_factory=dict)
    frame: pd.DataFrame | None = None
    seconds: float = 0.0


@dataclass
class FeaturesOutcome:
    """What the feature-factory job hands back (all real numbers)."""

    frame: pd.DataFrame | None = None
    families: dict = field(default_factory=dict)
    n_total: int = 0
    n_valid: int = 0
    dropped: int = 0
    seconds: float = 0.0
    refusal: str = ""  # non-empty: the batch guard refused (arithmetic inside)
    # (part8 §7.4) the complete feature-generation config rides the
    # outcome: families, descriptor subset, radius/bits/count, n_jobs,
    # plus RDKit/Mordred versions — the reproducibility record.
    config_record: dict = field(default_factory=dict)


class ChemistryWorkbenchPage(QWidget):
    """Standardize a SMILES column; compute multi-family features."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
        data_explorer: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._data_explorer = data_explorer
        self._frame: pd.DataFrame | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self._dataset_combo = QComboBox()
        for name in _BUILTIN:
            self._dataset_combo.addItem(name)
        self._dataset_combo.addItem("(table from Data Explorer)")
        self._dataset_combo.currentTextChanged.connect(self._load_dataset)
        # review P-12 fix: a visible route for the user's own ligand table
        self._open_btn = QPushButton("Open CSV / Excel…")
        self._open_btn.setToolTip(
            "Load any ligand table from disk (a SMILES column is "
            "auto-detected from the header)"
        )
        self._open_btn.clicked.connect(self._open_external)
        self._smiles_combo = QComboBox()
        self._families_combo = QComboBox()
        self._families_combo.setToolTip("Descriptor families to compute (descriptors, ECFP fingerprints, counts…)")
        for label in _FAMILY_PRESETS:
            self._families_combo.addItem(label)
        self._jobs_field = NJobsField(default=-1)

        # (part8 Phase D) independent fingerprint configuration — the
        # Morgan-style families accept radius / bit width / count mode;
        # non-Morgan families ignore these (the backend enforces it too).
        self._fp_radius = QSpinBox()
        self._fp_radius.setRange(0, 6)
        self._fp_radius.setValue(2)
        self._fp_radius.setToolTip(
            "Morgan-family radius (ECFP / ECFP3 / Morgan / FCFP): the atom-"
            "environment diameter the fingerprint sees. 2 is the ECFP4 "
            "convention; non-Morgan families ignore this."
        )
        self._fp_bits = QSpinBox()
        self._fp_bits.setRange(32, 65_536)
        self._fp_bits.setSingleStep(512)
        self._fp_bits.setValue(2048)
        self._fp_bits.setToolTip(
            "Fold length (bit count) for the Morgan-style families. "
            "Non-Morgan families keep their fixed widths."
        )
        self._fp_counts = QCheckBox("count fingerprints")
        self._fp_counts.setToolTip(
            "Emit COUNT vectors instead of binary bits for the Morgan-"
            "style families with a count flavour (values > 1 where an "
            "environment recurs)."
        )

        # (part8 Phase D) exact individual descriptor selection — the
        # picker lists the RDKit property catalog; empty selection means
        # ALL properties (the historical behaviour).
        self._descriptor_picker = QComboBox()
        self._descriptor_picker.setToolTip(
            "Descriptor-family scope: all RDKit properties (historical) or "
            "an exact subset picked in the dialog."
        )
        self._std_btn = QPushButton("Standardize column")
        self._std_btn.setToolTip("Standardize SMILES (canonical form, salts stripped per the RDKit pipeline)")
        self._std_btn.setMinimumHeight(34)
        self._std_btn.clicked.connect(self._start_standardize)
        self._feat_btn = QPushButton("Compute features")
        self._feat_btn.setToolTip("Compute the descriptor/fingerprint frame for the SMILES column")
        self._feat_btn.setMinimumHeight(34)
        self._feat_btn.clicked.connect(self._start_features)
        form.addRow("Dataset:", self._dataset_combo)
        form.addRow("", self._open_btn)
        form.addRow("SMILES column:", self._smiles_combo)
        form.addRow("Feature families:", self._families_combo)
        form.addRow("Descriptor scope:", self._descriptor_picker)
        fp_row = QHBoxLayout()
        fp_row.addWidget(self._fp_radius)
        fp_row.addWidget(QLabel("bits:"))
        fp_row.addWidget(self._fp_bits)
        fp_row.addWidget(self._fp_counts)
        form.addRow("Fingerprint radius/bits:", fp_row)
        form.addRow("Threads (n_jobs):", self._jobs_field)
        buttons = QHBoxLayout()
        buttons.addWidget(self._std_btn)
        buttons.addWidget(self._feat_btn)
        form.addRow("", buttons)
        lay.addLayout(form)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(QLabel("Feature distribution:"))
        self._column_combo = QComboBox()
        self._column_combo.currentIndexChanged.connect(self._plot_column)
        right_lay.addWidget(self._column_combo)
        self._plot = PlotPanel(title="Feature histogram")
        self._plot.set_empty_hint("Load a ligand table and press Compute features — the descriptor histogram lands here")

        right_lay.addWidget(self._plot, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([680, 420])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No chemistry run yet.")
        # (round 12) the requested molecule count behind the pending
        # feature batch -- the preserved-partial line reports against it
        self._pending_n_total = 0
        # (part8 Phase D) the descriptor-scope picker fills lazily —
        # RDKit's property catalog is not worth importing at page build
        # on machines that only browse datasets.
        self._fill_descriptor_picker()
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

        # (R30-c) the per-page chart export under the live palette —
        # created after _report so the status line exists to receive it
        from ..widgets.chart_export import attach_chart_export

        export_row = QHBoxLayout()
        attach_chart_export(
            self, self._plot, "chemistry-features",
            layout=export_row, status_label=self._report,
        )
        export_row.addStretch(1)
        right_lay.addLayout(export_row)

        self._load_dataset(self._dataset_combo.currentText())

    # -- data ---------------------------------------------------------------
    def _open_external(self) -> None:
        """Review P-12 fix: load the user's own ligand table directly."""
        from PySide6.QtWidgets import QFileDialog

        path, _filt = QFileDialog.getOpenFileName(
            self,
            "Open a ligand table",
            str(_TK / "examples" / "data"),
            "Tables (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*)",
        )
        if not path:
            return

        # (part8 Phase B) the read runs on the worker — the GUI thread
        # never parses a user ligand table, however large.
        if self._runner.busy:
            self._report.setText(
                f"Busy: '{getattr(self._runner, 'job_name', '') or 'another operation'}' "
                "is running — Stop it (or check Job Center) first."
            )
            return
        self._open_btn.setEnabled(False)

        def job(progress) -> pd.DataFrame:
            from pharmaml.utils.io import read_table_with_progress

            def report(done, total=None, _note=""):
                if total:
                    progress(min(done / total, 1.0), f"{done:,} rows")
                return None

            progress(0.05, "reading")
            return read_table_with_progress(Path(path), report)

        def done(frame) -> None:
            self._open_btn.setEnabled(True)
            if frame is None or frame.empty:
                self._report.setText(f"{Path(path).name} is empty.")
                return
            label = f"{Path(path).name} (file)"
            names = [self._dataset_combo.itemText(i) for i in range(self._dataset_combo.count())]
            if label not in names:
                self._dataset_combo.addItem(label)
            # hold the frame: _load_dataset re-runs on setCurrentText and the
            # "(file)" branch below picks this up instead of hitting examples/
            self._external_frame = frame
            self._dataset_combo.setCurrentText(label)
            self._report.setText(
                f"Loaded {Path(path).name} — {len(frame):,} rows; SMILES column auto-detected."
            )

        def failed(message) -> None:
            self._open_btn.setEnabled(True)
            self._report.setText(
                f"Could not read {Path(path).name}: {message.splitlines()[0]}"
            )

        def cancelled(_seconds) -> None:
            self._open_btn.setEnabled(True)

        self._runner.submit(f"load {Path(path).name}", job, done, failed, on_cancelled=cancelled)

    def _load_dataset(self, name: str) -> None:
        self._frame = None
        self._smiles_combo.clear()
        if name == "(table from Data Explorer)":
            if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
                self._frame = self._data_explorer.frame()
            if self._frame is None or self._frame.empty:
                self._report.setText("Data Explorer has no table loaded yet.")
                return
        elif name.endswith(" (file)") and getattr(self, "_external_frame", None) is not None:
            # review P-12: the user-loaded table, already in memory
            self._frame = self._external_frame
        else:
            path = _TK / "examples" / "data" / name
            if not path.exists():  # pragma: no cover - missing example
                self._report.setText(f"Example dataset missing on disk: {name}")
                return
            from pharmaml.utils.io import read_table

            self._frame = read_table(path)
        from pharmaml.chem.csv_standardize import detect_smiles_column

        # review C-3.4: detect_smiles_column RAISES DataValidationError on a
        # table without a molecular column (an assay table is a legitimate
        # thing to load); an uncaught raise inside this combo-refresh signal
        # handler would silently break the page. Model Analysis solved the
        # identical problem with a None-returning wrapper — same here: no
        # match leaves the pick unauto-populated with a reason in the status
        # line, every column still offered for manual choice.
        try:
            detected = detect_smiles_column(self._frame)
        except Exception:  # noqa: BLE001 - the engine's validation, surfaced
            detected = None
        if detected is not None:
            self._smiles_combo.addItem(detected)
        else:
            self._report.setText(
                "No SMILES-like column found — pick one manually "
                "(load a ligand table, or standardize a SMILES column first)."
            )
        for col in self._frame.columns:
            if str(col) != self._smiles_combo.currentText():
                self._smiles_combo.addItem(str(col))

    # -- job plumbing ---------------------------------------------------------
    def _on_started(self, name: str) -> None:
        self._std_btn.setEnabled(False)
        self._feat_btn.setEnabled(False)
        self._report.setText(f"Running {name}… on {self._dataset_combo.currentText()}")

    def _on_ended(self) -> None:
        self._std_btn.setEnabled(True)
        self._feat_btn.setEnabled(True)

    def _descriptor_selection(self) -> tuple[str, ...] | None:
        """(part8 Phase D) the picker's exact subset, or None = all.

        Scope 0 ('all properties') is the historical behaviour; any other
        index means the user picked exactly those catalog names (the
        picker stores the tuple in its item data).
        """
        data = self._descriptor_picker.currentData()
        return tuple(data) if data else None

    def _fill_descriptor_picker(self) -> None:
        """Fill the descriptor-scope picker (lazy: needs RDKit)."""
        combo = self._descriptor_picker
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("all properties (43)", None)
        try:
            from pharmaml.chem.features import descriptor_catalog

            names = descriptor_catalog()
            combo.setItemText(0, f"all properties ({len(names)})")
            for name in names:
                combo.addItem(name, [name])
        except Exception:  # noqa: BLE001 - RDKit absent: scope stays 'all'
            pass
        combo.blockSignals(False)
        combo.setCurrentIndex(0)

    def _smiles(self) -> list[str] | None:
        """The selected column as a plain SMILES list (None when unusable)."""
        if self._frame is None:
            self._report.setText("Load a dataset first.")
            return None
        col = self._smiles_combo.currentText()
        if not col or col not in self._frame.columns:
            self._report.setText("Pick a SMILES column.")
            return None
        return self._frame[col].astype(str).tolist()

    # -- standardize ------------------------------------------------------------
    def _start_standardize(self) -> None:
        smiles = self._smiles()
        if smiles is None:
            return
        col = self._smiles_combo.currentText()
        if self._runner.busy:
            self._report.setText(f"Busy: '{getattr(self._runner, 'job_name', '') or 'another operation'}' is running — Stop it (or check Job Center) first.")
            return
        frame = self._frame

        def job(progress) -> StandardizeOutcome:
            from pharmaml.chem.csv_standardize import standardize_csv

            progress(0.05, "standardizing")
            # standardize_csv is checkpoint-less: it ignores the callback
            # (the honest "no checkpoints" story, exactly like the card).
            report = standardize_csv(frame, smiles_column=col)
            progress(1.0, "done")
            return StandardizeOutcome(
                summary=report.summary(),
                frame=report.to_frame(),
                seconds=float(report.elapsed),
            )

        self._runner.submit(
            "SMILES standardization",
            job,
            on_finished=self._standardize_done,
            on_failed=self._failed,
        )

    def _standardize_done(self, outcome: object) -> None:
        assert isinstance(outcome, StandardizeOutcome)
        self._table.bind_frame(outcome.frame)
        parts = [f"{k} {v}" for k, v in outcome.summary.items()]
        self._report.setText("Standardization — " + " · ".join(parts))

    # -- feature factory ----------------------------------------------------------
    def _start_features(self) -> None:
        smiles = self._smiles()
        if smiles is None:
            return
        if self._runner.busy:
            self._report.setText(f"Busy: '{getattr(self._runner, 'job_name', '') or 'another operation'}' is running — Stop it (or check Job Center) first.")
            return
        # (round 12) the preserved-partial line reports against this
        # requested count when a cooperative stop lands mid-batch
        self._pending_n_total = len(smiles)
        preset = self._families_combo.currentText()
        families = _FAMILY_PRESETS[preset]
        # (part8 Phase 6) manual n_jobs: any nonzero integer the user
        # typed; 0/garbage is refused BEFORE the job starts (the backend
        # would raise the same error after the button went dark).
        n_jobs = self._jobs_field.value()
        if n_jobs is None:
            self._report.setText(
                "n_jobs must be a nonzero integer (e.g. -1 for all cores); "
                f"got {self._jobs_field.currentText().strip()!r}"
            )
            return
        # (part8 Phase D) the structured generation config: family list +
        # exact descriptor subset + Morgan radius/bits/count.  Built and
        # validated on the GUI thread so a bad selection refuses BEFORE
        # the worker spins up; the SAME object flows to the backend and
        # is persisted with the outcome (§7.4).
        fp_radius = int(self._fp_radius.value())
        fp_bits = int(self._fp_bits.value())
        fp_counts = bool(self._fp_counts.isChecked())
        descriptor_subset = self._descriptor_selection()
        from pharmaml.chem.features import FeatureConfig

        try:
            feature_config = FeatureConfig(
                families=tuple(
                    families if families is not None else __import__(
                        "pharmaml.chem", fromlist=["FEATURE_FAMILIES"]
                    ).FEATURE_FAMILIES
                ),
                descriptors=descriptor_subset,
                morgan_radius=fp_radius,
                morgan_bits=fp_bits,
                as_counts=fp_counts,
                n_jobs=n_jobs,
            )
        except ValueError as exc:
            self._report.setText(f"Feature configuration refused: {exc}")
            return

        def job(progress) -> FeaturesOutcome:
            import time as _time

            from pharmaml.chem.features import compute_feature_frame

            # A distinct local name: assigning the captured ``families``
            # would make it a job-local everywhere (UnboundLocalError).
            requested = families
            if requested is None:  # the "All families (heavy)" preset
                from pharmaml.chem import FEATURE_FAMILIES

                requested = FEATURE_FAMILIES
            started = _time.monotonic()
            progress(0.02, "probing")
            # Probe: the exact per-molecule column count from <=5 real
            # molecules, then the batch-guard arithmetic BEFORE the heavy
            # compute (the playground card's guard, backgrounded).
            probe = [s for s in smiles if s.strip()][:10]
            if not probe:
                return FeaturesOutcome(refusal="no parseable input: the SMILES column is empty")
            _probe_frame, probe_report = compute_feature_frame(
                probe[:5], config=feature_config
            )
            if probe_report.n_valid == 0:
                return FeaturesOutcome(
                    refusal="none of the first 5 SMILES parsed — check the column"
                )
            columns = probe_report.total_columns
            available = _meminfo_mb("MemAvailable")
            if available is not None:
                estimate = _feature_matrix_bytes(len(smiles), columns)
                budget = _GUARD_MEM_FRACTION * available * 1024 * 1024
                if estimate > budget:
                    return FeaturesOutcome(
                        refusal=(
                            f"batch refused by the memory guard: the dense matrix alone "
                            f"would need ≈ {estimate / 1024**3:.1f} GB ({len(smiles):,} "
                            f"molecules × {columns:,} columns × 8 B × {_MATRIX_COPIES} copies) "
                            f"but only {available:,.0f} MB is available and the guard budget "
                            f"is half of that"
                        )
                    )
            progress(0.05, "computing")

            def adapt(done, total=None, note=""):
                # the factory speaks (done, total, note); the Studio speaks
                # (fraction, label); the RETURN propagates the stop channel.
                fraction = (done / total) if total else 0.0
                return progress(fraction, note)

            frame, report = compute_feature_frame(
                smiles, config=feature_config, progress=adapt
            )
            seconds = _time.monotonic() - started
            progress(1.0, "done")
            record = feature_config.as_dict()
            try:  # §7.4: library versions where applicable
                import rdkit

                record["rdkit_version"] = rdkit.__version__
            except Exception:  # noqa: BLE001 - version is best-effort
                pass
            record["n_descriptors_selected"] = (
                len(descriptor_subset) if descriptor_subset is not None else "all"
            )
            record["dropped_molecules"] = len(report.dropped)
            record["feature_columns"] = list(frame.columns)[:200]
            record["n_feature_columns"] = int(frame.shape[1])
            return FeaturesOutcome(
                frame=frame,
                families=dict(report.families),
                n_total=len(smiles),
                n_valid=report.n_valid,
                dropped=len(report.dropped),
                seconds=seconds,
                config_record=record,
            )

        name = f"feature factory ({preset})"
        self._runner.submit(
            name,
            job,
            on_finished=self._features_done,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_preserved=self._features_preserved,
        )

    def _features_done(self, outcome: object) -> None:
        assert isinstance(outcome, FeaturesOutcome)
        if outcome.refusal:
            self._table.bind_frame(None, note="Batch refused — see the report line")
            self._report.setText(outcome.refusal)
            return
        assert outcome.frame is not None  # refusal is the only frame-less finish
        self._table.bind_frame(outcome.frame)
        numeric = [str(c) for c in outcome.frame.columns]
        listed = numeric[:400]
        self._column_combo.blockSignals(True)
        self._column_combo.clear()
        self._column_combo.addItems(listed)
        self._column_combo.blockSignals(False)
        if listed:
            self._plot_column(0)
        families = ", ".join(f"{k} {v}" for k, v in outcome.families.items())
        note = f" · showing first {len(listed)} columns" if len(numeric) > len(listed) else ""
        self._report.setText(
            f"Features — {outcome.n_valid}/{outcome.n_total} molecules kept · "
            f"{outcome.dropped} dropped · {families} · {outcome.seconds:.1f}s{note}"
        )

    # -- histogram -------------------------------------------------------------
    def _plot_column(self, index: int) -> None:
        frame = self.model_frame()
        if frame is None or index < 0:
            return
        name = self._column_combo.currentText()
        if not name or name not in frame.columns:
            return
        series = pd.to_numeric(frame[name], errors="coerce").dropna()
        if series.empty:
            return
        values = series.to_numpy(dtype=float, copy=False)
        self._plot.clear()
        self._plot.histogram(values, bins=min(40, max(10, int(len(values) ** 0.5))))
        self._plot.set_labels(x=name, y="count", title=f"Distribution of {name}")

    # -- failures / cancellation / theming -----------------------------------------
    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "job failed"
        self._report.setText(f"Job failed — {first}")

    def _cancelled(self, seconds: float) -> None:
        self._report.setText(f"Cancelled after {seconds:.0f} s (cooperative stop).")

    def _features_preserved(self, partial: object) -> None:
        """(round 12, §12/§13) surface what the descriptor batch preserved.

        The R6 contract attaches the partial ``(frame, report)`` to
        ``JobCancelled`` on a cooperative stop; the runner's
        ``preserved`` channel hands it to the page.  The partial frame
        is bound to the table (the completed rows are real, validated
        feature vectors) and the report line states exactly how many
        of the requested molecules survived the stop -- honest
        partial evidence, never a silent drop.
        """
        import pandas as pd

        if not isinstance(partial, tuple) or len(partial) != 2:
            return  # not this backend's payload: nothing to surface
        frame, report = partial
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return  # nothing had completed before the stop
        self._table.bind_frame(frame, note="Partial batch — stopped by request")
        self._report.setText(
            f"Cancelled — kept the {len(frame)} completed molecule(s) "
            f"({report.n_valid} valid, {len(report.dropped)} dropped) of "
            f"{self._pending_n_total} requested."
        )

    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    # -- test/inspection surface --------------------------------------------------
    def model_frame(self) -> pd.DataFrame | None:
        """The frame currently bound to the result table (or None)."""
        model = self._table.model()
        return model.frame if isinstance(model, PandasTableModel) else None

    def report_text(self) -> str:
        return self._report.text()
