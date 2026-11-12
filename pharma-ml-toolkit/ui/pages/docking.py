"""Docking Workbench: campaigns, clusters, DOCKSTRING boxes, libraries, contacts (audit row 5).

The Studio's increment-5 page (the playground Docking tab). Five of the
tab's engines, each wired to the SAME public backend APIs the cards
call -- nothing on this page is fabricated, and none of these workflows
execute an engine. The source/ZIP includes Windows engine binaries; actual
execution requires a compatible runtime and appropriate inputs:

- **Campaign summary** -- ``check_docking_completion`` +
  ``scan_docking_directory`` over the bundled synthetic AutoDock4
  campaign (14 ligand folders: 12 complete runs, one missing log, one
  run interrupted before clustering), then the rescore card's
  arithmetic on the OK rows: heavy-atom counts from the campaign's
  ``ligands.csv`` (RDKit), ``ligand_efficiency_series`` and
  ``consensus_rescore`` over the three evidence columns
  (``best_lowest_energy``, ``best_num_in_cluster``,
  ``ligand_efficiency``; more negative energy = better, exactly the
  engine's default direction). The ranking lands in the shared table,
  best-first.
- **Cluster analysis** -- ``parse_dlg_cluster_table`` on the chosen
  folder's ``r.dlg``: the CLUSTERING HISTOGRAM as a real table
  (rank / lowest / mean / #in-cluster) with the top-cluster and
  most-populated-cluster story in the report. The folder combo is fed
  by the campaign scan's OK folders (cross-tab state, GUI-thread
  writes only).
- **Ligand library** -- ``load_ligand_csv`` on the bundled library
  (8 valid + 2 deliberately invalid rows, whose reasons ride the
  table), plus the RDKit property columns the preparation card
  reports (heavy atoms, rotatable bonds) on the valid rows.
- **DOCKSTRING lab** -- ``target_frame`` (all 58 upstream Vina search
  boxes), ``get_target`` config previews, and ``parse_search_box``
  validation of a custom six-key config -- the same registry the
  playground's DOCKSTRING card prints.
- **Contacts & PLIF** -- the pose-level reading (r264): the real
  1AKI lysozyme (``read_pdb_atoms``) meets the bundled synthetic pose
  (``read_pdbqt_atoms``, ten clash-free atoms in the binding cleft,
  REMARK-provenance), ``compute_contacts`` classifies the 5 A shell,
  and the twin cards answer *where* and *how* -- the contact map
  (class-coloured scatter over the residue × ligand-atom grid, the
  inspector tooltip carries the exact pair + distance) and the PLIF
  fingerprint (grouped per-class bars per residue, the table rides the
  counts + the 0/1 bits twin).

All five heavy paths run in a :class:`~ui.workers.job_worker.JobWorker`.
They are single-shot engine calls without checkpoint loops, so the
progress channel reports honestly but the cooperative Stop only lands
for checkpoint-wired jobs -- the same disclosure the PK/PD and Biology
pages carry.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView, PandasTableModel
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[2]

#: The bundled synthetic AutoDock4 campaign: 14 ligand folders, each a
#: complete run holding ``r.dlg`` + ``r.glg`` except the audit cases
#: (08_benzocaine: no logs at all, 10_nicotine: interrupted before
#: clustering).
_CAMPAIGN_PATH = _TK / "examples" / "data" / "docking_campaign"

#: The ligand library the preparation card loads (8 valid SMILES +
#: 2 deliberately invalid rows).
_LIBRARY_PATH = _TK / "examples" / "data" / "ligand_library.csv"

#: The contacts & PLIF twin's receptor side: the real hen egg-white
#: lysozyme (PDB entry 1AKI, 1.5 A, public domain; the same file the
#: docking notebook prepares in its section 2).
_RECEPTOR_PDB = _TK / "examples" / "data" / "lysozyme_1aki.pdb"

#: The twin's pose side: ten synthetic atoms placed clash-free (>= 2.4 A)
#: at 2.9-4.2 A from real binding-cleft atoms -- a documented geometry
#: demo (the twin of the synthetic r.dlg campaign), REMARK-provenance
#: inside the file itself.
_POSE_PDBQT = _TK / "examples" / "data" / "contacts_demo_ligand.pdbqt"

#: The rescore card's evidence columns, in its own order.
_EVIDENCE_COLUMNS = ("best_lowest_energy", "best_num_in_cluster", "ligand_efficiency")


@dataclass
class CampaignOutcome:
    """What the campaign-scan job hands back (all real numbers)."""

    completion: pd.DataFrame = field(default_factory=pd.DataFrame)
    scan: pd.DataFrame = field(default_factory=pd.DataFrame)
    ranked: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_folders: int = 0
    n_ok: int = 0
    n_missing: int = 0
    n_incomplete: int = 0
    top_ligand: str = ""
    top_score: float = 0.0
    seconds: float = 0.0


@dataclass
class ClusterOutcome:
    """What the cluster-table job hands back for one folder."""

    clusters: pd.DataFrame = field(default_factory=pd.DataFrame)
    folder: str = ""
    top_energy: float = 0.0
    top_runs: int = 0
    best_rank: int = 0
    best_runs: int = 0
    seconds: float = 0.0


@dataclass
class RegistryOutcome:
    """What the DOCKSTRING registry job hands back."""

    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    objectives: tuple[str, ...] = ()
    first: str = ""
    last: str = ""
    seconds: float = 0.0


@dataclass
class ContactsOutcome:
    """What the contact-map job hands back (all real numbers).

    ``series`` holds one entry per interaction class -- the class name
    plus its (x, y) point arrays over the residue x ligand-atom grid
    and the per-point tooltip rows -- so the GUI thread only draws.
    """

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    series: tuple[tuple[str, object, object, list[str]], ...] = ()
    residues: tuple[str, ...] = ()
    ligand_atoms: tuple[str, ...] = ()
    n_contacts: int = 0
    n_residues: int = 0
    closest_pair: str = ""
    closest_distance: float = 0.0
    closest_interaction: str = ""
    cutoff: float = 5.0
    seconds: float = 0.0


@dataclass
class PlifOutcome:
    """What the PLIF job hands back (the counts and their 0/1 twin)."""

    counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    bits: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: tuple[str, ...] = ()
    class_totals: tuple[int, ...] = ()
    n_contacts: int = 0
    cutoff: float = 5.0
    seconds: float = 0.0


@dataclass
class LibraryOutcome:
    """What the library-audit job hands back."""

    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    invalid_names: list[str] = field(default_factory=list)
    seconds: float = 0.0


class DockingWorkbenchPage(QWidget):
    """Scan a campaign; parse clusters; load libraries; browse DOCKSTRING."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)

        # cross-tab state (written only from the GUI thread, in the
        # finished handlers; the jobs capture their inputs at submit)
        self._scan: pd.DataFrame | None = None
        self._ok_frame: pd.DataFrame | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_campaign_tab(), "Campaign summary")
        self._tabs.addTab(self._build_cluster_tab(), "Cluster analysis")
        self._tabs.addTab(self._build_library_tab(), "Ligand library")
        self._tabs.addTab(self._build_dockstring_tab(), "DOCKSTRING lab")
        self._tabs.addTab(self._build_contacts_tab(), "Contacts & PLIF")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Docking panel")
        splitter.addWidget(self._plot)
        splitter.setSizes([600, 500])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No docking run yet — scan the bundled campaign first.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

    # -- tab construction -----------------------------------------------------
    def _build_campaign_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._campaign_combo = QComboBox()
        self._campaign_combo.addItem("docking_campaign/ (bundled)")
        form.addRow("campaign:", self._campaign_combo)
        self._campaign_btn = QPushButton("Scan campaign & rescore")
        self._campaign_btn.setMinimumHeight(34)
        self._campaign_btn.clicked.connect(self._start_campaign)
        form.addRow("", self._campaign_btn)
        self._wizard_btn = QPushButton("Review a Vina campaign…")
        self._wizard_btn.clicked.connect(self._show_campaign_wizard)
        form.addRow("", self._wizard_btn)
        hint = QLabel(
            "check_docking_completion + scan_docking_directory over the "
            "bundled 14-folder AutoDock4 campaign, then the rescore card's "
            "arithmetic on the OK rows: heavy atoms from ligands.csv (RDKit), "
            "ligand efficiency, and the 3-column consensus (lower energy = "
            "better). The ranking lands best-first."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_cluster_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._cluster_ligand = QComboBox()
        form.addRow("ligand folder:", self._cluster_ligand)
        self._cluster_btn = QPushButton("Parse cluster table")
        self._cluster_btn.setMinimumHeight(34)
        self._cluster_btn.clicked.connect(self._start_cluster)
        form.addRow("", self._cluster_btn)
        hint = QLabel(
            "parse_dlg_cluster_table on the chosen folder's r.dlg: the "
            "CLUSTERING HISTOGRAM (rank, lowest/mean energy, #in-cluster), "
            "the top cluster and the most populated one. The folders load "
            "with the campaign scan."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_library_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._library_combo = QComboBox()
        self._library_combo.addItem("ligand_library.csv (bundled)")
        form.addRow("library:", self._library_combo)
        self._library_btn = QPushButton("Load & audit library")
        self._library_btn.setMinimumHeight(34)
        self._library_btn.clicked.connect(self._start_library)
        form.addRow("", self._library_btn)
        hint = QLabel(
            "load_ligand_csv (RDKit-validated SMILES) on the bundled "
            "library -- the two invalid rows ride the table with their "
            "reasons -- plus the RDKit properties the preparation card "
            "reports (heavy atoms, rotatable bonds) on the valid rows."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_dockstring_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._registry_btn = QPushButton("Load target registry (58 boxes)")
        self._registry_btn.setMinimumHeight(34)
        self._registry_btn.clicked.connect(self._start_registry)
        form.addRow("", self._registry_btn)
        self._target_combo = QComboBox()
        form.addRow("target:", self._target_combo)
        self._config_btn = QPushButton("Show target config")
        self._config_btn.setMinimumHeight(34)
        self._config_btn.clicked.connect(self._show_target_config)
        form.addRow("", self._config_btn)
        self._box_edit = QPlainTextEdit()
        self._box_edit.setMaximumHeight(120)
        self._box_edit.setPlaceholderText("paste a six-key Vina config…")
        form.addRow("custom box:", self._box_edit)
        self._box_btn = QPushButton("Validate custom box")
        self._box_btn.setMinimumHeight(34)
        self._box_btn.clicked.connect(self._validate_box)
        form.addRow("", self._box_btn)
        hint = QLabel(
            "The DOCKSTRING registry (target_frame, all 58 upstream Vina "
            "search boxes) with per-target config previews, and "
            "parse_search_box validation of a pasted six-key config -- "
            "center_x/y/z + size_x/y/z, positive sizes, finite values."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_contacts_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._receptor_combo = QComboBox()
        self._receptor_combo.addItem("lysozyme_1aki.pdb (real 1AKI, bundled)")
        form.addRow("receptor:", self._receptor_combo)
        self._pose_combo = QComboBox()
        self._pose_combo.addItem("contacts_demo_ligand.pdbqt (synthetic pose, bundled)")
        form.addRow("pose:", self._pose_combo)
        self._cutoff_spin = QDoubleSpinBox()
        self._cutoff_spin.setDecimals(1)
        self._cutoff_spin.setRange(3.0, 8.0)
        self._cutoff_spin.setSingleStep(0.5)
        self._cutoff_spin.setValue(5.0)
        self._cutoff_spin.setToolTip(
            "the contact shell in angstroms -- the classification bands "
            "(salt 4.0 / H-bond 3.5 / hydrophobic 4.5) stay fixed")
        form.addRow("contact shell (A):", self._cutoff_spin)
        self._contacts_btn = QPushButton("Compute contact map")
        self._contacts_btn.setMinimumHeight(34)
        self._contacts_btn.clicked.connect(self._start_contacts)
        form.addRow("", self._contacts_btn)
        self._plif_btn = QPushButton("Build PLIF fingerprint")
        self._plif_btn.setMinimumHeight(34)
        self._plif_btn.clicked.connect(self._start_plif)
        form.addRow("", self._plif_btn)
        hint = QLabel(
            "The pose-level reading: read_pdb_atoms on the real lysozyme "
            "+ read_pdbqt_atoms on the bundled synthetic pose (ten atoms "
            "placed clash-free in the 1AKI cleft, REMARK-provenance in "
            "the file), compute_contacts classifies every pair in the "
            "shell (salt bridge / H-bond / hydrophobic / vdw distance "
            "bands), the map is one class-coloured point per contact over "
            "the residue x ligand-atom grid (hover = the exact pair), and "
            "the PLIF twin aggregates it per residue -- the 0/1 bits are "
            "the fingerprint a pose comparison would Tanimoto over."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- job plumbing -----------------------------------------------------------
    def _on_started(self, name: str) -> None:
        for btn in (
            self._campaign_btn,
            self._cluster_btn,
            self._library_btn,
            self._registry_btn,
            self._contacts_btn,
            self._plif_btn,
        ):
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in (
            self._campaign_btn,
            self._cluster_btn,
            self._library_btn,
            self._registry_btn,
            self._contacts_btn,
            self._plif_btn,
        ):
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return True
        return False

    def _show_campaign_wizard(self) -> None:
        from .campaign_wizard import CampaignReviewWizard

        wizard = getattr(self, "_campaign_wizard", None)
        if wizard is None:
            wizard = CampaignReviewWizard(self._runner, self)
            self._campaign_wizard = wizard
        wizard.show()
        wizard.raise_()
        wizard.activateWindow()

    # -- campaign summary --------------------------------------------------------------
    def _start_campaign(self) -> None:
        if self._busy_check():
            return
        campaign_path = _CAMPAIGN_PATH

        def job(progress) -> CampaignOutcome:
            from rdkit import Chem

            from pharmaml.docking import (
                check_docking_completion,
                consensus_rescore,
                ligand_efficiency_series,
                scan_docking_directory,
            )

            started = _time.monotonic()
            progress(0.15, "auditing the campaign tree")
            completion = check_docking_completion(campaign_path)
            counts = {str(k): int(v) for k, v in completion["status"].value_counts().items()}
            progress(0.35, "scanning the DLG summaries")
            scan = scan_docking_directory(campaign_path)
            ok = scan[scan["status"] == "OK"].copy().reset_index(drop=True)
            progress(0.55, "heavy atoms from the campaign's ligands.csv")
            ligands = pd.read_csv(campaign_path / "ligands.csv")
            heavy: dict[str, int] = {}
            for _, row in ligands.iterrows():
                mol = Chem.MolFromSmiles(str(row["SMILES"]))
                heavy[str(row["name"])] = 0 if mol is None else int(mol.GetNumHeavyAtoms())
            ok["ligand"] = ok["folder"].str.split("_", n=1).str[1]
            ok["num_heavy_atoms"] = ok["ligand"].map(heavy)
            progress(0.75, "ligand efficiency + consensus rescore")
            ok["ligand_efficiency"] = ligand_efficiency_series(
                ok, affinity_col="best_lowest_energy", atoms_col="num_heavy_atoms"
            )
            ok["consensus"] = consensus_rescore(
                ok,
                list(_EVIDENCE_COLUMNS),
                higher_is_better={
                    "best_lowest_energy": False,
                    "best_num_in_cluster": True,
                    "ligand_efficiency": True,
                },
            )
            ranked = ok.sort_values("consensus", ascending=False).reset_index(drop=True)
            top = ranked.iloc[0]
            progress(1.0, "done")
            return CampaignOutcome(
                completion=completion,
                scan=scan,
                ranked=ranked,
                n_folders=int(completion["status"].size),
                n_ok=counts.get("OK", 0),
                n_missing=counts.get("MISSING_LOG", 0),
                n_incomplete=counts.get("INCOMPLETE", 0),
                top_ligand=str(top["ligand"]),
                top_score=float(top["consensus"]),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "campaign scan & rescore",
            job,
            on_finished=self._campaign_done,
            on_failed=self._failed,
        )

    def _campaign_done(self, outcome: object) -> None:
        assert isinstance(outcome, CampaignOutcome)
        # part5 §11: the parsed campaign becomes a saved run (best-effort).
        try:
            from pharmaml.studio import record_run

            record_run(
                "docking",
                "campaign scan",
                {
                    "n_folders": int(outcome.n_folders),
                    "n_ok": int(outcome.n_ok),
                    "n_missing": int(outcome.n_missing),
                    "n_incomplete": int(outcome.n_incomplete),
                    "top_ligand": str(outcome.top_ligand),
                    "top_score": float(outcome.top_score),
                },
            )
        except Exception:  # noqa: BLE001, S110 -- the record must never fail the render
            pass
        # cross-tab state, written on the GUI thread only
        self._scan = outcome.scan
        self._ok_frame = outcome.ranked
        folders = [str(v) for v in outcome.ranked["folder"]]
        self._cluster_ligand.blockSignals(True)
        self._cluster_ligand.clear()
        self._cluster_ligand.addItems(folders)
        self._cluster_ligand.blockSignals(False)

        self._table.bind_frame(outcome.ranked)
        self._plot.clear()
        self._plot.bars(
            outcome.ranked["consensus"].to_numpy(dtype=float),
            labels=[str(v) for v in outcome.ranked["ligand"]],
        )
        self._plot.set_labels(
            x="ligand",
            y="consensus score (0-1, higher = better)",
            title=f"Campaign ranking · {outcome.n_ok} complete runs of {outcome.n_folders} folders",
        )
        self._report.setText(
            f"Campaign — {outcome.n_folders} folders · {outcome.n_ok} OK / "
            f"{outcome.n_missing} missing log / {outcome.n_incomplete} incomplete · "
            f"consensus over {len(_EVIDENCE_COLUMNS)} evidence columns · "
            f"top: {outcome.top_ligand} ({outcome.top_score:.3f}) · "
            f"{outcome.seconds:.1f}s"
        )

    # -- cluster analysis ------------------------------------------------------------
    def _start_cluster(self) -> None:
        if self._scan is None:
            self._report.setText(
                "Scan a campaign first — the cluster tab parses its DLG logs."
            )
            return
        folder = self._cluster_ligand.currentText()
        if not folder:
            self._report.setText("Pick a ligand folder (the scan's OK folders load with it).")
            return
        if self._busy_check():
            return
        log_path = _CAMPAIGN_PATH / folder / "r.dlg"

        def job(progress) -> ClusterOutcome:
            from pharmaml.docking import parse_dlg_cluster_table

            started = _time.monotonic()
            progress(0.4, "parsing the clustering histogram")
            clusters = parse_dlg_cluster_table(log_path)
            progress(1.0, "done")
            best = max(clusters, key=lambda c: c.num_in_cluster)
            return ClusterOutcome(
                clusters=pd.DataFrame(
                    [
                        {
                            "rank": c.rank,
                            "lowest_energy": c.lowest_energy,
                            "mean_energy": c.mean_energy,
                            "num_in_cluster": c.num_in_cluster,
                        }
                        for c in clusters
                    ]
                ),
                folder=folder,
                top_energy=clusters[0].lowest_energy,
                top_runs=clusters[0].num_in_cluster,
                best_rank=best.rank,
                best_runs=best.num_in_cluster,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"cluster table ({folder})",
            job,
            on_finished=self._cluster_done,
            on_failed=self._failed,
        )

    def _cluster_done(self, outcome: object) -> None:
        assert isinstance(outcome, ClusterOutcome)
        self._table.bind_frame(outcome.clusters)
        self._plot.clear()
        self._plot.bars(
            outcome.clusters["lowest_energy"].to_numpy(dtype=float),
            labels=[str(r) for r in outcome.clusters["rank"]],
        )
        self._plot.set_labels(
            x="cluster rank",
            y="lowest energy (kcal/mol)",
            title=f"{outcome.folder} · {len(outcome.clusters)} clusters",
        )
        self._report.setText(
            f"{outcome.folder} — {len(outcome.clusters)} clusters · "
            f"rank 1: {outcome.top_energy:.2f} kcal/mol ({outcome.top_runs} runs) · "
            f"most populated: rank {outcome.best_rank} ({outcome.best_runs} runs) · "
            f"{outcome.seconds:.1f}s"
        )

    # -- ligand library ------------------------------------------------------------
    def _start_library(self) -> None:
        if self._busy_check():
            return
        library_path = _LIBRARY_PATH

        def job(progress) -> LibraryOutcome:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors

            from pharmaml.docking import load_ligand_csv

            started = _time.monotonic()
            progress(0.3, "loading & validating the library")
            library = load_ligand_csv(library_path)
            frame = library.frame.copy()
            progress(0.7, "RDKit properties on the valid rows")
            heavy = pd.Series(pd.NA, index=frame.index, dtype="Int64")
            rotatable = pd.Series(pd.NA, index=frame.index, dtype="Int64")
            for idx, row in frame.iterrows():
                if str(row["status"]) != "valid":
                    continue
                mol = Chem.MolFromSmiles(str(row["smiles"]))
                if mol is None:  # defensive: validation already ran RDKit
                    continue
                heavy.loc[idx] = int(mol.GetNumHeavyAtoms())
                rotatable.loc[idx] = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
            frame["num_heavy_atoms"] = heavy
            frame["n_rotatable"] = rotatable
            progress(1.0, "done")
            invalid = library.invalid_frame
            return LibraryOutcome(
                frame=frame,
                n_rows=len(library),
                n_valid=len(library.valid_frame),
                n_invalid=len(invalid),
                invalid_names=[str(v) for v in invalid["name"]],
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "ligand library audit",
            job,
            on_finished=self._library_done,
            on_failed=self._failed,
        )

    def _library_done(self, outcome: object) -> None:
        assert isinstance(outcome, LibraryOutcome)
        self._table.bind_frame(outcome.frame)
        valid = outcome.frame[outcome.frame["status"] == "valid"]
        self._plot.clear()
        self._plot.bars(
            valid["num_heavy_atoms"].astype(float).to_numpy(),
            labels=[str(v) for v in valid["name"]],
        )
        self._plot.set_labels(
            x="ligand",
            y="heavy atoms (RDKit)",
            title=f"Library · {outcome.n_valid} valid of {outcome.n_rows} rows",
        )
        invalid_note = ""
        if outcome.invalid_names:
            invalid_note = f" ({', '.join(outcome.invalid_names)})"
        self._report.setText(
            f"Library — {outcome.n_rows} rows · {outcome.n_valid} valid · "
            f"{outcome.n_invalid} invalid{invalid_note} · RDKit properties "
            f"on the valid rows · {outcome.seconds:.1f}s"
        )

    # -- DOCKSTRING lab ------------------------------------------------------------
    def _start_registry(self) -> None:
        if self._busy_check():
            return

        def job(progress) -> RegistryOutcome:
            from pharmaml.docking import DOCKSTRING_OBJECTIVES, target_frame

            started = _time.monotonic()
            progress(0.6, "reading the target registry")
            frame = target_frame()
            progress(1.0, "done")
            return RegistryOutcome(
                frame=frame,
                objectives=DOCKSTRING_OBJECTIVES,
                first=str(frame.iloc[0]["target"]),
                last=str(frame.iloc[-1]["target"]),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "DOCKSTRING registry",
            job,
            on_finished=self._registry_done,
            on_failed=self._failed,
        )

    def _registry_done(self, outcome: object) -> None:
        assert isinstance(outcome, RegistryOutcome)
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItems([str(v) for v in outcome.frame["target"]])
        self._target_combo.blockSignals(False)

        self._table.bind_frame(outcome.frame)
        volumes = (
            outcome.frame["size_x"].to_numpy(dtype=float)
            * outcome.frame["size_y"].to_numpy(dtype=float)
            * outcome.frame["size_z"].to_numpy(dtype=float)
        )
        self._plot.clear()
        self._plot.bars(volumes, labels=[str(v) for v in outcome.frame["target"]])
        self._plot.set_labels(
            x="target",
            y="search-box volume (angstrom^3)",
            title=f"DOCKSTRING registry · {len(outcome.frame)} targets",
        )
        self._report.setText(
            f"DOCKSTRING — {len(outcome.frame)} indexed targets · "
            f"{len(outcome.objectives)} objectives ({', '.join(outcome.objectives)}) · "
            f"first {outcome.first} · last {outcome.last} · {outcome.seconds:.1f}s"
        )

    def _show_target_config(self) -> None:
        """Instant: the registry lookup + its six-key config preview."""
        name = self._target_combo.currentText()
        if not name:
            self._report.setText(
                "Load the target registry first — the targets load with it."
            )
            return
        from pharmaml.docking import get_target

        target = get_target(name)
        self._report.setText(f"{name} config:\n{target.config_text().strip()}")

    # -- contacts & PLIF (r264) --------------------------------------------------------
    def _start_contacts(self) -> None:
        if self._busy_check():
            return
        cutoff = float(self._cutoff_spin.value())

        def job(progress) -> ContactsOutcome:
            import numpy as np

            from pharmaml.docking import (
                INTERACTION_CLASSES,
                compute_contacts,
                plif_matrix,
                read_pdb_atoms,
                read_pdbqt_atoms,
            )

            started = _time.monotonic()
            progress(0.25, "reading the real 1AKI receptor")
            receptor = read_pdb_atoms(_RECEPTOR_PDB)
            pose = read_pdbqt_atoms(_POSE_PDBQT)
            progress(0.6, f"classifying the {cutoff:g} A shell")
            contacts = compute_contacts(pose, receptor, cutoff=cutoff)
            residues = [str(i) for i in plif_matrix(contacts).index]
            ligand_atoms = [f"{a.residue_key} {a.name}" for a in pose]
            rank = {key: i for i, key in enumerate(residues)}

            table = pd.DataFrame(
                {
                    "residue": [c.receptor_atom.residue_key for c in contacts],
                    "receptor atom": [c.receptor_atom.name for c in contacts],
                    "ligand atom": [c.ligand_atom.name for c in contacts],
                    "distance (A)": [c.distance for c in contacts],
                    "interaction": [c.interaction for c in contacts],
                }
            )
            series = []
            for cls in INTERACTION_CLASSES:
                xs, ys, tips = [], [], []
                for c in contacts:
                    if c.interaction != cls:
                        continue
                    xs.append(ligand_atoms.index(f"{c.ligand_atom.residue_key} {c.ligand_atom.name}"))
                    ys.append(rank[c.receptor_atom.residue_key])
                    tips.append(
                        f"{c.receptor_atom.label} -- {c.ligand_atom.name} "
                        f"{c.distance:.2f} A ({cls})"
                    )
                if xs:
                    series.append((cls, np.array(xs, dtype=float),
                                   np.array(ys, dtype=float), tips))
            first = contacts[0]
            progress(1.0, "done")
            return ContactsOutcome(
                table=table,
                series=tuple(series),
                residues=tuple(residues),
                ligand_atoms=tuple(ligand_atoms),
                n_contacts=len(contacts),
                n_residues=len(residues),
                closest_pair=f"{first.receptor_atom.label} -- {first.ligand_atom.name}",
                closest_distance=float(first.distance),
                closest_interaction=str(first.interaction),
                cutoff=cutoff,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "contact map",
            job,
            on_finished=self._contacts_done,
            on_failed=self._failed,
        )

    def _contacts_done(self, outcome: object) -> None:
        import numpy as np

        assert isinstance(outcome, ContactsOutcome)
        self._table.bind_frame(outcome.table)
        self._plot.clear()
        for i, (cls, xs, ys, tips) in enumerate(outcome.series):
            self._plot.scatter(
                xs, ys, label=cls, symbol="o",
                size=8, index=i,
                meta=lambda _i, _tips=list(tips): {"contact": _tips[_i]},
            )
        if not outcome.series:
            self._plot.scatter(np.array([0.0]), np.array([0.0]), size=1)
        self._plot.set_labels(
            x="ligand atom (pose order)",
            y="receptor residue (closest-contact rank)",
            title=(
                f"Contact map -- {outcome.n_contacts} contact(s) over "
                f"{outcome.n_residues} residue(s), shell {outcome.cutoff:g} A"
            ),
        )
        self._report.setText(
            f"Contacts -- {outcome.n_contacts} pair(s) within "
            f"{outcome.cutoff:g} A of {outcome.n_residues} residue(s) -- "
            f"closest: {outcome.closest_pair} "
            f"{outcome.closest_distance:.2f} A ({outcome.closest_interaction}) -- "
            f"{outcome.seconds:.1f}s"
        )

    def _start_plif(self) -> None:
        if self._busy_check():
            return
        cutoff = float(self._cutoff_spin.value())

        def job(progress) -> PlifOutcome:
            from pharmaml.docking import (
                INTERACTION_CLASSES,
                compute_contacts,
                plif_bits,
                plif_matrix,
                read_pdb_atoms,
                read_pdbqt_atoms,
            )

            started = _time.monotonic()
            progress(0.3, "contacts first -- the PLIF aggregates them")
            receptor = read_pdb_atoms(_RECEPTOR_PDB)
            pose = read_pdbqt_atoms(_POSE_PDBQT)
            contacts = compute_contacts(pose, receptor, cutoff=cutoff)
            progress(0.7, "residue x class counts + the 0/1 twin")
            counts = plif_matrix(contacts)
            bits = plif_bits(contacts)
            progress(1.0, "done")
            return PlifOutcome(
                counts=counts,
                bits=bits,
                labels=tuple(str(i) for i in counts.index),
                class_totals=tuple(int(counts[c].sum()) for c in INTERACTION_CLASSES),
                n_contacts=len(contacts),
                cutoff=cutoff,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "PLIF fingerprint",
            job,
            on_finished=self._plif_done,
            on_failed=self._failed,
        )

    def _plif_done(self, outcome: object) -> None:
        import numpy as np

        assert isinstance(outcome, PlifOutcome)
        counts = outcome.counts.drop(columns=["total"])
        self._table.bind_frame(counts)
        self._plot.clear()
        n = len(counts)
        x = np.arange(n, dtype=float)
        width = 0.22
        for i, cls in enumerate(counts.columns):
            heights = counts[cls].to_numpy(dtype=float)
            self._plot.bars(
                heights,
                x=x + (i - 1.5) * width,
                width=width,
                index=i,
                labels=[str(v) for v in counts.index] if i == 0 else None,
                meta=lambda idx, _cls=str(cls), _h=heights: {
                    "class": _cls, "contacts": int(_h[idx]),
                },
            )
        self._plot.set_labels(
            x="receptor residue (closest-contact order)",
            y="contacts of the class",
            title=(
                f"PLIF fingerprint -- {len(counts)} residue(s), "
                f"{len(counts.columns)} classes, {outcome.n_contacts} contact(s)"
            ),
        )
        totals = " / ".join(
            f"{c} {t}" for c, t in zip(counts.columns, outcome.class_totals)
        )
        self._report.setText(
            f"PLIF -- {outcome.n_contacts} contact(s) classified over "
            f"{len(counts)} residue(s): {totals} -- the 0/1 bits twin rides "
            f"the table's counts -- {outcome.seconds:.1f}s"
        )

    def _validate_box(self) -> None:
        """Instant: parse_search_box validation of the pasted config."""
        from pharmaml.docking import parse_dockstring_search_box
        from pharmaml.exceptions import PharmaMLError

        text = self._box_edit.toPlainText()
        try:
            box = parse_dockstring_search_box(text)
        except PharmaMLError as exc:
            self._report.setText(f"Input — {exc}")
            return
        self._report.setText(
            f"Custom box — center ({box.center[0]:g}, {box.center[1]:g}, "
            f"{box.center[2]:g}) · size ({box.size[0]:g}, {box.size[1]:g}, "
            f"{box.size[2]:g})"
        )

    # -- failures / theming -----------------------------------------------------------
    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "job failed"
        self._report.setText(f"Job failed — {first}")

    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    # -- test/inspection surface -----------------------------------------------------
    def model_frame(self) -> pd.DataFrame | None:
        """The frame currently bound to the result table (or None)."""
        model = self._table.model()
        return model.frame if isinstance(model, PandasTableModel) else None

    def report_text(self) -> str:
        return self._report.text()
