"""Utilities Workbench: the rar's own file & table operations, Studio edition.

The round-227 port of the legacy Tkinter playground's Utilities tab
(the owner's "everything must be in studio.py" directive, increment 2):
the five workflow card families of the original ``others/General``
scripts, each wired to the SAME public backend APIs the cards call --

* **Table merge & split** -- ``merge_tables`` / ``merge_table_folder``
  (stack several tables or a whole folder, named columns first, the
  rest sorted) and ``split_table_into_chunks`` (stream one big CSV
  into fixed-size parts).
* **Table compare & annotate** -- ``compare_tables`` (the
  common / only-A / only-B reconciliation of one ligand-name column,
  name-normalised) and ``annotate_with_smiles`` (join SMILES onto
  name columns from a reference table).
* **Approved drugs → report** -- ``filter_approved_drugs`` over one
  or more regulatory-approval exports, with the CSV and the bordered
  Word-table (``export_docx``) reports.
* **Campaign file tools** -- ``collect_files`` / ``list_file_names``
  / ``distribute_files`` / ``delete_named_files`` / ``cleanup_tree``,
  dry-run by default on every destructive path.
* **Reindex by number** -- ``reindex_by_number``: align sparse
  per-ligand rows onto a dense 0..N index so missing runs show as
  gaps, not silence.

The reconciliation counts draw as a NATIVE PyQtGraph bar chart (the
§16.8 rule -- no matplotlib import anywhere on this page, so the
round-225 Qt-abort class cannot recur).  Split output goes to a
folder picked BEFORE the run (the Studio's honest pre-flight refusal
replaces the playground's mid-job dialog).  Everything runs in a
:class:`~ui.workers.job_worker.JobWorker`; nothing is fabricated.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
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

#: The merge & split card's modes (the rar's CSV_merger / filemerger /
#: Split CSV file trio) -- mirrored verbatim from the playground card.
MERGE_MODES: tuple[str, ...] = (
    "merge picked files",
    "merge whole folder",
    "split into chunks",
)

#: The compare & annotate card's modes (common_between_two_excel /
#: notcommon vs smilesgetter).
COMPARE_MODES: tuple[str, ...] = (
    "compare a column (A vs B)",
    "annotate names with SMILES",
)

#: The campaign file tools card's modes (extaacter / dlg_extracter /
#: ligand_pdbqt_extracter / Nameextracter / adder / deleter / delete).
TOOL_MODES: tuple[str, ...] = (
    "collect files (name/pattern)",
    "list file names",
    "distribute files into folders",
    "delete named files",
    "cleanup tree (keep list)",
)

#: The shipped example folder the campaign-tools card can load in one
#: click (14 ligand sub-folders, each carrying the campaign's r.dlg).
_CAMPAIGN_EXAMPLE = _TK / "examples" / "data" / "docking_campaign"


@dataclass
class MergeOutcome:
    """What the merge & split job hands back (all real records)."""

    kind: str = "merge"  # "merge" | "split"
    frame: pd.DataFrame | None = None
    parts: list[Path] = field(default_factory=list)
    label: str = ""


@dataclass
class CompareOutcome:
    """What the compare & annotate job hands back."""

    kind: str = "compare"  # "compare" | "annotate"
    frame: pd.DataFrame | None = None
    counts: tuple[int, int, int] = (0, 0, 0)  # common / only-A / only-B
    joined: int = 0  # annotate mode: name columns that found SMILES
    label: str = ""


@dataclass
class ApprovedOutcome:
    """What the approved-drugs job hands back (real filtered names)."""

    names: list[str] = field(default_factory=list)
    status: str = "approved"


@dataclass
class ToolOutcome:
    """What a campaign-tools job hands back."""

    mode: str = ""
    frame: pd.DataFrame | None = None  # None: the run reports a line only
    note: str = ""


@dataclass
class ReindexOutcome:
    """What the reindex job hands back (the dense table)."""

    frame: pd.DataFrame | None = None
    start: int = 0
    stop: int = 0
    missing: int = 0


def _parse_int(text: str, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    """Parse an int field the playground's way (message on failure)."""
    value = int(str(text).strip())
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be >= {minimum}" + (f" and <= {maximum}" if maximum else ""))
    return value


class UtilitiesWorkbenchPage(QWidget):
    """The playground Utilities tab, ported: tables, files, reports."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)

        # -- the picked state (the cards' file slots) ----------------------
        self._merge_files: list[Path] = []
        self._merge_folder: Path | None = None
        self._merge_split_file: Path | None = None
        self._split_out_dir: Path | None = None
        self._cmp_a: Path | None = None
        self._cmp_b: Path | None = None
        self._appr_files: list[Path] = []
        self._tool_folder: Path | None = None
        self._tool_files: list[Path] = []
        self._tool_out: Path | None = None
        self._reindex_path: Path | None = None
        self._frames: dict[str, pd.DataFrame] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_merge_tab(), "Merge & split")
        self._tabs.addTab(self._build_compare_tab(), "Compare & annotate")
        self._tabs.addTab(self._build_approved_tab(), "Approved drugs")
        self._tabs.addTab(self._build_tools_tab(), "Campaign tools")
        self._tabs.addTab(self._build_reindex_tab(), "Reindex")
        lay.addWidget(self._tabs)
        self._diagnostics_btn = QPushButton("Reproducibility & checkpoint diagnostics…")
        self._diagnostics_btn.clicked.connect(self._show_diagnostics)
        lay.addWidget(self._diagnostics_btn)

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
        right_lay.addWidget(QLabel("Ligand-set reconciliation:"))
        self._plot = PlotPanel(title="common / only-A / only-B")
        right_lay.addWidget(self._plot, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([680, 420])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No utilities run yet.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

    def _show_diagnostics(self) -> None:
        from .reproducibility_dialog import ReproducibilityDialog

        dialog = getattr(self, "_diagnostics_dialog", None)
        if dialog is None:
            dialog = ReproducibilityDialog(self._runner, self)
            self._diagnostics_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    # -- tab construction -------------------------------------------------------
    def _build_merge_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._merge_mode = QComboBox()
        self._merge_mode.addItems(MERGE_MODES)
        self._merge_mode.currentTextChanged.connect(self._refresh_merge_info)
        form.addRow("mode:", self._merge_mode)

        pick = QHBoxLayout()
        for label, slot in (
            ("Pick tables…", self._pick_merge_files),
            ("Pick folder…", self._pick_merge_folder),
            ("Pick big CSV…", self._pick_merge_split_file),
            ("Pick parts folder…", self._pick_split_out_dir),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            pick.addWidget(btn)
        example = QPushButton("Load example tables")
        example.clicked.connect(self._load_merge_example)
        pick.addWidget(example)
        form.addRow("", pick)

        self._merge_info = QLabel("no tables, folder or file picked yet")
        self._merge_info.setWordWrap(True)
        form.addRow("picked:", self._merge_info)

        self._merge_firstcols = QLineEdit("")
        self._merge_firstcols.setToolTip(
            "Comma-separated columns to keep first in the merged output (the "
            "CSV_merger column-ordering contract); blank keeps source order."
        )
        form.addRow("first columns:", self._merge_firstcols)
        self._merge_ext = QLineEdit(".csv")
        self._merge_ext.setToolTip("Folder-merge mode: the extension collected from the folder.")
        form.addRow("extension:", self._merge_ext)
        self._merge_chunk = QLineEdit("1500000")
        self._merge_chunk.setToolTip(
            "Split mode: maximum rows per part file (the rar's default was 1.5M)."
        )
        form.addRow("rows per part:", self._merge_chunk)

        run = QHBoxLayout()
        self._merge_btn = QPushButton("Run")
        self._merge_btn.setMinimumHeight(34)
        self._merge_btn.clicked.connect(self._start_merge)
        run.addWidget(self._merge_btn)
        self._merge_save_btn = QPushButton("Save merged CSV…")
        self._merge_save_btn.clicked.connect(lambda: self._save_frame("merge", "merged_output.csv"))
        run.addWidget(self._merge_save_btn)
        form.addRow("", run)

        hint = QLabel(
            "The rar's CSV_merger / filemerger / Split CSV file trio: stack several "
            "tables into one (named columns first, the rest sorted), merge every table "
            "in a folder, or stream one big CSV into fixed-size parts (parts land in "
            "the picked parts folder — the page refuses a split without one BEFORE "
            "any job starts). 'Load example tables' picks screening_library.csv + "
            "walkthrough_compounds.csv and the QSAR example as the split source."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_compare_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._cmp_mode = QComboBox()
        self._cmp_mode.addItems(COMPARE_MODES)
        form.addRow("mode:", self._cmp_mode)

        pick = QHBoxLayout()
        for label, slot in (
            ("Table A…", self._pick_cmp_a),
            ("Table B / reference…", self._pick_cmp_b),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            pick.addWidget(btn)
        form.addRow("", pick)

        self._cmp_info = QLabel("no tables picked yet")
        self._cmp_info.setWordWrap(True)
        form.addRow("picked:", self._cmp_info)

        self._cmp_column = QLineEdit("Ligand")
        self._cmp_column.setToolTip(
            "Compare mode: the ligand-name column reconciled across A and B "
            "(name-normalised: case, blanks, .pdbqt extensions)."
        )
        form.addRow("compare column:", self._cmp_column)
        self._cmp_names = QLineEdit("Ligand")
        self._cmp_names.setToolTip(
            "Annotate mode: comma-separated name columns that each get a SMILES column."
        )
        form.addRow("annotate columns:", self._cmp_names)
        self._cmp_ref_name = QLineEdit("Ligand")
        form.addRow("reference name col:", self._cmp_ref_name)
        self._cmp_ref_smiles = QLineEdit("SMILES")
        form.addRow("reference SMILES col:", self._cmp_ref_smiles)

        run = QHBoxLayout()
        self._cmp_btn = QPushButton("Run")
        self._cmp_btn.setMinimumHeight(34)
        self._cmp_btn.clicked.connect(self._start_compare)
        run.addWidget(self._cmp_btn)
        save = QPushButton("Save result table…")
        save.clicked.connect(lambda: self._save_frame("compare", "reconciliation.csv"))
        run.addWidget(save)
        form.addRow("", run)

        hint = QLabel(
            "common_between_two_excel + notcommon + smilesgetter: reconcile one "
            "ligand-name column across two tables (common / only-A / only-B, "
            "name-normalised — the counts draw as the bar chart), or join SMILES "
            "onto name columns from a reference table."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_approved_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        pick = QHBoxLayout()
        pick_btn = QPushButton("Pick approval exports…")
        pick_btn.clicked.connect(self._pick_appr_files)
        pick.addWidget(pick_btn)
        form.addRow("", pick)
        self._appr_info = QLabel("no export files picked yet")
        self._appr_info.setWordWrap(True)
        form.addRow("picked:", self._appr_info)

        self._appr_drug = QLineEdit("drug")
        form.addRow("drug column:", self._appr_drug)
        self._appr_col = QLineEdit("regulatory approval")
        form.addRow("approval column:", self._appr_col)
        self._appr_status = QLineEdit("approved")
        form.addRow("status:", self._appr_status)

        run = QHBoxLayout()
        self._appr_btn = QPushButton("Run")
        self._appr_btn.setMinimumHeight(34)
        self._appr_btn.clicked.connect(self._start_approved)
        run.addWidget(self._appr_btn)
        save_csv = QPushButton("Save CSV…")
        save_csv.clicked.connect(lambda: self._save_frame("approved", "approved_drugs.csv"))
        run.addWidget(save_csv)
        save_docx = QPushButton("Save Word report…")
        save_docx.clicked.connect(self._save_approved_docx)
        run.addWidget(save_docx)
        form.addRow("", run)

        hint = QLabel(
            "The rar's approved.py: filter a regulatory-approval column across one or "
            "more exports (TSV/CSV/DataFrame) and export the unique approved drug "
            "names — as CSV or the bordered Word table (python-docx; the CSV arm "
            "works without it)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_tools_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._tool_mode = QComboBox()
        self._tool_mode.addItems(TOOL_MODES)
        self._tool_mode.currentTextChanged.connect(self._refresh_tool_info)
        form.addRow("mode:", self._tool_mode)

        pick = QHBoxLayout()
        for label, slot in (
            ("Pick campaign folder…", self._pick_tool_folder),
            ("Pick files to distribute…", self._pick_tool_files),
            ("Pick collect-output folder…", self._pick_tool_out),
            ("Load example campaign", self._load_tool_example),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            pick.addWidget(btn)
        form.addRow("", pick)

        self._tool_info = QLabel(
            "pick a campaign folder (and files for distribute, an output folder for collect)"
        )
        self._tool_info.setWordWrap(True)
        form.addRow("picked:", self._tool_info)

        self._tool_filename = QLineEdit("r.dlg")
        self._tool_filename.setToolTip(
            "Collect/delete mode: the file name matched in every sub-folder."
        )
        form.addRow("filename:", self._tool_filename)
        self._tool_pattern = QLineEdit(r"\d+\.pdbqt")
        self._tool_pattern.setToolTip(
            "Collect/cleanup mode: the regex pattern (collect matches it, cleanup keeps it)."
        )
        form.addRow("pattern:", self._tool_pattern)
        self._tool_ext = QLineEdit(".csv")
        self._tool_ext.setToolTip("List mode: the extension whose names are listed, sorted.")
        form.addRow("extension (list):", self._tool_ext)
        self._tool_keep = QLineEdit("r.pdbqt,r.dlg,r.gpf,r.dpf")
        self._tool_keep.setToolTip(
            "Cleanup mode: comma-separated file names kept; everything else goes."
        )
        form.addRow("keep names (cleanup):", self._tool_keep)

        self._tool_dry = QCheckBox("dry run (report only, nothing deleted)")
        self._tool_dry.setChecked(True)
        form.addRow("", self._tool_dry)

        self._tool_btn = QPushButton("Run")
        self._tool_btn.setMinimumHeight(34)
        self._tool_btn.clicked.connect(self._start_tools)
        form.addRow("", self._tool_btn)

        hint = QLabel(
            "The rar's extaacter / dlg_extracter / ligand_pdbqt_extracter / "
            "Nameextracter / adder / deleter / delete family: collect matched files "
            "out of every sub-folder, list a folder's names, distribute files into "
            "every immediate sub-folder, delete stray named files, or clean the tree "
            "down to a keep list. Destructive modes run dry by default; a real run "
            "asks once before anything touches the disk."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_reindex_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        pick = QHBoxLayout()
        pick_btn = QPushButton("Pick the sparse results table…")
        pick_btn.clicked.connect(self._pick_reindex_file)
        pick.addWidget(pick_btn)
        form.addRow("", pick)
        self._reindex_info = QLabel("no table picked yet")
        self._reindex_info.setWordWrap(True)
        form.addRow("picked:", self._reindex_info)

        self._reindex_column = QLineEdit("ligand")
        self._reindex_column.setToolTip(
            "Column carrying the numbered names (e.g. '5_result.dlg'); its first "
            "number becomes the index."
        )
        form.addRow("column:", self._reindex_column)
        self._reindex_start = QLineEdit("0")
        form.addRow("start:", self._reindex_start)
        self._reindex_stop = QLineEdit("300")
        form.addRow("stop:", self._reindex_stop)
        self._reindex_pattern = QLineEdit(r"(\d+)")
        self._reindex_pattern.setToolTip(
            "Regex whose first capture group extracts the ligand number."
        )
        form.addRow("number pattern:", self._reindex_pattern)

        run = QHBoxLayout()
        self._reindex_btn = QPushButton("Run")
        self._reindex_btn.setMinimumHeight(34)
        self._reindex_btn.clicked.connect(self._start_reindex)
        run.addWidget(self._reindex_btn)
        save = QPushButton("Save dense table…")
        save.clicked.connect(lambda: self._save_frame("reindex", "reindexed.csv"))
        run.addWidget(save)
        form.addRow("", run)

        hint = QLabel(
            "The rar's Prettier_extracted_dlgs: align sparse per-ligand rows onto a "
            "dense start..stop index so missing runs show as gap rows, not silence."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- pickers (dialogs write the state slots; tests set them directly) -------
    def _pick_merge_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Pick the tables to merge", "", "Tables (*.csv *.tsv *.xlsx);;All files (*)"
        )
        if paths:
            self._merge_files = [Path(p) for p in paths]
            self._refresh_merge_info()

    def _pick_merge_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick the folder to merge")
        if path:
            self._merge_folder = Path(path)
            self._refresh_merge_info()

    def _pick_merge_split_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick the big table to split (CSV or Excel)", "", "Tables (*.csv *.tsv *.xlsx);;All files (*)"
        )
        if path:
            self._merge_split_file = Path(path)
            self._refresh_merge_info()

    def _pick_split_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick the folder for the part files")
        if path:
            self._split_out_dir = Path(path)
            self._refresh_merge_info()

    def _load_merge_example(self) -> None:
        """(round 201 arm, ported) every merge mode runnable in one click."""
        picked = [
            _TK / "examples" / "data" / "screening_library.csv",
            _TK / "examples" / "data" / "walkthrough_compounds.csv",
        ]
        split_source = _TK / "examples" / "data" / "qsar_regression.csv"
        missing = [p.name for p in (*picked, split_source) if not p.is_file()]
        if missing:
            self._report.setText("Missing examples: " + ", ".join(missing))
            return
        self._merge_files = list(picked)
        self._merge_split_file = split_source
        self._merge_mode.setCurrentText(MERGE_MODES[0])
        self._refresh_merge_info()
        self._report.setText("2 example tables picked + the QSAR split source — hit Run")

    def _refresh_merge_info(self) -> None:
        mode = self._merge_mode.currentText()
        parts: list[str] = []
        if mode.startswith("merge picked") and self._merge_files:
            names = ", ".join(p.name for p in self._merge_files[:4])
            parts.append(
                f"{len(self._merge_files)} table(s): {names}"
                + (" …" if len(self._merge_files) > 4 else "")
            )
        elif mode.startswith("merge whole") and self._merge_folder is not None:
            parts.append(f"folder: {self._merge_folder.name}")
        elif mode.startswith("split"):
            if self._merge_split_file is not None:
                parts.append(f"big CSV: {self._merge_split_file.name}")
            if self._split_out_dir is not None:
                parts.append(f"parts into: {self._split_out_dir.name}")
        self._merge_info.setText("\n".join(parts) or "no tables, folder or file picked yet")

    def _pick_cmp_a(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick table A", "", "Tables (*.csv *.tsv *.xlsx);;All files (*)"
        )
        if path:
            self._cmp_a = Path(path)
            self._refresh_cmp_info()

    def _pick_cmp_b(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick table B / the reference", "", "Tables (*.csv *.tsv *.xlsx);;All files (*)"
        )
        if path:
            self._cmp_b = Path(path)
            self._refresh_cmp_info()

    def _refresh_cmp_info(self) -> None:
        parts = []
        if self._cmp_a is not None:
            parts.append(f"A: {self._cmp_a.name}")
        if self._cmp_b is not None:
            parts.append(f"B: {self._cmp_b.name}")
        self._cmp_info.setText(" · ".join(parts) or "no tables picked yet")

    def _pick_appr_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Pick the approval exports", "", "Approval exports (*.tsv *.csv *.tab);;All files (*)"
        )
        if paths:
            self._appr_files = [Path(p) for p in paths]
            shown = ", ".join(p.name for p in self._appr_files[:3])
            if len(self._appr_files) > 3:
                shown += f" … (+{len(self._appr_files) - 3})"
            self._appr_info.setText(shown)

    def _pick_tool_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick the campaign folder")
        if path:
            self._tool_folder = Path(path)
            self._refresh_tool_info()

    def _pick_tool_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Pick the files to distribute", "", "All files (*)"
        )
        if paths:
            self._tool_files = [Path(p) for p in paths]
            self._refresh_tool_info()

    def _pick_tool_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick the collect-output folder")
        if path:
            self._tool_out = Path(path)
            self._refresh_tool_info()

    def _load_tool_example(self) -> None:
        """The shipped 14-ligand campaign tree, one click."""
        if not _CAMPAIGN_EXAMPLE.is_dir():
            self._report.setText("Example campaign missing on disk.")
            return
        self._tool_folder = _CAMPAIGN_EXAMPLE
        self._refresh_tool_info()
        self._report.setText(
            "Example campaign loaded (14 ligand folders) — collect r.dlg or list .csv, hit Run"
        )

    def _refresh_tool_info(self) -> None:
        mode = self._tool_mode.currentText()
        parts: list[str] = []
        if self._tool_folder is not None:
            parts.append(f"folder: {self._tool_folder.name}")
        if self._tool_files and mode.startswith("distribute"):
            names = ", ".join(p.name for p in self._tool_files[:3])
            parts.append(
                f"{len(self._tool_files)} file(s): {names}"
                + (" …" if len(self._tool_files) > 3 else "")
            )
        if self._tool_out is not None and mode.startswith("collect"):
            parts.append(f"collect into: {self._tool_out.name}")
        self._tool_info.setText(
            "\n".join(parts)
            or "pick a campaign folder (and files for distribute, an output folder for collect)"
        )

    def _pick_reindex_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick the sparse results table", "", "Tables (*.csv *.tsv *.xlsx);;All files (*)"
        )
        if path:
            self._reindex_path = Path(path)
            self._reindex_info.setText(path)

    # -- job plumbing ---------------------------------------------------------
    def _on_started(self, name: str) -> None:
        for btn in (
            self._merge_btn,
            self._cmp_btn,
            self._appr_btn,
            self._tool_btn,
            self._reindex_btn,
        ):
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in (
            self._merge_btn,
            self._cmp_btn,
            self._appr_btn,
            self._tool_btn,
            self._reindex_btn,
        ):
            btn.setEnabled(True)

    def _submit(self, name: str, job: Any, on_finished: Any) -> None:
        """Shared runner submit with the duplicate-submission guard."""
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return
        self._runner.submit(
            name,
            job,
            on_finished=on_finished,
            on_failed=self._failed,
        )

    def _failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "job failed"
        self._report.setText(f"Job failed — {first}")

    # -- merge & split ------------------------------------------------------------
    def _start_merge(self) -> None:
        mode = self._merge_mode.currentText()
        first_columns = (
            [t.strip() for t in self._merge_firstcols.text().split(",") if t.strip()] or None
        )
        if mode.startswith("merge picked"):
            sources: list[Any] = list(self._merge_files)  # TableSource: paths or frames
            if not sources:
                self._report.setText("Pick the tables to merge first.")
                return
            label = f"merging {len(sources)} table(s)"

            def merge_job(progress: Any) -> MergeOutcome:
                from pharmaml.chem import merge_tables

                progress(0.1, "reading tables")
                frame = merge_tables(sources, first_columns=first_columns)
                progress(1.0, "done")
                return MergeOutcome(kind="merge", frame=frame, label=label)

            self._submit(label, merge_job, self._merge_done)
            return
        if mode.startswith("merge whole"):
            folder = self._merge_folder
            if folder is None:
                self._report.setText("Pick the folder to merge first.")
                return
            extension = self._merge_ext.text().strip() or ".csv"
            label = f"merging {folder.name} ({extension})"

            def folder_job(progress: Any) -> MergeOutcome:
                from pharmaml.chem import merge_table_folder

                progress(0.1, "reading folder")
                frame = merge_table_folder(
                    folder, extension=extension, first_columns=first_columns
                )
                progress(1.0, "done")
                return MergeOutcome(kind="merge", frame=frame, label=label)

            self._submit(label, folder_job, self._merge_done)
            return
        # split mode -- the Studio's honest pre-flight: the parts folder
        # is picked BEFORE the job (the playground asked mid-run).
        big = self._merge_split_file
        if big is None:
            self._report.setText("Pick the big CSV to split first.")
            return
        out_dir = self._split_out_dir
        if out_dir is None:
            self._report.setText("Pick the parts folder first — the split refuses to guess.")
            return
        try:
            chunk_rows = _parse_int(self._merge_chunk.text(), "rows per part", minimum=1)
        except ValueError as exc:
            self._report.setText(str(exc))
            return
        label = f"splitting {big.name} every {chunk_rows:,} rows"

        def split_job(progress: Any) -> MergeOutcome:
            from pharmaml.chem.converters import split_table_into_chunks

            progress(0.1, "streaming parts")
            parts = split_table_into_chunks(big, out_dir, chunk_rows=chunk_rows)
            progress(1.0, "done")
            return MergeOutcome(kind="split", parts=parts, label=label)

        self._submit(label, split_job, self._merge_done)

    def _merge_done(self, outcome: object) -> None:
        assert isinstance(outcome, MergeOutcome)
        if outcome.kind == "merge":
            assert outcome.frame is not None
            self._frames["merge"] = outcome.frame
            self._table.bind_frame(outcome.frame)
            self._report.setText(
                f"Merged — {len(outcome.frame):,} rows × {len(outcome.frame.columns)} columns"
            )
        else:
            self._frames.pop("merge", None)
            parts = outcome.parts
            frame = pd.DataFrame(
                {
                    "part": [p.name for p in parts],
                    "bytes": [p.stat().st_size if p.exists() else 0 for p in parts],
                }
            )
            self._table.bind_frame(frame)
            self._report.setText(
                f"Split into {len(parts)} part file(s) — "
                f"see the table; parts live in the picked folder"
            )

    # -- compare & annotate -------------------------------------------------------
    def _start_compare(self) -> None:
        if self._cmp_a is None or self._cmp_b is None:
            self._report.setText("Pick table A and table B first.")
            return
        mode = self._cmp_mode.currentText()
        a_path, b_path = self._cmp_a, self._cmp_b

        if mode.startswith("compare"):
            column = self._cmp_column.text().strip()
            if not column:
                self._report.setText("Name the compare column.")
                return
            label = f"comparing '{column}'"

            def compare_job(progress: Any) -> CompareOutcome:
                from pharmaml.chem import compare_tables

                progress(0.1, "reconciling")
                comparison = compare_tables(a_path, b_path, column)
                progress(1.0, "done")
                records = (
                    [("common", name) for name in comparison.common]
                    + [("only in A", name) for name in comparison.only_in_a]
                    + [("only in B", name) for name in comparison.only_in_b]
                )
                return CompareOutcome(
                    kind="compare",
                    frame=pd.DataFrame(records, columns=["section", "name"]),
                    counts=(
                        len(comparison.common),
                        len(comparison.only_in_a),
                        len(comparison.only_in_b),
                    ),
                    label=label,
                )

            self._submit(label, compare_job, self._compare_done)
            return

        # annotate mode
        name_columns = (
            [t.strip() for t in self._cmp_names.text().split(",") if t.strip()] or None
        )
        if not name_columns:
            self._report.setText("Name the columns to annotate (comma-separated).")
            return
        ref_name = self._cmp_ref_name.text().strip() or "Ligand"
        ref_smiles = self._cmp_ref_smiles.text().strip() or "SMILES"
        label = "annotating with SMILES"

        def annotate_job(progress: Any) -> CompareOutcome:
            from pharmaml.chem import annotate_with_smiles
            from pharmaml.utils.io import read_table

            progress(0.1, "joining SMILES")
            frame = read_table(a_path)
            annotated = annotate_with_smiles(
                frame,
                b_path,
                name_columns,
                reference_name_column=ref_name,
                reference_smiles_column=ref_smiles,
            )
            progress(1.0, "done")
            joined = sum(
                1
                for column in name_columns
                if column + "_SMILES" in annotated.columns
                and annotated[column + "_SMILES"].astype(str).str.len().any()
            )
            return CompareOutcome(
                kind="annotate",
                frame=annotated,
                counts=(joined, len(name_columns), len(annotated)),
                label=label,
            )

        self._submit(label, annotate_job, self._compare_done)

    def _compare_done(self, outcome: object) -> None:
        assert isinstance(outcome, CompareOutcome)
        assert outcome.frame is not None
        self._frames["compare"] = outcome.frame
        self._table.bind_frame(outcome.frame)
        if outcome.kind == "compare":
            common, only_a, only_b = outcome.counts
            # the reconciliation figure, natively (§16.8: PyQtGraph, no matplotlib)
            self._plot.clear()
            self._plot.bars(
                pd.Series([common, only_a, only_b]).to_numpy(dtype=float),
                labels=["common", "only A", "only B"],
            )
            self._plot.set_labels(
                x="section", y="ligand names", title="Ligand-set reconciliation"
            )
            self._report.setText(
                f"Compared — common {common} · only A {only_a} · only B {only_b}"
            )
        else:
            joined, requested, rows = outcome.counts
            self._report.setText(
                f"Annotated — {rows:,} rows · {requested} name column(s) "
                f"annotated ({joined} found SMILES)"
            )

    # -- approved drugs -----------------------------------------------------------
    def _start_approved(self) -> None:
        if not self._appr_files:
            self._report.setText("Pick the approval export files first.")
            return
        files = list(self._appr_files)
        drug_column = self._appr_drug.text().strip() or "drug"
        approval_column = self._appr_col.text().strip() or "regulatory approval"
        status = self._appr_status.text().strip() or "approved"
        label = f"filtering {len(files)} file(s)"

        def job(progress: Any) -> ApprovedOutcome:
            from pharmaml.pharma import filter_approved_drugs

            progress(0.1, "filtering approvals")
            names = filter_approved_drugs(
                [str(p) for p in files],
                drug_column=drug_column,
                approval_column=approval_column,
                approval_status=status,
            )
            progress(1.0, "done")
            return ApprovedOutcome(names=list(names), status=status)

        self._submit(label, job, self._approved_done)

    def _approved_done(self, outcome: object) -> None:
        assert isinstance(outcome, ApprovedOutcome)
        frame = pd.DataFrame({"drug": outcome.names})
        self._frames["approved"] = frame
        self._table.bind_frame(frame)
        self._report.setText(
            f"Approved — {len(outcome.names)} drug name(s) (status '{outcome.status}')"
        )

    def _save_approved_docx(self) -> None:
        """The bordered Word-table report (export_docx; needs python-docx)."""
        frame = self._frames.get("approved")
        if frame is None:
            self._report.setText("Filter the approval files first.")
            return
        try:
            has_docx = importlib.util.find_spec("docx") is not None
        except (ImportError, ValueError):  # pragma: no cover
            has_docx = False
        if not has_docx:
            self._report.setText(
                "The Word export needs python-docx — pip install 'pharmaml[reports]' "
                "(the CSV export works without it)."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the approved-drug report", "approved_drugs.docx", "Word document (*.docx)"
        )
        if not path:
            return

        def job(progress: Any) -> Path:
            from pharmaml.viz import export_docx

            progress(0.1, "writing the DOCX report")
            written = export_docx(frame, path)
            progress(1.0, "done")
            return written

        def done(written: object) -> None:
            self._report.setText(f"Word report written to {written}")

        self._submit("writing the DOCX report", job, done)

    # -- campaign file tools --------------------------------------------------------
    def _start_tools(self) -> None:
        mode = self._tool_mode.currentText()
        folder = self._tool_folder
        if folder is None:
            self._report.setText("Pick the campaign folder first.")
            return
        dry_run = self._tool_dry.isChecked()
        filename = self._tool_filename.text().strip() or None
        pattern = self._tool_pattern.text().strip() or None
        extension = self._tool_ext.text().strip() or ".csv"

        if mode.startswith("collect"):
            if not filename and not pattern:
                self._report.setText("Give a filename and/or a pattern to collect.")
                return
            out_dir = self._tool_out or (folder / "collected")

            def collect_job(progress: Any) -> ToolOutcome:
                from pharmaml.docking import collect_files

                progress(0.1, "collecting")
                frame = collect_files(
                    folder, filename=filename, pattern=pattern, out_dir=out_dir, dry_run=dry_run
                )
                progress(1.0, "done")
                return ToolOutcome(mode="collect", frame=frame)

            def collect_done(outcome: object) -> None:
                assert isinstance(outcome, ToolOutcome)
                assert outcome.frame is not None
                self._frames["tools"] = outcome.frame
                self._table.bind_frame(outcome.frame)
                # dry-run rows all carry copied=False (every match is a
                # copy candidate) -- the honest dry-run count is the match
                # count, not the engine's literal copied sum.
                if dry_run:
                    copied = len(outcome.frame)
                    verb = "would copy"
                else:
                    copied = (
                        int(outcome.frame["copied"].sum())
                        if "copied" in outcome.frame
                        else 0
                    )
                    verb = "copied"
                self._report.setText(
                    f"Collected — {len(outcome.frame)} match(es) · {verb} {copied} "
                    f"into {out_dir.name}"
                )

            self._submit("collecting files", collect_job, collect_done)
            return

        if mode.startswith("list"):

            def list_job(progress: Any) -> ToolOutcome:
                from pharmaml.docking import list_file_names

                progress(0.1, "listing")
                names = list_file_names(folder, extension=extension)
                progress(1.0, "done")
                return ToolOutcome(
                    mode="list", frame=pd.DataFrame({"name": list(names)})
                )

            def list_done(outcome: object) -> None:
                assert isinstance(outcome, ToolOutcome)
                assert outcome.frame is not None
                self._frames["tools"] = outcome.frame
                self._table.bind_frame(outcome.frame)
                self._report.setText(
                    f"Listed — {len(outcome.frame)} {extension} name(s), sorted"
                )

            self._submit(f"listing {extension} files", list_job, list_done)
            return

        if mode.startswith("distribute"):
            if not self._tool_files:
                self._report.setText("Pick the files to distribute first (binaries, receptor, …).")
                return
            files: list[str | Path] = [str(p) for p in self._tool_files]
            if not dry_run:
                answer = QMessageBox.question(
                    self,
                    "Distribute for real",
                    f"Copy {len(files)} file(s) into EVERY immediate sub-folder of\n"
                    f"{folder}?\n(existing copies are kept)",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return

            def distribute_job(progress: Any) -> ToolOutcome:
                from pharmaml.docking import distribute_files

                progress(0.1, "distributing")
                copied = distribute_files(files, folder, overwrite=False)
                progress(1.0, "done")
                return ToolOutcome(
                    mode="distribute",
                    frame=pd.DataFrame({"destination": [str(p) for p in copied]}),
                )

            def distribute_done(outcome: object) -> None:
                assert isinstance(outcome, ToolOutcome)
                assert outcome.frame is not None
                self._frames["tools"] = outcome.frame
                self._table.bind_frame(outcome.frame)
                self._report.setText(
                    f"Distributed — {len(outcome.frame)} copy/copies across {folder.name}"
                )

            self._submit(f"distributing {len(files)} file(s)", distribute_job, distribute_done)
            return

        if mode.startswith("delete"):
            if not filename:
                self._report.setText("Name the stray file to delete.")
                return

            def delete_job(progress: Any) -> ToolOutcome:
                from pharmaml.docking import delete_named_files

                progress(0.1, "deleting")
                count = delete_named_files(folder, filename, dry_run=dry_run)
                progress(1.0, "done")
                return ToolOutcome(mode="delete", note=f"{count} {filename} file(s)")

            def delete_done(outcome: object) -> None:
                assert isinstance(outcome, ToolOutcome)
                self._frames.pop("tools", None)
                self._table.bind_frame(None, note="delete mode reports a count, not a table")
                verb = "would delete" if dry_run else "deleted"
                self._report.setText(f"Deleted — {verb} {outcome.note} under {folder.name}")

            self._submit(f"deleting '{filename}'", delete_job, delete_done)
            return

        if mode.startswith("cleanup"):
            keep = {
                token.strip() for token in self._tool_keep.text().split(",") if token.strip()
            }
            if not dry_run:
                answer = QMessageBox.question(
                    self,
                    "Cleanup for real",
                    f"Delete every file under\n{folder}\nthat is NOT one of the keep "
                    f"names / pattern?\n(empty folders can remain; this cannot be undone)",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return

            def cleanup_job(progress: Any) -> ToolOutcome:
                from pharmaml.docking import cleanup_tree

                progress(0.1, "cleaning")
                frame = cleanup_tree(folder, keep=keep, keep_pattern=pattern, dry_run=dry_run)
                progress(1.0, "done")
                return ToolOutcome(mode="cleanup", frame=frame)

            def cleanup_done(outcome: object) -> None:
                assert isinstance(outcome, ToolOutcome)
                assert outcome.frame is not None
                self._frames["tools"] = outcome.frame
                self._table.bind_frame(outcome.frame)
                verb = "would delete" if dry_run else "deleted"
                self._report.setText(f"Cleaned — {len(outcome.frame)} file(s) {verb}")

            self._submit("cleaning the tree", cleanup_job, cleanup_done)

    # -- reindex ----------------------------------------------------------------
    def _start_reindex(self) -> None:
        if self._reindex_path is None:
            self._report.setText("Pick the CSV first.")
            return
        column = self._reindex_column.text().strip()
        if not column:
            self._report.setText("Name the number column.")
            return
        try:
            start = _parse_int(self._reindex_start.text(), "start", minimum=0)
            stop = _parse_int(self._reindex_stop.text(), "stop", minimum=0)
        except ValueError as exc:
            self._report.setText(str(exc))
            return
        if stop < start:
            self._report.setText("stop must be >= start")
            return
        pattern = self._reindex_pattern.text().strip() or r"(\d+)"
        path = self._reindex_path
        label = f"reindexing {path.name}"

        def job(progress: Any) -> ReindexOutcome:
            from pharmaml.docking import reindex_by_number
            from pharmaml.utils.io import read_table

            progress(0.1, "reindexing")
            frame = read_table(path)
            dense = reindex_by_number(frame, column, start=start, stop=stop, pattern=pattern)
            progress(1.0, "done")
            missing = int(dense.iloc[:, 0].isna().sum()) if len(dense.columns) else 0
            return ReindexOutcome(frame=dense, start=start, stop=stop, missing=missing)

        self._submit(label, job, self._reindex_done)

    def _reindex_done(self, outcome: object) -> None:
        assert isinstance(outcome, ReindexOutcome)
        assert outcome.frame is not None
        self._frames["reindex"] = outcome.frame
        self._table.bind_frame(outcome.frame)
        self._report.setText(
            f"Reindexed — dense {outcome.start}…{outcome.stop} · {len(outcome.frame)} rows · "
            f"{outcome.missing} gap row(s)"
        )

    # -- saving (the library's six-format visual export owns the writer) -----------
    def _save_frame(self, key: str, default_name: str) -> None:
        """Ask a path and write the card's last frame (export_table)."""
        frame = self._frames.get(key)
        if frame is None:
            self._report.setText("Run the tool first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save the table",
            default_name,
            "CSV table (*.csv);;Excel workbook (*.xlsx);;TSV table (*.tsv);;"
            "Markdown table (*.md);;JSON records (*.json);;HTML table (*.html)",
        )
        if not path:
            return
        self._write_frame(frame, path, key)

    def _write_frame(self, frame: pd.DataFrame, path: str, subject: str) -> None:
        """The actual writer (tests drive this directly, dialog-free)."""
        try:
            from pharmaml.viz import export_table

            written = export_table(frame, path)
        except Exception as exc:  # noqa: BLE001
            self._report.setText(f"Could not save — {type(exc).__name__}: {exc}")
            return
        self._report.setText(f"Saved — {subject} table written to {written}")

    # -- theming / test surface ------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    def report_text(self) -> str:
        return self._report.text()

    def bound_frame(self) -> pd.DataFrame | None:
        """The frame currently bound to the result table (or None)."""
        model = self._table.model()
        return model.frame if isinstance(model, PandasTableModel) else None
