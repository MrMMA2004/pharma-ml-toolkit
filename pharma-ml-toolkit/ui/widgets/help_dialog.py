"""In-app help (review C-2/C-5 fix): the F1 help dialog.

One searchable pane owns everything a new user needs after the
welcome tour: a shortcut cheat-sheet (the five global shortcuts the
main window registers were previously invisible), a per-workspace
orientation line, and a pointer to the docs.  It is a plain QDialog
with token-styled widgets — no new QSS selectors required.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# -- the cheat-sheet (C-5): every global shortcut, one table ----------------
SHORTCUTS: list[tuple[str, str]] = [
    ("F1", "Open this help pane"),
    ("Ctrl+1 … Ctrl+9", "Jump to workspace 1-9 (sidebar order)"),
    ("Alt+Home", "Go to the nine-card home"),
    ("Alt+Left", "Go back to the previous workspace"),
    ("Ctrl+K", "Search capabilities (command palette)"),
    ("Ctrl+C", "Copy the selected table range (tables)"),
]

# -- one orientation line per workspace (C-2: guided entry points) ----------
ORIENTATION: dict[str, tuple[str, str]] = {
    "Data Explorer": (
        "Load a CSV/Excel file",
        "Open CSV / Excel… picks any table on disk; the audit, plots and "
        "every downstream workspace read the loaded frame. Bundled example "
        "datasets live in examples/data/.",
    ),
    "Chemistry": (
        "Featurize a ligand table",
        "Pick a SMILES column (or load one in Data Explorer first), then "
        "Compute features. Invalid SMILES are dropped with per-row reasons.",
    ),
    "Model Analysis": (
        "Train and validate a QSAR model",
        "Choose dataset + target column, press Train. The dataset combo "
        "also offers '(table from Data Explorer)' — the route for your own "
        "CSVs.",
    ),
    "PK/PD": (
        "Simulate dosing regimens",
        "Pick a model, set the dosing table, Run simulation. Completion "
        "reports land as a toast.",
    ),
    "Biology": (
        "Explore an expression matrix",
        "Load the shared matrix (row-select with live preview), then run "
        "the analyses.",
    ),
    "Docking": (
        "Prepare and dock ligands",
        "Docking needs the Vina engines on PATH (Settings ▸ Doctor checks). "
        "Start from a prepared receptor + ligand set.",
    ),
    "Utilities": (
        "Templates, pipelines, comparisons, reports",
        "The seven workflow templates chain existing backends; every "
        "destructive utility defaults to dry-run.",
    ),
    "Settings": (
        "Theme, providers, dependency doctor",
        "The Doctor button verifies optional dependencies and prints the "
        "exact pip line to install what is missing.",
    ),
}


class HelpDialog(QDialog):
    """F1 help: searchable cheat-sheet + workspace orientation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PharmaML Studio — Help")
        self.resize(720, 480)
        lay = QVBoxLayout(self)

        # -- search filter (Ctrl+K trains users to expect one) --------------
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter help topics and shortcuts…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._refilter)
        search_row.addWidget(self.search, stretch=1)
        lay.addLayout(search_row)

        # -- topic list + detail pane ----------------------------------------
        self.topics = QListWidget()
        self.detail = QTextBrowser()
        self.detail.setOpenExternalLinks(True)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.topics)
        split.addWidget(self.detail)
        split.setStretchFactor(1, 3)
        lay.addWidget(split, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)

        # -- content ----------------------------------------------------------
        self._pages: list[tuple[str, str]] = [
            ("Keyboard shortcuts", self._shortcuts_html()),
            ("Getting started", self._getting_started_html()),
        ]
        for name in sorted(ORIENTATION):
            first, rest = ORIENTATION[name]
            self._pages.append(
                (name, f"<h3>{name}</h3><p><b>Start here:</b> {first}</p><p>{rest}</p>")
            )
        for title, html in self._pages:
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, (title, html))
            self.topics.addItem(item)
        self.topics.currentRowChanged.connect(self._show_topic)
        self.topics.setCurrentRow(0)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _shortcuts_html() -> str:
        rows = "".join(
            f"<tr><td><code>{key}</code></td><td>{what}</td></tr>"
            for key, what in SHORTCUTS
        )
        return (
            "<h3>Keyboard shortcuts</h3>"
            "<p>These work everywhere in the Studio:</p>"
            f"<table border='0' cellpadding='4'>{rows}</table>"
            "<p>Table views copy their selection with Ctrl+C.</p>"
        )

    @staticmethod
    def _getting_started_html() -> str:
        return (
            "<h3>Getting started</h3>"
            "<ol>"
            "<li><b>Load data</b> — Data Explorer ▸ <i>Open CSV / Excel…</i> "
            "(or pick a bundled example from examples/data/).</li>"
            "<li><b>Try a workflow</b> — the nine-card home groups the 18 "
            "workspaces by intent; each card opens a guided workspace.</li>"
            "<li><b>Own data everywhere</b> — other workspaces offer "
            "'(table from Data Explorer)' in their dataset lists.</li>"
            "<li><b>Check the environment</b> — Settings ▸ Doctor verifies "
            "optional dependencies (RDKit, Vina, torch) and prints the "
            "exact install line for anything missing.</li>"
            "</ol>"
            "<p>Docs ship in the repository: README.md (overview), "
            "docs/INDEX.md (auto-generated catalog), CHANGELOG.md.</p>"
        )

    def _refilter(self, text: str) -> None:
        """Filter the topic list by substring; detail hides non-matches."""
        text = text.strip().lower()
        for i in range(self.topics.count()):
            item = self.topics.item(i)
            title, html = item.data(Qt.ItemDataRole.UserRole)
            hit = not text or text in title.lower() or text in html.lower()
            item.setHidden(not hit)
        if self.topics.currentItem() is not None and self.topics.currentItem().isHidden():
            for i in range(self.topics.count()):
                if not self.topics.item(i).isHidden():
                    self.topics.setCurrentRow(i)
                    break

    def _show_topic(self, row: int) -> None:
        if row < 0:
            self.detail.setHtml("")
            return
        item = self.topics.item(row)
        if item is None:
            return
        _title, html = item.data(Qt.ItemDataRole.UserRole)
        self.detail.setHtml(html)


class WelcomeDialog(QDialog):
    """First-run welcome (review C-1/C-2 fix): three buttons, no walls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to PharmaML Studio")
        self.setModal(True)
        self.resize(520, 300)
        lay = QVBoxLayout(self)

        head = QLabel("Welcome to PharmaML Studio")
        head.setObjectName("helpDialogTitle")
        head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(head)

        body = QLabel(
            "18 research workspaces, one job backend.\n\n"
            "Start from the nine-card home, load a table in Data Explorer, "
            "and press F1 anytime for the shortcut cheat-sheet and "
            "per-workspace guidance."
        )
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(body, stretch=1)

        row = QHBoxLayout()
        self._wants_help = False
        self.tour_btn = QPushButton("Open Help (F1)")
        self.tour_btn.setDefault(True)
        self.close_btn = QPushButton("Start working")

        def _choose_help() -> None:
            self._wants_help = True
            self.accept()

        self.tour_btn.clicked.connect(_choose_help)
        self.close_btn.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(self.tour_btn)
        row.addWidget(self.close_btn)
        row.addStretch(1)
        lay.addLayout(row)

    @staticmethod
    def maybe_welcome(settings_value: str | None) -> bool:
        """True when the welcome dialog should show (first run only)."""
        return not (isinstance(settings_value, str) and settings_value.lower() in ("true", "1"))
