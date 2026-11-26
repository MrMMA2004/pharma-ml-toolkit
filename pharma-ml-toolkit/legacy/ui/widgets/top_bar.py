"""The nine-card-first top navigation bar (part6 §5.2, Phase 1).

Replaces the permanently visible 18-item sidebar as the dominant
chrome: application identity, Home, Back (history), the current
location, a live jobs indicator fed from the real JobRunner, Settings
access, and the Advanced-navigation toggle that re-shows the legacy
sidebar for power users.  Every control routes through MainWindow
methods — this bar owns no navigation logic of its own.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class TopBar(QWidget):
    """Application-level navigation strip above every mode."""

    home_requested = Signal()
    back_requested = Signal()
    settings_requested = Signal()
    advanced_toggled = Signal(bool)
    search_requested = Signal()
    jobs_clicked = Signal()
    help_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("studioTopBar")
        # a plain QWidget subclass ignores QSS background rules unless it
        # opts in — required for the Catalyst navy shell strip
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(10)

        self._title = QLabel("PharmaML Studio")
        self._title.setObjectName("studioTopBarTitle")
        lay.addWidget(self._title)

        self.home_btn = QPushButton("Home")
        self.home_btn.setToolTip("Back to the nine-card home (Alt+Home)")
        self.home_btn.clicked.connect(self.home_requested.emit)
        lay.addWidget(self.home_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.setToolTip("Return to the previous workspace")
        self.back_btn.setEnabled(False)
        self.back_btn.clicked.connect(self.back_requested.emit)
        lay.addWidget(self.back_btn)

        # part6 §7 (Phase 4): the global search entry point
        self.search_btn = QPushButton("Search…")
        self.search_btn.setToolTip(
            "Search every capability, card and section (Ctrl+K)"
        )
        self.search_btn.clicked.connect(self.search_requested.emit)
        lay.addWidget(self.search_btn)

        # review C-2/C-5 fix: a visible Help entry point (F1) so the
        # cheat-sheet is discoverable without knowing the key.
        self.help_btn = QPushButton("Help")
        self.help_btn.setToolTip("Help, shortcuts and guidance (F1)")
        self.help_btn.clicked.connect(self.help_requested.emit)
        lay.addWidget(self.help_btn)

        self.location = QLabel("Nine-card home")
        self.location.setObjectName("studioTopBarLocation")
        self.location.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.location.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.location, stretch=1)

        # Live jobs indicator — MainWindow feeds this from the real
        # JobRunner signals; never a fabricated busy state.  The §10
        # job center opens on click (also reachable when idle: it owns
        # the run history, not just the live job).
        self.jobs_btn = QPushButton("Jobs")
        self.jobs_btn.setToolTip(
            "Open the unified job center: the live job, Stop, and the "
            "recorded run history with provenance"
        )
        self.jobs_btn.clicked.connect(self.jobs_clicked.emit)
        lay.addWidget(self.jobs_btn)
        self.jobs_label = QLabel("")
        self.jobs_label.setObjectName("studioTopBarJobs")
        lay.addWidget(self.jobs_label)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setToolTip("Open the Studio settings page")
        self.settings_btn.clicked.connect(self.settings_requested.emit)
        lay.addWidget(self.settings_btn)

        self.advanced_btn = QPushButton("Advanced navigation")
        self.advanced_btn.setToolTip(
            "Show the legacy 18-section sidebar (optional power-user mode)"
        )
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.toggled.connect(self.advanced_toggled.emit)
        lay.addWidget(self.advanced_btn)

    def set_location(self, text: str) -> None:
        self.location.setText(text)
        # Phase 5 accessibility: the location/jobs labels are read out
        # with context by screen readers, not as bare text fragments.
        self.location.setAccessibleName(f"Current location: {text}")

    def set_jobs(self, text: str) -> None:
        """Empty string means idle; MainWindow writes real runner state."""
        self.jobs_label.setText(text)
        self.jobs_label.setAccessibleName(
            f"Job status: {text}" if text else "Job status: idle"
        )

    def set_back_enabled(self, enabled: bool) -> None:
        self.back_btn.setEnabled(enabled)
