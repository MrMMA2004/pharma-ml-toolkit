"""The §17C experiment-comparison dialog.

Opened from the Job Center's history: the user multi-selects recorded
runs and gets the scientific comparison — compatibility groups with
side-by-side tables, per-group best-on-metric verdicts, and the
explicitly REFUSED cross-condition pairs (never silently merged).

The dialog renders :func:`pharmaml.studio.comparison.comparison_summary`
— the same text the §17D report surface embeds, one rendering path.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from pharmaml.studio.comparison import (
    RunView,
    comparison_summary,
    compare_runs,
)


class ExperimentCompareDialog(QDialog):
    """Compare selected saved runs (§17C) without mixing conditions."""

    def __init__(self, views: list[RunView], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Experiment comparison")
        self.resize(820, 560)
        self._views = views

        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self._headline = QLabel(f"{len(views)} run(s) selected")
        self._headline.setObjectName("studioStatus")
        self._headline.setWordWrap(True)
        head.addWidget(self._headline, stretch=1)
        lay.addLayout(head)

        self._result = compare_runs(views)
        n_groups = len(self._result.groups)
        n_refused = len(self._result.refused)
        self._headline.setText(
            f"{len(views)} run(s) → {n_groups} compatibility group(s)"
            + (f", {n_refused} cross-condition pair(s) not ranked"
               if n_refused else ""))

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlaceholderText(
            "The comparison appears here: groups of runs sharing task, "
            "metric and validation protocol, each with a side-by-side "
            "table, deltas against the best, and the runs that were NOT "
            "compared because their conditions differ.")
        self._text.setPlainText(comparison_summary(self._result))
        lay.addWidget(self._text, stretch=1)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
