"""Dashboard: the Studio home screen (directive 16.13).

Every number on this page is computed live from the real toolkit -- the
installed ``pharmaml`` version, the model registry, the Studio's own
page census, the example datasets that exist on disk, and the module
census of the package tree. No fabricated metrics.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.nine_cards import build_nine_card_grid

_TK = Path(__file__).resolve().parents[2]  # pharma-ml-toolkit/


def _module_census() -> tuple[int, int]:
    """Count python modules and source lines under src/pharmaml (real walk)."""
    files = list((_TK / "src" / "pharmaml").rglob("*.py"))
    lines = 0
    for f in files:
        try:
            lines += f.read_text(encoding="utf-8", errors="ignore").count("\n") + 1
        except OSError:  # pragma: no cover - unreadable file
            continue
    return len(files), lines


def _studio_census() -> tuple[int, int]:
    """Count the Studio's page modules and their source lines (real walk).

    (round 228) the playground's WORKFLOWS registry retired with
    ``playground.py`` — the dashboard now censuses the Studio itself:
    the pages that absorbed those eleven card workflows.
    """
    files = list((_TK / "ui").rglob("*.py"))
    lines = 0
    for f in files:
        try:
            lines += f.read_text(encoding="utf-8", errors="ignore").count("\n") + 1
        except OSError:  # pragma: no cover - unreadable file
            continue
    pages = len(list((_TK / "ui" / "pages").glob("*.py")))
    return pages, lines


def _model_registry_count() -> int:
    try:
        registry = importlib.import_module("pharmaml.ml.models")
    except Exception:  # noqa: BLE001 - broken-install fallback, census stays honest
        return 0
    return len(registry.MODEL_REGISTRY)


def _example_datasets() -> list[tuple[str, int]]:
    """(name, bytes) for every real example dataset on disk."""
    root = _TK / "examples" / "data"
    if not root.is_dir():  # pragma: no cover - relocated tree
        return []
    out: list[tuple[str, int]] = []
    for p in sorted(root.iterdir()):
        if p.is_file():
            out.append((p.name, p.stat().st_size))
    return out


def _stat_card(title: str, value: str, detail: str = "") -> QFrame:
    card = QFrame()
    card.setObjectName("studioCard")
    card.setFrameShape(QFrame.Shape.StyledPanel)
    card.setMinimumHeight(92)
    lay = QVBoxLayout(card)
    lay.setContentsMargins(16, 12, 16, 12)
    head = QLabel(title)
    head.setObjectName("studioCardTitle")
    big = QLabel(value)
    big.setObjectName("studioCardValue")
    big.setAlignment(Qt.AlignmentFlag.AlignLeft)
    sub = QLabel(detail)
    sub.setObjectName("studioCardDetail")
    sub.setWordWrap(True)
    lay.addWidget(head)
    lay.addWidget(big)
    lay.addWidget(sub)
    return card


class DashboardPage(QWidget):
    """The home screen: live toolkit census + dataset index."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(16)

        title = QLabel("PharmaML Studio")
        title.setObjectName("studioPageTitle")
        subtitle = QLabel("The PySide6 front end to the pharmaml computational-pharmacy toolkit")
        subtitle.setObjectName("studioPageSubtitle")
        lay.addWidget(title)
        lay.addWidget(subtitle)

        # §5: the nine high-level workflow cards — navigation entry points
        # onto the existing pages, never duplicated implementations.
        cards_header = QLabel("Workflow cards")
        cards_header.setObjectName("studioSectionTitle")
        lay.addWidget(cards_header)
        cards_host, _frames = build_nine_card_grid()
        lay.addWidget(cards_host)

        try:
            import pharmaml

            version = pharmaml.__version__
        except Exception:  # noqa: BLE001 - broken-install fallback, census stays honest
            version = "not installed"
        modules, lines = _module_census()
        pages, studio_lines = _studio_census()
        models = _model_registry_count()
        datasets = _example_datasets()

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(_stat_card("Toolkit version", version, "installed in this environment"), 0, 0)
        grid.addWidget(_stat_card("Package census", f"{modules} modules", f"{lines:,} source lines under src/pharmaml"), 0, 1)
        grid.addWidget(_stat_card("Model registry", f"{models} models", "training-ready estimators"), 0, 2)
        grid.addWidget(
            _stat_card(
                "Studio pages",
                f"{pages} pages",
                f"{studio_lines:,} source lines under ui/ — the retired Tk playground's "
                "11 card workflows live on across these pages",
            ),
            1,
            0,
        )
        grid.addWidget(_stat_card("Example datasets", f"{len(datasets)} files", "examples/data — ready to open in Data Explorer"), 1, 1)
        grid.addWidget(_stat_card("GUI framework", "PySide6 + PyQtGraph", "Qt-Material themes, QThread workers"), 1, 2)
        lay.addLayout(grid)

        data_header = QLabel("Example datasets")
        data_header.setObjectName("studioSectionTitle")
        lay.addWidget(data_header)
        for name, size in datasets:
            row = QHBoxLayout()
            label = QLabel(f"examples/data/{name}")
            label.setObjectName("studioDatasetName")
            size_label = QLabel(f"{size / 1024:.1f} KB")
            size_label.setObjectName("studioDatasetSize")
            size_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(label, stretch=1)
            row.addWidget(size_label)
            holder = QFrame()
            holder.setObjectName("studioRow")
            holder.setLayout(row)
            lay.addWidget(holder)

        migration = QLabel(
            "Migration status (directive #16): COMPLETE — the original seventeen pages were "
            "verified in increment 14 (round 228); the legacy Tkinter "
            "playground.py is retired and this Studio is THE front end "
            "(python studio.py)."
        )
        migration.setObjectName("studioMigrationNote")
        migration.setWordWrap(True)
        lay.addWidget(migration)
        lay.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll)
