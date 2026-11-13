"""§5 nine-card dashboard registry — the single source of truth.

Part-4 §5 of the directive requires the Studio home to organize the
existing eighteen-page architecture behind nine high-level cards that
act as navigation and workflow entry points, in an exact order, never
as duplicated implementations.  This module owns that contract:

* ``NINE_CARDS`` — the exact nine cards, in the directive's order, each
  mapped to the existing Studio sections (the ``SECTIONS`` keys of
  ``ui.main_window``) named by the directive's mapping table.
* ``probe_available`` — one honest, exception-safe import probe.
* ``live_preview`` — a real, computed summary line per card (no
  fabricated metrics; every value comes from the live tree or the
  installed environment).
* ``build_nine_card_grid`` — the Qt builder that renders the cards.

Phase-3 (part-4 §8, round 263) — responsive layouts and SMILES previews:

* :class:`NineCardGrid` re-flows the SAME card widgets row-major across
  three, two or one columns as the window narrows — the directive's
  canonical 3×3 stays the default, the §5 order survives every width,
  and no widget is ever rebuilt mid-flight;
* the Datasets card carries a real molecule thumbnail — the genuine
  first row of ``examples/data/screening_library.csv`` rendered through
  the Studio's own §22 :class:`MoleculeRenderer` (cached, honest
  refusal when rdkit or a QApplication is absent — the card then keeps
  its text preview and nothing is fabricated).

The cards never remove or duplicate the specialized pages: every entry
button routes through ``MainWindow.go_to(section)`` onto the page that
already owns the workflow.  The optional-dependency chips state the
real boundaries — including the class-F ones (the docking engine and
the MD trajectory data are external, never bundled).
"""

from __future__ import annotations

import csv
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtGui import QPixmap, QResizeEvent


@dataclass(frozen=True)
class CardEntry:
    """One direct entry point: a real Studio section key + button label."""

    section: str
    label: str


@dataclass(frozen=True)
class NineCard:
    """One §5 high-level card, in the directive's exact order."""

    card_id: str
    title: str
    description: str
    badge: str
    capabilities: str
    entries: tuple[CardEntry, ...]
    optional: tuple[tuple[str, str], ...] = ()
    external: str = ""


CARD_INTRODUCTION = NineCard(
    card_id="introduction",
    title="Introduction",
    description="The Studio home: live toolkit census, dataset index, and this card grid.",
    badge="01",
    capabilities="Installed-version report, package/Studio censuses, example-dataset index, "
    "and the navigation entry points for every workflow family below.",
    entries=(CardEntry("dashboard", "Open Dashboard"),),
)

CARD_DATASETS = NineCard(
    card_id="datasets",
    title="Datasets",
    description="Load, inspect, standardize, and convert screening data.",
    badge="02",
    capabilities="CSV/TSV/Excel ingestion, profiling and quality reports, descriptor and "
    "fingerprint computation, format conversion, campaign + approvals management.",
    entries=(
        CardEntry("data", "Open Data Explorer"),
        CardEntry("chemistry", "Open Chemistry Workbench"),
        CardEntry("utilities", "Open Utilities"),
    ),
    # openpyxl is a hard dependency — Excel support is core, not optional.
)

CARD_MACHINE_LEARNING = NineCard(
    card_id="machine-learning",
    title="Machine Learning",
    description="Train, analyze, and compare classical QSAR/ADMET models.",
    badge="03",
    capabilities="Model registry training, best-of sweeps, unsupervised exploration, "
    "foundations walkthroughs, and model analysis with metrics and explanations.",
    entries=(
        CardEntry("models", "Open Model Analysis"),
        CardEntry("mlworkbench", "Open ML Workbench"),
        CardEntry("unsupervised", "Open Unsupervised"),
        CardEntry("bestofml", "Open Best-of ML"),
        CardEntry("foundations", "Open Foundations"),
    ),
    optional=(("shap", "SHAP explanations in Model Analysis"),),
)

CARD_DEEP_LEARNING = NineCard(
    card_id="deep-learning",
    title="Deep Learning",
    description="Neural and graph models behind one workbench.",
    badge="04",
    capabilities="Neural-network training and evaluation over the registered datasets, "
    "with the class-C GNN/geometric backends available from the API layer.",
    entries=(CardEntry("deep_learning", "Open Deep Learning Workbench"),),
    optional=(("torch", "the neural backends (neural.py, class-C GNN modules)"),),
)

CARD_GENERATIVE = NineCard(
    card_id="generative-ai",
    title="Generative AI",
    description="De-novo design, scaffold hopping, and genetic exploration.",
    badge="05",
    capabilities="RNN generation, genetic optimization, scaffold hopping, and validity/"
    "uniqueness scoring over the standardization pipeline.",
    entries=(CardEntry("generative", "Open Generative Workbench"),),
    optional=(
        ("torch", "the RNN generator"),
        ("rdkit", "genetic/scaffold-hop chemistry scoring"),
    ),
)

CARD_DOCKING = NineCard(
    card_id="molecular-docking",
    title="Molecular Docking",
    description="Ligand/protein prep, output parsing, and consensus rescore.",
    badge="06",
    capabilities="GPF/DPF preparation, torsion QA, pose RMSD, docking-output parsing, "
    "and consensus rescoring; DOCKSTRING benchmark boxes for targets.",
    entries=(CardEntry("docking", "Open Docking Workbench"),),
    optional=(("rdkit", "the chemistry prep paths"),),
    external="Source/ZIP includes Windows Vina and AutoDock engines; execution needs a compatible runtime "
    "and user-supplied receptor PDBQT. Studio focuses on preparation and analysis; "
    "scores are user or engine-generated inputs.",
)

CARD_PKPD = NineCard(
    card_id="pkpd",
    title="PK/PD",
    description="Pharmacokinetic and pharmacodynamic modeling workflows.",
    badge="07",
    capabilities="Compartmental PK fitting, PD response modeling, and the biology and "
    "response-AI workbenches over the shared data explorer.",
    entries=(
        CardEntry("pkpd", "Open PK/PD Workbench"),
        CardEntry("biology", "Open Biology Workbench"),
        CardEntry("responseai", "Open Response AI"),
    ),
)

CARD_MD = NineCard(
    card_id="molecular-dynamics-analysis",
    title="Molecular Dynamics Analysis",
    description="Trajectory analysis over externally produced MD exports.",
    badge="08",
    capabilities="XVG parsing, alignment, RMSD/RMSF metrics, hydrogen bonds, PLIF, "
    "essential dynamics (dimred), ENM, and collective-variable studies.",
    entries=(CardEntry("molecular_dynamics", "Open Molecular Dynamics"),),
    external="Trajectory data comes from external GROMACS/PLUMED runs (class F) — "
    "the analysis engines are bundled, the simulations are not.",
)

CARD_VISUALIZATION = NineCard(
    card_id="visualization",
    title="Visualization",
    description="Image Lab and the shared plotting layer.",
    badge="09",
    capabilities="The PyQtGraph plot family (inspector, legends, tick budgets), the "
    "Image Lab canvas, and the pharmaml.viz chem/ML figure builders.",
    entries=(CardEntry("imagelab", "Open Image Lab"),),
    optional=(("matplotlib", "figure export (viz/export.py)"),),
)

#: The exact §5 order, left to right — pinned by tests.
NINE_CARDS: tuple[NineCard, ...] = (
    CARD_INTRODUCTION,
    CARD_DATASETS,
    CARD_MACHINE_LEARNING,
    CARD_DEEP_LEARNING,
    CARD_GENERATIVE,
    CARD_DOCKING,
    CARD_PKPD,
    CARD_MD,
    CARD_VISUALIZATION,
)

# part6 §6 (Phase 2): the card ORDER is owned by the pure-Python
# capability registry under src/pharmaml/studio/; this module owns the
# card PRESENTATION (titles, badges, entries).  The two sources must
# never diverge — this check fails the import loudly if they do.
from pharmaml.studio.capability_registry import CARD_ORDER as _REGISTRY_CARD_ORDER  # noqa: E402

assert tuple(c.card_id for c in NINE_CARDS) == _REGISTRY_CARD_ORDER, (
    "ui/nine_cards.py and pharmaml.studio.capability_registry disagree on the "
    "nine primary cards — fix the divergence before shipping (part6 §2.1/§6)"
)

#: Phase-3 responsive thresholds: at/above 880 logical px the directive's
#: canonical 3×3; from 560 to 879 two columns; below 560 a single column.
#: The order is row-major at every width — only the column count changes.
COLUMNS_3_MIN_WIDTH = 880
COLUMNS_2_MIN_WIDTH = 560

#: The Datasets card's molecule thumbnail (logical px; the §22 renderer
#: draws at 2x with devicePixelRatio for crisp bonds).
CARD_PREVIEW_SIZE = QSize(200, 110)

#: The real data source behind the Datasets card's SMILES preview — the
#: same file the Data Explorer opens.  Pinned by tests so a rename or a
#: column change fails the suite instead of silently blanking the card.
PREVIEW_SOURCE_NAME = "examples/data/screening_library.csv"


def columns_for_width(width: int) -> int:
    """The Phase-3 column count for a live widget width (3 / 2 / 1)."""
    if width >= COLUMNS_3_MIN_WIDTH:
        return 3
    if width >= COLUMNS_2_MIN_WIDTH:
        return 2
    return 1


def probe_available(module: str) -> bool:
    """Exception-safe import probe — the honest optional-dependency answer."""
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False
    return True


def live_preview(card_id: str, context: Mapping[str, Any] | None = None) -> str:
    """A real, computed summary for the card (never fabricated).

    part5 §11: previews come from actual loaded files, persisted runs, or
    parsed campaign outputs — :func:`pharmaml.studio.collect_evidence`
    gathers them (``context`` carries the live page state: the loaded
    dataset, the loaded MD study, the benchmark checkpoint dir).  A card
    without data shows its precise empty state instead of a module count.
    """
    try:
        if card_id == "introduction":
            import pharmaml

            return f"pharmaml {pharmaml.__version__} installed in this environment"
        from pharmaml.studio import collect_evidence

        board = collect_evidence(context)
        evidence = board.get(card_id)
        if evidence is not None:
            return evidence.text()
    except Exception:  # noqa: BLE001 - a broken install must not break the home
        return "live preview unavailable in this environment"
    return ""


def _toolkit_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def datasets_preview_smiles() -> tuple[str, str] | None:
    """The genuine first molecule of the screening library (never invented).

    Phase-3 asks for SMILES previews "where meaningful"; the Datasets
    card shows the real first row of ``examples/data/screening_library.csv``
    — the same file the Data Explorer opens.  Returns ``(smiles, name)``
    or ``None`` when the file, its SMILES column or its first row is
    gone (the card then keeps its text preview — no fabrication).
    """
    path = _toolkit_root() / PREVIEW_SOURCE_NAME
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [
                name
                for name in (reader.fieldnames or [])
                if name and name.strip().lower() in ("smiles", "smile")
            ]
            if not columns:
                return None
            column = columns[0]
            for row in reader:
                value = (row.get(column) or "").strip()
                if value:
                    name = (row.get("name") or "").strip() or "(unnamed)"
                    return value, name
    except Exception:  # noqa: BLE001 - moved/renamed file: honest refusal
        return None
    return None


#: One renderer per process for the card thumbnail — the §22 cache is
#: shared with every table preview, so repeated dashboard builds hit it.
_PREVIEW_RENDERER = None


def _preview_renderer():
    global _PREVIEW_RENDERER
    if _PREVIEW_RENDERER is None:
        from ui.widgets.smiles_preview import MoleculeRenderer

        _PREVIEW_RENDERER = MoleculeRenderer()
    return _PREVIEW_RENDERER


def _render_preview(smiles: str) -> tuple[QPixmap | None, str | None]:
    """One structure pixmap via the §22 renderer — honest refusal on any failure."""
    try:
        from PySide6.QtGui import QGuiApplication

        if QGuiApplication.instance() is None:
            return None, "no QApplication"
        return _preview_renderer().render(smiles, CARD_PREVIEW_SIZE)
    except Exception as exc:  # noqa: BLE001 - broken environment, honest refusal
        return None, str(exc)


def _datasets_molecule_visual() -> QWidget | None:
    """The Datasets card's real molecule thumbnail (part-4 Phase-3)."""
    found = datasets_preview_smiles()
    if found is None:
        return None
    smiles, name = found
    holder = QWidget()
    holder.setObjectName("nineCardVisual")
    lay = QVBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    image = QLabel()
    image.setObjectName("nineCardMolecule")
    image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    pixmap, _error = _render_preview(smiles)
    if pixmap is not None:
        image.setPixmap(pixmap)
    else:
        # §22.4 honesty: never a placeholder image — the reason, plus the
        # original SMILES preserved exactly in the caption below.
        image.setText("structure preview unavailable")
        image.setWordWrap(True)
    lay.addWidget(image)
    caption = QLabel(
        f"{name} · first row of {PREVIEW_SOURCE_NAME} · SMILES {smiles}"
    )
    caption.setObjectName("nineCardPreview")
    caption.setWordWrap(True)
    lay.addWidget(caption)
    return holder


def _dependency_chip(card: NineCard) -> str:
    """The honest optional/external indication §5 demands."""
    if card.external and not card.optional:
        return f"External: {card.external}"
    if not card.optional and not card.external:
        return "No optional dependencies — runs on the core stack"
    parts: list[str] = []
    for module, why in card.optional:
        state = "available" if probe_available(module) else "unavailable"
        parts.append(f"{module} {state} — {why}")
    if card.external:
        parts.append(f"External: {card.external}")
    return " · ".join(parts)


class NineCardGrid(QWidget):
    """The nine cards in a responsive row-major grid (part-4 Phase-3).

    The directive's canonical layout is the 3×3; a resize below
    :data:`COLUMNS_3_MIN_WIDTH` re-flows to two columns and below
    :data:`COLUMNS_2_MIN_WIDTH` to one — always row-major, always the
    §5 order.  The re-flow reuses the SAME card widgets (nothing is
    rebuilt), so frames addressed by ``card_id`` survive every width
    change and no state is ever lost mid-flight.

    part5 §11: the preview line per card is a :class:`QLabel` addressed
    by objectName ``nineCardLivePreview``; :meth:`refresh_evidence`
    re-reads the real sources and updates the SAME labels in place
    (the §8 widget-reuse contract, now for the evidence text too).

    part6 §5 (Phase 1): clicking anywhere on a card emits
    ``card_activated(card_id)`` — the nine-card-first home connects it
    to its CardWorkspace.  The entry buttons keep their exact routing.
    """

    card_activated = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        context_provider: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._context_provider = context_provider
        self._frames: dict[str, QFrame] = {}
        context = self._snapshot_context()
        for card in NINE_CARDS:
            frame = _build_card(card, context)
            frame.setObjectName("nineCard")
            self._frames[card.card_id] = frame
            # part6 §5: the whole card is clickable — the workspace opens;
            # the buttons inside keep their per-section routing.
            frame.setCursor(Qt.CursorShape.PointingHandCursor)
            frame.installEventFilter(self)
        self._previews: dict[str, QLabel] = {}
        for card in NINE_CARDS:
            labels = [
                widget
                for widget in self._frames[card.card_id].findChildren(QLabel)
                if widget.objectName() == "nineCardLivePreview"
            ]
            self._previews[card.card_id] = labels[0]
        self._grid = QGridLayout(self)
        self._grid.setSpacing(12)
        self._columns = 0
        self._stretch_row: int | None = None
        self.reflow(3)  # the §5 canonical layout is the default

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """A mouse press on a card frame opens that card's workspace."""
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(obj, QFrame):
            for card_id, frame in self._frames.items():
                if frame is obj:
                    self.card_activated.emit(card_id)
                    break
        return super().eventFilter(obj, event)

    def current_columns(self) -> int:
        """The live column count (3, 2 or 1)."""
        return self._columns

    def frames(self) -> dict[str, QFrame]:
        """The stable ``card_id -> QFrame`` map (a copy for safe addressing)."""
        return dict(self._frames)

    def preview_labels(self) -> dict[str, QLabel]:
        """The stable ``card_id -> preview QLabel`` map (§11 evidence text)."""
        return dict(self._previews)

    def set_card_context(
        self, provider: Callable[[], Mapping[str, Any] | None] | None
    ) -> None:
        """Wire (or clear) the live page-state provider, then refresh."""
        self._context_provider = provider
        self.refresh_evidence()

    def refresh_evidence(self) -> None:
        """Re-read the §11 evidence sources into the SAME preview labels.

        The collectors are deliberately cheap (in-memory frames, tiny
        JSON run records, one checkpoint read) so this runs on the GUI
        thread without a worker round-trip; the heavy jobs that CREATE
        the evidence stay in the pages' ``QThreadPool`` workers.
        """
        context = self._snapshot_context()
        for card_id, label in self._previews.items():
            label.setText(live_preview(card_id, context))

    def _snapshot_context(self) -> Mapping[str, Any] | None:
        if self._context_provider is None:
            return None
        try:
            return self._context_provider() or None
        except Exception:  # noqa: BLE001 - provider must not break the home
            return None

    def reflow(self, columns: int) -> None:
        """Re-lay the SAME frames row-major at 3, 2 or 1 columns."""
        if columns == self._columns:
            return
        if columns not in (1, 2, 3):
            raise ValueError(f"columns must be 1, 2 or 3 — got {columns!r}")
        if self._stretch_row is not None:
            self._grid.setRowStretch(self._stretch_row, 0)
            self._stretch_row = None
        for column in range(3):
            self._grid.setColumnStretch(column, 0)
        while self._grid.count():
            self._grid.takeAt(0)
        for index, card in enumerate(NINE_CARDS):
            self._grid.addWidget(
                self._frames[card.card_id], index // columns, index % columns
            )
        for column in range(columns):
            self._grid.setColumnStretch(column, 1)
        rows = -(-len(NINE_CARDS) // columns)  # ceil — the stretch row follows
        self._stretch_row = rows
        self._grid.setRowStretch(rows, 1)
        self._columns = columns

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Phase-3 responsive pass: re-flow when the width class changes."""
        super().resizeEvent(event)
        want = columns_for_width(event.size().width())
        if want != self._columns:
            self.reflow(want)


def build_nine_card_grid(
    context_provider: Callable[[], Mapping[str, Any] | None] | None = None,
) -> tuple[QWidget, dict[str, QFrame]]:
    """Render the nine cards responsive and row-major (the directive's order).

    Returns the grid widget and a ``card_id -> QFrame`` map so tests and
    the dashboard can address individual cards without fragile walks.
    """
    host = NineCardGrid(context_provider=context_provider)
    return host, host.frames()


def _build_card(card: NineCard, context: Mapping[str, Any] | None = None) -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setMinimumHeight(210)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(6)

    head = QHBoxLayout()
    badge = QLabel(card.badge)
    badge.setObjectName("nineCardBadge")
    title = QLabel(card.title)
    title.setObjectName("studioCardTitle")
    title.setWordWrap(True)
    head.addWidget(badge)
    head.addWidget(title, stretch=1)
    lay.addLayout(head)

    desc = QLabel(card.description)
    desc.setObjectName("studioCardDetail")
    desc.setWordWrap(True)
    lay.addWidget(desc)

    caps = QLabel(card.capabilities)
    caps.setObjectName("studioCardDetail")
    caps.setWordWrap(True)
    lay.addWidget(caps, stretch=1)

    # Phase-3 (§8): a dedicated SMILES preview where meaningful — the
    # Datasets card shows the real first molecule of the screening
    # library; every other card keeps its computed text preview.
    if card.card_id == "datasets":
        visual = _datasets_molecule_visual()
        if visual is not None:
            lay.addWidget(visual)

    preview = QLabel(live_preview(card.card_id, context))
    preview.setObjectName("nineCardLivePreview")
    preview.setWordWrap(True)
    lay.addWidget(preview)

    chips = QLabel(_dependency_chip(card))
    chips.setObjectName("nineCardChip")
    chips.setWordWrap(True)
    lay.addWidget(chips)

    buttons = QHBoxLayout()
    for entry in card.entries:
        btn = QPushButton(entry.label)
        btn.setObjectName(f"nineCardEntry-{entry.section}")
        btn.setToolTip(f"Navigate to the existing {entry.section} page")
        btn.clicked.connect(
            lambda _checked=False, section=entry.section, source=btn: _go(section, source)
        )
        buttons.addWidget(btn, stretch=1)
    lay.addLayout(buttons)
    return frame


def _go(section: str, source: QWidget) -> None:
    """Route through the live window — never a duplicated implementation."""
    go_to = getattr(source.window(), "go_to", None)
    if callable(go_to):
        go_to(section)


__all__ = [
    "CARD_PREVIEW_SIZE",
    "COLUMNS_2_MIN_WIDTH",
    "COLUMNS_3_MIN_WIDTH",
    "NINE_CARDS",
    "PREVIEW_SOURCE_NAME",
    "CardEntry",
    "NineCard",
    "NineCardGrid",
    "build_nine_card_grid",
    "columns_for_width",
    "datasets_preview_smiles",
    "live_preview",
    "probe_available",
]
