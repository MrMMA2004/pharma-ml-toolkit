"""The Introduction page (part8 Phase C, §12.2).

The Studio's opening screen: what this application IS, its
computational pharmaceutical purpose, the supported scientific
workflows, the key backend capabilities, the role of the nine
workspaces, and one clear entry point into each — an attractive,
informative, comprehensive, CONCISE orientation page.  Deliberately
NOT another nine-card dashboard (the directive's explicit rule): no
live-evidence preview grid, no card frames — a documented orientation
with entry buttons.
"""

from __future__ import annotations

from typing import Callable, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .nine_cards import NINE_CARDS

#: §12.2 orientation copy: what the Studio is, why it exists.  Concise
#: technical prose — hierarchy through typography, not walls of text.
_PURPOSE = (
    "PharmaML Studio is a computational-pharmaceutical workbench: an "
    "integrated desktop environment for ligand and screening data — from "
    "raw assay tables to featurized matrices, trained and validated "
    "models, applicability-domain analysis, docking, PK/PD and molecular "
    "dynamics inspection, and publication-ready figures."
)

_WORKFLOWS = (
    "Every workspace is a connected scientific workflow — data input, "
    "validation, computation, result inspection, visualization and "
    "export in sequence — over a shared dataset/result reference model "
    "that preserves provenance and stable sample identities between "
    "workspaces."
)

#: One line per workspace family: what it does, in workflow terms.
#: The tuple mirrors the §12.1 left-to-right workspace order.
_WORKSPACE_NOTES: dict[str, str] = {
    "introduction": "This page — orientation, census, and workspace entry points.",
    "datasets": "Import → inspect → validate → standardize → descriptors & "
                "fingerprints → feature selection → split/leakage audit → "
                "visualize → export.",
    "machine-learning": "Select dataset → target → features → model → train → "
                        "evaluate → compare → applicability domain → "
                        "explainability → export. Semi-supervised learning "
                        "lives here.",
    "deep-learning": "Architecture definition, training with monitored "
                     "learning curves, prediction inspection and checkpoint "
                     "export — separate from classical ML.",
    "generative-ai": "Generation method → starting molecules/constraints → "
                     "generate → validate → properties → filter → rank → "
                     "export, with synthetic-accessibility checks.",
    "molecular-docking": "Ligand preparation → receptor & binding site → "
                         "engine configuration → docking → pose inspection → "
                         "interaction analysis → export.",
    "pkpd": "Dose/time/response import → model selection → fit & simulate → "
            "diagnostics → exposure/response visualization → export.",
    "molecular-dynamics-analysis": "Topology/trajectory loading → validation → "
                                   "selected analyses → results → export "
                                   "(trajectory ANALYSIS, not simulation).",
    "visualization": "Any loaded dataset or model result → plot → configure "
                     "→ generate → inspect/compare → export, without "
                     "duplicating data between workspaces.",
}

#: Key backend capabilities (§12.2 "key backend capabilities") — the
#: real, verified ones, one phrase each.
_CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("Chemistry", "RDKit standardization, 43 descriptor properties, 20 feature "
                  "families, configurable Morgan fingerprints"),
    ("Classical ML", "train/evaluate registry models, hyperparameter search, "
                     "ensembles, persistence bundles, applicability domain"),
    ("Semi-supervised", "SelfTraining / LabelPropagation / LabelSpreading "
                        "with leakage-safe evaluation"),
    ("Deep Learning", "PyTorch training loops with live curves and checkpoints"),
    ("Docking", "AutoDock Vina preparation, docking and pose analysis"),
    ("PK/PD", "compartment fitting, diagnostics and simulation"),
    ("MD analysis", "trajectory-level analyses over loaded studies"),
    ("Visualization", "70+ matplotlib figure functions behind the Studio's "
                      "inspector-chart layer"),
)


class IntroductionPage(QWidget):
    """§12.2: the Studio's opening orientation screen."""

    workspace_requested = Signal(str)
    """The user asked for a workspace (the nine §12.1 keys)."""

    def __init__(
        self,
        parent: QWidget | None = None,
        context_provider: Callable[[], Mapping[str, object] | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._context_provider = context_provider

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(16)

        # -- headline block ------------------------------------------------
        title = QLabel("PharmaML Studio")
        title.setObjectName("studioPageTitle")
        subtitle = QLabel("A computational-pharmaceutical workbench, in nine connected workspaces")
        subtitle.setObjectName("studioPageSubtitle")
        subtitle.setWordWrap(True)
        purpose = QLabel(_PURPOSE)
        purpose.setObjectName("studioCardDetail")
        purpose.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addWidget(purpose)

        workflows = QLabel(_WORKFLOWS)
        workflows.setObjectName("studioCardDetail")
        workflows.setWordWrap(True)
        lay.addWidget(workflows)

        # -- capabilities strip (two columns of one-liners) -----------------
        cap_header = QLabel("Key backend capabilities")
        cap_header.setObjectName("studioSectionHeading")
        lay.addWidget(cap_header)
        caps = QHBoxLayout()
        caps.setSpacing(12)
        for column in (0, 1):
            col = QVBoxLayout()
            col.setSpacing(4)
            for i, (name, note) in enumerate(_CAPABILITIES):
                if i % 2 != column:
                    continue
                line = QLabel(f"<b>{name}</b> — {note}")
                line.setObjectName("studioCardDetail")
                line.setWordWrap(True)
                col.addWidget(line)
            col.addStretch(1)
            wrap = QWidget()
            wrap.setLayout(col)
            caps.addWidget(wrap, stretch=1)
        lay.addLayout(caps)

        # -- the nine workspaces: entry points, §12.1 order -----------------
        ws_header = QLabel("The nine workspaces")
        ws_header.setObjectName("studioSectionHeading")
        lay.addWidget(ws_header)
        note = QLabel(
            "Each workspace is a complete scientific workflow; the buttons "
            "below (and the horizontal bar above) enter them in the "
            "commission's exact order."
        )
        note.setObjectName("studioCardDetail")
        note.setWordWrap(True)
        lay.addWidget(note)

        for card in NINE_CARDS:  # the registry's own §12.1 order
            entry = QFrame()
            entry.setObjectName("introductionWorkspaceRow")
            row = QHBoxLayout(entry)
            row.setContentsMargins(12, 8, 12, 8)
            row.setSpacing(10)
            badge = QLabel(card.badge)
            badge.setObjectName("nineCardBadge")
            title_l = QLabel(f"<b>{card.title}</b>")
            title_l.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            desc = QLabel(_WORKSPACE_NOTES.get(card.card_id, card.description))
            desc.setObjectName("studioCardDetail")
            desc.setWordWrap(True)
            desc.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            go = QPushButton(f"Open {card.title}")
            go.setObjectName(f"introductionEntry-{card.card_id}")
            go.setCursor(Qt.CursorShape.PointingHandCursor)
            go.setMinimumHeight(32)
            go.clicked.connect(
                lambda _checked=False, k=card.card_id: self.workspace_requested.emit(k)
            )
            go.setToolTip(f"Enter the {card.title} workspace")
            row.addWidget(badge)
            row.addWidget(title_l)
            row.addWidget(desc, stretch=1)
            row.addWidget(go)
            lay.addWidget(entry)

        lay.addStretch(1)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # -- census hook (§12.2 "toolkit census" rides the live provider) --------
    def refresh(self) -> None:
        """Re-read the live context (kept for the shell's refresh calls)."""
        if self._context_provider is not None:
            self._context_provider()
