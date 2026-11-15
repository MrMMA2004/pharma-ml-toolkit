"""The §13 workflow-guide strip (part8 round 4, Phase Twelve).

The audit: every workspace already HAS its stages — import, validation,
computation, inspection, visualization and export buttons exist and are
wired — but the *sequence* the directive prescribes is invisible in the
app.  A first-time user sees a form, not a workflow.  This module gives
each of the nine workspaces a compact, collapsible strip that names the
directive's stage chain and says where each stage lives (which tab /
button), so the workspace reads as the connected scientific workflow it
is.

Design rules (from §13 and the house style):

* The registry below transcribes the directive's nine suggested
  workflows VERBATIM as stage names — no invented stages, no silently
  dropped ones.  Where the workspace splits a stage across subpages,
  ``where`` names every surface honestly.
* The strip is a QToolButton + one row of stage chips — it must not
  become a second dashboard (the §12.2 no-duplicate-dashboards rule).
* Collapsed by default after the first visit?  NO — always visible,
  one row tall; it is orientation, not a dialog.
* No Persian text, no RTL, English only (the directive's own rule).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

#: workspace key -> (heading, ((stage, where), ...), note)
#: The stage chains transcribe part8 §13.2–§13.9 verbatim; ``where``
#: names the REAL surfaces in this tree that implement each stage.
WORKFLOWS: dict[str, tuple[str, tuple[tuple[str, str], ...], str]] = {
    "datasets": (
        "Import → Inspect → Validate → Clean/Standardize → Select Columns "
        "→ Calculate Descriptors/Fingerprints → Feature Selection → "
        "Split/Leakage Audit → Visualize → Export",
        (
            ("Import", "Data Explorer — Open CSV / Excel… (background worker, cancellable)"),
            ("Inspect", "Data Explorer — preview table + column stats"),
            ("Validate", "Data Audit — per-column type/range/duplicate report"),
            ("Clean/Standardize", "Chemistry Lab — RDKit standardize, invalid rows dropped with reasons"),
            ("Select Columns", "Data Explorer — column visibility + partition ranges"),
            ("Descriptors/Fingerprints", "Chemistry Lab — 20 feature families, Morgan fingerprints"),
            ("Feature Selection", "ML Workbench — family benchmark leaderboard"),
            ("Split/Leakage Audit", "Data Explorer — split audit (scaffold/group folds); CV fold report"),
            ("Visualize", "Data Explorer — per-column plots; Image Lab matrices"),
            ("Export", "chart export on every plot; CSV/JSON from utilities"),
        ),
        "Dataset provenance rides the shared workspace registry: every load "
        "is noted with its real path and fingerprint, reusable from any "
        "workspace.",
    ),
    "machine-learning": (
        "Select Dataset → Select Target → Select Features → Preprocess → "
        "Define Split/Cross-Validation → Select Model → Configure "
        "Hyperparameters → Train → Evaluate → Compare → Applicability "
        "Domain → Explainability/Uncertainty → Visualize → Save/Export",
        (
            ("Select Dataset", "Model Analysis — dataset combo incl. (table from Data Explorer)"),
            ("Select Target", "Model Analysis — target column combo"),
            ("Select Features", "Chemistry Lab featurization; ML Workbench family selection"),
            ("Preprocess", "train_model(preprocess=True) — scaling/imputation inside the fit"),
            ("Split/CV", "Model Analysis — cv folds + cv strategy (scaffold/group-aware)"),
            ("Select Model", "Model Analysis — registry model combo"),
            ("Hyperparameters", "Model Analysis — manual hyperparameter table"),
            ("Train", "Model Analysis — Train (background job, cancellable)"),
            ("Evaluate", "predicted-vs-actual, residuals, metrics label"),
            ("Compare", "ML Workbench — Compare models (CV) leaderboard; Best of ML bake-offs"),
            ("Applicability Domain", "Model Analysis — applicability-domain analysis"),
            ("Explainability/Uncertainty", "permutation importance; uncertainty dialog"),
            ("Visualize", "learning curves, ROC/PR, calibration; Image Lab backend figures"),
            ("Save/Export", "persistence bundle save; chart exports"),
        ),
        "Semi-supervised learning lives in this workspace: the ML "
        "Workbench's self-training paths over partially labeled targets.",
    ),
    "deep-learning": (
        "Select Dataset → Validate → Preprocess → Define Architecture → "
        "Configure Training → Train → Monitor Learning Curves → Evaluate → "
        "Inspect Predictions → Visualize → Save Checkpoint/Export",
        (
            ("Select Dataset", "Deep Learning — dataset combo"),
            ("Validate", "same data-audit path as classical ML"),
            ("Preprocess", "the runboard's featurization step"),
            ("Define Architecture", "architecture tab — MLP dimension list"),
            ("Configure Training", "epochs, batch size, learning rate, device"),
            ("Train", "Train & evaluate tab (torch, background job)"),
            ("Monitor Curves", "live loss/epoch curves on the runboard"),
            ("Evaluate", "held-out metrics on the runboard"),
            ("Inspect Predictions", "prediction table per run"),
            ("Visualize", "learning-curve + prediction plots"),
            ("Save Checkpoint/Export", "checkpoint save/resume; run export"),
        ),
        "DL-specific configuration stays separate from classical ML — "
        "this workspace never reuses the scikit-learn registry forms.",
    ),
    "generative-ai": (
        "Select Generation Method → Define Starting Molecules/Conditions → "
        "Configure Constraints/Objectives → Generate → Validate → Calculate "
        "Properties → Filter → Rank → Visualize → Export",
        (
            ("Select Generation Method", "Generative — method tab (GA, SMILES RNN, scaffold hop)"),
            ("Starting Molecules", "seed SMILES / corpus fields"),
            ("Constraints/Objectives", "objective picker (QED, LogP band, SA, similarity)"),
            ("Generate", "Evolve / generate (background job)"),
            ("Validate", "invalid structures dropped, per-row reasons"),
            ("Properties", "descriptor panel over the results"),
            ("Filter", "objective thresholds; SA filter"),
            ("Rank", "leaderboard by score"),
            ("Visualize", "score history + property plots"),
            ("Export", "leaderboard CSV; chart exports"),
        ),
        "Synthetic accessibility uses the validated SA implementation, "
        "never a placeholder score.",
    ),
    "molecular-docking": (
        "Select Ligands → Validate/Prepare Ligands → Select/Prepare "
        "Receptor → Define Binding Site/Grid → Configure Engine → Run "
        "Docking → Inspect Poses → Analyze Interactions → Visualize → "
        "Export",
        (
            ("Select Ligands", "Docking — ligand set (SMILES table or prepared set)"),
            ("Validate/Prepare Ligands", "meeko PDBQT preparation, per-ligand errors surfaced"),
            ("Select/Prepare Receptor", "receptor picker + preparation"),
            ("Binding Site/Grid", "Vina box coordinates"),
            ("Configure Engine", "exhaustiveness, seeds, engine picker"),
            ("Run Docking", "Run docking (Vina on PATH; Doctor verifies)"),
            ("Inspect Poses", "pose table with scores"),
            ("Analyze Interactions", "Contacts & PLIF tab — contact maps, PLIF fingerprints"),
            ("Visualize", "pose + interaction figures"),
            ("Export", "pose/analysis exports"),
        ),
        "Preparation and docking errors are displayed per ligand — never "
        "a silent empty result.",
    ),
    "pkpd": (
        "Import Data → Inspect Dose/Time/Response → Select Model → "
        "Configure Parameters → Fit/Simulate → Evaluate Diagnostics → "
        "Visualize Exposure/Response → Export",
        (
            ("Import Data", "PK/PD — dataset/table picker incl. Data Explorer"),
            ("Inspect Dose/Time/Response", "dose table + concentration preview"),
            ("Select Model", "compartment model picker"),
            ("Configure Parameters", "parameter form"),
            ("Fit/Simulate", "Fit / Run simulation (background job)"),
            ("Evaluate Diagnostics", "residuals, parameter confidence"),
            ("Visualize", "exposure/response curves"),
            ("Export", "curve + table exports"),
        ),
        "Unsupported calculations are never displayed as completed "
        "features — the page refuses honestly.",
    ),
    "molecular-dynamics-analysis": (
        "Load Topology/Trajectory → Validate → Select Analysis → Execute → "
        "Inspect Results → Visualize → Export",
        (
            ("Load Topology/Trajectory", "Molecular Dynamics — open study folder (or example study)"),
            ("Validate", "per-series axis/units confirmation on load"),
            ("Select Analysis", "ligand + metric pickers (RMSD, RMSF, …)"),
            ("Execute", "Plot metric / Plot correlation"),
            ("Inspect Results", "metric stats panel"),
            ("Visualize", "figure preview pane"),
            ("Export", "Export figure… with validation"),
        ),
        "This workspace ANALYZES trajectories; it does not run "
        "simulations, and never claims to.",
    ),
    "visualization": (
        "Select Dataset/Model Result → Select Plot → Configure "
        "Variables/Style → Generate → Inspect → Filter/Compare → Export",
        (
            ("Select Dataset/Model Result", "Image Lab — source picker (datasets, model results)"),
            ("Select Plot", "the six tabs + Backend figures registry browser"),
            ("Configure Variables/Style", "axis labels, palette, per-plot options"),
            ("Generate", "each tab's compute/plot button"),
            ("Inspect", "the figure panes"),
            ("Filter/Compare", "matrix filters; experiment comparison in Utilities"),
            ("Export", "chart export everywhere (640×480 legibility floor)"),
        ),
        "Results from other workspaces arrive by REFERENCE — the shared "
        "dataset/result model, no duplicated DataFrames.",
    ),
    "introduction": (
        "Purpose → overview → workflow explanation → workspace navigation",
        (
            ("Purpose", "this page — what the Studio is"),
            ("Overview", "key backend capabilities"),
            ("Workflow explanation", "the nine stage chains (each workspace's strip)"),
            ("Navigation", "workspace entries + the horizontal bar, Ctrl+1..9"),
        ),
        "The Introduction workspace orients; the work happens in the "
        "other eight.",
    ),
}


class WorkflowGuide(QWidget):
    """One workspace's §13 stage chain as a collapsible strip.

    A heading button expands/collapses the stage table.  Each stage is
    a row: the stage NAME (verbatim from the directive's chain) and
    WHERE it lives in this workspace.  The note line carries the
    workspace's §13-specific instruction (provenance, honesty rules).
    """

    def __init__(self, workspace_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        entry = WORKFLOWS.get(workspace_key)
        if entry is None:
            # An unknown key is a programming error: fail loudly at
            # construction, never render an empty strip.
            raise ValueError(
                f"no §13 workflow registered for workspace {workspace_key!r} — "
                f"known: {', '.join(sorted(WORKFLOWS))}"
            )
        chain, stages, note = entry
        self._workspace_key = workspace_key

        self.setObjectName("workflowGuide")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self._toggle = QPushButton("Workflow — the stages of this workspace")
        self._toggle.setObjectName("workflowGuideToggle")
        self._toggle.setCheckable(True)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setToolTip(chain)
        self._toggle.setMinimumHeight(26)
        lay.addWidget(self._toggle)

        self._body = QWidget()
        self._body.setObjectName("workflowGuideBody")
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(8, 4, 8, 4)
        body_lay.setSpacing(1)

        chain_label = QLabel(chain)
        chain_label.setObjectName("workflowGuideChain")
        chain_label.setWordWrap(True)
        body_lay.addWidget(chain_label)

        for stage, where in stages:
            row = QHBoxLayout()
            row.setSpacing(8)
            name = QLabel(f"<b>{stage}</b>")
            name.setObjectName("workflowGuideStage")
            name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            place = QLabel(where)
            place.setObjectName("workflowGuideWhere")
            place.setWordWrap(True)
            place.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            row.addWidget(name)
            row.addWidget(place, stretch=1)
            holder = QWidget()
            holder.setLayout(row)
            body_lay.addWidget(holder)

        note_label = QLabel(note)
        note_label.setObjectName("workflowGuideNote")
        note_label.setWordWrap(True)
        body_lay.addWidget(note_label)

        lay.addWidget(self._body)

        # collapsed by default: one quiet row; expand on demand
        self._body.hide()
        self._toggle.toggled.connect(self._body.setVisible)

    # -- API -----------------------------------------------------------------
    def is_expanded(self) -> bool:
        """Whether the stage table is currently shown."""
        return self._body.isVisible() or self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)
        self._body.setVisible(expanded)

    def stage_count(self) -> int:
        """The number of stages in this workspace's chain (test hook)."""
        return len(WORKFLOWS[self._workspace_key][1])
