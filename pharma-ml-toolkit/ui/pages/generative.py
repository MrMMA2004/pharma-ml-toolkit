"""Generative Workbench: evolution, Pareto front, molecule surgery, oracles (audit row 9).

The Studio's increment-8 page (the playground Generative tab). The
audit row's clusters -- the objective & evolution run (GA / RL / MCTS),
the multi-objective trade-off set, the molecule surgery operators and
the GuacaMol scoring oracles -- each wired to the SAME public backend
APIs the playground cards call, over a shared corpus selection:

- **Evolve** -- the objective (QED, the LogP / MW / TPSA windows, the
  QED+LogP blend, the multi-objective triple, similarity to a
  reference) built exactly like the playground's ``build_objective``,
  then ``GeneticOptimizer`` (structure-space search), ``ReinventOptimizer``
  (policy-gradient fine-tuning of an on-the-fly SMILES RNN prior, the
  CLI idiom: seeded mode anchors the corpus ahead of the bundle) or
  ``MctsOptimizer`` (ChemTSv2-style MCTS over SMILES prefixes): the
  ranked leaderboard (``GenerationResult.to_dataframe()``) with the
  per-generation best / mean score curves from ``history``.
- **Pareto front** -- ``pareto_front`` over the LAST run's breakdown:
  the non-dominated set as a table + the trade-off scatter over the
  first two objectives; a single-objective run refuses honestly (the
  front needs the multi-objective breakdown).
- **Molecule surgery** -- ``ring_topology_moves`` (every distinct
  one-step ring-size mutant of one molecule: contraction and expansion
  candidates over saturated rings only) and ``scaffold_hop_crossover``
  (the BREED-style breeding of two parents: bridge / ring-merge /
  linker-transplant cuts compete for each draw), the mutants and
  children scored by ``QedObjective`` for the bars.
- **Scoring oracles** -- the GuacaMol benchmark score book for one
  molecule: ``qed_score`` / ``sa_score`` / ``penalized_logp`` /
  ``rediscovery_score`` plus the Lipinski-style ``property_filter``
  with its values and violations in the table.

Every heavy path runs in a :class:`~ui.workers.job_worker.JobWorker`;
they are single-shot engine calls without checkpoint loops, so the
cooperative Stop only lands for checkpoint-wired jobs -- the same
disclosure the PK/PD, Biology, Docking, Unsupervised and Deep Learning
pages carry.
"""

from __future__ import annotations

import random
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..font_loader import set_mono
from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import configure_splitter
from ..widgets.smiles_preview import SmilesDetailDialog, attach_smiles_previews
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[2]

#: The page's corpora: the bundled 12-molecule seed table or the
#: library's built-in 244-molecule drug-like corpus.
_SEED_CORPUS_ENTRY = "seed_molecules.csv (bundled 12)"
_BUILTIN_CORPUS_ENTRY = "built-in drug-like corpus (244)"

_OBJECTIVES: list[tuple[str, str]] = [
    ("qed", "QED drug-likeness"),
    ("logp", "LogP window"),
    ("mw", "Molecular weight window"),
    ("tpsa", "TPSA window"),
    ("blend", "QED + LogP blend"),
    ("multi", "Multi-objective (QED × MW × TPSA)"),
    ("similarity", "Similarity to reference SMILES"),
    # (part3 §24.5) the Ertl SA scale as an evolution reward: the window
    # entries become the easy/hard anchors (at/below easy -> 1.0).
    ("sa", "SA score (ease of synthesis)"),
]

_METHODS: list[tuple[str, str]] = [
    ("ga", "genetic algorithm"),
    ("rl", "reinforcement learning (RL)"),
    ("mcts", "monte carlo tree search (MCTS)"),
]


def _build_objective(key: str, lo: float, hi: float, reference: str) -> Any:
    """Instantiate the objective the playground's build_objective builds."""
    from pharmaml.generative import (
        LogPBandObjective,
        MolecularWeightBandObjective,
        MultiObjective,
        QedObjective,
        SAScoreObjective,
        SimilarityObjective,
        TpsaBandObjective,
    )

    if key == "qed":
        return QedObjective()
    if key == "logp":
        return LogPBandObjective(lo, hi)
    if key == "mw":
        return MolecularWeightBandObjective(lo, hi)
    if key == "tpsa":
        return TpsaBandObjective(lo, hi)
    if key == "blend":
        return MultiObjective([QedObjective(), LogPBandObjective(lo, hi)], weights=[0.7, 0.3])
    if key == "multi":
        # the CLI's 'druglike' recipe: equal-weight triple whose blended
        # leaderboard hides the trade-offs the Pareto front tab surfaces.
        return MultiObjective(
            [
                QedObjective(),
                MolecularWeightBandObjective(150, 450),
                TpsaBandObjective(30, 90),
            ]
        )
    if key == "similarity":
        return SimilarityObjective(reference)
    if key == "sa":
        # (part3 §24.5) the window entries are the easy/hard reward anchors
        # (1 <= easy < hard <= 10; sensible: 1.0/3.0 or 2.0/7.0).
        return SAScoreObjective(easy=lo, hard=hi)
    raise ValueError(f"unknown objective key: {key}")


def _seed_molecules() -> list[str]:
    import csv

    path = _TK / "examples" / "data" / "seed_molecules.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row["smiles"].strip() for row in rows if row.get("smiles", "").strip()]


@dataclass
class EvolveOutcome:
    """What an evolution job hands back."""

    result: Any = None
    method_label: str = ""
    objective_label: str = ""
    mode: str = ""
    seconds: float = 0.0


@dataclass
class FrontOutcome:
    """What the Pareto job hands back."""

    front: Any = None
    objectives: list[str] = field(default_factory=list)
    deepest: int = 0
    n_total: int = 0
    seconds: float = 0.0


@dataclass
class SurgeryOutcome:
    """What a surgery job hands back (topology moves or breeding)."""

    kind: str = ""
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    parent_score: float | None = None
    seconds: float = 0.0


@dataclass
class OracleOutcome:
    """What the oracle job hands back."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict: str = ""
    summary: str = ""
    seconds: float = 0.0


@dataclass
class SAScoreOutcome:
    """What the SA scoring job hands back (part3 §24)."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""
    seconds: float = 0.0


class GenerativeWorkbenchPage(QWidget):
    """The Generative tab's clusters over a shared corpus."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._seeds = _seed_molecules()
        self._last_run: EvolveOutcome | None = None
        self._last_result: Any = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # -- shared corpus selection --------------------------------------------
        top = QFormLayout()
        corpus_row = QWidget()
        corpus_lay = QHBoxLayout(corpus_row)
        corpus_lay.setContentsMargins(0, 0, 0, 0)
        self._corpus_combo = QComboBox()
        self._corpus_combo.addItem(_SEED_CORPUS_ENTRY)
        self._corpus_combo.addItem(_BUILTIN_CORPUS_ENTRY)
        corpus_lay.addWidget(self._corpus_combo)
        self._seeded_check = QCheckBox("seeded mode (the lines anchor the run)")
        self._seeded_check.setToolTip(
            "GA: the molecules seed the population. RL/MCTS: the prior's corpus is "
            "anchored with them ahead of the bundle (the CLI idiom)."
        )
        corpus_lay.addWidget(self._seeded_check)
        corpus_lay.addStretch(1)
        top.addRow("corpus:", corpus_row)
        lay.addLayout(top)

        # -- the engine tabs ------------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_evolve_tab(), "Evolve")
        self._tabs.addTab(self._build_front_tab(), "Pareto front")
        self._tabs.addTab(self._build_surgery_tab(), "Molecule surgery")
        self._tabs.addTab(self._build_oracle_tab(), "Scoring oracles")
        self._tabs.addTab(self._build_sa_tab(), "SA score")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        # Part-3 §22 (round 231): hover a SMILES cell briefly for the
        # 2-D structure preview; click it for the detail dialog.  The
        # controller resolves columns by name at hover time, so every
        # frame this shared table binds (SA board, evolution mutants,
        # topo/front children) gets the behaviour where a SMILES
        # column exists.
        self._smiles_previews = attach_smiles_previews(
            self._table, columns=("smiles", "canonical", "mutant", "child")
        )
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Generative panel")
        self._plot.set_empty_hint("Press Generate to sample molecules; scored results land here")

        splitter.addWidget(self._plot)
        splitter.setSizes([560, 540])
        configure_splitter(splitter, stretch=(0, 1))
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No generative run yet — pick an objective and evolve.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

        # (R30-c) the per-page chart export under the live palette
        from ..widgets.chart_export import attach_chart_export

        _export_row = QHBoxLayout()
        attach_chart_export(
            self, self._plot, "generative-panel",
            layout=_export_row, status_label=self._report,
        )
        _export_row.addStretch(1)
        lay.addLayout(_export_row)

        # §21.5 (round 236): clicking a plotted molecule opens its
        # §22.5 detail view (structure + calculated properties).
        self._sa_hover: list[str] = []
        self._sa_detail: SmilesDetailDialog | None = None
        self._plot.point_selected.connect(self._on_point_selected)

    # -- §21.5: molecule points on the charts ---------------------------------
    @staticmethod
    def _molecule_meta(smiles: list[str]) -> Any:
        """Per-point hover metadata for a molecule scatter (§21.5).

        The readout shows the SMILES plus formula and molecular weight
        calculated from it -- never invented; molecules RDKit cannot
        parse answer with the SMILES alone (the SA y-value is already
        in the readout).  Computed lazily per point and cached by the
        inspector (§21.6), so plotting 500 candidates costs nothing
        until the user actually hovers one.
        """

        def meta(i: int) -> dict[str, Any]:
            smi = smiles[i] if 0 <= i < len(smiles) else ""
            rows: dict[str, Any] = {"SMILES": smi}
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, rdMolDescriptors

                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is not None:
                    rows["formula"] = rdMolDescriptors.CalcMolFormula(mol)
                    rows["MW"] = f"{Descriptors.MolWt(mol):.2f} g/mol"  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001, S110 - hover metadata must never raise (nor log noise on a hover)
                pass
            return rows

        return meta

    def _on_point_selected(self, label: str, index: int) -> None:
        """§21.5 click-to-detail: the SA scatter's molecule points."""
        if label != "SA score" or not (0 <= index < len(self._sa_hover)):
            return
        self._sa_detail = SmilesDetailDialog(self._sa_hover[index], self)
        self._sa_detail.show()  # non-modal per §22.5

    # -- tab construction -----------------------------------------------------
    def _build_evolve_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._objective = QComboBox()
        for _key, label in _OBJECTIVES:
            self._objective.addItem(label)
        self._objective.setCurrentText("QED drug-likeness")
        form.addRow("objective:", self._objective)
        window_row = QWidget()
        window_lay = QHBoxLayout(window_row)
        window_lay.setContentsMargins(0, 0, 0, 0)
        self._obj_lo = QLineEdit("1.0")
        self._obj_lo.setMaximumWidth(70)
        self._obj_hi = QLineEdit("3.0")
        self._obj_hi.setMaximumWidth(70)
        window_lay.addWidget(QLabel("window low:"))
        window_lay.addWidget(self._obj_lo)
        window_lay.addWidget(QLabel("high:"))
        window_lay.addWidget(self._obj_hi)
        window_lay.addStretch(1)
        form.addRow("", window_row)
        self._obj_ref = QLineEdit("CC(=O)Oc1ccccc1C(=O)O")
        self._obj_ref.setToolTip(
            "The similarity objective's reference molecule (other objectives ignore it)."
        )
        form.addRow("reference SMILES:", self._obj_ref)
        self._method = QComboBox()
        for _key, label in _METHODS:
            self._method.addItem(label)
        form.addRow("optimiser:", self._method)
        knobs = QWidget()
        knobs_lay = QHBoxLayout(knobs)
        knobs_lay.setContentsMargins(0, 0, 0, 0)
        self._pop_size = QSpinBox()
        self._pop_size.setRange(8, 96)
        self._pop_size.setValue(24)
        knobs_lay.addWidget(QLabel("population:"))
        knobs_lay.addWidget(self._pop_size)
        self._generations = QSpinBox()
        self._generations.setRange(4, 48)
        self._generations.setValue(16)
        knobs_lay.addWidget(QLabel("generations:"))
        knobs_lay.addWidget(self._generations)
        self._rl_iterations = QSpinBox()
        self._rl_iterations.setRange(2, 60)
        self._rl_iterations.setValue(12)
        knobs_lay.addWidget(QLabel("RL iterations:"))
        knobs_lay.addWidget(self._rl_iterations)
        self._mcts_iterations = QSpinBox()
        self._mcts_iterations.setRange(4, 200)
        self._mcts_iterations.setValue(20)
        knobs_lay.addWidget(QLabel("MCTS iterations:"))
        knobs_lay.addWidget(self._mcts_iterations)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        knobs_lay.addWidget(QLabel("Seed:"))
        knobs_lay.addWidget(self._seed_spin)
        # §17 F (Phase 5): the search-budget knobs explain themselves
        from pharmaml.studio import apply_help as _help

        _help(self._mcts_iterations, "mcts_iterations")
        _help(self._rl_iterations, "epochs")
        _help(self._seed_spin, "random_seed")
        _help(self._generations, "population_size")
        form.addRow("budget:", knobs)
        self._evolve_btn = QPushButton("Evolve")
        self._evolve_btn.setMinimumHeight(34)
        self._evolve_btn.clicked.connect(self._start_evolve)
        form.addRow("", self._evolve_btn)
        hint = QLabel(
            "GeneticOptimizer (structure-space search: topology mutation + splice "
            "crossover over the corpus), ReinventOptimizer (policy-gradient "
            "fine-tuning of an on-the-fly SMILES RNN prior) or MctsOptimizer "
            "(ChemTSv2-style MCTS over SMILES prefixes) — the ranked leaderboard "
            "plus the per-generation best/mean score curves."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_front_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._front_btn = QPushButton("Extract the Pareto front of the last run")
        self._front_btn.setMinimumHeight(34)
        self._front_btn.clicked.connect(self._start_front)
        form.addRow("", self._front_btn)
        hint = QLabel(
            "pareto_front over the last run's per-objective breakdown: the "
            "non-dominated set (a >= b on every objective and a > b on at least "
            "one), leaderboard order preserved, the trade-off scatter over the "
            "first two objectives. A single-objective run refuses honestly — "
            "run 'QED + LogP blend' or the multi-objective triple first."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_surgery_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._topo_smiles = QLineEdit("C1CCNCC1")
        form.addRow("ring surgery SMILES:", self._topo_smiles)
        self._topo_btn = QPushButton("Ring topology moves")
        self._topo_btn.setMinimumHeight(34)
        self._topo_btn.clicked.connect(self._start_topology)
        form.addRow("", self._topo_btn)
        breed_row = QWidget()
        breed_lay = QHBoxLayout(breed_row)
        breed_lay.setContentsMargins(0, 0, 0, 0)
        self._breed_a = QLineEdit("CC(=O)Oc1ccccc1C(=O)O")
        self._breed_b = QLineEdit("CC(C)Cc1ccc(O)cc1")
        breed_lay.addWidget(QLabel("parent A:"))
        breed_lay.addWidget(self._breed_a)
        breed_lay.addWidget(QLabel("parent B:"))
        breed_lay.addWidget(self._breed_b)
        form.addRow("breed:", breed_row)
        breed_knobs = QWidget()
        breed_knobs_lay = QHBoxLayout(breed_knobs)
        breed_knobs_lay.setContentsMargins(0, 0, 0, 0)
        self._breed_count = QSpinBox()
        self._breed_count.setRange(1, 8)
        self._breed_count.setValue(4)
        breed_knobs_lay.addWidget(QLabel("children:"))
        breed_knobs_lay.addWidget(self._breed_count)
        self._breed_seed = QSpinBox()
        self._breed_seed.setRange(0, 999999)
        self._breed_seed.setValue(42)
        breed_knobs_lay.addWidget(QLabel("Seed:"))
        breed_knobs_lay.addWidget(self._breed_seed)
        breed_knobs_lay.addStretch(1)
        form.addRow("", breed_knobs)
        self._breed_btn = QPushButton("Breed (scaffold-hop crossover)")
        self._breed_btn.setMinimumHeight(34)
        self._breed_btn.clicked.connect(self._start_breed)
        form.addRow("", self._breed_btn)
        hint = QLabel(
            "ring_topology_moves: every distinct one-step ring-size mutant "
            "(contraction / expansion candidates over saturated rings only — "
            "aromatic and fused/spiro rings are never re-sized). "
            "scaffold_hop_crossover: the BREED-style breeding of two parents — "
            "bridge / ring-merge / linker-transplant / fused-ring-splice cuts "
            "compete for each draw. The bars score every product with the QED "
            "objective."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_oracle_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._oracle_smiles = QLineEdit("CC(=O)Oc1ccccc1C(=O)O")
        form.addRow("SMILES:", self._oracle_smiles)
        self._oracle_ref = QLineEdit("O=C(O)c1ccccc1O")
        form.addRow("rediscovery target:", self._oracle_ref)
        self._oracle_btn = QPushButton("Score (the GuacaMol book)")
        self._oracle_btn.setMinimumHeight(34)
        self._oracle_btn.clicked.connect(self._start_oracle)
        form.addRow("", self._oracle_btn)
        hint = QLabel(
            "Every oracle at once: qed_score (weighted desirability), sa_score "
            "(synthetic accessibility), penalized_logp, rediscovery_score (the "
            "Tanimoto similarity to the target) and the Lipinski-style "
            "property_filter with its values and violations."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_sa_tab(self) -> QWidget:
        """(part3 §24) the batch Synthetic Accessibility scoring tab."""
        tab = QWidget()
        form = QFormLayout(tab)
        self._sa_molecules = QPlainTextEdit(
            "CC(=O)Oc1ccccc1C(=O)O\n"          # aspirin
            "CC(C)Cc1ccc(O)cc1\n"              # ibuprofen core
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"      # caffeine
            "CC1=C(C[C@@H]2[C@]1(C[C@H]([C@@H](C2=O)O)"
            "NC(=O)c3ccccc3)O)OC(=O)C"          # the taxane-style complex
        )
        self._sa_molecules.setPlaceholderText("one SMILES per line")
        set_mono(self._sa_molecules)  # SMILES stay mono under every provider
        form.addRow("molecules:", self._sa_molecules)
        self._sa_max = QLineEdit("")
        self._sa_max.setPlaceholderText("blank = no filter (e.g. 4.0)")
        form.addRow("max SA:", self._sa_max)
        self._sa_btn = QPushButton("Score molecules")
        self._sa_btn.setMinimumHeight(34)
        self._sa_btn.clicked.connect(self._start_sa_score)
        form.addRow("", self._sa_btn)
        hint = QLabel(
            "Ertl–Schuffenhauer synthetic accessibility per molecule — "
            "lower = estimated easier to synthesise (a computational "
            "estimate from complexity and fragment familiarity, not a "
            "route plan). Invalid rows keep their reason; the optional "
            "max-SA bound flags rejects without discarding them. "
            "The chart plots every scored molecule (index vs SA): hover "
            "a point for its SMILES, formula and weight; click it for "
            "the detail dialog (structure + canonical SMILES + SA). "
            "Hover a SMILES cell in the table for the 2-D structure."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- job plumbing -----------------------------------------------------------
    def _buttons(self) -> tuple[QPushButton, ...]:
        return (
            self._evolve_btn,
            self._front_btn,
            self._topo_btn,
            self._breed_btn,
            self._oracle_btn,
            self._sa_btn,
        )

    def _on_started(self, name: str) -> None:
        for btn in self._buttons():
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in self._buttons():
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return True
        return False

    def _corpus_lines(self) -> list[str]:
        """The corpus selection as SMILES lines (empty = the built-in default)."""
        if self._corpus_combo.currentText() == _SEED_CORPUS_ENTRY:
            return list(self._seeds)
        return []

    # -- evolve ---------------------------------------------------------------------
    def _start_evolve(self) -> None:
        if self._busy_check():
            return
        objective_key = _OBJECTIVES[self._objective.currentIndex()][0]
        objective_label = _OBJECTIVES[self._objective.currentIndex()][1]
        try:
            lo = float(self._obj_lo.text() or "1.0")
            hi = float(self._obj_hi.text() or "3.0")
        except ValueError:
            self._report.setText("The window bounds must be numbers.")
            return
        if hi < lo:
            self._report.setText("The window's high bound must be >= the low bound.")
            return
        reference = self._obj_ref.text().strip()
        if objective_key == "similarity" and not reference:
            self._report.setText("The similarity objective needs a reference SMILES.")
            return
        method_key = _METHODS[self._method.currentIndex()][0]
        method_label = _METHODS[self._method.currentIndex()][1]
        population = self._pop_size.value()
        generations = self._generations.value()
        rl_iterations = self._rl_iterations.value()
        mcts_iterations = self._mcts_iterations.value()
        seed = self._seed_spin.value()
        lines = self._corpus_lines()
        seeded = self._seeded_check.isChecked() and bool(lines)

        def job(progress) -> EvolveOutcome:
            started = _time.monotonic()
            objective = _build_objective(objective_key, lo, hi, reference)
            optimizer: Any
            if method_key == "rl":
                from pharmaml.generative import DEFAULT_CORPUS, ReinventOptimizer

                # the CLI idiom (mirrored exactly): empty = the built-in
                # prior corpus; seeded mode anchors the lines ahead of the
                # bundle; unseeded lines are the sampling pool alone.
                if not lines:
                    corpus_arg: Any = None
                elif seeded:
                    corpus_arg = list(dict.fromkeys([*lines, *DEFAULT_CORPUS]))
                else:
                    corpus_arg = lines
                optimizer = ReinventOptimizer(
                    objective,
                    corpus=corpus_arg,
                    iterations=rl_iterations,
                    batch_size=16,
                    prior_epochs=8,
                    seed=seed,
                )
                result = optimizer.optimize()
                mode = f"RL · {'seeded' if seeded else ('custom corpus' if lines else 'built-in corpus')}"
            elif method_key == "mcts":
                from pharmaml.generative import DEFAULT_CORPUS, MctsOptimizer

                # the MCTS corpus semantics mirror the RL prior's
                if not lines:
                    corpus_arg = None
                elif seeded:
                    corpus_arg = list(dict.fromkeys([*lines, *DEFAULT_CORPUS]))
                else:
                    corpus_arg = lines
                optimizer = MctsOptimizer(
                    objective,
                    corpus=corpus_arg,
                    iterations=mcts_iterations,
                    seed=seed,
                )
                result = optimizer.optimize()
                mode = f"MCTS · {'seeded' if seeded else ('custom corpus' if lines else 'built-in corpus')}"
            else:
                from pharmaml.generative import GeneticOptimizer

                optimizer = GeneticOptimizer(
                    objective,
                    population_size=population,
                    generations=generations,
                    mutation="topology",
                    crossover="splice",
                    operator_selection="uniform",
                    seed=seed,
                )
                if not lines:
                    result = optimizer.optimize()
                    mode = "built-in corpus"
                elif seeded:
                    result = optimizer.optimize(seed_smiles=lines)
                    mode = "seeded"
                else:
                    result = optimizer.optimize(corpus=lines)
                    mode = "custom corpus"
            progress(1.0, "done")
            return EvolveOutcome(
                result=result,
                method_label=method_label,
                objective_label=objective_label,
                mode=mode,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"{method_label} · {objective_label}",
            job,
            on_finished=self._render_evolve,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_evolve(self, outcome: EvolveOutcome) -> None:
        self._last_run = outcome
        self._last_result = outcome.result
        result = outcome.result
        self._plot.clear()
        history = list(result.history)
        if history:
            generations = [int(rec.get("generation", i)) for i, rec in enumerate(history)]
            best = [float(rec.get("best_score", 0.0)) for rec in history]
            mean = [float(rec.get("mean_score", 0.0)) for rec in history]
            import numpy as np

            epochs = np.asarray(generations, dtype=float)
            self._plot.line(epochs, np.asarray(best), label="best score", width=2, index=0)
            self._plot.line(epochs, np.asarray(mean), label="mean score", width=1, index=1)
            self._plot.add_legend()
            self._plot.set_labels(
                x="generation" if "generation" in history[0] else "iteration",
                y="score",
                title=f"{outcome.method_label} · the run's progress",
            )
        frame = result.to_dataframe()
        self._table.bind_frame(frame.head(20))
        elapsed = result.metadata.get("elapsed_s")
        timing = f" · {elapsed:.1f} s" if isinstance(elapsed, (int, float)) else ""
        self._report.setText(
            f"{outcome.method_label} · {outcome.objective_label} · best "
            f"{result.best_score:.3f} — {result.best} · {len(result.smiles)} unique "
            f"molecules · {outcome.mode}{timing}"
        )

    # -- pareto front ---------------------------------------------------------------
    def _start_front(self) -> None:
        if self._busy_check():
            return
        if self._last_result is None:
            self._report.setText("Run an evolution first — the front reads the LAST run's breakdown.")
            return
        result = self._last_result
        if not any(result.breakdown):
            self._report.setText(
                "This run optimises a single objective — the Pareto front needs a "
                "multi-objective run ('QED + LogP blend' or 'Multi-objective "
                "(QED × MW × TPSA)')."
            )
            return

        def job(progress) -> FrontOutcome:
            from pharmaml.generative import pareto_front

            started = _time.monotonic()
            progress(0.5, "scanning dominance")
            front = pareto_front(result)
            progress(1.0, "done")
            deepest = max((result.smiles.index(s) for s in front.smiles), default=0) + 1
            return FrontOutcome(
                front=front,
                objectives=list(front.objectives),
                deepest=deepest,
                n_total=len(result),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "pareto front of the last run",
            job,
            on_finished=self._render_front,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_front(self, outcome: FrontOutcome) -> None:
        # part5 §11: the real front becomes a saved run (best-effort).
        try:
            from pharmaml.studio import record_run

            n_front = len(outcome.front.to_dataframe()) if outcome.front is not None else 0
            record_run(
                "generative",
                "Pareto front · " + ", ".join(outcome.objectives),
                {
                    "kind": "pareto-front",
                    "objectives": list(outcome.objectives),
                    "n_front": n_front,
                    "n_total": int(outcome.n_total),
                    "deepest": int(outcome.deepest),
                },
            )
        except Exception:  # noqa: BLE001, S110 -- the record must never fail the render
            pass
        self._plot.clear()
        front = outcome.front
        frame = front.to_dataframe()
        if len(outcome.objectives) >= 2:
            x_name, y_name = outcome.objectives[0], outcome.objectives[1]
            xs = frame[x_name].to_numpy(dtype=float)
            ys = frame[y_name].to_numpy(dtype=float)
            # §21.5: front members are molecules -- hover carries the
            # SMILES/formula/MW readout (click stays SA-chart-only:
            # the front's drill-down is the shared table beside it).
            self._plot.scatter(
                xs,
                ys,
                label="non-dominated",
                meta=self._molecule_meta(list(front.smiles)),
            )
            self._plot.add_legend()
            self._plot.set_labels(
                x=x_name,
                y=y_name,
                title=f"Pareto front · {len(front.smiles)} of {outcome.n_total} non-dominated",
            )
        self._table.bind_frame(frame)
        deepest = (
            f" · deepest front member sits at leaderboard rank {outcome.deepest} of "
            f"{outcome.n_total} — a trade-off the blended ranking hides"
            if outcome.deepest > 1
            else ""
        )
        self._report.setText(f"{front!s}{deepest} · extracted in {outcome.seconds:.1f}s")

    # -- molecule surgery -----------------------------------------------------------
    def _start_topology(self) -> None:
        if self._busy_check():
            return
        smiles = self._topo_smiles.text().strip()
        if not smiles:
            self._report.setText("Type a SMILES in the ring surgery row first.")
            return

        def job(progress) -> SurgeryOutcome:
            from pharmaml.generative import QedObjective, ring_topology_moves

            started = _time.monotonic()
            progress(0.4, "enumerating the move set")
            mutants = ring_topology_moves(smiles)
            progress(0.8, "scoring")
            objective = QedObjective()
            scores = [float(objective.score(mutant)) for mutant in mutants]
            parent_score = float(objective.score(smiles))
            rows = [
                {"mutant": mutant, "qed": score}
                for mutant, score in zip(mutants, scores)
            ]
            summary = (
                f"{len(mutants)} distinct one-step ring-size mutant(s) of {smiles} — "
                "contraction and expansion candidates, sorted"
                if mutants
                else f"{smiles} has no qualifying saturated ring — aromatic or "
                "fused/spiro rings are never re-sized, and acyclic molecules "
                "have no moves"
            )
            progress(1.0, "done")
            return SurgeryOutcome(
                kind="topology",
                table=pd.DataFrame(rows) if rows else pd.DataFrame(columns=["mutant", "qed"]),
                summary=summary,
                scores=scores,
                labels=[f"m{i + 1}" for i in range(len(mutants))],
                parent_score=parent_score,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"ring topology moves ({smiles})",
            job,
            on_finished=self._render_surgery,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _start_breed(self) -> None:
        if self._busy_check():
            return
        parent_a = self._breed_a.text().strip()
        parent_b = self._breed_b.text().strip()
        if not parent_a or not parent_b:
            self._report.setText("Both parent SMILES are required for breeding.")
            return
        count = self._breed_count.value()
        seed = self._breed_seed.value()

        def job(progress) -> SurgeryOutcome:
            from pharmaml.generative import QedObjective, scaffold_hop_crossover

            started = _time.monotonic()
            progress(0.4, "competing the cut geometries")
            rng = random.Random(seed)
            litter = [scaffold_hop_crossover(parent_a, parent_b, rng) for _ in range(count)]
            progress(0.8, "scoring")
            objective = QedObjective()
            viable = [child for child in litter if child]
            scores = [float(objective.score(child)) for child in viable]
            rows = [
                {"child": child, "qed": score} for child, score in zip(viable, scores)
            ]
            progress(1.0, "done")
            return SurgeryOutcome(
                kind="breed",
                table=pd.DataFrame(rows) if rows else pd.DataFrame(columns=["child", "qed"]),
                summary=(
                    f"{len(viable)}/{count} viable children — bridge / ring-merge / "
                    f"linker-transplant cuts competed for each draw (seed {seed})"
                ),
                scores=scores,
                labels=[f"c{i + 1}" for i in range(len(viable))],
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "scaffold-hop crossover",
            job,
            on_finished=self._render_surgery,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_surgery(self, outcome: SurgeryOutcome) -> None:
        import numpy as np

        self._plot.clear()
        if outcome.scores:
            self._plot.bars(
                np.asarray(outcome.scores, dtype=float), labels=outcome.labels
            )
            title = (
                f"ring mutants' QED (parent {outcome.parent_score:.3f})"
                if outcome.kind == "topology"
                else "children's QED"
            )
            self._plot.set_labels(x="product", y="QED", title=title)
        self._table.bind_frame(outcome.table)
        self._report.setText(f"{outcome.summary} · done in {outcome.seconds:.1f}s")

    # -- scoring oracles -------------------------------------------------------------
    def _start_oracle(self) -> None:
        if self._busy_check():
            return
        smiles = self._oracle_smiles.text().strip()
        if not smiles:
            self._report.setText("Type a SMILES to score first.")
            return
        reference = self._oracle_ref.text().strip()

        def job(progress) -> OracleOutcome:
            from pharmaml.chem.oracles import (
                penalized_logp,
                property_filter,
                qed_score,
                rediscovery_score,
                sa_score,
            )

            started = _time.monotonic()
            progress(0.5, "running the oracle book")
            qed = float(qed_score(smiles))
            sa = float(sa_score(smiles))
            logp = float(penalized_logp(smiles))
            redis = float(rediscovery_score(smiles, reference)) if reference else None
            props = property_filter(smiles)
            rows = [
                {"oracle": "QED", "value": qed},
                {"oracle": "SA score", "value": sa},
                {"oracle": "penalized logP", "value": logp},
            ]
            if redis is not None:
                rows.append({"oracle": f"rediscovery ({reference})", "value": redis})
            for key, value in props["values"].items():
                rows.append({"oracle": f"property {key}", "value": float(value)})
            progress(1.0, "done")
            verdict = "PASS" if props["ok"] else "VIOLATIONS"
            redis_note = f" · rediscovery {redis:.3f}" if redis is not None else ""
            summary = (
                f"QED {qed:.3f} · SA {sa:.2f} · logP {logp:.2f}{redis_note} · "
                f"property filter {verdict}"
            )
            if props["violations"]:
                summary += " — " + "; ".join(props["violations"])
            return OracleOutcome(
                table=pd.DataFrame(rows),
                verdict=verdict,
                summary=summary,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            f"the oracle book ({smiles})",
            job,
            on_finished=self._render_oracle,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_oracle(self, outcome: OracleOutcome) -> None:
        self._plot.clear()
        self._table.bind_frame(outcome.table)
        self._report.setText(f"{outcome.summary} · scored in {outcome.seconds:.1f}s")

    # -- SA score (part3 §24) ---------------------------------------------------------
    def _start_sa_score(self) -> None:
        if self._busy_check():
            return
        smiles = [
            line.strip()
            for line in self._sa_molecules.toPlainText().splitlines()
            if line.strip()
        ]
        if not smiles:
            self._report.setText("Paste at least one SMILES (one per line) first.")
            return
        raw_max = self._sa_max.text().strip()
        max_sa: float | None = None
        if raw_max:
            try:
                max_sa = float(raw_max)
            except ValueError:
                self._report.setText(f"max SA must be a number — got {raw_max!r}.")
                return
            if not 1.0 <= max_sa <= 10.0:
                self._report.setText("max SA must sit on the [1, 10] scale.")
                return

        def job(progress) -> SAScoreOutcome:
            from pharmaml.chem.sa_scoring import (
                filter_by_sa,
                sa_score_summary,
                sa_scores,
            )

            started = _time.monotonic()
            progress(0.4, "scoring the molecules")
            batch = sa_scores(smiles)
            summary = sa_score_summary(smiles)
            outcome = filter_by_sa(batch.results, max_sa=max_sa) if max_sa else None
            progress(0.8, "building the table")
            rows: list[dict[str, Any]] = []
            for index, result in enumerate(batch.sorted_by_score()):
                score = result.score if result.score is not None else float("nan")
                band = "easy" if score <= 3.5 else ("moderate" if score <= 6.0 else "hard")
                if outcome is not None:
                    rejected = any(r.smiles == result.smiles for r in outcome.rejected)
                    status = "rejected" if rejected else "kept"
                else:
                    status = "ok"
                rows.append(
                    {
                        "rank": index + 1,
                        "smiles": result.smiles,
                        "canonical": result.canonical_smiles or "",
                        "sa_score": score,
                        "band": band,
                        "status": status,
                    }
                )
            for result in batch.results:
                if not result.valid:
                    rows.append(
                        {
                            "rank": "",
                            "smiles": result.smiles,
                            "canonical": "",
                            "sa_score": float("nan"),
                            "band": "invalid",
                            "status": result.error or "invalid SMILES",
                        }
                    )
            frame = pd.DataFrame(
                rows,
                columns=["rank", "smiles", "canonical", "sa_score", "band", "status"],
            )
            if "mean" in summary:
                stats = (
                    f"{summary['n_scored']} scored / {summary['n_invalid']} invalid · "
                    f"mean {summary['mean']:.3f} ± {summary['std']:.3f} · "
                    f"range {summary['min']:.3f}–{summary['max']:.3f}"
                )
            else:
                stats = f"0 scored / {summary['n_invalid']} invalid · no valid molecules"
            if outcome is not None:
                stats += f" · filter kept {outcome.n_kept}, rejected {outcome.n_rejected}"
            stats += " · lower = easier · a computational estimate, not a route plan"
            progress(1.0, "done")
            return SAScoreOutcome(
                table=frame, summary=stats, seconds=_time.monotonic() - started
            )

        self._runner.submit(
            "the SA score batch",
            job,
            on_finished=self._render_sa_score,
            on_failed=self._failed,
            on_cancelled=self._cancelled,
            on_progress=self._progress.update_progress,
        )

    def _render_sa_score(self, outcome: SAScoreOutcome) -> None:
        self._plot.clear()
        # §21.5 (round 236): one point per SCORED molecule -- hover reads
        # the SMILES + formula + MW next to the SA value, click opens the
        # §22.5 detail view.  Invalid rows stay table-only (they have no
        # SA to plot); x is the table's row index so chart and board
        # agree on every molecule's identity.
        frame = outcome.table
        # the board is sorted easiest-first with integer ranks; invalid
        # rows ride below with rank "" and NaN SA -- table-only.
        scored = frame[frame["sa_score"].notna()] if not frame.empty else frame
        if not scored.empty:
            ranks = np.asarray([float(r) for r in scored["rank"]])
            sas = np.asarray([float(s) for s in scored["sa_score"]])
            self._sa_hover = [str(s) for s in scored["smiles"]]
            self._plot.scatter(ranks, sas, label="SA score",
                               meta=self._molecule_meta(self._sa_hover))
            self._plot.set_labels(
                x="rank (easiest first)",
                y="SA score",
                title=f"Ertl SA · {len(scored)} scored of {len(frame)}",
            )
        else:
            self._sa_hover = []
        self._table.bind_frame(outcome.table)
        self._report.setText(f"{outcome.summary} · scored in {outcome.seconds:.1f}s")
        # part5 §11: the batch's real SA stats — recomputed from the actual
        # outcome table, so the record can never drift from what was shown.
        try:
            from pharmaml.studio import record_run

            scored_values = (
                frame["sa_score"].dropna().to_numpy(dtype=float)
                if not frame.empty
                else np.zeros(0)
            )
            canonical = (
                frame["canonical"].replace("", np.nan).dropna()
                if not frame.empty
                else pd.Series(dtype=object)
            )
            record_run(
                "generative",
                "SA batch",
                {
                    "kind": "sa-batch",
                    "n_scored": int(scored_values.size),
                    "n_invalid": int((frame["band"] == "invalid").sum()) if not frame.empty else 0,
                    "unique": int(canonical.nunique()),
                    "mean": float(scored_values.mean()) if scored_values.size else None,
                    "std": float(scored_values.std()) if scored_values.size else None,
                    "min": float(scored_values.min()) if scored_values.size else None,
                    "max": float(scored_values.max()) if scored_values.size else None,
                },
            )
        except Exception:  # noqa: BLE001, S110 -- the record must never fail the render
            pass

    # -- shared finishers --------------------------------------------------------------
    def _failed(self, message: str) -> None:
        self._report.setText(f"Run failed — {message.splitlines()[0]}")

    def _cancelled(self, seconds: float) -> None:
        self._report.setText(f"Run cancelled after {seconds:.1f}s (cooperative stop).")

    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    # -- test accessors ------------------------------------------------------------
    def report_text(self) -> str:
        return self._report.text()
