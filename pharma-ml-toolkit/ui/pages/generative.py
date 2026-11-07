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

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
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

        # -- the four engine tabs ------------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_evolve_tab(), "Evolve")
        self._tabs.addTab(self._build_front_tab(), "Pareto front")
        self._tabs.addTab(self._build_surgery_tab(), "Molecule surgery")
        self._tabs.addTab(self._build_oracle_tab(), "Scoring oracles")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Generative panel")
        splitter.addWidget(self._plot)
        splitter.setSizes([560, 540])
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No generative run yet — pick an objective and evolve.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

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
        knobs_lay.addWidget(QLabel("seed:"))
        knobs_lay.addWidget(self._seed_spin)
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
        breed_knobs_lay.addWidget(QLabel("seed:"))
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

    # -- job plumbing -----------------------------------------------------------
    def _buttons(self) -> tuple[QPushButton, ...]:
        return (self._evolve_btn, self._front_btn, self._topo_btn, self._breed_btn, self._oracle_btn)

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
        self._plot.clear()
        front = outcome.front
        frame = front.to_dataframe()
        if len(outcome.objectives) >= 2:
            x_name, y_name = outcome.objectives[0], outcome.objectives[1]
            xs = frame[x_name].to_numpy(dtype=float)
            ys = frame[y_name].to_numpy(dtype=float)
            self._plot.scatter(xs, ys, label="non-dominated")
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
