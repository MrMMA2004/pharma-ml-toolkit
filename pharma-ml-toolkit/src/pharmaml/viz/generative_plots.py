"""Visual generative-design output — the optimiser's own learning curves.

The generative flagship's introspection artifacts, drawn as figures:

* :func:`plot_operator_trajectory` — the adaptive GA's operator mix per
  generation (the "GA that learns" learning curve), synced with the score
  trajectory any run already carries in ``history``.
* :func:`plot_pareto_front` — a MultiObjective run's trade-off scatter:
  every molecule muted, the non-dominated front
  (:func:`pharmaml.generative.pareto_front` over the plotted pair)
  highlighted and connected.
* :func:`plot_rl_introspection` — an RL run's prior-drift story: the
  score trajectory, the agent-vs-prior sequence log-probabilities, the
  squared drift the anchor penalises, and the exploration entropy — the
  four diagnostics every :func:`pharmaml.generative.run_rl_optimization`
  run records in ``history``.

Examples
--------
>>> from pharmaml.generative import GeneticOptimizer, QedObjective
>>> from pharmaml.viz import close_figure
>>> from pharmaml.viz.generative_plots import plot_operator_trajectory
>>> run = GeneticOptimizer(QedObjective(), population_size=8, generations=3,
...                        seed=42, mutation="topology",
...                        operator_selection="adaptive").optimize(
...     corpus=["CCO", "c1ccccc1", "CCCCN", "CC(=O)O"])
>>> fig = plot_operator_trajectory(run)          # score panel + mutation level
>>> (len(fig.axes), fig.axes[0].get_title(loc="left"))
(2, 'Adaptive operator trajectory')
>>> [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
['topology', 'character']
>>> round(run.metadata["operator_trajectory"][0]["mutation"]["topology"], 1)
0.5
>>> close_figure(fig)
>>> plot_operator_trajectory(run, levels=("nope",))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: level(s) ['nope'] not in this run's trajectory; available: ['geometry', 'level1', 'mutation']
>>> uniform = GeneticOptimizer(QedObjective(), population_size=8, generations=3,
...                            seed=42).optimize(corpus=["CCO", "c1ccccc1"])
>>> plot_operator_trajectory(uniform)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: metadata carries no operator_trajectory -- re-run with operator_selection='adaptive'

>>> from pharmaml.generative import GenerationResult
>>> from pharmaml.viz.generative_plots import plot_pareto_front
>>> multi = GenerationResult(
...     smiles=["A", "B", "C", "D"], scores=[0.9, 0.8, 0.7, 0.1],
...     breakdown=[{"qed": 0.9, "novelty": 0.2},
...                {"qed": 0.4, "novelty": 0.9},
...                {"qed": 0.3, "novelty": 0.3},
...                {"qed": 0.85, "novelty": 0.25}])
>>> fig = plot_pareto_front(multi, x="qed", y="novelty")
>>> (len(fig.axes), fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel())
(1, 'qed', 'novelty')
>>> [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
['dominated', 'Pareto front (3)']
>>> fig.axes[0].get_title(loc="left")
'Pareto front — 3 of 4 molecules non-dominated'
>>> close_figure(fig)
>>> plot_pareto_front(multi, x="qed", y="qed")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: x and y must name DIFFERENT objectives, both 'qed'

>>> from pharmaml.viz.generative_plots import plot_rl_introspection
>>> rl = GenerationResult(
...     smiles=["CCO"], scores=[0.7],
...     history=[
...         {"iteration": 0, "best_score": 0.4, "mean_score": 0.2,
...          "drift": 0.0, "entropy": 3.1,
...          "agent_logprob": -8.2, "prior_logprob": -8.2},
...         {"iteration": 1, "best_score": 0.6, "mean_score": 0.3,
...          "drift": 1.4, "entropy": 2.8,
...          "agent_logprob": -6.9, "prior_logprob": -8.2},
...     ],
...     metadata={"method": "reinforce_rl", "anchor_weight": 0.1})
>>> fig = plot_rl_introspection(rl)
>>> (len(fig.axes), fig.axes[0].get_title(loc="left"))
(4, 'RL introspection — policy drift from the prior')
>>> [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
['agent', 'prior (frozen)']
>>> fig.axes[2].get_ylabel()
'drift (mean (ΔlogP)²), anchor_weight=0.1'
>>> fig.axes[3].get_xlabel()
'iteration'
>>> close_figure(fig)
>>> ga = GenerationResult(smiles=["CCO"], scores=[0.5],
...     history=[{"iteration": 0, "best_score": 0.5, "mean_score": 0.3}])
>>> plot_rl_introspection(ga)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: history rows carry no ['drift', 'entropy'] diagnostics — only run_rl_optimization() results record them
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_operator_trajectory", "plot_pareto_front", "plot_rl_introspection"]

_LOG = get_logger(__name__)


def plot_operator_trajectory(
    result: Any,
    levels: Sequence[str] = ("mutation",),
    title: str = "Adaptive operator trajectory",
) -> Any:
    """One figure per run: score trajectory + the operator mix per generation.

    Stacks ``1 + len(levels)`` panels sharing the generation x-axis: the top
    panel plots ``result.history``'s best / population-mean scores, each
    further panel one adaptive level's per-operator selection probabilities
    from ``result.metadata["operator_trajectory"]`` (record 0 is the fair
    prior; the final record is the closing mix).

    Parameters
    ----------
    result:
        A :class:`~pharmaml.generative.GenerationResult` from an
        ``operator_selection="adaptive"`` run (anything exposing ``history``
        and ``metadata["operator_trajectory"]``; duck-typed like
        ``plot_table_comparison``'s ``comparison``).
    levels:
        Which trajectory levels to render (default ``("mutation",)`` — the
        flagship story; pass ``("level1", "mutation", "geometry")`` for all
        three).
    title:
        Figure title (left-anchored; retrieve with
        ``fig.axes[0].get_title(loc="left")``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``result`` carries no history, no ``operator_trajectory`` (a
        uniform run — the message names the fix), or unknown ``levels``.
    """
    history = getattr(result, "history", [])
    if not history:
        raise DataValidationError(
            "result carries no history -- pass the GenerationResult of a "
            "GeneticOptimizer.optimize() run"
        )
    metadata = getattr(result, "metadata", None) or {}
    trajectory = metadata.get("operator_trajectory")
    if not trajectory:
        raise DataValidationError(
            "metadata carries no operator_trajectory -- re-run with "
            "operator_selection='adaptive'"
        )
    levels = tuple(levels)
    known: set[str] = set()
    for record in trajectory:
        known.update(record)
    known.discard("generation")
    unknown = sorted(set(levels) - known)
    if unknown:
        raise DataValidationError(
            f"level(s) {unknown!r} not in this run's trajectory; available: "
            f"{sorted(known)!r}"
        )

    generations = [record["generation"] for record in trajectory]
    fig, raw_axes = new_figure(
        nrows=1 + len(levels),
        ncols=1,
        figsize=(8.4, 1.9 + 1.9 * len(levels)),
        sharex=True,
        gridspec_kw={"hspace": 0.35},
    )
    axes = list(np.atleast_1d(raw_axes))

    # -- top panel: the score trajectory every run already carries ---------
    ax_score = axes[0]
    ax_score.plot(
        generations,
        [entry["best_score"] for entry in history],
        color=PALETTE[0],
        marker="o",
        label="best score",
    )
    ax_score.plot(
        generations,
        [entry["mean_score"] for entry in history],
        color=PALETTE[1],
        marker=".",
        label="population mean",
    )
    ax_score.set_title(title, fontsize=11, loc="left", pad=10)
    ax_score.legend(fontsize=8, frameon=False)

    # -- one panel per requested level: p(operator) per generation ---------
    for index, level in enumerate(levels, start=1):
        ax = axes[index]
        operators: list[str] = []
        for record in trajectory:
            for operator in record[level]:
                if operator not in operators:  # first-seen order, stable
                    operators.append(operator)
        for j, operator in enumerate(operators):
            ax.plot(
                generations,
                [record[level][operator] for record in trajectory],
                color=PALETTE[j % len(PALETTE)],
                marker="o",
                markersize=3,
                label=operator,
            )
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"p({level})", fontsize=9)
        ax.legend(fontsize=8, frameon=False, title="operator")
    axes[-1].set_xlabel("generation", fontsize=9)
    return fig


def plot_pareto_front(
    result: Any,
    x: str,
    y: str,
    title: str = "Pareto front",
) -> Any:
    """One scatter per MultiObjective run: dominated points muted, front lit.

    Plots every molecule's ``x`` / ``y`` objective values as muted dots and
    the 2-objective Pareto front over that pair
    (:func:`pharmaml.generative.pareto_front` internally) as accent markers
    connected in ``x`` order -- the classic trade-off curve.  Molecules with
    non-finite values on either plotted objective are dropped from the
    scatter (the extraction's own exclusion rule; NaN would also break axis
    autoscaling).  The title reports front size over examined molecules.

    Parameters
    ----------
    result:
        A :class:`~pharmaml.generative.GenerationResult` carrying a
        per-objective ``breakdown`` (duck-typed like the trajectory plot's
        ``result``).
    x, y:
        The two objective names to plot (must be distinct breakdown keys).
    title:
        Figure title prefix (left-anchored; the front size is appended).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``x`` / ``y`` are equal, unknown in the breakdown, or the
        breakdown carries no objective keys at all.
    """
    from pharmaml.generative.pareto import pareto_front

    if x == y:
        raise DataValidationError(
            f"x and y must name DIFFERENT objectives, both {x!r}"
        )
    breakdown = list(getattr(result, "breakdown", []) or [])
    keys: list[str] = []
    for entry in breakdown:
        for key in entry:
            if key not in keys:
                keys.append(key)
    unknown = [name for name in (x, y) if name not in keys]
    if unknown:
        raise DataValidationError(
            f"objective(s) {unknown!r} not in this run's breakdown; "
            f"available: {keys!r}"
        )

    # the 2-objective front over the plotted pair (a projection: a molecule
    # dominated in 3D can be non-dominated in the 2D view -- that is the
    # honest reading of a trade-off scatter)
    front = pareto_front(result, objectives=[x, y])
    front_keys = set(front.smiles)

    def _finite(value: object) -> bool:
        return (
            isinstance(value, (int, float)) and math.isfinite(float(value))
        )

    fig, raw_axes = new_figure(figsize=(7.2, 5.4))
    ax = next(iter(np.atleast_1d(raw_axes)))
    smiles = list(getattr(result, "smiles", []) or [])
    # dominated points only (front members get their own accent trace);
    # non-finite rows never entered the dominance scan, and NaN would break
    # axis autoscaling -- they stay out of the scatter
    dominated = [
        (float(entry[x]), float(entry[y]))
        for name, entry in zip(smiles, breakdown)
        if name not in front_keys and _finite(entry.get(x)) and _finite(entry.get(y))
    ]
    ax.scatter(
        [pair[0] for pair in dominated],
        [pair[1] for pair in dominated],
        s=26, color="#B8B8B8", alpha=0.75,
        label="dominated", zorder=2,
    )
    order = sorted(range(len(front.smiles)), key=lambda i: front.values[i][x])
    ax.plot(
        [front.values[i][x] for i in order],
        [front.values[i][y] for i in order],
        color=PALETTE[0], marker="o", markersize=6, linewidth=1.6,
        label=f"Pareto front ({len(front)})", zorder=3,
    )
    ax.set_xlabel(x, fontsize=10)
    ax.set_ylabel(y, fontsize=10)
    ax.set_title(
        f"{title} — {len(front)} of {front.n_total} molecules non-dominated",
        fontsize=11, loc="left", pad=10,
    )
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_rl_introspection(
    result: Any,
    title: str = "RL introspection — policy drift from the prior",
) -> Any:
    """One figure per RL run: the score story + the prior-drift diagnostics.

    Stacks four panels sharing the iteration x-axis:

    1. the score trajectory every run already carries (best / batch mean);
    2. the agent's vs the frozen prior's mean sequence log-probability —
       the two curves whose gap is the fine-tuning itself;
    3. the squared log-probability drift ``history[...]["drift"]`` — the
       exact quantity the prior anchor penalises (the anchor weight is
       annotated in the panel's y-label when the run reports it); the
       curve starts at exactly ``0.0`` because the agent begins as a
       bit-exact copy of the prior;
    4. the mean per-step sampling entropy — the exploration meter (a
       collapsing run shows entropy diving toward 0 while the drift
       climbs: mode collapse, the brake's raison d'être).

    Parameters
    ----------
    result:
        A :class:`~pharmaml.generative.GenerationResult` from
        :func:`pharmaml.generative.run_rl_optimization` (duck-typed like
        ``plot_operator_trajectory``'s ``result``).
    title:
        Figure title (left-anchored; retrieve with
        ``fig.axes[0].get_title(loc="left")``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``result`` carries no history, or its rows lack the
        prior-drift fields (a ``GeneticOptimizer`` history does not
        record them — the message says so).
    """
    history = getattr(result, "history", []) or []
    if not history:
        raise DataValidationError(
            "result carries no history -- pass the GenerationResult of a "
            "run_rl_optimization() run"
        )
    missing = [key for key in ("drift", "entropy") if key not in history[0]]
    if missing:
        raise DataValidationError(
            f"history rows carry no {missing!r} diagnostics — only "
            "run_rl_optimization() results record them"
        )

    iterations = [entry["iteration"] for entry in history]
    anchor_weight = (getattr(result, "metadata", None) or {}).get("anchor_weight")

    fig, raw_axes = new_figure(
        nrows=4,
        ncols=1,
        figsize=(8.4, 7.6),
        sharex=True,
        gridspec_kw={"hspace": 0.42},
    )
    axes = list(np.atleast_1d(raw_axes))

    # -- panel 1: the score story every run already carries -------------
    ax_score = axes[0]
    ax_score.plot(
        iterations,
        [entry["best_score"] for entry in history],
        color=PALETTE[0],
        marker="o",
        label="best score",
    )
    ax_score.plot(
        iterations,
        [entry["mean_score"] for entry in history],
        color=PALETTE[1],
        marker=".",
        label="batch mean",
    )
    ax_score.set_title(title, fontsize=11, loc="left", pad=10)
    ax_score.legend(fontsize=8, frameon=False)

    # -- panel 2: agent vs frozen prior, the two curves whose gap IS ---
    #    the fine-tuning
    ax_lp = axes[1]
    ax_lp.plot(
        iterations,
        [entry["agent_logprob"] for entry in history],
        color=PALETTE[0],
        marker="o",
        label="agent",
    )
    ax_lp.plot(
        iterations,
        [entry["prior_logprob"] for entry in history],
        color=PALETTE[3],
        marker=".",
        linestyle="--",
        label="prior (frozen)",
    )
    ax_lp.set_ylabel("mean sequence log-prob", fontsize=9)
    ax_lp.legend(fontsize=8, frameon=False)

    # -- panel 3: the drift the anchor penalises (starts at exact 0) ---
    ax_drift = axes[2]
    ax_drift.plot(
        iterations,
        [entry["drift"] for entry in history],
        color=PALETTE[4],
        marker="o",
    )
    ylabel = "drift (mean (ΔlogP)²)"
    if anchor_weight is not None:
        ylabel += f", anchor_weight={anchor_weight}"
    ax_drift.set_ylabel(ylabel, fontsize=9)

    # -- panel 4: the exploration meter ---------------------------------
    ax_entropy = axes[3]
    ax_entropy.plot(
        iterations,
        [entry["entropy"] for entry in history],
        color=PALETTE[5],
        marker="o",
    )
    ax_entropy.set_ylabel("mean step entropy", fontsize=9)
    ax_entropy.set_xlabel("iteration", fontsize=9)
    return fig
