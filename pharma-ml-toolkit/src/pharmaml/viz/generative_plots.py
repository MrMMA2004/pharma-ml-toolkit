"""Visual generative-design output — the optimiser's own learning curves.

The generative flagship's introspection artifacts, drawn as figures:

* :func:`plot_operator_trajectory` — the adaptive GA's operator mix per
  generation (the "GA that learns" learning curve), synced with the score
  trajectory any run already carries in ``history``.

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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_operator_trajectory"]

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
