"""Visualization of multi-objective hyperparameter studies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.hpo import TrialState
from pharmaml.ml.multiobjective import MultiObjectiveStudy, pareto_indices
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_study_pareto"]


def plot_study_pareto(
    study: MultiObjectiveStudy,
    x: str,
    y: str,
    *,
    objective_names: Sequence[str] | None = None,
    title: str = "Multi-objective study",
) -> Any:
    """Plot completed trials and the full-dimensional Pareto front.

    All study objectives are minimized. ``x`` and ``y`` select two distinct
    names supplied in ``objective_names``, or the default ``objective_1``,
    ``objective_2``, etc. Dominance is computed using *all* objectives, not
    just the displayed pair. For three or more objectives, a full-front
    member may appear dominated in the two-dimensional projection. Points
    are not connected because such a line can misrepresent the front.

    Only completed trials are shown. The function rejects empty studies and
    invalid or ambiguous names. The caller owns the returned Matplotlib
    figure and may save it with ``pharmaml.viz.save_figure``.
    """
    if not isinstance(study, MultiObjectiveStudy):
        raise DataValidationError("study must be a MultiObjectiveStudy")
    names = tuple(objective_names) if objective_names is not None else tuple(
        f"objective_{index + 1}" for index in range(study.n_objectives)
    )
    if len(names) != study.n_objectives or any(not isinstance(n, str) or not n for n in names):
        raise DataValidationError("objective_names must supply one non-empty name per objective")
    if len(set(names)) != len(names):
        raise DataValidationError("objective_names must be unique")
    if x == y or x not in names or y not in names:
        raise DataValidationError(f"x and y must be distinct objective names from {list(names)!r}")
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        raise DataValidationError("study has no completed trials")
    try:
        matrix = np.asarray([trial.values for trial in completed], dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("completed trial values must form a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise DataValidationError("completed trial values must match the study's objective count")
    front_indices = pareto_indices(matrix)
    front_set = set(front_indices)
    dominated = [index for index in range(len(completed)) if index not in front_set]
    ix, iy = names.index(x), names.index(y)

    fig, ax = new_figure(figsize=(7.2, 5.4))
    if dominated:
        ax.scatter(matrix[dominated, ix], matrix[dominated, iy], s=32,
                   color="#777777", alpha=0.65, label="Dominated", zorder=2)
    ax.scatter(matrix[front_indices, ix], matrix[front_indices, iy], s=75,
               color=PALETTE[0], marker="D", edgecolor="white", linewidth=0.7,
               label=f"Full-objective Pareto front ({len(front_indices)})", zorder=3)
    ax.set_xlabel(f"{x} (minimize)")
    ax.set_ylabel(f"{y} (minimize)")
    ax.set_title(f"{title} — {len(front_indices)} of {len(completed)} completed trials",
                 loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
