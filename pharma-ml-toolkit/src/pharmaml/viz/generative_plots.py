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
  run records in ``history``;
* :func:`plot_sa_distribution` — a generation run's raw Ertl SA-score
  histogram (part3 section 24.3 "visualizing SA score distributions"):
  exact :func:`pharmaml.viz.statplots.histogram_stats` bars, the
  conventional 1-4 drug-like band shaded, the direction and the
  limitations shipped with the returned payload.

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
['dominated', 'Pareto front (3)', 'interior — size = crowding', 'corner — +inf crowding']
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
from pathlib import Path
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.viz._compat import PALETTE, new_figure, save_figure
from pharmaml.viz.statplots import histogram_stats

__all__ = [
    "plot_operator_trajectory",
    "plot_pareto_front",
    "plot_rl_introspection",
    "plot_sa_distribution",
]

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

    The front markers encode the NSGA-II **crowding distance** (computed
    over the plotted pair -- the honest reading of a 2D view): interior
    members are drawn with marker size proportional to their isolation
    (a sparse-region member stands large, a crowded one small), and the
    boundary corners (``+inf`` -- best or worst on a plotted objective)
    are amber diamonds that always survive a :meth:`ParetoFront.prune`
    selection.  A fully degenerate pair (no spread on either objective)
    renders every member at the floor size -- crowding measures spread.

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
    # the trade-off curve itself -- connective tissue, no markers (the
    # members carry their own crowding-encoded traces below)
    ax.plot(
        [front.values[i][x] for i in order],
        [front.values[i][y] for i in order],
        color=PALETTE[0], linewidth=1.6,
        label=f"Pareto front ({len(front)})", zorder=3,
    )
    # the crowding encoding (100x-69): size = isolation over the plotted
    # pair; the +inf corners are amber diamonds a prune always keeps
    distances = front.crowding_distance()
    interiors = [i for i in order if math.isfinite(distances[i])]
    corners = [i for i in order if math.isinf(distances[i])]
    if interiors:
        # 34 + 140 x crowding: a degenerate pair (all zeros) falls to the
        # 34-point floor by itself -- no special case needed
        sizes = [34.0 + 140.0 * distances[i] for i in interiors]
        ax.scatter(
            [front.values[i][x] for i in interiors],
            [front.values[i][y] for i in interiors],
            s=sizes, color=PALETTE[0], alpha=0.9, zorder=4,
            label="interior — size = crowding",
        )
    if corners:
        ax.scatter(
            [front.values[i][x] for i in corners],
            [front.values[i][y] for i in corners],
            s=120, marker="D", color=PALETTE[1], edgecolor="white",
            linewidth=0.8, zorder=5,
            label="corner — +inf crowding",
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


def plot_mcts_introspection(
    result: Any,
    title: str = "MCTS introspection — the search spending its budget",
) -> Any:
    """One figure per MCTS run: score story, search growth, dedup economy.

    Stacks three panels sharing the cycle x-axis:

    1. the reward story -- best-so-far score (the monotone frontier of
       the dedup cache) vs the mean rollout reward of each cycle (the
       raw explorer's meter: cycles that dip sampled prefixes the policy
       liked but the objective did not);
    2. the search growth -- tree nodes vs unique molecules found (the
       widening gap between them is the search revisiting: expansion
       without discovery);
    3. the dedup economy -- cumulative cache hits (ChemTSv2's
       ``generated_dict`` at work: molecules scored once, no matter how
       often the tree walks back to them).

    Parameters
    ----------
    result:
        A :class:`~pharmaml.generative.GenerationResult` from
        :class:`~pharmaml.generative.MctsOptimizer` (duck-typed like the
        sibling plot functions).
    title:
        Figure title (left-anchored; retrieve with
        ``fig.axes[0].get_title(loc="left")``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    DataValidationError
        If ``result`` carries no history, or its rows lack the MCTS
        search fields (a GA / RL history does not record them -- the
        message says so).

    Example
    -------
    >>> from pharmaml.generative import GenerationResult
    >>> from pharmaml.viz import close_figure
    >>> from pharmaml.viz.generative_plots import plot_mcts_introspection
    >>> mcts = GenerationResult(
    ...     smiles=["CCO"], scores=[0.7],
    ...     history=[
    ...         {"cycle": 0, "best_score": 0.4, "mean_score": 0.2,
    ...          "n_unique": 3, "n_nodes": 12, "cache_hits": 0},
    ...         {"cycle": 1, "best_score": 0.6, "mean_score": 0.3,
    ...          "n_unique": 5, "n_nodes": 20, "cache_hits": 4},
    ...     ],
    ...     metadata={"method": "mcts", "tree_policy": "uct", "c_val": 1.0})
    >>> fig = plot_mcts_introspection(mcts)
    >>> (len(fig.axes), fig.axes[0].get_title(loc="left"))
    (3, 'MCTS introspection — the search spending its budget')
    >>> [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    ['tree nodes', 'unique molecules']
    >>> fig.axes[2].get_xlabel()
    'cycle'
    >>> close_figure(fig)
    >>> plot_mcts_introspection(GenerationResult(smiles=["CCO"], scores=[0.5]))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: result carries no history -- pass the GenerationResult of an MctsOptimizer run
    """
    history = getattr(result, "history", []) or []
    if not history:
        raise DataValidationError(
            "result carries no history -- pass the GenerationResult of an "
            "MctsOptimizer run"
        )
    missing = [
        key for key in ("cycle", "n_unique", "n_nodes", "cache_hits")
        if key not in history[0]
    ]
    if missing:
        raise DataValidationError(
            f"history rows carry no {missing!r} search fields — only "
            "MctsOptimizer results record them"
        )

    cycles = [entry["cycle"] for entry in history]
    metadata = getattr(result, "metadata", None) or {}
    tree_policy = metadata.get("tree_policy")
    c_val = metadata.get("c_val")

    fig, raw_axes = new_figure(
        nrows=3,
        ncols=1,
        figsize=(8.4, 6.6),
        sharex=True,
        gridspec_kw={"hspace": 0.42},
    )
    axes = list(np.atleast_1d(raw_axes))

    # -- panel 1: the reward story ---------------------------------------
    ax_score = axes[0]
    ax_score.plot(
        cycles,
        [entry["best_score"] for entry in history],
        color=PALETTE[0],
        marker="o",
        label="best so far",
    )
    ax_score.plot(
        cycles,
        [entry["mean_score"] for entry in history],
        color=PALETTE[1],
        marker=".",
        label="mean rollout reward",
    )
    ax_score.set_ylim(-0.02, 1.02)
    ax_score.set_title(title, fontsize=11, loc="left", pad=10)
    ax_score.legend(fontsize=8, frameon=False)

    # -- panel 2: the search growth (expansion vs discovery) -------------
    ax_growth = axes[1]
    ax_growth.plot(
        cycles,
        [entry["n_nodes"] for entry in history],
        color=PALETTE[3],
        marker="o",
        label="tree nodes",
    )
    ax_growth.plot(
        cycles,
        [entry["n_unique"] for entry in history],
        color=PALETTE[2],
        marker=".",
        label="unique molecules",
    )
    ylabel = "search size"
    if tree_policy is not None and c_val is not None:
        ylabel += f" ({tree_policy}, c={c_val})"
    ax_growth.set_ylabel(ylabel, fontsize=9)
    ax_growth.legend(fontsize=8, frameon=False)

    # -- panel 3: the dedup economy ---------------------------------------
    ax_cache = axes[2]
    ax_cache.plot(
        cycles,
        [entry["cache_hits"] for entry in history],
        color=PALETTE[4],
        marker="o",
    )
    ax_cache.set_ylabel("cumulative cache hits", fontsize=9)
    ax_cache.set_xlabel("cycle", fontsize=9)
    return fig


def plot_sa_distribution(
    result: Any,
    path: str | Path | None = None,
    *,
    bins: str | int = "auto",
    color: str = PALETTE[0],
) -> dict[str, Any]:
    """Raw Ertl SA-score histogram for one generation run (section 24.3).

    The bars are drawn from :func:`~pharmaml.viz.statplots.histogram_stats`
    (the same exact-edge contract as :func:`~pharmaml.viz.statplots.plot_histogram`)
    over the run's *valid* molecules only; the conventional drug-like
    band 1-4 is shaded so the eye reads the distribution against it.  The
    x-axis names the direction ("lower = estimated easier to synthesise")
    and the returned payload carries ``method`` / ``direction`` /
    ``limitations`` plus ``n_invalid`` -- a favourable SA score is a
    computational estimate, not evidence a molecule can actually be
    synthesised (the spec's section 24.5 documentation rule).

    Parameters
    ----------
    result:
        A :class:`~pharmaml.generative.GenerationResult` (duck-typed like
        the sibling plot functions; only ``smiles`` is read).
    path:
        Optional save target; the figure is always also returned.
    bins:
        Passed through to :func:`histogram_stats` (``"auto"`` / ``"fd"``
        / ``"sturges"`` / ``"sqrt"`` or a fixed bin count).

    Returns
    -------
    dict
        :func:`histogram_stats`' keys plus ``n_invalid``, ``method``,
        ``direction`` and ``limitations``.

    Raises
    ------
    DataValidationError
        When fewer than two valid molecules can be scored (nothing to
        bin), or the bin rule is invalid.
    DependencyError
        When RDKit (or its Contrib SA_Score database) is unavailable.

    Example
    -------
    >>> from pharmaml.generative import GenerationResult
    >>> from pharmaml.viz import close_figure
    >>> from pharmaml.viz.generative_plots import plot_sa_distribution
    >>> run = GenerationResult(
    ...     smiles=["CC(=O)Oc1ccccc1C(=O)O", "CCO", "CCN",
    ...             "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "not-a-smiles"],
    ...     scores=[0.8, 0.6, 0.5, 0.4, 0.0])
    >>> payload = plot_sa_distribution(run)
    >>> (payload["n"], payload["n_invalid"], round(payload["mean"], 3))
    (4, 1, 2.008)
    >>> fig = payload["figure"]
    >>> (len(fig.axes), fig.axes[0].get_xlabel())
    (1, 'SA score (lower = estimated easier to synthesise)')
    >>> [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    ['conventional drug-like band (1-4)', 'molecules']
    >>> fig.axes[0].get_title(loc="left")
    'SA score distribution — n=4, mean=2.01, invalid=1'
    >>> close_figure(fig)
    >>> plot_sa_distribution(GenerationResult(smiles=["CCO"], scores=[0.5]))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: fewer than two valid molecules to score -- a distribution needs at least 2 (got 1)
    """
    from pharmaml.chem.sa_scoring import (
        SA_SCORE_LIMITATIONS,
        SA_SCORE_METHOD,
        sa_scores,
    )

    smiles = list(getattr(result, "smiles", []) or [])
    batch = sa_scores(smiles)
    scores = batch.valid_scores
    if len(scores) < 2:
        raise DataValidationError(
            "fewer than two valid molecules to score -- a distribution "
            f"needs at least 2 (got {len(scores)})"
        )
    stats = histogram_stats(scores, bins=bins)

    fig, raw_ax = new_figure(figsize=(7.0, 4.5))
    ax = np.atleast_1d(raw_ax)[0]
    ax.bar(
        stats["centers"], stats["counts"].astype(float),
        width=0.94 * stats["bin_width"], color=color, edgecolor="white",
        linewidth=0.6, label="molecules",
    )
    ax.axvspan(1.0, 4.0, color=PALETTE[1], alpha=0.12, zorder=0,
               label="conventional drug-like band (1-4)")
    ax.set_xlabel("SA score (lower = estimated easier to synthesise)", fontsize=10)
    ax.set_ylabel("molecules", fontsize=10)
    ax.set_title(
        f"SA score distribution — n={stats['n']}, mean={stats['mean']:.3g}, "
        f"invalid={batch.n_invalid}",
        fontsize=11, loc="left", pad=10,
    )
    ax.legend(fontsize=8, frameon=False)
    if path is not None:  # figure stays open for the caller either way
        save_figure(fig, path)

    payload = dict(stats)
    payload.update(
        n_invalid=batch.n_invalid,
        method=SA_SCORE_METHOD,
        direction="lower = estimated easier to synthesise",
        limitations=SA_SCORE_LIMITATIONS,
        figure=fig,
    )
    return payload
