#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/gen/mcts`` route.

Runs inside the toolkit virtualenv (real RDKit + torch + pharmaml) and
wires ONLY the public generative MCTS API the web twin mirrors — the
same ``MctsOptimizer`` surface the desktop Studio's Generative page
drives through its method picker ("monte carlo tree search (MCTS)"):

    MctsOptimizer(objective, corpus=..., iterations=..., seed=...)
    .optimize()

The objective registry (qed / logp / mw / tpsa / blend / multi /
similarity / sa, with the window lo/hi and reference-SMILES knobs) is
verbatim the Studio page's ``_build_objective``; the corpus semantics
mirror the Studio's too — empty = the built-in prior corpus, seeded
mode anchors the caller's lines ahead of the bundle, unseeded lines
are the sampling pool alone. The tree policy knob (uct / puct) is the
web card's honest extra over the Studio surface (the optimizer's own
public parameter, defaulted to "uct" exactly like the desktop).

Request (stdin, JSON)::

    {"objective": "qed", "iterations": 12, "treePolicy": "uct",
     "seed": 42, "corpus": "...", "seeded": false,
     "lo": 1.0, "hi": 3.0, "reference": "CC(=O)Oc1ccccc1C(=O)O"}

(``lo``/``hi`` feed the logp/mw/tpsa/sa windows — the SA anchors —
and are ignored elsewhere; ``reference`` feeds the similarity
objective only; ``corpus`` is optional newline-separated SMILES.)

Response (stdout, JSON, camelCase keys for the TypeScript route)::

    {"objective": "qed", "objectiveLabel": "QED drug-likeness",
     "iterations": 12, "treePolicy": "uct", "seed": 42,
     "corpusMode": "built-in", "best": "NC(...)", "bestScore": 0.49,
     "leaderboard": [{"rank": 1, "smiles": "...", "score": 0.49,
                       "qed": 0.66, ...}, ...],
     "breakdownColumns": ["qed", ...],        # [] when single-objective
     "history": {"cycle": [...], "bestScore": [...], "meanScore": [...],
                 "nUnique": [...]},
     "gridPng": "data:image/png;base64,...", "gridPanels": 9,
     "treeStats": {"nNodes": ..., "nRollouts": ..., "nValid": ...,
                   "nUnique": ..., "corpusSize": ..., "policy": "...",
                   "cVal": ..., "expansionThreshold": ...,
                   "simulationNum": ..., "maxLen": ...},
     "summary": "...", "elapsedSeconds": 6.1, "rdkitVersion": "..."}

Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit
code 2 so the route can surface them as HTTP 400/500 cleanly.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

EXIT_OK = 0
EXIT_ERROR = 2

#: the web card's latency budget — the desktop spinner runs 4..200, the
#: web keeps the search bounded (iterations=24 with the default
#: 8-epoch RNN prior measures ~6.5 s end to end on this stack).
MIN_ITERATIONS = 4
MAX_ITERATIONS = 24
MAX_SEED = 999_999
MAX_SMILES_CHARS = 300
#: a pasted corpus is capped the same way the Studio's lines editor is
#: (the page reads a whole textarea; the bridge keeps the first 200
#: lines and says so in the response when it truncates).
MAX_CORPUS_LINES = 200
MAX_CORPUS_LINE_CHARS = 80
#: grid panel budget — 3x3 panels of the best molecules; the
#: leaderboard cap below is 12 rows with an honest truncation note.
MAX_GRID_PANELS = 9
MAX_LEADERBOARD_ROWS = 12

_OBJECTIVES: dict[str, str] = {
    "qed": "QED drug-likeness",
    "logp": "LogP window",
    "mw": "Molecular weight window",
    "tpsa": "TPSA window",
    "blend": "QED + LogP blend",
    "multi": "Multi-objective (QED × MW × TPSA)",
    "similarity": "Similarity to reference SMILES",
    "sa": "SA score (ease of synthesis)",
}
_WINDOW_OBJECTIVES = frozenset({"logp", "mw", "tpsa", "sa"})


def _fail(message: str) -> int:
    print(json.dumps({"error": message}))
    return EXIT_ERROR


def _parseable(smiles: str) -> bool:
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles) is not None


def _build_objective(key: str, lo: float, hi: float, reference: str) -> Any:
    """Instantiate the objective the Studio's _build_objective builds."""
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
        return MultiObjective(
            [QedObjective(), LogPBandObjective(lo, hi)], weights=[0.7, 0.3]
        )
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
        # the window entries are the easy/hard reward anchors
        # (1 <= easy < hard <= 10; sensible: 1.0/3.0 or 2.0/7.0).
        return SAScoreObjective(easy=lo, hard=hi)
    raise ValueError(f"unknown objective key: {key}")


def _grid_figure(entries: list[tuple[str, str]], title: str) -> Any:
    """Compose the captioned 2D structure grid (the desktop recipe).

    Mirrors the surgery bridge's ``molecule_grid_figure``: 3 columns,
    one RDKit ``MolToImage`` raster per parseable entry, an honest note
    per unparseable one, and the classic ``nothing to draw`` line for
    an empty list.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Draw

    count = max(len(entries), 1)
    columns = 3
    rows = max(1, -(-count // columns))  # ceil division
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.1 * columns, 3.0 * rows + 0.5), squeeze=False
    )
    panels = [panel for row_axes in axes for panel in row_axes]
    for panel in panels:
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.set_xticks([])
        panel.set_yticks([])
    if not entries:
        panels[0].text(0.5, 0.5, "nothing to draw", ha="center", va="center", fontsize=11)
    for panel, (caption, smiles) in zip(panels, entries):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            panel.text(
                0.5,
                0.5,
                f"{caption}\n(unparseable SMILES)",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
            continue
        raster = np.asarray(Draw.MolToImage(mol, size=(300, 300)))
        panel.imshow(raster)
        panel.set_title(caption, fontsize=8, pad=6)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _grid_png(entries: list[tuple[str, str]], title: str) -> tuple[str, int]:
    """Render the grid to a ``data:`` URI (and close it defensively)."""
    from pharmaml.viz import figure_to_base64

    figure = _grid_figure(entries, title)
    try:
        return figure_to_base64(figure, "png"), len(entries)
    finally:
        from pharmaml.viz import close_figure

        close_figure(figure)


def _caption(text: str, width: int = 34) -> str:
    """Shorten a caption line to ``width`` characters with an ellipsis."""
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _fail(f"bridge request is not valid JSON: {exc}")

    if not isinstance(request, dict):
        return _fail("bridge request must be a JSON object")

    objective_key = request.get("objective")
    if objective_key not in _OBJECTIVES:
        return _fail(
            "objective must be one of " + ", ".join(sorted(_OBJECTIVES))
        )
    iterations = request.get("iterations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not MIN_ITERATIONS <= iterations <= MAX_ITERATIONS
    ):
        return _fail(
            f"iterations must be an integer between {MIN_ITERATIONS} "
            f"and {MAX_ITERATIONS}"
        )
    tree_policy = request.get("treePolicy", "uct")
    if tree_policy not in ("uct", "puct"):
        return _fail('treePolicy must be "uct" or "puct"')
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        return _fail(f"seed must be an integer between 0 and {MAX_SEED}")
    seeded = request.get("seeded", False)
    if not isinstance(seeded, bool):
        return _fail("seeded must be a boolean")
    corpus_text = request.get("corpus")
    if corpus_text is None:
        corpus_text = ""
    if not isinstance(corpus_text, str):
        return _fail("corpus must be a string of newline-separated SMILES")
    lines_all = [line.strip() for line in corpus_text.splitlines()]
    lines = [line for line in lines_all if line]
    truncated_corpus = False
    if len(lines) > MAX_CORPUS_LINES:
        truncated_corpus = True
        lines = lines[:MAX_CORPUS_LINES]
    for line in lines:
        if len(line) > MAX_CORPUS_LINE_CHARS:
            return _fail(
                f"a corpus line exceeds {MAX_CORPUS_LINE_CHARS} characters "
                "(the optimizer's max_len)"
            )

    lo_raw = request.get("lo", 1.0)
    hi_raw = request.get("hi", 3.0)
    try:
        lo = float(lo_raw)
        hi = float(hi_raw)
    except (TypeError, ValueError):
        return _fail("lo and hi must be numbers")
    if objective_key in _WINDOW_OBJECTIVES and not lo < hi:
        return _fail(
            f"the {objective_key} window needs lo < hi (got {lo} .. {hi})"
        )

    reference = request.get("reference", "")
    if not isinstance(reference, str):
        return _fail("reference must be a string")
    reference = reference.strip()
    if objective_key == "similarity":
        if not reference:
            return _fail("the similarity objective needs a reference SMILES")
        if len(reference) > MAX_SMILES_CHARS:
            return _fail(
                f"reference must be at most {MAX_SMILES_CHARS} characters"
            )
        if not _parseable(reference):
            return _fail(
                "the reference SMILES did not parse — check the string"
            )

    try:  # pragma: no cover - import guard for a broken environment
        from pharmaml.exceptions import DataValidationError
        from pharmaml.generative import DEFAULT_CORPUS, MctsOptimizer
    except ImportError as exc:
        return _fail(f"pharmaml.generative unavailable: {exc}")

    import rdkit

    started = time.perf_counter()

    # the MCTS corpus semantics mirror the desktop page's (and the RL
    # prior's): empty = the built-in prior corpus; seeded mode anchors
    # the lines ahead of the bundle; unseeded lines are the pool alone.
    if not lines:
        corpus_arg: Any = None
        corpus_mode = "built-in"
    elif seeded:
        corpus_arg = list(dict.fromkeys([*lines, *DEFAULT_CORPUS]))
        corpus_mode = "seeded"
    else:
        corpus_arg = lines
        corpus_mode = "custom"

    try:
        objective = _build_objective(objective_key, lo, hi, reference)
        optimizer = MctsOptimizer(
            objective,
            corpus=corpus_arg,
            iterations=iterations,
            tree_policy=tree_policy,
            seed=seed,
        )
        result = optimizer.optimize()
    except DataValidationError as exc:
        return _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced to the web UI verbatim
        return _fail(f"MCTS optimization failed: {exc}")

    frame = result.to_dataframe()
    breakdown_columns = [
        column
        for column in frame.columns
        if column not in ("rank", "smiles", "score")
    ]
    shown = frame.head(MAX_LEADERBOARD_ROWS)
    leaderboard: list[dict[str, Any]] = []
    for record in shown.to_dict("records"):
        leaderboard.append(
            {
                "rank": int(record["rank"]),
                "smiles": str(record["smiles"]),
                "score": round(float(record["score"]), 6),
                "breakdown": {
                    column: round(float(record[column]), 6)
                    for column in breakdown_columns
                },
            }
        )

    history: dict[str, list[Any]] = {
        "cycle": [],
        "bestScore": [],
        "meanScore": [],
        "nUnique": [],
    }
    for entry in result.history:
        history["cycle"].append(int(entry.get("cycle", len(history["cycle"]))))
        history["bestScore"].append(round(float(entry.get("best_score", 0.0)), 6))
        history["meanScore"].append(round(float(entry.get("mean_score", 0.0)), 6))
        history["nUnique"].append(int(entry.get("n_unique", 0)))

    grid_entries = [
        (
            (
                f"rank {int(record['rank'])} · {round(float(record['score']), 3)}\n"
                f"{_caption(str(record['smiles']), 30)}"
            ),
            str(record["smiles"]),
        )
        for _, record in frame.head(MAX_GRID_PANELS).iterrows()
    ]
    grid_png, grid_panels = _grid_png(
        grid_entries,
        f"MCTS leaderboard — top {len(grid_entries)} of {len(frame)} "
        f"({_OBJECTIVES[objective_key]}, seed {seed})",
    )

    metadata = dict(result.metadata)
    tree_stats = {
        "nNodes": int(metadata.get("n_nodes", 0)),
        "nRollouts": int(metadata.get("n_rollouts", 0)),
        "nValid": int(metadata.get("n_valid", 0)),
        "nUnique": int(metadata.get("n_unique", 0)),
        "nCacheHits": int(metadata.get("n_cache_hits", 0)),
        "maxDepth": int(metadata.get("max_depth", 0)),
        "corpusSize": int(metadata.get("corpus_size", 0)),
        "policy": str(metadata.get("policy", "")),
        "cVal": float(metadata.get("c_val", 1.0)),
        "expansionThreshold": float(metadata.get("expansion_threshold", 0.995)),
        "simulationNum": int(metadata.get("simulation_num", 3)),
        "maxLen": int(metadata.get("max_len", 80)),
    }

    n_total = len(frame)
    summary = (
        f"MCTS ({tree_policy.upper()}) over the {_OBJECTIVES[objective_key].lower()} "
        f"objective — {iterations} cycle(s) grew a {tree_stats['nNodes']}-node tree "
        f"({tree_stats['nRollouts']} rollout(s)); {n_total} unique valid molecule(s), "
        f"best score {round(result.best_score, 3)}."
    )
    if truncated_corpus:
        summary += f" (corpus truncated to the first {MAX_CORPUS_LINES} lines.)"
    if n_total > MAX_LEADERBOARD_ROWS:
        summary += (
            f" Leaderboard shows the top {MAX_LEADERBOARD_ROWS} of {n_total}."
        )

    response = {
        "objective": objective_key,
        "objectiveLabel": _OBJECTIVES[objective_key],
        "iterations": iterations,
        "treePolicy": tree_policy,
        "seed": seed,
        "corpusMode": corpus_mode,
        "best": result.best,
        "bestScore": round(float(result.best_score), 6),
        "leaderboard": leaderboard,
        "breakdownColumns": breakdown_columns,
        "history": history,
        "gridPng": grid_png,
        "gridPanels": grid_panels,
        "treeStats": tree_stats,
        "summary": summary,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "rdkitVersion": rdkit.__version__,
    }
    print(json.dumps(response))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
