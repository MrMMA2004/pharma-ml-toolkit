#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/gen/surgery`` route.

Runs inside the toolkit virtualenv (real RDKit + matplotlib + pharmaml)
and wires ONLY the public generative surgery API the web twin mirrors
(100x-130 — the same operators the desktop playground's ``Ring
topology moves`` and ``Breed two molecules`` cards drive, 100x-122):

    ring_topology_moves + scaffold_hop_crossover

The structure grids compose through the same captioned-panel recipe
the desktop's ``molecule_grid_figure`` helper uses (an RDKit 2D
drawing raster per panel, honest ``(unparseable SMILES)`` slots) and
render to a base64 PNG data URI through ``pharmaml.viz``.

Request (stdin, JSON) — exactly one of::

    {"mode": "topology", "smiles": "C1CCNCC1"}
    {"mode": "breed", "parentA": "...", "parentB": "...",
     "children": 4, "seed": 42}

Response (stdout, JSON, camelCase keys for the TypeScript route)::

    {"mode": "topology", "source": "C1CCNCC1", "mutants": ["C1CCCC1", ...],
     "summary": "3 distinct one-step ring-size mutant(s) of ... — ...",
     "gridPng": "data:image/png;base64,...", "gridPanels": 3,
     "elapsedSeconds": 0.4, "rdkitVersion": "..."}

    {"mode": "breed", "parentA": "...", "parentB": "...",
     "children": 4, "seed": 42, "litter": [child_or_null, ...],
     "viable": 3, "summary": "3/4 viable children — ...",
     "gridPng": "data:image/png;base64,...", "gridPanels": 5,
     "elapsedSeconds": 0.6, "rdkitVersion": "..."}

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

MAX_SMILES_CHARS = 300
MAX_CHILDREN = 8
MAX_SEED = 999_999
#: grid panel budget — the desktop helper renders every mutant, but a
#: many-ringed molecule can enumerate dozens; the web card keeps the
#: first 24 panels and says so in ``summary`` when it truncates.
MAX_GRID_PANELS = 24


def _fail(message: str) -> int:
    print(json.dumps({"error": message}))
    return EXIT_ERROR


def _caption(text: str, width: int = 34) -> str:
    """Shorten a caption line to ``width`` characters with an ellipsis."""
    return text if len(text) <= width else text[: width - 1] + "…"


def _grid_figure(entries: list[tuple[str, str]], title: str) -> Any:
    """Compose the captioned 2D structure grid (the desktop recipe).

    Mirrors ``playground.py``'s ``molecule_grid_figure``: 3 columns up
    to a 2-row litter (4 beyond), one RDKit ``MolToImage`` raster per
    parseable entry, an honest note per unparseable one, and the
    classic ``nothing to draw`` line for an empty list.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Draw

    count = max(len(entries), 1)
    columns = 3 if count <= 6 else 4
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


def _parseable(smiles: str) -> bool:
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles) is not None


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _fail(f"bridge request is not valid JSON: {exc}")

    if not isinstance(request, dict):
        return _fail("bridge request must be a JSON object")

    mode = request.get("mode")
    if mode not in ("topology", "breed"):
        return _fail('mode must be "topology" or "breed"')

    try:  # pragma: no cover - import guard for a broken environment
        from pharmaml.generative import ring_topology_moves, scaffold_hop_crossover
    except ImportError as exc:
        return _fail(f"pharmaml.generative unavailable: {exc}")

    import rdkit

    started = time.perf_counter()

    if mode == "topology":
        smiles = request.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            return _fail("smiles must be a non-empty string")
        smiles = smiles.strip()
        if len(smiles) > MAX_SMILES_CHARS:
            return _fail(f"smiles must be at most {MAX_SMILES_CHARS} characters")
        if not _parseable(smiles):
            return _fail("the molecule did not parse — check the SMILES string")
        try:
            mutants = ring_topology_moves(smiles)
        except ValueError as exc:
            return _fail(str(exc))
        except Exception as exc:  # noqa: BLE001 - surfaced to the web UI verbatim
            return _fail(f"ring topology enumeration failed: {exc}")

        shown = mutants[:MAX_GRID_PANELS]
        entries = [
            (f"mutant {index + 1}\n{_caption(value)}", value)
            for index, value in enumerate(shown)
        ]
        summary = (
            f"{len(mutants)} distinct one-step ring-size mutant(s) of "
            f"{_caption(smiles, 28)} — contraction and expansion "
            "candidates, sorted."
            if mutants
            else (
                f"{_caption(smiles, 28)} has no qualifying saturated ring — "
                "aromatic or fused/spiro rings are never re-sized, and "
                "acyclic molecules have no moves."
            )
        )
        if len(mutants) > MAX_GRID_PANELS:
            summary += f" (grid shows the first {MAX_GRID_PANELS}.)"
        grid_png, panels = _grid_png(
            entries,
            f"Ring topology moves — {len(mutants)} mutant(s) of {_caption(smiles, 28)}",
        )
        response = {
            "mode": "topology",
            "source": smiles,
            "mutants": mutants,
            "summary": summary,
            "gridPng": grid_png,
            "gridPanels": panels,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "rdkitVersion": rdkit.__version__,
        }
    else:
        parent_a = request.get("parentA")
        parent_b = request.get("parentB")
        children = request.get("children")
        seed = request.get("seed")
        for name, value in (("parentA", parent_a), ("parentB", parent_b)):
            if not isinstance(value, str) or not value.strip():
                return _fail(f"{name} must be a non-empty string")
            if len(value.strip()) > MAX_SMILES_CHARS:
                return _fail(f"{name} must be at most {MAX_SMILES_CHARS} characters")
        parent_a = parent_a.strip()
        parent_b = parent_b.strip()
        if isinstance(children, bool) or not isinstance(children, int) or not 1 <= children <= MAX_CHILDREN:
            return _fail(f"children must be an integer between 1 and {MAX_CHILDREN}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            return _fail(f"seed must be an integer between 0 and {MAX_SEED}")
        if not _parseable(parent_a):
            return _fail("parent A did not parse — check the SMILES string")
        if not _parseable(parent_b):
            return _fail("parent B did not parse — check the SMILES string")

        import random

        rng = random.Random(seed)
        litter: list[str | None] = []
        try:
            for _ in range(children):
                litter.append(scaffold_hop_crossover(parent_a, parent_b, rng))
        except ValueError as exc:
            return _fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            return _fail(f"scaffold-hop crossover failed: {exc}")

        viable = [child for child in litter if child]
        entries = [
            (f"parent A\n{_caption(parent_a)}", parent_a),
            (f"parent B\n{_caption(parent_b)}", parent_b),
        ]
        entries.extend(
            (f"child {index}\n{_caption(child)}", child)
            for index, child in enumerate(viable[: MAX_GRID_PANELS - 2], start=1)
        )
        summary = (
            f"{len(viable)}/{children} viable children — bridge / ring-merge / "
            f"linker-transplant cuts competed for each draw (seed {seed})."
        )
        if len(viable) > MAX_GRID_PANELS - 2:
            summary += f" (grid shows parents plus the first {MAX_GRID_PANELS - 2} children.)"
        grid_png, panels = _grid_png(
            entries,
            f"Scaffold-hop litter — {len(viable)}/{children} viable (seed {seed})",
        )
        response = {
            "mode": "breed",
            "parentA": parent_a,
            "parentB": parent_b,
            "children": children,
            "seed": seed,
            "litter": litter,
            "viable": len(viable),
            "summary": summary,
            "gridPng": grid_png,
            "gridPanels": panels,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "rdkitVersion": rdkit.__version__,
        }

    json.dump(response, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
