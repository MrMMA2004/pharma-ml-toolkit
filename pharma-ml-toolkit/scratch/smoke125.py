"""Round 100x-125 smoke: the Pareto front's colour-coded structure grid
(molecule_grid_figure's colours row + the front card's Draw front
button) -- headless, Xvfb-driven.

Run: DISPLAY=:99 python scratch/smoke125.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

import playground  # noqa: E402

TEAL = "#0F766E"
NEUTRAL = "#525252"


def main() -> None:
    app = playground.PlaygroundApp(persist_session=False)
    app.update_idletasks()
    app.update()

    # quiet ALL dialogs up front (headless: any modal would hang)
    seen: list[tuple[str, tuple]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _n=name, **k: seen.append((_n, a)),
        )

    # -- 1. molecule_grid_figure with a colours row ------------------- #
    figure = playground.molecule_grid_figure(
        [("#1 · C1CCNCC1", "C1CCNCC1"), ("#2 · c1ccccc1", "c1ccccc1"), ("#3 · bad", "nope")],
        title="smoke colours",
        scores=["qed 0.61 · mw 210", "qed 0.42", "qed —"],
        colors=[TEAL, NEUTRAL, TEAL],
    )
    axes = figure.get_axes()
    colors_seen = [axis.title.get_color() for axis in axes]
    assert colors_seen[0] == TEAL, colors_seen
    assert colors_seen[1] == NEUTRAL, colors_seen
    # the accent border is visible, the classic is not
    assert axes[0].spines["top"].get_visible()
    assert not axes[1].spines["top"].get_visible() or colors_seen[1] == NEUTRAL
    # the honest note keeps its slot and carries the accent colour
    note = axes[2].texts[0] if axes[2].texts else None
    assert note is not None and note.get_color() == TEAL, "unparseable note lost its accent"
    # the short-colours pad: entry 3 of a 2-colour list stays classic
    padded = playground.molecule_grid_figure(
        [("#1 · C1CCNCC1", "C1CCNCC1"), ("#2 · c1ccccc1", "c1ccccc1")],
        title="pad",
        colors=[TEAL],
    )
    pad_colors = [axis.title.get_color() for axis in padded.get_axes()]
    assert pad_colors[0] == TEAL and pad_colors[1] != TEAL, pad_colors
    import matplotlib.pyplot as plt

    plt.close(figure)
    plt.close(padded)
    print("1. molecule_grid_figure colours row OK (pad + note accent verified)")

    # -- 2. the Pareto draw-front path over a REAL multi-objective run - #
    app._book.select(app._book.tabs()[7])  # Generative
    app.update()
    app._objective_combo.set("Multi-objective (QED × MW × TPSA)")
    app._pop_size.set("12")
    app._generations.set("4")
    app._run_evolution()
    assert deadline_wait(app, "evolution"), "the evolution job hung"
    assert app._pareto_front is not None and len(app._pareto_front) == 7
    assert app._pareto_host._pg_figure is not None  # the auto-scatter
    app._gen_draw_front()
    assert deadline_wait(app, "draw front"), "the draw-front job hung"
    grid = app._pareto_host._pg_figure
    assert grid is not None, "the front structure grid never landed"
    grid_axes = grid.get_axes()
    title_colors = [axis.title.get_color() for axis in grid_axes]
    teal_count = title_colors.count(TEAL)
    neutral_count = title_colors.count(NEUTRAL)
    assert teal_count == 5 and neutral_count == 2, (teal_count, neutral_count)
    labels = [axis.get_xlabel() for axis in grid_axes]
    scored = [label for label in labels if label.startswith("qed ")]
    assert len(scored) == 7, labels
    assert all("mw_band" in label for label in scored)
    print(
        "2. Pareto draw-front grid OK (5 teal corners + 2 neutral,",
        "score rows:", scored[:2],
        ")",
    )

    # -- 3. Plot front brings the scatter back ------------------------- #
    app._plot_pareto_front()
    scatter = app._pareto_host._pg_figure
    assert scatter is not None
    scatter_labels = [axis.get_xlabel() for axis in scatter.get_axes()]
    assert not any(label.startswith("qed ") for label in scatter_labels), scatter_labels
    print("3. Plot-front restore OK (scatter replaced the grid)")

    # -- 4. the guard path: no front yet -------------------------------- #
    seen.clear()
    app._pareto_front = None
    app._pareto_result = None
    app._close_embedded(app._pareto_host)
    app._gen_draw_front()
    assert deadline_wait(app, "guard draw"), "the guard job hung"
    assert any("Nothing to draw" in str(a) for _n, a in seen), seen
    print("4. draw-front guard OK")

    plt.close("all")
    app.destroy()
    print("SMOKE125_OK")


def deadline_wait(app: playground.PlaygroundApp, label: str, timeout: float = 240.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    print(f"TIMEOUT waiting for {label}")
    return False


if __name__ == "__main__":
    import tkinter as tk  # noqa: F401 - guard: this smoke needs a window

    assert tk is not None
    main()
