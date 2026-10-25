"""Round 100x-124 smoke: the HTML session report, the GA top-molecule
grid and molecule_grid_figure's scores row -- headless, Xvfb-driven.

Run: DISPLAY=:99 python scratch/smoke124.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

import playground  # noqa: E402


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

    # -- 1. molecule_grid_figure with a scores row ------------------- #
    figure = playground.molecule_grid_figure(
        [("#1 · C1CCCC1", "C1CCCC1"), ("#2 · c1ccccc1", "c1ccccc1")],
        title="smoke grid",
        scores=["score 0.987", "score 0.500"],
    )
    labels = [axis.get_xlabel() for axis in figure.get_axes()]
    assert "score 0.987" in labels, labels
    assert "score 0.500" in labels, labels
    # the short-scores pad: entry 2 keeps no score row
    padded = playground.molecule_grid_figure(
        [("#1 · C1CCCC1", "C1CCCC1"), ("#2 · c1ccccc1", "c1ccccc1")],
        title="pad",
        scores=["score 0.987"],
    )
    pad_labels = [axis.get_xlabel() for axis in padded.get_axes()]
    assert "score 0.987" in pad_labels and "" in pad_labels, pad_labels
    import matplotlib.pyplot as plt

    plt.close(figure)
    plt.close(padded)
    print("1. molecule_grid_figure scores row OK")

    # -- 2. the GA draw-top path over a REAL evolution run ----------- #
    app._book.select(app._book.tabs()[7])  # Generative
    app.update()

    # a tiny GA run through the real card path
    app._objective_combo.set("QED drug-likeness")
    app._pop_size.set("8")
    app._generations.set("2")
    app._run_evolution()
    assert deadline_wait(app, "evolution"), "the evolution job hung"
    assert app._ga_result is not None
    app._gen_draw_top()
    assert deadline_wait(app, "draw top"), "the draw-top job hung"
    grid = getattr(app._ga_grid_host, "_pg_figure", None)
    assert grid is not None, "the top-molecule grid never landed"
    grid_labels = [axis.get_xlabel() for axis in grid.get_axes()]
    assert any(label.startswith("score ") for label in grid_labels), grid_labels
    titles = [axis.get_title() for axis in grid.get_axes()]
    assert any(title.startswith("#1") for title in titles), titles
    plt.close("all")
    print("2. GA draw-top grid OK (scores:", grid_labels[:3], ")")

    # -- 3. the HTML session report end-to-end ------------------------ #
    import tempfile
    from pathlib import Path

    monkey_path = Path(tempfile.mkdtemp()) / "session-report.html"

    def fake_dialog(*_a: object, **_k: object) -> str:
        return str(monkey_path)

    playground.filedialog.asksaveasfilename = fake_dialog  # type: ignore[assignment]
    app._menu_save_session_report_html()
    assert deadline_wait(app, "html report"), "the HTML report job hung"
    text = monkey_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert "PharmaML Playground session" in text
    assert "data:image" in text, "figures were not embedded"
    assert "<table" in text, "the tables never landed"
    assert "What this file contains" in text
    assert "cards on screen" in text, "the metric cards never landed"
    assert "self-contained HTML" in text
    messages = [str(a) for _n, a in seen]
    assert any("figure(s) + " in message for message in messages), messages
    print(f"3. HTML session report OK ({len(text)} chars, {monkey_path})")

    # -- 4. the guard path: nothing on screen ------------------------- #
    seen.clear()
    app._session_figures.clear()
    for widget in app._walk_widgets(app):
        if widget.winfo_class() == "Treeview":
            widget.delete(*widget.get_children())
    app._menu_save_session_report_html()
    assert any("Nothing to report yet" in str(a) for _n, a in seen), seen
    print("4. HTML report guard OK")

    app.destroy()
    print("SMOKE124_OK")


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
