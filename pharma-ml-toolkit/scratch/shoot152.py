"""SHOOT152 — the round-100x-152 desktop smoke + screenshot.

The ML Results card with the round's interpretability bundle live:
train ridge on the regression example (the report now carries the
overfit gap), press the new "Correlations…" button (the third button
row), and shoot the card with the signed-correlations figure on the
host.  Asserts along the way: the metrics grid carries the
Pearson r / train R² / R² gap labels, the note strip updated, the
figure title is the signed-correlations chart.

Run:  DISPLAY=:99 python scratch/shoot152.py  (Xvfb :99 already up)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PLAYGROUND_NO_SESSION", "1")

import playground  # noqa: E402

OUT = ROOT / "download"


def _grab(app: "playground.PlaygroundApp", filename: str) -> Path:
    from PIL import ImageGrab

    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    shot = OUT / filename
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shot)
    print(f"wrote {shot}")
    return shot


def _select_tab(app: "playground.PlaygroundApp", name: str) -> None:
    for widget in app._walk_widgets(app):
        if widget.winfo_class() == "TNotebook":
            for tab_id in widget.tabs():
                if name in str(widget.tab(tab_id, "text")):
                    widget.select(tab_id)
                    return
    raise RuntimeError(f"tab not found: {name}")


def _drain(app: "playground.PlaygroundApp", timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def _quiet() -> None:
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _n=name, **k: print(f"  [dialog {_n}]: {a}"),
        )


def _metrics_labels(app: "playground.PlaygroundApp") -> list[str]:
    labels: list[str] = []
    for widget in app._ml_metrics.winfo_children():
        if widget.winfo_class() == "TLabel":
            labels.append(str(widget.cget("text")))
    return labels


def main() -> int:
    _quiet()
    app = playground.PlaygroundApp(persist_session=False)
    ok_card = False
    ok_gap = False
    facts: list[str] = []
    try:
        app.geometry("1280x900+0+0")
        app.update_idletasks()
        app.update()
        OUT.mkdir(exist_ok=True)

        # -- train ridge on the regression example ----------------------- #
        import pandas as pd

        _select_tab(app, "Data & ML")
        frame = pd.read_csv(ROOT / "examples" / "data" / "qsar_regression.csv")
        app._ml_adopt_frame(frame, ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert _drain(app), "train job hung"

        # the overfit-gap labels ride the metrics grid (GAP 7, live)
        labels = _metrics_labels(app)
        ok_gap = (
            "Pearson r" in labels
            and "train R²" in labels
            and "R² gap (train−test)" in labels
        )
        facts.append(f"gap labels present: {ok_gap}")
        facts.append(f"n metric labels: {len(labels)}")

        # -- the new correlations figure (GAP 2, live) ------------------- #
        app._ml_plot_correlations()
        assert _drain(app), "correlations job hung"
        note = str(app._ml_result_note.cget("text"))
        figure = getattr(app._ml_host, "_pg_figure", None)
        title = figure.axes[0].get_title() if figure is not None else ""

        # -- the shot: the Results card, correlations on the host -------- #
        app._jump_to_card(5, "Results")
        app.update_idletasks()
        app.update()
        _grab(app, "pg152-ml-correlations.png")
        ok_card = (
            figure is not None
            and title.startswith("Top signed correlations")
            and "held-out" in title
            and "signed correlations" in note
            and "ranked by |r|" in note
        )
        facts.append(f"note: {note!r}")
        facts.append(f"figure title: {title!r}")

    finally:
        app.destroy()

    print()
    for key, value in (("card", ok_card), ("gap labels", ok_gap)):
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    for fact in facts:
        print(f"  · {fact}")
    if ok_card and ok_gap:
        print("SHOOT152_OK")
        return 0
    print("SHOOT152_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
