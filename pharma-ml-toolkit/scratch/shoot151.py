"""SHOOT151 -- screenshot the round-151 prediction-intervals card
(100x-151).

Opens the playground headlessly (Xvfb :99, same command), trains ridge
on the regression example, runs the quantile trio and captures:
  * pg151-ml-intervals.png -- the Data & ML Results card with the new
    buttons ("Prediction intervals" + the alpha entry, "Save
    intervals…"), the band figure (plot_prediction_intervals) in the
    card's figure host, and the honest status note naming coverage,
    mean width and the crossed bound(s).

Run:  DISPLAY=:99 python scratch/shoot151.py  (Xvfb :99 already up)
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


def main() -> int:
    _quiet()
    app = playground.PlaygroundApp(persist_session=False)
    ok_card = False
    facts: list[str] = []
    try:
        app.geometry("1280x900+0+0")
        app.update_idletasks()
        app.update()
        OUT.mkdir(exist_ok=True)

        # -- train ridge on the regression example, then the trio -------- #
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
        app._ml_plot_intervals()
        assert _drain(app), "intervals job hung"
        note = str(app._ml_result_note.cget("text"))
        table = app._ml_intervals_frame
        figure = getattr(app._ml_host, "_pg_figure", None)
        title = figure.axes[0].get_title() if figure is not None else ""

        # -- the shot: the Results card with the new buttons ------------- #
        app._jump_to_card(5, "Results")
        app.update_idletasks()
        app.update()
        _grab(app, "pg151-ml-intervals.png")
        ok_card = (
            table is not None
            and len(table) == len(app._ml_last_run[6])
            and "coverage" in note
            and "mean width" in note
            and "crossed bound(s) repaired" in note
            and figure is not None
            and title == "Prediction intervals — ridge (pIC50)"
            and app._interval_alpha.get() == "0.1"
        )
        facts.append(f"table rows: {len(table) if table is not None else 0}")
        facts.append(f"note: {note!r}")
        facts.append(f"figure title: {title!r}")

    finally:
        app.destroy()

    print()
    for key, value in (("card", ok_card),):
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    for fact in facts:
        print(f"  · {fact}")
    if ok_card:
        print("SHOOT151_OK")
        return 0
    print("SHOOT151_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
