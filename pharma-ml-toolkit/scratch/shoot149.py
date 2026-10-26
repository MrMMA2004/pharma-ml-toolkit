"""SHOOT149 -- screenshot the round-149 stacking card (100x-149).

Opens the playground headlessly (Xvfb :99, same command), runs the real
OOF stacking over the regression example (quantile target on, residual
correction on) and captures:
  * pg149-ml-board.png -- the shared Leaderboard card with the
    meta-leaderboard (the +residual row last) and the honest status
    note naming folds, counts and the transformed scale;
  * pg149-ml-stack.png -- the new "Stacking leaderboard (OOF +
    residual)" card itself: the base/meta/folds/test controls, the two
    toggles, and the notebook's step-18 parity scatter in the card's
    own figure host.

Run:  DISPLAY=:99 python scratch/shoot149.py  (Xvfb :99 already up)
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


def _scroll_to_widget(app: "playground.PlaygroundApp", widget: object, offset: int = 24) -> None:
    """Scroll a deep widget (a card / figure host) into the viewport."""
    page = app._book.nametowidget(app._book.select())
    app.update()
    body_y = widget.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - offset) / total)
    app.update()


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
    ok_board = ok_card = False
    facts: list[str] = []
    try:
        app.geometry("1280x900+0+0")
        app.update_idletasks()
        app.update()
        OUT.mkdir(exist_ok=True)

        # -- run the real OOF stacking on the regression example --------- #
        import pandas as pd

        _select_tab(app, "Data & ML")
        frame = pd.read_csv(ROOT / "examples" / "data" / "qsar_regression.csv")
        app._ml_adopt_frame(frame, ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")
        app._stack_base.delete(0, "end")
        app._stack_base.insert(0, "ridge, knn")
        app._stack_meta.delete(0, "end")
        app._stack_meta.insert(0, "ridge")
        app._stack_folds.delete(0, "end")
        app._stack_folds.insert(0, "2")
        app._stack_quantile.set(True)
        app._ml_stack_oof()
        assert _drain(app), "stacking job hung"
        status = str(app._board_status.cget("text"))
        rows = app._board_tree.get_children()

        # -- shot 1: the shared board with the meta-leaderboard ---------- #
        _scroll_to_widget(app, app._board_tree)
        _grab(app, "pg149-ml-board.png")
        ok_board = (
            len(rows) == 2
            and "2-fold OOF stacking" in status
            and "TRANSFORMED scale" in status
        )
        facts.append(f"board rows: {len(rows)}, status: {status!r}")

        # -- shot 2: the stacking card with its parity scatter ------------ #
        _scroll_to_widget(app, app._stack_host, offset=0)
        _grab(app, "pg149-ml-stack.png")
        figure = getattr(app._stack_host, "_pg_figure", None)
        title = figure.axes[0].get_title() if figure is not None else ""
        ok_card = figure is not None and title == "OOF stacking — ridge + residual"
        facts.append(f"scatter title: {title!r}")

    finally:
        app.destroy()

    print()
    for key, value in (("board", ok_board), ("card", ok_card)):
        print(f"  {'OK ' if value else 'FAIL'} {key}")
    for fact in facts:
        print(f"  · {fact}")
    if ok_board and ok_card:
        print("SHOOT149_OK")
        return 0
    print("SHOOT149_FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
