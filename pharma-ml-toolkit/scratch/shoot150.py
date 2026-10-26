"""SHOOT150 -- screenshot the round-150 board Save-table twin (100x-150).

Opens the playground headlessly (Xvfb :99, same command), runs the real
OOF stacking over the regression example (the fastest honest board
filler), then:
  * drives the new "Save table…" button through a monkeypatched save
    dialog -- the written leaderboard.csv is read back and its header
    row asserted (the end-to-end proof the button writes what the
    board shows);
  * pg150-ml-board-save.png -- the shared Leaderboard card with the
    meta-leaderboard rows AND the button row (Plot leaderboard / Save
    figure… / Save table…) in one shot.

Run:  DISPLAY=:99 python scratch/shoot150.py  (Xvfb :99 already up)
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


def main() -> int:
    _quiet_saved: list[str] = []
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _n=name, **k: print(f"  [dialog {_n}]: {a}"),
        )
    app = playground.PlaygroundApp(persist_session=False)
    ok_rows = ok_write = False
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
        app._ml_stack_oof()
        assert _drain(app), "stacking job hung"
        rows = app._board_tree.get_children()
        ok_rows = len(rows) == 2  # ridge + ridge+residual

        # -- drive the new Save table… button end to end ------------------ #
        target = OUT / "pg150_board_written.csv"
        playground.filedialog.asksaveasfilename = (  # type: ignore[method-assign]
            lambda t="", _p=target, **k: str(_p)
        )
        app._ml_board_save()
        app.update()
        if target.exists():
            content = target.read_text(encoding="utf-8")
            header = content.splitlines()[0]
            ok_write = header == "model,oof_r2,test_r2"
            _quiet_saved.append(header)
            print(f"  saved board -> {target.name}: header {header!r}")
            target.unlink()  # the proof ran; the CSV is not an artifact

        # -- the shot: board rows + the button row in one frame ----------- #
        _scroll_to_widget(app, app._board_tree, offset=170)
        _grab(app, "pg150-ml-board-save.png")
    finally:
        app.destroy()
    print("SHOOT150_OK" if (ok_rows and ok_write) else "SHOOT150_FAIL")
    return 0 if (ok_rows and ok_write) else 1


if __name__ == "__main__":
    sys.exit(main())
