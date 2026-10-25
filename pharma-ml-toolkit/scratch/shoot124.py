"""Screenshots + artifacts for the 100x-124 round: the GA top-molecule
scored structure grid and the HTML session report (the portable twin).

Run: DISPLAY=:99 python scratch/shoot124.py
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground

# the report's success dialog is modal -- silence it for headless runs
playground.messagebox.showinfo = lambda *a, **k: None

app = playground.PlaygroundApp()
app.geometry("1280x860+10+10")


def pump(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def shot(name: str) -> None:
    app.update()
    app.update_idletasks()
    for _ in range(24):
        app.update()
        time.sleep(0.01)
    from PIL import ImageGrab  # type: ignore

    img = ImageGrab.grab(xdisplay=":99")
    img.save(f"{Path(__file__).resolve().parents[2] / 'download'} / {name}.png")
    print("shot", name)


def scroll_to_card(title: str) -> None:
    page = app._book.nametowidget(app._book.select())
    shell = None
    for card_title, _subtitle, candidate in playground._CARD_INDEX:
        if card_title != title:
            continue
        try:
            if candidate.winfo_exists():
                shell = candidate
                break
        except Exception:  # noqa: BLE001, S110, S112 - destroyed shell
            pass  # an earlier app's shell is gone; keep looking
    assert shell is not None, title
    app.update()
    body_y = shell.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - 16) / total)
    app.update()


def scroll_to_widget(widget) -> None:
    """Scroll a deep widget (e.g. a figure host) into the viewport."""
    page = app._book.nametowidget(app._book.select())
    app.update()
    body_y = widget.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - 24) / total)
    app.update()


# 1) the GA Results card: a real evolution, then the scored top grid
app._jump_to_tab(7)
app._objective_combo.set("QED drug-likeness")
app._pop_size.set("12")
app._generations.set("3")
app._run_evolution()
assert pump(), "evolution hung"
assert app._ga_result is not None
app._gen_draw_top()
assert pump(), "draw-top hung"
grid = getattr(app._ga_grid_host, "_pg_figure", None)
assert grid is not None, "the grid never landed"
labels = [axis.get_xlabel() for axis in grid.get_axes()]
assert any(label.startswith("score ") for label in labels), labels[:4]
# scroll the GRID HOST itself into view (the Results card is taller
# than the viewport -- the card-top scroll leaves the grid off-screen)
scroll_to_widget(app._ga_grid_host)
shot("pg124-top-grid")

# 2) the HTML session report: both axes into one self-contained file
import tkinter.filedialog as fd

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

app._analyse_smiles()
assert pump()
assert app._live_session_tables(), "the descriptor table did not fill"
assert app._live_session_figures(), "the grids are not live figures"
tmp = Path(tempfile.mkdtemp(prefix="shoot124_"))
target = tmp / "session-report.html"
fd.asksaveasfilename = lambda **_k: str(target)
app._menu_save_session_report_html()
assert pump(), "HTML report hung"
assert target.exists() and target.stat().st_size > 50_000
html = target.read_text(encoding="utf-8")
assert "data:image" in html and "<table" in html and "cards on screen" in html
print("html bytes:", target.stat().st_size)

# keep the artifact for inspection (and the VLM/browser review)
shutil.copy(target, str(Path(__file__).resolve().parents[2] / "download") + "/pg124-session-report.html")
plt.close("all")

app.destroy()
print("SHOOT124_OK")
