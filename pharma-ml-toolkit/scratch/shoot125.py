"""Screenshots + artifacts for the 100x-125 round: the Pareto front's
colour-coded structure grid (boundary corners accented teal, interior
trade-offs muted) on the Generative tab.

Run: DISPLAY=:99 python scratch/shoot125.py
"""
import sys
import time

sys.path.insert(0, ".")
import playground

# keep any modal quiet in headless runs
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


def scroll_to_widget(widget) -> None:
    """Scroll a deep widget (e.g. a figure host) into the viewport."""
    page = app._book.nametowidget(app._book.select())
    app.update()
    body_y = widget.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - 24) / total)
    app.update()


# 1) the Pareto front card: a real multi-objective run, then the grid
app._jump_to_tab(7)
app._objective_combo.set("Multi-objective (QED × MW × TPSA)")
app._pop_size.set("12")
app._generations.set("4")
app._run_evolution()
assert pump(), "evolution hung"
assert app._pareto_front is not None and len(app._pareto_front) == 7
app._gen_draw_front()
assert pump(), "draw-front hung"
grid = getattr(app._pareto_host, "_pg_figure", None)
assert grid is not None, "the front grid never landed"
title_colors = [axis.title.get_color() for axis in grid.get_axes()]
assert title_colors.count("#0F766E") == 5, title_colors
assert title_colors.count("#525252") == 2, title_colors
labels = [axis.get_xlabel() for axis in grid.get_axes()]
assert sum(label.startswith("qed ") for label in labels) == 7, labels[:3]
# scroll the FRONT HOST itself into view (the card is taller than the
# viewport -- the card-top scroll leaves the host off-screen)
scroll_to_widget(app._pareto_host)
shot("pg125-front-grid")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

plt.close("all")
app.destroy()
print("SHOOT125_OK")
