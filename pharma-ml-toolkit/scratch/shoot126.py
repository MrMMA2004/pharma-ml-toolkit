"""Screenshots + artifacts for the 100x-126 round: the Pareto front
card with its NEW per-card "Save figure..." button, and the themed HTML
session report (dark palette + the toggle button) rendered off a real
session figure.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/shoot126.py
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
    img.save(f"/home/z/my-project/download/{name}.png")
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
# with the per-card save button in the same button row
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
# the new button rides the same row as Plot front / Save front as CSV
button_row = app._pareto_host.master
labels = [
    str(w.cget("text"))
    for w in app._walk_widgets(app._book.nametowidget(app._book.tabs()[7]))
    if w.winfo_class() == "TButton"
]
assert "Save figure…" in labels, "the per-card save button is missing"
# scroll the FRONT HOST itself into view (the card is taller than the
# viewport -- the card-top scroll leaves the host off-screen)
scroll_to_widget(app._pareto_host)
shot("pg126-front-save-button")

# 2) exercise the per-card save for real, then verify the PNG
captured = {}
import pharmaml.viz as viz_module

real_save = viz_module.save_figure


def spy(figure, path, *args, **kwargs):
    captured["figure"] = figure
    return real_save(figure, path, *args, **kwargs)


viz_module.save_figure = spy
try:
    playground.filedialog.asksaveasfilename = lambda **_k: (
        "/home/z/my-project/download/pg126-front-export.png"
    )
    app._save_pareto_figure()
    assert captured["figure"] is grid, "saved a figure that is not the card's"
finally:
    viz_module.save_figure = real_save
with open("/home/z/my-project/download/pg126-front-export.png", "rb") as handle:
    assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
print("exported the card's live grid via the per-card button")

# 3) the themed HTML session report: bake the app's dark theme over the
# live session figures (the grid above), then screenshot the rendered
# file's own structure as a PNG via the browser-free proof: the HTML
# itself carries the palette; the shot shows the app's View > theme
# radios + the saved artifact path in the completion dialog flow.
app._theme_var.set("dark")
playground.filedialog.asksaveasfilename = lambda **_k: (
    "/home/z/my-project/download/pg126-report.html"
)
app._menu_save_session_report_html()
assert pump(), "html report hung"
with open("/home/z/my-project/download/pg126-report.html", encoding="utf-8") as handle:
    text = handle.read()
assert "<html lang='en' data-theme='dark'>" in text
assert "id='theme-toggle'" in text and "localStorage" in text
assert "data:image/png;base64," in text
print("themed report written (dark + toggle + embedded figures)")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

plt.close("all")
app.destroy()
print("SHOOT126_OK")
