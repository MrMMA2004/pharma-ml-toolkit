"""Screenshots + artifacts for the 100x-128 round: the per-card
"Save figure..." export closing app-wide -- real drawn figures on the
PK/PD tab (IV bolus profile, Emax effect curve, the NCA profile) with
their NEW buttons, plus the ML results card's button row; the exports
exercise the real save path through the cards' own buttons.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/shoot128.py
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


captured = {}
import pharmaml.viz as viz_module

real_save = viz_module.save_figure


def spy(figure, path, *args, **kwargs):
    captured["figure"] = figure
    return real_save(figure, path, *args, **kwargs)


def real_export(saver, target, expect) -> None:
    viz_module.save_figure = spy
    try:
        playground.filedialog.asksaveasfilename = lambda **_k: target
        saver()
        assert pump(), "the card's export hung"
        assert captured["figure"] is expect, "saved a figure that is not the card's"
    finally:
        viz_module.save_figure = real_save
    with open(target, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
    print("exported", target, "via the card's own button")


app._jump_to_tab(2)  # PK / PD

# 1) the IV bolus card: a real decay curve, then the card with its own
# Save figure... button and the live profile
app._iv_c0.delete(0, "end"); app._iv_c0.insert(0, "10")
app._iv_thalf.delete(0, "end"); app._iv_thalf.insert(0, "4")
app._iv_hours.delete(0, "end"); app._iv_hours.insert(0, "24")
app._plot_iv_bolus()
assert pump(), "iv profile hung"
iv_figure = getattr(app._iv_host, "_pg_figure", None)
assert iv_figure is not None, "the IV decay curve never landed"
scroll_to_widget(app._iv_host)
shot("pg128-iv-save-button")
real_export(app._save_iv_figure, "/home/z/my-project/download/pg128-iv-export.png", iv_figure)

# 2) the Emax pharmacodynamics card: the real effect curve + button
app._emax_math()
assert pump(), "emax hung"
pd_figure = getattr(app._pd_host, "_pg_figure", None)
assert pd_figure is not None, "the Emax curve never landed"
scroll_to_widget(app._pd_host)
shot("pg128-pd-save-button")

# 3) the NCA card: the real concentration-time profile + button, and
# the card's own export for real
app._nca_run()
assert pump(), "nca hung"
app._nca_plot()
assert pump(), "nca plot hung"
nca_figure = getattr(app._nca_host, "_pg_figure", None)
assert nca_figure is not None, "the NCA profile never landed"
scroll_to_widget(app._nca_host)
shot("pg128-nca-save-button")
real_export(app._save_nca_figure, "/home/z/my-project/download/pg128-nca-export.png", nca_figure)

# 4) the Data & ML tab: the ML results card's button row now carries
# the per-card export beside Calibration curve / Learning curve
app._jump_to_tab(5)
ml_buttons = [
    str(w.cget("text"))
    for w in app._walk_widgets(app._book.nametowidget(app._book.tabs()[5]))
    if w.winfo_class() == "TButton"
]
assert ml_buttons.count("Save figure…") == 3, ml_buttons.count("Save figure…")
scroll_to_widget(app._ml_host)
shot("pg128-ml-save-button")

app.destroy()
print("SHOOT128_OK")
