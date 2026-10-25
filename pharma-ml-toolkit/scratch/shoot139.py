"""Screenshots + artifacts for the 100x-139 round: the return link's
POINTER confession + the web confession's DESKTOP ANCHOR (the web
strip's spotlight/confession shoots in the browser, not here).
(1) App #1 seeds three pages, moves a row, Saves -- the memory is
planted -- and the picklist re-opens with the confession the WEB strip
mirrors ("Your last saved order was applied — Reset to walking order
forgets it." -- the anchor the round's web twin paraphrases)
(pg139-anchor-confession.png).  (2) The HTML report saved from that
order carries the round's tooltip: EVERY page wrapper ends in the
SAME title-carrying "↑ Contents" link, one string minted once
(pg139-tooltip-report.html rides along; the links are asserted before
the shoot ends).

Run: DISPLAY=:99 python scratch/shoot139.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import gc

sys.path.insert(0, ".")
import playground  # noqa: E402

# an isolated HOME so the shoot never touches a real session file
_HOME = tempfile.mkdtemp(prefix="shoot139-home-")
os.environ["HOME"] = _HOME

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None
playground.messagebox.showerror = lambda *a, **k: None
playground.messagebox.showwarning = lambda *a, **k: None

OUT = Path(__file__).resolve().parents[1] / "download"
TARGET_HTML = OUT / "pg139-tooltip-report.html"

BACK_LINK = (
    "<a class='back-contents' href='#contents' "
    "title='Back to the contents block at the top of the report'>"
    "\u2191 Contents</a>"
)


def pump(app, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def run_for(app, seconds: float) -> None:
    """Pump the MAIN loop for a window so scheduled stages can fire."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.update()
        time.sleep(0.01)


def shot(app, name: str) -> None:
    from PIL import ImageGrab  # type: ignore

    for _ in range(12):
        app.update()
        time.sleep(0.01)
    img = ImageGrab.grab(xdisplay=":99")
    img.save(f"{OUT}/{name}.png")
    print("shot", name)


def widgets_of(app, dialog, cls):
    return [w for w in app._walk_widgets(dialog) if w.winfo_class() == cls]


def find_picklist(app):
    return next(
        child
        for child in app.winfo_children()
        if child.winfo_class() == "Toplevel"
        and str(child.title()) == "Session report contents"
    )


def confession_text(app, dialog):
    return next(
        str(widget.cget("text"))
        for widget in app._walk_widgets(dialog)
        if widget.winfo_class() == "TLabel" and "was applied" in str(widget.cget("text"))
    )


def seed(app):
    """The same three pages the standing shoots plant."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def figure_static(title):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    app._render_figure(
        app._topo_host,
        figure_static("the grid"),
        caption="Ring topology moves — 6 mutant(s)",
    )
    app._render_figure(
        app._breed_host, figure_static("beta plot"), caption="beta plot"
    )
    app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
    pump(app, timeout=30.0)


# -- App #1: seed, move a row, Save (the memory is planted) ------------- #
app = playground.PlaygroundApp(persist_session=True)
app.geometry("1280x860+10+10")
try:
    pump(app)
    seed(app)
    assert app._report_page_order == {}

    def stage_one():
        dialog = find_picklist(app)
        downs = [
            w
            for w in widgets_of(app, dialog, "TButton")
            if str(w.cget("text")) == "▼"
        ]
        downs[1].invoke()  # the mutants below beta (the remembered move)
        app.update()

        def stage_two():
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(600, stage_two)

    app.after(500, stage_one)
    app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    )
    assert app._report_page_order == {
        "table": ["SMILES inspector"],
        "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
    }, app._report_page_order
    print("the memory is planted (a completed Save with a moved row)")

    # the picklist re-opens with the ANCHOR wording -- the confession the
    # web strip's new order confession mirrors (probe-pinned there)
    def stage_three():
        dialog = find_picklist(app)
        confession = confession_text(app, dialog)
        assert confession == (
            "Your last saved order was applied — Reset to walking order forgets it."
        ), confession
        shot(app, "pg139-anchor-confession")

        def stage_four():
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(700, stage_four)

    app.after(500, stage_three)
    app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    )
    print("shot pg139-anchor-confession.png (the web twin's anchor wording)")

    # the HTML report saved from the remembered order carries the TOOLTIP
    playground.filedialog.asksaveasfilename = lambda **_k: str(TARGET_HTML)

    def stage_five():
        dialog = find_picklist(app)

        def stage_six():
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(700, stage_six())

    app.after(500, stage_five)
    app._menu_save_session_report_html()
    assert pump(app), "the HTML report job hung"
    assert TARGET_HTML.is_file(), "the tooltip report never landed"
    html = TARGET_HTML.read_text(encoding="utf-8")
    # three pages -> exactly three tooltip-carrying return links, one
    # string, no drift (the anchors themselves unchanged)
    assert html.count(BACK_LINK) == 3, html.count(BACK_LINK)
    assert (
        html.count("title='Back to the contents block at the top of the report'") == 3
    )
    assert "<nav class='contents' id='contents' tabindex='-1'" in html
    assert "<li><a href='#pg-1'>table 1 · SMILES inspector</a></li>" in html
    assert "<li><a href='#pg-2'>figure 1 · beta plot</a></li>" in html
    assert "<li><a href='#pg-3'>figure 2 · Ring topology moves — 6 mutant(s)</a></li>" in html
    print("artifact pg139-tooltip-report.html written (the pointer confession)")
finally:
    import contextlib

    with contextlib.suppress(Exception):
        if app.winfo_exists():
            app.destroy()
gc.collect()

import matplotlib.pyplot as plt  # noqa: E402

plt.close("all")
print("SHOOT139_OK")
