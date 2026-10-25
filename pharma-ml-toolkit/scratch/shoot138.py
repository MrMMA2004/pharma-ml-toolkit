"""Screenshots + artifacts for the 100x-138 round: the picklist
confession's PROVENANCE + the HTML twin's RETURN leg (the web strip's
kind census shoots in the browser, not here).
(1) App #1 seeds two figures + a filled table, moves a row, Saves --
the memory is planted -- then EXPORTS the standalone artifact
(pg138-shared-order.json rides along; its payload is asserted).
(2) App #2 starts FRESH (a new HOME -- no memory anywhere), the IMPORT
runs from the shared artifact, and the picklist re-opens PRE-ARRANGED
with the round's new confession naming the FILE it came from
(pg138-imported-confession.png).  (3) The Save completes -- the order
becomes the reader's OWN -- and the picklist re-opens with the retired
wording, "Your last saved order was applied" (pg138-own-order.png --
the retirement story, both states shot).  (4) The HTML report saved
from the own order carries the RETURN leg -- every page wrapper ends
in a "↑ Contents" link back to the header's nav (the
pg138-return-leg-report.html artifact rides along; the links are
asserted before the shoot ends).

Run: DISPLAY=:99 python scratch/shoot138.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground  # noqa: E402

# an isolated HOME so the shoot never touches a real session file
_HOME = tempfile.mkdtemp(prefix="shoot138-home-")
os.environ["HOME"] = _HOME

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None
playground.messagebox.showerror = lambda *a, **k: None
playground.messagebox.showwarning = lambda *a, **k: None

OUT = Path(__file__).resolve().parents[1] / "download"
SHARED_ORDER = OUT / "pg138-shared-order.json"
TARGET_HTML = OUT / "pg138-return-leg-report.html"

state = {}


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
    """The same three pages on both stages, same labels."""
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


# -- App #1: seed, move a row, Save (the memory is planted) ------------ #
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

    # the export: the standalone artifact lands
    playground.filedialog.asksaveasfilename = lambda **_k: str(SHARED_ORDER)
    app._export_report_order()
    assert SHARED_ORDER.is_file(), "the export never wrote the artifact"
    payload = json.loads(SHARED_ORDER.read_text(encoding="utf-8"))
    assert payload == {
        "format": playground.REPORT_ORDER_FORMAT,
        "kind": playground.REPORT_ORDER_KIND,
        "order": {
            "table": ["SMILES inspector"],
            "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
        },
    }, payload
    print("artifact pg138-shared-order.json written (the standalone order)")
finally:
    import contextlib

    with contextlib.suppress(Exception):
        if app.winfo_exists():
            app.destroy()

import gc  # noqa: E402

gc.collect()

# -- App #2: fresh HOME, the IMPORT + the confession's life -------------- #
_HOME2 = tempfile.mkdtemp(prefix="shoot138-import-")
os.environ["HOME"] = _HOME2
app = playground.PlaygroundApp(persist_session=True)
app.geometry("1280x860+10+10")
try:
    pump(app)
    assert app._report_page_order == {}  # a fresh launch remembers nothing
    assert app._report_order_origin is None
    playground.filedialog.askopenfilename = lambda **_k: str(SHARED_ORDER)
    app._import_report_order()
    assert app._report_page_order == {
        "table": ["SMILES inspector"],
        "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
    }, app._report_page_order
    # (100x-138) the adoption carries its PROVENANCE
    assert app._report_order_origin == str(SHARED_ORDER), app._report_order_origin
    print("the import adopted the colleague's order (with its provenance)")

    seed(app)

    # the picklist re-opens PRE-ARRANGED with the round's new confession
    def stage_three():
        dialog = find_picklist(app)
        confession = confession_text(app, dialog)
        assert confession == (
            "The imported order from pg138-shared-order.json was applied — "
            "Reset to walking order forgets it."
        ), confession
        shot(app, "pg138-imported-confession")

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
    # the completed Save retired the provenance -- the order is the reader's
    assert app._report_order_origin is None, "the Save did not retire the provenance"
    print("shot pg138-imported-confession.png (the confession names the file)")

    # the picklist re-opens with the RETIRED wording
    def stage_five():
        dialog = find_picklist(app)
        confession = confession_text(app, dialog)
        assert confession == (
            "Your last saved order was applied — Reset to walking order forgets it."
        ), confession
        shot(app, "pg138-own-order")

        def stage_six():
            cancel = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")) == "Cancel"
            )
            cancel.invoke()

        app.after(700, stage_six)

    app.after(500, stage_five)
    assert app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    ) is None
    print("shot pg138-own-order.png (the Save retired the import's wording)")

    # the HTML report saved from the own order carries the RETURN leg
    playground.filedialog.asksaveasfilename = lambda **_k: str(TARGET_HTML)

    def stage_seven():
        dialog = find_picklist(app)

        def stage_eight():
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(700, stage_eight)

    app.after(500, stage_seven)
    app._menu_save_session_report_html()
    assert pump(app), "the HTML report job hung"
    assert TARGET_HTML.is_file(), "the return-leg report never landed"
    html = TARGET_HTML.read_text(encoding="utf-8")
    assert "<nav class='contents' id='contents' tabindex='-1' aria-label='Contents'>" in html
    assert "<li><a href='#pg-1'>table 1 · SMILES inspector</a></li>" in html
    assert "<li><a href='#pg-2'>figure 1 · beta plot</a></li>" in html
    assert "<li><a href='#pg-3'>figure 2 · Ring topology moves — 6 mutant(s)</a></li>" in html
    # (100x-138) the RETURN leg: every page wrapper ends in the back link
    # (100x-139: the link carries its title tooltip -- three wrappers,
    # the same one-string discipline)
    assert (
        html.count(
            "<a class='back-contents' href='#contents' "
            "title='Back to the contents block at the top of the report'>"
            "\u2191 Contents</a>"
        )
        == 3
    )
    assert (
        "</table><a class='back-contents' href='#contents' "
        "title='Back to the contents block at the top of the report'>"
        "\u2191 Contents</a></section>" in html
    )
    assert (
        "<a class='back-contents' href='#contents' "
        "title='Back to the contents block at the top of the report'>"
        "\u2191 Contents</a></figure>" in html
    )
    assert "section.page:target, figure.page:target, nav.contents:target {" in html
    print("artifact pg138-return-leg-report.html written (the round trip)")
finally:
    import contextlib

    with contextlib.suppress(Exception):
        if app.winfo_exists():
            app.destroy()
gc.collect()

import matplotlib.pyplot as plt  # noqa: E402

plt.close("all")
print("SHOOT138_OK")
