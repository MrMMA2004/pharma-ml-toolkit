"""Screenshots + artifacts for the 100x-137 round: the order memory's
SHARE pair (File ▸ Export / Import remembered order) + the HTML twin's
clickable Contents anchors.
(1) App #1 seeds two figures + a filled table, moves a row, Saves --
the memory is planted.  (2) The FILE MENU is posted and shot with the
two new items sitting right after "Clear remembered report order…"
(pg137-file-menu.png).  (3) The EXPORT runs: the standalone artifact
lands (pg137-shared-order.json rides along; its payload is asserted --
marker + kind tag + the two label lists, none of the session's
inputs).  (4) App #2 starts FRESH (a new HOME -- no memory anywhere),
the IMPORT runs from the shared artifact, and the picklist re-opens
PRE-ARRANGED in the imported order with its confession showing
(pg137-imported-order.png).  (5) The HTML report saved from the
imported order carries the CLICKABLE anchors -- every Contents line
links to the id-carrying page wrapper (the pg137-anchors-report.html
artifact rides along; the anchors are asserted before the shoot ends).

Run: DISPLAY=:99 python scratch/shoot137.py
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
_HOME = tempfile.mkdtemp(prefix="shoot137-home-")
os.environ["HOME"] = _HOME

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None
playground.messagebox.showerror = lambda *a, **k: None
playground.messagebox.showwarning = lambda *a, **k: None

OUT = Path(__file__).resolve().parents[1] / "download"
SHARED_ORDER = OUT / "pg137-shared-order.json"
TARGET_HTML = OUT / "pg137-anchors-report.html"
playground.filedialog.asksaveasfilename = lambda **_k: str(TARGET_HTML)

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


def visual_check_labels(app, dialog):
    checks = widgets_of(app, dialog, "TCheckbutton")
    body = checks[0].master.master.master
    labels = []
    for group_frame in body.pack_slaves():
        for child in group_frame.pack_slaves():
            for check in checks:
                if check.master is child:
                    labels.append(str(check.cget("text")))
    return labels


def find_picklist(app):
    return next(
        child
        for child in app.winfo_children()
        if child.winfo_class() == "Toplevel"
        and str(child.title()) == "Session report contents"
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

    # the FILE MENU posted with the round's two new items
    def menu_stage():
        menubar = app.nametowidget(app.cget("menu"))
        index = menubar.index("File")
        file_menu = app.nametowidget(menubar.entrycget(index, "menu"))
        last = int(file_menu.index("end"))
        labels = [
            str(file_menu.entrycget(i, "label"))
            for i in range(last + 1)
            if str(file_menu.type(i)) == "command"
        ]
        assert "Export remembered order…" in labels, labels
        assert "Import remembered order…" in labels, labels
        assert labels.index("Export remembered order…") == (
            labels.index("Clear remembered report order…") + 1
        ), labels
        assert labels.index("Import remembered order…") == (
            labels.index("Export remembered order…") + 1
        ), labels
        file_menu.post(app.winfo_rootx() + 24, app.winfo_rooty() + 34)

        def after_shot():
            shot(app, "pg137-file-menu")
            file_menu.unpost()
            # the export runs: the standalone artifact lands
            playground.filedialog.asksaveasfilename = lambda **_k: str(SHARED_ORDER)
            app._export_report_order()
            state["exported"] = SHARED_ORDER.is_file()

        app.after(900, after_shot)

    app.after(400, menu_stage)
    run_for(app, 3.0)
    assert state.get("exported") is True
    payload = json.loads(SHARED_ORDER.read_text(encoding="utf-8"))
    assert payload == {
        "format": playground.REPORT_ORDER_FORMAT,
        "kind": playground.REPORT_ORDER_KIND,
        "order": {
            "table": ["SMILES inspector"],
            "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
        },
    }, payload
    print("artifact pg137-shared-order.json written (the standalone order)")
finally:
    import contextlib

    with contextlib.suppress(Exception):
        if app.winfo_exists():
            app.destroy()

import gc  # noqa: E402

gc.collect()

# -- App #2: fresh HOME, the IMPORT adopts the shared order -------------- #
_HOME2 = tempfile.mkdtemp(prefix="shoot137-import-")
os.environ["HOME"] = _HOME2
app = playground.PlaygroundApp(persist_session=True)
app.geometry("1280x860+10+10")
try:
    pump(app)
    assert app._report_page_order == {}  # a fresh launch remembers nothing
    # the import: a colleague's order file replaces the (empty) memory
    playground.filedialog.askopenfilename = lambda **_k: str(SHARED_ORDER)
    app._import_report_order()
    assert app._report_page_order == {
        "table": ["SMILES inspector"],
        "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
    }, app._report_page_order
    # the write-through landed (the next launch keeps the imported order)
    session_file = Path(_HOME2) / ".pharmaml" / "playground-session.json"
    assert session_file.is_file()
    stored = json.loads(session_file.read_text(encoding="utf-8"))
    assert json.loads(stored["texts"]["report_page_order"]) == {
        "table": ["SMILES inspector"],
        "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
    }
    print("the import adopted the colleague's order and wrote through")

    seed(app)

    # the picklist re-opens PRE-ARRANGED in the imported order
    def stage_five():
        dialog = find_picklist(app)
        state["imported"] = visual_check_labels(app, dialog)
        notes = [
            str(w.cget("text"))
            for w in app._walk_widgets(dialog)
            if w.winfo_class() == "TLabel" and w.winfo_manager() != ""
        ]
        assert any("last saved order" in n for n in notes), notes
        shot(app, "pg137-imported-order")

        def stage_six():
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(700, stage_six)

    app.after(500, stage_five)
    app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    )
    assert state["imported"] == [
        "table 1 · SMILES inspector",
        "figure 1 · beta plot",
        "figure 2 · Ring topology moves — 6 mutant(s)",
    ], state["imported"]
    print("the picklist re-opened pre-arranged in the imported order")

    # the HTML report saved from the imported order carries the anchors
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
    assert TARGET_HTML.is_file(), "the anchors report never landed"
    html = TARGET_HTML.read_text(encoding="utf-8")
    assert "<nav class='contents' id='contents' tabindex='-1' aria-label='Contents'>" in html
    assert "<li><a href='#pg-1'>table 1 · SMILES inspector</a></li>" in html
    assert "<li><a href='#pg-2'>figure 1 · beta plot</a></li>" in html
    assert "<li><a href='#pg-3'>figure 2 · Ring topology moves — 6 mutant(s)</a></li>" in html
    assert "<section class='page' id='pg-1' tabindex='-1'>" in html
    assert "<figure class='page' id='pg-2' tabindex='-1'>" in html
    assert "section.page:target, figure.page:target, nav.contents:target {" in html
    print("artifact pg137-anchors-report.html written (the clickable anchors)")
finally:
    import contextlib

    with contextlib.suppress(Exception):
        if app.winfo_exists():
            app.destroy()
gc.collect()

import matplotlib.pyplot as plt  # noqa: E402

plt.close("all")
print("SHOOT137_OK")
