"""Screenshots + artifacts for the 100x-136 round: the File menu's
Clear remembered report order + the HTML twin's Contents block.
(1) App #1 seeds two figures + a filled table, moves a row, Saves --
the memory is planted.  (2) The FILE MENU is posted and shot with the
new item sitting between "Clear saved session…" and "Export session…"
(pg136-file-menu.png).  (3) The new command runs: the memory is
dropped and written through, and the NEXT picklist opens in the
walking order with its confession gone (pg136-after-clear.png).
(4) The HTML report saved from the walk order carries the header's
new CONTENTS block -- the count-prefixed lines in the walking order
(the pg136-contents-report.html artifact rides along; the block's
lines are asserted before the shoot ends).

Run: DISPLAY=:99 python scratch/shoot136.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground  # noqa: E402

# an isolated HOME so the shoot never touches a real session file
_HOME = tempfile.mkdtemp(prefix="shoot136-home-")
os.environ["HOME"] = _HOME

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

OUT = Path(__file__).resolve().parents[1] / "download"
TARGET_HTML = OUT / "pg136-contents-report.html"
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
        downs[1].invoke()  # the mutants below beta (the 134 move)
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

    # the FILE MENU posted with the round's new item, then the clear
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
        assert "Clear remembered report order…" in labels, labels
        assert labels.index("Clear remembered report order…") == (
            labels.index("Clear saved session…") + 1
        ), labels
        file_menu.post(app.winfo_rootx() + 24, app.winfo_rooty() + 34)

        def after_shot():
            shot(app, "pg136-file-menu")
            file_menu.unpost()
            # the new command runs: the memory drops + writes through
            app._clear_report_order()
            assert app._report_page_order == {}
            state["cleared"] = True

        app.after(900, after_shot)

    app.after(400, menu_stage)
    run_for(app, 3.0)
    assert state.get("cleared") is True
    print("the File-menu clear dropped the memory and wrote through")

    # the NEXT picklist opens in the walk order, confession gone
    def stage_five():
        dialog = find_picklist(app)
        state["after_clear"] = visual_check_labels(app, dialog)
        notes = [
            str(w.cget("text"))
            for w in app._walk_widgets(dialog)
            if w.winfo_class() == "TLabel" and w.winfo_manager() != ""
        ]
        assert not any("last saved order" in n for n in notes)
        shot(app, "pg136-after-clear")

        def stage_six():
            cancel = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")) == "Cancel"
            )
            cancel.invoke()

        app.after(500, stage_six)

    app.after(500, stage_five)
    app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    )
    assert state["after_clear"] == [
        "table 1 · SMILES inspector",
        "figure 1 · Ring topology moves — 6 mutant(s)",
        "figure 2 · beta plot",
    ], state["after_clear"]
    print("the next picklist walked in the walk order, confession gone")
finally:
    import contextlib

    with contextlib.suppress(Exception):
        if app.winfo_exists():
            app.destroy()

import gc  # noqa: E402

gc.collect()

# -- App #2: the HTML report with the header's CONTENTS block ---------- #
app = playground.PlaygroundApp(persist_session=True)
app.geometry("1280x860+10+10")
try:
    pump(app)
    assert app._report_page_order == {}  # the clear's write-through held
    seed(app)

    def stage_seven():
        dialog = find_picklist(app)

        def stage_eight():
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(500, stage_eight)

    app.after(500, stage_seven)
    app._menu_save_session_report_html()
    assert pump(app), "the HTML report job hung"
    assert TARGET_HTML.is_file(), "the contents report never landed"
    html = TARGET_HTML.read_text(encoding="utf-8")
    assert "<nav class='contents' id='contents' tabindex='-1' aria-label='Contents'>" in html
    assert "<li>table 1 · SMILES inspector</li>" in html
    assert "<li>figure 1 · Ring topology moves — 6 mutant(s)</li>" in html
    assert "<li>figure 2 · beta plot</li>" in html
    print("artifact pg136-contents-report.html written (the Contents block)")
finally:
    if app.winfo_exists():
        app.destroy()
gc.collect()

import matplotlib.pyplot as plt  # noqa: E402

plt.close("all")
print("SHOOT136_OK")
