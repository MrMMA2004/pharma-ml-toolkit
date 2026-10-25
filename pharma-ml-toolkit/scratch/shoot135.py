"""Screenshots + artifacts for the 100x-135 round: the picklist's
order memory ACROSS A RESTART.  (1) App #1 seeds two figures + a
filled table, moves a row, Saves — the memory is written — and its
close-time auto-save persists it to an isolated session file.  (2)
App #2 — a FRESH launch from that file — replants the memory before
its first picklist ever opens: the dialog shows the pre-arranged
rows, the confession line, and the armed Reset button
(pg135-restart-remembered.png).  (3) The PDF report is saved from
that remembered order — its title page's new CONTENTS section lists
the very order the picklist showed (pg135-contents-page.png, the
page rasterised with pdftoppm; the PDF itself rides along as an
artifact).  (4) The Reset press on the replanted memory restores
the walk order and collapses the confession (pg135-restart-reset.png).

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/shoot135.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground  # noqa: E402

# an isolated HOME so the shoot never touches a real session file
_HOME = tempfile.mkdtemp(prefix="shoot135-home-")
os.environ["HOME"] = _HOME

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

OUT = Path("/home/z/my-project/pharma-ml-toolkit/download")
TARGET_PDF = OUT / "pg135-contents-report.pdf"
playground.filedialog.asksaveasfilename = lambda **_k: str(TARGET_PDF)

state = {}


def pump(app, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


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
    """The same three pages on both launches, same labels."""
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


# -- App #1: seed, move a row, Save, close (the file is written) ------- #
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
        # creation order: the table's (disabled) ▼, then the mutants'
        # — move the mutants below beta before saving (the 134 move)
        downs[1].invoke()
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
    # the memory now carries the moved order (per session, 100x-134)
    assert app._report_page_order == {
        "table": ["SMILES inspector"],
        "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
    }, app._report_page_order
    app._on_close()  # the close-time auto-save persists it
    print("app #1 closed with the memory in the session file")
finally:
    import contextlib

    with contextlib.suppress(Exception):
        if app.winfo_exists():
            app.destroy()

import gc  # noqa: E402

gc.collect()

# -- App #2: a FRESH launch — the memory replanted ---------------------- #
app = playground.PlaygroundApp(persist_session=True)
app.geometry("1280x860+10+10")
try:
    pump(app)
    # the fresh launch replanted the memory BEFORE any picklist opened
    assert app._report_page_order == {
        "table": ["SMILES inspector"],
        "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
    }, app._report_page_order
    seed(app)

    # the first picklist of the session opens PRE-ARRANGED
    def stage_three():
        dialog = find_picklist(app)
        state["open"] = visual_check_labels(app, dialog)
        shot(app, "pg135-restart-remembered")

        def stage_four():
            cancel = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")) == "Cancel"
            )
            cancel.invoke()

        app.after(500, stage_four)

    app.after(500, stage_three)
    app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    )
    assert state["open"] == [
        "table 1 · SMILES inspector",
        "figure 1 · beta plot",
        "figure 2 · Ring topology moves — 6 mutant(s)",
    ], state["open"]
    print("the restart's first picklist opened pre-arranged")

    # the PDF report from the REMEMBERED order — its Contents page will
    # list that very order (the dialog inside the menu rides callbacks)
    def stage_five():
        dialog = find_picklist(app)

        def stage_six():
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(500, stage_six)

    app.after(500, stage_five)
    app._menu_save_session_report()
    assert pump(app), "the session report job hung"
    assert TARGET_PDF.is_file(), "the contents report never landed"
    print("artifact pg135-contents-report.pdf written (remembered order)")

    # rasterise the title page (the Contents section) for review
    import subprocess

    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            "110",
            str(TARGET_PDF),
            str(OUT / "pg135-contents-page"),
        ],
        check=True,
        capture_output=True,
    )
    print("shot pg135-contents-page (pdftoppm page 1)")

    # the Reset press on the replanted memory
    def stage_seven():
        dialog = find_picklist(app)
        reset = next(
            w
            for w in widgets_of(app, dialog, "TButton")
            if str(w.cget("text")) == "Reset to walking order"
        )
        assert not reset.instate(("disabled",))  # a replanted memory IS drift
        reset.invoke()
        app.update()
        state["after_reset"] = visual_check_labels(app, dialog)
        shot(app, "pg135-restart-reset")

        def stage_eight():
            cancel = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")) == "Cancel"
            )
            cancel.invoke()

        app.after(500, stage_eight)

    app.after(500, stage_seven)
    app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    )
    assert state["after_reset"] == [
        "table 1 · SMILES inspector",
        "figure 1 · Ring topology moves — 6 mutant(s)",
        "figure 2 · beta plot",
    ], state["after_reset"]
    print("the Reset restored the walk order on the replanted memory")
finally:
    if app.winfo_exists():
        app.destroy()
gc.collect()

import matplotlib.pyplot as plt  # noqa: E402

plt.close("all")
print("SHOOT135_OK")
