"""SMOKE136 -- the 100x-136 round's headless driver.

Proves the round's three features end-to-end, no display needed for
the session-file axis (the picklist dialog itself rides Xvfb):

1. The File menu's Clear remembered report order -- a memory planted
   by a completed Save is dropped AND the session file is written
   through immediately (the forget survives a crash); the NEXT
   picklist opens in the walking order with its confession quiet.
2. The honest messages -- a memory that never existed gets "Nothing
   to forget yet"; persist_session=False names its scope; a failed
   file write confesses that the next launch restores the order.
3. The HTML twin's CONTENTS block (library axis -- the header nav,
   the count-prefixed lines in walking order, the page headings that
   match them, the skip when no table/figure pages exist).
4. The round's docstring pins.

Run:  DISPLAY=:99 python scratch/smoke136.py   ->  SMOKE136_OK
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import playground  # noqa: E402


def banner(text: str) -> None:
    print(f"  {text}")


def main() -> int:
    # an isolated HOME so the smoke never touches a real session file
    home = Path(tempfile.mkdtemp(prefix="smoke136-home-"))
    os.environ["HOME"] = str(home)

    # capture every messagebox instead of showing one
    dialogs: list[tuple[str, tuple]] = []
    for name in ("showinfo", "showwarning", "showerror"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _n=name, **_k: dialogs.append((_n, a)),
        )

    live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
    tables = [("T one", ["a"], [["1"]])]

    # -- 1: the File menu's Clear, end to end ----------------------------- #
    app = playground.PlaygroundApp(persist_session=True)
    for _ in range(4):
        app.update_idletasks()
        app.update()

    def first_save() -> None:
        dialog = next(
            child
            for child in list(app.winfo_children())
            if child.winfo_class() == "Toplevel"
            and str(child.title()) == "Session report contents"
        )
        ups = [
            w
            for w in app._walk_widgets(dialog)
            if w.winfo_class() == "TButton" and str(w.cget("text")) == "▲"
        ]
        ups[2].invoke()  # beta above alpha
        app.update()
        for w in app._walk_widgets(dialog):
            if w.winfo_class() == "TButton" and str(w.cget("text")) == "Save report…":
                w.invoke()
                break

    app.after(150, first_save)
    app._session_report_picklist(live, tables)
    assert app._report_page_order == {
        "table": ["T one"],
        "figure": ["beta plot", "alpha plot"],
    }, app._report_page_order
    banner("1a. a completed Save planted the memory (per session)")

    # the File-menu command on the planted memory
    app._clear_report_order()
    assert app._report_page_order == {}, app._report_page_order
    raw = json.loads(
        (home / ".pharmaml" / "playground-session.json").read_text(encoding="utf-8")
    )
    assert json.loads(raw["texts"]["report_page_order"]) == {}
    infos = [a for n, a in dialogs if n == "showinfo"]
    assert infos and infos[0][0] == "Remembered order cleared", dialogs
    assert "next launch forgets it too" in infos[0][1]
    assert "remembered report order cleared" in str(app._job_label.cget("text"))
    banner("1b. the File-menu clear dropped BOTH kinds and wrote through")

    # the next picklist walks in the WALKING order, confession quiet
    captured: dict[str, list[str]] = {}

    def quiet_visit() -> None:
        dialog = next(
            child
            for child in list(app.winfo_children())
            if child.winfo_class() == "Toplevel"
            and str(child.title()) == "Session report contents"
        )
        captured["labels"] = [
            str(w.cget("text"))
            for w in app._walk_widgets(dialog)
            if w.winfo_class() == "TCheckbutton"
        ]
        captured["notes"] = [
            str(w.cget("text"))
            for w in app._walk_widgets(dialog)
            if w.winfo_class() == "TLabel" and w.winfo_manager() != ""
        ]
        for w in app._walk_widgets(dialog):
            if w.winfo_class() == "TButton" and str(w.cget("text")) == "Cancel":
                w.invoke()
                break

    app.after(150, quiet_visit)
    app._session_report_picklist(live, tables)
    assert captured["labels"] == [
        "table 1 · T one",
        "figure 1 · alpha plot",
        "figure 2 · beta plot",
    ], captured
    assert not any("last saved order" in n for n in captured["notes"])
    banner("1c. the next picklist walked in the walk order, confession quiet")

    # the forget survived the restart (the write-through beat the close)
    app._on_close()
    gc.collect()
    reopened = playground.PlaygroundApp(persist_session=True)
    for _ in range(4):
        reopened.update_idletasks()
        reopened.update()
    assert reopened._report_page_order == {}, reopened._report_page_order
    reopened.destroy()
    gc.collect()
    banner("1d. the forget survived the restart")

    # -- 2: the honest messages -------------------------------------------- #
    dialogs.clear()
    fresh = playground.PlaygroundApp(persist_session=False)
    for _ in range(4):
        fresh.update_idletasks()
        fresh.update()
    fresh._clear_report_order()  # nothing remembered
    infos = [a for n, a in dialogs if n == "showinfo"]
    assert infos and infos[0][0] == "No remembered order", dialogs
    assert "Nothing to forget yet" in infos[0][1]
    banner("2a. no memory -> the honest 'nothing to forget'")

    dialogs.clear()
    fresh._report_page_order = {"figure": ["beta plot"], "table": []}
    fresh._clear_report_order()  # persist_session=False
    assert fresh._report_page_order == {}
    infos = [a for n, a in dialogs if n == "showinfo"]
    assert infos and "never reached a file" in infos[0][1], dialogs
    banner("2b. persist_session=False names its scope")

    dialogs.clear()
    writer = playground.PlaygroundApp(persist_session=True)
    for _ in range(4):
        writer.update_idletasks()
        writer.update()
    writer._report_page_order = {"figure": ["beta plot"], "table": []}
    original_write = playground.PlaygroundApp._write_session_file
    playground.PlaygroundApp._write_session_file = lambda self: False
    try:
        writer._clear_report_order()
    finally:
        playground.PlaygroundApp._write_session_file = original_write
    assert writer._report_page_order == {}
    infos = [a for n, a in dialogs if n == "showinfo"]
    assert infos and "restores the remembered order after all" in infos[0][1]
    banner("2c. the failed write confesses the launch restore")
    writer.destroy()
    gc.collect()
    fresh.destroy()
    gc.collect()

    # -- 3: the HTML twin's CONTENTS block (library axis) ------------------ #
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    from pharmaml.viz import HTMLReport

    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    report = (
        HTMLReport(
            "PharmaML Playground session",
            subtitle="2 figure(s) · 1 table(s) · smoke",
        )
        .add_metrics({"figures": "2", "tables": "1"})
        .add_text("provenance", heading="What this file contains")
        .add_table(
            pd.DataFrame([["MolWt", "180.16"]], columns=["descriptor", "value"]),
            heading="table 1 · SMILES inspector",
        )
        .add_figure(figure, caption="figure 1 · alpha plot")
    )
    html = report.render()
    assert "<nav class='contents' aria-label='Contents'>" in html
    # (100x-137) the standing pins updated: every Contents line rides
    # an anchor that links to its page's id
    assert "<li><a href='#pg-1'>table 1 · SMILES inspector</a></li>" in html
    assert "<li><a href='#pg-2'>figure 1 · alpha plot</a></li>" in html
    assert (
        html.index("<li><a href='#pg-1'>table 1 · SMILES inspector</a></li>")
        < html.index("<li><a href='#pg-2'>figure 1 · alpha plot</a></li>")
    )
    assert "<section class='page' id='pg-1' tabindex='-1'>" in html
    assert "<figure class='page' id='pg-2' tabindex='-1'>" in html
    assert "<h2>table 1 · SMILES inspector</h2>" in html
    assert html.index("aria-label='Contents'") < html.index("class='metrics'")
    prose = HTMLReport("prose").add_text("only prose").render()
    assert "<nav class='contents'" not in prose
    banner("3.  the HTML twin's Contents block (library axis)")

    plt.close("all")

    # -- 4: the docstring pins ---------------------------------------------- #
    doc = playground.__doc__ or ""
    assert "100x-136" in doc and "always-available FORGET" in doc
    clear_doc = playground.PlaygroundApp._clear_report_order.__doc__ or ""
    assert "100x-136" in clear_doc
    picklist_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
    assert "100x-136" in picklist_doc and "Clear remembered report order" in picklist_doc
    html_doc = playground.PlaygroundApp._menu_save_session_report_html.__doc__ or ""
    assert "100x-136" in html_doc and "CONTENTS block" in html_doc
    banner("4.  the round's docstring pins")

    print("SMOKE136_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
