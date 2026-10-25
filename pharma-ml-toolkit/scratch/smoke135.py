"""SMOKE135 -- the 100x-135 round's headless driver.

Proves the round's three features end-to-end, no display needed for
the session-file axis (the picklist dialog itself rides Xvfb):

1. The picklist's order memory SURVIVES RESTARTS -- a completed Save
   stores the order, the close-time auto-save persists it, and a
   FRESH PlaygroundApp replants it before the first picklist opens.
2. The Reset button's forget rides the file too -- the next launch
   starts at the walk order.
3. A corrupt/foreign ``report_page_order`` value resets per kind,
   never the restore.
4. The PDF twin's title page carries the CONTENTS section (library
   axis -- pypdf reads the page names back off page 0, walking order,
   tables first, a shared name listed twice).
5. The round's docstring pins.

Run:  DISPLAY=:99 python scratch/smoke135.py   ->  SMOKE135_OK
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
    home = Path(tempfile.mkdtemp(prefix="smoke135-home-"))
    os.environ["HOME"] = str(home)

    live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
    tables = [("T one", ["a"], [["1"]])]

    # -- 1: the memory rides the session file ---------------------------- #
    app = playground.PlaygroundApp(persist_session=True)
    for _ in range(4):
        app.update_idletasks()
        app.update()

    def first_save() -> None:
        dialog = None
        for child in list(app.winfo_children()):
            if child.winfo_class() == "Toplevel" and (
                str(child.title()) == "Session report contents"
            ):
                dialog = child
        assert dialog is not None
        ups = [
            w
            for w in app._walk_widgets(dialog)
            if w.winfo_class() == "TButton" and str(w.cget("text")) == "▲"
        ]
        ups[2].invoke()  # beta above alpha (creation: T one, alpha, beta)
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
    app._on_close()
    gc.collect()
    banner("1a. completed Save stored the dialog order")

    # the file itself carries the JSON key
    raw = json.loads(
        (home / ".pharmaml" / "playground-session.json").read_text(encoding="utf-8")
    )
    assert json.loads(raw["texts"]["report_page_order"]) == {
        "table": ["T one"],
        "figure": ["beta plot", "alpha plot"],
    }, raw["texts"]["report_page_order"]
    banner("1b. the session file carries the report_page_order key")

    # a FRESH launch replants it before the first picklist opens
    reopened = playground.PlaygroundApp(persist_session=True)
    for _ in range(4):
        reopened.update_idletasks()
        reopened.update()
    assert reopened._report_page_order == {
        "table": ["T one"],
        "figure": ["beta plot", "alpha plot"],
    }, reopened._report_page_order
    banner("1c. the fresh launch replanted the memory")

    # -- 2: the Reset forget rides the file ------------------------------- #
    def reset_visit() -> None:
        dialog = None
        for child in list(reopened.winfo_children()):
            if child.winfo_class() == "Toplevel" and (
                str(child.title()) == "Session report contents"
            ):
                dialog = child
        assert dialog is not None
        for w in app._walk_widgets(dialog):
            if w.winfo_class() == "TButton" and (
                str(w.cget("text")) == "Reset to walking order"
            ):
                w.invoke()
                break
        for w in app._walk_widgets(dialog):
            if w.winfo_class() == "TButton" and str(w.cget("text")) == "Cancel":
                w.invoke()
                break

    reopened.after(150, reset_visit)
    reopened._session_report_picklist(live, [])
    assert reopened._report_page_order == {}, reopened._report_page_order
    reopened._on_close()
    gc.collect()
    banner("2a. the Reset button forgot (Cancel never re-stored it)")

    third = playground.PlaygroundApp(persist_session=True)
    for _ in range(4):
        third.update_idletasks()
        third.update()
    assert third._report_page_order == {}, third._report_page_order
    third._on_close()
    gc.collect()
    banner("2b. the forget survived the restart (walk order on launch)")

    # -- 3: poison values reset per kind, never the restore ---------------- #
    poisons = [
        ("not json at all", {}),
        (json.dumps(["a", "list"]), {}),
        (
            json.dumps({"figure": "not a list", "table": ["ok one"]}),
            {"table": ["ok one"]},
        ),
    ]
    for poison, expected in poisons:
        payload = playground.dump_session(
            theme="light",
            geometry="1000x700+10+10",
            tab=0,
            variables={"_aff_value": "2.5"},
            families={},
            texts={"report_page_order": poison},
        )
        (home / ".pharmaml").mkdir(parents=True, exist_ok=True)
        (home / ".pharmaml" / "playground-session.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        app4 = playground.PlaygroundApp(persist_session=True)
        for _ in range(4):
            app4.update_idletasks()
            app4.update()
        assert app4._report_page_order == expected, (poison, app4._report_page_order)
        assert app4._aff_value.get() == "2.5"  # the RESTORE itself landed
        app4.destroy()
        gc.collect()
    banner("3.  poison values reset per kind, the restore itself landed")

    # -- 4: the PDF title page's CONTENTS section -------------------------- #
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    from pharmaml.viz import save_pdf_report

    out_dir = Path(tempfile.mkdtemp(prefix="smoke135-pdf-"))
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot([0, 1], [0, 1])
    axes.set_title("alpha")
    fig2 = plt.figure()
    axes2 = fig2.add_subplot(111)
    axes2.plot([0, 1], [1, 0])
    axes2.set_title("beta")
    pdf_path = save_pdf_report(
        {
            "figure 1 · alpha plot": fig,
            "figure 2 · beta plot": fig2,
        },
        tables={
            "table 1 · T one": pd.DataFrame({"a": ["1"]}),
            "table 2 · T two": pd.DataFrame({"b": ["2"]}),
        },
        out_path=out_dir / "smoke135.pdf",
        close=False,
    )
    plt.close("all")

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page0 = reader.pages[0].extract_text() or ""
    assert "Contents" in page0, page0
    # the walking order: tables first, then figures
    assert page0.index("Table 1 · T One") < page0.index("Table 2 · T Two")
    assert page0.index("Table 2 · T Two") < page0.index("Figure 1 · Alpha Plot")
    assert page0.index("Figure 1 · Alpha Plot") < page0.index("Figure 2 · Beta Plot")
    assert len(reader.pages) == 5  # title + 2 tables + 2 figures
    banner("4.  the PDF title page carries the Contents section (walk order)")

    # the shared-name honesty: one name, two pages, listed twice
    fig3 = plt.figure()
    fig3.add_subplot(111).plot([0, 1], [0, 1])
    shared = save_pdf_report(
        {"shared": fig3},
        tables={"shared": pd.DataFrame({"a": [1]})},
        out_path=out_dir / "shared.pdf",
        close=False,
    )
    plt.close("all")
    shared0 = PdfReader(str(shared)).pages[0].extract_text() or ""
    assert shared0.count("Shared") == 2, shared0
    banner("4b. a shared name lists twice (no silent dict-merge drop)")

    # -- 5: the docstring pins --------------------------------------------- #
    doc = playground.__doc__ or ""
    assert "100x-135" in doc and "survives RESTARTS" in doc
    pick_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
    assert "100x-135" in pick_doc and "survives RESTARTS" in pick_doc
    pdf_doc = playground.PlaygroundApp._menu_save_session_report.__doc__ or ""
    assert "100x-135" in pdf_doc and "CONTENTS" in pdf_doc
    banner("5.  the module/picklist/menu docstrings record the round")

    print("SMOKE135_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
