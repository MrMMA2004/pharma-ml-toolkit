#!/usr/bin/env python3
"""SMOKE 137 — the round's end-to-end story on the real Tk app.

Round 100x-137: the order memory's SHARE pair (File ▸ Export /
Import remembered order -- the standalone artifact), the HTML twin's
CLICKABLE Contents anchors, and the web preview's Contents twin (the
web side pins in the parity suite; this smoke drives the desktop half).

The story:
  1. the pure normalisers: dump/load roundtrip + the foreign rejections;
  2. the export: a remembered order writes the standalone artifact
     (marker + kind tag + the two label lists, NONE of the session's
     inputs); a memory that never existed confesses honestly;
  3. the import: a colleague's file REPLACES the memory and writes the
     session through (the next launch keeps it); a foreign file never
     touches the live memory; the empty order imports as the reset;
  4. the picklist re-opens PRE-ARRANGED in the imported order;
  5. the HTML twin's Contents lines are ANCHORS (the saved file's lines
     link to the id-carrying page wrappers);
  6. the round's docstring pins.

Run under Xvfb:  DISPLAY=:99 python scratch/smoke137.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import playground  # noqa: E402


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> int:
    # -- 1: the pure normalisers ------------------------------------------- #
    dumped = playground.dump_report_order(
        {"figure": ["beta plot", "alpha plot"], "table": ["T one"], "junk": ["x"]}
    )
    assert dumped == {
        "format": playground.REPORT_ORDER_FORMAT,
        "kind": playground.REPORT_ORDER_KIND,
        "order": {"table": ["T one"], "figure": ["beta plot", "alpha plot"]},
    }, dumped
    loaded = playground.load_report_order(dumped)
    assert loaded == {"table": ["T one"], "figure": ["beta plot", "alpha plot"]}, loaded
    assert playground.load_report_order({"format": 1}) is None
    assert playground.load_report_order(None) is None
    assert playground.load_report_order(playground.dump_report_order({})) == {}
    banner("1.  the pure normalisers (dump/load roundtrip + rejections)")

    # -- 2: the export ------------------------------------------------------ #
    import tkinter as tk
    import tkinter.filedialog as filedialog
    import tkinter.messagebox as messagebox

    app = playground.PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()

    seen: list[tuple[str, tuple]] = []
    for name in ("showinfo", "showerror"):
        def spy(title, message, _n=name, **_k):
            seen.append((_n, (title, message)))

        setattr(playground.messagebox, name, spy)

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="smoke137-"))
    target = tmp / "shared-order.json"
    playground.filedialog.asksaveasfilename = lambda **_k: str(target)  # type: ignore[assignment]

    # nothing remembered -> the honest nothing-to-share, no file
    app._export_report_order()
    assert not target.exists(), "an empty memory wrote a file"
    assert seen and seen[-1][1][0] == "No remembered order", seen
    assert "Nothing to share yet" in seen[-1][1][1]

    # a remembered order -> the standalone artifact
    app._report_page_order = {
        "figure": ["beta plot", "alpha plot"],
        "table": ["T one"],
    }
    app._export_report_order()
    assert target.is_file(), "the export never wrote the file"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "format": playground.REPORT_ORDER_FORMAT,
        "kind": playground.REPORT_ORDER_KIND,
        "order": {"table": ["T one"], "figure": ["beta plot", "alpha plot"]},
    }, payload
    assert "variables" not in payload and "texts" not in payload
    assert "report order exported" in str(app._job_label.cget("text"))
    banner("2.  the export (the standalone artifact + the honest empty)")

    # -- 3: the import ------------------------------------------------------ #
    # a foreign file never touches the live memory
    foreign = tmp / "foreign.json"
    foreign.write_text('{"format": 1, "theme": "light"}', encoding="utf-8")
    playground.filedialog.askopenfilename = lambda **_k: str(foreign)  # type: ignore[assignment]
    app._import_report_order()
    assert app._report_page_order == {
        "figure": ["beta plot", "alpha plot"],
        "table": ["T one"],
    }, app._report_page_order
    errors = [call for call in seen if call[0] == "showerror"]
    assert errors and errors[-1][1][0] == "Invalid report order file", seen
    assert "not touched" in errors[-1][1][1]

    # the colleague's file replaces the memory (in this process)
    colleague = tmp / "colleague.json"
    colleague.write_text(
        json.dumps(
            playground.dump_report_order(
                {"figure": ["beta plot", "alpha plot"], "table": ["T one"]}
            )
        ),
        encoding="utf-8",
    )
    playground.filedialog.askopenfilename = lambda **_k: str(colleague)  # type: ignore[assignment]
    app._report_page_order = {}
    app._import_report_order()
    assert app._report_page_order == {
        "figure": ["beta plot", "alpha plot"],
        "table": ["T one"],
    }, app._report_page_order
    assert "report order imported" in str(app._job_label.cget("text"))
    banner("3.  the import (foreign rejected + the colleague's order adopted)")

    # -- 4: the picklist re-opens pre-arranged ------------------------------ #
    captured: dict[str, list[str]] = {}

    def visit() -> None:
        dialog = None
        for child in list(app.winfo_children()):
            if child.winfo_class() == "Toplevel" and str(child.title()) == (
                "Session report contents"
            ):
                dialog = child
        assert dialog is not None
        checks = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TCheckbutton"
        ]
        captured["labels"] = [str(check.cget("text")) for check in checks]
        next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "Cancel"
        ).invoke()

    live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
    tables = [("T one", ["h"], [["r"]])]
    app.after(150, visit)
    assert app._session_report_picklist(live, tables) is None  # cancelled
    assert captured["labels"] == [
        "table 1 · T one",
        "figure 1 · beta plot",
        "figure 2 · alpha plot",
    ], captured["labels"]
    banner("4.  the picklist re-opens pre-arranged in the imported order")

    # -- 5: the HTML twin's clickable Contents anchors ----------------------- #
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
    # the lines are ANCHORS that resolve to real page ids
    assert "<li><a href='#pg-1'>table 1 · SMILES inspector</a></li>" in html
    assert "<li><a href='#pg-2'>figure 1 · alpha plot</a></li>" in html
    assert "<section class='page' id='pg-1' tabindex='-1'>" in html
    assert "<figure class='page' id='pg-2' tabindex='-1'>" in html
    # the link styling + the :target ring ride the theme variables
    assert "ul.contents-list a:hover, ul.contents-list a:focus-visible {" in html
    assert "section.page:target, figure.page:target {" in html
    assert "scroll-margin-top: 1.25rem;" in html
    banner("5.  the HTML twin's clickable Contents anchors")

    plt.close("all")
    app.destroy()

    # -- 6: the docstring pins ----------------------------------------------- #
    doc = playground.__doc__ or ""
    assert "100x-137" in doc and "SHAREABLE" in doc
    export_doc = playground.PlaygroundApp._export_report_order.__doc__ or ""
    assert "100x-137" in export_doc
    import_doc = playground.PlaygroundApp._import_report_order.__doc__ or ""
    assert "100x-137" in import_doc
    picklist_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
    assert "100x-137" in picklist_doc and "SHAREABLE" in picklist_doc
    html_doc = playground.PlaygroundApp._menu_save_session_report_html.__doc__ or ""
    assert "100x-137" in html_doc and "CLICKABLE anchor" in html_doc
    banner("6.  the round's docstring pins")

    print("SMOKE137_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
