#!/usr/bin/env python3
"""SMOKE 138 — the round's end-to-end story on the real Tk app.

Round 100x-138: the picklist confession's PROVENANCE (the imported
order names its file until the reader's own Save retires it), the HTML
twin's RETURN leg (every page wrapper ends in a "↑ Contents" link back
to the header's block), and the web strip's KIND CENSUS (the parity
suite pins the web half; this smoke drives the desktop half).

The story:
  1. the provenance attribute's life: fresh app -> None; the import
     carries its source; the empty import confesses nothing; a foreign
     file never touches it; Clear forgets it with the memory;
  2. the picklist's confession names the FILE after an import, retires
     to "Your last saved order" after a completed Save, and keeps the
     default wording across a RESTART (the session restore IS the last
     saved order -- a fresh launch never claims an import);
  3. the HTML twin's RETURN leg: every page wrapper ends in the back
     link, the nav is its target, the ring covers the nav;
  4. the round's docstring pins.

Run under Xvfb:  DISPLAY=:99 python scratch/smoke138.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import playground  # noqa: E402


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke138-"))

    # -- 1: the provenance attribute's life --------------------------------- #
    app = playground.PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()

    seen: list[tuple[str, tuple]] = []
    for name in ("showinfo", "showerror"):
        def spy(title, message, _n=name, **_k):
            seen.append((_n, (title, message)))

        setattr(playground.messagebox, name, spy)

    assert app._report_order_origin is None  # a fresh app is nobody's import

    colleague = tmp / "colleague-order.json"
    colleague.write_text(
        json.dumps(
            playground.dump_report_order(
                {"figure": ["beta plot", "alpha plot"], "table": ["T one"]}
            )
        ),
        encoding="utf-8",
    )
    playground.filedialog.askopenfilename = lambda **_k: str(colleague)  # type: ignore[assignment]
    app._import_report_order()
    assert app._report_order_origin == str(colleague), app._report_order_origin

    # a foreign file never touches the live memory OR its provenance
    foreign = tmp / "foreign.json"
    foreign.write_text('{"format": 1}', encoding="utf-8")
    playground.filedialog.askopenfilename = lambda **_k: str(foreign)  # type: ignore[assignment]
    app._import_report_order()
    assert app._report_order_origin == str(colleague), "the provenance was touched"
    errors = [call for call in seen if call[0] == "showerror"]
    assert errors and errors[-1][1][0] == "Invalid report order file"

    # the honest empty import confesses nothing
    empty = tmp / "empty.json"
    empty.write_text(json.dumps(playground.dump_report_order({})), encoding="utf-8")
    playground.filedialog.askopenfilename = lambda **_k: str(empty)  # type: ignore[assignment]
    app._import_report_order()
    assert app._report_page_order == {}
    assert app._report_order_origin is None

    # the import again (for the confession story below), then Clear
    playground.filedialog.askopenfilename = lambda **_k: str(colleague)  # type: ignore[assignment]
    app._import_report_order()
    assert app._report_order_origin == str(colleague)
    app._clear_report_order()
    assert app._report_page_order == {}
    assert app._report_order_origin is None, "Clear forgot the memory but not the origin"
    banner("1.  the provenance attribute's life (import/empty/foreign/clear)")

    # -- 2: the confession's wording ----------------------------------------- #
    playground.filedialog.askopenfilename = lambda **_k: str(colleague)  # type: ignore[assignment]
    app._import_report_order()
    live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
    tables = [("T one", ["h"], [["r"]])]
    captured: dict[str, str] = {}

    def memory_text(app_ref, dialog):
        return next(
            str(widget.cget("text"))
            for widget in app_ref._walk_widgets(dialog)
            if widget.winfo_class() == "TLabel" and "was applied" in str(widget.cget("text"))
        )

    def visit_imported() -> None:
        dialog = next(
            child
            for child in list(app.winfo_children())
            if child.winfo_class() == "Toplevel"
            and str(child.title()) == "Session report contents"
        )
        captured["imported"] = memory_text(app, dialog)
        next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "Cancel"
        ).invoke()

    app.after(150, visit_imported)
    assert app._session_report_picklist(live, tables) is None
    assert captured["imported"] == (
        "The imported order from colleague-order.json was applied — "
        "Reset to walking order forgets it."
    ), captured["imported"]

    # a completed Save retires the provenance
    def visit_save() -> None:
        dialog = next(
            child
            for child in list(app.winfo_children())
            if child.winfo_class() == "Toplevel"
            and str(child.title()) == "Session report contents"
        )
        next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton"
            and str(widget.cget("text")).startswith("Save report")
        ).invoke()

    app.after(150, visit_save)
    assert app._session_report_picklist(live, tables) is not None
    assert app._report_order_origin is None, "the Save did not retire the provenance"

    def visit_own() -> None:
        dialog = next(
            child
            for child in list(app.winfo_children())
            if child.winfo_class() == "Toplevel"
            and str(child.title()) == "Session report contents"
        )
        captured["own"] = memory_text(app, dialog)
        next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "Cancel"
        ).invoke()

    app.after(150, visit_own)
    assert app._session_report_picklist(live, tables) is None
    assert captured["own"] == (
        "Your last saved order was applied — Reset to walking order forgets it."
    ), captured["own"]
    app.destroy()
    banner("2.  the confession (imported -> the reader's own, honestly)")
    import gc

    gc.collect()

    # -- 2b: the RESTART contract -------------------------------------------- #
    # a seeding app imports + writes through; a FRESH launch restores from
    # the session file (which IS the last saved order) and keeps the
    # default wording -- a restart never claims an import
    import os

    home = Path(tempfile.mkdtemp(prefix="smoke138-home-"))
    os.environ["HOME"] = str(home)
    seeding = playground.PlaygroundApp()
    for _ in range(4):
        seeding.update_idletasks()
        seeding.update()
    playground.filedialog.askopenfilename = lambda **_k: str(colleague)  # type: ignore[assignment]
    seeding._import_report_order()
    seeding.destroy()
    gc.collect()
    session_file = home / ".pharmaml" / "playground-session.json"
    assert session_file.is_file(), "the import never wrote the session file"

    fresh = playground.PlaygroundApp()
    for _ in range(4):
        fresh.update_idletasks()
        fresh.update()
    assert fresh._report_page_order == {
        "figure": ["beta plot", "alpha plot"],
        "table": ["T one"],
    }
    assert fresh._report_order_origin is None, "a restart claimed the import"

    def visit_restart() -> None:
        dialog = next(
            child
            for child in list(fresh.winfo_children())
            if child.winfo_class() == "Toplevel"
            and str(child.title()) == "Session report contents"
        )
        captured["restart"] = memory_text(fresh, dialog)
        next(
            widget
            for widget in fresh._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "Cancel"
        ).invoke()

    fresh.after(150, visit_restart)
    assert fresh._session_report_picklist(live, tables) is None
    assert captured["restart"] == (
        "Your last saved order was applied — Reset to walking order forgets it."
    ), captured["restart"]
    fresh.destroy()
    gc.collect()
    banner("2b. the restart contract (the session restore keeps the wording)")

    # -- 3: the HTML twin's RETURN leg ---------------------------------------- #
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    from pharmaml.viz import HTMLReport

    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    html = (
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
        .render()
    )
    # the nav is the RETURN target; every page wrapper ends in the link
    assert "<nav class='contents' id='contents' tabindex='-1' aria-label='Contents'>" in html
    assert (
        "</table><a class='back-contents' href='#contents'>\u2191 Contents</a></section>" in html
    )
    assert (
        "<figcaption>figure 1 · alpha plot</figcaption>"
        "<a class='back-contents' href='#contents'>\u2191 Contents</a></figure>" in html
    )
    assert html.count("<a class='back-contents' href='#contents'>") == 2
    # the ring + the margin cover the nav (the return landing)
    assert "section.page:target, figure.page:target, nav.contents:target {" in html
    assert "section.page, figure.page, nav.contents { scroll-margin-top: 1.25rem; }" in html
    assert "a.back-contents:hover, a.back-contents:focus-visible {" in html
    banner("3.  the HTML twin's RETURN leg (the round trip)")

    plt.close("all")

    # -- 4: the docstring pins ------------------------------------------------- #
    doc = playground.__doc__ or ""
    assert "100x-138" in doc and "RETURN leg" in doc
    assert "KIND CENSUS" in doc and "PROVENANCE" in doc
    import_doc = playground.PlaygroundApp._import_report_order.__doc__ or ""
    assert "100x-138" in import_doc and "PROVENANCE" in import_doc
    picklist_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
    assert "100x-138" in picklist_doc and "PROVENANCE" in picklist_doc
    banner("4.  the round's docstring pins")

    print("SMOKE138_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
