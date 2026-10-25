#!/usr/bin/env python3
"""SMOKE 139 — the round's end-to-end story on the real Tk app.

Round 100x-139: the return link's POINTER confession (the "↑ Contents"
link carries a title tooltip -- the aria world always knew the
destination, a hovering pointer had no clue), the web preview's census
became CLICKABLE LEGS with the SPOTLIGHT (the parity suite pins the
web half; this smoke drives the desktop half), and the strip's
confession grew the ORDER's provenance -- the web twin of THIS app's
picklist memory label.

The story:
  1. the one-string discipline: the library's _BACK_CONTENTS_LINK is
     the ONLY return link (the playground never inlines one), and it
     carries the tooltip's exact wording;
  2. the web twin's ANCHOR: the picklist's confession wording ("Your
     last saved order was applied") stays -- the web strip's new
     confession mirrors it, so the desktop's half of the pair must stay
     findable;
  3. the app's own save path ships the tooltip: a planted figure + a
     filled table -> File ▸ Save session report -> every page wrapper
     in the written HTML ends in the SAME tooltip-carrying link (one
     figure + one table = exactly two), the anchors unchanged;
  4. the round's docstring pins.

Run under Xvfb:  DISPLAY=:99 python scratch/smoke139.py
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import playground  # noqa: E402


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke139-"))

    # -- 1: the one-string discipline ---------------------------------------- #
    from pharmaml.viz.report import _BACK_CONTENTS_LINK

    assert _BACK_CONTENTS_LINK == (
        "<a class='back-contents' href='#contents' "
        "title='Back to the contents block at the top of the report'>"
        "\u2191 Contents</a>"
    ), _BACK_CONTENTS_LINK
    source_text = (ROOT / "playground.py").read_text(encoding="utf-8")
    assert "back-contents" not in source_text, (
        "the playground grew an inline return link -- the one-string "
        "discipline broke"
    )
    banner("1.  the one-string link (the tooltip's exact wording, no inline twin)")

    # -- 2: the web twin's ANCHOR --------------------------------------------- #
    picklist_source = inspect.getsource(
        playground.PlaygroundApp._session_report_picklist
    )
    assert "Your last saved order was applied" in picklist_source
    assert "Reset to " in picklist_source and "walking order forgets it." in picklist_source
    # the imported-order wording stays too (the confession's other state)
    assert "The imported order from " in picklist_source
    banner("2.  the picklist wording stays the web twin's anchor")

    # -- 3: the app's own save path ships the tooltip -------------------------- #
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    app = playground.PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()

    seen: list[tuple[str, tuple]] = []
    for name in ("showinfo", "showerror"):
        def spy(title, message, _n=name, **_k):
            seen.append((_n, (title, message)))

        setattr(playground.messagebox, name, spy)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [0, 1])
    axes.set_title("alpha plot")
    app._render_figure(app._topo_host, figure, caption="alpha plot")
    app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
    for _ in range(4):
        app.update_idletasks()
        app.update()

    target = tmp / "session-report.html"
    playground.filedialog.asksaveasfilename = lambda **_k: str(target)  # type: ignore[assignment]

    # bypass the pre-save picklist (accept every page) -- the smoke's
    # target is the WRITTEN file, not the dialog flow (smoke138 drove it)
    original_picklist = playground.PlaygroundApp._session_report_picklist
    playground.PlaygroundApp._session_report_picklist = (
        lambda self, live, tables: (live, tables)
    )
    try:
        app._menu_save_session_report_html()
        deadline = time.monotonic() + 30.0
        while not target.is_file() and time.monotonic() < deadline:
            app.update()
            time.sleep(0.01)
        assert target.is_file(), "the save job never wrote the file"
        # the completion dialog rides the jobs queue's poller -- keep
        # pumping until the reader-facing showinfo fires (or a hard cap)
        done_deadline = time.monotonic() + 10.0
        while time.monotonic() < done_deadline:
            app.update()
            time.sleep(0.01)
            if any(
                call[0] == "showinfo" and call[1][0] == "Session report saved"
                for call in seen
            ):
                break
    finally:
        playground.PlaygroundApp._session_report_picklist = original_picklist

    html = target.read_text(encoding="utf-8")
    assert html.count(_BACK_CONTENTS_LINK) == 2, html.count(_BACK_CONTENTS_LINK)
    assert (
        html.count("title='Back to the contents block at the top of the report'") == 2
    )
    # the round only GREW the tooltip -- the navigation holds as before
    assert "<nav class='contents' id='contents' tabindex='-1'" in html
    assert "<li><a href='#pg-1' title='Jump to this page in the report'>table 1 · SMILES inspector</a></li>" in html
    assert "<li><a href='#pg-2' title='Jump to this page in the report'>figure 1 · alpha plot</a></li>" in html
    # the saved-file confession reached the reader (the completion dialog)
    saved = [call for call in seen if call[0] == "showinfo" and call[1][0] == "Session report saved"]
    assert saved, [call[1][0] for call in seen]
    app.destroy()
    import gc

    gc.collect()
    plt.close("all")
    banner("3.  the app's save path ships the tooltip (both wrappers, one string)")

    # -- 4: the round's docstring pins ----------------------------------------- #
    doc = playground.__doc__ or ""
    assert "100x-139" in doc
    assert "POINTER confession" in doc
    assert "SPOTLIGHTS" in doc
    assert "last saved walking order was applied" in doc
    banner("4.  the round's docstring pins")

    print("\nSMOKE139_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
