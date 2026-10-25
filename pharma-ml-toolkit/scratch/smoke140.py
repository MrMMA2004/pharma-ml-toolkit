#!/usr/bin/env python3
"""SMOKE 140 — the round's end-to-end story on the real Tk app.

Round 100x-140: the forward Contents lines gained the POINTER
confession (every line's anchor whispers "Jump to this page in the
report" to a hovering pointer -- the same tooltip discipline the
return link gained in 100x-139, applied symmetrically), the web
census's spotlight grew a KEYBOARD path (digits 1/2/3 through the same
pure map a click consumes; 0 clears -- the parity suite pins the web
half), and the web strip's confession grew its way out INSIDE the
dialog (a "Reset now" link -- web-side too).

The story (the desktop half):
  1. the one-place minting: the forward tooltip rides the library's
     render join and ONLY there (the playground never inlines the
     markup -- the one-string discipline the return link taught,
     applied to the whole line);
  2. the app's own save path ships it: a planted figure + a filled
     table -> File ▸ Save session report -> EVERY Contents line in
     the written HTML carries the tooltip (one figure + one table =
     exactly two), the return leg's tooltip still riding every
     wrapper too (the round trip confesses in BOTH directions);
  3. the round's docstring pins.

Run under Xvfb:  DISPLAY=:99 python scratch/smoke140.py
"""

from __future__ import annotations

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
    tmp = Path(tempfile.mkdtemp(prefix="smoke140-"))

    # -- 1: the one-place minting --------------------------------------------- #
    library = (ROOT / "src" / "pharmaml" / "viz" / "report.py").read_text(
        encoding="utf-8"
    )
    # the render join is the minting place: href + title + text, once
    assert (
        "f\"<li><a href='#{page_id}' title='Jump to this page in the report'>\"" in library
    ), "the render join lost the forward tooltip"
    # the exact wording rides the doctest pin too (single-line, the
    # standing doctest lesson)
    assert (
        "<li><a href='#pg-1' title='Jump to this page in the report'>Model leaderboard</a></li>"
        in library
    )
    source_text = (ROOT / "playground.py").read_text(encoding="utf-8")
    assert "title='Jump to this page in the report'" not in source_text, (
        "the playground inlined the forward tooltip markup -- the "
        "one-place minting broke"
    )
    banner("1.  the one-place minting (the render join, no inline twin)")

    # -- 2: the app's own save path ships the forward tooltip ------------------ #
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
    # target is the WRITTEN file, not the dialog flow (shoot140 drives it)
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
    tooltip = "title='Jump to this page in the report'"
    # one figure + one table -> exactly two Contents lines, EACH carrying
    # the forward tooltip
    assert html.count(tooltip) == 2, html.count(tooltip)
    assert (
        "<li><a href='#pg-1' title='Jump to this page in the report'>"
        "table 1 · SMILES inspector</a></li>" in html
    )
    assert (
        "<li><a href='#pg-2' title='Jump to this page in the report'>"
        "figure 1 · alpha plot</a></li>" in html
    )
    # the return leg still confesses on every wrapper (BOTH directions)
    assert (
        html.count("title='Back to the contents block at the top of the report'") == 2
    )
    # the navigation round trip is unchanged
    assert "<nav class='contents' id='contents' tabindex='-1'" in html
    assert "href='http" not in html
    # the saved-file confession reached the reader (the completion dialog)
    saved = [call for call in seen if call[0] == "showinfo" and call[1][0] == "Session report saved"]
    assert saved, [call[1][0] for call in seen]
    app.destroy()
    import gc

    gc.collect()
    plt.close("all")
    banner("2.  the app's save path ships the forward tooltip (every line)")

    # -- 3: the round's docstring pins ------------------------------------------ #
    doc = playground.__doc__ or ""
    assert "100x-140" in doc
    assert "the same POINTER" in doc
    assert "Jump to this page in the report" in doc
    assert "KEYBOARD" in doc
    assert "digits 1/2/3 toggle the census line's legs through the SAME" in doc
    assert '"Reset now" link beside' in doc
    banner("3.  the round's docstring pins")

    print("\nSMOKE140_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
