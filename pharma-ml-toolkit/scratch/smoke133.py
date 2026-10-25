"""Round 100x-133 smoke: the picklist's per-page REORDER (the desktop
twin of the web report card's drag) + the caption feed (the generative
grids' counts reaching the picklist's labels) — headless, Xvfb-driven.

Contracts walked:
  1. the rows ride the WALKING order — the table pages first, then the
     figure pages (both walkers write them that way), with the muted
     walking-order note visible under the header;
  2. the ▲ / ▼ arrows move a page among its OWN kind (the numbering
     rewrites itself, the visual packing order follows, and the return
     carries the dialog's order);
  3. the boundary arrows disable honestly (a one-row group has BOTH
     arrows disabled — a table can never cross into the figure block);
  4. the saved HTML walks its pages the way the dialog ordered them
     (the moved figure's caption appears in its new position);
  5. _render_figure(caption=...) reaches the registry, so the picklist
     offers the count-carrying label.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/smoke133.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")

import matplotlib

matplotlib.use("Agg", force=True)
import playground

OUT = Path("/home/z/my-project/pharma-ml-toolkit/download")


def deadline_wait(app, label: str, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    print(f"TIMEOUT waiting for {label}")
    return False


def figure_static(title: str):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot([0, 1], [0, 1])
    axes.set_title(title)
    return fig


def find_picklist(app):
    for child in list(app.winfo_children()):
        if child.winfo_class() != "Toplevel":
            continue  # Menu/Frame/Notebook carry no title()
        if str(child.title()) == "Session report contents":
            return child
    return None


def widgets_of(app, dialog, cls):
    return [w for w in app._walk_widgets(dialog) if w.winfo_class() == cls]


def visual_check_labels(app, dialog):
    """The checkbox texts in VISUAL (packing) order — the reorder's own
    axis, since creation order stays frozen while the rows move."""
    checks = widgets_of(app, dialog, "TCheckbutton")
    body = checks[0].master.master.master  # check -> row -> group -> body
    labels = []
    for group_frame in body.pack_slaves():
        for child in group_frame.pack_slaves():
            for check in checks:
                if check.master is child:
                    labels.append(str(check.cget("text")))
    return labels


def arrow_buttons(app, dialog):
    ups = [
        w
        for w in widgets_of(app, dialog, "TButton")
        if str(w.cget("text")) == "▲"
    ]
    downs = [
        w
        for w in widgets_of(app, dialog, "TButton")
        if str(w.cget("text")) == "▼"
    ]
    return ups, downs


def drive(app, extra=None, action="save"):
    captured = {}

    def interact():
        dialog = find_picklist(app)
        if dialog is None:
            captured["error"] = "the picklist never opened"
            return
        captured["note"] = [
            str(w.cget("text"))
            for w in widgets_of(app, dialog, "TLabel")
            if "walking order" in str(w.cget("text"))
        ]
        if extra is not None:
            extra(dialog, captured)
        try:
            alive = dialog.winfo_exists()
        except Exception:  # noqa: BLE001 - destroyed mid-callback
            alive = False
        if not alive:
            return
        buttons = {
            str(button.cget("text")): button
            for button in widgets_of(app, dialog, "TButton")
        }
        if action == "save":
            for text, button in buttons.items():
                if text.startswith("Save report"):
                    button.invoke()
                    break
        elif action == "cancel":
            buttons["Cancel"].invoke()

    app.after(150, interact)
    return captured


def seed_session(app):
    app._render_figure(app._topo_host, figure_static("alpha plot"))
    app._render_figure(app._breed_host, figure_static("beta plot"))
    app._session_figures[0][1] = "alpha plot"
    app._session_figures[1][1] = "beta plot"
    app._desc_tree.insert("", "end", values=("MolWt", "180.16"))


def main() -> int:
    app = playground.PlaygroundApp(persist_session=False)
    try:
        target = OUT / "smoke133-reordered.html"
        playground.filedialog.asksaveasfilename = lambda **_k: str(target)
        playground.messagebox.showinfo = lambda *a, **k: None

        # -- 1+2: the walking order + the reorder reach the saved file --- #
        seed_session(app)

        def move_and_save(dialog, bucket):
            bucket["before"] = visual_check_labels(app, dialog)
            _ups, downs = arrow_buttons(app, dialog)
            # creation order: the table's (disabled) arrows, then alpha's,
            # then beta's — alpha's ▼ moves it below beta
            downs[1].invoke()
            app.update()
            bucket["after"] = visual_check_labels(app, dialog)

        captured = drive(app, extra=move_and_save)
        app._menu_save_session_report_html()
        assert captured.get("error") is None, captured
        # the note confesses the walking order
        assert captured["note"], "the walking-order note never rendered"
        assert "table pages" in captured["note"][0]
        # the rows ride the walkers' order, table block first
        assert captured["before"] == [
            "table 1 · SMILES inspector",
            "figure 1 · alpha plot",
            "figure 2 · beta plot",
        ], captured["before"]
        # alpha moved below beta and the numbering followed
        assert captured["after"] == [
            "table 1 · SMILES inspector",
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], captured["after"]
        assert deadline_wait(app, "the reordered report job"), "job hung"
        text = target.read_text(encoding="utf-8")
        assert "alpha plot" in text and "beta plot" in text
        assert text.index("beta plot") < text.index("alpha plot"), (
            "the dialog's order never reached the report"
        )
        print("reorder path: rows in walking order, move + renumber + "
              "the saved file follows OK")

        # -- 3: the boundary arrows disable honestly ---------------------- #
        def boundaries(dialog, bucket):
            _ups, downs = arrow_buttons(app, dialog)
            bucket["table_down_disabled"] = downs[0].instate(("disabled",))
            bucket["alpha_down_enabled"] = not downs[1].instate(("disabled",))
            # a disabled invoke is a no-op — the order stands
            downs[0].invoke()
            app.update()
            bucket["stable"] = visual_check_labels(app, dialog)
            cancel = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")) == "Cancel"
            )
            cancel.invoke()

        found = drive(app, extra=boundaries, action="none")
        app._session_report_picklist(
            app._live_session_figures(), app._live_session_tables()
        )
        assert found.get("error") is None, found
        assert found["table_down_disabled"] is True, found
        assert found["alpha_down_enabled"] is True, found
        assert found["stable"] == [
            "table 1 · SMILES inspector",
            "figure 1 · alpha plot",
            "figure 2 · beta plot",
        ], found["stable"]
        print("boundary path: one-row group's ▼ disabled, invoke a no-op OK")

        # -- 4: the caption feed reaches the picklist's labels ------------ #
        app._render_figure(
            app._topo_host,
            figure_static("the grid"),
            caption="Ring topology moves — 6 mutant(s)",
        )
        seen = {}

        def caption_check(dialog, bucket):
            labels = visual_check_labels(app, dialog)
            seen["rows"] = labels
            cancel = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")) == "Cancel"
            )
            cancel.invoke()

        drive(app, extra=caption_check, action="none")
        app._session_report_picklist(
            app._live_session_figures(), app._live_session_tables()
        )
        assert any("Ring topology moves — 6 mutant(s)" in row for row in seen["rows"]), (
            "the caption never reached the picklist"
        )
        print("caption path: the count-carrying label offered OK")

        # -- 5: the documentation pins ------------------------------------ #
        doc = playground.__doc__ or ""
        assert "100x-133" in doc and "▲ / ▼" in doc and "PREVIEW" in doc
        pick_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-133" in pick_doc and "REORDER" in pick_doc
        render_doc = playground.PlaygroundApp._render_figure.__doc__ or ""
        assert "100x-133" in render_doc and "caption" in render_doc
        print("docstring pins OK")

        import matplotlib.pyplot as plt

        plt.close("all")
        target.unlink()
        print("SMOKE133_OK")
        return 0
    finally:
        app.destroy()
        deadline_wait(app, "final destroy", timeout=30.0)


if __name__ == "__main__":
    sys.exit(main())
