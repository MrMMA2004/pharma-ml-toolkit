"""Round 100x-134 smoke: the picklist's ORDER MEMORY (the web twin's
persisted walking order, mirrored per session on the desktop) + the
count-prefix page names (both report twins' "figure N · <caption>" /
"table N · <caption>") — headless, Xvfb-driven.

Contracts walked:
  1. a completed Save stores the dialog's per-kind order
     (``_report_page_order``), and the NEXT picklist opens pre-arranged
     in it — with the memory confession rendered and the Reset button
     enabled (a remembered order IS a drift);
  2. stale labels are skipped, NEW pages ride after the remembered ones
     in walk order;
  3. Cancel never touches the memory;
  4. "Reset to walking order" restores the walk order AND forgets the
     memory (the next dialog opens in the walk order too), and it
     disables itself at the walk order — enabled exactly while drifted
     (a manual ▲ / ▼ move drifts it, a reset un-drifts it);
  5. the saved HTML's captions/headings carry the count prefix
     ("figure 1 · ..." / "table 1 · ..."), in the dialog's order.

Run: DISPLAY=:99 python scratch/smoke134.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg", force=True)
import playground

OUT = Path(__file__).resolve().parents[1] / "download"


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


def button_of(app, dialog, text):
    return next(
        w
        for w in widgets_of(app, dialog, "TButton")
        if str(w.cget("text")) == text
    )


def drive(app, extra=None, action="save"):
    captured = {}

    def interact():
        dialog = find_picklist(app)
        if dialog is None:
            captured["error"] = "the picklist never opened"
            return
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
        target = OUT / "pg134-remembered-order.html"
        playground.filedialog.asksaveasfilename = lambda **_k: str(target)
        playground.messagebox.showinfo = lambda *a, **k: None
        seed_session(app)
        assert app._report_page_order == {}

        # -- 1: a Save stores the order; the next open applies it -------- #
        def move_and_save(dialog, bucket):
            _ups, downs = arrow_buttons(app, dialog)
            downs[1].invoke()  # alpha below beta (creation: table, alpha, beta)
            app.update()
            bucket["labels"] = visual_check_labels(app, dialog)

        first = drive(app, extra=move_and_save)
        app._menu_save_session_report_html()
        assert first.get("error") is None, first
        assert deadline_wait(app, "the first report job"), "job hung"
        # the memory now carries the dialog's order (ALL labels)
        assert app._report_page_order == {
            "table": ["SMILES inspector"],
            "figure": ["beta plot", "alpha plot"],
        }, app._report_page_order
        text = target.read_text(encoding="utf-8")
        # the count prefix reached the captions, in the dialog's order
        assert "figure 1 · beta plot" in text, "the count prefix never shipped"
        assert "figure 2 · alpha plot" in text, "the dialog order never shipped"
        assert "table 1 · SMILES inspector" in text
        print("memory store + count-prefix captions: OK")

        # -- 1b: the next open is pre-arranged, confession rendered ------- #
        def observe_and_cancel(dialog, bucket):
            bucket["labels"] = visual_check_labels(app, dialog)
            memory = next(
                w
                for w in widgets_of(app, dialog, "TLabel")
                if "last saved order" in str(w.cget("text"))
            )
            bucket["confession_packed"] = bool(memory.winfo_manager())
            bucket["reset_enabled"] = not button_of(
                app, dialog, "Reset to walking order"
            ).instate(("disabled",))

        second = drive(app, extra=observe_and_cancel, action="cancel")
        app._session_report_picklist(
            app._live_session_figures(), app._live_session_tables()
        )
        assert second.get("error") is None, second
        assert second["labels"] == [
            "table 1 · SMILES inspector",
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], second["labels"]
        assert second["confession_packed"] and second["reset_enabled"], second
        # Cancel left the memory untouched
        assert app._report_page_order["figure"] == ["beta plot", "alpha plot"]
        print("memory apply + confession + cancel-keeps-memory: OK")

        # -- 2: stale skipped, new appended -------------------------------- #
        app._report_page_order["figure"] = ["gone plot", "beta plot"]
        app._render_figure(
            app._topo_host, figure_static("gamma plot")
        )
        app._session_figures[0][1] = "gamma plot"

        def observe_stale(dialog, bucket):
            bucket["labels"] = visual_check_labels(app, dialog)

        stale = drive(app, extra=observe_stale, action="cancel")
        app._session_report_picklist(
            app._live_session_figures(), app._live_session_tables()
        )
        assert stale.get("error") is None, stale
        assert stale["labels"] == [
            "table 1 · SMILES inspector",
            "figure 1 · beta plot",
            "figure 2 · gamma plot",
        ], stale["labels"]
        print("stale-skip + new-append: OK")

        # -- 4: Reset restores the walk order AND forgets ----------------- #
        def reset_and_observe(dialog, bucket):
            reset = button_of(app, dialog, "Reset to walking order")
            bucket["before"] = visual_check_labels(app, dialog)
            bucket["reset_enabled_before"] = not reset.instate(("disabled",))
            reset.invoke()
            app.update()
            bucket["after"] = visual_check_labels(app, dialog)
            bucket["reset_disabled_after"] = reset.instate(("disabled",))
            # a manual move drifts it again — Reset re-enables.  The
            # arrow buttons are creation-frozen (the display order when
            # the rows were built: table, beta, gamma), so gamma's ▼ is
            # downs[2] — enabled now that gamma rides first
            _ups, downs = arrow_buttons(app, dialog)
            downs[2].invoke()  # gamma below beta -- away from walk order
            app.update()
            bucket["reset_enabled_after_move"] = not reset.instate(("disabled",))

        reset_visit = drive(app, extra=reset_and_observe, action="cancel")
        app._session_report_picklist(
            app._live_session_figures(), app._live_session_tables()
        )
        assert reset_visit.get("error") is None, reset_visit
        assert reset_visit["before"] == [
            "table 1 · SMILES inspector",
            "figure 1 · beta plot",
            "figure 2 · gamma plot",
        ], reset_visit["before"]
        assert reset_visit["after"] == [
            "table 1 · SMILES inspector",
            "figure 1 · gamma plot",  # gamma replaced alpha's host
            "figure 2 · beta plot",
        ], reset_visit["after"]
        assert reset_visit["reset_disabled_after"] is True
        assert reset_visit["reset_enabled_after_move"] is True
        assert app._report_page_order == {}, app._report_page_order
        print("reset: walk order restored, memory forgotten, drift-honest: OK")

        # -- docstring pins ------------------------------------------------- #
        doc = playground.__doc__ or ""
        assert "100x-134" in doc and "REMEMBERS" in doc and "hover INSPECT" in doc
        pick_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-134" in pick_doc and "order REMEMBERS" in pick_doc
        print("docstring pins: OK")

        print("SMOKE134_OK")
        return 0
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001 - already gone
            pass
        import matplotlib.pyplot as plt

        plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
