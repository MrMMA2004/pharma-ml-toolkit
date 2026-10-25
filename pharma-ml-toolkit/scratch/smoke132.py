"""Round 100x-132 smoke: the session reports' PRE-SAVE PICKLIST (the
desktop twin of the web report card's per-section control) — every
page the walk found is offered with a checkbox (all on by default),
the census footer recounts on every flip, Save disables at zero pages
(and a programmatic invoke of the disabled button stays a no-op — the
guarded confirm), Cancel aborts with no save started, and only the
kept pages walk into the HTML -- headless, Xvfb-driven.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/smoke132.py
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


def drive(app, flips=(), action="save", extra=None):
    """Schedule the modal interaction; the picklist blocks inside its
    own wait_window loop, which processes the after callback."""
    captured = {}

    def interact():
        dialog = find_picklist(app)
        if dialog is None:
            captured["error"] = "the picklist never opened"
            return
        checks = widgets_of(app, dialog, "TCheckbutton")
        captured["checks"] = [str(c.cget("text")) for c in checks]
        for index in flips:
            checks[index].invoke()
        app.update()
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
        import tkinter as tkmod  # noqa: F401 - asserted via winfo_class

        target = OUT / "smoke132-never.html"
        playground.filedialog.asksaveasfilename = lambda **_k: str(target)

        # -- every page offered, numbered by axis; cancel aborts -------- #
        seed_session(app)
        captured = drive(app, action="cancel")
        app._menu_save_session_report_html()
        assert captured.get("error") is None, captured
        assert captured["checks"] == [
            "figure 1 · alpha plot",
            "figure 2 · beta plot",
            "table 1 · SMILES inspector",
        ], captured["checks"]
        assert not target.is_file() and not app._busy
        print("cancel path: every page offered, no save started OK")

        # -- the unchecked page never ships -------------------------------- #
        target2 = OUT / "smoke132-filtered.html"
        playground.filedialog.asksaveasfilename = lambda **_k: str(target2)
        playground.messagebox.showinfo = lambda *a, **k: None
        seed_session(app)
        drive(app, flips=(0,))
        app._menu_save_session_report_html()
        assert deadline_wait(app, "the filtered report job"), "job hung"
        text = target2.read_text(encoding="utf-8")
        assert "beta plot" in text and "alpha plot" not in text
        assert "SMILES inspector table" in text
        assert "<!DOCTYPE html>" in text and "data:image" in text
        print("filter path: unchecked figure gone, kept pages shipped OK")

        # -- the census recounts + Save disables at zero (guarded) ------- #
        found = {}

        def extra(dialog, bucket):
            checks = widgets_of(app, dialog, "TCheckbutton")
            save = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            bucket["enabled"] = not save.instate(("disabled",))
            checks[0].invoke()
            app.update()
            labels = [str(w.cget("text")) for w in widgets_of(app, dialog, "TLabel")]
            bucket["two_of_three"] = any(
                "2 of 3 pages will ship" in label for label in labels
            )
            checks[1].invoke()
            checks[2].invoke()
            app.update()
            labels = [str(w.cget("text")) for w in widgets_of(app, dialog, "TLabel")]
            bucket["nothing"] = any(
                "Nothing would ship" in label for label in labels
            )
            bucket["disabled"] = bool(save.instate(("disabled",)))
            save.invoke()  # the guarded confirm: a no-op at zero pages
            app.update()
            bucket["still_open"] = bool(dialog.winfo_exists())
            cancel = next(
                w
                for w in widgets_of(app, dialog, "TButton")
                if str(w.cget("text")) == "Cancel"
            )
            cancel.invoke()
            found.update(bucket)

        seed_session(app)
        drive(app, action="none", extra=extra)
        app._menu_save_session_report_html()
        assert found["enabled"] is True
        assert found["two_of_three"] is True
        assert found["nothing"] is True
        assert found["disabled"] is True
        assert found["still_open"] is True
        assert not app._busy
        print("census path: recounts live, Save disables + guards at zero OK")

        # -- both menus consult it (source wiring) ------------------------ #
        import inspect

        pdf_src = inspect.getsource(
            playground.PlaygroundApp._menu_save_session_report
        )
        html_src = inspect.getsource(
            playground.PlaygroundApp._menu_save_session_report_html
        )
        assert "_session_report_picklist" in pdf_src
        assert "_session_report_picklist" in html_src
        print("wiring: both report menus consult the picklist OK")

        # -- the documentation pins ---------------------------------------- #
        doc = playground.__doc__ or ""
        assert "100x-132" in doc and "pre-save picklist" in doc
        pick_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-132" in pick_doc and "census footer" in pick_doc
        print("docstring pins OK")

        import matplotlib.pyplot as plt

        plt.close("all")
        target2.unlink()
        print("SMOKE132_OK")
        return 0
    finally:
        app.destroy()
        deadline_wait(app, "final destroy", timeout=30.0)


if __name__ == "__main__":
    sys.exit(main())
