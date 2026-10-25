"""Screenshots + artifacts for the 100x-132 round: the session
reports' pre-save PICKLIST — (1) the dialog with every page offered
(two live figures + one filled table, all checked, the census footer
reading "3 of 3 pages will ship"); (2) the moment after one checkbox
flips (the census recounting to "2 of 3"); (3) a REAL filtered
session-report HTML artifact (the unchecked figure genuinely absent,
the kept pages shipped) saved into download/ for inspection.

The dialog is MODAL (its own wait_window loop), so every interaction
and screenshot rides after-callbacks the loop processes while the
dialog is up.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/shoot132.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

OUT = Path("/home/z/my-project/pharma-ml-toolkit/download")
TARGET = OUT / "pg132-filtered-report.html"
playground.filedialog.asksaveasfilename = lambda **_k: str(TARGET)

app = playground.PlaygroundApp(persist_session=False)
app.geometry("1280x860+10+10")


def pump(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def shot(name: str) -> None:
    from PIL import ImageGrab  # type: ignore

    for _ in range(12):
        app.update()
        time.sleep(0.01)
    img = ImageGrab.grab(xdisplay=":99")
    img.save(f"{OUT}/{name}.png")
    print("shot", name)


def widgets_of(dialog, cls):
    return [w for w in app._walk_widgets(dialog) if w.winfo_class() == cls]


try:
    pump()
    # seed real content: two figures + one filled table
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def figure_static(title):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    app._render_figure(app._topo_host, figure_static("alpha plot"))
    app._render_figure(app._breed_host, figure_static("beta plot"))
    app._session_figures[0][1] = "alpha plot"
    app._session_figures[1][1] = "beta plot"
    app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
    pump(timeout=30.0)

    # the interactions + shots ride after-callbacks (the picklist's
    # wait_window loop processes them while the dialog is up)
    state = {}

    def stage_one():
        dialog = next(
            child
            for child in app.winfo_children()
            if child.winfo_class() == "Toplevel"
            and str(child.title()) == "Session report contents"
        )
        state["dialog"] = dialog
        shot("pg132-picklist-all-on")

        def stage_two():
            checks = widgets_of(state["dialog"], "TCheckbutton")
            checks[0].invoke()  # "figure 1 · alpha plot" off -> 2 of 3
            app.update()
            shot("pg132-picklist-filtered")

            def stage_three():
                save = next(
                    w
                    for w in widgets_of(state["dialog"], "TButton")
                    if str(w.cget("text")).startswith("Save report")
                )
                save.invoke()

            app.after(500, stage_three)

        app.after(700, stage_two)

    app.after(500, stage_one)
    app._menu_save_session_report_html()
    assert pump(), "the filtered report job hung"
    assert TARGET.is_file(), "the filtered report never landed"
    text = TARGET.read_text(encoding="utf-8")
    assert "beta plot" in text and "alpha plot" not in text
    assert "SMILES inspector table" in text
    print("artifact pg132-filtered-report.html written (alpha filtered out)")

    # close the static figures and clear the seeded table
    plt.close("all")
    app._desc_tree.delete(*app._desc_tree.get_children())

    print("SHOOT132_OK")
finally:
    app.destroy()
    pump(timeout=30.0)
