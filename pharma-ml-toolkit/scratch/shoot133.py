"""Screenshots + artifacts for the 100x-133 round: the picklist's
per-page REORDER — (1) the dialog with every page offered in the
WALKING order (the table block first, the walking-order note under the
header, all checked, the census reading "3 of 3 pages will ship");
(2) the moment after the ▼ move (alpha below beta, the numbering
rewritten — "figure 1 · beta plot" now rides above "figure 2 · alpha
plot"); (3) a REAL reordered session-report HTML artifact (beta's page
genuinely first) saved into download/ for inspection.

The dialog is MODAL (its own wait_window loop), so every interaction
and screenshot rides after-callbacks the loop processes while the
dialog is up.

Run: DISPLAY=:99 python scratch/shoot133.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

OUT = Path(__file__).resolve().parents[1] / "download"
TARGET = OUT / "pg133-reordered-report.html"
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


def visual_check_labels(dialog):
    """The checkbox texts in VISUAL (packing) order — the reorder's axis."""
    checks = widgets_of(dialog, "TCheckbutton")
    body = checks[0].master.master.master
    labels = []
    for group_frame in body.pack_slaves():
        for child in group_frame.pack_slaves():
            for check in checks:
                if check.master is child:
                    labels.append(str(check.cget("text")))
    return labels


try:
    pump()
    # seed real content: two figures + one filled table; the FIRST
    # figure carries a count caption the way the surgery grids do
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def figure_static(title):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    app._render_figure(
        app._topo_host,
        figure_static("the grid"),
        caption="Ring topology moves — 6 mutant(s)",
    )
    app._render_figure(
        app._breed_host, figure_static("beta plot"), caption="beta plot"
    )
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
        state["before"] = visual_check_labels(dialog)
        shot("pg133-picklist-walking-order")

        def stage_two():
            downs = [
                w
                for w in widgets_of(state["dialog"], "TButton")
                if str(w.cget("text")) == "▼"
            ]
            # creation order: the table's (disabled) ▼, then the
            # count-captioned figure's — move the mutants below beta
            downs[1].invoke()
            app.update()
            state["after"] = visual_check_labels(state["dialog"])
            shot("pg133-picklist-reordered")

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
    assert pump(), "the reordered report job hung"
    assert TARGET.is_file(), "the reordered report never landed"
    # the dialog offered the pages in the walking order, the caption
    # label among them
    assert state["before"] == [
        "table 1 · SMILES inspector",
        "figure 1 · Ring topology moves — 6 mutant(s)",
        "figure 2 · beta plot",
    ], state["before"]
    # the move renumbered and the packing order followed
    assert state["after"] == [
        "table 1 · SMILES inspector",
        "figure 1 · beta plot",
        "figure 2 · Ring topology moves — 6 mutant(s)",
    ], state["after"]
    text = TARGET.read_text(encoding="utf-8")
    assert "beta plot" in text and "Ring topology moves" in text
    assert text.index("beta plot") < text.index("Ring topology moves"), (
        "the dialog's order never reached the artifact"
    )
    assert "SMILES inspector table" in text
    print("artifact pg133-reordered-report.html written (beta moved first)")
finally:
    app.destroy()
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            app.update()
        except Exception:  # noqa: BLE001 - torn down
            break
print("SHOOT133_OK")
