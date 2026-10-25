"""Screenshots + artifacts for the 100x-134 round: the picklist's ORDER
MEMORY — (1) the first Save's dialog with a manual move (the memory
about to be written); (2) the NEXT dialog opening pre-arranged in the
remembered order — the confession line under the header, the Reset
button enabled (a remembered order IS a drift); (3) the moment after
Reset (the walk order restored, the confession collapsed, Reset
disabled again); plus a REAL remembered-order session-report HTML
artifact saved into download/ with the round's count-prefixed captions
("figure 1 · ..." / "table 1 · ...").

The dialog is MODAL (its own wait_window loop), so every interaction
and screenshot rides after-callbacks the loop processes while the
dialog is up.

Run: DISPLAY=:99 python scratch/shoot134.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

OUT = Path(__file__).resolve().parents[1] / "download"
TARGET = OUT / "pg134-remembered-report.html"
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
    checks = widgets_of(dialog, "TCheckbutton")
    body = checks[0].master.master.master
    labels = []
    for group_frame in body.pack_slaves():
        for child in group_frame.pack_slaves():
            for check in checks:
                if check.master is child:
                    labels.append(str(check.cget("text")))
    return labels


def find_picklist():
    return next(
        child
        for child in app.winfo_children()
        if child.winfo_class() == "Toplevel"
        and str(child.title()) == "Session report contents"
    )


state = {}

try:
    pump()
    # seed real content: two figures (one count-captioned the way the
    # surgery grids are) + one filled table
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
    assert app._report_page_order == {}

    # -- the FIRST visit: a manual move + Save (the memory moment) ----- #
    def stage_one():
        dialog = find_picklist()
        state["dialog"] = dialog
        state["first_before"] = visual_check_labels(dialog)
        downs = [
            w
            for w in widgets_of(dialog, "TButton")
            if str(w.cget("text")) == "▼"
        ]
        # creation order: the table's (disabled) ▼, then the mutants'
        # — move the mutants below beta before saving
        downs[1].invoke()
        app.update()
        state["first_after"] = visual_check_labels(dialog)
        shot("pg134-picklist-first-save")

        def stage_two():
            save = next(
                w
                for w in widgets_of(state["dialog"], "TButton")
                if str(w.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(600, stage_two)

    app.after(500, stage_one)
    app._menu_save_session_report_html()
    assert pump(), "the remembered report job hung"
    assert TARGET.is_file(), "the remembered report never landed"
    # the memory carries the moved order
    assert app._report_page_order == {
        "table": ["SMILES inspector"],
        "figure": ["beta plot", "Ring topology moves — 6 mutant(s)"],
    }, app._report_page_order
    assert state["first_before"] == [
        "table 1 · SMILES inspector",
        "figure 1 · Ring topology moves — 6 mutant(s)",
        "figure 2 · beta plot",
    ], state["first_before"]
    assert state["first_after"] == [
        "table 1 · SMILES inspector",
        "figure 1 · beta plot",
        "figure 2 · Ring topology moves — 6 mutant(s)",
    ], state["first_after"]
    text = TARGET.read_text(encoding="utf-8")
    assert "figure 1 · beta plot" in text, "the count prefix never shipped"
    assert "figure 2 · Ring topology moves — 6 mutant(s)" in text
    assert "table 1 · SMILES inspector" in text
    print("artifact pg134-remembered-report.html written (count-prefixed)")

    # -- the SECOND visit: the memory applied + the Reset ---------------- #
    def stage_three():
        dialog = find_picklist()
        state["second_open"] = visual_check_labels(dialog)
        shot("pg134-picklist-remembered")

        def stage_four():
            reset = next(
                w
                for w in widgets_of(state["dialog2"], "TButton")
                if str(w.cget("text")) == "Reset to walking order"
            )
            state["reset_enabled_before"] = not reset.instate(("disabled",))
            reset.invoke()
            app.update()
            state["second_reset"] = visual_check_labels(state["dialog2"])
            shot("pg134-picklist-reset")

            def stage_five():
                cancel = next(
                    w
                    for w in widgets_of(state["dialog2"], "TButton")
                    if str(w.cget("text")) == "Cancel"
                )
                cancel.invoke()

            app.after(500, stage_five)

        state["dialog2"] = dialog
        app.after(700, stage_four)

    app.after(500, stage_three)
    app._session_report_picklist(
        app._live_session_figures(), app._live_session_tables()
    )
    # the second dialog opened in the remembered order -- no moves made
    assert state["second_open"] == [
        "table 1 · SMILES inspector",
        "figure 1 · beta plot",
        "figure 2 · Ring topology moves — 6 mutant(s)",
    ], state["second_open"]
    assert state["reset_enabled_before"] is True, (
        "a remembered order IS a drift -- Reset should be enabled"
    )
    # the reset restored the walk order and forgot the memory
    assert state["second_reset"] == [
        "table 1 · SMILES inspector",
        "figure 1 · Ring topology moves — 6 mutant(s)",
        "figure 2 · beta plot",
    ], state["second_reset"]
    assert app._report_page_order == {}, app._report_page_order
    print("remembered-order dialog + reset shots taken")
finally:
    app.destroy()
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            app.update()
        except Exception:  # noqa: BLE001 - torn down
            break
    import matplotlib.pyplot as plt

    plt.close("all")
print("SHOOT134_OK")
