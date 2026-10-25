"""Screenshots + artifacts for the 100x-127 round: the two molecule
surgery cards and the evolution Results card with their NEW per-card
"Save figure..." buttons, each over a real drawn grid (the buttons ride
the cards' own button rows; the exports exercise the real save path).

Run: DISPLAY=:99 python scratch/shoot127.py
"""
import sys
import time

sys.path.insert(0, ".")
import playground

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

app = playground.PlaygroundApp()
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
    app.update()
    app.update_idletasks()
    for _ in range(24):
        app.update()
        time.sleep(0.01)
    from PIL import ImageGrab  # type: ignore

    img = ImageGrab.grab(xdisplay=":99")
    img.save(f"{Path(__file__).resolve().parents[2] / 'download'} / {name}.png")
    print("shot", name)


def scroll_to_widget(widget) -> None:
    """Scroll a deep widget (e.g. a figure host) into the viewport."""
    page = app._book.nametowidget(app._book.select())
    app.update()
    body_y = widget.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - 24) / total)
    app.update()


def tab_buttons() -> list[str]:
    return [
        str(w.cget("text"))
        for w in app._walk_widgets(app._book.nametowidget(app._book.tabs()[7]))
        if w.winfo_class() == "TButton"
    ]


app._jump_to_tab(7)

# 1) the Ring topology moves card: enumerate + draw the mutants, then
# the card with its own Save figure… button and the live grid
app._topo_smiles.delete(0, "end")
app._topo_smiles.insert(0, "C1CCNCC1")
app._gen_topology_moves()
assert pump(), "topology hung"
app._gen_draw_topology()
assert pump(), "draw-topology hung"
topo_grid = getattr(app._topo_host, "_pg_figure", None)
assert topo_grid is not None, "the mutant grid never landed"
assert tab_buttons().count("Save figure…") >= 2, "surgery save buttons missing"
scroll_to_widget(app._topo_host)
shot("pg127-topo-save-button")

# 2) exercise the topology card's export for real, then verify the PNG
captured = {}
import pharmaml.viz as viz_module

real_save = viz_module.save_figure


def spy(figure, path, *args, **kwargs):
    captured["figure"] = figure
    return real_save(figure, path, *args, **kwargs)


viz_module.save_figure = spy
try:
    playground.filedialog.asksaveasfilename = lambda **_k: (
        str(Path(__file__).resolve().parents[2] / "download") + "/pg127-topo-export.png"
    )
    app._save_topo_figure()
    assert pump(), "topo save hung"
    assert captured["figure"] is topo_grid, "saved a figure that is not the card's"
finally:
    viz_module.save_figure = real_save
with open(str(Path(__file__).resolve().parents[2] / "download") + "/pg127-topo-export.png", "rb") as handle:
    assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
print("exported the topology card's live grid via its own button")

# 3) the Breed card: breed + draw the litter, then the card with its
# button and the parents-and-children grid
app._gen_breed()
assert pump(), "breed hung"
app._gen_draw_breed()
assert pump(), "draw-litter hung"
breed_grid = getattr(app._breed_host, "_pg_figure", None)
assert breed_grid is not None, "the litter grid never landed"
scroll_to_widget(app._breed_host)
shot("pg127-breed-save-button")

# 4) the Results card: a real evolution, then Draw top molecules, then
# the card with the newest-of-two-hosts save button
app._objective_combo.set("QED + LogP blend")
app._pop_size.set("12")
app._generations.set("4")
app._run_evolution()
assert pump(), "evolution hung"
app._gen_draw_top()
assert pump(), "draw-top hung"
top_grid = getattr(app._ga_grid_host, "_pg_figure", None)
assert top_grid is not None, "the leaderboard grid never landed"
assert tab_buttons().count("Save figure…") == 4, "four save buttons expected"
scroll_to_widget(app._ga_grid_host)
shot("pg127-results-save-button")

# 5) exercise the Results card's two-host export (the leaderboard grid
# drew last on the card) for real, then verify the PNG
viz_module.save_figure = spy
try:
    playground.filedialog.asksaveasfilename = lambda **_k: (
        str(Path(__file__).resolve().parents[2] / "download") + "/pg127-results-export.png"
    )
    app._save_ga_figure()
    assert pump(), "results save hung"
    assert captured["figure"] is top_grid, "saved a figure that is not the card's"
finally:
    viz_module.save_figure = real_save
with open(str(Path(__file__).resolve().parents[2] / "download") + "/pg127-results-export.png", "rb") as handle:
    assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
print("exported the Results card's leaderboard grid via its own button")

app.destroy()
print("SHOOT127_OK")
