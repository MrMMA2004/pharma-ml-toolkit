"""Screenshots for the 100x-120 round: finder, learning curve, MAD, parse."""
import sys
import time

sys.path.insert(0, ".")
import playground

app = playground.PlaygroundApp()
app.geometry("1280x860+10+10")


def pump(timeout=300.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def shot(name):
    app.update()
    app.update_idletasks()
    app.after(240, lambda: None)
    for _ in range(24):
        app.update()
        time.sleep(0.01)
    from PIL import ImageGrab  # type: ignore

    img = ImageGrab.grab(xdisplay=":99")
    img.save(f"{Path(__file__).resolve().parents[2] / 'download'} / {name}.png")
    print("shot", name)


# 1) the card finder open with a query typed
app._find_card_dialog()
import tkinter as _tk

win = next(c for c in app.winfo_children()
            if isinstance(c, _tk.Toplevel) and c.title() == "Find a card")
entry = next(c for c in win.winfo_children()
             if c.winfo_class() in ("Entry", "TEntry"))
entry.insert(0, "learning")
entry.event_generate("<KeyRelease>")
app.update()
shot("pg120-finder")

# 2) jump to the ML card + flash (close dialog via Escape simulation)
tree = next(c for c in win.winfo_children()
             if c.winfo_class() == "Treeview")
rows = tree.get_children()
tree.selection_set(rows[0])
app._jump_to_card(5, "Results")
win.destroy()
shot("pg120-jump-flash")

# 3) learning curve after a train
app._ml_load_example()
app._model_combo.set("ridge")
app._ml_refresh_hyperparam_panel()
app._ml_train()
assert pump()
app._ml_plot_learning_curve()
assert pump()
shot("pg120-learning-curve")

# 4) MAD ranking
app._jump_to_tab(3)
app._bio_load_expr_example()
app._bio_rank_variable_genes()
assert pump()
shot("pg120-mad-genes")

# 5) Vina parse
app._jump_to_tab(4)
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="shoot120_"))
conf = tmp / "batch.conf"
conf.write_text(
    "receptor = r.pdbqt\ncenter_x = 2.5\ncenter_y = 3.5\ncenter_z = 4.5\n"
    "size_x = 20\nsize_y = 22\nsize_z = 24\nexhaustiveness = 12\n",
    encoding="utf-8")
app._dock_vbatch_conf = conf
app._dock_vbatch_parse()
shot("pg120-vina-parse")

app.destroy()
print("SHOTS_DONE")
