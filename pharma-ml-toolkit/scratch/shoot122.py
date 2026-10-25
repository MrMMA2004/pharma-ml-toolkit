"""Screenshots for the 100x-122 round: objectives, surgery, session report."""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground

# the report's success dialog is modal -- silence it for the headless run
playground.messagebox.showinfo = lambda *a, **k: None

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


# 1) the Generative tab: the reference row swapped in for the SMARTS objective
app._jump_to_tab(7)
app._objective_combo.set("Contains substructure (SMARTS)")
app._objective_hint()
shot("pg122-smarts-objective")

# 2) the ring topology moves card after an enumeration (scrolled to it)
def scroll_to_card(title):
    page = app._book.nametowidget(app._book.select())
    shell = None
    for card_title, _subtitle, candidate in playground._CARD_INDEX:
        if card_title != title:
            continue
        try:
            if candidate.winfo_exists():
                shell = candidate
                break
        except Exception:  # noqa: BLE001, S110, S112 - destroyed shell
            pass  # an earlier app's shell is gone; keep looking
    assert shell is not None, title
    app.update()
    body_y = shell.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - 16) / total)
    app.update()


app._gen_topology_moves()
assert pump()
scroll_to_card("Ring topology moves")
shot("pg122-ring-moves")

# 3) the breed card with its litter
app._gen_breed()
assert pump()
scroll_to_card("Breed two molecules")
shot("pg122-breed")

# 4) a session report written from two live figures
import tkinter.filedialog as fd

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

fig_a = plt.figure()
ax = fig_a.add_subplot(111)
ax.plot([0, 1], [0, 1])
ax.set_title("alpha")
fig_b = plt.figure()
ax = fig_b.add_subplot(111)
ax.plot([1, 0], [0, 1])
ax.set_title("beta")
app._job_label_base = "figure alpha"
app._render_figure(app._ml_host, fig_a)
app._job_label_base = "figure beta"
app._render_figure(app._feat_host, fig_b)
tmp = Path(tempfile.mkdtemp(prefix="shoot122_"))
target = tmp / "session-report.pdf"
fd.asksaveasfilename = lambda **_k: str(target)
app._menu_save_session_report()
assert pump()
assert target.exists() and target.stat().st_size > 1_000
print("pdf bytes:", target.stat().st_size)
plt.close("all")
app._jump_to_tab(5)
shot("pg122-two-figures")
# keep the artifact for inspection
import shutil

shutil.copy(target, str(Path(__file__).resolve().parents[2] / "download") + "/pg122-session-report.pdf")
shutil.rmtree(tmp, ignore_errors=True)

app.destroy()
print("SHOTS_DONE")
