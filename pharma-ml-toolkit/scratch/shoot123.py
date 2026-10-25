"""Screenshots for the 100x-123 round: surgery structure grids, the
report's table axis, and the report PDF artifact."""
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


def scroll_to_card(title: str) -> None:
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


# 1) the ring topology card: enumerate, then DRAW the mutant grid
app._jump_to_tab(7)
app._gen_topology_moves()
assert pump()
app._gen_draw_topology()
assert pump()
scroll_to_card("Ring topology moves")
shot("pg123-topology-grid")

# 2) the breed card: breed, then DRAW the litter (parents + children)
app._gen_breed()
assert pump()
app._gen_draw_breed()
assert pump()
scroll_to_card("Breed two molecules")
shot("pg123-breed-grid")

# 3) the report's table axis: a real analyse fills the descriptor table,
#    the surgery grids stay live as figures -> BOTH axes in one PDF
import tkinter.filedialog as fd

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

app._analyse_smiles()
assert pump()
assert app._live_session_tables(), "the descriptor table did not fill"
assert len(app._live_session_figures()) >= 2, "the surgery grids are not live"
tmp = Path(tempfile.mkdtemp(prefix="shoot123_"))
target = tmp / "session-report.pdf"
fd.asksaveasfilename = lambda **_k: str(target)
app._menu_save_session_report()
assert pump()
assert target.exists() and target.stat().st_size > 1_000
print("pdf bytes:", target.stat().st_size)

import pypdf

reader = pypdf.PdfReader(str(target))
joined = "\n".join((page.extract_text() or "").lower() for page in reader.pages)
assert "table page(s)" in joined and "smiles inspector table" in joined
assert "molwt" in joined and "more rows" in joined
print("report pages:", len(reader.pages), "(figures + real text tables)")

# keep the artifact for inspection
import shutil

shutil.copy(target, str(Path(__file__).resolve().parents[2] / "download") + "/pg123-session-report.pdf")
plt.close("all")

# 4) the SMILES inspector card full of walked rows (the table axis source)
app._jump_to_tab(1)


def scroll_chem(title: str) -> None:
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
            pass
    assert shell is not None, title
    app.update()
    body_y = shell.winfo_rooty() - page.body.winfo_rooty()
    total = page._canvas.bbox("all")[3] or 1
    page._canvas.yview_moveto(max(0, body_y - 16) / total)
    app.update()


scroll_chem("SMILES inspector")
shot("pg123-descriptor-table")

app.destroy()
print("SHOOT123_OK")
