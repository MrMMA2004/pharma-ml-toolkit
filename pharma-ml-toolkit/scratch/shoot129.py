"""Screenshots + artifacts for the 100x-129 round: the PDF session
report's dark-ink variant riding the theme radios -- the DARK app
window with a live IV figure (the palette the PDF matches), the
save-confirmation moment, and the round's real artifacts: the
dark-ink PDF and the classic light PDF written by the menu itself.

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/shoot129.py
"""
import shutil
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
    img.save(f"/home/z/my-project/pharma-ml-toolkit/download/{name}.png")
    print("shot", name)


def save_report(target: str) -> None:
    playground.filedialog.asksaveasfilename = lambda **_k: target
    app._menu_save_session_report()
    assert pump(), "the session report job hung"


# -- 1. the DARK app with a live IV figure (the PDF's own palette) ----- #
app._theme_var.set("dark")
app._on_theme_change()
# land on the PK/PD tab and draw a real decay curve
app._book.select(app._book.tabs()[2])
app._iv_c0.delete(0, "end"); app._iv_c0.insert(0, "12")
app._iv_thalf.delete(0, "end"); app._iv_thalf.insert(0, "6")
app._iv_hours.delete(0, "end"); app._iv_hours.insert(0, "24")
app._plot_iv_bolus()
assert pump(), "the IV profile job hung"
iv_host = app._iv_host
app.update()
body_y = iv_host.winfo_rooty() - app._book.nametowidget(app._book.select()).body.winfo_rooty()
page = app._book.nametowidget(app._book.select())
total = page._canvas.bbox("all")[3] or 1
page._canvas.yview_moveto(max(0, body_y - 120) / total)
shot("pg129-dark-app-iv")
print("1. dark app + live IV decay curve OK")

# -- 2. the round's real artifacts: dark-ink + classic PDFs ------------ #
app._desc_tree.delete(*app._desc_tree.get_children())
app._desc_tree.insert("", "end", values=("MolWt", "194.19"))
save_report("/tmp/shoot129-dark.pdf")
save_report_light_target = "/tmp/shoot129-light.pdf"
app._theme_var.set("light")
app._on_theme_change()
save_report(save_report_light_target)
for src, dst in (
    ("/tmp/shoot129-dark.pdf", "pg129-session-dark.pdf"),
    (save_report_light_target, "pg129-session-light.pdf"),
):
    shutil.copyfile(src, f"/home/z/my-project/pharma-ml-toolkit/download/{dst}")
    print("artifact", dst)
with open("/home/z/my-project/pharma-ml-toolkit/download/pg129-session-dark.pdf", "rb") as fh:
    head = fh.read(8)
assert head.startswith(b"%PDF-"), head
print("2. both PDF artifacts written by the menu itself OK")

# -- 3. the light app, same card (the classic paper's palette) --------- #
shot("pg129-light-app-iv")
print("3. light app + the same live curve OK")

app.destroy()
print("SHOOT129_OK")
