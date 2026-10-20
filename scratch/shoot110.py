import sys

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
import playground as pg

OUT = "/home/z/my-project/download"

app = pg.PlaygroundApp()
for _ in range(6):
    app.update_idletasks()
    app.update()


def shot(tab: int, name: str, theme: str = "light") -> None:
    app._theme_var.set(theme)
    app._on_theme_change()
    app._book.select(tab)
    for _ in range(8):
        app.update_idletasks()
        app.update()
    try:
        from PIL import ImageGrab

        img = ImageGrab.grab(xdisplay=":99")
        img.save(f"{OUT}/{name}")
        print("saved", name, img.size)
    except Exception as exc:  # noqa: BLE001
        print("grab failed:", exc)


# the new Utilities tab (light + dark)
shot(8, "utilities-tab-r110.png", "light")
shot(8, "utilities-tab-dark-r110.png", "dark")
# the Chemistry converters + similarity scan cards
shot(1, "chem-converters-r110.png", "light")
# the Docking audit + vina batch + single pdbqt cards
shot(4, "docking-audit-r110.png", "light")
app._on_close()
