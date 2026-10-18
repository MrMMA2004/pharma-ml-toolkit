import sys
from pathlib import Path
sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
import playground as pg

app = pg.PlaygroundApp()
for _ in range(6):
    app.update_idletasks()
    app.update()
app._book.select(1)  # chemistry tab
app._theme_var.set("dark")
app._on_theme_change()
for _ in range(6):
    app.update_idletasks()
    app.update()
try:
    from PIL import ImageGrab
    img = ImageGrab.grab(xdisplay=":91")
    img.save("/home/z/my-project/scratch/pg51/dark_chemistry.png")
    print("dark shot saved", img.size)
except Exception as exc:
    print("grab failed:", exc)
app._theme_var.set("light")
app._on_theme_change()
app._book.select(0)
for _ in range(6):
    app.update_idletasks()
    app.update()
img = ImageGrab.grab(xdisplay=":91")
img.save("/home/z/my-project/scratch/pg51/light_home.png")
print("light shot saved", img.size)
app._on_close()
