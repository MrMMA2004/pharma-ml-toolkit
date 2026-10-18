import sys
sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
import playground as pg

app = pg.PlaygroundApp(persist_session=False)
for _ in range(6):
    app.update_idletasks(); app.update()
app._theme_var.set("dark"); app._on_theme_change()
app._book.select(1)  # chemistry (readonly combos + entries + family checks)
for _ in range(6):
    app.update_idletasks(); app.update()
from PIL import ImageGrab
ImageGrab.grab(xdisplay=":91").save("/home/z/my-project/scratch/pg51/dark_chem2.png")
app._book.select(2)  # PK (spinboxes + comboboxes)
for _ in range(6):
    app.update_idletasks(); app.update()
ImageGrab.grab(xdisplay=":91").save("/home/z/my-project/scratch/pg51/dark_pk2.png")
app._theme_var.set("light"); app._on_theme_change()
app._book.select(1)
for _ in range(6):
    app.update_idletasks(); app.update()
ImageGrab.grab(xdisplay=":91").save("/home/z/my-project/scratch/pg51/light_chem2.png")
print("3 shots saved")
app.destroy()
