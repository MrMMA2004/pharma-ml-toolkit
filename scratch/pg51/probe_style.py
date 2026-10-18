import sys
sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
import playground as pg

app = pg.PlaygroundApp()
for _ in range(2):
    app.update_idletasks(); app.update()
app._theme_var.set("dark"); app._on_theme_change()
for _ in range(2):
    app.update_idletasks(); app.update()
style = app._build_style.__self__  # noqa
import tkinter.ttk as ttk
st = ttk.Style(app)
for name, opt in [("TEntry", "fieldbackground"), ("TEntry", "background"),
                  ("TCombobox", "fieldbackground"), ("TSpinbox", "fieldbackground"),
                  ("TScrollbar", "troughcolor"), ("TScrollbar", "background"),
                  ("TCombobox", "foreground"), ("TEntry", "foreground")]:
    print(f"{name}.{opt} =", repr(st.lookup(name, opt)),
          "| map:", st.map(name, opt) if opt in ("background", "fieldbackground") else "")
print("root:", app.cget("background"))
print("active theme:", pg._ACTIVE_THEME)
app.destroy()
