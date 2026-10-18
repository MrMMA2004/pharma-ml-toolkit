import sys
sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
import playground as pg

LIGHT_DEFAULTS = {"#dcdad5", "#bab5ab", "#4a6984", "#ffffff", "#fcfcfc",
                  "#dcdad5", "#e0ddd5", "#eeebe3", "#ccccc4", "#afafaf"}
app = pg.PlaygroundApp()
for _ in range(2):
    app.update_idletasks(); app.update()
app._theme_var.set("dark"); app._on_theme_change()
for _ in range(2):
    app.update_idletasks(); app.update()
import tkinter.ttk as ttk
st = ttk.Style(app)
dark = set(pg._DARK_PALETTE.values())
hits = []
for name in sorted(st.element_classes()) if hasattr(st, "element_classes") else []:
    pass
# walk layout-reachable styles instead: iterate over every style we configure
for name in ["TEntry", "TCombobox", "TSpinbox", "TScrollbar", "TButton",
             "TCheckbutton", "TRadiobutton", "TNotebook", "TNotebook.Tab",
             "Treeview", "Treeview.Heading", "TProgressbar", "TLabel",
             "TLabelframe", "TLabelframe.Label", "TFrame", "Status.TFrame"]:
    for opt in ["background", "fieldbackground", "foreground", "troughcolor",
                "arrowcolor", "lightcolor", "darkcolor", "bordercolor",
                "selectbackground", "selectforeground", "insertcolor",
                "indicatorcolor", "activebackground", "activeforeground"]:
        try:
            val = st.lookup(name, opt)
        except Exception:
            continue
        if val and isinstance(val, str) and val.lower() in LIGHT_DEFAULTS and val not in dark:
            hits.append(f"configure {name}.{opt} = {val}")
        try:
            mapping = st.map(name, opt)
        except Exception:
            mapping = []
        for entry in mapping or []:
            for piece in entry:
                if isinstance(piece, str) and piece.lower() in LIGHT_DEFAULTS:
                    hits.append(f"map {name}.{opt}: {entry}")
                    break
for hit in sorted(set(hits)):
    print(hit)
app.destroy()
