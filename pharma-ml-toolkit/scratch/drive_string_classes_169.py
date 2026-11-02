"""GUI drive (100x-169): bug 1 repro — heatmaps with STRING class names.

Loads a string-class CSV into the Data & ML tab, trains EVERY
classification model the combo offers, and reports which runs leave
the Results card's confusion heatmap un-rendered ("Plot failed").
Then the same via the Deep Learning tab and the tune=True path.
"""
import importlib.util
import sys
import time
from pathlib import Path
from tkinter import messagebox

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "scratch" / "string_classes_169.csv"

rng = np.random.default_rng(7)
n = 140
frame = pd.DataFrame({
    "f1": rng.normal(size=n),
    "f2": rng.normal(size=n),
    "f3": rng.normal(size=n),
    "activity": rng.choice(["active", "inactive"], size=n),
})
frame.to_csv(CSV, index=False)

spec = importlib.util.spec_from_file_location("playground_live", ROOT / "playground.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

captured: list[tuple[str, tuple]] = []
for name in ("showerror", "showwarning", "showinfo"):
    setattr(messagebox, name, (lambda n: lambda *a, **k: captured.append((n, a)))(name))

app = module.PlaygroundApp(persist_session=False)


def pump(extra: float = 1.0, timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    quiet_since = time.monotonic()
    while True:
        app.update_idletasks()
        app.update()
        now = time.monotonic()
        if app._busy:
            quiet_since = now
        if not app._busy and now - quiet_since >= extra:
            break
        if now > deadline:
            raise SystemExit("job never finished")
        time.sleep(0.02)


def clear_host(host) -> None:
    for child in host.winfo_children():
        child.destroy()


app._ml_load_path(CSV)
pump(0.5)
app._target_combo.set("activity")
app._task_combo.set("classification")

models = [v for v in app._model_combo["values"] if v != "auto"]
print(f"classification models offered: {models}", flush=True)

results = []
for model in models:
    captured.clear()
    clear_host(app._ml_host)
    app._model_combo.set(model)
    app._tune_var.set(False)
    app._ml_train()
    try:
        pump(0.4)
    except SystemExit:
        results.append((model, "TIMEOUT"))
        continue
    errors = [f"{kind}: {' / '.join(str(x) for x in args)}" for kind, args in captured]
    figure = len(app._ml_host.winfo_children()) > 0
    results.append((model, "figure" if figure and not errors else ("NO-FIGURE" if not figure else "err")))
    if errors or not figure:
        print(f"  {model}: errors={errors} figure={figure}", flush=True)

print("\n=== classic ML tab (string classes) ===")
for model, status in results:
    print(f"  {model:28s} {status}")

# --- tune=True path with string classes ---
captured.clear()
clear_host(app._ml_host)
app._model_combo.set("logistic_regression")
app._tune_var.set(True)
app._ml_train()
pump(1.0)
errors = [f"{kind}: {' / '.join(str(x) for x in args)}" for kind, args in captured]
figure = len(app._ml_host.winfo_children()) > 0
print(f"\ntune=True logistic_regression: figure={figure} errors={errors}")

# --- Deep Learning tab with string classes ---
captured.clear()
app._dl_load_path(CSV)
app._dl_target_combo.set("activity")
app._dl_task_combo.set("classification")
app._dl_epochs.set("40")
clear_host(app._dl_host)
app._dl_train()
pump(1.0)
errors = [f"{kind}: {' / '.join(str(x) for x in args)}" for kind, args in captured]
figure = len(app._dl_host.winfo_children()) > 0
print(f"\nDL tab (string classes): figure={figure} errors={errors}")

app.destroy()
print("\nDONE")
