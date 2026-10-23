"""E2E probe for the applicability-domain card (100x-118).

Drives the real GUI headless (Xvfb :99): loads the SMILES-carrying
library, trains ridge, runs the full AD analysis (bootstrap 8), the
kNN-only pass (bootstrap 0), and the bundle mode (save bundle -> load
-> train/test CSVs -> analyse), then saves the results CSV.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import tkinter.filedialog as fd
import tkinter.messagebox as mb

import pandas as pd

import playground

DIALOGS: list[tuple[str, tuple]] = []
mb.showerror = lambda *a, **k: DIALOGS.append(("error", a))
mb.showwarning = lambda *a, **k: DIALOGS.append(("warn", a))
mb.showinfo = lambda *a, **k: DIALOGS.append(("info", a))

app = playground.PlaygroundApp()


def pump_until_idle(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


# -- 1. load the SMILES library + train ridge -----------------------------
app._ml_load_path(Path("examples/data/library_with_descriptors.csv"))
assert "rows" in app._ml_info.cget("text"), app._ml_info.cget("text")
app._model_combo.set("logistic_regression")
app._ml_train()
assert pump_until_idle(), "ML train job hung"
assert app._ml_last_run is not None
n_test = len(app._ml_last_run[5])

info = app._ad_source_info.cget("text")
assert "logistic" in info and "train" in info, info
print("source line:", info)

# -- 2. full analysis (bootstrap 8 for speed) ------------------------------
app._ad_boot_spin.set("8")
app._ml_ad_run()
assert pump_until_idle(), "AD job hung"
result = app._ml_ad_result
assert result is not None
assert result.reliable is not None
rows = app._ad_tree.get_children()
assert len(rows) == n_test, (len(rows), n_test)
status = app._ad_status.cget("text")
assert "distance threshold" in status and "reliable" in status, status
print("full pass:", status)
assert len(app._ad_host.winfo_children()) == 1, "AD figure not embedded"
# the name column carries the library's SMILES
first_name = str(app._ad_tree.item(rows[0])["values"][0])
assert any(ch in first_name for ch in "CNc#()"), first_name[:40]
print("name column sample:", first_name[:48])
# verdict counts agree with the result object
inside = int(result.inside_ad.sum())
assert f"{inside}/{n_test} inside AD" in status

# -- 3. kNN-only quick pass (bootstrap 0) -----------------------------------
app._ad_boot_spin.set("0")
app._ml_ad_run()
assert pump_until_idle(), "kNN-only job hung"
result = app._ml_ad_result
assert result.reliable is None
assert result.pred_mean is None
status = app._ad_status.cget("text")
assert "kNN-only" in status, status
assert app._ml_ad_frame["reliable"].isna().all()
print("knn-only:", status)

# -- 4. bundle mode ----------------------------------------------------------
tmp = Path(tempfile.mkdtemp(prefix="ad_e2e_"))
fd.asksaveasfilename = lambda **k: str(tmp / "logistic_bundle.joblib")
app._ml_save_model_bundle()
assert pump_until_idle(), "bundle save hung"
assert (tmp / "logistic_bundle.joblib").is_file()

fd.askopenfilename = lambda **k: str(tmp / "logistic_bundle.joblib")
app._ml_load_model_bundle()
assert pump_until_idle(), "bundle load hung"
assert app._ml_bundle is not None

frame = pd.read_csv("examples/data/library_with_descriptors.csv")
frame.iloc[:60].to_csv(tmp / "train.csv", index=False)
frame.iloc[60:].to_csv(tmp / "test.csv", index=False)

fd.askopenfilename = lambda **k: str(tmp / "train.csv")
app._ml_ad_pick_train_csv()
assert app._ml_ad_train_path is not None
fd.askopenfilename = lambda **k: str(tmp / "test.csv")
app._ml_ad_pick_test_csv()
assert app._ml_ad_test_path is not None

app._ad_source_combo.set("loaded bundle")
app._ml_ad_source_changed()
assert app._ad_bundle_row.winfo_manager() == "grid", "bundle row not shown"
info = app._ad_source_info.cget("text")
assert "logistic" in info and "features" in info, info
print("bundle source:", info)

app._ad_boot_spin.set("6")
app._ml_ad_run()
assert pump_until_idle(), "bundle-mode AD job hung"
result = app._ml_ad_result
assert result is not None and result.pred_mean is not None
rows = app._ad_tree.get_children()
assert len(rows) == len(frame) - 60, (len(rows), len(frame) - 60)
status = app._ad_status.cget("text")
assert "ensemble" in status, status
print("bundle pass:", status)

# -- 5. save the results CSV (the upload's Applicability_Domain_Results.csv)
fd.asksaveasfilename = lambda **k: str(tmp / "Applicability_Domain_Results.csv")
app._ml_ad_save()
saved = tmp / "Applicability_Domain_Results.csv"
assert saved.is_file(), "results CSV not written"
out = pd.read_csv(saved)
assert list(out.columns) == [
    "name", "predicted", "prediction_sd", "knn_distance",
    "inside_ad", "low_uncertainty", "reliable",
], list(out.columns)
print("saved columns:", list(out.columns), "| rows:", len(out))

# -- 6. guard rails: no run / no bundle messages ----------------------------
app._ml_last_run = None
app._ad_source_combo.set("last trained model")
app._ml_ad_source_changed()
assert "no run yet" in app._ad_source_info.cget("text")
app._ml_ad_run()
assert any(kind == "warn" for kind, _ in DIALOGS), DIALOGS
print("no-run warning OK")

app._ml_bundle = None
app._ad_source_combo.set("loaded bundle")
app._ml_ad_source_changed()
assert "no bundle loaded" in app._ad_source_info.cget("text")
print("no-bundle hint OK")

app.destroy()
shutil.rmtree(tmp, ignore_errors=True)
errors = [d for d in DIALOGS if d[0] == "error"]
assert not errors, errors
print("AD E2E PROBE PASS — no error dialogs")
