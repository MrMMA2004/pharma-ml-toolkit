"""Probe the new playground features: SMILES array card, RISM card,
ensemble builder, model persistence, and the receptor bad-residue flow."""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import tkinter.filedialog as fd
import tkinter.messagebox as mb

DIALOGS = []
mb.showerror = lambda *a, **k: DIALOGS.append(("error", a))
mb.showwarning = lambda *a, **k: DIALOGS.append(("warn", a))
mb.showinfo = lambda *a, **k: DIALOGS.append(("info", a))

import playground

TMP = Path(tempfile.mkdtemp())

# a fake vina binary so the SMILES array run exercises the real flow
binroot = TMP / "bin"
binroot.mkdir()
(binroot / "vina").write_text(
    "#!/bin/bash\n"
    "out=''; lig=''; prev=''\n"
    "for arg in \"$@\"; do\n"
    "  [ \"$prev\" = --out ] && out=\"$arg\"\n"
    "  [ \"$prev\" = --ligand ] && lig=\"$arg\"\n"
    "  prev=\"$arg\"\n"
    "done\n"
    "{ echo 'REMARK VINA RESULT:    -8.2      0.000      0.000'\n"
    "  echo 'REMARK VINA RESULT:    -7.1      1.923      2.941'\n"
    "  grep '^ATOM\\|^HETATM' \"$lig\"; } > \"$out\"\n"
    "exit 0\n"
)
os.chmod(binroot / "vina", 0o755)
os.environ["PATH"] = f"{binroot}{os.pathsep}{os.environ.get('PATH', '')}"

app = playground.PlaygroundApp()


def pump(timeout=420.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def shot(name):
    out = TMP / f"{name}.png"
    try:
        app.update_idletasks(); app.update()
        from PIL import Image
        import subprocess
        xwininfo = subprocess.run(
            ["xwd", "-root", "-display", os.environ.get("DISPLAY", ":99")],
            capture_output=True)
        if xwininfo.returncode == 0 and xwininfo.stdout:
            with open(TMP / f"{name}.xwd", "wb") as fh:
                fh.write(xwininfo.stdout)
            subprocess.run(
                ["convert", str(TMP / f"{name}.xwd"), str(out)],
                capture_output=True)
            if out.is_file():
                print(f"shot: {out.name}")
                return
    except Exception as exc:  # noqa: BLE001
        print("shot failed:", exc)


# ---- 1. Chemistry: RISM-KH card -----------------------------------------
app._book.select(1)  # Chemistry
app.update()
app._rism_solvent_combo.set("spc")
app._rism_temp_entry.delete(0, "end"); app._rism_temp_entry.insert(0, "298.15")
app._rism_run()
assert pump(), "RISM run timed out"
rows = len(app._rism_tree.get_children())
summary = app._rism_summary.cget("text")
print(f"RISM: {rows} table rows | {summary}")
assert rows == 30, f"expected 30 RISM rows (3 molecules x 10), got {rows}"
assert "Gsolv" in summary
app._rism_frame.to_csv(TMP / "rism.csv", index=False)
print("RISM CSV saved:", (TMP / "rism.csv").stat().st_size, "bytes")

# ---- 2. Chemistry: feature factory now offers the rism family ----------- 
assert "rism" in app._feat_family_vars, "rism family checkbox missing"
app._feat_family_vars["rism"].set(True)
app._feat_update_width()
print("factory width label:", app._feat_width_label.cget("text"))
shot("chem_rism")

# ---- 3. Docking: receptor with auto-remove bad residues ----------------
app._book.select(4)  # Docking
app.update()
assert bool(app._dock_bad_res_var.get()), "bad-res checkbox should default on"

# break the example lysozyme: residues 40/50 lose atoms
text = (playground.EXAMPLES_DIR / "lysozyme_1aki.pdb").read_text()
broken = [
    line for line in text.splitlines()
    if not (line.startswith("ATOM") and line[22:26].strip() in ("40", "50")
            and line[12:16].strip() in ("CA", "CB", "C", "O"))
]
src = TMP / "broken.pdb"
src.write_text("\n".join(broken) + "\nTER\nEND\n")
app._dock_receptor_source = src
app._dock_bad_res_var.set(True)
fd.asksaveasfilename = lambda **k: str(TMP / "rec.pdbqt")
app._dock_prepare_receptor()
assert pump(300), "receptor preparation timed out"
report_text = app._dock_protein_text.get("1.0", "end")
print("receptor report mentions bad residues:", "bad residues" in report_text)
assert "A:40" in report_text and "A:50" in report_text, report_text[:400]

# ---- 4. Docking: SMILES array run ---------------------------------------
app._dock_smiles_receptor = TMP / "rec.pdbqt"
app._dock_smiles_out = TMP / "campaign"
app._dock_smiles_text.delete("1.0", "end")
app._dock_smiles_text.insert(
    "1.0", "CC(=O)Oc1ccccc1C(=O)O\nCC(=O)Nc1ccc(O)cc1\nbroken!!smiles")
app._dock_smiles_workers_entry.delete(0, "end")
app._dock_smiles_workers_entry.insert(0, "2")
DIALOGS.clear()
app._dock_smiles_run()
assert pump(600), "SMILES array docking timed out"
info = app._dock_results_info.cget("text")
print("array results:", info)
assert "3 SMILES docked" in info and "2 ok" in info, info
assert len(app._dock_results_tree.get_children()) == 3
assert (TMP / "campaign" / "01_aspirin" / "out.pdbqt").is_file()
print("folders:", sorted(p.name for p in (TMP / "campaign").iterdir()))
shot("docking_array")

# ---- 5. Data & ML: ensemble builder + persistence -----------------------
app._book.select(5)  # Data & ML
app.update()
app._example_combo.set("qsar_regression.csv")
app._ml_load_example()
assert pump(), "example load timed out"
app._target_combo.set("pIC50")
# voting with the default RF + SVM + KNN trio
app._ens_kind_combo.set("voting")
app._ens_train()
assert pump(600), "voting ensemble training timed out"
ens_summary = app._ens_summary.cget("text")
print("voting:", ens_summary[:90])
assert "voting" in ens_summary and "trained" in ens_summary
metrics = "\n".join(
    child.cget("text") for child in app._ml_metrics.winfo_children())
print("voting metrics head:", metrics.replace("\n", " | ")[:140])
assert "R²" in metrics

# stacking with a meta learner
app._ens_kind_combo.set("stacking")
app._ens_meta_combo.set("auto")
app._ens_train()
assert pump(600), "stacking ensemble training timed out"
print("stacking:", app._ens_summary.cget("text")[:90])

# save the bundle
fd.asksaveasfilename = lambda **k: str(TMP / "stacked.joblib")
app._ml_save_model_bundle()
assert pump(), "bundle save timed out"
assert (TMP / "stacked.joblib").is_file()
assert (TMP / "stacked.json").is_file()
print("bundle + sidecar:",
      (TMP / "stacked.joblib").stat().st_size, "+",
      (TMP / "stacked.json").stat().st_size, "bytes")

# load it back and predict on the training CSV
fd.askopenfilename = lambda **k: str(TMP / "stacked.joblib")
app._ml_load_model_bundle()
assert pump(), "bundle load timed out"
bundle_info = app._ml_bundle_info.cget("text")
print("bundle info:", bundle_info[:120])
assert "stacking" in bundle_info

fd.askopenfilename = lambda **k: str(
    playground.EXAMPLES_DIR / "qsar_regression.csv")
app._ml_predict_with_bundle()
errors = [d for d in DIALOGS if d[0] == "error"]
assert not errors, f"prediction raised: {errors}"
assert pump(), "bundle prediction timed out"
n_rows = len(app._predict_tree.get_children())
pred_info = app._ml_bundle_info.cget("text")
print(f"predicted {n_rows} rows | {pred_info[:100]}")
assert n_rows > 0 and "predicted" in pred_info
shot("ml_ensemble")

# ---- 6. RL advisory + registry listing -----------------------------------
names = [entry["name"] for entry in __import__(
    "pharmaml.ml.models", fromlist=["list_models"]).list_models(
    "regression", only_available=True)]
assert "voting" in names and "stacking" in names
print("registry regression count:", len(names), "| voting+stacking present")

model_values = list(app._model_combo["values"])
print("model combobox carries ensembles:",
      "voting" in model_values and "stacking" in model_values)

print("ALL PROBES PASSED")
app.destroy()
shutil.rmtree(TMP, ignore_errors=True)
