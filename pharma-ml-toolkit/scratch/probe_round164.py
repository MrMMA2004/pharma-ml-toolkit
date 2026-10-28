"""Round 100x-164 end-to-end probe: the REINVENT4 + DiffDock cards.

Exercises every new GUI surface headlessly (Xvfb):

* the Chemistry tab's score-transform lab (component + transform table);
* the custom-alerts check and the REINVENT4 standardization chain;
* the Generative tab's diversity filter + alerts wiring on a real GA run
  (the scaffold-memory stats must land in the stats text);
* the Docking tab's pose evaluation (symmetry-corrected RMSD table +
  the evaluate.py report), the randomize_position demo;
* the pocket extraction on the bundled lysozyme (example button ->
  extract -> save);
* the session-persistence map carrying the four new text boxes.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
import tkinter.filedialog as fd
import tkinter.messagebox as mb

import playground

DIALOGS: list[tuple[str, tuple]] = []
mb.showerror = lambda *a, **k: DIALOGS.append(("error", a))
mb.showwarning = lambda *a, **k: DIALOGS.append(("warn", a))
mb.showinfo = lambda *a, **k: DIALOGS.append(("info", a))

app = playground.PlaygroundApp(persist_session=False)


def pump_until_idle(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def run_job(label: str, starter) -> None:
    starter()
    assert pump_until_idle(), f"{label} timed out"
    assert not DIALOGS, f"{label} raised dialogs: {DIALOGS[:3]}"


# -- 1. the score-transform lab ------------------------------------------- #
app._transform_component.set("molecular weight (Da)")
app._transform_kind.set("double sigmoid (band)")
run_job("transform lab", app._transform_lab_run)
rows = app._transform_tree.get_children()
assert len(rows) > 0, "transform table empty"
summary = app._transform_summary.cget("text")
assert "molecular weight" in summary and "mean score" in summary, summary
print("transform lab:", summary)

# the step transform path (params filtered per transform)
app._transform_kind.set("step window")
run_job("transform lab (step)", app._transform_lab_run)
assert len(app._transform_tree.get_children()) > 0
print("transform lab (step):", app._transform_summary.cget("text"))

# -- 2. the custom-alerts check ------------------------------------------- #
run_job("alerts check", app._alerts_check_run)
rows = app._alerts_tree.get_children()
assert len(rows) > 0, "alerts table empty"
alerts_summary = app._alerts_summary.cget("text")
assert "blocked" in alerts_summary, alerts_summary
print("alerts check:", alerts_summary)

# -- 3. the REINVENT4 standardization chain -------------------------------- #
app._alert_mol_text.delete("1.0", "end")
app._alert_mol_text.insert(
    "1.0", "CC(=O)[O-].[Na+]\nCCO\nC[NH3+]\nC"
)
run_job("standardize", app._reinvent_standardize_run)
rows = app._alerts_tree.get_children()
texts = [app._alerts_tree.item(row, "values") for row in rows]
assert any("CC(=O)O" in str(values[1]) for values in texts), texts
assert any("rejected" in str(values[2]) for values in texts), texts
print("standardize:", app._alerts_summary.cget("text"))

# -- 4. the generative diversity filter + alerts ---------------------------- #
app._objective_combo.set("QED drug-likeness")
app._method_combo.set("genetic algorithm")
app._pop_size.set("8")
app._generations.set("3")
app._diversity_combo.set("penalize same SMILES")
app._diversity_bucket.set("2")
app._gen_alerts_text.delete("1.0", "end")
app._gen_alerts_text.insert("1.0", "[N+](=O)[O-]")
run_job("GA with diversity + alerts", app._run_evolution)
stats = app._ga_stats_text.get("1.0", "end")
assert "REINVENT4 dressings" in stats, stats[:400]
assert "custom alerts" in stats and "1 pattern" in stats, stats[:600]
assert "penalised scores" in stats, stats[:600]
print("GA dressings block present; summary:", app._ga_summary.cget("text")[:80])

# back to none for cleanliness
app._diversity_combo.set("none")
app._gen_alerts_text.delete("1.0", "end")

# -- 5. the pose evaluation -------------------------------------------------- #
from rdkit import Chem
from rdkit.Chem import AllChem

aspirin = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
params = AllChem.ETKDG()
params.randomSeed = 42  # type: ignore[assignment]
AllChem.EmbedMultipleConfs(aspirin, numConfs=4, params=params)
tmp = Path(tempfile.mkdtemp(prefix="pharmaml_probe164_"))
reference_sdf = tmp / "reference.sdf"
poses_sdf = tmp / "poses.sdf"
with open(reference_sdf, "w") as handle:
    handle.write(Chem.MolToMolBlock(aspirin, confId=aspirin.GetConformers()[0].GetId()))
# the poses: the other conformers, each slightly translated
import numpy as np
from rdkit.Geometry import Point3D

with open(poses_sdf, "w") as handle:
    conformers = [
        aspirin.GetConformer(i) for i in range(aspirin.GetNumConformers())
    ]
    for index, conf in enumerate(conformers[1:], start=1):
        shifted = Chem.Mol(aspirin)
        shifted.RemoveAllConformers()
        shifted.AddConformer(conf, assignId=True)
        positions = np.asarray(conf.GetPositions()) + float(index) * 0.5
        writer_conf = shifted.GetConformer()
        for atom_index in range(shifted.GetNumAtoms()):
            position = positions[atom_index]
            writer_conf.SetAtomPosition(
                atom_index,
                Point3D(*position.tolist()),
            )
        handle.write(Chem.MolToMolBlock(shifted))
        handle.write("$$$$\n")
app._pose_reference_entry.delete(0, "end")
app._pose_reference_entry.insert(0, str(reference_sdf))
app._pose_poses_entry.delete(0, "end")
app._pose_poses_entry.insert(0, str(poses_sdf))
run_job("pose evaluation", app._pose_evaluate_run)
rows = app._pose_tree.get_children()
assert len(rows) == 3, f"expected 3 poses, got {len(rows)}"
report_text = app._pose_report_text.get("1.0", "end")
assert "RMSD < 2 A" in report_text and "pose clusters" in report_text, report_text
assert "rotatable bond(s)" in report_text, report_text
print("pose summary:", app._pose_summary.cget("text")[:100])

# the randomize_position demo
run_job("pose randomization", app._pose_randomize_run)
summary = app._pose_summary.cget("text")
assert "randomize_position" in summary, summary
print("randomize:", summary[:100])

# -- 6. the pocket extraction ------------------------------------------------ #
run_job("pocket example", app._pocket_load_example)
run_job("pocket extraction", app._pocket_extract_run)
rows = app._pocket_tree.get_children()
assert len(rows) > 0, "pocket residue table empty"
pocket_summary = app._pocket_summary.cget("text")
assert "residue(s) in the pocket" in pocket_summary, pocket_summary
report_text = app._pocket_report_text.get("1.0", "end")
assert "pocket residues" in report_text and "closest ligand atom" in report_text
print("pocket:", pocket_summary[:110])

# the save button with a mocked dialog
target = tmp / "pocket.pdb"
fd.asksaveasfilename = lambda **kwargs: str(target)
app._pocket_save_run()
assert target.is_file() and target.read_text().startswith("ATOM")
print("pocket saved:", target.name, target.stat().st_size, "bytes")

# -- 7. the session map carries the new boxes -------------------------------- #
text_map = app._session_text_widgets()
for key in ("transform_mol", "alert_smarts", "alert_mol", "gen_alerts"):
    assert key in text_map and text_map[key] is not None, key
print("session map keys:", sorted(k for k in text_map if text_map[k] is not None))

print("PROBE 164 OK — every new REINVENT4/DiffDock surface works end-to-end")
app.destroy()
