"""PROBE 143-b: the MACCS render's dropped-message overwrite (live).

compute_maccs is stubbed (rdkit absent on this box) to return a
1-molecule matrix with ONE valid + ONE invalid SMILES -- exactly the
mixed case -- and the REAL render closure runs through the REAL
_start_job/_poll_jobs machinery.  Expected: the label ends with only
the dropped message, the "mean X/167" summary lost.

Run: DISPLAY=:99 python scratch/probe_audit_143_b.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

import playground  # noqa: E402
import pharmaml.chem as chem  # noqa: E402


def pump(app: playground.PlaygroundApp, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.02)


# stub: 2 input SMILES, only the first survives
def fake_compute_maccs(smiles: list[str]) -> tuple[Any, Any]:  # noqa: F821
    matrix = np.zeros((len(smiles) - 1, 167), dtype=int)
    matrix[:, :45] = 1  # 45 keys set
    return matrix, [0]


chem.compute_maccs = fake_compute_maccs

for name in ("showinfo", "showwarning", "showerror"):
    setattr(playground.messagebox, name, lambda *a, **k: None)

# the deps gate: rdkit is probed via import -- force the probe result
app = playground.PlaygroundApp(persist_session=False)
pump(app, 0.3)
app.deps["rdkit"] = True
app.deps["numpy"] = True

app._maccs_text.delete("1.0", "end")
app._maccs_text.insert("1.0", "CCO\nNOT_A_SMILES")
print("input lines:", playground.parse_smiles_lines(app._maccs_text.get("1.0", "end")))

app._maccs_run()
deadline = time.monotonic() + 30.0
while app._busy and time.monotonic() < deadline:
    pump(app, 0.1)
pump(app, 0.4)

final = app._maccs_result.cget("text")
print("final label text:", repr(final))
lost = "mean" not in str(final) and "1 SMILES failed to parse" in str(final)
print("summary lost + dropped-only message:", lost)
app.destroy()
print(f"PROBE143B={'CONFIRMED' if lost else 'REFUTED'}")
