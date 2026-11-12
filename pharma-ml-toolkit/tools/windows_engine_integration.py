"""Exercise bundled Vina, AutoGrid4 and AutoDock4 through toolkit APIs.

Windows only; runs in an ephemeral directory with synthetic, non-biological
one-carbon ligand/receptor fixtures. It is an integration smoke, not a chemical
benchmark. It makes no changes to the checked-in data or release archives.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

from windows_engine_smoke import EXPECTED, ROOT, atom


def main() -> int:
    if os.name != "nt":
        raise SystemExit("Windows required; integration engines not executed")
    from pharmaml.docking.preparation import GridBox, write_dpf, write_gpf
    from pharmaml.docking.workflow import run_autodock4, run_vina, run_vina_batch

    engines = ROOT / "engines"
    for name, expected in EXPECTED.items():
        if hashlib.sha256((engines / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"refusing unexpected bundled executable: {name}")
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pharmaml-engine-integration-") as folder:
        work = Path(folder)
        receptor, ligand = work / "r.pdbqt", work / "l.pdbqt"
        receptor.write_text("".join(atom(i + 1, *p)[:70] + f"{0.1:6.3f}" + atom(i + 1, *p)[76:] for i, p in enumerate([
            (4, 0, 0), (-4, 0, 0), (0, 4, 0), (0, -4, 0), (0, 0, 4), (0, 0, -4)
        ])), encoding="ascii")
        ligand.write_text("ROOT\n" + atom(1, 0, 0, 0) + "ENDROOT\nTORSDOF 0\n", encoding="ascii")

        def score_and_pose(path: Path) -> bool:
            if not path.is_file():
                return False
            text = path.read_text(encoding="utf-8", errors="replace")
            scores = re.findall(r"^REMARK VINA RESULT:\s+([-+0-9.eE]+)", text, re.MULTILINE)
            return bool(scores) and all(math.isfinite(float(x)) for x in scores) and "ATOM " in text

        result = run_vina(receptor, ligand, work / "single.pdbqt", (0, 0, 0), (12, 12, 12),
                          exhaustiveness=1, cpu=1, seed=42, engine_dir=engines, timeout_s=120)
        single_ok = result.ok and score_and_pose(work / "single.pdbqt")
        print(json.dumps({"stage": "toolkit run_vina", "ok": single_ok, "error": result.error}))
        if not single_ok:
            failures.append(f"toolkit run_vina failed: {result.error}")

        ligands = work / "ligands"
        ligands.mkdir()
        (ligands / ligand.name).write_bytes(ligand.read_bytes())
        config = work / "vina.conf"
        config.write_text("receptor = r.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\n"
                          "size_x = 12\nsize_y = 12\nsize_z = 12\nexhaustiveness = 1\n"
                          "cpu = 1\nseed = 42\n", encoding="ascii")
        batch = run_vina_batch(config, ligands, work / "batch", engine_dir=engines, timeout_s=120)
        batch_ok = len(batch) == 1 and batch[0].ok and score_and_pose(work / "batch/l_out.pdbqt")
        batch_error = batch[0].error if batch else "no batch results"
        print(json.dumps({"stage": "toolkit run_vina_batch", "ok": batch_ok, "error": batch_error}))
        if not batch_ok:
            probe = subprocess.run([str(engines / "vina.exe"), "--config", str(config),
                                    "--ligand", str(ligands / ligand.name), "--out", str(work / "probe.pdbqt"),
                                    "--log", str(work / "probe.log")], cwd=work,
                                   capture_output=True, text=True, errors="replace", timeout=120, check=False)
            failures.append(f"toolkit run_vina_batch failed: {batch_error}; direct probe exit "
                            f"{probe.returncode}, stdout {probe.stdout[-1500:]}, stderr {probe.stderr[:1500]}")

        write_gpf(ligand, receptor_file=receptor.name,
                  box=GridBox(0, 0, 0, npts=(20, 20, 20)), receptor_types="C")
        write_dpf(ligand, receptor_file=receptor.name, ga_run=1, ga_pop_size=10,
                  ga_num_evals=2500, ga_num_generations=100, sw_max_its=10)
        ad = run_autodock4(work, engine_dir=engines, timeout_s=120, skip_existing=False)
        maps_ok = all((work / name).is_file() and (work / name).stat().st_size > 0
                      for name in ("r.C.map", "r.e.map", "r.d.map", "r.maps.fld"))
        log = work / "r.dlg"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        energies = re.findall(r"Estimated Free Energy of Binding\s*=\s*([-+0-9.eE]+)", text)
        ad_ok = ad.ok and maps_ok and bool(energies) and all(math.isfinite(float(x)) for x in energies)
        print(json.dumps({"stage": "toolkit AutoGrid4 and AutoDock4", "ok": ad_ok,
                          "maps_present": maps_ok, "error": ad.error}))
        if not ad_ok:
            tails = []
            for name in ("r.glg", "r.dlg"):
                path = work / name
                if path.is_file():
                    tails.append(f"{name} tail: {path.read_text(encoding='utf-8', errors='replace')[-3000:]}")
            failures.append(f"AutoGrid4/AutoDock4 failed: {ad.error}; " + " ".join(tails))
    for failure in failures:
        escaped = failure.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{escaped}")
    if failures:
        return 1
    print("All toy integrations passed; this is not biological or scientific validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
