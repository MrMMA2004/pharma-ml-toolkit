"""Exercise bundled Vina, AutoGrid4 and AutoDock4 through toolkit APIs.

Windows only; runs in an ephemeral directory with synthetic, non-biological
artificial carbon-only ligand/receptor fixtures. It is an integration smoke, not a chemical
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

        # A three-atom synthetic ligand avoids the negative degrees of freedom
        # of a single atom in AutoDock. Pin this smoke fixture's seeds; production
        # defaults are deliberately not modified.
        ad_ligand = work / "ad_ligand.pdbqt"
        ad_ligand.write_text("ROOT\n" + "".join(atom(i + 1, *p) for i, p in enumerate([
            (0, 0, 0), (1.4, 0, 0), (0, 1.4, 0)
        ])) + "ENDROOT\nTORSDOF 0\n", encoding="ascii")
        ad_box = GridBox(0, 0, 0, npts=(30, 30, 30))
        write_gpf(ad_ligand, receptor_file=receptor.name, box=ad_box, receptor_types="C")
        dpf = write_dpf(ad_ligand, receptor_file=receptor.name, ga_run=1, ga_pop_size=10,
                        ga_num_evals=500, ga_num_generations=20, sw_max_its=10, ls_search_freq=0)
        dpf.write_text(dpf.read_text(encoding="utf-8").replace("seed pid time", "seed 123 456"),
                       encoding="utf-8")
        ad = run_autodock4(work, engine_dir=engines, timeout_s=120, skip_existing=False)
        glg = work / "r.glg"
        grid_log = glg.read_text(encoding="utf-8", errors="replace") if glg.is_file() else ""
        # Verify the real engine's fencepost interpretation rather than inferring it
        # from toolkit-generated files. 30 GPF intervals should yield 31 points.
        points_ok = bool(re.search(r"Number of grid points in x-direction:\s*31\b", grid_log))
        print(json.dumps({"stage": "AutoGrid npts interpretation", "gpf_npts": 30,
                          "x_grid_points_31": points_ok}))
        if not points_ok:
            failures.append("AutoGrid did not report 31 x-grid points for npts 30")
        map_path = work / "r.C.map"
        if map_path.is_file():
            header = map_path.read_text(encoding="utf-8", errors="replace").splitlines()[:8]
            # Surface the actual map metadata in annotations even on success.
            notice = json.dumps({"stage": "AutoGrid map header", "lines": header})
            print("::notice::" + notice.replace("%", "%25"))
            expected = ("SPACING 0.375", "NELEMENTS 30 30 30", "CENTER 0.000 0.000 0.000")
            if not all(any(line.strip() == value for line in header) for value in expected):
                failures.append("AutoGrid map header differs from expected center, spacing or npts: " + repr(header))
            matched = ad_box.as_vina_kwargs(convention="autogrid")
            if (matched["size_x"], matched["size_y"], matched["size_z"]) != (11.25, 11.25, 11.25):
                failures.append("opt-in Vina size does not match the 30-interval map extent")
            matched_result = run_vina(
                receptor, ligand, work / "matched.pdbqt",
                (matched["center_x"], matched["center_y"], matched["center_z"]),
                (matched["size_x"], matched["size_y"], matched["size_z"]),
                exhaustiveness=1, cpu=1, seed=42, engine_dir=engines, timeout_s=120,
            )
            if not matched_result.ok or not score_and_pose(work / "matched.pdbqt"):
                failures.append(f"Vina with opt-in matching geometry failed: {matched_result.error}")
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
        # Pin the specific odd-npts behavior observed for this hashed binary.
        # It does not establish behavior of other AutoGrid builds.
        odd = work / "odd-grid"
        odd.mkdir()
        (odd / "r.pdbqt").write_bytes(receptor.read_bytes())
        odd_ligand = odd / "l.pdbqt"
        odd_ligand.write_bytes(ad_ligand.read_bytes())
        write_gpf(odd_ligand, receptor_file="r.pdbqt",
                  box=GridBox(0, 0, 0, npts=(21, 21, 21)), receptor_types="C")
        odd_proc = subprocess.run([str(engines / "autogrid4.exe"), "-p", "r.gpf", "-l", "r.glg"],
                                  cwd=odd, capture_output=True, timeout=60, check=False)
        odd_log = (odd / "r.glg").read_text(encoding="utf-8", errors="replace") if (odd / "r.glg").is_file() else ""
        odd_match = re.search(r"Number of grid points in x-direction:\s*(\d+)", odd_log)
        odd_header = (odd / "r.C.map").read_text(encoding="utf-8", errors="replace").splitlines()[:8] if (odd / "r.C.map").is_file() else []
        notice = json.dumps({"stage": "odd npts probe", "gpf_npts": 21, "exit": odd_proc.returncode,
                             "x_grid_points": int(odd_match.group(1)) if odd_match else None,
                             "map_header": odd_header, "log_tail": odd_log[-500:]})
        print("::notice::" + notice.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))
        if (odd_proc.returncode != 0 or not odd_match or odd_match.group(1) != "21"
                or "NELEMENTS 20 20 20" not in odd_header):
            failures.append("bundled AutoGrid odd npts behavior differed from the observed 21-to-20 reduction")
    for failure in failures:
        escaped = failure.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{escaped}")
    if failures:
        return 1
    print("All toy integrations passed; this is not biological or scientific validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
