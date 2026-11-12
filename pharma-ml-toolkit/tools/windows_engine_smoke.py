"""Run bundled Windows engines and a tiny synthetic Vina calculation.

Windows only. No external tools or network access. The toy geometry is not a
biological receptor or meaningful scientific benchmark. AutoGrid/AutoDock are
only invoked for their help output; their docking workflows are NOT validated.
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

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "vina.exe": "e0c4b2715e0c1a74f6e92d0f3be0328ac97542eafbc111e6b1efad897a73cce5",
    "autogrid4.exe": "797efce687d1ae82df59726461e0e1966b3d8edb0f8b187f982fa1ab0c12da9e",
    "autodock4.exe": "36c0b16c04d7df8e6225737bae65bb04058d1f4c90accfaf2d230dbc913954bf",
}


def atom(serial: int, x: float, y: float, z: float) -> str:
    """One neutral carbon, formatted as an AutoDock PDBQT atom record."""
    return (f"ATOM  {serial:5d}  C   TOY A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}    {0.0:6.3f} C\n")


def main() -> int:
    if os.name != "nt":
        raise SystemExit("Windows is required; no engine was executed")
    engines = ROOT / "engines"
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((engines / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"refusing unexpected binary {name}: {actual}")
    with tempfile.TemporaryDirectory(prefix="pharmaml-engine-smoke-") as folder:
        work = Path(folder)
        for name in ("autogrid4.exe", "autodock4.exe"):
            # Historical engines vary in help exit codes. Record them, don't
            # mistake an invocation or a zero exit for a working docking run.
            run = subprocess.run([str(engines / name), "-h"], cwd=work,
                                 capture_output=True, text=True, timeout=30, check=False,
                                 errors="replace")
            print(json.dumps({"engine": name, "operation": "help only, not docking",
                              "exit_code": run.returncode, "stdout": run.stdout[:4000],
                              "stderr": run.stderr[:4000]}), flush=True)
        receptor = work / "receptor.pdbqt"
        ligand = work / "ligand.pdbqt"
        output = work / "out.pdbqt"
        receptor.write_text("".join(atom(i + 1, *point) for i, point in enumerate([
            (4, 0, 0), (-4, 0, 0), (0, 4, 0), (0, -4, 0), (0, 0, 4), (0, 0, -4)
        ])), encoding="ascii")
        ligand.write_text("ROOT\n" + atom(1, 0, 0, 0) + "ENDROOT\nTORSDOF 0\n", encoding="ascii")
        command = [str(engines / "vina.exe"), "--receptor", str(receptor),
                   "--ligand", str(ligand), "--center_x", "0", "--center_y", "0",
                   "--center_z", "0", "--size_x", "12", "--size_y", "12",
                   "--size_z", "12", "--exhaustiveness", "1", "--cpu", "1",
                   "--seed", "42", "--out", str(output)]
        run = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=120,
                             check=False, errors="replace")
        print(json.dumps({"engine": "vina.exe", "operation": "synthetic toy docking",
                          "exit_code": run.returncode, "stdout": run.stdout,
                          "stderr": run.stderr}), flush=True)
        if run.returncode != 0 or not output.is_file():
            raise SystemExit("Vina did not produce an output pose")
        text = output.read_text(encoding="utf-8", errors="replace")
        scores = re.findall(r"^REMARK VINA RESULT:\s+([-+0-9.eE]+)", text, re.MULTILINE)
        if not scores or not all(math.isfinite(float(score)) for score in scores) or "ATOM " not in text:
            raise SystemExit("Vina output missing finite result or atom record")
        print("Vina toy output contains a finite score and pose; not a scientific validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
