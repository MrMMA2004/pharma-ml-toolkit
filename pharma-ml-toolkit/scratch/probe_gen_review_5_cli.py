"""100x-21-b probe 5: CLI ``generative`` argument handling (bad combos).

Suggested probe #5 (--seeds table without a SMILES column / extra columns /
empty file) + own hunts: --iterations 0, --top negative, --out unwritable,
--seed huge, rl with --population (silently ignored?), --seeds .txt.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent


def cli(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pharmaml.cli", "generative", *args],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")},
    )
    out = (proc.stdout + proc.stderr).strip()
    tail = "\n".join(out.splitlines()[-6:])
    return proc.returncode, tail


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # -- seeds table WITHOUT a smiles column (name + ic50)
    no_smi = tmp / "no_smiles.csv"
    no_smi.write_text("name,ic50\naspirin,3.2\nibuprofen,12.5\n")
    code, out = cli("run", "--recipe", "similar", "--seeds", str(no_smi),
                    "--iterations", "1", "--population", "8")
    print(f"[no-smiles-col similar] exit={code}\n{out}\n")

    # -- same table but recipe qed (GA seeded mode: seeds dropped)
    code, out = cli("run", "--recipe", "qed", "--seeds", str(no_smi),
                    "--iterations", "1", "--population", "8")
    print(f"[no-smiles-col qed] exit={code}\n{out}\n")

    # -- seeds table WITH extra columns (name, smiles, mw)
    extra = tmp / "extra.csv"
    extra.write_text("name,smiles,mw\naspirin,CC(=O)Oc1ccccc1C(=O)O,180.16\n"
                     "ethanol,CCO,46.07\n")
    code, out = cli("run", "--recipe", "similar", "--seeds", str(extra),
                    "--iterations", "1", "--population", "8")
    print(f"[extra-cols similar] exit={code}\n{out}\n")

    # -- column named 'SMILES ' (case/space variants)
    spaced = tmp / "spaced.csv"
    spaced.write_text("SMILES ,note\nCCO,ethanol\n")
    code, out = cli("run", "--recipe", "similar", "--seeds", str(spaced),
                    "--iterations", "1", "--population", "8", "--seed", "3")
    print(f"[SMILES-with-space] exit={code}\n{out}\n")

    # -- empty file
    empty = tmp / "empty.csv"
    empty.write_text("")
    code, out = cli("run", "--recipe", "similar", "--seeds", str(empty),
                    "--iterations", "1")
    print(f"[empty-file] exit={code}\n{out}\n")

    # -- header-only file
    header_only = tmp / "header.csv"
    header_only.write_text("name,smiles\n")
    code, out = cli("run", "--recipe", "similar", "--seeds", str(header_only),
                    "--iterations", "1")
    print(f"[header-only] exit={code}\n{out}\n")

    # -- --iterations 0
    code, out = cli("run", "--iterations", "0")
    print(f"[iterations 0] exit={code}\n{out}\n")

    # -- --iterations 0 with rl
    code, out = cli("run", "--method", "rl", "--iterations", "0")
    print(f"[iterations 0 rl] exit={code}\n{out}\n")

    # -- negative values
    code, out = cli("run", "--iterations", "-3")
    print(f"[iterations -3] exit={code}\n{out}\n")
    code, out = cli("run", "--population", "-5")
    print(f"[population -5] exit={code}\n{out}\n")
    code, out = cli("run", "--top", "-5", "--iterations", "1",
                    "--population", "8")
    print(f"[top -5] exit={code}\n{out}\n")

    # -- huge seed through GA and RL
    code, out = cli("run", "--seed", "18446744073709551616", "--iterations",
                    "1", "--population", "8")
    print(f"[seed 2^64 ga] exit={code}\n{out}\n")
    code, out = cli("run", "--method", "rl", "--seed",
                    "18446744073709551616", "--iterations", "1",
                    "--batch-size", "4", "--prior-epochs", "1")
    print(f"[seed 2^64 rl] exit={code}\n{out}\n")

    # -- --out to an unwritable path (/proc/x)
    code, out = cli("run", "--iterations", "1", "--population", "8",
                    "--out", "/proc/definitely/not/writable.csv")
    print(f"[out unwritable] exit={code}\n{out}\n")

    # -- rl with --population (documented GA-only?)
    code, out = cli("run", "--method", "rl", "--population", "999",
                    "--iterations", "1", "--batch-size", "4",
                    "--prior-epochs", "1")
    print(f"[rl --population 999] exit={code}\n{out}\n")

    # -- --seeds pointing at a .txt (unsupported ext)
    txt = tmp / "seeds.txt"
    txt.write_text("CCO\nCCN\n")
    code, out = cli("run", "--recipe", "similar", "--seeds", str(txt),
                    "--iterations", "1")
    print(f"[seeds .txt] exit={code}\n{out}\n")

    # -- --seed-smiles with garbage
    code, out = cli("run", "--recipe", "similar", "--seed-smiles", "!!!,???",
                    "--iterations", "1", "--population", "8")
    print(f"[garbage seed-smiles] exit={code}\n{out}\n")

print("probe 5 complete")
