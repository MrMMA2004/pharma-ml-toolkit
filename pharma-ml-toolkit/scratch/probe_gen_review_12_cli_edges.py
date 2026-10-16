"""100x-21-b own-hunt probe C: CLI edges (mission item 5 + own hunts).

--iterations 0, huge/negative --seed, --top negative, empty seeds file,
seeds file whose smiles column is entirely NaN, --seed-smiles "" / "CCO,,",
--seeds with extra unknown columns, unwritable --out, help text quality.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PY = sys.executable


def cli(args: list[str], label: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [PY, "-m", "pharmaml.cli", *args],
            capture_output=True, text=True, timeout=180, cwd=tmp,
        )
        err = proc.stderr.strip().splitlines()
        out = proc.stdout.strip().splitlines()
        tail = (err[-1] if err else out[-1] if out else "")[:150]
        print(f"exit={proc.returncode} {label}: {tail}")


TMP = Path(tempfile.mkdtemp())
EMPTY = TMP / "empty.csv"
EMPTY.write_text("")
NAN_COL = TMP / "nan_col.csv"
NAN_COL.write_text("name,smiles\naspirin,\nethanol,\n")
EXTRA = TMP / "extra.csv"
EXTRA.write_text("name,smiles,ic50\naspirin,CC(=O)Oc1ccccc1C(=O)O,3.2\nethanol,CCO,999\n")
SPACE_HDR = TMP / "space_hdr.csv"
SPACE_HDR.write_text('name,"smiles "\naspirin,CCO\n')

cli(["generative", "run", "--recipe", "qed", "--iterations", "0"], "iterations=0 (ga)")
cli(["generative", "run", "--method", "rl", "--recipe", "qed",
     "--iterations", "0", "--prior-epochs", "1"], "iterations=0 (rl)")
cli(["generative", "run", "--recipe", "qed", "--iterations", "2",
     "--population", "8", "--seed", str(2**64)], "seed=2**64 (ga)")
cli(["generative", "run", "--method", "rl", "--recipe", "qed", "--iterations", "1",
     "--prior-epochs", "1", "--seed", str(2**64)], "seed=2**64 (rl)")
cli(["generative", "run", "--recipe", "qed", "--iterations", "1",
     "--population", "8", "--seed", "-7"], "seed=-7 (ga)")
cli(["generative", "run", "--recipe", "qed", "--iterations", "2",
     "--population", "8", "--seed", "3", "--top", "-5"], "top=-5")
cli(["generative", "run", "--recipe", "qed", "--iterations", "1",
     "--population", "8", "--seed", "3", "--top", "0"], "top=0")
cli(["generative", "run", "--recipe", "similar", "--seeds", str(EMPTY),
     "--iterations", "1"], "empty seeds file")
cli(["generative", "run", "--recipe", "similar", "--seeds", str(NAN_COL),
     "--iterations", "1"], "all-NaN smiles column")
cli(["generative", "run", "--recipe", "similar", "--seeds", str(EXTRA),
     "--iterations", "1", "--population", "8"], "extra ic50 column")
cli(["generative", "run", "--recipe", "similar", "--seeds", str(SPACE_HDR),
     "--iterations", "1", "--population", "8"], "'smiles ' header with space")
cli(["generative", "run", "--recipe", "similar", "--seed-smiles", "",
     "--iterations", "1"], "seed-smiles empty string")
cli(["generative", "run", "--recipe", "similar", "--seed-smiles", "CCO,,",
     "--iterations", "1", "--population", "8"], "seed-smiles 'CCO,,'")
cli(["generative", "run", "--recipe", "qed", "--iterations", "1", "--population", "8",
     "--out", "/proc/definitely/not/writable.csv"], "out unwritable path")
cli(["generative", "run", "--recipe", "qed", "--iterations", "1",
     "--population", "2"], "population=2 (below floor)")
cli(["generative", "run", "--method", "rl", "--recipe", "qed",
     "--batch-size", "1", "--iterations", "1", "--prior-epochs", "1"],
    "rl batch-size=1")
cli(["generative", "run", "--method", "rl", "--recipe", "qed",
     "--prior-epochs", "0", "--iterations", "1"], "rl prior-epochs=0")
cli(["generative", "--help"], "generative --help")
cli(["generative", "run", "--help"], "generative run --help")
cli(["generative", "corpus", "--out", "/proc/definitely/nope.csv"], "corpus out unwritable")
print("probe C complete")
