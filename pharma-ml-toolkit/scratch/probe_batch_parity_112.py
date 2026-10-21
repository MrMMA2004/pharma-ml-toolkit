"""Scratch probe (100x-112): run the REAL pharmaml write_config_batch /
write_gpf / write_dpf on the same example campaign the web BatchConfigs
card fabricates, so the TS mirror outputs can be diffed line by line."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pharma-ml-toolkit" / "src"))

from pharmaml.docking.campaign import write_config_batch  # noqa: E402

ROOT = Path("/tmp/pharmaml_probe_112_campaign")

ASPIRIN = (
    "REMARK  Name = 01_aspirin\n"
    "REMARK  SMILES: CC(=O)OC1=CC=CC=C1C(=O)O\n"
    "ROOT\n"
    "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.12 C\n"
    "ATOM      2  O   LIG A   1       2.000   0.000   0.000  1.00  0.00    -0.31 OA\n"
    "ENDROOT\n"
    "TORSDOF 1\n"
)
NAPROXEN = (
    "REMARK  Name = 03_naproxen\n"
    "REMARK  SMILES: COC1=CC2=C(C=C1)C(C)CC2=C(C)C(=O)O\n"
    "ROOT\n"
    "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.05 C\n"
    "ATOM      2  C   LIG A   1       3.000   1.000   0.500  1.00  0.00     0.02 C\n"
    "ATOM      3  O   LIG A   1       5.500   0.200   1.500  1.00  0.00    -0.40 OA\n"
    "ENDROOT\n"
    "TORSDOF 2\n"
)
RECEPTOR = (
    "ATOM      1  CA  REC A   1      10.000  10.000  10.000  1.00  0.00     0.10 C\n"
    "ATOM      2  N   REC A   1      12.300  10.500   9.800  1.00  0.00    -0.34 N\n"
)


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    (ROOT / "01_aspirin").mkdir(parents=True)
    (ROOT / "01_aspirin" / "l.pdbqt").write_text(ASPIRIN)
    (ROOT / "02_empty").mkdir()
    (ROOT / "03_naproxen").mkdir()
    (ROOT / "03_naproxen" / "r.pdbqt").write_text(RECEPTOR)
    (ROOT / "03_naproxen" / "l.pdbqt").write_text(NAPROXEN)

    report = write_config_batch(ROOT)
    print(report.to_string(index=False))
    assert report.loc[0, "status"] == "ok", report.loc[0, "status"]
    assert report.loc[1, "status"] == "no ligand PDBQT"
    assert report.loc[2, "status"] == "ok"
    assert report.loc[2, "ligand"] == "l.pdbqt", report.loc[2, "ligand"]
    print("\ncentres:", report["centre"].tolist())

    print("\n--- 01_aspirin/r.gpf ---")
    print((ROOT / "01_aspirin" / "r.gpf").read_text())
    print("--- 01_aspirin/r.dpf ---")
    print((ROOT / "01_aspirin" / "r.dpf").read_text())

    ni = write_config_batch(ROOT, receptor_types="A C HD N Ni OA SA")
    print("--- Ni receptor types line ---")
    for line in (ROOT / "01_aspirin" / "r.gpf").read_text().splitlines():
        if line.startswith("receptor_types"):
            print(line)
    assert ni.loc[0, "status"] == "ok"
    shutil.rmtree(ROOT)
    print("\nPROBE OK")


if __name__ == "__main__":
    main()
