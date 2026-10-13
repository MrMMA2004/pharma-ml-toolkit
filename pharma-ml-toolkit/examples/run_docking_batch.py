"""Docking-workflow example (dry-run parsing demo included).

The full docking loop needs AutoDock binaries and a receptor, so this
example demonstrates the two halves that are runnable anywhere:

1. GPF/grid preparation API (shows the objects, skips file writes for
   binaries we do not have),
2. DLG/Vina result parsing + ligand efficiency + consensus rescoring
   on bundled sample outputs.

For a real campaign::

    from pharmaml.docking import run_vina, scan_docking_directory
    run_vina(receptor="r.pdbqt", ligand="l.pdbqt", out="out.pdbqt",
             center=(-17.7, -0.8, 0.6), size=(22.5, 22.5, 22.5))
    summary = scan_docking_directory("docking_run/")

Usage::

    python examples/run_docking_batch.py
"""

from __future__ import annotations

import pandas as pd

from pharmaml.docking.parsing import parse_dlg_cluster_table, parse_vina_output
from pharmaml.docking.preparation import GridBox, compute_grid_box, write_gpf
from pharmaml.docking.scoring import consensus_rescore, ligand_efficiency
from pharmaml.logger import configure_logging

def _pdbqt_atom(
    serial: int, name: str, x: float, y: float, z: float, atom_type: str
) -> str:
    """Format one PDBQT ATOM record with strict PDB column alignment."""
    return (
        f"ATOM  {serial:5d} {name:>4s} LIG A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}    0.000 {atom_type:>2s}"
    )


SAMPLE_LIGAND_PDBQT = """\
REMARK  Name = sample_ligand
ROOT
{atom1}
{atom2}
{atom3}
{atom4}
ENDROOT
TORSDOF 2
""".format(
    atom1=_pdbqt_atom(1, "C", 2.345, -0.802, 0.645, "C"),
    atom2=_pdbqt_atom(2, "O", 3.456, -1.234, 1.789, "OA"),
    atom3=_pdbqt_atom(3, "N", 1.234, 0.123, -0.456, "N"),
    atom4=_pdbqt_atom(4, "HD", 0.987, 1.012, -1.234, "HD"),
)

SAMPLE_DLG = """\
AutoDock log file (synthetic sample)

DPF> for demonstration only

________
CLUSTERING HISTOGRAM
__________________
______________________________________________________________________
Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster
_____________|_______________|_____|_____________|________________
     1       |     -11.5     |  3  |    -11.2    |      7 |
     2       |     -10.8     |  9  |    -10.5    |     34 |
     3       |      -9.9     | 17  |     -9.7    |     52 |
     4       |      -9.1     | 22  |     -9.0    |     12 |
____________|_______________|_____|_____________|________________
"""

SAMPLE_VINA = """\
Estimated Free Energy of Binding   = -10.3 kcal/mol
Modes | Affinity | dist from best mode
      |          | rmsd l.b. | rmsd u.b.
------+----------+----------+----------
   1      -10.3          0          0
   2       -9.6      1.842      2.301
   3       -9.4      2.105      2.777
   4       -8.9      2.733      3.418
------+----------+----------+----------
"""


def main() -> None:
    configure_logging()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ---- 1. Grid box from a reference ligand ----------------------
        ligand_file = tmp_path / "l.pdbqt"
        ligand_file.write_text(SAMPLE_LIGAND_PDBQT, encoding="utf-8")
        box = compute_grid_box(ligand_file, padding=4.0)
        print(f"Grid centre:      ({box.center_x:.3f}, {box.center_y:.3f}, "
              f"{box.center_z:.3f})")
        print(f"Box size (A):     {tuple(round(s, 2) for s in box.size)}")

        # ---- 2. GPF writing ---------------------------------------------
        gpf = write_gpf(ligand_file, receptor_file="r.pdbqt", box=box)
        print(f"GPF written:      {gpf.name} "
              f"({len(gpf.read_text().splitlines())} lines)")

        # ---- 3. DLG parsing ---------------------------------------------
        clusters = parse_dlg_cluster_table(SAMPLE_DLG)
        table = pd.DataFrame(
            [
                {
                    "ligand": f"ligand_{c.rank:02d}",
                    "best_lowest_energy": c.lowest_energy,
                    "best_num_in_cluster": c.num_in_cluster,
                    "num_heavy_atoms": 18 + 3 * (c.rank % 3),
                }
                for c in clusters
            ]
        )

        # ---- 4. Ligand efficiency ----------------------------------------
        table["ligand_efficiency"] = table.apply(
            lambda row: ligand_efficiency(row["best_lowest_energy"], row["num_heavy_atoms"]),
            axis=1,
        )

        # ---- 5. Consensus rescoring ---------------------------------------
        table["consensus"] = consensus_rescore(
            table,
            columns=["best_lowest_energy", "best_num_in_cluster", "ligand_efficiency"],
            weights=[0.5, 0.25, 0.25],
        )
        print("\n=== Docking summary with consensus rescoring ===")
        print(table.to_string(index=False))
        best = table.sort_values("consensus", ascending=False).iloc[0]
        print(f"\nTop ligand by consensus: {best['ligand']} "
              f"(score {best['consensus']:.3f})")

        # ---- 6. Vina parsing -----------------------------------------------
        poses = parse_vina_output(SAMPLE_VINA)
        print(f"\nVina poses parsed: {len(poses)} (best affinity "
              f"{poses[0].affinity} kcal/mol)")


if __name__ == "__main__":
    main()
