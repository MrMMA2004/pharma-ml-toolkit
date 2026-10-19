/**
 * Realistic sample docking outputs for the playground parser.
 * Embedded on the client so "Load sample" works instantly.
 */

import type { DockingFormat } from "@/lib/toolkit-types";

export const VINA_SAMPLE = `AutoDock Vina v1.2.5
#################################################################
# If you used AutoDock Vina in your work, please cite:          #
#                                                               #
# J. Eberhardt, D. Santos-Martins, A. F. Tillack, and S. Forli   #
# AutoDock Vina 1.2.0: New Docking Methods, Expanded Binding    #
# Force Field, and Python Bindings, J. Chem. Inf. Model.        #
# (2021) DOI: 10.1021/acs.jcim.1c00203                         #
#                                                               #
# O. Trott, A. J. Olson,                                        #
# AutoDock Vina: improving the speed and accuracy of docking    #
# with a new scoring function, efficient optimization, and      #
# multithreading, J. Comp. Chem. (2010)                         #
# DOI: 10.1002/jcc.21334                                        #
#                                                               #
# Please see https://github.com/ccsb-scripps/AutoDock-Vina for  #
# more information.                                             #
#################################################################

Computing Vina grid ... done.
Performing docking (random seed: 42) ... done.
Estimated time of arrival (ETA): 8 seconds (44.941s elapsed)

mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b. | rmsd u.b.
-----+------------+----------+----------
   1        -10.3      0.000      0.000
   2        -9.8       1.386      2.554
   3        -9.5       1.873      3.112
   4        -9.1       2.308      3.882
   5        -8.9       2.642      4.408
   6        -8.7       3.171      5.203
   7        -8.4       3.594      5.789
   8        -8.2       4.033      6.455
   9        -8.0       4.572      7.319`;

export const DLG_SAMPLE = `AutoDock 4.2.6 — docking log file
Ligand: ligand_47.pdbqt   Receptor: receptor.pdbqt
DPF: ligand_47.dpf   Grid maps: maps.fld

OUTPUT:    47    -10.383    -9.154    -4.119    0.000    0.000

 final intermolecular energy =  -10.383 kcal/mol
 final total internal energy  =   -9.154 kcal/mol
 torsional free energy        =   -1.689 kcal/mol
 unbound system's energy      =   -4.119 kcal/mol

CLUSTERING HISTOGRAM
__________________

______________________________________________________________________
| Cluster Rank |  Lowest Energy  |  Run  |  Mean Energy  |  # in cluster |
|______________|_________________|_______|________________|_______________|
|      1       |     -10.383     |  47   |    -10.211     |       5       |
|      2       |      -9.862     |  12   |     -9.744     |       3       |
|      3       |      -9.415     |  31   |     -9.302     |       4       |
|      4       |      -8.912     |   3   |     -8.851     |       2       |
|      5       |      -8.447     |  86   |     -8.447     |       1       |
|      6       |      -8.103     |  22   |     -8.059     |       2       |
|      7       |      -7.852     |  55   |     -7.852     |       1       |
|      8       |      -7.518     |  71   |     -7.466     |       2       |

______________________________________________________________________
|    NOT CLUSTERED   |    -7.201  |   14   |    -7.201    |      1       |
|____________________|____________|________|______________|______________|`;

export const CSV_SAMPLE = `pose,ligand,score_vina,score_ad4
1,ligand_A,-10.2,-11.4
2,ligand_A,-9.4,-10.1
3,ligand_B,-9.8,-12.2
4,ligand_B,-9.1,-10.9
5,ligand_C,-8.7,-9.6
6,ligand_C,-8.2,-9.1
7,ligand_D,-7.9,-8.8
8,ligand_D,-7.3,-8.2`;

export function getSample(format: DockingFormat): string {
  switch (format) {
    case "vina":
      return VINA_SAMPLE;
    case "dlg":
      return DLG_SAMPLE;
    case "csv":
      return CSV_SAMPLE;
  }
}
