/**
 * Scratch probe (100x-112): run the TS mirrors (src/lib/docking-configs.ts)
 * on the SAME example campaign the Python probe used, and byte-diff the
 * generated r.gpf / r.dpf texts against the Python writers' output.
 */

import {
  computeGridBox,
  extractPdbqtAtomTypes,
  extractPdbqtCoords,
  readTorsdof,
  scanVinaLogs,
  writeConfigBatch,
} from "../src/lib/docking-configs";

const ASPIRIN = `REMARK  Name = 01_aspirin
REMARK  SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.12 C
ATOM      2  O   LIG A   1       2.000   0.000   0.000  1.00  0.00    -0.31 OA
ENDROOT
TORSDOF 1
`;

const NAPROXEN = `REMARK  Name = 03_naproxen
REMARK  SMILES: COC1=CC2=C(C=C1)C(C)CC2=C(C)C(=O)O
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.05 C
ATOM      2  C   LIG A   1       3.000   1.000   0.500  1.00  0.00     0.02 C
ATOM      3  O   LIG A   1       5.500   0.200   1.500  1.00  0.00    -0.40 OA
ENDROOT
TORSDOF 2
`;

const RECEPTOR = `ATOM      1  CA  REC A   1      10.000  10.000  10.000  1.00  0.00     0.10 C
ATOM      2  N   REC A   1      12.300  10.500   9.800  1.00  0.00    -0.34 N
`;

/* --- expected outputs, captured from the Python probe run ------------ */

const EXPECTED_GPF = `npts 60 60 60          # num.grid points in xyz
gridfld r.maps.fld                    # grid_data_file
spacing 0.375                        # spacing(A)
receptor_types A C HD N NA OA SA     # receptor atom types
ligand_types C OA            # ligand atom types
receptor r.pdbqt                     # macromolecule
gridcenter 1.0 0.0 0.0      # xyz-coordinates
smooth 0.5                           # store minimum energy w/in rad(A)
map r.C.map                          # atom-specific affinity map
map r.OA.map                          # atom-specific affinity map
elecmap r.e.map                      # electrostatic potential map
dsolvmap r.d.map              # desolvation potential map
dielectric -0.1465                   # <0, AD4 distance-dep.diel;>0, constant
`;

const EXPECTED_DPF = `autodock_parameter_version 4.2
outlev 1
intelec
seed pid time
ligand_types C OA
fld r.maps.fld
map r.C.map
map r.OA.map
elecmap r.e.map
desolvmap r.d.map
move l.pdbqt
about 1.0 0.0 0.0
tran0 random
quaternion0 random
dihe0 random
torsdof 1
rmstol 2.0
extnrg 1000.0
e0max 0.0 10000
ga_pop_size 150
ga_num_evals 2500000
ga_num_generations 27000
ga_elitism 1
ga_mutation_rate 0.02
ga_crossover_rate 0.8
ga_window_size 10
ga_cauchy_alpha 0.0
ga_cauchy_beta 1.0
set_ga
sw_max_its 300
sw_max_succ 4
sw_max_fail 4
sw_rho 1.0
sw_lb_rho 0.01
ls_search_freq 0.06
set_psw1
unbound_model bound
ga_run 100
analysis
`;

const EXPECTED_NI_LINE =
  "receptor_types A C HD N Ni OA SA     # receptor atom types";

/* --- unit-level checks ----------------------------------------------- */

let failures = 0;
function check(label: string, ok: boolean, detail?: string) {
  if (!ok) {
    failures += 1;
    console.error(`FAIL  ${label}${detail ? `\n      ${detail}` : ""}`);
  } else {
    console.log(`ok    ${label}`);
  }
}

const aspirinTypes = extractPdbqtAtomTypes(ASPIRIN);
check("atom types C OA", aspirinTypes.join(" ") === "C OA", aspirinTypes.join(" "));

const aspirinCoords = extractPdbqtCoords(ASPIRIN);
check(
  "coords fixed-column",
  aspirinCoords.length === 2 && aspirinCoords[0]!.join(",") === "0,0,0" && aspirinCoords[1]!.join(",") === "2,0,0",
  JSON.stringify(aspirinCoords)
);

check("torsdof 1", readTorsdof(ASPIRIN) === 1);
check("torsdof 2", readTorsdof(NAPROXEN) === 2);

const box = computeGridBox(aspirinCoords, 4.0, [60, 60, 60], 0.375);
check("box centre 1,0,0", box.centerX === 1 && box.centerY === 0 && box.centerZ === 0);
check("box npts 60³ (small ligand)", box.npts.join(",") === "60,60,60");
check("box size 22.125", box.size[0] === 22.125);

/* a ligand that forces growth: 30 Å extent + 8 Å padding = 38 Å →
 * ceil(38/0.375)+1 = 103 grid points on x */
const big = Array.from({ length: 31 }, (_, i) => [i, 0, 0]);
const bigBox = computeGridBox(big, 4.0, [60, 60, 60], 0.375);
check("box grows to 103 on x", bigBox.npts[0] === 103, String(bigBox.npts[0]));

/* --- batch parity ----------------------------------------------------- */

const report = writeConfigBatch(
  [
    { folder: "01_aspirin", files: [{ name: "l.pdbqt", content: ASPIRIN }] },
    { folder: "02_empty", files: [] },
    {
      folder: "03_naproxen",
      files: [
        { name: "r.pdbqt", content: RECEPTOR },
        { name: "l.pdbqt", content: NAPROXEN },
      ],
    },
  ],
  {
    receptorFile: "r.pdbqt",
    padding: 4.0,
    npts: [60, 60, 60],
    spacing: 0.375,
    receptorTypes: null,
    gaRun: 100,
    gaPopSize: 150,
  }
);

check("3 rows", report.length === 3);
check("row 0 ok", report[0]!.status === "ok", report[0]!.status);
check("row 1 no ligand PDBQT", report[1]!.status === "no ligand PDBQT", report[1]!.status);
check("row 2 ok", report[2]!.status === "ok", report[2]!.status);
check("row 2 ligand = l.pdbqt (receptor excluded)", report[2]!.ligand === "l.pdbqt", report[2]!.ligand);
check(
  "row 0 centre '1.000 0.000 0.000'",
  report[0]!.centre === "1.000 0.000 0.000",
  report[0]!.centre
);
check(
  "row 2 centre '2.750 0.500 0.750'",
  report[2]!.centre === "2.750 0.500 0.750",
  report[2]!.centre
);
check("GPF byte-parity", report[0]!.gpfText === EXPECTED_GPF, JSON.stringify(report[0]!.gpfText));
check("DPF byte-parity", report[0]!.dpfText === EXPECTED_DPF, JSON.stringify(report[0]!.dpfText));

const ni = writeConfigBatch(
  [{ folder: "01_aspirin", files: [{ name: "l.pdbqt", content: ASPIRIN }] }],
  {
    receptorFile: "r.pdbqt",
    padding: 4.0,
    npts: [60, 60, 60],
    spacing: 0.375,
    receptorTypes: "A C HD N Ni OA SA",
    gaRun: 100,
    gaPopSize: 150,
  }
);
const niLine = (ni[0]!.gpfText ?? "").split("\n").find((l) => l.startsWith("receptor_types"));
check("Ni receptor types line", niLine === EXPECTED_NI_LINE, String(niLine));

/* --- vina scan mirror -------------------------------------------------- */

const LOG_A = `AutoDock Vina v1.2.5
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b. | rmsd u.b.
-----+------------+----------+----------
   1        -7.9      0.000      0.000
   2        -7.4      1.204      2.31`;

const CRASHED = `AutoDock Vina v1.2.5
^C`;

const scan = scanVinaLogs([
  { path: "lig01.log", content: LOG_A },
  { path: "sub/lig02.log", content: LOG_A.replace(/-7\.9/g, "-8.35").replace(/-7\.4/g, "-8.1") },
  { path: "crashed.log", content: CRASHED },
]);

check("scan 3 rows", scan.length === 3);
check("scan best first (lig02)", scan[0]!.ligand === "lig02", scan.map((r) => r.ligand).join(","));
check("scan best -8.35", scan[0]!.bestAffinity === -8.35, String(scan[0]!.bestAffinity));
check("scan sub path kept", scan[0]!.log === "sub/lig02.log", scan[0]!.log);
check("scan crashed last with null", scan[2]!.bestAffinity === null && scan[2]!.ligand === "crashed");
check("scan nPoses 2", scan[1]!.nPoses === 2, String(scan[1]!.nPoses));

if (failures > 0) {
  console.error(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log("\nTS PROBE OK — all mirrors match the Python writers byte-for-byte");
