"use client";

import { useEffect, useMemo, useRef } from "react";
import { motion, useInView } from "framer-motion";
import {
  ArrowRight,
  BookOpen,
  Boxes,
  Brain,
  ChartNoAxesColumn,
  CircleCheck,
  Download,
  Dna,
  FileArchive,
  FileCode2,
  Microscope,
  NotebookText,
  Pill,
  ShieldCheck,
  Sigma,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CodeBlock } from "@/components/explorer/code-block";
import { StatChip } from "@/components/explorer/stat-chip";
import { useToolkitStore } from "@/lib/store";
import { fmtBytes, fmtInt } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TreeNode } from "@/lib/toolkit-types";

/** Depth-first lookup of a node by exact path (hero notebook metadata). */
function findNode(node: TreeNode, path: string): TreeNode | null {
  if (node.path === path) return node;
  for (const child of node.children ?? []) {
    const hit = findNode(child, path);
    if (hit) return hit;
  }
  return null;
}

const QUICKSTART = `# --- Bring your own data -----------------------------------------
from pharmaml.data import load_dataset, split_table

ds = load_dataset("my_compounds.csv", target="activity")
print(ds.task)          # 'classification' (2+ classes) or 'regression'
print(ds.class_counts)  # {'active': 120, 'inactive': 190, ...}

# keep string features (assay, cell line) for one-hot encoding:
ds = load_dataset("my_compounds.csv", keep_categorical=True)

# stratified 3-way split (upgrades 'Split CSV file.py'):
parts = split_table("my_compounds.csv", test_size=0.2, val_size=0.1)
parts.write("splits/", prefix="qsar")     # qsar_train/val/test.csv

# --- Feature calculation: SMILES -> descriptors -------------------
from pharmaml.chem import (
    compute_descriptors, compute_fingerprints, tanimoto_matrix,
)

desc = compute_descriptors(df["SMILES"])   # 217 RDKit descriptors
bits, valid = compute_fingerprints(df["SMILES"], kind="morgan")
sim  = tanimoto_matrix(bits, bits)          # Tanimoto heat-map

# --- Format conversion (SMI / CSV / SDF) ------------------------
from pharmaml.chem import smi_to_csv, table_to_sdf, sdf_to_table

smi_to_csv("library.smi", "library.csv")   # streaming, Excel-safe BOM
report = table_to_sdf("ligands.csv", "ligands.sdf",
                      name_col="ID")        # 3D embed + UFF + SD tags
frame = sdf_to_table("ligands.sdf")         # back to a DataFrame

# --- Pharmaceutical calculations -------------------------------
from pharmaml.pharma import (
    creatinine_clearance, bsa_mosteller,
    loading_dose, maintenance_dose, auc_trapezoid,
)
import numpy as np

crcl = creatinine_clearance(age_years=68, weight_kg=70,
                            serum_cr_mg_dl=1.1, sex="male")
bsa  = bsa_mosteller(height_cm=175, weight_kg=70)      # m^2
ld   = loading_dose(vd=45.0, target_concentration=12.0)  # mg
md   = maintenance_dose(clearance_=3.2, target_css=6.0, tau=8.0)

t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
c = np.array([0.0, 8.2, 6.1, 3.4, 1.1, 0.35])
auc = auc_trapezoid(t, c)                              # ug*h/mL

# --- QSAR machine learning --------------------------------------
from pharmaml.ml import (
    train_model, evaluate_model, compare_models,
    make_train_test_split,
)

X_tr, X_te, y_tr, y_te = make_train_test_split(ds.X, ds.y, test_size=0.2)
result = train_model("random_forest", ds.task,
                     X_tr, y_tr, cv=5, tune=True)
print(result.cv_mean, result.cv_std)      # 5-fold ROC-AUC mean +/- std
report = evaluate_model(result.model, X_te, y_te, ds.task)

# --- Docking -----------------------------------------------------
from pharmaml.docking import (
    load_ligand_csv, prepare_ligand_library, prepare_protein,
    compute_grid_box, scan_docking_directory, parse_vina_output,
    ligand_efficiency, consensus_rescore, run_vina_batch,
)

lib = load_ligand_csv("ligands.csv")          # SMILES col auto-detected
prepare_ligand_library(lib, "ligand_folders") # folder-per-ligand + l.pdbqt
prepare_protein("protein.pdb", "r.pdbqt")     # cleaned AD4 receptor
box    = compute_grid_box("ligand_folders/1_aspirin/l.pdbqt", padding=4.0)
summary = scan_docking_directory("docking_run/")   # one row per ligand
poses = parse_vina_output(open("vina.out").read())
cons  = consensus_rescore(summary, columns=["best_lowest_energy"])

# --- Campaign management & Vina batches -------------------------
from pharmaml.docking import (
    write_dpf, check_torsion_consistency, scan_torsion_consistency,
    collect_files, scan_vina_directory, parse_vina_config,
    distribute_files, cleanup_tree, generate_ligand_structures,
    convert_pdb_folder_to_pdbqt, vina_batch_progress,
)

write_dpf("ligand_folders/1/l.pdbqt", ga_run=100)   # AutoDock4 r.dpf
qa = scan_torsion_consistency("ligand_folders/")    # TORSDOF QA frame
scan_vina_directory("vina_results/")               # best affinity per log
collect_files("docking_run/", filename="r.dlg")     # gather artifacts
distribute_files(["receptor.pdbqt"], "ligand_folders/")  # push receptor
cleanup_tree("docking_run/", keep={"r.dlg"}, dry_run=True)  # safe sweep

# --- Ligand structures & resume (the 2026-10 upload twins) -------
structs = generate_ligand_structures(               # nigga.py: tautomer →
    lib, "structures/", num_confs=50,               # ETKDG conformers →
)                                                   # lowest-energy PDB+MOL2
tree = convert_pdb_folder_to_pdbqt(                 # niggaqt.py: MGLTools
    "pdbs/", "pdbqt_tree/", timeout_s=30.0,         # batch, numbered folders
)
run_vina_batch(conf, "pdbqt_tree/", skip_existing=True)  # resume batch
progress = vina_batch_progress(                    # pre-flight census:
    "pdbqt_tree/", "results/"                     # N docked, M to go
)

# --- Ligand-table reconciliation & standardization ---------------
from pharmaml.chem import (
    compare_tables, annotate_with_smiles, merge_table_folder,
    canonicalize_tautomer, smiles_to_pdb,
)

diff = compare_tables("d_summary.csv", "ligands.csv", column="Ligand")
merge_table_folder("batches/", first_columns=["SMILES", "pIC50"])
canonicalize_tautomer("CC(=O)CC(C)=O")   # one canonical tautomer
smiles_to_pdb("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin.pdb")

# --- Visual output: figures + HTML report ------------------------
from pharmaml.viz import (
    plot_confusion_matrix, plot_roc_curves, plot_calibration_curve,
    plot_learning_curve, plot_table_comparison, save_figure,
    export_table, save_visual_bundle,
)

fig = plot_confusion_matrix(report)        # 2+ classes, annotated cells
cal  = plot_calibration_curve(model, X_test, y_test)  # Brier scores
lc   = plot_learning_curve(model, X, y, cv=5)         # more data?
ovl  = plot_table_comparison(diff)         # ligand overlap bar + counts
save_figure(fig, "confusion.png")          # png / svg / pdf
export_table(leaderboard, "board.md")      # csv/tsv/md/json/html

save_visual_bundle(                        # report.html + all files
    figures={"confusion": fig},
    tables={"metrics": metrics_frame},
    metrics=report, out_dir="reports",
)`;

interface DomainCardSpec {
  icon: LucideIcon;
  title: string;
  tagline: string;
  bullets: string[];
  apis: string[];
  accent: string;
}

const DOMAINS: DomainCardSpec[] = [
  {
    icon: Brain,
    title: "Machine Learning",
    tagline: "pharmaml.ml — QSAR end-to-end",
    bullets: [
      "Cross-validated training with optional grid-search tuning",
      "26-entry model registry: sklearn core (incl. the GP regressor and its Laplace-approximated classifier) + optional XGBoost / LightGBM / CatBoost + 2 PyTorch deep families",
      "Metrics, comparison leaderboards & joblib persistence",
    ],
    apis: ["train_model()", "compare_models()", "list_models()"],
    accent: "bg-primary/10 text-primary",
  },
  {
    icon: Pill,
    title: "Pharma Calculations",
    tagline: "pharmaml.pharma — PK · PD · dosage",
    bullets: [
      "Non-compartmental PK: trapezoidal AUC, clearance, half-life",
      "Renal function (Cockcroft-Gault) & BSA-based dosing",
      "Loading / maintenance regimen design with bioavailability",
      "Regulatory-approval filtering from DGIdb TSV exports",
    ],
    apis: ["creatinine_clearance()", "auc_trapezoid()", "filter_approved_drugs()"],
    accent: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  },
  {
    icon: Boxes,
    title: "Molecular Docking",
    tagline: "pharmaml.docking — AutoDock4 & Vina",
    bullets: [
      "Ligand CSV import + protein preparation built in — CSV/PDB/mmCIF in, PDBQT out",
      "Grid-box, GPF & DPF prep, SMILES → 3D PDBQT, Vina/DLG parsers",
      "Ligand efficiency + consensus rescoring, one campaign notebook",
      "Torsion QA, campaign file management, config-driven Vina batches",
    ],
    apis: ["prepare_protein()", "write_gpf()", "write_dpf()", "scan_torsion_consistency()"],
    accent: "bg-teal-500/10 text-teal-600 dark:text-teal-400",
  },
  {
    icon: Microscope,
    title: "Cell Line Biology",
    tagline: "pharmaml.biology — GDSC/CCLE-style screens",
    bullets: [
      "Drug-response screens: load, audit (duplicates, missing metrics), collapse replicates",
      "Expression matrices: log2 / z-score normalisation, imputation, MAD feature selection",
      "Biomarker discovery: per-gene Spearman vs response with Benjamini–Hochberg FDR",
      "Synthetic 24-line × 10-tissue generator with a planted EGFR biomarker",
    ],
    apis: ["load_response_table()", "sensitivity_associations()", "top_biomarkers()"],
    accent: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  },
  {
    icon: Dna,
    title: "Cheminformatics",
    tagline: "pharmaml.chem — lazy RDKit extras",
    bullets: [
      "IC50 / Ki → pIC50 / pKi conversions with unit handling",
      "RDKit descriptors & fingerprints behind lazy imports",
      "SMI / CSV / SDF format conversion (streaming, Excel-safe) + Tanimoto scans",
      "Tautomer standardization, SMILES → 3D PDB ligands, table reconciliation",
    ],
    apis: ["ic50_to_pic50()", "compare_tables()", "smiles_to_pdb()", "tanimoto_from_sets()"],
    accent: "bg-rose-500/10 text-rose-600 dark:text-rose-400",
  },
  {
    icon: ChartNoAxesColumn,
    title: "Visualization & Reports",
    tagline: "pharmaml.viz — every domain, every format",
    bullets: [
      "Confusion, ROC, parity, residual, calibration (Brier) & learning-curve diagnostics",
      "PK profiles, dose accumulation, similarity heat-maps, docking rankings + receptor atom-type census",
      "Self-contained HTML reports + csv/tsv/md/json/html table export",
    ],
    apis: [
      "plot_confusion_matrix()",
      "plot_calibration_curve()",
      "plot_receptor_atom_types()",
      "save_visual_bundle()",
    ],
    accent: "bg-violet-500/10 text-violet-600 dark:text-violet-400",
  },
];

const UPGRADE_ROWS: Array<{ aspect: string; before: string; after: string }> = [
  {
    aspect: "Structure",
    before: "30+ flat, near-duplicate scripts in one folder",
    after: "Typed package pharmaml with 6 subpackages + CLI",
  },
  {
    aspect: "Typing",
    before: "No type hints, no docstrings",
    after: "Full annotations (Python ≥ 3.10 syntax) + Google docstrings",
  },
  {
    aspect: "Logging",
    before: "Bare print() with mixed Persian/English messages",
    after: "Structured logger — console + optional file handler",
  },
  {
    aspect: "Paths",
    before: "Hardcoded C:\\Users\\... Windows paths, os.chdir side effects",
    after: "ToolConfig paths, CLI arguments, no working-directory mutation",
  },
  {
    aspect: "Testing",
    before: "0 tests — changes verified by eyeball",
    after: "5426 passing pytest cases across every domain (incl. viz, notebook & CLI guards, the no-leakage and database-audit regression guards, the applicability-domain suite, the SHAP/Optuna and XAI absorption suites, the runboard/umap/curation/express quartet, the round-189 hpo/pruning/quantization/harness engines with the gnn/gnnxai/featurize orphan suite, the Part-3 §18 hyperparameter-recommendations suite, the Part-3 §19 cross-validation results suite, the round-226 studio CV-interface port suite, the round-228 ML-Workbench + retirement suite, the round-229 §20 checkpoint engine suite (36 engine + 6 integration + 1 GUI resume-loop tests), the round-230 §24 synthetic-accessibility suite (37 engine + 6 GUI + the web bridge pins), and the round-231 pair: the Qt console-noise filter suite (the user-reported v1.2.142 Windows startup noise, 5 tests incl. a live qInstallMessageHandler round-trip) the Part-3 §22 SMILES-preview suite (12 tests over the real MolDraw2DCairo renderer, hover popup, and detail dialog), and the round-232 pair: the Windows platform decision-matrix suite (7 tests pinning the black-window fix — the headless bootstrap now forces offscreen only on displayless Linux, never on Windows/macOS/Wayland desktops) and the Part-3 §21 hover-inspection suite (15 tests over the ChartInspector: crosshair, real-value readout, cached metadata, bar-body snapping, series toggles, click-to-select), and the round-233 Part-3 §23 responsive-policy suite (13 tests: the splitter audit over all 20 studio splitters — non-collapsible children, minimum extents, chart-side stretch — the 640×480 export legibility floor with vector-scene re-render and preserved aspect, the §23.5 data-integrity pin that exports never touch the plotted data, and the ImagePanel aspect lock), and the round-235 Part-3 §23 scroll-policy slice (3 tests: the wrap_in_scroll contract — widget-resizable, sideways overflow made impossible rather than scrollable, frameless, a hard 240 px region minimum — the §23.5 remount-invariants pin (rows untouched, never more space required), and the structural audit that every ML Workbench tab body rides the scroll policy after the probe measured the tuner form at 479 px and Leaderboards at 775 px minimum against the ~450 px the 860×560 minimum window leaves), and the round-236 Part-3 §21.5 molecule-readout suite (4 tests: the PlotPanel point_selected forwarding pin, the SA-chart hover carrying SMILES + RDKit formula + molecular weight beside the SA value, the click-to-detail opening the §22.5 molecule dialog with a foreign-series guard, and the Pareto-front hover readout, and the round-237 launch-hardening suite (7 tests over the Qt import guard and the studio --doctor self-check: both fix-recipe branches, the report format and exit codes, the live 16-check collection, and two PYTHONPATH-shadowed end-to-end legs that fake the exact Windows 'DLL load failed while importing Shiboken' break — exit code 2 with the reinstall recipe, and the doctor diagnosing it without a traceback), and the round-238 Part-3 §23.6 adaptive-tick-density suite (10 tests over PlotPanel's category axes: width-derived stride thinning so no two rendered labels can overlap, pixel-budget ellipsization of the kept labels, the §21.3 full-list hover-meta contract surviving the thinning, the below-floor full-list state, and a shown-panel leg pinning the resizeEvent re-fit wiring), and the round-239 Part-3 §23.7 legend-hygiene suite (11 tests over the PlotPanel legend: pixel-budget label elision with verbatim restoration on widening and the 220px ceiling no width can lift, column escalation capped at 3 for 40 stacked entries, the no-legend no-op, the pinned pyqtgraph-0.13.7 clear-empties behavior with stale-id pruning, the §21 inspector-keys contract surviving the relabel, and fit idempotence), and the round-240 Part-3 §21.4 image pixel-inspection suite (20 tests over the ImagePanel hover probe: the pure pixel_at mapping — exact values, fractional flooring, outside refusal, NaN disclosure, axis labels riding the payload — the scene wiring through mapFromScene with per-pixel debouncing and the leaveEvent reset, the image_metadata at-rest facts, RGB(A) per-channel payloads with the honest NaN refusal, and the Image Lab page readout answering a real engine payload), and the round-241 Part-3 §21 per-point meta-adoption suite (16 tests: the parity/residual/fold/dose/cluster/sample/embedding scatter factories — signed residuals, 0-based source rows, 1-based folds and samples, the original-scale dose behind a log10 axis, the cluster's member count — the out-of-range-empty contract, NaN-bearing inputs, and three inspector-flow pins that the metadata reaches the readout verbatim), and the round-242 Part-3 §23.8 title pixel-budget suite (12 tests over PlotPanel's last unbounded string: the middle elide measured with the live 9pt LabelItem font to the view column's width — both ends kept because titles are centered — the verbatim full title preserved in _title_full + tooltip, BOTH-ways width following, the setTitle-rebuild guard, set_labels adoption, clear() survival, the empty-title no-op, and a shown-panel resize leg), and the round-243 Part-3 §23.9 legend-placement suite (17 tests over the docked legend: the grid row below the bottom axis — never over the data at any width, pinned at three — with the probed size-pin fix (the legacy size=(10, 10) call froze updateSize into a no-op, leaving a 10×10 frame with entries outside it), the 35% band cap with hard maximumHeight/maximumWidth so the band never eats the plot nor widens the grid, the per-column-share label budget following the width both ways with the full names surviving behind the panel, the empty-legend no-band contract with clear() releasing it, and the §21 hover readout re-anchoring above the band through show/resize/clear) — the Tk playground's 980 coupled tests left with it), and the round-244 lint-backlog close (ruff check src/ tests/ ui/ zero findings — the standing hygiene debt the round-243 baseline listed first paid in full: the final 8 all in tests/, two dead F841 assignments removed with the env-applying call kept, two import blocks organized, four unused imports dropped, mypy re-verified clean over 193 source files, no behavioural change — 8 insertions / 10 deletions), and the round-245 Part-3 §23.10 sub-pixel disclosure suite (16 tests: the ImagePanel hover payload now carries image_px_screen — the live on-screen size of one image pixel from 1/ViewBox.viewPixelSize(); zoomed out past 0.9 the readout discloses the ≤ceil coverage '1 screen px spans ≤N×M image px (one shown)', zoomed in past 4.0 it explains '1 image px ≈ N×M screen px (value steady in block)' so the debounced readout no longer looks frozen; the scale refreshes on the next pixel transition after a zoom; four Image Lab page legs append/silence the note), and the round-246 Part-3 §21.4 original-resolution suite (16 tests: the true 1:1 pixel-perfect toggle — set_original_resolution lands the view range where ONE image pixel renders at ONE screen pixel per axis, centred on the current viewport, the achieved scale measured back and disclosed; is_original_resolution tracks the live ≈1:1 band within RESOLUTION_TOLERANCE 0.05; the hidden-panel honest refusal (a hidden widget answers a fake default-geometry scale, ~40 screen px per image px — probed); every Image Lab matrix tab carries the 1:1 button with the fit-restoring second press and the out-loud refusal), and the round-247 multi-objective HPO suite (27 tests over ml/multiobjective.py: the Pareto order — strict/equal/partial dominance with length and finiteness contracts — the pairwise front checked against an independently-coded brute force on seeded 2-D and 4-D clouds with duplicates surviving together, the exact 2-D hypervolume staircase (hand-computed unions, the dominated-adds-zero rule, the WFG reference contract, monotonicity in the point set), NSGA-II crowding (corner infinities, the degenerate-span objective), the Tchebycheff decomposition and its corner weights, the Das-Dennis lattice spread and the seeded simplex draws, the study bookkeeping (the first-trial-is-its-own-ideal scalar of 0, weight cycling, validation contracts, the valley-pair front), the report (the nadir+10% reference, the monotone hypervolume curve, the 3-objective None contract, the tidy frame), and the sklearn adapter over the cross_validate ledger with its deterministic search under seed), and the round-249 molecular-dynamics suite (55 tests over the pharmaml.md package — the user's own GROMACS post-processing pipeline productionized against the MDAnalysis/mdtraj/GROMACS/cpptraj/pytraj/getcontacts/gmx_MMPBSA/ProDy/deeptime/ProLIF reference stack: the bit-faithful .xvg/.ndx parsers with the time-column heuristic and the ten-metric classifier, the Topology/Trajectory selection mini-language with wildcards and boolean algebra, Kabsch-aligned RMSD/RMSF/Rg timelines, contact maps and native contacts Q, g(r) and dihedrals, geometric H-bonds and salt bridges, trajectory PCA and tICA, ANM/GNM elastic network modes, protein–ligand interaction fingerprints, and the ligand-study pipeline from folders to parsed series to correlations to CSV), and the round-250 MD viz-twin suite (29 tests over viz/md_plots.py: the two headline panels of the uploaded pipeline as first-class pharmaml figures — plot_md_series with hand-computed ramp stats (exact population std 0.05·√(399/12)), classifier-driven styling for the ten standard exports plus substring inference and honest generics, the centered moving-average overlay checked against moving_average itself, byte-reproducible saves (identical bytes across two saves, png/svg/pdf), NaN-aware stats, and the full validation contract (window ≥ 2 bounds, non-series inputs, under-supplied mappings, common length < 2); plot_md_correlation with designed ±1 correlations, the min-length alignment disclosure, the constant-series NaN disclosure (2k−1 cells), and the untangled display labels); and the round-252 H-bond-timeline suite (3 tests over md/trajectory.py's synthetic_hbond_trajectory + the bridge op: the designed 12-frame schedule pinned exactly — 0,0,0,0,1,1,1,2,2,1,1,1 with frame 0 breaking both bonds, frame 6 keeping only bond 2 and frame 9 only bond 1 — determinism, the counts re-deriving through the public hbond_timeline, the D1/H1/A1/D2/H2/A2 topology, and the n_frames/noise validation; the web H-bond timeline card rides the new /api/md/hbond route over the same real toolkit functions, and the card-directory census grows to 31 with the md-hbond source mapping pinned), and the round-255 PLUMED-CV suite (65 tests over md/cv.py: the enhanced-sampling collective-variable vocabulary - cv_distance/cv_angle/cv_dihedral on designed geometries (the 3-4-5 triangle apex, the eclipsed/trans dihedrals), cv_coordination hand-counted with the inclusive cutoff edge, cv_rmsd/cv_gyration composed over the metrics module with mass weighting, the CVTable with its axes-must-match refusal and finite-only contract (NaN in a bias target is refused outright, not tolerated-and-disclosed like a figure), byte-exact COLVAR text, and the XvgSeries bridge into the shipped viz twins; the plumed.dat mini-parser pins PLUMED 1-based indices with the atom-0 refusal, a-b range expansion, the MASS flag topology-masses requirement, and honest refusal of everything unsupported (METAD, REFERENCE files, the rational switching keywords) with messages that name the supported set)), and the round-257 salt-bridge-timeline suite (5 tests over md/hbond.py's salt_bridge_timeline + md/trajectory.py's synthetic_salt_bridge_trajectory: the designed 12-frame schedule pinned exactly — 0,0,0,0,0,0,1,2,2,2,2,1, a four-frame two-bridge plateau with bridge 2 leading by a sixth of a period, a deliberately different silhouette from the H-bond demo's narrow 2-frame peak — determinism, the counts re-deriving through the public salt_bridge_timeline, the LYS/GLU/ARG/ASP topology, the validation contract, and the timeline twin's near/far + out-of-range/cutoff refusal pins; the web salt-bridge timeline card rides the new /api/md/saltbridge route over the same real toolkit functions, and the card-directory census grows to 32 with the md-saltbridge source mapping pinned), and the round-258 COLVAR-playground suite (8 tests over the md_bridge 'cv' op + the web wiring: the default five-action plumed.dat snippet pinned to the COLVAR header — #! FIELDS time d_ca bend twist rg fold — seeded determinism, the RMSD-frame-0-is-zero custom snippet with GYRATION MASS riding the demo topology's masses, eight malformed-plumed client errors, the method footnote naming parse_plumed_cvs, and the route/type/card wiring pins; the web COLVAR card rides the new /api/md/cv route over the same real toolkit functions, and the card-directory census grows to 33 with the md-cv source mapping pinned), and the round-260 part4-§4 integrity suite (22 tests over the checkpoint/restore/zip tooling: the dual-layout restore contract — the root-level regression that had broken --restore against every shipped zip since round 210, prefix stripping, the mixed/absolute/traversal/backslash refusals, dry-run zero-writes, the staged version-mismatch refusal, user-data survival — the ship-gate pins — real commands only, the retired playground and its invented --self-test gone, the offscreen pages-smoke GUI gate, the notebook version-freshness contract — the zip policy — both builders exclude scratch/download/dist-archive (37% of the archive was research history), the canonical build utilities relocated to tools/ with scratch/ forwarders — and the metadata pins — eighteen Studio pages with molecular_dynamics in the documented list, no stale (1.2.125), the pyproject homepage equal to the origin remote), and the round-261 part4-§4.5 feature-inventory suite (12 drift-guard pins over the new docs/FEATURES.md traceability matrix: the six honest status classes and the three verbatim honesty rules, the 18 Studio nav keys / 33 web card ids / 27 API routes / 10 packages re-derived LIVE from the registries and asserted present in the matrix, the class-C backend-only honesty check in the other direction, and the status-summary pins 34/7/5/1/1/3), and the round-262 part4-§5 nine-card suite (19 tests: the NINE_CARDS registry — exactly the directive's nine cards in the exact order, every entry routed to a live SECTIONS key, the mapping covering every specialized page with settings excluded by the directive's own table — the rendered 3×3 grid row-major with working entry buttons that navigate the real MainWindow through go_to, the exception-safe dependency probes, the computed live previews, the honesty pins that the docking card names the external Vina engine and the MD card names the external GROMACS/PLUMED data, and five FEATURES.md drift-guard pins re-deriving the nine-card mapping from the live source with the status summary trued to 35/6/5/0/1/3)",
  },
  {
    aspect: "Interface",
    before: "Command-line only",
    after: "CLI + notebooks + studio.py — the PySide6 Studio: eighteen verified pages over every domain (the legacy Tkinter playground retired in v1.2.141; the eighteen-page SECTIONS registry is pinned live by the UI tests and the §4.5 drift-guards)",
  },
  {
    aspect: "Distribution",
    before: "Copy-paste between machines",
    after: "pip-installable extras, Dockerfile, GitHub Actions CI matrix",
  },
];

const DESIGN_CHOICES: Array<{ title: string; rationale: string }> = [
  {
    title: "scikit-learn as the ML backbone",
    rationale:
      "QSAR tabular data is the sweet spot for sklearn — the registry covers every family from logistic to random forests, and ann (MLPClassifier) handles the neural case. Trade-off: no GPU training for tabular models, which is rarely needed at this data scale.",
  },
  {
    title: "Lazily-imported optional dependencies",
    rationale:
      "The package imports cleanly on a bare Python + numpy/pandas/sklearn environment; RDKit, Mordred, XGBoost/LightGBM/CatBoost and Meeko fail at call time with an informative DependencyError including install hints. Trade-off: a slightly more complex import path, worth it for portability.",
  },
  {
    title: "Pure-Python docking parsers",
    rationale:
      "DLG / Vina / PDBQT parsing has zero external dependencies, so result analysis works even on machines without AutoDock installed. Execution wrappers locate binaries via shutil.which and raise DockingError with guidance instead of shell=True surprises.",
  },
  {
    title: "Explicit unit conventions over a unit library",
    rationale:
      "PK functions document their units in the docstring instead of pulling in pint. Trade-off: caller responsibility — but zero-dependency math functions are trivially testable and fast.",
  },
  {
    title: "One shared validation layer",
    rationale:
      "pharmaml.utils.validation raises the same DataValidationError — with the parameter name in the message — from every domain, so one except PharmaMLError clause catches everything and error messages stay uniform.",
  },
];

export function Overview() {
  const setTab = useToolkitStore((s) => s.setTab);
  const ensureTree = useToolkitStore((s) => s.ensureTree);
  const stats = useToolkitStore((s) => s.stats);
  const treeStatus = useToolkitStore((s) => s.treeStatus);
  const tree = useToolkitStore((s) => s.tree);
  const openFileAtLine = useToolkitStore((s) => s.openFileAtLine);

  useEffect(() => {
    void ensureTree();
  }, [ensureTree]);

  /* Sparkline animates only once the featured card scrolls into view. */
  const sparkRef = useRef<HTMLDivElement>(null);
  const sparkInView = useInView(sparkRef, { once: true, margin: "-10% 0px" });

  const fileCount = stats ? fmtInt(stats.files) : "1102";
  const loc = stats ? fmtInt(stats.totalLines) : "334,346";

  /* Live metadata for the featured ML workflow notebook in the hero. */
  const notebookMeta = useMemo(
    () => (tree ? findNode(tree, "examples/complete_ml_workflow.ipynb") : null),
    [tree]
  );
  /* ...and for its bandwidth-friendly no-output twin. */
  const twinMeta = useMemo(
    () => (tree ? findNode(tree, "examples/complete_ml_workflow_no_output.ipynb") : null),
    [tree]
  );

  const heroStagger = {
    hidden: { opacity: 0, y: 14 },
    show: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: { delay: 0.08 + i * 0.07, duration: 0.32, ease: "easeOut" as const },
    }),
  };

  return (
    <div className="space-y-10">
      {/* ------------------------------ Hero ------------------------------ */}
      <section
        aria-labelledby="hero-title"
        className="hero-panel relative space-y-6 rounded-2xl border border-border/70 p-6 shadow-sm sm:p-10"
      >
        <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_300px]">
          {/* -- Left: headline, stats, CTAs -- */}
          <div className="space-y-4">
            <motion.div variants={heroStagger} initial="hidden" animate="show" custom={0}>
              <Badge variant="secondary" className="gap-1.5 border border-primary/25 bg-primary/10 text-primary">
                <ShieldCheck className="size-3.5" aria-hidden />
                Python ≥ 3.10 · numpy / pandas / sklearn core
              </Badge>
            </motion.div>
            <motion.h1
              variants={heroStagger}
              initial="hidden"
              animate="show"
              custom={1}
              id="hero-title"
              className="text-3xl font-bold tracking-tight text-balance sm:text-4xl lg:text-5xl"
            >
              One toolkit.{" "}
              <span className="hero-gradient-text">
                Six scientific domains.
              </span>
            </motion.h1>
            <motion.p
              variants={heroStagger}
              initial="hidden"
              animate="show"
              custom={2}
              className="max-w-2xl text-base text-muted-foreground sm:text-lg"
            >
              A production-ready, typed, tested Python package for computational pharmacy — QSAR
              machine learning, pharmaceutical PK/PD &amp; dosage calculations, and molecular
              docking workflows, upgraded from a flat collection of research scripts.
            </motion.p>

            <motion.div
              variants={heroStagger}
              initial="hidden"
              animate="show"
              custom={3}
              className="grid grid-cols-2 gap-3 pt-2 lg:grid-cols-4"
            >
              <StatChip
                icon={FileCode2}
                value={treeStatus === "loading" ? "…" : fileCount}
                countTo={treeStatus === "loading" ? undefined : stats?.files}
                label="source files"
                hint="Whole package: src, tests, examples, CI"
              />
              <StatChip
                icon={Sigma}
                value={treeStatus === "loading" ? "…" : loc}
                countTo={treeStatus === "loading" ? undefined : stats?.totalLines}
                label="lines of code"
                hint="Counted live from the file tree"
              />
              <StatChip
                icon={CircleCheck}
                value="5426"
                countTo={5426}
                label="passing tests"
                hint="pytest suite covering every domain"
              />
              <StatChip
                icon={Brain}
                value="26"
                countTo={26}
                label="registry models"
                hint="24 sklearn + 2 PyTorch deep families"
              />
            </motion.div>

            <motion.div
              variants={heroStagger}
              initial="hidden"
              animate="show"
              custom={4}
              className="flex flex-wrap gap-3 pt-2"
            >
              <Button
                size="lg"
                className="group h-11 gap-2"
                onClick={() => setTab("explorer")}
              >
                <BookOpen className="size-4" aria-hidden />
                Explore the code
                <ArrowRight
                  className="size-4 transition-transform duration-200 group-hover:translate-x-1"
                  aria-hidden
                />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="group h-11 gap-2"
                onClick={() => setTab("download")}
              >
                <FileArchive
                  className="size-4 transition-transform duration-200 group-hover:translate-y-[-1px]"
                  aria-hidden
                />
                Download ZIP
              </Button>
            </motion.div>
          </div>

          {/* -- Right: featured complete ML workflow notebook (deep-links into the explorer) -- */}
          <motion.aside
            initial={{ opacity: 0, x: 18 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.28, duration: 0.4, ease: "easeOut" }}
            className="featured-card relative overflow-hidden rounded-xl border bg-card/80 shadow-sm"
            aria-label="Featured complete ML workflow notebook"
          >
            <div className="space-y-3 p-5">
              <div className="flex items-center gap-3">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <NotebookText className="size-5" aria-hidden />
                </span>
                <div className="min-w-0">
                  <p className="text-sm font-semibold">Complete ML workflow</p>
                  <p className="break-all font-mono text-[11px] leading-snug text-muted-foreground">
                    examples/complete_ml_workflow.ipynb
                  </p>
                </div>
              </div>
              <p className="text-xs leading-relaxed text-muted-foreground">
                Zero to one hundred in one notebook: import your own file → SMILES →
                descriptors &amp; fingerprints → EDA → split → grid-searched training →
                multiclass evaluation → leaderboard → feature importance → save/load →
                HTML report. All outputs committed, figures inline.
              </p>

              {/* Mock notebook preview: two cells + animated ROC-AUC sparkline. */}
              <div
                ref={sparkRef}
                className={cn(
                  "space-y-1.5 rounded-lg border bg-muted/30 p-2.5",
                  sparkInView && "spark-live"
                )}
                aria-hidden
              >
                <div className="flex items-center gap-1.5 font-mono text-[9px] leading-none text-muted-foreground">
                  <span className="rounded-sm bg-emerald-500/15 px-1 py-0.5 text-emerald-600 dark:text-emerald-400">
                    In [1–38]
                  </span>
                  <span className="ml-auto">15 figures · executed top-to-bottom</span>
                </div>
                <div className="rounded-md border bg-background/70 px-2.5 py-1.5 font-mono text-[10px] leading-relaxed">
                  <span className="text-primary">from</span>{" "}
                  <span className="text-muted-foreground">pharmaml.data</span>{" "}
                  <span className="text-primary">import</span>{" "}
                  <span className="text-muted-foreground">load_dataset</span>
                </div>
                <div className="space-y-1.5 rounded-md border bg-background/70 px-2.5 py-1.5">
                  <div className="font-mono text-[10px] leading-relaxed">
                    <span className="text-primary">plot_confusion_matrix</span>
                    <span className="text-muted-foreground">(report)</span>
                  </div>
                  {/* Sparkline: CV ROC-AUC progression across the workflow stages. */}
                  <svg
                    viewBox="0 0 230 36"
                    className="block h-9 w-full"
                    preserveAspectRatio="none"
                    role="presentation"
                  >
                    <defs>
                      <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.22" />
                        <stop offset="100%" stopColor="var(--primary)" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                    <path
                      d="M0,20.2 L32,16.6 L64,14.4 L96,12.1 L128,9 L160,7.4 L192,5.7 L224,4.8 L224,36 L0,36 Z"
                      fill="url(#sparkFill)"
                    />
                    <polyline
                      points="0,20.2 32,16.6 64,14.4 96,12.1 128,9 160,7.4 192,5.7 224,4.8"
                      fill="none"
                      stroke="var(--primary)"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="sparkline-draw"
                    />
                    <circle
                      cx="224"
                      cy="4.8"
                      r="2.75"
                      fill="var(--primary)"
                      className="spark-dot"
                    />
                    <circle
                      cx="224"
                      cy="4.8"
                      r="5.5"
                      fill="var(--primary)"
                      className="spark-dot-pulse"
                    />
                  </svg>
                  <div className="flex items-center justify-between font-mono text-[9px] leading-none text-muted-foreground">
                    <span>CV ROC-AUC · stages 1→13</span>
                    <span className="font-semibold text-primary">0.50 → 0.92</span>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
                <span className="rounded-md border bg-muted/40 px-1.5 py-0.5 font-mono tabular-nums">
                  {notebookMeta ? fmtBytes(notebookMeta.size) : "…"}
                </span>
                {typeof notebookMeta?.lines === "number" ? (
                  <span className="rounded-md border bg-muted/40 px-1.5 py-0.5 font-mono tabular-nums">
                    {fmtInt(notebookMeta.lines)} lines
                  </span>
                ) : null}
                <span
                  className="rounded-md border border-primary/25 bg-primary/5 px-1.5 py-0.5 font-mono tabular-nums text-primary"
                  title="Same cells, all outputs stripped — the lightweight twin shipped next to the executed notebook"
                >
                  {twinMeta ? `${fmtBytes(twinMeta.size)} no-output twin` : "no-output twin available"}
                </span>
                <span className="rounded-md border bg-primary/10 px-1.5 py-0.5 font-mono text-primary">
                  Jupyter
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="h-9 w-full gap-1.5 text-xs"
                onClick={() => {
                  openFileAtLine("examples/complete_ml_workflow.ipynb", null);
                  setTab("explorer");
                }}
              >
                <FileCode2 className="size-3.5" aria-hidden />
                Open in the Code Explorer
                <ArrowRight className="size-3.5" aria-hidden />
              </Button>
            </div>
          </motion.aside>
        </div>
      </section>

      {/* --------------------------- Domain cards --------------------------- */}
      <section aria-labelledby="domains-title" className="space-y-4">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow">The package</p>
            <h2 id="domains-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
              Six domains, one import
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Every module ships with type hints, docstrings, structured logging and tests.
            </p>
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {DOMAINS.map((d, i) => (
            <motion.div
              key={d.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06, duration: 0.25 }}
              whileHover={{ y: -4 }}
              className="h-full"
            >
              <Card className="h-full border-border/80 shadow-sm">
                <CardHeader className="pb-3">
                  <span
                    className={`mb-2 flex size-10 items-center justify-center rounded-lg ${d.accent}`}
                  >
                    <d.icon className="size-5" aria-hidden />
                  </span>
                  <CardTitle className="text-base">{d.title}</CardTitle>
                  <CardDescription className="font-mono text-xs">{d.tagline}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 pb-4">
                  <ul className="space-y-1.5 text-sm text-muted-foreground">
                    {d.bullets.map((b) => (
                      <li key={b} className="flex gap-2">
                        <CircleCheck className="mt-0.5 size-3.5 shrink-0 text-primary" aria-hidden />
                        <span>{b}</span>
                      </li>
                    ))}
                  </ul>
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {d.apis.map((api) => (
                      <code
                        key={api}
                        className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-foreground/80"
                      >
                        {api}
                      </code>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ------------------------- Upgrade comparison ------------------------- */}
      <section aria-labelledby="upgrade-title" className="space-y-4">
        <div>
          <p className="eyebrow">The re-engineering</p>
          <h2 id="upgrade-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
            Before → After: the upgrade
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            What changed when the flat script collection was re-engineered into the{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">pharmaml</code>{" "}
            package.
          </p>
        </div>
        <Card className="overflow-hidden py-0 shadow-sm">
          <div className="thin-scrollbar overflow-x-auto">
            <Table className="data-table">
              <TableHeader>
                <TableRow className="bg-muted/40 hover:bg-muted/40">
                  <TableHead className="w-[110px] font-semibold">Aspect</TableHead>
                  <TableHead className="font-semibold text-rose-600 dark:text-rose-400">
                    Before — flat scripts
                  </TableHead>
                  <TableHead className="font-semibold text-primary">
                    After — pharmaml package
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {UPGRADE_ROWS.map((row) => (
                  <TableRow key={row.aspect}>
                    <TableCell className="whitespace-nowrap font-medium">{row.aspect}</TableCell>
                    <TableCell className="text-muted-foreground">{row.before}</TableCell>
                    <TableCell>{row.after}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>
      </section>

      {/* --------------------------- Design choices --------------------------- */}
      <section aria-labelledby="design-title" className="space-y-4">
        <div>
          <p className="eyebrow">Architecture rationale</p>
          <h2 id="design-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
            Design choices &amp; trade-offs
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            The reasoning behind the architecture — straight from the README.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {DESIGN_CHOICES.map((choice, i) => (
            <motion.div
              key={choice.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.25 }}
              whileHover={{ y: -3 }}
            >
              <Card className="h-full border-border/80 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-start gap-2 text-sm leading-snug">
                    <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/10 font-mono text-[11px] font-semibold text-primary">
                      {i + 1}
                    </span>
                    {choice.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {choice.rationale}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ------------------------------ Quickstart ------------------------------ */}
      <section aria-labelledby="quickstart-title" className="space-y-4">
        <div>
          <p className="eyebrow">Get running</p>
          <h2 id="quickstart-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
            Quickstart
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            One import per domain — deterministic examples, no external datasets required.
          </p>
        </div>
        <CodeBlock code={QUICKSTART} language="python" fileName="quickstart.py" showLineNumbers maxHeightClass="max-h-[520px]" />
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground sm:gap-x-4">
          <span>Want the full picture?</span>
          <Button variant="link" className="h-auto gap-1 p-0 text-sm" onClick={() => setTab("explorer")}>
            Browse the actual source <ArrowRight className="size-3.5" aria-hidden />
          </Button>
          <span aria-hidden className="hidden text-border sm:inline">·</span>
          <Button variant="link" className="h-auto gap-1 p-0 text-sm" onClick={() => setTab("reference")}>
            Browse the API reference <ArrowRight className="size-3.5" aria-hidden />
          </Button>
        </div>
      </section>

      {/* ------------------------------ Download CTA ------------------------------ */}
      <section
        aria-labelledby="dl-cta-title"
        className="rounded-xl border border-primary/20 bg-gradient-to-br from-primary/5 via-transparent to-amber-500/5 p-6 sm:p-8"
      >
        <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
          <div className="space-y-1">
            <h2 id="dl-cta-title" className="flex items-center gap-2 text-lg font-semibold">
              <Download className="size-5 text-primary" aria-hidden />
              Get the whole toolkit
            </h2>
            <p className="text-sm text-muted-foreground">
              Full source, tests, examples, Dockerfile and CI — packaged as a single ZIP.
            </p>
          </div>
          <Button size="lg" className="h-11 gap-2" onClick={() => setTab("download")}>
            <FileArchive className="size-4" aria-hidden />
            Go to download
          </Button>
        </div>
      </section>
    </div>
  );
}
