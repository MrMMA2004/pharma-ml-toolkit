/* ------------------------------------------------------------------ */
/* Playground card directory (100x-129)                                */
/* ------------------------------------------------------------------ */
/**
 * The web twin of the desktop Ctrl+F card finder: every ALWAYS-RENDERED
 * card of the Playground, one row each, with the sub-tab (and PK inner
 * tab) it lives on and the exact `match` string its rendered card
 * carries. The finder filters this directory, switches the tabs, then
 * scrolls to the card and flashes it — `match` is resolved against the
 * live `[data-slot="card"]` elements' textContent, so a rename drift
 * surfaces honestly as a "not found" toast instead of a silent jump to
 * the wrong card.
 *
 * (Historically the desktop registry was the retired Tk playground's
 * `_CARD_INDEX`, built at UI construction time so it could never drift; the
 * Studio pages carry those cards now.) The web catalog is static,
 * so the E2E probe walks EVERY entry and asserts the jump resolves —
 * drift is caught the round it happens, not by the user.
 *
 * Cards that only render AFTER a run (the ML results / leaderboard /
 * comparison / tuning readouts, the biology matrix cards) are
 * deliberately absent: the finder lists what is on screen NOW, the
 * same honesty rule as the desktop finder listing the cards it built.
 */

/** Sub-tab values of the Playground tab strip. */
export type PlaygroundTabId =
  | "calculators"
  | "pk"
  | "ml"
  | "biology"
  | "docking"
  | "features"
  | "ligandprep"
  | "reconciler"
  | "generative"
  | "md";

export interface PlaygroundCardEntry {
  /** Stable slug (also the finder row's React key). */
  id: string;
  /** Sub-tab the card lives on (null: rides above the strip, e.g. the
   * session-report card — no sub-tab switch on jump). */
  tab: PlaygroundTabId | null;
  /** Inner tab of the PK / PD Explorer, when the card lives on one. */
  inner?: "regimen" | "pd" | "nca";
  /** Display title (the card's own heading). */
  title: string;
  /** One-line blurb shown under the title in the finder rows. */
  blurb: string;
  /** Unique substring of the card's rendered text — the jump anchor. */
  match: string;
  /** Extra fuzzy-match target (synonyms the title does not carry). */
  keywords?: string;
}

export const PLAYGROUND_CARDS: readonly PlaygroundCardEntry[] = [
  /* -- Calculators ------------------------------------------------- */
  {
    id: "crcl",
    tab: "calculators",
    title: "Creatinine clearance",
    blurb: "Cockcroft–Gault renal dosing anchor",
    match: "Creatinine clearance",
    keywords: "cockcroft gault renal kidney crcl",
  },
  {
    id: "bsa",
    tab: "calculators",
    title: "Body surface area",
    blurb: "Mosteller formula",
    match: "Body surface area",
    keywords: "mosteller bsa surface",
  },
  {
    id: "pediatric",
    tab: "calculators",
    title: "Pediatric dose",
    blurb: "mg/kg scaling with optional adult cap",
    match: "Pediatric dose",
    keywords: "child weight mgkg scaling",
  },
  {
    id: "loading",
    tab: "calculators",
    title: "Loading & maintenance dose",
    blurb: "bolus + steady-state maintenance pair",
    match: "Loading & maintenance dose",
    keywords: "bolus maintenance steady state",
  },
  {
    id: "halflife",
    tab: "calculators",
    title: "Half-life ↔ elimination rate",
    blurb: "t½ ↔ k conversion both ways",
    match: "Half-life ↔ elimination rate",
    keywords: "halflife elimination kel rate",
  },
  {
    id: "auc",
    tab: "calculators",
    title: "AUC₀→last — trapezoidal rule",
    blurb: "area under the observed curve",
    match: "AUC₀→last — trapezoidal rule",
    keywords: "auc trapezoid area curve",
  },
  {
    id: "pd-emax",
    tab: "calculators",
    title: "PD Emax / sigmoid Hill simulator",
    blurb: "Emax model with Hill coefficient",
    match: "PD Emax / sigmoid Hill simulator",
    keywords: "emax hill sigmoid pd effect",
  },
  {
    id: "units",
    tab: "calculators",
    title: "Unit converter",
    blurb: "dimension-aware pharma unit conversion",
    match: "Unit converter",
    keywords: "units convert dimension mg ml",
  },
  /* -- PK / PD Explorer (inner tabs) -------------------------------- */
  {
    id: "pk-regimen",
    tab: "pk",
    inner: "regimen",
    title: "Regimen parameters",
    blurb: "dose · Vd · CL · τ — the superposition inputs",
    match: "Regimen parameters",
    keywords: "regimen dose vd clearance superposition",
  },
  {
    id: "pk-profile",
    tab: "pk",
    inner: "regimen",
    title: "Concentration–time profile",
    blurb: "exact first-order superposition plot",
    match: "Concentration–time profile",
    keywords: "concentration time profile plot pk",
  },
  {
    id: "pk-pd-effect",
    tab: "pk",
    inner: "pd",
    title: "Concentration & effect–time profile",
    blurb: "live PD layer over the C(t) curve",
    match: "Concentration & effect–time profile",
    keywords: "effect time pd emax sigmoid live",
  },
  {
    id: "pk-nca",
    tab: "pk",
    inner: "nca",
    title: "Observed profile & terminal fit",
    blurb: "NCA over pasted concentration–time rows",
    match: "Observed profile & terminal fit",
    keywords: "nca noncompartmental observed terminal auc aumc mrt",
  },
  /* -- ML Training --------------------------------------------------- */
  {
    id: "ml-setup",
    tab: "ml",
    title: "Experiment setup",
    blurb: "dataset, task, model and hyperparameters",
    match: "Experiment setup",
    keywords: "experiment dataset task model hyperparameters train",
  },
  {
    id: "ml-calibration",
    tab: "ml",
    title: "Probability Calibration",
    blurb: "reliability diagram + Brier score",
    match: "Probability Calibration",
    keywords: "calibration reliability brier probability",
  },
  {
    id: "ml-drugresponse",
    tab: "ml",
    title: "Response AI — drug-response learning",
    blurb: "Hill 4PL fit · metrics · CaDRReS model · grouped CV",
    match: "Response AI — drug-response learning",
    keywords: "drug response hill ic50 cadres metrics pearson spearman kendall cross-validation",
  },
  /* -- Cell Line Bio -------------------------------------------------- */
  {
    id: "bio-biomarkers",
    tab: "biology",
    title: "Biomarker discovery",
    blurb: "expression → response associations",
    match: "Biomarker discovery",
    keywords: "biomarker expression response discovery genes",
  },
  /* -- Docking Parser -------------------------------------------------- */
  {
    id: "dock-parse",
    tab: "docking",
    title: "Docking result parser",
    blurb: "DLG / Vina log → binding table",
    match: "Docking result parser",
    keywords: "docking parser dlg vina log results",
  },
  {
    id: "dock-torsion",
    tab: "docking",
    title: "Torsion QA",
    blurb: "torsion-consistency audit of prepared ligands",
    match: "Torsion QA",
    keywords: "torsion consistency audit qa ligands",
  },
  {
    id: "dock-vina-scan",
    tab: "docking",
    title: "Vina results-folder scan",
    blurb: "whole-campaign Vina output folder",
    match: "Vina results-folder scan",
    keywords: "vina folder scan campaign batch",
  },
  {
    id: "dock-rescore",
    tab: "docking",
    title: "Consensus rescore — the explainable ranking",
    blurb: "consensus scoring with visible weights",
    match: "Consensus rescore",
    keywords: "consensus rescore ranking weights explainable",
  },
  /* -- Features -------------------------------------------------------- */
  {
    id: "feat-factory",
    tab: "features",
    title: "Feature factory",
    blurb: "10,000+ descriptor/fingerprint columns, ranked",
    match: "Feature factory",
    keywords: "feature factory descriptors fingerprints ranked topk",
  },
  /* -- Ligand Prep ------------------------------------------------------ */
  {
    id: "ligand-prep",
    tab: "ligandprep",
    title: "Docking ligand prep",
    blurb: "SMILES → PDBQT with protonation states",
    match: "Docking ligand prep",
    keywords: "ligand prep pdbqt protonation smiles",
  },
  {
    id: "ligand-batch",
    tab: "ligandprep",
    title: "Batch GPF/DPF for a campaign",
    blurb: "grid + docking configs for every ligand",
    match: "Batch GPF/DPF for a campaign",
    keywords: "batch gpf dpf campaign configs grid",
  },
  /* -- Ligand Sets ------------------------------------------------------ */
  {
    id: "ligand-sets",
    tab: "reconciler",
    title: "Ligand-set reconciliation",
    blurb: "three-way compare of ligand columns",
    match: "Ligand-set reconciliation",
    keywords: "reconciliation compare ligand sets intersection",
  },
  /* -- Surgery (100x-130 — the desktop Generative tab's twin) ---------- */
  {
    id: "gen-topology",
    tab: "generative",
    title: "Ring topology moves",
    blurb: "one-step ring-size mutants, drawn",
    match: "Ring topology moves",
    keywords: "ring topology moves mutants contraction expansion generative",
  },
  {
    id: "gen-breed",
    tab: "generative",
    title: "Breed two molecules",
    blurb: "seeded scaffold-hop crossover litter",
    match: "Breed two molecules",
    keywords: "breed crossover scaffold hop children litter parents generative",
  },
  /* -- SA score (part3 §24 — Ertl-Schuffenhauer over the bridge) ------- */
  {
    id: "gen-sa-score",
    tab: "generative",
    title: "Synthetic Accessibility score",
    blurb: "Ertl–Schuffenhauer SA with filter + CSV export",
    match: "Synthetic Accessibility score",
    keywords:
      "sa score sascore ertl schuffenhauer synthetic accessibility ease of synthesis complexity",
  },
  /* -- MCTS search (the Studio Generative method's web twin) ---------- */
  {
    id: "gen-mcts",
    tab: "generative",
    title: "MCTS molecule search",
    blurb: "ChemTSv2-style tree search over an RNN prior",
    match: "MCTS molecule search",
    keywords:
      "mcts monte carlo tree search chemts uct puct rnn prior generative optimization leaderboard",
  },
  /* -- Molecular Dynamics (round 249 — the pharmaml.md web twins) ----- */
  {
    id: "md-xvg",
    tab: "md",
    title: "GROMACS export parser",
    blurb: "past .xvg/.ndx → stats, plots, correlations",
    match: "GROMACS export parser",
    keywords: "xvg ndx gromacs parser rmsd hbond correlation",
  },
  {
    id: "md-traj",
    tab: "md",
    title: "Trajectory analysis lab",
    blurb: "helix demo: RMSD/Rg/Q, RMSF, contacts, PCA",
    match: "Trajectory analysis lab",
    keywords: "trajectory rmsd rmsf rg contact map pca anm essential dynamics",
  },
  {
    id: "md-hbond",
    tab: "md",
    title: "H-bond timeline",
    blurb: "gmx hbond -num: two designed bonds across the 3.5 Å cutoff",
    match: "H-bond timeline",
    keywords: "hbond hbond timeline hydrogen bond gmx hbnum distance cutoff",
  },
  {
    id: "md-saltbridge",
    tab: "md",
    title: "Salt-bridge timeline",
    blurb: "getcontacts ionic pairs: two designed bridges across the 4.0 Å cutoff",
    match: "Salt-bridge timeline",
    keywords: "salt bridge saltbridge ionic contact getcontacts lysine glutamate arginine aspartate charged cutoff",
  },
  {
    id: "md-cv",
    tab: "md",
    title: "Collective variables (COLVAR)",
    blurb: "edit a plumed.dat: DISTANCE/ANGLE/DIHEDRAL/COORDINATION/RMSD/GYRATION over the helix demo",
    match: "Collective variables (COLVAR)",
    keywords: "collective variable colvar plumed metadynamics umbrella sampling enhanced bias distance angle dihedral coordination rmsd gyration",
  },
  {
    id: "md-eco",
    tab: "md",
    title: "The MD ecosystem",
    blurb: "the reference stack behind pharmaml.md",
    match: "The MD ecosystem",
    keywords: "mdanalysis mdtraj gromacs prody prolif deeptime ecosystem links",
  },
  /* -- The session-report strip (rides above every sub-tab) ------------- */
  {
    id: "session-report",
    tab: null,
    title: "Session report",
    blurb: "the whole playground session as one HTML file",
    match: "Session report",
    keywords: "session report html export download",
  },
];
