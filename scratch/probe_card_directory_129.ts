/**
 * 100x-129 — the card-directory drift probe (web side).
 *
 * Imports the SHIPPED catalog and checks every entry against the
 * component sources: the match string must still appear in the file
 * that renders the card (JSX entities normalized), ids must be unique,
 * tab values must be the real sub-tab values. The paired pytest guard
 * (tests/test_web_parity.py::TestCardFinderParity) fails CI when any
 * check breaks; the E2E probe (probe_card_finder_129.sh) walks the
 * live UI on top of this.
 */
import { PLAYGROUND_CARDS } from "../src/components/explorer/playground/card-directory";
import { readFileSync } from "node:fs";

/** entry id -> the source file whose render contains the card. */
const SOURCES: Record<string, string> = {
  crcl: "src/components/explorer/playground/pharma-calcs.tsx",
  bsa: "src/components/explorer/playground/pharma-calcs.tsx",
  pediatric: "src/components/explorer/playground/pharma-calcs.tsx",
  loading: "src/components/explorer/playground/pharma-calcs.tsx",
  halflife: "src/components/explorer/playground/pharma-calcs.tsx",
  auc: "src/components/explorer/playground/pharma-calcs.tsx",
  "pd-emax": "src/components/explorer/playground/pharma-calcs.tsx",
  units: "src/components/explorer/playground/pharma-calcs.tsx",
  "pk-regimen": "src/components/explorer/playground/pk-simulator.tsx",
  "pk-profile": "src/components/explorer/playground/pk-simulator.tsx",
  "pk-pd-effect": "src/components/explorer/playground/pd-effect.tsx",
  "pk-nca": "src/components/explorer/playground/nca-panel.tsx",
  "ml-setup": "src/components/explorer/playground/ml-playground.tsx",
  "ml-calibration": "src/components/explorer/playground/calibration-widget.tsx",
  "bio-biomarkers": "src/components/explorer/playground/biology-playground.tsx",
  "dock-parse": "src/components/explorer/playground/docking-parser.tsx",
  "dock-torsion": "src/components/explorer/playground/docking-parser.tsx",
  "dock-vina-scan": "src/components/explorer/playground/vina-folder-scan.tsx",
  "dock-rescore": "src/components/explorer/playground/consensus-rescore.tsx",
  "feat-factory": "src/components/explorer/playground/feature-factory.tsx",
  "ligand-prep": "src/components/explorer/playground/ligand-prep.tsx",
  "ligand-batch": "src/components/explorer/playground/batch-configs.tsx",
  "ligand-sets": "src/components/explorer/playground/ligand-reconciler.tsx",
  "gen-topology": "src/components/explorer/playground/surgery-cards.tsx",
  "gen-breed": "src/components/explorer/playground/surgery-cards.tsx",
  "gen-sa-score": "src/components/explorer/playground/sa-score-card.tsx",
  "md-xvg": "src/components/explorer/playground/md-analysis.tsx",
  "md-traj": "src/components/explorer/playground/md-analysis.tsx",
  "md-hbond": "src/components/explorer/playground/md-analysis.tsx",
  "md-eco": "src/components/explorer/playground/md-analysis.tsx",
  "session-report": "src/components/explorer/playground/session-report.tsx",
};

const VALID_TABS = new Set([
  "calculators", "pk", "ml", "biology", "docking",
  "features", "ligandprep", "reconciler", "generative", "md", null,
]);
const VALID_INNER = new Set(["regimen", "pd", "nca", undefined]);

const problems: string[] = [];
const seenIds = new Set<string>();
const cache = new Map<string, string>();
const source = (p: string): string => {
  if (!cache.has(p)) cache.set(p, readFileSync(p, "utf8"));
  return cache.get(p)!;
};

for (const entry of PLAYGROUND_CARDS) {
  if (seenIds.has(entry.id)) problems.push(`duplicate id ${entry.id}`);
  seenIds.add(entry.id);

  if (!VALID_TABS.has(entry.tab)) problems.push(`${entry.id}: bad tab ${String(entry.tab)}`);
  if (!VALID_INNER.has(entry.inner)) problems.push(`${entry.id}: bad inner ${String(entry.inner)}`);
  if (entry.tab === null && entry.inner !== undefined) {
    problems.push(`${entry.id}: strip entry cannot carry an inner tab`);
  }

  const path = SOURCES[entry.id];
  if (!path) {
    problems.push(`${entry.id}: no source mapping`);
    continue;
  }
  // JSX renders "&" as "&amp;"; both spellings count as present.
  const text = source(path);
  const variants = [entry.match, entry.match.replace(/&/g, "&amp;")];
  if (!variants.some((v) => text.includes(v))) {
    problems.push(`${entry.id}: match "${entry.match}" absent from ${path}`);
  }
  // the finder's rows display the title — the source should carry it
  // too (entity-normalized), else the catalog title has drifted.
  const titleVariants = [entry.title, entry.title.replace(/&/g, "&amp;")];
  if (!titleVariants.some((v) => text.includes(v))) {
    problems.push(`${entry.id}: title "${entry.title}" absent from ${path}`);
  }
}

const result = {
  entries: PLAYGROUND_CARDS.length,
  ids: seenIds.size,
  problems,
};
console.log(JSON.stringify(result, null, 2));
if (problems.length > 0) {
  console.error(`CARD_DIRECTORY_DRIFT: ${problems.length} problem(s)`);
  process.exit(1);
}
console.log("CARD_DIRECTORY_OK");
