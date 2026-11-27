/* R30-a probe — the ligand-set reconciliation session-report group.
 *
 * Exercises the full chain the round added:
 *   1. the store slice (setReconcileResult / sessionStorage persist)
 *   2. the collector's ligands group (metrics + per-bucket tables)
 *   3. the group's availability + toggle + walking-order behaviour
 *
 * Run with: bunx tsx scratch/probe_session_report_r30a.ts
 */
import {
  collectSessionReport,
  sanitizeSessionOrder,
  sessionGroupsAvailable,
  sessionReportCensus,
  sessionReportContents,
  SESSION_GROUP_ORDER,
} from "@/lib/session-report";
import type { SessionSnapshot } from "@/lib/session-report";
import type { ReconcileResultState } from "@/lib/store";
import { RECONCILE_BUCKET_CAP } from "@/lib/store";

/* The fixture: a reconciliation in the shape the Ligand Sets tab
 * produces — 4 common, 2 only in A, 1 only in B, both normalisation
 * flags on.  The onlyA bucket carries 60 names to prove the report
 * table caps at RECONCILE_BUCKET_CAP and confesses the remainder. */
const bigOnlyA: string[] = Array.from(
  { length: 60 },
  (_, i) => `ligand_A_only_${String(i + 1).padStart(2, "0")}`,
);
const reconciliation: ReconcileResultState = {
  stripExtension: true,
  lowercase: true,
  labelA: "A",
  labelB: "B",
  counts: { common: 4, onlyA: 60, onlyB: 1 },
  common: ["aspirin", "caffeine", "ibuprofen", "metformin"],
  onlyA: bigOnlyA,
  onlyB: ["warfarin"],
};

const snapshot: SessionSnapshot = {
  mlResult: null,
  lcResult: null,
  mlHistory: [],
  pkParams: {
    route: "iv",
    dose: 500,
    vd: 40,
    cl: 3,
    tau: 12,
    doses: 5,
    f: 1,
    ka: 1,
  },
  pdParams: {
    model: "emax",
    view: "both",
    slope: 1,
    e0: 4,
    emax: 100,
    ec50: 2,
    hill: 1,
    imax: 80,
    ic50: 2,
  },
  ncaState: { rows: [], dose: "250", f: "0.9", terminalN: 3 },
  surgery: null,
  ligands: reconciliation,
};

/* -- 1. the collector walks the ligands group -------------------------- */
const payload = collectSessionReport(snapshot);
const ligandSections = payload.sections.filter(
  (s) =>
    s.kind === "metrics" &&
    s.title.startsWith("Ligand-set reconciliation"),
);
if (ligandSections.length !== 1) {
  throw new Error(`expected exactly 1 reconciliation metrics card, got ${ligandSections.length}`);
}
const card = ligandSections[0];
if (card.kind !== "metrics") throw new Error("unreachable");
const itemMap = new Map(card.items);
if (itemMap.get("common (both tables)") !== "4") throw new Error("common count wrong");
if (itemMap.get("only in A") !== "60") throw new Error("onlyA count wrong");
if (itemMap.get("only in B") !== "1") throw new Error("onlyB count wrong");
if (itemMap.get("unique names compared") !== "65") throw new Error("total wrong");
if (itemMap.get("normalisation") !== "extensions stripped, case folded") {
  throw new Error("normalisation text wrong");
}
console.log("1. metrics card:", card.items.length, "items, counts + norms correct");

/* -- 2. bucket tables, the cap and the +N confession ------------------- */
const tables = payload.sections.filter(
  (s) =>
    s.kind === "table" &&
    (s.title.startsWith("Common ligands") ||
      s.title.startsWith("Only in")),
);
if (tables.length !== 3) throw new Error(`expected 3 bucket tables, got ${tables.length}`);
const onlyATable = tables.find((s) => s.kind === "table" && s.title.startsWith("Only in A"));
if (!onlyATable || onlyATable.kind !== "table") throw new Error("only-A table missing");
if (onlyATable.rows.length !== RECONCILE_BUCKET_CAP + 1) {
  throw new Error(`capped table should be cap+1 rows (last = +N confession), got ${onlyATable.rows.length}`);
}
const lastRow = onlyATable.rows[onlyATable.rows.length - 1];
if (String(lastRow[1]) !== `+ ${60 - RECONCILE_BUCKET_CAP} more (see the web console's Ligand Sets tab or report settings)`) {
  throw new Error(`confession row wrong: ${JSON.stringify(lastRow)}`);
}
const emptyBucket = tables.filter(
  (s) => s.kind === "table" && s.title.startsWith("Only in B"),
);
const onlyBTable = emptyBucket[0];
if (
  emptyBucket.length !== 1 ||
  onlyBTable.kind !== "table" ||
  onlyBTable.rows.length !== 1
) {
  throw new Error("only-B (count 1) should still render its single row");
}
console.log(
  "2. bucket tables:",
  tables.length,
  "| only-A capped at",
  RECONCILE_BUCKET_CAP,
  "+ confession row | only-B ships its 1 row",
);

/* -- 3. availability, toggles and the walking order -------------------- */
const available = sessionGroupsAvailable(snapshot);
if (available.ligands !== true) throw new Error("ligands should be available");
if (available.ml !== false) throw new Error("ml should be unavailable");

const toggled = collectSessionReport(snapshot, { ligands: false });
if (toggled.sections.some((s) => "title" in s && s.title.startsWith("Ligand-set"))) {
  throw new Error("ligands: false did not skip the group");
}

const census = sessionReportCensus(snapshot, {}, SESSION_GROUP_ORDER);
if (census.tables !== 3) {
  throw new Error(`census should count the 3 ligand bucket tables, got ${census.tables}`);
}
if (census.metrics < 1) {
  throw new Error("census should count the reconciliation metrics card");
}
const contents = sessionReportContents(payload.sections);
if (!contents.some((entry) => entry.title.startsWith("Only in A"))) {
  throw new Error("contents strip should list the only-A bucket table");
}

const ligandsFirst = sanitizeSessionOrder(["ligands", "ml"]);
if (ligandsFirst[0] !== "ligands") throw new Error("custom order should honor ligands first");

const restoredOrder = sanitizeSessionOrder(["ml"]);
if (restoredOrder[restoredOrder.length - 1] !== "ligands") {
  throw new Error("a pre-ligands stored order must append the group last, never drop it");
}
console.log(
  "3. availability/toggle/order: ligands available + skip-on-false +",
  "census lists it + sanitize appends it last for stored orders",
);

/* -- 4. the comparison figure rides the report figure contract -------- */
const figure = payload.sections.find(
  (s) => s.kind === "figure" && s.title.includes("bucket counts"),
);
if (!figure || figure.kind !== "figure") {
  throw new Error("the comparison figure section is missing");
}
if (figure.traces.length !== 1 || figure.traces[0].style !== "bar") {
  throw new Error("the comparison figure needs exactly one bar trace");
}
const barPoints = figure.traces[0].points as Array<[number, number]>;
const byY = new Map(barPoints.map(([y, v]) => [y, v]));
if (byY.get(2) !== 4 || byY.get(1) !== 60 || byY.get(0) !== 1) {
  throw new Error(`bar values wrong: ${JSON.stringify(barPoints)}`);
}
if (figure.xLabel !== "number of unique ligand names") {
  throw new Error("figure x label wrong");
}
console.log("4. comparison figure: one bar trace, [y, value] pairs 2/1/0 = 4/60/1");

/* -- 5. null reconciliation contributes nothing ------------------------ */
const withoutLigands: SessionSnapshot = { ...snapshot, ligands: null };
const bare = collectSessionReport(withoutLigands);
if (bare.sections.some((s) => "title" in s && s.title.startsWith("Ligand-set"))) {
  throw new Error("null ligands should contribute no sections");
}
console.log("5. null reconciliation: no sections, no crash");

console.log("\nprobe_session_report_r30a: ALL CHECKS PASSED");
