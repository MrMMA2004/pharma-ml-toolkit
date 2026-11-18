import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Boxes,
  Cpu,
  Dna,
  FileSearch,
  LineChart,
  Network,
  Pill,
  Play,
  Target,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useJob } from "@/hooks/useJobTracking";
import { jobsApi } from "@/services/jobs/client";
import { ApiError } from "@/services/api/client";
import {
  artifactsApi,
  type ArtifactView,
} from "@/services/artifacts/client";
import { getEventStream } from "@/services/websocket/client";
import {
  dockingApi,
  parseDockRows,
  parseJobProgressPayload,
  type ConsensusResponse,
  type ContactAtomFields,
  type ContactsResponse,
  type DlgParseResponse,
  type DockRowsView,
  type DockstringTargetRow,
  type DockstringTargetView,
  type EnginesResponse,
  type JobProgressPayload,
  type LigandEfficiencyResponse,
  type LigandPdbqtResponse,
  type ObjectiveResponse,
  type PoseRmsdResponse,
  type ReceptorPrepareResponse,
  type VinaParseResponse,
} from "@/services/docking/client";
import {
  Molecule3D,
  type Molecule3DAtom,
} from "@/components/molecular3d/Molecule3D";
import {
  BOND_INFERENCE_ATOM_LIMIT,
  inferDisplayBonds,
  parsePdbqtAtoms,
  type InferredBond,
} from "@/services/molecular3d/pdbqt";

/**
 * The Molecular Docking workspace — part11 §11.5's complete sequential
 * workflow:
 *
 * Select Ligands → Prepare Ligands → Prepare Receptor → Define Binding
 * Site/Grid → Configure Engine → Run Docking → Inspect Poses → Analyze
 * Interactions → Visualize → Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons).
 * The preparation lanes are the backend's own (`smiles_to_pdbqt`'s
 * RDKit ETKDG + Meeko typing for the ligand, `prepare_protein` for
 * the receptor — each registering a downloadable, digest-verified
 * artifact); the binding site is either the manual center/size box or
 * a REAL DOCKSTRING target box; the engine catalog is the route's
 * own honest verdict (`GET /docking/engines` — on a host where no
 * bundled engine launches, the boundary note is rendered verbatim,
 * the dock button still submits, and the route's own 422 surfaces
 * as the stage's error state; a docking result is never faked). The
 * dock lane defaults to job mode — the COMPLETED job's REST result
 * is authoritative (the WS `job.progress` envelopes drive the
 * per-ligand progress bar only); the route's sync mode answers the
 * same rows directly, no pretend job tracking. Cancelling is native:
 * the job lands CANCELLED, never FAILED, with partial results
 * preserved. The Inspect stage parses REAL pasted engine logs (the
 * Vina mode table + the AutoDock4 DLG cluster histogram — the
 * backend's own parsers, honest 422s when a log carries no table);
 * the Analyze stage computes REAL contacts + the PLIF matrix, the
 * symmetry-aware pose RMSD, ligand efficiency and the consensus
 * normalization; the Visualize stage registers a REAL figure
 * artifact (the parsed poses' affinity profile through the §17
 * four-format export stack); the Export stage rides the rows-based
 * table lane with the backend's own 5000-row cap mirrored
 * client-side.
 */

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

/** The pasted-contact CSV's editable fields, in the route's own order. */
const CONTACT_ATOM_FIELDS: readonly (keyof ContactAtomFields)[] = [
  "serial",
  "name",
  "element",
  "resname",
  "chain",
  "resseq",
  "x",
  "y",
  "z",
  "autodock_type",
];

/** The pose table's own numeric keys (the consensus lane's columns). */
const POSE_COLUMNS: readonly string[] = [
  "mode",
  "affinity",
  "rmsd_ub",
  "rmsd_lb",
];

/** One blank editable atom row (the editor's own empty state). */
const EMPTY_ATOM: ContactAtomFields = {
  serial: "",
  name: "",
  element: "",
  resname: "",
  chain: "",
  resseq: "",
  x: "",
  y: "",
  z: "",
  autodock_type: "",
};

/**
 * A realistic minimal Vina output block — the mode table the REAL
 * parser (`parse_vina_output`) reads (the `Writing output` header +
 * the `mode | affinity | dist from best mode` table, real format
 * lines; rmsd l.b./u.b. per pose).
 */
const SAMPLE_VINA_LOG = [
  "Estimated Free Energy of Binding   = -10.3 kcal/mol",
  "Writing output ... ligand_out.pdbqt.",
  "mode | affinity | dist from best mode",
  "     |          | rmsd l.b. | rmsd u.b.",
  "-----+----------+----------+----------",
  "   1      -10.3          0          0",
  "   2       -9.6      1.842      2.301",
  "   3       -9.4      2.105      2.777",
].join("\n");

/**
 * A realistic minimal AutoDock4 DLG snippet — the `CLUSTERING`
 * histogram table the REAL parser (`parse_dlg_cluster_table`)
 * reads (rank | lowest energy | run | mean energy | # in cluster).
 */
const SAMPLE_DLG_LOG = [
  "AutoDock4 log file (sample)",
  "CLUSTERING HISTOGRAM",
  "______________________________________________________________________",
  "Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster",
  "     1       |     -11.5     |  3  |    -11.2    |      7 |",
  "     2       |     -10.8     |  9  |    -10.5    |     34 |",
  "     3       |      -9.9     | 15  |     -9.6    |     12 |",
].join("\n");

/** Editable sample ligand atoms (the CSV lane's format, verbatim). */
const SAMPLE_LIGAND_ATOMS_CSV = [
  "serial name element resname chain resseq x y z type",
  "1 C1 C LIG A 1 0.0 0.0 0.0 C",
  "2 O1 O LIG A 1 1.0 0.0 0.0 OA",
].join("\n");

/** Editable sample receptor atoms (a far residue shows the cutoff). */
const SAMPLE_RECEPTOR_ATOMS_CSV = [
  "serial name element resname chain resseq x y z type",
  "10 CA C ASP A 30 2.0 0.0 0.0 C",
  "11 OD1 O ASP A 30 2.5 0.5 0.0 OA",
  "12 CB C PHE A 31 40.0 0.0 0.0 C",
].join("\n");

type DockMode = "job" | "sync";

type ExportSource = "vina-poses" | "dock-rows";

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail;
    if (
      typeof detail === "object" &&
      detail !== null &&
      "detail" in detail &&
      typeof (detail as Record<string, unknown>)["detail"] === "string"
    ) {
      return (detail as Record<string, string>)["detail"] ?? `HTTP ${error.status}`;
    }
    if (typeof detail === "string") {
      return detail;
    }
    return `HTTP ${error.status}`;
  }
  return error instanceof Error ? error.message : String(error);
}

/**
 * The Phase Eighteen 3D scene builder — the PDBQT text that actually
 * arrived parsed client-side for display only (the backend's own
 * preparation stays the scientific authority); the bonds are the
 * documented distance heuristic, never scientific output.
 */
interface Scene3d {
  readonly atoms: readonly Molecule3DAtom[];
  readonly bonds: readonly InferredBond[];
  readonly diagnostics: string;
}

function pdbqtToScene(text: string, label: string): Scene3d {
  const parsed = parsePdbqtAtoms(text, label);
  const atoms: Molecule3DAtom[] = parsed.atoms.map((atom) => ({
    element: atom.element,
    x: atom.x,
    y: atom.y,
    z: atom.z,
    label:
      [atom.name, atom.residue]
        .filter((part) => part !== null && part.trim() !== "")
        .join(" ") || null,
  }));
  const bonds = inferDisplayBonds(parsed.atoms);
  const diagnostics: string[] = [];
  if (parsed.skipped > 0) {
    diagnostics.push(`${parsed.skipped} malformed records skipped (counted, never silent)`);
  }
  if (parsed.truncated > 0) {
    diagnostics.push(`${parsed.truncated} records beyond the 4,000-atom render limit not drawn`);
  }
  if (bonds.length === 0 && parsed.atoms.length > BOND_INFERENCE_ATOM_LIMIT) {
    diagnostics.push("bond inference skipped above the 2,000-atom display limit");
  }
  return { atoms, bonds, diagnostics: diagnostics.join("; ") };
}

/** The Phase Eighteen display-heuristic honesty note (required). */
const DOCKING_3D_BOND_NOTE =
  "Bonds are drawn by a distance heuristic (covalent radii × 1.25) — display only, never scientific output; the scene renders only the ATOM/HETATM records that actually arrived. Pose coordinates are not parsed from the engine logs, so only the prepared ligand and receptor structures render — never an imagined complex.";

function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-sm text-[var(--color-foreground)]"
    >
      <span className="font-medium">Request failed:</span>{" "}
      <span className="text-[var(--color-muted-foreground)]">{message}</span>
    </div>
  );
}

/** The §13 live job card: WS-driven state, progress, cancel, result relay. */
function JobRunCard({
  jobId,
  resultLabel,
  onResult,
}: {
  jobId: string;
  resultLabel: string;
  onResult: (result: unknown) => void;
}) {
  const { snapshot, cancel } = useJob(jobId);
  const relayed = useRef(false);
  const [relayError, setRelayError] = useState<string | null>(null);
  const view = snapshot?.view ?? null;

  useEffect(() => {
    if (
      view !== null &&
      view.state === "COMPLETED" &&
      view.result_ready &&
      !relayed.current
    ) {
      relayed.current = true;
      jobsApi
        .result(jobId)
        .then((response) => onResult(response.result))
        .catch((error) => setRelayError(describeError(error)));
    }
  }, [view, jobId, onResult]);

  if (view === null) {
    return <p className="text-sm text-[var(--color-muted-foreground)]">Tracking job…</p>;
  }
  const terminal = view.state === "COMPLETED" || view.state === "CANCELLED" || view.state === "FAILED";
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] font-semibold tracking-wide text-[var(--color-muted-foreground)]">
          {view.state}
        </span>
        <span className="text-xs text-[var(--color-muted-foreground)]">
          {view.operation} · {resultLabel}
        </span>
        {!terminal ? (
          <Button size="sm" variant="outline" onClick={() => void cancel()}>
            Cancel
          </Button>
        ) : null}
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-[var(--color-muted)]">
        <div
          className="h-full rounded-full bg-[var(--color-primary)] transition-[width]"
          style={{ width: `${Math.min(100, Math.max(0, view.progress))}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        {view.progress}% · {view.stage || "queued"}
      </p>
      {view.error !== null ? (
        <p className="mt-2 text-xs text-[var(--color-destructive)]">{view.error}</p>
      ) : null}
      {relayError !== null ? <ErrorBanner message={relayError} /> : null}
    </div>
  );
}

/** A numbered workflow stage header (§11.5's fixed order). */
function StageHeader({
  index,
  title,
  hint,
  icon,
  children,
}: {
  index: number;
  title: string;
  hint: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-8">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-[var(--color-muted)] text-xs font-semibold text-[var(--color-muted-foreground)]">
          {index}
        </span>
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">{title}</h2>
        {icon ?? null}
        <span className="text-xs text-[var(--color-muted-foreground)]">{hint}</span>
      </div>
      <div className="mt-3">{children}</div>
    </section>
  );
}

/** One table cell's honest rendering (§17.3: type-aware, bounded). */
function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "—";
    }
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (Array.isArray(value)) {
    return `[${value.map((v) => formatValue(v)).join(", ")}]`;
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

/** The union of key names across a rows list (column discovery). */
function unionRowKeys(rows: readonly Record<string, unknown>[]): string[] {
  const keys: string[] = [];
  const seen = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        keys.push(key);
      }
    }
  }
  return keys;
}

/**
 * Parse one pasted atom CSV into editable rows — columnar text in the
 * route's `ContactAtomSpec` order (a `serial …` header line is
 * skipped; every other malformed row is this parser's own labeled
 * error, never a silent drop).
 */
function parseAtomRows(text: string, label: string): ContactAtomFields[] {
  const rows: ContactAtomFields[] = [];
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
  lines.forEach((line, i) => {
    const cells = line.split(/[\s,]+/).filter((cell) => cell !== "");
    if (cells[0] === "serial") {
      return; // the header row
    }
    if (cells.length < 9) {
      throw new Error(
        `${label} row ${i + 1}: expected ≥ 9 cells ` +
          "(serial name element resname chain resseq x y z [type])",
      );
    }
    rows.push({
      serial: cells[0] ?? "",
      name: cells[1] ?? "",
      element: cells[2] ?? "",
      resname: cells[3] ?? "",
      chain: cells[4] ?? "",
      resseq: cells[5] ?? "",
      x: cells[6] ?? "",
      y: cells[7] ?? "",
      z: cells[8] ?? "",
      autodock_type: cells[9] ?? "",
    });
  });
  return rows;
}

/** Parse pasted `x y z` triples (exactly three finite numbers per row). */
function parseTriples(text: string, label: string): number[][] {
  const rows: number[][] = [];
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
  lines.forEach((line, i) => {
    const cells = line.split(/[\s,]+/).filter((cell) => cell !== "");
    const values = cells.map((cell) => Number(cell));
    if (cells.length !== 3 || values.some((v) => !Number.isFinite(v))) {
      throw new Error(
        `${label} row ${i + 1}: expected exactly 3 finite numbers (x y z)`,
      );
    }
    rows.push(values);
  });
  return rows;
}

/** Parse pasted poses — blank-line-separated blocks of `x y z` rows. */
function parsePoseBlocks(text: string): number[][][] {
  return text
    .split(/\n[ \t]*\n+/)
    .map((block) => block.trim())
    .filter((block) => block !== "")
    .map((block, i) => parseTriples(block, `pose ${i + 1}`));
}

/**
 * Parse the objective lane's pasted scores — one `KEY value` pair
 * per line (the objective's own required keys are the route's 422
 * to state, never guessed here).
 */
function parseScoresInput(
  text: string,
  label: string,
): Record<string, number> {
  const scores: Record<string, number> = {};
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
  lines.forEach((line, i) => {
    const match = /^(\S+)[\s=]+(\S+)$/.exec(line);
    if (match === null) {
      throw new Error(
        `${label} line ${i + 1}: expected "KEY value" (e.g. F2 -9.5)`,
      );
    }
    const value = Number(match[2]);
    if (!Number.isFinite(value)) {
      throw new Error(
        `${label} line ${i + 1}: ${match[2] ?? ""} is not a finite score`,
      );
    }
    scores[match[1] ?? ""] = value;
  });
  return scores;
}

/** One editable atom-rows table (the `ContactAtomSpec` editor). */
function AtomRowsEditor({
  label,
  rows,
  onChange,
}: {
  label: string;
  rows: readonly ContactAtomFields[];
  onChange: (next: readonly ContactAtomFields[]) => void;
}) {
  const updateRow = (
    index: number,
    field: keyof ContactAtomFields,
    value: string,
  ) => {
    onChange(
      rows.map((row, i) => (i === index ? { ...row, [field]: value } : row)),
    );
  };
  return (
    <div className="mt-2">
      {rows.length === 0 ? (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          no {label} rows yet — paste the CSV lane above or add a blank row
        </p>
      ) : (
        <div className="max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
          <table className="w-full border-collapse text-[11px]">
            <thead className="sticky top-0 bg-[var(--color-muted)]">
              <tr>
                {CONTACT_ATOM_FIELDS.map((field) => (
                  <th
                    key={field}
                    scope="col"
                    className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]"
                  >
                    {field === "autodock_type" ? "type" : field}
                  </th>
                ))}
                <th scope="col" className="px-2 py-1" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr
                  key={index}
                  className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                >
                  {CONTACT_ATOM_FIELDS.map((field) => (
                    <td key={field} className="px-1 py-0.5">
                      <input
                        value={row[field]}
                        onChange={(event) =>
                          updateRow(index, field, event.target.value)
                        }
                        className="w-16 rounded border border-[var(--color-border)] bg-[var(--color-background)] px-1 py-0.5 font-mono text-[11px]"
                      />
                    </td>
                  ))}
                  <td className="px-1 py-0.5">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        onChange(rows.filter((_, i) => i !== index))
                      }
                    >
                      ×
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="mt-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => onChange([...rows, EMPTY_ATOM])}
        >
          Add blank row
        </Button>
      </div>
    </div>
  );
}

export function DockingPage() {
  // stage 1 — the ligand list
  const [ligandText, setLigandText] = useState<string>(
    "CCO\nCCN\nCC(=O)O\nc1ccccc1C\nCCO",
  );

  // stage 2 — the ligand PDBQT lane
  const [selectedLigand, setSelectedLigand] = useState<string>("");
  const [ligName, setLigName] = useState<string>("");
  const [ligCharge, setLigCharge] = useState<string>("");
  const [ligSeed, setLigSeed] = useState<number>(0);
  const [ligandResult, setLigandResult] = useState<LigandPdbqtResponse | null>(null);
  const [ligandError, setLigandError] = useState<string | null>(null);
  const [ligandBusy, setLigandBusy] = useState(false);
  const [ligandDigest, setLigandDigest] = useState<string | null>(null);

  // stage 9 — the Phase Eighteen 3D structure lane (Three.js): the
  // prepared ligand's PDBQT text fetched through the digest-verified
  // raw lane and the pasted receptor rendered as ball-and-stick
  const [ligand3dText, setLigand3dText] = useState<string | null>(null);
  const [ligand3dBusy, setLigand3dBusy] = useState(false);
  const [ligand3dError, setLigand3dError] = useState<string | null>(null);

  // stage 3 — the receptor preparation lane
  const [receptorText, setReceptorText] = useState<string>("");
  const [receptorFilename, setReceptorFilename] = useState<string>("receptor.pdbqt");
  const [removeWaters, setRemoveWaters] = useState(true);
  const [keepHetatm, setKeepHetatm] = useState(false);
  const [receptorChain, setReceptorChain] = useState<string>("");
  const [stripHydrogens, setStripHydrogens] = useState(true);
  const [receptorResult, setReceptorResult] = useState<ReceptorPrepareResponse | null>(null);
  const [receptorError, setReceptorError] = useState<string | null>(null);
  const [receptorBusy, setReceptorBusy] = useState(false);
  const [receptorDigest, setReceptorDigest] = useState<string | null>(null);

  // stage 4 — the binding site / grid box
  const [centerX, setCenterX] = useState(0);
  const [centerY, setCenterY] = useState(0);
  const [centerZ, setCenterZ] = useState(0);
  const [sizeX, setSizeX] = useState(22.5);
  const [sizeY, setSizeY] = useState(22.5);
  const [sizeZ, setSizeZ] = useState(22.5);
  const [targets, setTargets] = useState<readonly DockstringTargetRow[]>([]);
  const [targetsError, setTargetsError] = useState<string | null>(null);
  const [targetsBusy, setTargetsBusy] = useState(true);
  const [targetQuery, setTargetQuery] = useState<string>("");
  const [targetView, setTargetView] = useState<DockstringTargetView | null>(null);
  const [targetViewError, setTargetViewError] = useState<string | null>(null);
  const [targetBusy, setTargetBusy] = useState(false);

  // stage 5 — the engine catalog + the benchmark objective lane
  const [enginesCatalog, setEnginesCatalog] = useState<EnginesResponse | null>(null);
  const [enginesError, setEnginesError] = useState<string | null>(null);
  const [enginesBusy, setEnginesBusy] = useState(true);
  const [objectiveName, setObjectiveName] = useState<string>("");
  const [objectiveScoresText, setObjectiveScoresText] = useState<string>("F2 -9.5");
  const [objectiveQed, setObjectiveQed] = useState<number>(0.8);
  const [objectiveResult, setObjectiveResult] = useState<ObjectiveResponse | null>(null);
  const [objectiveError, setObjectiveError] = useState<string | null>(null);
  const [objectiveBusy, setObjectiveBusy] = useState(false);

  // stage 6 — the dock lane (job by default, sync as the route declares)
  const [dockEngine, setDockEngine] = useState<"vina" | "autodock4">("vina");
  const [exhaustiveness, setExhaustiveness] = useState(8);
  const [numModes, setNumModes] = useState(9);
  const [workers, setWorkers] = useState(1);
  const [dockSeed, setDockSeed] = useState(0);
  const [timeoutS, setTimeoutS] = useState(3600);
  const [dockMode, setDockMode] = useState<DockMode>("job");
  const [dockJobId, setDockJobId] = useState<string | null>(null);
  const [dockRowsView, setDockRowsView] = useState<DockRowsView | null>(null);
  const [dockError, setDockError] = useState<string | null>(null);
  const [dockBusy, setDockBusy] = useState(false);
  const [dockProgress, setDockProgress] = useState<readonly JobProgressPayload[]>([]);

  // stage 7 — the pasted engine logs
  const [vinaText, setVinaText] = useState<string>("");
  const [vinaResult, setVinaResult] = useState<VinaParseResponse | null>(null);
  const [vinaError, setVinaError] = useState<string | null>(null);
  const [vinaBusy, setVinaBusy] = useState(false);
  const [dlgText, setDlgText] = useState<string>("");
  const [dlgResult, setDlgResult] = useState<DlgParseResponse | null>(null);
  const [dlgError, setDlgError] = useState<string | null>(null);
  const [dlgBusy, setDlgBusy] = useState(false);

  // stage 8a — the contacts + PLIF lane
  const [ligandCsvText, setLigandCsvText] = useState<string>(SAMPLE_LIGAND_ATOMS_CSV);
  const [receptorCsvText, setReceptorCsvText] = useState<string>(SAMPLE_RECEPTOR_ATOMS_CSV);
  const [ligandAtoms, setLigandAtoms] = useState<readonly ContactAtomFields[]>([]);
  const [receptorAtoms, setReceptorAtoms] = useState<readonly ContactAtomFields[]>([]);
  const [contactsCutoff, setContactsCutoff] = useState(5);
  const [contactsResult, setContactsResult] = useState<ContactsResponse | null>(null);
  const [contactsError, setContactsError] = useState<string | null>(null);
  const [contactsBusy, setContactsBusy] = useState(false);

  // stage 8b — the pose RMSD lane
  const [rmsdSmiles, setRmsdSmiles] = useState<string>("CCO");
  const [rmsdRefText, setRmsdRefText] = useState<string>(
    "0.0 0.0 0.0\n1.5 0.0 0.0\n0.0 1.5 0.0",
  );
  const [rmsdPosesText, setRmsdPosesText] = useState<string>(
    "0.0 0.0 0.0\n1.5 0.0 0.0\n0.0 1.5 0.0\n\n2.0 0.0 0.0\n3.5 0.0 0.0\n2.0 1.5 0.0",
  );
  const [rmsdResult, setRmsdResult] = useState<PoseRmsdResponse | null>(null);
  const [rmsdError, setRmsdError] = useState<string | null>(null);
  const [rmsdBusy, setRmsdBusy] = useState(false);

  // stage 8c — the scoring lane
  const [leAffinity, setLeAffinity] = useState<string>("-8.5");
  const [leHeavyAtoms, setLeHeavyAtoms] = useState<string>("17");
  const [leResult, setLeResult] = useState<LigandEfficiencyResponse | null>(null);
  const [leError, setLeError] = useState<string | null>(null);
  const [leBusy, setLeBusy] = useState(false);
  const [consensusSelected, setConsensusSelected] = useState<
    Record<string, boolean>
  >({ mode: false, affinity: true, rmsd_ub: false, rmsd_lb: false });
  const [consensusDirections, setConsensusDirections] = useState<
    Record<string, boolean>
  >({ mode: true, affinity: false, rmsd_ub: false, rmsd_lb: false });
  const [consensusWeightsText, setConsensusWeightsText] = useState<string>("");
  const [consensusResult, setConsensusResult] = useState<ConsensusResponse | null>(null);
  const [consensusError, setConsensusError] = useState<string | null>(null);
  const [consensusBusy, setConsensusBusy] = useState(false);

  // stage 9 — the figure lane
  const [figureFormat, setFigureFormat] = useState<"PNG" | "SVG" | "TIFF" | "JPEG">("PNG");
  const [poseFigure, setPoseFigure] = useState<ArtifactView | null>(null);
  const [figureError, setFigureError] = useState<string | null>(null);
  const [figureBusy, setFigureBusy] = useState(false);
  const [figureDigest, setFigureDigest] = useState<string | null>(null);

  // stage 10 — the export lane
  const [exportFormat, setExportFormat] = useState<"csv" | "tsv" | "xlsx">("csv");
  const [exportSource, setExportSource] = useState<ExportSource>("vina-poses");
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [exportDigest, setExportDigest] = useState<string | null>(null);

  // stage 5: the live engine catalog (the route's own honest verdict)
  useEffect(() => {
    let cancelled = false;
    setEnginesBusy(true);
    dockingApi
      .engines()
      .then((response) => {
        if (cancelled) {
          return;
        }
        setEnginesCatalog(response);
        setEnginesError(null);
        setObjectiveName((current) =>
          current === "" ? (response.dockstring.objectives[0] ?? "") : current,
        );
      })
      .catch((error) => {
        if (!cancelled) {
          setEnginesError(describeError(error));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setEnginesBusy(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // stage 4: the DOCKSTRING target catalog (best-effort at mount)
  useEffect(() => {
    let cancelled = false;
    setTargetsBusy(true);
    dockingApi
      .dockstringTargets()
      .then((response) => {
        if (!cancelled) {
          setTargets(response.targets);
          setTargetsError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setTargetsError(describeError(error));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setTargetsBusy(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // stage 1: the deduplicated ligand list (one SMILES per row)
  const ligands = useMemo(() => {
    const seen = new Set<string>();
    const unique: string[] = [];
    for (const line of ligandText.split("\n")) {
      const trimmed = line.trim();
      if (trimmed !== "" && !seen.has(trimmed)) {
        seen.add(trimmed);
        unique.push(trimmed);
      }
    }
    return unique;
  }, [ligandText]);

  const parsedSmilesCount = useMemo(
    () =>
      ligandText
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line !== "").length,
    [ligandText],
  );

  const selectedSmiles = useMemo(
    () =>
      ligands.includes(selectedLigand)
        ? selectedLigand
        : (ligands[0] ?? ""),
    [ligands, selectedLigand],
  );

  // stage 2: the ligand PDBQT lane
  const handleLigandPdbqt = useCallback(async () => {
    if (selectedSmiles === "") {
      return;
    }
    const chargeValue =
      ligCharge.trim() === "" ? null : Number(ligCharge.trim());
    if (chargeValue !== null && !Number.isFinite(chargeValue)) {
      setLigandError("charge must be a number (−10…10, the route's own bound)");
      return;
    }
    setLigandBusy(true);
    setLigandError(null);
    setLigandResult(null);
    setLigandDigest(null);
    try {
      const response = await dockingApi.ligandPdbqt({
        smiles: selectedSmiles,
        name: ligName.trim() === "" ? null : ligName.trim(),
        charge: chargeValue,
        seed: ligSeed,
      });
      setLigandResult(response);
    } catch (error) {
      setLigandError(describeError(error));
    } finally {
      setLigandBusy(false);
    }
  }, [selectedSmiles, ligName, ligCharge, ligSeed]);

  // stage 3: the receptor preparation lane
  const handleReceptorPrepare = useCallback(async () => {
    if (receptorText.trim() === "") {
      return;
    }
    setReceptorBusy(true);
    setReceptorError(null);
    setReceptorResult(null);
    setReceptorDigest(null);
    try {
      const response = await dockingApi.receptorPrepare({
        pdbqtText: receptorText,
        filename: receptorFilename.trim() === "" ? "receptor.pdbqt" : receptorFilename.trim(),
        removeWaters,
        keepHetatm,
        chain: receptorChain.trim() === "" ? null : receptorChain.trim(),
        stripHydrogens,
      });
      setReceptorResult(response);
    } catch (error) {
      setReceptorError(describeError(error));
    } finally {
      setReceptorBusy(false);
    }
  }, [receptorText, receptorFilename, removeWaters, keepHetatm, receptorChain, stripHydrogens]);

  // stage 4: the DOCKSTRING target picker (searchable, live)
  const filteredTargets = useMemo(() => {
    const query = targetQuery.trim().toLowerCase();
    if (query === "") {
      return targets;
    }
    return targets.filter((row) =>
      row.target.toLowerCase().includes(query),
    );
  }, [targets, targetQuery]);

  const selectTarget = useCallback(async (name: string) => {
    setTargetBusy(true);
    setTargetViewError(null);
    try {
      const view = await dockingApi.dockstringTarget(name);
      setTargetView(view);
      setCenterX(view.center[0] ?? 0);
      setCenterY(view.center[1] ?? 0);
      setCenterZ(view.center[2] ?? 0);
      setSizeX(view.size[0] ?? 22.5);
      setSizeY(view.size[1] ?? 22.5);
      setSizeZ(view.size[2] ?? 22.5);
    } catch (error) {
      setTargetView(null);
      setTargetViewError(describeError(error));
    } finally {
      setTargetBusy(false);
    }
  }, []);

  // stage 5: the benchmark objective lane
  const handleObjective = useCallback(async () => {
    setObjectiveBusy(true);
    setObjectiveError(null);
    setObjectiveResult(null);
    try {
      if (objectiveName === "") {
        throw new Error("pick one of the catalog's own objective names");
      }
      const scores = parseScoresInput(
        objectiveScoresText,
        "the objective scores",
      );
      const response = await dockingApi.dockstringObjective({
        name: objectiveName,
        scores,
        qed: objectiveQed,
      });
      setObjectiveResult(response);
    } catch (error) {
      setObjectiveError(describeError(error));
    } finally {
      setObjectiveBusy(false);
    }
  }, [objectiveName, objectiveScoresText, objectiveQed]);

  // stage 6: the dock lane (job by default; sync answers directly)
  const handleDock = useCallback(async () => {
    if (ligands.length === 0 || receptorText.trim() === "") {
      return;
    }
    setDockBusy(true);
    setDockError(null);
    setDockRowsView(null);
    setDockJobId(null);
    setDockProgress([]);
    try {
      const response = await dockingApi.dock({
        smiles: ligands,
        receptorPdbqt: receptorText,
        engine: dockEngine,
        center: [centerX, centerY, centerZ],
        size: [sizeX, sizeY, sizeZ],
        exhaustiveness,
        numModes,
        workers,
        seed: dockSeed,
        timeoutS,
        mode: dockMode,
      });
      if (response.mode === "sync") {
        setDockRowsView(response);
      } else {
        setDockJobId(response.job.job_id);
      }
    } catch (error) {
      setDockError(describeError(error));
    } finally {
      setDockBusy(false);
    }
  }, [
    ligands, receptorText, dockEngine, centerX, centerY, centerZ,
    sizeX, sizeY, sizeZ, exhaustiveness, numModes, workers, dockSeed,
    timeoutS, dockMode,
  ]);

  const handleDockResult = useCallback((result: unknown) => {
    try {
      setDockRowsView(parseDockRows(result, "dock job result"));
    } catch (error) {
      setDockError(describeError(error));
    }
  }, []);

  // stage 6's live telemetry: the dock job's per-ligand job.progress
  // envelopes (progress-only — the rows come from the REST relay)
  useEffect(() => {
    if (dockJobId === null) {
      return;
    }
    const stream = getEventStream();
    const unsubscribe = stream.subscribe((envelope) => {
      if (envelope.job_id !== dockJobId) {
        return;
      }
      if (envelope.event_type === "job.progress") {
        try {
          const payload = parseJobProgressPayload(
            envelope.payload,
            "job.progress",
          );
          setDockProgress((prev) => [...prev, payload]);
        } catch (error) {
          setDockError(describeError(error));
        }
      }
    });
    return unsubscribe;
  }, [dockJobId]);

  // stage 7: the pasted-log parse lanes
  const handleParseVina = useCallback(async () => {
    setVinaBusy(true);
    setVinaError(null);
    setVinaResult(null);
    try {
      const response = await dockingApi.parseVina(vinaText);
      setVinaResult(response);
    } catch (error) {
      setVinaError(describeError(error));
    } finally {
      setVinaBusy(false);
    }
  }, [vinaText]);

  const handleParseDlg = useCallback(async () => {
    setDlgBusy(true);
    setDlgError(null);
    setDlgResult(null);
    try {
      const response = await dockingApi.parseDlg(dlgText);
      setDlgResult(response);
    } catch (error) {
      setDlgError(describeError(error));
    } finally {
      setDlgBusy(false);
    }
  }, [dlgText]);

  // stage 8a: the contacts + PLIF lane
  const handleParseLigandCsv = useCallback(() => {
    setContactsError(null);
    try {
      setLigandAtoms(parseAtomRows(ligandCsvText, "the ligand CSV"));
    } catch (error) {
      setContactsError(describeError(error));
    }
  }, [ligandCsvText]);

  const handleParseReceptorCsv = useCallback(() => {
    setContactsError(null);
    try {
      setReceptorAtoms(parseAtomRows(receptorCsvText, "the receptor CSV"));
    } catch (error) {
      setContactsError(describeError(error));
    }
  }, [receptorCsvText]);

  const handleContacts = useCallback(async () => {
    setContactsBusy(true);
    setContactsError(null);
    setContactsResult(null);
    try {
      const response = await dockingApi.contacts({
        ligand: ligandAtoms,
        receptor: receptorAtoms,
        cutoff: contactsCutoff,
      });
      setContactsResult(response);
    } catch (error) {
      setContactsError(describeError(error));
    } finally {
      setContactsBusy(false);
    }
  }, [ligandAtoms, receptorAtoms, contactsCutoff]);

  // stage 8b: the pose RMSD lane
  const handlePoseRmsd = useCallback(async () => {
    setRmsdBusy(true);
    setRmsdError(null);
    setRmsdResult(null);
    try {
      const coordsRef = parseTriples(rmsdRefText, "the reference coords");
      const poses = parsePoseBlocks(rmsdPosesText);
      if (coordsRef.length === 0 || poses.length === 0) {
        throw new Error("the RMSD lane needs reference coords and at least one pose");
      }
      const response = await dockingApi.poseRmsd({
        smiles: rmsdSmiles.trim(),
        coordsRef,
        poses,
      });
      setRmsdResult(response);
    } catch (error) {
      setRmsdError(describeError(error));
    } finally {
      setRmsdBusy(false);
    }
  }, [rmsdSmiles, rmsdRefText, rmsdPosesText]);

  // stage 8c: the scoring lane
  const bestAffinity = useMemo(() => {
    if (vinaResult === null || vinaResult.poses.length === 0) {
      return null;
    }
    let best: number | null = null;
    for (const pose of vinaResult.poses) {
      if (
        typeof pose.affinity === "number" &&
        Number.isFinite(pose.affinity) &&
        (best === null || pose.affinity < best)
      ) {
        best = pose.affinity;
      }
    }
    return best;
  }, [vinaResult]);

  const handleLigandEfficiency = useCallback(async () => {
    setLeBusy(true);
    setLeError(null);
    setLeResult(null);
    try {
      const affinity = Number(leAffinity.trim());
      const heavyAtoms = Number(leHeavyAtoms.trim());
      if (!Number.isFinite(affinity)) {
        throw new Error("affinity must be a finite number (kcal/mol)");
      }
      if (!Number.isFinite(heavyAtoms) || !Number.isInteger(heavyAtoms) || heavyAtoms < 1) {
        throw new Error("heavy atoms must be an integer ≥ 1");
      }
      const response = await dockingApi.ligandEfficiency(affinity, heavyAtoms);
      setLeResult(response);
    } catch (error) {
      setLeError(describeError(error));
    } finally {
      setLeBusy(false);
    }
  }, [leAffinity, leHeavyAtoms]);

  const handleConsensus = useCallback(async () => {
    setConsensusBusy(true);
    setConsensusError(null);
    setConsensusResult(null);
    try {
      if (vinaResult === null || vinaResult.poses.length === 0) {
        throw new Error("the consensus lane needs parsed Vina poses first");
      }
      const columns = POSE_COLUMNS.filter(
        (column) => consensusSelected[column] === true,
      );
      if (columns.length === 0) {
        throw new Error("the consensus needs at least one selected column");
      }
      const rows = vinaResult.poses.map((pose) => ({
        mode: pose.mode,
        affinity: pose.affinity,
        rmsd_ub: pose.rmsd_ub,
        rmsd_lb: pose.rmsd_lb,
      }));
      const higherIsBetter: Record<string, boolean> = {};
      for (const column of columns) {
        higherIsBetter[column] = consensusDirections[column] === true;
      }
      const weightsText = consensusWeightsText.trim();
      let weights: number[] | null = null;
      if (weightsText !== "") {
        const parsed = weightsText
          .split(/[\s,]+/)
          .filter((cell) => cell !== "")
          .map((cell) => Number(cell));
        if (parsed.some((value) => !Number.isFinite(value))) {
          throw new Error("the weights must be numbers, comma- or space-separated");
        }
        weights = parsed;
      }
      const response = await dockingApi.consensus({
        rows,
        columns,
        weights,
        higherIsBetter,
      });
      setConsensusResult(response);
    } catch (error) {
      setConsensusError(describeError(error));
    } finally {
      setConsensusBusy(false);
    }
  }, [vinaResult, consensusSelected, consensusDirections, consensusWeightsText]);

  // the digest-verified download lane (§15: the token-header raw lane)
  const verifiedDownload = useCallback(
    async (artifact: ArtifactView): Promise<string> => {
      const download = await artifactsApi.download(artifact.artifact_id);
      let status: string;
      if (download.digest_verified === true) {
        status = `digest verified: ${download.actual_sha256?.slice(0, 16)}…`;
      } else if (download.digest_verified === null) {
        status =
          "downloaded; local digest unavailable (insecure context) — the route's X-Artifact-SHA256 was received";
      } else {
        status =
          "DIGEST MISMATCH — the downloaded bytes differ from the route's promise";
      }
      const url = URL.createObjectURL(download.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = download.filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      return status;
    },
    [],
  );

  const handleLigandDownload = useCallback(async () => {
    if (ligandResult === null) {
      return;
    }
    setLigandDigest(null);
    try {
      setLigandDigest(await verifiedDownload(ligandResult.artifact));
    } catch (error) {
      setLigandError(describeError(error));
    }
  }, [ligandResult, verifiedDownload]);

  // stage 9: load the prepared ligand's PDBQT text for the 3D view —
  // the same token-header raw lane the download rides, the text kept
  // for display (the backend's own preparation stays the authority)
  const handleLigand3dLoad = useCallback(async () => {
    if (ligandResult === null) {
      return;
    }
    setLigand3dBusy(true);
    setLigand3dError(null);
    try {
      const download = await artifactsApi.download(
        ligandResult.artifact.artifact_id,
      );
      setLigand3dText(await download.blob.text());
    } catch (error) {
      setLigand3dError(describeError(error));
    } finally {
      setLigand3dBusy(false);
    }
  }, [ligandResult]);

  // the parsed 3D scenes — rebuilt from the texts that actually
  // arrived; a parse failure renders the honest error, never a guess
  const ligand3dScene = useMemo(() => {
    if (ligand3dText === null) {
      return null;
    }
    try {
      return pdbqtToScene(ligand3dText, "the ligand PDBQT");
    } catch (error) {
      return { error: describeError(error) } as const;
    }
  }, [ligand3dText]);

  const receptor3dScene = useMemo(() => {
    if (receptorText.trim() === "") {
      return null;
    }
    try {
      return pdbqtToScene(receptorText, "the receptor PDBQT");
    } catch (error) {
      return { error: describeError(error) } as const;
    }
  }, [receptorText]);

  const handleReceptorDownload = useCallback(async () => {
    if (receptorResult === null) {
      return;
    }
    setReceptorDigest(null);
    try {
      setReceptorDigest(await verifiedDownload(receptorResult.artifact));
    } catch (error) {
      setReceptorError(describeError(error));
    }
  }, [receptorResult, verifiedDownload]);

  // stage 9: the figure lane
  const handleVisualize = useCallback(async () => {
    setFigureBusy(true);
    setFigureError(null);
    setPoseFigure(null);
    setFigureDigest(null);
    try {
      if (vinaResult === null || vinaResult.poses.length < 1) {
        throw new Error("the figure lane needs at least one parsed Vina pose");
      }
      const artifact = await dockingApi.poseAffinityFigure(
        vinaResult.poses,
        figureFormat,
      );
      setPoseFigure(artifact);
    } catch (error) {
      setFigureError(describeError(error));
    } finally {
      setFigureBusy(false);
    }
  }, [vinaResult, figureFormat]);

  const handleFigureDownload = useCallback(async () => {
    if (poseFigure === null) {
      return;
    }
    setFigureDigest(null);
    try {
      setFigureDigest(await verifiedDownload(poseFigure));
    } catch (error) {
      setFigureError(describeError(error));
    }
  }, [poseFigure, verifiedDownload]);

  // stage 10: the export lane (the parsed poses / the dock rows)
  const exportRows = useMemo(() => {
    if (exportSource === "dock-rows") {
      return dockRowsView === null ? [] : [...dockRowsView.rows];
    }
    if (vinaResult === null) {
      return [];
    }
    return vinaResult.poses.map((pose) => ({
      mode: pose.mode,
      affinity: pose.affinity,
      rmsd_ub: pose.rmsd_ub,
      rmsd_lb: pose.rmsd_lb,
    }));
  }, [exportSource, vinaResult, dockRowsView]);

  const handleExport = useCallback(async () => {
    setExportBusy(true);
    setExportError(null);
    setExportArtifact(null);
    setExportDigest(null);
    try {
      if (exportRows.length === 0) {
        throw new Error("the export carries no rows");
      }
      if (exportRows.length > TABLE_EXPORT_MAX_ROWS) {
        throw new Error(
          `the export carries ${exportRows.length} rows; the table export route's own cap is ${TABLE_EXPORT_MAX_ROWS}`,
        );
      }
      const artifact = await dockingApi.exportTable({
        rows: exportRows,
        format: exportFormat,
        filename:
          exportSource === "dock-rows" ? "docking-dock-rows" : "docking-poses",
      });
      setExportArtifact(artifact);
    } catch (error) {
      setExportError(describeError(error));
    } finally {
      setExportBusy(false);
    }
  }, [exportRows, exportFormat, exportSource]);

  const handleExportDownload = useCallback(async () => {
    if (exportArtifact === null) {
      return;
    }
    setExportDigest(null);
    try {
      setExportDigest(await verifiedDownload(exportArtifact));
    } catch (error) {
      setExportError(describeError(error));
    }
  }, [exportArtifact, verifiedDownload]);

  const noEngineLaunchable = useMemo(
    () =>
      enginesCatalog !== null &&
      enginesCatalog.engines.length > 0 &&
      enginesCatalog.engines.every((entry) => !entry.launchable),
    [enginesCatalog],
  );

  const dockRowColumns = useMemo(
    () =>
      dockRowsView === null ? [] : unionRowKeys(dockRowsView.rows),
    [dockRowsView],
  );

  const consensusRowColumns = useMemo(
    () =>
      consensusResult === null ? [] : unionRowKeys(consensusResult.rows),
    [consensusResult],
  );

  const lastProgress =
    dockProgress.length > 0
      ? (dockProgress[dockProgress.length - 1] ?? null)
      : null;

  return (
    <div className="mx-auto max-w-7xl px-4 pb-16 pt-6">
      <h1 className="text-xl font-semibold text-[var(--color-foreground)]">
        Molecular Docking
      </h1>
      <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
        The §11.5 workflow on the REAL backend: ligand and receptor
        preparation with registered artifacts, a DOCKSTRING-or-manual
        binding box, the honest engine catalog, docking as a managed
        job, pasted-log pose inspection, contacts + PLIF + RMSD +
        scoring analysis, and digest-verified figure/table exports.
      </p>

      <StageHeader
        index={1}
        title="Select Ligands"
        hint="paste one SMILES per row — parsed to the deduplicated list the dock lane docks"
        icon={<Pill aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Ligand list</CardTitle>
            <CardDescription>
              One SMILES per row. Duplicates collapse into the
              deduplicated list — the count badges state both numbers,
              never a silent edit.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <textarea
              value={ligandText}
              onChange={(event) => setLigandText(event.target.value)}
              rows={6}
              spellCheck={false}
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
            />
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 font-semibold text-[var(--color-foreground)]">
                {parsedSmilesCount} rows
              </span>{" "}
              <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 font-semibold text-[var(--color-foreground)]">
                {ligands.length} unique
              </span>{" "}
              <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 font-semibold text-[var(--color-foreground)]">
                {parsedSmilesCount - ligands.length} duplicates
              </span>{" "}
              — the route caps the request at 2000 SMILES, its own bound.
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={2}
        title="Prepare Ligands"
        hint="the REAL RDKit ETKDG embedding + Meeko typing — a registered, digest-verified artifact"
        icon={<Boxes aria-hidden className="h-4 w-4" />}
      >
        {ligandError !== null ? <ErrorBanner message={ligandError} /> : null}
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Ligand PDBQT</CardTitle>
              <CardDescription>
                The `LigandPdbqtRequest` bounds, verbatim from the route
                (name ≤ 80 chars, charge −10…10, seed 0…2³¹−1).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <label className="flex flex-col gap-1 text-xs">
                the ligand to prepare
                <select
                  value={selectedSmiles}
                  onChange={(event) => setSelectedLigand(event.target.value)}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  {ligands.length === 0 ? (
                    <option value="">paste ligands above first…</option>
                  ) : null}
                  {ligands.map((smiles) => (
                    <option key={smiles} value={smiles}>
                      {smiles}
                    </option>
                  ))}
                </select>
              </label>
              <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
                <label className="flex flex-col gap-1">
                  name (blank = route default)
                  <input
                    value={ligName}
                    onChange={(event) => setLigName(event.target.value)}
                    placeholder="ethanol"
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  charge (blank = auto)
                  <input
                    value={ligCharge}
                    onChange={(event) => setLigCharge(event.target.value)}
                    placeholder="0"
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  seed
                  <input
                    type="number"
                    min={0}
                    value={ligSeed}
                    onChange={(event) => setLigSeed(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleLigandPdbqt()}
                  disabled={ligandBusy || selectedSmiles === ""}
                >
                  {ligandBusy ? "embedding…" : "Prepare ligand PDBQT"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/ligand/pdbqt — an invalid SMILES is
                  the route&apos;s own honest 422, surfaced verbatim
                </span>
              </div>
            </CardContent>
          </Card>
          {ligandResult !== null ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  {ligandResult.artifact.filename}
                </CardTitle>
                <CardDescription>
                  {ligandResult.smiles} · {ligandResult.size_bytes} B ·{" "}
                  {ligandResult.artifact.sha256.slice(0, 16)}… · artifact{" "}
                  {ligandResult.artifact.artifact_id}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-3">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => void handleLigandDownload()}
                  >
                    <ArrowDownToLine aria-hidden className="mr-1 h-3 w-3" />
                    Download
                  </Button>
                  {ligandDigest !== null ? (
                    <span className="text-[11px] text-[var(--color-muted-foreground)]">
                      {ligandDigest}
                    </span>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">The ligand artifact</CardTitle>
                <CardDescription>
                  The prepared ligand PDBQT registers here — size in bytes,
                  sha256, and the digest-verified download lane.
                </CardDescription>
              </CardHeader>
            </Card>
          )}
        </div>
      </StageHeader>

      <StageHeader
        index={3}
        title="Prepare Receptor"
        hint="the REAL prepare_protein audit + the cleaned receptor artifact"
        icon={<Dna aria-hidden className="h-4 w-4" />}
      >
        {receptorError !== null ? <ErrorBanner message={receptorError} /> : null}
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Receptor PDBQT text</CardTitle>
              <CardDescription>
                Paste the receptor PDBQT; the preparation runs
                server-side and registers the cleaned receptor as a
                downloadable artifact.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <textarea
                value={receptorText}
                onChange={(event) => setReceptorText(event.target.value)}
                rows={8}
                spellCheck={false}
                placeholder="REMARK  Name = receptor&#10;ATOM      1  CA  ALA A   1 ..."
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
              />
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                <label className="flex flex-col gap-1">
                  filename
                  <input
                    value={receptorFilename}
                    onChange={(event) => setReceptorFilename(event.target.value)}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  chain (blank = all)
                  <input
                    value={receptorChain}
                    onChange={(event) => setReceptorChain(event.target.value)}
                    placeholder="A"
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={removeWaters}
                    onChange={(event) => setRemoveWaters(event.target.checked)}
                  />
                  remove waters
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={keepHetatm}
                    onChange={(event) => setKeepHetatm(event.target.checked)}
                  />
                  keep HETATM
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={stripHydrogens}
                    onChange={(event) => setStripHydrogens(event.target.checked)}
                  />
                  strip hydrogens
                </label>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleReceptorPrepare()}
                  disabled={receptorBusy || receptorText.trim() === ""}
                >
                  {receptorBusy ? "preparing…" : "Prepare receptor"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/receptor/prepare — the four
                  preparation options are the route&apos;s own
                  `ReceptorPrepareRequest` fields, verbatim
                </span>
              </div>
            </CardContent>
          </Card>
          {receptorResult !== null ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  {receptorResult.artifact.filename}
                </CardTitle>
                <CardDescription>
                  mode {receptorResult.report.mode} ·{" "}
                  {receptorResult.report.input_atoms} input atoms ·{" "}
                  {receptorResult.report.output_atoms} output atoms ·
                  source {receptorResult.report.source}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  {receptorResult.report.waters_removed} waters removed ·{" "}
                  {receptorResult.report.hydrogens_removed} hydrogens
                  stripped · chains{" "}
                  {receptorResult.report.chains.join(", ") || "—"} ·{" "}
                  {receptorResult.report.residues} residues ·{" "}
                  {receptorResult.artifact.size_bytes} B
                </p>
                <div className="mt-2 flex items-center gap-3">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => void handleReceptorDownload()}
                  >
                    <ArrowDownToLine aria-hidden className="mr-1 h-3 w-3" />
                    Download
                  </Button>
                  {receptorDigest !== null ? (
                    <span className="text-[11px] text-[var(--color-muted-foreground)]">
                      {receptorDigest}
                    </span>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  The receptor report + artifact
                </CardTitle>
                <CardDescription>
                  The preparation audit (input/output atoms, removed
                  waters/hydrogens, chains) and the cleaned receptor
                  artifact register here.
                </CardDescription>
              </CardHeader>
            </Card>
          )}
        </div>
      </StageHeader>

      <StageHeader
        index={4}
        title="Define Binding Site/Grid"
        hint="a manual center/size box or a REAL DOCKSTRING target box"
        icon={<Target aria-hidden className="h-4 w-4" />}
      >
        {targetViewError !== null ? <ErrorBanner message={targetViewError} /> : null}
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Manual grid box</CardTitle>
              <CardDescription>
                The `DockRequest` defaults, verbatim: center 0, 0, 0 and
                size 22.5, 22.5, 22.5 (Å).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <label className="flex flex-col gap-1">
                  center x
                  <input
                    type="number"
                    step={0.1}
                    value={centerX}
                    onChange={(event) => setCenterX(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  center y
                  <input
                    type="number"
                    step={0.1}
                    value={centerY}
                    onChange={(event) => setCenterY(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  center z
                  <input
                    type="number"
                    step={0.1}
                    value={centerZ}
                    onChange={(event) => setCenterZ(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  size x
                  <input
                    type="number"
                    step={0.5}
                    value={sizeX}
                    onChange={(event) => setSizeX(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  size y
                  <input
                    type="number"
                    step={0.5}
                    value={sizeY}
                    onChange={(event) => setSizeY(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  size z
                  <input
                    type="number"
                    step={0.5}
                    value={sizeZ}
                    onChange={(event) => setSizeZ(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">DOCKSTRING target box</CardTitle>
              <CardDescription>
                The 58-target catalog — picking a row loads its box +
                config text (a GET per target; the route&apos;s own 422
                for unknown names).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <input
                value={targetQuery}
                onChange={(event) => setTargetQuery(event.target.value)}
                placeholder="search the target catalog…"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
              />
              <div className="mt-2 max-h-56 overflow-auto rounded-lg border border-[var(--color-border)]">
                <table className="w-full border-collapse text-xs">
                  <thead className="sticky top-0 bg-[var(--color-muted)]">
                    <tr>
                      {["Target", "Center (x, y, z)", "Size (x, y, z)"].map(
                        (header) => (
                          <th
                            key={header}
                            scope="col"
                            className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                          >
                            {header}
                          </th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTargets.map((row) => (
                      <tr
                        key={row.target}
                        onClick={() => void selectTarget(row.target)}
                        className={`cursor-pointer border-t border-[var(--color-border)] ${
                          targetView?.name === row.target
                            ? "bg-[var(--color-accent)]"
                            : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                        }`}
                      >
                        <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                          {row.target}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {formatValue(row.center_x)}, {formatValue(row.center_y)},{" "}
                          {formatValue(row.center_z)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {formatValue(row.size_x)}, {formatValue(row.size_y)},{" "}
                          {formatValue(row.size_z)}
                        </td>
                      </tr>
                    ))}
                    {filteredTargets.length === 0 ? (
                      <tr>
                        <td
                          colSpan={3}
                          className="px-3 py-4 text-center text-[var(--color-muted-foreground)]"
                        >
                          {targetsBusy
                            ? "loading the target catalog…"
                            : targetsError !== null
                              ? "the catalog failed to load — its error is surfaced above"
                              : "no target matches the search"}
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
              {targetBusy ? (
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  fetching the target box…
                </p>
              ) : null}
              {targetView !== null ? (
                <div className="mt-3">
                  <p className="text-xs text-[var(--color-muted-foreground)]">
                    {targetView.name} — the box filled the manual grid
                    inputs above; the six-key config text below is the
                    target&apos;s own:
                  </p>
                  <pre className="mt-2 overflow-auto rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-[11px] text-[var(--color-foreground)]">
                    {targetView.config}
                  </pre>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </StageHeader>

      <StageHeader
        index={5}
        title="Configure Engine"
        hint="the route's own honest catalog + the DOCKSTRING benchmark objectives"
        icon={<Cpu aria-hidden className="h-4 w-4" />}
      >
        {enginesError !== null ? <ErrorBanner message={enginesError} /> : null}
        <div className="mt-1 overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-64 overflow-auto">
            <table className="w-full border-collapse text-xs">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["Engine", "Bundled", "Binary", "Launchable", "Boundary note"].map(
                    (header) => (
                      <th
                        key={header}
                        scope="col"
                        className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                      >
                        {header}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {(enginesCatalog?.engines ?? []).map((entry) => (
                  <tr
                    key={entry.name}
                    className={`border-t border-[var(--color-border)] ${
                      entry.launchable
                        ? "bg-[var(--color-card)]"
                        : "bg-[var(--color-card)] opacity-80"
                    }`}
                  >
                    <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                      {entry.name}
                    </td>
                    <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      {entry.bundled ? "yes" : "no"}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      {entry.binary ?? "—"}
                    </td>
                    <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      {entry.launchable ? "yes" : "no"}
                    </td>
                    <td className="px-3 py-1.5 text-[10px] text-[var(--color-muted-foreground)]">
                      {entry.boundary_note}
                    </td>
                  </tr>
                ))}
                {(enginesCatalog?.engines ?? []).length === 0 ? (
                  <tr>
                    <td
                      colSpan={5}
                      className="px-3 py-4 text-center text-[var(--color-muted-foreground)]"
                    >
                      {enginesBusy
                        ? "loading the engine catalog…"
                        : "the engine catalog carried no entries"}
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
        {enginesCatalog !== null ? (
          <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
            platform {enginesCatalog.platform} · DOCKSTRING{" "}
            {enginesCatalog.dockstring.n_targets} targets · objectives{" "}
            {enginesCatalog.dockstring.objectives.join(", ")} — only the
            engines that are actually available and correctly configured
            are launchable, and the catalog says which (the boundary note
            is the route&apos;s own verdict, rendered verbatim).
          </p>
        ) : null}
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Benchmark objective evaluator
              </CardTitle>
              <CardDescription>
                The DOCKSTRING original objectives over precomputed
                docking scores — the objective&apos;s own required score
                keys are the route&apos;s honest 422 to state (never
                guessed here).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <label className="flex flex-col gap-1 text-xs">
                objective
                <select
                  value={objectiveName}
                  onChange={(event) => setObjectiveName(event.target.value)}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  {(enginesCatalog?.dockstring.objectives ?? []).map(
                    (name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ),
                  )}
                </select>
              </label>
              <label className="mt-3 flex flex-col gap-1 text-xs">
                scores — one KEY value per line (e.g. F2 -9.5)
                <textarea
                  value={objectiveScoresText}
                  onChange={(event) =>
                    setObjectiveScoresText(event.target.value)
                  }
                  rows={4}
                  spellCheck={false}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
                />
              </label>
              <label className="mt-3 flex w-40 flex-col gap-1 text-xs">
                QED (0–1)
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={objectiveQed}
                  onChange={(event) =>
                    setObjectiveQed(Number(event.target.value))
                  }
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                />
              </label>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleObjective()}
                  disabled={objectiveBusy || objectiveName === ""}
                >
                  {objectiveBusy ? "evaluating…" : "Evaluate objective"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/dockstring/objective — stays fully
                  usable whatever the engine boundary says
                </span>
              </div>
              {objectiveResult !== null ? (
                <p className="mt-3 text-sm text-[var(--color-foreground)]">
                  {objectiveResult.objective} ={" "}
                  {formatValue(objectiveResult.value)} (QED{" "}
                  {formatValue(objectiveResult.qed)} ·{" "}
                  {Object.entries(objectiveResult.scores)
                    .map(([key, value]) => `${key}=${formatValue(value)}`)
                    .join(" ·")}
                  )
                </p>
              ) : null}
              {objectiveError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={objectiveError} />
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </StageHeader>

      <StageHeader
        index={6}
        title="Run Docking"
        hint="the REAL dock_smiles_array — job mode by default, REST-authoritative"
        icon={<Play aria-hidden className="h-4 w-4" />}
      >
        {dockError !== null ? <ErrorBanner message={dockError} /> : null}
        {noEngineLaunchable ? (
          <div
            role="note"
            className="mb-3 rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-sm text-[var(--color-foreground)]"
          >
            <span className="font-medium">
              No docking engine launches on this host.
            </span>{" "}
            <span className="text-[var(--color-muted-foreground)]">
              The catalog&apos;s own boundary notes, verbatim from the
              engines endpoint — never a hardcoded client guess. The
              dock button below still submits: the route pre-validates
              the launch and answers its honest 422, whose detail this
              stage surfaces as its error state. A docking result is
              never faked.
            </span>
            <ul className="mt-2 list-disc pl-5 text-xs text-[var(--color-muted-foreground)]">
              {(enginesCatalog?.engines ?? [])
                .filter((entry) => !entry.launchable)
                .map((entry) => (
                  <li key={entry.name}>
                    {entry.name}: {entry.boundary_note}
                  </li>
                ))}
            </ul>
          </div>
        ) : null}
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Docking parameters</CardTitle>
              <CardDescription>
                The `DockRequest` bounds, verbatim from the route.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <label className="flex flex-col gap-1">
                  engine
                  <select
                    value={dockEngine}
                    onChange={(event) =>
                      setDockEngine(
                        event.target.value === "autodock4"
                          ? "autodock4"
                          : "vina",
                      )
                    }
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  >
                    <option value="vina">vina</option>
                    <option value="autodock4">autodock4</option>
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  mode
                  <select
                    value={dockMode}
                    onChange={(event) =>
                      setDockMode(event.target.value === "sync" ? "sync" : "job")
                    }
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  >
                    <option value="job">job (tracked)</option>
                    <option value="sync">sync (direct answer)</option>
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  exhaustiveness (1–64)
                  <input
                    type="number"
                    min={1}
                    max={64}
                    value={exhaustiveness}
                    onChange={(event) =>
                      setExhaustiveness(Number(event.target.value))
                    }
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  num modes (1–36)
                  <input
                    type="number"
                    min={1}
                    max={36}
                    value={numModes}
                    onChange={(event) => setNumModes(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  workers (1–16)
                  <input
                    type="number"
                    min={1}
                    max={16}
                    value={workers}
                    onChange={(event) => setWorkers(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  seed
                  <input
                    type="number"
                    min={0}
                    value={dockSeed}
                    onChange={(event) => setDockSeed(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  timeout s (0–7200)
                  <input
                    type="number"
                    min={1}
                    max={7200}
                    step={60}
                    value={timeoutS}
                    onChange={(event) => setTimeoutS(Number(event.target.value))}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleDock()}
                  disabled={
                    dockBusy ||
                    dockJobId !== null ||
                    ligands.length === 0 ||
                    receptorText.trim() === ""
                  }
                >
                  {dockBusy ? "submitting…" : "Dock ligands"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/dock — {ligands.length} ligands ·
                  box {formatValue(centerX)}/{formatValue(centerY)}/
                  {formatValue(centerZ)} ± {formatValue(sizeX)}/
                  {formatValue(sizeY)}/{formatValue(sizeZ)}
                </span>
              </div>
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                Job mode is the default — the COMPLETED job&apos;s REST
                result is authoritative; the WS job.progress envelopes
                drive the progress bar per ligand only. The route also
                declares a sync mode: it answers the same rows directly,
                no pretend job tracking. Cancelling is native — the job
                lands CANCELLED, never FAILED, partial results
                preserved.
              </p>
            </CardContent>
          </Card>
          {dockJobId !== null ? (
            <div>
              <JobRunCard
                jobId={dockJobId}
                resultLabel="the dock rows relay"
                onResult={handleDockResult}
              />
              {lastProgress !== null ? (
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  {dockProgress.length} job.progress envelopes · last:{" "}
                  {lastProgress.progress}% · {lastProgress.stage}
                </p>
              ) : (
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  the backend&apos;s own progress callback drives one
                  job.progress envelope per finished ligand
                </p>
              )}
            </div>
          ) : null}
        </div>
        {dockRowsView !== null ? (
          <div className="mt-3">
            <p className="text-sm text-[var(--color-foreground)]">
              {dockRowsView.n_ligands} ligands · {dockRowsView.n_ok} ok ·{" "}
              {dockRowsView.n_failed} failed — the backend&apos;s own
              rows, verbatim (never faked, never re-typed)
            </p>
            <div className="mt-2 max-h-72 overflow-auto rounded-lg border border-[var(--color-border)]">
              <table className="w-full border-collapse text-xs">
                <thead className="sticky top-0 bg-[var(--color-muted)]">
                  <tr>
                    {dockRowColumns.map((header) => (
                      <th
                        key={header}
                        scope="col"
                        className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dockRowsView.rows.map((row, i) => (
                    <tr
                      key={i}
                      className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                    >
                      {dockRowColumns.map((column) => (
                        <td
                          key={column}
                          className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]"
                        >
                          {formatValue(row[column])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <p className="mt-3 text-sm text-[var(--color-muted-foreground)]">
            The dock lane needs the ligand list (stage 1) and the
            receptor PDBQT text (stage 3) — the box comes from stage 4
            and the engine from stage 5.
          </p>
        )}
      </StageHeader>

      <StageHeader
        index={7}
        title="Inspect Poses"
        hint="the REAL parsers over pasted engine logs — vina table + AutoDock4 DLG"
        icon={<FileSearch aria-hidden className="h-4 w-4" />}
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Vina log</CardTitle>
              <CardDescription>
                Paste a Vina output log (or output PDBQT REMARKs); the
                mode table is parsed server-side — a log with no pose
                table is the route&apos;s honest 422.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <textarea
                value={vinaText}
                onChange={(event) => setVinaText(event.target.value)}
                rows={8}
                spellCheck={false}
                placeholder="mode | affinity | dist from best mode"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
              />
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleParseVina()}
                  disabled={vinaBusy || vinaText.trim() === ""}
                >
                  {vinaBusy ? "parsing…" : "Parse Vina log"}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setVinaText(SAMPLE_VINA_LOG)}
                >
                  Load sample
                </Button>
              </div>
              {vinaError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={vinaError} />
                </div>
              ) : null}
              {vinaResult !== null ? (
                <div className="mt-3">
                  <p className="text-xs text-[var(--color-muted-foreground)]">
                    {vinaResult.n_poses} poses parsed — the table below
                    feeds Visualize, Analyze and Export verbatim
                  </p>
                  <div className="mt-2 max-h-56 overflow-auto rounded-md border border-[var(--color-border)]">
                    <table className="w-full border-collapse text-xs">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          {["mode", "affinity", "rmsd_ub", "rmsd_lb"].map(
                            (header) => (
                              <th
                                key={header}
                                scope="col"
                                className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                              >
                                {header}
                              </th>
                            ),
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {vinaResult.poses.map((pose) => (
                          <tr
                            key={pose.mode}
                            className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                          >
                            <td className="px-3 py-1.5 text-[var(--color-foreground)]">
                              {pose.mode}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {formatValue(pose.affinity)}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {formatValue(pose.rmsd_ub)}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {formatValue(pose.rmsd_lb)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">AutoDock4 DLG log</CardTitle>
              <CardDescription>
                Paste a DLG; the CLUSTERING histogram table + the
                summary (top / most-populated cluster) are parsed
                server-side — a run that stopped before clustering 422s
                honestly.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <textarea
                value={dlgText}
                onChange={(event) => setDlgText(event.target.value)}
                rows={8}
                spellCheck={false}
                placeholder="CLUSTERING HISTOGRAM"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
              />
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleParseDlg()}
                  disabled={dlgBusy || dlgText.trim() === ""}
                >
                  {dlgBusy ? "parsing…" : "Parse DLG log"}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setDlgText(SAMPLE_DLG_LOG)}
                >
                  Load sample
                </Button>
              </div>
              {dlgError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={dlgError} />
                </div>
              ) : null}
              {dlgResult !== null ? (
                <div className="mt-3">
                  <p className="text-xs text-[var(--color-muted-foreground)]">
                    {dlgResult.summary.n_clusters} clusters · top{" "}
                    {formatValue(dlgResult.summary.top_lowest_energy)}
                    kcal/mol · most populated cluster{" "}
                    {dlgResult.summary.best_rank} with{" "}
                    {dlgResult.summary.best_num_in_cluster} poses
                  </p>
                  <div className="mt-2 max-h-56 overflow-auto rounded-md border border-[var(--color-border)]">
                    <table className="w-full border-collapse text-xs">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          {[
                            "rank",
                            "lowest_energy",
                            "mean_energy",
                            "num_in_cluster",
                          ].map((header) => (
                            <th
                              key={header}
                              scope="col"
                              className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                            >
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {dlgResult.clusters.map((cluster) => (
                          <tr
                            key={cluster.rank}
                            className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                          >
                            <td className="px-3 py-1.5 text-[var(--color-foreground)]">
                              {cluster.rank}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {formatValue(cluster.lowest_energy)}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {formatValue(cluster.mean_energy)}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {cluster.num_in_cluster}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </StageHeader>

      <StageHeader
        index={8}
        title="Analyze Interactions"
        hint="contacts + PLIF, symmetry-aware pose RMSD, ligand efficiency, consensus"
        icon={<Network aria-hidden className="h-4 w-4" />}
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Contact atoms (CSV)</CardTitle>
              <CardDescription>
                Paste the two atom lists — columnar text in the route&apos;s
                `ContactAtomSpec` order (serial name element resname
                chain resseq x y z [type]); the parsed rows stay editable
                below.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <label className="flex flex-col gap-1 text-xs">
                ligand atoms
                <textarea
                  value={ligandCsvText}
                  onChange={(event) => setLigandCsvText(event.target.value)}
                  rows={4}
                  spellCheck={false}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
                />
              </label>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <Button size="sm" variant="outline" onClick={handleParseLigandCsv}>
                  Parse ligand rows
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setLigandCsvText(SAMPLE_LIGAND_ATOMS_CSV)}
                >
                  Load sample
                </Button>
              </div>
              <AtomRowsEditor
                label="ligand"
                rows={ligandAtoms}
                onChange={setLigandAtoms}
              />
              <label className="mt-4 flex flex-col gap-1 text-xs">
                receptor atoms
                <textarea
                  value={receptorCsvText}
                  onChange={(event) => setReceptorCsvText(event.target.value)}
                  rows={4}
                  spellCheck={false}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
                />
              </label>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleParseReceptorCsv}
                >
                  Parse receptor rows
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setReceptorCsvText(SAMPLE_RECEPTOR_ATOMS_CSV)}
                >
                  Load sample
                </Button>
              </div>
              <AtomRowsEditor
                label="receptor"
                rows={receptorAtoms}
                onChange={setReceptorAtoms}
              />
              <label className="mt-3 flex w-40 flex-col gap-1 text-xs">
                cutoff Å (0–20)
                <input
                  type="number"
                  min={0.5}
                  max={20}
                  step={0.5}
                  value={contactsCutoff}
                  onChange={(event) =>
                    setContactsCutoff(Number(event.target.value))
                  }
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                />
              </label>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleContacts()}
                  disabled={
                    contactsBusy ||
                    ligandAtoms.length === 0 ||
                    receptorAtoms.length === 0
                  }
                >
                  {contactsBusy ? "computing…" : "Compute contacts + PLIF"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/contacts — the interaction classes
                  legend and the PLIF matrix are the backend&apos;s own
                </span>
              </div>
              {contactsError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={contactsError} />
                </div>
              ) : null}
              {contactsResult !== null ? (
                <div className="mt-3">
                  <p className="text-xs text-[var(--color-muted-foreground)]">
                    {contactsResult.n_contacts} contacts within{" "}
                    {formatValue(contactsResult.cutoff)} Å · classes:{" "}
                    {contactsResult.interaction_classes.join(" · ")}
                  </p>
                  <div className="mt-2 max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
                    <table className="w-full border-collapse text-xs">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          {[
                            "ligand_serial",
                            "ligand_atom",
                            "receptor_residue",
                            "receptor_atom",
                            "distance",
                            "interaction",
                          ].map((header) => (
                            <th
                              key={header}
                              scope="col"
                              className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                            >
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {contactsResult.contacts.map((contact, i) => (
                          <tr
                            key={i}
                            className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                          >
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {contact.ligand_serial}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {contact.ligand_atom}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {contact.receptor_residue}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {contact.receptor_atom}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {formatValue(contact.distance)}
                            </td>
                            <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {contact.interaction}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {contactsResult.plif.length > 0 ? (
                    <div className="mt-3">
                      <p className="text-xs text-[var(--color-muted-foreground)]">
                        PLIF matrix — per-residue interaction counts (the
                        backend&apos;s own classes + total)
                      </p>
                      <div className="mt-2 max-h-40 overflow-auto rounded-md border border-[var(--color-border)]">
                        <table className="w-full border-collapse text-xs">
                          <thead className="sticky top-0 bg-[var(--color-muted)]">
                            <tr>
                              {[
                                "residue",
                                ...contactsResult.interaction_classes,
                                "total",
                              ].map((header) => (
                                <th
                                  key={header}
                                  scope="col"
                                  className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                                >
                                  {header}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {contactsResult.plif.map((row) => (
                              <tr
                                key={row.residue}
                                className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                              >
                                <td className="px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                                  {row.residue}
                                </td>
                                {contactsResult.interaction_classes.map(
                                  (interactionClass) => (
                                    <td
                                      key={interactionClass}
                                      className="px-3 py-1.5 text-[var(--color-muted-foreground)]"
                                    >
                                      {formatValue(row[interactionClass])}
                                    </td>
                                  ),
                                )}
                                <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                                  {row.total}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Pose RMSD (symmetry-aware)</CardTitle>
              <CardDescription>
                The reference coords (one x y z row per atom) and the
                poses (blocks separated by a blank line) — every pose
                must carry the reference&apos;s atom count, the route&apos;s
                own rule.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <label className="flex flex-col gap-1 text-xs">
                SMILES
                <input
                  value={rmsdSmiles}
                  onChange={(event) => setRmsdSmiles(event.target.value)}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
                />
              </label>
              <label className="mt-2 flex flex-col gap-1 text-xs">
                reference coords
                <textarea
                  value={rmsdRefText}
                  onChange={(event) => setRmsdRefText(event.target.value)}
                  rows={4}
                  spellCheck={false}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
                />
              </label>
              <label className="mt-2 flex flex-col gap-1 text-xs">
                poses (blank line between poses)
                <textarea
                  value={rmsdPosesText}
                  onChange={(event) => setRmsdPosesText(event.target.value)}
                  rows={6}
                  spellCheck={false}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
                />
              </label>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button onClick={() => void handlePoseRmsd()} disabled={rmsdBusy}>
                  {rmsdBusy ? "computing…" : "Compute pose RMSD"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/pose-rmsd
                </span>
              </div>
              {rmsdError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={rmsdError} />
                </div>
              ) : null}
              {rmsdResult !== null ? (
                <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                  {rmsdResult.smiles} · {rmsdResult.n_atoms} atoms ·{" "}
                  {rmsdResult.n_poses} poses · RMSD (Å):{" "}
                  {rmsdResult.rmsd.map((v) => formatValue(v)).join(", ")}
                </p>
              ) : null}
            </CardContent>
          </Card>
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Ligand efficiency</CardTitle>
              <CardDescription>
                The REAL `ligand_efficiency` — affinity ÷ heavy atoms.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <label className="flex flex-col gap-1">
                  affinity (kcal/mol)
                  <input
                    value={leAffinity}
                    onChange={(event) => setLeAffinity(event.target.value)}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  heavy atoms (1–500)
                  <input
                    value={leHeavyAtoms}
                    onChange={(event) => setLeHeavyAtoms(event.target.value)}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  />
                </label>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={bestAffinity === null}
                  onClick={() =>
                    setLeAffinity(bestAffinity === null ? "" : String(bestAffinity))
                  }
                >
                  Use best parsed affinity
                </Button>
                <Button onClick={() => void handleLigandEfficiency()} disabled={leBusy}>
                  {leBusy ? "computing…" : "Compute ligand efficiency"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/scoring/ligand-efficiency
                </span>
              </div>
              {leError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={leError} />
                </div>
              ) : null}
              {leResult !== null ? (
                <p className="mt-3 text-sm text-[var(--color-foreground)]">
                  {formatValue(leResult.affinity_kcal_mol)} kcal/mol ÷{" "}
                  {leResult.num_heavy_atoms} heavy atoms ={" "}
                  <span className="font-semibold">
                    {formatValue(leResult.ligand_efficiency)}
                  </span>{" "}
                  kcal/mol per heavy atom
                </p>
              ) : null}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Consensus scoring</CardTitle>
              <CardDescription>
                The REAL `consensus_breakdown` over the parsed Vina poses
                — columns default to affinity (lower is better, the
                honest default), weights optional and must align (the
                route&apos;s own 422).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-hidden rounded-md border border-[var(--color-border)]">
                <table className="w-full border-collapse text-xs">
                  <thead className="bg-[var(--color-muted)]">
                    <tr>
                      {["column", "include", "direction"].map((header) => (
                        <th
                          key={header}
                          scope="col"
                          className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                        >
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {POSE_COLUMNS.map((column) => (
                      <tr
                        key={column}
                        className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                      >
                        <td className="px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                          {column}
                        </td>
                        <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          <input
                            type="checkbox"
                            checked={consensusSelected[column] === true}
                            onChange={(event) =>
                              setConsensusSelected((prev) => ({
                                ...prev,
                                [column]: event.target.checked,
                              }))
                            }
                          />
                        </td>
                        <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          <select
                            value={
                              consensusDirections[column] === true
                                ? "higher"
                                : "lower"
                            }
                            onChange={(event) =>
                              setConsensusDirections((prev) => ({
                                ...prev,
                                [column]: event.target.value === "higher",
                              }))
                            }
                            className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 text-xs"
                          >
                            <option value="lower">lower is better</option>
                            <option value="higher">higher is better</option>
                          </select>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <label className="mt-3 flex flex-col gap-1 text-xs">
                weights (blank = none; must align with the columns)
                <input
                  value={consensusWeightsText}
                  onChange={(event) =>
                    setConsensusWeightsText(event.target.value)
                  }
                  placeholder="e.g. 1, 1"
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                />
              </label>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  onClick={() => void handleConsensus()}
                  disabled={consensusBusy || vinaResult === null}
                >
                  {consensusBusy ? "scoring…" : "Score consensus"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  POST /api/v1/docking/scoring/consensus over the stage-7
                  pose table
                </span>
              </div>
              {consensusError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={consensusError} />
                </div>
              ) : null}
              {consensusResult !== null ? (
                <div className="mt-3">
                  <p className="text-xs text-[var(--color-muted-foreground)]">
                    {consensusResult.n_rows} rows · columns{" "}
                    {consensusResult.columns.join(", ")} · the
                    `{consensusResult.columns.join("_")}_norm` + consensus
                    values are the backend&apos;s own normalization
                  </p>
                  <div className="mt-2 max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
                    <table className="w-full border-collapse text-xs">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          {consensusRowColumns.map((header) => (
                            <th
                              key={header}
                              scope="col"
                              className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                            >
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {consensusResult.rows.map((row, i) => (
                          <tr
                            key={i}
                            className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                          >
                            {consensusRowColumns.map((column) => (
                              <td
                                key={column}
                                className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]"
                              >
                                {formatValue(row[column])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </StageHeader>

      <StageHeader
        index={9}
        title="Visualize"
        hint="the parsed poses' REAL affinity profile — a bar chart per mode"
        icon={<LineChart aria-hidden className="h-4 w-4" />}
      >
        {figureError !== null ? <ErrorBanner message={figureError} /> : null}
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-xs">
            format
            <select
              value={figureFormat}
              onChange={(event) =>
                setFigureFormat(
                  event.target.value as "PNG" | "SVG" | "TIFF" | "JPEG",
                )
              }
              className="w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            >
              {(["PNG", "SVG", "TIFF", "JPEG"] as const).map((format) => (
                <option key={format} value={format}>
                  {format}
                </option>
              ))}
            </select>
          </label>
          <Button
            onClick={() => void handleVisualize()}
            disabled={figureBusy || vinaResult === null}
          >
            {figureBusy ? "rendering…" : "Register pose figure"}
          </Button>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            POST /api/v1/artifacts/figure — the §17 four-format export
            stack; the lane needs at least one parsed Vina pose
          </span>
        </div>
        {poseFigure !== null ? (
          <Card className="mt-3">
            <CardHeader>
              <CardTitle className="text-base">{poseFigure.filename}</CardTitle>
              <CardDescription>
                {poseFigure.size_bytes} B · {poseFigure.sha256.slice(0, 16)}… ·
                artifact {poseFigure.artifact_id}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void handleFigureDownload()}
                >
                  <ArrowDownToLine aria-hidden className="mr-1 h-3 w-3" />
                  Download
                </Button>
                {figureDigest !== null ? (
                  <span className="text-[11px] text-[var(--color-muted-foreground)]">
                    {figureDigest}
                  </span>
                ) : null}
              </div>
            </CardContent>
          </Card>
        ) : (
          <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
            The figure artifact registers from the stage-7 pose table —
            one bar per parsed mode, the affinity the parser actually
            read out of the log.
          </p>
        )}

        {/* Phase Eighteen — the genuine 3D structure lane (Three.js):
            the prepared ligand (its PDBQT artifact fetched through the
            digest-verified raw lane) and the pasted receptor rendered
            as ball-and-stick; the pose coordinates are NOT parsed
            from the engine logs, so only the structures that
            actually arrived render — never an imagined complex. */}
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                variant="outline"
                disabled={ligandResult === null || ligand3dBusy}
                onClick={() => void handleLigand3dLoad()}
              >
                {ligand3dBusy ? "loading…" : "Load the prepared ligand in 3D"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                the stage-2 ligand PDBQT artifact — ball-and-stick, click an
                atom to inspect it
              </span>
            </div>
            {ligand3dError !== null ? (
              <ErrorBanner message={ligand3dError} />
            ) : null}
            {ligand3dScene === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                The ligand 3D view draws only after the stage-2 preparation
                registers its PDBQT artifact and the text is loaded here —
                never before, and never from a guessed structure.
              </p>
            ) : "error" in ligand3dScene ? (
              <ErrorBanner message={ligand3dScene.error} />
            ) : (
              <Molecule3D
                title="Ligand structure (3D)"
                atoms={ligand3dScene.atoms}
                bonds={ligand3dScene.bonds}
                bondNote={`${DOCKING_3D_BOND_NOTE}${ligand3dScene.diagnostics !== "" ? ` — ${ligand3dScene.diagnostics}` : ""}`}
                height={300}
              />
            )}
          </div>
          <div className="flex flex-col gap-2">
            {receptor3dScene === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                The receptor 3D view renders the pasted stage-3 PDBQT text —
                paste ATOM/HETATM records and the ball-and-stick scene draws
                from exactly those records.
              </p>
            ) : "error" in receptor3dScene ? (
              <ErrorBanner message={receptor3dScene.error} />
            ) : (
              <Molecule3D
                title="Receptor structure (3D)"
                atoms={receptor3dScene.atoms}
                bonds={receptor3dScene.bonds}
                bondNote={`${DOCKING_3D_BOND_NOTE}${receptor3dScene.diagnostics !== "" ? ` — ${receptor3dScene.diagnostics}` : ""}`}
                height={300}
              />
            )}
          </div>
        </div>
      </StageHeader>

      <StageHeader
        index={10}
        title="Export"
        hint="the pose/dock rows through the rows-based table lane, digest-verified"
        icon={<ArrowDownToLine aria-hidden className="h-4 w-4" />}
      >
        {exportError !== null ? <ErrorBanner message={exportError} /> : null}
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-xs">
            rows
            <select
              value={exportSource}
              onChange={(event) =>
                setExportSource(
                  event.target.value === "dock-rows"
                    ? "dock-rows"
                    : "vina-poses",
                )
              }
              className="w-40 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            >
              <option value="vina-poses">parsed Vina poses</option>
              <option value="dock-rows" disabled={dockRowsView === null}>
                dock rows
              </option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs">
            format
            <select
              value={exportFormat}
              onChange={(event) =>
                setExportFormat(
                  event.target.value as "csv" | "tsv" | "xlsx",
                )
              }
              className="w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            >
              {(["csv", "tsv", "xlsx"] as const).map((format) => (
                <option key={format} value={format}>
                  {format}
                </option>
              ))}
            </select>
          </label>
          <Button onClick={() => void handleExport()} disabled={exportBusy}>
            {exportBusy ? "exporting…" : "Export rows"}
          </Button>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            POST /api/v1/visualization/table — {exportRows.length} rows ·
            the backend&apos;s own 5000-row cap mirrored client-side
          </span>
        </div>
        {exportArtifact !== null ? (
          <Card className="mt-3">
            <CardHeader>
              <CardTitle className="text-base">
                {exportArtifact.filename}
              </CardTitle>
              <CardDescription>
                {exportArtifact.size_bytes} B ·{" "}
                {exportArtifact.sha256.slice(0, 16)}… · artifact{" "}
                {exportArtifact.artifact_id}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void handleExportDownload()}
                >
                  <ArrowDownToLine aria-hidden className="mr-1 h-3 w-3" />
                  Download
                </Button>
                {exportDigest !== null ? (
                  <span className="text-[11px] text-[var(--color-muted-foreground)]">
                    {exportDigest}
                  </span>
                ) : null}
              </div>
            </CardContent>
          </Card>
        ) : (
          <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
            The export rides the stage-7 pose table (or the stage-6 dock
            rows) — the artifact downloads through the token-header lane
            with digest verification.
          </p>
        )}
      </StageHeader>
    </div>
  );
}
