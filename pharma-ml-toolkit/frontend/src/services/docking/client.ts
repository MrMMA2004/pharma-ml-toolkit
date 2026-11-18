/**
 * The Docking client — part11 §11.5's workflow service (the
 * ligand/receptor preparation, engine, parse, scoring, contacts and
 * dock surface over the REAL backend).
 *
 * The backend contract (mirrored field-for-field from the route source
 * `routes/docking.py`; the request bodies come from the generated
 * OpenAPI types — the same one-DTO-source rule the Python drift guards
 * in `tests/test_docking_workspace.py` pin for this client):
 *
 * * `GET /api/v1/docking/engines` — the honest engine catalog: what
 *   `locate_engine_binary` finds on this host, whether `engine_argv`
 *   can actually launch it (the bundled `vina`/`autogrid4`/
 *   `autodock4` are Windows PE executables — on a non-Windows host
 *   without Wine the launch check fails honestly and the route's own
 *   `boundary_note` says so), plus the DOCKSTRING target/objective
 *   inventory.
 * * `GET /api/v1/docking/dockstring/targets` — the 58-target catalog
 *   (`target_frame`: `target` + the six center/size columns); the
 *   single-target view (`name`, `center`, `size`, `config`) at
 *   `/dockstring/targets/{name}`.
 * * `POST /api/v1/docking/dockstring/objective` — `ObjectiveRequest`
 *   (the F2 / promiscuous_PPAR / selective_JAK2 benchmark objectives
 *   over precomputed docking scores).
 * * `POST /api/v1/docking/receptor/prepare` — `ReceptorPrepareRequest`
 *   over pasted PDBQT text; the cleaned receptor registers as a
 *   downloadable artifact.
 * * `POST /api/v1/docking/ligand/pdbqt` — `LigandPdbqtRequest` (RDKit
 *   ETKDG embedding + Meeko typing); the ligand PDBQT registers as an
 *   artifact.
 * * `POST /api/v1/docking/parse/vina` / `POST /api/v1/docking/parse/dlg`
 *   — `ParseTextRequest` (the REAL `parse_vina_output` /
 *   `parse_dlg_summary` + `parse_dlg_cluster_table` parsers over pasted
 *   engine logs).
 * * `POST /api/v1/docking/scoring/consensus` /
 *   `POST /api/v1/docking/scoring/ligand-efficiency` —
 *   `ConsensusRequest` / `LigandEfficiencyRequest`.
 * * `POST /api/v1/docking/contacts` — `ContactsRequest` (the
 *   `ContactAtomSpec` rows + cutoff): contacts, interaction classes
 *   and the PLIF matrix.
 * * `POST /api/v1/docking/pose-rmsd` — `PoseRmsdRequest` (the
 *   symmetry-aware `best_pose_rmsd`).
 * * `POST /api/v1/docking/dock` — `DockRequest` →
 *   `{mode: "job", job: JobView}` (the default lane) /
 *   `{mode: "sync", n_ligands, n_ok, n_failed, rows}`; the COMPLETED
 *   job's REST result payload is the same `{n_ligands, n_ok,
 *   n_failed, rows}` body — the page relays the authoritative REST
 *   result through the jobs client (§13), never the WS event payload
 *   alone (the WS `job.progress` envelopes drive the progress bar per
 *   ligand, narrowed here by `parseJobProgressPayload`). The route
 *   pre-validates the engine launch up front: an unlaunchable engine
 *   is an environmental boundary — a 422 surfaced verbatim, never a
 *   faked docking result.
 *
 * The figure lane (the parsed Vina poses' REAL affinity profile as a
 * bar chart) rides `POST /api/v1/artifacts/figure`; the export rides
 * the rows-based `POST /api/v1/visualization/table` — both posted from
 * the generated types, parsed here with `parseArtifactView`.
 */

import { api } from "@/services/api/client";
import { parseArtifactView, type ArtifactView } from "@/services/artifacts/client";
import { type JobView, parseJobView } from "@/services/jobs/types";
import type { components } from "@/types/api.d";

type ReceptorPrepareRequest = components["schemas"]["ReceptorPrepareRequest"];
type LigandPdbqtRequest = components["schemas"]["LigandPdbqtRequest"];
type ParseTextRequest = components["schemas"]["ParseTextRequest"];
type ConsensusRequest = components["schemas"]["ConsensusRequest"];
type LigandEfficiencyRequest =
  components["schemas"]["LigandEfficiencyRequest"];
type ContactAtomSpec = components["schemas"]["ContactAtomSpec"];
type ContactsRequest = components["schemas"]["ContactsRequest"];
type PoseRmsdRequest = components["schemas"]["PoseRmsdRequest"];
type DockRequest = components["schemas"]["DockRequest"];
type ObjectiveRequest = components["schemas"]["ObjectiveRequest"];
type FigureRequest = components["schemas"]["FigureRequest"];
type VisualizationTableRequest =
  components["schemas"]["VisualizationTableRequest"];

// -- the response views (hand-written, runtime-narrowed) --------------

/** One `_engine_catalog()` row — the route's own keys. */
export interface EngineCatalogEntry {
  readonly name: string;
  /** The bundled binary was found on this host. */
  readonly bundled: boolean;
  /** The located binary's filename (null when not bundled). */
  readonly binary: string | null;
  /** Whether `engine_argv` can actually launch it here. */
  readonly launchable: boolean;
  /** The route's own environmental boundary note, verbatim. */
  readonly boundary_note: string;
}

/** `GET /engines` — the honest catalog + the DOCKSTRING inventory. */
export interface EnginesResponse {
  readonly engines: readonly EngineCatalogEntry[];
  readonly platform: string;
  readonly dockstring: {
    readonly n_targets: number;
    readonly objectives: readonly string[];
  };
}

/** One `target_frame()` row — the catalog's own columns. */
export interface DockstringTargetRow {
  readonly target: string;
  readonly center_x: number;
  readonly center_y: number;
  readonly center_z: number;
  readonly size_x: number;
  readonly size_y: number;
  readonly size_z: number;
}

export interface DockstringTargetsResponse {
  readonly total: number;
  readonly targets: readonly DockstringTargetRow[];
}

/** `GET /dockstring/targets/{name}` — one target's box + config text. */
export interface DockstringTargetView {
  readonly name: string;
  readonly center: readonly number[];
  readonly size: readonly number[];
  readonly config: string;
}

/** `POST /dockstring/objective` — the benchmark objective's own answer. */
export interface ObjectiveResponse {
  readonly objective: string;
  readonly scores: Readonly<Record<string, number>>;
  readonly qed: number;
  readonly value: number;
}

/** `ProteinReport.as_dict()` — the preparation audit, mirrored. */
export interface ReceptorReportView {
  readonly source: string;
  readonly output_pdbqt: string;
  readonly mode: string;
  readonly input_atoms: number;
  readonly input_hetatms: number;
  readonly waters_removed: number;
  readonly hetatm_removed: number;
  readonly altloc_removed: number;
  readonly hydrogens_removed: number;
  readonly bad_residues_removed: number;
  readonly chains: readonly string[];
  readonly residues: number;
  readonly output_atoms: number;
  readonly atom_types: Readonly<Record<string, number>>;
  readonly charge_min: number;
  readonly charge_max: number;
  readonly charge_mean: number;
  readonly notes: readonly string[];
}

/** `POST /receptor/prepare` — the report + the registered artifact. */
export interface ReceptorPrepareResponse {
  readonly report: ReceptorReportView;
  readonly artifact: ArtifactView;
}

/** `POST /ligand/pdbqt` — the registered ligand artifact lane. */
export interface LigandPdbqtResponse {
  readonly smiles: string;
  readonly name: string;
  readonly size_bytes: number;
  readonly artifact: ArtifactView;
}

/** One `VinaPose` row — the parser's own keys. */
export interface VinaPoseRow {
  readonly mode: number;
  readonly affinity: number | null;
  readonly rmsd_ub: number | null;
  readonly rmsd_lb: number | null;
}

/** `POST /parse/vina` — the parsed pose table. */
export interface VinaParseResponse {
  readonly n_poses: number;
  readonly poses: readonly VinaPoseRow[];
}

/** `parse_dlg_summary`'s own dict — the DLG summary card. */
export interface DlgSummaryView {
  readonly n_clusters: number;
  readonly top_rank: number;
  readonly top_lowest_energy: number;
  readonly top_mean_energy: number;
  readonly top_num_in_cluster: number;
  readonly best_rank: number;
  readonly best_lowest_energy: number;
  readonly best_mean_energy: number;
  readonly best_num_in_cluster: number;
}

/** One `DockingCluster` row — the cluster table's own keys. */
export interface DlgClusterRow {
  readonly rank: number;
  readonly lowest_energy: number;
  readonly mean_energy: number;
  readonly num_in_cluster: number;
}

/** `POST /parse/dlg` — the summary + the cluster table. */
export interface DlgParseResponse {
  readonly summary: DlgSummaryView;
  readonly clusters: readonly DlgClusterRow[];
}

/** `POST /scoring/consensus` — the normalized breakdown rows. */
export interface ConsensusResponse {
  readonly columns: readonly string[];
  readonly n_rows: number;
  readonly rows: readonly Record<string, unknown>[];
}

/** `POST /scoring/ligand-efficiency` — the efficiency answer. */
export interface LigandEfficiencyResponse {
  readonly affinity_kcal_mol: number;
  readonly num_heavy_atoms: number;
  readonly ligand_efficiency: number;
}

/** One contact row — `compute_contacts`'s own emitted keys. */
export interface ContactRow {
  readonly ligand_serial: number;
  readonly ligand_atom: string;
  readonly receptor_residue: string;
  readonly receptor_atom: string;
  readonly distance: number;
  readonly interaction: string;
}

/** One PLIF row: the residue + the per-class counts + the total. */
export interface PlifRow {
  readonly residue: string;
  readonly total: number;
  readonly [interactionClass: string]: number | string;
}

/** `POST /contacts` — the contacts + classes + the PLIF matrix. */
export interface ContactsResponse {
  readonly cutoff: number;
  readonly n_contacts: number;
  readonly interaction_classes: readonly string[];
  readonly contacts: readonly ContactRow[];
  readonly plif: readonly PlifRow[];
}

/** `POST /pose-rmsd` — the symmetry-aware RMSD answer. */
export interface PoseRmsdResponse {
  readonly smiles: string;
  readonly n_atoms: number;
  readonly n_poses: number;
  readonly rmsd: readonly number[];
}

/**
 * The `_dock_body` rows answer — the sync body AND the COMPLETED
 * job's authoritative REST result payload (`{n_ligands, n_ok,
 * n_failed, rows}`).
 */
export interface DockRowsView {
  readonly n_ligands: number;
  readonly n_ok: number;
  readonly n_failed: number;
  readonly rows: readonly Record<string, unknown>[];
}

export interface DockJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export interface DockSyncResponse extends DockRowsView {
  readonly mode: "sync";
}

export type DockResponse = DockJobResponse | DockSyncResponse;

/**
 * One `job.progress` envelope payload — the dock lane's REAL
 * per-ligand telemetry (`JobManager.report_progress`'s own
 * `{progress, stage}` dict; progress-only, never the result).
 */
export interface JobProgressPayload {
  readonly progress: number;
  readonly stage: string;
}

// -- the option payloads ----------------------------------------------

export interface ReceptorPrepareOptions {
  readonly pdbqtText: string;
  readonly filename: string;
  readonly removeWaters: boolean;
  readonly keepHetatm: boolean;
  readonly chain: string | null;
  readonly stripHydrogens: boolean;
}

export interface LigandPdbqtOptions {
  readonly smiles: string;
  readonly name: string | null;
  readonly charge: number | null;
  readonly seed: number;
}

export interface ObjectiveOptions {
  readonly name: string;
  readonly scores: Readonly<Record<string, number>>;
  readonly qed: number;
}

export interface ConsensusOptions {
  readonly rows: readonly Record<string, unknown>[];
  readonly columns: readonly string[];
  readonly weights: readonly number[] | null;
  readonly higherIsBetter: Readonly<Record<string, boolean>> | null;
}

/**
 * One editable atom row — the pasted/edited `ContactAtomSpec` fields
 * as raw strings (narrowed to the contract type by `parseContactAtom`
 * at request time, with labeled errors, never silent defaults).
 */
export interface ContactAtomFields {
  readonly serial: string;
  readonly name: string;
  readonly element: string;
  readonly resname: string;
  readonly chain: string;
  readonly resseq: string;
  readonly x: string;
  readonly y: string;
  readonly z: string;
  readonly autodock_type: string;
}

export interface ContactsOptions {
  readonly ligand: readonly ContactAtomFields[];
  readonly receptor: readonly ContactAtomFields[];
  readonly cutoff: number;
}

export interface PoseRmsdOptions {
  readonly smiles: string;
  readonly coordsRef: readonly (readonly number[])[];
  readonly poses: readonly (readonly (readonly number[])[])[];
}

export interface DockOptions {
  readonly smiles: readonly string[];
  readonly receptorPdbqt: string;
  readonly engine: "vina" | "autodock4";
  readonly center: readonly number[];
  readonly size: readonly number[];
  readonly exhaustiveness: number;
  readonly numModes: number;
  readonly workers: number;
  readonly seed: number;
  readonly timeoutS: number;
  readonly mode?: "job" | "sync";
}

export interface TableExportOptions {
  /** The pose/dock rows the export carries. */
  readonly rows: Readonly<Record<string, unknown>[]>;
  /** The backend's own export enum (visualization.py's own literals). */
  readonly format: "csv" | "tsv" | "xlsx";
  readonly filename: string;
}

export const dockingApi = {
  /**
   * `GET /api/v1/docking/engines` — the honest engine catalog: what is
   * bundled, what actually launches on this host (the boundary note is
   * the route's own environmental verdict, rendered verbatim — never a
   * hardcoded client-side guess), plus the DOCKSTRING inventory.
   */
  async engines(): Promise<EnginesResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/docking/engines",
    );
    return parseEnginesResponse(raw, "engines catalog");
  },

  /** `GET /api/v1/docking/dockstring/targets` — the 58-target catalog. */
  async dockstringTargets(): Promise<DockstringTargetsResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/docking/dockstring/targets",
    );
    const r = expectObject(raw, "dockstring targets");
    const targets = r["targets"];
    if (!Array.isArray(targets)) {
      return { total: 0, targets: [] };
    }
    return {
      total: typeof r["total"] === "number" ? r["total"] : targets.length,
      targets: targets.map((row, i) =>
        parseDockstringTargetRow(row, `targets[${i}]`),
      ),
    };
  },

  /**
   * `GET /api/v1/docking/dockstring/targets/{name}` — one target's
   * center/size box + its six-key config text (the name is the
   * case-sensitive upstream key; an unknown name is the route's own
   * honest 422).
   */
  async dockstringTarget(name: string): Promise<DockstringTargetView> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/docking/dockstring/targets/${encodeURIComponent(name)}`,
    );
    return parseDockstringTargetView(raw, "dockstring target view");
  },

  /**
   * `POST /api/v1/docking/dockstring/objective` — one benchmark
   * objective over precomputed docking scores (the objective's own
   * required score keys are the route's honest 422 to state).
   */
  async dockstringObjective(
    options: ObjectiveOptions,
  ): Promise<ObjectiveResponse> {
    const request: ObjectiveRequest = {
      name: options.name,
      scores: { ...options.scores },
      qed: options.qed,
    };
    const raw = (await api.post(
      "/api/v1/docking/dockstring/objective",
      request,
    )) as Record<string, unknown>;
    return parseObjectiveResponse(raw, "dockstring objective");
  },

  /**
   * `POST /api/v1/docking/receptor/prepare` — the REAL
   * `prepare_protein` over pasted PDBQT text; the cleaned receptor
   * registers as a downloadable artifact (digest-verified at the
   * download lane).
   */
  async receptorPrepare(
    options: ReceptorPrepareOptions,
  ): Promise<ReceptorPrepareResponse> {
    const request: ReceptorPrepareRequest = {
      pdbqt_text: options.pdbqtText,
      filename: options.filename,
      remove_waters: options.removeWaters,
      keep_hetatm: options.keepHetatm,
      chain: options.chain,
      strip_hydrogens: options.stripHydrogens,
    };
    const raw = (await api.post(
      "/api/v1/docking/receptor/prepare",
      request,
    )) as Record<string, unknown>;
    const r = expectObject(raw, "receptor prepare");
    const artifact = r["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("receptor prepare: the route answered without an artifact");
    }
    return {
      report: parseReceptorReport(r["report"], "receptor prepare report"),
      artifact: parseArtifactView(artifact, "receptor prepare artifact"),
    };
  },

  /**
   * `POST /api/v1/docking/ligand/pdbqt` — the REAL RDKit ETKDG
   * embedding + Meeko typing of one SMILES; the ligand PDBQT
   * registers as a downloadable artifact.
   */
  async ligandPdbqt(options: LigandPdbqtOptions): Promise<LigandPdbqtResponse> {
    const request: LigandPdbqtRequest = {
      smiles: options.smiles,
      name: options.name,
      charge: options.charge,
      seed: options.seed,
    };
    const raw = (await api.post(
      "/api/v1/docking/ligand/pdbqt",
      request,
    )) as Record<string, unknown>;
    const r = expectObject(raw, "ligand pdbqt");
    const artifact = r["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("ligand pdbqt: the route answered without an artifact");
    }
    return {
      smiles: typeof r["smiles"] === "string" ? r["smiles"] : "",
      name: typeof r["name"] === "string" ? r["name"] : "",
      size_bytes: typeof r["size_bytes"] === "number" ? r["size_bytes"] : 0,
      artifact: parseArtifactView(artifact, "ligand pdbqt artifact"),
    };
  },

  /**
   * `POST /api/v1/docking/parse/vina` — the REAL `parse_vina_output`
   * over a pasted Vina log/output PDBQT (mode/affinity/rmsd_ub/
   * rmsd_lb per pose; a log without a pose table is the route's own
   * honest 422).
   */
  async parseVina(text: string): Promise<VinaParseResponse> {
    const request: ParseTextRequest = {
      text: text,
    };
    const raw = (await api.post(
      "/api/v1/docking/parse/vina",
      request,
    )) as Record<string, unknown>;
    return parseVinaParseResponse(raw, "parse vina");
  },

  /**
   * `POST /api/v1/docking/parse/dlg` — the REAL `parse_dlg_summary` +
   * `parse_dlg_cluster_table` over a pasted AutoDock4 DLG (a log that
   * stopped before clustering 422s honestly, verbatim).
   */
  async parseDlg(text: string): Promise<DlgParseResponse> {
    const request: ParseTextRequest = {
      text: text,
    };
    const raw = (await api.post(
      "/api/v1/docking/parse/dlg",
      request,
    )) as Record<string, unknown>;
    return parseDlgParseResponse(raw, "parse dlg");
  },

  /**
   * `POST /api/v1/docking/scoring/consensus` — the REAL
   * `consensus_breakdown` normalization over the pose/dock rows (the
   * weights must align with the columns — the route's own 422).
   */
  async consensus(options: ConsensusOptions): Promise<ConsensusResponse> {
    const request: ConsensusRequest = {
      rows: [...options.rows],
      columns: [...options.columns],
      weights: options.weights === null ? null : [...options.weights],
      higher_is_better: options.higherIsBetter,
    };
    const raw = (await api.post(
      "/api/v1/docking/scoring/consensus",
      request,
    )) as Record<string, unknown>;
    const r = expectObject(raw, "consensus");
    const rows = r["rows"];
    return {
      columns: Array.isArray(r["columns"])
        ? r["columns"].map((column) => String(column))
        : [],
      n_rows: typeof r["n_rows"] === "number" ? r["n_rows"] : 0,
      rows: Array.isArray(rows)
        ? rows.map((row, i) => expectObject(row, `consensus rows[${i}]`))
        : [],
    };
  },

  /**
   * `POST /api/v1/docking/scoring/ligand-efficiency` — the REAL
   * `ligand_efficiency` (affinity ÷ heavy atoms).
   */
  async ligandEfficiency(
    affinityKcalMol: number,
    numHeavyAtoms: number,
  ): Promise<LigandEfficiencyResponse> {
    const request: LigandEfficiencyRequest = {
      affinity_kcal_mol: affinityKcalMol,
      num_heavy_atoms: numHeavyAtoms,
    };
    const raw = (await api.post(
      "/api/v1/docking/scoring/ligand-efficiency",
      request,
    )) as Record<string, unknown>;
    return parseLigandEfficiencyResponse(raw, "ligand efficiency");
  },

  /**
   * `POST /api/v1/docking/contacts` — the REAL `compute_contacts` +
   * `plif_matrix` over the two pasted atom lists (each row narrowed
   * to the route's `ContactAtomSpec`; a malformed row is this
   * parser's own labeled error, never a silent guess).
   */
  async contacts(options: ContactsOptions): Promise<ContactsResponse> {
    const request: ContactsRequest = {
      ligand: options.ligand.map((row, i) =>
        parseContactAtom(row, `ligand atom ${i + 1}`),
      ),
      receptor: options.receptor.map((row, i) =>
        parseContactAtom(row, `receptor atom ${i + 1}`),
      ),
      cutoff: options.cutoff,
    };
    const raw = (await api.post(
      "/api/v1/docking/contacts",
      request,
    )) as Record<string, unknown>;
    return parseContactsResponse(raw, "contacts");
  },

  /**
   * `POST /api/v1/docking/pose-rmsd` — the REAL symmetry-aware
   * `best_pose_rmsd` (every pose must match the reference's atom
   * count; the route's own 422 says so, surfaced verbatim).
   */
  async poseRmsd(options: PoseRmsdOptions): Promise<PoseRmsdResponse> {
    const request: PoseRmsdRequest = {
      smiles: options.smiles,
      coords_ref: options.coordsRef.map((row) => [...row]),
      poses: options.poses.map((pose) => pose.map((row) => [...row])),
    };
    const raw = (await api.post(
      "/api/v1/docking/pose-rmsd",
      request,
    )) as Record<string, unknown>;
    return parsePoseRmsdResponse(raw, "pose rmsd");
  },

  /**
   * `POST /api/v1/docking/dock` — the REAL `dock_smiles_array` over
   * the pasted receptor text. Job mode (the default) answers
   * `{mode: "job", job: JobView}` — the COMPLETED job's REST result
   * is the authoritative rows relay (`parseDockRows`); the WS
   * `job.progress` envelopes drive the per-ligand progress bar only.
   * Sync mode answers the same rows directly. The route pre-validates
   * the engine launch up front: on a host where no engine launches,
   * the 422 detail is surfaced verbatim — a docking result is never
   * faked.
   */
  async dock(options: DockOptions): Promise<DockResponse> {
    const request: DockRequest = {
      smiles: [...options.smiles],
      receptor_pdbqt: options.receptorPdbqt,
      engine: options.engine,
      center: [...options.center],
      size: [...options.size],
      exhaustiveness: options.exhaustiveness,
      num_modes: options.numModes,
      workers: options.workers,
      seed: options.seed,
      timeout_s: options.timeoutS,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/docking/dock",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return { mode: "sync", ...parseDockRows(raw, "dock(sync)") };
    }
    return { mode: "job", job: parseJobView(raw["job"], "dock(job)") };
  },

  /**
   * `POST /api/v1/artifacts/figure` — the parsed Vina poses' REAL
   * affinity profile as one bar chart (x = the mode rank, y = the
   * affinity the parser actually read out of the log — a bar per
   * pose). The lane needs at least one parsed pose with a finite
   * affinity — the honest minimum, stated before the request fires.
   */
  async poseAffinityFigure(
    poses: readonly VinaPoseRow[],
    format: "PNG" | "SVG" | "TIFF" | "JPEG",
  ): Promise<ArtifactView> {
    if (poses.length < 1) {
      throw new Error("a pose affinity figure needs at least one parsed pose");
    }
    const x: (number | string)[] = [];
    const y: number[] = [];
    const color: string[] = [];
    for (const pose of poses) {
      if (typeof pose.affinity === "number" && Number.isFinite(pose.affinity)) {
        x.push(pose.mode);
        y.push(pose.affinity);
        // the route's own validator: color must be the same length as x
        color.push("affinity");
      }
    }
    if (y.length < 1) {
      throw new Error(
        "a pose affinity figure needs at least one pose with a finite affinity",
      );
    }
    const request: FigureRequest = {
      kind: "bar",
      x,
      y,
      color,
      title: "Molecular Docking — Vina pose affinity per mode",
      format,
      dpi: 150,
      filename: "docking-pose-affinity",
    };
    const raw = (await api.post(
      "/api/v1/artifacts/figure",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the figure lane answered without an artifact");
    }
    return parseArtifactView(artifact, "pose affinity figure");
  },

  /**
   * `POST /api/v1/visualization/table` — the parsed poses / the dock
   * rows as a REAL table artifact (the §17 export stack). The caller
   * carries the backend's own 5000-row cap — mirrored at the page,
   * enforced before the request fires (the error names both numbers,
   * never a silent truncation).
   */
  async exportTable(options: TableExportOptions): Promise<ArtifactView> {
    if (options.rows.length === 0) {
      throw new Error("the export carries no rows");
    }
    const request: VisualizationTableRequest = {
      rows: [...options.rows],
      format: options.format,
      filename: options.filename,
    };
    const raw = (await api.post(
      "/api/v1/visualization/table",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the table lane answered without an artifact");
    }
    return parseArtifactView(artifact, "table export");
  },
};

// -- validation helpers (§8.1) ------------------------------------------

function expectObject(raw: unknown, what: string): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${what}: expected an object`);
  }
  return raw as Record<string, unknown>;
}

function asNumber(raw: unknown): number | null {
  return typeof raw === "number" ? raw : null;
}

function asString(raw: unknown): string {
  return typeof raw === "string" ? raw : "";
}

/** Narrow one `_engine_catalog()` row (the route's own keys). */
function parseEngineCatalogEntry(
  raw: unknown,
  what: string,
): EngineCatalogEntry {
  const r = expectObject(raw, what);
  const name = r["name"];
  if (typeof name !== "string") {
    throw new Error(`${what}: name is not a string`);
  }
  const binary = r["binary"];
  return {
    name,
    bundled: r["bundled"] === true,
    binary: typeof binary === "string" ? binary : null,
    launchable: r["launchable"] === true,
    boundary_note:
      typeof r["boundary_note"] === "string" ? r["boundary_note"] : "",
  };
}

/** Narrow the engines catalog answer (catalog + platform + dockstring). */
export function parseEnginesResponse(
  raw: unknown,
  what: string,
): EnginesResponse {
  const r = expectObject(raw, what);
  const engines = r["engines"];
  const dockstring = expectObject(r["dockstring"], `${what} dockstring`);
  const objectives = dockstring["objectives"];
  return {
    engines: Array.isArray(engines)
      ? engines.map((row, i) => parseEngineCatalogEntry(row, `${what}[${i}]`))
      : [],
    platform: asString(r["platform"]),
    dockstring: {
      n_targets:
        typeof dockstring["n_targets"] === "number"
          ? dockstring["n_targets"]
          : 0,
      objectives: Array.isArray(objectives)
        ? objectives.map((value) => String(value))
        : [],
    },
  };
}

/** Narrow one `target_frame()` row (the catalog's own columns). */
function parseDockstringTargetRow(
  raw: unknown,
  what: string,
): DockstringTargetRow {
  const r = expectObject(raw, what);
  const target = r["target"];
  if (typeof target !== "string") {
    throw new Error(`${what}: target is not a string`);
  }
  return {
    target,
    center_x: asNumber(r["center_x"]) ?? 0,
    center_y: asNumber(r["center_y"]) ?? 0,
    center_z: asNumber(r["center_z"]) ?? 0,
    size_x: asNumber(r["size_x"]) ?? 0,
    size_y: asNumber(r["size_y"]) ?? 0,
    size_z: asNumber(r["size_z"]) ?? 0,
  };
}

/** Narrow the single-target view (`name` + box + config text). */
export function parseDockstringTargetView(
  raw: unknown,
  what: string,
): DockstringTargetView {
  const r = expectObject(raw, what);
  const name = r["name"];
  if (typeof name !== "string") {
    throw new Error(`${what}: name is not a string`);
  }
  const center = r["center"];
  const size = r["size"];
  return {
    name,
    center: Array.isArray(center) ? center.map((v) => asNumber(v) ?? 0) : [],
    size: Array.isArray(size) ? size.map((v) => asNumber(v) ?? 0) : [],
    config: asString(r["config"]),
  };
}

/** Narrow the benchmark objective's own answer. */
export function parseObjectiveResponse(
  raw: unknown,
  what: string,
): ObjectiveResponse {
  const r = expectObject(raw, what);
  const objective = r["objective"];
  const value = r["value"];
  if (typeof objective !== "string" || typeof value !== "number") {
    throw new Error(`${what}: objective/value are not the route's shape`);
  }
  const scores = r["scores"];
  return {
    objective,
    scores:
      typeof scores === "object" && scores !== null
        ? (scores as Record<string, number>)
        : {},
    qed: asNumber(r["qed"]) ?? 0,
    value,
  };
}

/** Narrow the preparation audit dict (`ProteinReport.as_dict()`). */
export function parseReceptorReport(
  raw: unknown,
  what: string,
): ReceptorReportView {
  const r = expectObject(raw, what);
  const chains = r["chains"];
  const atomTypes = r["atom_types"];
  const notes = r["notes"];
  return {
    source: asString(r["source"]),
    output_pdbqt: asString(r["output_pdbqt"]),
    mode: asString(r["mode"]),
    input_atoms: asNumber(r["input_atoms"]) ?? 0,
    input_hetatms: asNumber(r["input_hetatms"]) ?? 0,
    waters_removed: asNumber(r["waters_removed"]) ?? 0,
    hetatm_removed: asNumber(r["hetatm_removed"]) ?? 0,
    altloc_removed: asNumber(r["altloc_removed"]) ?? 0,
    hydrogens_removed: asNumber(r["hydrogens_removed"]) ?? 0,
    bad_residues_removed: asNumber(r["bad_residues_removed"]) ?? 0,
    chains: Array.isArray(chains) ? chains.map((v) => String(v)) : [],
    residues: asNumber(r["residues"]) ?? 0,
    output_atoms: asNumber(r["output_atoms"]) ?? 0,
    atom_types:
      typeof atomTypes === "object" && atomTypes !== null
        ? (atomTypes as Record<string, number>)
        : {},
    charge_min: asNumber(r["charge_min"]) ?? 0,
    charge_max: asNumber(r["charge_max"]) ?? 0,
    charge_mean: asNumber(r["charge_mean"]) ?? 0,
    notes: Array.isArray(notes) ? notes.map((v) => String(v)) : [],
  };
}

/** Narrow the vina parse answer (one row per parsed pose). */
export function parseVinaParseResponse(
  raw: unknown,
  what: string,
): VinaParseResponse {
  const r = expectObject(raw, what);
  const poses = r["poses"];
  if (!Array.isArray(poses)) {
    throw new Error(`${what}: poses is not an array`);
  }
  return {
    n_poses: typeof r["n_poses"] === "number" ? r["n_poses"] : poses.length,
    poses: poses.map((row, i) => parseVinaPoseRow(row, `${what} poses[${i}]`)),
  };
}

/** Narrow one parsed pose (mode/affinity/rmsd_ub/rmsd_lb). */
function parseVinaPoseRow(raw: unknown, what: string): VinaPoseRow {
  const r = expectObject(raw, what);
  const mode = r["mode"];
  if (typeof mode !== "number") {
    throw new Error(`${what}: mode is not a number`);
  }
  return {
    mode,
    affinity: asNumber(r["affinity"]),
    rmsd_ub: asNumber(r["rmsd_ub"]),
    rmsd_lb: asNumber(r["rmsd_lb"]),
  };
}

/** Narrow the DLG parse answer (summary + the cluster table). */
export function parseDlgParseResponse(
  raw: unknown,
  what: string,
): DlgParseResponse {
  const r = expectObject(raw, what);
  const summary = expectObject(r["summary"], `${what} summary`);
  const clusters = r["clusters"];
  return {
    summary: {
      n_clusters: asNumber(summary["n_clusters"]) ?? 0,
      top_rank: asNumber(summary["top_rank"]) ?? 0,
      top_lowest_energy: asNumber(summary["top_lowest_energy"]) ?? 0,
      top_mean_energy: asNumber(summary["top_mean_energy"]) ?? 0,
      top_num_in_cluster: asNumber(summary["top_num_in_cluster"]) ?? 0,
      best_rank: asNumber(summary["best_rank"]) ?? 0,
      best_lowest_energy: asNumber(summary["best_lowest_energy"]) ?? 0,
      best_mean_energy: asNumber(summary["best_mean_energy"]) ?? 0,
      best_num_in_cluster: asNumber(summary["best_num_in_cluster"]) ?? 0,
    },
    clusters: Array.isArray(clusters)
      ? clusters.map((row, i) =>
          parseDlgClusterRow(row, `${what} clusters[${i}]`),
        )
      : [],
  };
}

/** Narrow one cluster row (rank/lowest/mean/count). */
function parseDlgClusterRow(raw: unknown, what: string): DlgClusterRow {
  const r = expectObject(raw, what);
  const rank = r["rank"];
  if (typeof rank !== "number") {
    throw new Error(`${what}: rank is not a number`);
  }
  return {
    rank,
    lowest_energy: asNumber(r["lowest_energy"]) ?? 0,
    mean_energy: asNumber(r["mean_energy"]) ?? 0,
    num_in_cluster: asNumber(r["num_in_cluster"]) ?? 0,
  };
}

/** Narrow the ligand-efficiency answer. */
export function parseLigandEfficiencyResponse(
  raw: unknown,
  what: string,
): LigandEfficiencyResponse {
  const r = expectObject(raw, what);
  const ligandEfficiency = r["ligand_efficiency"];
  if (typeof ligandEfficiency !== "number") {
    throw new Error(`${what}: ligand_efficiency is not a number`);
  }
  return {
    affinity_kcal_mol: asNumber(r["affinity_kcal_mol"]) ?? 0,
    num_heavy_atoms: asNumber(r["num_heavy_atoms"]) ?? 0,
    ligand_efficiency: ligandEfficiency,
  };
}

/** Narrow the contacts answer (contacts + classes + PLIF rows). */
export function parseContactsResponse(
  raw: unknown,
  what: string,
): ContactsResponse {
  const r = expectObject(raw, what);
  const interactionClasses = r["interaction_classes"];
  const contacts = r["contacts"];
  const plif = r["plif"];
  return {
    cutoff: asNumber(r["cutoff"]) ?? 0,
    n_contacts: asNumber(r["n_contacts"]) ?? 0,
    interaction_classes: Array.isArray(interactionClasses)
      ? interactionClasses.map((v) => String(v))
      : [],
    contacts: Array.isArray(contacts)
      ? contacts.map((row, i) => parseContactRow(row, `${what} contacts[${i}]`))
      : [],
    plif: Array.isArray(plif)
      ? plif.map((row, i) => parsePlifRow(row, `${what} plif[${i}]`))
      : [],
  };
}

/** Narrow one contact row (the route's own emitted keys). */
function parseContactRow(raw: unknown, what: string): ContactRow {
  const r = expectObject(raw, what);
  const ligandSerial = r["ligand_serial"];
  if (typeof ligandSerial !== "number") {
    throw new Error(`${what}: ligand_serial is not a number`);
  }
  return {
    ligand_serial: ligandSerial,
    ligand_atom: asString(r["ligand_atom"]),
    receptor_residue: asString(r["receptor_residue"]),
    receptor_atom: asString(r["receptor_atom"]),
    distance: asNumber(r["distance"]) ?? 0,
    interaction: asString(r["interaction"]),
  };
}

/** Narrow one PLIF row (the residue + per-class counts + total). */
function parsePlifRow(raw: unknown, what: string): PlifRow {
  const r = expectObject(raw, what);
  const residue = r["residue"];
  if (typeof residue !== "string") {
    throw new Error(`${what}: residue is not a string`);
  }
  const counts: Record<string, number | string> = {};
  for (const [key, value] of Object.entries(r)) {
    counts[key] = typeof value === "number" ? value : String(value);
  }
  return {
    residue,
    total: asNumber(r["total"]) ?? 0,
    ...counts,
  } as PlifRow;
}

/** Narrow the pose-rmsd answer (one RMSD per pose, verbatim). */
export function parsePoseRmsdResponse(
  raw: unknown,
  what: string,
): PoseRmsdResponse {
  const r = expectObject(raw, what);
  const smiles = r["smiles"];
  const rmsd = r["rmsd"];
  if (typeof smiles !== "string" || !Array.isArray(rmsd)) {
    throw new Error(`${what}: smiles/rmsd are not the route's shape`);
  }
  return {
    smiles,
    n_atoms: asNumber(r["n_atoms"]) ?? 0,
    n_poses: asNumber(r["n_poses"]) ?? 0,
    rmsd: rmsd.map((v) => asNumber(v) ?? 0),
  };
}

/**
 * Narrow the dock rows body — the sync answer's own payload AND the
 * COMPLETED job's authoritative REST result (`{n_ligands, n_ok,
 * n_failed, rows}`; every row is the backend's own row dict,
 * verbatim — never re-typed, never faked).
 */
export function parseDockRows(raw: unknown, what: string): DockRowsView {
  const r = expectObject(raw, what);
  const rows = r["rows"];
  if (!Array.isArray(rows)) {
    throw new Error(`${what}: rows is not an array`);
  }
  return {
    n_ligands: asNumber(r["n_ligands"]) ?? rows.length,
    n_ok: asNumber(r["n_ok"]) ?? 0,
    n_failed: asNumber(r["n_failed"]) ?? 0,
    rows: rows.map((row, i) => expectObject(row, `${what} rows[${i}]`)),
  };
}

/**
 * Narrow one `job.progress` envelope payload — the dock lane's REAL
 * per-ligand telemetry (progress %, stage). Progress-only: the rows
 * come from the authoritative REST result relay, never the envelope.
 */
export function parseJobProgressPayload(
  raw: unknown,
  what: string,
): JobProgressPayload {
  const r = expectObject(raw, what);
  const progress = r["progress"];
  if (typeof progress !== "number") {
    throw new Error(`${what}: progress is not a number`);
  }
  return {
    progress,
    stage: typeof r["stage"] === "string" ? r["stage"] : "",
  };
}

// -- the ContactAtomSpec builder ---------------------------------------

function parseIntField(raw: string, label: string, minimum: number): number {
  const value = Number(raw.trim());
  if (!Number.isFinite(value) || !Number.isInteger(value) || value < minimum) {
    throw new Error(
      `${label}: ${JSON.stringify(raw)} is not an integer ≥ ${minimum}`,
    );
  }
  return value;
}

function parseCoordinate(raw: string, label: string): number {
  const value = Number(raw.trim());
  if (!Number.isFinite(value)) {
    throw new Error(`${label}: ${JSON.stringify(raw)} is not a finite number`);
  }
  return value;
}

/**
 * Narrow one pasted atom row to the route's `ContactAtomSpec` (the
 * generated contract type — every field, verbatim; a malformed cell
 * is this parser's own labeled error, never a silent default).
 */
export function parseContactAtom(
  fields: ContactAtomFields,
  label: string,
): ContactAtomSpec {
  const spec: ContactAtomSpec = {
    serial: parseIntField(fields.serial, `${label} serial`, 1),
    name: fields.name.trim(),
    element: fields.element.trim(),
    resname: fields.resname.trim(),
    chain: fields.chain.trim(),
    resseq: parseIntField(fields.resseq, `${label} resseq`, 0),
    x: parseCoordinate(fields.x, `${label} x`),
    y: parseCoordinate(fields.y, `${label} y`),
    z: parseCoordinate(fields.z, `${label} z`),
    autodock_type:
      fields.autodock_type.trim() === "" ? null : fields.autodock_type.trim(),
  };
  return spec;
}
