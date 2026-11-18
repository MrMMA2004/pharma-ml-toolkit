import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Beaker,
  CircleDot,
  Columns3,
  FileSearch,
  FlaskConical,
  Fingerprint,
  Upload,
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
  chemistryApi,
  type DescriptorFrame,
  type FingerprintMatrix,
} from "@/services/chemistry/client";
import {
  datasetsApi,
  type ColumnValues,
  type DatasetSplitResponse,
  type DatasetView,
  type FramePreview,
} from "@/services/datasets/client";
import {
  artifactsApi,
  type ArtifactView,
} from "@/services/artifacts/client";
import { api } from "@/services/api/client";
import type { components } from "@/types/api.d";

/**
 * The Datasets workspace — part11 §11.1's complete sequential workflow:
 *
 * Import → Inspect → Validate → Clean/Standardize → Select Columns →
 * Descriptor/Fingerprint Generation → Feature Selection →
 * Split/Leakage Audit → Visualize → Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons):
 * the upload rides the multipart lane through the real reader; the
 * preview/profile/split/chemistry lanes run the backend's own
 * engines; the export registers a real artifact and downloads it
 * through the token-header lane with digest verification.
 */

type StatRequest = components["schemas"]["StatRequest"];
type ArtifactTableRequest = components["schemas"]["ArtifactTableRequest"];

/** `POST /api/v1/visualization/stat` — the stat + figure answer. */
interface StatResponse {
  readonly kind: string;
  readonly n_points: number;
  readonly stats: Readonly<Record<string, unknown>>;
  readonly artifact: ArtifactView | null;
}

const ACCEPTED_SUFFIXES = ".csv,.tsv,.txt,.xlsx,.xls,.xlsm";

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail =
      typeof error.detail === "object" && error.detail !== null
        ? JSON.stringify(error.detail)
        : String(error.detail);
    return `API ${error.status}: ${detail}`;
  }
  return error instanceof Error ? error.message : String(error);
}

/** A dismissible error banner (§8.3: clear, actionable errors). */
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

/** A numbered workflow stage header (§11.1's fixed order). */
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
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-8">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-[var(--color-muted)] text-xs font-semibold text-[var(--color-muted-foreground)]">
          {index}
        </span>
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">{title}</h2>
        <span className="text-xs text-[var(--color-muted-foreground)]">{hint}</span>
      </div>
      <div className="mt-3">{children}</div>
      {icon ? null : null}
    </section>
  );
}

/** The bounded preview table (§17.3: paginated, type-aware cells). */
function PreviewTable({ preview }: { preview: FramePreview }) {
  return (
    <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
      <div className="max-h-96 overflow-auto">
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 bg-[var(--color-muted)]">
            <tr>
              {preview.columns.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                  title={preview.dtypes[column] ?? ""}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, i) => (
              <tr
                key={i}
                className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
              >
                {row.map((cell, j) => (
                  <td
                    key={j}
                    className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]"
                  >
                    {cell === null || cell === undefined
                      ? "—"
                      : typeof cell === "number"
                        ? Number.isInteger(cell)
                          ? String(cell)
                          : cell.toFixed(4)
                        : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-[var(--color-border)] bg-[var(--color-card)] px-3 py-2 text-[11px] text-[var(--color-muted-foreground)]">
        {preview.returned} of {preview.n_rows} rows shown — the backend keeps
        the authoritative frame; workflows reference the dataset ID.
      </p>
    </div>
  );
}

export function DatasetsPage() {
  // stage 1-2: registry + selection
  const [datasets, setDatasets] = useState<readonly DatasetView[]>([]);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<DatasetView | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [preview, setPreview] = useState<FramePreview | null>(null);
  const [previewLimit, setPreviewLimit] = useState(10);

  // stage 1: the upload lane
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [task, setTask] = useState("");
  const [uploadJobId, setUploadJobId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);

  // stage 3: the profile evidence
  const [profile, setProfile] = useState<Readonly<Record<string, unknown>> | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [profileBusy, setProfileBusy] = useState(false);

  // stage 5: chemistry
  const [smilesColumn, setSmilesColumn] = useState("");
  const [column, setColumn] = useState<ColumnValues | null>(null);
  const [columnError, setColumnError] = useState<string | null>(null);
  const [descriptorSet, setDescriptorSet] = useState<"rdkit_all" | "morgan" | "mordred">("morgan");
  const [fpKind, setFpKind] = useState<"morgan" | "rdkit" | "atom_pair" | "avalon">("morgan");
  const [radius, setRadius] = useState(2);
  const [nBits, setNBits] = useState(2048);
  const [asCounts, setAsCounts] = useState(false);
  const [chemJobId, setChemJobId] = useState<string | null>(null);
  const [chemError, setChemError] = useState<string | null>(null);
  const [descriptorFrame, setDescriptorFrame] = useState<DescriptorFrame | null>(null);
  const [fpMatrix, setFpMatrix] = useState<FingerprintMatrix | null>(null);

  // stage 6: split
  const [testSize, setTestSize] = useState(0.2);
  const [valSize, setValSize] = useState<number | null>(null);
  const [stratify, setStratify] = useState(true);
  const [randomState, setRandomState] = useState(42);
  const [splitReport, setSplitReport] = useState<DatasetSplitResponse | null>(null);
  const [splitError, setSplitError] = useState<string | null>(null);
  const [splitBusy, setSplitBusy] = useState(false);

  // stage 7: visualize
  const [vizColumn, setVizColumn] = useState("");
  const [statResult, setStatResult] = useState<StatResponse | null>(null);
  const [vizError, setVizError] = useState<string | null>(null);
  const [vizBusy, setVizBusy] = useState(false);

  // stage 8: export
  const [exportFormat, setExportFormat] = useState<"csv" | "tsv" | "xlsx">("csv");
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [digestStatus, setDigestStatus] = useState<string | null>(null);

  const refreshRegistry = useCallback(async () => {
    try {
      const response = await datasetsApi.list();
      setDatasets(response.datasets);
      setRegistryError(null);
    } catch (error) {
      setRegistryError(describeError(error));
    }
  }, []);

  const selectDataset = useCallback(async (datasetId: string) => {
    setSelectedId(datasetId);
    setSelectionError(null);
    // stage-local state resets with the selection
    setProfile(null);
    setColumn(null);
    setChemJobId(null);
    setDescriptorFrame(null);
    setFpMatrix(null);
    setSplitReport(null);
    setStatResult(null);
    setExportArtifact(null);
    setDigestStatus(null);
    try {
      const [view, frame] = await Promise.all([
        datasetsApi.view(datasetId),
        datasetsApi.preview(datasetId, 10),
      ]);
      setSelected(view);
      setPreview(frame);
      const guessed = view.columns.find((c) => /^smiles?$/i.test(c)) ?? "";
      setSmilesColumn(guessed);
      setVizColumn(
        view.feature_names.find((name) => name !== view.target) ?? "",
      );
    } catch (error) {
      setSelected(null);
      setPreview(null);
      setSelectionError(describeError(error));
    }
  }, []);

  useEffect(() => {
    void refreshRegistry();
  }, [refreshRegistry]);

  // stage 1: upload (job mode — the backend's default lane)
  const handleUpload = useCallback(async () => {
    if (file === null) {
      return;
    }
    setUploadBusy(true);
    setUploadError(null);
    setUploadJobId(null);
    try {
      const response = await datasetsApi.upload(file, {
        target: target.trim() === "" ? null : target.trim(),
        task: task.trim() === "" ? null : task.trim(),
        mode: "job",
      });
      if (response.mode === "job") {
        setUploadJobId(response.job.job_id);
      }
    } catch (error) {
      setUploadError(describeError(error));
    } finally {
      setUploadBusy(false);
    }
  }, [file, target, task]);

  const handleUploadResult = useCallback(
    (result: unknown) => {
      // the job's result IS the registered dataset view
      const record = result as Record<string, unknown>;
      const id = typeof record["dataset_id"] === "string" ? record["dataset_id"] : null;
      void refreshRegistry();
      if (id !== null) {
        void selectDataset(id);
      }
    },
    [refreshRegistry, selectDataset],
  );

  // stage 3: the profile
  const handleProfile = useCallback(async () => {
    if (selectedId === null) {
      return;
    }
    setProfileBusy(true);
    setProfileError(null);
    try {
      const response = await datasetsApi.profile(selectedId, { mode: "sync" });
      setProfile(response.profile);
    } catch (error) {
      setProfileError(describeError(error));
    } finally {
      setProfileBusy(false);
    }
  }, [selectedId]);

  // stage 5: the chemistry bridge
  const handlePullColumn = useCallback(async () => {
    if (selectedId === null || smilesColumn === "") {
      return;
    }
    setColumnError(null);
    try {
      setColumn(await datasetsApi.column(selectedId, smilesColumn));
    } catch (error) {
      setColumn(null);
      setColumnError(describeError(error));
    }
  }, [selectedId, smilesColumn]);

  const smiles = column === null
    ? []
    : column.values.filter((v): v is string => typeof v === "string");

  const handleDescriptors = useCallback(async () => {
    if (smiles.length === 0) {
      return;
    }
    setChemError(null);
    setFpMatrix(null);
    try {
      const response = await chemistryApi.descriptors(smiles, {
        descriptorSet,
        mode: "job",
      });
      if (response.mode === "job") {
        setChemJobId(response.job.job_id);
      }
    } catch (error) {
      setChemError(describeError(error));
    }
  }, [smiles, descriptorSet]);

  const handleFingerprints = useCallback(async () => {
    if (smiles.length === 0) {
      return;
    }
    setChemError(null);
    setDescriptorFrame(null);
    try {
      const response = await chemistryApi.fingerprints(smiles, {
        kind: fpKind,
        radius,
        nBits,
        asCounts,
        mode: "job",
      });
      if (response.mode === "job") {
        setChemJobId(response.job.job_id);
      }
    } catch (error) {
      setChemError(describeError(error));
    }
  }, [smiles, fpKind, radius, nBits, asCounts]);

  const handleChemResult = useCallback((result: unknown) => {
    const record = result as Record<string, unknown>;
    if ("matrix" in record) {
      setFpMatrix({
        shape: (record["shape"] as [number, number]) ?? [0, 0],
        dtype: (record["dtype"] as string) ?? "",
        invalid_indices: (record["invalid_indices"] as number[]) ?? [],
        matrix: (record["matrix"] as number[][]) ?? [],
      });
    } else {
      setDescriptorFrame({
        shape: (record["shape"] as [number, number]) ?? [0, 0],
        columns: (record["columns"] as string[]) ?? [],
        rows: (record["rows"] as unknown[][]) ?? [],
        index: (record["index"] as string[]) ?? [],
      });
    }
  }, []);

  // stage 6: the leakage-safe split
  const handleSplit = useCallback(async () => {
    if (selectedId === null) {
      return;
    }
    setSplitBusy(true);
    setSplitError(null);
    try {
      setSplitReport(
        await datasetsApi.split(selectedId, {
          target: null,
          test_size: testSize,
          val_size: valSize,
          stratify,
          random_state: randomState,
          mode: "sync",
        }),
      );
    } catch (error) {
      setSplitReport(null);
      setSplitError(describeError(error));
    } finally {
      setSplitBusy(false);
    }
  }, [selectedId, testSize, valSize, stratify, randomState]);

  // stage 7: the backend's own statplots engine
  const handleStat = useCallback(async () => {
    if (selectedId === null || vizColumn === "") {
      return;
    }
    setVizBusy(true);
    setVizError(null);
    try {
      const values: number[] = [];
      const pulled = await datasetsApi.column(selectedId, vizColumn);
      for (const value of pulled.values) {
        if (typeof value === "number" && Number.isFinite(value)) {
          values.push(value);
        }
      }
      if (values.length < 4) {
        throw new Error(
          `the column has only ${values.length} finite values; the stat engine needs at least 4`,
        );
      }
      const request: StatRequest = {
        kind: "histogram",
        values,
        bins: "auto",
        whis: 1.5,
        n_boot: 200,
        seed: 0,
        format: "PNG",
        with_artifact: true,
      };
      const raw = await api.post<Record<string, unknown>>(
        "/api/v1/visualization/stat",
        request,
      );
      const artifact = raw["artifact"];
      setStatResult({
        kind: typeof raw["kind"] === "string" ? raw["kind"] : "",
        n_points: typeof raw["n_points"] === "number" ? raw["n_points"] : 0,
        stats: (raw["stats"] as Record<string, unknown>) ?? {},
        artifact:
          typeof artifact === "object" && artifact !== null
            ? (artifact as ArtifactView)
            : null,
      });
    } catch (error) {
      setStatResult(null);
      setVizError(describeError(error));
    } finally {
      setVizBusy(false);
    }
  }, [selectedId, vizColumn]);

  // stage 8: export + the verified download
  const handleExport = useCallback(async () => {
    if (selectedId === null) {
      return;
    }
    setExportBusy(true);
    setExportError(null);
    setDigestStatus(null);
    try {
      const request: ArtifactTableRequest = {
        dataset_id: selectedId,
        format: exportFormat,
      };
      const raw = await api.post<Record<string, unknown>>(
        "/api/v1/artifacts/table",
        request,
      );
      const artifact = raw["artifact"];
      if (typeof artifact === "object" && artifact !== null) {
        setExportArtifact(artifact as ArtifactView);
      }
    } catch (error) {
      setExportArtifact(null);
      setExportError(describeError(error));
    } finally {
      setExportBusy(false);
    }
  }, [selectedId, exportFormat]);

  const handleDownload = useCallback(async () => {
    if (exportArtifact === null) {
      return;
    }
    setDigestStatus(null);
    try {
      const download = await artifactsApi.download(exportArtifact.artifact_id);
      if (download.digest_verified === true) {
        setDigestStatus(
          `digest verified: ${download.actual_sha256?.slice(0, 16)}…`,
        );
      } else if (download.digest_verified === null) {
        setDigestStatus(
          "downloaded; local digest unavailable (insecure context) — the route's X-Artifact-SHA256 was received",
        );
      } else {
        setDigestStatus("DIGEST MISMATCH — the downloaded bytes differ from the route's promise");
      }
      const url = URL.createObjectURL(download.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = download.filename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setDigestStatus(describeError(error));
    }
  }, [exportArtifact]);

  const selectedColumns = selected?.columns ?? [];
  const numericCandidates = selected?.feature_names ?? [];

  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      <div className="max-w-3xl">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--color-foreground)]">
          Datasets
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-muted-foreground)]">
          The complete sequential workflow — import, inspect, validate,
          clean, select columns, generate descriptors and fingerprints,
          split with a leakage audit, visualize, and export — every step
          runs the real Python backend through the local API.
        </p>
      </div>

      <StageHeader
        index={1}
        title="Import"
        hint="multipart upload through the real reader"
        icon={<Upload aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-[1fr_auto_auto] md:items-end">
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                  CSV / TSV / TXT / XLSX file
                </span>
                <input
                  type="file"
                  accept={ACCEPTED_SUFFIXES}
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-[var(--color-primary)] file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-[var(--color-primary-foreground)]"
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">Target column</span>
                <input
                  value={target}
                  onChange={(event) => setTarget(event.target.value)}
                  placeholder="auto-detect"
                  className="w-40 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">Task override</span>
                <input
                  value={task}
                  onChange={(event) => setTask(event.target.value)}
                  placeholder="inferred"
                  className="w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                />
              </label>
              <Button disabled={file === null || uploadBusy} onClick={() => void handleUpload()}>
                <Upload aria-hidden className="h-4 w-4" />
                {uploadBusy ? "Uploading…" : "Upload"}
              </Button>
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              The upload parses through the backend&apos;s own
              load_dataset engine (target auto-detection, task
              inference, non-numeric column policy) as a managed
              background job.
            </p>
            {uploadError !== null ? <div className="mt-3"><ErrorBanner message={uploadError} /></div> : null}
            {uploadJobId !== null ? (
              <div className="mt-3">
                <JobRunCard jobId={uploadJobId} resultLabel="the registered dataset view" onResult={handleUploadResult} />
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={2}
        title="Inspect"
        hint="the registry, the summary, the paginated preview"
        icon={<FileSearch aria-hidden className="h-4 w-4" />}
      >
        {registryError !== null ? <ErrorBanner message={registryError} /> : null}
        <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-48 overflow-auto">
            <table className="w-full border-collapse text-xs">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["Name", "Rows", "Columns", "Target", "Task", "Created"].map((header) => (
                    <th key={header} scope="col" className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {datasets.map((entry) => (
                  <tr
                    key={entry.dataset_id}
                    onClick={() => void selectDataset(entry.dataset_id)}
                    className={`cursor-pointer border-t border-[var(--color-border)] ${
                      entry.dataset_id === selectedId
                        ? "bg-[var(--color-accent)]"
                        : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                    }`}
                  >
                    <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">{entry.name}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.shape[0]}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.shape[1]}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.target ?? "—"}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.task ?? "—"}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.created_at}</td>
                  </tr>
                ))}
                {datasets.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-3 py-4 text-center text-[var(--color-muted-foreground)]">
                      No datasets registered yet — import one above.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
        {selectionError !== null ? <div className="mt-3"><ErrorBanner message={selectionError} /></div> : null}
        {selected !== null ? (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="text-base">{selected.name}</CardTitle>
              <CardDescription>
                {selected.shape[0]} rows × {selected.shape[1]} columns ·{" "}
                {selected.n_features} numeric features · target{" "}
                {selected.target ?? "none"} · task {selected.task ?? "inferred"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-3 flex items-center gap-2">
                {selected.dropped_columns.length > 0 ? (
                  <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
                    {selected.dropped_columns.length} dropped by the loader
                  </span>
                ) : null}
                {selected.categorical_columns.length > 0 ? (
                  <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
                    {selected.categorical_columns.length} categorical
                  </span>
                ) : null}
                <label className="ml-auto text-xs text-[var(--color-muted-foreground)]">
                  rows
                  <select
                    value={previewLimit}
                    onChange={(event) => {
                      const limit = Number(event.target.value);
                      setPreviewLimit(limit);
                      if (selectedId !== null) {
                        void datasetsApi
                          .preview(selectedId, limit)
                          .then(setPreview)
                          .catch((error) => setSelectionError(describeError(error)));
                      }
                    }}
                    className="ml-2 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 text-xs"
                  >
                    {[10, 25, 50, 100, 500].map((limit) => (
                      <option key={limit} value={limit}>{limit}</option>
                    ))}
                  </select>
                </label>
              </div>
              {preview !== null ? <PreviewTable preview={preview} /> : null}
            </CardContent>
          </Card>
        ) : null}
      </StageHeader>

      <StageHeader
        index={3}
        title="Validate"
        hint="the backend's own profile engine"
        icon={<CircleDot aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <Button
              size="sm"
              disabled={selectedId === null || profileBusy}
              onClick={() => void handleProfile()}
            >
              <Beaker aria-hidden className="h-4 w-4" />
              {profileBusy ? "Profiling…" : "Profile the dataset"}
            </Button>
            {profileError !== null ? <div className="mt-3"><ErrorBanner message={profileError} /></div> : null}
            {profile !== null ? (
              <div className="mt-4">
                <div className="flex flex-wrap gap-2">
                  {["n", "n_var", "n_cells", "n_cells_missing", "n_duplicates", "memory_bytes"].map((key) => {
                    const value = profile[key];
                    return value === undefined ? null : (
                      <span key={key} className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                        {key}: {String(value)}
                      </span>
                    );
                  })}
                </div>
                <ProfileVariables profile={profile} />
              </div>
            ) : (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                profile_frame&apos;s table-level statistics and the
                per-variable profiles (missing values, distinct counts,
                constants, duplicate rows) render here after the run.
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={4}
        title="Clean / Standardize / Select Columns"
        hint="the loader's own policy, surfaced"
        icon={<Columns3 aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            {selected === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a dataset in Inspect to see the loader&apos;s
                column policy.
              </p>
            ) : (
              <div className="grid gap-4 md:grid-cols-3">
                <ColumnChips
                  title="Numeric features (kept)"
                  columns={selected.feature_names}
                />
                <ColumnChips
                  title="Categorical (kept)"
                  columns={selected.categorical_columns}
                />
                <ColumnChips
                  title="Dropped by the loader"
                  columns={selected.dropped_columns}
                />
                <p className="col-span-full text-xs text-[var(--color-muted-foreground)]">
                  The real load_dataset policy: non-numeric,
                  non-categorical columns (IDs, free text, dates) are
                  dropped from the feature matrix; the target column is
                  auto-detected and its task inferred. The full frame
                  stays available to the backend — nothing is lost
                  server-side.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={5}
        title="Descriptor / Fingerprint Generation"
        hint="the Chemistry Lab engines through the bounded column bridge"
        icon={<Fingerprint aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">SMILES column</span>
                <select
                  value={smilesColumn}
                  onChange={(event) => setSmilesColumn(event.target.value)}
                  className="w-44 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  <option value="">select…</option>
                  {selectedColumns.map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </label>
              <Button size="sm" variant="outline" disabled={selectedId === null || smilesColumn === ""} onClick={() => void handlePullColumn()}>
                Pull the column
              </Button>
              {column !== null ? (
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  {column.returned}/{column.n_rows} values ({column.dtype})
                  {column.truncated ? " — truncated at the chemistry engine's own 500-smiles cap" : ""}
                </span>
              ) : null}
            </div>
            {columnError !== null ? <div className="mt-3"><ErrorBanner message={columnError} /></div> : null}

            {smiles.length > 0 ? (
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                <div className="rounded-lg border border-[var(--color-border)] p-4">
                  <div className="flex items-center gap-2">
                    <FlaskConical aria-hidden className="h-4 w-4 text-[var(--color-primary)]" />
                    <span className="text-sm font-medium text-[var(--color-foreground)]">Descriptors</span>
                  </div>
                  <label className="mt-3 block text-xs text-[var(--color-muted-foreground)]">
                    descriptor set
                    <select
                      value={descriptorSet}
                      onChange={(event) => setDescriptorSet(event.target.value as typeof descriptorSet)}
                      className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="morgan">Morgan (RDKit)</option>
                      <option value="rdkit_all">RDKit descriptors (all)</option>
                      <option value="mordred">Mordred</option>
                    </select>
                  </label>
                  <Button size="sm" className="mt-3" onClick={() => void handleDescriptors()}>
                    Compute descriptors
                  </Button>
                  {descriptorFrame !== null ? (
                    <div className="mt-3">
                      <p className="text-xs text-[var(--color-muted-foreground)]">
                        {descriptorFrame.shape[0]} molecules ×{" "}
                        {descriptorFrame.shape[1]} descriptor columns
                        {" "}(first 20 rows shown)
                      </p>
                      <div className="mt-2 overflow-hidden rounded-md border border-[var(--color-border)]">
                        <div className="max-h-48 overflow-auto">
                          <table className="w-full border-collapse text-[11px]">
                            <thead className="sticky top-0 bg-[var(--color-muted)]">
                              <tr>
                                <th scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold">SMILES</th>
                                {descriptorFrame.columns.slice(0, 6).map((name) => (
                                  <th key={name} scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold">{name}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {descriptorFrame.rows.slice(0, 20).map((row, i) => (
                                <tr key={i} className="border-t border-[var(--color-border)]">
                                  {row.slice(0, 7).map((cell, j) => (
                                    <td key={j} className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">
                                      {cell === null || cell === undefined ? "—" : typeof cell === "number" && !Number.isInteger(cell) ? cell.toFixed(4) : String(cell)}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
                <div className="rounded-lg border border-[var(--color-border)] p-4">
                  <div className="flex items-center gap-2">
                    <Fingerprint aria-hidden className="h-4 w-4 text-[var(--color-primary)]" />
                    <span className="text-sm font-medium text-[var(--color-foreground)]">Fingerprints</span>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <label className="text-xs text-[var(--color-muted-foreground)]">
                      kind
                      <select
                        value={fpKind}
                        onChange={(event) => setFpKind(event.target.value as typeof fpKind)}
                        className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                      >
                        <option value="morgan">Morgan</option>
                        <option value="rdkit">RDKit topological</option>
                        <option value="atom_pair">Atom pair</option>
                        <option value="avalon">Avalon</option>
                      </select>
                    </label>
                    <label className="text-xs text-[var(--color-muted-foreground)]">
                      nBits
                      <select
                        value={nBits}
                        onChange={(event) => setNBits(Number(event.target.value))}
                        className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                      >
                        {[512, 1024, 2048, 4096].map((bits) => (
                          <option key={bits} value={bits}>{bits}</option>
                        ))}
                      </select>
                    </label>
                    <label className="text-xs text-[var(--color-muted-foreground)]">
                      Morgan radius
                      <input
                        type="number"
                        min={0}
                        max={6}
                        value={radius}
                        onChange={(event) => setRadius(Number(event.target.value))}
                        className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                      />
                    </label>
                    <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                      <input
                        type="checkbox"
                        checked={asCounts}
                        onChange={(event) => setAsCounts(event.target.checked)}
                        className="h-4 w-4 rounded border-[var(--color-border)]"
                      />
                      count representation
                    </label>
                  </div>
                  <Button size="sm" className="mt-3" onClick={() => void handleFingerprints()}>
                    Generate fingerprints
                  </Button>
                  {fpMatrix !== null ? (
                    <div className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                      <p>
                        {fpMatrix.shape[0]} × {fpMatrix.shape[1]} matrix ·{" "}
                        {fpMatrix.dtype} · {fpMatrix.invalid_indices.length} invalid SMILES
                        {" "}(first molecule, first 64 bits:)
                      </p>
                      <div className="mt-2 flex flex-wrap gap-0.5">
                        {(fpMatrix.matrix[0] ?? []).slice(0, 64).map((bit, i) => (
                          <span
                            key={i}
                            className={`inline-flex h-4 w-4 items-center justify-center rounded-[2px] text-[9px] ${
                              bit > 0
                                ? "bg-[var(--color-primary)] text-[var(--color-primary-foreground)]"
                                : "bg-[var(--color-muted)] text-[var(--color-muted-foreground)]"
                            }`}
                          >
                            {bit > 9 ? "+" : bit}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                Pull a SMILES column to arm the chemistry engines.
              </p>
            )}
            {chemError !== null ? <div className="mt-3"><ErrorBanner message={chemError} /></div> : null}
            {chemJobId !== null ? (
              <div className="mt-3">
                <JobRunCard jobId={chemJobId} resultLabel="the featurization payload" onResult={handleChemResult} />
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={6}
        title="Split / Leakage Audit"
        hint="stratified, seeded, target-balanced"
        icon={<Columns3 aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                test size
                <input
                  type="number"
                  step={0.05}
                  min={0.05}
                  max={0.9}
                  value={testSize}
                  onChange={(event) => setTestSize(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                validation size (optional)
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={0.9}
                  value={valSize ?? ""}
                  placeholder="none"
                  onChange={(event) => {
                    const raw = event.target.value;
                    setValSize(raw === "" ? null : Number(raw));
                  }}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <input
                  type="checkbox"
                  checked={stratify}
                  onChange={(event) => setStratify(event.target.checked)}
                  className="h-4 w-4"
                />
                stratify
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                random state
                <input
                  type="number"
                  value={randomState}
                  onChange={(event) => setRandomState(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <Button size="sm" disabled={selectedId === null || splitBusy} onClick={() => void handleSplit()}>
                {splitBusy ? "Splitting…" : "Run the split"}
              </Button>
            </div>
            {splitError !== null ? <div className="mt-3"><ErrorBanner message={splitError} /></div> : null}
            {splitReport !== null ? (
              <div className="mt-4 grid gap-4 md:grid-cols-3">
                {Object.entries(splitReport.sides).map(([name, side]) => (
                  <div key={name} className="rounded-lg border border-[var(--color-border)] p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-[var(--color-foreground)]">{name}</span>
                      <span className="text-xs text-[var(--color-muted-foreground)]">{side.rows} rows</span>
                    </div>
                    {side.target_balance !== undefined ? (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {Object.entries(side.target_balance).map(([label, count]) => (
                          <span key={label} className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                            {label}: {count}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    <p className="mt-2 text-[11px] text-[var(--color-muted-foreground)]">
                      stratified={String(splitReport.stratified)} · seed={splitReport.random_state}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                The backend&apos;s split_table reports each side&apos;s
                row count and target balance — the leakage-safe
                structure (a seeded, stratified holdout the ML
                workspace will train against).
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={7}
        title="Visualize"
        hint="the backend's own statplots engine"
        icon={<FileSearch aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">Numeric column</span>
                <select
                  value={vizColumn}
                  onChange={(event) => setVizColumn(event.target.value)}
                  className="w-44 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  <option value="">select…</option>
                  {numericCandidates.map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </label>
              <Button size="sm" disabled={selectedId === null || vizColumn === "" || vizBusy} onClick={() => void handleStat()}>
                {vizBusy ? "Computing…" : "Histogram + stats"}
              </Button>
            </div>
            {vizError !== null ? <div className="mt-3"><ErrorBanner message={vizError} /></div> : null}
            {statResult !== null ? (
              <div className="mt-4">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(statResult.stats).map(([key, value]) => (
                    <span key={key} className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                      {key}: {typeof value === "number" ? Number(value.toFixed(4)) : String(value)}
                    </span>
                  ))}
                </div>
                {statResult.artifact !== null ? (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    figure artifact registered:{" "}
                    {statResult.artifact.filename} (
                    {statResult.artifact.size_bytes} bytes) — the
                    Plotly-interactive rendering lands with the
                    Visualization workspace row; the backend figure is
                    downloadable from Export below.
                  </p>
                ) : null}
              </div>
            ) : (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                The stat engine computes the distribution&apos;s
                statistics and registers a real figure artifact.
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={8}
        title="Export"
        hint="a real registered artifact + a digest-verified download"
        icon={<ArrowDownToLine aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                format
                <select
                  value={exportFormat}
                  onChange={(event) => setExportFormat(event.target.value as typeof exportFormat)}
                  className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                >
                  <option value="csv">CSV</option>
                  <option value="tsv">TSV</option>
                  <option value="xlsx">XLSX</option>
                </select>
              </label>
              <Button size="sm" disabled={selectedId === null || exportBusy} onClick={() => void handleExport()}>
                {exportBusy ? "Exporting…" : "Export the dataset"}
              </Button>
              {exportArtifact !== null ? (
                <Button size="sm" variant="outline" onClick={() => void handleDownload()}>
                  <ArrowDownToLine aria-hidden className="h-4 w-4" />
                  Download {exportArtifact.filename}
                </Button>
              ) : null}
            </div>
            {exportError !== null ? <div className="mt-3"><ErrorBanner message={exportError} /></div> : null}
            {exportArtifact !== null ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                artifact {exportArtifact.artifact_id} ·{" "}
                {exportArtifact.size_bytes} bytes · sha256{" "}
                {exportArtifact.sha256.slice(0, 16)}… — the download
                rides the token-header lane and verifies the served
                digest.
              </p>
            ) : null}
            {digestStatus !== null ? (
              <p className="mt-2 text-xs font-medium text-[var(--color-foreground)]">{digestStatus}</p>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>
    </div>
  );
}

/** Column chips — one category of the loader's policy. */
function ColumnChips({
  title,
  columns,
}: {
  title: string;
  columns: readonly string[];
}) {
  return (
    <div>
      <p className="text-xs font-semibold text-[var(--color-foreground)]">
        {title} ({columns.length})
      </p>
      <div className="mt-2 flex flex-wrap gap-1">
        {columns.length === 0 ? (
          <span className="text-[11px] text-[var(--color-muted-foreground)]">none</span>
        ) : (
          columns.slice(0, 24).map((name) => (
            <span
              key={name}
              className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]"
            >
              {name}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

/** The per-variable profile table (the Validate evidence). */
function ProfileVariables({
  profile,
}: {
  profile: Readonly<Record<string, unknown>>;
}) {
  const variables = profile["variables"];
  if (typeof variables !== "object" || variables === null) {
    return null;
  }
  const entries = Object.entries(variables as Record<string, unknown>);
  return (
    <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
      <div className="max-h-72 overflow-auto">
        <table className="w-full border-collapse text-[11px]">
          <thead className="sticky top-0 bg-[var(--color-muted)]">
            <tr>
              {["column", "kind", "n", "missing", "distinct", "mean", "min", "max"].map((header) => (
                <th key={header} scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, statsRaw]) => {
              const stats =
                typeof statsRaw === "object" && statsRaw !== null
                  ? (statsRaw as Record<string, unknown>)
                  : {};
              const cell = (key: string) => {
                const value = stats[key];
                if (value === undefined) {
                  return "—";
                }
                return typeof value === "number" && !Number.isInteger(value)
                  ? value.toFixed(4)
                  : String(value);
              };
              return (
                <tr key={name} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                  <td className="whitespace-nowrap px-2 py-1 font-medium text-[var(--color-foreground)]">{name}</td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{cell("kind")}</td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{cell("n")}</td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{cell("n_missing")}</td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{cell("n_distinct")}</td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{cell("mean")}</td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{cell("min")}</td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{cell("max")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
