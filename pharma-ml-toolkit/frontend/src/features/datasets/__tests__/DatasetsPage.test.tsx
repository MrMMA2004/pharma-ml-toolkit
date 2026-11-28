import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DatasetsPage } from "@/features/datasets/DatasetsPage";
import { ApiError } from "@/services/api/client";
import type { DatasetView, FramePreview } from "@/services/datasets/client";

/**
 * part13 Phase 3 — the Datasets workspace's first workflow stage as a
 * complete component-level user flow:
 *
 *   dataset import    — file selection gates the Upload button; the
 *                       multipart request carries the real File;
 *   validation errors — the typed ApiError surfaces verbatim;
 *   progress display  — the JobRunCard renders the tracked view's
 *                       state chip, percentage, stage, and Cancel;
 *   job cancellation  — the cooperative cancel action fires;
 *   job completion    — the completed result relay selects the
 *                       registered dataset end to end;
 *   the registry      — the honest empty state, the real row table,
 *                       row selection, and the preview's row-limit
 *                       control (the table interaction lane);
 *   failure states    — the registry's own API error banner.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, and the DOM are the real ones.
 */

const jobState = vi.hoisted(() => {
  const state: {
    snapshots: Record<string, { view: Record<string, unknown> } | null>;
    cancels: string[];
    refreshes: string[];
  } = { snapshots: {}, cancels: [], refreshes: [] };
  return state;
});

vi.mock("@/hooks/useJobTracking", () => ({
  useJob: (jobId: string | null) => ({
    snapshot: jobId === null ? null : jobState.snapshots[jobId] ?? null,
    cancel: jobId === null
      ? vi.fn()
      : vi.fn(async () => {
          jobState.cancels.push(jobId);
        }),
    refresh: jobId === null
      ? vi.fn()
      : vi.fn(async () => {
          jobState.refreshes.push(jobId);
        }),
  }),
}));

const datasetsMocks = vi.hoisted(() => ({
  list: vi.fn(),
  upload: vi.fn(),
  view: vi.fn(),
  preview: vi.fn(),
  profile: vi.fn(),
  split: vi.fn(),
  column: vi.fn(),
  remove: vi.fn(),
}));

vi.mock("@/services/datasets/client", () => ({
  datasetsApi: {
    list: datasetsMocks.list,
    upload: datasetsMocks.upload,
    view: datasetsMocks.view,
    preview: datasetsMocks.preview,
    profile: datasetsMocks.profile,
    split: datasetsMocks.split,
    column: datasetsMocks.column,
    remove: datasetsMocks.remove,
  },
}));

const chemistryMocks = vi.hoisted(() => ({
  descriptors: vi.fn(),
  fingerprints: vi.fn(),
  standardize: vi.fn(),
  registerFrames: vi.fn(),
}));
vi.mock("@/services/chemistry/client", async (importOriginal) => {
  // the real module keeps its types + parsers (the page narrows
  // untrusted payloads through the REAL parseStandardizeReport /
  // parseFramesRegisterReport); only the networked chemistryApi
  // surface is mocked
  const actual = await importOriginal<typeof import("@/services/chemistry/client")>();
  return {
    ...actual,
    chemistryApi: {
      descriptors: chemistryMocks.descriptors,
      fingerprints: chemistryMocks.fingerprints,
      standardize: chemistryMocks.standardize,
      registerFrames: chemistryMocks.registerFrames,
    },
  };
});

const artifactsMocks = vi.hoisted(() => ({ download: vi.fn() }));
vi.mock("@/services/artifacts/client", () => ({
  artifactsApi: { download: artifactsMocks.download },
}));

const jobsMocks = vi.hoisted(() => ({ result: vi.fn() }));
vi.mock("@/services/jobs/client", () => ({
  jobsApi: { result: jobsMocks.result },
}));

// the stat lane's raw REST client — stubbed so no accidental fetch
vi.mock("@/services/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api/client")>();
  return { ...actual, api: { get: vi.fn(), post: vi.fn() } };
});

const DATASET: DatasetView = {
  dataset_id: "ds-1",
  name: "solubility.csv",
  source_filename: "solubility.csv",
  size_bytes: 2048,
  shape: [128, 6],
  columns: ["smiles", "solubility"],
  target: "solubility",
  task: "regression",
  n_features: 4,
  feature_names: ["molwt", "logp"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: [],
  summary: {},
  created_at: "2026-11-21T00:00:00Z",
};

const FRAME: FramePreview = {
  columns: ["smiles", "solubility"],
  dtypes: { smiles: "object", solubility: "float64" },
  n_rows: 128,
  returned: 10,
  rows: [["CCO", -1.2], ["CCC", -0.4]],
};

beforeEach(() => {
  jobState.snapshots = {};
  jobState.cancels = [];
  jobState.refreshes = [];
  datasetsMocks.list.mockReset().mockResolvedValue({ datasets: [] });
  datasetsMocks.upload.mockReset();
  datasetsMocks.view.mockReset();
  datasetsMocks.preview.mockReset();
  datasetsMocks.remove.mockReset();
  jobsMocks.result.mockReset();
  chemistryMocks.standardize.mockReset();
  chemistryMocks.registerFrames.mockReset();
});

describe("the Import stage — file selection, submission, validation errors", () => {
  it("renders the honest empty registry until a file is imported", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [] });
    render(<DatasetsPage />);

    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalledTimes(1);
    });
    expect(
      screen.getByText("No datasets registered yet — import one above."),
    ).toBeInTheDocument();
  });

  it("keeps Upload disabled until a file is selected, then submits the real File", async () => {
    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalled();
    });

    const uploadButton = screen.getByRole("button", { name: "Upload" });
    expect(uploadButton).toBeDisabled();

    const fileInput = screen.getByLabelText("CSV / TSV / TXT / XLSX file");
    const file = new File(["smiles,solubility\nCCO,-1.2\n"], "solubility.csv", {
      type: "text/csv",
    });
    await userEvent.upload(fileInput, file);
    expect(uploadButton).toBeEnabled();

    datasetsMocks.upload.mockResolvedValueOnce({
      mode: "job",
      job: { job_id: "job-77" },
    });
    await userEvent.click(uploadButton);

    await waitFor(() => {
      expect(datasetsMocks.upload).toHaveBeenCalledWith(file, {
        target: null,
        task: null,
        mode: "job",
      });
    });
    // the tracked job card appears — the honest "waiting" state
    expect(await screen.findByText("Tracking job…")).toBeInTheDocument();
  });

  it("surfaces the typed ApiError verbatim — never a silent failure", async () => {
    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalled();
    });

    const fileInput = screen.getByLabelText("CSV / TSV / TXT / XLSX file");
    await userEvent.upload(
      fileInput,
      new File(["bad"], "bad.csv", { type: "text/csv" }),
    );
    datasetsMocks.upload.mockRejectedValueOnce(
      new ApiError(422, "non-numeric target column"),
    );
    await userEvent.click(screen.getByRole("button", { name: "Upload" }));

    expect(await screen.findByText(/API 422: non-numeric target column/)).toBeInTheDocument();
    // no job card was born — the failed request leaves nothing tracked
    expect(screen.queryByText("Tracking job…")).not.toBeInTheDocument();
  });
});

describe("the JobRunCard — progress display + cooperative cancellation", () => {
  it("renders the tracked view's state, percentage, stage, and the Cancel action", async () => {
    jobState.snapshots["job-77"] = {
      view: {
        state: "RUNNING",
        operation: "dataset_upload",
        progress: 42,
        stage: "parsing rows",
        error: null,
        result_ready: false,
      },
    };
    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalled();
    });

    const fileInput = screen.getByLabelText("CSV / TSV / TXT / XLSX file");
    await userEvent.upload(
      fileInput,
      new File(["smiles\nCCO\n"], "mini.csv", { type: "text/csv" }),
    );
    datasetsMocks.upload.mockResolvedValueOnce({
      mode: "job",
      job: { job_id: "job-77" },
    });
    await userEvent.click(screen.getByRole("button", { name: "Upload" }));

    expect(await screen.findByText("RUNNING")).toBeInTheDocument();
    expect(screen.getByText(/42% · parsing rows/)).toBeInTheDocument();
    expect(screen.getByText(/dataset_upload · the registered dataset view/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(jobState.cancels).toEqual(["job-77"]);
  });

  it("relays the completed job's result into the registry selection — the workflow's end-to-end", async () => {
    jobState.snapshots["job-77"] = {
      view: {
        state: "COMPLETED",
        operation: "dataset_upload",
        progress: 100,
        stage: "done",
        error: null,
        result_ready: true,
      },
    };
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    jobsMocks.result.mockResolvedValueOnce({ result: { dataset_id: "ds-1" } });

    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalled();
    });

    const fileInput = screen.getByLabelText("CSV / TSV / TXT / XLSX file");
    await userEvent.upload(
      fileInput,
      new File(["smiles\nCCO\n"], "mini.csv", { type: "text/csv" }),
    );
    datasetsMocks.upload.mockResolvedValueOnce({
      mode: "job",
      job: { job_id: "job-77" },
    });
    await userEvent.click(screen.getByRole("button", { name: "Upload" }));

    // the completed relay fetched the result, refreshed the registry,
    // and selected the registered dataset — all through the real page
    await waitFor(() => {
      expect(jobsMocks.result).toHaveBeenCalledWith("job-77");
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
      expect(datasetsMocks.preview).toHaveBeenCalledWith("ds-1", 10);
    });
    // the selected card renders the real summary line
    expect(
      await screen.findByText("128 rows × 6 columns · 4 numeric features · target solubility · task regression"),
    ).toBeInTheDocument();
  });
});

describe("the Inspect stage — the registry table and the row-limit control", () => {
  it("lists the registered datasets and selects one on a row click", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    render(<DatasetsPage />);

    const row = await screen.findByText("solubility.csv");
    const registryRow = row.closest("tr")!;
    expect(within(registryRow).getByText("128")).toBeInTheDocument(); // rows
    expect(within(registryRow).getByText("6")).toBeInTheDocument(); // columns

    await userEvent.click(row);
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
      expect(datasetsMocks.preview).toHaveBeenCalledWith("ds-1", 10);
    });
    expect(
      screen.getByText("128 rows × 6 columns · 4 numeric features · target solubility · task regression"),
    ).toBeInTheDocument();
  });

  it("re-fetches the preview with the chosen row limit — the table interaction lane", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    render(<DatasetsPage />);

    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.preview).toHaveBeenCalledWith("ds-1", 10);
    });

    datasetsMocks.preview.mockClear();
    const rowsLimit = screen.getByLabelText(/rows/);
    await userEvent.selectOptions(rowsLimit, "25");

    await waitFor(() => {
      expect(datasetsMocks.preview).toHaveBeenCalledWith("ds-1", 25);
    });
  });

  it("shows the registry's own API error honestly when the list fails", async () => {
    datasetsMocks.list.mockRejectedValueOnce(new ApiError(500, "database locked"));
    render(<DatasetsPage />);

    expect(await screen.findByText(/API 500: database locked/)).toBeInTheDocument();
    expect(
      screen.getByText("No datasets registered yet — import one above."),
    ).toBeInTheDocument();
  });
});

describe("the §3.1 deletion lane — the two-step gate over the typed evict", () => {
  it("arms the confirmation with the dataset's name and metadata — no request yet", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    render(<DatasetsPage />);

    const deleteAction = await screen.findByRole("button", {
      name: "Delete dataset solubility.csv",
    });
    await userEvent.click(deleteAction);

    expect(
      screen.getByText(/Delete dataset .solubility\.csv.\?/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/128 rows × 6 columns/),
    ).toBeInTheDocument();
    // the confirmation's scope note carries the dataset id too
    expect(
      screen.getAllByText(/ds-1/).length,
    ).toBeGreaterThanOrEqual(2);
    expect(datasetsMocks.remove).not.toHaveBeenCalled();
  });

  it("the row's Delete action never selects the row (the propagation guard)", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    render(<DatasetsPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Delete dataset solubility.csv" }),
    );

    expect(datasetsMocks.view).not.toHaveBeenCalled();
    expect(datasetsMocks.preview).not.toHaveBeenCalled();
  });

  it("Cancel disarms the gate without any request", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    render(<DatasetsPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Delete dataset solubility.csv" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByText(/Delete dataset .solubility\.csv.\?/)).toBeNull();
    expect(datasetsMocks.remove).not.toHaveBeenCalled();
  });

  it("confirm evicts through the typed client, refreshes the registry, and reports it", async () => {
    datasetsMocks.list.mockResolvedValueOnce({ datasets: [DATASET] });
    datasetsMocks.list.mockResolvedValue({ datasets: [] });
    datasetsMocks.remove.mockResolvedValue(undefined);
    render(<DatasetsPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Delete dataset solubility.csv" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Delete dataset" }));

    await waitFor(() => {
      expect(datasetsMocks.remove).toHaveBeenCalledTimes(1);
    });
    expect(datasetsMocks.remove).toHaveBeenCalledWith("ds-1");
    await waitFor(() => {
      // the registry refresh re-lists after the evict
      expect(datasetsMocks.list.mock.calls.length).toBeGreaterThanOrEqual(2);
    });
    expect(
      await screen.findByText(/Deleted .solubility\.csv. from the registry\./),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Delete dataset .solubility\.csv.\?/)).toBeNull();
  });

  it("deleting the selected dataset clears the selection — never a dangling record", async () => {
    datasetsMocks.list.mockResolvedValueOnce({ datasets: [DATASET] });
    datasetsMocks.list.mockResolvedValue({ datasets: [] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    datasetsMocks.remove.mockResolvedValue(undefined);
    render(<DatasetsPage />);

    // select the dataset first (the selection card renders)
    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });
    // then delete that same dataset through the gate
    await userEvent.click(
      screen.getByRole("button", { name: "Delete dataset solubility.csv" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Delete dataset" }));

    await waitFor(() => {
      expect(datasetsMocks.remove).toHaveBeenCalledWith("ds-1");
    });
    await waitFor(() => {
      // the selection card is gone — the deleted record cannot stay selected
      expect(screen.queryByText("solubility.csv")).toBeNull();
    });
  });

  it("the honest 404 lane — already-gone reconciles the registry instead of erroring", async () => {
    datasetsMocks.list.mockResolvedValueOnce({ datasets: [DATASET] });
    datasetsMocks.list.mockResolvedValue({ datasets: [] });
    datasetsMocks.remove.mockRejectedValueOnce(
      new ApiError(404, { detail: "Unknown dataset_id." }),
    );
    render(<DatasetsPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Delete dataset solubility.csv" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Delete dataset" }));

    await waitFor(() => {
      expect(
        screen.getByText(/was already gone \(404\) — registry refreshed\./),
      ).toBeInTheDocument();
    });
    expect(screen.queryByText(/Request failed:/)).toBeNull();
    expect(screen.queryByText(/Delete dataset .solubility\.csv.\?/)).toBeNull();
  });

  it("a hard failure stays on the confirmation card, surfaced verbatim", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.remove.mockRejectedValueOnce(new ApiError(500, "locked"));
    render(<DatasetsPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Delete dataset solubility.csv" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Delete dataset" }));

    expect(await screen.findByText(/API 500: locked/)).toBeInTheDocument();
    // the gate stays armed for a retry or an explicit cancel
    expect(
      screen.getByText(/Delete dataset .solubility\.csv.\?/),
    ).toBeInTheDocument();
  });
});

// ------------------------------------------------- DS-1: the strategy lane

describe("the Split stage's DS-1 strategy lane — scaffold/group-honest splits", () => {
  const SCAFFOLD_REPORT = {
    target: "solubility",
    stratified: false,
    random_state: 42,
    strategy: "scaffold" as const,
    smiles_column: "smiles",
    group_column: null,
    group_census: {
      sides: {
        train: { n_groups: 5, top_groups: [["c1ccccc1", 40]] },
        test: { n_groups: 2, top_groups: [["", 10]] },
      },
      shared_groups: [],
    },
    sides: {
      train: { rows: 96, columns: ["smiles", "solubility"], preview: FRAME },
      test: { rows: 32, columns: ["smiles", "solubility"], preview: FRAME },
    },
    n_rows: 128,
  };

  it("sends the scaffold strategy through and renders the group census + leakage verdict", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    datasetsMocks.split.mockReset().mockResolvedValueOnce(SCAFFOLD_REPORT);
    render(<DatasetsPage />);

    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });

    // the strategy selector: auto → scaffold
    await userEvent.selectOptions(screen.getByLabelText("strategy"), "scaffold");
    // the SMILES column select appears beside the chemistry stage's own
    // (and DS-2's stage-4 select — same label there; assert the trio,
    // then drive the split one, the last of the three)
    const smilesSelects = screen.getAllByLabelText("SMILES column");
    expect(smilesSelects).toHaveLength(3);
    expect(smilesSelects[2] ?? null).toHaveValue(""); // the auto-detect default

    await userEvent.click(screen.getByRole("button", { name: "Run the split" }));

    await waitFor(() => {
      expect(datasetsMocks.split).toHaveBeenCalledWith(
        "ds-1",
        expect.objectContaining({
          strategy: "scaffold",
          test_size: 0.2,
          random_state: 42,
          mode: "sync",
        }),
      );
    });
    // the auto-detect lane sends no explicit smiles_column
    expect(
      (datasetsMocks.split.mock.calls[0]![1] as Record<string, unknown>)
        .smiles_column,
    ).toBeUndefined();

    // per-side group counts + the leakage verdict line
    expect(await screen.findByText(/5 groups/)).toBeInTheDocument();
    expect(screen.getByText(/2 groups/)).toBeInTheDocument();
    expect(
      screen.getByText(/0 shared groups across sides/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Bemis–Murcko families kept whole/),
    ).toBeInTheDocument();
  });

  it("the group strategy keeps the run gated until a group column is picked", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    render(<DatasetsPage />);

    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });

    await userEvent.selectOptions(screen.getByLabelText("strategy"), "group");
    const run = screen.getByRole("button", { name: "Run the split" });
    expect(run).toBeDisabled();
    expect(screen.getByText(/group labels are never guessed/)).toBeInTheDocument();

    await userEvent.selectOptions(
      screen.getByLabelText("group column"),
      "smiles",
    );
    expect(screen.getByRole("button", { name: "Run the split" })).toBeEnabled();
  });

  it("the legacy auto lane sends no strategy — the boolean behavior unchanged", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    datasetsMocks.split.mockReset().mockResolvedValueOnce({
      ...SCAFFOLD_REPORT,
      strategy: null,
      smiles_column: null,
      group_census: null,
    });
    render(<DatasetsPage />);

    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });

    await userEvent.click(screen.getByRole("button", { name: "Run the split" }));
    await waitFor(() => {
      expect(datasetsMocks.split).toHaveBeenCalledWith(
        "ds-1",
        expect.objectContaining({ test_size: 0.2, mode: "sync" }),
      );
    });
    const body = datasetsMocks.split.mock.calls[0]![1] as Record<string, unknown>;
    expect(body.strategy).toBeUndefined();
    // the legacy answer renders without any census line
    expect(screen.queryByText(/shared groups/)).toBeNull();
  });
});

describe("the Clean/Standardize stage — DS-2's derived-dataset action", () => {
  it("runs the standardization job and relays the report end-to-end: the honest counts, the rejected rows, the derived dataset's select action, the registry refresh", async () => {
    jobState.snapshots["job-std"] = {
      view: {
        state: "COMPLETED",
        operation: "chemistry.standardize",
        progress: 100,
        stage: "done",
        error: null,
        result_ready: true,
      },
    };
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    chemistryMocks.standardize.mockResolvedValueOnce({
      mode: "job",
      job: { job_id: "job-std" },
    });
    jobsMocks.result.mockResolvedValueOnce({
      result: {
        source_dataset_id: "ds-1",
        column: "smiles",
        ops: ["reinvent"],
        counts: { n_rows: 5, changed: 2, unchanged: 1, rejected: 2 },
        rejected: [
          { row: 2, input: "not-a-smiles", reason: "unparseable" },
          { row: 3, input: "C", reason: "rejected-by-standardization-chain" },
        ],
        provenance: {
          kind: "standardized",
          source_dataset_id: "ds-1",
          column: "smiles",
          ops: ["reinvent"],
          neutralize: true,
        },
        dataset: {
          dataset_id: "ds-2-std",
          name: "solubility.csv (standardized)",
          shape: [3, 2],
          columns: ["smiles", "solubility"],
          target: "solubility",
        },
      },
    });

    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalledTimes(1);
    });

    // select the registered dataset — the guessed SMILES column arms
    // the stage-4 action (stdColumn prefilled from the view)
    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });

    const runButton = await screen.findByRole("button", {
      name: "Standardize → derived dataset",
    });
    // the default lanes: the REINVENT chain on, uncharge on, no
    // tautomer pass — the pre-guessed column keeps it enabled
    expect(runButton).toBeEnabled();
    await userEvent.click(runButton);

    await waitFor(() => {
      expect(chemistryMocks.standardize).toHaveBeenCalledWith("ds-1", {
        column: "smiles",
        ops: ["reinvent"],
        neutralize: true,
        mode: "job",
      });
    });

    // the completed job relayed the report: the counts line, the
    // honest rejected table, the derived dataset identity, and the
    // registry refresh (the derived record lands in the list call)
    await waitFor(() => {
      expect(jobsMocks.result).toHaveBeenCalledWith("job-std");
      expect(datasetsMocks.list).toHaveBeenCalledTimes(2);
    });
    expect(
      await screen.findByText(/5 rows → 2 changed · 1 unchanged · 2 rejected/),
    ).toBeInTheDocument();
    // the honest rejected table: unparseable garbage and the
    // chain-rejected methane row, each with its reason
    expect(screen.getByText(/not-a-smiles/)).toBeInTheDocument();
    expect(screen.getByText(/rejected-by-standardization-chain/)).toBeInTheDocument();
    expect(
      screen.getByText(/solubility\.csv \(standardized\) \(ds-2-std\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/the source is untouched/),
    ).toBeInTheDocument();

    // the derived-dataset select action re-selects through the real lane
    await userEvent.click(
      screen.getByRole("button", { name: "Select the derived dataset" }),
    );
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-2-std");
      expect(datasetsMocks.preview).toHaveBeenCalledWith("ds-2-std", 10);
    });
  });

  it("gates the run honestly: no selected dataset, no column, no pass — no request", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalled();
    });

    // before a selection the stage-4 action is inert
    expect(
      screen.queryByRole("button", { name: "Standardize → derived dataset" }),
    ).toBeNull();

    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });

    // both passes off -> no ops -> the button stays disabled
    await userEvent.click(screen.getByLabelText(/REINVENT4 chain/));
    const runButton = screen.getByRole("button", {
      name: "Standardize → derived dataset",
    });
    expect(runButton).toBeDisabled();
    expect(chemistryMocks.standardize).not.toHaveBeenCalled();
  });
});

describe("the Descriptor/Fingerprint register lane — DS-3's derived-dataset action", () => {
  it("runs the register job and relays the report end-to-end: the honest counts, the rejected rows, the derived dataset's select action, the Machine-Learning navigate hint, the registry refresh", async () => {
    jobState.snapshots["job-reg"] = {
      view: {
        state: "COMPLETED",
        operation: "chemistry.frames.register",
        progress: 100,
        stage: "done",
        error: null,
        result_ready: true,
      },
    };
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    chemistryMocks.registerFrames.mockResolvedValueOnce({
      mode: "job",
      job: { job_id: "job-reg" },
    });
    jobsMocks.result.mockResolvedValueOnce({
      result: {
        source_dataset_id: "ds-1",
        column: "smiles",
        featurization: "descriptors",
        counts: { n_rows: 8, kept: 6, rejected: 2, features_appended: 217 },
        rejected: [
          { row: 1, input: "not-a-smiles", reason: "unparseable" },
          { row: 2, input: "[H]", reason: "descriptor-computation-failure" },
        ],
        provenance: {
          kind: "derived-descriptors",
          source_dataset_id: "ds-1",
          column: "smiles",
          descriptor_set: "morgan",
          n_features: 217,
          target: "solubility",
        },
        dataset: {
          dataset_id: "ds-2-desc",
          name: "solubility.csv (descriptors)",
          shape: [6, 219],
          columns: ["smiles", "solubility"],
          target: "solubility",
        },
      },
    });
    const onNavigate = vi.fn();

    render(<DatasetsPage onNavigate={onNavigate} />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalledTimes(1);
    });

    // select the registered dataset — the guessed SMILES column arms
    // the register lane (no pull needed: the engine re-reads the
    // source column server-side)
    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });

    const registerButton = await screen.findByRole("button", {
      name: "Register as derived dataset",
    });
    expect(registerButton).toBeEnabled();
    await userEvent.click(registerButton);

    await waitFor(() => {
      expect(chemistryMocks.registerFrames).toHaveBeenCalledWith("ds-1", {
        column: "smiles",
        featurization: "descriptors",
        descriptorSet: "morgan",
        kind: "morgan",
        radius: 2,
        nBits: 2048,
        asCounts: false,
        includeChirality: false,
        target: null,
        mode: "job",
      });
    });

    // the completed job relayed the report: the counts line, the
    // honest rejected table, the derived identity, the registry
    // refresh
    await waitFor(() => {
      expect(jobsMocks.result).toHaveBeenCalledWith("job-reg");
      expect(datasetsMocks.list).toHaveBeenCalledTimes(2);
    });
    expect(
      await screen.findByText(
        /8 rows → 6 kept · 2 rejected · 217 feature columns appended/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/not-a-smiles/)).toBeInTheDocument();
    expect(
      screen.getByText(/descriptor-computation-failure/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/solubility\.csv \(descriptors\) \(ds-2-desc\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/kind "derived-descriptors" from ds-1/),
    ).toBeInTheDocument();

    // the navigate hint: the shell's own workspace switcher hands
    // the workflow over to the ML stage
    await userEvent.click(
      screen.getByRole("button", { name: "Continue in Machine Learning" }),
    );
    expect(onNavigate).toHaveBeenCalledWith("machine-learning");

    // the derived-dataset select action re-selects through the real lane
    await userEvent.click(
      screen.getByRole("button", { name: "Select the derived dataset" }),
    );
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-2-desc");
      expect(datasetsMocks.preview).toHaveBeenCalledWith("ds-2-desc", 10);
    });
  });

  it("switches the featurization select to the fingerprints branch — the request carries the branch + the fp knobs", async () => {
    jobState.snapshots["job-reg-fp"] = {
      view: {
        state: "COMPLETED",
        operation: "chemistry.frames.register",
        progress: 100,
        stage: "done",
        error: null,
        result_ready: true,
      },
    };
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    chemistryMocks.registerFrames.mockResolvedValueOnce({
      mode: "job",
      job: { job_id: "job-reg-fp" },
    });
    jobsMocks.result.mockResolvedValueOnce({
      result: {
        source_dataset_id: "ds-1",
        column: "smiles",
        featurization: "fingerprints",
        counts: { n_rows: 8, kept: 7, rejected: 1, features_appended: 2048 },
        rejected: [
          { row: 1, input: "not-a-smiles", reason: "unparseable" },
        ],
        provenance: {
          kind: "derived-fingerprints",
          source_dataset_id: "ds-1",
          column: "smiles",
          fingerprint_kind: "morgan",
          radius: 2,
          n_bits: 2048,
          as_counts: false,
          include_chirality: false,
          target: "solubility",
        },
        dataset: {
          dataset_id: "ds-2-fp",
          name: "solubility.csv (fingerprints)",
          shape: [7, 2050],
          columns: ["smiles", "solubility"],
          target: "solubility",
        },
      },
    });

    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalledTimes(1);
    });
    await userEvent.click(await screen.findByText("solubility.csv"));
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });

    // the featurization select: switch to the fingerprints branch
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "featurization" }),
      "fingerprints",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Register as derived dataset" }),
    );
    await waitFor(() => {
      expect(chemistryMocks.registerFrames).toHaveBeenCalledWith("ds-1", {
        column: "smiles",
        featurization: "fingerprints",
        descriptorSet: "morgan",
        kind: "morgan",
        radius: 2,
        nBits: 2048,
        asCounts: false,
        includeChirality: false,
        target: null,
        mode: "job",
      });
    });
    expect(
      await screen.findByText(
        /kind "derived-fingerprints" from ds-1/,
      ),
    ).toBeInTheDocument();
    // no navigate prop -> the hint button stays absent
    expect(
      screen.queryByRole("button", { name: "Continue in Machine Learning" }),
    ).toBeNull();
  });

  it("gates the run honestly: no selected dataset or column — no request", async () => {
    datasetsMocks.list.mockResolvedValue({ datasets: [DATASET] });
    datasetsMocks.view.mockResolvedValue(DATASET);
    datasetsMocks.preview.mockResolvedValue(FRAME);
    render(<DatasetsPage />);
    await waitFor(() => {
      expect(datasetsMocks.list).toHaveBeenCalled();
    });

    // before a selection the register action is inert
    const registerButton = screen.getByRole("button", {
      name: "Register as derived dataset",
    });
    expect(registerButton).toBeDisabled();
    expect(chemistryMocks.registerFrames).not.toHaveBeenCalled();
  });
});
