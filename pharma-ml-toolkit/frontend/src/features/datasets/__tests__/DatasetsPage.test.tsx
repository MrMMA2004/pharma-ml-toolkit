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
}));
vi.mock("@/services/chemistry/client", () => ({
  chemistryApi: {
    descriptors: chemistryMocks.descriptors,
    fingerprints: chemistryMocks.fingerprints,
  },
}));

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
