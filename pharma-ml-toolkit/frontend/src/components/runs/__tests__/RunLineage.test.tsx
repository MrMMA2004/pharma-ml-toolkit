/**
 * part16 §15.2 (Phase 12) — the Experiment log card's own contract.
 *
 * The lineage surface as a component-level user flow:
 *
 *   the lineage table — the run summaries render with the status
 *                       badge, the model/dataset identity, the CV
 *                       mean (the honest dash when absent), the mode,
 *                       the seed, the creation stamp, and the count
 *                       in the header row;
 *   the empty state  — no runs recorded yet states the honest next
 *                       step (train in stage 7) instead of a fabricated
 *                       row;
 *   the §15.2 detail — a row click fetches the full record on demand
 *                       and expands the definition-style lineage
 *                       blocks (dataset lineage, algorithm, the seed),
 *                       caching per run_id so a re-expand never
 *                       refetches;
 *   the failure lane — a rejected listing surfaces through the
 *                       error banner, never a fabricated table.
 *
 * The runs service is mocked at the module boundary only; the
 * component, its state machine, and the DOM are the real ones.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const runsMocks = vi.hoisted(() => ({
  list: vi.fn(),
  detail: vi.fn(),
}));
vi.mock("@/services/runs/client", () => ({
  runsApi: { list: runsMocks.list, detail: runsMocks.detail },
}));

import { RunLineage } from "@/components/runs/RunLineage";
import type { ExperimentRun, RunSummary } from "@/services/runs/types";

const COMPLETED_SUMMARY: RunSummary = {
  run_id: "run-4f0a9c21b7d3",
  family: "ml.train",
  mode: "sync",
  status: "completed",
  job_id: null,
  dataset_id: "ds-20250601-0001",
  dataset_name: "Walkthrough classification",
  model_name: "logistic_regression",
  task: "classification",
  metric: 0.9123,
  cv_folds: 5,
  random_seed: 42,
  created_at: "2025-06-01T10:00:00+00:00",
  started_at: "2025-06-01T10:00:01+00:00",
  completed_at: "2025-06-01T10:00:03+00:00",
  error: null,
};

const FAILED_SUMMARY: RunSummary = {
  ...COMPLETED_SUMMARY,
  run_id: "run-9911aabbccdd",
  mode: "job",
  status: "failed",
  job_id: "job-20250601-0007",
  model_name: "random_forest",
  metric: null,
  error: "n_estimators must be greater than 0, got 0",
};

const FULL_RUN: ExperimentRun = {
  ...COMPLETED_SUMMARY,
  dataset: {
    dataset_id: "ds-20250601-0001",
    name: "Walkthrough classification",
    source_filename: "walkthrough_classification.csv",
    size_bytes: 43883,
    shape: [120, 9],
    target: "activity",
    task: "classification",
    n_features: 8,
    created_at: "2025-06-01T09:00:00+00:00",
  },
  feature_configuration: {
    feature_names: ["molwt", "logp", "tpsa"],
    n_features: 8,
    target: "activity",
  },
  preprocessing: {
    enabled: true,
    pipeline:
      "median imputation -> StandardScaler (non-numeric columns: " +
      "most_frequent imputation + one-hot encoding, handle_unknown=ignore)",
  },
  algorithm: {
    name: "logistic_regression",
    task: "classification",
    toolkit_version: "1.2.240",
  },
  hyperparameters: {
    requested: { C: 0.5 },
    tuned: true,
    best_params: { C: 0.5 },
  },
  cross_validation: {
    cv_folds: 5,
    tuning: true,
    n_jobs_requested: 1,
    n_jobs_effective: 1,
    parallelism_advisory: {
      requested: 1,
      executed: 1,
      reason: null,
      estimated_bytes: null,
      available_bytes: 6_200_000_000,
    },
  },
  evaluation: {
    cv_scores: [0.91, 0.92, 0.9, 0.93, 0.89],
    cv_mean: 0.9123,
    cv_std: 0.0141,
    primary_metric: "roc_auc_ovo (balanced_accuracy without predict_proba)",
  },
  model_artifact: {
    model_id: "model-20250601-0001",
    name: "logistic_regression",
    task: "classification",
    n_train_rows: 96,
    fit_seconds: 0.21,
    created_at: "2025-06-01T10:00:03+00:00",
    fetch: "GET /api/v1/ml/trained/model-20250601-0001",
  },
  environment: {
    toolkit: "1.2.240",
    python: "3.11.9",
    platform: "Linux-6.5.0-x86_64",
    machine: "x86_64",
    packages: { numpy: "1.26.4", rdkit: null },
    numeric_environment: { OPENBLAS_NUM_THREADS: null },
  },
  logs: {
    job_id: null,
    stage: "completed",
    history: "sync mode (no job history)",
  },
  schema: "pharmaml-experiment-run/1",
};

beforeEach(() => {
  runsMocks.list.mockReset();
  runsMocks.detail.mockReset();
});

describe("RunLineage (part16 §15.2)", () => {
  it("renders the summary table after load — every column and the count", async () => {
    runsMocks.list.mockResolvedValue({
      runs: [COMPLETED_SUMMARY, FAILED_SUMMARY],
      count: 2,
    });

    render(<RunLineage />);

    expect(runsMocks.list).toHaveBeenCalledTimes(1);
    expect(runsMocks.list).toHaveBeenCalledWith();
    // the header row: the title, then the count once the listing lands
    expect(screen.getByText("Experiment log")).toBeInTheDocument();
    // the lineage table: the model and dataset identities
    expect(await screen.findByText("logistic_regression")).toBeInTheDocument();
    expect(screen.getByText("random_forest")).toBeInTheDocument();
    expect(screen.getByText("2 runs")).toBeInTheDocument();
    // both runs trained on the same dataset — the column repeats it
    expect(screen.getAllByText("Walkthrough classification")).toHaveLength(2);
    // the metric column: the CV mean, and the honest dash for the failed run
    expect(screen.getByText("0.9123")).toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
    // the status badges render their own vocabulary
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    // the mode and seed columns (the seed rides both rows)
    expect(screen.getByText("sync")).toBeInTheDocument();
    expect(screen.getByText("job")).toBeInTheDocument();
    expect(screen.getAllByText("42")).toHaveLength(2);
    // the creation stamp
    expect(screen.getAllByText("2025-06-01T10:00:00+00:00")).toHaveLength(2);
    // the detail blocks stay closed until a row is clicked
    expect(
      screen.queryByRole("region", { name: "Dataset lineage" }),
    ).not.toBeInTheDocument();
  });

  it("shows the honest empty state before any run is recorded", async () => {
    runsMocks.list.mockResolvedValue({ runs: [], count: 0 });

    render(<RunLineage />);

    await screen.findByText(
      "No runs recorded yet — train a model in stage 7 and the lineage records itself.",
    );
    // the honest zero, never a fabricated row
    expect(screen.getByText("0 runs")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("fetches the §15.2 detail on demand and expands the lineage blocks", async () => {
    runsMocks.list.mockResolvedValue({
      runs: [COMPLETED_SUMMARY],
      count: 1,
    });
    runsMocks.detail.mockResolvedValue(FULL_RUN);

    render(<RunLineage />);
    const row = (await screen.findByText("logistic_regression")).closest("tr");
    expect(row).not.toBeNull();
    await userEvent.click(row as HTMLTableRowElement);

    // the detail fetched on demand with the row's own run_id
    expect(runsMocks.detail).toHaveBeenCalledWith("run-4f0a9c21b7d3");

    // the dataset lineage block: name/shape/target/source_filename
    const dataset = await screen.findByRole("region", {
      name: "Dataset lineage",
    });
    expect(
      within(dataset).getByText("walkthrough_classification.csv"),
    ).toBeInTheDocument();
    expect(within(dataset).getByText("120 × 9")).toBeInTheDocument();
    expect(within(dataset).getByText("activity")).toBeInTheDocument();

    // the algorithm block with the toolkit version
    const algorithm = screen.getByRole("region", { name: "Algorithm" });
    expect(
      within(algorithm).getByText("logistic_regression"),
    ).toBeInTheDocument();
    expect(within(algorithm).getByText("1.2.240")).toBeInTheDocument();

    // the seed rides the cross-validation configuration block
    const cv = screen.getByRole("region", { name: "Cross-validation" });
    expect(within(cv).getByText("Random seed")).toBeInTheDocument();
    expect(within(cv).getByText("42")).toBeInTheDocument();

    // the model artifact reference carries the fetch hint
    const artifact = screen.getByRole("region", { name: "Model artifact" });
    expect(
      within(artifact).getByText("GET /api/v1/ml/trained/model-20250601-0001"),
    ).toBeInTheDocument();

    // the job reference rides the history string
    const job = screen.getByRole("region", { name: "Job reference" });
    expect(
      within(job).getByText("sync mode (no job history)"),
    ).toBeInTheDocument();
  });

  it("caches the detail per run_id — a re-expand never refetches", async () => {
    runsMocks.list.mockResolvedValue({
      runs: [COMPLETED_SUMMARY],
      count: 1,
    });
    runsMocks.detail.mockResolvedValue(FULL_RUN);

    render(<RunLineage />);
    const row = (await screen.findByText("logistic_regression")).closest(
      "tr",
    ) as HTMLTableRowElement;

    // expand, collapse, expand again — one fetch, ever
    await userEvent.click(row);
    await screen.findByRole("region", { name: "Dataset lineage" });
    await userEvent.click(row);
    expect(
      screen.queryByRole("region", { name: "Dataset lineage" }),
    ).not.toBeInTheDocument();
    await userEvent.click(row);
    await screen.findByRole("region", { name: "Dataset lineage" });
    expect(runsMocks.detail).toHaveBeenCalledTimes(1);
  });

  it("surfaces a listing failure through the error banner — no fabricated table", async () => {
    runsMocks.list.mockRejectedValue(
      new Error("the backend endpoint is not configured yet"),
    );

    render(<RunLineage />);

    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("Request failed:");
    expect(banner.textContent).toContain(
      "the backend endpoint is not configured yet",
    );
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("surfaces a detail failure through the error banner and retries on re-open", async () => {
    runsMocks.list.mockResolvedValue({
      runs: [COMPLETED_SUMMARY],
      count: 1,
    });
    runsMocks.detail.mockRejectedValueOnce(new Error("API 404: Unknown run_id."));

    render(<RunLineage />);
    const row = (await screen.findByText("logistic_regression")).closest(
      "tr",
    ) as HTMLTableRowElement;

    await userEvent.click(row);
    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("Unknown run_id.");
    expect(
      screen.queryByRole("region", { name: "Dataset lineage" }),
    ).not.toBeInTheDocument();

    // the failed fetch never caches — a re-open retries the record
    runsMocks.detail.mockResolvedValue(FULL_RUN);
    await userEvent.click(row); // collapse
    await userEvent.click(row); // re-open
    await screen.findByRole("region", { name: "Dataset lineage" });
    expect(runsMocks.detail).toHaveBeenCalledTimes(2);
  });

  it("re-fires the listing through the Refresh button", async () => {
    runsMocks.list.mockResolvedValue({
      runs: [COMPLETED_SUMMARY],
      count: 1,
    });

    render(<RunLineage />);
    await screen.findByText("logistic_regression");

    await userEvent.click(
      screen.getByRole("button", { name: "refresh the experiment log" }),
    );

    await waitFor(() => expect(runsMocks.list).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("logistic_regression")).toBeInTheDocument();
  });
});
