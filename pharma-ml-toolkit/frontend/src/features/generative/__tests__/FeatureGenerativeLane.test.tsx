import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GenerativePage } from "@/features/generative/GenerativePage";

/**
 * part18 §5.6 — the feature-space GAN / diffusion lanes of the
 * Generative AI workspace as component-level user flows:
 *
 *   the honest scope   — FEATURE-space, never molecules, stated where
 *                        the lanes live (the cards + every response);
 *   the GAN lane      — the train submission over the SELECTED
 *                        dataset's feature matrix (job mode) and the
 *                        sample lane with the distributional parity
 *                        report;
 *   the diffusion lane — the DDPM train + reverse-chain sample with
 *                        the same parity contract.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, and the DOM are the real ones.
 */

const jobState = vi.hoisted(() => {
  const state: {
    snapshots: Record<string, { view: Record<string, unknown> } | null>;
  } = { snapshots: {} };
  return state;
});

vi.mock("@/hooks/useJobTracking", () => ({
  useJob: (jobId: string | null) => ({
    snapshot: jobId === null ? null : jobState.snapshots[jobId] ?? null,
    cancel: vi.fn(),
    refresh: vi.fn(),
  }),
}));

const generativeMocks = vi.hoisted(() => ({
  objectives: vi.fn(),
  priors: vi.fn(),
  ganTrain: vi.fn(),
  ganSample: vi.fn(),
  diffusionTrain: vi.fn(),
  diffusionSample: vi.fn(),
}));

vi.mock("@/services/generative/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/generative/client")>();
  return {
    ...actual,
    generativeApi: {
      ...actual.generativeApi,
      objectives: generativeMocks.objectives,
      priors: generativeMocks.priors,
      ganTrain: generativeMocks.ganTrain,
      ganSample: generativeMocks.ganSample,
      diffusionTrain: generativeMocks.diffusionTrain,
      diffusionSample: generativeMocks.diffusionSample,
    },
  };
});

const datasetsMocks = vi.hoisted(() => ({ list: vi.fn(), view: vi.fn() }));
vi.mock("@/services/datasets/client", () => ({
  datasetsApi: { list: datasetsMocks.list, view: datasetsMocks.view },
}));

const jobsMocks = vi.hoisted(() => ({ result: vi.fn() }));
vi.mock("@/services/jobs/client", () => ({
  jobsApi: { result: jobsMocks.result },
}));

vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => ({
    subscribe: () => () => undefined,
  }),
}));

vi.mock("@/services/artifacts/client", () => ({
  artifactsApi: { download: vi.fn() },
  parseArtifactView: vi.fn(),
}));

const DATASET_VIEW = {
  dataset_id: "ds-1",
  name: "demo",
  source_filename: "demo.csv",
  size_bytes: 1024,
  shape: [60, 2],
  columns: ["x1", "x2", "y"],
  target: "y",
  task: "classification",
  n_features: 2,
  feature_names: ["x1", "x2"],
  classes: ["0", "1"],
  class_counts: { "0": 30, "1": 30 },
  dropped_columns: [],
  categorical_columns: [],
  summary: {},
  created_at: "2026-11-27T00:00:00Z",
};

const GAN_PRIOR_RESULT = {
  prior: {
    prior_id: "prior-ganaa000111",
    name: "feature-gan",
    kind: "feature_gan",
    params: { latent_dim: 8, epochs: 40 },
    n_train_rows: 60,
    n_epochs: 40,
    final_loss: 0.65,
    final_d_loss: 0.7,
    losses: [1.1, 0.8, 0.65],
    dataset_source: "dataset:ds-1",
    created_at: "2026-11-27T00:00:00Z",
    sample_url: "/api/v1/generative/gan/sample",
    molecular: false,
    note: "feature-space generative model — synthetic feature rows, NOT molecules (the SMILES lanes own molecular generation)",
  },
};

const GAN_SAMPLE_RESULT = {
  priorId: "prior-ganaa000111",
  nRequested: 24,
  nSampled: 24,
  rows: [[1.0, 2.0], [0.5, 1.5]],
  quality: {
    nRef: 60,
    meanDelta: [0.1, 0.2],
    stdDelta: [0.05, 0.1],
    corrL1: 0.12,
  },
  seed: 0,
  note: "synthetic feature rows — a distributional parity report, not real data; feature-space generation, not molecules",
};

const DIFFUSION_SAMPLE_RESULT = {
  priorId: "prior-diffbb000222",
  nRequested: 24,
  nSampled: 24,
  rows: [[1.1, 1.9], [0.4, 1.6]],
  quality: {
    nRef: 60,
    meanDelta: [0.05, 0.08],
    stdDelta: [0.03, 0.07],
    corrL1: 0.09,
  },
  seed: 0,
  note: "synthetic feature rows — a distributional parity report, not real data; feature-space generation, not molecules",
};

const GAN_PRIOR_PARSED = {
  priorId: "prior-ganaa000111",
  name: "feature-gan",
  kind: "feature_gan",
  params: { latent_dim: 8, epochs: 40 },
  nTrainRows: 60,
  nEpochs: 40,
  finalLoss: 0.65,
  finalDLoss: 0.7,
  losses: [1.1, 0.8, 0.65],
  datasetSource: "dataset:ds-1",
  createdAt: "2026-11-27T00:00:00Z",
  sampleUrl: "/api/v1/generative/gan/sample",
  molecular: false,
  note: "feature-space generative model — synthetic feature rows, NOT molecules (the SMILES lanes own molecular generation)",
};

const DIFFUSION_PRIOR_PARSED = {
  priorId: "prior-diffbb000222",
  name: "feature-diffusion",
  kind: "feature_diffusion",
  params: { timesteps: 50, epochs: 40 },
  nTrainRows: 60,
  nEpochs: 40,
  finalLoss: 1.17,
  finalDLoss: null,
  losses: [1.4, 1.3, 1.17],
  datasetSource: "dataset:ds-1",
  createdAt: "2026-11-27T00:00:00Z",
  sampleUrl: "/api/v1/generative/diffusion/sample",
  molecular: false,
  note: "feature-space generative model — synthetic feature rows, NOT molecules (the SMILES lanes own molecular generation)",
};

function completedJobView(jobId: string, operation: string): Record<string, unknown> {
  return {
    job_id: jobId,
    operation,
    category: "generative-ai",
    state: "COMPLETED",
    progress: 100,
    stage: "done",
    error: null,
    created_at: "2026-11-27T00:00:00Z",
    started_at: "2026-11-27T00:00:01Z",
    finished_at: "2026-11-27T00:00:04Z",
    configuration: {},
    result_ready: true,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  generativeMocks.objectives.mockResolvedValue({
    objectives: [
      { kind: "qed", class: "QedObjective", params: {}, description: "quantified drug-likeness (0..1, higher better)" },
    ],
  });
  generativeMocks.priors.mockResolvedValue({ priors: [] });
  datasetsMocks.list.mockResolvedValue({ datasets: [DATASET_VIEW] });
  datasetsMocks.view.mockResolvedValue(DATASET_VIEW);
});

describe("the feature-space generative lanes — the honest scope", () => {
  it("renders both lanes with the feature-space, not-molecules statements", async () => {
    render(<GenerativePage />);
    expect(
      await screen.findByRole("heading", {
        name: "Feature-space GAN & diffusion — the generative model lanes",
      }),
    ).toBeVisible();
    expect(
      screen.getByText(/synthetic feature ROWS, never\s+molecules/i),
    ).toBeVisible();
    expect(
      screen.getByText(/never experimentally validated data/i),
    ).toBeVisible();
  });

  it("gates the GAN train on a selected dataset", async () => {
    render(<GenerativePage />);
    const trainButton = await screen.findByRole("button", {
      name: "Train the feature GAN",
    });
    expect(trainButton).toBeDisabled();
    expect(
      screen.getByText(/select a dataset in the prior lane above/i),
    ).toBeVisible();
  });
});

describe("the GAN lane — train, sample, the parity report", () => {
  it("trains over the SELECTED dataset in job mode and relays the prior view", async () => {
    const user = userEvent.setup();
    generativeMocks.ganTrain.mockResolvedValue({
      mode: "job",
      job: completedJobView("job-gan-1", "generative.gan_train"),
    });
    jobState.snapshots["job-gan-1"] = {
      view: completedJobView("job-gan-1", "generative.gan_train"),
    };
    jobsMocks.result.mockResolvedValue({ result: GAN_PRIOR_RESULT });
    render(<GenerativePage />);
    // select the dataset via the registry row
    const nameCell = await screen.findByText("demo");
    const row = nameCell.closest("tr");
    expect(row).not.toBeNull();
    await user.click(row as HTMLElement);
    await waitFor(() => {
      expect(datasetsMocks.view).toHaveBeenCalledWith("ds-1");
    });
    const trainButton = await screen.findByRole("button", {
      name: "Train the feature GAN",
    });
    await waitFor(() => expect(trainButton).toBeEnabled());
    await user.click(trainButton);
    await waitFor(() => {
      expect(generativeMocks.ganTrain).toHaveBeenCalledTimes(1);
    });
    const options = generativeMocks.ganTrain.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(options["datasetId"]).toBe("ds-1");
    expect(options["latentDim"]).toBe(8);
    // the relayed prior view: kind, final g_loss, molecular false
    expect(
      await screen.findByText(/prior-ganaa000111/i),
    ).toBeVisible();
    expect(screen.getByText(/no — feature-space/i)).toBeVisible();
  });

  it("samples with the honest parity report", async () => {
    const user = userEvent.setup();
    generativeMocks.ganTrain.mockResolvedValue({
      mode: "sync",
      prior: GAN_PRIOR_PARSED,
    });
    generativeMocks.ganSample.mockResolvedValue(GAN_SAMPLE_RESULT);
    render(<GenerativePage />);
    const nameCell = await screen.findByText("demo");
    await user.click(nameCell.closest("tr") as HTMLElement);
    const trainButton = await screen.findByRole("button", {
      name: "Train the feature GAN",
    });
    await waitFor(() => expect(trainButton).toBeEnabled());
    await user.click(trainButton);
    const sampleButton = await screen.findByRole("button", {
      name: "Sample rows",
    });
    await waitFor(() => expect(sampleButton).toBeEnabled());
    await user.click(sampleButton);
    await waitFor(() => {
      expect(generativeMocks.ganSample).toHaveBeenCalledTimes(1);
    });
    const opts = generativeMocks.ganSample.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(opts["priorId"]).toBe("prior-ganaa000111");
    expect(opts["n"]).toBe(24);
    // the parity report renders: reference rows, the corr L1, the note
    expect(await screen.findByText(/60 reference rows/i)).toBeVisible();
    expect(screen.getByText(/corr L1 0\.120/i)).toBeVisible();
    expect(screen.getByText(/not real data/i)).toBeVisible();
  });
});

describe("the diffusion lane — train and the reverse chain", () => {
  it("trains and samples through the DDPM lane", async () => {
    const user = userEvent.setup();
    generativeMocks.diffusionTrain.mockResolvedValue({
      mode: "sync",
      prior: DIFFUSION_PRIOR_PARSED,
    });
    generativeMocks.diffusionSample.mockResolvedValue(DIFFUSION_SAMPLE_RESULT);
    render(<GenerativePage />);
    const nameCell = await screen.findByText("demo");
    await user.click(nameCell.closest("tr") as HTMLElement);
    const trainButton = await screen.findByRole("button", {
      name: "Train the diffusion model",
    });
    await waitFor(() => expect(trainButton).toBeEnabled());
    await user.click(trainButton);
    await waitFor(() => {
      expect(generativeMocks.diffusionTrain).toHaveBeenCalledTimes(1);
    });
    const options = generativeMocks.diffusionTrain.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(options["datasetId"]).toBe("ds-1");
    expect(options["timesteps"]).toBe(50);
    expect(await screen.findByText(/prior-diffbb000222/i)).toBeVisible();
    const sampleButton = await screen.findByRole("button", {
      name: "Run the reverse chain",
    });
    await waitFor(() => expect(sampleButton).toBeEnabled());
    await user.click(sampleButton);
    await waitFor(() => {
      expect(generativeMocks.diffusionSample).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByText(/corr L1 0\.090/i)).toBeVisible();
  });
});
