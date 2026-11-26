import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ReinforceLane, UnsupervisedLearningLane } from "@/features/ml/UnsupervisedLearningLane";
import type { DatasetView } from "@/services/datasets/client";
import {
  type AnomalyResultView,
  type ClusterResultView,
  type ReinforceRunView,
} from "@/services/ml/unsupervised";

/**
 * part18 §4.3/§4.5/§6 — the Unsupervised Learning lane (the three
 * clustering families + Isolation Forest) and the EXPERIMENTAL
 * reinforcement-learning lane as complete component-level user flows:
 *
 *   the clustering round trip — the method selects + the eps /
 *                            min-samples controls become the honest
 *                            request, and the parsed answer renders
 *                            (the honest-null diagnostics, the noise
 *                            count/ratio, the eps source, the summary
 *                            line);
 *   the anomaly round trip   — the contamination control becomes the
 *                            honest request, the parsed answer renders
 *                            (flagged rows, the feature contrast
 *                            table, the honesty note that labels
 *                            contamination-threshold verdicts as not
 *                            ground truth);
 *   the RL round trip       — the agent/episodes/seed controls become
 *                            the honest request, the parsed answer
 *                            renders with the EXPERIMENTAL label, the
 *                            reward-curve evidence and the honest
 *                            "never settled" convergence phrasing;
 *   the gating lanes        — no dataset selected never leaves the
 *                            page (validated BEFORE any request);
 *   typed errors            — a rejected request surfaces verbatim
 *                            through the alert role.
 *
 * Services are mocked at the network boundary only; the components,
 * their state machines, and the DOM are the real ones.
 */

const unsupervisedMocks = vi.hoisted(() => ({
  cluster: vi.fn(),
  anomaly: vi.fn(),
  methods: vi.fn(),
}));
const reinforceMocks = vi.hoisted(() => ({
  train: vi.fn(),
  evaluate: vi.fn(),
  methods: vi.fn(),
}));

vi.mock("@/services/ml/unsupervised", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/ml/unsupervised")>();
  return {
    ...actual,
    unsupervisedApi: {
      ...actual.unsupervisedApi,
      cluster: unsupervisedMocks.cluster,
      anomaly: unsupervisedMocks.anomaly,
      methods: unsupervisedMocks.methods,
    },
    reinforceApi: {
      ...actual.reinforceApi,
      train: reinforceMocks.train,
      evaluate: reinforceMocks.evaluate,
      methods: reinforceMocks.methods,
    },
  };
});

const DATASET: DatasetView = {
  dataset_id: "ds-test",
  name: "blobs.csv",
  source_filename: "blobs.csv",
  size_bytes: 2048,
  shape: [163, 2],
  columns: ["x", "y"],
  target: null,
  task: null,
  n_features: 2,
  feature_names: ["x", "y"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: [],
  summary: {},
} as unknown as DatasetView;

const CLUSTER_RESULT: ClusterResultView = {
  method: "dbscan",
  dataset_id: "ds-test",
  n_samples: 163,
  diagnostics: {
    n_clusters: 2,
    silhouette: 0.951,
    davies_bouldin: 0.069,
    calinski_harabasz: 21100,
    noise_count: 3,
    noise_ratio: 0.018,
    cluster_sizes: { "0": 80, "1": 80, noise: 3 },
  },
  labels: { n: 163, preview: [0, 0, 1], unique: [-1, 0, 1] },
  summary: "DBSCAN -- eps=1 (manual), min_samples=4, 163 rows",
  eps: 1.0,
  eps_source: "manual",
  min_samples: 4,
  bandwidth: null,
  bandwidth_source: null,
  n_modes: null,
  n_clusters: null,
  affinity: null,
  gamma: null,
  n_neighbors: null,
};

const ANOMALY_RESULT: AnomalyResultView = {
  method: "isolation_forest",
  dataset_id: "ds-test",
  contamination: "0.03",
  n_estimators: 200,
  seed: 42,
  n_samples: 150,
  n_features: 4,
  n_outliers: 4,
  anomaly_fraction: 0.027,
  score_threshold: -0.35,
  outlier_positions: [1, 2, 0, 82],
  scores_preview: [-0.28, -0.35],
  summary: "IsolationForest -- 200 trees, seed=42, contamination=0.03",
  honesty_note:
    "anomaly labels are contamination-threshold verdicts, not ground truth",
  feature_report: [
    {
      feature: "w",
      inlier_mean: 0.1,
      outlier_mean: 18.1,
      shift_sigma: 2.57,
    },
    {
      feature: "x",
      inlier_mean: -0.2,
      outlier_mean: -20.2,
      shift_sigma: -2.9,
    },
  ],
};

const RL_RESULT: ReinforceRunView = {
  method: "q_learning",
  env: "GridWorldEnv(rows=4, cols=5)",
  success_rate: 1.0,
  reward_curve: {
    episodes: 150,
    first_20_mean: -1.2,
    last_20_mean: 0.69,
    max: 1.0,
    min: -2.0,
  },
  convergence_episode: 95,
  hyperparams: { alpha: 0.1, gamma: 0.99, epsilon: 0.1, seed: 42 },
  policy: { "0": 0, "1": 0, "2": 2 },
  value_table_preview: [0.98, 0.96],
  notes: [],
  experimental: true,
};

describe("the Unsupervised Learning lane (part18 §4.3 + §4.5)", () => {
  beforeEach(() => {
    unsupervisedMocks.cluster.mockReset();
    unsupervisedMocks.anomaly.mockReset();
  });

  it("runs the DBSCAN round trip and renders the honest diagnostics", async () => {
    unsupervisedMocks.cluster.mockResolvedValue({ result: CLUSTER_RESULT });
    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.type(screen.getByLabelText(/eps \(blank/i), "1.0");
    // min_samples keeps its default (5): a controlled numeric input's
    // clear+type concatenates; the request asserts the default travels
    await userEvent.click(screen.getByRole("button", { name: "Fit clustering" }));
    await waitFor(() => {
      expect(screen.getByText(/DBSCAN -- eps=1 \(manual\)/)).toBeInTheDocument();
    });
    expect(screen.getByText(/noise 3 \(1\.8%\)/)).toBeInTheDocument();
    expect(screen.getByText(/silhouette 0\.951/)).toBeInTheDocument();
    const request = unsupervisedMocks.cluster.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(request).toMatchObject({
      dataset_id: "ds-test",
      method: "dbscan",
      eps: 1.0,
      min_samples: 5,
      mode: "sync",
    });
  });

  it("sends eps=null (the k-distance estimate) when the control is blank", async () => {
    unsupervisedMocks.cluster.mockResolvedValue({ result: CLUSTER_RESULT });
    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.click(screen.getByRole("button", { name: "Fit clustering" }));
    await waitFor(() => {
      expect(unsupervisedMocks.cluster).toHaveBeenCalled();
    });
    expect((unsupervisedMocks.cluster.mock.calls[0]?.[0] as Record<string, unknown>)["eps"]).toBeNull();
  });

  it("refuses to fire without a dataset (client-side gating)", async () => {
    render(<UnsupervisedLearningLane selected={null} />);
    await userEvent.click(screen.getByRole("button", { name: "Fit clustering" }));
    expect(unsupervisedMocks.cluster).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/select a dataset first/i);
  });

  it("renders the anomaly answer with the honesty note", async () => {
    unsupervisedMocks.anomaly.mockResolvedValue({ result: ANOMALY_RESULT });
    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.clear(screen.getByLabelText(/contamination/i));
    await userEvent.type(screen.getByLabelText(/contamination/i), "0.03");
    await userEvent.click(screen.getByRole("button", { name: "Detect anomalies" }));
    await waitFor(() => {
      expect(screen.getByText(/IsolationForest -- 200 trees/)).toBeInTheDocument();
    });
    expect(screen.getByText(/flagged rows \(most anomalous first\): 1, 2, 0, 82/)).toBeInTheDocument();
    expect(screen.getByText(/not ground truth/)).toBeInTheDocument();
    expect(screen.getByRole("table")).toHaveTextContent("shift (σ)");
    const request = unsupervisedMocks.anomaly.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(request).toMatchObject({
      dataset_id: "ds-test",
      contamination: 0.03,
      mode: "sync",
    });
  });

  it("surfaces a rejected request verbatim through the alert", async () => {
    unsupervisedMocks.cluster.mockRejectedValue(
      new Error("eps must be positive, got -1.0"),
    );
    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.type(screen.getByLabelText(/eps \(blank/i), "-1.0");
    await userEvent.click(screen.getByRole("button", { name: "Fit clustering" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "eps must be positive, got -1.0",
    );
  });
});

describe("the EXPERIMENTAL Reinforcement Learning lane (part18 §6)", () => {
  beforeEach(() => {
    reinforceMocks.train.mockReset();
  });

  it("labels itself experimental and never claims pharmaceutical validity", () => {
    render(<ReinforceLane />);
    expect(screen.getByText(/Reinforcement Learning — experimental/i)).toBeInTheDocument();
    expect(screen.getByText(/NOT supervised registry entries and NOT pharmaceutical models/i)).toBeInTheDocument();
  });

  it("runs the training round trip and renders the curve evidence", async () => {
    reinforceMocks.train.mockResolvedValue({ result: RL_RESULT });
    render(<ReinforceLane />);
    // episodes/seed keep their defaults (150/42): the request asserts
    // the defaults travel honestly to the backend
    await userEvent.click(screen.getByRole("button", { name: "Train agent" }));
    await waitFor(() => {
      expect(screen.getByText(/success rate 100\.0%/)).toBeInTheDocument();
    });
    expect(screen.getByText(/convergence episode 95/)).toBeInTheDocument();
    expect(screen.getByText(/first-20 mean -1\.200/)).toBeInTheDocument();
    expect(screen.getByText(/last-20 mean 0\.690/)).toBeInTheDocument();
    const request = reinforceMocks.train.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(request).toMatchObject({
      method: "q_learning",
      episodes: 150,
      seed: 42,
      mode: "sync",
    });
  });

  it("renders the honest never-settled convergence wording", async () => {
    reinforceMocks.train.mockResolvedValue({
      result: { ...RL_RESULT, convergence_episode: null, success_rate: 0.42 },
    });
    render(<ReinforceLane />);
    await userEvent.click(screen.getByRole("button", { name: "Train agent" }));
    await waitFor(() => {
      expect(screen.getByText(/never settled \(reported honestly\)/)).toBeInTheDocument();
    });
    expect(screen.getByText(/success rate 42\.0%/)).toBeInTheDocument();
  });

  it("surfaces a typed backend error verbatim", async () => {
    reinforceMocks.train.mockRejectedValue(
      new Error("episodes must be 1..10000, got 0"),
    );
    render(<ReinforceLane />);
    await userEvent.click(screen.getByRole("button", { name: "Train agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "episodes must be 1..10000, got 0",
    );
  });
});
