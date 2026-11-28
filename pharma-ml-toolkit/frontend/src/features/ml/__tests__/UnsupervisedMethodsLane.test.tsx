import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { UnsupervisedLearningLane } from "@/features/ml/UnsupervisedLearningLane";
import type { DatasetView } from "@/services/datasets/client";
import {
  type ClusterResultView,
  type PcaResultView,
} from "@/services/ml/unsupervised";

/**
 * ML-4 (the WORKFLOW_ENHANCEMENT_ROADMAP's row 11) — the remaining
 * unsupervised families as complete component-level user flows:
 *
 *   the chips        — the method select carries the six methods
 *                      (dbscan / meanshift / spectral / kmeans / gmm /
 *                      hierarchical), the engine list mirrored;
 *   the kmeans lane  — blank k POSTs null (the engine's own silhouette
 *                      scan); a manual k POSTs verbatim; the parsed
 *                      answer renders the k/inertia/silhouette line +
 *                      the elbow/silhouette scan table;
 *   the gmm lane     — the covariance/criterion selects ride the body;
 *                      the parsed answer renders the components line +
 *                      the AIC/BIC table + the converged verdict;
 *   the hierarchical
 *                    — the linkage select + the n_clusters ride the
 *                      body; the parsed answer renders the merge-height
 *                      line;
 *   the pca sibling  — the components/scale controls ride the body; the
 *                      parsed answer renders the explained-variance
 *                      ladder + the loadings preview + the honesty note;
 *   the degenerate   — a single-cluster answer renders its honest
 *                      nulls ("n/a"), never a fabricated score.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers, and the DOM are the real
 * ones.
 */

const unsupervisedMocks = vi.hoisted(() => ({
  cluster: vi.fn(),
  anomaly: vi.fn(),
  pca: vi.fn(),
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
      pca: unsupervisedMocks.pca,
      methods: unsupervisedMocks.methods,
    },
  };
});

const DATASET: DatasetView = {
  dataset_id: "ds-blobs",
  name: "blobs.csv",
  source_filename: "blobs.csv",
  size_bytes: 2048,
  shape: [160, 3],
  columns: ["x", "y", "target"],
  target: "target",
  task: "regression",
  n_features: 2,
  feature_names: ["x", "y"],
  classes: null,
  class_counts: null,
  dropped_columns: [],
  categorical_columns: [],
  summary: {},
  created_at: "2026-11-28T00:00:00Z",
} as unknown as DatasetView;

function clusterAnswer(overrides: Record<string, unknown> = {}) {
  return {
    result: {
      method: "kmeans",
      dataset_id: "ds-blobs",
      n_samples: 160,
      diagnostics: {
        n_clusters: 2,
        silhouette: 0.94,
        davies_bouldin: 0.07,
        calinski_harabasz: 21000,
        noise_count: 0,
        noise_ratio: 0.0,
        cluster_sizes: { "0": 80, "1": 80 },
      },
      labels: { n: 160, preview: [0, 0, 1], unique: [0, 1] },
      summary: "k-means (kmeans) -- k=2 (best silhouette 0.941 in scan), 160 rows",
      eps: null,
      eps_source: null,
      min_samples: null,
      bandwidth: null,
      bandwidth_source: null,
      n_modes: null,
      n_clusters: null,
      affinity: null,
      gamma: null,
      n_neighbors: null,
      k: 2,
      k_source: "the silhouette scan over the engine's own k range",
      kmeans_method: "kmeans",
      inertia: 10.58,
      kmeans_silhouette: 0.941,
      scan: [
        { k: 2, inertia: 10.58, silhouette: 0.941 },
        { k: 3, inertia: 6.61, silhouette: 0.752 },
      ],
      sizes: { "2": 80, "3": 80 },
      n_components: null,
      covariance_type: null,
      chosen_by: null,
      criterion: null,
      converged: false,
      log_likelihood: null,
      weights: null,
      bic_table: [],
      linkage: null,
      merge_heights: [],
      ...overrides,
    } as ClusterResultView,
  };
}

const PCA_RESULT: PcaResultView = {
  method: "pca",
  dataset_id: "ds-blobs",
  n_components: 2,
  scaled: true,
  n_samples: 160,
  n_features: 2,
  explained_variance_ratio: [0.71, 0.29],
  cumulative_variance: [0.71, 1.0],
  total_variance_explained: 1.0,
  column_names: ["x", "y"],
  loadings_preview: [
    { "": "x", PC1: 0.707, PC2: -0.707 },
    { "": "y", PC1: 0.707, PC2: 0.707 },
  ],
  scores_preview: [],
  summary: "PCA(2 components over 2 features, 160 rows, scaled=True)",
  honesty_note:
    "the components are the axes of maximum retained variance, not causal directions",
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("the ML-4 unsupervised families (kmeans / gmm / hierarchical + the pca sibling)", () => {
  it("carries the six clustering methods in the select — the engine list mirrored", () => {
    render(<UnsupervisedLearningLane selected={DATASET} />);

    const select = screen.getByLabelText("the clustering method");
    for (const option of [
      "DBSCAN",
      "Mean Shift",
      "Spectral",
      "k-means",
      "Gaussian Mixture",
      "Hierarchical",
    ]) {
      expect(
        screen.getByRole("option", { name: option }),
      ).toBeInTheDocument();
    }
    expect(select).toHaveValue("dbscan");
  });

  it("POSTs kmeans with a null k (blank = the engine's own scan) and renders the scan table", async () => {
    unsupervisedMocks.cluster.mockResolvedValueOnce(clusterAnswer());

    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.selectOptions(
      screen.getByLabelText("the clustering method"),
      "kmeans",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Fit clustering" }),
    );

    await waitFor(() => {
      expect(unsupervisedMocks.cluster).toHaveBeenCalledTimes(1);
    });
    const body = unsupervisedMocks.cluster.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(body["method"]).toBe("kmeans");
    expect(body["k"]).toBeNull();
    expect(body["mode"]).toBe("sync");

    // the parsed answer renders: the k/inertia/silhouette line + the
    // elbow/silhouette scan table
    expect(
      await screen.findByText(/k 2 \(the silhouette scan over the engine's own k range\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/inertia 10\.580/)).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: "silhouette" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "10.580" })).toBeInTheDocument();
  });

  it("POSTs a manual k verbatim when the k field carries one", async () => {
    unsupervisedMocks.cluster.mockResolvedValueOnce(
      clusterAnswer({
        k: 3,
        k_source: "manual",
        scan: [],
        summary: "k-means (kmeans) -- k=3 (manual), 160 rows",
      }),
    );

    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.selectOptions(
      screen.getByLabelText("the clustering method"),
      "kmeans",
    );
    await userEvent.type(screen.getByLabelText("the k-means k"), "3");
    await userEvent.click(
      screen.getByRole("button", { name: "Fit clustering" }),
    );

    const body = unsupervisedMocks.cluster.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(body["k"]).toBe(3);
    expect(
      await screen.findByText(/k 3 \(manual\)/),
    ).toBeInTheDocument();
  });

  it("POSTs gmm with the covariance/criterion selects and renders the AIC/BIC table", async () => {
    unsupervisedMocks.cluster.mockResolvedValueOnce(
      clusterAnswer({
        method: "gmm",
        summary:
          "GaussianMixture(2 components, covariance=full, chosen_by=bic)",
        k: null,
        k_source: null,
        kmeans_method: null,
        inertia: null,
        kmeans_silhouette: null,
        scan: [],
        sizes: null,
        n_components: 2,
        covariance_type: "full",
        chosen_by: "bic",
        criterion: "bic",
        converged: true,
        log_likelihood: -248.2,
        weights: { "0": 0.5, "1": 0.5 },
        bic_table: [
          { n_components: 1, aic: 808.0, bic: 814.2 },
          { n_components: 2, aic: 248.2, bic: 263.6 },
        ],
      }),
    );

    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.selectOptions(
      screen.getByLabelText("the clustering method"),
      "gmm",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Fit clustering" }),
    );

    await waitFor(() => {
      expect(unsupervisedMocks.cluster).toHaveBeenCalledTimes(1);
    });
    const body = unsupervisedMocks.cluster.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(body["method"]).toBe("gmm");
    expect(body["n_components"]).toBeNull();
    expect(body["covariance_type"]).toBe("full");
    expect(body["criterion"]).toBe("bic");

    // the parsed answer renders: the components line, the AIC/BIC
    // table, the converged verdict, the mixing weights
    expect(
      await screen.findByText(/2 components \(bic · full\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/EM converged/)).toBeInTheDocument();
    expect(
      screen.getByText(/mixing weights: 0 0\.50 · 1 0\.50/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: "bic" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "263.6" })).toBeInTheDocument();
  });

  it("POSTs hierarchical with the linkage select and renders the merge-height line", async () => {
    unsupervisedMocks.cluster.mockResolvedValueOnce(
      clusterAnswer({
        method: "hierarchical",
        summary: "hierarchical (ward) k=3 over 160 rows",
        k: 3,
        k_source: "manual",
        inertia: 20.9,
        linkage: "ward",
        merge_heights: [41.2, 88.4],
      }),
    );

    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.selectOptions(
      screen.getByLabelText("the clustering method"),
      "hierarchical",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Fit clustering" }),
    );

    await waitFor(() => {
      expect(unsupervisedMocks.cluster).toHaveBeenCalledTimes(1);
    });
    const body = unsupervisedMocks.cluster.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(body["method"]).toBe("hierarchical");
    expect(body["n_clusters"]).toBe(3);
    expect(body["linkage"]).toBe("ward");

    expect(
      await screen.findByText(/linkage ward · inertia 20\.900/),
    ).toBeInTheDocument();
    expect(screen.getByText(/41\.20, 88\.40/)).toBeInTheDocument();
  });

  it("POSTs the pca sibling with the engine-cap default and renders the variance ladder", async () => {
    unsupervisedMocks.pca.mockResolvedValueOnce({ result: PCA_RESULT });

    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.click(
      screen.getByRole("button", { name: "Reduce (PCA)" }),
    );

    await waitFor(() => {
      expect(unsupervisedMocks.pca).toHaveBeenCalledTimes(1);
    });
    const body = unsupervisedMocks.pca.mock.calls[0]?.[0] as Record<
      string,
      unknown
    >;
    expect(body["dataset_id"]).toBe("ds-blobs");
    expect(body["n_components"]).toBeNull();
    expect(body["scale"]).toBe(true);

    // the parsed answer renders: the ladder, the loadings preview,
    // the honesty note
    expect(
      (await screen.findAllByText(/71\.0%/)).length,
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/100\.0%/).length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getByText(
        /the components are the axes of maximum retained variance, not causal directions/,
      ),
    ).toBeInTheDocument();
  });

  it("renders a degenerate single-cluster answer with its honest nulls — never a fabricated score", async () => {
    unsupervisedMocks.cluster.mockResolvedValueOnce(
      clusterAnswer({
        summary: "k-means (kmeans) -- k=1 (manual), 160 rows",
        k: 1,
        k_source: "manual",
        inertia: 0.0,
        kmeans_silhouette: null,
        scan: [],
        diagnostics: {
          n_clusters: 1,
          silhouette: null,
          davies_bouldin: null,
          calinski_harabasz: null,
          noise_count: 0,
          noise_ratio: 0.0,
          cluster_sizes: { "0": 160 },
        },
      }),
    );

    render(<UnsupervisedLearningLane selected={DATASET} />);
    await userEvent.selectOptions(
      screen.getByLabelText("the clustering method"),
      "kmeans",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Fit clustering" }),
    );

    // the honest nulls surface as n/a — the single-cluster lane never
    // fabricates a similarity score (both the diagnostics line and
    // the k-means line carry the honest n/a)
    expect(
      (await screen.findAllByText(/n\/a/)).length,
    ).toBeGreaterThanOrEqual(3);
    expect(
      screen.getAllByText(/silhouette n\/a/).length,
    ).toBeGreaterThanOrEqual(1);
  });
});
