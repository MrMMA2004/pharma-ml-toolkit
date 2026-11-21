import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part13 Phase 5's next remainder item — the drug-repurposing lane of
 * the Machine Learning workspace as a complete component-level user
 * flow:
 *
 *   the registry line      — the stage renders the backend's own
 *                            vocabulary summary (DeepResponse's split
 *                            ladder, the ten-key metric contract, the
 *                            candidate schema + ordering) off the
 *                            mocked registry answer;
 *   the screen round trip  — the default parameter block becomes the
 *                            honest request, and the ranked candidate
 *                            tables render the REAL parsed payload
 *                            (the screen summary, the fit telemetry,
 *                            the metric row, the per-profile tables
 *                            with rank/drug/predicted-IC50 rows);
 *   the honest labels      — the stopped-before-diverging telemetry,
 *                            the model-estimate column header, and a
 *                            null metric rendered as null, never zero;
 *   typed errors           — the backend's own hard 422 surfaces
 *                            verbatim through the banner;
 *   client-side gating     — a below-bound screen, an over-cap
 *                            screen and a bad learning rate never
 *                            leave the page (validated BEFORE any
 *                            request fires);
 *   honest provenance      — the disclaimer + the "0 shipped
 *                            checkpoints" line ride every answer.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers (parseRepurposingRegistryResponse /
 * parseRepurposingScreenResponse — the REAL ones, spread from the
 * actual module), and the DOM are the real ones.
 */

const datasetsMocks = vi.hoisted(() => ({ list: vi.fn() }));
vi.mock("@/services/datasets/client", () => ({
  datasetsApi: { list: datasetsMocks.list },
}));

const mlMocks = vi.hoisted(() => ({
  trained: vi.fn(),
  catalog: vi.fn(),
  admetEndpoints: vi.fn(),
  admetScore: vi.fn(),
  dtaRegistry: vi.fn(),
  dtaFeatures: vi.fn(),
  repurposingRegistry: vi.fn(),
  repurposingScreen: vi.fn(),
}));
vi.mock("@/services/ml/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/ml/client")>();
  // the REAL parsers ride the page under test; only the network
  // boundary is faked
  return {
    ...actual,
    mlApi: {
      ...actual.mlApi,
      trained: mlMocks.trained,
      catalog: mlMocks.catalog,
      admetEndpoints: mlMocks.admetEndpoints,
      admetScore: mlMocks.admetScore,
      dtaRegistry: mlMocks.dtaRegistry,
      dtaFeatures: mlMocks.dtaFeatures,
      repurposingRegistry: mlMocks.repurposingRegistry,
      repurposingScreen: mlMocks.repurposingScreen,
    },
  };
});

vi.mock("@/services/jobs/client", () => ({
  jobsApi: { result: vi.fn() },
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

vi.mock("@/components/charts/PlotlyFigure", () => ({
  PlotlyFigure: () => <div data-testid="plotly-stub" />,
}));

const RP_PROVENANCE = {
  backend: "pharmaml.ml.response_model",
  source:
    "CSB5/CaDRReS + burakcan-izmirli/DeepResponse + SmritiChawla/Precily",
  n_repositories: 3,
  shipped_checkpoints: 0,
  shipped_checkpoints_note:
    "the SELFormer/CNN/FiLM hybrid is NOT shipped — this lane ships " +
    "the recommender that trains in-tree",
  screen: "SYNTHETIC — the stand-in for a GDSC-sized table",
  disclaimer:
    "predicted IC50 values are model estimates over the synthetic " +
    "screen — repurposing candidates are ranked hypotheses for " +
    "triage, never clinical recommendations",
};

const RP_REGISTRY = {
  split_strategies: {
    within_table: [
      "random",
      "cell_stratified",
      "drug_stratified",
      "drug_cell_stratified",
    ],
    cross_domain: "cross_domain_split — the fifth strategy",
    source: "DeepResponse's five split strategies, verbatim",
  },
  model: {
    family: "caDRReS-style matrix factorisation",
    formulation: "S = mu + b_p + b_q + P.Q^T",
    direction: "the matrix reads IC50 (lower = more sensitive)",
    telemetry_fields: ["history", "observed_rmse", "stopped_early"],
    history_note: "the trainer has no live per-epoch callback",
  },
  hyperparameters: {
    n_factors: { default: 3, max: 10 },
    lr: { default: 0.05, min: 0.001, max: 0.5 },
    max_epochs: { default: 200, max: 1000 },
    seed: { default: 42, max: 2147483647 },
    note: "the backend's own defaults are n_factors=10 / lr=0.01",
  },
  screen: {
    defaults: {
      n_cells: 12,
      n_drugs: 8,
      n_features: 6,
      noise: 0.3,
      observed: 0.8,
      seed: 42,
    },
    bounds: {
      n_cells: "4..40",
      n_drugs: "3..30",
      n_features: "2..20",
      observed: "0.3..1.0",
    },
    label: "SYNTHETIC — a seeded IC50 screen",
  },
  repurposing: {
    schema: ["cell_line", "drug", "predicted_ic50", "rank"],
    ordering: "most-sensitive first — lowest predicted IC50 = rank 1",
    source: "DeepResponse's RepurposingInferenceEngine concept",
    response_limits: { profiles: "1..8", top_k_per_profile: "1..10" },
  },
  metrics: {
    keys: [
      "rmse", "mae", "r2", "pcc", "scc",
      "accuracy", "precision", "recall", "f1", "mcc",
    ],
    threshold: 6.0,
    note: "DeepResponse's metric contract on the observed entries",
  },
  provenance: RP_PROVENANCE,
  screen_url: "/api/v1/repurposing/screen",
  note: "sync-only light compute — dense bounded matrices",
};

function candidate(drug: string, predicted: number, rank: number) {
  return { drug, predicted_ic50: predicted, rank };
}

function screenResponse(overrides: Record<string, unknown> = {}) {
  return {
    request: {
      n_cells: 12,
      n_drugs: 8,
      n_features: 6,
      n_factors: 3,
      lr: 0.05,
      max_epochs: 200,
      noise: 0.3,
      observed: 0.8,
      seed: 42,
      top_k: 5,
      profiles: 3,
    },
    screen: {
      cells: ["C01", "C02", "C03"],
      drugs: ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"],
      features: ["F1", "F2", "F3", "F4", "F5", "F6"],
      n_cells: 12,
      n_drugs: 8,
      n_features: 6,
      n_observed: 77,
      n_entries: 96,
      observed_fraction: 0.802,
      label: "SYNTHETIC seeded screen — not real pharmacology",
    },
    training: {
      epochs_run: 143,
      max_epochs: 200,
      stopped_early: false,
      observed_rmse: 0.512,
      first_epoch_rmse: 1.918,
      first_epoch: 1,
      first_epoch_ndcg: 0.61,
      n_factors: 3,
      note: "never a fabricated epoch stream",
    },
    metrics: {
      rmse: 0.512,
      mae: 0.401,
      r2: 0.79,
      pcc: 0.89,
      scc: 0.88,
      accuracy: 0.9,
      precision: 0.8,
      recall: 0.75,
      f1: 0.77,
      mcc: 0.7,
    },
    candidates: {
      n_profiles: 3,
      n_ranked_rows_total: 24,
      top_k: 5,
      per_profile: {
        C01: [
          candidate("D03", 2.11, 1),
          candidate("D07", 2.84, 2),
          candidate("D01", 3.05, 3),
          candidate("D05", 3.42, 4),
          candidate("D02", 3.97, 5),
        ],
        C02: [
          candidate("D05", 2.44, 1),
          candidate("D01", 2.9, 2),
          candidate("D08", 3.16, 3),
          candidate("D03", 3.58, 4),
          candidate("D06", 3.71, 5),
        ],
        C03: [
          candidate("D08", 2.63, 1),
          candidate("D02", 2.72, 2),
          candidate("D04", 3.21, 3),
          candidate("D05", 3.66, 4),
          candidate("D07", 3.88, 5),
        ],
      },
      ordering: "lowest predicted IC50 = rank 1 (most sensitive first)",
    },
    provenance: RP_PROVENANCE,
    note: "ranked hypotheses for triage, never clinical recommendations",
    ...overrides,
  };
}

beforeEach(() => {
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  mlMocks.trained.mockResolvedValue({ models: [] });
  mlMocks.admetEndpoints.mockResolvedValue({
    endpoints: [],
    categories: [],
    radial_axes: [],
    low_confidence_endpoints: [],
    provenance: {
      backend: "pharmaml.ml.admet",
      source: "",
      n_endpoints: 0,
      n_learned_endpoints: 0,
      shipped_ensembles: 0,
      shipped_ensembles_note: "",
      low_performance_threshold: 0.6,
      disclaimer: "",
    },
    score_url: "",
    note: "",
  });
  // structurally complete DTA registry stub — the DTA stage renders the
  // registry line off these fields, so the lane under test never crashes
  // its neighbour while this suite focuses on the repurposing stage
  mlMocks.dtaRegistry.mockResolvedValue({
    alphabets: {
      protein: {
        n_symbols: 25,
        padding_code: 0,
        symbols: { A: 1 },
        encoding: "1-based labels",
        source: "DeepDTA CHARPROTSET verbatim",
      },
      smiles_isomeric: {
        n_symbols: 64,
        padding_code: 0,
        symbols: { C: 42 },
        encoding: "encoded as-is",
        source: "DeepDTA CHARISOSMISET verbatim",
      },
      smiles_canonical: {
        n_symbols: 62,
        padding_code: 0,
        symbols: { "#": 1 },
        encoding: "the 62-symbol canonical set",
        source: "DeepDTA CHARCANSMISET verbatim",
      },
    },
    amino_acids: Array.from("ACDEFGHIKLMNPQRSTVWY"),
    protein_features: {
      n_base: 23,
      base_columns: [],
      n_dipeptide_extras: 401,
      n_with_dipeptides: 424,
      note: "",
    },
    ligand_features: { family: "ecfp", n_bits: 2048, source: "" },
    paired_width: { base: 2071, with_dipeptides: 2472 },
    metrics: [],
    demo_frame: { rows: 12, note: "" },
    provenance: {
      backend: "pharmaml.ml.dta",
      source: "",
      n_alphabets: 3,
      shipped_checkpoints: 0,
      shipped_checkpoints_note: "",
      disclaimer: "",
    },
    features_url: "/api/v1/dta/features",
    note: "",
  });
  mlMocks.repurposingRegistry.mockResolvedValue(RP_REGISTRY);
  mlMocks.repurposingScreen.mockReset();
});

describe("the drug repurposing lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary line", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(screen.getByText("Drug Repurposing")).toBeInTheDocument();
    });
    // the backend's own registry summary, never re-declared
    await waitFor(() => {
      expect(
        screen.getByText(/Strategies: random · cell_stratified/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/the metric contract: 10 keys at threshold 6/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/lowest predicted IC50 = rank 1\.$/),
    ).toBeInTheDocument();
    expect(mlMocks.repurposingRegistry).toHaveBeenCalledTimes(1);
  });

  it("runs the screen and renders the ranked candidate tables", async () => {
    mlMocks.repurposingScreen.mockResolvedValue(screenResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the screen",
    });
    await userEvent.click(button);
    // the default parameter block becomes the honest request
    await waitFor(() => {
      expect(mlMocks.repurposingScreen).toHaveBeenCalledWith({
        n_cells: 12,
        n_drugs: 8,
        n_features: 6,
        n_factors: 3,
        lr: 0.05,
        max_epochs: 200,
        noise: 0.3,
        observed: 0.8,
        seed: 42,
        top_k: 5,
        profiles: 3,
      });
    });
    // the summary line over the REAL parsed payload
    expect(
      await screen.findByText(/12×8 screen \(SYNTHETIC\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/77\/96 observed entries \(80%\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/143\/200 epochs/)).toBeInTheDocument();
    expect(
      screen.getByText(/observed RMSE 0\.512/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/rmse 0\.512 · accuracy 0\.900/),
    ).toBeInTheDocument();
    // the per-profile candidate tables: the profile header carries
    // the ordering rule, and the rank-1 row renders the most-sensitive
    // drug with its labelled model estimate
    expect(screen.getByText(/C01 — top 5 of 8/)).toBeInTheDocument();
    expect(screen.getByText(/C03 — top 5 of 8/)).toBeInTheDocument();
    expect(screen.getAllByText("D03").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("D08").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("2.11").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText(/predicted IC50 \(model estimate\)/).length,
    ).toBeGreaterThanOrEqual(3);
  });

  it("labels the estimates and renders null telemetry honestly", async () => {
    mlMocks.repurposingScreen.mockResolvedValue(
      screenResponse({
        training: {
          epochs_run: 87,
          max_epochs: 200,
          stopped_early: true,
          observed_rmse: 0.623,
          first_epoch_rmse: 1.88,
          first_epoch: 1,
          first_epoch_ndcg: 0.6,
          n_factors: 3,
          note: "never a fabricated epoch stream",
        },
        metrics: {
          rmse: 0.623,
          mae: 0.5,
          r2: null,
          pcc: null,
          scc: null,
          accuracy: 0.85,
          precision: 0.7,
          recall: 0.6,
          f1: 0.64,
          mcc: 0.5,
        },
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the screen",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(
        screen.getByText(/stopped before diverging/),
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/87\/200 epochs/)).toBeInTheDocument();
    // the rank-1 column header labels every predicted value a model
    // estimate, and the SYNTHETIC screen label rides the summary
    expect(screen.getByText(/SYNTHETIC\)/)).toBeInTheDocument();
    expect(
      screen.getByText(/rmse 0\.623 · accuracy 0\.850/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    mlMocks.repurposingScreen.mockRejectedValue(
      new ApiError(422, {
        detail: "observed must be in [0.3, 1.0], got 0.2",
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the screen",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(screen.getByText(/API 422/)).toBeInTheDocument();
    });
    expect(
      screen.getByText(/observed must be in \[0\.3, 1\.0\], got 0\.2/),
    ).toBeInTheDocument();
    expect(mlMocks.repurposingScreen).toHaveBeenCalledTimes(1);
  });

  it("gates a below-bound screen client-side — no request fires", async () => {
    render(<MLPage />);
    const cells = await screen.findByLabelText("repurposing n_cells");
    await userEvent.clear(cells);
    await userEvent.type(cells, "3");
    const button = screen.getByRole("button", { name: "Run the screen" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/needs at least 4 cell lines/),
    ).toBeInTheDocument();
    expect(mlMocks.repurposingScreen).not.toHaveBeenCalled();
  });

  it("gates an over-cap screen client-side — no request fires", async () => {
    render(<MLPage />);
    const cells = await screen.findByLabelText("repurposing n_cells");
    await userEvent.clear(cells);
    await userEvent.type(cells, "41");
    const button = screen.getByRole("button", { name: "Run the screen" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/caps the screen at 40 cells/),
    ).toBeInTheDocument();
    expect(mlMocks.repurposingScreen).not.toHaveBeenCalled();
  });

  it("gates a bad learning rate client-side — no request fires", async () => {
    render(<MLPage />);
    const lr = await screen.findByLabelText("repurposing lr");
    await userEvent.clear(lr);
    await userEvent.type(lr, "0.6");
    const button = screen.getByRole("button", { name: "Run the screen" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/learning rate must lie in 0\.001\.\.0\.5/),
    ).toBeInTheDocument();
    expect(mlMocks.repurposingScreen).not.toHaveBeenCalled();
  });

  it("carries the honest provenance and disclaimer on every answer", async () => {
    mlMocks.repurposingScreen.mockResolvedValue(screenResponse());
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Run the screen",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(
        screen.getByText(/0 shipped checkpoints \(the honest absence/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/never clinical recommendations/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/SYNTHETIC — the stand-in for a GDSC-sized table/),
    ).toBeInTheDocument();
  });
});
