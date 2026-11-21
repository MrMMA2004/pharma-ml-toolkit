import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part13 Phase 5's next remainder item — the ADMET lane of the
 * Machine Learning workspace as a complete component-level user flow:
 *
 *   the registry line    — the card renders the backend's own
 *                          vocabulary summary (52 endpoints, the
 *                          six categories, the low-confidence
 *                          count) off the mocked registry answer;
 *   the score submission — the textarea's SMILES lines become the
 *                          honest request (aspirin + caffeine), and
 *                          the per-molecule cards render the REAL
 *                          parsed payload (the physchem tables, the
 *                          radial bars, the valid badges);
 *   the invalid verdict  — an invalid SMILES row reports itself
 *                          per-row (never dropped, never silent);
 *   typed errors         — the backend's own 422 surfaces verbatim
 *                          through the error banner;
 *   client-side gating   — an empty batch and an over-cap batch
 *                          never leave the page (validated BEFORE
 *                          any request fires);
 *   honest provenance    — the disclaimer + the "0 shipped
 *                          ensembles" line ride every answer.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers (parseAdmetEndpointsResponse
 * / parseAdmetScoreResponse — the REAL ones, spread from the actual
 * module), and the DOM are the real ones.
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

const ADMET_REGISTRY = {
  endpoints: Array.from({ length: 52 }, (_, i) => ({
    id: `endpoint_${i}`,
    category: [
      "Physicochemical",
      "Absorption",
      "Distribution",
      "Excretion",
      "Metabolism",
      "Toxicity",
    ][i % 6],
    task_type: "classification",
    units: "-",
    minimum: 0.0,
    maximum: 1.0,
    species: "human",
    size: 1000 + i,
    quality: i % 7 === 0 ? 0.5 : 0.9,
    low_confidence: i % 7 === 0,
  })),
  categories: [
    "Physicochemical",
    "Absorption",
    "Distribution",
    "Excretion",
    "Metabolism",
    "Toxicity",
  ],
  radial_axes: [
    { endpoint: "BBB_Martins", label: "Blood-Brain Barrier Safe", invert: true },
    { endpoint: "ClinTox", label: "Non-Toxic", invert: true },
    { endpoint: "Solubility_AqSolDB", label: "Soluble", invert: false },
    { endpoint: "Bioavailability_Ma", label: "Bioavailable", invert: false },
    { endpoint: "hERG", label: "hERG Safe", invert: true },
  ],
  low_confidence_endpoints: ["endpoint_0", "endpoint_7", "endpoint_14"],
  provenance: {
    backend: "pharmaml.ml.admet",
    source: "swansonk14/admet_ai v2.0.1",
    n_endpoints: 52,
    n_learned_endpoints: 41,
    shipped_ensembles: 0,
    shipped_ensembles_note:
      "the 41 TDC-trained chemprop ensembles are NOT shipped (torch checkpoints)",
    low_performance_threshold: 0.6,
    disclaimer:
      "predicted ADMET properties are model estimates, not experimentally established facts",
  },
  score_url: "/api/v1/admet/score",
  note: "sync-only light compute",
};

const ASPIRIN_CARD = {
  smiles: "CC(=O)Oc1ccccc1C(=O)O",
  valid: true,
  physchem: {
    molecular_weight: 180.157,
    logP: 1.31,
    hydrogen_bond_acceptors: 3.0,
    hydrogen_bond_donors: 1.0,
    Lipinski: 4.0,
    QED: 0.855,
    stereo_centers: 0.0,
    tpsa: 63.6,
    PAINS_alert: 0.0,
    BRENK_alert: 0.0,
    NIH_alert: 0.0,
  },
  radial: {
    "Blood-Brain Barrier Safe": 88.9,
    "Non-Toxic": 14.5,
    Soluble: 27.3,
    Bioavailable: 85.5,
    "hERG Safe": 84.1,
  },
  low_confidence_endpoints: ["endpoint_0", "endpoint_7", "endpoint_14"],
};

const CAFFEINE_CARD = {
  smiles: "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
  valid: true,
  physchem: {
    molecular_weight: 194.195,
    logP: 0.07,
    hydrogen_bond_acceptors: 5.0,
    hydrogen_bond_donors: 0.0,
    Lipinski: 4.0,
    QED: 0.773,
    stereo_centers: 0.0,
    tpsa: 61.8,
    PAINS_alert: 0.0,
    BRENK_alert: 0.0,
    NIH_alert: 0.0,
  },
  radial: {
    "Blood-Brain Barrier Safe": 78.4,
    "Non-Toxic": 22.7,
    Soluble: 45.9,
    Bioavailable: 77.3,
    "hERG Safe": 80.2,
  },
  low_confidence_endpoints: ["endpoint_0", "endpoint_7", "endpoint_14"],
};

function scoreResponse(molecules: unknown[]) {
  const nValid = molecules.filter(
    (m) => (m as { valid: boolean }).valid === true,
  ).length;
  return {
    n_smiles: molecules.length,
    n_valid: nValid,
    n_invalid: molecules.length - nValid,
    molecules,
    provenance: ADMET_REGISTRY.provenance,
    note: "the radial axes use the offline physchem stand-ins",
  };
}

beforeEach(() => {
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  mlMocks.trained.mockResolvedValue({ models: [] });
  mlMocks.admetEndpoints.mockResolvedValue(ADMET_REGISTRY);
  mlMocks.admetScore.mockReset();
});

describe("the ADMET lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary line", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(screen.getByText("ADMET Profile")).toBeInTheDocument();
    });
    // the backend's own registry summary, never re-declared
    await waitFor(() => {
      expect(
        screen.getByText(/Registry: 52 endpoints across 6 categories/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/3 endpoints below the vendor's 0\.6 quality/),
    ).toBeInTheDocument();
    expect(mlMocks.admetEndpoints).toHaveBeenCalledTimes(1);
  });

  it("scores the textarea's SMILES and renders the per-molecule cards", async () => {
    mlMocks.admetScore.mockResolvedValue(
      scoreResponse([ASPIRIN_CARD, CAFFEINE_CARD]),
    );
    render(<MLPage />);
    const scoreButton = await screen.findByRole("button", {
      name: "Score ADMET",
    });
    await userEvent.click(scoreButton);
    // the default textarea (aspirin + caffeine, one per line)
    await waitFor(() => {
      expect(mlMocks.admetScore).toHaveBeenCalledWith([
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
      ]);
    });
    // the REAL parsed payload renders: the physchem rows (both
    // molecules' cards — the shared labels appear once per card)
    expect(await screen.findAllByText("MW (Da)")).toHaveLength(2);
    expect(screen.getAllByText("Crippen LogP")).toHaveLength(2);
    // the radial axis bars (the backend's own labels as keys)
    expect(screen.getAllByText("Blood-Brain Barrier Safe")).toHaveLength(2);
    expect(screen.getAllByText("hERG Safe")).toHaveLength(2);
    // the verdict line
    expect(screen.getByText(/2 valid · 0 invalid/)).toBeInTheDocument();
    expect(
      screen.getAllByText("CC(=O)Oc1ccccc1C(=O)O").length,
    ).toBeGreaterThanOrEqual(1);
  });

  it("reports an invalid SMILES row per-row, never silently", async () => {
    mlMocks.admetScore.mockResolvedValue(
      scoreResponse([
        ASPIRIN_CARD,
        {
          smiles: "not-a-molecule",
          valid: false,
          error: "invalid SMILES: 'not-a-molecule'",
        },
      ]),
    );
    render(<MLPage />);
    const scoreButton = await screen.findByRole("button", {
      name: "Score ADMET",
    });
    await userEvent.click(scoreButton);
    expect(await screen.findByText(/1 valid · 1 invalid/)).toBeInTheDocument();
    expect(
      screen.getByText("invalid SMILES — reported per-row"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/invalid SMILES: 'not-a-molecule'/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own 422 verbatim through the banner", async () => {
    mlMocks.admetScore.mockRejectedValue(
      new ApiError(422, {
        detail: "Every SMILES must be a non-empty string.",
      }),
    );
    render(<MLPage />);
    const scoreButton = await screen.findByRole("button", {
      name: "Score ADMET",
    });
    await userEvent.click(scoreButton);
    await waitFor(() => {
      expect(screen.getByText(/API 422/)).toBeInTheDocument();
    });
    expect(mlMocks.admetScore).toHaveBeenCalledTimes(1);
  });

  it("gates an empty batch client-side — no request fires", async () => {
    render(<MLPage />);
    const textarea = await screen.findByLabelText("ADMET SMILES batch");
    await userEvent.clear(textarea);
    const scoreButton = screen.getByRole("button", { name: "Score ADMET" });
    await userEvent.click(scoreButton);
    expect(
      await screen.findByText(/needs at least one SMILES line/),
    ).toBeInTheDocument();
    expect(mlMocks.admetScore).not.toHaveBeenCalled();
  });

  it("gates an over-cap batch client-side — no request fires", async () => {
    render(<MLPage />);
    const textarea = await screen.findByLabelText("ADMET SMILES batch");
    // one change event, not ~2,000 simulated keystrokes (the r316 CI
    // catch: the DTA sibling timed out at the 5 s test budget typing
    // a 101-line batch into the sixteen-stage page — the gate is the
    // point, the keystroke simulation is not)
    const lines = Array.from({ length: 101 }, () => "CCO").join("\n");
    fireEvent.change(textarea, { target: { value: lines } });
    const scoreButton = screen.getByRole("button", { name: "Score ADMET" });
    await userEvent.click(scoreButton);
    expect(
      await screen.findByText(/caps a batch at 100 SMILES; got 101/),
    ).toBeInTheDocument();
    expect(mlMocks.admetScore).not.toHaveBeenCalled();
  });

  it("carries the honest provenance and disclaimer on every answer", async () => {
    mlMocks.admetScore.mockResolvedValue(scoreResponse([ASPIRIN_CARD]));
    render(<MLPage />);
    const scoreButton = await screen.findByRole("button", {
      name: "Score ADMET",
    });
    await userEvent.click(scoreButton);
    await waitFor(() => {
      expect(
        screen.getByText(/0 shipped ensembles \(the honest absence/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(
        /predicted ADMET properties are model estimates, not experimentally established facts/,
      ),
    ).toBeInTheDocument();
  });
});
