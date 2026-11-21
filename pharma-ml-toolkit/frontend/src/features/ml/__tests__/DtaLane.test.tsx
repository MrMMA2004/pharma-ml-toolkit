import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MLPage } from "@/features/ml/MLPage";
import { ApiError } from "@/services/api/client";

/**
 * part13 Phase 5's next remainder item — the drug–target interaction
 * lane of the Machine Learning workspace as a complete component-level
 * user flow:
 *
 *   the registry line      — the card renders the backend's own
 *                            vocabulary summary (the three DeepDTA
 *                            alphabets, the ECFP+AAC pairing
 *                            arithmetic) off the mocked registry
 *                            answer;
 *   the featurization      — the textarea's SMILES,PROTEIN[,AFFINITY]
 *                            lines become the honest request, and the
 *                            per-pair cards render the REAL parsed
 *                            payload (the protein composition, the top
 *                            AAC bars, the affinity badges);
 *   the invalid verdict    — an invalid SMILES row reports itself
 *                            per-row (dropped together with its
 *                            protein and affinity, never silent);
 *   typed errors           — the backend's own hard 422 (a protein
 *                            symbol outside DeepDTA's alphabet)
 *                            surfaces verbatim through the banner;
 *   client-side gating     — an empty batch, an over-cap batch and a
 *                            malformed line never leave the page
 *                            (validated BEFORE any request fires);
 *   honest provenance      — the disclaimer + the "0 shipped
 *                            checkpoints" line ride every answer.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parsers (parseDtaRegistryResponse /
 * parseDtaFeaturesResponse — the REAL ones, spread from the actual
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
  dtaRegistry: vi.fn(),
  dtaFeatures: vi.fn(),
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

const DTA_PROVENANCE = {
  backend: "pharmaml.ml.dta",
  source: "hkmztrk/DeepDTA — the alphabets and rm2 ported verbatim",
  n_alphabets: 3,
  shipped_checkpoints: 0,
  shipped_checkpoints_note:
    "the DeepDTA CNN, DeepPurpose, DGL-LifeSci and TorchDrug " +
    "checkpoints are NOT shipped — the feature/metric boundary",
  disclaimer:
    "paired DTA features are deterministic encodings, not learned " +
    "affinity predictions — no Kd/Ki/pIC50 value is produced here",
};

const DTA_REGISTRY = {
  alphabets: {
    protein: {
      n_symbols: 25,
      padding_code: 0,
      symbols: { A: 1, C: 2, X: 24, Z: 25 },
      encoding: "1-based labels, zero reserved for padding",
      source: "DeepDTA CHARPROTSET verbatim",
    },
    smiles_isomeric: {
      n_symbols: 64,
      padding_code: 0,
      symbols: { C: 42, "@": 8, y: 64 },
      encoding: "the supplied string is encoded as-is",
      source: "DeepDTA CHARISOSMISET verbatim",
    },
    smiles_canonical: {
      n_symbols: 62,
      padding_code: 0,
      symbols: { "#": 1, y: 62 },
      encoding: "same contract, the 62-symbol canonical set",
      source: "DeepDTA CHARCANSMISET verbatim",
    },
  },
  amino_acids: Array.from("ACDEFGHIKLMNPQRSTVWY"),
  protein_features: {
    n_base: 23,
    base_columns: [
      "prot_length",
      "prot_standard_fraction",
      "prot_nonstandard_fraction",
    ],
    n_dipeptide_extras: 401,
    n_with_dipeptides: 424,
    note: "non-standard residues are counted explicitly",
  },
  ligand_features: {
    family: "ecfp",
    n_bits: 2048,
    source: "the toolkit's own compute_feature_frame",
  },
  paired_width: { base: 2071, with_dipeptides: 2472 },
  metrics: [
    {
      name: "deepdta_rm2",
      description: "DeepDTA's get_rm2 external-prediction metric",
    },
  ],
  demo_frame: {
    rows: 12,
    note: "twelve synthetic, clearly non-benchmark pairs",
  },
  provenance: DTA_PROVENANCE,
  features_url: "/api/v1/dta/features",
  note: "sync-only light compute; up to 100 pairs per call",
};

function proteinRow(overrides: Record<string, unknown> = {}) {
  const aac: Record<string, number> = {};
  for (const symbol of "ACDEFGHIKLMNPQRSTVWY") {
    aac[symbol] = 0.05;
  }
  return {
    prot_length: 22,
    prot_standard_fraction: 1.0,
    prot_nonstandard_fraction: 0.0,
    aac,
    ...overrides,
  };
}

function pairCard(index: number, smiles: string, protein: string) {
  return {
    index,
    smiles,
    protein,
    valid: true,
    protein_row: proteinRow(),
    affinity: 5.0 + index * 0.5,
  };
}

function featuresResponse(
  rows: unknown[],
  dropped: unknown[] = [],
  overrides: Record<string, unknown> = {},
) {
  const keptIndices = rows
    .map((row, i) => ((row as { valid: boolean }).valid ? i : -1))
    .filter((i) => i >= 0);
  const affinityValues = keptIndices.map(
    (i) => (rows[i] as { affinity: number | null }).affinity ?? 0,
  );
  return {
    n_pairs: rows.length,
    n_kept: keptIndices.length,
    n_dropped: dropped.length,
    kept_indices: keptIndices,
    dropped,
    n_ligand_features: 2048,
    n_protein_features: 23,
    n_features: 2071,
    dipeptides: false,
    rows,
    affinity: {
      present: true,
      n: affinityValues.length,
      min: Math.min(...affinityValues),
      max: Math.max(...affinityValues),
      mean:
        affinityValues.reduce((a, b) => a + b, 0) / affinityValues.length,
    },
    provenance: DTA_PROVENANCE,
    note: "the feature matrix is the product",
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
  mlMocks.admetScore.mockReset();
  mlMocks.dtaRegistry.mockResolvedValue(DTA_REGISTRY);
  mlMocks.dtaFeatures.mockReset();
});

describe("the drug–target interaction lane of the Machine Learning workspace", () => {
  it("renders the stage with the registry vocabulary line", async () => {
    render(<MLPage />);
    await waitFor(() => {
      expect(screen.getByText("Drug–Target Interaction")).toBeInTheDocument();
    });
    // the backend's own registry summary, never re-declared
    await waitFor(() => {
      expect(
        screen.getByText(/Alphabets: 25 protein symbols/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/64 isomeric \/ 62 canonical SMILES symbols/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/2048-bit ECFP \+ 23 AAC columns → 2071 features/),
    ).toBeInTheDocument();
    expect(mlMocks.dtaRegistry).toHaveBeenCalledTimes(1);
  });

  it("featurizes the textarea's pairs and renders the per-pair cards", async () => {
    mlMocks.dtaFeatures.mockResolvedValue(
      featuresResponse([
        pairCard(0, "CCO", "MKTAYIAKQRQISFVKSHFSRQ"),
        pairCard(1, "CCN", "GAVLILKKGGSSHHTRPFSL"),
        pairCard(2, "CCC", "ACDEFGHIKLMNPQRSTVWY"),
      ]),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Featurize pairs",
    });
    await userEvent.click(button);
    // the default textarea (three pairs, one per line, with affinity)
    await waitFor(() => {
      expect(mlMocks.dtaFeatures).toHaveBeenCalledWith(
        [
          { smiles: "CCO", protein: "MKTAYIAKQRQISFVKSHFSRQ", affinity: 5.0 },
          { smiles: "CCN", protein: "GAVLILKKGGSSHHTRPFSL", affinity: 5.5 },
          {
            smiles: "CCC",
            protein: "ACDEFGHIKLMNPQRSTVWY",
            affinity: 6.0,
          },
        ],
        false,
      );
    });
    // the summary line over the REAL parsed payload
    expect(
      await screen.findByText(/3 pairs → 3 kept · 0 dropped/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/the matrix: 2071 features = 2048 ECFP \+ 23/),
    ).toBeInTheDocument();
    // the per-pair cards: the verdict badge + the affinity + the
    // protein composition
    expect(screen.getAllByText("kept")).toHaveLength(3);
    expect(screen.getAllByText(/affinity 5/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/22 residues/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/top AAC components/).length).toBe(3);
  });

  it("reports an invalid SMILES row per-row, never silently", async () => {
    mlMocks.dtaFeatures.mockResolvedValue(
      featuresResponse(
        [
          pairCard(0, "CCO", "MKTAYIAKQRQISFVKSHFSRQ"),
          {
            index: 1,
            smiles: "not-a-molecule",
            protein: "GAVLILKKGGSSHHTRPFSL",
            valid: false,
            protein_row: proteinRow({ prot_length: 19 }),
            affinity: null,
          },
        ],
        [
          {
            index: 1,
            smiles: "not-a-molecule",
            reason:
              "invalid SMILES — dropped together with its protein and " +
              "affinity value by the ECFP featurization (never silently)",
          },
        ],
      ),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Featurize pairs",
    });
    await userEvent.click(button);
    expect(
      await screen.findByText(/2 pairs → 1 kept · 1 dropped/),
    ).toBeInTheDocument();
    // the dropped row reports itself through the card's own badge
    expect(screen.getByText("dropped (invalid SMILES)")).toBeInTheDocument();
    // and through the dropped list's reason line
    expect(
      screen.getByText(/invalid SMILES — dropped together with its protein/),
    ).toBeInTheDocument();
  });

  it("surfaces the backend's own hard 422 verbatim through the banner", async () => {
    mlMocks.dtaFeatures.mockRejectedValue(
      new ApiError(422, {
        detail:
          "protein sequence contains symbols outside DeepDTA's " +
          "alphabet: J; accepted symbols are ACDEFGHIKLMNPQRSTVWYBXZOU",
      }),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Featurize pairs",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(screen.getByText(/API 422/)).toBeInTheDocument();
    });
    expect(
      screen.getByText(/symbols outside DeepDTA's alphabet: J/),
    ).toBeInTheDocument();
    expect(mlMocks.dtaFeatures).toHaveBeenCalledTimes(1);
  });

  it("gates an empty batch client-side — no request fires", async () => {
    render(<MLPage />);
    const textarea = await screen.findByLabelText("DTA pairs batch");
    await userEvent.clear(textarea);
    const button = screen.getByRole("button", { name: "Featurize pairs" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/needs at least one SMILES,PROTEIN line/),
    ).toBeInTheDocument();
    expect(mlMocks.dtaFeatures).not.toHaveBeenCalled();
  });

  it("gates an over-cap batch client-side — no request fires", async () => {
    render(<MLPage />);
    const textarea = await screen.findByLabelText("DTA pairs batch");
    const lines = Array.from({ length: 101 }, () => "CCO,MK").join("\n");
    await userEvent.clear(textarea);
    await userEvent.type(textarea, lines);
    const button = screen.getByRole("button", { name: "Featurize pairs" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/caps a batch at 100 pairs; got 101/),
    ).toBeInTheDocument();
    expect(mlMocks.dtaFeatures).not.toHaveBeenCalled();
  });

  it("gates a malformed line client-side — no request fires", async () => {
    render(<MLPage />);
    const textarea = await screen.findByLabelText("DTA pairs batch");
    await userEvent.clear(textarea);
    await userEvent.type(textarea, "CCO-only-no-protein-field");
    const button = screen.getByRole("button", { name: "Featurize pairs" });
    await userEvent.click(button);
    expect(
      await screen.findByText(/line 1 needs at least SMILES,PROTEIN/),
    ).toBeInTheDocument();
    expect(mlMocks.dtaFeatures).not.toHaveBeenCalled();
  });

  it("carries the honest provenance and disclaimer on every answer", async () => {
    mlMocks.dtaFeatures.mockResolvedValue(
      featuresResponse([pairCard(0, "CCO", "MKTAYIAKQRQISFVKSHFSRQ")]),
    );
    render(<MLPage />);
    const button = await screen.findByRole("button", {
      name: "Featurize pairs",
    });
    await userEvent.click(button);
    await waitFor(() => {
      expect(
        screen.getByText(/0 shipped checkpoints \(the honest absence/),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(
        /deterministic encodings, not learned affinity predictions/,
      ),
    ).toBeInTheDocument();
  });
});
