import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GenerativePage } from "@/features/generative/GenerativePage";

/**
 * GEN-2 — the roadmap's row 5: the Validate stage's MOSES metrics card
 * as a component-level user flow over the real page mount:
 *
 *   the working set  — a sync optimize answer fills it (the same lane
 *                      the Generate stage rides);
 *   the metrics card — the five headline metrics render from a mocked
 *                      route payload (validity / uniqueness / novelty /
 *                      IntDiv1 / scaffold diversity), the property-
 *                      shift table rides below;
 *   the honest nulls — without a reference the novelty cell renders
 *                      "not measured (no reference)" and the
 *                      reference-mean / shift columns render "—",
 *                      never a fake 1.0;
 *   the reference lane — the inline corpus picked, the request carries
 *                      it and the novelty renders the measured value.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parser (the REAL one, spread from
 * the actual module), and the DOM are the real ones.
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
  optimize: vi.fn(),
  metrics: vi.fn(),
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
      optimize: generativeMocks.optimize,
      metrics: generativeMocks.metrics,
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

const OPTIMIZE_SYNC = {
  mode: "sync",
  optimization: {
    best: "CCO",
    best_score: 0.5,
    generations: 2,
    history: [],
    metadata: { method: "genetic_algorithm", mutation: "uniform", crossover: "splice" },
    top: [
      { smiles: "CCO", score: 0.5, breakdown: {} },
      { smiles: "CCN", score: 0.4, breakdown: {} },
      { smiles: "CCS", score: 0.3, breakdown: {} },
    ],
  },
};

const METRICS_NO_REFERENCE = {
  n_generated: 3,
  n_valid: 3,
  valid_fraction: 1.0,
  n_unique: 3,
  unique_fraction: 1.0,
  n_novel: 0,
  novel_fraction: 0.0,
  internal_diversity_1: 0.667,
  internal_diversity_2: 0.68,
  scaffold_diversity: 1.0,
  n_scaffolds: 3,
  pairwise_truncated: false,
  property_shifts: [
    { property: "MW", generated_mean: 46.1, reference_mean: null, shift: null },
    { property: "MolLogP", generated_mean: 0.5, reference_mean: null, shift: null },
    { property: "TPSA", generated_mean: 20.2, reference_mean: null, shift: null },
    { property: "QED", generated_mean: 0.4, reference_mean: null, shift: null },
  ],
  reference: { source: null, n: 0 },
};

const METRICS_WITH_REFERENCE = {
  ...METRICS_NO_REFERENCE,
  n_novel: 1,
  novel_fraction: 1.0 / 3.0,
  property_shifts: [
    { property: "MW", generated_mean: 46.1, reference_mean: 45.5, shift: 0.6 },
    { property: "MolLogP", generated_mean: 0.5, reference_mean: 0.4, shift: 0.1 },
    { property: "TPSA", generated_mean: 20.2, reference_mean: 20.0, shift: 0.2 },
    { property: "QED", generated_mean: 0.4, reference_mean: 0.39, shift: 0.01 },
  ],
  reference: { source: "inline", n: 2 },
};

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  generativeMocks.objectives.mockResolvedValue({
    objectives: [
      { kind: "qed", class: "QedObjective", params: {}, description: "quantified drug-likeness (0..1, higher better)" },
    ],
  });
  generativeMocks.priors.mockResolvedValue({ priors: [] });
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  datasetsMocks.view.mockResolvedValue(null);
});

async function driveToWorkingSet() {
  render(<GenerativePage />);
  const gaCard = await screen.findByRole("heading", {
    name: "Genetic optimization",
  });
  await userEvent.click(gaCard.closest("div")!);
  // the sync optimize answer fills the working set (the Generate
  // stage's own lane)
  generativeMocks.optimize.mockResolvedValue(OPTIMIZE_SYNC);
  await userEvent.click(
    screen.getByRole("button", { name: /Run evolution/ }),
  );
  // the working set line renders in the metrics card
  await waitFor(() => {
    expect(
      screen.getByText(/3 molecules in the working set/),
    ).toBeInTheDocument();
  });
}

describe("the GEN-2 MOSES metrics card of the Generative workspace", () => {
  it("renders the five headline metrics with honest nulls without a reference", async () => {
    await driveToWorkingSet();

    generativeMocks.metrics.mockResolvedValue(METRICS_NO_REFERENCE);
    await userEvent.click(
      screen.getByRole("button", { name: /Compute MOSES metrics/ }),
    );

    await waitFor(() => {
      expect(generativeMocks.metrics).toHaveBeenCalledTimes(1);
    });
    const request = generativeMocks.metrics.mock.calls[0]![0] as {
      generated: string[];
      reference: string[] | null;
      datasetId: string | null;
    };
    // the no-reference wire contract: the working set rides, the
    // reference lanes stay null
    expect(request.generated).toEqual(["CCO", "CCN", "CCS"]);
    expect(request.reference).toBeNull();
    expect(request.datasetId).toBeNull();

    // the five headline cells render
    const cells = await screen.findByLabelText("moses headline metrics");
    expect(within(cells).getByText("validity")).toBeInTheDocument();
    expect(within(cells).getByText("uniqueness")).toBeInTheDocument();
    expect(within(cells).getByText("novelty")).toBeInTheDocument();
    expect(within(cells).getByText("IntDiv1")).toBeInTheDocument();
    expect(within(cells).getByText("scaffold diversity")).toBeInTheDocument();
    // validity and uniqueness both render 100.0% (3/3) over the fixture
    expect(within(cells).getAllByText("100.0% (3/3)")).toHaveLength(2);

    // the honest nulls: the novelty cell states not-measured, the
    // reference-mean / shift columns render the em dash
    expect(
      within(cells).getByText("not measured (no reference)"),
    ).toBeInTheDocument();
    const table = await screen.findByLabelText("moses property shifts");
    const dashes = within(table).getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(8); // 4 rows x 2 nulls
    expect(
      screen.getByText(/none supplied — novelty and the reference means/),
    ).toBeInTheDocument();
  });

  it("carries the inline reference through and renders the measured novelty", async () => {
    await driveToWorkingSet();

    await userEvent.selectOptions(
      screen.getByLabelText("metrics reference source"),
      "inline",
    );
    generativeMocks.metrics.mockResolvedValue(METRICS_WITH_REFERENCE);
    await userEvent.click(
      screen.getByRole("button", { name: /Compute MOSES metrics/ }),
    );

    await waitFor(() => {
      expect(generativeMocks.metrics).toHaveBeenCalledTimes(1);
    });
    const request = generativeMocks.metrics.mock.calls[0]![0] as {
      reference: string[] | null;
    };
    // the inline starting molecules ride as the reference (the
    // stage-two corpus, 5 entries in the default fixture)
    expect(request.reference).not.toBeNull();
    expect(request.reference!.length).toBeGreaterThanOrEqual(5);

    // the measured novelty renders
    const cells = await screen.findByLabelText("moses headline metrics");
    expect(within(cells).getByText("33.3% (1/3)")).toBeInTheDocument();
    const table = screen.getByLabelText("moses property shifts");
    expect(within(table).getAllByText("45.500")).toHaveLength(1);
    expect(
      screen.getByText(/inline \(2 SMILES\)/),
    ).toBeInTheDocument();
  });
});
