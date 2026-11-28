import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GenerativePage } from "@/features/generative/GenerativePage";

/**
 * GEN-3 (the roadmap's row 17) — the REINVENT-style RL lane as a
 * component-level user flow over the real page mount:
 *
 *   the third method card — the method selector renders THREE cards
 *                          (the RL card selected by its own click);
 *   the prior select     — the session priors list + the engine's
 *                          own on-the-fly default option;
 *   the run              — the rl-optimize request carries the
 *                          objective spec + iterations/batch/seed
 *                          through the typed service layer;
 *   the verdict card     — the job's relayed result renders the
 *                          leaderboard + the per-iteration step curve
 *                          through the PlotlyFigure component.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, the runtime parser (parseRlOptimizeView — the
 * REAL one, spread from the actual module), and the DOM are the
 * real ones.
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
  rlOptimize: vi.fn(),
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
      rlOptimize: generativeMocks.rlOptimize,
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

vi.mock("@/components/charts/PlotlyFigure", () => ({
  PlotlyFigure: () => <div data-testid="plotly-stub" />,
}));

const RL_RESULT = {
  mode: "sync",
  optimization: {
    best: "CCCC(=O)Nc1ccc2c(n1)S2",
    best_score: 0.743,
    iterations: 10,
    history: [
      { iteration: 0, best_score: 0.617, mean_score: 0.156, valid_fraction: 0.83, drift: 0.01, entropy: 3.2, loss: 0.4 },
      { iteration: 1, best_score: 0.743, mean_score: 0.2, valid_fraction: 0.87, drift: 0.03, entropy: 3.1, loss: 0.38 },
      { iteration: 2, best_score: 0.64, mean_score: 0.24, valid_fraction: 0.9, drift: 0.05, entropy: 3.0, loss: 0.36 },
    ],
    metadata: {
      method: "reinforce_rl",
      seed: 3,
      iterations: 10,
      unique_generated: 107,
    },
    top: [
      { smiles: "CCCC(=O)Nc1ccc2c(n1)S2", score: 0.743, breakdown: {} },
      { smiles: "CCO", score: 0.52, breakdown: {} },
    ],
  },
  prior: null,
  note: "the REINVENT-style recipe over the REAL engine",
};

beforeEach(() => {
  vi.clearAllMocks();
  jobState.snapshots = {};
  generativeMocks.objectives.mockResolvedValue({
    objectives: [
      { kind: "qed", class: "QedObjective", params: {}, description: "quantified drug-likeness (0..1, higher better)" },
    ],
  });
  generativeMocks.priors.mockResolvedValue({
    priors: [
      {
        prior_id: "prior-000111",
        name: "smiles-prior",
        n_train_smiles: 10,
        params: {},
        losses: [0.9, 0.5],
        corpus_source: { source: "inline", n: 10 },
        created_at: "2026-11-28T00:00:00Z",
      },
    ],
  });
  datasetsMocks.list.mockResolvedValue({ datasets: [] });
  datasetsMocks.view.mockResolvedValue(null);
});

describe("the GEN-3 RL lane of the Generative workspace", () => {
  it("renders the third method card and arms the lane", async () => {
    render(<GenerativePage />);

    // THREE method cards now — the RL card among them
    const rlCard = await screen.findByRole("heading", {
      name: "RL optimization (REINVENT-style)",
    });
    await userEvent.click(rlCard.closest("div")!);

    // the RL controls render: the prior select over the session list
    const priorSelect = await screen.findByLabelText(
      "the RL lane's prior",
    );
    expect(
      Array.from(priorSelect.querySelectorAll("option")).map(
        (o) => o.textContent,
      ),
    ).toEqual([
      "the engine's own (on-the-fly, DEFAULT_CORPUS)",
      "smiles-prior · 10 SMILES",
    ]);
    expect(screen.getByLabelText("rl iterations")).toBeInTheDocument();
    expect(screen.getByLabelText("rl batch")).toBeInTheDocument();
    expect(screen.getByLabelText("rl seed")).toBeInTheDocument();
    expect(screen.getByLabelText("rl top_n")).toBeInTheDocument();
  });

  it("runs the RL optimization and renders the leaderboard + the step curve", async () => {
    render(<GenerativePage />);

    const rlCard = await screen.findByRole("heading", {
      name: "RL optimization (REINVENT-style)",
    });
    await userEvent.click(rlCard.closest("div")!);

    generativeMocks.rlOptimize.mockResolvedValue({
      mode: "job",
      job: {
        job_id: "job-gen3",
        operation: "generative.rl-optimize",
        category: "generative-ai",
        state: "RUNNING",
        progress: 0,
        stage: "queued",
        error: null,
        created_at: "2026-11-29T00:00:00Z",
        started_at: "2026-11-29T00:00:01Z",
        finished_at: null,
        configuration: {},
        result_ready: false,
      },
    });
    jobsMocks.result.mockResolvedValue({ result: RL_RESULT });
    jobState.snapshots["job-gen3"] = {
      view: {
        job_id: "job-gen3",
        operation: "generative.rl-optimize",
        category: "generative-ai",
        state: "COMPLETED",
        progress: 100,
        stage: "done",
        error: null,
        created_at: "2026-11-29T00:00:00Z",
        started_at: "2026-11-29T00:00:01Z",
        finished_at: "2026-11-29T00:00:10Z",
        configuration: {},
        result_ready: true,
      },
    };

    await userEvent.click(
      screen.getByRole("button", { name: /Run the RL optimization/ }),
    );

    await waitFor(() => {
      expect(generativeMocks.rlOptimize).toHaveBeenCalledTimes(1);
    });
    const options = generativeMocks.rlOptimize.mock.calls[0]![0] as {
      objective: Record<string, unknown>;
      priorId: string | null;
      iterations: number;
      batchSize: number;
      seed: number;
      topN: number;
      mode: string;
    };
    expect(options.objective).toEqual({ kind: "qed" });
    expect(options.priorId).toBeNull();
    expect(options.iterations).toEqual(30);
    expect(options.batchSize).toEqual(32);
    expect(options.seed).toEqual(42);
    expect(options.topN).toEqual(20);
    expect(options.mode).toEqual("job");

    // the job's relayed result renders the verdict card + the
    // leaderboard + the step curve stub
    expect(
      await screen.findByText("The RL run's verdict"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/10 iterations · 2 ranked molecules/),
    ).toBeInTheDocument();
    expect(screen.getByText("CCCC(=O)Nc1ccc2c(n1)S2")).toBeInTheDocument();
    // the step curve through PlotlyFigure (the roadmap's own clause)
    expect(screen.getByTestId("plotly-stub")).toBeInTheDocument();
    expect(
      screen.getAllByText(/the RL run's top 2 molecules/).length,
    ).toBeGreaterThan(0);
  });
});
