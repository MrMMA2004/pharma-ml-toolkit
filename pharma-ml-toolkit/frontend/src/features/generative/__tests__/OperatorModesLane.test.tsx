import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GenerativePage } from "@/features/generative/GenerativePage";

/**
 * GEN-1 — the roadmap's row 4: the Configure stage's operator-mode
 * selects as a component-level user flow over the real page mount.
 *
 *   the two selects — the mutation/crossover modes over the engine's
 *                      own vocabulary (the route Literal's set), the
 *                      option labels carrying the engine's own
 *                      descriptions;
 *   the conditional gate — the min-heavy-atoms input appears only
 *                      for the scaffold_hop crossover;
 *   the passthrough    — the optimize request carries the picked
 *                      modes + the gate through the typed service
 *                      layer verbatim (mutationMode / crossoverMode /
 *                      minHeavyAtoms on the wire).
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
  optimize: vi.fn(),
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

describe("the GEN-1 operator-mode selects of the Generative workspace", () => {
  it("passes the picked modes + the gate through the optimize request", async () => {
    render(<GenerativePage />);

    // stage 1: select the Genetic optimization method
    const gaCard = await screen.findByRole("heading", {
      name: "Genetic optimization",
    });
    await userEvent.click(gaCard.closest("div")!);

    // the two selects render over the engine's own vocabulary
    const mutationSelect = await screen.findByLabelText("ga mutation mode");
    expect(
      Array.from(mutationSelect.querySelectorAll("option")).map(
        (o) => o.textContent,
      ),
    ).toEqual([
      "uniform (character moves)",
      "scaffold (decoration resampling)",
      "topology (ring-size moves)",
      "bioisostere (matched pairs)",
    ]);
    const crossoverSelect = screen.getByLabelText("ga crossover mode");
    expect(
      Array.from(crossoverSelect.querySelectorAll("option")).map(
        (o) => o.textContent,
      ),
    ).toEqual([
      "splice (single-point)",
      "scaffold_hop (BREED-style hop)",
    ]);

    // the gate input is absent for the plain splice mode
    expect(
      screen.queryByLabelText("ga min heavy atoms"),
    ).toBeNull();

    // pick the structural pair: topology mutation + scaffold-hop
    // crossover; the gate input appears with the crossover
    await userEvent.selectOptions(mutationSelect, "topology");
    await userEvent.selectOptions(crossoverSelect, "scaffold_hop");
    const gateInput = await screen.findByLabelText("ga min heavy atoms");
    await userEvent.clear(gateInput);
    await userEvent.type(gateInput, "12");

    // run the evolution — the request carries the picked modes
    generativeMocks.optimize.mockResolvedValue({
      mode: "job",
      job: {
        job_id: "job-gen1",
        operation: "generative.optimize",
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
    await userEvent.click(
      screen.getByRole("button", { name: /Run evolution/ }),
    );

    await waitFor(() => {
      expect(generativeMocks.optimize).toHaveBeenCalledTimes(1);
    });
    const options = generativeMocks.optimize.mock.calls[0]![0] as {
      mutationMode: string;
      crossoverMode: string;
      minHeavyAtoms: number;
    };
    expect(options.mutationMode).toBe("topology");
    expect(options.crossoverMode).toBe("scaffold_hop");
    expect(options.minHeavyAtoms).toBe(12);
  });

  it("keeps the default lane honest (uniform/splice, gate off at 1)", async () => {
    render(<GenerativePage />);

    const gaCard = await screen.findByRole("heading", {
      name: "Genetic optimization",
    });
    await userEvent.click(gaCard.closest("div")!);

    // the defaults: no picks made — the request answers the engine's
    // own defaults (uniform / splice / the gate off)
    generativeMocks.optimize.mockResolvedValue({
      mode: "job",
      job: {
        job_id: "job-gen1b",
        operation: "generative.optimize",
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
    await userEvent.click(
      screen.getByRole("button", { name: /Run evolution/ }),
    );

    await waitFor(() => {
      expect(generativeMocks.optimize).toHaveBeenCalledTimes(1);
    });
    const options = generativeMocks.optimize.mock.calls[0]![0] as {
      mutationMode: string;
      crossoverMode: string;
      minHeavyAtoms: number;
    };
    expect(options.mutationMode).toBe("uniform");
    expect(options.crossoverMode).toBe("splice");
    expect(options.minHeavyAtoms).toBe(1);
  });
});
