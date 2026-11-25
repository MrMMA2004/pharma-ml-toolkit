import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VisualizationPage } from "@/features/visualization/VisualizationPage";
import { ApiError } from "@/services/api/client";
import type { ArtifactView } from "@/services/artifacts/client";

/**
 * part17 §3.2 (Round 360, the G1 lane) — the consumption hub's
 * view/delete consumers as a component-level user flow:
 *
 *   the registry    — the stage-one list renders the session's rows
 *                     (the kind line + the Select/Download actions);
 *   the view lane   — the row's View action fetches the FULL record
 *                     through the typed artifactsApi.view and the
 *                     metadata panel renders (artifact_id, the full
 *                     sha256, the meta JSON, the download route); a
 *                     404 surfaces verbatim, never a silent blank;
 *   the two-step    — the row's Delete action only ARMS the
 *                     confirmation (the typed delete has NOT been
 *                     called yet); the card states the name, the
 *                     kind/size metadata, and the honest scope note;
 *   the confirm     — only the confirm button fires the typed
 *                     artifactsApi.delete; the registry refreshes and
 *                     the notice names the deleted row;
 *   the 404 lane    — an already-gone row reconciles the registry and
 *                     says so (never a hidden failure);
 *   the disarm      — Cancel never fires the request;
 *   the stray-fetch
 *                    — no other network lane fires during the flows.
 *
 * Services are mocked at the network boundary only; the component,
 * its state machine, and the DOM are the real ones.
 */

const artifactsMocks = vi.hoisted(() => ({
  list: vi.fn(),
  view: vi.fn(),
  delete: vi.fn(),
  download: vi.fn(),
}));
vi.mock("@/services/artifacts/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/artifacts/client")>();
  // the REAL parseArtifactView rides the page under test; only the
  // network boundary is faked
  return {
    ...actual,
    artifactsApi: {
      ...actual.artifactsApi,
      list: artifactsMocks.list,
      view: artifactsMocks.view,
      delete: artifactsMocks.delete,
      download: artifactsMocks.download,
    },
  };
});

const vizMocks = vi.hoisted(() => ({
  catalog: vi.fn(),
  stat: vi.fn(),
  concentrationProfile: vi.fn(),
  doseAccumulation: vi.fn(),
  mdSeries: vi.fn(),
  exportTable: vi.fn(),
  report: vi.fn(),
}));
vi.mock("@/services/visualization/client", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/visualization/client")>();
  return {
    ...actual,
    vizApi: {
      ...actual.vizApi,
      catalog: vizMocks.catalog,
      stat: vizMocks.stat,
      concentrationProfile: vizMocks.concentrationProfile,
      doseAccumulation: vizMocks.doseAccumulation,
      mdSeries: vizMocks.mdSeries,
      exportTable: vizMocks.exportTable,
      report: vizMocks.report,
    },
  };
});

vi.mock("@/hooks/useJobTracking", () => ({
  useJob: () => ({
    snapshot: null,
    cancel: vi.fn(),
    refresh: vi.fn(),
  }),
}));

vi.mock("@/services/jobs/client", () => ({
  jobsApi: { result: vi.fn() },
}));

vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => ({
    subscribe: () => () => undefined,
  }),
}));

// the raw REST client — stubbed so no accidental fetch ever leaves
vi.mock("@/services/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api/client")>();
  return { ...actual, api: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } };
});

const CATALOG = {
  families: [
    {
      family: "stat",
      kinds: ["histogram"],
      formats: ["PNG", "SVG"],
      backend: "matplotlib",
    },
  ],
  palette: [],
  export_formats: ["csv"],
  report_themes: ["auto"],
};

const artifact = (overrides: Partial<ArtifactView> = {}): ArtifactView => ({
  artifact_id: "art-0001",
  kind: "figure",
  filename: "solubility-histogram.png",
  content_type: "image/png",
  size_bytes: 2048,
  sha256: "a".repeat(64),
  created_at: "2026-11-25T09:00:00Z",
  meta: { engine: "matplotlib" },
  download_url: "/api/v1/artifacts/art-0001/download",
  ...overrides,
});

const ROWS: readonly ArtifactView[] = [
  artifact(),
  artifact({
    artifact_id: "art-0002",
    kind: "table",
    filename: "prediction-table.csv",
    content_type: "text/csv",
    sha256: "b".repeat(64),
  }),
];

describe("the Visualization consumption hub's view/delete lanes (part17 §3.2)", () => {
  beforeEach(() => {
    artifactsMocks.list.mockReset();
    artifactsMocks.view.mockReset();
    artifactsMocks.delete.mockReset();
    artifactsMocks.download.mockReset();
    vizMocks.catalog.mockReset();
    artifactsMocks.list.mockResolvedValue(ROWS);
    vizMocks.catalog.mockResolvedValue(CATALOG);
  });

  it("renders the registry rows with the View and Delete actions", async () => {
    render(<VisualizationPage />);
    expect(await screen.findByText("solubility-histogram.png")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "View artifact solubility-histogram.png" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Delete artifact solubility-histogram.png" }),
    ).toBeInTheDocument();
  });

  it("the View action fetches the full record and renders the metadata panel", async () => {
    artifactsMocks.view.mockResolvedValue(ROWS[0]);
    render(<VisualizationPage />);
    await screen.findByText("solubility-histogram.png");

    await userEvent.click(
      screen.getByRole("button", { name: "View artifact solubility-histogram.png" }),
    );
    await waitFor(() => {
      expect(artifactsMocks.view).toHaveBeenCalledWith("art-0001");
    });
    // the FULL registry record: the id, the full sha256, the meta
    // JSON, and the download route
    expect(await screen.findByText("The registry record — solubility-histogram.png")).toBeInTheDocument();
    expect(screen.getByText("art-0001")).toBeInTheDocument();
    expect(screen.getByText("a".repeat(64))).toBeInTheDocument();
    expect(screen.getByText(/"engine":"matplotlib"/)).toBeInTheDocument();
    expect(screen.getByText("/api/v1/artifacts/art-0001/download")).toBeInTheDocument();
  });

  it("the View lane surfaces a 404 verbatim — the row can be gone between list and click", async () => {
    artifactsMocks.view.mockRejectedValueOnce(
      new ApiError(404, { detail: "Unknown artifact_id." }),
    );
    render(<VisualizationPage />);
    await screen.findByText("solubility-histogram.png");

    await userEvent.click(
      screen.getByRole("button", { name: "View artifact solubility-histogram.png" }),
    );
    // the honest banner names the lane's own answer (the demo-data
    // card's own alert may be present — the view banner is the one
    // that carries the artifact answer)
    const banners = await screen.findAllByRole("alert");
    expect(
      banners.some((b) => b.textContent?.includes("Unknown artifact_id.")),
    ).toBe(true);
    // the panel never renders a fabricated record
    expect(screen.queryByText(/^The registry record/)).toBeNull();
  });

  it("the Delete action only ARMS the confirmation — no request fires yet", async () => {
    render(<VisualizationPage />);
    await screen.findByText("solubility-histogram.png");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete artifact solubility-histogram.png" }),
    );
    expect(
      await screen.findByText(/Delete artifact “solubility-histogram.png”\?/),
    ).toBeInTheDocument();
    // the honest metadata: the kind, the size, the id
    expect(screen.getByText(/figure · image\/png · 2\.0 KiB · id art-0001/)).toBeInTheDocument();
    // the honest scope note — the action never deletes another record
    expect(
      screen.getByText(/never deletes another record/),
    ).toBeInTheDocument();
    expect(artifactsMocks.delete).not.toHaveBeenCalled();
  });

  it("Cancel disarms the gate without any request", async () => {
    render(<VisualizationPage />);
    await screen.findByText("solubility-histogram.png");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete artifact solubility-histogram.png" }),
    );
    await screen.findByText(/Delete artifact “solubility-histogram.png”\?/);
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(artifactsMocks.delete).not.toHaveBeenCalled();
    expect(screen.queryByText(/Delete artifact “solubility-histogram.png”\?/)).toBeNull();
  });

  it("only the Confirm button fires the typed delete, refreshes the registry, and reports it", async () => {
    artifactsMocks.list.mockResolvedValueOnce(ROWS);
    artifactsMocks.list.mockResolvedValue([]);
    artifactsMocks.delete.mockResolvedValue({ evicted: true, artifact_id: "art-0001" });
    render(<VisualizationPage />);
    await screen.findByText("solubility-histogram.png");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete artifact solubility-histogram.png" }),
    );
    await screen.findByText(/Delete artifact “solubility-histogram.png”\?/);
    await userEvent.click(screen.getByRole("button", { name: "Delete artifact" }));

    await waitFor(() => {
      expect(artifactsMocks.delete).toHaveBeenCalledTimes(1);
      expect(artifactsMocks.delete).toHaveBeenCalledWith("art-0001");
    });
    expect(
      await screen.findByText(/Deleted "solubility-histogram.png" from the artifact registry\./),
    ).toBeInTheDocument();
    // the registry refreshed — the row left the list
    expect(artifactsMocks.list).toHaveBeenCalledTimes(2);
    await waitFor(() => {
      expect(screen.queryByText("solubility-histogram.png")).toBeNull();
    });
  });

  it("an already-gone row (404) reconciles the registry honestly", async () => {
    artifactsMocks.delete.mockRejectedValueOnce(
      new ApiError(404, { detail: "Unknown artifact_id." }),
    );
    render(<VisualizationPage />);
    await screen.findByText("solubility-histogram.png");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete artifact solubility-histogram.png" }),
    );
    await screen.findByText(/Delete artifact “solubility-histogram.png”\?/);
    await userEvent.click(screen.getByRole("button", { name: "Delete artifact" }));

    expect(
      await screen.findByText(
        /"solubility-histogram.png" was already gone \(404\) — registry refreshed\./,
      ),
    ).toBeInTheDocument();
    expect(artifactsMocks.list).toHaveBeenCalledTimes(2);
  });

  it("every other failure stays on the confirmation card verbatim", async () => {
    artifactsMocks.delete.mockRejectedValueOnce(
      new ApiError(500, { detail: "vault unreadable" }),
    );
    render(<VisualizationPage />);
    await screen.findByText("solubility-histogram.png");

    await userEvent.click(
      screen.getByRole("button", { name: "Delete artifact solubility-histogram.png" }),
    );
    await screen.findByText(/Delete artifact “solubility-histogram.png”\?/);
    await userEvent.click(screen.getByRole("button", { name: "Delete artifact" }));

    // the gate stays armed with the typed error surfaced verbatim
    const banners = await screen.findAllByRole("alert");
    expect(
      banners.some((b) => b.textContent?.includes("vault unreadable")),
    ).toBe(true);
    expect(
      screen.getByText(/Delete artifact “solubility-histogram.png”\?/),
    ).toBeInTheDocument();
    expect(artifactsMocks.list).toHaveBeenCalledTimes(1);
  });
});
