import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import type { StartupInfo } from "@/services/api/startup";

/**
 * part13 Phase 3 — the shell navigation lane.
 *
 * The App component's own contract is under test: the §10.1 nav strip
 * renders the nine registry entries, Introduction is the first page,
 * clicking a nav entry mounts exactly that workspace's page, and the
 * active state follows aria-current. The nine workspace pages are
 * mocked as one-line stubs so this file tests the SHELL, not 5,000
 * lines of workspace internals (each page's own lane lives beside it).
 */

vi.mock("@/services/api/startup", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api/startup")>();
  return { ...actual, initializeBackend: vi.fn() };
});

const idleStream = {
  status: { state: "idle", lastSequence: -1, attempts: 0, hello: null },
  onStateChange: () => () => undefined,
};
vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => idleStream,
}));

vi.mock("@/features/introduction/IntroductionPage", () => ({
  IntroductionPage: () => <div data-testid="page-introduction" />,
}));
vi.mock("@/features/datasets/DatasetsPage", () => ({
  DatasetsPage: () => <div data-testid="page-datasets" />,
}));
vi.mock("@/features/ml/MLPage", () => ({
  MLPage: () => <div data-testid="page-machine-learning" />,
}));
vi.mock("@/features/dl/DLPage", () => ({
  DLPage: () => <div data-testid="page-deep-learning" />,
}));
vi.mock("@/features/generative/GenerativePage", () => ({
  GenerativePage: () => <div data-testid="page-generative-ai" />,
}));
vi.mock("@/features/docking/DockingPage", () => ({
  DockingPage: () => <div data-testid="page-molecular-docking" />,
}));
vi.mock("@/features/pkpd/PKPDPage", () => ({
  PKPDPage: () => <div data-testid="page-pkpd" />,
}));
vi.mock("@/features/md/MDPage", () => ({
  MDPage: () => <div data-testid="page-molecular-dynamics" />,
}));
vi.mock("@/features/visualization/VisualizationPage", () => ({
  VisualizationPage: () => <div data-testid="page-visualization" />,
}));

const { initializeBackend } = await import("@/services/api/startup");
const READY_INFO: StartupInfo = {
  source: "dev-same-origin",
  version: "1.2.238",
  scientificCore: true,
};
vi.mocked(initializeBackend).mockResolvedValue(READY_INFO);

describe("the application shell (§10)", () => {
  it("renders Introduction first with the nine-entry nav in §10.1 order", async () => {
    render(<App />);

    await screen.findByTestId("page-introduction");
    const nav = screen.getByRole("navigation", { name: "Workspaces" });
    const entries = nav.querySelectorAll("button");
    expect(entries).toHaveLength(9);
    expect([...entries].map((b) => b.textContent)).toEqual([
      "Introduction",
      "Datasets",
      "Machine Learning",
      "Deep Learning",
      "Generative AI",
      "Molecular Docking",
      "PK/PD",
      "Molecular Dynamics Analysis",
      "Visualization",
    ]);
    expect(screen.getByRole("button", { name: "Introduction" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("switches the mounted page on a nav click — one page at a time", async () => {
    render(<App />);
    await screen.findByTestId("page-introduction");

    await userEvent.click(screen.getByRole("button", { name: "Datasets" }));
    expect(screen.getByTestId("page-datasets")).toBeInTheDocument();
    expect(screen.queryByTestId("page-introduction")).not.toBeInTheDocument();
    // the active state followed the selection
    expect(screen.getByRole("button", { name: "Datasets" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("button", { name: "Introduction" })).not.toHaveAttribute(
      "aria-current",
    );

    // and back — navigation is lossless in both directions
    await userEvent.click(screen.getByRole("button", { name: "Introduction" }));
    expect(screen.getByTestId("page-introduction")).toBeInTheDocument();
    expect(screen.queryByTestId("page-datasets")).not.toBeInTheDocument();
  });

  it("reaches the last workspace in the strip (Visualization) through keyboard focus + click", async () => {
    render(<App />);
    await screen.findByTestId("page-introduction");

    // the strip is horizontally scrollable, never truncating entries
    const visualization = screen.getByRole("button", { name: "Visualization" });
    await userEvent.click(visualization);
    expect(screen.getByTestId("page-visualization")).toBeInTheDocument();
  });
});
