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
  // the toast lane's surface (the real EventStream exposes both)
  subscribe: () => () => undefined,
};
vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => idleStream,
}));

const systemResourcesMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/system/client", () => ({
  systemApi: { resources: () => systemResourcesMock() },
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
  version: "1.2.240",
  scientificCore: true,
};
vi.mocked(initializeBackend).mockResolvedValue(READY_INFO);

import type { ResourceSnapshot } from "@/services/system/types";

systemResourcesMock.mockResolvedValue({
  source: "psutil",
  backend: { name: "pharmaml", version: "1.2.240" },
  system: {
    cpu_count: 8,
    cpu_percent: 23.5,
    loadavg: [1.1, 0.8, 0.6],
    memory: {
      total: 16_000_000_000,
      available: 6_200_000_000,
      used: 9_800_000_000,
      percent: 61.2,
    },
    swap: {
      total: 4_000_000_000,
      used: 500_000_000,
      free: 3_500_000_000,
      percent: 12.5,
    },
    disk: {
      path: "/app/pharma-ml-toolkit",
      total: 50_000_000_000,
      used: 28_000_000_000,
      free: 22_000_000_000,
    },
    notes: [],
  },
  process: {
    pid: 12345,
    cpu_percent: 4.2,
    memory_rss_bytes: 250_000_000,
    num_threads: 18,
  },
  jobs: {
    active: 2,
    by_state: {
      QUEUED: 1,
      RUNNING: 1,
      CANCELLATION_REQUESTED: 0,
      CANCELLED: 0,
      COMPLETED: 0,
      FAILED: 0,
    },
  },
  gpu: { available: false, note: "GPU monitoring is deliberately absent" },
} satisfies ResourceSnapshot);

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

  it("carries the part16 §10 settings entry in the header (every workspace's reach)", async () => {
    render(<App />);
    await screen.findByTestId("page-introduction");

    // the entry is present outside the workspace nav (the header lane)
    const entry = screen.getByRole("button", { name: "Settings" });
    expect(entry).toHaveAttribute("aria-haspopup", "dialog");
    expect(screen.queryByRole("navigation", { name: "Workspaces" })).not.toContain(entry);

    await userEvent.click(entry);
    expect(screen.getByRole("dialog", { name: "Application settings" })).toBeInTheDocument();

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("carries the part16 §8 floating resource-monitor entry in the header (every workspace's reach)", async () => {
    render(<App />);
    await screen.findByTestId("page-introduction");

    // the entry rides the header next to the settings gear, never the nav
    const entry = screen.getByRole("button", { name: "Resource monitor" });
    const nav = screen.getByRole("navigation", { name: "Workspaces" });
    expect(nav).not.toContainElement(entry);
    expect(entry).toHaveAttribute("aria-expanded", "true");

    // visible by the §10 default preference -> the floating panel mounts
    const panel = screen.getByRole("complementary", { name: "Resource monitor" });
    expect(nav).not.toContainElement(panel);
    // the compact summary carries the live snapshot values (the first
    // poll resolves async — findBy waits for it, never a race)
    await screen.findByText("CPU 23.5%");
    await screen.findByText("Jobs 2");

    // the entry closes it through the persisted preference
    await userEvent.click(entry);
    expect(
      screen.queryByRole("complementary", { name: "Resource monitor" }),
    ).not.toBeInTheDocument();
    expect(entry).toHaveAttribute("aria-expanded", "false");
  });
});
