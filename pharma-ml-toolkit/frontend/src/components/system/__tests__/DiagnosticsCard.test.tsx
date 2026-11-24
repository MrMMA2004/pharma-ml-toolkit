/**
 * part16 §15.5 (Phase 12) — the Diagnostics card's own contract.
 *
 * The §15.5 surface as a component-level user flow:
 *
 *   the load          — the mount fetches through the service and
 *                       renders the backend identity, the scientific
 *                       tools with their probe results, the storage
 *                       health, the dependency table under its
 *                       details, the crash rows, the log config;
 *   the broken rows   — an unimportable dependency renders its
 *                       honest error, an absent tool renders the
 *                       dash, never a fabricated version;
 *   the cleanup        — the click POSTs through the service and
 *                       renders the cleared count with the guarantee;
 *   the failure lane   — a rejected fetch surfaces the error banner;
 *   the crash rows     — the seeded failure renders verbatim, and
 *                       the empty state says so honestly;
 *   the mutation rule  — the ONLY mutation control is the safe
 *                       cache cleanup; no other control exists.
 *
 * The system service is mocked at the module boundary only; the
 * component, its state machine, and the DOM are the real ones.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const diagnosticsMock = vi.hoisted(() => vi.fn());
const cleanupMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/system/client", () => ({
  systemApi: {
    diagnostics: diagnosticsMock,
    cleanupCache: cleanupMock,
  },
}));

import { DiagnosticsCard } from "@/components/system/DiagnosticsCard";
import type { DiagnosticsSnapshot } from "@/services/system/client";

function snapshot(
  overrides: Partial<DiagnosticsSnapshot> = {},
): DiagnosticsSnapshot {
  return {
    backend: { name: "pharmaml", version: "1.2.240", python: "3.12.16" },
    dependencies: [
      {
        name: "numpy",
        role: "the numeric core (the 1.26.4 pin)",
        tier: "required",
        importable: true,
        version: "1.26.4",
        error: null,
      },
      {
        name: "mordred",
        role: "the descriptor engine",
        tier: "optional",
        importable: false,
        version: null,
        error: "ModuleNotFoundError",
      },
    ],
    scientific_tools: [
      {
        name: "rdkit",
        version: "2026.9.1",
        importable: true,
        functional_probe: true,
        probe: "Chem.MolFromSmiles('CCO') is not None",
      },
      {
        name: "torch",
        version: null,
        importable: false,
        functional_probe: null,
        probe: "torch.tensor([1,2]) + torch.tensor([1,1]) == [2,3]",
      },
    ],
    storage: {
      persistent: true,
      data_dir: "/home/z/.local/share/pharmaml-studio",
      reason: null,
      disk: {
        path: "/app",
        total: 50_000_000_000,
        used: 28_000_000_000,
        free: 22_000_000_000,
      },
      database: {
        file: "studio.db",
        size_bytes: 94_208,
        schema_version: 3,
        journal_mode: "wal",
      },
      tables: { datasets: 2, jobs: 1 },
      vault_bytes: 4_096,
      integrity: "ok",
    },
    crash: {
      source: "sqlite job history (the FAILED rows survive restarts)",
      recent_failed: [
        {
          job_id: "job-abc123",
          operation: "ml.train",
          workspace: "machine-learning",
          error: "RuntimeError: the seeded failure",
          completed_at: "2026-11-24T08:00:00Z",
        },
      ],
    },
    logs: {
      level: "WARNING",
      handlers: [{ kind: "console", path: null }],
      log_file: null,
      note: "read-only configuration report",
    },
    recovery: {
      persisted_by_state: { FAILED: 1 },
      in_session_by_state: { QUEUED: 0, RUNNING: 0, COMPLETED: 0, FAILED: 0 },
      source: "sqlite job history (survives the process)",
      note: "the two views never pretend to be one",
    },
    scope: {
      implemented: ["dependency availability checks"],
      not_implemented: ["automatic job resume after a crash"],
      note: "Do not delete user data during diagnostic or cleanup operations.",
    },
    ...overrides,
  };
}

describe("DiagnosticsCard", () => {
  beforeEach(() => {
    diagnosticsMock.mockReset();
    cleanupMock.mockReset();
  });

  it("loads the snapshot on mount and renders the identity + tools", async () => {
    diagnosticsMock.mockResolvedValue(snapshot());
    render(<DiagnosticsCard />);
    await waitFor(() => {
      expect(diagnosticsMock).toHaveBeenCalledTimes(1);
    });
    expect(await screen.findByText("1.2.240")).toBeInTheDocument();
    expect(screen.getByText(/Python 3\.12\.16/)).toBeInTheDocument();
    // the tool rows: the probe result + the version or the absent dash
    expect(screen.getByText("rdkit")).toBeInTheDocument();
    expect(screen.getByText("2026.9.1")).toBeInTheDocument();
    expect(screen.getByText(/absent/i)).toBeInTheDocument();
    expect(screen.getByText(/Functional probes:/)).toBeInTheDocument();
  });

  it("renders the storage health with the durable numbers", async () => {
    diagnosticsMock.mockResolvedValue(snapshot());
    render(<DiagnosticsCard />);
    await screen.findByText(/20\.49 GB/);
    expect(screen.getByText(/46\.57 GB/)).toBeInTheDocument();
    expect(screen.getByText(/92\.0 kB/)).toBeInTheDocument();
    expect(screen.getByText(/ok/)).toBeInTheDocument();
    expect(screen.getByText(/4\.0 kB/)).toBeInTheDocument();
  });

  it("renders the honest ephemeral storage line", async () => {
    diagnosticsMock.mockResolvedValue(
      snapshot({
        storage: {
          persistent: false,
          data_dir: null,
          reason: "ephemeral session — no data_dir was configured",
          disk: {
            path: "/app",
            total: 50_000_000_000,
            used: 28_000_000_000,
            free: 22_000_000_000,
          },
          database: null,
          tables: null,
          vault_bytes: null,
          integrity: null,
        },
      }),
    );
    render(<DiagnosticsCard />);
    expect(
      await screen.findByText(/Ephemeral session — no data_dir/i),
    ).toBeInTheDocument();
  });

  it("opens the dependency table and renders the broken row honestly", async () => {
    diagnosticsMock.mockResolvedValue(snapshot());
    render(<DiagnosticsCard />);
    const summary = await screen.findByText(
      /The dependency availability table \(2 probes\)/i,
    );
    await userEvent.click(summary);
    expect(screen.getByText("numpy")).toBeInTheDocument();
    expect(screen.getByText("mordred")).toBeInTheDocument();
    expect(
      screen.getByText(/unavailable \(ModuleNotFoundError\)/i),
    ).toBeInTheDocument();
  });

  it("renders the crash rows verbatim and the empty state honestly", async () => {
    diagnosticsMock.mockResolvedValue(snapshot());
    render(<DiagnosticsCard />);
    expect(
      await screen.findByText(/RuntimeError: the seeded failure/),
    ).toBeInTheDocument();
    expect(screen.getByText(/job-abc123/)).toBeInTheDocument();

    diagnosticsMock.mockResolvedValue(
      snapshot({
        crash: {
          source: "sqlite job history",
          recent_failed: [],
        },
      }),
    );
    render(<DiagnosticsCard />);
    expect(
      await screen.findAllByText(/No failed jobs recorded/i),
    ).toHaveLength(1);
  });

  it("runs the safe cleanup and renders the guarantee with the count", async () => {
    diagnosticsMock.mockResolvedValue(snapshot());
    cleanupMock.mockResolvedValue({
      cleared: { samples_stats_entries: 17 },
      preserved: ["datasets", "trained models", "the job history"],
      note: "Do not delete user data during diagnostic or cleanup operations.",
    });
    render(<DiagnosticsCard />);
    await screen.findByText("1.2.240");
    await userEvent.click(
      screen.getByRole("button", { name: /Clear the derived cache/i }),
    );
    await waitFor(() => {
      expect(cleanupMock).toHaveBeenCalledTimes(1);
    });
    expect(
      await screen.findByText(/Cleared 17 cached sample stats/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/user data preserved/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Do not delete user data during diagnostic/i),
    ).toBeInTheDocument();
  });

  it("surfaces a rejected fetch through the error banner", async () => {
    diagnosticsMock.mockRejectedValue(new Error("backend unreachable"));
    render(<DiagnosticsCard />);
    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toContain("Diagnostics failed:");
    expect(banner.textContent).toContain("backend unreachable");
  });

  it("offers exactly one mutation control — the safe cleanup", async () => {
    diagnosticsMock.mockResolvedValue(snapshot());
    render(<DiagnosticsCard />);
    await screen.findByText("1.2.240");
    const buttons = screen.getAllByRole("button");
    // Refresh + Clear derived cache — read-only diagnostics plus the
    // single §15.5 cleanup lane, nothing else mutates anything
    expect(buttons).toHaveLength(2);
    expect(
      screen.getByRole("button", { name: /Clear the derived cache/i }),
    ).toBeInTheDocument();
  });
});
