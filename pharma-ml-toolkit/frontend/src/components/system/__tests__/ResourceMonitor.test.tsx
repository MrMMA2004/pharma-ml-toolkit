/**
 * part16 §8 — the floating Resource Monitor panel's own contract.
 *
 * The interaction vocabulary of the directive, one lane per clause:
 * compact summary + detailed view, collapsible/expandable, movable by
 * dragging (and by keyboard arrows), resizable within the clamped
 * limits, openable/closable through the persisted visible preference,
 * position + preferred size remembered across remounts. The metrics
 * vocabulary: system-level and application-process clearly separated,
 * GPU degrading gracefully to its honest "not available" line, absent
 * stdlib-fallback numbers rendered as the dash — NEVER simulated —
 * and the polling cadence read from the settings store's interval
 * (the §10 preference the settings panel already owns).
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const resourcesMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/system/client", () => ({
  systemApi: { resources: () => resourcesMock() },
}));

import { ResourceMonitorHost } from "@/components/system/ResourceMonitor";
import { getSettingsStore } from "@/services/settings/store";
import {
  PANEL_MAX_WIDTH,
  PANEL_MIN_WIDTH,
  PANEL_STORAGE_KEY,
} from "@/services/system/panelState";
import type { ResourceSnapshot } from "@/services/system/types";

const DISK = {
  path: "/app/pharma-ml-toolkit",
  total: 50_000_000_000,
  used: 28_000_000_000,
  free: 22_000_000_000,
};

function psutilSnapshot(): ResourceSnapshot {
  return {
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
      disk: DISK,
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
  };
}

function fallbackSnapshot(): ResourceSnapshot {
  return {
    source: "stdlib-fallback",
    backend: { name: "pharmaml", version: "1.2.240" },
    system: {
      cpu_count: 8,
      cpu_percent: null,
      loadavg: [0.9, 0.7, 0.5],
      memory: null,
      swap: null,
      disk: DISK,
      notes: [
        "memory, swap and cpu_percent are absent without psutil — nulled, never fabricated",
      ],
    },
    process: {
      pid: 12345,
      cpu_percent: null,
      memory_rss_bytes: 260_000_000,
      num_threads: null,
    },
    jobs: {
      active: 0,
      by_state: {
        QUEUED: 0,
        RUNNING: 0,
        CANCELLATION_REQUESTED: 0,
        CANCELLED: 0,
        COMPLETED: 0,
        FAILED: 0,
      },
    },
    gpu: { available: false, note: "GPU monitoring is deliberately absent" },
  };
}

beforeEach(() => {
  window.localStorage.clear();
  getSettingsStore().reset();
  resourcesMock.mockReset().mockResolvedValue(psutilSnapshot());
});

afterEach(() => {
  vi.useRealTimers();
});

async function renderPanel() {
  render(<ResourceMonitorHost />);
  const panel = await screen.findByRole("complementary", {
    name: "Resource monitor",
  });
  return panel;
}

describe("the floating Resource Monitor (§8)", () => {
  it("mounts visible by default with the compact summary from the first snapshot", async () => {
    const panel = await renderPanel();

    expect(resourcesMock).toHaveBeenCalledTimes(1);
    expect(panel).toBeInTheDocument();
    // the compact summary: CPU, RAM, active jobs, backend version
    expect(screen.getByText("CPU 23.5%")).toBeInTheDocument();
    expect(screen.getByText("RAM 61.2%")).toBeInTheDocument();
    expect(screen.getByText("Jobs 2")).toBeInTheDocument();
    expect(screen.getByText("v1.2.240")).toBeInTheDocument();
    // compact means compact: the detailed sections are absent
    expect(
      screen.queryByRole("region", { name: "System — host" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Application — backend process" }),
    ).not.toBeInTheDocument();
  });

  it("expands into the two clearly separated metric sections", async () => {
    await renderPanel();

    await userEvent.click(screen.getByRole("button", { name: "Toggle detail" }));

    const system = screen.getByRole("region", { name: "System — host" });
    const application = screen.getByRole("region", {
      name: "Application — backend process",
    });
    expect(system).toBeInTheDocument();
    expect(application).toBeInTheDocument();
    // system-level vocabulary
    expect(screen.getByText("CPU cores")).toBeInTheDocument();
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getByText("1.1 / 0.8 / 0.6")).toBeInTheDocument();
    expect(screen.getByText(/9.1 GB \/ 14.9 GB/)).toBeInTheDocument();
    // application-process vocabulary
    expect(screen.getByText("Memory (RSS)")).toBeInTheDocument();
    expect(screen.getByText("238.4 MB")).toBeInTheDocument();
    expect(screen.getByText("Threads")).toBeInTheDocument();
    expect(screen.getByText("18")).toBeInTheDocument();
    // the jobs-by-state rows ride the application section
    expect(application.textContent).toContain("QUEUED1");
    expect(application.textContent).toContain("RUNNING1");
    // GPU degrades gracefully to the honest line
    expect(
      screen.getByText("GPU — not available (CPU-only operation supported)"),
    ).toBeInTheDocument();
    // collapse again — the sections fold away
    await userEvent.click(screen.getByRole("button", { name: "Toggle detail" }));
    expect(
      screen.queryByRole("region", { name: "System — host" }),
    ).not.toBeInTheDocument();
  });

  it("renders the honest dash for the stdlib-fallback absent numbers — never zero", async () => {
    resourcesMock.mockResolvedValue(fallbackSnapshot());
    const panel = await renderPanel();

    // CPU percent is absent: the dash, not a fabricated 0.0%
    expect(screen.getByText("CPU —")).toBeInTheDocument();
    expect(screen.getByText("RAM —")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Toggle detail" }));
    expect(panel.textContent).toContain("nulled, never fabricated");
    // the source chip states the degraded origin
    expect(screen.getByText("stdlib-fallback")).toBeInTheDocument();
  });

  it("renders the honest unavailable state when the fetch fails — no fabricated numbers", async () => {
    resourcesMock.mockRejectedValue(
      new Error("the backend endpoint is not configured yet"),
    );
    const panel = await renderPanel();

    expect(
      screen.getByText("Resource snapshot unavailable"),
    ).toBeInTheDocument();
    expect(panel.textContent).toContain("the backend endpoint is not configured yet");
    // nothing simulated: no percentage numbers anywhere
    expect(panel.textContent?.includes("%")).toBe(false);
  });

  it("polls at the settings interval and stops entirely once closed", async () => {
    vi.useFakeTimers();
    render(<ResourceMonitorHost />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByRole("complementary", { name: "Resource monitor" })).toBeInTheDocument();
    expect(resourcesMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    expect(resourcesMock).toHaveBeenCalledTimes(2);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    expect(resourcesMock).toHaveBeenCalledTimes(3);

    // the interval preference re-keys the cadence live (§10's setting)
    act(() => {
      getSettingsStore().update({ resourceMonitor: { updateIntervalMs: 500 } });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const afterRekey = resourcesMock.mock.calls.length;
    expect(afterRekey).toBeGreaterThanOrEqual(4);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(resourcesMock.mock.calls.length).toBeGreaterThan(afterRekey);

    // closing stops every cadence — no background polling while closed
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    const closedBaseline = resourcesMock.mock.calls.length;
    fireEvent.click(
      screen.getByRole("button", { name: "Close resource monitor" }),
    );
    expect(
      screen.queryByRole("complementary", { name: "Resource monitor" }),
    ).not.toBeInTheDocument();
    expect(getSettingsStore().snapshot().resourceMonitor.visible).toBe(false);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    expect(resourcesMock.mock.calls.length).toBe(closedBaseline);
  });

  it("moves by dragging the header strip and remembers the position", async () => {
    const panel = await renderPanel();
    const strip = screen.getByTestId("monitor-drag");

    // the default pocket: x = innerWidth - 320 - 16 (jsdom: 1024 -> 688)
    expect(panel.style.left).toBe("688px");
    expect(panel.style.top).toBe("88px");

    fireEvent.pointerDown(strip, { clientX: 698, clientY: 98 });
    fireEvent.pointerMove(document.documentElement, {
      clientX: 718,
      clientY: 138,
    });
    fireEvent.pointerUp(document.documentElement, {
      clientX: 718,
      clientY: 138,
    });

    // dragged by (+20, +40) and clamped live
    expect(panel.style.left).toBe("708px");
    expect(panel.style.top).toBe("128px");
    // "remembers its last position" — persisted to the pocket
    const saved = JSON.parse(
      window.localStorage.getItem(PANEL_STORAGE_KEY) ?? "{}",
    );
    expect(saved.x).toBe(708);
    expect(saved.y).toBe(128);
  });

  it("moves by keyboard arrows on the drag strip (the accessible move)", async () => {
    const panel = await renderPanel();
    const strip = screen.getByTestId("monitor-drag");
    strip.focus();

    await userEvent.keyboard("{ArrowLeft}");
    expect(panel.style.left).toBe("678px");
    await userEvent.keyboard("{ArrowDown}");
    expect(panel.style.top).toBe("98px");
    const saved = JSON.parse(
      window.localStorage.getItem(PANEL_STORAGE_KEY) ?? "{}",
    );
    expect(saved.x).toBe(678);
    expect(saved.y).toBe(98);
  });

  it("resizes within the clamped limits and the size survives a remount", async () => {
    const first = render(<ResourceMonitorHost />);
    const panel = await screen.findByRole("complementary", {
      name: "Resource monitor",
    });
    const handle = screen.getByTestId("monitor-resize");

    fireEvent.pointerDown(handle, { clientX: 1008, clientY: 200 });
    fireEvent.pointerMove(document.documentElement, {
      clientX: 1308,
      clientY: 200,
    });
    fireEvent.pointerUp(document.documentElement, {
      clientX: 1308,
      clientY: 200,
    });

    // +300 wanted over the limit — clamped at the max width
    expect(panel.style.width).toBe(`${PANEL_MAX_WIDTH}px`);
    const saved = JSON.parse(
      window.localStorage.getItem(PANEL_STORAGE_KEY) ?? "{}",
    );
    expect(saved.width).toBe(PANEL_MAX_WIDTH);

    // "remembers its preferred size" across a full remount
    first.unmount();
    render(<ResourceMonitorHost />);
    const remounted = await screen.findByRole("complementary", {
      name: "Resource monitor",
    });
    expect(remounted.style.width).toBe(`${PANEL_MAX_WIDTH}px`);
  });

  it("keeps the width inside the limits on a keyboard resize", async () => {
    const panel = await renderPanel();
    const handle = screen.getByTestId("monitor-resize");
    handle.focus();

    await userEvent.keyboard("{ArrowRight}");
    // the keyboard lane also honors the clamp (320 default -> 330)
    expect(panel.style.width).toBe("330px");
    for (let i = 0; i < 30; i += 1) {
      await userEvent.keyboard("{ArrowRight}");
    }
    expect(parseInt(panel.style.width, 10)).toBe(PANEL_MAX_WIDTH);
    await userEvent.keyboard("{ArrowLeft}");
    expect(parseInt(panel.style.width, 10)).toBe(PANEL_MAX_WIDTH - 10);
    expect(PANEL_MIN_WIDTH).toBe(280);
    expect(PANEL_MAX_WIDTH).toBe(480);
  });

  it("toggles open/closed through the persisted visible preference", async () => {
    await renderPanel();
    const entry = screen.getByRole("button", { name: "Resource monitor" });

    expect(entry).toHaveAttribute("aria-expanded", "true");
    await userEvent.click(entry);
    expect(
      screen.queryByRole("complementary", { name: "Resource monitor" }),
    ).not.toBeInTheDocument();
    expect(entry).toHaveAttribute("aria-expanded", "false");
    expect(getSettingsStore().snapshot().resourceMonitor.visible).toBe(false);

    await userEvent.click(entry);
    await screen.findByRole("complementary", { name: "Resource monitor" });
    expect(entry).toHaveAttribute("aria-expanded", "true");
  });

  it("closes on Escape while the panel holds focus — but not for a page-wide Escape", async () => {
    await renderPanel();
    const strip = screen.getByTestId("monitor-drag");

    // an Escape aimed elsewhere (the page, a dialog) never sweeps the
    // floating panel away
    await userEvent.keyboard("{Escape}");
    expect(
      screen.getByRole("complementary", { name: "Resource monitor" }),
    ).toBeInTheDocument();

    // the panel holds focus -> its own Escape closes it
    strip.focus();
    await userEvent.keyboard("{Escape}");
    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: "Resource monitor" }),
      ).not.toBeInTheDocument(),
    );
    expect(getSettingsStore().snapshot().resourceMonitor.visible).toBe(false);
  });
});
