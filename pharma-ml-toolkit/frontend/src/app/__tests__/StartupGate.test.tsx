import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { StartupGate } from "@/app/StartupGate";
import type { StartupInfo } from "@/services/api/startup";

/**
 * part13 Phase 3 — the §14.1 startup handshake's three honest states
 * plus the §14.2 controlled restart:
 *
 *   loading  — the connecting card while the handshake is in flight;
 *   failure  — the typed stage + message verbatim + a Retry button
 *             (NEVER a silent fake or a page reload);
 *   success  — the workspaces mount, the sr-only ready marker for
 *             screen readers, the WS banner stays silent while idle;
 *   restart  — Retry re-arms the connecting state from the click
 *             handler and a second attempt can succeed.
 *
 * The service is mocked exactly at the network boundary
 * (`initializeBackend`); everything below it — the gate's own state
 * machine, the banner, the DOM — is the real component.
 */

vi.mock("@/services/api/startup", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api/startup")>();
  return { ...actual, initializeBackend: vi.fn() };
});

// The Tauri host wires the backend before the workspaces mount; tests
// provide the same wiring through a controllable fake stream.
const stateListeners = new Set<(status: unknown) => void>();
const fakeStream = {
  status: { state: "idle", lastSequence: -1, attempts: 0, hello: null },
  onStateChange(cb: (status: unknown) => void) {
    stateListeners.add(cb);
    return () => {
      stateListeners.delete(cb);
    };
  },
};
vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => fakeStream,
}));

const { initializeBackend, StartupError } = await import("@/services/api/startup");
const mockBoot = vi.mocked(initializeBackend);

const READY_INFO: StartupInfo = {
  source: "dev-same-origin",
  version: "1.2.240",
  scientificCore: true,
};

function emitStreamState(status: unknown) {
  act(() => {
    fakeStream.status = status as typeof fakeStream.status;
    for (const listener of stateListeners) {
      listener(status);
    }
  });
}

describe("StartupGate — the §14.1 handshake states", () => {
  it("shows the honest connecting card while the handshake is in flight", async () => {
    let release: (() => void) | undefined;
    mockBoot.mockImplementationOnce(
      () => new Promise<StartupInfo>((resolve) => { release = () => resolve(READY_INFO); }),
    );
    render(
      <StartupGate>
        <div>workspace-shell</div>
      </StartupGate>,
    );

    expect(screen.getByText("Starting the local scientific backend")).toBeInTheDocument();
    // the screen-reader status lane (part13's a11y bar)
    expect(screen.getByRole("status")).toHaveTextContent("Backend startup in progress");
    expect(screen.queryByText("workspace-shell")).not.toBeInTheDocument();

    release?.();
    await screen.findByText("workspace-shell");
  });

  it("mounts the workspaces with the sr-only ready marker once healthy", async () => {
    mockBoot.mockResolvedValueOnce(READY_INFO);
    render(
      <StartupGate>
        <div>workspace-shell</div>
      </StartupGate>,
    );

    await screen.findByText("workspace-shell");
    const marker = screen.getByTestId("startup-ready");
    expect(marker).toHaveTextContent("backend 1.2.240 ready via dev-same-origin");
    // the WS banner stays silent while the stream is idle/healthy
    expect(screen.queryByText(/live event stream/)).not.toBeInTheDocument();
  });

  it("surfaces the typed failure stage + message verbatim with a Retry", async () => {
    mockBoot.mockRejectedValueOnce(new StartupError("health", "backend health probe failed: connection refused"));
    render(
      <StartupGate>
        <div>workspace-shell</div>
      </StartupGate>,
    );

    expect(await screen.findByText("The backend did not start")).toBeInTheDocument();
    expect(screen.getByText(/stage: health/)).toBeInTheDocument();
    expect(screen.getByText(/backend health probe failed: connection refused/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry the startup handshake" })).toBeInTheDocument();
    expect(screen.queryByText("workspace-shell")).not.toBeInTheDocument();
  });

  it("treats a non-StartupError rejection as an acquire-stage failure, not a crash", async () => {
    mockBoot.mockRejectedValueOnce(new Error("the sidecar exited before the handshake"));
    render(
      <StartupGate>
        <div>workspace-shell</div>
      </StartupGate>,
    );

    expect(await screen.findByText(/stage: acquire/)).toBeInTheDocument();
    expect(screen.getByText(/the sidecar exited before the handshake/)).toBeInTheDocument();
  });

  it("re-arms the handshake from the Retry click — the §14.2 controlled restart", async () => {
    mockBoot.mockRejectedValueOnce(new StartupError("acquire", "first attempt failed"));
    render(
      <StartupGate>
        <div>workspace-shell</div>
      </StartupGate>,
    );

    await screen.findByText("The backend did not start");
    mockBoot.mockResolvedValueOnce(READY_INFO);
    await userEvent.click(screen.getByRole("button", { name: "Retry the startup handshake" }));

    expect(await screen.findByText("workspace-shell")).toBeInTheDocument();
    expect(mockBoot).toHaveBeenCalledTimes(2);
  });
});

describe("StartupGate — the live-update health strip", () => {
  it("stays silent while the stream is open, and degrades visibly when it drops", async () => {
    mockBoot.mockResolvedValueOnce(READY_INFO);
    render(
      <StartupGate>
        <div>workspace-shell</div>
      </StartupGate>,
    );
    await screen.findByText("workspace-shell");
    // Drain the banner's passive subscription effect BEFORE any emission.
    // The runner-load flake (CI run 45056299925) exposed the interleaving
    // window: both store emissions landed before useSyncExternalStore had
    // attached its listener, the silence assertion could not catch the
    // loss, and the reconnecting assertion failed. An empty async act
    // flushes the pending effect queue, making the subscription live
    // deterministically first. The assertions below are unchanged —
    // they remain the spec.
    await act(async () => {});

    // healthy + open — the banner is absent by design
    emitStreamState({ state: "open", lastSequence: 3, attempts: 0, hello: null });
    expect(screen.queryByText(/live event stream/)).not.toBeInTheDocument();

    // a drop degrades to a visible, honest strip
    emitStreamState({ state: "reconnecting", lastSequence: 3, attempts: 2, hello: null });
    expect(screen.getByText("reconnecting the live event stream (attempt 2)")).toBeInTheDocument();

    // terminal close — the REST fallback notice, never a blocker
    emitStreamState({ state: "closed", lastSequence: 3, attempts: 2, hello: null });
    expect(screen.getByText(/job progress falls back to REST polling/)).toBeInTheDocument();
  });
});
