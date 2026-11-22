import { beforeEach, describe, expect, it, vi } from "vitest";

import { initializeBackend, StartupError } from "@/services/api/startup";
import { ApiError } from "@/services/api/client";

/**
 * part14 Phase 4 — the D-3 regression tests (§3.1 item 12: errors are
 * reported accurately, never mislabeled as backend startup failures).
 *
 * The health probe's rejection is classified by its real cause —
 *   auth      — a 401/403 answer: the backend is UP and rejected the
 *               token/origin (the packaged app's origin incident class);
 *   network   — a fetch TypeError: unreachable or CORS-blocked;
 *   http      — another non-ok HTTP status;
 *   protocol  — anything else (including an ok-shaped but wrong
 *               payload).
 *
 * Everything is mocked at the network boundary (invoke + api.get),
 * exactly like the StartupGate lane; the classification itself — the
 * code the packaged frontend runs — is real.
 */

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

vi.mock("@/services/api/client", async (importOriginal) => {
  // keep the REAL ApiError (the instanceof arm must classify the real
  // client's error class) and stub only the api surface
  const actual = await importOriginal<typeof import("@/services/api/client")>();
  return { ...actual, api: { get: vi.fn() } };
});

vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => ({ connect: vi.fn() }),
}));

import { invoke } from "@tauri-apps/api/core";
import { api } from "@/services/api/client";

const mockedInvoke = vi.mocked(invoke);
const mockedGet = vi.mocked(api.get);

const HANDSHAKE = {
  base_url: "http://127.0.0.1:54127",
  token: "test-session-token-0123456789",
  version: "1.2.240",
};

beforeEach(() => {
  // the Tauri runtime flag is checked at call time — the packaged
  // app's lane (the IPC acquire channel) is the one under test
  (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
  mockedInvoke.mockResolvedValue({ ...HANDSHAKE });
  mockedGet.mockReset();
});

async function failure(): Promise<StartupError> {
  try {
    await initializeBackend();
    throw new Error("initializeBackend unexpectedly resolved");
  } catch (err) {
    if (err instanceof StartupError) {
      return err;
    }
    throw err;
  }
}

describe("initializeBackend — the D-3 honest health-failure classification", () => {
  it("classifies a 403 as an authorization rejection, not a startup failure", async () => {
    mockedGet.mockRejectedValueOnce(
      new ApiError(403, { detail: "Non-loopback Origin rejected." }),
    );
    const err = await failure();
    expect(err.stage).toBe("health");
    expect(err.cause).toBe("auth");
    expect(err.message).toContain("HTTP 403");
    expect(err.message).toContain("Non-loopback Origin rejected.");
    expect(err.message).toContain("not a backend startup failure");
  });

  it("classifies a 401 as an authorization rejection", async () => {
    mockedGet.mockRejectedValueOnce(
      new ApiError(401, { detail: "Invalid or missing session token." }),
    );
    const err = await failure();
    expect(err.cause).toBe("auth");
    expect(err.message).toContain("HTTP 401");
  });

  it("classifies the browser's CORS/network block (Failed to fetch) as a reachability failure", async () => {
    mockedGet.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const err = await failure();
    expect(err.cause).toBe("network");
    expect(err.message).toContain("could not reach the backend");
    expect(err.message).toContain("Failed to fetch");
    expect(err.message).toContain("not established");
  });

  it("classifies a plain message mentioning fetch as a reachability failure too", async () => {
    mockedGet.mockRejectedValueOnce(new Error("NetworkError when attempting to fetch resource."));
    const err = await failure();
    expect(err.cause).toBe("network");
  });

  it("classifies another HTTP status as an honest unhealthy answer", async () => {
    mockedGet.mockRejectedValueOnce(new ApiError(500, null));
    const err = await failure();
    expect(err.cause).toBe("http");
    expect(err.message).toContain("HTTP 500");
    expect(err.message).toContain("answered, but not with ok");
  });

  it("classifies an unexpected error shape as a protocol failure", async () => {
    mockedGet.mockRejectedValueOnce(new Error("some odd library error"));
    const err = await failure();
    expect(err.cause).toBe("protocol");
    expect(err.message).toContain("some odd library error");
  });

  it("classifies a 200 answer whose payload is not ok as a protocol failure", async () => {
    mockedGet.mockResolvedValueOnce({
      status: "degraded",
      version: "1.2.240",
      scientific_core: true,
    });
    const err = await failure();
    expect(err.stage).toBe("health");
    expect(err.cause).toBe("protocol");
    expect(err.message).toContain("status=degraded");
  });

  it("keeps the acquire-stage failures on the handshake cause (backward compatibility)", async () => {
    mockedInvoke.mockRejectedValueOnce(new Error("the sidecar exited before the handshake"));
    const err = await failure();
    expect(err.stage).toBe("acquire");
    expect(err.cause).toBe("handshake");
    expect(err.message).toContain("backend_endpoint bridge failed");
  });

  it("resolves through the full sequence when the backend is healthy", async () => {
    mockedGet.mockResolvedValueOnce({
      status: "ok",
      version: "1.2.240",
      scientific_core: true,
    });
    const info = await initializeBackend();
    expect(info.source).toBe("tauri-ipc");
    expect(info.version).toBe("1.2.240");
    expect(info.scientificCore).toBe(true);
  });
});
