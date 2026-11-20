import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { EventStreamClient } from "@/services/websocket/client";

/**
 * part13 Phase 3 — the §13 event-stream engine, tested end to end
 * through a controllable fake socket at the exact network boundary.
 *
 * Everything under test is the REAL client: the hello handshake, the
 * replay cursor, duplicate-event protection, the exponential backoff,
 * the liveness dead-peer detection, graceful shutdown, and listener
 * isolation. Only the browser's WebSocket constructor is faked.
 */

interface Frame {
  event_id: string;
  sequence: number;
  job_id: string | null;
  event_type: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

function frame(sequence: number, event_type: string, payload: Record<string, unknown> = {}, event_id = `e${sequence}`): Frame {
  return { event_id, sequence, job_id: null, event_type, timestamp: "2026-11-21T00:00:00Z", payload };
}

function hello(last_sequence: number, replayed: number): string {
  return JSON.stringify(frame(-1, "heartbeat", { hello: true, last_sequence, replayed }, "hello-1"));
}

class FakeSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeSocket.CONNECTING;
  readonly url: string;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readonly sent: string[] = [];
  closeCalls: Array<{ code?: number; reason?: string }> = [];

  constructor(url: string) {
    this.url = url;
    sockets.push(this);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(code?: number, reason?: string): void {
    this.closeCalls.push({ code, reason });
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.();
  }

  serverSend(raw: string): void {
    this.readyState = FakeSocket.OPEN;
    this.onmessage?.({ data: raw });
  }

  serverHello(last = 0, replayed = 0): void {
    this.serverSend(hello(last, replayed));
  }

  serverEvent(envelope: Frame): void {
    this.serverSend(JSON.stringify(envelope));
  }
}

const sockets: FakeSocket[] = [];

const FAST = {
  heartbeatTimeoutMs: 60,
  maxReconnectAttempts: 3,
  reconnectBaseMs: 4,
  reconnectMaxMs: 16,
};

function makeClient(): EventStreamClient {
  return new EventStreamClient("http://127.0.0.1:8317", "session-token-1", FAST);
}

beforeEach(() => {
  sockets.length = 0;
  vi.stubGlobal("WebSocket", FakeSocket);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("the §13 hello handshake", () => {
  it("connects with the token in the query string, and no `since` on the first open", () => {
    const client = makeClient();
    client.connect();

    expect(client.status.state).toBe("connecting");
    expect(sockets).toHaveLength(1);
    const url = new URL(sockets[0]!.url);
    expect(url.pathname).toBe("/api/v1/events");
    expect(url.searchParams.get("token")).toBe("session-token-1");
    expect(url.searchParams.has("since")).toBe(false);
    // the ws:// scheme is derived from the REST base
    expect(url.protocol).toBe("ws:");
  });

  it("flips to open on the hello frame and records the payload", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.serverHello(7, 2);

    expect(client.status.state).toBe("open");
    expect(client.status.hello).toEqual({ hello: true, last_sequence: 7, replayed: 2 });
  });

  it("resets the attempt counter on a successful hello", () => {
    vi.useFakeTimers();
    const client = makeClient();
    client.connect();
    sockets[0]!.close(); // server drop
    expect(client.status.state).toBe("reconnecting");
    expect(client.status.attempts).toBe(1);
    vi.advanceTimersByTime(4);
    sockets[1]!.serverHello(0, 0);
    expect(client.status.attempts).toBe(0);
    expect(client.status.state).toBe("open");
  });
});

describe("event delivery + the replay cursor", () => {
  it("delivers accepted envelopes to listeners and advances the cursor", () => {
    const client = makeClient();
    const seen: Frame[] = [];
    client.subscribe((envelope) => { seen.push(envelope as unknown as Frame); });
    client.connect();
    sockets[0]!.serverHello();
    sockets[0]!.serverEvent(frame(1, "job.progress", { percent: 10 }));
    sockets[0]!.serverEvent(frame(2, "job.completed", { result_ready: true }));

    expect(seen.map((e) => e.sequence)).toEqual([1, 2]);
    expect(client.status.lastSequence).toBe(2);
  });

  it("never delivers duplicates: sequences at or below the cursor are dropped", () => {
    const client = makeClient();
    const seen: Frame[] = [];
    client.subscribe((envelope) => { seen.push(envelope as unknown as Frame); });
    client.connect();
    sockets[0]!.serverHello();
    sockets[0]!.serverEvent(frame(5, "job.progress", {}));
    sockets[0]!.serverEvent(frame(5, "job.progress", {})); // duplicate
    sockets[0]!.serverEvent(frame(3, "job.progress", {})); // replay below the cursor

    expect(seen).toHaveLength(1);
    expect(client.status.lastSequence).toBe(5);
  });

  it("drops malformed frames without killing the stream", () => {
    const client = makeClient();
    const seen: unknown[] = [];
    client.subscribe((envelope) => { seen.push(envelope); });
    client.connect();
    sockets[0]!.serverHello();
    sockets[0]!.serverSend("not json at all");
    sockets[0]!.serverSend(JSON.stringify({ no: "shape" }));
    sockets[0]!.serverSend(JSON.stringify(frame(9, "not.in.vocabulary", {})));
    sockets[0]!.serverEvent(frame(10, "warning", {}));

    expect(seen).toHaveLength(1);
    expect(client.status.state).toBe("open");
    expect(client.status.lastSequence).toBe(10);
  });

  it("isolates a throwing listener — one broken consumer never breaks the stream", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const client = makeClient();
    const good: number[] = [];
    client.subscribe(() => { throw new Error("broken consumer"); });
    client.subscribe((envelope) => { good.push(envelope.sequence); });
    client.connect();
    sockets[0]!.serverHello();
    sockets[0]!.serverEvent(frame(1, "job.progress", {}));

    expect(good).toEqual([1]);
    expect(consoleError).toHaveBeenCalled();
    consoleError.mockRestore();
  });
});

describe("reconnection: backoff, replay, and the terminal states", () => {
  it("schedules the reconnect with doubling backoff and carries `since` on the retry", () => {
    vi.useFakeTimers();
    const client = makeClient();
    client.connect();
    sockets[0]!.serverHello();
    sockets[0]!.serverEvent(frame(11, "job.progress", {}));
    sockets[0]!.close(); // server drop

    expect(client.status.state).toBe("reconnecting");
    expect(client.status.attempts).toBe(1);

    vi.advanceTimersByTime(4); // baseMs * 2^0
    expect(sockets).toHaveLength(2);
    const retryUrl = new URL(sockets[1]!.url);
    expect(retryUrl.searchParams.get("since")).toBe("11"); // exactly the missed tail

    sockets[1]!.close(); // drop again — the second backoff step is 8 ms
    vi.advanceTimersByTime(4);
    expect(sockets).toHaveLength(2); // too early
    vi.advanceTimersByTime(4);
    expect(sockets).toHaveLength(3);
  });

  it("caps the backoff at reconnectMaxMs", () => {
    vi.useFakeTimers();
    // a dedicated budget so the cap is observable BEFORE give-up:
    // delays would be 4, 8, 16, 32 without the cap; with max 10 the
    // third step and beyond are all 10 ms.
    const client = new EventStreamClient("http://127.0.0.1:8317", "session-token-1", {
      heartbeatTimeoutMs: 60,
      maxReconnectAttempts: 8,
      reconnectBaseMs: 4,
      reconnectMaxMs: 10,
    });
    client.connect();
    sockets[0]!.close(); // attempts=1, delay 4 ms
    vi.advanceTimersByTime(4);
    expect(sockets).toHaveLength(2);
    sockets[1]!.close(); // attempts=2, delay 8 ms
    vi.advanceTimersByTime(8);
    expect(sockets).toHaveLength(3);
    sockets[2]!.close(); // attempts=3, delay 16 ms uncapped -> 10 ms capped
    vi.advanceTimersByTime(9);
    expect(sockets).toHaveLength(3); // one millimeter short of the cap
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(4);
    sockets[3]!.close(); // attempts=4, delay 32 ms uncapped -> still 10 ms
    vi.advanceTimersByTime(10);
    expect(sockets).toHaveLength(5);
  });

  it("gives up to `closed` after maxReconnectAttempts", () => {
    vi.useFakeTimers();
    const client = makeClient();
    client.connect();
    for (let i = 0; i < FAST.maxReconnectAttempts; i += 1) {
      sockets[sockets.length - 1]!.close();
      vi.advanceTimersByTime(20); // past any capped delay
    }
    sockets[sockets.length - 1]!.close(); // the attempt that exhausts the budget
    expect(client.status.state).toBe("closed");
    const before = sockets.length;
    vi.advanceTimersByTime(200);
    expect(sockets).toHaveLength(before); // terminal: no further retries
  });

  it("never reconnects after the backend's graceful shutdown envelope", () => {
    vi.useFakeTimers();
    const client = makeClient();
    client.connect();
    sockets[0]!.serverHello();
    sockets[0]!.serverEvent(frame(3, "shutdown", { reason: "restart" }));

    expect(client.status.state).toBe("closed");
    expect(sockets[0]!.closeCalls[0]?.code).toBe(1000);
    const before = sockets.length;
    vi.advanceTimersByTime(200);
    expect(sockets).toHaveLength(before); // graceful shutdown: no reconnect, ever
  });

  it("never reconnects after a user close()", () => {
    vi.useFakeTimers();
    const client = makeClient();
    client.connect();
    sockets[0]!.serverHello();
    client.close();

    expect(client.status.state).toBe("closed");
    expect(sockets[0]!.closeCalls[0]?.code).toBe(1000);
    const before = sockets.length;
    vi.advanceTimersByTime(200);
    expect(sockets).toHaveLength(before);
  });
});

describe("liveness + the keepalive probe", () => {
  it("closes the dead peer and reconnects when no frame arrives in the heartbeat window", () => {
    vi.useFakeTimers();
    const client = makeClient();
    client.connect();
    sockets[0]!.serverHello();
    expect(client.status.state).toBe("open");

    // silence longer than heartbeatTimeoutMs (60 ms): the 5 s liveness
    // tick fires at t=5000 under fake timers and detects the dead peer
    vi.advanceTimersByTime(5_000);

    expect(sockets[0]!.closeCalls[0]?.code).toBe(4008);
    expect(client.status.state).toBe("reconnecting");
    vi.advanceTimersByTime(4); // the baseMs reconnect delay
    expect(sockets).toHaveLength(2);
    // the reconnect replays from the last accepted sequence
    sockets[1]!.serverHello(0, 0);
    expect(client.status.state).toBe("open");
  });

  it("sends the literal `ping` probe while open (the backend answers pong beats)", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.serverHello();
    client.ping();

    expect(sockets[0]!.sent).toEqual(["ping"]);
  });
});

describe("state notifications", () => {
  it("notifies state listeners on every transition", () => {
    const client = makeClient();
    const states: string[] = [];
    client.onStateChange((status) => { states.push(status.state); });
    client.connect();
    sockets[0]!.serverHello();
    sockets[0]!.serverEvent(frame(1, "job.progress", {}));

    expect(states).toEqual(["connecting", "open", "open"]);
    // the trailing "open" is the lastSequence notification (§13: status
    // readers refresh when the cursor moves, even mid-open)
    expect(client.status.lastSequence).toBe(1);
  });

  it("unsubscribes cleanly", () => {
    const client = makeClient();
    const states: string[] = [];
    const off = client.onStateChange((status) => { states.push(status.state); });
    client.connect();
    off();
    sockets[0]!.serverHello();

    expect(states).toEqual(["connecting"]); // the hello notification never arrived
  });
});
