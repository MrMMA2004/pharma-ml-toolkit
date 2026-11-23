import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { JobCompletionToasts } from "@/components/settings/JobCompletionToasts";
import { getSettingsStore } from "@/services/settings/store";
import type { EventEnvelope } from "@/services/websocket/events";

/**
 * part16 §10 — the notification preference's real consumer.
 *
 * The job-completion toast lane: the shell-mounted watcher subscribes
 * the REAL event stream (the same surface the JobTracker rides),
 * surfaces one polite toast per TERMINAL job event
 * (job.completed / job.failed / job.cancelled — the vocabulary's own
 * terminal three), dedups by event id, and stays silent when the
 * notification preference is off. Non-terminal events never toast.
 */

type Listener = (envelope: EventEnvelope) => void;

const listeners = new Set<Listener>();
const unsubscribe = vi.fn();

vi.mock("@/services/websocket/client", () => ({
  getEventStream: () => ({
    subscribe: (cb: Listener) => {
      listeners.add(cb);
      return unsubscribe;
    },
  }),
}));

function envelope(
  event_type: string,
  event_id: string,
  job_id: string | null,
): EventEnvelope {
  return {
    event_id,
    sequence: 1,
    job_id,
    event_type: event_type as EventEnvelope["event_type"],
    timestamp: "2026-11-23T12:00:00Z",
    payload: {},
  };
}

describe("the job completion toast lane (§10 notifications)", () => {
  beforeEach(() => {
    listeners.clear();
    unsubscribe.mockClear();
    vi.useFakeTimers();
    window.localStorage.clear();
    getSettingsStore().reset();
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("toasts the three terminal job event kinds", async () => {
    render(<JobCompletionToasts />);
    const cb = [...listeners][0]!;

    await act(async () => {
      cb(envelope("job.completed", "e1", "job-a"));
    });
    expect(screen.getByText(/job-a.*completed/i)).toBeInTheDocument();

    await act(async () => {
      cb(envelope("job.failed", "e2", "job-b"));
    });
    expect(screen.getByText(/job-b.*failed/i)).toBeInTheDocument();

    await act(async () => {
      cb(envelope("job.cancelled", "e3", "job-c"));
    });
    expect(screen.getByText(/job-c.*cancelled/i)).toBeInTheDocument();
  });

  it("never toasts non-terminal or session-level events", () => {
    render(<JobCompletionToasts />);
    const cb = [...listeners][0]!;

    cb(envelope("job.progress", "e4", "job-a"));
    cb(envelope("job.started", "e5", "job-a"));
    cb(envelope("heartbeat", "e6", null));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("dedups by event id (replays and reconnect repeats stay silent)", async () => {
    render(<JobCompletionToasts />);
    const cb = [...listeners][0]!;

    await act(async () => {
      cb(envelope("job.completed", "e1", "job-a"));
      cb(envelope("job.completed", "e1", "job-a"));
    });
    expect(screen.getAllByText(/job-a.*completed/i)).toHaveLength(1);
  });

  it("stays silent when the notification preference is off (the §10 toggle is real)", () => {
    getSettingsStore().update({ notifications: { jobCompletionToasts: false } });
    render(<JobCompletionToasts />);
    const cb = [...listeners][0]!;

    cb(envelope("job.completed", "e1", "job-a"));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("auto-dismisses a toast after its lifetime", async () => {
    render(<JobCompletionToasts />);
    const cb = [...listeners][0]!;

    await act(async () => {
      cb(envelope("job.completed", "e1", "job-a"));
    });
    expect(screen.getByRole("status")).toBeInTheDocument();
    await act(async () => {
      vi.advanceTimersByTime(6_000);
    });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("unsubscribes from the stream on unmount", () => {
    const { unmount } = render(<JobCompletionToasts />);
    unmount();
    expect(unsubscribe).toHaveBeenCalled();
  });
});
