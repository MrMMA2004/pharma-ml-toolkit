/**
 * part16 §10 — the notification preference's real consumer.
 *
 * The shell-mounted job-completion toast lane: it subscribes the REAL
 * event stream (the same surface the JobTracker rides — this
 * component mounts inside the StartupGate's ready phase, so the
 * stream is configured), surfaces one polite toast per TERMINAL job
 * event (job.completed / job.failed / job.cancelled — the event
 * vocabulary's own terminal three), dedups by event id (reconnect
 * replays stay silent), caps the stack, auto-dismisses, and stays
 * entirely silent while the notification preference is off.
 */

import { useEffect, useRef, useState } from "react";

import { getEventStream } from "@/services/websocket/client";
import { getSettingsStore } from "@/services/settings/store";
import type { EventEnvelope } from "@/services/websocket/events";

type ToastKind = "completed" | "failed" | "cancelled";

const TERMINAL_KINDS: Record<string, ToastKind> = {
  "job.completed": "completed",
  "job.failed": "failed",
  "job.cancelled": "cancelled",
};

const TOAST_LIFETIME_MS = 6_000;
const MAX_VISIBLE = 3;

interface ToastEntry {
  readonly id: string;
  readonly kind: ToastKind;
  readonly jobId: string;
}

const KIND_CLASS: Record<ToastKind, string> = {
  completed: "border-[var(--color-primary)]",
  failed: "border-[var(--color-destructive)]",
  cancelled: "border-[var(--color-border)]",
};

export function JobCompletionToasts() {
  const [toasts, setToasts] = useState<readonly ToastEntry[]>([]);
  const timersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());

  useEffect(() => {
    const stream = getEventStream();
    const seen = new Set<string>();
    const unsubscribe = stream.subscribe((envelope: EventEnvelope) => {
      const kind = TERMINAL_KINDS[envelope.event_type];
      if (kind === undefined || envelope.job_id === null) {
        return;
      }
      if (seen.has(envelope.event_id)) {
        return; // reconnect replay — the tracker reconciles; a toast never repeats
      }
      seen.add(envelope.event_id);
      if (!getSettingsStore().snapshot().notifications.jobCompletionToasts) {
        return;
      }
      const id = envelope.event_id;
      setToasts((prev) => {
        const next = [...prev, { id, kind, jobId: envelope.job_id as string }];
        return next.slice(-MAX_VISIBLE);
      });
      const timer = setTimeout(() => {
        setToasts((prev) => prev.filter((toast) => toast.id !== id));
        timersRef.current.delete(timer);
      }, TOAST_LIFETIME_MS);
      timersRef.current.add(timer);
    });
    return () => {
      unsubscribe();
      for (const timer of timersRef.current) {
        clearTimeout(timer);
      }
      timersRef.current.clear();
    };
  }, []);

  if (toasts.length === 0) {
    return null;
  }

  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed bottom-4 right-4 z-40 flex flex-col gap-2"
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          role="status"
          data-kind={toast.kind}
          className={`rounded-md border bg-[var(--color-card)] px-3 py-2 text-xs text-[var(--color-foreground)] shadow-lg ${KIND_CLASS[toast.kind]}`}
        >
          Job {toast.jobId} {toast.kind}
        </div>
      ))}
    </div>
  );
}
