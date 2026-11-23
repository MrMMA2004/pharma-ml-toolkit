/**
 * part16 §8 — the floating Resource Monitor panel.
 *
 * A compact floating panel, never a full-window surface: movable by
 * dragging the header strip (and by keyboard arrows — the accessible
 * move), collapsible between the compact summary and the detailed
 * view, resizable within the clamped 280..480px limits, openable and
 * closable through the §10 persisted `resourceMonitor.visible`
 * preference, remembering its last position and preferred size in the
 * validated `pharmaml.studio.resmon.v1` pocket. Every number comes
 * from the backend's existing `/api/v1/system/resources` service (the
 * JobManager queue depth included) — never simulated; absent
 * stdlib-fallback numbers render as the honest dash; GPU degrades to
 * its stated "not available" line. The polling cadence is the §10
 * interval preference (the settings panel owns it), paused entirely
 * while closed and skipped while the document is hidden — no
 * aggressive polling, no monitoring overhead when nobody watches.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Activity, ChevronDown, ChevronUp, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useSettings } from "@/hooks/useSettings";
import { getSettingsStore } from "@/services/settings/store";
import { systemApi } from "@/services/system/client";
import {
  KEYBOARD_GEOMETRY_STEP,
  clampGeometry,
  clampWidth,
  loadPanelState,
  savePanelState,
  type PanelGeometry,
} from "@/services/system/panelState";
import type { ResourceSnapshot } from "@/services/system/types";

const POLL_FLOOR_MS = 250;

function formatBytes(bytes: number | null): string {
  if (bytes === null) {
    return "—";
  }
  const units = ["B", "KB", "MB", "GB", "TB"] as const;
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = unit === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

function formatPercent(percent: number | null): string {
  return percent === null ? "—" : `${percent.toFixed(1)}%`;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 py-0.5">
      <span className="text-[11px] text-[var(--color-muted-foreground)]">{label}</span>
      <span className="font-mono text-[11px] text-[var(--color-foreground)]">{value}</span>
    </div>
  );
}

export function ResourceMonitorPanel() {
  const store = getSettingsStore();
  const settings = useSettings();
  const intervalMs = settings.resourceMonitor.updateIntervalMs;
  const [snapshot, setSnapshot] = useState<ResourceSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [geometry, setGeometry] = useState<PanelGeometry>(() => loadPanelState());
  const asideRef = useRef<HTMLElement | null>(null);

  const persist = useCallback((next: PanelGeometry) => {
    const clamped = clampGeometry(next);
    setGeometry(clamped);
    savePanelState(clamped);
  }, []);

  // the single §8 poll loop: cadence = the §10 interval preference,
  // skipped while the document is hidden, stopped entirely when closed
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      if (!document.hidden) {
        try {
          const snap = await systemApi.resources();
          if (!cancelled) {
            setSnapshot(snap);
            setError(null);
          }
        } catch (e) {
          if (!cancelled) {
            setError(e instanceof Error ? e.message : String(e));
          }
        }
      }
      if (!cancelled) {
        timer = setTimeout(tick, Math.max(POLL_FLOOR_MS, intervalMs));
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        clearTimeout(timer);
      }
    };
  }, [intervalMs]);

  // a viewport change re-clamps the remembered geometry (never off-screen)
  useEffect(() => {
    const onResize = () => {
      setGeometry((current) => clampGeometry(current));
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const close = useCallback(() => {
    store.update({ resourceMonitor: { visible: false } });
  }, [store]);

  // Escape closes the panel only while the panel holds focus — an
  // Escape aimed at a dialog, a form, or the page never sweeps this
  // floating panel away (the popover precedent's own scope rule)
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") {
        return;
      }
      const target = event.target as Node | null;
      if (target !== null && asideRef.current?.contains(target) === true) {
        close();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [close]);

  const onStripPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const origin = geometry;
    const onMove = (move: PointerEvent) => {
      const next = clampGeometry({
        ...origin,
        x: origin.x + (move.clientX - startX),
        y: origin.y + (move.clientY - startY),
      });
      setGeometry(next);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      setGeometry((current) => {
        savePanelState(current);
        return current;
      });
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const onStripKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const step = KEYBOARD_GEOMETRY_STEP;
    const origin = geometry;
    const deltas: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const delta = deltas[event.key];
    if (delta === undefined) {
      return;
    }
    event.preventDefault();
    persist({
      ...origin,
      x: origin.x + delta[0],
      y: origin.y + delta[1],
    });
  };

  const onResizePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const origin = geometry;
    const onMove = (move: PointerEvent) => {
      const next = clampGeometry({
        ...origin,
        width: clampWidth(origin.width + (move.clientX - startX)),
      });
      setGeometry(next);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      setGeometry((current) => {
        savePanelState(current);
        return current;
      });
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const onResizeKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    event.preventDefault();
    const origin = geometry;
    const delta = event.key === "ArrowRight" ? KEYBOARD_GEOMETRY_STEP : -KEYBOARD_GEOMETRY_STEP;
    persist({ ...origin, width: clampWidth(origin.width + delta) });
  };

  const healthy = error === null && snapshot !== null;
  const system = snapshot?.system ?? null;

  return (
    <aside
      ref={asideRef}
      aria-label="Resource monitor"
      style={{
        position: "fixed",
        left: geometry.x,
        top: geometry.y,
        width: geometry.width,
      }}
      className="z-40 flex flex-col rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] text-[var(--color-foreground)] shadow-lg"
    >
      <div
        data-testid="monitor-drag"
        tabIndex={0}
        aria-label="Move panel (drag or arrow keys)"
        role="toolbar"
        onPointerDown={onStripPointerDown}
        onKeyDown={onStripKeyDown}
        className="flex cursor-grab select-none items-center gap-2 rounded-t-lg border-b border-[var(--color-border)] bg-[var(--color-muted)] px-2.5 py-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
      >
        <span
          aria-hidden="true"
          title={healthy ? "backend live" : "snapshot failing"}
          className={`inline-block h-2 w-2 shrink-0 rounded-full ${
            healthy
              ? "bg-[var(--color-primary)]"
              : "bg-[var(--color-destructive)]"
          }`}
        />
        <span className="text-xs font-semibold">Resource monitor</span>
        <span className="ml-auto flex items-center gap-1.5">
          <span className="rounded border border-[var(--color-border)] px-1 py-0.5 font-mono text-[10px] text-[var(--color-muted-foreground)]">
            {Math.max(POLL_FLOOR_MS, intervalMs)}ms
          </span>
          {snapshot ? (
            <span className="rounded border border-[var(--color-border)] px-1 py-0.5 font-mono text-[10px] text-[var(--color-muted-foreground)]">
              {snapshot.source}
            </span>
          ) : null}
          <button
            type="button"
            aria-label="Toggle detail"
            aria-expanded={geometry.expanded}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={() => persist({ ...geometry, expanded: !geometry.expanded })}
            className="rounded p-1 text-[var(--color-muted-foreground)] hover:bg-[var(--color-accent)] hover:text-[var(--color-foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
          >
            {geometry.expanded ? (
              <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
            )}
          </button>
          <button
            type="button"
            aria-label="Close resource monitor"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={close}
            className="rounded p-1 text-[var(--color-muted-foreground)] hover:bg-[var(--color-accent)] hover:text-[var(--color-foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
          >
            <X className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </span>
      </div>

      {snapshot === null && error !== null ? (
        <div className="px-3 py-3 text-[11px] text-[var(--color-muted-foreground)]">
          <div className="font-semibold text-[var(--color-foreground)]">
            Resource snapshot unavailable
          </div>
          <div className="mt-1 break-words">{error}</div>
          <div className="mt-2">
            the panel keeps retrying at the configured interval — no values are
            simulated
          </div>
        </div>
      ) : snapshot === null ? (
        <div className="px-3 py-3 text-[11px] text-[var(--color-muted-foreground)]">
          fetching the first snapshot…
        </div>
      ) : (
        <div className="px-3 py-2">
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5" aria-label="Compact summary">
            <span className="font-mono text-[11px] text-[var(--color-foreground)]">
              CPU {formatPercent(system?.cpu_percent ?? null)}
            </span>
            <span className="font-mono text-[11px] text-[var(--color-foreground)]">
              RAM {formatPercent(system?.memory?.percent ?? null)}
            </span>
            <span className="font-mono text-[11px] text-[var(--color-foreground)]">
              Jobs {snapshot.jobs.active}
            </span>
            <span className="font-mono text-[11px] text-[var(--color-foreground)]">
              v{snapshot.backend.version}
            </span>
          </div>
          {geometry.expanded ? (
            <div className="mt-2 max-h-72 overflow-y-auto border-t border-[var(--color-border)] pt-2">
              <section
                aria-label="System — host"
                className="mb-2"
              >
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
                  System — host
                </div>
                <Row label="CPU cores" value={system?.cpu_count === null || system?.cpu_count === undefined ? "—" : String(system.cpu_count)} />
                <Row label="CPU utilization" value={formatPercent(system?.cpu_percent ?? null)} />
                <Row
                  label="Load average (1/5/15)"
                  value={
                    system?.loadavg
                      ? system.loadavg.map((x) => x.toFixed(1)).join(" / ")
                      : "—"
                  }
                />
                <Row
                  label="Memory used / total"
                  value={
                    system?.memory
                      ? `${formatBytes(system.memory.used)} / ${formatBytes(system.memory.total)} · ${formatPercent(system.memory.percent)}`
                      : "—"
                  }
                />
                <Row
                  label="Memory available"
                  value={system?.memory ? formatBytes(system.memory.available) : "—"}
                />
                <Row
                  label="Swap used / total"
                  value={
                    system?.swap
                      ? `${formatBytes(system.swap.used)} / ${formatBytes(system.swap.total)} · ${formatPercent(system.swap.percent)}`
                      : "—"
                  }
                />
                <Row
                  label="Disk used / total"
                  value={
                    system
                      ? `${formatBytes(system.disk.used)} / ${formatBytes(system.disk.total)}`
                      : "—"
                  }
                />
                <Row label="Disk free" value={system ? formatBytes(system.disk.free) : "—"} />
              </section>
              <section aria-label="Application — backend process">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
                  Application — backend process
                </div>
                <Row label="Backend" value={`${snapshot.backend.name} v${snapshot.backend.version}`} />
                <Row label="PID" value={String(snapshot.process.pid)} />
                <Row label="Process CPU" value={formatPercent(snapshot.process.cpu_percent)} />
                <Row label="Memory (RSS)" value={formatBytes(snapshot.process.memory_rss_bytes)} />
                <Row label="Threads" value={snapshot.process.num_threads === null ? "—" : String(snapshot.process.num_threads)} />
                <Row label="Jobs (active)" value={String(snapshot.jobs.active)} />
                <div className="mt-1 grid grid-cols-2 gap-x-3">
                  {Object.entries(snapshot.jobs.by_state).map(([state, count]) => (
                    <div key={state} className="flex items-center justify-between">
                      <span className="text-[11px] text-[var(--color-muted-foreground)]">{state}</span>
                      <span className="font-mono text-[11px] text-[var(--color-foreground)]">{count}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                  GPU — not available (CPU-only operation supported)
                </div>
              </section>
              <div className="mt-2 border-t border-[var(--color-border)] pt-2 text-[10px] leading-snug text-[var(--color-muted-foreground)]">
                {system ? (system.notes ?? []).map((note) => (
                  <div key={note}>{note}</div>
                )) : null}
                <div>
                  All values from /api/v1/system/resources — never simulated.
                </div>
              </div>
            </div>
          ) : null}
        </div>
      )}

      <div
        data-testid="monitor-resize"
        tabIndex={0}
        aria-label="Resize panel (drag or arrow keys)"
        role="separator"
        onPointerDown={onResizePointerDown}
        onKeyDown={onResizeKeyDown}
        className="mt-auto flex h-3 w-full cursor-col-resize items-center justify-center rounded-b-lg border-t border-[var(--color-border)] text-[var(--color-muted-foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
      >
        <span aria-hidden="true" className="pointer-events-none select-none">
          ⌃⌄
        </span>
      </div>
    </aside>
  );
}

export function ResourceMonitorHost() {
  const store = getSettingsStore();
  const settings = useSettings();
  const visible = settings.resourceMonitor.visible;

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        aria-label="Resource monitor"
        aria-expanded={visible}
        onClick={() => {
          store.update({ resourceMonitor: { visible: !visible } });
        }}
      >
        <Activity className="h-4 w-4" aria-hidden="true" />
      </Button>
      {visible ? <ResourceMonitorPanel /> : null}
    </>
  );
}
