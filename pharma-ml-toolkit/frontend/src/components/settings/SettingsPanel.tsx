/**
 * part16 §10 — the compact settings panel (the popover's content).
 *
 * §10's inventory, one compact section each: Appearance (theme +
 * density), Charts (the colorblind-safe palette + the default export
 * format), Scientific execution defaults (the ML form's n_jobs
 * prefill), Notifications (the job-completion toast toggle), the
 * Resource-monitor preferences (consumed by the Phase-8 lane when
 * it ships — the roadmap's #5), the read-only Diagnostics block
 * (version / build / data + cache locations / privacy), and
 * Reset-to-default.
 *
 * Every control writes through the store's validate-before-apply
 * `update()`; out-of-band values surface the store's own errors
 * inline (role=alert) and nothing applies.
 */

import { useState, type KeyboardEvent } from "react";

import { Button } from "@/components/ui/button";
import { useSettings } from "@/hooks/useSettings";
import { APP_VERSION } from "@/lib/version";
import { getSettingsStore } from "@/services/settings/store";
import { getLastStartupInfo } from "@/services/api/startup";

/** One compact section heading. */
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-[var(--color-border)] pt-3 first:border-t-0 first:pt-0">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
        {title}
      </h3>
      {children}
    </section>
  );
}

/** One read-only diagnostics row. */
function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5 text-xs">
      <span className="shrink-0 text-[var(--color-muted-foreground)]">{label}</span>
      <span className="text-right text-[var(--color-foreground)]">{value}</span>
    </div>
  );
}

/** A labeled radio option. */
function Radio({
  name,
  value,
  checked,
  onChange,
  label,
}: {
  name: string;
  value: string;
  checked: boolean;
  onChange: () => void;
  label: string;
}) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-[var(--color-border)] px-2.5 py-1.5 text-xs has-[:checked]:border-[var(--color-primary)]">
      <input
        type="radio"
        name={name}
        value={value}
        checked={checked}
        onChange={onChange}
        className="accent-[var(--color-primary)]"
      />
      {label}
    </label>
  );
}

/** A labeled toggle row. */
function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 py-1 text-xs text-[var(--color-foreground)]">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-4 w-4 accent-[var(--color-primary)]"
      />
      {label}
    </label>
  );
}

export function SettingsPanel() {
  const settings = useSettings();
  const store = getSettingsStore();
  const [errors, setErrors] = useState<string[]>([]);
  const [parallelismDraft, setParallelismDraft] = useState(() =>
    String(settings.execution.defaultParallelism),
  );
  const [intervalDraft, setIntervalDraft] = useState(() =>
    String(settings.resourceMonitor.updateIntervalMs),
  );
  const backend = getLastStartupInfo();

  const commit = (apply: () => { ok: boolean; errors?: string[] }) => {
    const verdict = apply();
    setErrors(verdict.ok ? [] : verdict.errors ?? ["invalid value"]);
  };

  const commitNumber = (
    draft: string,
    apply: (value: number) => void,
  ) => {
    const parsed = Number(draft);
    if (draft.trim() === "" || !Number.isInteger(parsed)) {
      setErrors(["the value must be a whole number"]);
      return;
    }
    apply(parsed);
  };

  const onDraftKey = (event: KeyboardEvent<HTMLInputElement>, commitFn: () => void) => {
    if (event.key === "Enter") {
      commitFn();
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <Section title="Appearance">
        <fieldset className="mb-2">
          <legend className="mb-1 text-xs text-[var(--color-muted-foreground)]">Theme</legend>
          <div className="flex flex-wrap gap-1.5">
            <Radio name="settings-theme" value="light" label="Light" checked={settings.theme === "light"} onChange={() => { void store.update({ theme: "light" }); }} />
            <Radio name="settings-theme" value="dark" label="Dark" checked={settings.theme === "dark"} onChange={() => { void store.update({ theme: "dark" }); }} />
            <Radio name="settings-theme" value="system" label="System" checked={settings.theme === "system"} onChange={() => { void store.update({ theme: "system" }); }} />
          </div>
        </fieldset>
        <fieldset>
          <legend className="mb-1 text-xs text-[var(--color-muted-foreground)]">Density</legend>
          <div className="flex flex-wrap gap-1.5">
            <Radio name="settings-density" value="comfortable" label="Comfortable" checked={settings.density === "comfortable"} onChange={() => { void store.update({ density: "comfortable" }); }} />
            <Radio name="settings-density" value="compact" label="Compact" checked={settings.density === "compact"} onChange={() => { void store.update({ density: "compact" }); }} />
          </div>
        </fieldset>
      </Section>

      <Section title="Charts">
        <Toggle
          label="Colorblind-safe chart palette"
          checked={settings.charts.colorblindSafe}
          onChange={() => { void store.update({ charts: { colorblindSafe: !settings.charts.colorblindSafe } }); }}
        />
        <fieldset className="mt-1">
          <legend className="mb-1 text-xs text-[var(--color-muted-foreground)]">Default chart export</legend>
          <div className="flex flex-wrap gap-1.5">
            <Radio name="settings-export-format" value="png" label="PNG" checked={settings.charts.defaultExportFormat === "png"} onChange={() => { void store.update({ charts: { defaultExportFormat: "png" } }); }} />
            <Radio name="settings-export-format" value="svg" label="SVG" checked={settings.charts.defaultExportFormat === "svg"} onChange={() => { void store.update({ charts: { defaultExportFormat: "svg" } }); }} />
          </div>
        </fieldset>
      </Section>

      <Section title="Scientific execution defaults">
        <label className="flex items-center gap-2 py-1 text-xs text-[var(--color-foreground)]">
          Default parallelism (n_jobs)
          <input
            key={settings.execution.defaultParallelism}
            type="number"
            min={-1}
            max={64}
            step={1}
            value={parallelismDraft}
            onChange={(event) => setParallelismDraft(event.target.value)}
            onBlur={() => commitNumber(parallelismDraft, (value) => { commit(() => store.update({ execution: { defaultParallelism: value } })); })}
            onKeyDown={(event) => onDraftKey(event, () => commitNumber(parallelismDraft, (value) => { commit(() => store.update({ execution: { defaultParallelism: value } })); }))}
            className="w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
          />
        </label>
        <p className="mt-0.5 text-[11px] leading-snug text-[var(--color-muted-foreground)]">
          1 is conservative; -1 uses all cores (the memory-aware clamp stays automatic
          and advisory-reported).
        </p>
      </Section>

      <Section title="Notifications">
        <Toggle
          label="Job completion toasts"
          checked={settings.notifications.jobCompletionToasts}
          onChange={() => { void store.update({ notifications: { jobCompletionToasts: !settings.notifications.jobCompletionToasts } }); }}
        />
      </Section>

      <Section title="Resource monitor">
        <label className="flex items-center gap-2 py-1 text-xs text-[var(--color-foreground)]">
          Monitor update interval (ms)
          <input
            key={settings.resourceMonitor.updateIntervalMs}
            type="number"
            min={250}
            max={60000}
            step={50}
            value={intervalDraft}
            onChange={(event) => setIntervalDraft(event.target.value)}
            onBlur={() => commitNumber(intervalDraft, (value) => { commit(() => store.update({ resourceMonitor: { updateIntervalMs: value } })); })}
            onKeyDown={(event) => onDraftKey(event, () => commitNumber(intervalDraft, (value) => { commit(() => store.update({ resourceMonitor: { updateIntervalMs: value } })); }))}
            className="w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
          />
        </label>
        <Toggle
          label="Show on launch"
          checked={settings.resourceMonitor.visible}
          onChange={() => { void store.update({ resourceMonitor: { visible: !settings.resourceMonitor.visible } }); }}
        />
        <p className="mt-0.5 text-[11px] leading-snug text-[var(--color-muted-foreground)]">
          Consumed by the Phase-8 floating resource monitor when that lane ships
          (the roadmap's #5); persisted here so the choice survives it.
        </p>
      </Section>

      <Section title="Diagnostics">
        <Row label="App version" value={APP_VERSION} />
        <Row
          label="Backend version"
          value={backend === null ? "not connected yet" : `${backend.version} (${backend.source})`}
        />
        <Row label="Settings storage" value="browser local storage (pharmaml.studio.settings.v1)" />
        <Row label="Data + cache location" value="backend-managed (the artifact registry + dataset store); this app adds no local files beyond settings" />
        <Row label="Privacy + security" value="offline application: no telemetry, no external network calls; the backend runs locally with session-token auth" />
      </Section>

      {errors.length > 0 ? (
        <div role="alert" className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-muted)] px-3 py-2 text-xs text-[var(--color-foreground)]">
          {errors.map((error) => (
            <p key={error}>{error}</p>
          ))}
        </div>
      ) : null}

      <div className="border-t border-[var(--color-border)] pt-3">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            store.reset();
            setErrors([]);
          }}
        >
          Reset all settings to defaults
        </Button>
      </div>
    </div>
  );
}
