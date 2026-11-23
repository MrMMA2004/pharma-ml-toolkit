/**
 * part16 §10 — the Phase-7 settings store + persistence layer.
 *
 * The house pattern (part11 §8.2): a framework-free singleton the
 * React binding reads through `useSyncExternalStore` — no new
 * state-management dependency. The persistence layer is the WebView's
 * localStorage (works identically in the Vite dev lane and the
 * Tauri 2 host; the key is versioned so migrations stay explicit).
 *
 * The §10 contract enforced HERE, mechanically:
 *
 * * validate-before-apply — `update()` validates the merged WHOLE;
 *   an invalid candidate is rejected whole with its errors listed,
 *   and the snapshot, the document and the storage stay untouched;
 * * persisted settings are validated on read — corrupt or
 *   forward-schema junk degrades to the defaults, never a crash and
 *   never a partially-applied state;
 * * reset-to-default — one call, applied + persisted + notified;
 * * the DOM applications ride the SAME documentElement protocol the
 *   app already speaks: the `.dark` class (the Mineral tokens), the
 *   `data-density` attribute, and the `.charts-cvd` palette class
 *   (the PlotlyFigure theme observer re-draws live charts on class
 *   mutations — the palette flip re-themes every mounted chart).
 */

import {
  DEFAULT_SETTINGS,
  type AppSettings,
  type SettingsPatch,
  type ThemeSetting,
} from "@/services/settings/types";

export { DEFAULT_SETTINGS };
export type { AppSettings, SettingsPatch };

/** The one localStorage key (schema-versioned). */
export const SETTINGS_STORAGE_KEY = "pharmaml.studio.settings.v1";

export type ValidationVerdict =
  | { readonly ok: true; readonly value: AppSettings }
  | { readonly ok: false; readonly errors: string[] };

const THEMES: readonly ThemeSetting[] = ["light", "dark", "system"];
const DENSITIES: readonly string[] = ["comfortable", "compact"];
const EXPORT_FORMATS: readonly string[] = ["png", "svg"];
const TOP_LEVEL_KEYS = [
  "schemaVersion",
  "theme",
  "density",
  "charts",
  "notifications",
  "execution",
  "resourceMonitor",
] as const;
const PARALLELISM_MIN = -1;
const PARALLELISM_MAX = 64;
const INTERVAL_MIN_MS = 250;
const INTERVAL_MAX_MS = 60_000;

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value);

/**
 * Validate one settings candidate. Unknown keys are rejected (the
 * §10 no-unsafe-values rule — nothing unreviewed ever persists), and
 * every leaf is type- and range-checked.
 */
export function validateSettings(candidate: unknown): ValidationVerdict {
  const errors: string[] = [];
  if (!isPlainObject(candidate)) {
    return { ok: false, errors: ["settings must be an object"] };
  }
  for (const key of Object.keys(candidate)) {
    if (!(TOP_LEVEL_KEYS as readonly string[]).includes(key)) {
      errors.push(`unknown setting "${key}" (rejected — no unsafe values)`);
    }
  }
  if (candidate["schemaVersion"] !== 1) {
    errors.push("schemaVersion must be 1");
  }
  const theme = candidate["theme"];
  if (typeof theme !== "string" || !THEMES.includes(theme as ThemeSetting)) {
    errors.push('theme must be one of "light" | "dark" | "system"');
  }
  const density = candidate["density"];
  if (typeof density !== "string" || !DENSITIES.includes(density)) {
    errors.push('density must be "comfortable" | "compact"');
  }
  const group = (
    name: string,
    keys: readonly string[],
    check: (group: Record<string, unknown>) => void,
  ): void => {
    const raw = candidate[name];
    if (!isPlainObject(raw)) {
      errors.push(`${name} must be an object`);
      return;
    }
    for (const key of Object.keys(raw)) {
      if (!keys.includes(key)) {
        errors.push(`unknown setting "${name}.${key}"`);
      }
    }
    check(raw);
  };
  group("charts", ["colorblindSafe", "defaultExportFormat"], (charts) => {
    if (typeof charts["colorblindSafe"] !== "boolean") {
      errors.push("charts.colorblindSafe must be a boolean");
    }
    const format = charts["defaultExportFormat"];
    if (typeof format !== "string" || !EXPORT_FORMATS.includes(format)) {
      errors.push('charts.defaultExportFormat must be "png" | "svg"');
    }
  });
  group("notifications", ["jobCompletionToasts"], (notifications) => {
    if (typeof notifications["jobCompletionToasts"] !== "boolean") {
      errors.push("notifications.jobCompletionToasts must be a boolean");
    }
  });
  group("execution", ["defaultParallelism"], (execution) => {
    const parallelism = execution["defaultParallelism"];
    if (
      !isInteger(parallelism) ||
      parallelism < PARALLELISM_MIN ||
      parallelism > PARALLELISM_MAX
    ) {
      errors.push(
        `execution.defaultParallelism must be an integer in ${PARALLELISM_MIN}..${PARALLELISM_MAX}`,
      );
    }
  });
  group("resourceMonitor", ["updateIntervalMs", "visible"], (monitor) => {
    const interval = monitor["updateIntervalMs"];
    if (
      !isInteger(interval) ||
      interval < INTERVAL_MIN_MS ||
      interval > INTERVAL_MAX_MS
    ) {
      errors.push(
        `resourceMonitor.updateIntervalMs must be an integer in ${INTERVAL_MIN_MS}..${INTERVAL_MAX_MS}`,
      );
    }
    if (typeof monitor["visible"] !== "boolean") {
      errors.push("resourceMonitor.visible must be a boolean");
    }
  });
  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    value: candidate as unknown as AppSettings,
  };
}

/** Read the persisted settings — validated on read, null on anything invalid. */
function readPersisted(): AppSettings | null {
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (raw === null) {
      return null;
    }
    const verdict = validateSettings(JSON.parse(raw) as unknown);
    return verdict.ok ? verdict.value : null;
  } catch {
    // corrupt JSON, storage unavailable, private-mode refusals: the
    // settings degrade to the in-session defaults, never a crash
    return null;
  }
}

/** Persist — best-effort: a storage failure degrades to in-memory settings. */
function persist(settings: AppSettings): void {
  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // quota or privacy mode: the app keeps running on the memory copy
  }
}

/** The effective theme under the `system` lane (no matchMedia -> light). */
function systemPrefersDark(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function effectiveTheme(theme: ThemeSetting): "light" | "dark" {
  if (theme === "system") {
    return systemPrefersDark() ? "dark" : "light";
  }
  return theme;
}

/**
 * The settings store. `createSettingsStore()` builds an isolated
 * instance (the tests' lane); `getSettingsStore()` is the app's
 * singleton over the real localStorage.
 */
export class SettingsStore {
  private current: AppSettings;
  private readonly listeners = new Set<() => void>();
  private mediaListener: (() => void) | null = null;

  constructor() {
    this.current = readPersisted() ?? DEFAULT_SETTINGS;
    this.applyToDocument();
    if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
      // the system lane: follow the OS while theme === "system"; an
      // explicit choice simply stops matching (the listener re-checks)
      const query = window.matchMedia("(prefers-color-scheme: dark)");
      const onchange = (): void => {
        if (this.current.theme === "system") {
          this.applyTheme();
        }
      };
      query.addEventListener?.("change", onchange);
      this.mediaListener = () => query.removeEventListener?.("change", onchange);
    }
  }

  /** The referentially-stable snapshot (useSyncExternalStore-ready). */
  snapshot(): AppSettings {
    return this.current;
  }

  /** Subscribe to settings changes. Returns the unsubscribe fn. */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Update with a patch: the merged whole is validated FIRST; on any
   * error nothing changes (§10 validate-before-apply).
   */
  update(patch: SettingsPatch): ValidationVerdict {
    const merged: AppSettings = {
      ...this.current,
      ...(patch.theme !== undefined ? { theme: patch.theme } : {}),
      ...(patch.density !== undefined ? { density: patch.density } : {}),
      charts: { ...this.current.charts, ...(patch.charts ?? {}) },
      notifications: { ...this.current.notifications, ...(patch.notifications ?? {}) },
      execution: { ...this.current.execution, ...(patch.execution ?? {}) },
      resourceMonitor: { ...this.current.resourceMonitor, ...(patch.resourceMonitor ?? {}) },
    };
    const verdict = validateSettings(merged);
    if (!verdict.ok) {
      return verdict;
    }
    this.current = verdict.value;
    this.commit();
    return verdict;
  }

  /** Reset every setting to the shipped defaults. */
  reset(): void {
    this.current = DEFAULT_SETTINGS;
    this.commit();
  }

  private commit(): void {
    persist(this.current);
    this.applyToDocument();
    for (const listener of this.listeners) {
      listener();
    }
  }

  /** All three DOM applications (theme class, density, CVD palette). */
  private applyToDocument(): void {
    this.applyTheme();
    document.documentElement.dataset.density = this.current.density;
    document.documentElement.classList.toggle(
      "charts-cvd",
      this.current.charts.colorblindSafe,
    );
  }

  private applyTheme(): void {
    document.documentElement.classList.toggle(
      "dark",
      effectiveTheme(this.current.theme) === "dark",
    );
  }

  /** Tests only: detach the matchMedia listener. */
  dispose(): void {
    this.mediaListener?.();
    this.mediaListener = null;
    this.listeners.clear();
  }
}

/** The singleton (the app's store). */
let singleton: SettingsStore | null = null;

export function getSettingsStore(): SettingsStore {
  if (singleton === null) {
    singleton = new SettingsStore();
  }
  return singleton;
}

/** An isolated instance for tests (its own document applications). */
export function createSettingsStore(): SettingsStore {
  return new SettingsStore();
}
