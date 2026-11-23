/**
 * part16 §10 — the Phase-7 settings schema.
 *
 * The single typed shape every settings consumer reads. The §10
 * inventory this lane persists:
 *
 * * theme — light / dark / system (the persistent theme preference);
 * * density — the interface-density scale;
 * * charts — the colorblind-safe palette switch + the default chart
 *   export format (the plotly modebar camera honors it);
 * * notifications — the job-completion toast toggle (a REAL consumer:
 *   the shell's toast watcher);
 * * execution — the scientific execution default (the ML form's
 *   n_jobs prefill; the -1..64 passthrough contract of the r344
 *   Phase-2 fix, default 1 conservative);
 * * resourceMonitor — the Phase-8 floating monitor's preferences
 *   (roadmap #5 — validated + persisted here, consumed when that
 *   lane ships).
 *
 * §10's no-unsafe-values rule: the schema carries NO secrets, no
 * URLs, no tokens — validation rejects unknown keys outright.
 */

/** The §10 theme vocabulary. `system` follows `prefers-color-scheme`. */
export type ThemeSetting = "light" | "dark" | "system";

/** The interface-density scale (the root font-size lane). */
export type DensitySetting = "comfortable" | "compact";

/** The chart export formats plotly's modebar camera supports. */
export type ChartExportFormat = "png" | "svg";

/** The persisted settings object. `schemaVersion` gates migrations. */
export interface AppSettings {
  readonly schemaVersion: 1;
  readonly theme: ThemeSetting;
  readonly density: DensitySetting;
  readonly charts: {
    readonly colorblindSafe: boolean;
    readonly defaultExportFormat: ChartExportFormat;
  };
  readonly notifications: {
    readonly jobCompletionToasts: boolean;
  };
  readonly execution: {
    readonly defaultParallelism: number;
  };
  readonly resourceMonitor: {
    readonly updateIntervalMs: number;
    readonly visible: boolean;
  };
}

/** The shipped defaults — conservative, honest, offline-first. */
export const DEFAULT_SETTINGS: AppSettings = {
  schemaVersion: 1,
  theme: "system",
  density: "comfortable",
  charts: {
    colorblindSafe: false,
    defaultExportFormat: "png",
  },
  notifications: {
    jobCompletionToasts: true,
  },
  execution: {
    defaultParallelism: 1,
  },
  resourceMonitor: {
    updateIntervalMs: 2_000,
    visible: true,
  },
};

/**
 * A partial update — one honest surface for the panel: any subset of
 * the leaf values. `update()` validates the MERGED whole before
 * anything applies (§10 validate-before-apply).
 */
export interface SettingsPatch {
  readonly theme?: ThemeSetting;
  readonly density?: DensitySetting;
  readonly charts?: { readonly colorblindSafe?: boolean; readonly defaultExportFormat?: ChartExportFormat };
  readonly notifications?: { readonly jobCompletionToasts?: boolean };
  readonly execution?: { readonly defaultParallelism?: number };
  readonly resourceMonitor?: { readonly updateIntervalMs?: number; readonly visible?: boolean };
}
