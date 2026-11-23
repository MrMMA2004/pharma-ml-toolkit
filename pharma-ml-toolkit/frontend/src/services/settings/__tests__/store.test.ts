import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_SETTINGS,
  SETTINGS_STORAGE_KEY,
  createSettingsStore,
  validateSettings,
} from "@/services/settings/store";
import type { AppSettings } from "@/services/settings/types";

/**
 * part16 §10 — the Phase-7 settings store contract.
 *
 * The store is the whole lane's spine: the §10 rules pinned here are
 * (a) defaults on a clean storage, (b) validate-BEFORE-apply (an
 * invalid candidate is rejected whole — the snapshot never lands in
 * a partial state), (c) the persistence round-trip through the
 * localStorage layer, (d) reset-to-default, and (e) the three DOM
 * applications (the `.dark` theme class incl. the system lane, the
 * density dataset, the `.charts-cvd` palette class) — each asserted
 * against the live documentElement.
 */

const CLEAN: AppSettings = DEFAULT_SETTINGS;

describe("validateSettings (the §10 validate-before-apply rule)", () => {
  it("accepts the defaults object as-is", () => {
    const verdict = validateSettings(CLEAN);
    expect(verdict.ok).toBe(true);
  });

  it("rejects a wrong schemaVersion (forward-junk storage falls back to defaults)", () => {
    const verdict = validateSettings({ ...CLEAN, schemaVersion: 2 });
    expect(verdict.ok).toBe(false);
  });

  it("rejects out-of-band parallelism (the -1..64 contract of the ML form)", () => {
    for (const bad of [200, -2, 1.5, Number.NaN]) {
      const verdict = validateSettings({
        ...CLEAN,
        execution: { defaultParallelism: bad },
      });
      expect(verdict.ok).toBe(false);
    }
  });

  it("rejects a monitor interval below the 250 ms floor (busy-loop protection)", () => {
    const verdict = validateSettings({
      ...CLEAN,
      resourceMonitor: { updateIntervalMs: 100 },
    });
    expect(verdict.ok).toBe(false);
  });

  it("rejects unknown enum values and non-boolean toggles", () => {
    expect(
      validateSettings({ ...CLEAN, theme: "sepia" }).ok,
    ).toBe(false);
    expect(
      validateSettings({
        ...CLEAN,
        charts: { colorblindSafe: "yes" },
      }).ok,
    ).toBe(false);
  });

  it("rejects unknown top-level keys (the no-unsafe-values rule)", () => {
    const verdict = validateSettings({
      ...CLEAN,
      apiToken: "should-not-exist",
    });
    expect(verdict.ok).toBe(false);
  });
});

describe("the settings store (createSettingsStore)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.className = "";
    delete document.documentElement.dataset.density;
  });

  it("serves the defaults on a clean storage and applies them to the document", () => {
    const store = createSettingsStore();
    expect(store.snapshot()).toEqual(CLEAN);
    // system theme with no matchMedia preference -> light at first paint
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(document.documentElement.dataset.density).toBe("comfortable");
  });

  it("applies + persists a valid update and notifies the subscribers", () => {
    const store = createSettingsStore();
    let notified = 0;
    store.subscribe(() => {
      notified += 1;
    });

    const verdict = store.update({ theme: "dark", density: "compact" });
    expect(verdict.ok).toBe(true);
    expect(store.snapshot().theme).toBe("dark");
    expect(notified).toBe(1);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.dataset.density).toBe("compact");

    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw!).theme).toBe("dark");
  });

  it("rejects an invalid update WHOLE — the snapshot, the document and the storage stay untouched", () => {
    const store = createSettingsStore();
    store.update({ theme: "dark" });

    const before = store.snapshot();
    const verdict = store.update({ execution: { defaultParallelism: 99 } });
    expect(verdict.ok).toBe(false);
    expect((verdict as { errors: string[] }).errors.length).toBeGreaterThan(0);
    expect(store.snapshot()).toEqual(before);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("round-trips: a persisted store is re-read by the next store instance", () => {
    const first = createSettingsStore();
    first.update({
      theme: "light",
      charts: { colorblindSafe: true },
    });
    const second = createSettingsStore();
    expect(second.snapshot().theme).toBe("light");
    expect(second.snapshot().charts.colorblindSafe).toBe(true);
    // the CVD palette class follows the setting
    expect(document.documentElement.classList.contains("charts-cvd")).toBe(
      true,
    );
  });

  it("treats corrupt/invalid storage as absent (defaults, never a crash)", () => {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, "{not json");
    const store = createSettingsStore();
    expect(store.snapshot()).toEqual(CLEAN);

    window.localStorage.setItem(
      SETTINGS_STORAGE_KEY,
      JSON.stringify({ ...CLEAN, theme: "bogus" }),
    );
    expect(createSettingsStore().snapshot()).toEqual(CLEAN);
  });

  it("reset() restores the defaults, persists them and notifies", () => {
    const store = createSettingsStore();
    store.update({ theme: "dark", density: "compact" });
    let notified = 0;
    store.subscribe(() => {
      notified += 1;
    });

    store.reset();
    expect(store.snapshot()).toEqual(CLEAN);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(document.documentElement.dataset.density).toBe("comfortable");
    expect(notified).toBe(1);
    expect(JSON.parse(window.localStorage.getItem(SETTINGS_STORAGE_KEY)!)).toEqual(CLEAN);
  });

  it("follows the OS preference while theme is system (matchMedia change -> class flip)", () => {
    const handlers: Array<(event: { matches: boolean }) => void> = [];
    let dark = false;
    window.matchMedia = ((query: string) => ({
      media: query,
      matches: dark,
      addEventListener: (_: string, cb: (event: { matches: boolean }) => void) => {
        handlers.push(cb);
      },
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
      onchange: null,
    })) as unknown as typeof window.matchMedia;
    const fire = (matches: boolean): void => {
      for (const handler of handlers) {
        handler({ matches });
      }
    };

    const store = createSettingsStore();
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    dark = true;
    fire(true);
    expect(document.documentElement.classList.contains("dark")).toBe(true);

    // an explicit choice stops following the OS
    store.update({ theme: "light" });
    dark = false;
    fire(false);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    dark = true;
    fire(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });
});
