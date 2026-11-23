/**
 * part16 §8 — the floating panel's geometry persistence contract.
 *
 * "Remembers its last position and preferred size where practical" —
 * the schema-validated localStorage pocket `pharmaml.studio.resmon.v1`
 * (deliberately SEPARATE from the settings store's section-10 schema,
 * which stays byte-frozen): validate-on-read so corrupt storage degrades
 * to the defaults, never a crash; clamp-on-write so a stale saved
 * geometry from a larger screen still lands inside the viewport and
 * within the width limits.
 */
import { beforeEach, describe, expect, it } from "vitest";

import {
  clampGeometry,
  DEFAULT_PANEL_WIDTH,
  loadPanelState,
  PANEL_MAX_WIDTH,
  PANEL_MIN_WIDTH,
  PANEL_STORAGE_KEY,
  savePanelState,
  type PanelGeometry,
} from "@/services/system/panelState";

beforeEach(() => {
  window.localStorage.clear();
});

describe("the panel geometry pocket (§8)", () => {
  it("degrades to the defaults on fresh or corrupt storage — never a crash", () => {
    const fresh = loadPanelState();
    expect(fresh.width).toBe(DEFAULT_PANEL_WIDTH);
    expect(fresh.expanded).toBe(false);
    expect(fresh.x).toBe(window.innerWidth - DEFAULT_PANEL_WIDTH - 16);
    expect(fresh.y).toBe(88);

    window.localStorage.setItem(PANEL_STORAGE_KEY, "{not json at all");
    const corrupt = loadPanelState();
    expect(corrupt).toEqual(fresh);

    window.localStorage.setItem(
      PANEL_STORAGE_KEY,
      JSON.stringify({ x: "left", y: null, width: [] }),
    );
    expect(loadPanelState()).toEqual(fresh);
  });

  it("round-trips a valid geometry through save and load", () => {
    const wanted: PanelGeometry = { x: 100, y: 200, width: 400, expanded: true };
    savePanelState(wanted);

    expect(loadPanelState()).toEqual(wanted);
    expect(window.localStorage.getItem(PANEL_STORAGE_KEY)).toBeTruthy();
  });

  it("clamps an out-of-bounds geometry inside the viewport and width limits", () => {
    const clamped = clampGeometry({ x: -50, y: -10, width: 9_999, expanded: false });
    expect(clamped.x).toBe(0);
    expect(clamped.y).toBe(0);
    expect(clamped.width).toBe(PANEL_MAX_WIDTH);
    expect(clamped.expanded).toBe(false);

    const under = clampGeometry({ x: 10_000, y: 4_000, width: 10, expanded: true });
    expect(under.width).toBe(PANEL_MIN_WIDTH);
    expect(under.x).toBeLessThanOrEqual(window.innerWidth - 60);
    expect(under.y).toBeLessThanOrEqual(window.innerHeight - 80);
    expect(under.expanded).toBe(true);
  });

  it("persisted geometry survives a remount round trip", () => {
    savePanelState({ x: 140, y: 260, width: 420, expanded: true });
    // a fresh load (the remount path) restores exactly what was saved
    expect(loadPanelState()).toEqual({ x: 140, y: 260, width: 420, expanded: true });
  });
});
