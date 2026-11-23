/**
 * part16 §8 — the floating panel's geometry pocket.
 *
 * "Remembers its last position and preferred size where practical."
 * A schema-validated localStorage pocket SEPARATE from the settings
 * store (`pharmaml.studio.resmon.v1`): the §10 settings schema stays
 * byte-frozen — geometry is presentation state, not a preference.
 * Validate-on-read so corrupt storage degrades to the defaults,
 * never a crash; clamp-on-write so a geometry saved on a larger
 * screen still lands inside the viewport and within the width
 * limits on the current one.
 */

export const PANEL_STORAGE_KEY = "pharmaml.studio.resmon.v1";
export const PANEL_MIN_WIDTH = 280;
export const PANEL_MAX_WIDTH = 480;
export const DEFAULT_PANEL_WIDTH = 320;

export interface PanelGeometry {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly expanded: boolean;
}

const KEYBOARD_STEP = 10;
const VIEWPORT_RIGHT_MARGIN = 60;
const VIEWPORT_BOTTOM_MARGIN = 80;
const DEFAULT_Y = 88;
const DEFAULT_RIGHT_MARGIN = 16;

export function clampWidth(width: unknown): number {
  const value = typeof width === "number" && Number.isFinite(width)
    ? width
    : DEFAULT_PANEL_WIDTH;
  return Math.min(PANEL_MAX_WIDTH, Math.max(PANEL_MIN_WIDTH, Math.round(value)));
}

/** Clamp a geometry inside the viewport (at most a grip of it stays reachable). */
export function clampGeometry(geometry: PanelGeometry): PanelGeometry {
  const width = clampWidth(geometry.width);
  const maxX = Math.max(0, window.innerWidth - VIEWPORT_RIGHT_MARGIN);
  const maxY = Math.max(0, window.innerHeight - VIEWPORT_BOTTOM_MARGIN);
  const x = Math.min(maxX, Math.max(0, Math.round(geometry.x)));
  const y = Math.min(maxY, Math.max(0, Math.round(geometry.y)));
  return {
    x,
    y,
    width,
    expanded: geometry.expanded === true,
  };
}

function defaultGeometry(): PanelGeometry {
  return clampGeometry({
    x: window.innerWidth - DEFAULT_PANEL_WIDTH - DEFAULT_RIGHT_MARGIN,
    y: DEFAULT_Y,
    width: DEFAULT_PANEL_WIDTH,
    expanded: false,
  });
}

/** Load the pocket — validate-on-read, degrade to defaults, never crash. */
export function loadPanelState(): PanelGeometry {
  try {
    const raw = window.localStorage.getItem(PANEL_STORAGE_KEY);
    if (raw === null) {
      return defaultGeometry();
    }
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) {
      return defaultGeometry();
    }
    const r = parsed as Record<string, unknown>;
    if (typeof r["x"] !== "number" || typeof r["y"] !== "number") {
      return defaultGeometry();
    }
    return clampGeometry({
      x: r["x"],
      y: r["y"],
      width: typeof r["width"] === "number" ? r["width"] : DEFAULT_PANEL_WIDTH,
      expanded: r["expanded"] === true,
    });
  } catch {
    return defaultGeometry();
  }
}

/** Save the pocket — clamped first, so only reachable geometry persists. */
export function savePanelState(geometry: PanelGeometry): void {
  try {
    window.localStorage.setItem(
      PANEL_STORAGE_KEY,
      JSON.stringify(clampGeometry(geometry)),
    );
  } catch {
    // storage unavailable (private mode / quota) — the monitor still works,
    // it just will not remember this session's geometry
  }
}

/** The keyboard lane's step (the accessible move / resize). */
export const KEYBOARD_GEOMETRY_STEP = KEYBOARD_STEP;
