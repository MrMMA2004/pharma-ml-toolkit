/**
 * Small formatting helpers shared across the UI.
 */

/** Format a number to `sig` significant digits without float soup. */
export function fmtNum(value: number | null | undefined, sig = 3): string {
  if (value === null || value === undefined) return "—";
  if (Number.isNaN(value)) return "—";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "-∞";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) {
    return value.toExponential(2).replace("e+", "e");
  }
  return String(Number(value.toPrecision(sig)));
}

/** Format bytes as B / KB / MB. */
export function fmtBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Integer with thousands separators. */
export function fmtInt(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

/** Chart axis tick formatter (2 significant digits). */
export function fmtTick(value: number): string {
  return fmtNum(value, 2);
}

/** Parse a user-typed numeric string; null when not a finite number. */
export function parsePositiveNumber(raw: string): number | null {
  const v = Number.parseFloat(raw);
  if (!Number.isFinite(v) || v <= 0) return null;
  return v;
}

/** Parse a user-typed numeric string; null when not finite or negative. */
export function parseNonNegativeNumber(raw: string): number | null {
  const v = Number.parseFloat(raw);
  if (!Number.isFinite(v) || v < 0) return null;
  return v;
}

/** Copy text to the clipboard with a legacy fallback. */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the legacy path
  }
  try {
    const el = document.createElement("textarea");
    el.value = text;
    el.setAttribute("readonly", "");
    el.style.position = "fixed";
    el.style.opacity = "0";
    document.body.appendChild(el);
    el.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(el);
    return ok;
  } catch {
    return false;
  }
}
