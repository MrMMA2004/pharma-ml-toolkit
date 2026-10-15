/**
 * Small formatting helpers shared across the UI.
 */

/** Format a number to `sig` significant digits without float soup.
 *  `sig` is clamped to the 1–100 toPrecision() domain so a stray extra
 *  argument (e.g. a recharts tick INDEX passed as the second parameter)
 *  can never raise a RangeError. */
export function fmtNum(value: number | null | undefined, sig = 3): string {
  if (value === null || value === undefined) return "—";
  if (Number.isNaN(value)) return "—";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "-∞";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) {
    return value.toExponential(2).replace("e+", "e");
  }
  const digits = Math.min(Math.max(Math.round(sig), 1), 100);
  return String(Number(value.toPrecision(digits)));
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

/**
 * Subsequence fuzzy match (case-insensitive). Returns `null` when `query` is
 * not a subsequence of `target`; otherwise a score where higher is better:
 * exact substring > word-start / consecutive runs > scattered subsequence.
 */
export function fuzzyScore(query: string, target: string): number | null {
  const q = query.trim().toLowerCase();
  const t = target.toLowerCase();
  if (q.length === 0) return 0;
  const subIdx = t.indexOf(q);
  if (subIdx >= 0) return 1000 - Math.min(50, subIdx * 2);
  let qi = 0;
  let score = 0;
  let prevTi = -2;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] !== q[qi]) continue;
    score += 10;
    if (ti === prevTi + 1) score += 8; // consecutive run
    const prevChar = t[ti - 1] ?? "";
    if (ti === 0 || /[\s_\-.]/.test(prevChar)) score += 6; // word start
    prevTi = ti;
    qi += 1;
  }
  if (qi < q.length) return null;
  return score - Math.min(20, (t.length - q.length) * 0.5);
}

/**
 * Indices of the characters in `target` matched by the fuzzy query
 * (case-insensitive), consistent with `fuzzyScore`'s walk: an exact substring
 * highlights the whole substring; otherwise the subsequence positions.
 * Returns `null` when the query does not match, `[]` for an empty query.
 */
export function fuzzyMatchIndices(query: string, target: string): number[] | null {
  const q = query.trim().toLowerCase();
  const t = target.toLowerCase();
  if (q.length === 0) return [];
  const subIdx = t.indexOf(q);
  if (subIdx >= 0) {
    return Array.from({ length: q.length }, (_, i) => subIdx + i);
  }
  const indices: number[] = [];
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] !== q[qi]) continue;
    indices.push(ti);
    qi += 1;
  }
  return qi === q.length ? indices : null;
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
