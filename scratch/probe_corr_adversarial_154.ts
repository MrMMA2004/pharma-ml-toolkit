/**
 * 100x-154 audit probe A, TS side -- the SHIPPED pearsonCorrelation
 * against the adversarial numerics grid the round-36 parity grid
 * missed (huge magnitudes, huge offsets, near-constant, tiny values,
 * exact cancellation, n=2, n=100_000, mirrored zero).
 *
 * Reads /tmp/probe_a_154_in.json (name + two float64 columns), writes
 * /tmp/probe_a_154_ts.json (name + r or null when the module returns
 * NaN/undefined -- the honest undefined verdict).  PRIVATE paths: the
 * 100x-154 parity twin owns /tmp/corr_adv_154_*.json -- sharing them
 * let a stale probe output poison the twin's cross-language check.
 */
import { pearsonCorrelation } from "../src/lib/ml-math";
import { readFileSync, writeFileSync } from "node:fs";

interface Case {
  name: string;
  a: number[];
  b: number[];
}

const cases = JSON.parse(readFileSync("/tmp/probe_a_154_in.json", "utf8")) as Case[];
const out = cases.map((c) => {
  const r = pearsonCorrelation(c.a, c.b);
  return { name: c.name, r: Number.isFinite(r) ? r : null };
});
writeFileSync("/tmp/probe_a_154_ts.json", JSON.stringify(out));
console.log("TS_SIDE_DONE", out.length);
