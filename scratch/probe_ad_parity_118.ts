/**
 * Bun twin of the ml-ad.ts parity pair (100x-118): runs the SHIPPED
 * TS module (fitStandardizer + knnDomain) on the SAME deterministic
 * LCG dataset and compares against the Python side's JSON readout —
 * the browser's kNN math must match pharmaml.ml.applicability's to
 * 1e-9 (both are IEEE doubles; the loop order is mirrored).
 *
 * Run AFTER the python side:
 *     python ../pharma-ml-toolkit/scratch/probe_ad_parity_118.py
 *     bun run scratch/probe_ad_parity_118.ts
 */
import { readFileSync } from "fs";
import { fitStandardizer, knnDomain } from "../src/lib/ml-ad";

/* The same minstd (Lehmer) generator — bit-exact with the Python side
 * (products < 2^46: doubles never lose a bit). */
let state = 42;
const MOD = 2 ** 31 - 1;
function lcg(): number {
  state = (48271 * state) % MOD;
  return state / MOD;
}

const rows: number[][] = [];
for (let i = 0; i < 60; i += 1) rows.push([lcg(), lcg(), lcg()]);
const query: number[][] = [rows[0]!, rows[7]!, rows[15]!, [5.0, -4.0, 6.0]];

const scaler = fitStandardizer(rows);
const trainZ = rows.map((r) => scaler.z(r));
const testZ = query.map((r) => scaler.z(r));
const domain = knnDomain(trainZ, testZ, 5, 95);

const expected = JSON.parse(
  readFileSync("/tmp/ad_parity_py.json", "utf8")
) as {
  trainMean: number[];
  testMean: number[];
  threshold: number;
  inside: boolean[];
  featureNames: string[];
};

let failures = 0;
function close(a: number, b: number): boolean {
  const scale = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= 1e-9 * scale;
}
function check(name: string, ok: boolean, detail: string): void {
  if (!ok) {
    failures += 1;
    console.error(`FAIL ${name}: ${detail}`);
  } else {
    console.log(`ok   ${name} ${detail}`);
  }
}

check("threshold", close(domain.threshold, expected.threshold),
  `ts=${domain.threshold.toPrecision(12)} py=${expected.threshold.toPrecision(12)}`);
check("trainMean length", domain.trainMean.length === expected.trainMean.length,
  `${domain.trainMean.length} vs ${expected.trainMean.length}`);
let worstTrain = 0;
for (let i = 0; i < Math.min(domain.trainMean.length, expected.trainMean.length); i += 1) {
  worstTrain = Math.max(worstTrain, Math.abs(domain.trainMean[i]! - expected.trainMean[i]!));
}
check("trainMean parity", worstTrain <= 1e-9, `max|d|=${worstTrain.toExponential(2)}`);
let worstTest = 0;
for (let i = 0; i < Math.min(domain.testMean.length, expected.testMean.length); i += 1) {
  worstTest = Math.max(worstTest, Math.abs(domain.testMean[i]! - expected.testMean[i]!));
}
check("testMean parity", worstTest <= 1e-9, `max|d|=${worstTest.toExponential(2)}`);
check("inside verdicts", JSON.stringify(domain.inside) === JSON.stringify(expected.inside),
  `ts=${domain.inside.map((v) => (v ? "T" : "F")).join("")} py=${expected.inside.map((v) => (v ? "T" : "F")).join("")}`);
/* The far query row must be outside in both languages. */
check("far row outside", domain.inside[3] === false && expected.inside[3] === false,
  `ts=${domain.inside[3]} py=${expected.inside[3]}`);

/* The self-exclusion invariant: a uniform grid's interior row has mean
 * distance exactly 1.8 * spacing / std (the documented upgrade over the
 * upload's self-including math). */
const grid: number[][] = [];
for (let i = 0; i < 101; i += 1) grid.push([i / 10]);
const gridScaler = fitStandardizer(grid);
const gridDomain = knnDomain(
  grid.map((r) => gridScaler.z(r)),
  [[0.5]],
  5,
  95
);
let sum = 0;
for (const r of grid) sum += r[0]!;
const mean = sum / grid.length;
let varAcc = 0;
for (const r of grid) varAcc += (r[0]! - mean) ** 2;
const std = Math.sqrt(varAcc / grid.length);
const expectedInterior = (1.8 * 0.1) / std;
check("self-exclusion grid", Math.abs(gridDomain.trainMean[50]! - expectedInterior) < 1e-12,
  `ts=${gridDomain.trainMean[50]!.toPrecision(12)} closed-form=${expectedInterior.toPrecision(12)}`);

if (failures > 0) {
  console.error(`AD PARITY FAIL: ${failures} check(s)`);
  process.exit(1);
}
console.log("AD PARITY PASS — ml-ad.ts matches pharmaml.ml.applicability (kNN half)");
