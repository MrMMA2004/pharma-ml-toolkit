"use client";

import { useState } from "react";
import {
  Activity,
  BarChart3,
  BookOpen,
  Copy,
  Download,
  ExternalLink,
  FileSearch,
  FileText,
  Grid3x3,
  Info,
  Loader2,
  Network,
  Orbit,
  Play,
  Plus,
  RotateCcw,
  ScatterChart,
  TriangleAlert,
  X,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CHART, ParamSlider } from "@/components/explorer/playground/calc-ui";
import { copyToClipboard, fmtNum } from "@/lib/format";
import type {
  MdTrajectoryResponse,
  MdXvgResponse,
  MdXvgSeries,
} from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Molecular Dynamics (round 249) — the pharmaml.md web twins          */
/* ------------------------------------------------------------------ */
/**
 * The "Molecular Dynamics" playground sub-tab, three cards:
 *
 *  (A) GROMACS export parser — paste .xvg/.ndx exports, one POST runs
 *      the REAL pharmaml.md.study pipeline on the server (the
 *      productionised upload/MD_analysis.py script): kJ→kcal, ps→ns,
 *      3-point H-bond smoothing, per-metric stats, the cross-metric
 *      correlation matrix and the summary CSV.
 *  (B) Trajectory analysis lab — a deterministic synthetic alpha-helix
 *      analysed by the real toolkit (RMSD after Kabsch superposition,
 *      Rg, native-contact Q, RMSF, contact map, PCA, ANM) behind four
 *      inner tabs.
 *  (C) The MD ecosystem — the static reference-card grid of the open
 *      source stack behind pharmaml.md.
 *
 * Charts are hand-rolled inline SVG in the house style of the
 * ml-playground cards (viewBox + w-full h-auto, muted-foreground mono
 * ticks, the CHART palette — no blue/indigo, no new chart libraries);
 * every async card follows the sa-score-card discipline: idle/running/
 * done/error, Loader2 spinner buttons, destructive Alert on failure,
 * honest stats and a method footnote that never lies.
 */

type Status = "idle" | "running" | "done" | "error";

/** Relative-URL POST with the sa-score-card unknown-narrowing discipline. */
async function postMdJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data: unknown = await res.json();
  if (!res.ok) {
    const message =
      typeof data === "object" && data !== null && "error" in data
        ? String((data as { error: unknown }).error)
        : `HTTP ${res.status}`;
    throw new Error(message);
  }
  return data as T;
}

/** Metric-key → series accent (house palette; kind order mirrors the
 * backend's classify_metric: "energy" before "rg" because "eneRGy"
 * contains those letters). */
function metricColor(key: string): string {
  const k = key.toLowerCase();
  if (k.includes("rmsd")) return CHART.rose;
  if (k.includes("rmsf")) return CHART.emerald;
  if (k.includes("hbnum") || k.includes("hbond")) return CHART.teal;
  if (k.includes("energy")) return CHART.emeraldDark;
  if (k.includes("rg")) return CHART.amber;
  if (k.includes("sasa")) return CHART.teal;
  return CHART.stone;
}

/** Metric-key → display unit ("" when the label already says it). The
 * kind order mirrors the backend's classify_metric: "energy" is tested
 * before "rg" because "eneRGy" itself contains those letters. */
function metricUnit(key: string): string {
  const k = key.toLowerCase();
  if (k.includes("rmsd") || k.includes("rmsf")) return "Å";
  if (k.includes("sasa")) return "Å²";
  if (k.includes("energy")) return "kcal/mol";
  if (k.includes("rg")) return "Å";
  return "";
}

/** Metric-key → x-axis label: RMSF exports index atoms/residues, every
 * other metric is a normalised time axis (ps→ns happened server-side). */
function metricXLabel(key: string): string {
  return key.toLowerCase().includes("rmsf") ? "atom / residue index" : "time (ns)";
}

/* ------------------------------------------------------------------ */
/* Demo study — deterministic GROMACS-style exports (~60 rows each)    */
/* ------------------------------------------------------------------ */

/** Deterministic LCG in [-0.5, 0.5): the demo text is byte-identical
 * on every click and every machine — no Math.random anywhere. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

/** ligand_rmsd.xvg — 0–60 000 ps axis, RMSD rising 0.05→0.35 Å with
 * noise over the first 18 ns, then the plateau. */
function demoLigandRmsd(): string {
  const rnd = lcg(7);
  const rows: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const target = i <= 18 ? 0.05 + (0.3 * i) / 18 : 0.35;
    rows.push(`   ${(i * 1000).toFixed(1)}   ${(target + rnd() * 0.03).toFixed(4)}`);
  }
  return `# This file was created Thu May  8 09:14:22 2025
# by the following command:
#               gmx rms -s md_0_1.tpr -f md_0_1.xtc -n index.ndx
#
@    title "Ligand RMSD"
@    xaxis  label "Time (ps)"
@    yaxis  label "RMSD (A)"
@    s0 legend "ligand RMSD"
@TYPE xy
${rows.join("\n")}
`;
}

/** hbnum_intermolecular.ndx — time + integer H-bond counts (2–6) + a
 * decoy third float column, so the card exercises the .ndx column
 * heuristic (monotonic time column, near-integer small-mean H-bond
 * column — the decoy distance column is ignored). */
function demoHbnumNdx(): string {
  const rnd = lcg(11);
  const rows: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const hb = Math.max(2, Math.min(6, Math.round(4 + Math.sin(i * 0.55) * 1.6 + rnd() * 0.9)));
    const decoy = 0.29 + rnd() * 0.03;
    rows.push(`   ${(i * 1000).toFixed(1)}   ${hb}   ${decoy.toFixed(3)}`);
  }
  return `# gmx hbond -f md_0_1.xtc -s md_0_1.tpr -num hbnum_intermolecular
# columns: time (ps) | intermolecular H-bonds | mean donor-acceptor distance (nm)
@    title "H Bonds"
@    xaxis  label "Time (ps)"
@    yaxis  label "H Bonds"
@    s0 legend "Hydrogen bonds"
${rows.join("\n")}
`;
}

/** binding_free_energy.xvg — time + ΔG in kJ/mol around −60 ± 15, so
 * the parsed card demonstrates the pipeline's kJ→kcal conversion. */
function demoBindingEnergy(): string {
  const rnd = lcg(23);
  const rows: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const v = -60 + Math.sin(i * 0.31) * 8 + rnd() * 14;
    rows.push(`   ${(i * 1000).toFixed(1)}   ${v.toFixed(3)}`);
  }
  return `# gmx_MMPBSA -o binding_free_energy.xvg
# DELTA TOTAL (kJ/mol) — converted to kcal/mol by the study reader
@    title "dG bind"
@    xaxis  label "Time (ps)"
@    yaxis  label "Energy (kJ/mol)"
@    s0 legend "DELTA TOTAL"
${rows.join("\n")}
`;
}

/* ------------------------------------------------------------------ */
/* Shared inline-SVG chart atoms (ml-playground house style)           */
/* ------------------------------------------------------------------ */

const MONO_TICK = { fontSize: 9, fontFamily: "var(--font-mono, monospace)" } as const;
const MONO_AXIS = { fontSize: 10, fontFamily: "var(--font-mono, monospace)" } as const;

/** One-series line chart, y scaled to the series' own min/max. */
function MiniLineChart({
  time,
  values,
  color,
  xLabel,
  yLabel,
  ariaLabel,
}: {
  time: number[];
  values: number[];
  color: string;
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
}) {
  const W = 560;
  const H = 168;
  const pad = { l: 46, r: 12, top: 16, b: 24 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;

  const dataLo = Math.min(...values);
  const dataHi = Math.max(...values);
  const span = dataHi - dataLo;
  const guard = span || Math.max(Math.abs(dataLo) || 1, 1);
  const lo = dataLo - guard * 0.08;
  const hi = dataHi + guard * 0.08;

  const t0 = time[0] ?? 0;
  const t1 = time.length > 1 ? (time[time.length - 1] ?? t0) : t0 + 1;
  const tSpan = t1 - t0 || 1;
  const px = (t: number) => pad.l + ((t - t0) / tSpan) * pw;
  const py = (v: number) => pad.top + ph - ((v - lo) / (hi - lo)) * ph;

  const yTicks = Array.from({ length: 5 }, (_, i) => dataLo + (span * i) / 4);
  const xTicks = Array.from({ length: 5 }, (_, i) => t0 + (tSpan * i) / 4);

  const lastIdx = time.length - 1;
  const lastT = time[lastIdx] ?? t0;
  const lastV = values[lastIdx] ?? dataLo;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      {yTicks.map((v, i) => (
        <g key={`gy-${i}`}>
          <line
            x1={pad.l}
            y1={py(v)}
            x2={pad.l + pw}
            y2={py(v)}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text
            x={pad.l - 6}
            y={py(v) + 3}
            textAnchor="end"
            className="fill-muted-foreground tabular-nums"
            style={MONO_TICK}
          >
            {fmtNum(v, 3)}
          </text>
        </g>
      ))}
      {xTicks.map((t, i) => (
        <g key={`gx-${i}`}>
          <line
            x1={px(t)}
            y1={pad.top}
            x2={px(t)}
            y2={pad.top + ph}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text
            x={px(t)}
            y={pad.top + ph + 12}
            textAnchor="middle"
            className="fill-muted-foreground tabular-nums"
            style={MONO_TICK}
          >
            {fmtNum(t, 3)}
          </text>
        </g>
      ))}
      <rect
        x={pad.l}
        y={pad.top}
        width={pw}
        height={ph}
        fill="none"
        stroke={CHART.gridStroke}
        strokeOpacity={0.4}
      />
      <polyline
        points={time
          .map((t, i) => `${px(t).toFixed(2)},${py(values[i] ?? dataLo).toFixed(2)}`)
          .join(" ")}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={px(lastT)} cy={py(lastV)} r={3.5} fill={color} stroke="currentColor" strokeOpacity={0.4} strokeWidth={1}>
        <title>{`last point: ${fmtNum(lastV, 4)}`}</title>
      </circle>
      <text x={pad.l + pw} y={H - 2} textAnchor="end" className="fill-muted-foreground" style={MONO_AXIS}>
        {xLabel}
      </text>
      <text x={pad.l} y={9} textAnchor="start" className="fill-muted-foreground" style={MONO_AXIS}>
        {yLabel}
      </text>
    </svg>
  );
}

/** Multi-series timeline: each series on its OWN vertical scale (the
 * legend carries the per-series ranges — three different units share
 * one honest canvas without lying about magnitudes). */
function TimelineChart({
  time,
  series,
  ariaLabel,
}: {
  time: number[];
  series: Array<{ label: string; values: number[]; color: string }>;
  ariaLabel: string;
}) {
  const W = 560;
  const H = 240;
  const pad = { l: 40, r: 14, top: 14, b: 26 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;

  const t0 = time[0] ?? 0;
  const t1 = time.length > 1 ? (time[time.length - 1] ?? t0) : t0 + 1;
  const tSpan = t1 - t0 || 1;
  const px = (t: number) => pad.l + ((t - t0) / tSpan) * pw;

  const scales = series.map((s) => {
    const lo = Math.min(...s.values);
    const hi = Math.max(...s.values);
    const span = hi - lo || Math.max(Math.abs(lo) || 1, 1) * 0.1;
    return { lo: lo - span * 0.06, hi: hi + span * 0.06 };
  });
  const py = (sIdx: number, v: number) => {
    const sc = scales[sIdx] ?? { lo: 0, hi: 1 };
    return pad.top + ph - ((v - sc.lo) / (sc.hi - sc.lo)) * ph;
  };

  const xTicks = Array.from({ length: 5 }, (_, i) => t0 + (tSpan * i) / 4);
  const fracs = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      {fracs.map((f, i) => {
        const y = pad.top + ph - f * ph;
        return (
          <g key={`gy-${i}`}>
            <line
              x1={pad.l}
              y1={y}
              x2={pad.l + pw}
              y2={y}
              stroke={CHART.gridStroke}
              strokeOpacity={CHART.gridStrokeOpacity}
            />
          </g>
        );
      })}
      {xTicks.map((t, i) => (
        <g key={`gx-${i}`}>
          <line
            x1={px(t)}
            y1={pad.top}
            x2={px(t)}
            y2={pad.top + ph}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text
            x={px(t)}
            y={pad.top + ph + 13}
            textAnchor="middle"
            className="fill-muted-foreground tabular-nums"
            style={MONO_TICK}
          >
            {fmtNum(t, 3)}
          </text>
        </g>
      ))}
      <rect
        x={pad.l}
        y={pad.top}
        width={pw}
        height={ph}
        fill="none"
        stroke={CHART.gridStroke}
        strokeOpacity={0.4}
      />
      {series.map((s, si) => {
        const sc = scales[si] ?? { lo: 0, hi: 1 };
        const lastIdx = s.values.length - 1;
        const lastV = s.values[lastIdx] ?? sc.lo;
        return (
          <g key={s.label}>
            <polyline
              points={time
                .map((t, i) => `${px(t).toFixed(2)},${py(si, s.values[i] ?? sc.lo).toFixed(2)}`)
                .join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <circle cx={px(t1)} cy={py(si, lastV)} r={3.5} fill={s.color} stroke="currentColor" strokeOpacity={0.4} strokeWidth={1}>
              <title>{`${s.label}: ${fmtNum(lastV, 4)}`}</title>
            </circle>
          </g>
        );
      })}
      <text x={pad.l + pw} y={H - 2} textAnchor="end" className="fill-muted-foreground" style={MONO_AXIS}>
        frame
      </text>
    </svg>
  );
}

/** Vertical bar chart (RMSF / ANM): bars rise from a zero baseline,
 * per-bar <title> tooltips, optional amber highlight indices. */
function MiniBarChart({
  values,
  color,
  highlightIndices,
  highlightColor = CHART.amber,
  labelEvery = 1,
  barTitle,
  xLabel,
  yLabel,
  ariaLabel,
}: {
  values: number[];
  color: string;
  highlightIndices?: ReadonlySet<number>;
  highlightColor?: string;
  labelEvery?: number;
  barTitle: (index: number) => string;
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
}) {
  const W = 560;
  const H = 200;
  const pad = { l: 46, r: 10, top: 14, b: 22 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;
  const n = values.length;
  const dataHi = Math.max(...values, 0);
  const hi = dataHi * 1.12 || 1;
  const slot = pw / Math.max(n, 1);
  const bw = Math.max(2, slot * 0.68);
  const baseY = pad.top + ph;
  const py = (v: number) => pad.top + ph - (v / hi) * ph;
  const yTicks = Array.from({ length: 5 }, (_, i) => (dataHi * i) / 4);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      {yTicks.map((v, i) => (
        <g key={`gy-${i}`}>
          <line
            x1={pad.l}
            y1={py(v)}
            x2={pad.l + pw}
            y2={py(v)}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text
            x={pad.l - 6}
            y={py(v) + 3}
            textAnchor="end"
            className="fill-muted-foreground tabular-nums"
            style={MONO_TICK}
          >
            {fmtNum(v, 3)}
          </text>
        </g>
      ))}
      <rect
        x={pad.l}
        y={pad.top}
        width={pw}
        height={ph}
        fill="none"
        stroke={CHART.gridStroke}
        strokeOpacity={0.4}
      />
      {values.map((v, i) => {
        const highlighted = highlightIndices?.has(i) ?? false;
        const barX = pad.l + i * slot + (slot - bw) / 2;
        const barH = Math.max(1, baseY - py(v));
        return (
          <rect
            key={`bar-${i}`}
            x={barX}
            y={baseY - barH}
            width={bw}
            height={barH}
            rx={1.5}
            fill={highlighted ? highlightColor : color}
          >
            <title>{barTitle(i)}</title>
          </rect>
        );
      })}
      {values.map((v, i) =>
        i % labelEvery === 0 || i === n - 1 ? (
          <text
            key={`bx-${i}`}
            x={pad.l + i * slot + slot / 2}
            y={baseY + 12}
            textAnchor="middle"
            className="fill-muted-foreground tabular-nums"
            style={MONO_TICK}
          >
            {i + 1}
          </text>
        ) : null,
      )}
      <text x={pad.l + pw} y={H - 2} textAnchor="end" className="fill-muted-foreground" style={MONO_AXIS}>
        {xLabel}
      </text>
      <text x={pad.l} y={9} textAnchor="start" className="fill-muted-foreground" style={MONO_AXIS}>
        {yLabel}
      </text>
    </svg>
  );
}

/** PCA scatter — the frames on PC1 × PC2, coloured light→dark by frame
 * index; a faint dashed polyline joins consecutive frames (the slow
 * rotational drift reads as the path). */
const PCA_LIGHT = [153, 246, 228];
const PCA_DARK = [15, 118, 110];

function frameColor(index: number, total: number): string {
  const t = total > 1 ? index / (total - 1) : 0;
  const r = Math.round(PCA_LIGHT[0] + (PCA_DARK[0] - PCA_LIGHT[0]) * t);
  const g = Math.round(PCA_LIGHT[1] + (PCA_DARK[1] - PCA_LIGHT[1]) * t);
  const b = Math.round(PCA_LIGHT[2] + (PCA_DARK[2] - PCA_LIGHT[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function PcaScatter({
  projection,
  evr,
  ariaLabel,
}: {
  projection: number[][];
  evr: number[];
  ariaLabel: string;
}) {
  const W = 560;
  const H = 260;
  const pad = { l: 48, r: 14, top: 16, b: 26 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;

  const xs = projection.map((p) => p[0] ?? 0);
  const ys = projection.map((p) => p[1] ?? 0);
  const xLo0 = Math.min(...xs);
  const xHi0 = Math.max(...xs);
  const yLo0 = Math.min(...ys);
  const yHi0 = Math.max(...ys);
  const xSpan = xHi0 - xLo0 || 1;
  const ySpan = yHi0 - yLo0 || 1;
  const xLo = xLo0 - xSpan * 0.05;
  const xHi = xHi0 + xSpan * 0.05;
  const yLo = yLo0 - ySpan * 0.05;
  const yHi = yHi0 + ySpan * 0.05;
  const px = (v: number) => pad.l + ((v - xLo) / (xHi - xLo)) * pw;
  const py = (v: number) => pad.top + ph - ((v - yLo) / (yHi - yLo)) * ph;

  const xTicks = Array.from({ length: 5 }, (_, i) => xLo0 + (xSpan * i) / 4);
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo0 + (ySpan * i) / 4);

  const evr1 = (evr[0] ?? 0) * 100;
  const evr2 = (evr[1] ?? 0) * 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      {yTicks.map((v, i) => (
        <g key={`gy-${i}`}>
          <line
            x1={pad.l}
            y1={py(v)}
            x2={pad.l + pw}
            y2={py(v)}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text
            x={pad.l - 6}
            y={py(v) + 3}
            textAnchor="end"
            className="fill-muted-foreground tabular-nums"
            style={MONO_TICK}
          >
            {fmtNum(v, 2)}
          </text>
        </g>
      ))}
      {xTicks.map((v, i) => (
        <g key={`gx-${i}`}>
          <line
            x1={px(v)}
            y1={pad.top}
            x2={px(v)}
            y2={pad.top + ph}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text
            x={px(v)}
            y={pad.top + ph + 12}
            textAnchor="middle"
            className="fill-muted-foreground tabular-nums"
            style={MONO_TICK}
          >
            {fmtNum(v, 2)}
          </text>
        </g>
      ))}
      <rect
        x={pad.l}
        y={pad.top}
        width={pw}
        height={ph}
        fill="none"
        stroke={CHART.gridStroke}
        strokeOpacity={0.4}
      />
      {/* the temporal path — consecutive frames joined faintly */}
      <polyline
        points={projection.map((p, i) => `${px(p[0] ?? 0).toFixed(2)},${py(p[1] ?? 0).toFixed(2)}`).join(" ")}
        fill="none"
        stroke={CHART.stone}
        strokeOpacity={0.45}
        strokeWidth={1}
        strokeDasharray="3 3"
      />
      {projection.map((p, i) => (
        <circle
          key={`pt-${i}`}
          cx={px(p[0] ?? 0)}
          cy={py(p[1] ?? 0)}
          r={3.5}
          fill={frameColor(i, projection.length)}
          stroke="currentColor"
          strokeOpacity={0.35}
          strokeWidth={0.75}
        >
          <title>{`frame ${i}: PC1 ${fmtNum(p[0] ?? 0, 3)} · PC2 ${fmtNum(p[1] ?? 0, 3)}`}</title>
        </circle>
      ))}
      <text x={pad.l + pw} y={H - 2} textAnchor="end" className="fill-muted-foreground" style={MONO_AXIS}>
        PC1 ({evr1.toFixed(1)} %)
      </text>
      <text x={pad.l} y={9} textAnchor="start" className="fill-muted-foreground" style={MONO_AXIS}>
        PC2 ({evr2.toFixed(1)} %)
      </text>
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Card A — GROMACS export parser                                      */
/* ------------------------------------------------------------------ */

const MAX_FILES = 12;

interface ExportFileDraft {
  id: number;
  name: string;
  content: string;
}

let draftCounter = 0;

function nextDraftId(): number {
  draftCounter += 1;
  return draftCounter;
}

/** Diverging correlation cell colour: rose (negative) → card → teal
 * (positive), alpha scaled by |r| — transparent over the theme's own
 * background, so it reads in light AND dark mode. */
function corrCellBg(r: number): string {
  const alpha = 0.08 + 0.72 * Math.min(Math.abs(r), 1);
  return r >= 0
    ? `rgba(20, 184, 166, ${alpha.toFixed(3)})`
    : `rgba(244, 63, 94, ${alpha.toFixed(3)})`;
}

function XvgCard() {
  const [files, setFiles] = useState<ExportFileDraft[]>([
    { id: nextDraftId(), name: "", content: "" },
  ]);
  const [smoothHbonds, setSmoothHbonds] = useState(true);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<MdXvgResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  function addFile() {
    if (files.length >= MAX_FILES) return;
    setFiles((prev) => [...prev, { id: nextDraftId(), name: "", content: "" }]);
  }

  function removeFile(id: number) {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }

  function updateFile(id: number, patch: Partial<Pick<ExportFileDraft, "name" | "content">>) {
    setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, ...patch } : f)));
  }

  function loadDemo() {
    setFiles([
      { id: nextDraftId(), name: "ligand_rmsd.xvg", content: demoLigandRmsd() },
      { id: nextDraftId(), name: "hbnum_intermolecular.ndx", content: demoHbnumNdx() },
      { id: nextDraftId(), name: "binding_free_energy.xvg", content: demoBindingEnergy() },
    ]);
    toast.success("Demo study loaded — three GROMACS exports.");
  }

  async function run() {
    if (files.length === 0) {
      setStatus("error");
      setError("Add at least one export file (or load the demo study).");
      return;
    }
    const incomplete = files.filter((f) => f.name.trim() === "" || f.content.trim() === "");
    if (incomplete.length > 0) {
      setStatus("error");
      setError(
        `Every export file needs both a name and content — ${incomplete.length} row${
          incomplete.length === 1 ? "" : "s"
        } incomplete.`
      );
      return;
    }
    setStatus("running");
    setError(null);
    setResult(null);
    try {
      const data = await postMdJson<MdXvgResponse>("/api/md/xvg", {
        files: files.map((f) => ({ name: f.name.trim(), content: f.content })),
        smoothHbonds,
      });
      setResult(data);
      setStatus("done");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "MD export parsing failed.");
    }
  }

  async function copyCsv() {
    if (!result) return;
    const ok = await copyToClipboard(result.summaryCsv);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Copy failed — the clipboard refused the CSV.");
    }
  }

  function downloadCsv() {
    if (!result) return;
    const blob = new Blob([result.summaryCsv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "md-study-summary.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  const metricEntries: Array<[string, MdXvgSeries]> = result
    ? Object.entries(result.ligand.series)
    : [];
  const nMetrics = metricEntries.length;

  return (
    <Card data-slot="card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
          <FileText className="size-5 shrink-0" aria-hidden="true" />
          GROMACS export parser
        </CardTitle>
        <CardDescription>
          Paste GROMACS / MM-PBSA exports (<code className="font-mono text-xs">.xvg</code> /{" "}
          <code className="font-mono text-xs">.ndx</code>) — parsed by the real{" "}
          <code className="font-mono text-xs">pharmaml.md.study</code> pipeline: kJ→kcal, ps→ns,
          3-point H-bond smoothing, per-metric stats, cross-metric correlations and one summary
          CSV.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          {files.length} of {MAX_FILES} export slots used — recognised names such as{" "}
          <code className="font-mono">ligand_rmsd.xvg</code>,{" "}
          <code className="font-mono">hbnum_intermolecular.ndx</code>,{" "}
          <code className="font-mono">binding_free_energy.xvg</code>, or any name containing
          rmsd / rmsf / hbnum / sasa / energy / rg.
        </p>

        <div className="thin-scrollbar max-h-96 space-y-3 overflow-y-auto pr-1">
          {files.map((f, index) => (
            <div key={f.id} className="space-y-2 rounded-md border p-3">
              <div className="flex items-center gap-2">
                <Input
                  id={`md-file-name-${f.id}`}
                  aria-label={`Export file ${index + 1} name`}
                  value={f.name}
                  onChange={(e) => updateFile(f.id, { name: e.target.value })}
                  placeholder="ligand_rmsd.xvg"
                  spellCheck={false}
                  autoComplete="off"
                  className="h-8 flex-1 font-mono text-xs"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => removeFile(f.id)}
                  disabled={files.length === 0}
                  aria-label={`Remove export file ${f.name || `row ${index + 1}`}`}
                  className="h-8 px-2"
                >
                  <X className="size-3.5" aria-hidden="true" />
                </Button>
              </div>
              <textarea
                id={`md-file-content-${f.id}`}
                aria-label={`Export file ${index + 1} content`}
                value={f.content}
                onChange={(e) => updateFile(f.id, { content: e.target.value })}
                rows={6}
                spellCheck={false}
                className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs leading-relaxed shadow-sm outline-none focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring"
                placeholder={"@ xaxis label \"Time (ps)\"\n0.0  0.05\n1000.0  0.12"}
              />
            </div>
          ))}
          {files.length === 0 && (
            <p className="rounded-md border border-dashed px-3 py-4 text-center text-xs text-muted-foreground">
              No export files — add one or load the demo study.
            </p>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={addFile} disabled={files.length >= MAX_FILES}>
            <Plus className="size-4" aria-hidden="true" />
            Add export file
          </Button>
          <Button variant="outline" size="sm" onClick={loadDemo} disabled={status === "running"}>
            <BookOpen className="size-4" aria-hidden="true" />
            Load demo study
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-3 rounded-md border bg-muted/40 p-3">
          <div className="flex items-center gap-2">
            <Switch
              id="md-smooth-hbonds"
              checked={smoothHbonds}
              onCheckedChange={setSmoothHbonds}
              aria-label="Toggle 3-point H-bond smoothing"
            />
            <Label htmlFor="md-smooth-hbonds" className="text-sm font-normal">
              3-point H-bond smoothing
            </Label>
          </div>
          <p className="text-xs text-muted-foreground">
            The uploaded pipeline&apos;s moving average over H-bond counts.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button onClick={run} disabled={status === "running"}>
            {status === "running" ? (
              <>
                <Loader2 className="size-4 animate-spin" aria-hidden="true" />
                Parsing…
              </>
            ) : (
              <>
                <FileSearch className="size-4" aria-hidden="true" />
                Parse exports
              </>
            )}
          </Button>
          {result && (
            <>
              <Button variant="outline" onClick={copyCsv}>
                <Copy className="size-4" aria-hidden="true" />
                {copied ? "Copied" : "Copy CSV"}
              </Button>
              <Button variant="outline" onClick={downloadCsv}>
                <Download className="size-4" aria-hidden="true" />
                Download CSV
              </Button>
            </>
          )}
        </div>

        {status === "error" && (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden="true" />
            <AlertTitle>Export parsing failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Ligand &ldquo;{result.ligand.name}&rdquo; · {nMetrics} metric{nMetrics === 1 ? "" : "s"}{" "}
              parsed · {result.elapsedSeconds.toFixed(2)} s on the server.
            </p>

            {nMetrics > 0 && (
              <div className="thin-scrollbar max-h-96 overflow-y-auto pr-1">
                <div className="grid gap-3 md:grid-cols-2">
                  {metricEntries.map(([key, series]) => {
                    const unit = metricUnit(key);
                    const color = metricColor(key);
                    return (
                      <div key={key} className="space-y-2 rounded-md border p-3">
                        <div className="flex items-baseline justify-between gap-2">
                          <p className="flex items-center gap-1.5 text-sm font-medium">
                            <span
                              className="inline-block h-0.5 w-4 shrink-0 rounded-full"
                              style={{ backgroundColor: color }}
                              aria-hidden="true"
                            />
                            {series.label}
                          </p>
                          <code
                            className="max-w-[45%] truncate font-mono text-[10px] text-muted-foreground"
                            title={series.source}
                          >
                            {series.source}
                          </code>
                        </div>
                        <div className="flex flex-wrap gap-1.5 text-[11px]">
                          <span className="rounded border bg-muted/40 px-1.5 py-0.5 font-mono tabular-nums">
                            {series.time.length} pts
                          </span>
                          <span className="rounded border bg-muted/40 px-1.5 py-0.5 font-mono tabular-nums">
                            mean {fmtNum(series.stats.mean, 3)} ± {fmtNum(series.stats.std, 3)}
                            {unit ? ` ${unit}` : ""}
                          </span>
                          <span className="rounded border bg-muted/40 px-1.5 py-0.5 font-mono tabular-nums">
                            CV {fmtNum(series.stats.cv, 3)}
                          </span>
                        </div>
                        <MiniLineChart
                          time={series.time}
                          values={series.values}
                          color={color}
                          xLabel={metricXLabel(key)}
                          yLabel={unit ? `${series.label} (${unit})` : series.label}
                          ariaLabel={`${series.label} versus ${metricXLabel(key)} line chart, ${series.time.length} points, mean ${fmtNum(series.stats.mean, 3)}`}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {result.correlations !== null && result.correlations.labels.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Cross-metric Pearson correlations
                </p>
                <div className="thin-scrollbar overflow-x-auto">
                  <table className="border-separate border-spacing-px font-mono text-[10px]">
                    <thead>
                      <tr>
                        <th className="bg-card" aria-label="row label" />
                        {result.correlations.labels.map((col) => (
                          <th
                            key={col}
                            className="max-w-24 truncate bg-card px-1 py-0.5 text-center font-medium text-muted-foreground"
                            title={col}
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.correlations.matrix.map((row, i) => (
                        <tr key={result.correlations?.labels[i] ?? i}>
                          <th
                            className="sticky left-0 z-[1] max-w-28 truncate bg-card px-1 py-0.5 text-left font-medium text-muted-foreground"
                            title={result.correlations?.labels[i] ?? ""}
                          >
                            {result.correlations?.labels[i] ?? ""}
                          </th>
                          {row.map((v, j) => {
                            const darkCell = Math.abs(v) >= 0.55;
                            return (
                              <td
                                key={`${i}-${j}`}
                                className="px-1.5 py-1 text-center tabular-nums"
                                style={{ background: corrCellBg(v) }}
                                title={`r = ${fmtNum(v, 4)}`}
                              >
                                {darkCell ? (
                                  <span className="text-white">{fmtNum(v, 2)}</span>
                                ) : (
                                  fmtNum(v, 2)
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="flex items-center gap-2 pt-1">
                  <span className="font-mono text-[10px] text-muted-foreground">−1</span>
                  <div
                    className="h-1.5 flex-1 rounded-full"
                    style={{
                      background:
                        "linear-gradient(90deg, rgba(244,63,94,0.85), transparent, rgba(20,184,166,0.85))",
                    }}
                    aria-hidden="true"
                  />
                  <span className="font-mono text-[10px] text-muted-foreground">+1</span>
                </div>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Computed over the metrics&apos; overlapping time axis in the same request — hover
                  a cell for the exact r.
                </p>
              </div>
            )}

            {result.warnings.length > 0 && (
              <Alert>
                <TriangleAlert className="size-4 shrink-0" aria-hidden="true" />
                <AlertTitle>Parser warnings</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc space-y-0.5 pl-4 font-mono text-xs">
                    {result.warnings.map((w, i) => (
                      <li key={`${w}-${i}`}>{w}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            <Alert className="border-l-4">
              <Info className="size-4 shrink-0" aria-hidden="true" />
              <AlertDescription>
                <span className="font-medium">{result.method}</span> — H-bond smoothing{" "}
                {smoothHbonds ? "on" : "off"} for this run · {result.elapsedSeconds.toFixed(2)} s.
              </AlertDescription>
            </Alert>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Card B — Trajectory analysis lab                                    */
/* ------------------------------------------------------------------ */

/** Zero-based indices of residues 5–7 — the demo helix's flexible loop. */
const LOOP_RESIDUES: ReadonlySet<number> = new Set([4, 5, 6]);

const DEFAULT_FRAMES = "80";
const DEFAULT_SEED = "0";
const DEFAULT_NOISE = 0.25;
const DEFAULT_CUTOFF = "4.5";

function TrajectoryCard() {
  const [framesRaw, setFramesRaw] = useState(DEFAULT_FRAMES);
  const [seedRaw, setSeedRaw] = useState(DEFAULT_SEED);
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [cutoffRaw, setCutoffRaw] = useState(DEFAULT_CUTOFF);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<MdTrajectoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [innerTab, setInnerTab] = useState("timelines");

  function resetParams() {
    setFramesRaw(DEFAULT_FRAMES);
    setSeedRaw(DEFAULT_SEED);
    setNoise(DEFAULT_NOISE);
    setCutoffRaw(DEFAULT_CUTOFF);
  }

  async function run() {
    const frames = Number.parseInt(framesRaw, 10);
    if (!Number.isFinite(frames) || frames < 4 || frames > 200) {
      setStatus("error");
      setError("Frames must be an integer between 4 and 200.");
      return;
    }
    const seed = Number.parseInt(seedRaw, 10);
    if (!Number.isFinite(seed) || seed < 0 || seed > 2 ** 31 - 1) {
      setStatus("error");
      setError("Seed must be an integer between 0 and 2147483647.");
      return;
    }
    const cutoff = Number.parseFloat(cutoffRaw);
    if (!Number.isFinite(cutoff) || cutoff < 1 || cutoff > 12) {
      setStatus("error");
      setError("Contact cutoff must be between 1 and 12 Å.");
      return;
    }
    setStatus("running");
    setError(null);
    setResult(null);
    try {
      const data = await postMdJson<MdTrajectoryResponse>("/api/md/trajectory", {
        nFrames: frames,
        seed,
        noise,
        contactCutoff: cutoff,
      });
      setResult(data);
      setStatus("done");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "MD trajectory analysis failed.");
    }
  }

  return (
    <Card data-slot="card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
          <Orbit className="size-5 shrink-0" aria-hidden="true" />
          Trajectory analysis lab
        </CardTitle>
        <CardDescription>
          A deterministic 10-residue alpha-helix (30 atoms, residues 5–7 a flexible loop) analysed
          by the real toolkit on the server — RMSD after Kabsch superposition, Rg, native-contact
          Q, RMSF, the contact-frequency map, PCA and ANM. Same seed, same numbers, every run.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-x-4 gap-y-3 rounded-md border bg-muted/40 p-3">
          <div className="space-y-1">
            <Label htmlFor="md-traj-frames" className="text-xs text-muted-foreground">
              Frames <span className="text-[10px]">(4–200)</span>
            </Label>
            <Input
              id="md-traj-frames"
              type="number"
              min={4}
              max={200}
              step={1}
              value={framesRaw}
              onChange={(e) => setFramesRaw(e.target.value)}
              className="h-8 w-24 font-mono text-xs tabular-nums"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="md-traj-seed" className="text-xs text-muted-foreground">
              Seed
            </Label>
            <Input
              id="md-traj-seed"
              type="number"
              min={0}
              max={2147483647}
              step={1}
              value={seedRaw}
              onChange={(e) => setSeedRaw(e.target.value)}
              className="h-8 w-24 font-mono text-xs tabular-nums"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="md-traj-cutoff" className="text-xs text-muted-foreground">
              Cutoff <span className="text-[10px]">(Å)</span>
            </Label>
            <Input
              id="md-traj-cutoff"
              type="number"
              min={1}
              max={12}
              step={0.1}
              value={cutoffRaw}
              onChange={(e) => setCutoffRaw(e.target.value)}
              className="h-8 w-24 font-mono text-xs tabular-nums"
            />
          </div>
          <div className="min-w-[170px] flex-1 basis-40">
            <ParamSlider
              id="md-traj-noise"
              label="Noise"
              unit=""
              min={0}
              max={1}
              step={0.05}
              value={noise}
              onChange={setNoise}
              format={(v) => v.toFixed(2)}
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={resetParams}
            aria-label="Reset trajectory parameters to defaults"
            className="ml-auto"
          >
            <RotateCcw className="size-3.5" aria-hidden="true" />
            Reset
          </Button>
        </div>

        <Button onClick={run} disabled={status === "running"}>
          {status === "running" ? (
            <>
              <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              Analysing…
            </>
          ) : (
            <>
              <Play className="size-4" aria-hidden="true" />
              Analyse trajectory
            </>
          )}
        </Button>

        {status === "error" && (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden="true" />
            <AlertTitle>Trajectory analysis failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              {result.nFrames} frames · {result.nAtoms} atoms · {result.nResidues} residues · seed{" "}
              {result.seed} · noise {result.noise.toFixed(2)} · cutoff {result.contactCutoff} Å ·{" "}
              {result.elapsedSeconds.toFixed(2)} s on the server.
            </p>

            <Tabs value={innerTab} onValueChange={setInnerTab} className="w-full">
              <TabsList className="thin-scrollbar mx-auto grid h-auto w-full max-w-lg grid-cols-4 gap-1 overflow-x-auto p-1">
                <TabsTrigger
                  value="timelines"
                  className="min-h-10 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
                >
                  <Activity className="size-4 shrink-0" aria-hidden="true" />
                  Timelines
                </TabsTrigger>
                <TabsTrigger
                  value="rmsf"
                  className="min-h-10 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
                >
                  <BarChart3 className="size-4 shrink-0" aria-hidden="true" />
                  RMSF
                </TabsTrigger>
                <TabsTrigger
                  value="contacts"
                  className="min-h-10 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
                >
                  <Grid3x3 className="size-4 shrink-0" aria-hidden="true" />
                  <span className="sm:hidden">Contacts</span>
                  <span className="hidden sm:inline">Contact map</span>
                </TabsTrigger>
                <TabsTrigger
                  value="dynamics"
                  className="min-h-10 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
                >
                  <ScatterChart className="size-4 shrink-0" aria-hidden="true" />
                  <span className="sm:hidden">Dynamics</span>
                  <span className="hidden sm:inline">Essential dynamics</span>
                </TabsTrigger>
              </TabsList>

              <TabsContent value="timelines" className="mt-3 space-y-3">
                <TimelineTab result={result} />
              </TabsContent>
              <TabsContent value="rmsf" className="mt-3 space-y-3">
                <RmsfTab result={result} />
              </TabsContent>
              <TabsContent value="contacts" className="mt-3 space-y-3">
                <ContactMapTab result={result} />
              </TabsContent>
              <TabsContent value="dynamics" className="mt-3 space-y-3">
                <EssentialDynamicsTab result={result} />
              </TabsContent>
            </Tabs>

            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-base font-semibold tabular-nums">
                  {result.stats.rmsdMean.toFixed(3)} Å
                </p>
                <p className="text-[11px] leading-tight text-muted-foreground">RMSD mean</p>
              </div>
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-base font-semibold tabular-nums">
                  {result.stats.rgMean.toFixed(3)} Å
                </p>
                <p className="text-[11px] leading-tight text-muted-foreground">Rg mean</p>
              </div>
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-base font-semibold tabular-nums">
                  {result.stats.qMean.toFixed(3)}
                </p>
                <p className="text-[11px] leading-tight text-muted-foreground">Q mean</p>
              </div>
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-base font-semibold tabular-nums">
                  {result.stats.contactPairs}
                </p>
                <p className="text-[11px] leading-tight text-muted-foreground">Contact pairs</p>
              </div>
            </div>

            <Alert className="border-l-4">
              <Info className="size-4 shrink-0" aria-hidden="true" />
              <AlertDescription>
                <span className="font-medium">{result.method}</span> — {result.elapsedSeconds.toFixed(2)} s.
              </AlertDescription>
            </Alert>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ---- Card B inner tabs -------------------------------------------- */

function TimelineTab({ result }: { result: MdTrajectoryResponse }) {
  const series = [
    { label: "RMSD", unit: " Å", color: CHART.rose, values: result.rmsd },
    { label: "Rg", unit: " Å", color: CHART.amber, values: result.rg },
    { label: "Q", unit: "", color: CHART.teal, values: result.q },
  ];
  return (
    <div className="space-y-3">
      <TimelineChart
        time={result.time}
        series={series}
        ariaLabel={`Timelines over ${result.nFrames} frames: RMSD in angstrom, radius of gyration in angstrom and native-contact fraction Q, each series on its own scale`}
      />
      <div className="flex flex-wrap gap-2">
        {series.map((s) => {
          const lastV = s.values[s.values.length - 1] ?? 0;
          return (
            <span
              key={`chip-${s.label}`}
              className="flex items-center gap-1.5 rounded-md border bg-muted/40 px-2 py-1 font-mono text-[11px] tabular-nums"
            >
              <span
                className="inline-block size-2 rounded-full"
                style={{ backgroundColor: s.color }}
                aria-hidden="true"
              />
              {s.label} final {fmtNum(lastV, 3)}
              {s.unit}
            </span>
          );
        })}
      </div>
      <ul className="flex list-none flex-wrap gap-x-4 gap-y-1 text-[11px]">
        {series.map((s) => (
          <li key={`legend-${s.label}`} className="flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded-full"
              style={{ backgroundColor: s.color }}
              aria-hidden="true"
            />
            <span className="text-muted-foreground">
              {s.label}
              {s.unit}{" "}
              <span className="font-mono tabular-nums">
                {fmtNum(Math.min(...s.values), 3)}–{fmtNum(Math.max(...s.values), 3)}
              </span>{" "}
              (own scale)
            </span>
          </li>
        ))}
      </ul>
      <p className="text-xs leading-snug text-muted-foreground">
        Three unit-different series (RMSD Å, Rg Å, native-contact Q) on one frame axis — each line
        is scaled to its own min–max, so shapes compare, not magnitudes.
      </p>
    </div>
  );
}

function RmsfTab({ result }: { result: MdTrajectoryResponse }) {
  return (
    <div className="space-y-3">
      <MiniBarChart
        values={result.rmsfResidues}
        color={CHART.emerald}
        highlightIndices={LOOP_RESIDUES}
        highlightColor={CHART.amber}
        labelEvery={1}
        barTitle={(i) => `residue ${i + 1}: RMSF ${fmtNum(result.rmsfResidues[i] ?? 0, 3)} Å`}
        xLabel="residue"
        yLabel="RMSF (Å)"
        ariaLabel={`Per-residue RMSF bar chart over ${result.nResidues} residues; residues 5 to 7 — the flexible loop — highlighted in amber`}
      />
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span
          className="inline-block size-2 shrink-0 rounded-sm"
          style={{ backgroundColor: CHART.amber }}
          aria-hidden="true"
        />
        Residues 5–7 are the model&apos;s flexible loop — the flexible loop carries the motion.
      </p>
      <p className="text-xs leading-snug text-muted-foreground">
        Per-residue RMSF after least-squares superposition on the reference frame; hover a bar for
        the exact value. The helix body stays put, the loop does the moving.
      </p>
    </div>
  );
}

function ContactMapTab({ result }: { result: MdTrajectoryResponse }) {
  const n = result.contactMap.length;
  return (
    <div className="space-y-3">
      <div className="mx-auto w-full max-w-[440px]">
        <div
          role="img"
          aria-label={`Contact frequency map: ${n} by ${n} atoms; brighter teal means the atom pair is within ${result.contactCutoff} angstrom in more of the ${result.nFrames} frames`}
        >
          <div
            className="grid gap-px rounded-md bg-muted/60 p-1"
            style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}
          >
            {result.contactMap.flatMap((row, i) =>
              row.map((freq, j) => (
                <div
                  key={`${i}-${j}`}
                  className="aspect-square rounded-[1px]"
                  title={`${i},${j}: ${freq.toFixed(2)}`}
                  style={{ backgroundColor: CHART.teal, opacity: Math.max(freq, 0.03) }}
                />
              )),
            )}
          </div>
        </div>
      </div>
      <p className="text-center text-xs leading-snug text-muted-foreground">
        Cell intensity = the fraction of the {result.nFrames} frames in which atom pair (i, j) sits
        within {result.contactCutoff} Å — hover any cell for its exact frequency.
      </p>
      <p className="text-center">
        <span className="rounded-md border bg-muted/40 px-2 py-1 font-mono text-[11px] tabular-nums">
          {result.stats.contactPairs} persistent contact pairs (frequency &gt; 0.05)
        </span>
      </p>
    </div>
  );
}

function EssentialDynamicsTab({ result }: { result: MdTrajectoryResponse }) {
  const evr1 = (result.pca.explainedVarianceRatio[0] ?? 0) * 100;
  const evr2 = (result.pca.explainedVarianceRatio[1] ?? 0) * 100;
  const anmMax = Math.max(...result.anmFluctuations, 0);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-md border bg-muted/40 px-2 py-1 font-mono text-[11px] tabular-nums">
          PC1 {evr1.toFixed(1)} % of variance
        </span>
        <span className="rounded-md border bg-muted/40 px-2 py-1 font-mono text-[11px] tabular-nums">
          PC2 {evr2.toFixed(1)} % of variance
        </span>
      </div>
      <PcaScatter
        projection={result.pca.projection}
        evr={result.pca.explainedVarianceRatio}
        ariaLabel={`PCA scatter of the ${result.nFrames} frames on the first two principal components, coloured from light to dark by frame index, consecutive frames joined by a dashed path`}
      />
      <div className="flex items-center gap-2">
        <span className="font-mono text-[10px] text-muted-foreground">frame 0</span>
        <div
          className="h-1.5 flex-1 rounded-full"
          style={{ background: "linear-gradient(90deg, rgb(153,246,228), rgb(15,118,110))" }}
          aria-hidden="true"
        />
        <span className="font-mono text-[10px] text-muted-foreground">
          frame {result.nFrames - 1}
        </span>
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        The {result.nFrames} frames projected on the first two principal components of their
        coordinates; colour runs light→dark with frame order and the dashed path joins consecutive
        frames — the helix&apos;s slow rotational drift reads as the arc.
      </p>
      <div className="space-y-2 border-t pt-3">
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          ANM fluctuations (first frame)
        </p>
        <MiniBarChart
          values={result.anmFluctuations}
          color={CHART.teal}
          labelEvery={5}
          barTitle={(i) => `atom ${i + 1}: ${fmtNum(result.anmFluctuations[i] ?? 0, 3)}`}
          xLabel="atom"
          yLabel="fluctuation"
          ariaLabel={`ANM fluctuations of the ${result.nAtoms} atoms from the first frame, tallest bar ${fmtNum(anmMax, 3)}`}
        />
        <p className="text-xs leading-snug text-muted-foreground">
          Anisotropic network model of the first frame — the coarse-grained spring model whose
          per-atom fluctuations mirror the explicit-simulation RMSF above.
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card C — The MD ecosystem                                           */
/* ------------------------------------------------------------------ */

const ECOSYSTEM: ReadonlyArray<{ name: string; url: string; role: string }> = [
  { name: "MDAnalysis", url: "https://github.com/MDAnalysis/mdanalysis", role: "trajectory core & selections" },
  { name: "mdtraj", url: "https://github.com/mdtraj/mdtraj", role: "trajectory I/O & metrics" },
  { name: "GROMACS", url: "https://github.com/gromacs/gromacs", role: "the simulation engine & .xvg exports" },
  { name: "cpptraj", url: "https://github.com/Amber-MD/cpptraj", role: "Amber's analysis suite" },
  { name: "pytraj", url: "https://github.com/Amber-MD/pytraj", role: "cpptraj's Python face" },
  { name: "PLUMED", url: "https://github.com/plumed/plumed2", role: "enhanced sampling & collective variables" },
  { name: "ProLIF", url: "https://github.com/chemosim-lab/ProLIF", role: "protein–ligand interaction fingerprints" },
  { name: "getcontacts", url: "https://github.com/getcontacts/getcontacts", role: "contact & H-bond timelines" },
  { name: "gmx_MMPBSA", url: "https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA", role: "MM-PBSA binding energies" },
  { name: "ProDy", url: "https://github.com/prody/ProDy", role: "elastic network models" },
  { name: "deeptime", url: "https://github.com/deeptime-ml/deeptime", role: "PCA/tICA & MSMs" },
  { name: "nglview", url: "https://github.com/nglviewer/nglview", role: "interactive 3D viewing" },
  { name: "PyMOL (open source)", url: "https://github.com/schrodinger/pymol-open-source", role: "structure visualisation" },
  { name: "VMD (python)", url: "https://github.com/Eigenstate/vmd-python", role: "visual molecular dynamics" },
  { name: "MDAnalysisData", url: "https://github.com/MDAnalysis/MDAnalysisData", role: "reference test datasets" },
];

function EcosystemCard() {
  return (
    <Card data-slot="card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
          <Network className="size-5 shrink-0" aria-hidden="true" />
          The MD ecosystem
        </CardTitle>
        <CardDescription>
          The reference stack behind the MD cards — seeded by the user&apos;s uploaded{" "}
          <code className="font-mono text-xs">MD_analysis.py</code> pipeline, now shipped as{" "}
          <code className="font-mono text-xs">pharmaml.md</code> in the toolkit. Each tile opens
          the project&apos;s GitHub in a new tab.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
          {ECOSYSTEM.map((entry) => (
            <a
              key={entry.name}
              href={entry.url}
              target="_blank"
              rel="noreferrer"
              className="group flex items-start gap-2 rounded-lg border bg-card p-2.5 shadow-sm transition-colors hover:border-primary/40 hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <ExternalLink
                className="mt-0.5 size-3.5 shrink-0 text-muted-foreground transition-colors group-hover:text-primary"
                aria-hidden="true"
              />
              <span className="min-w-0">
                <span className="block text-xs font-medium leading-tight">{entry.name}</span>
                <span className="mt-0.5 block text-[11px] leading-snug text-muted-foreground">
                  {entry.role}
                </span>
              </span>
            </a>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Shell — the Molecular Dynamics sub-tab                              */
/* ------------------------------------------------------------------ */

export function MdAnalysis() {
  return (
    <div className="space-y-4 sm:space-y-6">
      <XvgCard />
      <TrajectoryCard />
      <EcosystemCard />
    </div>
  );
}
