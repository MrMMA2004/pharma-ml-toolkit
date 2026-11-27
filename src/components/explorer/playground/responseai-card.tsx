"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  CircleCheck,
  Copy,
  Loader2,
  Play,
  TriangleAlert,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { CHART } from "@/components/explorer/playground/calc-ui";
import { StatChip } from "@/components/explorer/stat-chip";
import { copyToClipboard, fmtNum } from "@/lib/format";
import type {
  DrugResponseAnalysis,
  DrugResponseResult,
  HillFitResponse,
  ResponseMetricsResponse,
  ResponseModelResponse,
  ResponseCvResponse,
} from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Response AI — separate Next.js web-console demo                     */
/* ------------------------------------------------------------------ */
/**
 * This web-console card demonstrates four analyses over
 * `pharmaml.ml.drugresponse`: a four-parameter Hill fit, response
 * metrics and rank correlations, CaDRReS bilinear factorisation on a
 * seeded 12×4 screen, and cell-line-grouped cross-validation. A POST
 * runs the module on the server. The demo exposes only these operations;
 * it is separate from the React/Tauri desktop app and does not imply
 * feature parity with Studio or the Python package. The page's deeper
 * diagnostics (informative regression, the response VAE, multi-omics
 * fusion) stay desktop-side.
 */

const ANALYSIS_OPTIONS: ReadonlyArray<{
  value: DrugResponseAnalysis;
  label: string;
  blurb: string;
}> = [
  { value: "hill", label: "Hill curve fit", blurb: "the 4PL the screens actually measure" },
  { value: "metrics", label: "Response metrics", blurb: "Pearson · Spearman · RMSE · MAE + Kendall" },
  { value: "model", label: "Response model", blurb: "CaDRReS bilinear factorisation" },
  { value: "cv", label: "Cross-validation", blurb: "cell-line-grouped folds" },
];

const DEFAULT_EPOCHS = "120";
const DEFAULT_FACTORS = "2";
const DEFAULT_FOLDS = "4";
const DEFAULT_SEED = "42";
const DEFAULT_NOISE = "0.01";

type Status = "idle" | "running" | "done" | "error";

/* ------------------------------ charts ------------------------------ */

/** The fitted Hill curve: measured points plus the fitted curve on a log10-dose axis. */
function HillChart({ result }: { result: HillFitResponse }) {
  const W = 560;
  const H = 220;
  const pad = { l: 44, r: 14, top: 12, b: 28 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;

  const xs = [...result.curveDose, ...result.doses].map((d) => Math.log10(d));
  const ys = [...result.curveResponse, ...result.response];
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys, 0);
  const y1 = Math.max(...ys, 1);
  const xSpan = x1 - x0 || 1;
  const ySpan = y1 - y0 || 1;
  const px = (v: number) => pad.l + ((v - x0) / xSpan) * pw;
  const py = (v: number) => pad.top + ph - ((v - y0) / ySpan) * ph;

  const curve = result.curveDose
    .map((d, i) => `${i === 0 ? "M" : "L"}${fmtNum(px(Math.log10(d)), 1)},${fmtNum(py(result.curveResponse[i] ?? 0), 1)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label={`The fitted Hill curve over ${result.nPoints} measured points`}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={`gy-${f}`}>
          <line
            x1={pad.l}
            y1={pad.top + ph - f * ph}
            x2={pad.l + pw}
            y2={pad.top + ph - f * ph}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text
            x={pad.l - 6}
            y={pad.top + ph - f * ph + 4}
            textAnchor="end"
            className="fill-muted-foreground"
            style={CHART.axisTick}
          >
            {fmtNum(y0 + f * ySpan, 2)}
          </text>
        </g>
      ))}
      <path d={curve} fill="none" stroke={CHART.emerald} strokeWidth={2} />
      {result.doses.map((d, i) => (
        <circle
          key={`pt-${i}`}
          cx={px(Math.log10(d))}
          cy={py(result.response[i] ?? 0)}
          r={3}
          fill={CHART.amber}
        />
      ))}
      <text x={pad.l + pw / 2} y={H - 6} textAnchor="middle" className="fill-muted-foreground" style={CHART.axisTick}>
        log10 dose
      </text>
      <text x={10} y={pad.top + ph / 2} textAnchor="middle" className="fill-muted-foreground" style={CHART.axisTick} transform={`rotate(-90 10 ${pad.top + ph / 2})`}>
        response
      </text>
    </svg>
  );
}

/** Observed-vs-predicted scatter with an identity reference, shared by this demo's metrics and model results. */
function PairScatter({
  x,
  y,
  xLabel,
  yLabel,
  ariaLabel,
}: {
  x: number[];
  y: number[];
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
}) {
  const W = 560;
  const H = 220;
  const pad = { l: 44, r: 14, top: 12, b: 28 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;
  const finite = x.flatMap((xv, i) => {
    const yv = y[i] ?? Number.NaN;
    return Number.isFinite(xv) && Number.isFinite(yv) ? [[xv, yv] as const] : [];
  });
  const lo = Math.min(...finite.map(([a]) => a), ...finite.map(([, b]) => b));
  const hi = Math.max(...finite.map(([a]) => a), ...finite.map(([, b]) => b));
  const span = hi - lo || 1;
  const px = (v: number) => pad.l + ((v - lo) / span) * pw;
  const py = (v: number) => pad.top + ph - ((v - lo) / span) * ph;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label={ariaLabel}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={`gy-${f}`}>
          <line
            x1={pad.l}
            y1={pad.top + ph - f * ph}
            x2={pad.l + pw}
            y2={pad.top + ph - f * ph}
            stroke={CHART.gridStroke}
            strokeOpacity={CHART.gridStrokeOpacity}
          />
          <text x={pad.l - 6} y={pad.top + ph - f * ph + 4} textAnchor="end" className="fill-muted-foreground" style={CHART.axisTick}>
            {fmtNum(lo + f * span, 2)}
          </text>
        </g>
      ))}
      <line
        x1={px(lo)}
        y1={py(lo)}
        x2={px(hi)}
        y2={py(hi)}
        stroke={CHART.stone}
        strokeDasharray="4 3"
        strokeWidth={1.5}
      />
      {finite.map(([a, b], i) => (
        <circle key={`pt-${i}`} cx={px(a)} cy={py(b)} r={2.6} fill={CHART.emerald} fillOpacity={0.85} />
      ))}
      <text x={pad.l + pw / 2} y={H - 6} textAnchor="middle" className="fill-muted-foreground" style={CHART.axisTick}>
        {xLabel}
      </text>
      <text x={10} y={pad.top + ph / 2} textAnchor="middle" className="fill-muted-foreground" style={CHART.axisTick} transform={`rotate(-90 10 ${pad.top + ph / 2})`}>
        {yLabel}
      </text>
    </svg>
  );
}

/** The model's loss history over epochs (one point per epoch; a long
 * run decimates to ~120 points so the SVG stays light). */
function LossChart({ losses }: { losses: number[] }) {
  const W = 560;
  const H = 180;
  const pad = { l: 52, r: 14, top: 12, b: 24 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;
  const step = Math.max(1, Math.ceil(losses.length / 120));
  const shown = losses.filter((_, i) => i % step === 0);
  const y0 = Math.min(...shown);
  const y1 = Math.max(...shown);
  const ySpan = y1 - y0 || 1;
  const px = (i: number) => pad.l + (i / Math.max(1, shown.length - 1)) * pw;
  const py = (v: number) => pad.top + ph - ((v - y0) / ySpan) * ph;
  const path = shown
    .map((v, i) => `${i === 0 ? "M" : "L"}${fmtNum(px(i), 1)},${fmtNum(py(v), 1)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`The training loss over ${losses.length} epochs`}>
      {[0, 0.5, 1].map((f) => (
        <g key={`gy-${f}`}>
          <line x1={pad.l} y1={pad.top + ph - f * ph} x2={pad.l + pw} y2={pad.top + ph - f * ph} stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
          <text x={pad.l - 6} y={pad.top + ph - f * ph + 4} textAnchor="end" className="fill-muted-foreground" style={CHART.axisTick}>
            {fmtNum(y0 + f * ySpan, 4)}
          </text>
        </g>
      ))}
      <path d={path} fill="none" stroke={CHART.rose} strokeWidth={2} />
      <text x={pad.l + pw / 2} y={H - 5} textAnchor="middle" className="fill-muted-foreground" style={CHART.axisTick}>
        epoch{step > 1 ? ` (every ${step})` : ""}
      </text>
    </svg>
  );
}

/** Per-fold Pearson bars with a dashed mean line. */
function CvBars({ result }: { result: ResponseCvResponse }) {
  const W = 560;
  const H = 200;
  const pad = { l: 44, r: 14, top: 14, b: 28 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;
  const values = result.folds.map((f) => f.pearson);
  const lo = Math.min(-1, ...values);
  const hi = Math.max(1, ...values);
  const span = hi - lo || 1;
  const n = Math.max(1, values.length);
  const slot = pw / n;
  const barW = Math.min(0.5 * slot, 48);
  const py = (v: number) => pad.top + ph - ((v - lo) / span) * ph;
  const zeroY = py(0);
  const meanY = py(result.meanPearson);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Per-fold pearson over ${result.nFolds} folds, mean ${fmtNum(result.meanPearson, 3)}`}>
      {[-1, -0.5, 0, 0.5, 1].filter((v) => v >= lo && v <= hi).map((v) => (
        <g key={`gy-${v}`}>
          <line x1={pad.l} y1={py(v)} x2={pad.l + pw} y2={py(v)} stroke={CHART.gridStroke} strokeOpacity={v === 0 ? 0.45 : CHART.gridStrokeOpacity} />
          <text x={pad.l - 6} y={py(v) + 4} textAnchor="end" className="fill-muted-foreground" style={CHART.axisTick}>
            {fmtNum(v, 1)}
          </text>
        </g>
      ))}
      <line x1={pad.l} y1={meanY} x2={pad.l + pw} y2={meanY} stroke={CHART.amber} strokeDasharray="5 4" strokeWidth={1.6} />
      {result.folds.map((f, i) => {
        const cx = pad.l + slot * (i + 0.5);
        const yTop = Math.min(py(f.pearson), zeroY);
        const h = Math.abs(py(f.pearson) - zeroY) || 1;
        const positive = f.pearson >= 0;
        return (
          <g key={`fold-${f.fold}`}>
            <rect
              x={cx - barW / 2}
              y={yTop}
              width={barW}
              height={h}
              rx={2}
              fill={positive ? CHART.emerald : CHART.rose}
              fillOpacity={0.85}
            />
            <text x={cx} y={H - 8} textAnchor="middle" className="fill-muted-foreground" style={CHART.axisTick}>
              fold {f.fold}
            </text>
            <text x={cx} y={yTop - 4} textAnchor="middle" className="fill-muted-foreground" style={CHART.axisTick}>
              {fmtNum(f.pearson, 2)}
            </text>
          </g>
        );
      })}
      <text x={pad.l + pw} y={meanY - 5} textAnchor="end" style={{ ...CHART.axisTick, fill: CHART.amber }}>
        mean {fmtNum(result.meanPearson, 3)}
      </text>
    </svg>
  );
}

/* ------------------------------ the card ------------------------------ */

export function ResponseAICard() {
  const [analysis, setAnalysis] = useState<DrugResponseAnalysis>("hill");
  const [dose, setDose] = useState("");
  const [response, setResponse] = useState("");
  const [yTrue, setYTrue] = useState("");
  const [yPred, setYPred] = useState("");
  const [noise, setNoise] = useState(DEFAULT_NOISE);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [factors, setFactors] = useState(DEFAULT_FACTORS);
  const [folds, setFolds] = useState(DEFAULT_FOLDS);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<DrugResponseResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const seedParsed = Number.parseInt(seed, 10);
  const seedValid = Number.isInteger(seedParsed) && seedParsed >= 0 && seedParsed <= 999_999;
  const running = status === "running";
  const hill = result?.analysis === "hill" ? (result as HillFitResponse) : null;
  const metrics = result?.analysis === "metrics" ? (result as ResponseMetricsResponse) : null;
  const model = result?.analysis === "model" ? (result as ResponseModelResponse) : null;
  const cv = result?.analysis === "cv" ? (result as ResponseCvResponse) : null;

  async function run() {
    if (!seedValid) {
      setError("The seed must be an integer between 0 and 999999.");
      toast.error("Invalid seed");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const res = await fetch("/api/ml/drugresponse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis,
          dose: analysis === "hill" ? dose : "",
          response: analysis === "hill" ? response : "",
          yTrue: analysis === "metrics" ? yTrue : "",
          yPred: analysis === "metrics" ? yPred : "",
          noise: Number.parseFloat(noise) || 0.01,
          epochs: Number.parseInt(epochs, 10) || 120,
          factors: Number.parseInt(factors, 10) || 2,
          folds: Number.parseInt(folds, 10) || 4,
          seed: seedParsed,
        }),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? String((data as { error: unknown }).error)
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const payload = data as DrugResponseResult;
      if (payload.analysis !== analysis) {
        throw new Error("the bridge answered with an unexpected analysis");
      }
      setResult(payload);
      setStatus("done");
      toast.success("Response AI analysis complete", {
        description: payload.summary,
      });
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "The analysis failed.";
      setError(message);
      toast.error("The analysis failed", { description: message });
    }
  }

  async function copyResult() {
    if (!result) return;
    let header = "";
    const lines: string[] = [];
    if (hill) {
      header = `# Hill fit — ${hill.source} · r² ${fmtNum(hill.rSquared, 4)} · ${hill.nPoints} points`;
      lines.push("parameter,fitted" + (hill.truth ? ",true" : ""));
      const p = hill.parameters;
      const t = hill.truth;
      for (const key of ["hill_coef", "ic50", "top", "bottom"] as const) {
        lines.push(`${key},${fmtNum(p[key], 6)}${t ? `,${fmtNum(t[key], 6)}` : ""}`);
      }
    } else if (metrics) {
      header = `# Response metrics — ${metrics.source} · n=${metrics.n}`;
      lines.push("metric,value");
      lines.push(`pearson,${fmtNum(metrics.pearson, 6)}`);
      lines.push(`spearman,${fmtNum(metrics.spearman, 6)}`);
      lines.push(`kendall,${fmtNum(metrics.rank.kendall, 6)}`);
      lines.push(`rmse,${fmtNum(metrics.rmse, 6)}`);
      lines.push(`mae,${fmtNum(metrics.mae, 6)}`);
    } else if (model) {
      header = `# Response model — ${model.epochs} epochs · ${model.nObserved} observed pairs · pearson ${fmtNum(model.pearsonR, 4)}`;
      lines.push("cell," + model.drugs.join(","));
      for (const row of model.predictedHead) {
        lines.push(row.cell + "," + row.values.map((v) => fmtNum(v, 6)).join(","));
      }
    } else if (cv) {
      header = `# Response cross-validation — ${cv.nFolds} cell-line-grouped folds · mean pearson ${fmtNum(cv.meanPearson, 4)}`;
      lines.push("fold,test_cells,observed,pearson,rmse");
      for (const f of cv.folds) {
        lines.push(`${f.fold},${f.nTestCells},${f.nObserved},${fmtNum(f.pearson, 6)},${fmtNum(f.rmse, 6)}`);
      }
    }
    const ok = await copyToClipboard([header, ...lines].join("\n"));
    if (ok) {
      setCopied(true);
      toast.success("Copied the result table");
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  return (
    <Card className="overflow-hidden py-0 shadow-sm">
      <CardHeader className="border-b bg-muted/30 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Activity className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">Response AI — drug-response learning</CardTitle>
            <CardDescription className="text-xs">
              A separate Next.js web-console demo calling{" "}
              <code className="font-mono">pharmaml.ml.drugresponse</code>. Its four analyses are
              not the React/Tauri Studio and do not imply feature parity.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pb-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="responseai-analysis" className="text-xs font-medium">
              Analysis
            </Label>
            <Select value={analysis} onValueChange={(v) => setAnalysis(v as DrugResponseAnalysis)}>
              <SelectTrigger id="responseai-analysis" className="h-9">
                <SelectValue placeholder="pick an analysis" />
              </SelectTrigger>
              <SelectContent>
                {ANALYSIS_OPTIONS.map((a) => (
                  <SelectItem key={a.value} value={a.value} textValue={a.label}>
                    <div className="flex flex-col items-start py-1">
                      <span className="text-sm">{a.label}</span>
                      <span className="text-[11px] text-muted-foreground">{a.blurb}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="responseai-seed" className="text-xs font-medium">
              Seed
            </Label>
            <Input
              id="responseai-seed"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              inputMode="numeric"
              className="h-9 font-mono text-sm"
              aria-describedby="responseai-seed-hint"
            />
            <p id="responseai-seed-hint" className="text-[11px] text-muted-foreground">
              Varies the seeded demos and the model&apos;s weight seed — the same seed regrows
              the same fit and training.
            </p>
          </div>
        </div>

        {analysis === "hill" ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="responseai-dose" className="text-xs font-medium">
                Dose series <span className="text-muted-foreground">(empty = seeded demo)</span>
              </Label>
              <Textarea
                id="responseai-dose"
                value={dose}
                onChange={(e) => setDose(e.target.value)}
                placeholder="e.g. 0.1 0.3 1 3 10 30 100 — comma, space or semicolon separated"
                className="min-h-20 font-mono text-xs"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="responseai-response" className="text-xs font-medium">
                Response series <span className="text-muted-foreground">(empty = seeded demo)</span>
              </Label>
              <Textarea
                id="responseai-response"
                value={response}
                onChange={(e) => setResponse(e.target.value)}
                placeholder="one value per dose, strictly positive doses"
                className="min-h-20 font-mono text-xs"
              />
            </div>
            <div className="space-y-1.5 sm:col-span-2">
              <Label htmlFor="responseai-noise" className="text-xs font-medium">
                Demo noise σ <span className="text-muted-foreground">(seeded demo only)</span>
              </Label>
              <Input
                id="responseai-noise"
                value={noise}
                onChange={(e) => setNoise(e.target.value)}
                inputMode="decimal"
                className="h-9 w-28 font-mono text-sm"
              />
              <p className="text-[11px] text-muted-foreground">
                Both fields empty runs the page&apos;s seeded 21-point 4PL demo (truth
                hill 1.2 · IC50 0.8 · top 1.0 · bottom 0.05).
              </p>
            </div>
          </div>
        ) : null}

        {analysis === "metrics" ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="responseai-ytrue" className="text-xs font-medium">
                Observed <span className="text-muted-foreground">(empty = seeded pair)</span>
              </Label>
              <Textarea
                id="responseai-ytrue"
                value={yTrue}
                onChange={(e) => setYTrue(e.target.value)}
                placeholder="e.g. 0.1 0.4 nan 0.9 — NaN tokens drop their pair"
                className="min-h-20 font-mono text-xs"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="responseai-ypred" className="text-xs font-medium">
                Predicted <span className="text-muted-foreground">(empty = seeded pair)</span>
              </Label>
              <Textarea
                id="responseai-ypred"
                value={yPred}
                onChange={(e) => setYPred(e.target.value)}
                placeholder="one value per observed entry"
                className="min-h-20 font-mono text-xs"
              />
            </div>
          </div>
        ) : null}

        {analysis === "model" ? (
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label htmlFor="responseai-epochs" className="text-xs font-medium">
                Epochs <span className="text-muted-foreground">(20–400)</span>
              </Label>
              <Input
                id="responseai-epochs"
                value={epochs}
                onChange={(e) => setEpochs(e.target.value)}
                inputMode="numeric"
                className="h-9 font-mono text-sm"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="responseai-factors" className="text-xs font-medium">
                Latent factors <span className="text-muted-foreground">(1–4)</span>
              </Label>
              <Input
                id="responseai-factors"
                value={factors}
                onChange={(e) => setFactors(e.target.value)}
                inputMode="numeric"
                className="h-9 font-mono text-sm"
              />
            </div>
            <div className="space-y-1.5 sm:col-span-3">
              <p className="text-[11px] text-muted-foreground">
                Trains on the page&apos;s seeded 12×4 screen (15% NaN mask — the CaDRReS
                data contract in miniature); full-batch gradient descent over the observed
                pairs only, deterministic per seed (lr 0.1 · l2 0.001).
              </p>
            </div>
          </div>
        ) : null}

        {analysis === "cv" ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="responseai-folds" className="text-xs font-medium">
                Folds <span className="text-muted-foreground">(2–8)</span>
              </Label>
              <Input
                id="responseai-folds"
                value={folds}
                onChange={(e) => setFolds(e.target.value)}
                inputMode="numeric"
                className="h-9 w-28 font-mono text-sm"
              />
            </div>
            <div className="space-y-1.5">
              <p className="text-[11px] text-muted-foreground">
                Folds split CELL LINES, never (cell line, drug) pairs — the leak the
                grouped design exists to prevent. Each fold trains a fresh model with the
                page&apos;s exact factory; the mean pearson is reported honestly even
                when the toy screen barely carries signal.
              </p>
            </div>
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={run} disabled={running} size="sm" className="h-9">
            {running ? (
              <>
                <Loader2 className="size-4 animate-spin" aria-hidden />
                Running…
              </>
            ) : (
              <>
                <Play className="size-4" aria-hidden />
                Run analysis
              </>
            )}
          </Button>
          {result ? (
            <Button onClick={copyResult} size="sm" variant="outline" className="h-9">
              <Copy className="size-4" aria-hidden />
              {copied ? "Copied" : "Copy result"}
            </Button>
          ) : null}
          <span className="ml-auto text-[11px] text-muted-foreground">
            This web-console demo omits other analyses available through the Python API; see
            package documentation for the separate workflow scope.
          </span>
        </div>

        {error ? (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden />
            <AlertTitle>Analysis failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {hill ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <StatChip icon={CircleCheck} value={fmtNum(hill.rSquared, 4)} label="r²" />
              <StatChip icon={CircleCheck} value={hill.converged ? "yes" : "NO"} label="converged" />
              <StatChip icon={CircleCheck} value={String(hill.nPoints)} label="points" />
              <StatChip icon={CircleCheck} value={fmtNum(hill.fittedAuc, 3)} label="AUC (fitted)" />
            </div>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full min-w-80 text-sm">
                <thead>
                  <tr className="border-b bg-muted/40">
                    <th className="px-3 py-2 text-left font-medium">parameter</th>
                    <th className="px-3 py-2 text-right font-medium">fitted</th>
                    {hill.truth ? <th className="px-3 py-2 text-right font-medium">true</th> : null}
                  </tr>
                </thead>
                <tbody>
                  {(["hill_coef", "ic50", "top", "bottom"] as const).map((k) => (
                    <tr key={k} className="border-b last:border-0">
                      <td className="px-3 py-1.5 font-mono text-xs">{k}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">
                        {fmtNum(hill.parameters[k], 4)}
                      </td>
                      {hill.truth ? (
                        <td className="px-3 py-1.5 text-right font-mono text-xs tabular-nums text-muted-foreground">
                          {fmtNum(hill.truth[k], 4)}
                        </td>
                      ) : null}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <HillChart result={hill} />
            <p className="text-[11px] leading-snug text-muted-foreground">
              {hill.summary} · top = the low-dose plateau, bottom = the high-dose plateau
              (the module&apos;s own convention).
            </p>
          </div>
        ) : null}

        {metrics ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
              <StatChip icon={CircleCheck} value={fmtNum(metrics.pearson, 4)} label="pearson" />
              <StatChip icon={CircleCheck} value={fmtNum(metrics.spearman, 4)} label="spearman" />
              <StatChip icon={CircleCheck} value={fmtNum(metrics.rank.kendall, 4)} label="kendall τ" />
              <StatChip icon={CircleCheck} value={fmtNum(metrics.rmse, 4)} label="RMSE" />
              <StatChip icon={CircleCheck} value={`${metrics.n}${metrics.nDropped ? ` (+${metrics.nDropped} NaN)` : ""}`} label="pairs" />
            </div>
            <PairScatter
              x={metrics.yTrue}
              y={metrics.yPred}
              xLabel="observed"
              yLabel="predicted"
              ariaLabel={`observed vs predicted over ${metrics.n} surviving pairs`}
            />
            <p className="text-[11px] leading-snug text-muted-foreground">
              {metrics.summary} — NaN pairs drop before every metric (the untested pairs
              of a screen; the rank trio rides the finite pairs); MAE {fmtNum(metrics.mae, 4)}.
            </p>
          </div>
        ) : null}

        {model ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <StatChip icon={CircleCheck} value={fmtNum(model.pearsonR, 4)} label="in-sample pearson" />
              <StatChip icon={CircleCheck} value={fmtNum(model.finalLoss, 4)} label="final loss" />
              <StatChip icon={CircleCheck} value={String(model.nObserved)} label="observed pairs" />
              <StatChip icon={CircleCheck} value={String(model.epochs)} label="epochs" />
            </div>
            <LossChart losses={model.trainLosses} />
            <PairScatter
              x={model.observed}
              y={model.predicted}
              xLabel="observed"
              yLabel="predicted"
              ariaLabel={`observed vs predicted over ${model.nPairs} observed pairs of the screen`}
            />
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full min-w-64 text-sm">
                <thead>
                  <tr className="border-b bg-muted/40">
                    <th className="px-3 py-2 text-left font-medium">cell line</th>
                    {model.drugs.map((d) => (
                      <th key={d} className="px-3 py-2 text-right font-medium">
                        {d}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {model.predictedHead.map((row) => (
                    <tr key={row.cell} className="border-b last:border-0">
                      <td className="px-3 py-1.5 font-mono text-xs">{row.cell}</td>
                      {row.values.map((v, i) => (
                        <td key={i} className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">
                          {fmtNum(v, 3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">{model.summary}</p>
          </div>
        ) : null}

        {cv ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <StatChip icon={CircleCheck} value={String(cv.nFolds)} label="folds" />
              <StatChip icon={CircleCheck} value={fmtNum(cv.meanPearson, 4)} label="mean pearson" />
              <StatChip
                icon={CircleCheck}
                value={String(cv.folds.reduce((a, f) => a + f.nObserved, 0))}
                label="observed pairs"
              />
            </div>
            <CvBars result={cv} />
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full min-w-72 text-sm">
                <thead>
                  <tr className="border-b bg-muted/40">
                    <th className="px-3 py-2 text-left font-medium">fold</th>
                    <th className="px-3 py-2 text-right font-medium">test cells</th>
                    <th className="px-3 py-2 text-right font-medium">observed</th>
                    <th className="px-3 py-2 text-right font-medium">pearson</th>
                    <th className="px-3 py-2 text-right font-medium">rmse</th>
                  </tr>
                </thead>
                <tbody>
                  {cv.folds.map((f) => (
                    <tr key={f.fold} className="border-b last:border-0">
                      <td className="px-3 py-1.5 font-mono text-xs">{f.fold}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">{f.nTestCells}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">{f.nObserved}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">{fmtNum(f.pearson, 4)}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">{fmtNum(f.rmse, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">{cv.summary}</p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
