"use client";

import { useEffect, useMemo, useState } from "react";
import { Eraser, FlaskConical, Flame, Gauge, TriangleAlert } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import { fmtNum, fmtTick } from "@/lib/format";
import { toast } from "sonner";

/* ------------------------------------------------------------------ *
 * Probability Calibration mini-widget — a client-side mirror of the
 * Python API pharmaml.viz.plot_calibration_curve (binary one-vs-rest
 * case): sklearn's uniform-strategy reliability diagram + Brier score.
 * ------------------------------------------------------------------ */

/** Selectable equal-width bin counts (the classic diagram uses 10). */
type CalibrationBins = 5 | 10 | 15;

/** One pasted row: predicted probability + binary label. */
interface CalibrationRow {
  prob: number;
  label: 0 | 1;
}

/** First malformed line of the paste (1-based, counted on raw lines). */
interface CalibrationError {
  line: number;
  message: string;
}

/** One NON-EMPTY bin of the reliability diagram (empty bins are skipped,
 * exactly like sklearn.calibration.calibration_curve). */
interface CalibrationBin {
  bin: number;
  lo: number;
  hi: number;
  /** Rows in this bin (0 = empty/skipped). */
  count: number;
  /** Mean predicted probability (sklearn's prob_pred). */
  meanProb: number;
  /** Observed fraction of positives (sklearn's prob_true). */
  fractionPositive: number;
}

/** sessionStorage key for the calibration input (survives reloads). */
const CALIBRATION_STORAGE_KEY = "pharmaml.calibration";

interface CalibrationPersisted {
  content: string;
  nBins: CalibrationBins;
}

/** Perfect-calibration reference line (the dashed diagonal). */
const DIAGONAL: Array<{ x: number; y: number }> = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
];

/** Fixed [0, 1] axis ticks for both axes. */
const AXIS_TICKS: number[] = [0, 0.25, 0.5, 0.75, 1];

/** Golden-ratio fractional step for the deterministic sample generators. */
const PHI = 0.6180339887498949;

/** Read the persisted calibration input (client only, failure-tolerant). */
function loadCalibrationInput(): CalibrationPersisted | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(CALIBRATION_STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    const p = parsed as Partial<CalibrationPersisted>;
    const nBins: CalibrationBins =
      p.nBins === 5 || p.nBins === 10 || p.nBins === 15 ? p.nBins : 10;
    const content = typeof p.content === "string" ? p.content.slice(0, 200_000) : "";
    return { content, nBins };
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

/** Parse `probability,label` lines (comma, tab or space separated).
 *  Blank lines are skipped; the first malformed line is reported with its
 *  1-based line number so the user can jump straight to it. */
function parseCalibrationRows(text: string): {
  rows: CalibrationRow[];
  error: CalibrationError | null;
} {
  const rows: CalibrationRow[] = [];
  let error: CalibrationError | null = null;
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length === 0) continue;
    const tokens = line.split(/[\s,]+/).filter(Boolean);
    if (tokens.length !== 2) {
      error ??= {
        line: i + 1,
        message: `expected two values (probability,label), got ${tokens.length}`,
      };
      continue;
    }
    const prob = Number(tokens[0]);
    if (!Number.isFinite(prob) || prob < 0 || prob > 1) {
      error ??= {
        line: i + 1,
        message: `probability must be a number in [0, 1] (got "${tokens[0]}")`,
      };
      continue;
    }
    if (tokens[1] !== "0" && tokens[1] !== "1") {
      error ??= {
        line: i + 1,
        message: `label must be 0 or 1 — binary only (got "${tokens[1]}")`,
      };
      continue;
    }
    rows.push({ prob, label: tokens[1] === "1" ? 1 : 0 });
  }
  return { rows, error };
}

/** Uniform-strategy reliability bins + Brier score — a line-by-line mirror
 *  of sklearn.calibration.calibration_curve(strategy="uniform") and
 *  sklearn.metrics.brier_score_loss as called by plot_calibration_curve.
 *
 *  Bin assignment mirrors sklearn's `np.searchsorted(bin_edges[1:-1], prob)`
 *  with its default side="left": the index counts interior edges STRICTLY
 *  below the probability, so a value exactly on an edge lands in the lower
 *  bin (bin i covers (i/n, (i+1)/n]; bin 0 covers [0, 1/n]). Empty bins are
 *  dropped from the diagram, exactly like sklearn. */
function computeCalibration(rows: CalibrationRow[], nBins: number) {
  const sums = Array.from({ length: nBins }, () => ({ n: 0, p: 0, y: 0 }));
  let brierSum = 0;
  for (const { prob, label } of rows) {
    let idx = 0;
    while (idx < nBins - 1 && (idx + 1) / nBins < prob) idx++;
    sums[idx].n += 1;
    sums[idx].p += prob;
    sums[idx].y += label;
    brierSum += (prob - label) ** 2;
  }
  const allBins: CalibrationBin[] = sums.map((s, bin) => ({
    bin,
    lo: bin / nBins,
    hi: (bin + 1) / nBins,
    count: s.n,
    meanProb: s.n > 0 ? s.p / s.n : 0,
    fractionPositive: s.n > 0 ? s.y / s.n : 0,
  }));
  const points = allBins.filter((b) => b.count > 0);
  return { allBins, points, brier: rows.length > 0 ? brierSum / rows.length : 0 };
}

/** Human label for a bin's range (matches the side="left" edge semantics). */
function binRangeLabel(bin: number, lo: number, hi: number): string {
  return bin === 0
    ? `[${fmtNum(lo, 2)}, ${fmtNum(hi, 2)}]`
    : `(${fmtNum(lo, 2)}, ${fmtNum(hi, 2)}]`;
}

/** Teal dot renderer for the reliability points (matches MoonDot's idiom). */
function ReliabilityDot(color: string) {
  return function renderReliabilityDot(props: unknown) {
    const { cx, cy } = props as { cx?: number; cy?: number };
    if (cx === undefined || cy === undefined) return <g />;
    return (
      <circle cx={cx} cy={cy} r={3.2} fill={color} fillOpacity={0.95} stroke={color} strokeWidth={0.6} />
    );
  };
}

/** Renders nothing — the diagonal reference series shows its dashed line
 *  only, never dots (and therefore never captures tooltip hovers). */
function renderHiddenDot() {
  return <g />;
}

/** Custom tooltip for the reliability points: bin range, row count, mean
 *  predicted probability and observed fraction of positives. Hovering is
 *  dot-based (ScatterChart item tooltips), so the payload entries all point
 *  at the hovered bin. */
function CalibrationTooltipContent(props: { active?: boolean; payload?: unknown }) {
  if (!props.active) return null;
  const entries = Array.isArray(props.payload)
    ? (props.payload as Array<{ payload?: unknown }>)
    : [];
  const datum = entries
    .map((e) => e.payload as CalibrationBin | undefined)
    .find((p): p is CalibrationBin => typeof p?.bin === "number");
  if (!datum) return null;
  return (
    <div className="rounded-lg border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-sm">
      <p className="font-mono text-[11px] font-semibold">
        bin {binRangeLabel(datum.bin, datum.lo, datum.hi)} · n {datum.count}
      </p>
      <p className="mt-1 font-mono text-[11px] leading-relaxed tabular-nums">
        mean p {fmtNum(datum.meanProb, 3)} → positives {fmtNum(datum.fractionPositive, 3)}
      </p>
    </div>
  );
}

/** Mild miscalibration sample: 40 rows (4 per decile) with probabilities
 *  jittered around the decile centres; the per-decile positive count is
 *  round(4·σ(3.2·(p̄ − 0.5))) — a sigmoid compression that bows the curve
 *  ~±0.2 away from the diagonal (predicted 0.85 delivers ~0.75 positives).
 *  Golden-ratio ranks pick *which* rows of each decile are positive, so the
 *  paste never looks artificially sorted. Fully deterministic. */
function buildCalibrationSample(): string {
  const lines: string[] = [];
  for (let j = 0; j < 10; j++) {
    const center = (j + 0.5) / 10;
    const trueRate = 1 / (1 + Math.exp(-3.2 * (center - 0.5)));
    const positives = Math.round(4 * trueRate);
    const ranks = [0, 1, 2, 3]
      .map((i) => ({ i, u: ((j * 4 + i + 0.5) * PHI) % 1 }))
      .sort((a, b) => a.u - b.u);
    const positiveSet = new Set(ranks.slice(0, positives).map((r) => r.i));
    for (let i = 0; i < 4; i++) {
      const jitter = 0.012 * (((i * 3 + j) % 5) - 2);
      const p = Math.min(0.99, Math.max(0.01, center + jitter));
      lines.push(`${p.toFixed(2)},${positiveSet.has(i) ? 1 : 0}`);
    }
  }
  return lines.join("\n");
}

/** Overconfident sample: 40 rows split between ~0.02–0.07 and ~0.93–0.98
 *  while the labels stay scattered ~50/50 (golden-ratio threshold), so both
 *  populated ends of the diagram sit at fraction ≈ 0.5 — far off the
 *  diagonal, and the Brier score ends up worse than uninformed guessing.
 *  Fully deterministic. */
function buildOverconfidentSample(): string {
  const lines: string[] = [];
  for (let i = 0; i < 40; i++) {
    const jitter = 0.05 * (((i * 7) % 5) / 4); // 0 … 0.05, deterministic
    const prob = i % 2 === 1 ? 0.98 - jitter : 0.02 + jitter;
    const u = ((i + 0.5) * PHI) % 1;
    lines.push(`${prob.toFixed(2)},${u < 0.5 ? 1 : 0}`);
  }
  return lines.join("\n");
}

/** Built once at module load — never recomputed during render. */
const CALIBRATION_SAMPLE = buildCalibrationSample();
const OVERCONFIDENT_SAMPLE = buildOverconfidentSample();

export function ProbabilityCalibration() {
  const [restored] = useState(loadCalibrationInput);
  const [input, setInput] = useState(restored?.content ?? "");
  const [nBins, setNBins] = useState<CalibrationBins>(restored?.nBins ?? 10);

  /* Persist the paste + bin count like the sibling widgets (optional,
   * quota-tolerant) so a reload or tab switch doesn't lose the input.
   * Hydration-safe: this card only mounts inside the ML Training tab,
   * never during the SSR/hydration render. */
  useEffect(() => {
    try {
      window.sessionStorage.setItem(
        CALIBRATION_STORAGE_KEY,
        JSON.stringify({ content: input, nBins })
      );
    } catch {
      /* quota exceeded or private mode — persistence is optional */
    }
  }, [input, nBins]);

  /* Pure parse + binning — recomputed on every keystroke, no debounce. */
  const parse = useMemo(() => parseCalibrationRows(input), [input]);
  const stats = useMemo(
    () => computeCalibration(parse.rows, nBins),
    [parse.rows, nBins]
  );

  const hasData = parse.rows.length > 0 && parse.error === null;
  const maxCount = Math.max(...stats.allBins.map((b) => b.count), 1);
  /* Chart series (stable shape for recharts: x/y + the bin payload). */
  const series = useMemo(
    () =>
      stats.points.map((p) => ({
        x: p.meanProb,
        y: p.fractionPositive,
        ...p,
      })),
    [stats.points]
  );

  function loadExample(kind: "mild" | "overconfident") {
    setInput(kind === "mild" ? CALIBRATION_SAMPLE : OVERCONFIDENT_SAMPLE);
    toast.success(
      kind === "mild" ? "Calibration example loaded" : "Overconfident example loaded",
      {
        description:
          kind === "mild"
            ? "40 rows · sigmoid compression — Brier ≈ 0.22"
            : "40 rows · extremes wrong half the time — Brier ≈ 0.46",
      }
    );
  }

  return (
    <Card className="overflow-hidden py-0 shadow-sm">
      <CardHeader className="border-b bg-muted/30 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Gauge className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">Probability Calibration</CardTitle>
            <CardDescription className="text-xs">
              Reliability diagram (binary, one-vs-rest) — mirroring{" "}
              <code className="font-mono">pharmaml.viz.plot_calibration_curve</code>
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pb-4">
        <div className="grid gap-3 sm:grid-cols-[minmax(220px,0.85fr)_minmax(0,1.15fr)]">
          {/* Input column: paste + samples + bin count */}
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Label htmlFor="calibration-input" className="text-xs font-medium">
                Probability,label pairs
              </Label>
              <div className="flex flex-wrap gap-1.5">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 px-2.5 text-[11px]"
                  onClick={() => loadExample("mild")}
                >
                  <FlaskConical className="size-3" aria-hidden />
                  Load example
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 px-2.5 text-[11px]"
                  onClick={() => loadExample("overconfident")}
                >
                  <Flame className="size-3" aria-hidden />
                  Load overconfident example
                </Button>
                {input.length > 0 ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-8 gap-1.5 px-2.5 text-[11px] text-muted-foreground"
                    onClick={() => setInput("")}
                  >
                    <Eraser className="size-3" aria-hidden />
                    Clear
                  </Button>
                ) : null}
              </div>
            </div>
            <Textarea
              id="calibration-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={"0.82,1\n0.67,0\n0.13 0\n…"}
              spellCheck={false}
              className="thin-scrollbar min-h-[140px] max-h-56 resize-y font-mono text-xs leading-relaxed"
              aria-label="Paste probability,label pairs"
              aria-invalid={parse.error ? true : undefined}
            />
            <p className="text-[10px] leading-snug text-muted-foreground">
              One pair per line · probability in [0, 1] · label 0 or 1 · comma, tab or
              space separated
            </p>
            {parse.error ? (
              <p
                role="alert"
                className="flex items-start gap-1.5 text-[11px] leading-snug text-destructive"
              >
                <TriangleAlert className="mt-0.5 size-3 shrink-0" aria-hidden />
                Line {parse.error.line}: {parse.error.message}.
              </p>
            ) : null}
            <div className="space-y-1">
              <Label className="text-[11px] font-medium">Bins (equal width)</Label>
              <Select
                value={String(nBins)}
                onValueChange={(v) => setNBins(Number.parseInt(v, 10) as CalibrationBins)}
              >
                <SelectTrigger className="h-9 text-xs" aria-label="Probability bins">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="5">5 bins</SelectItem>
                  <SelectItem value="10">10 bins · classic</SelectItem>
                  <SelectItem value="15">15 bins</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Chart column: reliability diagram or the empty-state hint */}
          <div className="flex min-w-0 flex-col gap-1.5">
            {hasData ? (
              <>
                <div
                  className="h-[260px] w-full"
                  role="img"
                  aria-label={`Reliability diagram: ${nBins} bins, ${stats.points.length} non-empty, Brier ${fmtNum(stats.brier, 2)}`}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
                      <CartesianGrid
                        stroke={CHART.gridStroke}
                        strokeOpacity={CHART.gridStrokeOpacity}
                      />
                      <XAxis
                        type="number"
                        dataKey="x"
                        domain={[0, 1]}
                        ticks={AXIS_TICKS}
                        tick={CHART.axisTick}
                        tickFormatter={fmtTick}
                        tickLine={false}
                        axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                        label={{
                          value: "Mean predicted probability",
                          position: "insideBottomRight",
                          offset: -2,
                          fill: "#78716c",
                          fontSize: 11,
                        }}
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        width={44}
                        domain={[0, 1]}
                        ticks={AXIS_TICKS}
                        tick={CHART.axisTick}
                        tickFormatter={fmtTick}
                        tickLine={false}
                        axisLine={false}
                        label={{
                          value: "Fraction of positives",
                          angle: -90,
                          position: "insideBottomLeft",
                          offset: 18,
                          fill: "#78716c",
                          fontSize: 11,
                        }}
                      />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        content={<CalibrationTooltipContent />}
                      />
                      <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                      <Scatter
                        name={`class 1 — Brier ${fmtNum(stats.brier, 3)}`}
                        data={series}
                        line={{ stroke: CHART.teal, strokeWidth: 2 }}
                        shape={ReliabilityDot(CHART.teal)}
                        fill={CHART.teal}
                        isAnimationActive={false}
                      />
                      <Scatter
                        name="perfectly calibrated"
                        data={DIAGONAL}
                        line={{ stroke: "#9CA3AF", strokeWidth: 1.5, strokeDasharray: "6 4" }}
                        shape={renderHiddenDot}
                        fill="#9CA3AF"
                        isAnimationActive={false}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                {/* Bin-occupancy mini histogram (subtle, decorative) */}
                <div
                  className="flex h-7 items-end gap-[2px] pl-11 pr-3.5"
                  aria-hidden
                >
                  {stats.allBins.map((b) => (
                    <div
                      key={b.bin}
                      className="flex-1 rounded-t-[2px]"
                      style={{
                        height: `${
                          b.count === 0
                            ? 2
                            : Math.max(3, Math.round((26 * b.count) / maxCount))
                        }px`,
                        backgroundColor: CHART.teal,
                        opacity: b.count === 0 ? 0.15 : 0.4,
                      }}
                      title={`bin ${binRangeLabel(b.bin, b.lo, b.hi)}: ${b.count} row${
                        b.count === 1 ? "" : "s"
                      }`}
                    />
                  ))}
                </div>
                <p className="pl-11 pr-3.5 text-[10px] leading-snug text-muted-foreground">
                  Bin occupancy — one bar per equal-width bin, height = rows in that bin
                  (hover a bar for the count).
                </p>
              </>
            ) : (
              <div className="flex min-h-[160px] flex-1 items-center justify-center rounded-lg border border-dashed bg-muted/20 px-4 py-6 text-center">
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Paste{" "}
                  <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
                    probability,label
                  </code>{" "}
                  pairs to draw a reliability diagram — does a predicted 0.8 really mean
                  80 % positives? Parity:{" "}
                  <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
                    pharmaml.viz.plot_calibration_curve(model, X_test, y_test)
                  </code>
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Chips row */}
        {hasData ? (
          <div className="flex flex-wrap gap-2">
            <div className="min-w-[190px] rounded-lg border bg-card px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Brier score
              </p>
              <p className="font-mono text-lg font-semibold leading-tight tabular-nums">
                {fmtNum(stats.brier, 3)}
              </p>
              <p className="mt-0.5 text-[10px] leading-snug text-muted-foreground">
                lower is better · 0.25 ≈ random guessing for balanced data
              </p>
            </div>
            <div className="min-w-[86px] rounded-lg border bg-card px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Rows
              </p>
              <p className="font-mono text-lg font-semibold leading-tight tabular-nums">
                {parse.rows.length}
              </p>
            </div>
            <div className="min-w-[124px] rounded-lg border bg-card px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Non-empty bins
              </p>
              <p className="font-mono text-lg font-semibold leading-tight tabular-nums">
                {stats.points.length}
                <span className="text-xs font-normal text-muted-foreground"> / {nBins}</span>
              </p>
            </div>
          </div>
        ) : null}

        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Parity:{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
            pharmaml.viz.plot_calibration_curve(model, X_test, y_test, n_bins={nBins})
          </code>{" "}
          /{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
            pharmaml.ml.train(..., plots=True)
          </code>
        </p>
      </CardContent>
    </Card>
  );
}
