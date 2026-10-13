"use client";

import { useMemo } from "react";
import {
  Activity,
  Info,
  RotateCcw,
  TriangleAlert,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useToolkitStore, type PkParams } from "@/lib/store";
import { CHART } from "@/components/explorer/playground/calc-ui";

/* ------------------------------------------------------------------ */
/* Types & maths                                                       */
/* ------------------------------------------------------------------ */

type Route = PkParams["route"];

interface Derived {
  ke: number; // 1/h
  halfLife: number; // h
  accumulation: number; // R_ac
  cssAvg: number; // mg/L
  cMaxFirst: number; // mg/L
  cMaxSS: number; // mg/L
}

/** One compartment, first-order elimination, exact superposition. */
function concentrationAt(t: number, p: PkParams, ke: number): number {
  let c = 0;
  for (let i = 0; i < p.doses; i++) {
    const since = t - i * p.tau;
    if (since < 0) break;
    if (p.route === "iv") {
      c += (p.dose / p.vd) * Math.exp(-ke * since);
    } else {
      const denom = p.vd * (p.ka - ke);
      if (Math.abs(denom) < 1e-9) continue;
      c += ((p.f * p.dose * p.ka) / denom) * (Math.exp(-ke * since) - Math.exp(-p.ka * since));
    }
  }
  return c;
}

function derive(p: PkParams): Derived {
  const ke = p.cl / p.vd;
  const halfLife = Math.log(2) / ke;
  const accumulation = 1 / (1 - Math.exp(-ke * p.tau));
  const cssAvg = (p.route === "iv" ? 1 : p.f) * (p.dose / (p.cl * p.tau));

  const endOfFirst = p.route === "iv" ? 0 : Math.log(p.ka / ke) / (p.ka - ke);
  const cMaxFirst = concentrationAt(endOfFirst, { ...p, doses: 1 }, ke);
  const lastDose = (p.doses - 1) * p.tau;
  const cMaxSS =
    p.doses > 4
      ? concentrationAt(lastDose + endOfFirst, p, ke)
      : Number.NaN; // not yet near steady state — don't mislead
  return { ke, halfLife, accumulation, cssAvg, cMaxFirst, cMaxSS };
}

/* ------------------------------------------------------------------ */
/* Slider row                                                          */
/* ------------------------------------------------------------------ */

interface ParamSliderProps {
  id: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}

function ParamSlider({ id, label, unit, min, max, step, value, onChange, format }: ParamSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <Label htmlFor={id} className="text-xs font-medium text-muted-foreground">
          {label}
        </Label>
        <span className="rounded-md border border-primary/15 bg-primary/5 px-1.5 py-0.5 font-mono text-xs font-semibold tabular-nums">
          {format ? format(value) : value}
          <span className="ml-0.5 font-normal text-muted-foreground">{unit}</span>
        </span>
      </div>
      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(vals) => onChange(vals[0] ?? value)}
        aria-label={`${label} in ${unit}`}
        className="py-1"
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */

const SAMPLES = 420;

/** Multi-dose PK simulator: exact superposition, 1-compartment.
 * Regimen state lives in the zustand store, so sliders keep their values
 * when the user switches playground tabs or navigates away and back. */
export function PkSimulator() {
  const params = useToolkitStore((s) => s.pkParams);
  const setPkParam = useToolkitStore((s) => s.setPkParam);
  const resetPkParams = useToolkitStore((s) => s.resetPkParams);

  const { route, dose, vd, cl, tau, doses, f, ka } = params;

  const derived = useMemo(() => derive(params), [params]);

  const series = useMemo(() => {
    const ke = derived.ke;
    const tail = Math.max(3 * derived.halfLife, 2 * params.tau);
    const total = (params.doses - 1) * params.tau + tail;
    const step = total / SAMPLES;
    return Array.from({ length: SAMPLES + 1 }, (_, i) => {
      const t = i * step;
      return {
        t: Number(t.toFixed(2)),
        c: Number(concentrationAt(t, params, ke).toFixed(4)),
      };
    });
  }, [params, derived]);

  const flipFlop = route === "oral" && ka <= 1.25 * derived.ke;

  const readouts: { label: string; value: string; unit: string }[] = [
    { label: "ke", value: derived.ke.toFixed(4), unit: "h⁻¹" },
    { label: "t½", value: derived.halfLife.toFixed(2), unit: "h" },
    { label: "Accumulation R", value: derived.accumulation.toFixed(2), unit: "×" },
    { label: "Css,avg", value: derived.cssAvg.toFixed(2), unit: "mg/L" },
    { label: "Cmax (1st dose)", value: derived.cMaxFirst.toFixed(2), unit: "mg/L" },
  ];
  if (Number.isFinite(derived.cMaxSS)) {
    readouts.push({ label: "Cmax (≈ss)", value: derived.cMaxSS.toFixed(2), unit: "mg/L" });
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
      {/* ------------------------------ Controls ------------------------------ */}
      <Card className="h-fit py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="size-4 text-primary" aria-hidden />
            Regimen parameters
          </CardTitle>
          <CardDescription className="flex items-center justify-between gap-2 text-xs">
            <span>One compartment · first-order elimination · exact superposition</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 py-4">
          <Tabs value={route} onValueChange={(v) => setPkParam("route", v as Route)}>
            <TabsList className="grid h-10 w-full grid-cols-2">
              <TabsTrigger value="iv" className="min-h-9 text-xs">
                IV bolus
              </TabsTrigger>
              <TabsTrigger value="oral" className="min-h-9 text-xs">
                Oral (1-cmt)
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <ParamSlider id="pk-dose" label="Dose" unit="mg" min={25} max={1000} step={25} value={dose} onChange={(v) => setPkParam("dose", v)} />
          <ParamSlider id="pk-vd" label="Volume Vd" unit="L" min={5} max={150} step={1} value={vd} onChange={(v) => setPkParam("vd", v)} />
          <ParamSlider id="pk-cl" label="Clearance CL" unit="L/h" min={0.5} max={12} step={0.1} value={cl} onChange={(v) => setPkParam("cl", v)} format={(v) => v.toFixed(1)} />
          <ParamSlider id="pk-tau" label="Interval τ" unit="h" min={2} max={24} step={0.5} value={tau} onChange={(v) => setPkParam("tau", v)} format={(v) => v.toFixed(1)} />
          <ParamSlider id="pk-doses" label="Doses" unit="" min={1} max={12} step={1} value={doses} onChange={(v) => setPkParam("doses", v)} />

          {route === "oral" ? (
            <>
              <ParamSlider id="pk-f" label="Bioavailability F" unit="" min={0.1} max={1} step={0.05} value={f} onChange={(v) => setPkParam("f", v)} format={(v) => v.toFixed(2)} />
              <ParamSlider id="pk-ka" label="Absorption ka" unit="h⁻¹" min={0.1} max={3} step={0.05} value={ka} onChange={(v) => setPkParam("ka", v)} format={(v) => v.toFixed(2)} />
            </>
          ) : null}

          <Button
            variant="outline"
            size="sm"
            className="w-full gap-2 text-xs"
            onClick={resetPkParams}
            aria-label="Reset all regimen parameters to defaults"
          >
            <RotateCcw className="size-3.5" aria-hidden />
            Reset parameters
          </Button>
        </CardContent>
      </Card>

      {/* ------------------------------ Chart + readouts ------------------------------ */}
      <div className="space-y-4">
        <Card className="py-0 shadow-sm">
          <CardHeader className="border-b bg-muted/30 py-3">
            <CardTitle className="text-base">Concentration–time profile</CardTitle>
            <CardDescription className="text-xs">
              {route === "iv"
                ? "C(t) = Σᵢ (D/Vd)·e^(−ke·(t−iτ)) — repeated IV boluses"
                : "C(t) = Σᵢ (F·D·ka)/(Vd(ka−ke))·(e^(−ke·s) − e^(−ka·s)) — repeated oral doses"}
              {" · dashed line = average steady-state concentration"}
            </CardDescription>
          </CardHeader>
          <CardContent className="py-4">
            <div className="h-[300px] w-full sm:h-[340px]" role="img" aria-label="Plasma concentration versus time chart">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={series} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <defs>
                    <linearGradient id="pkConcFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART.emerald} stopOpacity={0.28} />
                      <stop offset="100%" stopColor={CHART.emerald} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
                  <XAxis
                    dataKey="t"
                    type="number"
                    domain={[0, "dataMax"]}
                    tick={CHART.axisTick}
                    tickLine={false}
                    axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                    label={{ value: "time (h)", position: "insideBottomRight", offset: -2, fill: "#78716c", fontSize: 11 }}
                    minTickGap={32}
                  />
                  <YAxis
                    tick={CHART.axisTick}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    label={{ value: "C (mg/L)", angle: -90, position: "insideLeft", fill: "#78716c", fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={CHART.tooltipStyle}
                    formatter={(value: number | string) => [`${Number(value).toFixed(2)} mg/L`, "Concentration"]}
                    labelFormatter={(label: number | string) => `t = ${Number(label).toFixed(2)} h`}
                  />
                  <ReferenceLine
                    y={derived.cssAvg}
                    stroke={CHART.amber}
                    strokeDasharray="6 4"
                    strokeWidth={1.5}
                    label={{
                      value: `Css,avg ≈ ${derived.cssAvg.toFixed(2)}`,
                      position: "insideTopRight",
                      fill: CHART.amber,
                      fontSize: 11,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="c"
                    stroke="none"
                    fill="url(#pkConcFill)"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="c"
                    stroke={CHART.emerald}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {readouts.map((r) => (
            <div key={r.label} className="readout-card rounded-lg border bg-card p-3 pl-4 shadow-sm">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{r.label}</p>
              <p className="mt-0.5 font-mono text-lg font-semibold tabular-nums leading-tight">
                {r.value}
                {r.unit ? <span className="ml-1 text-xs font-normal text-muted-foreground">{r.unit}</span> : null}
              </p>
            </div>
          ))}
        </div>

        <div
          className={cn(
            "flex items-start gap-2 rounded-lg border p-3 text-xs leading-relaxed",
            flipFlop
              ? "border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200"
              : "bg-muted/40 text-muted-foreground"
          )}
          role="note"
        >
          {flipFlop ? (
            <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
          ) : (
            <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
          )}
          <p>
            {flipFlop ? (
              <>
                <strong>Flip-flop kinetics warning:</strong> ka ≤ 1.25·ke — absorption is now
                rate-limiting, so the terminal slope reflects <em>ka</em>, not elimination.
                Increase ka or CL to restore ka &gt; ke.
              </>
            ) : (
              <>
                <strong>Superposition principle:</strong> because the model is linear, the
                multi-dose profile is the exact sum of each dose&apos;s single-dose curve.
                Accumulation ratio R = 1/(1 − e^(−ke·τ)) ≈ {derived.accumulation.toFixed(2)}× and
                ~95 % of steady state is reached after ≈ {Math.ceil((Math.log(1 / 0.05) / derived.ke) / params.tau)} doses
                ({(Math.log(1 / 0.05) / derived.ke).toFixed(1)} h ≈ {Math.ceil((Math.log(1 / 0.05) / derived.ke) / params.tau)}·τ).
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
