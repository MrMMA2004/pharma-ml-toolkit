"use client";

import { useMemo } from "react";
import {
  Activity,
  Info,
  RotateCcw,
  Sigma,
  TriangleAlert,
  Waves,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useToolkitStore, type PkParams } from "@/lib/store";
import { CHART, ParamSlider } from "@/components/explorer/playground/calc-ui";
import { PdEffect } from "@/components/explorer/playground/pd-effect";
import { NcaPanel } from "@/components/explorer/playground/nca-panel";
import { concentrationAt, derivePk } from "@/lib/pk-pd";

/* ------------------------------------------------------------------ */
/* Regimen panel (one-compartment multi-dose simulator)                */
/* ------------------------------------------------------------------ */

type Route = PkParams["route"];

const SAMPLES = 420;

/** Regimen PK panel: exact superposition, 1-compartment. Regimen state
 * lives in the zustand store, so sliders keep their values when the user
 * switches playground tabs or navigates away and back. */
function RegimenPanel() {
  const params = useToolkitStore((s) => s.pkParams);
  const setPkParam = useToolkitStore((s) => s.setPkParam);
  const resetPkParams = useToolkitStore((s) => s.resetPkParams);

  const { route, dose, vd, cl, tau, doses, f, ka } = params;

  const derived = useMemo(() => derivePk(params), [params]);

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
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
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
                    tooltipType="none"
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

/* ------------------------------------------------------------------ */
/* Shell: PK / PD Explorer inner tabs                                  */
/* ------------------------------------------------------------------ */

/** PK/PD Explorer shell: three inner tabs — regimen simulator, live PD
 * effect layer, and NCA over user-observed data. The active inner tab is
 * persisted in the store (like the playground sub-tab itself). */
export function PkSimulator() {
  const innerTab = useToolkitStore((s) => s.pkInnerTab);
  const setPkInnerTab = useToolkitStore((s) => s.setPkInnerTab);

  return (
    <Tabs value={innerTab} onValueChange={setPkInnerTab} className="w-full">
      <TabsList className="thin-scrollbar mx-auto grid h-auto w-full max-w-md grid-cols-3 gap-1 overflow-x-auto p-1">
        <TabsTrigger
          value="regimen"
          className="min-h-10 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
        >
          <Activity className="size-4 shrink-0" aria-hidden />
          Regimen PK
        </TabsTrigger>
        <TabsTrigger
          value="pd"
          className="min-h-10 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
        >
          <Waves className="size-4 shrink-0" aria-hidden />
          PD effect
        </TabsTrigger>
        <TabsTrigger
          value="nca"
          className="min-h-10 gap-1.5 px-2 text-xs font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
        >
          <Sigma className="size-4 shrink-0" aria-hidden />
          {/* Short label on ≤sm: "NCA (observed)" clips inside its
           * grid cell at 360 px (109 px needed vs 104 available). */}
          <span className="sm:hidden">NCA</span>
          <span className="hidden sm:inline">NCA (observed)</span>
        </TabsTrigger>
      </TabsList>
      <TabsContent value="regimen" className="mt-4 sm:mt-6">
        <RegimenPanel />
      </TabsContent>
      <TabsContent value="pd" className="mt-4 sm:mt-6">
        <PdEffect />
      </TabsContent>
      <TabsContent value="nca" className="mt-4 sm:mt-6">
        <NcaPanel />
      </TabsContent>
    </Tabs>
  );
}
