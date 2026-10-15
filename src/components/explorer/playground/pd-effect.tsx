"use client";

import { useMemo } from "react";
import { ArrowRight, Info, RotateCcw, TriangleAlert, Waves } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useToolkitStore, type PdView } from "@/lib/store";
import { fmtNum, fmtTick } from "@/lib/format";
import { CHART, DedupeLegend, ParamSlider } from "@/components/explorer/playground/calc-ui";
import {
  concentrationAt,
  derivePk,
  pdEffect,
  pdHalfMaxEffect,
  pdPotency,
  timeAbove,
  PD_MODELS,
  type PdModelId,
} from "@/lib/pk-pd";

const SAMPLES = 420;

/** Advisory note tone (mirrors the flip-flop note in the regimen panel). */
function AdvisoryNote({
  tone,
  children,
}: {
  tone: "amber" | "info";
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-lg border p-3 text-xs leading-relaxed",
        tone === "amber"
          ? "border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200"
          : "bg-muted/40 text-muted-foreground",
      )}
      role="note"
    >
      {tone === "amber" ? (
        <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
      ) : (
        <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
      )}
      <p>{children}</p>
    </div>
  );
}

/**
 * PD effect layer: the four pharmaml.pharma.pd models (linear / Emax /
 * sigmoid Emax / inhibitory Emax) evaluated on the LIVE concentration
 * curve of the regimen simulator. Model + parameters live in the store.
 */
export function PdEffect() {
  const pk = useToolkitStore((s) => s.pkParams);
  const pd = useToolkitStore((s) => s.pdParams);
  const setPdParam = useToolkitStore((s) => s.setPdParam);
  const resetPdParams = useToolkitStore((s) => s.resetPdParams);
  const setPkInnerTab = useToolkitStore((s) => s.setPkInnerTab);

  const model = PD_MODELS.find((m) => m.id === pd.model) ?? PD_MODELS[1];
  const derived = useMemo(() => derivePk(pk), [pk]);

  const series = useMemo(() => {
    const ke = derived.ke;
    const tail = Math.max(3 * derived.halfLife, 2 * pk.tau);
    const total = (pk.doses - 1) * pk.tau + tail;
    const step = total / SAMPLES;
    return Array.from({ length: SAMPLES + 1 }, (_, i) => {
      const t = i * step;
      const c = concentrationAt(t, pk, ke);
      return {
        t: Number(t.toFixed(2)),
        c: Number(c.toFixed(4)),
        e: Number(pdEffect(c, pd.model, pd).toFixed(4)),
      };
    });
  }, [pk, derived, pd]);

  /* Observed headline numbers over the sampled curve. */
  const cMaxObs = useMemo(() => Math.max(...series.map((p) => p.c)), [series]);
  const potency = pdPotency(pd.model, pd);
  const potencyLabel = pd.model === "inhibitory" ? "IC50" : "EC50";
  const eAtCmax = pdEffect(cMaxObs, pd.model, pd);
  const eAtCss = pdEffect(derived.cssAvg, pd.model, pd);
  const tAbove50raw = potency !== null ? timeAbove(series, potency) : null;
  const tAbove50 =
    tAbove50raw !== null && Number.isFinite(tAbove50raw) && tAbove50raw >= 0
      ? tAbove50raw
      : null;
  const halfMax = pdHalfMaxEffect(pd.model, pd);
  const ratio = potency !== null && cMaxObs > 0 ? potency / cMaxObs : null;

  /* Degenerate-combination notes (amber, never NaN on screen). */
  const barelyActivates = potency !== null && potency > 10 * cMaxObs;
  const nearSaturated =
    potency !== null && cMaxObs > 50 * potency && pd.model !== "linear";

  const showConc = pd.view !== "effect";
  const showEffect = pd.view !== "conc";
  const effectMin = Math.min(0, ...series.map((p) => p.e));

  const readouts: Array<{ label: string; value: string; unit: string }> = [
    { label: "Cmax (observed)", value: fmtNum(cMaxObs, 3), unit: "mg/L" },
    { label: "E at Cmax", value: fmtNum(eAtCmax, 3), unit: "E" },
    { label: "E at Css,avg", value: fmtNum(eAtCss, 3), unit: "E" },
    potency !== null
      ? {
          label: `t > ${potencyLabel}`,
          value: tAbove50 !== null ? fmtNum(tAbove50, 3) : "—",
          unit: "h",
        }
      : { label: "t > EC50", value: "—", unit: "h" },
    potency !== null
      ? {
          label: `${potencyLabel}/Cmax`,
          value: ratio !== null ? fmtNum(ratio, 3) : "—",
          unit: "×",
        }
      : { label: "EC50/Cmax", value: "—", unit: "×" },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
      {/* ------------------------------ Controls ------------------------------ */}
      <Card className="h-fit py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Waves className="size-4 text-primary" aria-hidden />
            PD model
          </CardTitle>
          <CardDescription className="text-xs">
            pharmaml.pharma.pd — effect driven by the live regimen C(t)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 py-4">
          <div className="space-y-1.5">
            <Label htmlFor="pd-model" className="text-xs font-medium">
              Effect model
            </Label>
            <Select value={pd.model} onValueChange={(v) => setPdParam("model", v as PdModelId)}>
              <SelectTrigger id="pd-model" className="h-10" aria-label="PD effect model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PD_MODELS.map((m) => (
                  <SelectItem key={m.id} value={m.id} textValue={m.label}>
                    <span className="text-sm">{m.label}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="font-mono text-[11px] leading-snug text-muted-foreground" aria-label="PD model equation">
              {model.equation}
            </p>
          </div>

          {pd.model === "linear" ? (
            <ParamSlider
              id="pd-slope"
              label="Slope"
              unit="E/(mg/L)"
              min={0.5}
              max={20}
              step={0.5}
              value={pd.slope}
              onChange={(v) => setPdParam("slope", v)}
              format={(v) => v.toFixed(1)}
            />
          ) : null}

          {pd.model !== "linear" ? (
            <>
              <ParamSlider
                id="pd-e0"
                label="Baseline E0"
                unit="E"
                min={0}
                max={100}
                step={1}
                value={pd.e0}
                onChange={(v) => setPdParam("e0", v)}
              />
              {pd.model === "inhibitory" ? (
                <ParamSlider
                  id="pd-imax"
                  label="Max inhibition Imax"
                  unit="E"
                  min={0}
                  max={100}
                  step={1}
                  value={pd.imax}
                  onChange={(v) => setPdParam("imax", v)}
                />
              ) : (
                <ParamSlider
                  id="pd-emax"
                  label="Max effect Emax"
                  unit="E"
                  min={0}
                  max={100}
                  step={1}
                  value={pd.emax}
                  onChange={(v) => setPdParam("emax", v)}
                />
              )}
              <ParamSlider
                id={`pd-${pd.model === "inhibitory" ? "ic50" : "ec50"}`}
                label={`Potency ${pd.model === "inhibitory" ? "IC50" : "EC50"}`}
                unit="mg/L"
                min={0.05}
                max={20}
                step={0.05}
                value={pd.model === "inhibitory" ? pd.ic50 : pd.ec50}
                onChange={(v) =>
                  setPdParam(pd.model === "inhibitory" ? "ic50" : "ec50", v)
                }
                format={(v) => v.toFixed(2)}
              />
            </>
          ) : null}

          {pd.model === "sigmoid" ? (
            <ParamSlider
              id="pd-hill"
              label="Hill coefficient h"
              unit=""
              min={0.5}
              max={5}
              step={0.1}
              value={pd.hill}
              onChange={(v) => setPdParam("hill", v)}
              format={(v) => v.toFixed(1)}
            />
          ) : null}

          {/* Compact regimen summary + jump to the regimen sliders. */}
          <div className="rounded-lg border bg-muted/30 p-3">
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Driving regimen
            </p>
            <p className="mt-1 font-mono text-xs tabular-nums leading-relaxed">
              {pk.route === "iv" ? "IV" : "Oral"} · D {pk.dose} mg · Vd {pk.vd} L · CL{" "}
              {pk.cl.toFixed(1)} L/h · τ {pk.tau} h × {pk.doses}
            </p>
            <Button
              type="button"
              variant="link"
              size="sm"
              className="mt-1 h-auto gap-1 p-0 text-xs"
              onClick={() => setPkInnerTab("regimen")}
              aria-label="Switch to the regimen parameters tab"
            >
              Adjust regimen parameters
              <ArrowRight className="size-3" aria-hidden />
            </Button>
          </div>

          <Button
            variant="outline"
            size="sm"
            className="w-full gap-2 text-xs"
            onClick={resetPdParams}
            aria-label="Reset the PD model and parameters to defaults"
          >
            <RotateCcw className="size-3.5" aria-hidden />
            Reset PD parameters
          </Button>
        </CardContent>
      </Card>

      {/* ------------------------------ Chart + readouts ------------------------------ */}
      <div className="space-y-4">
        <Card className="py-0 shadow-sm">
          <CardHeader className="border-b bg-muted/30 py-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <CardTitle className="text-base">Concentration &amp; effect–time profile</CardTitle>
                <CardDescription className="mt-0.5 font-mono text-[11px]">
                  {model.equation} · effect evaluated on the live C(t)
                </CardDescription>
              </div>
              <Tabs
                value={pd.view}
                onValueChange={(v) => setPdParam("view", v as PdView)}
              >
                <TabsList className="h-9 grid grid-cols-3">
                  <TabsTrigger value="conc" className="px-2 text-xs">
                    Conc
                  </TabsTrigger>
                  <TabsTrigger value="both" className="px-2 text-xs">
                    Both
                  </TabsTrigger>
                  <TabsTrigger value="effect" className="px-2 text-xs">
                    Effect
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </CardHeader>
          <CardContent className="py-4">
            <div
              className="h-[300px] w-full sm:h-[340px]"
              role="img"
              aria-label="Plasma concentration and drug effect versus time chart"
            >
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={series} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
                  <defs>
                    <linearGradient id="pdConcFill" x1="0" y1="0" x2="0" y2="1">
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
                    label={{
                      value: "time (h)",
                      position: "insideBottomRight",
                      offset: -2,
                      fill: "#78716c",
                      fontSize: 11,
                    }}
                    tickFormatter={fmtTick}
                    minTickGap={32}
                  />
                  {showConc ? (
                    <YAxis
                      yAxisId="conc"
                      tick={CHART.axisTick}
                      tickFormatter={fmtTick}
                      tickLine={false}
                      axisLine={false}
                      width={44}
                      label={{
                        value: "C (mg/L)",
                        angle: -90,
                        position: "insideLeft",
                        fill: "#78716c",
                        fontSize: 11,
                      }}
                    />
                  ) : null}
                  {showEffect ? (
                    <YAxis
                      yAxisId="effect"
                      orientation="right"
                      domain={[effectMin, "dataMax"]}
                      tick={{ fill: CHART.rose, fontSize: 11 }}
                      tickFormatter={fmtTick}
                      tickLine={false}
                      axisLine={false}
                      width={44}
                      label={{
                        value: "Effect E",
                        angle: 90,
                        position: "insideRight",
                        fill: CHART.rose,
                        fontSize: 11,
                      }}
                    />
                  ) : null}
                  <Tooltip
                    contentStyle={CHART.tooltipStyle}
                    formatter={(value: number | string, name: string) => {
                      const v = Number(value);
                      if (!Number.isFinite(v)) return ["—", name];
                      return name === "Effect"
                        ? [`${v.toFixed(2)} E`, "Effect"]
                        : [`${v.toFixed(2)} mg/L`, "Concentration"];
                    }}
                    labelFormatter={(label: number | string) => `t = ${Number(label).toFixed(2)} h`}
                  />
                  <Legend content={<DedupeLegend />} />
                  {showConc ? (
                    <ReferenceLine
                      yAxisId="conc"
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
                  ) : null}
                  {showConc && potency !== null ? (
                    <ReferenceLine
                      yAxisId="conc"
                      y={potency}
                      stroke={CHART.rose}
                      strokeDasharray="5 4"
                      strokeWidth={1.5}
                      label={{
                        value: `${potencyLabel} = ${potency.toFixed(2)}`,
                        position: "insideTopLeft",
                        fill: CHART.rose,
                        fontSize: 11,
                      }}
                    />
                  ) : null}
                  {showEffect && halfMax !== null ? (
                    <ReferenceLine
                      yAxisId="effect"
                      y={halfMax}
                      stroke={CHART.stone}
                      strokeOpacity={0.6}
                      strokeDasharray="3 4"
                      label={{
                        value: "half-max",
                        position: "insideBottomRight",
                        fill: "#78716c",
                        fontSize: 10,
                      }}
                    />
                  ) : null}
                  {showConc ? (
                    <Area
                      yAxisId="conc"
                      type="monotone"
                      dataKey="c"
                      name="Concentration"
                      stroke="none"
                      fill="url(#pdConcFill)"
                      legendType="none"
                      tooltipType="none"
                      isAnimationActive={false}
                    />
                  ) : null}
                  {showConc ? (
                    <Line
                      yAxisId="conc"
                      type="monotone"
                      dataKey="c"
                      name="Concentration"
                      stroke={CHART.emerald}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ) : null}
                  {showEffect ? (
                    <Line
                      yAxisId="effect"
                      type="monotone"
                      dataKey="e"
                      name="Effect"
                      stroke={CHART.rose}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ) : null}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {readouts.map((r) => (
            <div key={r.label} className="readout-card min-w-0 rounded-lg border bg-card p-3 pl-4 shadow-sm">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                {r.label}
              </p>
              <p className="mt-0.5 font-mono text-lg font-semibold tabular-nums leading-tight">
                {r.value}
                {r.unit ? (
                  <span className="ml-1 text-xs font-normal text-muted-foreground">{r.unit}</span>
                ) : null}
              </p>
            </div>
          ))}
        </div>

        {barelyActivates ? (
          <AdvisoryNote tone="amber">
            <strong>{potencyLabel} ≫ Cmax:</strong> the potency parameter is{" "}
            {fmtNum(ratio, 2)}× the observed peak concentration, so the model barely activates —
            the effect stays within{" "}
            {Math.abs(pdEffect(cMaxObs, pd.model, pd) - pdEffect(0, pd.model, pd)).toFixed(1)} E of
            baseline across the whole profile. Lower {potencyLabel} or raise the dose.
          </AdvisoryNote>
        ) : null}
        {nearSaturated ? (
          <AdvisoryNote tone="amber">
            <strong>Near saturation:</strong> Cmax is{" "}
            {fmtNum(cMaxObs / (potency ?? 1), 1)}× {potencyLabel} — the effect sits close to its
            ceiling (E ≈ {fmtNum(pdEffect(cMaxObs, pd.model, pd), 1)} E) for most of the profile,
            so concentration changes barely translate into effect changes.
          </AdvisoryNote>
        ) : null}
        <AdvisoryNote tone="info">
          <strong>Link model:</strong> the effect curve is the PD model evaluated at every point of
          the live concentration curve — E(t) = E(C(t)). The dashed {potencyLabel} line marks the
          concentration at half the maximal drug effect; the grey line on the effect axis marks
          that half-maximum (E0 {pd.model === "inhibitory" ? "−" : "+"}{" "}
          {pd.model === "inhibitory" ? "Imax" : "Emax"}/2).
          {pd.model === "linear"
            ? " The linear model has no potency parameter — effect grows without bound with C (small-dose approximation)."
            : null}
        </AdvisoryNote>
      </div>
    </div>
  );
}
