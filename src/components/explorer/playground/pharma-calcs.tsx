"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  ArrowLeftRight,
  LineChart as LineChartIcon,
  Pill,
  Plus,
  RefreshCw,
  Ruler,
  Sigma,
  Timer,
  Trash2,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
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
import { Slider } from "@/components/ui/slider";
import { CHART, NumberField, ResultBlock } from "@/components/explorer/playground/calc-ui";
import { fmtNum, fmtTick, parseNonNegativeNumber, parsePositiveNumber } from "@/lib/format";
import {
  UNIT_SCALES,
  UnitConversionError,
  convertUnits,
  unitsForDimension,
  type Dimension,
} from "@/lib/unit-converter";

/* ------------------------------------------------------------------ */
/* Shared card shell                                                   */
/* ------------------------------------------------------------------ */

function CalcCard({
  icon: Icon,
  title,
  description,
  children,
  className,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={`overflow-hidden py-0 shadow-sm ${className ?? ""}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Icon className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">{title}</CardTitle>
            <CardDescription className="text-xs">{description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pb-4">{children}</CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* 1. Creatinine clearance (Cockcroft–Gault)                           */
/* ------------------------------------------------------------------ */

const CRCL_BANDS: Array<{ min: number; label: string; className: string }> = [
  { min: 90, label: "Normal renal function", className: "bg-primary/10 text-primary border-primary/30" },
  { min: 60, label: "Mild reduction", className: "bg-teal-500/10 text-teal-700 dark:text-teal-400 border-teal-500/30" },
  { min: 30, label: "Moderate reduction", className: "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30" },
  { min: 0, label: "Severe reduction", className: "bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-500/30" },
];

function CrclCard() {
  const [age, setAge] = useState("68");
  const [weight, setWeight] = useState("70");
  const [scr, setScr] = useState("1.1");
  const [sex, setSex] = useState<"male" | "female">("male");

  const a = parsePositiveNumber(age);
  const w = parsePositiveNumber(weight);
  const s = parsePositiveNumber(scr);

  const err = (parsed: number | null, fallback: string) =>
    parsed === null ? (fallback.length === 0 ? "Enter a number" : "Must be a positive number") : null;

  const valid = a !== null && w !== null && s !== null;
  const crcl = valid ? ((140 - (a as number)) * (w as number)) / (72 * (s as number)) : null;
  const female = sex === "female" ? 0.85 : 1;
  const crclFinal = crcl !== null ? crcl * female : null;

  const band =
    crclFinal !== null ? (CRCL_BANDS.find((b) => crclFinal >= b.min) ?? CRCL_BANDS[3]) : null;

  return (
    <CalcCard
      icon={Activity}
      title="Creatinine clearance"
      description="Cockcroft–Gault estimate of renal function"
    >
      <div className="grid grid-cols-2 gap-3">
        <NumberField id="crcl-age" label="Age" unit="years" value={age} onChange={setAge} error={err(a, age)} placeholder="68" />
        <NumberField id="crcl-wt" label="Weight" unit="kg" value={weight} onChange={setWeight} error={err(w, weight)} placeholder="70" />
        <NumberField id="crcl-scr" label="Serum creatinine" unit="mg/dL" value={scr} onChange={setScr} error={err(s, scr)} placeholder="1.1" />
        <div className="space-y-1.5">
          <Label htmlFor="crcl-sex" className="text-xs font-medium">
            Sex
          </Label>
          <Select value={sex} onValueChange={(v) => setSex(v === "female" ? "female" : "male")}>
            <SelectTrigger id="crcl-sex" className="h-10">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="male">Male</SelectItem>
              <SelectItem value="female">Female (× 0.85)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <ResultBlock
        label="CrCl"
        value={crclFinal !== null ? fmtNum(crclFinal, 4) : null}
        unit="mL/min"
        copyText={crclFinal !== null ? `CrCl = ${fmtNum(crclFinal, 4)} mL/min` : null}
        formula={`CrCl = (140 − age) × wt / (72 × SCr)${female === 1 ? "" : " × 0.85 (female)"}`}
      >
        {band && crclFinal !== null ? (
          <div className="mt-2">
            <Badge variant="outline" className={`border font-normal ${band.className}`}>
              {band.label}
            </Badge>
          </div>
        ) : null}
      </ResultBlock>
    </CalcCard>
  );
}

/* ------------------------------------------------------------------ */
/* 2. BSA (Mosteller)                                                  */
/* ------------------------------------------------------------------ */

function BsaCard() {
  const [weight, setWeight] = useState("70");
  const [height, setHeight] = useState("175");

  const w = parsePositiveNumber(weight);
  const h = parsePositiveNumber(height);
  const valid = w !== null && h !== null;
  const bsa = valid ? Math.sqrt(((h as number) * (w as number)) / 3600) : null;

  return (
    <CalcCard icon={Ruler} title="Body surface area" description="Mosteller formula">
      <div className="grid grid-cols-2 gap-3">
        <NumberField id="bsa-wt" label="Weight" unit="kg" value={weight} onChange={setWeight} error={w === null ? "Must be a positive number" : null} placeholder="70" />
        <NumberField id="bsa-ht" label="Height" unit="cm" value={height} onChange={setHeight} error={h === null ? "Must be a positive number" : null} placeholder="175" />
      </div>
      <ResultBlock
        label="BSA"
        value={bsa !== null ? fmtNum(bsa, 4) : null}
        unit="m²"
        copyText={bsa !== null ? `BSA = ${fmtNum(bsa, 4)} m²` : null}
        formula="BSA = √( height(cm) × weight(kg) / 3600 )"
      />
    </CalcCard>
  );
}

/* ------------------------------------------------------------------ */
/* 3. Pediatric dose (mg/kg)                                           */
/* ------------------------------------------------------------------ */

const FREQUENCIES: Array<{ value: string; label: string; hours: number }> = [
  { value: "6", label: "q6h (4×/day)", hours: 6 },
  { value: "8", label: "q8h (3×/day)", hours: 8 },
  { value: "12", label: "q12h (2×/day)", hours: 12 },
  { value: "24", label: "q24h (1×/day)", hours: 24 },
];

function PediatricDoseCard() {
  const [mgPerKg, setMgPerKg] = useState("15");
  const [weight, setWeight] = useState("22");
  const [maxAdult, setMaxAdult] = useState("500");
  const [freq, setFreq] = useState("8");

  const d = parsePositiveNumber(mgPerKg);
  const w = parsePositiveNumber(weight);
  const max = maxAdult.trim() === "" ? null : parsePositiveNumber(maxAdult);
  const hours = FREQUENCIES.find((f) => f.value === freq)?.hours ?? 8;

  const valid = d !== null && w !== null && (maxAdult.trim() === "" || max !== null);
  const rawDose = valid ? (d as number) * (w as number) : null;
  const capped = rawDose !== null && max !== null ? Math.min(rawDose, max) : rawDose;
  const perDay = capped !== null ? (capped * 24) / hours : null;

  return (
    <CalcCard icon={Pill} title="Pediatric dose" description="mg/kg scaling with optional adult cap">
      <div className="grid grid-cols-2 gap-3">
        <NumberField id="ped-dose" label="Dose" unit="mg/kg" value={mgPerKg} onChange={setMgPerKg} error={d === null ? "Must be a positive number" : null} placeholder="15" />
        <NumberField id="ped-wt" label="Weight" unit="kg" value={weight} onChange={setWeight} error={w === null ? "Must be a positive number" : null} placeholder="22" />
        <NumberField
          id="ped-max"
          label="Max adult dose"
          unit="mg"
          optional
          value={maxAdult}
          onChange={setMaxAdult}
          error={maxAdult.trim() !== "" && max === null ? "Must be a positive number" : null}
          placeholder="500"
        />
        <div className="space-y-1.5">
          <Label htmlFor="ped-freq" className="text-xs font-medium">
            Frequency
          </Label>
          <Select value={freq} onValueChange={setFreq}>
            <SelectTrigger id="ped-freq" className="h-10">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {FREQUENCIES.map((f) => (
                <SelectItem key={f.value} value={f.value}>
                  {f.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <ResultBlock
          label="Per dose"
          value={capped !== null ? fmtNum(capped, 4) : null}
          unit="mg"
          copyText={capped !== null ? `Dose = ${fmtNum(capped, 4)} mg` : null}
          formula="dose = mg/kg × weight (≤ adult max)"
        />
        <ResultBlock
          label="Per day"
          value={perDay !== null ? fmtNum(perDay, 4) : null}
          unit="mg/day"
          copyText={perDay !== null ? `Daily total = ${fmtNum(perDay, 4)} mg/day` : null}
          formula={`daily = dose × 24 / τ (τ = ${hours} h)`}
        />
      </div>
    </CalcCard>
  );
}

/* ------------------------------------------------------------------ */
/* 4. Loading & maintenance dose                                       */
/* ------------------------------------------------------------------ */

function LoadingMaintenanceCard() {
  const [targetConc, setTargetConc] = useState("12");
  const [vd, setVd] = useState("45");
  const [clearance, setClearance] = useState("3.2");
  const [css, setCss] = useState("6");
  const [tau, setTau] = useState("8");
  const [f, setF] = useState("1.0");

  const tc = parsePositiveNumber(targetConc);
  const v = parsePositiveNumber(vd);
  const cl = parsePositiveNumber(clearance);
  const cs = parsePositiveNumber(css);
  const t = parsePositiveNumber(tau);
  const fv = parsePositiveNumber(f);

  const fError =
    fv === null
      ? "Must be > 0"
      : (fv as number) > 1
        ? "Bioavailability cannot exceed 1"
        : null;

  const valid = tc !== null && v !== null && cl !== null && cs !== null && t !== null && fError === null;

  const ld = valid ? (tc as number) * (v as number) : null;
  const md = valid ? ((cl as number) * (cs as number) * (t as number)) / (fv as number) : null;

  return (
    <CalcCard
      icon={Timer}
      title="Loading & maintenance dose"
      description="One-compartment regimen design"
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <NumberField id="ld-tc" label="Target conc." unit="mg/L" value={targetConc} onChange={setTargetConc} error={tc === null ? "Must be > 0" : null} />
        <NumberField id="ld-vd" label="Vd" unit="L" value={vd} onChange={setVd} error={v === null ? "Must be > 0" : null} />
        <NumberField id="ld-cl" label="Clearance" unit="L/h" value={clearance} onChange={setClearance} error={cl === null ? "Must be > 0" : null} />
        <NumberField id="ld-css" label="Target Css" unit="mg/L" value={css} onChange={setCss} error={cs === null ? "Must be > 0" : null} />
        <NumberField id="ld-tau" label="Interval τ" unit="h" value={tau} onChange={setTau} error={t === null ? "Must be > 0" : null} />
        <NumberField id="ld-f" label="Bioavail. F" unit="0–1" value={f} onChange={setF} error={fError} />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <ResultBlock
          label="Loading dose"
          value={ld !== null ? fmtNum(ld, 4) : null}
          unit="mg"
          copyText={ld !== null ? `Loading dose = ${fmtNum(ld, 4)} mg` : null}
          formula="LD = C_target × Vd"
        />
        <ResultBlock
          label="Maintenance dose"
          value={md !== null ? fmtNum(md, 4) : null}
          unit="mg per τ"
          copyText={md !== null ? `Maintenance dose = ${fmtNum(md, 4)} mg per ${fmtNum(t as number, 2)} h` : null}
          formula="MD = CL × Css × τ / F"
        />
      </div>
    </CalcCard>
  );
}

/* ------------------------------------------------------------------ */
/* 5. Half-life ↔ ke converter                                         */
/* ------------------------------------------------------------------ */

function HalfLifeCard() {
  const [tHalf, setTHalf] = useState("6");
  const [ke, setKe] = useState("0.1155");

  const th = parsePositiveNumber(tHalf);
  const k = parsePositiveNumber(ke);

  const keOut = th !== null ? Math.LN2 / (th as number) : null;
  const thOut = k !== null ? Math.LN2 / (k as number) : null;

  return (
    <CalcCard
      icon={RefreshCw}
      title="Half-life ↔ elimination rate"
      description="Two-way converter, first-order kinetics"
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-3">
          <NumberField
            id="hl-thalf"
            label="Half-life t½"
            unit="h"
            value={tHalf}
            onChange={setTHalf}
            error={th === null ? "Must be a positive number" : null}
          />
          <ResultBlock
            label="ke"
            value={keOut !== null ? fmtNum(keOut, 4) : null}
            unit="h⁻¹"
            copyText={keOut !== null ? `ke = ${fmtNum(keOut, 4)} /h` : null}
            formula="ke = ln(2) / t½"
          />
        </div>
        <div className="space-y-3">
          <NumberField
            id="hl-ke"
            label="Elimination rate ke"
            unit="h⁻¹"
            value={ke}
            onChange={setKe}
            error={k === null ? "Must be a positive number" : null}
          />
          <ResultBlock
            label="Half-life"
            value={thOut !== null ? fmtNum(thOut, 4) : null}
            unit="h"
            copyText={thOut !== null ? `t½ = ${fmtNum(thOut, 4)} h` : null}
            formula="t½ = ln(2) / ke"
          />
        </div>
      </div>
    </CalcCard>
  );
}

/* ------------------------------------------------------------------ */
/* 6. AUC₀→last (trapezoidal) with concentration–time chart            */
/* ------------------------------------------------------------------ */

interface AucRow {
  id: number;
  t: string;
  c: string;
}

const SAMPLE_ROWS: Array<{ t: string; c: string }> = [
  { t: "0", c: "0" },
  { t: "1", c: "8.2" },
  { t: "2", c: "6.1" },
  { t: "4", c: "3.4" },
  { t: "8", c: "1.1" },
  { t: "12", c: "0.35" },
];

function AucCard() {
  const [rows, setRows] = useState<AucRow[]>(() =>
    SAMPLE_ROWS.map((r, i) => ({ id: i + 1, ...r }))
  );

  const parsed = useMemo(
    () => rows.map((r) => ({ t: parseNonNegativeNumber(r.t), c: parseNonNegativeNumber(r.c) })),
    [rows]
  );

  const rowError = (v: number | null) => (v === null ? "≥ 0" : null);

  const timesIncreasing = useMemo(() => {
    const ts = parsed.map((p) => p.t).filter((v): v is number => v !== null);
    if (ts.length !== parsed.length || ts.length < 2) return false;
    for (let i = 1; i < ts.length; i++) {
      if (ts[i] <= ts[i - 1]) return false;
    }
    return true;
  }, [parsed]);

  const allValid = parsed.length >= 2 && parsed.every((p) => p.t !== null && p.c !== null) && timesIncreasing;

  const auc = useMemo(() => {
    if (!allValid) return null;
    let acc = 0;
    for (let i = 1; i < parsed.length; i++) {
      const t0 = parsed[i - 1]?.t as number;
      const c0 = parsed[i - 1]?.c as number;
      const t1 = parsed[i]?.t as number;
      const c1 = parsed[i]?.c as number;
      acc += (t1 - t0) * ((c0 + c1) / 2);
    }
    return acc;
  }, [allValid, parsed]);

  const chartData = useMemo(
    () =>
      parsed
        .map((p) => ({ t: p.t, c: p.c }))
        .filter((p): p is { t: number; c: number } => p.t !== null && p.c !== null),
    [parsed]
  );

  const nextId = rows.length > 0 ? Math.max(...rows.map((r) => r.id)) + 1 : 1;

  function updateRow(id: number, field: "t" | "c", value: string) {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, [field]: value } : r)));
  }

  return (
    <CalcCard
      icon={Sigma}
      title="AUC₀→last — trapezoidal rule"
      description="Editable concentration–time pairs with a live chart"
      className="xl:col-span-2"
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(280px,380px)_minmax(0,1fr)]">
        {/* Pair editor */}
        <div className="space-y-2">
          <div className="grid grid-cols-[28px_1fr_1fr_36px] items-center gap-2 px-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            <span>#</span>
            <span>time (h)</span>
            <span>conc (µg/mL)</span>
            <span className="sr-only">Remove row</span>
          </div>
          <div className="thin-scrollbar max-h-64 space-y-1.5 overflow-y-auto pr-1">
            {rows.map((r, i) => (
              <div key={r.id} className="grid grid-cols-[28px_1fr_1fr_36px] items-start gap-2">
                <span className="flex h-10 items-center justify-center rounded-md bg-muted font-mono text-xs text-muted-foreground tabular-nums">
                  {i + 1}
                </span>
                <div>
                  <Input
                    aria-label={`Row ${i + 1} time in hours`}
                    inputMode="decimal"
                    value={r.t}
                    onChange={(e) => updateRow(r.id, "t", e.target.value)}
                    className="h-10 font-mono text-sm tabular-nums"
                  />
                  {rowError(parsed[i]?.t ?? null) ? (
                    <p className="mt-0.5 text-[10px] text-destructive">Must be ≥ 0</p>
                  ) : null}
                </div>
                <div>
                  <Input
                    aria-label={`Row ${i + 1} concentration`}
                    inputMode="decimal"
                    value={r.c}
                    onChange={(e) => updateRow(r.id, "c", e.target.value)}
                    className="h-10 font-mono text-sm tabular-nums"
                  />
                  {rowError(parsed[i]?.c ?? null) ? (
                    <p className="mt-0.5 text-[10px] text-destructive">Must be ≥ 0</p>
                  ) : null}
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-10 w-9 text-muted-foreground hover:text-destructive"
                  onClick={() => setRows((prev) => prev.filter((x) => x.id !== r.id))}
                  aria-label={`Remove row ${i + 1}`}
                  disabled={rows.length <= 2}
                >
                  <Trash2 className="size-4" aria-hidden />
                </Button>
              </div>
            ))}
          </div>
          <div className="flex items-center justify-between gap-2 pt-1">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-1.5"
              onClick={() => setRows((prev) => [...prev, { id: nextId, t: "", c: "" }])}
            >
              <Plus className="size-3.5" aria-hidden />
              Add pair
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-9 gap-1.5 text-xs text-muted-foreground"
              onClick={() => setRows(SAMPLE_ROWS.map((r, i) => ({ id: i + 1, ...r })))}
            >
              <RefreshCw className="size-3.5" aria-hidden />
              Reset sample
            </Button>
          </div>
          {!timesIncreasing && parsed.every((p) => p.t !== null && p.c !== null) ? (
            <p className="flex items-center gap-1.5 text-xs text-destructive">
              Times must be strictly increasing.
            </p>
          ) : null}
          <ResultBlock
            label="AUC₀→last"
            value={auc !== null ? fmtNum(auc, 4) : null}
            unit="µg·h/mL"
            copyText={auc !== null ? `AUC0-last = ${fmtNum(auc, 4)} µg·h/mL` : null}
            formula="AUC = Σ (tᵢ₊₁ − tᵢ) · (Cᵢ + Cᵢ₊₁) / 2"
          />
        </div>

        {/* Chart */}
        <div className="min-w-0">
          <div className="h-[300px] w-full" role="img" aria-label="Concentration versus time chart with trapezoid shading">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
                <XAxis
                  dataKey="t"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tick={CHART.axisTick}
                  tickFormatter={fmtTick}
                  tickLine={false}
                  axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                  label={{ value: "time (h)", position: "insideBottomRight", offset: -2, fill: (typeof window === "undefined" ? "#78716c" : getComputedStyle(document.documentElement).getPropertyValue("--muted-foreground").trim() || "#78716c"), fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  width={44}
                  tick={CHART.axisTick}
                  tickFormatter={fmtTick}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={CHART.tooltipStyle}
                  formatter={(value) => fmtNum(Number(value), 4)}
                  labelFormatter={(label) => `t = ${fmtNum(Number(label), 2)} h`}
                />
                <Area
                  type="linear"
                  dataKey="c"
                  name="concentration"
                  stroke={CHART.emerald}
                  strokeWidth={2}
                  fill={CHART.emerald}
                  fillOpacity={0.14}
                  isAnimationActive={false}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
                <Scatter dataKey="c" name="samples" fill={CHART.emeraldDark} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-center text-[11px] text-muted-foreground">
            Shaded area = trapezoidal AUC (linear interpolation between samples).
          </p>
        </div>
      </div>
    </CalcCard>
  );
}

/* ------------------------------------------------------------------ */
/* 7. PD Emax / Hill simulator                                         */
/* ------------------------------------------------------------------ */

function PdCard() {
  const [emax, setEmax] = useState(80);
  const [ec50, setEc50] = useState(5);
  const [hill, setHill] = useState(2.5);

  const data = useMemo(() => {
    const pts: Array<{ conc: number; plain: number; sigmoid: number }> = [];
    const N = 80;
    for (let i = 0; i < N; i++) {
      const conc = 0.01 * Math.pow(100 / 0.01, i / (N - 1)); // 0.01 → 100
      const plain = (emax * conc) / (ec50 + conc);
      const sigmoid = (emax * Math.pow(conc, hill)) / (Math.pow(ec50, hill) + Math.pow(conc, hill));
      pts.push({ conc, plain, sigmoid });
    }
    return pts;
  }, [emax, ec50, hill]);

  const copyText = `Emax = ${emax}, EC50 = ${ec50}, Hill = ${hill.toFixed(1)}`;

  return (
    <CalcCard
      icon={LineChartIcon}
      title="PD Emax / sigmoid Hill simulator"
      description="Dose–response curves on a log concentration axis"
      className="xl:col-span-2"
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(260px,320px)_minmax(0,1fr)]">
        <div className="space-y-5">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="pd-emax" className="text-xs font-medium">
                Emax (max effect)
              </Label>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">{emax}</span>
            </div>
            <Slider
              id="pd-emax"
              min={1}
              max={100}
              step={1}
              value={[emax]}
              onValueChange={(v) => setEmax(v[0] ?? emax)}
              aria-label="Emax maximum effect"
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="pd-ec50" className="text-xs font-medium">
                EC50
              </Label>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {ec50.toFixed(1)}
              </span>
            </div>
            <Slider
              id="pd-ec50"
              min={0.1}
              max={50}
              step={0.1}
              value={[ec50]}
              onValueChange={(v) => setEc50(v[0] ?? ec50)}
              aria-label="EC50 concentration"
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="pd-hill" className="text-xs font-medium">
                Hill coefficient
              </Label>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {hill.toFixed(1)}
              </span>
            </div>
            <Slider
              id="pd-hill"
              min={0.2}
              max={5}
              step={0.1}
              value={[hill]}
              onValueChange={(v) => setHill(v[0] ?? hill)}
              aria-label="Hill coefficient"
            />
          </div>
          <ResultBlock
            label="Model parameters"
            value={`h = ${hill.toFixed(1)}`}
            copyText={copyText}
            formula="E = Emax · C / (EC50 + C)   ·   E = Emax · C^h / (EC50^h + C^h)"
          >
            <p className="mt-1 text-[11px] text-muted-foreground">
              Emerald = sigmoid Hill (h = {hill.toFixed(1)}); teal dashed = plain Emax (h = 1).
              At C = EC50 the effect is Emax/2 for both models.
            </p>
          </ResultBlock>
        </div>

        <div className="min-w-0">
          <div className="h-[300px] w-full" role="img" aria-label="Effect versus concentration curves for plain Emax and sigmoid Hill models">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
                <XAxis
                  dataKey="conc"
                  type="number"
                  scale="log"
                  domain={[0.01, 100]}
                  allowDataOverflow
                  ticks={[0.01, 0.1, 1, 10, 100]}
                  tickFormatter={fmtTick}
                  tick={CHART.axisTick}
                  tickLine={false}
                  axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                  label={{ value: "concentration (log scale)", position: "insideBottomRight", offset: -2, fill: (typeof window === "undefined" ? "#78716c" : getComputedStyle(document.documentElement).getPropertyValue("--muted-foreground").trim() || "#78716c"), fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  domain={[0, 105]}
                  width={40}
                  tick={CHART.axisTick}
                  tickFormatter={fmtTick}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={CHART.tooltipStyle}
                  formatter={(value) => fmtNum(Number(value), 4)}
                  labelFormatter={(label) => `C = ${fmtNum(Number(label), 3)}`}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: (typeof window === "undefined" ? "#78716c" : getComputedStyle(document.documentElement).getPropertyValue("--muted-foreground").trim() || "#78716c") }} />
                <ReferenceLine
                  x={ec50}
                  stroke={CHART.amber}
                  strokeDasharray="4 3"
                  label={{ value: "EC50", fill: CHART.amber, fontSize: 11, position: "top" }}
                />
                <Line
                  type="linear"
                  dataKey="plain"
                  name="Emax (h = 1)"
                  stroke={CHART.teal}
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="linear"
                  dataKey="sigmoid"
                  name={`Hill (h = ${hill.toFixed(1)})`}
                  stroke={CHART.emerald}
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-center text-[11px] text-muted-foreground">
            Concentrations sampled 0.01 → 100 (80 log-spaced points).
          </p>
        </div>
      </div>
    </CalcCard>
  );
}

/* ------------------------------------------------------------------ */
/* 8. Dimension-aware unit converter for this web-console demo        */
/* ------------------------------------------------------------------ */

const UNIT_DIMENSIONS: Array<{ value: Dimension; label: string; preset: [string, string] }> = [
  { value: "mass", label: "Mass (kg … pg)", preset: ["mg", "g"] },
  { value: "volume", label: "Volume (L … nL)", preset: ["mL", "L"] },
  { value: "amount", label: "Amount (mol … nmol)", preset: ["mmol", "mol"] },
  { value: "concentration", label: "Concentration (M … pM)", preset: ["uM", "nM"] },
  { value: "time", label: "Time (s … day)", preset: ["h", "min"] },
];

function UnitConverterCard() {
  const [dimension, setDimension] = useState<Dimension>("mass");
  const [value, setValue] = useState("500");
  const [fromUnit, setFromUnit] = useState("mg");
  const [toUnit, setToUnit] = useState("g");

  const units = unitsForDimension(dimension);

  const parsed = parseNonNegativeNumber(value);
  let result: number | null = null;
  let error: string | null =
    parsed === null ? (value.length === 0 ? "Enter a number" : "Must be ≥ 0") : null;
  if (parsed !== null) {
    try {
      result = convertUnits(parsed, fromUnit, toUnit);
    } catch (err) {
      // unreachable via the closed selects (same dimension, known
      // units) — kept honest anyway: the twin's contract IS to throw.
      error = err instanceof UnitConversionError ? err.message : String(err);
    }
  }
  const factor =
    UNIT_SCALES[dimension][fromUnit] !== undefined &&
    UNIT_SCALES[dimension][toUnit] !== undefined
      ? UNIT_SCALES[dimension][fromUnit] / UNIT_SCALES[dimension][toUnit]
      : null;

  const onDimensionChange = (next: string) => {
    const entry = UNIT_DIMENSIONS.find((d) => d.value === next);
    if (!entry) return;
    setDimension(entry.value);
    setFromUnit(entry.preset[0]);
    setToUnit(entry.preset[1]);
  };

  return (
    <CalcCard
      icon={ArrowLeftRight}
      title="Unit converter"
      description="Dimension-aware conversion — mass, volume, amount, concentration, time"
    >
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="uc-dimension" className="text-xs font-medium">
            Dimension
          </Label>
          <Select value={dimension} onValueChange={onDimensionChange}>
            <SelectTrigger id="uc-dimension" className="h-10">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {UNIT_DIMENSIONS.map((d) => (
                <SelectItem key={d.value} value={d.value}>
                  {d.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <NumberField
          id="uc-value"
          label="Value"
          unit={fromUnit}
          value={value}
          onChange={setValue}
          error={error}
          placeholder="500"
        />
        <div className="space-y-1.5">
          <Label htmlFor="uc-from" className="text-xs font-medium">
            From
          </Label>
          <Select value={fromUnit} onValueChange={setFromUnit}>
            <SelectTrigger id="uc-from" className="h-10 font-mono">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {units.map((unit) => (
                <SelectItem key={unit} value={unit} className="font-mono">
                  {unit}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="uc-to" className="text-xs font-medium">
            To
          </Label>
          <Select value={toUnit} onValueChange={setToUnit}>
            <SelectTrigger id="uc-to" className="h-10 font-mono">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {units.map((unit) => (
                <SelectItem key={unit} value={unit} className="font-mono">
                  {unit}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <ResultBlock
        label="Result"
        value={result !== null ? fmtNum(result, 6) : null}
        unit={toUnit}
        copyText={
          result !== null && parsed !== null
            ? `${parsed} ${fromUnit} = ${fmtNum(result, 6)} ${toUnit}`
            : null
        }
        formula={
          factor !== null
            ? `${dimension} · factor ×${fmtNum(factor, 6)} (dimension-checked — cross-dimension pairs are refused)`
            : undefined
        }
      />
      <p className="text-xs text-muted-foreground">
        Units in this dimension: {units.join(", ")}. The conversion runs through
        the same single front-door as{" "}
        <code className="rounded bg-muted px-1 font-mono text-xs">pharmaml.pharma.UnitConverter</code>{" "}
        — the same-dimension conversion matrix is exact-tested against the library in
        <code className="rounded bg-muted px-1 font-mono text-xs">TestUnitConverterParity</code>.
      </p>
    </CalcCard>
  );
}

export function PharmaCalcs() {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-amber-500/25 bg-amber-500/5 px-4 py-3 text-sm text-muted-foreground">
        <strong className="font-medium text-foreground">Research &amp; education only.</strong>{" "}
        These calculators mirror <code className="rounded bg-muted px-1 font-mono text-xs">pharmaml.pharma</code>{" "}
        and must not be used for clinical dosing decisions.
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <CrclCard />
        <BsaCard />
        <PediatricDoseCard />
        <LoadingMaintenanceCard />
        <HalfLifeCard />
        <AucCard />
        <PdCard />
        <UnitConverterCard />
      </div>
    </div>
  );
}
