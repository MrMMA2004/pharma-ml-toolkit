"use client";

import { useMemo } from "react";
import {
  Info,
  Pill,
  Plus,
  RotateCcw,
  Sigma,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import {
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { nextRowId, useToolkitStore, type NcaRow } from "@/lib/store";
import { fmtNum, fmtTick, parseNonNegativeNumber, parsePositiveNumber } from "@/lib/format";
import { CHART, DedupeLegend, NumberField } from "@/components/explorer/playground/calc-ui";
import { computeNca, NCA_PRESETS, type NcaPoint } from "@/lib/pk-pd";

/** Hard cap on editable rows (store-side cap is the same). */
const ROWS_CAP = 40;

interface ChartDatum {
  t: number;
  /** Observed concentration (undefined beyond the last sample). */
  c?: number;
  /** Fitted terminal-tail concentration (only within the tail span). */
  fit?: number;
  /** Terminal-fit point marker. */
  terminal?: boolean;
}

/**
 * NCA (observed data) panel: an editable time/concentration table plus a
 * client-side non-compartmental analysis mirroring pharmaml.pharma —
 * trapezoid AUC, log-linear terminal fit (r² reported), extrapolated
 * AUC∞, and CL/Vz from a supplied dose. Cross-validated against the
 * Python package (see the pinned numbers in src/lib/pk-pd.ts).
 */
export function NcaPanel() {
  const rows = useToolkitStore((s) => s.ncaState.rows);
  const dose = useToolkitStore((s) => s.ncaState.dose);
  const f = useToolkitStore((s) => s.ncaState.f);
  const terminalN = useToolkitStore((s) => s.ncaState.terminalN);
  const setNcaState = useToolkitStore((s) => s.setNcaState);
  const resetNcaState = useToolkitStore((s) => s.resetNcaState);

  /* ---- Parse the editable rows (invalid cells are skipped + counted). ---- */
  const parsed = useMemo(() => {
    const points: NcaPoint[] = [];
    let invalid = 0;
    for (const row of rows) {
      const t = Number.parseFloat(row.t);
      const c = Number.parseFloat(row.c);
      const bad =
        row.t.trim() === "" ||
        row.c.trim() === "" ||
        !Number.isFinite(t) ||
        !Number.isFinite(c) ||
        t < 0 ||
        c < 0;
      if (bad) invalid += 1;
      else points.push({ t, c });
    }
    return { points, invalid };
  }, [rows]);

  const doseNum = dose.trim() === "" ? null : parsePositiveNumber(dose);
  const fNum = parseNonNegativeNumber(f);
  const fValid = fNum !== null && fNum > 0 && fNum <= 1;
  const doseInvalid = dose.trim() !== "" && doseNum === null;

  const result = useMemo(
    () =>
      computeNca(parsed.points, {
        terminalN,
        dose: doseNum !== null && fValid ? doseNum : null,
        f: fValid ? (fNum as number) : 1,
      }),
    [parsed, terminalN, doseNum, fValid, fNum],
  );

  /* ---- Row mutations (kept in the store so edits survive tab switches). ---- */
  const updateRow = (id: number, field: "t" | "c", value: string) => {
    setNcaState({
      rows: rows.map((r) => (r.id === id ? { ...r, [field]: value } : r)),
    });
  };
  const removeRow = (id: number) => {
    if (rows.length <= 2) return; // keep at least two rows for a valid series
    setNcaState({ rows: rows.filter((r) => r.id !== id) });
  };
  const addRow = () => {
    if (rows.length >= ROWS_CAP) return;
    const id = nextRowId(rows);
    const last = rows[rows.length - 1];
    const nextT =
      last && Number.isFinite(Number.parseFloat(last.t))
        ? String(Number(Number.parseFloat(last.t) + 1).toFixed(2))
        : "0";
    setNcaState({ rows: [...rows, { id, t: nextT, c: "" }] });
  };
  const loadPreset = (presetId: string) => {
    const preset = NCA_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setNcaState({
      rows: preset.rows.map(([t, c], id) => ({ id, t, c })),
      dose: preset.dose,
      f: preset.f,
    });
  };

  /* ---- Chart data: observations + the fitted exponential tail. ---- */
  const chartData = useMemo<ChartDatum[]>((): ChartDatum[] => {
    const data: ChartDatum[] = result.points.map((p, i) => ({
      t: p.t,
      c: p.c,
      terminal: result.terminalIdx.includes(i),
    }));
    const { ke, a0, from } = result.fit;
    if (ke !== null && a0 !== null && from !== null && result.tailEnd !== null) {
      const span = result.tailEnd - from;
      const steps = Math.max(2, Math.min(48, Math.ceil(span * 2)));
      for (let i = 0; i <= steps; i++) {
        const t = from + (span * i) / steps;
        const fit = a0 * Math.exp(-ke * t);
        data.push({ t, fit: Number(fit.toFixed(5)) });
      }
    }
    return data.sort((a, b) => a.t - b.t);
  }, [result]);

  const hasFit = result.fit.ke !== null && result.tailEnd !== null;

  /* ---- Metric cards (AUC / Cmax / Tmax / ke / t½ / r² / AUC∞ / CL / Vz). ---- */
  const metrics: Array<{ label: string; value: string; unit: string }> = [
    {
      label: "AUC₀₋last",
      value: result.aucLast !== null ? fmtNum(result.aucLast, 4) : "—",
      unit: "mg·h/L",
    },
    {
      label: "Cmax",
      value: result.cMax !== null ? fmtNum(result.cMax, 3) : "—",
      unit: "mg/L",
    },
    { label: "Tmax", value: result.tMax !== null ? fmtNum(result.tMax, 3) : "—", unit: "h" },
    {
      label: "λz (terminal)",
      value: result.fit.ke !== null ? fmtNum(result.fit.ke, 3) : "—",
      unit: "h⁻¹",
    },
    {
      label: "t½ terminal",
      value: result.fit.halfLife !== null ? fmtNum(result.fit.halfLife, 3) : "—",
      unit: "h",
    },
    {
      label: "fit r²",
      value: result.fit.r2 !== null ? result.fit.r2.toFixed(4) : "—",
      unit: "",
    },
    {
      label: "AUC₀₋∞",
      value: result.aucInf !== null ? fmtNum(result.aucInf, 4) : "—",
      unit: "mg·h/L",
    },
    {
      label: "% extrapolated",
      value: result.pctExtrap !== null ? fmtNum(result.pctExtrap, 3) : "—",
      unit: "%",
    },
    {
      label: "CL (F·D/AUC∞)",
      value: result.cl !== null ? fmtNum(result.cl, 4) : "—",
      unit: "L/h",
    },
    { label: "Vz = CL/λz", value: result.vz !== null ? fmtNum(result.vz, 4) : "—", unit: "L" },
  ];

  const activePresetId = useMemo(() => {
    const sig = rows.map((r) => `${r.t}|${r.c}`).join(";");
    return NCA_PRESETS.find(
      (p) =>
        p.rows.map(([t, c]) => `${t}|${c}`).join(";") === sig &&
        p.dose === dose &&
        p.f === f,
    )?.id;
  }, [rows, dose, f]);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
      {/* ------------------------------ Observations editor ------------------------------ */}
      <Card className="h-fit py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Sigma className="size-4 text-primary" aria-hidden />
            Observed data
          </CardTitle>
          <CardDescription className="text-xs">
            Editable time–concentration table · NCA computed live in your browser
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 py-4">
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Preset datasets</Label>
            <div className="grid grid-cols-3 gap-2">
              {NCA_PRESETS.map((p) => (
                <Button
                  key={p.id}
                  type="button"
                  variant={activePresetId === p.id ? "default" : "outline"}
                  size="sm"
                  className="h-9 gap-1 px-2 text-xs"
                  onClick={() => loadPreset(p.id)}
                  aria-label={`Load the ${p.label} preset dataset`}
                  title={p.description}
                >
                  <Pill className="size-3" aria-hidden />
                  {p.label}
                </Button>
              ))}
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">
              {NCA_PRESETS.find((p) => p.id === activePresetId)?.description ??
                "Custom data — edit any cell, add or remove rows."}
            </p>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <Label htmlFor="nca-table" className="text-xs font-medium">
                Observations
              </Label>
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {rows.length} rows
              </span>
            </div>
            <div
              id="nca-table"
              className="thin-scrollbar max-h-72 overflow-auto rounded-lg border"
              role="group"
              aria-label="Editable concentration-time observations"
            >
              <Table className="data-table min-w-[300px]">
                <TableHeader className="sticky top-0 z-10 bg-muted/80 backdrop-blur">
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="h-9 w-8 text-right">#</TableHead>
                    <TableHead className="h-9">time (h)</TableHead>
                    <TableHead className="h-9">C (mg/L)</TableHead>
                    <TableHead className="h-9 w-12">
                      <span className="sr-only">Remove row</span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((row, i) => {
                    const badRow =
                      row.t.trim() === "" ||
                      row.c.trim() === "" ||
                      !Number.isFinite(Number.parseFloat(row.t)) ||
                      !Number.isFinite(Number.parseFloat(row.c)) ||
                      Number.parseFloat(row.t) < 0 ||
                      Number.parseFloat(row.c) < 0;
                    return (
                      <TableRow key={row.id} className="hover:bg-transparent">
                        <TableCell className="p-1 text-right font-mono text-xs tabular-nums text-muted-foreground">
                          {i + 1}
                        </TableCell>
                        <TableCell className="p-1">
                          <Input
                            aria-label={`Row ${i + 1} time in hours`}
                            inputMode="decimal"
                            autoComplete="off"
                            value={row.t}
                            onChange={(e) => updateRow(row.id, "t", e.target.value)}
                            aria-invalid={badRow ? true : undefined}
                            className={cn(
                              "h-8 border-transparent bg-transparent px-2 font-mono text-xs tabular-nums focus-visible:bg-background",
                              badRow && "border-destructive/60",
                            )}
                          />
                        </TableCell>
                        <TableCell className="p-1">
                          <Input
                            aria-label={`Row ${i + 1} concentration in mg per litre`}
                            inputMode="decimal"
                            autoComplete="off"
                            value={row.c}
                            onChange={(e) => updateRow(row.id, "c", e.target.value)}
                            aria-invalid={badRow ? true : undefined}
                            className={cn(
                              "h-8 border-transparent bg-transparent px-2 font-mono text-xs tabular-nums focus-visible:bg-background",
                              badRow && "border-destructive/60",
                            )}
                          />
                        </TableCell>
                        <TableCell className="p-1">
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className="size-8"
                            onClick={() => removeRow(row.id)}
                            disabled={rows.length <= 2}
                            aria-label={`Remove row ${i + 1}`}
                            title={rows.length <= 2 ? "Keep at least two rows" : "Remove row"}
                          >
                            <Trash2 className="size-3.5" aria-hidden />
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
            <div className="flex items-center justify-between gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 gap-1 text-xs"
                onClick={addRow}
                disabled={rows.length >= ROWS_CAP}
                aria-label="Add an observation row"
              >
                <Plus className="size-3.5" aria-hidden />
                Add row
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 gap-1 text-xs"
                onClick={resetNcaState}
                aria-label="Reset the observations to the default preset"
              >
                <RotateCcw className="size-3.5" aria-hidden />
                Reset
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <NumberField
              id="nca-dose"
              label="Dose"
              unit="mg"
              optional
              value={dose}
              onChange={(v) => setNcaState({ dose: v })}
              error={doseInvalid ? "Positive number (or leave empty)." : null}
              placeholder="250"
            />
            <NumberField
              id="nca-f"
              label="Bioavailability F"
              unit="0–1"
              optional
              value={f}
              onChange={(v) => setNcaState({ f: v })}
              error={fValid || f.trim() === "" ? null : "Between 0 and 1."}
              placeholder="1"
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="nca-terminal-n" className="text-xs font-medium">
              Terminal points (log-linear fit)
            </Label>
            <Select
              value={String(terminalN)}
              onValueChange={(v) => setNcaState({ terminalN: Number.parseInt(v, 10) })}
            >
              <SelectTrigger id="nca-terminal-n" className="h-10" aria-label="Number of terminal points for the fit">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[2, 3, 4, 5, 6].map((n) => (
                  <SelectItem key={n} value={String(n)}>
                    last {n} point{n > 1 ? "s" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11px] leading-snug text-muted-foreground">
              Positive samples after Tmax are regressed as ln C vs t; ke = −slope, t½ = ln2/ke.
              N = 2 reproduces pharmaml&apos;s <span className="font-mono">nca_summary</span>.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* ------------------------------ Chart + metrics + warnings ------------------------------ */}
      <div className="space-y-4">
        <Card className="py-0 shadow-sm">
          <CardHeader className="border-b bg-muted/30 py-3">
            <CardTitle className="text-base">Observed profile &amp; terminal fit</CardTitle>
            <CardDescription className="text-xs">
              {hasFit
                ? `Amber points: the ${result.fit.n}-point terminal fit · dashed line: C(t) = ${fmtNum(result.fit.a0, 3)}·e^(−${fmtNum(result.fit.ke, 3)}·t), extended to 1 % of Cmax`
                : "Observations — no valid terminal fit (see the notes below)"}
            </CardDescription>
          </CardHeader>
          <CardContent className="py-4">
            <div
              className="h-[280px] w-full sm:h-[320px]"
              role="img"
              aria-label="Scatter plot of observed concentrations versus time with the fitted terminal exponential tail"
            >
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
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
                      fill: (typeof window === "undefined" ? "#78716c" : getComputedStyle(document.documentElement).getPropertyValue("--muted-foreground").trim() || "#78716c"),
                      fontSize: 11,
                    }}
                    tickFormatter={fmtTick}
                    minTickGap={32}
                  />
                  <YAxis
                    dataKey="c"
                    tick={CHART.axisTick}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    label={{
                      value: "C (mg/L)",
                      angle: -90,
                      position: "insideLeft",
                      fill: (typeof window === "undefined" ? "#78716c" : getComputedStyle(document.documentElement).getPropertyValue("--muted-foreground").trim() || "#78716c"),
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    contentStyle={CHART.tooltipStyle}
                    formatter={(value: number | string, name: string) => [
                      `${Number(value).toFixed(3)} mg/L`,
                      name,
                    ]}
                    labelFormatter={(label: number | string) => `t = ${Number(label).toFixed(2)} h`}
                  />
                  <Legend content={<DedupeLegend />} />
                  {result.fit.from !== null && result.tailEnd !== null ? (
                    <ReferenceLine
                      x={result.fit.from}
                      stroke={CHART.stone}
                      strokeOpacity={0.4}
                      strokeDasharray="3 3"
                      label={{
                        value: "terminal",
                        fill: (typeof window === "undefined" ? "#78716c" : getComputedStyle(document.documentElement).getPropertyValue("--muted-foreground").trim() || "#78716c"),
                        fontSize: 10,
                        position: "insideTopLeft",
                      }}
                    />
                  ) : null}
                  <Line
                    name="observations"
                    type="monotone"
                    dataKey="c"
                    stroke={CHART.emerald}
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  {hasFit ? (
                    <Line
                      name="terminal fit"
                      type="monotone"
                      dataKey="fit"
                      stroke={CHART.amber}
                      strokeWidth={1.5}
                      strokeDasharray="6 4"
                      dot={false}
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                  ) : null}
                  <Scatter
                    name="observations"
                    dataKey="c"
                    fill={CHART.emerald}
                    fillOpacity={0.9}
                    isAnimationActive={false}
                  >
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={d.terminal ? CHART.amber : CHART.emerald} />
                    ))}
                  </Scatter>
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {metrics.map((m) => (
            <div key={m.label} className="readout-card min-w-0 rounded-lg border bg-card p-3 pl-4 shadow-sm">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                {m.label}
              </p>
              <p className="mt-0.5 font-mono text-lg font-semibold tabular-nums leading-tight">
                {m.value}
                {m.unit ? (
                  <span className="ml-1 text-xs font-normal text-muted-foreground">{m.unit}</span>
                ) : null}
              </p>
            </div>
          ))}
        </div>

        {/* Input-level advisories (invalid rows / dose / F). */}
        {parsed.invalid > 0 ? (
          <div
            className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-xs leading-relaxed text-destructive"
            role="alert"
          >
            <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
            <p>
              <strong>{parsed.invalid} row{parsed.invalid === 1 ? "" : "s"} skipped:</strong>{" "}
              time and concentration must be non-negative numbers (empty cells count as invalid).
              Fix or delete the highlighted rows to include them in the analysis.
            </p>
          </div>
        ) : null}
        {doseInvalid || (!fValid && f.trim() !== "") ? (
          <div
            className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-xs leading-relaxed text-destructive"
            role="alert"
          >
            <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
            <p>
              <strong>Invalid dose inputs:</strong>{" "}
              {[
                doseInvalid ? "dose must be a positive number" : null,
                !fValid && f.trim() !== "" ? "F must lie between 0 and 1" : null,
              ]
                .filter(Boolean)
                .join(" · ")}
              . CL and Vz are hidden until the dose inputs parse.
            </p>
          </div>
        ) : null}

        {/* NCA warnings (amber notes / destructive alerts). */}
        {result.warnings.map((w, i) => (
          <div
            key={i}
            className={cn(
              "flex items-start gap-2 rounded-lg border p-3 text-xs leading-relaxed",
              w.level === "amber"
                ? "border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200"
                : "border-destructive/40 bg-destructive/5 text-destructive",
            )}
            role={w.level === "amber" ? "note" : "alert"}
          >
            <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
            <p>{w.text}</p>
          </div>
        ))}

        {doseNum === null && !doseInvalid ? (
          <div
            className="flex items-start gap-2 rounded-lg border bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground"
            role="note"
          >
            <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
            <p>
              <strong>Tip:</strong> enter the administered dose (and F for oral) to also compute
              CL = F·Dose/AUC∞ and the terminal volume Vz = CL/λz — mirroring{" "}
              <span className="font-mono">pharmaml.pharma.clearance</span> /{" "}
              <span className="font-mono">volume_of_distribution</span>.
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}
