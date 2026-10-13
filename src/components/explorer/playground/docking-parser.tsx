"use client";

import { useMemo, useState } from "react";
import {
  Boxes,
  CircleCheck,
  Loader2,
  ScanSearch,
  Sigma,
  Star,
  TriangleAlert,
  TrendingDown,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { Textarea } from "@/components/ui/textarea";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CHART } from "@/components/explorer/playground/calc-ui";
import { StatChip } from "@/components/explorer/stat-chip";
import { CSV_SAMPLE, DLG_SAMPLE, VINA_SAMPLE } from "@/lib/docking-samples";
import { fmtNum, parsePositiveNumber } from "@/lib/format";
import type { DockingFormat, DockingParseResponse, DockingPose } from "@/lib/toolkit-types";
import { toast } from "sonner";

const FORMATS: Array<{ value: DockingFormat; label: string; sample: string }> = [
  { value: "vina", label: "Vina output", sample: VINA_SAMPLE },
  { value: "dlg", label: "AutoDock4 DLG cluster table", sample: DLG_SAMPLE },
  { value: "csv", label: "Generic CSV", sample: CSV_SAMPLE },
];

function PoseValue({ value, sig = 3 }: { value: number | undefined; sig?: number }) {
  if (value === undefined || !Number.isFinite(value)) return <span className="text-muted-foreground">—</span>;
  return <span className="tabular-nums">{fmtNum(value, sig)}</span>;
}

export function DockingParser() {
  const [format, setFormat] = useState<DockingFormat>("vina");
  const [content, setContent] = useState("");
  const [heavyAtoms, setHeavyAtoms] = useState("32");

  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DockingParseResponse | null>(null);

  const ha = parsePositiveNumber(heavyAtoms);
  const haValid = ha !== null && Number.isInteger(ha);

  const extraColumns = useMemo(() => {
    if (!result || result.format !== "csv") return [] as string[];
    const keys = new Set<string>();
    for (const p of result.poses) {
      for (const k of Object.keys(p.extraScores ?? {})) keys.add(k);
    }
    return Array.from(keys);
  }, [result]);

  function loadSample(f: DockingFormat) {
    const spec = FORMATS.find((x) => x.value === f);
    if (!spec) return;
    setFormat(f);
    setContent(spec.sample);
    setStatus("idle");
    setResult(null);
    setError(null);
    toast.success(`${spec.label} sample loaded`);
  }

  async function parse() {
    if (!haValid) {
      setError("Heavy atoms must be a positive integer (1 – 500).");
      toast.error("Invalid heavy-atom count");
      return;
    }
    if (content.trim().length === 0) {
      setError("Paste docking output first — or load one of the samples.");
      toast.error("Nothing to parse");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const res = await fetch("/api/docking/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          format,
          content,
          heavyAtoms: Math.round(ha as number),
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
      setResult(data as DockingParseResponse);
      setStatus("done");
      toast.success(`Parsed ${((data as DockingParseResponse).poses ?? []).length} poses`);
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Parsing request failed.";
      setError(message);
      toast.error("Parsing failed", { description: message });
    }
  }

  const running = status === "running";
  const formatLabel = FORMATS.find((f) => f.value === format)?.label ?? format;

  return (
    <div className="space-y-4">
      {/* ------------------------------ Input ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <ScanSearch className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Docking result parser</CardTitle>
              <CardDescription className="text-xs">
                Parses Vina tables, AutoDock4 DLG cluster tables and generic CSV — mirroring{" "}
                <code className="font-mono">pharmaml.docking.parsing</code>
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pb-4">
          <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_180px]">
            <div className="space-y-1.5">
              <Label htmlFor="dock-format" className="text-xs font-medium">
                Input format
              </Label>
              <Select value={format} onValueChange={(v) => setFormat(v as DockingFormat)}>
                <SelectTrigger id="dock-format" className="h-10" aria-label="Docking output format">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {FORMATS.map((f) => (
                    <SelectItem key={f.value} value={f.value}>
                      {f.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="dock-ha" className="text-xs font-medium">
                Heavy atoms (for LE)
              </Label>
              <Input
                id="dock-ha"
                inputMode="numeric"
                value={heavyAtoms}
                onChange={(e) => setHeavyAtoms(e.target.value)}
                aria-invalid={haValid ? undefined : true}
                className={`h-10 font-mono text-sm tabular-nums ${haValid ? "" : "border-destructive/60"}`}
              />
              {!haValid ? (
                <p className="text-[11px] text-destructive">Positive integer.</p>
              ) : null}
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Label htmlFor="dock-input" className="text-xs font-medium">
                Paste {formatLabel} output
              </Label>
              <div className="flex flex-wrap gap-1.5">
                {FORMATS.map((f) => (
                  <Button
                    key={f.value}
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 px-2.5 text-[11px]"
                    onClick={() => loadSample(f.value)}
                  >
                    <Boxes className="size-3" aria-hidden />
                    {f.value === "vina" ? "Vina" : f.value === "dlg" ? "DLG" : "CSV"} sample
                  </Button>
                ))}
              </div>
            </div>
            <Textarea
              id="dock-input"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder={`Paste ${formatLabel} text here — or load a sample above…`}
              className="thin-scrollbar min-h-[160px] max-h-72 resize-y font-mono text-xs leading-relaxed"
              aria-label="Docking output text"
            />
          </div>

          <Button size="lg" className="h-11 gap-2" onClick={() => void parse()} disabled={running}>
            {running ? (
              <>
                <Loader2 className="size-4 animate-spin" aria-hidden />
                Parsing…
              </>
            ) : (
              <>
                <ScanSearch className="size-4" aria-hidden />
                Parse results
              </>
            )}
          </Button>

          {error ? (
            <Alert variant="destructive">
              <TriangleAlert className="size-4" aria-hidden />
              <AlertTitle>Could not parse input</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}
        </CardContent>
      </Card>

      {/* ------------------------------ Results ------------------------------ */}
      {result && !running ? (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatChip icon={Star} value={fmtNum(result.stats.best, 3)} label="best affinity (kcal/mol)" />
            <StatChip icon={Sigma} value={fmtNum(result.stats.mean, 3)} label="mean affinity" />
            <StatChip icon={TrendingDown} value={fmtNum(result.stats.worst, 3)} label="worst affinity" />
            <StatChip icon={CircleCheck} value={String(result.stats.n)} label="poses parsed" />
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            {/* Poses table */}
            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <CardTitle className="text-sm">Ranked poses</CardTitle>
                <CardDescription className="text-xs">
                  Lower (more negative) affinity = better. LE = ligand efficiency,{" "}
                  {haValid ? `${heavyAtoms} heavy atoms` : "?"} (kcal/mol/HA).
                </CardDescription>
              </CardHeader>
              <div className="thin-scrollbar max-h-96 overflow-auto">
                <Table className="data-table">
                  <TableHeader className="sticky top-0 z-10 bg-muted/80 backdrop-blur">
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="h-9">#</TableHead>
                      {result.format === "vina" ? (
                        <>
                          <TableHead className="h-9 text-right">Affinity</TableHead>
                          <TableHead className="h-9 text-right">RMSD l.b.</TableHead>
                          <TableHead className="h-9 text-right">RMSD u.b.</TableHead>
                        </>
                      ) : null}
                      {result.format === "dlg" ? (
                        <>
                          <TableHead className="h-9 text-right">Lowest E</TableHead>
                          <TableHead className="h-9 text-right">Run</TableHead>
                          <TableHead className="h-9 text-right">Mean E</TableHead>
                          <TableHead className="h-9 text-right">Cluster</TableHead>
                        </>
                      ) : null}
                      {result.format === "csv" ? (
                        <>
                          <TableHead className="h-9">Ligand</TableHead>
                          <TableHead className="h-9 text-right">Pose</TableHead>
                          <TableHead className="h-9 text-right">Affinity</TableHead>
                          {extraColumns.map((col) => (
                            <TableHead key={col} className="h-9 text-right">
                              {col}
                            </TableHead>
                          ))}
                          {result.consensus?.applied ? (
                            <>
                              <TableHead className="h-9 text-right">z-score</TableHead>
                              <TableHead className="h-9 text-right">Consensus</TableHead>
                            </>
                          ) : null}
                        </>
                      ) : null}
                      <TableHead className="h-9 text-right">LE</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.poses.map((pose: DockingPose) => (
                      <TableRow
                        key={`${pose.rank}-${pose.mode ?? ""}`}
                        style={
                          pose.rank === 1
                            ? { background: "color-mix(in oklch, var(--primary) 8%, transparent)" }
                            : undefined
                        }
                      >
                        <TableCell className="py-2 font-mono text-xs tabular-nums">
                          {pose.rank === 1 ? (
                            <Badge className="bg-primary/15 text-primary hover:bg-primary/15" variant="secondary">
                              best
                            </Badge>
                          ) : (
                            pose.rank
                          )}
                        </TableCell>
                        {result.format === "vina" ? (
                          <>
                            <TableCell className="py-2 text-right font-mono text-xs">
                              <PoseValue value={pose.affinity} />
                            </TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs text-muted-foreground">
                              <PoseValue value={pose.rmsdLb} />
                            </TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs text-muted-foreground">
                              <PoseValue value={pose.rmsdUb} />
                            </TableCell>
                          </>
                        ) : null}
                        {result.format === "dlg" ? (
                          <>
                            <TableCell className="py-2 text-right font-mono text-xs">
                              <PoseValue value={pose.affinity} />
                            </TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs text-muted-foreground">
                              <PoseValue value={pose.run} />
                            </TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs text-muted-foreground">
                              <PoseValue value={pose.meanEnergy} />
                            </TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs text-muted-foreground">
                              <PoseValue value={pose.clusterSize} sig={2} />
                            </TableCell>
                          </>
                        ) : null}
                        {result.format === "csv" ? (
                          <>
                            <TableCell className="py-2 font-mono text-xs">{pose.ligand ?? "—"}</TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                              {pose.mode ?? "—"}
                            </TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs">
                              <PoseValue value={pose.affinity} />
                            </TableCell>
                            {extraColumns.map((col) => (
                              <TableCell key={col} className="py-2 text-right font-mono text-xs text-muted-foreground">
                                <PoseValue value={pose.extraScores?.[col]} />
                              </TableCell>
                            ))}
                            {result.consensus?.applied ? (
                              <>
                                <TableCell className="py-2 text-right font-mono text-xs">
                                  <PoseValue value={pose.zScore} sig={4} />
                                </TableCell>
                                <TableCell className="py-2 text-right font-mono text-xs font-semibold text-primary">
                                  {pose.consensusRank ?? "—"}
                                </TableCell>
                              </>
                            ) : null}
                          </>
                        ) : null}
                        <TableCell className="py-2 text-right font-mono text-xs text-muted-foreground">
                          <PoseValue value={pose.ligandEfficiency} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </Card>

            {/* Chart + consensus */}
            <div className="space-y-4">
              <Card className="overflow-hidden py-0 shadow-sm">
                <CardHeader className="border-b bg-muted/30 py-3">
                  <CardTitle className="text-sm">Binding affinities</CardTitle>
                  <CardDescription className="text-xs">
                    More negative = stronger binding. Best pose in emerald.
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-3 sm:p-4">
                  <div
                    className="h-[260px] w-full"
                    role="img"
                    aria-label="Bar chart of docking affinities per pose"
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={result.chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                        <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} vertical={false} />
                        <XAxis
                          dataKey="mode"
                          tick={CHART.axisTick}
                          tickLine={false}
                          axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                          interval={0}
                          angle={result.chartData.length > 6 ? -25 : 0}
                          textAnchor={result.chartData.length > 6 ? "end" : "middle"}
                          height={result.chartData.length > 6 ? 44 : 24}
                        />
                        <YAxis
                          width={44}
                          tick={CHART.axisTick}
                          tickFormatter={(v: number) => fmtNum(v, 2)}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          cursor={{ fill: "rgba(120,113,108,0.08)" }}
                          contentStyle={CHART.tooltipStyle}
                          formatter={(value) => [fmtNum(Number(value), 3), "affinity (kcal/mol)"]}
                        />
                        <Bar dataKey="affinity" radius={[0, 0, 4, 4]} isAnimationActive={false}>
                          {result.chartData.map((d, i) => (
                            <Cell key={i} fill={d.best ? CHART.emerald : CHART.stone} fillOpacity={d.best ? 1 : 0.55} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {result.consensus ? (
                <Card
                  className={`border py-0 shadow-sm ${
                    result.consensus.applied ? "border-primary/30" : "border-dashed"
                  }`}
                >
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <Sigma className="size-4 text-primary" aria-hidden />
                      Consensus z-score rescoring
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pb-4">
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {result.consensus.note}
                    </p>
                    {result.consensus.applied ? (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {result.consensus.columns.map((c) => (
                          <Badge key={c} variant="secondary" className="font-mono text-[10px]">
                            {c}
                          </Badge>
                        ))}
                      </div>
                    ) : null}
                  </CardContent>
                </Card>
              ) : null}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
