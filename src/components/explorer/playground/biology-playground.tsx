"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  Check,
  Dices,
  Dna,
  FlaskConical,
  Loader2,
  Microscope,
  Play,
  RotateCcw,
  TrendingDown,
  TrendingUp,
  TriangleAlert,
} from "lucide-react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
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
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CHART } from "@/components/explorer/playground/calc-ui";
import { fmtNum, fmtTick } from "@/lib/format";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import type {
  BiomarkerPoint,
  BiomarkerScatter,
  BiomarkersResponse,
  BiologyCompound,
} from "@/lib/toolkit-types";

/* ------------------------------------------------------------------ */
/* Compound panel                                                      */
/* ------------------------------------------------------------------ */

const COMPOUNDS: Array<{
  id: BiologyCompound;
  label: string;
  hint: string;
  egfr: boolean;
}> = [
  { id: "erlotinib", label: "Erlotinib", hint: "EGFR inhibitor — planted biomarker active", egfr: true },
  { id: "gefitinib", label: "Gefitinib", hint: "EGFR inhibitor — planted biomarker active", egfr: true },
  {
    id: "cisplatin",
    label: "Cisplatin",
    hint: "tissue shifts: blood −0.6 · skin +0.4",
    egfr: false,
  },
  { id: "doxorubicin", label: "Doxorubicin", hint: "no tissue shift — plain baseline", egfr: false },
  {
    id: "paclitaxel",
    label: "Paclitaxel",
    hint: "tissue shifts: prostate −0.5 · breast −0.3",
    egfr: false,
  },
  {
    id: "5-fluorouracil",
    label: "5-Fluorouracil",
    hint: "tissue shift: colon −0.4",
    egfr: false,
  },
  { id: "gemcitabine", label: "Gemcitabine", hint: "tissue shift: pancreas −0.7", egfr: false },
  { id: "sorafenib", label: "Sorafenib", hint: "tissue shift: liver −0.5", egfr: false },
];

/** Panel-size options (canonical line slices, 6–24 like the Python CLI). */
const PANEL_SIZES: Array<{ value: string; label: string }> = [
  { value: "6", label: "6 lines — lung + breast" },
  { value: "9", label: "9 lines — + colon" },
  { value: "12", label: "12 lines — + blood & skin" },
  { value: "15", label: "15 lines — + pancreas & liver" },
  { value: "18", label: "18 lines — + ovary & prostate" },
  { value: "21", label: "21 lines — + brain (partial)" },
  { value: "24", label: "24 lines — full panel" },
];

/** Categorical tissue colors (10 hues, no blue/indigo anywhere). */
const TISSUE_COLORS: Record<string, string> = {
  lung: "#10b981",
  breast: "#ec4899",
  colon: "#f59e0b",
  blood: "#ef4444",
  skin: "#f97316",
  pancreas: "#84cc16",
  liver: "#a16207",
  ovary: "#d946ef",
  prostate: "#14b8a6",
  brain: "#8b5cf6",
};

const DEFAULTS = {
  compound: "erlotinib" as BiologyCompound,
  genes: 300,
  seed: "42",
  nCellLines: "24",
};

/** Chart datum: recharts x/y plus the payload fields the tooltip shows. */
interface ScatterDatum {
  x: number;
  y: number;
  cellLine: string;
  tissue: string;
  value: number;
  lnIc50: number;
  auc: number;
}

function toDatum(p: BiomarkerPoint): ScatterDatum {
  return {
    x: p.value,
    y: p.lnIc50,
    cellLine: p.cellLine,
    tissue: p.tissue,
    value: p.value,
    lnIc50: p.lnIc50,
    auc: p.auc,
  };
}

/* ------------------------------------------------------------------ */
/* Small presentational helpers                                        */
/* ------------------------------------------------------------------ */

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-xl font-semibold tabular-nums">{value}</p>
    </div>
  );
}

/** Custom scatter tooltip: cell line + tissue + both values. */
function ScatterTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: ScatterDatum }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0]?.payload;
  if (!p) return null;
  const color = TISSUE_COLORS[p.tissue] ?? CHART.stone;
  return (
    <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
      <p className="font-mono font-semibold">{p.cellLine}</p>
      <p className="mt-0.5 flex items-center gap-1.5 text-muted-foreground">
        <span className="size-2 rounded-full" style={{ backgroundColor: color }} aria-hidden />
        {p.tissue}
      </p>
      <p className="mt-1 font-mono tabular-nums">
        expression {fmtNum(p.value, 3)} · LN_IC50 {fmtNum(p.lnIc50, 3)}
      </p>
      <p className="font-mono text-[11px] tabular-nums text-muted-foreground">
        AUC {fmtNum(p.auc, 3)}
      </p>
    </div>
  );
}

/** rho cell coloring: negative = sensitizing (emerald), positive = resisting (rose). */
function RhoCell({ rho }: { rho: number }) {
  const negative = rho < 0;
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-xs tabular-nums">
      <span className={negative ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>
        {negative ? <TrendingDown className="size-3" aria-hidden /> : <TrendingUp className="size-3" aria-hidden />}
        {fmtNum(rho, 3)}
      </span>
      <Badge
        variant="secondary"
        className={cn(
          "h-4 gap-0.5 px-1 text-[9px] font-medium",
          negative
            ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
            : "bg-rose-500/10 text-rose-600 dark:text-rose-400"
        )}
      >
        {negative ? "sensitizing" : "resisting"}
      </Badge>
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Panel                                                               */
/* ------------------------------------------------------------------ */

export function BiologyPlayground() {
  /* Transient panel state — same convention as the docking parser and the
   * ligand reconciler (only the ML/PK setups persist in the store). */
  const [compound, setCompound] = useState<BiologyCompound>(DEFAULTS.compound);
  const [genes, setGenes] = useState(DEFAULTS.genes);
  const [seed, setSeed] = useState(DEFAULTS.seed);
  const [nCellLines, setNCellLines] = useState(DEFAULTS.nCellLines);
  const [result, setResult] = useState<BiomarkersResponse | null>(null);
  /* (100x-124) the MAD variability ranking toggle — the desktop's
   * "Rank variable genes (MAD)" button mirrored as a disclosure. */
  const [showMad, setShowMad] = useState(false);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  /** Which gene the scatter shows: the top hit, or EGFR specifically. */
  const [chartGene, setChartGene] = useState<"top" | "egfr">("top");

  const running = status === "running";
  const seedValid = /^\d{1,10}$/.test(seed.trim());

  /** Tissue-grouped series (stable references for recharts). */
  const series = useMemo(() => {
    if (!result) return null;
    const scatter: BiomarkerScatter =
      chartGene === "egfr" && result.egfrScatter ? result.egfrScatter : result.topHit;
    const byTissue = new Map<string, ScatterDatum[]>();
    for (const p of scatter.points) {
      const list = byTissue.get(p.tissue) ?? [];
      list.push(toDatum(p));
      byTissue.set(p.tissue, list);
    }
    return {
      scatter,
      groups: result.tissues
        .filter((t) => byTissue.has(t))
        .map((t) => ({ tissue: t, data: byTissue.get(t) ?? [] })),
    };
  }, [result, chartGene]);

  function randomizeSeed() {
    setSeed(String(Math.floor(Math.random() * 2_147_483_648)));
  }

  function resetSetup() {
    setCompound(DEFAULTS.compound);
    setGenes(DEFAULTS.genes);
    setSeed(DEFAULTS.seed);
    setNCellLines(DEFAULTS.nCellLines);
    setStatus("idle");
    setResult(null);
    setError(null);
    setChartGene("top");
  }

  async function discover() {
    if (!seedValid) {
      setError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const res = await fetch("/api/biology/biomarkers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          compound,
          genes,
          seed: Number.parseInt(seed, 10) || 0,
          nCellLines: Number.parseInt(nCellLines, 10) || 24,
        }),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? `${String((data as { error: unknown }).error)}${
                "detail" in data && Array.isArray((data as { detail: unknown }).detail)
                  ? ` ${((data as { detail: unknown[] }).detail as string[]).join("; ")}`
                  : ""
              }`
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const payload = data as BiomarkersResponse;
      setResult(payload);
      setStatus("done");
      setChartGene("top");
      toast.success("Biomarkers ranked", {
        description:
          payload.meta.plantedHit === "EGFR"
            ? `${payload.meta.nGenes} genes screened — planted EGFR biomarker recovered at rank #1.`
            : `${payload.meta.nGenes} genes screened — ${
                payload.meta.nSignificant
              } significant at FDR < 0.05.`,
      });
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Discovery request failed.";
      setError(message);
      toast.error("Biomarker discovery failed", { description: message });
    }
  }

  const panelLabel =
    PANEL_SIZES.find((p) => p.value === nCellLines)?.label.split(" — ")[1] ?? `${nCellLines} lines`;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(300px,360px)_minmax(0,1fr)]">
      {/* ------------------------------ Controls ------------------------------ */}
      <Card className="h-fit overflow-hidden py-0 shadow-sm xl:sticky xl:top-24">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Dna className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Biomarker discovery</CardTitle>
              <CardDescription className="text-xs">
                Synthetic GDSC/CCLE panel — 24 cell lines × 10 tissues, planted EGFR signal
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-5 pb-4">
          {/* Compound */}
          <div className="space-y-1.5">
            <Label htmlFor="bio-compound" className="text-xs font-medium">
              Compound
            </Label>
            <Select value={compound} onValueChange={(v) => setCompound(v as BiologyCompound)}>
              <SelectTrigger id="bio-compound" className="h-10" aria-label="Screened compound">
                {/* Short label in the trigger; the rich item (hint + badge)
                 * only renders inside the dropdown itself. */}
                <span className="truncate">
                  {COMPOUNDS.find((c) => c.id === compound)?.label ?? compound}
                </span>
              </SelectTrigger>
              <SelectContent>
                {COMPOUNDS.map((c) => (
                  <SelectItem key={c.id} value={c.id} textValue={c.label} className="items-start py-2">
                    <span className="flex min-w-0 flex-col gap-0.5">
                      <span className="text-sm">
                        {c.label}
                        {c.egfr ? (
                          <span className="ml-1.5 rounded bg-emerald-500/10 px-1 py-0.5 text-[10px] font-medium text-emerald-600 dark:text-emerald-400">
                            EGFR-linked
                          </span>
                        ) : null}
                      </span>
                      <span className="text-[11px] leading-snug text-muted-foreground">{c.hint}</span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11px] leading-snug text-muted-foreground">
              LN_IC50 = baseline + tissue effect + EGFR coefficient × EGFR z + noise σ 0.25
            </p>
          </div>

          {/* Genes */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="bio-genes" className="text-xs font-medium">
                Genes screened
              </Label>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">{genes}</span>
            </div>
            <Slider
              id="bio-genes"
              min={60}
              max={600}
              step={10}
              value={[genes]}
              onValueChange={(v) => setGenes(v[0] ?? genes)}
            />
            <p className="text-[11px] text-muted-foreground">
              GENE_001… noise columns + EGFR + KRT8 (breast-hot)
            </p>
          </div>

          {/* Seed + dice */}
          <div className="space-y-1.5">
            <Label htmlFor="bio-seed" className="text-xs font-medium">
              Seed
            </Label>
            <div className="flex gap-2">
              <Input
                id="bio-seed"
                inputMode="numeric"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                aria-invalid={seedValid ? undefined : true}
                className={`h-10 font-mono text-sm tabular-nums ${seedValid ? "" : "border-destructive/60"}`}
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="size-10 shrink-0"
                onClick={randomizeSeed}
                disabled={running}
                aria-label="Randomize the seed"
                title="Randomize the seed"
              >
                <Dices className="size-4" aria-hidden />
              </Button>
            </div>
            {!seedValid ? (
              <p className="text-[11px] text-destructive">Non-negative integer.</p>
            ) : null}
          </div>

          {/* Panel size */}
          <div className="space-y-1.5">
            <Label htmlFor="bio-panel" className="text-xs font-medium">
              Cell line panel
            </Label>
            <Select value={nCellLines} onValueChange={setNCellLines}>
              <SelectTrigger id="bio-panel" className="h-10" aria-label="Cell line panel size">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PANEL_SIZES.map((p) => (
                  <SelectItem key={p.value} value={p.value}>
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex gap-2">
            <Button
              size="lg"
              className="h-11 flex-1 gap-2"
              onClick={() => void discover()}
              disabled={running}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  Screening…
                </>
              ) : (
                <>
                  <Play className="size-4" aria-hidden />
                  Discover biomarkers
                </>
              )}
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="h-11 shrink-0 gap-2"
              onClick={resetSetup}
              disabled={running}
              aria-label="Reset the discovery setup to defaults"
              title="Reset setup to defaults"
            >
              <RotateCcw className="size-4" aria-hidden />
              <span className="hidden xl:inline">Reset</span>
            </Button>
          </div>

          {error ? (
            <Alert variant="destructive">
              <TriangleAlert className="size-4" aria-hidden />
              <AlertTitle>Request failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}
        </CardContent>
      </Card>

      {/* ------------------------------ Results ------------------------------ */}
      <div className="min-w-0 space-y-4">
        {!result && status !== "running" ? (
          <Card className="py-0 shadow-sm">
            <CardContent className="flex flex-col items-center justify-center gap-2 py-16 text-center">
              <span className="ping-ring flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Microscope className="size-6" aria-hidden />
              </span>
              <p className="text-sm font-medium">No discovery run yet</p>
              <p className="max-w-sm text-xs text-muted-foreground">
                Pick a compound and press{" "}
                <strong className="font-semibold text-foreground">Discover biomarkers</strong>. Every
                gene in the expression matrix is correlated (Spearman, BH-corrected) with the
                compound&apos;s LN_IC50 across the panel — erlotinib should recover the planted EGFR
                biomarker, cisplatin should not.
              </p>
            </CardContent>
          </Card>
        ) : null}

        {status === "running" ? (
          <Card className="py-0 shadow-sm">
            <CardContent className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
              <p className="text-sm font-medium">Screening…</p>
              <p className="text-xs text-muted-foreground">
                Generating the {panelLabel} screen + {genes}-gene expression matrix and ranking every
                gene against {compound}.
              </p>
            </CardContent>
          </Card>
        ) : null}

        {result && status !== "running" ? (
          <>
            {/* Stats strip */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatCard label="genes screened" value={String(result.meta.nGenes)} />
              <StatCard
                label="significant (FDR<0.05)"
                value={String(result.meta.nSignificant)}
              />
              <StatCard label="strongest |ρ|" value={fmtNum(result.meta.strongestRho, 3)} />
              <StatCard label="runtime" value={`${result.meta.runtimeMs} ms`} />
            </div>

            {/* Scatter chart */}
            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
                      <FlaskConical className="size-3.5 text-primary" aria-hidden />
                      {series?.scatter.gene ?? result.topHit.gene} expression vs LN_IC50 —{" "}
                      {result.meta.compound}
                      {result.meta.plantedHit === "EGFR" ? (
                        <Badge
                          variant="secondary"
                          className="gap-1 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                        >
                          <Check className="size-3" aria-hidden />
                          planted biomarker recovered
                        </Badge>
                      ) : null}
                    </CardTitle>
                    <CardDescription className="text-xs">
                      {result.meta.nCellLines} cell lines · ρ ={" "}
                      {fmtNum((series?.scatter ?? result.topHit).rho, 3)} · Spearman rank trend
                      dashed · points colored by tissue
                    </CardDescription>
                  </div>
                  {result.egfrScatter && result.topHit.gene !== "EGFR" ? (
                    <div
                      className="grid gap-1 rounded-lg border bg-muted/40 p-1"
                      role="tablist"
                      aria-label="Scatter gene"
                    >
                      {(
                        [
                          ["top", `top hit (${result.topHit.gene})`],
                          ["egfr", "EGFR"],
                        ] as const
                      ).map(([value, label]) => (
                        <button
                          key={value}
                          type="button"
                          role="tab"
                          aria-selected={chartGene === value}
                          onClick={() => setChartGene(value)}
                          className={`h-7 rounded-md px-2.5 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                            chartGene === value
                              ? "bg-background text-foreground shadow-sm"
                              : "text-muted-foreground hover:text-foreground"
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>
              </CardHeader>
              <CardContent className="p-3 sm:p-4">
                <div
                  className="h-[340px] w-full"
                  role="img"
                  aria-label={`Scatter plot of ${series?.scatter.gene ?? "gene"} expression versus LN_IC50 across ${result.meta.nCellLines} cell lines, points colored by tissue, with the fitted rank trend line`}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
                      <CartesianGrid
                        stroke={CHART.gridStroke}
                        strokeOpacity={CHART.gridStrokeOpacity}
                      />
                      <XAxis
                        type="number"
                        dataKey="x"
                        domain={["auto", "auto"]}
                        tick={CHART.axisTick}
                        tickFormatter={fmtTick}
                        tickLine={false}
                        axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                        label={{
                          value: `${series?.scatter.gene ?? "gene"} (log2 expression)`,
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
                        domain={["auto", "auto"]}
                        tick={CHART.axisTick}
                        tickFormatter={fmtTick}
                        tickLine={false}
                        axisLine={false}
                        label={{
                          value: "LN_IC50",
                          angle: -90,
                          position: "insideBottomLeft",
                          offset: 18,
                          fill: "#78716c",
                          fontSize: 11,
                        }}
                      />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        content={<ScatterTooltip />}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 11, color: "#78716c", paddingTop: 4 }}
                        iconSize={8}
                      />
                      {series?.groups.map((g) => (
                        <Scatter
                          key={g.tissue}
                          name={g.tissue}
                          data={g.data}
                          fill={TISSUE_COLORS[g.tissue] ?? CHART.stone}
                          fillOpacity={0.85}
                          isAnimationActive={false}
                        />
                      ))}
                      <Line
                        name="rank trend"
                        data={(series?.scatter ?? result.topHit).trend}
                        dataKey="y"
                        stroke={CHART.stone}
                        strokeWidth={2}
                        strokeDasharray="6 3"
                        dot={false}
                        activeDot={false}
                        isAnimationActive={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Ranked biomarker table */}
            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <CardTitle className="text-sm">
                  Ranked biomarkers — top {result.biomarkers.length} of{" "}
                  {result.meta.nGenes} genes
                </CardTitle>
                <CardDescription className="text-xs">
                  Spearman ρ vs {result.meta.compound} LN_IC50 · two-sided normal-approximation p
                  with tie correction · Benjamini–Hochberg FDR · significant at FDR &lt; 0.05
                </CardDescription>
              </CardHeader>
              <div className="thin-scrollbar max-h-96 overflow-auto">
                <Table className="data-table min-w-[560px]">
                  <TableHeader className="sticky top-0 z-10 bg-muted/60 backdrop-blur">
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="h-9 w-10 text-right">#</TableHead>
                      <TableHead className="h-9">Gene</TableHead>
                      <TableHead className="h-9">Spearman ρ</TableHead>
                      <TableHead className="h-9 text-right">p-value</TableHead>
                      <TableHead className="h-9 text-right">FDR</TableHead>
                      <TableHead className="h-9 w-14 text-center">hit</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.biomarkers.map((row) => (
                      <TableRow
                        key={row.gene}
                        className={cn(row.gene === "EGFR" && "bg-emerald-500/5 hover:bg-emerald-500/5")}
                      >
                        <TableCell className="py-2 text-right font-mono text-xs tabular-nums text-muted-foreground">
                          {row.rank}
                        </TableCell>
                        <TableCell className="py-2 font-mono text-xs font-medium">
                          {row.gene}
                        </TableCell>
                        <TableCell className="py-2">
                          <RhoCell rho={row.rho} />
                        </TableCell>
                        <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                          {fmtNum(row.pValue, 3)}
                        </TableCell>
                        <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                          {fmtNum(row.fdr, 3)}
                        </TableCell>
                        <TableCell className="py-2 text-center">
                          {row.significant ? (
                            <span
                              className="inline-flex items-center justify-center rounded-full bg-emerald-500/15 p-0.5 text-emerald-600 dark:text-emerald-400"
                              title="Significant at FDR < 0.05"
                            >
                              <Check className="size-3.5" aria-hidden />
                              <span className="sr-only">significant</span>
                            </span>
                          ) : (
                            <span className="text-muted-foreground/40" aria-hidden>
                              —
                            </span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <CardContent className="border-t bg-muted/20 py-2">
                <p className="text-[11px] leading-snug text-muted-foreground">
                  {result.egfr
                    ? `EGFR ranks #${result.egfr.rank} of ${result.meta.nGenes} (ρ = ${fmtNum(
                        result.egfr.rho,
                        3
                      )}, FDR = ${fmtNum(result.egfr.fdr, 3)})${
                        result.meta.plantedHit === "EGFR"
                          ? " — the planted biomarker, exactly what a working discovery pipeline should surface for an EGFR inhibitor."
                          : " — no planted signal for this compound, so EGFR behaves like any noise gene."
                      }`
                    : "EGFR is not in the matrix."}{" "}
                  Panel: {result.meta.nCellLines} lines · seed {result.meta.seed} · runtime{" "}
                  {result.meta.runtimeMs} ms.
                </p>
              </CardContent>
            </Card>

            {/* Variable genes (MAD) — the most_variable_genes twin (100x-124) */}
            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
                      <Activity className="size-4 shrink-0 text-primary" aria-hidden />
                      Variable genes (MAD)
                    </CardTitle>
                    <CardDescription className="mt-0.5 text-xs">
                      Median absolute deviation across the {result.meta.nCellLines}-line panel — the
                      robust pre-filter asking “which probes carry signal at all?” before any
                      response is correlated
                    </CardDescription>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 shrink-0 gap-1.5 text-xs"
                    onClick={() => setShowMad((v) => !v)}
                    aria-expanded={showMad}
                  >
                    {showMad ? "Hide" : "Rank variable genes (MAD)"}
                  </Button>
                </div>
              </CardHeader>
              {showMad ? (
                <>
                  <div className="thin-scrollbar max-h-72 overflow-auto">
                    <Table className="data-table min-w-[360px]">
                      <TableHeader className="sticky top-0 z-10 bg-muted/60 backdrop-blur">
                        <TableRow className="hover:bg-transparent">
                          <TableHead className="h-9 w-10 text-right">#</TableHead>
                          <TableHead className="h-9">Gene</TableHead>
                          <TableHead className="h-9 text-right">MAD</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {result.variableGenes.map((row) => (
                          <TableRow
                            key={row.gene}
                            className={cn(
                              row.gene === "EGFR" &&
                                "bg-emerald-500/5 hover:bg-emerald-500/5"
                            )}
                          >
                            <TableCell className="py-2 text-right font-mono text-xs tabular-nums text-muted-foreground">
                              {row.rank}
                            </TableCell>
                            <TableCell className="py-2 font-mono text-xs font-medium">
                              {row.gene}
                            </TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                              {fmtNum(row.mad, 3)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  <CardContent className="border-t bg-muted/20 py-2">
                    <p className="text-[11px] leading-snug text-muted-foreground">
                      Top {result.variableGenes.length} of {result.meta.nGenes} genes by MAD — most
                      variable: {result.variableGenes[0]?.gene} (MAD ={" "}
                      {fmtNum(result.variableGenes[0]?.mad ?? 0, 3)}). Deterministic ranking: MAD
                      descending, ties by gene name — the{" "}
                      <span className="font-mono">most_variable_genes</span> contract.
                    </p>
                  </CardContent>
                </>
              ) : (
                <CardContent className="bg-muted/20 py-2">
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    The ranking ships with every screen — press{" "}
                    <span className="font-medium text-foreground">Rank variable genes (MAD)</span>{" "}
                    to browse the leaderboard.
                  </p>
                </CardContent>
              )}
            </Card>
          </>
        ) : null}
      </div>
    </div>
  );
}
