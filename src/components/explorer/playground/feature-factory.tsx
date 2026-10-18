"use client";

import { useMemo, useState } from "react";
import {
  CircleCheck,
  Copy,
  Eraser,
  FlaskConical,
  Layers,
  Loader2,
  Sigma,
  Sparkles,
  TriangleAlert,
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { CHART } from "@/components/explorer/playground/calc-ui";
import { StatChip } from "@/components/explorer/stat-chip";
import { copyToClipboard, fmtNum } from "@/lib/format";
import type { FeatureFamily, FeatureFactoryResponse } from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Constants (mirror pharmaml.chem.features)                           */
/* ------------------------------------------------------------------ */

/** Column budget per family — matches the library's bit constants. */
const FAMILY_BITS: Record<FeatureFamily, number> = {
  descriptors: 43, // RDKit rdMolDescriptors.Properties() on this build
  atompair_count: 2048,
  ecfp: 2048,
  avalon_count: 512,
  rdkit_fingerprint: 2048,
  morgan_full: 2048,
  morgan_scaffold: 2048,
};

const FAMILY_LABELS: Record<FeatureFamily, string> = {
  descriptors: "RDKit descriptors",
  atompair_count: "Atom-pair counts",
  ecfp: "ECFP bits",
  avalon_count: "Avalon counts",
  rdkit_fingerprint: "RDKit fingerprint",
  morgan_full: "Morgan bits",
  morgan_scaffold: "Morgan-on-scaffold",
};

const ALL_FAMILIES = Object.keys(FAMILY_BITS) as FeatureFamily[];

const K_PRESETS = [50, 200, 500, 1000, 1500];

/** Web-calculator cap: every molecule is a real RDKit computation. */
const MAX_MOLECULES = 20;

const DEFAULT_SMILES = [
  "CC(=O)OC1=CC=CC=C1C(=O)O", // aspirin
  "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", // caffeine
  "CC(=O)NC1=CC=C(C=C1)O", // paracetamol
].join("\n");

const TABLE_ROW_CAP = 200;

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

type Status = "idle" | "running" | "done" | "error";

/** Feature factory: SMILES list -> 7-family 10,000+ column matrix ->
 *  variance ranking -> top-K selection, computed with real RDKit through
 *  the pharmaml.chem.features API on the server. */
export function FeatureFactory() {
  const [smilesText, setSmilesText] = useState(DEFAULT_SMILES);
  const [families, setFamilies] = useState<FeatureFamily[]>(ALL_FAMILIES);
  const [kText, setKText] = useState("200");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<FeatureFactoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const kParsed = Number.parseInt(kText, 10);
  const kValid = Number.isInteger(kParsed) && kParsed >= 1 && kParsed <= 1500;

  const smilesList = useMemo(
    () =>
      smilesText
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0 && !l.startsWith("#")),
    [smilesText]
  );

  const widthEstimate = useMemo(
    () => families.reduce((acc, f) => acc + FAMILY_BITS[f], 0),
    [families]
  );

  const familyOrder = useMemo(
    () => ALL_FAMILIES.filter((f) => (result ? (result.familiesComputed[f] ?? 0) > 0 : families.includes(f))),
    [families, result]
  );

  const chartData = useMemo(() => {
    if (!result) return [];
    return ALL_FAMILIES.filter((f) => (result.familiesComputed[f] ?? 0) > 0).map((f) => ({
      family: FAMILY_LABELS[f],
      short: FAMILY_LABELS[f].split(" ")[0],
      computed: result.familiesComputed[f] ?? 0,
      selected: result.familyCounts[f] ?? 0,
    }));
  }, [result]);

  function toggleFamily(f: FeatureFamily) {
    setFamilies((prev) =>
      prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]
    );
    setResult(null);
  }

  function clearAll() {
    setSmilesText("");
    setResult(null);
    setError(null);
    setStatus("idle");
  }

  async function compute() {
    if (!kValid) {
      setError("K must be an integer between 1 and 1500.");
      toast.error("Invalid K");
      return;
    }
    if (families.length === 0) {
      setError("Select at least one feature family.");
      toast.error("No families selected");
      return;
    }
    if (smilesList.length === 0) {
      setError("Paste at least one SMILES — one per line (# comments are ignored).");
      toast.error("Nothing to compute");
      return;
    }
    if (smilesList.length > MAX_MOLECULES) {
      setError(`The web calculator handles up to ${MAX_MOLECULES} molecules per run (you pasted ${smilesList.length}). Use the desktop playground.py for larger libraries.`);
      toast.error("Too many molecules");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const res = await fetch("/api/chem/features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          smiles: smilesList,
          families,
          k: kParsed,
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
      setResult(data as FeatureFactoryResponse);
      setStatus("done");
      toast.success(
        `Ranked ${(data as FeatureFactoryResponse).totalFeatures.toLocaleString("en-US")} features — kept top ${(data as FeatureFactoryResponse).k.toLocaleString("en-US")}`
      );
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Feature computation failed.";
      setError(message);
      toast.error("Feature factory failed", { description: message });
    }
  }

  async function copyCsv() {
    if (!result) return;
    const header = "rank,feature,family,importance";
    const lines = result.selected.map(
      (r) => `${r.rank},${r.feature},${r.family},${r.importance}`
    );
    const ok = await copyToClipboard([header, ...lines].join("\n"));
    if (ok) {
      setCopied(true);
      toast.success(`Copied ${result.selected.length} selected features as CSV`);
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  const running = status === "running";
  const shownRows = result ? result.selected.slice(0, TABLE_ROW_CAP) : [];

  return (
    <div className="space-y-4">
      {/* ------------------------------ Input ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <FlaskConical className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Feature factory</CardTitle>
              <CardDescription className="text-xs">
                SMILES → {widthEstimate.toLocaleString("en-US")}+ descriptors &amp; fingerprint
                columns → ranked → keep your top K — mirroring{" "}
                <code className="font-mono">pharmaml.chem.features</code> (computed with real
                RDKit on the server)
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pb-4">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Label htmlFor="feat-smiles" className="text-xs font-medium">
                Molecules — one SMILES per line
              </Label>
              <span className="text-[11px] text-muted-foreground">
                {smilesList.length}/{MAX_MOLECULES} · # comments ignored
              </span>
            </div>
            <Textarea
              id="feat-smiles"
              value={smilesText}
              onChange={(e) => setSmilesText(e.target.value)}
              spellCheck={false}
              className="min-h-24 resize-y font-mono text-xs leading-relaxed"
              placeholder={"CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C)C(=O)N2C"}
              aria-label="SMILES list, one per line"
            />
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-medium">
              Feature families{" "}
              <span className="font-normal text-muted-foreground">
                ({families.length}/7 · ≈{widthEstimate.toLocaleString("en-US")} columns)
              </span>
            </Label>
            <div className="flex flex-wrap gap-1.5">
              {ALL_FAMILIES.map((f) => {
                const active = families.includes(f);
                return (
                  <button
                    key={f}
                    type="button"
                    onClick={() => toggleFamily(f)}
                    aria-pressed={active}
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                      active
                        ? "border-primary/40 bg-primary/10 text-primary"
                        : "border-border bg-muted/40 text-muted-foreground hover:bg-muted"
                    }`}
                  >
                    <span
                      className={`inline-block size-1.5 rounded-full ${active ? "bg-primary" : "bg-muted-foreground/40"}`}
                      aria-hidden
                    />
                    {FAMILY_LABELS[f]}
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {FAMILY_BITS[f].toLocaleString("en-US")}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-[minmax(0,200px)_minmax(0,1fr)]">
            <div className="space-y-1.5">
              <Label htmlFor="feat-k" className="text-xs font-medium">
                Keep top K features
              </Label>
              <Input
                id="feat-k"
                inputMode="numeric"
                value={kText}
                onChange={(e) => setKText(e.target.value)}
                aria-invalid={kValid ? undefined : true}
                className={`h-10 font-mono text-sm tabular-nums ${kValid ? "" : "border-destructive/60"}`}
              />
              {!kValid ? (
                <p className="text-[11px] text-destructive">Integer between 1 and 1500.</p>
              ) : null}
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">K presets</Label>
              <div className="flex flex-wrap gap-1.5">
                {K_PRESETS.map((p) => (
                  <Button
                    key={p}
                    type="button"
                    variant={kParsed === p ? "default" : "outline"}
                    size="sm"
                    className="h-8 px-2.5 font-mono text-xs tabular-nums"
                    onClick={() => {
                      setKText(String(p));
                      setResult(null);
                    }}
                  >
                    {p.toLocaleString("en-US")}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              onClick={compute}
              disabled={running}
              className="h-10 gap-2"
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  Computing {widthEstimate.toLocaleString("en-US")} columns…
                </>
              ) : (
                <>
                  <Sparkles className="size-4" aria-hidden />
                  Compute &amp; rank features
                </>
              )}
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-9 gap-1.5"
              onClick={clearAll}
              disabled={running}
            >
              <Eraser className="size-3.5" aria-hidden />
              Clear
            </Button>
            <p className="text-[11px] text-muted-foreground">
              Unsupervised variance ranking · supervised RF / LightGBM ranking lives in the
              desktop <code className="font-mono">playground.py</code>
            </p>
          </div>
        </CardContent>
      </Card>

      {/* ------------------------------ Error ------------------------------ */}
      {error ? (
        <Alert variant="destructive">
          <TriangleAlert className="size-4" aria-hidden />
          <AlertTitle>Feature factory failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {/* ------------------------------ Results ------------------------------ */}
      {result ? (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
            <StatChip
              icon={Layers}
              value={result.totalFeatures.toLocaleString("en-US")}
              label="features computed"
              hint={`${result.rdkitVersion} · ${result.method} ranking`}
            />
            <StatChip
              icon={CircleCheck}
              value={String(result.nMolecules)}
              label="molecules"
              hint={
                result.dropped.length > 0
                  ? `${result.dropped.length} invalid SMILES dropped`
                  : "all SMILES valid"
              }
            />
            <StatChip
              icon={Sparkles}
              value={result.k.toLocaleString("en-US")}
              label="kept (top K)"
            />
            <StatChip
              icon={Sigma}
              value={result.survivors.toLocaleString("en-US")}
              label="median survivors"
              hint="per-family median-importance prefilter"
            />
            <StatChip
              icon={FlaskConical}
              value={`${fmtNum(result.elapsedSeconds, 1)} s`}
              label="compute time"
              hint="real RDKit run on the server"
            />
          </div>

          {result.dropped.length > 0 ? (
            <Alert>
              <TriangleAlert className="size-4" aria-hidden />
              <AlertTitle>
                {result.dropped.length} molecule{result.dropped.length > 1 ? "s" : ""} dropped
              </AlertTitle>
              <AlertDescription>
                <span className="font-mono text-xs">
                  {result.dropped.map((d) => `row ${d.row + 1}: ${d.smiles}`).join(" · ")}
                </span>
              </AlertDescription>
            </Alert>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <CardTitle className="text-sm">Selected features per family</CardTitle>
                <CardDescription className="text-xs">
                  After the per-family median filter, top {result.k.toLocaleString("en-US")}{" "}
                  overall
                </CardDescription>
              </CardHeader>
              <CardContent className="pb-4 pt-3">
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                      <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} vertical={false} />
                      <XAxis
                        dataKey="short"
                        tick={{ ...CHART.axisTick, fontSize: 10 }}
                        interval={0}
                        tickLine={false}
                        axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                      />
                      <YAxis
                        width={36}
                        allowDecimals={false}
                        tick={{ ...CHART.axisTick, fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        cursor={{ fill: "rgba(120,113,108,0.08)" }}
                        contentStyle={CHART.tooltipStyle}
                        formatter={(value: number | string, name) => [
                          Number(value).toLocaleString("en-US"),
                          String(name),
                        ]}
                        labelFormatter={(label) => String(label)}
                      />
                      <Bar dataKey="computed" name="computed" fill={CHART.stone} fillOpacity={0.45} radius={[3, 3, 0, 0]} />
                      <Bar dataKey="selected" name="selected" radius={[3, 3, 0, 0]}>
                        {chartData.map((d) => (
                          <Cell key={d.family} fill={d.selected > 0 ? CHART.emerald : CHART.gridStroke} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <ul className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 text-[11px] text-muted-foreground sm:grid-cols-2">
                  {familyOrder.map((f) => {
                    const computed = result.familiesComputed[f] ?? 0;
                    const selected = result.familyCounts[f] ?? 0;
                    const pct = computed > 0 ? Math.round((selected / computed) * 100) : 0;
                    return (
                      <li key={f} className="flex items-baseline justify-between gap-2">
                        <span className="truncate">{FAMILY_LABELS[f]}</span>
                        <span className="font-mono tabular-nums">
                          {computed.toLocaleString("en-US")} → {selected.toLocaleString("en-US")}{" "}
                          ({pct}%)
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </CardContent>
            </Card>

            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <CardTitle className="text-sm">Top {result.k.toLocaleString("en-US")} features</CardTitle>
                    <CardDescription className="text-xs">
                      Variance ranking, descending — rank {shownRows[0]?.rank ?? 1}…
                    </CardDescription>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 px-2.5 text-[11px]"
                    onClick={copyCsv}
                  >
                    {copied ? (
                      <>
                        <CircleCheck className="size-3.5 text-primary" aria-hidden />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="size-3.5" aria-hidden />
                        Copy CSV
                      </>
                    )}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="pb-4 pt-0">
                <div className="max-h-96 overflow-y-auto thin-scrollbar">
                  <Table>
                    <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur">
                      <TableRow>
                        <TableHead className="h-9 w-14 text-right font-medium">#</TableHead>
                        <TableHead className="h-9 font-medium">Feature</TableHead>
                        <TableHead className="h-9 font-medium">Family</TableHead>
                        <TableHead className="h-9 text-right font-medium">Importance</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {shownRows.map((r) => (
                        <TableRow key={r.rank} className="hover:bg-muted/40">
                          <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums text-muted-foreground">
                            {r.rank}
                          </TableCell>
                          <TableCell className="max-w-56 truncate py-1.5 font-mono text-xs" title={r.feature}>
                            {r.feature}
                          </TableCell>
                          <TableCell className="py-1.5 text-xs text-muted-foreground">
                            {FAMILY_LABELS[r.family as FeatureFamily] ?? r.family}
                          </TableCell>
                          <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                            {fmtNum(r.importance, 4)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                {result.selected.length > TABLE_ROW_CAP ? (
                  <p className="mt-2 text-[11px] text-muted-foreground">
                    Showing first {TABLE_ROW_CAP.toLocaleString("en-US")} of{" "}
                    {result.selected.length.toLocaleString("en-US")} — copy the CSV for the full
                    list.
                  </p>
                ) : null}
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}
