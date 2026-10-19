"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowDownWideNarrow,
  ArrowUpNarrowWide,
  ClipboardPaste,
  Crown,
  FileSpreadsheet,
  Loader2,
  RotateCcw,
  Scale,
  Sparkles,
  Trophy,
} from "lucide-react";
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
import { Checkbox } from "@/components/ui/checkbox";
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
import { Textarea } from "@/components/ui/textarea";
import { fmtNum } from "@/lib/format";
import type { DockingRescoreResponse } from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* The §9 demo table — the shipped example campaign, enriched with     */
/* heavy-atom ligand efficiencies exactly like the notebook cell.      */
/* ------------------------------------------------------------------ */
const EXAMPLE_CSV = `ligand,best_lowest_energy,best_num_in_cluster,ligand_efficiency
aspirin,-4.9100,15,0.3777
ibuprofen,-9.6300,15,0.6420
naproxen,-9.5200,13,0.5600
paracetamol,-5.6800,13,0.5164
caffeine,-7.1400,15,0.5100
theophylline,-11.5100,15,0.8854
metformin,-8.1300,19,0.9033
propranolol,-7.8300,13,0.4121
atenolol,-8.1100,16,0.4268
lidocaine,-5.9700,14,0.3512
coumarin,-7.3400,20,0.6673
vanillin,-8.6400,14,0.7855`;

/** Columns auto-preselected when a table is pasted (preference order). */
const PREFERRED_EVIDENCE = [
  "ligand_efficiency",
  "LE (kcal/mol/atom)",
  "best_lowest_energy",
  "lowest_energy",
  "affinity",
  "best_num_in_cluster",
  "num_in_cluster",
  "cluster_size",
];

const MAX_CONTENT = 200_000;
const STORAGE_KEY = "pharmaml.rescoreInput";

interface SniffedColumn {
  name: string;
  numeric: boolean;
}

interface ColumnConfig {
  weight: string;
  direction: "auto" | "higher" | "lower";
}

type Status = "idle" | "running" | "done" | "error";

interface Persisted {
  content: string;
}

function loadPersisted(): Persisted | null {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    const content =
      typeof (parsed as Partial<Persisted>).content === "string"
        ? ((parsed as Partial<Persisted>).content as string).slice(0, MAX_CONTENT)
        : "";
    return { content };
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

/** Sniff the delimiter of the header line (comma/semicolon/tab/pipe). */
function sniffDelimiter(line: string): string {
  let best = ",";
  let bestCount = -1;
  for (const delimiter of [",", ";", "\t", "|"]) {
    const count = line.split(delimiter).length - 1;
    if (count > bestCount) {
      best = delimiter;
      bestCount = count;
    }
  }
  return bestCount > 0 ? best : ",";
}

/** Parse the header + probe the first rows for numeric-ness. */
function sniffColumns(content: string): SniffedColumn[] {
  const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length === 0) return [];
  const delimiter = sniffDelimiter(lines[0]);
  const headers = lines[0].split(delimiter).map((h) => h.trim()).filter(Boolean);
  if (headers.length === 0) return [];
  const sample = lines.slice(1, 8).map((line) => line.split(delimiter));
  return headers.map((name, index) => {
    let numericCount = 0;
    let seen = 0;
    for (const cells of sample) {
      const raw = (cells[index] ?? "").trim();
      if (raw.length === 0) continue;
      seen += 1;
      if (Number.isFinite(Number(raw))) numericCount += 1;
    }
    return { name, numeric: seen > 0 && numericCount === seen };
  });
}

/** A one-dimensional 0-1 meter rendered as a tiny bar under the value. */
function UnitMeter({ value }: { value: number | null }) {
  if (value === null || !Number.isFinite(value)) {
    return <span className="text-muted-foreground">—</span>;
  }
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const tone =
    value >= 0.75 ? "bg-emerald-500" : value >= 0.4 ? "bg-amber-500" : "bg-muted-foreground/50";
  return (
    <span className="inline-flex items-center justify-end gap-2">
      <span className="tabular-nums">{fmtNum(value, 3)}</span>
      <span className="h-1.5 w-10 overflow-hidden rounded-full bg-muted" aria-hidden>
        <span className={`block h-full rounded-full ${tone}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="sr-only">{`${Math.round(pct)}% of the 0-1 unit range`}</span>
    </span>
  );
}

export function ConsensusRescore() {
  const [restored] = useState(loadPersisted);
  const [content, setContent] = useState(restored?.content ?? "");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [config, setConfig] = useState<Record<string, ColumnConfig>>({});
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<DockingRescoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* Persist the pasted table so a reload doesn't lose it (hydration-safe:
   * only rendered inside the Playground tab, never during SSR). */
  useEffect(() => {
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ content }));
    } catch {
      /* quota exceeded or private mode — persistence is optional */
    }
  }, [content]);

  const columns = useMemo(() => sniffColumns(content), [content]);
  const identity = columns.length > 0 ? columns[0] : null;
  const evidenceColumns = useMemo(
    () => columns.slice(1).filter((column) => column.numeric),
    [columns]
  );

  /* When a new table lands, preselect the toolkit's canonical evidence
   * columns (or every numeric column when none of them match). */
  useEffect(() => {
    const preferred = PREFERRED_EVIDENCE.filter((name) =>
      evidenceColumns.some((column) => column.name === name)
    );
    const picks =
      preferred.length > 0
        ? preferred
        : evidenceColumns.slice(0, 5).map((column) => column.name);
    setSelected(new Set(picks));
    setResult(null);
    setStatus("idle");
    setError(null);
  }, [evidenceColumns]);

  function toggleColumn(name: string, checked: boolean) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (checked) next.add(name);
      else next.delete(name);
      return next;
    });
  }

  function weightOf(name: string): string {
    return config[name]?.weight ?? "";
  }

  function setWeight(name: string, value: string) {
    setConfig((previous) => ({
      ...previous,
      [name]: { weight: value, direction: previous[name]?.direction ?? "auto" },
    }));
  }

  function directionOf(name: string): ColumnConfig["direction"] {
    return config[name]?.direction ?? "auto";
  }

  function setDirection(name: string, value: ColumnConfig["direction"]) {
    setConfig((previous) => ({
      ...previous,
      [name]: { weight: previous[name]?.weight ?? "", direction: value },
    }));
  }

  function loadExample() {
    setContent(EXAMPLE_CSV);
    toast.info("Example loaded", {
      description: "The §9 docking-campaign table with ligand efficiencies.",
    });
  }

  function clearAll() {
    setContent("");
    setResult(null);
    setError(null);
    setStatus("idle");
  }

  async function rescore() {
    if (!content.trim()) {
      setError("Paste a docking summary table first — one header line plus one row per ligand.");
      toast.error("Nothing to rescore");
      return;
    }
    const picked = [...selected];
    if (picked.length === 0) {
      setError("Tick at least one evidence column — the consensus combines them.");
      toast.error("No evidence selected");
      return;
    }
    // Weights: all blank -> uniform (the library default); any set -> a
    // full list with blanks read as 1.
    const rawWeights = picked.map(weightOf);
    const anyWeight = rawWeights.some((raw) => raw.trim().length > 0);
    let weights: number[] | undefined;
    if (anyWeight) {
      weights = [];
      for (const raw of rawWeights) {
        const value = raw.trim().length === 0 ? 1 : Number(raw);
        if (!Number.isFinite(value) || value < 0 || value > 100) {
          setError("Weights must be numbers between 0 and 100 (blank = 1).");
          toast.error("Invalid weights");
          return;
        }
        weights.push(value);
      }
    }
    const higher = picked.filter((name) => directionOf(name) === "higher");

    setStatus("running");
    setError(null);
    try {
      const res = await fetch("/api/docking/rescore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, columns: picked, weights, higher }),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? String((data as { error: unknown }).error)
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const payload = data as DockingRescoreResponse;
      setResult(payload);
      setStatus("done");
      if (payload.verdict) {
        toast.success(`${payload.verdict.identity ?? "The top row"} tops the table`, {
          description: `${payload.verdict.driver} = ${fmtNum(payload.verdict.driverValue, 3)} is the driving evidence.`,
        });
      } else {
        toast.success(`${payload.nRows} row(s) rescored`);
      }
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Consensus rescoring failed.";
      setError(message);
      toast.error("Rescore failed", { description: message });
    }
  }

  const running = status === "running";
  const normColumns = result ? result.columns.map((column) => `${column}_norm`) : [];

  return (
    <div className="space-y-4">
      {/* ------------------------------ Input ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Scale className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Consensus rescore — the explainable ranking</CardTitle>
              <CardDescription className="text-xs">
                Combine a summary table&apos;s evidence into one 0–1 consensus, with the
                per-column units that explain the winner. Runs the real{" "}
                <code className="font-mono">consensus_breakdown</code> in the toolkit venv —
                the same shared core as the CLI.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 p-3 sm:p-4">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Label htmlFor="rescore-csv" className="text-xs font-medium text-muted-foreground">
                Docking summary table (CSV — one row per ligand)
              </Label>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={loadExample}>
                  <Sparkles className="size-3.5" aria-hidden />
                  Load §9 example
                </Button>
                <Button variant="ghost" size="sm" onClick={clearAll}>
                  <RotateCcw className="size-3.5" aria-hidden />
                  Clear
                </Button>
              </div>
            </div>
            <Textarea
              id="rescore-csv"
              value={content}
              onChange={(event) => setContent(event.target.value.slice(0, MAX_CONTENT))}
              placeholder={"ligand,best_lowest_energy,best_num_in_cluster,ligand_efficiency\naspirin,-4.91,15,0.378\n..."}
              className="min-h-[140px] font-mono text-xs leading-relaxed"
              spellCheck={false}
              aria-describedby="rescore-csv-hint"
            />
            <p id="rescore-csv-hint" className="text-xs text-muted-foreground">
              {content.trim()
                ? `${columns.length} column(s) detected · first column "${identity?.name ?? "—"}" is the ligand identity`
                : "Paste a scan/rescored CSV, or load the shipped example campaign's §9 table."}
            </p>
          </div>

          {evidenceColumns.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground">
                Evidence columns — tick to combine, set weights and directions:
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                {evidenceColumns.map((column) => {
                  const checked = selected.has(column.name);
                  return (
                    <div
                      key={column.name}
                      className="flex flex-wrap items-center gap-2 rounded-lg border bg-card px-3 py-2"
                    >
                      <Checkbox
                        id={`ev-${column.name}`}
                        checked={checked}
                        onCheckedChange={(value) => toggleColumn(column.name, value === true)}
                        aria-label={`Use ${column.name} as evidence`}
                      />
                      <Label
                        htmlFor={`ev-${column.name}`}
                        className="min-w-0 flex-1 truncate font-mono text-xs"
                        title={column.name}
                      >
                        {column.name}
                      </Label>
                      <Input
                        value={weightOf(column.name)}
                        onChange={(event) => setWeight(column.name, event.target.value)}
                        disabled={!checked}
                        placeholder="w=1"
                        inputMode="decimal"
                        aria-label={`Weight for ${column.name}`}
                        className="h-8 w-16 px-2 font-mono text-xs tabular-nums"
                      />
                      <Select
                        value={directionOf(column.name)}
                        onValueChange={(value) =>
                          setDirection(column.name, value as ColumnConfig["direction"])
                        }
                        disabled={!checked}
                      >
                        <SelectTrigger
                          className="h-8 w-[104px] text-xs"
                          aria-label={`Direction for ${column.name}`}
                        >
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">
                            <span className="flex items-center gap-1.5">
                              <ArrowDownWideNarrow className="size-3" aria-hidden />
                              auto
                            </span>
                          </SelectItem>
                          <SelectItem value="higher">
                            <span className="flex items-center gap-1.5">
                              <ArrowUpNarrowWide className="size-3" aria-hidden />
                              higher=better
                            </span>
                          </SelectItem>
                          <SelectItem value="lower">
                            <span className="flex items-center gap-1.5">
                              <ArrowDownWideNarrow className="size-3" aria-hidden />
                              lower=better
                            </span>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  );
                })}
              </div>
              <p className="text-xs text-muted-foreground">
                auto = the library&apos;s energy/affinity heuristic (columns ending in
                &ldquo;energy&rdquo; or &ldquo;affinity&rdquo; are lower-is-better).
              </p>
            </div>
          ) : null}

          {error ? (
            <Alert variant="destructive">
              <AlertTitle>Rescoring failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={rescore} disabled={running || selected.size === 0}>
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  Rescoring…
                </>
              ) : (
                <>
                  <ClipboardPaste className="size-4" aria-hidden />
                  Rescore with pharmaml
                </>
              )}
            </Button>
            <span className="text-xs text-muted-foreground">
              {selected.size > 0
                ? `${selected.size} evidence column(s) selected · blank weights = uniform`
                : "select evidence columns above"}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* ----------------------------- Result ------------------------------ */}
      {result ? (
        <Card className="overflow-hidden py-0 shadow-sm">
          <CardHeader className="border-b bg-muted/30 py-3">
            <div className="flex items-center gap-2.5">
              <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <Trophy className="size-4.5" aria-hidden />
              </span>
              <div className="min-w-0">
                <CardTitle className="text-base">Why the winner wins</CardTitle>
                <CardDescription className="text-xs">
                  One 0–1 <code className="font-mono">&lt;column&gt;_norm</code> unit per evidence
                  column; the consensus is their weighted mean.
                </CardDescription>
              </div>
              {result.pharmamlVersion ? (
                <Badge variant="secondary" className="ml-auto shrink-0 font-mono text-[10px]">
                  pharmaml {result.pharmamlVersion}
                </Badge>
              ) : null}
            </div>
          </CardHeader>
          <CardContent className="space-y-4 p-3 sm:p-4">
            {result.verdict ? (
              <div
                role="status"
                className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border bg-primary/5 px-4 py-3"
              >
                <Crown className="size-4 shrink-0 text-primary" aria-hidden />
                <p className="min-w-0 flex-1 text-sm">
                  <span className="font-semibold">{result.verdict.identity ?? "The top row"}</span>{" "}
                  tops the table (consensus{" "}
                  <span className="font-semibold tabular-nums">
                    {fmtNum(result.verdict.consensus, 3)}
                  </span>
                  ):{" "}
                  <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
                    {result.verdict.driver}
                  </code>{" "}
                  = <span className="tabular-nums">{fmtNum(result.verdict.driverValue, 3)}</span>{" "}
                  is the driving evidence.
                </p>
              </div>
            ) : (
              <div role="status" className="rounded-lg border bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
                No complete consensus in the table — every row kept a NaN input.
              </div>
            )}

            <p className="font-mono text-xs text-muted-foreground">
              <span className="font-semibold">evidence:</span> {result.recipe}
            </p>

            <div className="max-h-96 overflow-auto rounded-lg border">
              <Table>
                <TableHeader className="sticky top-0 z-10 bg-background shadow-[0_1px_0_0_hsl(var(--border))]">
                  <TableRow>
                    <TableHead className="h-9">{result.identityColumn}</TableHead>
                    {normColumns.map((column) => (
                      <TableHead key={column} className="h-9 text-right">
                        <span className="font-mono">{column}</span>
                      </TableHead>
                    ))}
                    <TableHead className="h-9 text-right">
                      <span className="font-mono font-semibold">consensus</span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.rows.map((row, index) => {
                    const consensus = typeof row["consensus"] === "number" ? row["consensus"] : null;
                    const isWinner =
                      index === 0 && consensus !== null && result.verdict !== null;
                    return (
                      <TableRow
                        key={`${String(row[result.identityColumn])}-${index}`}
                        style={
                          isWinner
                            ? {
                                background:
                                  "color-mix(in oklch, var(--primary) 8%, transparent)",
                              }
                            : undefined
                        }
                      >
                        <TableCell className="py-2 font-mono text-xs">
                          {isWinner ? (
                            <Badge
                              className="bg-primary/15 text-primary hover:bg-primary/15"
                              variant="secondary"
                            >
                              {String(row[result.identityColumn] ?? "—")}
                            </Badge>
                          ) : (
                            <span className={consensus === null ? "text-muted-foreground/60" : ""}>
                              {String(row[result.identityColumn] ?? "—")}
                            </span>
                          )}
                        </TableCell>
                        {normColumns.map((column) => {
                          const value = typeof row[column] === "number" ? (row[column] as number) : null;
                          return (
                            <TableCell key={column} className="py-2 text-right font-mono text-xs">
                              <UnitMeter value={value} />
                            </TableCell>
                          );
                        })}
                        <TableCell className="py-2 text-right font-mono text-xs font-semibold text-primary">
                          {consensus === null ? (
                            <span className="font-normal text-muted-foreground">—</span>
                          ) : (
                            <span className="tabular-nums">{fmtNum(consensus, 3)}</span>
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <FileSpreadsheet className="size-3.5 shrink-0" aria-hidden />
              {result.nRows} row(s) rescored · consensus = weighted mean of the _norm columns,
              higher is better · NaN rows sort last.
            </p>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
