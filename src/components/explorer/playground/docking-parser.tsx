"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Boxes,
  CircleCheck,
  Eraser,
  ListTree,
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
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

/** sessionStorage key for the pasted docking input (survives reloads). */
const DOCKING_STORAGE_KEY = "pharmaml.dockingInput";

interface DockingPersisted {
  format: DockingFormat;
  content: string;
  heavyAtoms: string;
}

/** Read the persisted docking input (client only, failure-tolerant). */
function loadDockingInput(): DockingPersisted | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(DOCKING_STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    const p = parsed as Partial<DockingPersisted>;
    const format =
      p.format === "vina" || p.format === "dlg" || p.format === "csv" ? p.format : "vina";
    const content = typeof p.content === "string" ? p.content.slice(0, 200_000) : "";
    const heavyAtoms = typeof p.heavyAtoms === "string" ? p.heavyAtoms.slice(0, 6) : "32";
    return { format, content, heavyAtoms };
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

/* ------------------------------------------------------------------ *
 * Torsion QA mini-widget — a client-side mirror of the Python API
 * pharmaml.docking.check_torsion_consistency: identical counting
 * rules (per trimmed line) and identical convention resolution.
 * ------------------------------------------------------------------ */

type TorsionConvention = "auto" | "mgltools" | "meeko";

/** sessionStorage key for the torsion-QA input (survives reloads). */
const TORSION_STORAGE_KEY = "pharmaml.torsionQa";

interface TorsionPersisted {
  content: string;
  convention: TorsionConvention;
}

const TORSION_CONVENTIONS: Array<{ value: TorsionConvention; hint: string }> = [
  { value: "auto", hint: "MGLTools rule when active REMARKs exist, else Meeko" },
  { value: "mgltools", hint: "TORSDOF = remarks = branches" },
  { value: "meeko", hint: "TORSDOF = branches" },
];

/** MGLTools-style ligand: 8 ATOM records, 2 active torsions and a
 * matching 2 BRANCH pairs + TORSDOF 2 → the OK state (2/2/2). */
const TORSION_OK_SAMPLE = `REMARK  Name = ligand_47
REMARK  SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
REMARK  2 active torsions.
REMARK       1  A  between atoms: LIG_1_LIG_2
REMARK       2  A  between atoms: LIG_5_LIG_6
ROOT
ATOM      1  C1  LIG     1      12.345   4.210   2.947  1.00  0.00     0.123 C
ATOM      2  C2  LIG     1      11.210   3.890   3.752  1.00  0.00     0.142 C
ATOM      3  C3  LIG     1      10.987   4.650   4.995  1.00  0.00     0.098 C
ATOM      4  O1  LIG     1      13.110   3.001   2.014  1.00  0.00    -0.312 OA
ENDROOT
BRANCH    1    2
ATOM      5  C4  LIG     1      10.234   5.912   5.311  1.00  0.00     0.105 C
ATOM      6  C5  LIG     1       9.187   6.043   6.210  1.00  0.00     0.097 C
ENDBRANCH   1    2
BRANCH    1    4
ATOM      7  C6  LIG     1      14.003   2.544   1.356  1.00  0.00     0.118 C
ATOM      8  O2  LIG     1      14.512   3.001   0.321  1.00  0.00    -0.298 OA
ENDBRANCH   1    4
TORSDOF 2`;

/** Same ligand after one active-torsion REMARK was lost in a manual
 * edit: TORSDOF 2 vs 1 remark vs 2 branches → MISMATCH under the
 * auto/MGLTools rule, but OK under the Meeko rule (remarks-only
 * disagreement) — so both badge states are one click apart. */
const TORSION_MISMATCH_SAMPLE = `REMARK  Name = ligand_47
REMARK  SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
REMARK  2 active torsions.
REMARK       1  A  between atoms: LIG_1_LIG_2
ROOT
ATOM      1  C1  LIG     1      12.345   4.210   2.947  1.00  0.00     0.123 C
ATOM      2  C2  LIG     1      11.210   3.890   3.752  1.00  0.00     0.142 C
ATOM      3  C3  LIG     1      10.987   4.650   4.995  1.00  0.00     0.098 C
ATOM      4  O1  LIG     1      13.110   3.001   2.014  1.00  0.00    -0.312 OA
ENDROOT
BRANCH    1    2
ATOM      5  C4  LIG     1      10.234   5.912   5.311  1.00  0.00     0.105 C
ATOM      6  C5  LIG     1       9.187   6.043   6.210  1.00  0.00     0.097 C
ENDBRANCH   1    2
BRANCH    1    4
ATOM      7  C6  LIG     1      14.003   2.544   1.356  1.00  0.00     0.118 C
ATOM      8  O2  LIG     1      14.512   3.001   0.321  1.00  0.00    -0.298 OA
ENDBRANCH   1    4
TORSDOF 2`;

/** Read the persisted torsion-QA input (client only, failure-tolerant). */
function loadTorsionInput(): TorsionPersisted | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(TORSION_STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    const p = parsed as Partial<TorsionPersisted>;
    const convention: TorsionConvention =
      p.convention === "auto" || p.convention === "mgltools" || p.convention === "meeko"
        ? p.convention
        : "auto";
    const content = typeof p.content === "string" ? p.content.slice(0, 200_000) : "";
    return { content, convention };
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

interface TorsionAudit {
  torsdof: number;
  activeRemarks: number;
  branches: number;
  /** Rule actually applied once "auto" is resolved from the remark count. */
  rule: "mgltools" | "meeko";
  ok: boolean;
  /** One-line invariant report (failed invariant when mismatched). */
  explanation: string;
}

/** Mirror of pharmaml.docking.check_torsion_consistency, line by line:
 * per trimmed line the last well-formed `TORSDOF <int>` wins (malformed
 * values are ignored), `REMARK … A …` lines mark active torsions and
 * `BRANCH` lines grow the torsion tree; the convention then applies the
 * MGLTools three-way invariant or the Meeko two-way one. */
function auditTorsionTree(content: string, convention: TorsionConvention): TorsionAudit {
  let torsdof = 0;
  let activeRemarks = 0;
  let branches = 0;
  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (line.startsWith("TORSDOF")) {
      const value = line.split(/\s+/)[1];
      // Python int() tolerates an optional sign; anything else raises → ignored.
      if (value !== undefined && /^[+-]?\d+$/.test(value)) torsdof = Number(value);
    } else if (line.startsWith("REMARK") && line.includes(" A ")) {
      activeRemarks += 1;
    } else if (line.startsWith("BRANCH")) {
      branches += 1;
    }
  }
  const rule: "mgltools" | "meeko" =
    convention === "auto" ? (activeRemarks > 0 ? "mgltools" : "meeko") : convention;
  const ok =
    rule === "mgltools"
      ? torsdof === activeRemarks && activeRemarks === branches
      : torsdof === branches;
  return {
    torsdof,
    activeRemarks,
    branches,
    rule,
    ok,
    explanation: torsionExplanation(torsdof, activeRemarks, branches, rule, ok),
  };
}

/** Human one-liner naming the invariant that held or failed. */
function torsionExplanation(
  torsdof: number,
  activeRemarks: number,
  branches: number,
  rule: "mgltools" | "meeko",
  ok: boolean
): string {
  if (ok) {
    return rule === "mgltools"
      ? `TORSDOF ${torsdof} = ${activeRemarks} active torsion remarks = ${branches} BRANCH records — the torsion tree is consistent.`
      : `TORSDOF ${torsdof} = BRANCH count ${branches} — the torsion tree is consistent under the Meeko convention${
          activeRemarks > 0 && activeRemarks !== torsdof
            ? " (the REMARK count is not part of this invariant)"
            : ""
        }.`;
  }
  if (rule === "mgltools") {
    if (torsdof !== activeRemarks && torsdof !== branches && activeRemarks !== branches) {
      return `TORSDOF ${torsdof}, ${activeRemarks} active torsion remarks and ${branches} BRANCH records all disagree — the torsion tree is inconsistent.`;
    }
    if (torsdof !== activeRemarks) {
      return `TORSDOF ${torsdof} ≠ ${activeRemarks} active torsion remarks — the torsion tree is inconsistent.`;
    }
    return `TORSDOF ${torsdof} ≠ ${branches} BRANCH records — the torsion tree is inconsistent.`;
  }
  return `TORSDOF ${torsdof} ≠ ${branches} BRANCH records — the torsion tree is inconsistent (Meeko convention).`;
}

function PoseValue({ value, sig = 3 }: { value: number | undefined; sig?: number }) {
  if (value === undefined || !Number.isFinite(value)) return <span className="text-muted-foreground">—</span>;
  return <span className="tabular-nums">{fmtNum(value, sig)}</span>;
}

export function DockingParser() {
  const [restored] = useState(loadDockingInput);
  const [format, setFormat] = useState<DockingFormat>(restored?.format ?? "vina");
  const [content, setContent] = useState(restored?.content ?? "");
  const [heavyAtoms, setHeavyAtoms] = useState(restored?.heavyAtoms ?? "32");

  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DockingParseResponse | null>(null);

  /* Persist the input (format + pasted text + heavy atoms) so a reload
   * doesn't lose work. Hydration-safe: this component only renders inside
   * the Playground tab, never during the SSR/hydration render. */
  useEffect(() => {
    try {
      window.sessionStorage.setItem(
        DOCKING_STORAGE_KEY,
        JSON.stringify({ format, content, heavyAtoms })
      );
    } catch {
      /* quota exceeded or private mode — persistence is optional */
    }
  }, [format, content, heavyAtoms]);

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

  /* ------------------ Torsion QA mini-widget (client-side) ------------------ */
  const [torsionRestored] = useState(loadTorsionInput);
  const [torsionInput, setTorsionInput] = useState(torsionRestored?.content ?? "");
  const [convention, setConvention] = useState<TorsionConvention>(
    torsionRestored?.convention ?? "auto"
  );

  /* Persist the torsion-QA input like the parser above (optional,
   * quota-tolerant) so a reload or tab switch doesn't lose the paste. */
  useEffect(() => {
    try {
      window.sessionStorage.setItem(
        TORSION_STORAGE_KEY,
        JSON.stringify({ content: torsionInput, convention })
      );
    } catch {
      /* quota exceeded or private mode — persistence is optional */
    }
  }, [torsionInput, convention]);

  /* Pure string counting — recomputed on every keystroke, no debounce. */
  const torsion = useMemo(
    () => auditTorsionTree(torsionInput, convention),
    [torsionInput, convention]
  );

  function loadTorsionSample(mismatch: boolean) {
    setTorsionInput(mismatch ? TORSION_MISMATCH_SAMPLE : TORSION_OK_SAMPLE);
    toast.success(mismatch ? "Mismatch example loaded" : "Consistent example loaded");
  }

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
                {content.length > 0 ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-8 gap-1.5 px-2.5 text-[11px] text-muted-foreground"
                    onClick={() => {
                      setContent("");
                      setStatus("idle");
                      setResult(null);
                      setError(null);
                    }}
                  >
                    <Eraser className="size-3" aria-hidden />
                    Clear
                  </Button>
                ) : null}
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
            <p className="text-[11px] text-muted-foreground">
              Input survives page reloads within this browser session — use Clear to wipe it.
            </p>
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

      {/* ------------------------------ Torsion QA ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <ListTree className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Torsion QA</CardTitle>
              <CardDescription className="text-xs">
                Ligand PDBQT torsion-tree audit — mirroring{" "}
                <code className="font-mono">pharmaml.docking.check_torsion_consistency</code>
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pb-4">
          <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_212px]">
            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Label htmlFor="torsion-input" className="text-xs font-medium">
                  Ligand PDBQT content
                </Label>
                <div className="flex flex-wrap gap-1.5">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 px-2.5 text-[11px]"
                    onClick={() => loadTorsionSample(false)}
                  >
                    <CircleCheck className="size-3" aria-hidden />
                    Load example
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 px-2.5 text-[11px]"
                    onClick={() => loadTorsionSample(true)}
                  >
                    <TriangleAlert className="size-3" aria-hidden />
                    Load mismatch
                  </Button>
                  {torsionInput.length > 0 ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-8 gap-1.5 px-2.5 text-[11px] text-muted-foreground"
                      onClick={() => setTorsionInput("")}
                    >
                      <Eraser className="size-3" aria-hidden />
                      Clear
                    </Button>
                  ) : null}
                </div>
              </div>
              <Textarea
                id="torsion-input"
                value={torsionInput}
                onChange={(e) => setTorsionInput(e.target.value)}
                placeholder="Paste a prepared ligand PDBQT (REMARK / ROOT / BRANCH / TORSDOF records) — counts update live…"
                className="thin-scrollbar min-h-[140px] max-h-64 resize-y font-mono text-xs leading-relaxed"
                aria-label="Paste ligand PDBQT content"
              />
            </div>
            <div className="space-y-1.5">
              <p id="torsion-convention-label" className="text-xs font-medium">
                Convention
              </p>
              <RadioGroup
                value={convention}
                onValueChange={(v) => setConvention(v as TorsionConvention)}
                className="gap-2.5"
                aria-labelledby="torsion-convention-label"
              >
                {TORSION_CONVENTIONS.map((c) => (
                  <div key={c.value} className="flex items-start gap-2">
                    <RadioGroupItem value={c.value} id={`torsion-conv-${c.value}`} className="mt-0.5" />
                    <div className="grid gap-0.5">
                      <Label
                        htmlFor={`torsion-conv-${c.value}`}
                        className="cursor-pointer text-xs font-medium leading-none"
                      >
                        {c.value}
                      </Label>
                      <p className="text-[10px] leading-snug text-muted-foreground">{c.hint}</p>
                    </div>
                  </div>
                ))}
              </RadioGroup>
            </div>
          </div>

          {torsionInput.trim().length === 0 ? (
            <p className="rounded-lg border border-dashed bg-muted/20 px-3 py-2.5 text-xs leading-relaxed text-muted-foreground">
              Paste a prepared ligand PDBQT to audit its torsion tree — TORSDOF must equal the
              active-torsion REMARKs and BRANCH count (MGLTools convention) or the BRANCH count
              alone (Meeko convention).
            </p>
          ) : (
            <div className="space-y-2.5 rounded-lg border bg-muted/30 p-3">
              <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
                <div className="flex flex-wrap gap-2">
                  {[
                    { label: "TORSDOF", value: torsion.torsdof },
                    { label: "Active REMARKs", value: torsion.activeRemarks },
                    { label: "BRANCHes", value: torsion.branches },
                  ].map((chip) => (
                    <div key={chip.label} className="min-w-[86px] rounded-lg border bg-card px-3 py-2">
                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        {chip.label}
                      </p>
                      <p className="font-mono text-lg font-semibold leading-tight tabular-nums">
                        {chip.value}
                      </p>
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <div className="min-w-[124px]">
                    <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      Convention
                    </p>
                    <p className="text-sm font-medium leading-tight">
                      {torsion.rule}
                      <span className="ml-1 text-[11px] font-normal text-muted-foreground">
                        {convention === "auto" ? "(auto-detected)" : "(selected)"}
                      </span>
                    </p>
                  </div>
                  <span
                    aria-live="polite"
                    className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
                      torsion.ok
                        ? "border-emerald-200 bg-emerald-50 text-emerald-600 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-400"
                        : "border-destructive/30 bg-destructive/10 text-destructive"
                    }`}
                  >
                    <span
                      className={`size-1.5 rounded-full ${torsion.ok ? "bg-emerald-500" : "bg-destructive"}`}
                      aria-hidden
                    />
                    {torsion.ok ? "OK" : "MISMATCH"}
                  </span>
                </div>
              </div>
              <p
                role="status"
                className={`text-xs leading-relaxed ${
                  torsion.ok ? "text-muted-foreground" : "font-medium text-destructive"
                }`}
              >
                {torsion.explanation}
              </p>
            </div>
          )}

          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Parity:{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
              pharmaml.docking.check_torsion_consistency(path, convention={JSON.stringify(convention)})
            </code>{" "}
            /{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
              pharmaml docking torsion --dir &lt;folder&gt;
            </code>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
