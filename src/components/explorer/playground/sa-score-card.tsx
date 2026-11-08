"use client";

import { useMemo, useState } from "react";
import {
  CircleCheck,
  Copy,
  Download,
  FlaskConical,
  Info,
  Loader2,
  Sigma,
  TriangleAlert,
} from "lucide-react";
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
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/format";
import type { SAScoreResponse, SAScoreRow } from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Synthetic Accessibility score — part3 §24, the web twin             */
/* ------------------------------------------------------------------ */
/**
 * MASTER_PROMPT part3 section 24: SA scoring over the generative
 * results. One POST runs the REAL Ertl-Schuffenhauer scorer
 * (pharmaml.chem.sa_scoring) with real RDKit on the server — the same
 * fragment database the official RDKit Contrib implementation loads.
 *
 * The card mirrors the section's acceptance list: scores for every
 * valid molecule (invalid rows kept with their reason — never a
 * misleading number), the direction and limitations copy always
 * visible, easiest-first ordering, mean/min/max/median/std summary,
 * and an inclusive min/max filter whose rejected candidates stay on
 * screen ("do not permanently discard generated candidates unless the
 * user explicitly enables such filtering").
 */

const DEFAULT_SMILES = [
  "CC(=O)Oc1ccccc1C(=O)O", // aspirin
  "CC(C)Cc1ccc(O)cc1", // ibuprofen core
  "Cn1cnc2c1c(=O)n(C)c(=O)n2C", // caffeine
  "CC1=C(C[C@@H]2[C@]1(C[C@H]([C@@H](C2=O)O)NC(=O)c3ccccc3)O)OC(=O)C", // taxane
].join("\n");

const MAX_MOLECULES = 50;

type Status = "idle" | "running" | "done" | "error";

function scoreColor(score: number): string {
  if (score <= 3.5) return "text-emerald-600 dark:text-emerald-400";
  if (score <= 6.0) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

function bandLabel(score: number): string {
  if (score <= 3.5) return "easy";
  if (score <= 6.0) return "moderate";
  return "hard";
}

async function postSaScore(body: unknown): Promise<SAScoreResponse> {
  const res = await fetch("/api/chem/sa-score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data: unknown = await res.json();
  if (!res.ok) {
    const message =
      typeof data === "object" && data !== null && "error" in data
        ? String((data as { error: unknown }).error)
        : `HTTP ${res.status}`;
    throw new Error(message);
  }
  return data as SAScoreResponse;
}

function rowsToCsv(rows: SAScoreRow[]): string {
  const header = "smiles,canonical_smiles,sa_score,valid,error";
  const lines = rows.map((r) =>
    [
      r.smiles,
      r.canonicalSmiles ?? "",
      r.score !== null && r.valid ? String(r.score) : "",
      r.valid ? "true" : "false",
      (r.error ?? "").replace(/,/g, ";"),
    ].join(",")
  );
  return [header, ...lines].join("\n");
}

export function SaScoreCard() {
  const [smiles, setSmiles] = useState(DEFAULT_SMILES);
  const [filterOn, setFilterOn] = useState(false);
  const [maxSa, setMaxSa] = useState("4.0");
  const [minSa, setMinSa] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<SAScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const parsedLines = useMemo(
    () =>
      smiles
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0),
    [smiles]
  );

  async function run() {
    if (parsedLines.length === 0) {
      setStatus("error");
      setError("Pass at least one SMILES string (one per line).");
      return;
    }
    if (parsedLines.length > MAX_MOLECULES) {
      setStatus("error");
      setError(`The web calculator handles at most ${MAX_MOLECULES} molecules per run.`);
      return;
    }
    setStatus("running");
    setError(null);
    setResult(null);
    try {
      const body: Record<string, unknown> = { smiles: parsedLines };
      if (filterOn) {
        const hi = Number.parseFloat(maxSa);
        const lo = minSa.trim() === "" ? undefined : Number.parseFloat(minSa);
        if (Number.isFinite(hi)) body.maxSa = hi;
        if (lo !== undefined && Number.isFinite(lo)) body.minSa = lo;
      }
      const data = await postSaScore(body);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "SA scoring failed.");
    }
  }

  async function copyCsv() {
    if (!result) return;
    const ok = await copyToClipboard(rowsToCsv(result.results));
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Copy failed — the clipboard refused the CSV.");
    }
  }

  function downloadCsv() {
    if (!result) return;
    const blob = new Blob([rowsToCsv(result.results)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "sa-scores.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  const sortedRows = result?.sorted ?? [];
  const summary = result?.summary;

  return (
    <Card data-slot="card" className="lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
          <Sigma className="size-5 shrink-0" aria-hidden="true" />
          Synthetic Accessibility score
        </CardTitle>
        <CardDescription>
          Ertl–Schuffenhauer SA score for every SMILES (lower = estimated
          easier to synthesise) — scored by{" "}
          <code className="font-mono text-xs">pharmaml.chem.sa_scoring</code>{" "}
          with the RDKit Contrib fragment database.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="sa-smiles">SMILES (one per line, max {MAX_MOLECULES})</Label>
          <textarea
            id="sa-smiles"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            rows={5}
            spellCheck={false}
            className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs leading-relaxed shadow-sm outline-none focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring"
            placeholder={"CCO\nCC(=O)Oc1ccccc1C(=O)O"}
          />
          <p className="text-xs text-muted-foreground">
            {parsedLines.length} molecule{parsedLines.length === 1 ? "" : "s"} ready —
            invalid rows are reported, never scored.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-3 rounded-md border bg-muted/40 p-3">
          <div className="flex items-center gap-2">
            <Switch
              id="sa-filter"
              checked={filterOn}
              onCheckedChange={setFilterOn}
              aria-label="Enable SA filtering"
            />
            <Label htmlFor="sa-filter" className="text-sm font-normal">
              Filter candidates
            </Label>
          </div>
          <div
            className={cn(
              "flex items-end gap-2 transition-opacity",
              filterOn ? "opacity-100" : "pointer-events-none opacity-40"
            )}
          >
            <div className="space-y-1">
              <Label htmlFor="sa-min" className="text-xs text-muted-foreground">
                Min SA
              </Label>
              <Input
                id="sa-min"
                value={minSa}
                onChange={(e) => setMinSa(e.target.value)}
                inputMode="decimal"
                placeholder="1.0"
                className="h-8 w-24 font-mono text-xs"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="sa-max" className="text-xs text-muted-foreground">
                Max SA
              </Label>
              <Input
                id="sa-max"
                value={maxSa}
                onChange={(e) => setMaxSa(e.target.value)}
                inputMode="decimal"
                placeholder="4.0"
                className="h-8 w-24 font-mono text-xs"
              />
            </div>
          </div>
          <p className="ml-auto hidden text-xs text-muted-foreground sm:block">
            Inclusive bounds; rejected candidates stay listed.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button onClick={run} disabled={status === "running"}>
            {status === "running" ? (
              <>
                <Loader2 className="size-4 animate-spin" aria-hidden="true" />
                Scoring…
              </>
            ) : (
              <>
                <FlaskConical className="size-4" aria-hidden="true" />
                Score molecules
              </>
            )}
          </Button>
          {result && (
            <>
              <Button variant="outline" onClick={copyCsv}>
                <Copy className="size-4" aria-hidden="true" />
                {copied ? "Copied" : "Copy CSV"}
              </Button>
              <Button variant="outline" onClick={downloadCsv}>
                <Download className="size-4" aria-hidden="true" />
                Download CSV
              </Button>
            </>
          )}
        </div>

        {status === "error" && (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden="true" />
            <AlertTitle>SA scoring failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">{result.nScored}</p>
                <p className="text-[11px] leading-tight text-muted-foreground">
                  scored
                </p>
              </div>
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">
                  {summary?.mean !== undefined ? summary.mean.toFixed(2) : "—"}
                </p>
                <p className="text-[11px] leading-tight text-muted-foreground">
                  mean ± {summary?.std !== undefined ? summary.std.toFixed(2) : "—"}
                </p>
              </div>
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">
                  {summary?.min !== undefined ? summary.min.toFixed(2) : "—"}–
                  {summary?.max !== undefined ? summary.max.toFixed(2) : "—"}
                </p>
                <p className="text-[11px] leading-tight text-muted-foreground">
                  range
                </p>
              </div>
              <div className="rounded-md border bg-muted/40 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">
                  {result.nInvalid}
                </p>
                <p className="text-[11px] leading-tight text-muted-foreground">
                  invalid
                </p>
              </div>
            </div>

            {filterOn && result.kept !== undefined && (
              <Alert>
                <CircleCheck className="size-4" aria-hidden="true" />
                <AlertTitle>
                  Filter kept {result.nKept} of {result.nTotal}
                </AlertTitle>
                <AlertDescription>
                  {result.nRejected} candidate
                  {result.nRejected === 1 ? "" : "s"} rejected by the SA bounds —
                  listed below with scores intact, nothing discarded.
                </AlertDescription>
              </Alert>
            )}

            <div className="max-h-96 overflow-y-auto rounded-md border">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-muted/95 backdrop-blur">
                <tr className="border-b text-left">
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">SMILES</th>
                  <th className="px-3 py-2 text-right font-medium">SA score</th>
                  <th className="px-3 py-2 font-medium">Band</th>
                  {filterOn && result.kept !== undefined && (
                    <th className="px-3 py-2 font-medium">Filter</th>
                  )}
                </tr>
                </thead>
                <tbody>
                  {sortedRows.map((row, index) => {
                    const full = result.results.find(
                      (r) => r.smiles === row.smiles
                    );
                    const rejected =
                      result.rejected?.some((r) => r.smiles === row.smiles) ?? false;
                    const kept =
                      result.kept?.some((r) => r.smiles === row.smiles) ?? false;
                    return (
                      <tr
                        key={`${row.smiles}-${index}`}
                        className="border-b last:border-0 hover:bg-muted/50"
                      >
                        <td className="px-3 py-1.5 tabular-nums text-muted-foreground">
                          {index + 1}
                        </td>
                        <td
                          className="max-w-[220px] truncate px-3 py-1.5 font-mono sm:max-w-[360px]"
                          title={full?.canonicalSmiles ?? row.smiles}
                        >
                          {row.smiles}
                          {full && !full.valid && (
                            <span className="ml-2 text-red-600 dark:text-red-400">
                              ({full.error})
                            </span>
                          )}
                        </td>
                        <td
                          className={cn(
                            "px-3 py-1.5 text-right font-mono tabular-nums",
                            row.score !== null ? scoreColor(row.score) : ""
                          )}
                        >
                          {row.score !== null ? row.score.toFixed(3) : "—"}
                        </td>
                        <td className="px-3 py-1.5 text-muted-foreground">
                          {row.score !== null ? bandLabel(row.score) : "invalid"}
                        </td>
                        {filterOn && result.kept !== undefined && (
                          <td className="px-3 py-1.5">
                            {rejected ? (
                              <span className="text-red-600 dark:text-red-400">
                                rejected
                              </span>
                            ) : kept ? (
                              <span className="text-emerald-600 dark:text-emerald-400">
                                kept
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <Alert className="border-l-4">
              <Info className="size-4 shrink-0" aria-hidden="true" />
              <AlertDescription>
                <span className="font-medium">{result.direction}</span>
                {" — "}
                {result.limitations} Method: {result.method} · RDKit{" "}
                {result.rdkitVersion} · {result.elapsedSeconds.toFixed(2)} s.
              </AlertDescription>
            </Alert>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
