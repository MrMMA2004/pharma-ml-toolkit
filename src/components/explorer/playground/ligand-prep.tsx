"use client";

import { useMemo, useState } from "react";
import {
  Boxes,
  CircleCheck,
  Eraser,
  Loader2,
  TriangleAlert,
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
import { CodeBlock } from "@/components/explorer/code-block";
import { NumberField } from "@/components/explorer/playground/calc-ui";
import { StatChip } from "@/components/explorer/stat-chip";
import { fmtNum } from "@/lib/format";
import type { LigandPrepResponse } from "@/lib/toolkit-types";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const DEFAULT_SMILES = [
  "CC(=O)OC1=CC=CC=C1C(=O)O", // aspirin
  "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", // caffeine
  "CC(C)Cc1ccc(cc1)C(C)C(=O)O", // ibuprofen
].join("\n");

/** Web-calculator cap: every ligand is a real meeko 3D preparation. */
const MAX_LIGANDS = 10;

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

type Status = "idle" | "running" | "done" | "error";

/** Docking ligand prep: SMILES list -> PDBQT ligands (real meeko 3D embed)
 *  -> grid box + GPF/DPF config texts — the pharmaml.docking preparation
 *  chain run with real RDKit on the server. */
export function LigandPrep() {
  const [smilesText, setSmilesText] = useState(DEFAULT_SMILES);
  const [padding, setPadding] = useState("4.0");
  const [spacing, setSpacing] = useState("0.375");
  const [gaRun, setGaRun] = useState("100");
  const [gaPop, setGaPop] = useState("150");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<LigandPrepResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);

  const smilesList = useMemo(
    () =>
      smilesText
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0 && !l.startsWith("#")),
    [smilesText]
  );

  const paddingNum = Number.parseFloat(padding);
  const spacingNum = Number.parseFloat(spacing);
  const gaRunNum = Number.parseInt(gaRun, 10);
  const gaPopNum = Number.parseInt(gaPop, 10);

  const paramsValid =
    Number.isFinite(paddingNum) &&
    paddingNum >= 0 &&
    paddingNum <= 20 &&
    Number.isFinite(spacingNum) &&
    spacingNum >= 0.1 &&
    spacingNum <= 2 &&
    Number.isInteger(gaRunNum) &&
    gaRunNum >= 1 &&
    gaRunNum <= 10000 &&
    Number.isInteger(gaPopNum) &&
    gaPopNum >= 1 &&
    gaPopNum <= 10000;

  const okLigands = result?.ligands.filter((l) => l.ok) ?? [];
  const selectedLigand = okLigands[Math.min(selected, Math.max(0, okLigands.length - 1))];

  function clearAll() {
    setSmilesText("");
    setResult(null);
    setError(null);
    setStatus("idle");
    setSelected(0);
  }

  async function prepare() {
    if (smilesList.length === 0) {
      setError("Paste at least one SMILES — one per line (# comments are ignored).");
      toast.error("Nothing to prepare");
      return;
    }
    if (smilesList.length > MAX_LIGANDS) {
      setError(
        `The web calculator handles up to ${MAX_LIGANDS} ligands per run (you pasted ${smilesList.length}). Use the desktop playground.py for full libraries.`
      );
      toast.error("Too many ligands");
      return;
    }
    if (!paramsValid) {
      setError(
        "Check the grid-box and GA parameters: padding 0–20 Å, spacing 0.1–2 Å, GA runs/population 1–10000."
      );
      toast.error("Invalid parameters");
      return;
    }
    setStatus("running");
    setError(null);
    try {
      const res = await fetch("/api/docking/prep", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          smiles: smilesList,
          padding: paddingNum,
          spacing: spacingNum,
          gaRun: gaRunNum,
          gaPopSize: gaPopNum,
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
      setResult(data as LigandPrepResponse);
      setSelected(0);
      setStatus("done");
      toast.success(
        `Prepared ${(data as LigandPrepResponse).nOk}/${(data as LigandPrepResponse).ligands.length} ligands as PDBQT`
      );
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Ligand preparation failed.";
      setError(message);
      toast.error("Ligand prep failed", { description: message });
    }
  }

  const running = status === "running";
  const box = result?.gridBox;

  return (
    <div className="space-y-4">
      {/* ------------------------------ Input ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Boxes className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Docking ligand prep</CardTitle>
              <CardDescription className="text-xs">
                SMILES → 3D-embedded PDBQT ligands → grid box + AutoDock GPF/DPF configs —
                mirroring <code className="font-mono">pharmaml.docking</code> (real RDKit + meeko
                on the server)
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pb-4">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Label htmlFor="prep-smiles" className="text-xs font-medium">
                Ligands — one SMILES per line
              </Label>
              <span className="text-[11px] text-muted-foreground">
                {smilesList.length}/{MAX_LIGANDS} · # comments ignored
              </span>
            </div>
            <Textarea
              id="prep-smiles"
              value={smilesText}
              onChange={(e) => setSmilesText(e.target.value)}
              spellCheck={false}
              className="min-h-24 resize-y font-mono text-xs leading-relaxed"
              placeholder={"CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C)C(=O)N2C"}
              aria-label="SMILES list, one per line"
            />
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <NumberField
              id="prep-padding"
              label="Padding"
              unit="Å"
              value={padding}
              onChange={setPadding}
              placeholder="4.0"
            />
            <NumberField
              id="prep-spacing"
              label="Spacing"
              unit="Å"
              value={spacing}
              onChange={setSpacing}
              placeholder="0.375"
            />
            <NumberField
              id="prep-garun"
              label="GA runs"
              value={gaRun}
              onChange={setGaRun}
              placeholder="100"
            />
            <NumberField
              id="prep-gapop"
              label="GA population"
              value={gaPop}
              onChange={setGaPop}
              placeholder="150"
            />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" onClick={prepare} disabled={running} className="h-10 gap-2">
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  Embedding 3D structures…
                </>
              ) : (
                <>
                  <Boxes className="size-4" aria-hidden />
                  Prepare ligands
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
              The box + GPF/DPF are computed from the first successfully prepared ligand — the
              campaign runner lives in the desktop <code className="font-mono">playground.py</code>
            </p>
          </div>
        </CardContent>
      </Card>

      {/* ------------------------------ Error ------------------------------ */}
      {error ? (
        <Alert variant="destructive">
          <TriangleAlert className="size-4" aria-hidden />
          <AlertTitle>Ligand prep failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {/* ------------------------------ Results ------------------------------ */}
      {result ? (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <StatChip
              icon={CircleCheck}
              value={`${result.nOk}/${result.ligands.length}`}
              label="ligands prepared"
              hint={`${result.rdkitVersion} · meeko 3D embedding`}
            />
            <StatChip
              icon={Boxes}
              value={selectedLigand ? String(selectedLigand.atoms) : "—"}
              label="atoms (selected)"
              hint={selectedLigand ? `${selectedLigand.smiles.slice(0, 60)}` : undefined}
            />
            <StatChip
              icon={TriangleAlert}
              value={String(result.nFailed)}
              label="failed SMILES"
              hint={result.nFailed === 0 ? "every SMILES prepared" : "dropped from the campaign"}
            />
            <StatChip
              icon={Boxes}
              value={`${fmtNum(result.elapsedSeconds, 1)} s`}
              label="prep time"
              hint="real RDKit + meeko run on the server"
            />
          </div>

          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="border-b bg-muted/30 py-3">
              <CardTitle className="text-sm">Prepared ligands</CardTitle>
              <CardDescription className="text-xs">
                Atom types feed the GPF <code className="font-mono">ligand_types</code> line; pick a
                ligand to view its PDBQT
              </CardDescription>
            </CardHeader>
            <CardContent className="pb-4 pt-0">
              <div className="max-h-72 overflow-y-auto thin-scrollbar">
                <Table>
                  <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur">
                    <TableRow>
                      <TableHead className="h-9 font-medium">Ligand</TableHead>
                      <TableHead className="h-9 text-right font-medium">Atoms</TableHead>
                      <TableHead className="h-9 text-right font-medium">Torsions</TableHead>
                      <TableHead className="h-9 font-medium">Atom types</TableHead>
                      <TableHead className="h-9 font-medium">PDBQT</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.ligands.map((lig, idx) => {
                      const okIdx = lig.ok ? okLigands.indexOf(lig) : -1;
                      return (
                        <TableRow
                          key={`${lig.name}-${idx}`}
                          className={`cursor-pointer hover:bg-muted/40 ${okIdx === selected ? "bg-primary/5" : ""}`}
                          onClick={() => lig.ok && setSelected(okIdx)}
                        >
                          <TableCell className="max-w-52 truncate py-1.5 font-mono text-xs" title={lig.smiles}>
                            {lig.smiles}
                          </TableCell>
                          <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                            {lig.ok ? lig.atoms : "—"}
                          </TableCell>
                          <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                            {lig.ok ? lig.torsdof : "—"}
                          </TableCell>
                          <TableCell className="py-1.5">
                            {lig.ok ? (
                              <span className="flex flex-wrap gap-1">
                                {lig.atomTypes.map((t) => (
                                  <Badge
                                    key={t}
                                    variant="outline"
                                    className="px-1.5 py-0 font-mono text-[10px]"
                                  >
                                    {t}
                                  </Badge>
                                ))}
                              </span>
                            ) : (
                              <span
                                className="text-[11px] text-destructive"
                                title={lig.error ?? undefined}
                              >
                                {(lig.error ?? "failed").slice(0, 40)}
                              </span>
                            )}
                          </TableCell>
                          <TableCell className="py-1.5">
                            {lig.ok ? (
                              <Button
                                type="button"
                                variant={okIdx === selected ? "default" : "outline"}
                                size="sm"
                                className="h-7 px-2 text-[11px]"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelected(okIdx);
                                }}
                              >
                                View
                              </Button>
                            ) : (
                              <span className="text-[11px] text-muted-foreground">—</span>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          {box ? (
            <Card className="py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <CardTitle className="text-sm">Grid box & docking configs</CardTitle>
                <CardDescription className="text-xs">
                  Box from the first prepared ligand (padding {padding} Å, spacing{" "}
                  {box.spacing} Å) — the same <code className="font-mono">compute_grid_box</code> /
                  <code className="font-mono">write_gpf</code> /
                  <code className="font-mono">write_dpf</code> chain as the CLI
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 pb-4 pt-3">
                <div className="grid gap-2 font-mono text-xs sm:grid-cols-2">
                  <div className="rounded-lg border bg-muted/30 p-3">
                    <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                      Box centre (Å)
                    </p>
                    <p className="tabular-nums">
                      x {fmtNum(box.centerX, 3)} · y {fmtNum(box.centerY, 3)} · z{" "}
                      {fmtNum(box.centerZ, 3)}
                    </p>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-3">
                    <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                      Box size (Å) · npts
                    </p>
                    <p className="tabular-nums">
                      {fmtNum(box.sizeX, 3)} × {fmtNum(box.sizeY, 3)} × {fmtNum(box.sizeZ, 3)} ·{" "}
                      {box.npts.join(" ")}
                    </p>
                  </div>
                </div>
                <div className="grid gap-3 lg:grid-cols-2">
                  {result.gpfText ? (
                    <CodeBlock
                      code={result.gpfText}
                      language="text"
                      fileName="r.gpf"
                      maxHeightClass="max-h-64"
                      showLineNumbers={false}
                    />
                  ) : null}
                  {result.dpfText ? (
                    <CodeBlock
                      code={result.dpfText}
                      language="text"
                      fileName="r.dpf"
                      maxHeightClass="max-h-64"
                      showLineNumbers={false}
                    />
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ) : null}

          {selectedLigand?.pdbqtText ? (
            <Card className="py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <CardTitle className="text-sm">
                      PDBQT — {selectedLigand.name}
                      <span className="ml-2 font-mono text-[11px] font-normal text-muted-foreground">
                        {selectedLigand.atoms} atoms · TORSDOF {selectedLigand.torsdof}
                      </span>
                    </CardTitle>
                    <CardDescription className="max-w-xl truncate font-mono text-[11px]">
                      {selectedLigand.smiles}
                    </CardDescription>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {okLigands.map((lig, idx) => (
                      <Button
                        key={lig.name}
                        type="button"
                        variant={idx === selected ? "default" : "outline"}
                        size="sm"
                        className="h-7 px-2 font-mono text-[11px]"
                        onClick={() => setSelected(idx)}
                        aria-pressed={idx === selected}
                      >
                        {lig.name.replace("ligand_", "L")}
                      </Button>
                    ))}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pb-4 pt-3">
                <CodeBlock
                  code={selectedLigand.pdbqtText}
                  language="text"
                  fileName={`${selectedLigand.name}.pdbqt`}
                  maxHeightClass="max-h-80"
                  showLineNumbers
                />
              </CardContent>
            </Card>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
