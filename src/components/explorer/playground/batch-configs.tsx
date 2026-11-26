"use client";

import { useMemo, useRef, useState } from "react";
import {
  CircleCheck,
  Download,
  Eraser,
  FolderTree,
  Grid3X3,
  Loader2,
  Star,
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
import { CodeBlock } from "@/components/explorer/code-block";
import { NumberField } from "@/components/explorer/playground/calc-ui";
import {
  DEFAULT_RECEPTOR_TYPES,
  groupCampaignFiles,
  writeConfigBatch,
} from "@/lib/docking-configs";
import type { BatchConfigRow, CampaignFolder } from "@/lib/docking-configs";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Limits — honest caps for the in-browser batch                       */
/* ------------------------------------------------------------------ */

const MAX_PDBQT = 200;
const MAX_TOTAL_CHARS = 4_000_000; // ~4 MB of PDBQT text

/* ------------------------------------------------------------------ */
/* Example campaign — one folder per ligand, as the desktop walk sees   */
/* ------------------------------------------------------------------ */

const ASPIRIN_PDBQT = `REMARK  Name = 01_aspirin
REMARK  SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.12 C
ATOM      2  O   LIG A   1       2.000   0.000   0.000  1.00  0.00    -0.31 OA
ENDROOT
TORSDOF 1
`;

const NAPROXEN_PDBQT = `REMARK  Name = 03_naproxen
REMARK  SMILES: COC1=CC2=C(C=C1)C(C)CC2=C(C)C(=O)O
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.05 C
ATOM      2  C   LIG A   1       3.000   1.000   0.500  1.00  0.00     0.02 C
ATOM      3  O   LIG A   1       5.500   0.200   1.500  1.00  0.00    -0.40 OA
ENDROOT
TORSDOF 2
`;

const RECEPTOR_PDBQT = `ATOM      1  CA  REC A   1      10.000  10.000  10.000  1.00  0.00     0.10 C
ATOM      2  N   REC A   1      12.300  10.500   9.800  1.00  0.00    -0.34 N
`;

/** 03_naproxen's ligand lives next to the receptor file — the batch must
 *  exclude r.pdbqt from the ligand search, exactly like the desktop. */
const EXAMPLE_CAMPAIGN: CampaignFolder[] = [
  {
    folder: "01_aspirin",
    files: [{ name: "l.pdbqt", content: ASPIRIN_PDBQT }],
  },
  { folder: "02_empty", files: [] },
  {
    folder: "03_naproxen",
    files: [
      { name: "r.pdbqt", content: RECEPTOR_PDBQT },
      { name: "l.pdbqt", content: NAPROXEN_PDBQT },
    ],
  },
];

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

interface BatchParams {
  padding: string;
  npts: string;
  spacing: string;
  receptorFile: string;
  gaRun: string;
  gaPop: string;
  receptorTypes: string;
}

const DEFAULT_PARAMS: BatchParams = {
  padding: "4.0",
  npts: "60 60 60",
  spacing: "0.375",
  receptorFile: "r.pdbqt",
  gaRun: "100",
  gaPop: "150",
  receptorTypes: "",
};

/** Batch GPF/DPF for a campaign — the client-side mirror of
 *  pharmaml.docking.write_config_batch: pick the campaign root (one
 *  sub-folder per ligand), every sub-folder's reference ligand gets its
 *  own grid box and r.gpf / r.dpf pair, generated in the browser. */
export function BatchConfigs() {
  const [params, setParams] = useState<BatchParams>(DEFAULT_PARAMS);
  const [folders, setFolders] = useState<CampaignFolder[] | null>(null);
  const [reading, setReading] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  function set<K extends keyof BatchParams>(key: K, value: string) {
    setParams((p) => ({ ...p, [key]: value }));
  }

  /* ---- validation: the desktop's parse_float / parse_int_triple ---- */
  const paddingNum = Number.parseFloat(params.padding);
  const spacingNum = Number.parseFloat(params.spacing);
  const gaRunNum = Number.parseInt(params.gaRun, 10);
  const gaPopNum = Number.parseInt(params.gaPop, 10);
  const nptsParts = params.npts.trim().split(/[\s,]+/).filter((p) => p.length > 0);
  const nptsNums = nptsParts.map((p) => Number.parseInt(p, 10));
  const nptsValid =
    nptsNums.length === 3 && nptsNums.every((v) => Number.isInteger(v) && v >= 1);

  const paddingError =
    !Number.isFinite(paddingNum) || paddingNum < 0 ? "padding must be ≥ 0" : null;
  const nptsError = !nptsValid ? "three integers ≥ 1 (x y z)" : null;
  const spacingError =
    !Number.isFinite(spacingNum) || spacingNum < 0.01 ? "spacing must be ≥ 0.01" : null;
  const receptorError =
    params.receptorFile.trim().length === 0 ? "pick a file name (e.g. r.pdbqt)" : null;
  const gaRunError =
    !Number.isInteger(gaRunNum) || gaRunNum < 1 ? "ga_run must be ≥ 1" : null;
  const gaPopError =
    !Number.isInteger(gaPopNum) || gaPopNum < 1 ? "population must be ≥ 1" : null;
  const typesError =
    params.receptorTypes.trim().length > 0 &&
    !params.receptorTypes
      .trim()
      .split(/[\s,]+/)
      .every((t) => /^[A-Za-z]{1,3}$/.test(t))
      ? "space-separated atom types (e.g. A C HD N Ni OA SA)"
      : null;

  const paramsValid =
    !paddingError &&
    !nptsError &&
    !spacingError &&
    !receptorError &&
    !gaRunError &&
    !gaPopError &&
    !typesError;

  /* ---- the batch itself: recomputed on every param keystroke ------- */
  const report: BatchConfigRow[] = useMemo(() => {
    if (!folders || !paramsValid) return [];
    return writeConfigBatch(folders, {
      receptorFile: params.receptorFile.trim(),
      padding: paddingNum,
      npts: [nptsNums[0] ?? 60, nptsNums[1] ?? 60, nptsNums[2] ?? 60],
      spacing: spacingNum,
      receptorTypes: params.receptorTypes.trim() || null,
      gaRun: gaRunNum,
      gaPopSize: gaPopNum,
    });
  }, [
    folders,
    paramsValid,
    params.receptorFile,
    params.receptorTypes,
    params.padding,
    params.npts,
    params.spacing,
    params.gaRun,
    params.gaPop,
  ]);

  const okCount = report.filter((r) => r.status === "ok").length;
  const noLigand = report.filter((r) => r.status === "no ligand PDBQT").length;
  const failed = report.filter((r) => r.status.startsWith("failed")).length;
  const selectedRow = report.find((r) => r.folder === selected) ?? null;

  async function readCampaign(list: FileList | null) {
    if (!list || list.length === 0) return;
    const picked = Array.from(list);
    const candidates = picked.filter((f) => /\.pdbqt$/i.test(f.name));
    if (candidates.length === 0) {
      toast.error("No PDBQT files found", {
        description: "The batch walks sub-folders of ligand .pdbqt files — pick the campaign root.",
      });
      return;
    }
    if (candidates.length > MAX_PDBQT) {
      toast.error("Too many PDBQT files for the browser batch", {
        description: `Pick at most ${MAX_PDBQT} files (you chose ${candidates.length}). Use the CLI (pharmaml docking prep-protein / prep-ligands) for full campaigns.`,
      });
      return;
    }
    setReading(true);
    try {
      const files: Array<{ path: string; content: string }> = [];
      let total = 0;
      for (const file of candidates) {
        const content = await file.text();
        total += content.length;
        if (total > MAX_TOTAL_CHARS) {
          toast.error("Campaign payload too large", {
            description: `The browser batch reads at most ~4 MB of PDBQT text (exceeded at ${file.name}).`,
          });
          return;
        }
        files.push({ path: file.webkitRelativePath || file.name, content });
      }
      const grouped = groupCampaignFiles(files);
      if (grouped.length === 0) {
        toast.error("No ligand sub-folders found", {
          description:
            "The campaign root must hold one sub-folder per ligand (01_aspirin/l.pdbqt, …) — files at the root are not ligand folders.",
        });
        return;
      }
      setFolders(grouped);
      setSelected(null);
      toast.success(`Campaign loaded: ${grouped.length} ligand folder${grouped.length === 1 ? "" : "s"}`, {
        description: "Configs recompute live as you tune the parameters.",
      });
    } catch (err) {
      toast.error("Could not read the campaign", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setReading(false);
    }
  }

  function loadExample() {
    setFolders(EXAMPLE_CAMPAIGN.map((f) => ({ ...f, files: f.files.map((x) => ({ ...x })) })));
    setSelected("01_aspirin");
    toast.success("Example campaign loaded", {
      description: "3 folders — ok, no ligand PDBQT, and receptor-file exclusion.",
    });
  }

  function downloadText(fileName: string, content: string, label: string) {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(label, { description: fileName });
  }

  function downloadBundle() {
    const chunks: string[] = [];
    for (const row of report) {
      if (row.gpfText) chunks.push(`=== ${row.folder}/r.gpf ===\n${row.gpfText}`);
      if (row.dpfText) chunks.push(`=== ${row.folder}/r.dpf ===\n${row.dpfText}`);
    }
    if (chunks.length === 0) return;
    downloadText(
      "campaign_configs.txt",
      `${chunks.join("\n")}\n`,
      `Bundle downloaded (${okCount} pair${okCount === 1 ? "" : "s"})`
    );
  }

  return (
    <Card className="overflow-hidden py-0 shadow-sm">
      <CardHeader className="border-b bg-muted/30 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Grid3X3 className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">Batch GPF/DPF for a campaign</CardTitle>
            <CardDescription className="text-xs">
              Every sub-folder&apos;s ligand PDBQT gets its own grid box and{" "}
              <code className="font-mono">r.gpf</code>/<code className="font-mono">r.dpf</code>{" "}
              pair — mirroring{" "}
              <code className="font-mono">pharmaml.docking.write_config_batch</code>
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pb-4">
        <div className="flex flex-wrap gap-1.5">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 px-2.5 text-[11px]"
            onClick={() => folderInputRef.current?.click()}
          >
            <FolderTree className="size-3" aria-hidden />
            Choose campaign folder…
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 px-2.5 text-[11px]"
            onClick={loadExample}
          >
            <Star className="size-3" aria-hidden />
            Load example campaign
          </Button>
          {report.length > 0 && okCount > 0 ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 gap-1.5 px-2.5 text-[11px]"
              onClick={downloadBundle}
            >
              <Download className="size-3" aria-hidden />
              Download bundle
            </Button>
          ) : null}
          {folders ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 gap-1.5 px-2.5 text-[11px] text-muted-foreground"
              onClick={() => {
                setFolders(null);
                setSelected(null);
              }}
            >
              <Eraser className="size-3" aria-hidden />
              Clear
            </Button>
          ) : null}
          {reading ? (
            <span className="inline-flex h-8 items-center gap-1.5 px-2 text-[11px] text-muted-foreground">
              <Loader2 className="size-3 animate-spin" aria-hidden />
              Reading campaign…
            </span>
          ) : null}
        </div>

        <input
          ref={folderInputRef}
          type="file"
          multiple
          accept=".pdbqt,text/plain"
          className="sr-only"
          aria-label="Choose a campaign folder (one sub-folder per ligand)"
          onChange={(e) => {
            void readCampaign(e.target.files);
            e.target.value = "";
          }}
          {...({ webkitdirectory: "true", directory: "true" } as Record<string, string>)}
        />

        {/* ---- parameters: the desktop grid card's fields ------------- */}
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <NumberField
            id="batch-padding"
            label="padding"
            unit="Å"
            value={params.padding}
            onChange={(v) => set("padding", v)}
            error={folders ? paddingError : null}
          />
          <div className="space-y-1.5">
            <Label htmlFor="batch-npts" className="text-xs font-medium">
              npts (x y z)
            </Label>
            <Input
              id="batch-npts"
              value={params.npts}
              onChange={(e) => set("npts", e.target.value)}
              aria-invalid={folders && !nptsValid ? true : undefined}
              className={`h-10 font-mono text-sm tabular-nums ${
                folders && !nptsValid ? "border-destructive/60" : ""
              }`}
            />
            {folders && !nptsValid ? (
              <p className="text-[11px] text-destructive">{nptsError}</p>
            ) : null}
          </div>
          <NumberField
            id="batch-spacing"
            label="spacing"
            unit="Å"
            value={params.spacing}
            onChange={(v) => set("spacing", v)}
            error={folders ? spacingError : null}
          />
          <div className="space-y-1.5">
            <Label htmlFor="batch-receptor" className="text-xs font-medium">
              receptor file
            </Label>
            <Input
              id="batch-receptor"
              value={params.receptorFile}
              onChange={(e) => set("receptorFile", e.target.value)}
              aria-invalid={folders && receptorError ? true : undefined}
              className={`h-10 font-mono text-sm ${
                folders && receptorError ? "border-destructive/60" : ""
              }`}
            />
            {folders && receptorError ? (
              <p className="text-[11px] text-destructive">{receptorError}</p>
            ) : null}
          </div>
          <NumberField
            id="batch-ga-run"
            label="ga_run"
            value={params.gaRun}
            onChange={(v) => set("gaRun", v)}
            error={folders ? gaRunError : null}
          />
          <NumberField
            id="batch-ga-pop"
            label="ga population"
            value={params.gaPop}
            onChange={(v) => set("gaPop", v)}
            error={folders ? gaPopError : null}
          />
          <div className="space-y-1.5 sm:col-span-2">
            <Label htmlFor="batch-types" className="text-xs font-medium">
              receptor types{" "}
              <span className="text-[10px] font-normal text-muted-foreground">
                (blank = {DEFAULT_RECEPTOR_TYPES})
              </span>
            </Label>
            <Input
              id="batch-types"
              value={params.receptorTypes}
              onChange={(e) => set("receptorTypes", e.target.value)}
              placeholder="A C HD N Ni OA SA — the Ni-bearing case"
              aria-invalid={folders && typesError ? true : undefined}
              className={`h-10 font-mono text-sm ${
                folders && typesError ? "border-destructive/60" : ""
              }`}
            />
            {folders && typesError ? (
              <p className="text-[11px] text-destructive">{typesError}</p>
            ) : null}
          </div>
        </div>

        {folders && !paramsValid ? (
          <Alert variant="destructive">
            <TriangleAlert className="size-4" aria-hidden />
            <AlertTitle>Parameters need fixing</AlertTitle>
            <AlertDescription>
              The batch waits until every field is valid — same rules the desktop&apos;s
              parse_float / parse_int_triple apply.
            </AlertDescription>
          </Alert>
        ) : null}

        {!folders ? (
          <p className="rounded-lg border border-dashed bg-muted/20 px-3 py-2.5 text-xs leading-relaxed text-muted-foreground">
            Pick the campaign root — one sub-folder per ligand, each holding its prepared{" "}
            <code className="font-mono">.pdbqt</code> (the receptor file is excluded from the
            ligand search). The grid box, <code className="font-mono">r.gpf</code> and{" "}
            <code className="font-mono">r.dpf</code> texts are computed right here in your
            browser and recompute live as you tune the parameters — the desktop writes the
            pair to disk, the web hands you the same bytes to download. Set receptor types for
            Ni-bearing receptors (the <code className="font-mono">gpferreceptorni.py</code>{" "}
            case).
          </p>
        ) : report.length > 0 ? (
          <>
            <p role="status" className="text-xs font-medium">
              {`configured ${okCount}/${report.length} folders`}
              {noLigand > 0 ? ` · ${noLigand} without a ligand` : ""}
              {failed > 0 ? ` · ${failed} failed` : ""}
              <span className="ml-1 font-normal text-muted-foreground">
                — click a row to preview and download its config pair
              </span>
            </p>

            <div className="thin-scrollbar max-h-96 overflow-auto rounded-lg border">
              <Table className="data-table">
                <TableHeader className="sticky top-0 z-10 bg-muted/80 backdrop-blur">
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="h-9">Folder</TableHead>
                    <TableHead className="h-9">Ligand</TableHead>
                    <TableHead className="h-9">Status</TableHead>
                    <TableHead className="h-9 text-right">Box centre (x y z)</TableHead>
                    <TableHead className="h-9 text-right">npts</TableHead>
                    <TableHead className="h-9">Ligand types</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.map((row) => (
                    <TableRow
                      key={row.folder}
                      onClick={() => setSelected(row.folder)}
                      className="cursor-pointer"
                      aria-selected={selected === row.folder}
                    >
                      <TableCell className="py-2 font-mono text-xs">{row.folder}</TableCell>
                      <TableCell className="py-2 font-mono text-xs text-muted-foreground">
                        {row.ligand || "—"}
                      </TableCell>
                      <TableCell className="py-2">
                        {row.status === "ok" ? (
                          <Badge className="gap-1 border-emerald-200 bg-emerald-50 text-emerald-600 hover:bg-emerald-50 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-400">
                            <CircleCheck className="size-3" aria-hidden />
                            ok
                          </Badge>
                        ) : row.status === "no ligand PDBQT" ? (
                          <Badge variant="secondary" className="text-amber-600 dark:text-amber-400">
                            no ligand PDBQT
                          </Badge>
                        ) : (
                          <Badge
                            variant="secondary"
                            className="border-destructive/30 bg-destructive/10 text-destructive"
                          >
                            {row.status}
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                        {row.centre || "—"}
                      </TableCell>
                      <TableCell className="py-2 text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {row.npts ? row.npts.join(" ") : "—"}
                      </TableCell>
                      <TableCell className="py-2 font-mono text-xs text-muted-foreground">
                        {row.ligandTypes.length > 0 ? row.ligandTypes.join(" ") : "—"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {selectedRow && selectedRow.status === "ok" ? (
              <div className="space-y-3 rounded-lg border bg-muted/30 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-xs font-medium">
                    Config pair for{" "}
                    <code className="font-mono">{selectedRow.folder}</code>
                    <span className="ml-1 font-normal text-muted-foreground">
                      — grid box from {selectedRow.ligand}
                    </span>
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedRow.gpfText ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 gap-1.5 px-2 text-[11px]"
                        onClick={() =>
                          downloadText(
                            "r.gpf",
                            selectedRow.gpfText ?? "",
                            "GPF downloaded"
                          )
                        }
                      >
                        <Download className="size-3" aria-hidden />
                        r.gpf
                      </Button>
                    ) : null}
                    {selectedRow.dpfText ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 gap-1.5 px-2 text-[11px]"
                        onClick={() =>
                          downloadText(
                            "r.dpf",
                            selectedRow.dpfText ?? "",
                            "DPF downloaded"
                          )
                        }
                      >
                        <Download className="size-3" aria-hidden />
                        r.dpf
                      </Button>
                    ) : null}
                  </div>
                </div>
                {selectedRow.gpfText ? (
                  <CodeBlock
                    code={selectedRow.gpfText}
                    language="text"
                    fileName={`${selectedRow.folder}/r.gpf`}
                    showLineNumbers={false}
                    maxHeightClass="max-h-56"
                  />
                ) : null}
                {selectedRow.dpfText ? (
                  <CodeBlock
                    code={selectedRow.dpfText}
                    language="text"
                    fileName={`${selectedRow.folder}/r.dpf`}
                    showLineNumbers={false}
                    maxHeightClass="max-h-56"
                  />
                ) : null}
              </div>
            ) : null}
          </>
        ) : (
          <Alert>
            <TriangleAlert className="size-4" aria-hidden />
            <AlertTitle>No ligand folders</AlertTitle>
            <AlertDescription>
              The chosen folder holds no sub-folders with PDBQT ligands — one bad pick must
              never sink the campaign, but there is nothing to configure here.
            </AlertDescription>
          </Alert>
        )}

        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Parity:{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
            pharmaml.docking.write_config_batch(campaign_root, receptor_file=…,
            receptor_types=…)
          </code>{" "}
          — statuses ok / no ligand PDBQT / failed: …; the receptor file is excluded from the
          ligand search. PDBQT files are read locally in your browser (≤ {MAX_PDBQT} files,
          ~4 MB) and never uploaded.
        </p>
      </CardContent>
    </Card>
  );
}
