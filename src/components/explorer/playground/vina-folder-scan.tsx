"use client";

import { useMemo, useRef, useState } from "react";
import {
  CircleSlash,
  Eraser,
  FileDown,
  Files,
  FolderOpen,
  FolderSearch,
  Loader2,
  SquareArrowOutUpRight,
  Star,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { StatChip } from "@/components/explorer/stat-chip";
import type { VinaScanFile, VinaScanRow } from "@/lib/docking-configs";
import { scanVinaLogs } from "@/lib/docking-configs";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Limits — honest caps for the in-browser scan                        */
/* ------------------------------------------------------------------ */

const MAX_LOGS = 200;
const MAX_TOTAL_CHARS = 2_000_000; // ~2 MB of pasted log text

/* ------------------------------------------------------------------ */
/* Example scan — three scored logs plus one crashed run               */
/* ------------------------------------------------------------------ */

function vinaLog(poses: Array<[number, number, number, number]>): string {
  const rows = poses
    .map(
      ([mode, aff, lb, ub]) =>
        `${String(mode).padStart(4)} ${String(aff).padStart(10)} ${String(lb).padStart(10)} ${String(ub).padStart(10)}`
    )
    .join("\n");
  return `AutoDock Vina v1.2.5

Computing Vina grid ... done.
Performing docking (random seed: 42) ... done.

mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b. | rmsd u.b.
-----+------------+----------+----------
${rows}`;
}

const CRASHED_LOG = `AutoDock Vina v1.2.5

Computing Vina grid ... done.
Performing docking (random seed: 7) ...

^C

# process killed before the pose table was written — no modes scored`;

/** Example scan fixture: lig02 is the strongest binder at -8.35 kcal/mol
 * and the crashed log keeps its row. */
const EXAMPLE_SCAN: VinaScanFile[] = [
  {
    path: "lig01.log",
    content: vinaLog([
      [1, -7.9, 0.0, 0.0],
      [2, -7.4, 1.204, 2.31],
      [3, -7.1, 1.87, 3.11],
      [4, -6.9, 2.31, 3.88],
      [5, -6.6, 2.64, 4.41],
      [6, -6.4, 3.17, 5.2],
      [7, -6.2, 3.59, 5.79],
      [8, -6.0, 4.03, 6.46],
      [9, -5.8, 4.57, 7.32],
    ]),
  },
  {
    path: "lig02.log",
    content: vinaLog([
      [1, -8.35, 0.0, 0.0],
      [2, -8.1, 1.386, 2.554],
      [3, -7.9, 1.873, 3.112],
      [4, -7.6, 2.308, 3.882],
      [5, -7.4, 2.642, 4.408],
      [6, -7.1, 3.171, 5.203],
      [7, -6.9, 3.594, 5.789],
      [8, -6.7, 4.033, 6.455],
      [9, -6.5, 4.572, 7.319],
    ]),
  },
  {
    path: "lig03.log",
    content: vinaLog([
      [1, -6.2, 0.0, 0.0],
      [2, -5.9, 1.12, 2.2],
      [3, -5.7, 1.9, 3.4],
      [4, -5.4, 2.6, 4.1],
      [5, -5.1, 3.3, 5.0],
    ]),
  },
  { path: "crashed.log", content: CRASHED_LOG },
];

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

interface VinaFolderScanProps {
  /** Loads one log into the main parser (format = vina). */
  onOpenLog?: (name: string, content: string) => void;
}

/** Vina results-folder scan — the client-side mirror of
 *  pharmaml.docking.scan_vina_directory: pick a folder (or a set of
 *  logs), every .log is parsed in the browser, and the best affinity
 *  per ligand ranks strongest-binder-first with crashed logs kept. */
export function VinaFolderScan({ onOpenLog }: VinaFolderScanProps) {
  const [logs, setLogs] = useState<VinaScanFile[]>([]);
  const [reading, setReading] = useState(false);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const filesInputRef = useRef<HTMLInputElement>(null);

  const scan: VinaScanRow[] = useMemo(() => scanVinaLogs(logs), [logs]);
  const scored = scan.filter((r) => r.bestAffinity !== null);
  const crashed = scan.length - scored.length;
  const best = scored[0];

  async function readFiles(list: FileList | null) {
    if (!list || list.length === 0) return;
    const picked = Array.from(list);
    const candidates = picked.filter((f) => /\.(log|txt)$/i.test(f.name));
    const skipped = picked.length - candidates.length;
    if (candidates.length === 0) {
      toast.error("No Vina logs found", {
        description: "The scan reads .log and .txt files — pick the results folder again.",
      });
      return;
    }
    if (candidates.length > MAX_LOGS) {
      toast.error("Too many logs for the browser scan", {
        description: `Pick at most ${MAX_LOGS} logs (you chose ${candidates.length}). Use the Python package's campaign-inspection workflow for larger result trees; this browser demo has its own limits.`,
      });
      return;
    }
    setReading(true);
    try {
      const collected: VinaScanFile[] = [];
      let total = 0;
      for (const file of candidates) {
        const content = await file.text();
        total += content.length;
        if (total > MAX_TOTAL_CHARS) {
          toast.error("Scan payload too large", {
            description: `The browser scan reads at most ~2 MB of log text (exceeded at ${file.name}). Split the folder or use the Python campaign-inspection workflow.`,
          });
          return;
        }
        const rel = file.webkitRelativePath || file.name;
        const parts = rel.split("/").filter((p) => p.length > 0);
        // folder picks carry the picked root as the first segment —
        // strip it so `log` is relative to the scanned root
        const path = parts.length > 1 ? parts.slice(1).join("/") : (parts[0] ?? file.name);
        collected.push({ path, content });
      }
      setLogs(collected);
      toast.success(`Scanned ${collected.length} Vina log${collected.length === 1 ? "" : "s"}`, {
        description:
          skipped > 0
            ? `${skipped} non-log file${skipped === 1 ? "" : "s"} ignored · ranked best-binder-first`
            : "Ranked best-binder-first",
      });
    } catch (err) {
      toast.error("Could not read the logs", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setReading(false);
    }
  }

  function loadExample() {
    setLogs(EXAMPLE_SCAN.map((f) => ({ ...f })));
    toast.success("Example scan loaded", {
      description: "3 scored logs + 1 crashed run — the crashed row is kept, not dropped.",
    });
  }

  function downloadCsv() {
    const header = "ligand,log,best_affinity,n_poses";
    const lines = scan.map((r) =>
      [r.ligand, r.log, r.bestAffinity === null ? "" : String(r.bestAffinity), String(r.nPoses)]
        .map((cell) => (cell.includes(",") ? `"${cell}"` : cell))
        .join(",")
    );
    const blob = new Blob([`${[header, ...lines].join("\n")}\n`], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "vina_folder_scan.csv";
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Ranked table downloaded", { description: "vina_folder_scan.csv" });
  }

  const info =
    scan.length === 0
      ? "no Vina logs scanned yet"
      : best && best.bestAffinity !== null
        ? `${scan.length} Vina log${scan.length === 1 ? "" : "s"} · best ${(best.bestAffinity ?? 0).toFixed(2)} kcal/mol (${best.ligand}) — strongest binder first`
        : `${scan.length} Vina log${scan.length === 1 ? "" : "s"} · none scored a pose (check the logs for crashes)`;

  return (
    <Card className="overflow-hidden py-0 shadow-sm">
      <CardHeader className="border-b bg-muted/30 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <FolderSearch className="size-4.5" aria-hidden />
          </span>
          <div>
            <CardTitle className="text-base">Vina results-folder scan</CardTitle>
            <CardDescription className="text-xs">
              Every <code className="font-mono">.log</code> under a results tree — best affinity per
              ligand, strongest binder first, mirroring{" "}
              <code className="font-mono">pharmaml.docking.scan_vina_directory</code>
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
            <FolderOpen className="size-3" aria-hidden />
            Choose folder…
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 px-2.5 text-[11px]"
            onClick={() => filesInputRef.current?.click()}
          >
            <Files className="size-3" aria-hidden />
            Choose logs…
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 px-2.5 text-[11px]"
            onClick={loadExample}
          >
            <Star className="size-3" aria-hidden />
            Load example scan
          </Button>
          {logs.length > 0 ? (
            <>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 px-2.5 text-[11px]"
                onClick={downloadCsv}
              >
                <FileDown className="size-3" aria-hidden />
                Download CSV
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-8 gap-1.5 px-2.5 text-[11px] text-muted-foreground"
                onClick={() => setLogs([])}
              >
                <Eraser className="size-3" aria-hidden />
                Clear
              </Button>
            </>
          ) : null}
          {reading ? (
            <span className="inline-flex h-8 items-center gap-1.5 px-2 text-[11px] text-muted-foreground">
              <Loader2 className="size-3 animate-spin" aria-hidden />
              Reading logs…
            </span>
          ) : null}
        </div>

        <input
          ref={folderInputRef}
          type="file"
          multiple
          accept=".log,.txt,text/plain"
          className="sr-only"
          aria-label="Choose a Vina results folder"
          onChange={(e) => {
            void readFiles(e.target.files);
            e.target.value = "";
          }}
          {...({ webkitdirectory: "true", directory: "true" } as Record<string, string>)}
        />
        <input
          ref={filesInputRef}
          type="file"
          multiple
          accept=".log,.txt,text/plain"
          className="sr-only"
          aria-label="Choose Vina log files"
          onChange={(e) => {
            void readFiles(e.target.files);
            e.target.value = "";
          }}
        />

        {scan.length === 0 ? (
          <p className="rounded-lg border border-dashed bg-muted/20 px-3 py-2.5 text-xs leading-relaxed text-muted-foreground">
            Pick the folder your Vina runs wrote their logs to (or select the log files
            themselves) — every log is parsed right here in the browser, the table ranks the
            strongest binder first, and crashed runs keep their row instead of disappearing.
            Nothing is uploaded: the logs never leave this page.
          </p>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <StatChip icon={FolderSearch} value={String(scan.length)} label="logs scanned" />
              <StatChip
                icon={Star}
                value={best?.bestAffinity !== undefined ? (best.bestAffinity ?? NaN).toFixed(2) : "—"}
                label="best affinity (kcal/mol)"
              />
              <StatChip icon={SquareArrowOutUpRight} value={String(best?.nPoses ?? 0)} label="poses in best log" />
              <StatChip icon={CircleSlash} value={String(crashed)} label="crashed / unscored" />
            </div>

            <p role="status" className="text-xs font-medium">
              {info}
              {crashed > 0 ? (
                <span className="ml-1 font-normal text-muted-foreground">
                  · {crashed} log{crashed === 1 ? " has" : "s have"} no parseable pose table
                </span>
              ) : null}
            </p>

            <div className="thin-scrollbar max-h-96 overflow-auto rounded-lg border">
              <Table className="data-table">
                <TableHeader className="sticky top-0 z-10 bg-muted/80 backdrop-blur">
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="h-9">#</TableHead>
                    <TableHead className="h-9">Ligand</TableHead>
                    <TableHead className="h-9">Log</TableHead>
                    <TableHead className="h-9 text-right">Best affinity</TableHead>
                    <TableHead className="h-9 text-right">Poses</TableHead>
                    {onOpenLog ? <TableHead className="h-9 text-right">Open</TableHead> : null}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {scan.map((row, i) => (
                    <TableRow
                      key={`${row.log}-${i}`}
                      style={
                        i === 0 && row.bestAffinity !== null
                          ? { background: "color-mix(in oklch, var(--primary) 8%, transparent)" }
                          : undefined
                      }
                    >
                      <TableCell className="py-2 font-mono text-xs tabular-nums">
                        {i === 0 && row.bestAffinity !== null ? (
                          <Badge
                            className="bg-primary/15 text-primary hover:bg-primary/15"
                            variant="secondary"
                          >
                            best
                          </Badge>
                        ) : (
                          i + 1
                        )}
                      </TableCell>
                      <TableCell className="py-2 font-mono text-xs">{row.ligand}</TableCell>
                      <TableCell className="py-2 font-mono text-xs text-muted-foreground">
                        {row.log}
                      </TableCell>
                      <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                        {row.bestAffinity === null ? (
                          <span
                            className="text-muted-foreground"
                            title="No pose table parsed — the run crashed or was interrupted"
                          >
                            —
                          </span>
                        ) : (
                          row.bestAffinity.toFixed(2)
                        )}
                      </TableCell>
                      <TableCell className="py-2 text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {row.nPoses}
                      </TableCell>
                      {onOpenLog ? (
                        <TableCell className="py-2 text-right">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 gap-1 px-2 text-[11px]"
                            onClick={() => {
                              const file = logs.find((f) => f.path === row.log);
                              if (file) onOpenLog(row.ligand, file.content);
                            }}
                          >
                            <SquareArrowOutUpRight className="size-3" aria-hidden />
                            In parser
                          </Button>
                        </TableCell>
                      ) : null}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </>
        )}

        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Parity:{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
            pharmaml.docking.scan_vina_directory(results_root)
          </code>{" "}
          — columns ligand / log / best_affinity / n_poses, NaN rows sorted last. Logs are read
          locally in your browser (≤ {MAX_LOGS} files, ~2 MB) and never uploaded.
        </p>
      </CardContent>
    </Card>
  );
}
