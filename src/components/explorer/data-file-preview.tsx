"use client";

/**
 * Data-file previews for the Code Explorer — the shipped .csv datasets
 * and .pdb structures the notebooks consume ("notebooks must work
 * entirely based on the files I give them"). Renders the structured
 * payloads from /api/toolkit/file:
 *  - CSV: shadcn-style table (sticky header, zebra, numeric right-align,
 *    truncated-cell tooltips) in its own scroll container;
 *  - PDB: summary chips (atoms / residues / lines / size) + the first
 *    records in the standard monospace CodeBlock chrome.
 */

import { useMemo } from "react";
import { AlignLeft, Atom, Boxes, HardDrive, Table2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CodeBlock } from "@/components/explorer/code-block";
import { cn } from "@/lib/utils";
import { fmtBytes, fmtInt } from "@/lib/format";
import type {
  CsvPreviewResponse,
  DataPreviewResponse,
  PdbPreviewResponse,
} from "@/lib/toolkit-types";

/** A cell counts as numeric when it parses to a finite number. */
function isNumericCell(cell: string): boolean {
  if (cell === "") return false;
  return Number.isFinite(Number(cell));
}

/** Per-column numeric detection over the preview rows (mirrors the
 *  ≥ 90 %-parseable heuristic from lib/csv.ts). */
function numericColumns(columns: string[], rows: string[][]): boolean[] {
  return columns.map((_, colIdx) => {
    let nonEmpty = 0;
    let parseable = 0;
    for (const row of rows) {
      const cell = row[colIdx] ?? "";
      if (cell === "") continue;
      nonEmpty += 1;
      if (isNumericCell(cell)) parseable += 1;
    }
    return nonEmpty > 0 && parseable / nonEmpty >= 0.9;
  });
}

function baseName(relPath: string): string {
  return relPath.split("/").pop() ?? relPath;
}

/* ------------------------------------------------------------------ */
/* Empty state                                                          */
/* ------------------------------------------------------------------ */

function EmptyPreview({ icon: Icon, title, hint }: { icon: typeof Table2; title: string; hint: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed px-4 py-10 text-center">
      <Icon className="size-6 text-muted-foreground/50" aria-hidden />
      <p className="text-sm font-medium">{title}</p>
      <p className="max-w-sm text-xs leading-relaxed text-muted-foreground">{hint}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* CSV table                                                            */
/* ------------------------------------------------------------------ */

function CsvPreview({ data }: { data: CsvPreviewResponse }) {
  const fileName = baseName(data.path);
  const numeric = useMemo(
    () => numericColumns(data.columns, data.previewRows),
    [data.columns, data.previewRows]
  );

  if (data.columns.length === 0) {
    return (
      <EmptyPreview
        icon={Table2}
        title="No data rows"
        hint="This CSV contains no parseable records — the file is empty or holds only blank lines. Pick another file in the tree to keep exploring."
      />
    );
  }

  return (
    <div className="space-y-2">
      {/* Own scroll container (project convention): the table scrolls
       * horizontally on narrow screens and vertically past ~70vh, while
       * the sticky header stays pinned. border-separate is required for
       * the sticky header borders to survive scrolling. */}
      <div className="thin-scrollbar max-h-[70vh] overflow-auto rounded-lg border bg-card">
        <table
          className="data-table w-full border-separate border-spacing-0 text-sm"
          aria-label={`CSV preview of ${fileName}, first ${data.previewRows.length} of ${data.rowCount} rows`}
        >
          <TableHeader className="sticky top-0 z-10">
            <TableRow className="bg-muted/95 backdrop-blur hover:bg-muted/95">
              <TableHead
                scope="col"
                aria-label="Row index"
                className="h-9 w-10 border-b bg-muted/95 px-2 text-right font-mono text-[10px] font-normal tabular-nums text-muted-foreground"
              >
                #
              </TableHead>
              {data.columns.map((col) => (
                <TableHead
                  scope="col"
                  key={col}
                  title={col}
                  className="h-9 max-w-[20ch] truncate border-b bg-muted/95 px-2 font-mono text-[11px]"
                >
                  {col}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.previewRows.map((row, rowIdx) => (
              <TableRow key={rowIdx}>
                <TableCell className="border-b border-border/60 py-1.5 px-2 text-right font-mono text-[10px] tabular-nums text-muted-foreground/60">
                  {rowIdx + 1}
                </TableCell>
                {data.columns.map((col, colIdx) => {
                  const cell = row[colIdx] ?? "";
                  return (
                    <TableCell
                      key={col}
                      title={cell}
                      className={cn(
                        "max-w-[28ch] truncate border-b border-border/60 py-1.5 px-2 font-mono text-xs",
                        numeric[colIdx] && "text-right tabular-nums"
                      )}
                    >
                      {cell === "" ? (
                        <span className="text-muted-foreground/40" aria-label="empty cell">
                          —
                        </span>
                      ) : (
                        cell
                      )}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </table>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 px-1">
        <p className="font-mono text-[11px] leading-snug text-muted-foreground">
          showing {fmtInt(data.previewRows.length)} of {fmtInt(data.rowCount)} rows ·{" "}
          {fmtInt(data.columns.length)} columns · {fmtBytes(data.totalBytes)}
        </p>
        {data.truncated ? (
          <Badge variant="outline" className="tabular-nums">
            +{fmtInt(data.rowCount - data.previewRows.length)} more rows in file
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* PDB structure preview                                                */
/* ------------------------------------------------------------------ */

function PdbPreview({ data }: { data: PdbPreviewResponse }) {
  const fileName = baseName(data.path);
  const previewLineCount = useMemo(
    () => (data.previewText.length === 0 ? 0 : data.previewText.split("\n").length),
    [data.previewText]
  );

  if (data.totalLineCount === 0) {
    return (
      <EmptyPreview
        icon={Atom}
        title="Empty structure file"
        hint="This PDB contains no records. Pick another file in the tree to keep exploring."
      />
    );
  }

  return (
    <div className="space-y-2">
      <div
        className="flex flex-wrap items-center gap-1.5"
        aria-label="Structure summary"
      >
        <Badge
          variant="outline"
          className="gap-1.5 tabular-nums"
          title="ATOM + HETATM records in the file"
        >
          <Atom className="size-3 text-primary" aria-hidden />
          {fmtInt(data.atomCount)} atoms
        </Badge>
        <Badge
          variant="outline"
          className="gap-1.5 tabular-nums"
          title="Distinct chain / residue numbers across ATOM/HETATM records"
        >
          <Boxes className="size-3 text-primary" aria-hidden />
          {fmtInt(data.residueCount)} residues
        </Badge>
        <Badge
          variant="outline"
          className="gap-1.5 tabular-nums"
          title="Total records (lines) in the file"
        >
          <AlignLeft className="size-3 text-primary" aria-hidden />
          {fmtInt(data.totalLineCount)} lines
        </Badge>
        <Badge variant="outline" className="gap-1.5 tabular-nums" title="File size">
          <HardDrive className="size-3 text-primary" aria-hidden />
          {fmtBytes(data.totalBytes)}
        </Badge>
      </div>
      <CodeBlock
        code={data.previewText}
        language="text"
        fileName={fileName}
        showLineNumbers
        maxHeightClass="max-h-[70vh]"
      />
      <p className="px-1 font-mono text-[11px] leading-snug text-muted-foreground">
        showing {fmtInt(previewLineCount)} of {fmtInt(data.totalLineCount)} lines ·{" "}
        {fmtBytes(data.totalBytes)}
        {data.truncated ? " · full structure ships in the downloadable archive" : ""}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Entry point                                                          */
/* ------------------------------------------------------------------ */

export function DataFilePreview({ data }: { data: DataPreviewResponse }) {
  return data.kind === "csv" ? <CsvPreview data={data} /> : <PdbPreview data={data} />;
}
