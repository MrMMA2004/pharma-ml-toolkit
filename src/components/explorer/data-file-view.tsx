"use client";

import { useMemo } from "react";
import {
  Atom,
  Dna,
  HardDrive,
  ListOrdered,
  Table2,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { CodeBlock } from "@/components/explorer/code-block";
import { fmtBytes, fmtInt } from "@/lib/format";
import { isNumericColumn, type DataFileKind } from "@/lib/data-file";
import type {
  CsvPreviewResponse,
  DataFileResponse,
  PdbPreviewResponse,
  TextPreviewResponse,
} from "@/lib/toolkit-types";

/* ------------------------------------------------------------------ */
/* Header badges (kind-aware, mirrors the source-file badge idiom)     */
/* ------------------------------------------------------------------ */

/** Kind badge + the counts that matter for each preview kind. */
export function DataFileBadges({ data }: { data: DataFileResponse }) {
  if (data.kind === "csv") {
    return (
      <>
        <Badge variant="secondary" className="font-mono text-[10px] uppercase">
          CSV
        </Badge>
        <Badge variant="outline" className="tabular-nums">
          {fmtInt(data.rowCount)} rows
        </Badge>
        <Badge variant="outline" className="tabular-nums">
          {fmtInt(data.columns.length)} columns
        </Badge>
        <Badge variant="outline" className="tabular-nums">
          {fmtBytes(data.totalBytes)}
        </Badge>
      </>
    );
  }
  if (data.kind === "pdb") {
    return (
      <>
        <Badge variant="secondary" className="font-mono text-[10px] uppercase">
          PDB
        </Badge>
        <Badge variant="outline" className="tabular-nums">
          {fmtInt(data.atomCount)} atoms
        </Badge>
        <Badge variant="outline" className="tabular-nums">
          {fmtInt(data.residueCount)} residues
        </Badge>
        <Badge variant="outline" className="tabular-nums">
          {fmtBytes(data.totalBytes)}
        </Badge>
      </>
    );
  }
  return (
    <>
      <Badge variant="secondary" className="font-mono text-[10px] uppercase">
        Text
      </Badge>
      <Badge variant="outline" className="tabular-nums">
        {fmtInt(data.totalLineCount)} lines
      </Badge>
      <Badge variant="outline" className="tabular-nums">
        {fmtBytes(data.totalBytes)}
      </Badge>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Loading skeletons (table-shaped for CSV, chips+lines for PDB)       */
/* ------------------------------------------------------------------ */

/** Kind-aware loading placeholder shown while the preview streams in. */
export function DataFileSkeleton({ kind }: { kind: DataFileKind }) {
  if (kind === "csv") {
    return (
      <div
        className="overflow-hidden rounded-lg border bg-card"
        aria-label="Loading CSV preview"
      >
        <div className="flex gap-3 border-b bg-muted/50 px-3 py-2.5">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-3.5 w-20 shrink-0" />
          ))}
        </div>
        <div className="space-y-2.5 p-3">
          {Array.from({ length: 7 }).map((_, r) => (
            <div key={r} className="flex gap-3">
              {Array.from({ length: 6 }).map((_, c) => (
                <Skeleton
                  key={c}
                  className="h-3.5 shrink-0"
                  style={{ width: `${34 + ((r * 7 + c * 13) % 9) * 7}%`, maxWidth: 130 }}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (kind === "pdb") {
    return (
      <div className="space-y-3" aria-label="Loading PDB preview">
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-28 rounded-lg" />
          ))}
        </div>
        <div className="space-y-2.5">
          <Skeleton className="h-4 w-2/3" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-2.5" aria-label="Loading text preview">
      <Skeleton className="h-4 w-2/3" />
      <Skeleton className="h-4 w-5/6" />
      <Skeleton className="h-4 w-1/2" />
      <Skeleton className="h-4 w-3/4" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* CSV table                                                           */
/* ------------------------------------------------------------------ */

function CsvEmptyState({ data }: { data: CsvPreviewResponse }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed px-4 py-10 text-center">
      <Table2 className="size-6 text-muted-foreground/50" aria-hidden />
      <p className="text-sm text-muted-foreground">
        No data rows — this CSV contains only a header row.
      </p>
      <p className="font-mono text-xs text-muted-foreground/70">
        {data.columns.join(", ") || "(empty file)"}
      </p>
    </div>
  );
}

function CsvTableView({ data }: { data: CsvPreviewResponse }) {
  const fileName = data.path.split("/").pop() ?? data.path;
  const numericColumns = useMemo(
    () => data.columns.map((_, i) => isNumericColumn(data.previewRows, i)),
    [data]
  );
  const shown = data.previewRows.length;
  const more = Math.max(0, data.rowCount - shown);

  if (data.rowCount === 0) return <CsvEmptyState data={data} />;

  return (
    <div className="space-y-2">
      <div className="overflow-hidden rounded-lg border bg-card focus-within:ring-1 focus-within:ring-ring">
        {/* Inner scroll is the project convention: wide tables scroll
         * horizontally inside this container, never the page. */}
        <div className="thin-scrollbar max-h-[62vh] overflow-auto">
          <table
            className="w-full border-separate border-spacing-0 text-xs sm:text-[13px]"
            aria-label={`${fileName} — first ${fmtInt(shown)} of ${fmtInt(data.rowCount)} rows, ${fmtInt(data.columns.length)} columns`}
          >
            <caption className="sr-only">
              Preview of {fileName}: showing {fmtInt(shown)} of {fmtInt(data.rowCount)} rows and{" "}
              {fmtInt(data.columns.length)} columns ({fmtBytes(data.totalBytes)}).
            </caption>
            <thead>
              <tr>
                {/* Row-index column (the 100x-154 fold: the preview
                 * twin's one genuinely-new idea — a stable reference
                 * for "row N of the file" while scrolling). */}
                <th
                  scope="col"
                  aria-label="Row index"
                  className="sticky top-0 z-10 w-10 whitespace-nowrap border-b bg-muted px-2.5 py-2 text-right font-mono text-[11px] font-normal tabular-nums text-muted-foreground/70"
                >
                  #
                </th>
                {data.columns.map((column, i) => (
                  <th
                    key={`${column}-${i}`}
                    scope="col"
                    title={column}
                    className={
                      "sticky top-0 z-10 max-w-[24ch] truncate whitespace-nowrap border-b bg-muted px-2.5 py-2 font-mono text-[11px] font-semibold uppercase tracking-wide text-foreground/80 " +
                      (numericColumns[i] ? "text-right" : "text-left")
                    }
                  >
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.previewRows.map((row, r) => (
                <tr
                  key={r}
                  className="odd:bg-muted/50 transition-colors hover:bg-primary/10"
                >
                  <td className="whitespace-nowrap border-b border-border/60 px-2.5 py-1.5 text-right font-mono text-[11px] tabular-nums text-muted-foreground/60">
                    {r + 1}
                  </td>
                  {data.columns.map((_, c) => {
                    const value = row[c] ?? "";
                    return (
                      <td
                        key={c}
                        title={value}
                        className={
                          "max-w-[28ch] truncate whitespace-nowrap border-b border-border/60 px-2.5 py-1.5 font-mono text-muted-foreground " +
                          (numericColumns[c] ? "text-right tabular-nums" : "text-left")
                        }
                      >
                        {value === "" ? (
                          <span
                            className="text-muted-foreground/40"
                            aria-label="empty cell"
                          >
                            —
                          </span>
                        ) : (
                          value
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* Visible caption strip mirrors the screen-reader caption. */}
        <div className="border-t bg-muted/40 px-3 py-1.5 text-[11px] text-muted-foreground">
          Showing {fmtInt(shown)} of {fmtInt(data.rowCount)} rows ·{" "}
          {fmtInt(data.columns.length)} columns · {fmtBytes(data.totalBytes)}
        </div>
      </div>
      {data.truncated ? (
        <div className="flex justify-center">
          <Badge variant="secondary" className="gap-1 tabular-nums">
            +{fmtInt(more)} more rows in file
          </Badge>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* PDB structure view                                                  */
/* ------------------------------------------------------------------ */

function SummaryChip({
  icon: Icon,
  value,
  label,
}: {
  icon: LucideIcon;
  value: string;
  label: string;
}) {
  return (
    <div
      role="listitem"
      className="flex min-w-24 items-center gap-2 rounded-lg border bg-card px-3 py-2 shadow-sm"
    >
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
        <Icon className="size-4" aria-hidden />
      </span>
      <span className="min-w-0">
        <span className="block text-[10px] font-medium uppercase leading-tight tracking-wide text-muted-foreground">
          {label}
        </span>
        <span className="block text-sm font-semibold leading-tight tabular-nums">
          {value}
        </span>
      </span>
    </div>
  );
}

function PdbView({ data }: { data: PdbPreviewResponse }) {
  const fileName = data.path.split("/").pop() ?? data.path;
  const previewLineCount =
    data.previewText.length === 0 ? 0 : data.previewText.split("\n").length;
  const more = Math.max(0, data.totalLineCount - previewLineCount);

  return (
    <div className="space-y-3">
      <div
        className="flex flex-wrap gap-2"
        role="list"
        aria-label={`${fileName} structure summary`}
      >
        <SummaryChip icon={Atom} value={fmtInt(data.atomCount)} label="Atoms" />
        <SummaryChip icon={Dna} value={fmtInt(data.residueCount)} label="Residues" />
        <SummaryChip
          icon={ListOrdered}
          value={fmtInt(data.totalLineCount)}
          label="Lines"
        />
        <SummaryChip icon={HardDrive} value={fmtBytes(data.totalBytes)} label="Size" />
      </div>
      <CodeBlock
        code={data.previewText}
        language="text"
        fileName={fileName}
        showLineNumbers
        maxHeightClass="max-h-[70vh]"
      />
      {more > 0 ? (
        <div className="flex justify-center">
          <Badge variant="secondary" className="gap-1 tabular-nums">
            +{fmtInt(more)} more lines in file
          </Badge>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Plain-text view                                                     */
/* ------------------------------------------------------------------ */

function TextView({ data }: { data: TextPreviewResponse }) {
  const fileName = data.path.split("/").pop() ?? data.path;
  const shown =
    data.previewText.length === 0 ? 0 : data.previewText.split("\n").length;
  const more = Math.max(0, data.totalLineCount - shown);

  return (
    <div className="space-y-2">
      <CodeBlock
        code={data.previewText}
        language="text"
        fileName={fileName}
        showLineNumbers
        maxHeightClass="max-h-[70vh]"
      />
      {data.truncated && more > 0 ? (
        <div className="flex justify-center">
          <Badge variant="secondary" className="gap-1 tabular-nums">
            +{fmtInt(more)} more lines in file
          </Badge>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Dispatcher                                                          */
/* ------------------------------------------------------------------ */

/** Kind-aware viewer for shipped data files (CSV tables, PDB structures,
 *  docking logs). Rendered by the Code Explorer in place of the source
 *  CodeBlock when `/api/toolkit/file` returns a `kind`-tagged payload. */
export function DataFileView({ data }: { data: DataFileResponse }) {
  switch (data.kind) {
    case "csv":
      return <CsvTableView data={data} />;
    case "pdb":
      return <PdbView data={data} />;
    case "text":
      return <TextView data={data} />;
  }
}
