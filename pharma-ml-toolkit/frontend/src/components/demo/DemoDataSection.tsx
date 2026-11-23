/**
 * The shared Demo Data section — part16 §8 (Phase 5): every relevant
 * workflow gets a compact, accessible sample-data entry point.
 *
 * One component, mounted per workspace with the workspace's own
 * tag: the backend catalog (over the REAL shipped `examples/data`
 * files — never invented datasets) is fetched once and rendered as
 * a compact list; every entry offers preview-before-load (the RAW
 * file head — a preview registers nothing) and the one-click load
 * whose payload kind decides the consumer:
 *
 * * `tabular` → a FRESH dataset record (the same import path as a
 *   manual upload; never an overwrite) — `onDatasetLoaded` lets the
 *   host page refresh its dataset registry so the record appears in
 *   its selector immediately;
 * * `series` → the times/conc arrays (PK/PD and plotting forms);
 * * `library` → the name+SMILES columns (chemistry, generative
 *   corpora, docking ligand lists);
 * * `text_file` → the raw file content (docking logs, structures),
 *   with a copy button for paste lanes.
 *
 * The load itself always runs through an explicit confirm step that
 * states what will be imported (and that nothing user-owned is
 * replaced), per the directive's confirmation requirement.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Check, ClipboardCopy, FlaskConical, LoaderCircle, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  samplesApi,
  type SampleLoadResponse,
  type SamplePreview,
  type SampleView,
} from "@/services/samples/client";

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function formatBytes(size: number | null): string {
  if (size === null) {
    return "—";
  }
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KiB`;
  }
  return `${(size / (1024 * 1024)).toFixed(2)} MiB`;
}

const KIND_LABEL: Record<SampleView["kind"], string> = {
  tabular: "table → dataset",
  series: "series → PK/PD arrays",
  library: "library → SMILES list",
  text_file: "file → content",
};

export interface DemoDataSectionProps {
  /** The workspace tag (filters the catalog); omit for the global list. */
  workflow?: string;
  /** A heading override (defaults to "Demo data"). */
  heading?: string;
  /** Fires after a tabular sample registers a dataset record. */
  onDatasetLoaded?: () => void;
  /** Fires with the series payload (times/conc arrays). */
  onSeriesLoaded?: (payload: Extract<SampleLoadResponse, { kind: "series" }>) => void;
  /** Fires with the library payload (name+SMILES columns). */
  onLibraryLoaded?: (payload: Extract<SampleLoadResponse, { kind: "library" }>) => void;
  /** Fires with the text-file payload (raw content). */
  onTextLoaded?: (payload: Extract<SampleLoadResponse, { kind: "text_file" }>) => void;
}

function PreviewBlock({ sampleId }: { sampleId: string }) {
  const [preview, setPreview] = useState<SamplePreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    if (busy || preview !== null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setPreview(await samplesApi.preview(sampleId, 8));
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  }, [busy, preview, sampleId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (busy) {
    return (
      <p className="text-xs text-[var(--color-muted-foreground)]">
        <LoaderCircle aria-hidden className="mr-1 inline h-3 w-3 animate-spin" />
        loading the raw preview…
      </p>
    );
  }
  if (error !== null) {
    return (
      <p className="text-xs text-[var(--color-destructive)]" role="alert">
        preview failed: {error}
      </p>
    );
  }
  if (preview === null) {
    return null;
  }
  if (preview.kind === "text_file") {
    return (
      <div>
        <p className="text-xs text-[var(--color-muted-foreground)]">
          {preview.n_rows} lines total; first {preview.returned}
          {preview.truncated ? " (truncated)" : ""}:
        </p>
        <pre className="mt-1 max-h-40 overflow-auto rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] p-2 font-mono text-[11px] leading-snug text-[var(--color-foreground)]">
          {(preview.lines ?? []).join("\n")}
        </pre>
      </div>
    );
  }
  return (
    <div>
      <p className="text-xs text-[var(--color-muted-foreground)]">
        {preview.n_rows} rows × {preview.columns.length} columns — raw head
        {preview.truncated ? ` (first ${preview.returned})` : ""}:
      </p>
      <div className="mt-1 max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
        <table className="w-full text-left font-mono text-[11px]">
          <thead className="bg-[var(--color-muted)] text-[var(--color-foreground)]">
            <tr>
              {preview.columns.slice(0, 6).map((column) => (
                <th key={column} className="px-2 py-1 font-semibold">
                  {column}
                </th>
              ))}
              {preview.columns.length > 6 ? (
                <th className="px-2 py-1 font-semibold">
                  +{preview.columns.length - 6} more
                </th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {(preview.rows ?? []).slice(0, 8).map((row, index) => (
              <tr key={index} className="border-t border-[var(--color-border)]">
                {row.slice(0, 6).map((cell, cellIndex) => (
                  <td key={cellIndex} className="px-2 py-1 text-[var(--color-muted-foreground)]">
                    {cell === null || cell === undefined ? "—" : String(cell).slice(0, 18)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SampleRow({
  sample,
  onLoaded,
}: {
  sample: SampleView;
  onLoaded: (payload: SampleLoadResponse) => void;
}) {
  const [open, setOpen] = useState(false);
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SampleLoadResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copyTimer.current !== null) {
        clearTimeout(copyTimer.current);
      }
    };
  }, []);

  const copyContent = useCallback(async () => {
    if (result === null || result.kind !== "text_file") {
      return;
    }
    try {
      await navigator.clipboard.writeText(result.content);
      setCopied(true);
      if (copyTimer.current !== null) {
        clearTimeout(copyTimer.current);
      }
      copyTimer.current = setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("clipboard unavailable — select the text manually");
    }
  }, [result]);

  const runLoad = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const payload = await samplesApi.load(sample.sample_id);
      setResult(payload);
      setArmed(false);
      onLoaded(payload);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  }, [onLoaded, sample.sample_id]);

  return (
    <li className="border-t border-[var(--color-border)] px-4 py-3 first:border-t-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-[var(--color-foreground)]">
            {sample.title}
          </p>
          <p className="text-xs text-[var(--color-muted-foreground)]">
            {sample.filename} · {formatBytes(sample.size_bytes)} ·{" "}
            {KIND_LABEL[sample.kind]}
            {sample.target !== null ? ` · target ${sample.target}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? "Hide preview" : "Preview"}
          </Button>
          {armed ? (
            <>
              <Button size="sm" disabled={busy} onClick={() => void runLoad()}>
                {busy ? (
                  <>
                    <LoaderCircle aria-hidden className="h-3.5 w-3.5 animate-spin" />
                    loading…
                  </>
                ) : (
                  "Confirm load"
                )}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setArmed(false)}
                aria-label="cancel the demo load"
              >
                Cancel
              </Button>
            </>
          ) : (
            <Button
              size="sm"
              disabled={busy || !sample.available}
              onClick={() => {
                setArmed(true);
                setResult(null);
                setError(null);
              }}
            >
              <FlaskConical aria-hidden className="h-3.5 w-3.5" />
              {sample.available ? "Load demo" : "Unavailable"}
            </Button>
          )}
        </div>
      </div>
      <p className="mt-1 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
        {sample.description}
      </p>
      {open ? (
        <div className="mt-2">
          <PreviewBlock sampleId={sample.sample_id} />
        </div>
      ) : null}
      {armed ? (
        <p className="mt-2 rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] px-3 py-2 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
          {sample.kind === "tabular"
            ? "Imports this file as a NEW dataset record through the backend's own import engine — nothing you own is replaced; loaded demo datasets are removable in the Datasets workspace."
            : sample.kind === "series"
              ? "Delivers the file's two numeric columns as the times/concentration arrays this workflow's forms take — nothing is registered or replaced."
              : sample.kind === "library"
                ? "Delivers the file's name + SMILES columns as the list this workflow's inputs take — nothing is registered or replaced."
                : "Delivers the file's content verbatim (sha256-pinned) for this workflow's paste lanes — nothing is registered or replaced."}
        </p>
      ) : null}
      {error !== null ? (
        <p className="mt-2 rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-xs text-[var(--color-foreground)]" role="alert">
          load failed: {error}
        </p>
      ) : null}
      {result !== null ? (
        <div className="mt-2 rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] px-3 py-2 text-xs leading-relaxed text-[var(--color-foreground)]">
          <p className="font-medium">
            <Check aria-hidden className="mr-1 inline h-3 w-3 text-[var(--color-primary)]" />
            {result.kind === "tabular"
              ? `registered dataset ${(result.dataset as { dataset_id?: string }).dataset_id ?? ""}`
              : result.kind === "series"
                ? `delivered ${result.n_points} time/concentration points`
                : result.kind === "library"
                  ? `delivered ${result.n_molecules} compounds (name + SMILES)`
                  : `delivered ${formatBytes(result.size_bytes)} of ${result.file_type} content (${result.n_lines} lines)`}
          </p>
          {result.kind === "text_file" ? (
            <button
              type="button"
              onClick={() => void copyContent()}
              className="mt-1 inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 text-[11px] font-medium"
            >
              <ClipboardCopy aria-hidden className="h-3 w-3" />
              {copied ? "copied" : "copy content to clipboard"}
            </button>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

export function DemoDataSection({
  workflow,
  heading = "Demo data",
  onDatasetLoaded,
  onSeriesLoaded,
  onLibraryLoaded,
  onTextLoaded,
}: DemoDataSectionProps) {
  const [samples, setSamples] = useState<readonly SampleView[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const catalog = await samplesApi.list(workflow);
      setSamples(catalog.samples);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  }, [workflow]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleLoaded = useCallback(
    (payload: SampleLoadResponse) => {
      if (payload.kind === "tabular") {
        onDatasetLoaded?.();
      } else if (payload.kind === "series") {
        onSeriesLoaded?.(payload);
      } else if (payload.kind === "library") {
        onLibraryLoaded?.(payload);
      } else {
        onTextLoaded?.(payload);
      }
    },
    [onDatasetLoaded, onSeriesLoaded, onLibraryLoaded, onTextLoaded],
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <FlaskConical aria-hidden className="h-4 w-4 text-[var(--color-primary)]" />
          {heading}
          <Button
            size="sm"
            variant="outline"
            onClick={() => void refresh()}
            disabled={busy}
            aria-label="refresh the demo catalog"
            className="ml-auto"
          >
            <RotateCw aria-hidden className="h-3.5 w-3.5" />
          </Button>
        </CardTitle>
        <CardDescription>
          Sample data shipped with the application (the backend's own{" "}
          <code className="font-mono">examples/data</code> files) — preview
          first, then load one into this workflow. Nothing user-owned is
          ever replaced.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {error !== null ? (
          <p className="text-sm text-[var(--color-destructive)]" role="alert">
            catalog failed: {error}
          </p>
        ) : null}
        {samples === null ? (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            {busy ? "loading the demo catalog…" : "no demo data"}
          </p>
        ) : samples.length === 0 ? (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            no sample data applies to this workspace yet
          </p>
        ) : (
          <ul className="max-h-96 overflow-y-auto rounded-lg border border-[var(--color-border)]">
            {samples.map((sample) => (
              <SampleRow key={sample.sample_id} sample={sample} onLoaded={handleLoaded} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
