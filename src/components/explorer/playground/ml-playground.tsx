"use client";

import { useMemo, useRef, useState } from "react";
import {
  Brain,
  ChevronDown,
  ClipboardPaste,
  Download,
  Eraser,
  FileSpreadsheet,
  FileUp,
  History,
  Loader2,
  Play,
  RotateCcw,
  Table2,
  TrendingDown,
  TrendingUp,
  TriangleAlert,
  X,
  ZoomIn,
} from "lucide-react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CHART } from "@/components/explorer/playground/calc-ui";
import {
  MAX_CSV_TEXT,
  MIN_CSV_ROWS,
  numericColumnNames,
  parseCsv,
} from "@/lib/csv";
import { fmtBytes, fmtNum, fmtTick } from "@/lib/format";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { DEFAULT_ML_CONFIG, useToolkitStore, type MlRunRecord } from "@/lib/store";
import type {
  DatasetPreviewResponse,
  MLTrainResponse,
  MLTrainRequest,
} from "@/lib/toolkit-types";

type Task = "regression" | "classification";
type RegModel = "linear" | "polynomial" | "ridge";

const REG_METRICS: Array<{ key: string; label: string }> = [
  { key: "r2", label: "R²" },
  { key: "rmse", label: "RMSE" },
  { key: "mae", label: "MAE" },
];

const CLS_METRICS: Array<{ key: string; label: string }> = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
];

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-xl font-semibold tabular-nums">{fmtNum(value, 4)}</p>
    </div>
  );
}

/** Factory for tiny boundary dot renderers (60×60 grid → 3600 points). */
function TinyDot(color: string) {
  return function renderTinyDot(props: unknown) {
    const { cx, cy } = props as { cx?: number; cy?: number };
    if (cx === undefined || cy === undefined) return <g />;
    return <circle cx={cx} cy={cy} r={1.5} fill={color} fillOpacity={0.3} />;
  };
}

/** Factory for the larger moon-class dots. */
function MoonDot(color: string) {
  return function renderMoonDot(props: unknown) {
    const { cx, cy } = props as { cx?: number; cy?: number };
    if (cx === undefined || cy === undefined) return <g />;
    return <circle cx={cx} cy={cy} r={3.2} fill={color} fillOpacity={0.9} stroke={color} strokeWidth={0.6} />;
  };
}

/** Qualitative label for a Pearson correlation coefficient. */
function corrStrength(r: number): string {
  const a = Math.abs(r);
  if (a >= 0.9) return "very strong";
  if (a >= 0.7) return "strong";
  if (a >= 0.5) return "moderate";
  return "weak";
}

/** Compact "time ago" label for run-history rows (client-side only). */
function timeAgo(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function MlPlayground() {
  /* Experiment setup + results live in the store, so the whole playground
   * (config, trained model, dataset preview) survives tab switches. */
  const mlConfig = useToolkitStore((s) => s.mlConfig);
  const setMlConfig = useToolkitStore((s) => s.setMlConfig);
  const resetMlConfig = useToolkitStore((s) => s.resetMlConfig);
  const result = useToolkitStore((s) => s.mlResult);
  const setResult = useToolkitStore((s) => s.setMlResult);
  const preview = useToolkitStore((s) => s.mlPreview);
  const setPreview = useToolkitStore((s) => s.setMlPreview);
  const mlHistory = useToolkitStore((s) => s.mlHistory);
  const pushMlRun = useToolkitStore((s) => s.pushMlRun);
  const clearMlHistory = useToolkitStore((s) => s.clearMlHistory);
  const mlCurrentRunId = useToolkitStore((s) => s.mlCurrentRunId);
  const setMlCurrentRun = useToolkitStore((s) => s.setMlCurrentRun);
  const mlCsv = useToolkitStore((s) => s.mlCsv);
  const setMlCsv = useToolkitStore((s) => s.setMlCsv);

  const {
    task,
    source,
    samples,
    noise,
    regModel,
    alpha,
    l2,
    testSizePct,
    cv,
    seed,
    featureColumn,
    featureColumn2,
    targetColumn,
  } = mlConfig;

  /* Transient request state — not worth persisting across tab switches. */
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">(
    "idle"
  );
  const [error, setError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(true);

  /* X-range zoom on the preview scatter — null = full range. */
  const [xZoom, setXZoom] = useState<[number, number] | null>(null);

  /* Uploaded-CSV state: drag feedback, paste area, hidden file input. */
  const [dragOver, setDragOver] = useState(false);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* The loaded CSV parsed once per file (memoized) + client-side sanity. */
  const csvTable = useMemo(() => {
    if (!mlCsv) return null;
    try {
      const table = parseCsv(mlCsv.text);
      return table.columns.length > 0 && table.rows.length > 0 ? table : null;
    } catch {
      return null;
    }
  }, [mlCsv]);
  const csvNumericCols = useMemo(
    () => (csvTable ? numericColumnNames(csvTable) : []),
    [csvTable]
  );
  const csvUsable =
    csvTable !== null &&
    csvTable.rows.length >= MIN_CSV_ROWS &&
    csvNumericCols.length >= 2;

  /** Classification preview: the sampled points split by class for the chart. */
  const previewClasses = useMemo(() => {
    if (!preview || preview.task !== "classification") return null;
    return {
      cls0: preview.points.filter((p) => p.label === 0),
      cls1: preview.points.filter((p) => p.label === 1),
    };
  }, [preview]);

  /** Windowed view of the preview dataset (a true brush): the x-window
   *  filters the series data AND re-derives both axis domains, so zooming
   *  rescales the plot instead of merely clipping it. */
  const previewView = useMemo(() => {
    if (!preview || preview.points.length === 0) return null;
    const xs = preview.points.map((p) => p.x);
    const fullX: [number, number] = [Math.min(...xs), Math.max(...xs)];
    const yPad = preview.task === "regression" ? 0.4 : 0.3;
    const zoomed = xZoom !== null && (xZoom[0] > fullX[0] || xZoom[1] < fullX[1]);
    const win: [number, number] = xZoom ?? fullX;

    const inside =
      zoomed ? preview.points.filter((p) => p.x >= win[0] && p.x <= win[1]) : preview.points;
    const ySource = inside.length > 0 ? inside : preview.points;
    const ys = ySource.map((p) => p.y);
    // Pad the window slightly so edge dots aren't cut by the plot boundary.
    const xPad = zoomed ? Math.max((win[1] - win[0]) * 0.04, 0.05) : yPad;

    const inWin = (p: { x: number }) => !zoomed || (p.x >= win[0] && p.x <= win[1]);
    return {
      fullX,
      win,
      zoomed,
      xDomain: [win[0] - xPad, win[1] + xPad] as [number, number],
      yDomain: [Math.min(...ys) - yPad, Math.max(...ys) + yPad] as [number, number],
      points: inside,
      cls0: previewClasses ? previewClasses.cls0.filter(inWin) : [],
      cls1: previewClasses ? previewClasses.cls1.filter(inWin) : [],
      trend: (preview.trend ?? []).filter(inWin),
    };
  }, [preview, previewClasses, xZoom]);

  /** Export the full deterministic dataset (all rows) as a CSV file. */
  async function downloadCsv() {
    if (!seedValid) {
      setError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    setCsvLoading(true);
    setError(null);
    const seedNum = Number.parseInt(seed, 10) || 0;
    const fileName = `pharmaml_${task}_n${samples}_seed${seedNum}.csv`;
    try {
      const qs = new URLSearchParams({
        task,
        samples: String(samples),
        noise: String(noise),
        seed: String(seedNum),
      });
      const res = await fetch(`/api/ml/dataset/csv?${qs.toString()}`);
      if (!res.ok) {
        let message = `HTTP ${res.status}`;
        try {
          const data: unknown = await res.json();
          if (typeof data === "object" && data !== null && "error" in data) {
            message = String((data as { error: unknown }).error);
          }
        } catch {
          /* body wasn't JSON — keep the status message */
        }
        throw new Error(message);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = fileName;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
      toast.success("Dataset exported", {
        description: `${fileName} · ${samples} rows · ${fmtBytes(blob.size)}`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "CSV export failed.");
    } finally {
      setCsvLoading(false);
    }
  }

  const noiseMax = task === "regression" ? 2 : 1.5;
  const metricSpecs = task === "regression" ? REG_METRICS : CLS_METRICS;

  function switchTask(next: Task) {
    setMlConfig({ task: next });
    if (next === "classification" && noise > 1.5) setMlConfig({ noise: 0.3 });
    if (next === "regression" && regModel !== "ridge" && regModel !== "linear" && regModel !== "polynomial") {
      setMlConfig({ regModel: "linear" });
    }
    setStatus("idle");
    setResult(null);
    setError(null);
    setPreview(null);
    setXZoom(null);
  }

  /** Restore the default experiment setup and drop any stored results. */
  function resetSetup() {
    resetMlConfig();
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
  }

  /* ------------------------- CSV data loading ------------------------- */

  /** Load CSV text into the store, reset column picks, drop stale results.
   *  Returns false (with a toast) when the text is unusable. */
  function loadCsv(fileName: string, text: string): boolean {
    if (text.trim().length < 4 || text.length > MAX_CSV_TEXT) {
      toast.error("CSV not loaded", {
        description: "Expected a non-empty file of at most 256 KB.",
      });
      return false;
    }
    const table = parseCsv(text);
    if (table.columns.length === 0 || table.rows.length === 0) {
      toast.error("Could not parse the CSV", {
        description: "Expected a header row followed by data rows.",
      });
      return false;
    }
    setMlCsv({ fileName, text });
    setMlConfig({ featureColumn: "", featureColumn2: "", targetColumn: "" });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
    toast.success("CSV loaded", {
      description: `${fileName} — ${table.rows.length} rows × ${table.columns.length} columns · ${numericColumnNames(table).length} numeric`,
    });
    return true;
  }

  /** Read a picked/dropped File as text and load it. */
  function handleFile(file: File) {
    if (file.size > MAX_CSV_TEXT) {
      toast.error("File too large", {
        description: `CSV uploads are capped at 256 KB — this file is ${fmtBytes(file.size)}.`,
      });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => loadCsv(file.name, String(reader.result ?? ""));
    reader.onerror = () =>
      toast.error("Could not read the file", {
        description: "Try re-saving it as a plain UTF-8 CSV.",
      });
    reader.readAsText(file);
  }

  /** Load the pasted textarea contents as "pasted.csv". */
  function loadPasted() {
    const text = pasteText.trim();
    if (text.length >= 4 && loadCsv("pasted.csv", text)) {
      setPasteOpen(false);
      setPasteText("");
    }
  }

  /** Drop the loaded CSV and return to the empty upload state. */
  function removeCsv() {
    setMlCsv(null);
    setMlConfig({ featureColumn: "", featureColumn2: "", targetColumn: "" });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
  }

  /** Switch the data source; stale results/previews are dropped. */
  function switchSource(next: "synthetic" | "csv") {
    if (source === next) return;
    setMlConfig({ source: next });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
  }

  const seedValid = /^\d{1,10}$/.test(seed.trim());
  const alphaValid = Number.isFinite(Number.parseFloat(alpha)) && Number.parseFloat(alpha) >= 0;
  const l2Valid = Number.isFinite(Number.parseFloat(l2)) && Number.parseFloat(l2) >= 0;

  async function run() {
    if (source === "synthetic" && !seedValid) {
      setError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    if (source === "csv" && (!mlCsv || !csvUsable)) {
      setError(
        `Load a CSV with at least ${MIN_CSV_ROWS} rows and 2 numeric columns before training.`
      );
      return;
    }
    if (task === "regression" && regModel === "ridge" && !alphaValid) {
      setError("Ridge alpha must be a number ≥ 0.");
      return;
    }
    if (task === "classification" && !l2Valid) {
      setError("L2 strength must be a number ≥ 0.");
      return;
    }
    setStatus("running");
    setError(null);
    const body: MLTrainRequest = {
      task,
      samples,
      noise,
      model: task === "regression" ? regModel : "logistic",
      testSize: testSizePct / 100,
      cv: (Number.parseInt(cv, 10) || 0) as 0 | 3 | 5 | 10,
      seed: Number.parseInt(seed, 10) || 0,
    };
    if (task === "regression" && regModel === "ridge") body.alpha = Number.parseFloat(alpha);
    if (task === "classification") body.l2 = Number.parseFloat(l2);
    if (source === "csv" && mlCsv) {
      body.csv = mlCsv.text;
      body.csvName = mlCsv.fileName;
      if (featureColumn) body.featureColumn = featureColumn;
      if (featureColumn2) body.featureColumn2 = featureColumn2;
      if (targetColumn) body.targetColumn = targetColumn;
    }

    try {
      const res = await fetch("/api/ml/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? `${String((data as { error: unknown }).error)}${
                "detail" in data && Array.isArray((data as { detail: unknown }).detail)
                  ? ` ${((data as { detail: unknown[] }).detail as string[]).join("; ")}`
                  : ""
              }`
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const payload = data as MLTrainResponse;
      setResult(payload);
      setStatus("done");
      /* Record the completed run in the history strip (restorable). */
      const key = payload.task === "regression" ? "r2" : "accuracy";
      const keyValue = payload.metrics[key];
      if (typeof keyValue === "number" && Number.isFinite(keyValue)) {
        pushMlRun({
          id: Date.now(),
          at: new Date().toISOString(),
          task: payload.task,
          model: payload.model,
          keyMetric: payload.task === "regression" ? "R²" : "Accuracy",
          keyValue,
          config: { ...mlConfig },
          result: payload,
          ...(source === "csv" && mlCsv
            ? {
                csv: {
                  fileName: mlCsv.fileName,
                  featureColumn,
                  featureColumn2,
                  targetColumn,
                },
              }
            : {}),
        });
      }
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Training request failed.";
      setError(message);
    }
  }

  /** Restore a previous run: config snapshot + full result + charts. */
  function restoreRun(rec: MlRunRecord) {
    setMlConfig(rec.config);
    setResult(rec.result);
    setMlCurrentRun(rec.id);
    setStatus("done");
    setError(null);
    setXZoom(null);
    toast.success("Run restored", {
      description: `${rec.model} · ${rec.csv ? rec.csv.fileName : `seed ${rec.config.seed}`} · ${
        rec.keyMetric
      } ${fmtNum(rec.keyValue, 4)}`,
    });
  }  /** Fetch the first rows + summary stats of the (deterministic) dataset. */
  async function previewDataset() {
    if (source === "synthetic" && !seedValid) {
      setError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    if (source === "csv" && (!mlCsv || !csvUsable)) {
      setError(
        `Load a CSV with at least ${MIN_CSV_ROWS} rows and 2 numeric columns before previewing.`
      );
      return;
    }
    setPreviewLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/ml/dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task,
          samples,
          noise,
          seed: Number.parseInt(seed, 10) || 0,
          ...(source === "csv" && mlCsv
            ? {
                csv: mlCsv.text,
                csvName: mlCsv.fileName,
                ...(featureColumn ? { featureColumn } : {}),
                ...(featureColumn2 ? { featureColumn2 } : {}),
                ...(targetColumn ? { targetColumn } : {}),
              }
            : {}),
        }),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? `${String((data as { error: unknown }).error)}${
                "detail" in data && Array.isArray((data as { detail: unknown }).detail)
                  ? ` ${((data as { detail: unknown[] }).detail as string[]).join("; ")}`
                  : ""
              }`
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      setPreview(data as DatasetPreviewResponse);
      setPreviewOpen(true);
      setXZoom(null); // fresh dataset — reset the x-range zoom
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preview request failed.");
    } finally {
      setPreviewLoading(false);
    }
  }

  const running = status === "running";

  /* Chart series (memoized splits so recharts gets stable references). */
  const clsSeries = useMemo(() => {
    if (!result || result.task !== "classification") return null;
    const boundary = result.series.boundary ?? [];
    return {
      grid0: boundary.filter((p) => p.pred === 0),
      grid1: boundary.filter((p) => p.pred === 1),
      moon0: result.series.test.filter((p) => p.label === 0),
      moon1: result.series.test.filter((p) => p.label === 1),
    };
  }, [result]);

  const coefficientsText = useMemo(() => {
    if (!result) return null;
    const { intercept, coefs } = result.coefficients;
    if (result.task === "regression") {
      const terms = coefs
        .map((c, i) => `${fmtNum(c, 4)}·x̂${coefs.length > 1 ? `^${i + 1}` : ""}`)
        .join(" + ");
      return `ŷ = ${fmtNum(intercept, 4)} + ${terms}   (x̂ = standardized x)`;
    }
    return `σ(b₀ + w₁x̂₁ + w₂x̂₂)   b₀=${fmtNum(intercept, 4)}, w=[${coefs
      .map((c) => fmtNum(c, 4))
      .join(", ")}]`;
  }, [result]);

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(300px,360px)_minmax(0,1fr)]">
      {/* ------------------------------ Controls ------------------------------ */}
      <Card className="h-fit overflow-hidden py-0 shadow-sm xl:sticky xl:top-24">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Brain className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Experiment setup</CardTitle>
              <CardDescription className="text-xs">
                Seeded generator or your own CSV — setup & results survive reloads
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-5 pb-4">
          {/* Task */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Task</Label>
            <div className="grid grid-cols-2 gap-1 rounded-lg border bg-muted/40 p-1" role="tablist" aria-label="ML task">
              {(["regression", "classification"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  role="tab"
                  aria-selected={task === t}
                  onClick={() => switchTask(t)}
                  className={`h-9 rounded-md text-sm font-medium capitalize transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                    task === t ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Data source: seeded generator or the user's own CSV */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Data source</Label>
            <div className="grid grid-cols-2 gap-1 rounded-lg border bg-muted/40 p-1" role="tablist" aria-label="Data source">
              {(["synthetic", "csv"] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  role="tab"
                  aria-selected={source === s}
                  onClick={() => switchSource(s)}
                  className={`h-9 rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                    source === s ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {s === "synthetic" ? "Synthetic" : "My CSV"}
                </button>
              ))}
            </div>
          </div>

          {/* Uploaded CSV dataset (drop / browse / paste + column picks) */}
          {source === "csv" ? (
            <div className="space-y-2.5">
              <Label className="text-xs font-medium">Your dataset</Label>
              {mlCsv ? (
                <div className="space-y-2.5">
                  <div className="rounded-lg border bg-muted/20 p-2.5">
                    <div className="flex items-center gap-2.5">
                      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                        <FileSpreadsheet className="size-4" aria-hidden />
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-mono text-xs font-medium" title={mlCsv.fileName}>
                          {mlCsv.fileName}
                        </p>
                        <p className="text-[11px] leading-snug text-muted-foreground">
                          {csvTable
                            ? `${csvTable.rows.length} rows × ${csvTable.columns.length} cols · ${csvNumericCols.length} numeric`
                            : "Unreadable — re-upload the file"}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={removeCsv}
                        aria-label="Remove the loaded CSV"
                        title="Remove CSV"
                        className="flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      >
                        <X className="size-3.5" aria-hidden />
                      </button>
                    </div>
                    {csvTable && !csvUsable ? (
                      <p className="mt-2 rounded-md border border-destructive/25 bg-destructive/5 px-2 py-1.5 text-[11px] leading-snug text-destructive">
                        Needs ≥ {MIN_CSV_ROWS} rows and ≥ 2 numeric columns to train.
                      </p>
                    ) : null}
                  </div>

                  {csvTable ? (
                    task === "regression" ? (
                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Feature (x)</Label>
                          <Select
                            value={featureColumn || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ featureColumn: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="Feature column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · first numeric</SelectItem>
                              {csvNumericCols.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Target (y)</Label>
                          <Select
                            value={targetColumn || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ targetColumn: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="Target column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · last numeric</SelectItem>
                              {csvNumericCols.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Feature x₁</Label>
                          <Select
                            value={featureColumn || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ featureColumn: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="First feature column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · first numeric</SelectItem>
                              {csvNumericCols.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Feature x₂</Label>
                          <Select
                            value={featureColumn2 || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ featureColumn2: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="Second feature column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · second numeric</SelectItem>
                              {csvNumericCols.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Label (2 classes)</Label>
                          <Select
                            value={targetColumn || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ targetColumn: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="Label column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · 2-value column</SelectItem>
                              {csvTable.columns.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    )
                  ) : null}
                </div>
              ) : (
                <div className="space-y-2">
                  <div
                    role="button"
                    tabIndex={0}
                    aria-label="Upload a CSV file"
                    onClick={() => fileInputRef.current?.click()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        fileInputRef.current?.click();
                      }
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(true);
                    }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragOver(false);
                      const file = e.dataTransfer.files?.[0];
                      if (file) void handleFile(file);
                    }}
                    className={cn(
                      "flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed px-4 py-6 text-center outline-none transition-colors focus-visible:ring-1 focus-visible:ring-ring",
                      dragOver
                        ? "border-primary bg-primary/5"
                        : "border-border bg-muted/20 hover:border-primary/40 hover:bg-primary/5"
                    )}
                  >
                    <FileUp className="size-5 text-primary" aria-hidden />
                    <p className="text-xs font-medium">Drop a CSV here or click to browse</p>
                    <p className="text-[11px] leading-snug text-muted-foreground">
                      .csv · .tsv · .txt · ≤ 256 KB · parsed in your browser
                    </p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.tsv,.txt,text/csv,text/plain"
                    className="sr-only"
                    aria-label="Choose a CSV file"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) void handleFile(file);
                      e.target.value = "";
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setPasteOpen((v) => !v)}
                    aria-expanded={pasteOpen}
                    className="w-full rounded-sm text-center text-[11px] font-medium text-primary underline-offset-2 transition-colors hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    {pasteOpen ? "Hide paste area" : "…or paste CSV text instead"}
                  </button>
                  {pasteOpen ? (
                    <div className="space-y-1.5">
                      <textarea
                        value={pasteText}
                        onChange={(e) => setPasteText(e.target.value)}
                        rows={5}
                        spellCheck={false}
                        placeholder={"dose,clearance\n250,3.2\n180,2.4\n…"}
                        aria-label="Paste CSV text"
                        className="thin-scrollbar w-full resize-y rounded-md border bg-background p-2 font-mono text-[11px] leading-relaxed outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      />
                      <Button
                        size="sm"
                        className="h-8 w-full gap-1.5 text-xs"
                        onClick={loadPasted}
                        disabled={pasteText.trim().length < 4}
                      >
                        <ClipboardPaste className="size-3.5" aria-hidden />
                        Load pasted CSV
                      </Button>
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          ) : null}

          {/* Samples + noise — synthetic mode only (CSV rows come from the file) */}
          {source === "synthetic" ? (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="ml-samples" className="text-xs font-medium">
                    Samples
                  </Label>
                  <span className="font-mono text-xs tabular-nums text-muted-foreground">{samples}</span>
                </div>
                <Slider
                  id="ml-samples"
                  min={100}
                  max={1000}
                  step={50}
                  value={[samples]}
                  onValueChange={(v) => setMlConfig({ samples: v[0] ?? samples })}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="ml-noise" className="text-xs font-medium">
                    Noise {task === "classification" ? "(moons)" : "σ"}
                  </Label>
                  <span className="font-mono text-xs tabular-nums text-muted-foreground">
                    {noise.toFixed(2)}
                  </span>
                </div>
                <Slider
                  id="ml-noise"
                  min={0}
                  max={noiseMax}
                  step={0.05}
                  value={[Math.min(noise, noiseMax)]}
                  onValueChange={(v) => setMlConfig({ noise: v[0] ?? noise })}
                />
              </div>
            </>
          ) : null}

          {/* Model */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Model</Label>
            {task === "regression" ? (
              <Select value={regModel} onValueChange={(v) => setMlConfig({ regModel: v as RegModel })}>
                <SelectTrigger className="h-10" aria-label="Regression model">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="linear">Linear regression</SelectItem>
                  <SelectItem value="polynomial">Polynomial (degree 2)</SelectItem>
                  <SelectItem value="ridge">Ridge (α regularized)</SelectItem>
                </SelectContent>
              </Select>
            ) : (
              <div className="space-y-2">
                <div className="flex h-10 items-center rounded-md border bg-muted/30 px-3 text-sm text-muted-foreground">
                  Logistic regression (batch GD, 800 epochs)
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="ml-l2" className="text-xs font-medium">
                    L2 strength λ
                  </Label>
                  <Input
                    id="ml-l2"
                    inputMode="decimal"
                    value={l2}
                    onChange={(e) => setMlConfig({ l2: e.target.value })}
                    aria-invalid={l2Valid ? undefined : true}
                    className={`h-10 font-mono text-sm tabular-nums ${l2Valid ? "" : "border-destructive/60"}`}
                  />
                  {!l2Valid ? (
                    <p className="text-[11px] text-destructive">Must be a number ≥ 0.</p>
                  ) : null}
                </div>
              </div>
            )}
          </div>

          {task === "regression" && regModel === "ridge" ? (
            <div className="space-y-1.5">
              <Label htmlFor="ml-alpha" className="text-xs font-medium">
                Ridge α
              </Label>
              <Input
                id="ml-alpha"
                inputMode="decimal"
                value={alpha}
                onChange={(e) => setMlConfig({ alpha: e.target.value })}
                aria-invalid={alphaValid ? undefined : true}
                className={`h-10 font-mono text-sm tabular-nums ${alphaValid ? "" : "border-destructive/60"}`}
              />
              {!alphaValid ? <p className="text-[11px] text-destructive">Must be a number ≥ 0.</p> : null}
            </div>
          ) : null}

          {/* Test size */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="ml-test" className="text-xs font-medium">
                Test split
              </Label>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {testSizePct}%
              </span>
            </div>
            <Slider
              id="ml-test"
              min={10}
              max={50}
              step={5}
              value={[testSizePct]}
              onValueChange={(v) => setMlConfig({ testSizePct: v[0] ?? testSizePct })}
            />
          </div>

          {/* CV + seed */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">k-fold CV</Label>
              <Select value={cv} onValueChange={(v) => setMlConfig({ cv: v })}>
                <SelectTrigger className="h-10" aria-label="Cross-validation folds">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Off</SelectItem>
                  <SelectItem value="3">3-fold</SelectItem>
                  <SelectItem value="5">5-fold</SelectItem>
                  <SelectItem value="10">10-fold</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {source === "synthetic" ? (
              <div className="space-y-1.5">
                <Label htmlFor="ml-seed" className="text-xs font-medium">
                  Seed
                </Label>
                <Input
                  id="ml-seed"
                  inputMode="numeric"
                  value={seed}
                  onChange={(e) => setMlConfig({ seed: e.target.value })}
                  aria-invalid={seedValid ? undefined : true}
                  className={`h-10 font-mono text-sm tabular-nums ${seedValid ? "" : "border-destructive/60"}`}
                />
                {!seedValid ? (
                  <p className="text-[11px] text-destructive">Non-negative integer.</p>
                ) : null}
              </div>
            ) : (
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Rows</Label>
                <div
                  className="flex h-10 items-center rounded-md border bg-muted/30 px-3 font-mono text-sm tabular-nums text-muted-foreground"
                  title={mlCsv?.fileName ?? "No CSV loaded"}
                >
                  {csvTable ? `${csvTable.rows.length} rows` : "No CSV loaded"}
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <Button
              size="lg"
              className="h-11 flex-1 gap-2"
              onClick={() => void run()}
              disabled={running}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  Training…
                </>
              ) : (
                <>
                  <Play className="size-4" aria-hidden />
                  Train model
                </>
              )}
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="h-11 shrink-0 gap-2"
              onClick={() => void previewDataset()}
              disabled={previewLoading}
              aria-label="Preview the dataset before training"
            >
              {previewLoading ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <Table2 className="size-4" aria-hidden />
              )}
              <span className="hidden sm:inline">Preview data</span>
              <span className="sm:hidden">Data</span>
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="h-11 shrink-0 gap-2"
              onClick={resetSetup}
              disabled={running || previewLoading}
              aria-label="Reset the experiment setup to defaults"
              title="Reset setup to defaults"
            >
              <RotateCcw className="size-4" aria-hidden />
              <span className="hidden xl:inline">Reset</span>
            </Button>
          </div>

          {error ? (
            <Alert variant="destructive">
              <TriangleAlert className="size-4" aria-hidden />
              <AlertTitle>Request failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}
        </CardContent>
      </Card>

      {/* ------------------------------ Results ------------------------------ */}
      <div className="min-w-0 space-y-4">
        {preview ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="border-b bg-muted/30 py-3">
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <CardTitle className="text-sm">Dataset preview</CardTitle>
                  <CardDescription className="text-xs">
                    {preview.source
                      ? `First ${preview.rows.length} of ${preview.source.nRows} usable rows · ${
                          preview.source.fileName ?? "pasted.csv"
                        } · ${preview.source.droppedRows} malformed row${
                          preview.source.droppedRows === 1 ? "" : "s"
                        } dropped — raw column values`
                      : `First ${preview.rows.length} of ${preview.nTotal} rows · seed ${preview.seed} ·
                    noise σ ${preview.noise.toFixed(2)} — same generator the trainer uses`}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-1.5">
                {!preview.source ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 shrink-0 gap-1.5 px-2.5 text-xs"
                    onClick={() => void downloadCsv()}
                    disabled={csvLoading}
                    aria-label="Download the full dataset as CSV"
                    title="Download the full dataset as CSV"
                  >
                    {csvLoading ? (
                      <Loader2 className="size-3.5 animate-spin" aria-hidden />
                    ) : (
                      <Download className="size-3.5" aria-hidden />
                    )}
                    <span className="hidden sm:inline">CSV</span>
                  </Button>
                ) : null}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 shrink-0 p-0"
                  onClick={() => setPreviewOpen((v) => !v)}
                  aria-expanded={previewOpen}
                  aria-label={previewOpen ? "Collapse dataset preview" : "Expand dataset preview"}
                >
                  <ChevronDown
                    className={cn(
                      "size-4 transition-transform duration-200",
                      previewOpen ? "" : "-rotate-90"
                    )}
                    aria-hidden
                  />
                </Button>
                </div>
              </div>
            </CardHeader>
            {previewOpen ? (
              <CardContent className="space-y-3 p-3 sm:p-4">
                <div className="grid gap-3 lg:grid-cols-2">
                  <div className="thin-scrollbar max-h-72 overflow-auto rounded-md border">
                    <Table className="data-table">
                      <TableHeader>
                        <TableRow className="bg-muted/40 hover:bg-muted/40">
                          {preview.columns.map((c) => (
                            <TableHead key={c} className="h-8 font-mono text-[11px]">
                              {c}
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {preview.rows.map((row, idx) => (
                          <TableRow key={String(row[preview.columns[0] ?? "i"] ?? idx)}>
                            {preview.columns.map((c) => (
                              <TableCell key={c} className="py-1.5 font-mono text-xs tabular-nums">
                                {c === (preview.source?.targetColumn ?? "label") ? (
                                  <span className="inline-flex items-center gap-1.5">
                                    <span
                                      className={cn(
                                        "size-1.5 rounded-full",
                                        row[c] === 1 ? "bg-chart-1" : "bg-chart-2"
                                      )}
                                      aria-hidden
                                    />
                                    {row[c]}
                                  </span>
                                ) : (
                                  fmtNum(row[c] ?? 0, 4)
                                )}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  <div className="flex flex-col rounded-md border bg-muted/10 p-2">
                    <p className="px-1 pb-1 font-mono text-[11px] leading-snug text-muted-foreground">
                      all {preview.nTotal} points · {preview.nPoints} evenly-spaced shown
                      {preview.task === "regression" ? " · true line dashed" : ""}
                    </p>
                    <div className="h-56 min-h-0 lg:h-auto lg:flex-1">
                      <ResponsiveContainer width="100%" height="100%">
                        {preview.task === "regression" ? (
                          <ComposedChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                            <CartesianGrid
                              stroke={CHART.gridStroke}
                              strokeOpacity={CHART.gridStrokeOpacity}
                            />
                            <XAxis
                              type="number"
                              dataKey="x"
                              domain={previewView?.xDomain ?? ["dataMin - 0.4", "dataMax + 0.4"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                              label={{
                                value: "x",
                                position: "insideBottomRight",
                                offset: -2,
                                fill: "#78716c",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              type="number"
                              dataKey="y"
                              width={44}
                              domain={previewView?.yDomain ?? ["dataMin - 0.4", "dataMax + 0.4"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={false}
                            />
                            <Tooltip
                              cursor={{ strokeDasharray: "3 3" }}
                              contentStyle={CHART.tooltipStyle}
                              formatter={(value) => fmtNum(Number(value), 4)}
                            />
                            <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                            <Scatter
                              name="sampled points"
                              data={previewView?.points ?? preview.points}
                              fill={CHART.stone}
                              fillOpacity={0.7}
                              isAnimationActive={false}
                            />
                            <Line
                              name="true line (noiseless)"
                              data={previewView?.trend ?? preview.trend ?? []}
                              dataKey="y"
                              stroke={CHART.emerald}
                              strokeWidth={2}
                              strokeDasharray="6 3"
                              dot={false}
                              activeDot={false}
                              isAnimationActive={false}
                            />
                          </ComposedChart>
                        ) : (
                          <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                            <CartesianGrid
                              stroke={CHART.gridStroke}
                              strokeOpacity={CHART.gridStrokeOpacity}
                            />
                            <XAxis
                              type="number"
                              dataKey="x"
                              domain={previewView?.xDomain ?? ["dataMin - 0.3", "dataMax + 0.3"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                            />
                            <YAxis
                              type="number"
                              dataKey="y"
                              width={44}
                              domain={previewView?.yDomain ?? ["dataMin - 0.3", "dataMax + 0.3"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={false}
                            />
                            <Tooltip
                              cursor={{ strokeDasharray: "3 3" }}
                              contentStyle={CHART.tooltipStyle}
                              formatter={(value) => fmtNum(Number(value), 3)}
                            />
                            <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                            {previewView ? (
                              <Scatter
                                name={
                                  preview.source?.labelMap
                                    ? `class 0 · ${preview.source.labelMap[0]}`
                                    : "class 0 (outer moon)"
                                }
                                data={previewView.cls0}
                                fill={CHART.emerald}
                                fillOpacity={0.75}
                                isAnimationActive={false}
                              />
                            ) : null}
                            {previewView ? (
                              <Scatter
                                name={
                                  preview.source?.labelMap
                                    ? `class 1 · ${preview.source.labelMap[1]}`
                                    : "class 1 (inner moon)"
                                }
                                data={previewView.cls1}
                                fill={CHART.rose}
                                fillOpacity={0.75}
                                isAnimationActive={false}
                              />
                            ) : null}
                          </ScatterChart>
                        )}
                      </ResponsiveContainer>
                    </div>
                    {previewView ? (
                      <div className="mt-1.5 rounded-md border border-primary/15 bg-primary/5 px-2.5 pb-1.5 pt-2">
                        <div className="flex items-center justify-between gap-2 pb-1">
                          <p className="flex items-center gap-1.5 font-mono text-[11px] font-medium text-primary">
                            <ZoomIn className="size-3 shrink-0" aria-hidden />
                            x-window zoom
                          </p>
                          <div className="flex min-w-0 items-center gap-2">
                            <span
                              className="truncate font-mono text-[10px] tabular-nums text-muted-foreground"
                              aria-live="polite"
                            >
                              [{fmtNum(previewView.win[0], 2)} … {fmtNum(previewView.win[1], 2)}]
                              <span className="sr-only">
                                ; drag the handles to zoom the chart
                              </span>
                            </span>
                            {previewView.zoomed ? (
                              <button
                                type="button"
                                onClick={() => setXZoom(null)}
                                aria-label="Reset the chart zoom window"
                                title="Reset zoom"
                                className="flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                              >
                                <RotateCcw className="size-3" aria-hidden />
                              </button>
                            ) : null}
                          </div>
                        </div>
                        <Slider
                          aria-label="Zoom the preview chart x-axis window"
                          min={previewView.fullX[0]}
                          max={previewView.fullX[1]}
                          step={(previewView.fullX[1] - previewView.fullX[0]) / 200}
                          value={[previewView.win[0], previewView.win[1]]}
                          onValueChange={(v) =>
                            setXZoom([
                              v[0] ?? previewView.fullX[0],
                              v[1] ?? previewView.fullX[1],
                            ])
                          }
                        />
                      </div>
                    ) : null}
                  </div>
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {preview.stats.map((s) => (
                    <div key={s.column} className="rounded-md border bg-muted/20 px-3 py-2">
                      <p className="font-mono text-xs font-medium">{s.column}</p>
                      <p className="mt-0.5 font-mono text-[11px] leading-snug tabular-nums text-muted-foreground">
                        mean {fmtNum(s.mean, 4)} · σ {fmtNum(s.std, 4)} · min {fmtNum(s.min, 4)} ·
                        max {fmtNum(s.max, 4)}
                      </p>
                    </div>
                  ))}
                  {typeof preview.correlation === "number" ? (
                    <div className="rounded-md border border-primary/25 bg-primary/5 px-3 py-2">
                      <p className="font-mono text-xs font-medium text-primary">Pearson r(x, y)</p>
                      <p className="mt-0.5 font-mono text-[11px] tabular-nums text-primary/90">
                        {fmtNum(preview.correlation, 4)} — {corrStrength(preview.correlation)} linear
                        signal
                      </p>
                    </div>
                  ) : null}
                  {preview.classBalance ? (
                    <div className="rounded-md border border-primary/25 bg-primary/5 px-3 py-2">
                      <p className="font-mono text-xs font-medium text-primary">Class balance</p>
                      <p className="mt-0.5 font-mono text-[11px] tabular-nums text-primary/90">
                        {preview.classBalance
                          .map(
                            (c) =>
                              `class ${c.label}: ${c.count} (${c.pct.toFixed(1)}%)`
                          )
                          .join(" · ")}
                      </p>
                    </div>
                  ) : null}
                </div>
              </CardContent>
            ) : null}
          </Card>
        ) : null}

        {!result && status !== "running" ? (
          <Card className="py-0 shadow-sm">
            <CardContent className="flex flex-col items-center justify-center gap-2 py-16 text-center">
              <span className="ping-ring flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Brain className="size-6" aria-hidden />
              </span>
              <p className="text-sm font-medium">No results yet</p>
              <p className="max-w-sm text-xs text-muted-foreground">
                Pick a data source — the seeded server-side generator or your own CSV — then press{" "}
                <strong className="font-semibold text-foreground">Train model</strong>. Regression
                fits a closed-form least-squares model; classification trains logistic regression
                on the chosen columns.
              </p>
            </CardContent>
          </Card>
        ) : null}

        {status === "running" ? (
          <Card className="py-0 shadow-sm">
            <CardContent className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
              <p className="text-sm font-medium">Training…</p>
              <p className="text-xs text-muted-foreground">
                {source === "csv" && mlCsv
                  ? `Training on ${mlCsv.fileName} (${csvTable?.rows.length ?? 0} rows), fitting the model, evaluating the held-out split${
                      cv !== "0" ? ` and running ${cv}-fold CV` : ""
                    }.`
                  : `Generating ${samples} samples, fitting the model, evaluating the held-out split${
                      cv !== "0" ? ` and running ${cv}-fold CV` : ""
                    }.`}
              </p>
            </CardContent>
          </Card>
        ) : null}

        {result && status !== "running" ? (
          <>
            {/* Metrics */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {metricSpecs.map((m) => (
                <MetricCard key={m.key} label={m.label} value={result.metrics[m.key] ?? Number.NaN} />
              ))}
              {result.cv ? (
                <div className="col-span-2 rounded-lg border border-primary/25 bg-primary/5 px-3 py-2.5 shadow-sm sm:col-span-4">
                  <p className="text-xs text-muted-foreground">
                    {result.cv.k}-fold CV {result.cv.metricName}
                  </p>
                  <p className="font-mono text-xl font-semibold tabular-nums text-primary">
                    {fmtNum(result.cv.mean, 4)}{" "}
                    <span className="text-sm font-normal">± {fmtNum(result.cv.std, 3)}</span>
                  </p>
                </div>
              ) : null}
            </div>

            {/* Chart */}
            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <CardTitle className="text-sm">
                  {result.task === "regression"
                    ? "Test points & fitted curve"
                    : result.datasetMeta.source
                      ? "Test points & decision boundary"
                      : "Two moons & decision boundary"}
                </CardTitle>
                <CardDescription className="text-xs">
                  {result.datasetMeta.source
                    ? `${result.datasetMeta.nTotal} rows from ${
                        result.datasetMeta.source.fileName ?? "pasted.csv"
                      } · ${result.datasetMeta.nTrain} train · ${result.datasetMeta.nTest} test · ${
                        result.datasetMeta.source.droppedRows
                      } dropped`
                    : `${result.datasetMeta.nTotal} samples · ${result.datasetMeta.nTrain} train · ${
                        result.datasetMeta.nTest
                      } test · seed ${result.datasetMeta.seed}`}
                </CardDescription>
              </CardHeader>
              <CardContent className="p-3 sm:p-4">
                <div
                  className="h-[340px] w-full"
                  role="img"
                  aria-label={
                    result.task === "regression"
                      ? "Scatter plot of test points with the fitted regression curve"
                      : result.datasetMeta.source
                        ? "Scatter plot of the uploaded-data test classes with the model decision boundary"
                        : "Scatter plot of the two moon classes with model decision boundary"
                  }
                >
                  <ResponsiveContainer width="100%" height="100%">
                    {result.task === "regression" ? (
                      <ScatterChart margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
                        <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={["auto", "auto"]}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                          label={{ value: "x", position: "insideBottomRight", offset: -2, fill: "#78716c", fontSize: 11 }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          width={44}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          contentStyle={CHART.tooltipStyle}
                          formatter={(value) => fmtNum(Number(value), 4)}
                        />
                        <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                        <Scatter
                          name="fitted curve"
                          data={result.series.curve ?? []}
                          line={{ stroke: CHART.emerald, strokeWidth: 2.5 }}
                          shape="square"
                          fill={CHART.emerald}
                          fillOpacity={0.5}
                          isAnimationActive={false}
                        />
                        <Scatter
                          name="test points"
                          data={result.series.test}
                          fill={CHART.stone}
                          fillOpacity={0.8}
                          isAnimationActive={false}
                        />
                      </ScatterChart>
                    ) : (
                      <ScatterChart margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
                        <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={["auto", "auto"]}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          width={44}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          contentStyle={CHART.tooltipStyle}
                          formatter={(value) => fmtNum(Number(value), 3)}
                        />
                        <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                        {clsSeries ? (
                          <Scatter
                            name="boundary: class 0"
                            data={clsSeries.grid0}
                            fill={CHART.teal}
                            shape={TinyDot(CHART.teal)}
                            isAnimationActive={false}
                          />
                        ) : null}
                        {clsSeries ? (
                          <Scatter
                            name="boundary: class 1"
                            data={clsSeries.grid1}
                            fill={CHART.amber}
                            shape={TinyDot(CHART.amber)}
                            isAnimationActive={false}
                          />
                        ) : null}
                        {clsSeries ? (
                          <Scatter
                            name={
                              result.datasetMeta.source?.labelMap
                                ? `class 0 · ${result.datasetMeta.source.labelMap[0]} (test)`
                                : "moon 0 (test)"
                            }
                            data={clsSeries.moon0}
                            fill={CHART.emerald}
                            shape={MoonDot(CHART.emerald)}
                            isAnimationActive={false}
                          />
                        ) : null}
                        {clsSeries ? (
                          <Scatter
                            name={
                              result.datasetMeta.source?.labelMap
                                ? `class 1 · ${result.datasetMeta.source.labelMap[1]} (test)`
                                : "moon 1 (test)"
                            }
                            data={clsSeries.moon1}
                            fill={CHART.rose}
                            shape={MoonDot(CHART.rose)}
                            isAnimationActive={false}
                          />
                        ) : null}
                      </ScatterChart>
                    )}
                  </ResponsiveContainer>
                </div>
                <p className="mt-1 text-center text-[11px] text-muted-foreground">
                  {result.task === "regression"
                    ? "Stone dots = held-out test points; emerald line = model prediction (100-point sweep)."
                    : result.datasetMeta.source
                      ? "Emerald/rose = test-set classes; faint teal/amber dots = 60×60 grid classified by the model."
                      : "Emerald/rose = test-set moons; faint teal/amber dots = 60×60 grid classified by the model."}
                </p>
              </CardContent>
            </Card>

            {/* CV folds + coefficients */}
            <div className="grid gap-4 lg:grid-cols-2">
              {result.cv ? (
                <Card className="overflow-hidden py-0 shadow-sm">
                  <CardHeader className="border-b bg-muted/30 py-3">
                    <CardTitle className="text-sm">
                      {result.cv.k}-fold CV — {result.cv.metricName} per fold
                    </CardTitle>
                  </CardHeader>
                  <div className="thin-scrollbar max-h-72 overflow-y-auto">
                    <Table className="data-table">
                      <TableHeader className="sticky top-0 bg-muted/60 backdrop-blur">
                        <TableRow className="hover:bg-transparent">
                          <TableHead className="h-9">Fold</TableHead>
                          <TableHead className="h-9 text-right">{result.cv.metricName}</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {result.cv.folds.map((f) => (
                          <TableRow key={f.fold}>
                            <TableCell className="py-2 font-mono text-xs">fold {f.fold}</TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                              {fmtNum(f.value, 4)}
                            </TableCell>
                          </TableRow>
                        ))}
                        <TableRow className="bg-muted/30 hover:bg-muted/30">
                          <TableCell className="py-2 text-xs font-medium">mean ± std</TableCell>
                          <TableCell className="py-2 text-right font-mono text-xs font-semibold tabular-nums text-primary">
                            {fmtNum(result.cv?.mean ?? 0, 4)} ± {fmtNum(result.cv?.std ?? 0, 3)}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </div>
                </Card>
              ) : null}

              <Card className="overflow-hidden py-0 shadow-sm">
                <CardHeader className="border-b bg-muted/30 py-3">
                  <CardTitle className="text-sm">Learned parameters</CardTitle>
                  <CardDescription className="text-xs">
                    Standardized feature space (train-set μ, σ)
                  </CardDescription>
                </CardHeader>
                <CardContent className="pb-4">
                  {coefficientsText ? (
                    <pre className="thin-scrollbar overflow-x-auto rounded-md border bg-muted/30 p-3 font-mono text-xs leading-relaxed">
                      {coefficientsText}
                    </pre>
                  ) : null}
                  <p className="mt-2 text-[11px] text-muted-foreground">
                    {result.task === "regression"
                      ? "Solved in closed form via normal equations (Gaussian elimination with partial pivoting); ridge adds α to the diagonal, skipping the intercept."
                      : "Batch gradient descent: lr = 0.1, 800 epochs, L2-penalized weights (bias exempt)."}
                  </p>
                </CardContent>
              </Card>
            </div>
          </>
        ) : null}

        {/* ------------------------------ Run history ------------------------------ */}
        {mlHistory.length > 0 ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 border-b bg-muted/30 py-3">
              <div className="flex min-w-0 items-center gap-2.5">
                <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <History className="size-4.5" aria-hidden />
                </span>
                <div className="min-w-0">
                  <CardTitle className="text-base">Run history</CardTitle>
                  <CardDescription className="text-xs">
                    Last {mlHistory.length} training{" "}
                    {mlHistory.length === 1 ? "run" : "runs"} · survives reloads · Restore
                    reloads the config and results
                  </CardDescription>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 shrink-0 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
                onClick={clearMlHistory}
                aria-label="Clear the run history"
              >
                <Eraser className="size-3.5" aria-hidden />
                Clear
              </Button>
            </CardHeader>
            <CardContent className="py-0">
              <ul className="divide-y">
                {mlHistory.map((rec, i) => {
                  /* Newest-first list: the run *before* this one is i + 1. */
                  const prev = mlHistory[i + 1];
                  const delta = prev ? rec.keyValue - prev.keyValue : null;
                  const isCurrent = result !== null && rec.id === mlCurrentRunId;
                  const up = delta !== null && delta >= 0;
                  return (
                    <li
                      key={rec.id}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2.5 transition-colors hover:bg-primary/5",
                        isCurrent && "bg-primary/5"
                      )}
                    >
                      <span
                        className={cn(
                          "flex size-8 shrink-0 items-center justify-center rounded-md border font-mono text-[10px] font-semibold uppercase",
                          isCurrent
                            ? "border-primary/30 bg-primary/10 text-primary"
                            : "bg-muted/40 text-muted-foreground"
                        )}
                        title={rec.task}
                      >
                        {rec.task === "regression" ? "reg" : "cls"}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-mono text-xs font-medium">
                          {rec.model}
                          <span className="font-normal text-muted-foreground">
                            {" "}
                            {rec.csv
                              ? ` · ${rec.csv.fileName} · target: ${
                                  rec.csv.targetColumn || "auto"
                                }`
                              : ` · seed ${rec.config.seed} · n${rec.config.samples} · noise ${
                                  rec.config.noise
                                }`}
                          </span>
                        </p>
                        <p className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                          {timeAgo(rec.at)}
                          {isCurrent ? (
                            <span className="rounded-sm border border-primary/30 bg-primary/10 px-1 font-mono text-[9px] font-semibold uppercase text-primary">
                              current
                            </span>
                          ) : null}
                        </p>
                      </div>
                      <div className="shrink-0 text-right">
                        <p className="font-mono text-sm font-semibold tabular-nums">
                          {fmtNum(rec.keyValue, 4)}
                          <span className="ml-1 text-[10px] font-normal text-muted-foreground">
                            {rec.keyMetric}
                          </span>
                        </p>
                        {delta !== null ? (
                          <p
                            className={cn(
                              "flex items-center justify-end gap-0.5 font-mono text-[10px] tabular-nums",
                              up ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"
                            )}
                            title="Change vs the previous run"
                          >
                            {up ? (
                              <TrendingUp className="size-3" aria-hidden />
                            ) : (
                              <TrendingDown className="size-3" aria-hidden />
                            )}
                            {up ? "+" : "−"}
                            {fmtNum(Math.abs(delta), 3)}
                          </p>
                        ) : (
                          <p className="font-mono text-[10px] text-muted-foreground/60">—</p>
                        )}
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 shrink-0 px-2.5 text-xs"
                        onClick={() => restoreRun(rec)}
                        aria-label={`Restore run ${rec.model}${
                          rec.csv ? ` from ${rec.csv.fileName}` : ` seed ${rec.config.seed}`
                        }`}
                        disabled={isCurrent}
                      >
                        <RotateCcw className="size-3.5" aria-hidden />
                        <span className="hidden sm:inline">
                          {isCurrent ? "Current" : "Restore"}
                        </span>
                      </Button>
                    </li>
                  );
                })}
              </ul>
            </CardContent>
          </Card>
        ) : null}
      </div>
    </div>
  );
}
