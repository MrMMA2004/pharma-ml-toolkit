"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Check,
  CircleCheck,
  Copy,
  FileArchive,
  FileCode2,
  FileDown,
  FileText,
  Loader2,
  NotebookText,
  Scale,
  ShieldCheck,
  TerminalSquare,
  TriangleAlert,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { StatChip } from "@/components/explorer/stat-chip";
import { useToolkitStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { copyToClipboard, fmtBytes, fmtInt } from "@/lib/format";
import type { FileContentResponse, TreeNode } from "@/lib/toolkit-types";
import { toast } from "sonner";
import type { LucideIcon } from "lucide-react";

/** One numbered step of the install guide. */
interface InstallStep {
  title: string;
  description: string;
  commands: string[];
  /** Optional pip extras — each chip copies its own install command. */
  extras?: Array<{ name: string; note: string }>;
}

const INSTALL_STEPS: InstallStep[] = [
  {
    title: "Unpack the archive",
    description: "Creates a pharmaml/ working copy and moves the terminal into it.",
    commands: ["unzip pharma-ml-toolkit.zip -d pharmaml && cd pharmaml"],
  },
  {
    title: "Install the package",
    description: "Editable core install — ML, pharma math, and docking parsing/scoring.",
    commands: ["pip install -e ."],
    extras: [
      { name: ".[chem]", note: "cheminformatics (RDKit)" },
      { name: ".[docking]", note: "docking prep (RDKit + Meeko)" },
      { name: ".[boosting]", note: "gradient-boosting model families" },
    ],
  },
  {
    title: "Verify the install",
    description: "1128 tests should pass and the CLI answers with its version.",
    commands: ["python -m pytest", "pharmaml --version"],
  },
];

/**
 * Compact terminal command strip: a header row naming the command context
 * ("terminal", "terminal · macOS", …) with an icon-only copy button, then the
 * command itself behind a `$` prompt in monospace. Copy feedback swaps the
 * icon to a check, flips aria-pressed and announces itself via a live region.
 */
function CommandStrip({
  command,
  context = "terminal",
  hint,
  className,
}: {
  command: string;
  context?: string;
  hint?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    const ok = await copyToClipboard(command);
    if (ok) {
      setCopied(true);
      toast.success("Command copied", { description: command });
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  return (
    <div className={cn("overflow-hidden rounded-lg border bg-card text-left", className)}>
      <div className="flex items-center justify-between gap-2 border-b bg-muted/40 py-1 pl-2.5 pr-1.5">
        <span className="flex min-w-0 items-center gap-1.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
          <TerminalSquare className="size-3 shrink-0 text-primary" aria-hidden />
          <span className="truncate">{context}</span>
        </span>
        <button
          type="button"
          onClick={() => void copy()}
          aria-label={`Copy command to clipboard: ${command}`}
          aria-pressed={copied}
          className="flex size-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {copied ? (
            <Check className="size-3.5 text-primary" aria-hidden />
          ) : (
            <Copy className="size-3.5" aria-hidden />
          )}
          <span aria-live="polite" className="sr-only">
            {copied ? "Copied to clipboard" : ""}
          </span>
        </button>
      </div>
      <div className="flex items-baseline gap-2 px-2.5 py-2">
        <span
          aria-hidden
          className="shrink-0 select-none font-mono text-[13px] leading-6 text-muted-foreground/60"
        >
          $
        </span>
        <code className="min-w-0 flex-1 break-all font-mono text-[13px] leading-6">{command}</code>
      </div>
      {hint ? (
        <p className="border-t bg-muted/20 px-2.5 py-1.5 text-[11px] leading-snug text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/**
 * A pip extra rendered as a copy-to-clipboard chip (e.g. `.[chem]`) — the
 * full `pip install -e "…"` command is copied, the chip label stays compact.
 */
function ExtraChip({ name, note }: { name: string; note: string }) {
  const [copied, setCopied] = useState(false);
  const command = `pip install -e "${name}"`;

  async function copy() {
    const ok = await copyToClipboard(command);
    if (ok) {
      setCopied(true);
      toast.success("Command copied", { description: command });
      window.setTimeout(() => setCopied(false), 1600);
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  return (
    <button
      type="button"
      onClick={() => void copy()}
      aria-label={`Copy install command for the ${note} extra: ${command}`}
      aria-pressed={copied}
      title={`${note} — copies ${command}`}
      className={cn(
        "inline-flex items-center gap-1 rounded-md border bg-muted/40 px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        copied && "border-primary/40 bg-primary/5 text-foreground"
      )}
    >
      {name}
      {copied ? (
        <Check className="size-3 text-primary" aria-hidden />
      ) : (
        <Copy className="size-3 opacity-50" aria-hidden />
      )}
    </button>
  );
}

function countFiles(node: TreeNode): number {
  if (node.type === "file") return 1;
  return (node.children ?? []).reduce((acc, child) => acc + countFiles(child), 0);
}

/** Look up a node in the tree by its toolkit-relative path. */
function findNode(node: TreeNode, path: string): TreeNode | null {
  if (node.path === path) return node;
  for (const child of node.children ?? []) {
    const hit = findNode(child, path);
    if (hit) return hit;
  }
  return null;
}

/** Hand-picked highlights of the archive, with live tree metadata. */
const FEATURED_FILES: Array<{
  path: string;
  title: string;
  description: string;
  icon: LucideIcon;
  /** Tailwind tint classes for the icon badge. */
  badgeClass: string;
}> = [
  {
    path: "examples/complete_ml_workflow.ipynb",
    title: "Complete ML workflow (zero → one hundred)",
    description:
      "Executed Jupyter notebook — the full ML process in structured order: import your own file → SMILES → descriptors & fingerprints (feature calculation) → EDA → pIC50 → split → tuned training → multiclass evaluation → leaderboard → feature importance → save/load → HTML report. 15 figures committed inline.",
    icon: NotebookText,
    badgeClass: "bg-primary/10 text-primary",
  },
  {
    path: "examples/complete_docking_workflow.ipynb",
    title: "Complete docking workflow (zero → one hundred)",
    description:
      "Executed Jupyter notebook — starts from a ligands.csv and a real lysozyme PDB: library import with failed-SMILES accounting, receptor cleaning/typing via Meeko, then the full grid/engine/parse/rescore/report workflow. Runs without a docking engine.",
    icon: NotebookText,
    badgeClass: "bg-teal-500/10 text-teal-600 dark:text-teal-400",
  },
  {
    path: "examples/complete_pk_workflow.ipynb",
    title: "Complete PK/PD workflow (zero → one hundred)",
    description:
      "Executed Jupyter notebook — the pharma walkthrough the legacy archive never had: import your concentration–time file, NCA (AUC/t½/Cmax), derived CL & Vd, IV routes, accumulation to steady state, Emax PD, dosage design, renal & TDM adjustments, one-click HTML report. 3 figures committed inline.",
    icon: NotebookText,
    badgeClass: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  },
  {
    path: "examples/complete_ml_workflow_no_output.ipynb",
    title: "No-output notebook twins (lightweight)",
    description:
      "Bandwidth-friendly twins of the three workflows — identical cells, every output stripped: ML 44 KB (4% of the executed original), docking 80 KB, PK 42 KB. Open the full twin to read results, run this one to produce them yourself.",
    icon: NotebookText,
    badgeClass: "bg-teal-500/10 text-teal-600 dark:text-teal-400",
  },
  {
    path: "examples/pharmaml_walkthrough.ipynb",
    title: "Interactive walkthrough",
    description:
      "Executed Jupyter notebook — an every-domain tour of the package on synthetic data: ML, PK/PD dosage, docking rescoring and endpoint conversion.",
    icon: NotebookText,
    badgeClass: "bg-primary/10 text-primary",
  },
  {
    path: "docs/INDEX.md",
    title: "Public API index",
    description:
      "Auto-generated module index — every public symbol with a one-line summary, kept in sync by the docs drift-guard tests.",
    icon: FileText,
    badgeClass: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  },
];

function collectDirSummary(
  root: TreeNode
): Array<{ name: string; files: number; size: number; note: string }> {
  const notes: Record<string, string> = {
    src: "the pharmaml package (data, chem, ml, pharma, docking, viz, utils + CLI)",
    tests: "1128 pytest cases across all domains",
    examples: "four executed notebooks + three no-output twins + runnable demos",
    ".github": "GitHub Actions CI matrix (Python 3.10–3.12)",
    docs: "module index — auto-generated API reference",
    scratch: "idempotent build & patch scripts (notebooks, docs index)",
  };
  const out: Array<{ name: string; files: number; size: number; note: string }> = [];
  let rootFiles = 0;
  let rootBytes = 0;
  for (const child of root.children ?? []) {
    if (child.type === "dir") {
      out.push({
        name: child.name,
        files: countFiles(child),
        size: child.size,
        note: notes[child.name] ?? "",
      });
    } else {
      rootFiles++;
      rootBytes += child.size;
    }
  }
  if (rootFiles > 0) {
    out.push({
      name: "(root)",
      files: rootFiles,
      size: rootBytes,
      note: "README, pyproject.toml, Dockerfile, requirements.txt, .gitignore",
    });
  }
  return out;
}

export function DownloadPanel() {
  const ensureTree = useToolkitStore((s) => s.ensureTree);
  const tree = useToolkitStore((s) => s.tree);
  const stats = useToolkitStore((s) => s.stats);
  const treeStatus = useToolkitStore((s) => s.treeStatus);
  const setTab = useToolkitStore((s) => s.setTab);
  const openFileAtLine = useToolkitStore((s) => s.openFileAtLine);

  const [downloading, setDownloading] = useState(false);
  /** Path of the featured file currently being fetched (spinner state). */
  const [fileBusy, setFileBusy] = useState<string | null>(null);

  useEffect(() => {
    void ensureTree();
  }, [ensureTree]);

  const dirSummary = useMemo(() => (tree ? collectDirSummary(tree) : []), [tree]);

  /** Total files across the top-level folders — denominator for the share bars. */
  const totalFiles = useMemo(
    () => dirSummary.reduce((acc, d) => acc + d.files, 0),
    [dirSummary]
  );

  /** Live metadata (size / lines) for each featured file, from the tree. */
  const featuredMeta = useMemo(() => {
    const map = new Map<string, TreeNode | null>();
    for (const f of FEATURED_FILES) {
      map.set(f.path, tree ? findNode(tree, f.path) : null);
    }
    return map;
  }, [tree]);

  async function download() {
    setDownloading(true);
    try {
      const res = await fetch("/api/toolkit/download");
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
      if (blob.size < 1024) {
        throw new Error("The archive came back suspiciously small — try again.");
      }
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "pharma-ml-toolkit.zip";
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
      toast.success("Download started", {
        description: `pharma-ml-toolkit.zip · ${fmtBytes(blob.size)}`,
      });
    } catch (err) {
      toast.error("Download failed", {
        description: err instanceof Error ? err.message : "Unknown error.",
      });
    } finally {
      setDownloading(false);
    }
  }

  /** Deep link a featured file into the Code Explorer. */
  function viewFeatured(path: string, title: string) {
    openFileAtLine(path, null);
    setTab("explorer");
    toast(`Opened ${title} in the Code Explorer`, { description: path });
  }

  /** Fetch a single toolkit file and save it locally as a text blob. */
  async function downloadSingleFile(path: string) {
    setFileBusy(path);
    try {
      const res = await fetch(`/api/toolkit/file?path=${encodeURIComponent(path)}`);
      const body: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof body === "object" && body !== null && "error" in body
            ? String((body as { error: unknown }).error)
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const payload = body as FileContentResponse;
      const blob = new Blob([payload.content], { type: "text/plain;charset=utf-8" });
      const fileName = path.split("/").pop() ?? "file.txt";
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = fileName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast.success("Download started", { description: `${fileName} · ${fmtBytes(blob.size)}` });
    } catch (err) {
      toast.error("Download failed", {
        description: err instanceof Error ? err.message : "Unknown error.",
      });
    } finally {
      setFileBusy(null);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {/* ------------------------------ CTA ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="dot-grid border-b bg-gradient-to-br from-primary/10 via-transparent to-amber-500/10 py-6">
          <div className="flex flex-col items-center gap-4 text-center">
            <span className="flex size-14 items-center justify-center rounded-xl bg-primary/15 text-primary">
              <FileArchive className="size-7" aria-hidden />
            </span>
            <div className="space-y-1">
              <CardTitle className="text-xl">Download pharma-ml-toolkit.zip</CardTitle>
              <CardDescription>
                The complete Python package — source, tests, examples, Dockerfile and CI.
              </CardDescription>
            </div>
            <Button
              size="lg"
              className="h-12 gap-2 px-8 text-base"
              onClick={() => void download()}
              disabled={downloading}
            >
              {downloading ? (
                <>
                  <Loader2 className="size-5 animate-spin" aria-hidden />
                  Packaging…
                </>
              ) : (
                <>
                  <FileArchive className="size-5" aria-hidden />
                  Download ZIP
                  {stats ? (
                    <span className="ml-1 rounded-full bg-primary-foreground/15 px-2 py-0.5 font-mono text-xs font-normal tabular-nums">
                      ≈ {fmtBytes(stats.totalBytes)}
                    </span>
                  ) : null}
                </>
              )}
            </Button>
            <p className="text-xs text-muted-foreground">
              Generated on the fly from the server copy — always in sync with the explorer.
            </p>
          </div>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3 py-4 lg:grid-cols-4">
          <StatChip icon={FileCode2} value={treeStatus === "ready" && stats ? fmtInt(stats.files) : "…"} label="files" />
          <StatChip icon={CircleCheck} value="1128" label="passing tests" />
          <StatChip icon={Scale} value="MIT" label="license" />
          <StatChip
            icon={FileArchive}
            value={stats ? fmtBytes(stats.totalBytes) : "…"}
            label="uncompressed"
          />
        </CardContent>
      </Card>

      {/* --------------------------- Featured files --------------------------- */}
      <section aria-labelledby="featured-title" className="space-y-3">
        <h2 id="featured-title" className="flex items-center gap-2 text-lg font-semibold">
          <FileDown className="size-5 text-primary" aria-hidden />
          Featured files
        </h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {FEATURED_FILES.map((f) => {
            const meta = featuredMeta.get(f.path) ?? null;
            const busy = fileBusy === f.path;
            const Icon = f.icon;
            return (
              <Card key={f.path} className="featured-card overflow-hidden py-0 shadow-sm">
                <CardHeader className="border-b bg-muted/30 py-3">
                  <div className="flex items-start gap-2.5">
                    <span className={`flex size-9 shrink-0 items-center justify-center rounded-lg ${f.badgeClass}`}>
                      <Icon className="size-4.5" aria-hidden />
                    </span>
                    <div className="min-w-0">
                      <CardTitle className="text-sm">{f.title}</CardTitle>
                      <p className="mt-0.5 break-all font-mono text-[11px] leading-snug text-muted-foreground">
                        {f.path}
                      </p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3 py-3">
                  <p className="text-xs leading-relaxed text-muted-foreground">{f.description}</p>
                  <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <span className="rounded-md border bg-muted/40 px-1.5 py-0.5 font-mono tabular-nums">
                      {meta ? fmtBytes(meta.size) : "…"}
                    </span>
                    {typeof meta?.lines === "number" ? (
                      <span className="rounded-md border bg-muted/40 px-1.5 py-0.5 font-mono tabular-nums">
                        {fmtInt(meta.lines)} lines
                      </span>
                    ) : null}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 flex-1 gap-1.5 text-xs"
                      onClick={() => viewFeatured(f.path, f.title)}
                    >
                      <FileCode2 className="size-3.5" aria-hidden />
                      View
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 flex-1 gap-1.5 text-xs"
                      onClick={() => void downloadSingleFile(f.path)}
                      disabled={busy}
                    >
                      {busy ? (
                        <Loader2 className="size-3.5 animate-spin" aria-hidden />
                      ) : (
                        <FileDown className="size-3.5" aria-hidden />
                      )}
                      Download
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* ------------------------------ Contents ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <CardTitle className="text-base">What&apos;s inside</CardTitle>
          <CardDescription className="text-xs">
            {treeStatus === "loading"
              ? "Counting files…"
              : treeStatus === "error"
                ? "Tree unavailable — using the documented layout."
                : "File counts computed live from the server file tree."}
          </CardDescription>
        </CardHeader>
        <CardContent className="pb-4 pt-3">
          {treeStatus === "loading" ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-6 animate-pulse rounded bg-muted" />
              ))}
            </div>
          ) : dirSummary.length > 0 ? (
            <ul className="divide-y divide-border">
              {dirSummary.map((d) => {
                const share = Math.round((d.files / Math.max(1, totalFiles)) * 100);
                return (
                  <li
                    key={d.name}
                    className="py-2 transition-colors hover:bg-muted/40"
                  >
                    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                      <code className="font-mono text-sm font-medium">
                        {d.name === "(root)" ? d.name : `${d.name}/`}
                      </code>
                      <span className="rounded-md border bg-muted/40 px-1.5 py-0.5 font-mono text-xs tabular-nums text-muted-foreground">
                        {fmtInt(d.files)} {d.files === 1 ? "file" : "files"}
                      </span>
                      <span className="rounded-md border bg-muted/40 px-1.5 py-0.5 font-mono text-xs tabular-nums text-muted-foreground">
                        {fmtBytes(d.size)}
                      </span>
                      <span
                        className="font-mono text-[10px] tabular-nums text-muted-foreground/70"
                        title={`${fmtInt(d.files)} of ${fmtInt(totalFiles)} files`}
                      >
                        {fmtInt(share)}%
                      </span>
                      {d.note ? (
                        <span className="w-full text-xs text-muted-foreground/80 sm:w-auto">
                          — {d.note}
                        </span>
                      ) : null}
                    </div>
                    {/* File-count share of the whole archive (decorative). */}
                    <div
                      aria-hidden
                      className="mt-1.5 h-1 overflow-hidden rounded-full bg-muted"
                    >
                      <div
                        className="h-full rounded-full bg-primary/35"
                        style={{ width: `${Math.max(share, 2)}%` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          ) : (
            <div className="flex items-start gap-2 text-sm text-muted-foreground">
              <TriangleAlert className="mt-0.5 size-4 shrink-0 text-amber-500" aria-hidden />
              <span>
                src/ (chem, ml, pharma, docking, viz, utils), tests/, examples/, .github/, Dockerfile,
                pyproject.toml, requirements.txt.
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ------------------------------ Verify ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <ShieldCheck className="size-4 text-primary" aria-hidden />
            Verify the archive
          </CardTitle>
          <CardDescription className="text-xs">
            The ZIP is rebuilt on the fly from the live tree — check your copy landed intact.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 py-3">
          <CommandStrip
            command="unzip -t pharma-ml-toolkit.zip"
            hint="Tests the CRC-32 checksum of every entry — a healthy archive reports “No errors detected”."
          />
          <div className="grid gap-2 sm:grid-cols-2">
            <CommandStrip
              command="shasum -a 256 pharma-ml-toolkit.zip"
              context="terminal · macOS"
            />
            <CommandStrip
              command="sha256sum pharma-ml-toolkit.zip"
              context="terminal · linux"
            />
          </div>
          {/* Expected contents — counted client-side from the existing tree API. */}
          <div className="flex flex-wrap items-end gap-x-8 gap-y-3 rounded-lg border bg-muted/20 px-3 py-2.5">
            {[
              { label: "entries", value: stats ? fmtInt(stats.files + stats.dirs) : "…" },
              { label: "files", value: stats ? fmtInt(stats.files) : "…" },
              { label: "folders", value: stats ? fmtInt(stats.dirs) : "…" },
            ].map((s) => (
              <div key={s.label}>
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{s.label}</p>
                <p className="font-mono text-lg font-semibold leading-tight tabular-nums">{s.value}</p>
              </div>
            ))}
            <p className="w-full text-[11px] leading-snug text-muted-foreground/80 sm:ml-auto sm:w-auto sm:text-right">
              expected contents, counted live from the server tree — mirrors “What’s inside” above.
            </p>
          </div>
          <p className="text-[11px] leading-snug text-muted-foreground">
            Because the archive is regenerated per request, its SHA-256 differs between downloads — hash your own
            copy to re-verify it later or share it with a colleague.
          </p>
        </CardContent>
      </Card>

      {/* ------------------------------ Install ------------------------------ */}
      <section aria-labelledby="install-title" className="space-y-3">
        <h2 id="install-title" className="flex items-center gap-2 text-lg font-semibold">
          <TerminalSquare className="size-5 text-primary" aria-hidden />
          Install &amp; verify
        </h2>
        <Card className="py-0 shadow-sm">
          <CardContent className="py-4">
            <ol>
              {INSTALL_STEPS.map((step, i) => (
                <li key={step.title} className="flex gap-3.5 sm:gap-4">
                  {/* Self-numbering circle + connector line down to the next step. */}
                  <div className="flex flex-col items-center">
                    <span
                      aria-hidden
                      className="flex size-6 shrink-0 items-center justify-center rounded-full border border-primary/30 bg-primary/10 font-mono text-[11px] font-semibold text-primary tabular-nums"
                    >
                      {i + 1}
                    </span>
                    {i < INSTALL_STEPS.length - 1 ? (
                      <span aria-hidden className="mt-1 w-px flex-1 bg-border" />
                    ) : null}
                  </div>
                  <div
                    className={cn(
                      "min-w-0 flex-1 space-y-2",
                      i < INSTALL_STEPS.length - 1 && "pb-5"
                    )}
                  >
                    <div>
                      <p className="text-sm font-medium leading-6">{step.title}</p>
                      <p className="text-xs leading-snug text-muted-foreground">{step.description}</p>
                    </div>
                    <div
                      className={cn("grid gap-2", step.commands.length > 1 && "sm:grid-cols-2")}
                    >
                      {step.commands.map((command) => (
                        <CommandStrip key={command} command={command} />
                      ))}
                    </div>
                    {step.extras ? (
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                          extras
                        </span>
                        {step.extras.map((extra) => (
                          <ExtraChip key={extra.name} name={extra.name} note={extra.note} />
                        ))}
                      </div>
                    ) : null}
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      </section>

      {/* ------------------------------ License ------------------------------ */}
      <Card className="border py-0 shadow-sm">
        <CardContent className="flex flex-wrap items-center justify-between gap-3 py-4">
          <div className="flex items-center gap-3">
            <Scale className="size-5 text-muted-foreground" aria-hidden />
            <div>
              <p className="text-sm font-medium">MIT License</p>
              <p className="text-xs text-muted-foreground">
                Free for research and education — attribution appreciated.
              </p>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Research &amp; education only — not medical advice.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
