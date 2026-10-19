"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Braces,
  Check,
  ClipboardCopy,
  Download,
  FileCode2,
  RefreshCw,
  Search,
  SearchX,
  TriangleAlert,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useToolkitStore } from "@/lib/store";
import { copyToClipboard, fmtInt, fuzzyScore } from "@/lib/format";
import { getReference, peekReference } from "@/lib/reference-cache";
import { HighlightText } from "@/components/explorer/highlight-text";
import type { ReferenceResponse, SymbolKind } from "@/lib/toolkit-types";

const KIND_FILTERS: Array<{ id: SymbolKind | "all"; label: string }> = [
  { id: "all", label: "All" },
  { id: "function", label: "Functions" },
  { id: "class", label: "Classes" },
  { id: "constant", label: "Constants" },
];

const KIND_STYLES: Record<SymbolKind, string> = {
  function: "border-primary/30 bg-primary/10 text-primary",
  class: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  module: "border-teal-500/40 bg-teal-500/10 text-teal-700 dark:text-teal-300",
  constant: "border-stone-500/40 bg-stone-500/10 text-stone-600 dark:text-stone-300",
};

/** Browsable, searchable rendering of docs/INDEX.md with view-source links. */
export function ApiReference() {
  const setTab = useToolkitStore((s) => s.setTab);
  const openFileAtLine = useToolkitStore((s) => s.openFileAtLine);

  const [data, setData] = useState<ReferenceResponse | null>(() => peekReference());
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    peekReference() ? "ready" : "loading"
  );
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<SymbolKind | "all">("all");
  /** Symbol whose reference was just copied (drives the check-icon swap). */
  const [copiedSym, setCopiedSym] = useState<string | null>(null);

  useEffect(() => {
    if (peekReference()) return;
    let stale = false;
    getReference()
      .then((payload) => {
        if (stale) return;
        setData(payload);
        setStatus("ready");
      })
      .catch((err: unknown) => {
        if (stale) return;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Unknown error.");
      });
    return () => {
      stale = true;
    };
  }, []);

  const trimmed = query.trim().toLowerCase();
  const searchActive = trimmed.length > 0;

  /** Unfiltered kind counts for the filter pills. */
  const kindCounts = useMemo(() => {
    const counts = new Map<string, number>();
    if (!data) return counts;
    for (const mod of data.modules) {
      for (const sym of mod.symbols) {
        counts.set(sym.kind, (counts.get(sym.kind) ?? 0) + 1);
      }
    }
    return counts;
  }, [data]);

  /** Unfiltered symbol count per module — powers the "n of m" badge while filtering. */
  const moduleTotals = useMemo(() => {
    const map = new Map<string, number>();
    if (!data) return map;
    for (const mod of data.modules) map.set(mod.name, mod.symbols.length);
    return map;
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.modules
      .map((mod) => {
        const scored = mod.symbols.map((sym) => {
          let keep = true;
          let score = 0;
          if (kindFilter !== "all" && sym.kind !== kindFilter) {
            keep = false;
          } else if (searchActive) {
            // Fuzzy match on the symbol / module name, substring on summaries.
            const nameScore = fuzzyScore(trimmed, sym.name);
            const modScore = fuzzyScore(trimmed, mod.name);
            const inSummary = sym.summary.toLowerCase().includes(trimmed);
            if (nameScore === null && modScore === null && !inSummary) {
              keep = false;
            } else {
              // Direct name hits outrank module-name hits, which outrank summaries.
              score = Math.max(
                nameScore ?? -1,
                modScore === null ? -1 : modScore - 400,
                inSummary ? 1 : -1
              );
            }
          }
          return { sym, score, keep };
        });
        const symbols = scored
          .filter((s) => s.keep)
          .sort((a, b) => (searchActive ? b.score - a.score : 0))
          .map((s) => s.sym);
        return { ...mod, symbols };
      })
      .filter((mod) => mod.symbols.length > 0);
  }, [data, kindFilter, trimmed, searchActive]);

  const visibleSymbols = filtered.reduce((acc, m) => acc + m.symbols.length, 0);

  /** Deep link: open the module source at the symbol's definition line. */
  function viewSource(sourcePath: string, moduleName: string, line?: number | null) {
    openFileAtLine(sourcePath, line ?? null);
    setTab("explorer");
    toast(
      line
        ? `Jumped to ${moduleName} · line ${line} in the Code Explorer`
        : `Opened ${moduleName} source in the Code Explorer`
    );
  }

  /** Copy a shareable `path#LNN` reference straight from the list. */
  async function copyReference(name: string, sourcePath: string, line?: number | null) {
    const reference = line ? `${sourcePath}#L${line}` : sourcePath;
    const ok = await copyToClipboard(reference);
    if (ok) {
      setCopiedSym(name);
      window.setTimeout(() => setCopiedSym((cur) => (cur === name ? null : cur)), 1600);
      toast.success("Line reference copied", { description: reference });
    } else {
      toast.error("Could not copy — your browser blocked clipboard access");
    }
  }

  /** Download the structured reference as a JSON file. */
  function exportJson() {
    if (!data) return;
    try {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "pharmaml-api-reference.json";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast.success("Exported pharmaml-api-reference.json");
    } catch {
      toast.error("Could not export the reference as JSON.");
    }
  }

  if (status === "error") {
    return (
      <Card className="py-0 shadow-sm">
        <CardContent className="space-y-3 py-8">
          <div className="flex items-start gap-2 text-sm text-destructive">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
            <div>
              <p className="font-medium">Could not load the API reference.</p>
              <p className="mt-0.5 text-xs text-muted-foreground">{error}</p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={() => {
              setStatus("loading");
              setError(null);
              window.setTimeout(() => window.location.reload(), 100);
            }}
          >
            <RefreshCw className="size-3.5" aria-hidden />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* ------------------------------ Header ------------------------------ */}
      <section aria-labelledby="reference-title" className="space-y-4">
        <div>
          <p className="eyebrow">Public API</p>
          <h2 id="reference-title" className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
            API reference
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Every public symbol of the <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">pharmaml</code>{" "}
            package, auto-derived from each module&apos;s <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">__all__</code>{" "}
            exports in <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">docs/INDEX.md</code>. Click a
            symbol to jump straight to its definition — or use the clipboard button on a row to copy its{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">path#LNN</code> reference.
          </p>
        </div>

        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="gap-1.5 tabular-nums">
              <Braces className="size-3 text-primary" aria-hidden />
              {data ? fmtInt(data.stats.modules) : "…"} modules
            </Badge>
            <Badge variant="outline" className="tabular-nums">
              {data ? fmtInt(data.stats.symbols) : "…"} public symbols
            </Badge>
            {searchActive || kindFilter !== "all" ? (
              <Badge
                variant="outline"
                className="border-primary/30 bg-primary/5 text-primary tabular-nums"
              >
                {fmtInt(visibleSymbols)} shown
              </Badge>
            ) : null}
          </div>

          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <div className="relative">
              <Search
                className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
                aria-hidden
              />
              <Input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search symbols (fuzzy works)…"
                aria-label="Search API symbols"
                className="h-9 w-full pl-8 pr-8 text-sm sm:w-56"
              />
              {searchActive ? (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear symbol search"
                  className="absolute right-2 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <X className="size-3.5" aria-hidden />
                </button>
              ) : null}
            </div>
            <div
              role="group"
              aria-label="Filter symbols by kind"
              className="flex gap-1 rounded-lg border bg-muted/40 p-1"
            >
              {KIND_FILTERS.map((f) => {
                const count =
                  f.id === "all"
                    ? data?.stats.symbols
                    : kindCounts.get(f.id);
                return (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => setKindFilter(f.id)}
                    aria-pressed={kindFilter === f.id}
                    className={cn(
                      "flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      kindFilter === f.id
                        ? "bg-background shadow-sm text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {f.label}
                    {typeof count === "number" ? (
                      <span className="font-mono text-[10px] tabular-nums opacity-60">
                        {fmtInt(count)}
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-9 shrink-0 gap-1.5"
              onClick={exportJson}
              disabled={!data}
              aria-label="Export the API reference as JSON"
            >
              <Download className="size-3.5" aria-hidden />
              <span className="hidden lg:inline">JSON</span>
            </Button>
          </div>
        </div>
      </section>

      {/* ------------------------------ Body ------------------------------ */}
      {status === "loading" || status === "idle" ? (
        <div className="space-y-4" aria-label="Loading API reference">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="py-0 shadow-sm">
              <CardContent className="space-y-2.5 py-4">
                <Skeleton className="h-5 w-48" />
                <Skeleton className="h-4 w-full max-w-md" />
                {Array.from({ length: 4 + (i % 3) }).map((__, j) => (
                  <Skeleton key={j} className="h-9 w-full" />
                ))}
              </CardContent>
            </Card>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card className="py-0 shadow-sm" role="status">
          <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
            <span className="flex size-12 items-center justify-center rounded-full border bg-muted/50">
              <SearchX className="size-6 text-muted-foreground" aria-hidden />
            </span>
            <div className="max-w-md space-y-1.5">
              <p className="text-sm font-medium">
                {searchActive ? "No symbols match your search" : "No symbols in this view"}
              </p>
              <p className="text-sm text-muted-foreground">
                {searchActive ? (
                  <>
                    Nothing matches{" "}
                    <span className="font-mono font-medium text-foreground">{query.trim()}</span>
                    {kindFilter !== "all"
                      ? ` with the ${KIND_FILTERS.find((f) => f.id === kindFilter)?.label.toLowerCase()} filter`
                      : ""}
                    .
                  </>
                ) : (
                  <>
                    The{" "}
                    {KIND_FILTERS.find((f) => f.id === kindFilter)?.label.toLowerCase()} filter hides every
                    symbol.
                  </>
                )}
              </p>
              <p className="text-xs text-muted-foreground/80">
                Fuzzy search tolerates typos and partial words — try a shorter query.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={() => {
                setQuery("");
                setKindFilter("all");
              }}
            >
              <X className="size-3.5" aria-hidden />
              Clear filters
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div>
          {filtered.map((mod, i) => {
            const moduleTotal = moduleTotals.get(mod.name);
            return (
              <Fragment key={mod.name}>
                {/* Subtle separator between module groups (decorative). */}
                {i > 0 ? (
                  <div aria-hidden className="my-2 flex items-center justify-center gap-2">
                    <span className="h-px w-10 bg-border/70" />
                    <Braces className="size-3 text-muted-foreground/40" />
                    <span className="h-px w-10 bg-border/70" />
                  </div>
                ) : null}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Math.min(i * 0.05, 0.25), duration: 0.2 }}
                >
                  <Card className="overflow-hidden py-0 shadow-sm">
                    <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 border-b bg-muted/30 py-3">
                      <div className="min-w-0">
                        <p className="flex flex-wrap items-center gap-2">
                          <span className="font-mono text-sm font-semibold">{mod.name}</span>
                          <Badge
                            variant="outline"
                            className="shrink-0 bg-muted/40 px-1.5 font-mono text-[10px] tabular-nums text-muted-foreground"
                          >
                            {moduleTotal !== undefined && moduleTotal !== mod.symbols.length
                              ? `${fmtInt(mod.symbols.length)} of ${fmtInt(moduleTotal)} symbols`
                              : `${fmtInt(mod.symbols.length)} symbol${mod.symbols.length === 1 ? "" : "s"}`}
                          </Badge>
                        </p>
                        <p className="mt-0.5 break-words text-xs leading-snug text-muted-foreground">{mod.blurb}</p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 shrink-0 gap-1.5 text-xs"
                        onClick={() => viewSource(mod.sourcePath, mod.name)}
                        aria-label={`View source of ${mod.name}`}
                      >
                        <FileCode2 className="size-3.5" aria-hidden />
                        View source
                      </Button>
                    </CardHeader>
                    <CardContent className="py-0">
                      <ul className="divide-y">
                        {mod.symbols.map((sym) => {
                          const symPath = sym.sourcePath ?? mod.sourcePath;
                          const symRef = sym.line ? `${symPath}#L${sym.line}` : symPath;
                          const copied = copiedSym === sym.name;
                          return (
                            <li key={sym.name} className="group/row flex items-stretch">
                              <button
                                type="button"
                                onClick={() => viewSource(symPath, mod.name, sym.line)}
                                aria-label={
                                  sym.line
                                    ? `Jump to ${sym.name} definition at line ${sym.line} of ${mod.name}`
                                    : `Open source of ${mod.name} (${sym.name})`
                                }
                                className="group flex min-w-0 flex-1 flex-col gap-1 px-3 py-2.5 text-left transition-colors hover:bg-primary/5 focus-visible:outline-none focus-visible:bg-primary/5 focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring sm:flex-row sm:items-center sm:gap-3"
                              >
                                <div className="flex min-w-0 items-center gap-2 sm:w-[240px] sm:shrink-0">
                                  <code className="break-all font-mono text-[13px] font-medium text-foreground underline decoration-primary/40 decoration-2 underline-offset-4 group-hover:underline group-focus-visible:underline">
                                    <HighlightText text={sym.name} query={query} />
                                  </code>
                                </div>
                              <div className="flex shrink-0 items-center gap-2">
                                <Badge
                                  variant="outline"
                                  className={cn(
                                    "px-1.5 py-0 font-mono text-[10px] uppercase tracking-wide",
                                    KIND_STYLES[sym.kind]
                                  )}
                                >
                                  {sym.kind}
                                </Badge>
                                {sym.line ? (
                                  <span className="font-mono text-[10px] tabular-nums text-muted-foreground/70">
                                    L{sym.line}
                                  </span>
                                ) : null}
                              </div>
                              <p className="min-w-0 flex-1 text-xs leading-relaxed text-muted-foreground">
                                {sym.summary || "—"}
                              </p>
                              <ArrowRight
                                className="hidden size-3.5 shrink-0 -translate-x-1 text-primary opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100 group-focus-visible:translate-x-0 group-focus-visible:opacity-100 sm:block"
                                aria-hidden
                              />
                            </button>
                            <button
                              type="button"
                              onClick={() => void copyReference(sym.name, symPath, sym.line)}
                              aria-label={`Copy line reference ${symRef}`}
                              title={`Copy ${symRef}`}
                              className={cn(
                                "flex w-10 shrink-0 items-center justify-center border-l border-transparent text-muted-foreground/60 transition-all hover:border-l-border hover:bg-primary/5 hover:text-primary focus-visible:outline-none focus-visible:bg-primary/5 focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring sm:opacity-0 sm:group-hover/row:opacity-100 sm:group-focus-within/row:opacity-100 sm:focus-visible:opacity-100",
                                copied && "text-primary sm:opacity-100"
                              )}
                            >
                              {copied ? (
                                <Check className="size-3.5" aria-hidden />
                              ) : (
                                <ClipboardCopy className="size-3.5" aria-hidden />
                              )}
                            </button>
                          </li>
                        );
                      })}
                      </ul>
                    </CardContent>
                  </Card>
                </motion.div>
              </Fragment>
            );
          })}
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        Generated from <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">docs/INDEX.md</code>{" "}
        — regenerate after adding public symbols with{" "}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">python scratch/build_docs_index.py</code>.
      </p>
    </div>
  );
}
