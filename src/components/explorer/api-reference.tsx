"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Braces,
  FileCode2,
  RefreshCw,
  Search,
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
import { fmtInt } from "@/lib/format";
import type { ReferenceResponse, SymbolKind } from "@/lib/toolkit-types";

/** Module-level cache so tab switches never refetch. */
let refCache: ReferenceResponse | null = null;

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
  const setSelectedFile = useToolkitStore((s) => s.setSelectedFile);

  const [data, setData] = useState<ReferenceResponse | null>(refCache);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    refCache ? "ready" : "loading"
  );
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<SymbolKind | "all">("all");

  useEffect(() => {
    if (refCache) return;
    let stale = false;
    const load = async () => {
      try {
        const res = await fetch("/api/toolkit/reference");
        const body: unknown = await res.json();
        if (!res.ok) {
          const message =
            typeof body === "object" && body !== null && "detail" in body
              ? String((body as { detail: unknown }).detail)
              : `HTTP ${res.status}`;
          throw new Error(message);
        }
        const payload = body as ReferenceResponse;
        if (stale) return;
        refCache = payload;
        setData(payload);
        setStatus("ready");
      } catch (err: unknown) {
        if (stale) return;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Unknown error.");
      }
    };
    void load();
    return () => {
      stale = true;
    };
  }, []);

  const trimmed = query.trim().toLowerCase();
  const searchActive = trimmed.length > 0;

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.modules
      .map((mod) => ({
        ...mod,
        symbols: mod.symbols.filter((sym) => {
          if (kindFilter !== "all" && sym.kind !== kindFilter) return false;
          if (!searchActive) return true;
          return (
            sym.name.toLowerCase().includes(trimmed) ||
            sym.summary.toLowerCase().includes(trimmed) ||
            mod.name.toLowerCase().includes(trimmed)
          );
        }),
      }))
      .filter((mod) => mod.symbols.length > 0);
  }, [data, kindFilter, trimmed, searchActive]);

  const visibleSymbols = filtered.reduce((acc, m) => acc + m.symbols.length, 0);

  function viewSource(sourcePath: string, moduleName: string) {
    setSelectedFile(sourcePath);
    setTab("explorer");
    toast(`Opened ${moduleName} source in the Code Explorer`);
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
              refCache = null;
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
            exports in <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">docs/INDEX.md</code>.
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
                placeholder="Search symbols…"
                aria-label="Search API symbols"
                className="h-9 w-full pl-8 pr-8 text-sm sm:w-56"
              />
              {searchActive ? (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear symbol search"
                  className="absolute right-2 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
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
              {KIND_FILTERS.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => setKindFilter(f.id)}
                  aria-pressed={kindFilter === f.id}
                  className={cn(
                    "h-7 rounded-md px-2.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    kindFilter === f.id
                      ? "bg-background shadow-sm text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {f.label}
                </button>
              ))}
            </div>
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
        <Card className="py-0 shadow-sm">
          <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
            <Search className="size-7 text-muted-foreground/50" aria-hidden />
            <p className="text-sm text-muted-foreground">
              No symbols match{" "}
              <span className="font-mono font-medium text-foreground">
                {query.trim()}
              </span>
              {kindFilter !== "all" ? ` (${kindFilter}s only)` : ""}.
            </p>
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
        <div className="space-y-4">
          {filtered.map((mod, i) => (
            <motion.div
              key={mod.name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.05, 0.25), duration: 0.2 }}
            >
              <Card className="overflow-hidden py-0 shadow-sm">
                <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 border-b bg-muted/30 py-3">
                  <div className="min-w-0">
                    <p className="flex flex-wrap items-baseline gap-2">
                      <span className="font-mono text-sm font-semibold">{mod.name}</span>
                      <span className="text-xs text-muted-foreground">
                        {mod.symbols.length} symbol{mod.symbols.length === 1 ? "" : "s"}
                      </span>
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
                    {mod.symbols.map((sym) => (
                      <li
                        key={sym.name}
                        className="flex flex-col gap-1 py-2.5 transition-colors hover:bg-primary/5 sm:flex-row sm:items-center sm:gap-3"
                      >
                        <div className="flex min-w-0 items-center gap-2 sm:w-[240px] sm:shrink-0">
                          <code className="break-all font-mono text-[13px] font-medium text-foreground">
                            {sym.name}
                          </code>
                        </div>
                        <Badge
                          variant="outline"
                          className={cn(
                            "w-fit shrink-0 px-1.5 py-0 font-mono text-[10px] uppercase tracking-wide",
                            KIND_STYLES[sym.kind]
                          )}
                        >
                          {sym.kind}
                        </Badge>
                        <p className="min-w-0 flex-1 text-xs leading-relaxed text-muted-foreground">
                          {sym.summary || "—"}
                        </p>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </motion.div>
          ))}
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
