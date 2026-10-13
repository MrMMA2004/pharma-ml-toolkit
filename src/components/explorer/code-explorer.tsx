"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CornerDownLeft,
  FileCode2,
  History,
  RefreshCw,
  Search,
  TriangleAlert,
  X,
} from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { CodeBlock } from "@/components/explorer/code-block";
import { FileTree } from "@/components/explorer/file-tree";
import { useToolkitStore } from "@/lib/store";
import { fmtBytes, fmtInt } from "@/lib/format";
import type { FileContentResponse, TreeNode } from "@/lib/toolkit-types";

/** Module-level cache of file contents so tab switches never refetch. */
const fileCache = new Map<string, FileContentResponse>();

/** Count files in a (sub)tree. */
function countFiles(node: TreeNode): number {
  if (node.type === "file") return 1;
  return (node.children ?? []).reduce((acc, child) => acc + countFiles(child), 0);
}

/** Collect every file path in the tree into a Set. */
function collectPaths(node: TreeNode, into: Set<string>): void {
  if (node.type === "file") {
    into.add(node.path);
    return;
  }
  for (const child of node.children ?? []) collectPaths(child, into);
}

/** Max chips rendered in the recent strip. */
const RECENT_SHOWN = 6;

/**
 * Parse a pasted line reference like `src/pharmaml/chem/converters.py#L94`.
 * Returns null when the query is a plain filter string.
 */
export function parsePermalink(raw: string): { path: string; line: number } | null {
  const match = /^\s*(.+?)\s*#L(\d+)\s*$/i.exec(raw);
  if (!match) return null;
  const line = Number(match[2]);
  return line > 0 ? { path: match[1], line } : null;
}

export function CodeExplorer() {
  const ensureTree = useToolkitStore((s) => s.ensureTree);
  const tree = useToolkitStore((s) => s.tree);
  const stats = useToolkitStore((s) => s.stats);
  const treeStatus = useToolkitStore((s) => s.treeStatus);
  const treeError = useToolkitStore((s) => s.treeError);

  const selectedFile = useToolkitStore((s) => s.selectedFile);
  const setSelectedFile = useToolkitStore((s) => s.setSelectedFile);
  const targetLine = useToolkitStore((s) => s.targetLine);
  const setTargetLine = useToolkitStore((s) => s.setTargetLine);
  const openFileAtLine = useToolkitStore((s) => s.openFileAtLine);
  const recentFiles = useToolkitStore((s) => s.recentFiles);
  const clearRecentFiles = useToolkitStore((s) => s.clearRecentFiles);

  const [content, setContent] = useState<FileContentResponse | null>(null);
  const [fileStatus, setFileStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [fileError, setFileError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const fetchSeq = useRef(0);

  useEffect(() => {
    void ensureTree();
  }, [ensureTree]);

  const loadFile = useCallback(
    async (path: string) => {
      const cached = fileCache.get(path);
      if (cached) {
        setContent(cached);
        setFileStatus("ready");
        setFileError(null);
        return;
      }
      const seq = ++fetchSeq.current;
      setFileStatus("loading");
      setFileError(null);
      try {
        const res = await fetch(`/api/toolkit/file?path=${encodeURIComponent(path)}`);
        const data: unknown = await res.json();
        if (!res.ok) {
          const message =
            typeof data === "object" && data !== null && "error" in data
              ? String((data as { error: unknown }).error)
              : `HTTP ${res.status}`;
          throw new Error(message);
        }
        const payload = data as FileContentResponse;
        if (seq !== fetchSeq.current) return; // stale response
        fileCache.set(path, payload);
        setContent(payload);
        setFileStatus("ready");
      } catch (err) {
        if (seq !== fetchSeq.current) return;
        setFileStatus("error");
        setFileError(err instanceof Error ? err.message : "Failed to load the file.");
      }
    },
    []
  );

  useEffect(() => {
    void loadFile(selectedFile);
  }, [selectedFile, loadFile]);

  const trimmed = query.trim().toLowerCase();
  const permalink = parsePermalink(query);
  const searchActive = trimmed.length > 0 && !permalink;

  /** All file paths — used to validate pasted references against the tree. */
  const allPaths = useMemo(() => {
    const set = new Set<string>();
    if (tree) collectPaths(tree, set);
    return set;
  }, [tree]);

  /** Resolve a pasted path: exact, then case-insensitive fallback. */
  const resolvePastedPath = useCallback(
    (path: string): string | null => {
      if (allPaths.has(path)) return path;
      const lower = path.toLowerCase();
      for (const candidate of allPaths) {
        if (candidate.toLowerCase() === lower) return candidate;
      }
      return null;
    },
    [allPaths]
  );

  const jumpTarget = useMemo(() => {
    if (!permalink) return null;
    const resolved = resolvePastedPath(permalink.path);
    return { requested: permalink, resolved };
  }, [permalink, resolvePastedPath]);

  const jumpToPermalink = useCallback(() => {
    if (!jumpTarget) return;
    const { requested, resolved } = jumpTarget;
    if (resolved) {
      openFileAtLine(resolved, requested.line);
      setQuery("");
      toast.success("Opened line reference", {
        description: `${resolved}#L${requested.line}`,
      });
    } else {
      toast.error("No such file in the toolkit", {
        description: requested.path,
      });
    }
  }, [jumpTarget, openFileAtLine]);

  const filteredTree = useMemo(() => {
    if (!tree || !searchActive) return tree;
    const matchesPath = (p: string) => p.toLowerCase().includes(trimmed);
    const prune = (node: TreeNode): TreeNode | null => {
      if (node.type === "file") return matchesPath(node.path) ? node : null;
      const children = (node.children ?? [])
        .map((child) => prune(child))
        .filter((child): child is TreeNode => child !== null);
      return children.length > 0 ? { ...node, children } : null;
    };
    const rootChildren = (tree.children ?? [])
      .map((child) => prune(child))
      .filter((child): child is TreeNode => child !== null);
    return { ...tree, children: rootChildren };
  }, [tree, trimmed, searchActive]);
  const matchCount = filteredTree ? countFiles(filteredTree) : 0;

  return (
    <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
      {/* ------------------------------ Sidebar ------------------------------ */}
      <Card className="h-fit overflow-hidden py-0 shadow-sm lg:sticky lg:top-24">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2">
            <FileCode2 className="size-4 text-primary" aria-hidden />
            <p className="truncate font-mono text-sm font-semibold">pharma-ml-toolkit/</p>
          </div>
          <div className="relative mt-2">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <Input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && permalink) {
                  e.preventDefault();
                  jumpToPermalink();
                }
              }}
              placeholder="Filter files… or paste path#L94"
              aria-label="Filter files in the toolkit tree, or paste a path#LNN line reference"
              className="h-9 pl-8 pr-8 text-sm"
            />
            {searchActive || permalink ? (
              <button
                type="button"
                onClick={() => setQuery("")}
                aria-label="Clear filter"
                className="absolute right-2 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <X className="size-3.5" aria-hidden />
              </button>
            ) : null}
          </div>
          {permalink ? (
            <button
              type="button"
              onClick={jumpToPermalink}
              aria-label={
                jumpTarget?.resolved
                  ? `Jump to ${jumpTarget.resolved} line ${jumpTarget.requested.line}`
                  : `No file matches ${permalink.path}`
              }
              className={
                "hint-enter mt-2 flex w-full items-center gap-2 rounded-md border px-2.5 py-2 text-left text-xs transition-colors " +
                (jumpTarget?.resolved
                  ? "border-primary/30 bg-primary/5 text-foreground hover:bg-primary/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                  : "border-destructive/30 bg-destructive/5 text-destructive")
              }
            >
              <CornerDownLeft className="size-3.5 shrink-0" aria-hidden />
              <span className="min-w-0 flex-1 truncate font-mono">
                {jumpTarget?.resolved ?? permalink.path}
                <span className="font-semibold">#L{permalink.line}</span>
              </span>
              <kbd className="hidden shrink-0 rounded border bg-muted px-1 font-mono text-[10px] text-muted-foreground sm:inline">
                ⏎
              </kbd>
            </button>
          ) : null}
          <p className="text-xs text-muted-foreground">
            {treeStatus === "loading" && "Scanning directory…"}
            {treeStatus === "ready" && stats && !searchActive && (
              <>
                {fmtInt(stats.files)} files · {fmtInt(stats.totalLines)} lines ·{" "}
                {fmtBytes(stats.totalBytes)}
              </>
            )}
            {treeStatus === "ready" && searchActive && (
              <span className={matchCount === 0 ? "text-destructive" : undefined}>
                {matchCount} file{matchCount === 1 ? "" : "s"} match
                {matchCount === 1 ? "" : "es"} “{query.trim()}”
              </span>
            )}
            {treeStatus === "error" && "Could not load the tree"}
          </p>
        </CardHeader>
        <CardContent className="p-2">
          {!searchActive && !permalink && recentFiles.some((p) => p !== selectedFile) ? (
            <div className="mb-2 border-b pb-2">
              <div className="flex items-center justify-between px-1 pb-1.5">
                <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  <History className="size-3" aria-hidden />
                  Recently viewed
                </p>
                <button
                  type="button"
                  onClick={clearRecentFiles}
                  aria-label="Clear recently viewed files"
                  className="flex size-5 items-center justify-center rounded-md text-muted-foreground/70 transition-colors hover:bg-muted hover:text-foreground"
                >
                  <X className="size-3" aria-hidden />
                </button>
              </div>
              <div className="thin-scrollbar flex gap-1.5 overflow-x-auto pb-1" role="list" aria-label="Recently viewed files">
                {recentFiles.slice(0, RECENT_SHOWN).map((path, i) => {
                  const isActive = path === selectedFile;
                  const name = path.split("/").pop() ?? path;
                  return (
                    <motion.button
                      key={path}
                      type="button"
                      role="listitem"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: Math.min(i * 0.04, 0.16), duration: 0.18 }}
                      onClick={() => setSelectedFile(path)}
                      title={path}
                      aria-label={`Open ${path}`}
                      aria-current={isActive ? "true" : undefined}
                      className={
                        "shrink-0 rounded-md border px-2 py-1 font-mono text-[11px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring " +
                        (isActive
                          ? "border-primary/40 bg-primary/10 text-primary"
                          : "border-border/70 bg-muted/40 text-muted-foreground hover:border-primary/30 hover:bg-primary/5 hover:text-foreground")
                      }
                    >
                      {name}
                    </motion.button>
                  );
                })}
              </div>
            </div>
          ) : null}
          {treeStatus === "loading" ? (
            <div className="space-y-2 p-2" aria-label="Loading file tree">
              {Array.from({ length: 9 }).map((_, i) => (
                <Skeleton key={i} className="h-9" style={{ marginLeft: (i % 3) * 14 }} />
              ))}
            </div>
          ) : treeStatus === "error" ? (
            <div className="space-y-3 p-3">
              <div className="flex items-start gap-2 text-sm text-destructive">
                <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
                <span>{treeError ?? "Unknown error"}</span>
              </div>
              <Button variant="outline" size="sm" className="gap-2" onClick={() => void ensureTree()}>
                <RefreshCw className="size-3.5" aria-hidden />
                Retry
              </Button>
            </div>
          ) : searchActive && matchCount === 0 ? (
            <div className="flex flex-col items-center gap-2 px-3 py-10 text-center">
              <Search className="size-6 text-muted-foreground/50" aria-hidden />
              <p className="text-sm text-muted-foreground">
                No files match <span className="font-mono font-medium text-foreground">{query.trim()}</span>
              </p>
              <Button variant="outline" size="sm" className="gap-1.5" onClick={() => setQuery("")}>
                <X className="size-3.5" aria-hidden />
                Clear filter
              </Button>
            </div>
          ) : filteredTree ? (
            <FileTree
              node={filteredTree}
              selectedPath={selectedFile}
              onSelect={setSelectedFile}
              searchActive={searchActive}
            />
          ) : null}
        </CardContent>
      </Card>

      {/* ------------------------------ File view ------------------------------ */}
      <Card className="overflow-hidden py-0 shadow-sm">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <nav aria-label="File path" className="min-w-0">
              <ol className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
                <li className="font-mono">pharma-ml-toolkit</li>
                {selectedFile.split("/").map((segment, i, arr) => (
                  <li key={`${segment}-${i}`} className="flex items-center gap-1">
                    <span aria-hidden>/</span>
                    <span
                      className={
                        i === arr.length - 1
                          ? "font-mono font-medium text-foreground"
                          : "font-mono"
                      }
                    >
                      {segment}
                    </span>
                  </li>
                ))}
              </ol>
            </nav>
            <div className="flex flex-wrap items-center gap-1.5">
              {content ? (
                <>
                  <Badge variant="secondary" className="font-mono text-[10px] uppercase">
                    {content.language}
                  </Badge>
                  <Badge variant="outline" className="tabular-nums">
                    {fmtInt(content.lines)} lines
                  </Badge>
                  <Badge variant="outline" className="tabular-nums">
                    {fmtBytes(content.size)}
                  </Badge>
                </>
              ) : null}
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-3 sm:p-4">
          {fileStatus === "loading" ? (
            <div className="space-y-2.5" aria-label="Loading file content">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-4 w-5/6" />
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-4 w-4/5" />
              <Skeleton className="h-4 w-3/5" />
            </div>
          ) : fileStatus === "error" ? (
            <div className="space-y-3 py-6">
              <div className="flex items-start gap-2 text-sm text-destructive">
                <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
                <span>{fileError ?? "Unknown error"}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
                onClick={() => setSelectedFile("README.md")}
              >
                <RefreshCw className="size-3.5" aria-hidden />
                Back to README.md
              </Button>
            </div>
          ) : content ? (
            <CodeBlock
              code={content.content}
              language={content.language}
              fileName={content.path.split("/").pop() ?? content.path}
              showLineNumbers
              maxHeightClass="max-h-[70vh]"
              highlightLine={targetLine}
              onClearHighlight={() => setTargetLine(null)}
              permalinkPath={content.path}
            />
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
