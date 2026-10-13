"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileCode2, RefreshCw, Search, TriangleAlert, X } from "lucide-react";
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

export function CodeExplorer() {
  const ensureTree = useToolkitStore((s) => s.ensureTree);
  const tree = useToolkitStore((s) => s.tree);
  const stats = useToolkitStore((s) => s.stats);
  const treeStatus = useToolkitStore((s) => s.treeStatus);
  const treeError = useToolkitStore((s) => s.treeError);

  const selectedFile = useToolkitStore((s) => s.selectedFile);
  const setSelectedFile = useToolkitStore((s) => s.setSelectedFile);

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
  const searchActive = trimmed.length > 0;
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
              placeholder="Filter files by name…"
              aria-label="Filter files in the toolkit tree"
              className="h-9 pl-8 pr-8 text-sm"
            />
            {searchActive ? (
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
            />
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
