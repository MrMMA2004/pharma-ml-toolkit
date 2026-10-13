"use client";

import { useState } from "react";
import { ChevronRight, FileCode2, FileText, Folder, FolderOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TreeNode } from "@/lib/toolkit-types";

interface FileTreeProps {
  node: TreeNode;
  selectedPath: string;
  onSelect: (path: string) => void;
}

const CODE_EXTENSIONS = new Set([".py", ".toml", ".yml", ".yaml", ".cfg", ".ini"]);

function fileIconClass(node: TreeNode): string {
  if (node.language === "python") return "text-primary";
  if (node.language && node.language !== "text") return "text-amber-600 dark:text-amber-400";
  return "text-muted-foreground";
}

function FileIcon({ node }: { node: TreeNode }) {
  const ext = node.name.slice(node.name.lastIndexOf("."));
  if (CODE_EXTENSIONS.has(ext) || node.name === "Dockerfile") {
    return <FileCode2 className={cn("size-4 shrink-0", fileIconClass(node))} aria-hidden />;
  }
  return <FileText className="size-4 shrink-0 text-muted-foreground" aria-hidden />;
}

function TreeNodeRow({
  node,
  depth,
  selectedPath,
  onSelect,
  expanded,
  toggleDir,
}: {
  node: TreeNode;
  depth: number;
  selectedPath: string;
  onSelect: (path: string) => void;
  expanded: Set<string>;
  toggleDir: (path: string) => void;
}) {
  const paddingLeft = 8 + depth * 16;

  if (node.type === "dir") {
    const isOpen = expanded.has(node.path);
    const childCount = node.children?.length ?? 0;
    return (
      <li className="list-none" role="none">
        <button
          type="button"
          role="treeitem"
          aria-expanded={isOpen}
          aria-selected={false}
          aria-label={`${node.name} directory, ${childCount} entries`}
          onClick={() => toggleDir(node.path)}
          className="flex min-h-11 w-full items-center gap-1.5 rounded-md pr-2 text-left text-sm text-foreground/90 transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          style={{ paddingLeft }}
        >
          <ChevronRight
            className={cn(
              "size-3.5 shrink-0 text-muted-foreground transition-transform",
              isOpen && "rotate-90"
            )}
            aria-hidden
          />
          {isOpen ? (
            <FolderOpen className="size-4 shrink-0 text-amber-500" aria-hidden />
          ) : (
            <Folder className="size-4 shrink-0 text-amber-500/80" aria-hidden />
          )}
          <span className="truncate font-medium">{node.name}</span>
          <span className="ml-auto shrink-0 text-[10px] tabular-nums text-muted-foreground">
            {childCount}
          </span>
        </button>
        {isOpen && childCount > 0 ? (
          <ul role="group" className="space-y-0.5">
            {(node.children ?? []).map((child) => (
              <TreeNodeRow
                key={child.path}
                node={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelect={onSelect}
                expanded={expanded}
                toggleDir={toggleDir}
              />
            ))}
          </ul>
        ) : null}
      </li>
    );
  }

  const selected = selectedPath === node.path;
  return (
    <li className="list-none" role="none">
      <button
        type="button"
        role="treeitem"
        aria-selected={selected}
        aria-label={
          node.lines !== undefined
            ? `${node.name}, ${node.lines} lines`
            : node.name
        }
        onClick={() => onSelect(node.path)}
        className={cn(
          "flex min-h-11 w-full items-center gap-1.5 rounded-md pr-2 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          selected
            ? "bg-primary/10 font-medium text-primary"
            : "text-foreground/80 hover:bg-muted"
        )}
        style={{ paddingLeft: paddingLeft + 18 }}
      >
        <FileIcon node={node} />
        <span className="truncate font-mono text-[13px]">{node.name}</span>
        {node.lines !== undefined ? (
          <span
            className={cn(
              "ml-auto shrink-0 text-[10px] tabular-nums",
              selected ? "text-primary/70" : "text-muted-foreground/70"
            )}
          >
            {node.lines}L
          </span>
        ) : null}
      </button>
    </li>
  );
}

/** Collapsible file tree with line counts and emerald selection highlight. */
export function FileTree({ node, selectedPath, onSelect }: FileTreeProps) {
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set(["", "src", "src/pharmaml"])
  );

  function toggleDir(path: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  return (
    <nav aria-label="Toolkit file tree" className="thin-scrollbar max-h-[70vh] overflow-y-auto pr-1">
      <ul role="tree" className="space-y-0.5">
        {(node.children ?? []).map((child) => (
          <TreeNodeRow
            key={child.path}
            node={child}
            depth={0}
            selectedPath={selectedPath}
            onSelect={onSelect}
            expanded={expanded}
            toggleDir={toggleDir}
          />
        ))}
      </ul>
    </nav>
  );
}
