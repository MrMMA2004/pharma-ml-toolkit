"use client";

import { useState } from "react";
import {
  Atom,
  ChevronRight,
  FileCode2,
  FileText,
  Folder,
  FolderOpen,
  Table2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { isTreeDataNode } from "@/lib/data-file";
import { fmtBytes } from "@/lib/format";
import type { TreeNode } from "@/lib/toolkit-types";

interface FileTreeProps {
  node: TreeNode;
  selectedPath: string;
  onSelect: (path: string) => void;
  /** When true (active search), every directory renders expanded. */
  searchActive?: boolean;
}

const CODE_EXTENSIONS = new Set([".py", ".toml", ".yml", ".yaml", ".cfg", ".ini"]);

/** Data extensions rendered with the Table2 (tabular) tree icon. */
const TABLE2_DATA_EXTENSIONS = new Set([".csv", ".gpf", ".dpf"]);
/** Data extensions rendered with the Atom (molecular) tree icon. */
const ATOM_DATA_EXTENSIONS = new Set([".pdb", ".dlg", ".glg"]);

/** Human labels for data-node aria descriptions. */
const DATA_KIND_LABELS: Record<string, string> = {
  ".csv": "CSV dataset",
  ".pdb": "PDB structure",
  ".dlg": "AutoDock log",
  ".glg": "AutoGrid log",
  ".gpf": "AutoGrid parameter file",
  ".dpf": "AutoDock parameter file",
};

function fileIconClass(node: TreeNode): string {
  if (node.language === "python") return "text-primary";
  if (node.language && node.language !== "text") return "text-amber-600 dark:text-amber-400";
  return "text-muted-foreground";
}

function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

/** Data nodes (.csv/.pdb/.dlg/.glg/.gpf/.dpf) preview in the explorer —
 *  they get data icons in the primary accent instead of code glyphs. */
function DataFileIcon({ node }: { node: TreeNode }) {
  const ext = fileExtension(node.name);
  if (ATOM_DATA_EXTENSIONS.has(ext)) {
    return <Atom className="size-4 shrink-0 text-primary" aria-hidden />;
  }
  return <Table2 className="size-4 shrink-0 text-primary" aria-hidden />;
}

function FileIcon({ node }: { node: TreeNode }) {
  if (isTreeDataNode(node.path)) return <DataFileIcon node={node} />;
  const ext = fileExtension(node.name);
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
  searchActive,
}: {
  node: TreeNode;
  depth: number;
  selectedPath: string;
  onSelect: (path: string) => void;
  expanded: Set<string>;
  toggleDir: (path: string) => void;
  searchActive?: boolean;
}) {
  const paddingLeft = 8 + depth * 16;

  if (node.type === "dir") {
    const isOpen = searchActive || expanded.has(node.path);
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
                searchActive={searchActive}
              />
            ))}
          </ul>
        ) : null}
      </li>
    );
  }

  const selected = selectedPath === node.path;
  const isData = isTreeDataNode(node.path);
  const dataLabel = isData
    ? (DATA_KIND_LABELS[fileExtension(node.name)] ?? "data file")
    : null;
  return (
    <li className="list-none" role="none">
      <button
        type="button"
        role="treeitem"
        aria-selected={selected}
        aria-label={
          isData
            ? `${node.name}, ${dataLabel}, ${fmtBytes(node.size)}, click to preview`
            : node.lines !== undefined
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
        {isData ? (
          <span
            className={cn(
              "ml-auto shrink-0 text-[10px] tabular-nums",
              selected ? "text-primary/70" : "text-muted-foreground/70"
            )}
          >
            {fmtBytes(node.size)}
          </span>
        ) : node.lines !== undefined ? (
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
export function FileTree({ node, selectedPath, onSelect, searchActive }: FileTreeProps) {
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
            searchActive={searchActive}
          />
        ))}
      </ul>
    </nav>
  );
}
