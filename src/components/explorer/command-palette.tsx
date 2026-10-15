"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useTheme } from "next-themes";
import {
  Activity,
  Beaker,
  BookMarked,
  BookOpen,
  Boxes,
  Braces,
  Brain,
  Calculator,
  Download,
  Eraser,
  FileCode2,
  GitCompare,
  Hash,
  LayoutDashboard,
  Loader2,
  Microscope,
  Moon,
  NotebookText,
  ScanSearch,
  Search,
  SquareFunction,
  Sun,
  Terminal,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { copyToClipboard, fuzzyScore } from "@/lib/format";
import { getReference, peekReference } from "@/lib/reference-cache";
import { TABS, useToolkitStore, type TabId } from "@/lib/store";
import { HighlightText } from "@/components/explorer/highlight-text";
import { toast } from "sonner";
import type { ReferenceResponse, TreeNode } from "@/lib/toolkit-types";

/* ------------------------------------------------------------------ */
/* Palette items                                                       */
/* ------------------------------------------------------------------ */

type PaletteGroup = "navigate" | "recent" | "files" | "symbols" | "actions";

const GROUP_TAG: Record<PaletteGroup, string> = {
  navigate: "tab",
  recent: "recent",
  files: "file",
  symbols: "symbol",
  actions: "action",
};

interface PaletteItem {
  id: string;
  group: PaletteGroup;
  label: string;
  /** Secondary fuzzy-match target (dir path, module, summary, keywords). */
  sublabel?: string;
  icon: LucideIcon;
  run: () => void;
}

/** Flat file list from the toolkit tree. */
function flattenFiles(node: TreeNode, out: TreeNode[] = []): TreeNode[] {
  for (const child of node.children ?? []) {
    if (child.type === "file") out.push(child);
    else flattenFiles(child, out);
  }
  return out;
}

const TAB_ICONS: Record<TabId, LucideIcon> = {
  overview: LayoutDashboard,
  explorer: BookOpen,
  reference: BookMarked,
  playground: Beaker,
  download: Download,
};

const SUB_TABS: Array<{ value: string; label: string; icon: LucideIcon }> = [
  { value: "calculators", label: "Calculators", icon: Calculator },
  { value: "pk", label: "PK / PD Explorer", icon: Activity },
  { value: "ml", label: "ML Training", icon: Brain },
  { value: "biology", label: "Cell Line Bio", icon: Microscope },
  { value: "docking", label: "Docking Parser", icon: ScanSearch },
  { value: "reconciler", label: "Ligand Sets", icon: GitCompare },
];

const SYMBOL_ICONS: Record<string, LucideIcon> = {
  function: SquareFunction,
  class: Braces,
  module: Boxes,
  constant: Hash,
};

/** Maximum rows rendered in query mode (footer reports the full match count). */
const MAX_QUERY_ITEMS = 12;
/** Maximum rows per group in the empty-query view. */
const MAX_RECENT_ITEMS = 6;

/* ------------------------------------------------------------------ */
/* Trigger button                                                      */
/* ------------------------------------------------------------------ */

/** Header button that opens the command palette (⌘K / Ctrl-K). */
export function CommandPaletteTrigger({
  compact = false,
  className,
}: {
  compact?: boolean;
  className?: string;
}) {
  const setPaletteOpen = useToolkitStore((s) => s.setPaletteOpen);
  if (compact) {
    return (
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={cn("size-11 shrink-0 rounded-full", className)}
        onClick={() => setPaletteOpen(true)}
        aria-label="Open command palette (Control K)"
        title="Command palette — Ctrl/⌘ K"
      >
        <Search className="size-5" aria-hidden />
      </Button>
    );
  }
  return (
    <Button
      type="button"
      variant="outline"
      className={cn(
        "h-11 shrink-0 gap-2 rounded-full px-4 text-sm text-muted-foreground hover:text-foreground",
        className
      )}
      onClick={() => setPaletteOpen(true)}
      aria-label="Open command palette (Control K)"
      title="Command palette — Ctrl/⌘ K"
    >
      <Search className="size-4" aria-hidden />
      <span className="hidden lg:inline">Search…</span>
      <kbd className="hidden rounded-md border bg-muted px-1.5 font-mono text-[10px] text-muted-foreground md:inline-flex">
        Ctrl ⌘ K
      </kbd>
    </Button>
  );
}

/* ------------------------------------------------------------------ */
/* Palette (global hotkey + dialog lifecycle)                          */
/* ------------------------------------------------------------------ */

/**
 * Global command palette (Ctrl/Cmd+K): fuzzy search across navigation tabs,
 * playground sub-tabs, recently opened files, the whole toolkit file tree,
 * public API symbols, and quick actions. Rendered once at the page level;
 * the dialog itself is a child that mounts fresh on every open (so the
 * query and active row always start clean).
 */
export function CommandPalette() {
  const open = useToolkitStore((s) => s.paletteOpen);
  const setPaletteOpen = useToolkitStore((s) => s.setPaletteOpen);
  const close = useCallback(() => setPaletteOpen(false), [setPaletteOpen]);

  /* Global hotkey: Ctrl/Cmd+K toggles the palette. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen(!useToolkitStore.getState().paletteOpen);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPaletteOpen]);

  return (
    <AnimatePresence>
      {open ? <PaletteDialog key="palette" close={close} /> : null}
    </AnimatePresence>
  );
}

/* ------------------------------------------------------------------ */
/* Dialog body (own state, mounts fresh per open via the store flag)   */
/* ------------------------------------------------------------------ */

/**
 * The palette dialog body. Mounts only while the palette is open (and stays
 * mounted through the exit animation), so all local state (query, active
 * row, loaded indexes) starts clean on every open.
 */
function PaletteDialog({ close }: { close: () => void }) {
  const setTab = useToolkitStore((s) => s.setTab);
  const setSelectedFile = useToolkitStore((s) => s.setSelectedFile);
  const openFileAtLine = useToolkitStore((s) => s.openFileAtLine);
  const setPlaygroundTab = useToolkitStore((s) => s.setPlaygroundTab);
  const clearRecentFiles = useToolkitStore((s) => s.clearRecentFiles);
  const recentFiles = useToolkitStore((s) => s.recentFiles);
  const tree = useToolkitStore((s) => s.tree);
  const treeStatus = useToolkitStore((s) => s.treeStatus);
  const ensureTree = useToolkitStore((s) => s.ensureTree);

  const { resolvedTheme, setTheme } = useTheme();

  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [reference, setReference] = useState<ReferenceResponse | null>(() => peekReference());
  const [referenceError, setReferenceError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const trimmed = query.trim();

  /* Warm both search indexes as soon as the dialog appears. */
  useEffect(() => {
    void ensureTree();
    if (peekReference() || referenceError) return;
    let stale = false;
    getReference()
      .then((data) => {
        if (!stale) setReference(data);
      })
      .catch((err: unknown) => {
        if (!stale) {
          setReferenceError(err instanceof Error ? err.message : "Failed to load symbol index.");
        }
      });
    return () => {
      stale = true;
    };
  }, [ensureTree, referenceError]);

  /* Lock body scroll while the dialog is open (restored on unmount). */
  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  /* Focus the input when the dialog appears. */
  useEffect(() => {
    const t = window.setTimeout(() => inputRef.current?.focus(), 30);
    return () => window.clearTimeout(t);
  }, []);

  /* ---------------- Item assembly (stable, re-run only on data change) --------------- */

  const runAndClose = useCallback(
    (run: () => void) => {
      close();
      // Run after the close state settles so focus returns to the page first.
      requestAnimationFrame(() => run());
    },
    [close]
  );

  const items = useMemo<PaletteItem[]>(() => {
    const list: PaletteItem[] = [];

    for (const t of TABS) {
      const Icon = TAB_ICONS[t.id];
      list.push({
        id: `tab:${t.id}`,
        group: "navigate",
        label: t.label,
        sublabel: t.id === "playground" ? "ml · pharma · biology · docking" : undefined,
        icon: Icon,
        run: () => setTab(t.id),
      });
    }
    for (const sub of SUB_TABS) {
      list.push({
        id: `sub:${sub.value}`,
        group: "navigate",
        label: sub.label,
        sublabel: "Playground sub-tab",
        icon: sub.icon,
        run: () => {
          setPlaygroundTab(sub.value);
          setTab("playground");
        },
      });
    }

    for (const path of recentFiles.slice(0, MAX_RECENT_ITEMS)) {
      list.push({
        id: `recent:${path}`,
        group: "recent",
        label: path.split("/").pop() ?? path,
        sublabel: path,
        icon: FileCode2,
        run: () => {
          setSelectedFile(path);
          setTab("explorer");
        },
      });
    }

    if (tree) {
      for (const file of flattenFiles(tree)) {
        const dir = file.path.includes("/")
          ? file.path.slice(0, file.path.lastIndexOf("/"))
          : "";
        list.push({
          id: `file:${file.path}`,
          group: "files",
          label: file.name,
          sublabel: dir || "root",
          icon: FileCode2,
          run: () => {
            setSelectedFile(file.path);
            setTab("explorer");
          },
        });
      }
    }

    if (reference) {
      for (const mod of reference.modules) {
        for (const sym of mod.symbols) {
          const source = sym.sourcePath ?? mod.sourcePath;
          const Icon = SYMBOL_ICONS[sym.kind] ?? SquareFunction;
          const summary = sym.summary.length > 72 ? `${sym.summary.slice(0, 72)}…` : sym.summary;
          list.push({
            id: `sym:${mod.name}.${sym.name}`,
            group: "symbols",
            label: sym.name,
            sublabel: `${mod.name}${summary ? ` · ${summary}` : ""}`,
            icon: Icon,
            run: () => {
              openFileAtLine(source, sym.line ?? null);
              setTab("explorer");
            },
          });
        }
      }
    }

    list.push(
      {
        id: "action:download",
        group: "actions",
        label: "Download the toolkit ZIP",
        sublabel: "api/toolkit/download · full source + tests + examples",
        icon: Download,
        run: () => {
          window.location.assign("/api/toolkit/download");
        },
      },
      {
        id: "action:install",
        group: "actions",
        label: "Copy install command",
        sublabel: "pip install -e .",
        icon: Terminal,
        run: () => {
          void copyToClipboard("pip install -e .").then((ok) => {
            if (ok) toast.success("Copied install command", { description: "pip install -e ." });
            else toast.error("Could not copy — clipboard blocked");
          });
        },
      },
      {
        id: "action:notebook",
        group: "actions",
        label: "Open the walkthrough notebook",
        sublabel: "examples/pharmaml_walkthrough.ipynb",
        icon: NotebookText,
        run: () => {
          openFileAtLine("examples/pharmaml_walkthrough.ipynb", null);
          setTab("explorer");
        },
      },
      {
        id: "action:theme",
        group: "actions",
        label: "Toggle light / dark theme",
        sublabel: "switch the color scheme",
        icon: resolvedTheme === "dark" ? Sun : Moon,
        run: () => setTheme(resolvedTheme === "dark" ? "light" : "dark"),
      },
      {
        id: "action:clear-recent",
        group: "actions",
        label: "Clear recent files",
        sublabel: `${recentFiles.length} entr${recentFiles.length === 1 ? "y" : "ies"}`,
        icon: Eraser,
        run: () => {
          clearRecentFiles();
          toast.success("Recent files cleared");
        },
      }
    );

    return list;
  }, [
    tree,
    reference,
    recentFiles,
    resolvedTheme,
    setTab,
    setPlaygroundTab,
    setSelectedFile,
    openFileAtLine,
    setTheme,
    clearRecentFiles,
  ]);

  /* ---------------- Scoring / filtering (query mode) ---------------- */

  const filtered = useMemo(() => {
    if (trimmed.length === 0) return null;
    const scored: Array<{ item: PaletteItem; score: number }> = [];
    for (const item of items) {
      const labelScore = fuzzyScore(trimmed, item.label);
      const subScore = item.sublabel ? fuzzyScore(trimmed, item.sublabel) : null;
      if (labelScore === null && subScore === null) continue;
      const score = Math.max(labelScore ?? 0, (subScore ?? 0) * 0.75);
      scored.push({ item, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored;
  }, [items, trimmed]);

  const isQueryMode = filtered !== null;

  /* ---------------- Rows to render ---------------- */

  const rows = useMemo(() => {
    if (filtered) {
      return filtered.slice(0, MAX_QUERY_ITEMS).map(({ item }) => item);
    }
    // Empty query: Navigate → Recent → Actions (files/symbols need a query).
    const groups: Array<{ label: string; items: PaletteItem[] }> = [];
    const navigate = items.filter((i) => i.group === "navigate");
    const recent = items.filter((i) => i.group === "recent").slice(0, MAX_RECENT_ITEMS);
    const actions = items.filter((i) => i.group === "actions");
    groups.push({ label: "Navigate", items: navigate });
    if (recent.length > 0) groups.push({ label: "Recent files", items: recent });
    groups.push({ label: "Actions", items: actions });
    return groups;
  }, [filtered, items]);

  const flatRows = useMemo(
    () =>
      isQueryMode
        ? (rows as PaletteItem[])
        : (rows as Array<{ items: PaletteItem[] }>).flatMap((g) => g.items),
    [rows, isQueryMode]
  );
  const matchCount = filtered ? filtered.length : flatRows.length;

  /* Clamp the active row during render (rows can shrink after filtering). */
  const maxIndex = Math.max(flatRows.length - 1, 0);
  const active = Math.min(activeIndex, maxIndex);

  /* Keep the active row visible in the scrollable list. */
  useEffect(() => {
    if (flatRows.length === 0) return;
    listRef.current
      ?.querySelector(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [active, flatRows.length]);

  const runActive = useCallback(() => {
    const item = flatRows[active];
    if (item) runAndClose(item.run);
  }, [active, flatRows, runAndClose]);

  const onInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex(flatRows.length === 0 ? 0 : (active + 1) % flatRows.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex(flatRows.length === 0 ? 0 : (active - 1 + flatRows.length) % flatRows.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      runActive();
    }
  };

  /* Simple focus trap: Tab stays on the input (rows are reachable via arrows). */
  const onDialogKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      inputRef.current?.focus();
    }
  };

  const renderRow = (item: PaletteItem, flatIndex: number) => {
    const Icon = item.icon;
    const isActive = flatIndex === active;
    return (      <button
        key={item.id}
        type="button"
        role="option"
        aria-selected={isActive}
        data-index={flatIndex}
        id={`palette-opt-${flatIndex}`}
        onMouseMove={() => {
          if (active !== flatIndex) setActiveIndex(flatIndex);
        }}
        onClick={() => runAndClose(item.run)}
        className={cn(
          "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          isActive ? "bg-primary/10 text-primary" : "text-foreground hover:bg-muted/60",
          (item.group === "symbols" || item.group === "files") && "font-mono text-[13px]"
        )}
      >
        <Icon
          className={cn("size-4 shrink-0", isActive ? "text-primary" : "text-muted-foreground")}
          aria-hidden
        />
        <span className="min-w-0 flex-1 truncate">
          <HighlightText text={item.label} query={query} className="font-medium" />
          {item.sublabel ? (
            <span className="ml-2 hidden text-xs text-muted-foreground sm:inline">
              <HighlightText
                text={item.sublabel}
                query={query}
                className="font-normal text-muted-foreground"
              />
            </span>
          ) : null}
        </span>
        {isQueryMode ? (
          <span className="shrink-0 rounded-md border bg-muted/50 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
            {GROUP_TAG[item.group]}
          </span>
        ) : null}
      </button>
    );
  };

  let flatCounter = -1;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[10vh] sm:pt-[14vh]">
      {/* Overlay */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.14 }}
        className="absolute inset-0 bg-background/70 backdrop-blur-[3px]"
        onClick={close}
        aria-hidden
      />
      {/* Dialog */}
      <motion.div
        initial={{ opacity: 0, y: -10, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -6, scale: 0.98 }}
        transition={{ duration: 0.18, ease: "easeOut" }}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onKeyDown={onDialogKeyDown}
        className="relative w-full max-w-xl overflow-hidden rounded-xl border bg-card shadow-2xl ring-1 ring-primary/10"
      >
        {/* Gradient hairline */}
        <div
          className="h-px w-full bg-gradient-to-r from-transparent via-primary/50 to-transparent"
          aria-hidden
        />
        {/* Input row */}
        <div className="flex h-12 items-center gap-2.5 border-b px-3.5">
          <Search className="size-4 shrink-0 text-primary" aria-hidden />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            onKeyDown={onInputKeyDown}
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-listbox"
            aria-activedescendant={flatRows.length > 0 ? `palette-opt-${active}` : undefined}
            aria-label="Search files, symbols and commands"
            placeholder="Search files, symbols, tabs…"
            spellCheck={false}
            autoComplete="off"
            className="h-full min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <kbd className="hidden shrink-0 rounded-md border bg-muted px-1.5 font-mono text-[10px] text-muted-foreground sm:inline-flex">
            esc
          </kbd>
        </div>
        {/* Rows */}
        <div
          ref={listRef}
          id="palette-listbox"
          role="listbox"
          aria-label="Search results"
          className="thin-scrollbar max-h-[min(52vh,400px)] overflow-y-auto p-1.5"
        >
          {flatRows.length === 0 && trimmed.length === 0 ? (
            <p className="px-3 py-6 text-center text-sm text-muted-foreground">
              Nothing to show yet.
            </p>
          ) : flatRows.length === 0 && trimmed.length > 0 ? (
            <div className="space-y-2 px-3 py-6 text-center">
              <p className="text-sm text-muted-foreground">No matches for “{trimmed}”.</p>
              <p className="text-xs text-muted-foreground/70">
                Try a shorter query — files, symbols, tabs and actions are all searchable.
              </p>
            </div>
          ) : isQueryMode ? (
            (rows as PaletteItem[]).map((item) => {
              flatCounter += 1;
              return renderRow(item, flatCounter);
            })
          ) : (
            (rows as Array<{ label: string; items: PaletteItem[] }>).map((group) => (
              <div key={group.label} className="mb-1">
                <p className="px-2.5 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {group.label}
                </p>
                {group.items.map((item) => {
                  flatCounter += 1;
                  return renderRow(item, flatCounter);
                })}
              </div>
            ))
          )}
          {treeStatus === "loading" ? (
            <p className="flex items-center gap-2 px-2.5 py-2 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" aria-hidden />
              Loading file index…
            </p>
          ) : null}
          {referenceError ? (
            <p className="flex items-center gap-2 px-2.5 py-2 text-xs text-muted-foreground">
              <TriangleAlert className="size-3.5 shrink-0 text-amber-500" aria-hidden />
              Symbol index unavailable — file search still works.
            </p>
          ) : null}
        </div>
        {/* Footer */}
        <div className="flex items-center justify-between gap-3 border-t bg-muted/30 px-3.5 py-1.5 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <kbd className="rounded border bg-muted px-1 font-mono text-[10px]">↑↓</kbd>
              navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="rounded border bg-muted px-1 font-mono text-[10px]">⏎</kbd>
              select
            </span>
          </span>
          <span className="tabular-nums">
            {trimmed.length > 0
              ? filtered && filtered.length > MAX_QUERY_ITEMS
                ? `${MAX_QUERY_ITEMS} of ${filtered.length} matches`
                : `${matchCount} match${matchCount === 1 ? "" : "es"}`
              : `${matchCount} commands`}
          </span>
        </div>
      </motion.div>
    </div>
  );
}

