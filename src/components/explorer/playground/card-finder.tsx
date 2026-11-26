"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Eraser, History, Search, SquareArrowOutUpRight } from "lucide-react";
import { toast } from "sonner";
import { HighlightText } from "@/components/explorer/highlight-text";
import { cn } from "@/lib/utils";
import { fuzzyScore } from "@/lib/format";
import { useToolkitStore } from "@/lib/store";
import {
  PLAYGROUND_CARDS,
  type PlaygroundCardEntry,
} from "@/components/explorer/playground/card-directory";

/* ------------------------------------------------------------------ */
/* Card finder — Next.js Playground navigation                         */
/* ------------------------------------------------------------------ */
/**
 * Filters this web Playground's cards by title, blurb, or keywords and
 * navigates to a match across the top tab, Playground sub-tab, and—when
 * needed—the PK / PD inner tab. It then scrolls to the live card and
 * flashes it. The browser owns Ctrl+F, so this dialog uses
 * Ctrl/Cmd+Shift+F. Catalog drift surfaces as an explicit "could not
 * find" message instead of a silent jump to the wrong card.
 */

const HOTKEY_LABEL = "Ctrl+Shift+F";
const JUMP_TIMEOUT_MS = 2_500;

/** Recency row's localStorage key + cap: record cards reached through
 * this web-console finder, newest first, de-duplicated, and capped. A
 * failed resolve is never remembered. */
const RECENT_STORAGE_KEY = "pharmaml.cardFinder.recent";
const RECENT_CAP = 6;

function loadRecentIds(): string[] {
  try {
    const raw = window.localStorage.getItem(RECENT_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((id): id is string => typeof id === "string").slice(0, RECENT_CAP);
  } catch {
    return [];
  }
}

function saveRecentIds(ids: string[]): void {
  try {
    window.localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(ids.slice(0, RECENT_CAP)));
  } catch {
    /* a blocked localStorage (private mode) just loses the memory */
  }
}

function rememberJump(id: string): void {
  const next = [id, ...loadRecentIds().filter((x) => x !== id)];
  saveRecentIds(next);
}

/** (100x-131) Empty the recency memory — the row's own "Clear" chip.
 * The next open reads an empty list, so the row honestly disappears
 * (a vanished memory renders no row at all, the same honest rule). */
function clearRecentIds(): void {
  try {
    window.localStorage.removeItem(RECENT_STORAGE_KEY);
  } catch {
    /* a blocked localStorage has nothing to clear */
  }
}

/** Sub-tab strip labels (the jump-readiness gate matches them on the
 * live Radix triggers). */
const SUB_TAB_LABELS: Record<string, string> = {
  calculators: "Calculators",
  pk: "PK / PD",
  ml: "ML Training",
  biology: "Cell Line Bio",
  docking: "Docking Parser",
  features: "Features",
  ligandprep: "Ligand Prep",
  reconciler: "Ligand Sets",
  generative: "Surgery",
  md: "Molecular Dynamics",
};

/** PK / PD Explorer inner-tab labels (the NCA trigger shortens on ≤sm). */
const INNER_TAB_LABELS: Record<string, string> = {
  regimen: "Regimen PK",
  pd: "PD effect",
  nca: "NCA",
};

/** True once the DOM shows the tabs the entry lives on — the guard
 * against the exit-animation race: the page-level AnimatePresence is
 * mode="wait", so while the PREVIOUS top tab is still animating out its
 * cards are matchable, and a naive resolve can hit a card from the tab
 * you are LEAVING (an Overview blurb contains "Torsion QA"). The gate
 * requires the target sub-tab's trigger to be active AND an active
 * Radix panel to exist — which only happens once the Playground content
 * has actually mounted (the old content is gone by then). */
function domReadyFor(entry: PlaygroundCardEntry): boolean {
  const nav = document.querySelector('header button[aria-current="page"]');
  if (!nav?.textContent?.includes("Playground")) return false;
  if (document.querySelector('[role="tabpanel"][data-state="active"]') === null) {
    return false;
  }
  if (entry.tab === null) return true; // the strip rides above the panels
  const active = Array.from(document.querySelectorAll<HTMLElement>('[role="tab"]')).filter(
    (t) => t.getAttribute("data-state") === "active"
  );
  if (!active.some((t) => t.textContent?.includes(SUB_TAB_LABELS[entry.tab!] ?? "§"))) {
    return false;
  }
  if (entry.inner !== undefined) {
    if (!active.some((t) => t.textContent?.includes(INNER_TAB_LABELS[entry.inner!] ?? "§"))) {
      return false;
    }
  }
  return true;
}

/** Resolve one catalog entry in the live DOM, or null.
 *
 * Two precision passes over the card pool: a card TITLE containing the
 * match string beats a body-text mention (headings are the anchor), and
 * the pool is scoped — cards inside an active Radix panel for sub-tab
 * entries, cards OUTSIDE any panel (the session-report strip) for the
 * strip entry — so a match can never cross tabs.
 *
 * Exposed for the E2E probe: the probe walks EVERY catalog entry through
 * this resolver so a rename drift fails the round it happens in. */
export function resolveCardElement(entry: PlaygroundCardEntry): HTMLElement | null {
  const all = Array.from(document.querySelectorAll<HTMLElement>('[data-slot="card"]'));
  const inPanel = (el: HTMLElement) => el.closest('[role="tabpanel"]') !== null;
  const pool = entry.tab === null ? all.filter((c) => !inPanel(c)) : all.filter((c) => inPanel(c));
  for (const card of pool) {
    const title = card.querySelector("[data-slot=card-title]")?.textContent ?? "";
    if (title.includes(entry.match)) return card;
  }
  for (const card of pool) {
    if (card.textContent?.includes(entry.match)) return card;
  }
  return null;
}

/** Switch the tabs an entry needs, wait for its card to mount, then
 * scroll + flash. The wait is a polling loop gated on domReadyFor (the
 * exit-animation race guard), capped at JUMP_TIMEOUT_MS. */
async function jumpToCard(entry: PlaygroundCardEntry): Promise<boolean> {
  const store = useToolkitStore.getState();
  if (store.tab !== "playground") store.setTab("playground");
  if (entry.tab !== null) store.setPlaygroundTab(entry.tab);
  if (entry.inner !== undefined) store.setPkInnerTab(entry.inner);
  const deadline = Date.now() + JUMP_TIMEOUT_MS;
  do {
    await new Promise((r) => window.setTimeout(r, 60));
    if (!domReadyFor(entry)) continue;
    const card = resolveCardElement(entry);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.add("pg-card-flash");
      window.setTimeout(() => card.classList.remove("pg-card-flash"), 1_800);
      return true;
    }
  } while (Date.now() < deadline);
  return false;
}

/** The visible trigger (rides beside the Playground sub-tab strip). */
export function CardFinderTrigger() {
  const open = useToolkitStore((s) => s.cardFinderOpen);
  const setOpen = useToolkitStore((s) => s.setCardFinderOpen);

  return (
    <button
      type="button"
      onClick={() => setOpen(!open)}
      aria-haspopup="dialog"
      aria-expanded={open}
      title={`Find a card (${HOTKEY_LABEL})`}
      className="flex h-11 shrink-0 items-center gap-1.5 rounded-full border bg-background px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:text-sm"
    >
      <Search className="size-4 shrink-0" aria-hidden />
      <span className="hidden sm:inline">Find a card</span>
      <kbd className="hidden shrink-0 rounded-md border bg-muted px-1.5 font-mono text-[10px] text-muted-foreground lg:inline-flex">
        {HOTKEY_LABEL}
      </kbd>
    </button>
  );
}

interface ScoredEntry {
  entry: PlaygroundCardEntry;
  score: number;
}

/** The dialog itself — mounted once at the page level (like the command
 * palette) so the hotkey works from ANY top tab: opening it from
 * elsewhere first switches to the Playground. */
export function CardFinder() {
  const open = useToolkitStore((s) => s.cardFinderOpen);
  const setOpen = useToolkitStore((s) => s.setCardFinderOpen);

  /* The hotkey: Ctrl/Cmd+Shift+F opens (or closes) from anywhere. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setOpen(!useToolkitStore.getState().cardFinderOpen);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setOpen]);

  /* The dialog body mounts FRESH on every open (AnimatePresence swaps it
   * in), so its local query/selection state resets by remounting — no
   * reset-in-effect cascade. */
  return (
    <AnimatePresence>
      {open ? <CardFinderDialog onClose={() => setOpen(false)} /> : null}
    </AnimatePresence>
  );
}

function CardFinderDialog({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  /* Focus the input on mount (an external-DOM sync, not state). */
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const scored = useMemo<ScoredEntry[]>(() => {
    const trimmed = query.trim();
    if (trimmed.length === 0) {
      return PLAYGROUND_CARDS.map((entry) => ({ entry, score: 0 }));
    }
    const out: ScoredEntry[] = [];
    for (const entry of PLAYGROUND_CARDS) {
      const targets = [entry.title, entry.blurb, entry.keywords ?? ""];
      let best: number | null = null;
      for (const target of targets) {
        const s = fuzzyScore(trimmed, target);
        if (s !== null && (best === null || s > best)) best = s;
      }
      if (best !== null) out.push({ entry, score: best });
    }
    out.sort((a, b) => b.score - a.score);
    return out;
  }, [query]);

  /* Clamp the active row during render (rows shrink after filtering). */
  const maxIndex = Math.max(scored.length - 1, 0);
  const active = Math.min(activeIndex, maxIndex);

  /* Keep the active row visible in the scrollable list. */
  useEffect(() => {
    if (scored.length === 0) return;
    listRef.current
      ?.querySelector(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [active, scored.length]);

  const close = useCallback(() => onClose(), [onClose]);

  /* (100x-130) the recency row — read once per open (the dialog body
   * mounts fresh, so a jump made from THIS dialog shows next time).
   * (100x-131) the row's "Clear" affordance empties the memory live
   * (state flip hides the row without a remount; the storage is
   * already gone, so the next open starts honestly empty). */
  const recentIds = useMemo(() => loadRecentIds(), []);
  const [recentCleared, setRecentCleared] = useState(false);
  const recentEntries = useMemo(() => {
    const byId = new Map(PLAYGROUND_CARDS.map((e) => [e.id, e] as const));
    return recentIds
      .map((id) => byId.get(id))
      .filter((e): e is PlaygroundCardEntry => e !== undefined)
      .slice(0, RECENT_CAP);
  }, [recentIds]);

  const runAndClose = useCallback(
    async (entry: PlaygroundCardEntry) => {
      close();
      const found = await jumpToCard(entry);
      if (found) {
        // (100x-130) only a jump that actually LANDED is remembered —
        // a failed resolve never pollutes the recency row.
        rememberJump(entry.id);
      } else {
        toast.error("Card not found", {
          description: `“${entry.title}” is not on screen — its heading may have been renamed.`,
        });
      }
    },
    [close]
  );

  const onInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex(scored.length === 0 ? 0 : (active + 1) % scored.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex(scored.length === 0 ? 0 : (active - 1 + scored.length) % scored.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const item = scored[active];
      if (item) void runAndClose(item.entry);
    }
  };

  /* Simple focus trap: Tab stays on the input (rows via arrows). */
  const onDialogKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      inputRef.current?.focus();
    }
  };

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
        aria-label="Find a Playground card"
        onKeyDown={onDialogKeyDown}
        className="relative w-full max-w-xl overflow-hidden rounded-xl border bg-card shadow-2xl ring-1 ring-primary/10"
      >
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
                aria-controls="card-finder-listbox"
                aria-activedescendant={
                  scored.length > 0 ? `card-finder-opt-${active}` : undefined
                }
                aria-label="Find a card by title or keyword"
                placeholder="Find a card — title, blurb or keyword…"
                spellCheck={false}
                autoComplete="off"
                className="h-full min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              />
              <kbd className="hidden shrink-0 rounded-md border bg-muted px-1.5 font-mono text-[10px] text-muted-foreground sm:inline-flex">
                esc
              </kbd>
            </div>
            {/* Recent cards reached through this web-console finder.
                The row survives reloads; entries no longer in the catalog
                are dropped rather than shown as dead chips. */}
            {recentEntries.length > 0 && !recentCleared ? (
              <div className="flex flex-wrap items-center gap-1.5 border-b bg-muted/20 px-3.5 py-2">
                <span className="flex shrink-0 items-center gap-1 text-[11px] font-medium text-muted-foreground">
                  <History className="size-3.5" aria-hidden />
                  Recently jumped
                </span>
                {recentEntries.map((entry) => (
                  <button
                    key={`recent-${entry.id}`}
                    type="button"
                    onClick={() => void runAndClose(entry)}
                    className="inline-flex max-w-44 items-center gap-1 truncate rounded-full border bg-background px-2.5 py-1 text-[11px] font-medium text-foreground transition-colors hover:border-primary/40 hover:bg-primary/10 hover:text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    title={`Jump back to ${entry.title} — one of your most recent finder jumps (the row remembers up to ${RECENT_CAP}, newest first)`}
                  >
                    <span className="truncate">{entry.title}</span>
                    <span className="shrink-0 font-mono text-[9px] text-muted-foreground">
                      {entry.tab ?? "session"}
                    </span>
                  </button>
                ))}
                {/* (100x-131) the clear affordance — empties the memory
                    for good (the honest empty: no row on the next open) */}
                <button
                  type="button"
                  onClick={() => {
                    clearRecentIds();
                    setRecentCleared(true);
                  }}
                  className="ml-auto inline-flex shrink-0 items-center gap-1 rounded-full border border-dashed px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:border-destructive/40 hover:text-destructive focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  title="Forget every remembered jump — the row disappears until you land on cards again"
                >
                  <Eraser className="size-3" aria-hidden />
                  Clear
                </button>
              </div>
            ) : null}
            {/* Rows */}
            <div
              ref={listRef}
              id="card-finder-listbox"
              role="listbox"
              aria-label="Playground cards"
              className="thin-scrollbar max-h-[52vh] overflow-y-auto p-1.5"
            >
              {scored.length === 0 ? (
                <div className="space-y-2 px-3 py-6 text-center">
                  <p className="text-sm text-muted-foreground">
                    No card matches “{query.trim()}”.
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    Cards that appear only after a run (ML results, the biology
                    matrix) are not listed — the finder shows what is on screen.
                  </p>
                </div>
              ) : (
                scored.map(({ entry }, index) => {
                  const isActive = index === active;
                  return (
                    <button
                      key={entry.id}
                      type="button"
                      role="option"
                      aria-selected={isActive}
                      data-index={index}
                      id={`card-finder-opt-${index}`}
                      onMouseMove={() => {
                        if (active !== index) setActiveIndex(index);
                      }}
                      onClick={() => void runAndClose(entry)}
                      className={cn(
                        "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
                        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                        isActive
                          ? "bg-primary/10 text-primary"
                          : "text-foreground hover:bg-muted/60"
                      )}
                    >
                      <SquareArrowOutUpRight
                        className={cn(
                          "size-4 shrink-0",
                          isActive ? "text-primary" : "text-muted-foreground"
                        )}
                        aria-hidden
                      />
                      <span className="min-w-0 flex-1">
                        <HighlightText
                          text={entry.title}
                          query={query}
                          className="font-medium"
                        />
                        <span className="block truncate text-xs font-normal text-muted-foreground">
                          <HighlightText
                            text={entry.blurb}
                            query={query}
                            className="font-normal text-muted-foreground"
                          />
                        </span>
                      </span>
                      <span className="shrink-0 rounded-md border bg-muted/50 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                        {entry.tab ?? "session"}
                      </span>
                    </button>
                  );
                })
              )}
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
                  jump
                </span>
                <span className="hidden items-center gap-1 sm:flex">
                  <kbd className="rounded border bg-muted px-1 font-mono text-[10px]">
                    {HOTKEY_LABEL}
                  </kbd>
                  the browser owns Ctrl+F
                </span>
              </span>
              <span className="tabular-nums">
                {scored.length} of {PLAYGROUND_CARDS.length} cards
              </span>
            </div>
      </motion.div>
    </div>
  );
}
