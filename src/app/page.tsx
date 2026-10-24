"use client";

import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useTheme } from "next-themes";
import {
  Beaker,
  BookOpen,
  BookMarked,
  Download,
  FlaskConical,
  LayoutDashboard,
  Moon,
  Sun,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Overview } from "@/components/explorer/overview";
import { CodeExplorer } from "@/components/explorer/code-explorer";
import { ApiReference } from "@/components/explorer/api-reference";
import { Playground } from "@/components/explorer/playground";
import { DownloadPanel } from "@/components/explorer/download-panel";
import {
  CommandPalette,
  CommandPaletteTrigger,
} from "@/components/explorer/command-palette";
import { CardFinder } from "@/components/explorer/playground/card-finder";
import { TABS, useToolkitStore } from "@/lib/store";
import type { TabId } from "@/lib/store";
import { cn } from "@/lib/utils";
import { SHIPPED_TOOLKIT_VERSION } from "@/lib/toolkit-version"

const TAB_ICONS: Record<TabId, typeof LayoutDashboard> = {
  overview: LayoutDashboard,
  explorer: BookOpen,
  reference: BookMarked,
  playground: Beaker,
  download: Download,
};

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();

  return (
    <Button
      variant="ghost"
      size="icon"
      className="size-11 shrink-0 rounded-full"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      aria-label="Toggle color theme"
      title="Toggle theme"
    >
      {/* CSS-driven icon swap — no hydration mismatch, no mounted state. */}
      <Sun className="size-5 hidden dark:block" aria-hidden />
      <Moon className="size-5 block dark:hidden" aria-hidden />
    </Button>
  );
}

const FALLBACK_STATS = { files: 365, lines: "145,505" };

export default function Home() {
  const tab = useToolkitStore((s) => s.tab);
  const setTab = useToolkitStore((s) => s.setTab);
  const stats = useToolkitStore((s) => s.stats);

  /* Tabs behave like pages: always start a fresh tab at the top (also
   * keeps deep links to the code explorer from inheriting stale scroll). */
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [tab]);

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      {/* ------------------------------ Header ------------------------------ */}
      <header className="sticky top-0 z-40 border-b bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-4 py-2.5 sm:px-6 md:flex-row md:items-center md:justify-between md:gap-4">
          <div className="flex items-center justify-between gap-3 md:justify-start">
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/12 text-primary">
                <FlaskConical className="size-5" aria-hidden />
              </span>
              <div className="flex min-w-0 items-center gap-2">
                <h1 className="truncate text-base font-semibold tracking-tight sm:text-lg">
                  PharmaML Toolkit
                </h1>
                <Badge
                  variant="outline"
                  className="hidden shrink-0 border-primary/30 bg-primary/5 font-mono text-[10px] font-normal text-primary sm:inline-flex"
                >
                  v{stats?.version ?? SHIPPED_TOOLKIT_VERSION}
                </Badge>
              </div>
            </div>
            <div className="md:hidden">
              <div className="flex items-center gap-1">
                <CommandPaletteTrigger compact />
                <ThemeToggle />
              </div>
            </div>
          </div>

          <nav
            aria-label="Primary"
            className="pill-scroll -mx-1 flex gap-1.5 overflow-x-auto px-1 pb-0.5 md:pb-0"
          >
            {TABS.map((t) => {
              const Icon = TAB_ICONS[t.id];
              const active = tab === t.id;
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setTab(t.id)}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "relative flex h-11 shrink-0 items-center gap-1.5 rounded-full px-3.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:px-4",
                    active
                      ? "text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  {active ? (
                    <motion.span
                      layoutId="nav-active-pill"
                      className="absolute inset-0 rounded-full bg-primary/12"
                      transition={{ type: "spring", bounce: 0.25, duration: 0.5 }}
                      aria-hidden
                    />
                  ) : null}
                  <span className="relative z-10 flex items-center gap-1.5">
                    <Icon className="size-4 shrink-0" aria-hidden />
                    {t.label}
                  </span>
                </button>
              );
            })}
          </nav>

          <div className="hidden shrink-0 items-center gap-1 md:flex">
            <CommandPaletteTrigger />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* ------------------------------ Main ------------------------------ */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            {tab === "overview" ? <Overview /> : null}
            {tab === "explorer" ? <CodeExplorer /> : null}
            {tab === "reference" ? <ApiReference /> : null}
            {tab === "playground" ? <Playground /> : null}
            {tab === "download" ? <DownloadPanel /> : null}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Global command palette (Ctrl/Cmd+K), mounted once at the page level. */}
      <CommandPalette />

      {/* (100x-129) the Playground card finder (Ctrl/Cmd+Shift+F) — mounted
          at the page level so the hotkey works from any top tab; opening it
          from elsewhere switches to the Playground first. */}
      <CardFinder />

      {/* ------------------------------ Footer ------------------------------ */}
      <footer className="mt-auto border-t bg-muted/30 pb-[max(1.5rem,env(safe-area-inset-bottom))]">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-6 sm:px-6 md:flex-row md:items-start md:justify-between">
          <div className="max-w-md space-y-1.5">
            <p className="flex items-center gap-2 text-sm font-medium">
              <FlaskConical className="size-4 text-primary" aria-hidden />
              PharmaML Toolkit
            </p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Research &amp; education only — not medical advice. Always verify pharmaceutical
              calculations against approved references.
            </p>
            <p className="text-xs text-muted-foreground">Released under the MIT License.</p>
          </div>

          <nav aria-label="Footer quick links" className="flex flex-wrap gap-2">
            {TABS.map((t) => (
              <Button
                key={t.id}
                variant="ghost"
                size="sm"
                className="h-9 px-3 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </Button>
            ))}
          </nav>

          <p className="flex shrink-0 items-center gap-2 font-mono text-xs tabular-nums text-muted-foreground">
            {stats ? `${stats.files} files · ${stats.totalLines.toLocaleString("en-US")} LOC` : `${FALLBACK_STATS.files} files · ${FALLBACK_STATS.lines} LOC`}
            {" · "}
            <span className="ping-ring relative inline-flex size-2 rounded-full bg-primary" aria-hidden />
            <span className="text-primary">2471 tests</span>
          </p>
        </div>
      </footer>
    </div>
  );
}
