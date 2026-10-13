/**
 * Client-side app state (single-page tab navigation + toolkit tree cache
 * + selected file in the code explorer). Zustand keeps the selected file
 * and loaded tree alive across tab switches.
 */

import { create } from "zustand";
import type { TreeNode, TreeResponse, TreeStats } from "@/lib/toolkit-types";

export type TabId = "overview" | "explorer" | "reference" | "playground" | "download";

/** Persisted one-compartment PK regimen (survives playground tab switches). */
export interface PkParams {
  route: "iv" | "oral";
  dose: number; // mg
  vd: number; // L
  cl: number; // L/h
  tau: number; // h
  doses: number;
  f: number; // bioavailability (oral)
  ka: number; // 1/h (oral)
}

export const DEFAULT_PK_PARAMS: PkParams = {
  route: "iv",
  dose: 250,
  vd: 45,
  cl: 3.2,
  tau: 8,
  doses: 6,
  f: 0.9,
  ka: 1.2,
};

/** Max entries kept in the "recently viewed" strip. */
export const RECENT_LIMIT = 8;

export const TABS: Array<{ id: TabId; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "explorer", label: "Code Explorer" },
  { id: "reference", label: "API Reference" },
  { id: "playground", label: "Playground" },
  { id: "download", label: "Download" },
];

type TreeStatus = "idle" | "loading" | "ready" | "error";

interface ToolkitStore {
  tab: TabId;
  setTab: (tab: TabId) => void;

  /** Currently selected file (relative path) in the code explorer. */
  selectedFile: string;
  setSelectedFile: (path: string) => void;

  /** Definition line to highlight + scroll to (API-reference deep links). */
  targetLine: number | null;
  setTargetLine: (line: number | null) => void;
  /** Open a file and (optionally) jump to a definition line. */
  openFileAtLine: (path: string, line: number | null) => void;

  /** Most-recently-viewed files (newest first, deduplicated). */
  recentFiles: string[];
  /** Clear the recent-files strip. */
  clearRecentFiles: () => void;

  /** Active playground sub-tab (restored when returning to Playground). */
  playgroundTab: string;
  setPlaygroundTab: (tab: string) => void;

  /** PK simulator regimen — persisted so sliders survive tab switches. */
  pkParams: PkParams;
  setPkParam: <K extends keyof PkParams>(key: K, value: PkParams[K]) => void;
  resetPkParams: () => void;

  tree: TreeNode | null;
  stats: TreeStats | null;
  treeStatus: TreeStatus;
  treeError: string | null;
  /** Fetch the toolkit tree once; concurrent calls are collapsed. */
  ensureTree: () => Promise<void>;
}

/** Newest-first, deduplicated, capped recent list (pure helper). */
function pushRecent(recent: string[], path: string): string[] {
  const next = [path, ...recent.filter((p) => p !== path)];
  return next.slice(0, RECENT_LIMIT);
}

export const useToolkitStore = create<ToolkitStore>((set, get) => ({
  tab: "overview",
  setTab: (tab) => set({ tab }),

  selectedFile: "README.md",
  setSelectedFile: (path) =>
    set((s) => ({
      selectedFile: path,
      targetLine: null,
      recentFiles: pushRecent(s.recentFiles, path),
    })),

  targetLine: null,
  setTargetLine: (line) => set({ targetLine: line }),
  openFileAtLine: (path, line) =>
    set((s) => ({
      selectedFile: path,
      targetLine: line,
      recentFiles: pushRecent(s.recentFiles, path),
    })),

  recentFiles: [],
  clearRecentFiles: () => set({ recentFiles: [] }),

  playgroundTab: "calculators",
  setPlaygroundTab: (tab) => set({ playgroundTab: tab }),

  pkParams: DEFAULT_PK_PARAMS,
  setPkParam: (key, value) =>
    set((s) => ({ pkParams: { ...s.pkParams, [key]: value } })),
  resetPkParams: () => set({ pkParams: DEFAULT_PK_PARAMS }),

  tree: null,
  stats: null,
  treeStatus: "idle",
  treeError: null,
  ensureTree: async () => {
    const { treeStatus, tree } = get();
    if (treeStatus === "loading" || tree) return;
    set({ treeStatus: "loading", treeError: null });
    try {
      const res = await fetch("/api/toolkit/tree");
      if (!res.ok) {
        throw new Error(`Tree API responded with HTTP ${res.status}`);
      }
      const data = (await res.json()) as TreeResponse;
      set({ tree: data.root, stats: data.stats, treeStatus: "ready" });
    } catch (err) {
      set({
        treeStatus: "error",
        treeError:
          err instanceof Error ? err.message : "Unknown error while loading the file tree.",
      });
    }
  },
}));
