/**
 * Client-side app state (single-page tab navigation + toolkit tree cache
 * + selected file in the code explorer). Zustand keeps the selected file
 * and loaded tree alive across tab switches.
 */

import { create } from "zustand";
import type { TreeNode, TreeResponse, TreeStats } from "@/lib/toolkit-types";

export type TabId = "overview" | "explorer" | "reference" | "playground" | "download";

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

  tree: TreeNode | null;
  stats: TreeStats | null;
  treeStatus: TreeStatus;
  treeError: string | null;
  /** Fetch the toolkit tree once; concurrent calls are collapsed. */
  ensureTree: () => Promise<void>;
}

export const useToolkitStore = create<ToolkitStore>((set, get) => ({
  tab: "overview",
  setTab: (tab) => set({ tab }),

  selectedFile: "README.md",
  setSelectedFile: (path) => set({ selectedFile: path, targetLine: null }),

  targetLine: null,
  setTargetLine: (line) => set({ targetLine: line }),
  openFileAtLine: (path, line) => set({ selectedFile: path, targetLine: line }),

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
