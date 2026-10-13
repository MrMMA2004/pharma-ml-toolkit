/**
 * Server-side filesystem helpers for reading the pharma-ml-toolkit
 * Python package. Shared by the /api/toolkit/* routes.
 */

import { promises as fs } from "fs";
import path from "path";
import type { TreeNode, TreeStats } from "@/lib/toolkit-types";

export const TOOLKIT_ROOT = path.join(process.cwd(), "pharma-ml-toolkit");
export const TOOLKIT_NAME = "pharma-ml-toolkit";

/** Generated artifacts that should never appear in the explorer or ZIP. */
export const EXCLUDED_DIRS = new Set([".pytest_cache", "__pycache__", "node_modules"]);

const ALLOWED_EXTENSIONS = new Set([
  ".py",
  ".toml",
  ".txt",
  ".md",
  ".yml",
  ".yaml",
  ".cfg",
  ".ini",
  ".ipynb",
  ".json",
]);

const ALLOWED_BASENAMES = new Set(["Dockerfile", ".dockerignore", ".gitignore"]);

export function isViewableFile(relPath: string): boolean {
  const base = path.posix.basename(relPath);
  if (ALLOWED_BASENAMES.has(base)) return true;
  const ext = path.posix.extname(relPath).toLowerCase();
  return ALLOWED_EXTENSIONS.has(ext);
}

/** Map a relative path to a syntax-highlighting grammar id. */
export function detectLanguage(relPath: string): string {
  const base = path.posix.basename(relPath).toLowerCase();
  if (base === "dockerfile") return "docker";
  if (base === ".gitignore" || base === ".dockerignore") return "text";
  const ext = path.posix.extname(relPath).toLowerCase();
  switch (ext) {
    case ".py":
      return "python";
    case ".toml":
      return "toml";
    case ".yml":
    case ".yaml":
      return "yaml";
    case ".md":
      return "markdown";
    case ".cfg":
    case ".ini":
      return "ini";
    case ".ipynb":
      return "json";
    case ".json":
      return "json";
    default:
      return "text";
  }
}

function countLines(content: string): number {
  if (content.length === 0) return 0;
  return content.split("\n").length;
}

interface WalkAccumulator {
  files: number;
  dirs: number;
  totalLines: number;
  totalBytes: number;
}

async function walk(dirAbs: string, relDir: string, acc: WalkAccumulator): Promise<TreeNode> {
  const entries = await fs.readdir(dirAbs, { withFileTypes: true });
  const children: TreeNode[] = [];

  // Sort: directories first, then files, both alphabetical.
  const sorted = [...entries].sort((a, b) => {
    if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
    return a.name.localeCompare(b.name);
  });

  for (const entry of sorted) {
    const rel = relDir ? `${relDir}/${entry.name}` : entry.name;
    const abs = path.join(dirAbs, entry.name);
    if (entry.isDirectory()) {
      // Build/install metadata (e.g. pharmaml.egg-info) never ships.
      if (EXCLUDED_DIRS.has(entry.name) || entry.name.endsWith(".egg-info")) continue;
      acc.dirs += 1;
      const node = await walk(abs, rel, acc);
      children.push(node);
    } else if (entry.isFile()) {
      const stat = await fs.stat(abs);
      acc.files += 1;
      acc.totalBytes += stat.size;
      let lines: number | undefined;
      let language: string | undefined;
      if (isViewableFile(rel)) {
        try {
          const content = await fs.readFile(abs, "utf8");
          lines = countLines(content);
          language = detectLanguage(rel);
        } catch {
          lines = undefined;
        }
      }
      if (lines !== undefined) acc.totalLines += lines;
      children.push({
        name: entry.name,
        path: rel,
        type: "file",
        size: stat.size,
        lines,
        language,
      });
    }
  }

  const size = children.reduce((a, c) => a + c.size, 0);
  return { name: path.basename(dirAbs), path: relDir, type: "dir", size, children };
}

/** Recursively build the toolkit file tree plus aggregate stats. */
export async function buildTree(): Promise<{ root: TreeNode; stats: TreeStats }> {
  const acc: WalkAccumulator = { files: 0, dirs: 0, totalLines: 0, totalBytes: 0 };
  const root = await walk(TOOLKIT_ROOT, "", acc);
  root.name = TOOLKIT_NAME;
  return {
    root,
    stats: {
      files: acc.files,
      dirs: acc.dirs,
      totalLines: acc.totalLines,
      totalBytes: acc.totalBytes,
    },
  };
}

/**
 * Resolve a relative path inside the toolkit with traversal protection.
 * @throws Error with a user-facing message on illegal paths.
 */
export function resolveToolkitPath(relPath: string): string {
  if (relPath.length === 0 || relPath.startsWith("/") || relPath.includes("\0")) {
    throw new Error("Invalid path: expected a relative path inside the toolkit.");
  }
  const resolved = path.resolve(TOOLKIT_ROOT, relPath);
  if (resolved !== TOOLKIT_ROOT && !resolved.startsWith(TOOLKIT_ROOT + path.sep)) {
    throw new Error("Invalid path: escaping the toolkit root is not allowed.");
  }
  return resolved;
}
