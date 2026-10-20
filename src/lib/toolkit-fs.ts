/**
 * Server-side filesystem helpers for reading the pharma-ml-toolkit
 * Python package. Shared by the /api/toolkit/* routes.
 */

import { spawn } from "child_process";
import { createReadStream, promises as fs } from "fs";
import path from "path";
import { isTreeDataNode } from "@/lib/data-file";
import type { TreeNode, TreeStats } from "@/lib/toolkit-types";

export const TOOLKIT_ROOT = path.join(process.cwd(), "pharma-ml-toolkit");
export const TOOLKIT_NAME = "pharma-ml-toolkit";

/** Generated artifacts that should never appear in the explorer or ZIP.
 * The virtualenv entries matter: a stray `uv venv` / `python -m venv` inside
 * the toolkit once inflated the live stats from 93 to 6,732 files — local
 * environments must never leak into the served tree. `artifacts` covers
 * generated example output (e.g. examples/artifacts/ leaderboards), which
 * is .gitignored on the Python side but was leaking into tree/ZIP stats. */
export const EXCLUDED_DIRS = new Set([
  ".pytest_cache",
  "__pycache__",
  "node_modules",
  "catboost_info",
  "artifacts",
  ".venv",
  "venv",
  "env",
  ".git",
  ".ruff_cache",
  ".mypy_cache",
  ".tox",
  ".idea",
  ".vscode",
  ".cache",
]);

/** Exclusion arguments shared by the ZIP streamer (/api/toolkit/download)
 *  and the archive-size measurement — ONE list so the measured number is
 *  always the byte count the client actually receives. Order mirrors the
 *  `zip -x` convention: each pattern applies at every archive depth. */
export const ZIP_EXCLUDE_ARGS: readonly string[] = [
  ".pytest_cache/*",
  "*/.pytest_cache/*",
  "__pycache__/*",
  "*/__pycache__/*",
  "*.pyc",
  "catboost_info/*",
  "*/catboost_info/*",
  // Generated example output (leaderboards etc.) — .gitignored on the
  // Python side, excluded here so the archive ships source only.
  "artifacts/*",
  "*/artifacts/*",
  "*.egg-info",
  "*.egg-info/*",
  "*/src/*.egg-info/*",
  // Local virtualenvs / tool caches must never ship — a stray `uv venv`
  // inside the toolkit once added 265 MB to the archive.
  ".venv/*",
  "*/.venv/*",
  "venv/*",
  "*/venv/*",
  "env/*",
  "*/env/*",
  ".git/*",
  "*/.git/*",
  ".ruff_cache/*",
  "*/.ruff_cache/*",
  ".mypy_cache/*",
  "*/.mypy_cache/*",
  ".tox/*",
  "*/.tox/*",
  ".idea/*",
  "*/.idea/*",
  ".vscode/*",
  "*/.vscode/*",
];

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

/** Language marker for tree data nodes (.csv/.pdb/.dlg/.glg/.gpf/.dpf).
 *  These nodes are previewed by /api/toolkit/file, never line-counted. */
export function detectDataLanguage(relPath: string): string {
  const ext = path.posix.extname(relPath).toLowerCase();
  if (ext === ".csv") return "csv";
  if (ext === ".pdb") return "pdb";
  return "text";
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
      if (isTreeDataNode(rel)) {
        // Data files carry a language marker for the explorer UI but are
        // NEVER line-counted — the LOC stat stays source-only.
        language = detectDataLanguage(rel);
      } else if (isViewableFile(rel)) {
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

/** The toolkit's release version: the `version = "x.y.z"` line inside
 *  pyproject.toml's [project] section, falling back to the VERSION file,
 *  then "0.0.0" when neither is readable. */
export async function readToolkitVersion(): Promise<string> {
  const fromPyproject = await readProjectVersion();
  if (fromPyproject) return fromPyproject;
  try {
    return (await fs.readFile(path.join(TOOLKIT_ROOT, "VERSION"), "utf8")).trim();
  } catch {
    return "0.0.0";
  }
}

/** Extract `version = "…"` from the [project] section of pyproject.toml —
 *  any other section's version keys are ignored. Returns null when the file
 *  is unreadable or carries no project version. */
async function readProjectVersion(): Promise<string | null> {
  try {
    const content = await fs.readFile(path.join(TOOLKIT_ROOT, "pyproject.toml"), "utf8");
    let inProject = false;
    for (const rawLine of content.split("\n")) {
      const line = rawLine.trim();
      if (line.startsWith("[")) {
        inProject = line === "[project]";
        continue;
      }
      if (!inProject) continue;
      const match = /^version\s*=\s*"([^"]+)"/.exec(line);
      if (match) return match[1];
    }
    return null;
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Archive size measurement                                            */
/* ------------------------------------------------------------------ */

/** How long a successful ZIP-size measurement stays fresh (10 min). */
const ZIP_SIZE_TTL_MS = 10 * 60_000;
/** Backoff after a failed measurement (utility missing, non-zero exit) —
 *  short enough that the next visitor retries, long enough that a broken
 *  environment does not re-spawn `zip` on every request. */
const ZIP_SIZE_FAILURE_TTL_MS = 60_000;

let zipSizeCache: { bytes: number; expiresAt: number } | null = null;
let zipSizeFailureUntil = 0;
let zipSizeInFlight: Promise<number | null> | null = null;

/** Byte size of the archive /api/toolkit/download would stream right now —
 *  the SAME `zip` invocation with the SAME exclusion list, stdout counted
 *  in-process (nothing written to disk). Results are cached 10 minutes and
 *  concurrent callers share a single measurement; resolves null when the
 *  utility is unavailable or exits non-zero. */
export function measureZipBytes(): Promise<number | null> {
  const now = Date.now();
  if (zipSizeCache && zipSizeCache.expiresAt > now) return Promise.resolve(zipSizeCache.bytes);
  if (now < zipSizeFailureUntil) return Promise.resolve(null);
  if (zipSizeInFlight) return zipSizeInFlight;

  zipSizeInFlight = new Promise<number | null>((resolve) => {
    let total = 0;
    let settled = false;
    const child = spawn(
      "zip",
      ["-r", "-q", "-", ".", "-x", ...ZIP_EXCLUDE_ARGS],
      { cwd: TOOLKIT_ROOT, stdio: ["ignore", "pipe", "pipe"] }
    );
    const finish = (value: number | null) => {
      if (settled) return;
      settled = true;
      zipSizeInFlight = null;
      const at = Date.now();
      if (value !== null) {
        zipSizeCache = { bytes: value, expiresAt: at + ZIP_SIZE_TTL_MS };
      } else {
        zipSizeFailureUntil = at + ZIP_SIZE_FAILURE_TTL_MS;
      }
      resolve(value);
    };
    child.stdout?.on("data", (chunk: Buffer) => {
      total += chunk.length;
    });
    child.once("error", () => finish(null));
    child.once("close", (code) => finish(code === 0 ? total : null));
  });
  return zipSizeInFlight;
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
      version: await readToolkitVersion(),
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

/** Sentinel thrown by {@link readToolkitFileBounded} when a stream exceeds
 *  its byte budget (stat raced with a growing file, etc.). */
export const FILE_TOO_LARGE = "FILE_TOO_LARGE";

/**
 * Stream at most `maxBytes` bytes of a toolkit file into a Buffer. Uses the
 * TOP-LEVEL `createReadStream` from "fs" — `fs.promises` has no stream API
 * (a real bug in a previous implementation). The hard byte stop guarantees
 * a bounded read even if the file grows between stat() and open().
 * @throws Error with message FILE_TOO_LARGE when the budget is exceeded.
 */
export function readToolkitFileBounded(absPath: string, maxBytes: number): Promise<Buffer> {
  return new Promise<Buffer>((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let settled = false;
    const stream = createReadStream(absPath);
    const fail = (err: Error) => {
      if (settled) return;
      settled = true;
      stream.destroy();
      reject(err);
    };
    stream.on("data", (chunk: Buffer) => {
      if (settled) return;
      total += chunk.length;
      if (total > maxBytes) {
        fail(new Error(FILE_TOO_LARGE));
        return;
      }
      chunks.push(chunk);
    });
    stream.on("error", fail);
    stream.on("end", () => {
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks));
    });
    stream.on("close", () => {
      // Covers destroy()-without-end paths; no-op after a clean end.
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks));
    });
  });
}
