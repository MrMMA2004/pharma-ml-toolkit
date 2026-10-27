/**
 * Server-side filesystem helpers for reading the pharma-ml-toolkit
 * Python package. Shared by the /api/toolkit/* routes.
 */

import { spawn } from "child_process";
import { createReadStream, promises as fs } from "fs";
import type { FileHandle } from "fs/promises";
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
  // (100x-138) machine-local run logs must never ship — the venv repair
  // trace (absolute pip paths) rode the archive once and tripped the
  // round-end leak scan; logs are diagnostics for THIS machine only.
  "*.log",
  "*/*.log",
  // Generated example output (leaderboards etc.) — .gitignored on the
  // Python side, excluded here so the archive ships source only.
  "artifacts/*",
  "*/artifacts/*",
  "*.egg-info",
  "*.egg-info/*",
  "*/src/*.egg-info/*",
  // (100x-155) Release archives must never nest inside the streamed ZIP.
  // The toolkit's download/ folder accumulates one historical archive per
  // round (~12 MB each, already deflate-compressed — re-compression is
  // useless), and they silently rode every download: the streamed archive
  // ballooned from ~13 MB to 107 MB and the gateway reset the connection
  // mid-stream (zip exit 14, "Connection reset by peer"). Info-ZIP's -x
  // patterns match at any depth, so a bare "*.zip" keeps them all out —
  // the same guard the release builder (scratch/build_zip_*.py) has had
  // since round 36, finally mirrored at the route.
  "*.zip",
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
/* Release artifact serving                                            */
/* ------------------------------------------------------------------ */

/** Directory holding the round-end release artifacts (the verification
 *  screenshots + the versioned archive). It sits INSIDE the toolkit tree,
 *  but `*.zip` is excluded from the streamed archive (ZIP_EXCLUDE_ARGS),
 *  so the prebuilt release zip never nests inside itself. */
const RELEASE_ZIP_DIR = path.join(TOOLKIT_ROOT, "download");

/** Absolute path of the prebuilt release archive for a given version —
 *  the round-end builder's output (testzip-verified, leak-scanned). */
export function releaseZipAbsPath(version: string): string {
  return path.join(RELEASE_ZIP_DIR, `pharma-ml-toolkit-v${version}.zip`);
}

export interface ReleaseZipHandle {
  /** The open file handle — the caller streams from it or closes it. */
  fh: FileHandle;
  /** Exact byte size OF THE OPEN INODE — becomes the download's
   *  Content-Length. Because the size is read via fstat of the handle
   *  the caller streams from, it always describes the exact bytes that
   *  get sent: the stat-then-open race (a same-version archive replaced
   *  between the two calls made Content-Length describe a different
   *  inode than the stream read — probe B, 100x-156) is closed by
   *  construction. */
  size: number;
  /** mtime of the open inode — folded into the ETag so an in-place
   *  rebuild revalidates. */
  mtimeMs: number;
}

/** The End-of-Central-Directory signature every COMPLETE zip carries
 *  ("PK\x05\x06", 22 bytes + an optional ≤65 535-byte comment). The
 *  release artifact is verified against it before serving: a partially
 *  written archive (a future non-atomic writer — the shipped builder
 *  renames atomically, but the route must not trust that) is refused in
 *  favour of the on-the-fly fallback instead of streaming truncated
 *  bytes behind a clean 200 whose Content-Length even "matches" the
 *  truncation (probe A, 100x-156: 100 KB of the real archive served as
 *  a 200, unzip failed downstream). */
const ZIP_EOCD_SIGNATURE = Buffer.from([0x50, 0x4b, 0x05, 0x06]);
/** The EOCD may sit anywhere in the last 65 557 bytes of the file —
 *  read that whole window (plus a little slack) when validating. */
const ZIP_TAIL_WINDOW = 65_600;

/** Open and validate the prebuilt release archive for `version`.
 *
 * Resolves null when the file is absent, unreadable, implausibly
 *  empty, or INCOMPLETE (no zip End-of-Central-Directory in its tail) —
 * callers fall back to the on-the-fly streamer, which regenerates a
 *  complete archive over the current tree instead of serving the
 *  broken artifact. The returned handle is either streamed from (the
 *  caller passes it as `createReadStream`'s `fd` — the stream then OWNS
 *  it and closes it on end/error/destroy, verified by the 100x-156
 *  fd-lifecycle probe covering both the clean-drain and client-abort
 *  paths) or closed by the caller itself (the 304 / size-stat paths
 *  where no stream ever takes ownership). */
export async function openReleaseZip(version: string): Promise<ReleaseZipHandle | null> {
  let fh: FileHandle;
  try {
    fh = await fs.open(releaseZipAbsPath(version), "r");
  } catch {
    return null;
  }
  try {
    const st = await fh.stat();
    if (!st.isFile() || st.size <= 0) {
      await fh.close();
      return null;
    }
    /* Completeness: the EOCD must be present in the tail window. The
     * read goes through the OPEN handle, so the verdict describes the
     * exact inode the caller would stream. */
    const tailLen = Math.min(st.size, ZIP_TAIL_WINDOW);
    const tail = Buffer.alloc(tailLen);
    await fh.read(tail, 0, tailLen, st.size - tailLen);
    if (!tail.includes(ZIP_EOCD_SIGNATURE)) {
      await fh.close();
      return null;
    }
    return { fh, size: st.size, mtimeMs: st.mtimeMs };
  } catch {
    await fh.close().catch(() => undefined);
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
