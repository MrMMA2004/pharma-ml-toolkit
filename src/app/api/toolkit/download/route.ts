import { spawn } from "child_process";
import { createReadStream, promises as fs } from "fs";
import { Readable } from "stream";
import { NextResponse } from "next/server";
import {
  TOOLKIT_ROOT,
  ZIP_EXCLUDE_ARGS,
  openReleaseZip,
  readToolkitVersion,
  releaseZipAbsPath,
} from "@/lib/toolkit-fs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Stream the toolkit archive — the PREBUILT release artifact when it exists,
 * a fresh on-the-fly build otherwise.
 *
 * Release path (the normal case): the round-end builder leaves a
 * testzip-verified, leak-scanned archive at
 * download/pharma-ml-toolkit-v<version>.zip. It is streamed straight from
 * disk with Content-Length (the browser gets a real progress bar), a strong
 * ETag (repeat clicks revalidate — a matching If-None-Match is a zero-byte
 * 304), and the exact bytes of the shipped release: a stable SHA-256 per
 * version, so a checksum verified once matches every later download. This
 * replaced a rebuild-per-request streamer that once nested the historical
 * release zips (97 MB of already-compressed bytes) inside every download —
 * the gateway reset the connection mid-stream and the download died.
 *
 * Since 100x-156 the release path is handle-anchored: the archive is
 *  OPENED first (openReleaseZip in toolkit-fs.ts), Content-Length and
 *  the ETag come from fstat of that exact open handle, and the stream
 *  reads the same handle — a same-version archive replaced between the
 *  old stat-then-open pair used to make Content-Length describe bytes
 *  that were never streamed (probe B), and a partially-written artifact
 *  used to stream behind a clean 200 (probe A; now refused by the EOCD
 *  tail check in favour of the fallback).
 *
 * Fallback (release file missing): `zip -r` over the tree piped to stdout
 * with the SAME exclusion list the release builder uses (ZIP_EXCLUDE_ARGS
 * in toolkit-fs.ts, including *.zip). This path regenerates per request —
 * no Content-Length, no stable hash, `no-store` — but it is fast (~1 s for
 * the ~13 MB archive) and stays correct while a release is mid-rebuild.
 *
 * X-Toolkit-Zip-Source tells tooling which path served ("release"|"stream")
 * and X-Toolkit-Version exposes the release without opening the archive.
 *
 * An explicit HEAD handler ships alongside GET: Next's default HEAD
 * runs GET and discards the body, which strands the release artifact's
 * FileHandle until garbage collection (DEP0137 — "closing a FileHandle
 * on garbage collection is deprecated"). The HEAD path never creates a
 * stream: it opens + validates the artifact, closes the handle itself,
 * and answers with the exact headers a GET would send.
 */

/** Headers describing the release artifact — shared by GET (which adds
 *  the body) and HEAD (which never opens a stream), so the two can never
 *  drift apart. */
function releaseHeaderSet(version: string, size: number, mtimeMs: number) {
  const etag = `"zip-v${version}-${size}-${Math.floor(mtimeMs)}"`;
  const headers: Record<string, string> = {
    ETag: etag,
    "X-Toolkit-Version": version,
    "X-Toolkit-Zip-Source": "release",
    // no-cache: the browser may store the body but must revalidate —
    // an in-place rebuild (new size/mtime) revalidates and re-streams.
    "Cache-Control": "no-cache",
  };
  return { etag, headers };
}

/** Headers of the on-the-fly fallback — no Content-Length (the size is
 *  unknowable before the compression pass), no stable hash, no-store. */
function fallbackHeaderSet(version: string): Record<string, string> {
  return {
    "Content-Type": "application/zip",
    "Content-Disposition": `attachment; filename="pharma-ml-toolkit-v${version}.zip"`,
    "X-Toolkit-Version": version,
    "X-Toolkit-Zip-Source": "stream",
    "Cache-Control": "no-store",
  };
}

export async function GET(req: Request) {
  try {
    const info = await fs.stat(TOOLKIT_ROOT);
    if (!info.isDirectory()) {
      throw new Error("Toolkit directory is missing on the server.");
    }

    // Version the filename (and expose X-Toolkit-Version) so releases can be
    // told apart — and updated in place — without opening the archive.
    const version = await readToolkitVersion();

    // ---- Preferred: the prebuilt release artifact ---------------------
    const release = await openReleaseZip(version);
    if (release) {
      const { etag, headers } = releaseHeaderSet(version, release.size, release.mtimeMs);
      if (req.headers.get("if-none-match") === etag) {
        // Nothing streams on the 304 path — the route keeps ownership of
        // the handle and closes it itself.
        await release.fh.close();
        return new Response(null, { status: 304, headers });
      }
      // The stream takes OWNERSHIP of the handle (fd + autoClose closes
      // it on end/error/destroy — including a client abort, verified by
      // the 100x-156 fd-lifecycle probe) and reads exactly
      // [0, release.size - 1] of the inode the fstat described: the
      // Content-Length cannot lie about the bytes streamed.
      const nodeStream = createReadStream(releaseZipAbsPath(version), {
        fd: release.fh,
        start: 0,
        end: release.size - 1,
      });
      const webStream = Readable.toWeb(nodeStream) as unknown as ReadableStream<Uint8Array>;
      return new Response(webStream, {
        status: 200,
        headers: {
          ...headers,
          "Content-Type": "application/zip",
          "Content-Length": String(release.size),
          "Content-Disposition": `attachment; filename="pharma-ml-toolkit-v${version}.zip"`,
        },
      });
    }

    // ---- Fallback: on-the-fly build -----------------------------------
    const child = spawn(
      "zip",
      ["-r", "-q", "-", ".", "-x", ...ZIP_EXCLUDE_ARGS],
      { cwd: TOOLKIT_ROOT }
    );

    // Wait until the process actually spawned so launch failures (ENOENT
    // etc.) surface as a clean 500 before any bytes are sent.
    await new Promise<void>((resolve, reject) => {
      child.once("spawn", () => resolve());
      child.once("error", (err: NodeJS.ErrnoException) =>
        reject(new Error(`Failed to launch the zip utility: ${err.message}`))
      );
    });

    // Non-zero exit after streaming started can no longer change the
    // response — log it for diagnostics instead of throwing unhandled.
    let stderr = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString()).slice(-4096);
    });
    child.once("close", (code) => {
      if (code !== 0) {
        console.error(`[toolkit/download] zip exited with code ${code}: ${stderr.trim()}`);
      }
    });

    const webStream = Readable.toWeb(child.stdout) as unknown as ReadableStream<Uint8Array>;

    return new Response(webStream, {
      status: 200,
      headers: fallbackHeaderSet(version),
    });
  } catch (err) {
    return NextResponse.json(
      {
        error: "Failed to build the toolkit ZIP archive.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

/** Headers-only twin of GET — never creates a stream, so nothing can be
 *  abandoned mid-response. Next's implicit HEAD (run GET, discard the
 *  body) strands the release FileHandle until GC (DEP0137); this export
 *  opens + validates the artifact, closes the handle, and answers with
 *  the exact headers a GET would send. */
export async function HEAD(req: Request) {
  try {
    const info = await fs.stat(TOOLKIT_ROOT);
    if (!info.isDirectory()) {
      throw new Error("Toolkit directory is missing on the server.");
    }
    const version = await readToolkitVersion();

    const release = await openReleaseZip(version);
    if (release) {
      const { etag, headers } = releaseHeaderSet(version, release.size, release.mtimeMs);
      // HEAD never streams — whatever the cache answer, close the handle
      // right here (nothing else will).
      await release.fh.close();
      if (req.headers.get("if-none-match") === etag) {
        return new Response(null, { status: 304, headers });
      }
      return new Response(null, {
        status: 200,
        headers: {
          ...headers,
          "Content-Type": "application/zip",
          "Content-Length": String(release.size),
          "Content-Disposition": `attachment; filename="pharma-ml-toolkit-v${version}.zip"`,
        },
      });
    }

    // Fallback: the size is unknowable before the compression pass —
    // report the version + source so tooling still knows what would serve.
    return new Response(null, {
      status: 200,
      headers: fallbackHeaderSet(version),
    });
  } catch {
    return new Response(null, { status: 500 });
  }
}
