import { spawn } from "child_process";
import { Readable } from "stream";
import { promises as fs } from "fs";
import { NextResponse } from "next/server";
import { TOOLKIT_ROOT } from "@/lib/toolkit-fs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Stream a ZIP archive of the whole toolkit directory.
 * Uses the `zip` CLI piped to stdout so the archive streams to the client
 * without buffering the entire file in memory. Generated cache artifacts
 * (.pytest_cache, __pycache__) are excluded — matching the explorer tree
 * and the package's .gitignore.
 */
export async function GET() {
  try {
    const info = await fs.stat(TOOLKIT_ROOT);
    if (!info.isDirectory()) {
      throw new Error("Toolkit directory is missing on the server.");
    }

    const child = spawn(
      "zip",
      [
        "-r",
        "-q",
        "-",
        ".",
        "-x",
        ".pytest_cache/*",
        "*/.pytest_cache/*",
        "__pycache__/*",
        "*/__pycache__/*",
        "*.pyc",
      ],
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
      headers: {
        "Content-Type": "application/zip",
        "Content-Disposition": 'attachment; filename="pharma-ml-toolkit.zip"',
        "Cache-Control": "no-store",
      },
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
