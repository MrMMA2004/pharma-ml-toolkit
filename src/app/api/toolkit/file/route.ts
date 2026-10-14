import { promises as fs } from "fs";
import { NextRequest, NextResponse } from "next/server";
import { detectLanguage, isViewableFile, resolveToolkitPath } from "@/lib/toolkit-fs";
import type { FileContentResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const relPath = request.nextUrl.searchParams.get("path") ?? "";
  try {
    if (!relPath) {
      return NextResponse.json(
        { error: "Missing required query parameter: path" },
        { status: 400 }
      );
    }
    if (relPath.includes("..")) {
      return NextResponse.json(
        { error: "Invalid path: '..' segments are not allowed." },
        { status: 400 }
      );
    }

    let absPath: string;
    try {
      absPath = resolveToolkitPath(relPath);
    } catch (err) {
      return NextResponse.json(
        { error: err instanceof Error ? err.message : "Invalid path." },
        { status: 400 }
      );
    }

    if (!isViewableFile(relPath)) {
      return NextResponse.json(
        {
          error:
            "This file type is not viewable. Allowed: .py, .toml, .txt, .md, .yml, .yaml, .cfg, .ini, Dockerfile, .dockerignore.",
        },
        { status: 400 }
      );
    }

    let stat;
    try {
      stat = await fs.stat(absPath);
    } catch {
      return NextResponse.json(
        { error: `File not found in the toolkit: ${relPath}` },
        { status: 404 }
      );
    }
    if (!stat.isFile()) {
      return NextResponse.json(
        { error: "The requested path is not a file." },
        { status: 400 }
      );
    }

    const content = await fs.readFile(absPath, "utf8");
    const payload: FileContentResponse = {
      path: relPath,
      content,
      lines: content.length === 0 ? 0 : content.split("\n").length,
      size: stat.size,
      language: detectLanguage(relPath),
    };
    return NextResponse.json(payload);
  } catch (err) {
    return NextResponse.json(
      {
        error: "Failed to read the file.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

export const runtime = "nodejs";
