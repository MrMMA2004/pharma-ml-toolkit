import { promises as fs } from "fs";
import { NextRequest, NextResponse } from "next/server";
import {
  dataFileKindFromPath,
  dataFileMaxBytes,
  parseCsvText,
  parsePdbText,
  previewPlainText,
} from "@/lib/data-file";
import { fmtBytes } from "@/lib/format";
import {
  FILE_TOO_LARGE,
  detectLanguage,
  isViewableFile,
  readToolkitFileBounded,
  resolveToolkitPath,
} from "@/lib/toolkit-fs";
import type {
  CsvPreviewResponse,
  FileContentResponse,
  PdbPreviewResponse,
  TextPreviewResponse,
} from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";

const NOT_VIEWABLE_MESSAGE =
  "This file type is not viewable. Allowed: .py, .toml, .txt, .md, .yml, " +
  ".yaml, .cfg, .ini, .ipynb, .json, Dockerfile, .dockerignore — shipped " +
  "data files (.csv, .pdb, .dlg, .glg, .gpf, .dpf) render as previews.";

/** Actionable message for files beyond the preview byte caps. */
function oversizedMessage(fileName: string, size: number, maxBytes: number): string {
  const kindLabel = maxBytes > 512 * 1024 ? "PDB/text" : "CSV";
  return (
    `${fileName} is ${fmtBytes(size)} — ${kindLabel} previews are capped at ` +
    `${fmtBytes(maxBytes)}. Use the Download tab to get the full toolkit archive.`
  );
}

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

    /* ------------------------- Data-file preview ------------------------- */
    const dataKind = dataFileKindFromPath(relPath);
    if (dataKind !== null) {
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
      const maxBytes = dataFileMaxBytes(dataKind);
      const fileName = relPath.split("/").pop() ?? relPath;
      if (stat.size > maxBytes) {
        return NextResponse.json(
          { error: oversizedMessage(fileName, stat.size, maxBytes) },
          { status: 400 }
        );
      }

      let buffer: Buffer;
      try {
        buffer = await readToolkitFileBounded(absPath, maxBytes);
      } catch (err) {
        if (err instanceof Error && err.message === FILE_TOO_LARGE) {
          return NextResponse.json(
            { error: oversizedMessage(fileName, stat.size, maxBytes) },
            { status: 400 }
          );
        }
        throw err;
      }
      const text = buffer.toString("utf8");

      if (dataKind === "csv") {
        const parsed = parseCsvText(text);
        const payload: CsvPreviewResponse = {
          kind: "csv",
          path: relPath,
          columns: parsed.columns,
          rowCount: parsed.rowCount,
          previewRows: parsed.previewRows,
          truncated: parsed.truncated,
          totalBytes: stat.size,
        };
        return NextResponse.json(payload);
      }
      if (dataKind === "pdb") {
        const parsed = parsePdbText(text);
        const payload: PdbPreviewResponse = {
          kind: "pdb",
          path: relPath,
          totalBytes: stat.size,
          atomCount: parsed.atomCount,
          residueCount: parsed.residueCount,
          totalLineCount: parsed.totalLineCount,
          previewText: parsed.previewText,
        };
        return NextResponse.json(payload);
      }
      const parsed = previewPlainText(text);
      const payload: TextPreviewResponse = {
        kind: "text",
        path: relPath,
        previewText: parsed.previewText,
        truncated: parsed.truncated,
        totalBytes: stat.size,
        totalLineCount: parsed.totalLineCount,
      };
      return NextResponse.json(payload);
    }

    /* ------------------ Source file (byte-compatible) -------------------- */
    if (!isViewableFile(relPath)) {
      return NextResponse.json(
        { error: NOT_VIEWABLE_MESSAGE },
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
