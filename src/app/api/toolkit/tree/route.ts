import { NextResponse } from "next/server";
import { buildTree } from "@/lib/toolkit-fs";
import type { TreeResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";

/** Tiny in-memory TTL cache so repeated tree walks are cheap. */
const CACHE_TTL_MS = 60_000;
let cache: { data: TreeResponse; expiresAt: number } | null = null;

export async function GET() {
  try {
    if (cache && cache.expiresAt > Date.now()) {
      return NextResponse.json(cache.data);
    }
    const { root, stats } = await buildTree();
    const data: TreeResponse = {
      root,
      stats,
      generatedAt: new Date().toISOString(),
    };
    cache = { data, expiresAt: Date.now() + CACHE_TTL_MS };
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json(
      {
        error: "Failed to walk the toolkit directory.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}
