import { NextResponse } from "next/server";
import { buildTree } from "@/lib/toolkit-fs";
import type { TreeResponse } from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";

/** Tiny in-memory TTL cache so repeated tree walks are cheap. */
const CACHE_TTL_MS = 60_000;
let cache: { data: TreeResponse; expiresAt: number } | null = null;

export async function GET(request: Request) {
  try {
    // `?fresh=1` bypasses the TTL cache (the round-sync ritual: right
    // after a round's edits the cached counts are up to 60 s stale).
    // The fresh walk also REPLACES the cache, so normal browsing that
    // follows immediately serves the same numbers.
    const fresh = new URL(request.url).searchParams.get("fresh") === "1";
    if (!fresh && cache && cache.expiresAt > Date.now()) {
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
