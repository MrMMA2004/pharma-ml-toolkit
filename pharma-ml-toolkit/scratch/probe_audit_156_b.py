#!/usr/bin/env python3
"""(100x-156) Post-fix verification for the handle-anchored download route.

Verifies the round-39 audit fixes close the verified findings:
  A'  the truncated-artifact RED: a partially-written zip at the release
      path must now be REFUSED (fallback stream, a complete fresh archive)
      instead of a clean 200 streaming truncated bytes.
  B'  the 304 revalidation still works off the fstat-anchored ETag.
  C'  the intact artifact still serves from the release path with the
      stable per-version sha256 (61ca3e38… — the immutable shipped bytes).
  D'  zip-size consistency: exact:true + source:release on the intact
      artifact; exact:false + source:stream while the artifact is broken
      (the badge follows the path that would actually serve).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = ROOT / "download"
ROUTE = "http://localhost:3000/api/toolkit/download"
SIZE_ROUTE = "http://localhost:3000/api/toolkit/zip-size"
VERSION = (ROOT / "VERSION").read_text().strip()
ZIP = DOWNLOAD / f"pharma-ml-toolkit-v{VERSION}.zip"
ROUND38_SHA = "61ca3e38"

results: list[tuple[str, str, str]] = []


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["curl", *args], capture_output=True, text=True, timeout=90)


def headers_of(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def testzip_ok(path: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-c", f"import sys, zipfile; sys.exit(0 if zipfile.ZipFile({path!r}).testzip() is None else 1)"],
        capture_output=True,
    )
    return r.returncode == 0


def main() -> int:
    if not ZIP.exists():
        print(f"release zip missing: {ZIP}")
        return 1

    # C' + B': intact artifact — release path, stable sha, 304 revalidation
    out = curl(["-s", "-D", "-", "-o", "/tmp/pg156_fix_full.zip", ROUTE])
    status = out.stdout.splitlines()[0] if out.stdout else "?"
    h = headers_of(out.stdout)
    sha = sha256_of(Path("/tmp/pg156_fix_full.zip")) if os.path.exists("/tmp/pg156_fix_full.zip") else "?"
    ok_c = (
        "200" in status
        and h.get("x-toolkit-zip-source") == "release"
        and h.get("content-length") == str(os.path.getsize("/tmp/pg156_fix_full.zip"))
        and testzip_ok("/tmp/pg156_fix_full.zip")
        and sha.startswith(ROUND38_SHA)
    )
    results.append((
        "C' intact release serve",
        "ok" if ok_c else "RED",
        f"{status} source={h.get('x-toolkit-zip-source')} cl={h.get('content-length')} "
        f"testzip={'ok' if testzip_ok('/tmp/pg156_fix_full.zip') else 'FAIL'} sha256={sha[:16]}…",
    ))

    etag = h.get("etag", "")
    out304 = curl(["-s", "-D", "-", "-o", "/dev/null", "-H", f"If-None-Match: {etag}", ROUTE])
    st304 = out304.stdout.splitlines()[0] if out304.stdout else "?"
    n304 = len(out304.stdout.encode()) if out304.stdout else -1
    ok_b = "304" in st304
    results.append((
        "B' 304 revalidation",
        "ok" if ok_b else "AMBER",
        f"{st304} etag={etag} body_bytes≈0 (headers only: {n304}B of header text)",
    ))

    # D' baseline: zip-size on the intact artifact
    out_size = curl(["-s", SIZE_ROUTE])
    try:
        payload = json.loads(out_size.stdout)
    except json.JSONDecodeError:
        payload = {}
    ok_d1 = payload.get("exact") is True and payload.get("source") == "release"
    results.append((
        "D' zip-size intact",
        "ok" if ok_d1 else "RED",
        f"{payload}",
    ))

    # A': truncated artifact — must be refused in favour of the fallback
    backup = Path(tempfile.mkdtemp()) / ZIP.name
    original_sha = sha256_of(ZIP)
    try:
        shutil.move(str(ZIP), str(backup))
        with open(backup, "rb") as src, open(ZIP, "wb") as dst:
            dst.write(src.read(100 * 1024))  # the non-atomic writer's mid-copy state

        out = curl(["-s", "-D", "-", "-o", "/tmp/pg156_fix_trunc.zip", ROUTE])
        status = out.stdout.splitlines()[0] if out.stdout else "?"
        h = headers_of(out.stdout)
        served = os.path.getsize("/tmp/pg156_fix_trunc.zip") if os.path.exists("/tmp/pg156_fix_trunc.zip") else -1
        tz = testzip_ok("/tmp/pg156_fix_trunc.zip")
        # the fix: the release path REFUSES the truncated artifact (no
        # x-toolkit-zip-source: release), the fallback serves a COMPLETE
        # fresh archive (testzip ok, size ~13 MB)
        ok_a = (
            h.get("x-toolkit-zip-source") == "stream"
            and served > 1_000_000
            and tz
        )
        results.append((
            "A' truncated refused",
            "ok" if ok_a else "RED",
            f"{status} source={h.get('x-toolkit-zip-source')} served={served} "
            f"testzip={'ok' if tz else 'FAIL'} (a complete fresh archive, not the 100 KB truncation)",
        ))

        # D' broken: zip-size must follow (exact:false, source:stream)
        out_size = curl(["-s", SIZE_ROUTE])
        try:
            payload = json.loads(out_size.stdout)
        except json.JSONDecodeError:
            payload = {}
        ok_d2 = payload.get("exact") is False and payload.get("source") == "stream"
        results.append((
            "D' zip-size truncated",
            "ok" if ok_d2 else "AMBER",
            f"{payload}",
        ))
    finally:
        if ZIP.exists():
            ZIP.unlink()
        shutil.move(str(backup), str(ZIP))
        for tmp in ("/tmp/pg156_fix_full.zip", "/tmp/pg156_fix_trunc.zip"):
            Path(tmp).unlink(missing_ok=True)

    restored = sha256_of(ZIP)
    ok = restored == original_sha and restored.startswith(ROUND38_SHA)
    results.append((
        "restore",
        "ok" if ok else "RED",
        f"sha256 {restored[:16]}… ({'byte-identical' if restored == original_sha else 'CHANGED!'})",
    ))

    print()
    fails = 0
    for probe, verdict, evidence in results:
        if verdict != "ok":
            fails += 1
        print(f"[{verdict:>6}] {probe}: {evidence}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
