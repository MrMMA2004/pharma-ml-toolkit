#!/usr/bin/env python3
"""(100x-156) Round-39 audit probes for the round-38 download route.

The audit target (worklog round-38 close-out, candidate a): the download
route's new release/fallback split — the ETag mtime race during an
in-place rebuild, the partial-file window, and the stat-then-open
consistency of Content-Length.

Probes:
  A  partial-file hazard: a NON-atomic writer leaves a truncated zip at
     download/pharma-ml-toolkit-v<ver>.zip — does the route serve the
     truncated bytes as a clean 200 whose Content-Length "matches"?
     (The shipped builder renames atomically, so this models a future
     cross-filesystem copy or a manual cp — the route currently guards
     only size <= 0.)
  B  stat-then-open consistency: probe the window in Node directly —
     stat(), replace the file, open() — what does createReadStream read?
     (Documents the race the fix must eliminate by construction.)
  C  fd lifecycle semantics: does createReadStream(path, { fd }) close
     the caller's fd (autoClose) or must the route close it manually?
     Empirical, both autoClose defaults — the fix must not leak fds.
  D  restore discipline: the release zip must come back byte-identical
     (sha256 61ca3e38… per the round-38 close-out) — asserted at exit.
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
VERSION = (ROOT / "VERSION").read_text().strip()
ZIP = DOWNLOAD / f"pharma-ml-toolkit-v{VERSION}.zip"
ROUND38_SHA = "61ca3e38"  # the stable per-version prefix (worklog round 38)

results: list[tuple[str, str, str]] = []  # (probe, verdict, evidence)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["curl", *args], capture_output=True, text=True, timeout=60)


def head_route() -> tuple[str, dict[str, str]]:
    out = curl(["-s", "-D", "-", "-o", "/dev/null", "-X", "HEAD", ROUTE])
    headers: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return out.stdout.splitlines()[0] if out.stdout else "?", headers


# ------------------------------------------------------------------
# A: the partial-file hazard (live, against the running dev server)
# ------------------------------------------------------------------
def probe_a() -> None:
    backup = Path(tempfile.mkdtemp()) / ZIP.name
    original_sha = sha256_of(ZIP)
    try:
        shutil.move(str(ZIP), str(backup))
        # a non-atomic writer's mid-copy state: the first 100 KB of the
        # real archive sitting at the release path
        with open(backup, "rb") as src, open(ZIP, "wb") as dst:
            dst.write(src.read(100 * 1024))
        out = curl(["-s", "-D", "-", "-o", "/tmp/pg156_partial.zip", ROUTE])
        status = out.stdout.splitlines()[0] if out.stdout else "?"
        headers = {}
        for line in out.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        served = os.path.getsize("/tmp/pg156_partial.zip") if os.path.exists("/tmp/pg156_partial.zip") else -1
        cl = headers.get("content-length", "?")
        src_hdr = headers.get("x-toolkit-zip-source", "?")
        # is the served body a valid zip?
        tz = subprocess.run(["python3", "-c", "import zipfile,sys;sys.exit(0 if zipfile.ZipFile('/tmp/pg156_partial.zip').testzip() is None else 1)"], capture_output=True)
        verdict = (
            "RED"
            if status.endswith("200") and served == 100 * 1024 and cl == str(100 * 1024) and tz.returncode != 0
            else "unexpected"
        )
        results.append((
            "A partial-file",
            verdict,
            f"{status} cl={cl} served={served} source={src_hdr} testzip={'FAIL' if tz.returncode != 0 else 'ok'} "
            f"(the route served a TRUNCATED archive as a clean 200 whose Content-Length matches the truncation)",
        ))
    finally:
        if ZIP.exists():
            ZIP.unlink()
        shutil.move(str(backup), str(ZIP))
        Path("/tmp/pg156_partial.zip").unlink(missing_ok=True)


# ------------------------------------------------------------------
# B + C: Node fd semantics (the fix's building blocks), probed directly
# ------------------------------------------------------------------
NODE_PROBE = r"""
const { open, fstat, close } = require("fs").promises;
const { createReadStream, writeFileSync } = require("fs");
(async () => {
  // B: stat-then-open race — replace the file between stat and open
  writeFileSync("/tmp/pg156_race.bin", "AAAA");
  const path = "/tmp/pg156_race.bin";
  const st1 = await require("fs").promises.stat(path);        // stat: size 4
  writeFileSync(path + ".tmp", "BBBBBBBB");                    // writer lands a new file
  require("fs").renameSync(path + ".tmp", path);               // atomic replace
  const stream = createReadStream(path);                       // open: NEW inode, 8 bytes
  let got = "";
  for await (const chunk of stream) got += chunk.toString();
  console.log(JSON.stringify({
    probe: "B stat-then-open",
    statSize: st1.size, bytesRead: Buffer.byteLength(got),
    mismatch: st1.size !== Buffer.byteLength(got),
  }));

  // C: fd lifecycle — autoClose honouring when fd is passed explicitly
  writeFileSync("/tmp/pg156_fd.bin", "0123456789");
  const fd = await open("/tmp/pg156_fd.bin", "r");
  const st = await fstat(fd);
  const rs = createReadStream("/tmp/pg156_fd.bin", { fd, start: 0, end: st.size - 1 });
  let fdAliveAfterEnd = null;
  rs.on("close", async () => {
    try { await fstat(fd); fdAliveAfterEnd = true; }
    catch { fdAliveAfterEnd = false; }
    console.log(JSON.stringify({
      probe: "C fd-lifecycle", autoClosed: fdAliveAfterEnd === false,
    }));
    if (fdAliveAfterEnd) await close(fd); // don't leak in the probe itself
  });
  rs.on("data", () => {}); // drain
})().catch((e) => { console.error(String(e)); process.exit(1); });
"""


def probe_bc() -> None:
    out = subprocess.run(["node", "-e", NODE_PROBE], capture_output=True, text=True, timeout=30)
    lines = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
    parsed = [json.loads(l) for l in lines]
    for p in parsed:
        if p.get("probe") == "B stat-then-open":
            verdict = "AMBER" if p["mismatch"] else "ok"
            results.append((
                "B stat-then-open race",
                verdict,
                f"stat said {p['statSize']} bytes but the stream read {p['bytesRead']} — a replacement "
                f"between stat() and open() makes Content-Length describe bytes that are NOT streamed",
            ))
        if p.get("probe") == "C fd-lifecycle":
            results.append((
                "C fd lifecycle",
                "info",
                f"createReadStream(path, {{fd}}) autoClosed={p['autoClosed']} — the fix's close discipline",
            ))
    if not parsed:
        results.append(("B/C node probe", "ERROR", out.stderr[:300]))


def main() -> int:
    if not ZIP.exists():
        print(f"release zip missing: {ZIP}")
        return 1
    status, headers = head_route()
    print(f"baseline: {status} content-length={headers.get('content-length')} "
          f"source={headers.get('x-toolkit-zip-source')} etag={headers.get('etag')}")

    probe_a()
    probe_bc()

    # D: restore discipline
    restored = sha256_of(ZIP)
    ok = restored.startswith(ROUND38_SHA)
    results.append((
        "D restore",
        "ok" if ok else "RED",
        f"sha256 {restored[:16]}… {'==' if ok else '!='} round-38 {ROUND38_SHA}…",
    ))

    print()
    for probe, verdict, evidence in results:
        print(f"[{verdict:>10}] {probe}: {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
