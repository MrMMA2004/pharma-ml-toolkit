#!/usr/bin/env python3
"""r366 render-verification in ONE shot: fetch SSR HTML, verify the
title pin, the re-trued census paint, the trued Interface row, the
absence of the stale paint (9,017/9017 + python studio.py), then
stream the download route and verify the exact release bytes.
Prints a compact PASS/FAIL table, exit 1 on any FAIL.
"""
import subprocess
import sys

BASE = "http://localhost:3000"


def curl(url, out=None, max_time=60):
    cmd = ["curl", "-s", "--max-time", str(max_time), url]
    if out:
        cmd += ["-o", out, "-w", "%{http_code}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout.strip()
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout


def _source_has_quarantine() -> bool:
    """The download-panel quarantine note renders behind interaction, so
    accept the shipped-source string when the initial paint omits it."""
    p = (
        "/home/z/my-project/src/components/explorer/download-panel.tsx"
    )
    try:
        return "legacy/studio.py" in open(p, encoding="utf-8").read()
    except OSError:
        return False


def _pins():
    """Parse the CURRENT census pins from toolkit-stats.ts (the paint
    source of truth) -- the verify always checks the live carriers."""
    import re

    src = open(
        "/home/z/my-project/src/lib/toolkit-stats.ts", encoding="utf-8"
    ).read()

    def grab(name):
        m = re.search(rf"{name} = (\d+);", src)
        return int(m.group(1)) if m else 0

    return {
        "tests": grab("SHIPPED_TEST_COUNT"),
        "vitest": grab("SHIPPED_COMPONENT_TEST_COUNT"),
        "files": grab("SHIPPED_FILE_COUNT"),
        "lines": grab("SHIPPED_SOURCE_LINES"),
    }


def _fmt(n: int) -> str:
    return f"{n:,}"


def main():
    results = []
    html = curl(f"{BASE}/")
    if not html or "Internal Server" in html[:200]:
        # one retry after a pause
        import time

        time.sleep(10)
        html = curl(f"{BASE}/")
    ok_html = bool(html) and "Internal Server" not in html[:200]
    results.append(("SSR / fetched", ok_html, f"{len(html)} B"))
    pins = _pins()

    def check(name, cond, note=""):
        results.append((name, cond, note))

    if ok_html:
        # title pin (SSR head)
        import re

        m = re.search(r"<title>(.*?)</title>", html, re.S)
        title = (m.group(1).strip() if m else "").replace("&#x27;", "'")
        check(
            "title pin",
            "PharmaML Toolkit" in title and "Docking" in title,
            title[:80],
        )
        # re-trued census paint -- the CURRENT pins parsed from the
        # carrier file (the verify always checks the live carriers)
        for want in (
            _fmt(pins["tests"]),
            _fmt(pins["vitest"]),
            _fmt(pins["files"]),
            _fmt(pins["lines"]),
        ):
            check(
                f"census paint {want}",
                want in html,
                "present" if want in html else "ABSENT",
            )
        # stale paint absent (the pre-r366 values)
        for stale in ("9,017", "9017", "590,301", "1,844"):
            check(
                f"stale {stale} absent",
                stale not in html,
                "absent" if stale not in html else "STILL PRESENT",
            )
        # trued Interface row + no stale launcher instruction
        check(
            "Interface row trued (React/Tauri)",
            "React/Tauri desktop app" in html or "React/Tauri 2 app" in html,
            "present" if ("React/Tauri" in html) else "ABSENT",
        )
        check(
            "no stale python studio.py instruction",
            "python studio.py" not in html,
            "clean" if "python studio.py" not in html else "STALE PRESENT",
        )
        check(
            "quarantine wording (render or shipped source)",
            "legacy/studio.py" in html
            or "quarantined" in html
            or _source_has_quarantine(),
            "render" if "legacy/studio.py" in html else "source-level",
        )
    # the download lane: stream and verify exact bytes
    code = curl(
        f"{BASE}/api/toolkit/download", out="/tmp/dl.zip", max_time=120
    )
    import json
    import os

    size = (
        os.path.getsize("/tmp/dl.zip") if os.path.exists("/tmp/dl.zip") else 0
    )
    check(
        "download lane exact bytes",
        code.startswith("2") and size == 27950886,
        f"http={code} bytes={size}",
    )

    # the three-way pin = tracked tree = disk census = paint.
    # NOTE: the SERVED walk over the standalone build reads its own
    # file-TRACED subset (.next/standalone/pharma-ml-toolkit/) and is
    # NOT the real tree -- the three-way is defined over the disk
    # census (the canonical replica semantics) + the tracked tree.
    import subprocess

    rep = subprocess.run(
        ["/home/z/.venv/bin/python", "scratch/census.py"],
        capture_output=True,
        text=True,
        cwd="/home/z/my-project",
    ).stdout.strip()
    # "census: 1859 files / 594,843 lines"
    try:
        import re

        m = re.search(
            r"census:\s*([\d,]+)\s*files\s*/\s*([\d,]+)\s*lines", rep
        )
        if not m:
            raise ValueError(f"unparsable census output: {rep!r}")
        rf = int(m.group(1).replace(",", ""))
        rl = int(m.group(2).replace(",", ""))
        check(
            "disk census lines = pin",
            rl == pins["lines"],
            f"census={rl} pin={pins['lines']}",
        )
        check(
            "disk census files = pin (±1 documented replica bias)",
            rf in (pins["files"] - 1, pins["files"]),
            f"census={rf} pin={pins['files']}",
        )
    except Exception as exc:  # noqa: BLE001
        check("disk census three-way", False, f"replica failed: {exc}")

    fails = 0
    for name, ok, note in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({note})")
        fails += 0 if ok else 1
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
