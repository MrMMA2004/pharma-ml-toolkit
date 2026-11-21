#!/usr/bin/env python3
"""R320 verify -- the 04:30 2h-sync housekeeping firing's evidence pass
over the unchanged tree (the r305 verify-only precedent):

  1. /api/toolkit/zip-size  -> 19,796,029 / exact:true / source:release / 1.2.240
  2. /api/toolkit/tree?fresh=1 -> stats.files 1,675 / stats.lines 563,174
  3. the page pins render at 1,675 / 563,174 / 8,484 / 94 / 1.2.240
     (desktop 1440x900 + 390px mobile, zero page/console app errors,
     zero mobile overflow), screenshots into scratch/ ONLY (the
     verify-only round must not touch the toolkit tree -- no re-ship).

Dev server must already be up on 127.0.0.1:3000 (dev_daemon.py)."""
import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = "http://127.0.0.1:3000"
SCRATCH = Path("/home/z/my-project/scratch")
EXPECT_ZIP = 19_796_029
EXPECT_FILES = 1_675
EXPECT_LINES = 563_174
EXPECT_TESTS = 8_484
EXPECT_VITEST = 94
EXPECT_VERSION = "1.2.240"


def get_json(path: str):
    with urllib.request.urlopen(WEB + path, timeout=120) as r:
        return json.load(r)


def main() -> int:
    ok = True

    # -- leg 1: the zip-size API (the release path)
    zs = get_json("/api/toolkit/zip-size")
    print(f"zip-size: {zs}")
    if (zs.get("zipBytes") != EXPECT_ZIP or zs.get("exact") is not True
            or zs.get("source") != "release"
            or str(zs.get("version")) != EXPECT_VERSION):
        print("ZIP-SIZE MISMATCH")
        ok = False

    # -- leg 2: the live tree walk (fresh, bypassing the TTL cache)
    tr = get_json("/api/toolkit/tree?fresh=1")
    stats = tr.get("stats", {})
    print(f"tree stats: files={stats.get('files')} dirs={stats.get('dirs')} "
          f"totalLines={stats.get('totalLines')} "
          f"totalBytes={stats.get('totalBytes')} version={stats.get('version')}")
    if (stats.get("files") != EXPECT_FILES
            or stats.get("totalLines") != EXPECT_LINES
            or str(stats.get("version")) != EXPECT_VERSION):
        print("TREE STATS MISMATCH")
        ok = False

    # -- leg 3: the page pins render (desktop + 390px mobile)
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.goto(WEB, wait_until="domcontentloaded")
        page.wait_for_selector(f"text={EXPECT_FILES:,}", timeout=60_000)
        page.wait_for_selector(f"text={EXPECT_LINES:,}", timeout=60_000)
        body = page.inner_text("body")
        for needle in (f"{EXPECT_FILES:,}", f"{EXPECT_LINES:,}",
                       f"{EXPECT_TESTS:,}", str(EXPECT_VITEST),
                       EXPECT_VERSION):
            if needle not in body:
                print(f"PIN MISSING: {needle}")
                ok = False
        print(f"page pins render: {EXPECT_FILES:,} files · {EXPECT_LINES:,} "
              f"lines · {EXPECT_TESTS:,} tests · {EXPECT_VITEST} vitest · "
              f"v{EXPECT_VERSION}")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SCRATCH / "r320-pins-desktop.png"))

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on("console", lambda m: console_errors.append(m.text)
                  if m.type == "error" else None)
        mobile.on("pageerror", lambda e: page_errors.append(str(e)))
        mobile.goto(WEB, wait_until="domcontentloaded")
        mobile.wait_for_selector(f"text={EXPECT_FILES:,}", timeout=60_000)
        overflow = mobile.evaluate(
            "() => document.documentElement.scrollWidth"
            " - document.documentElement.clientWidth")
        print(f"mobile 390px overflow: {overflow}")
        if overflow > 0:
            print("MOBILE OVERFLOW")
            ok = False
        mobile.wait_for_timeout(1200)
        mobile.screenshot(path=str(SCRATCH / "r320-pins-mobile.png"))
        browser.close()

    hmr = [e for e in console_errors if "_next/webpack-hmr" in e]
    app_console = [e for e in console_errors if "_next/webpack-hmr" not in e]
    print(f"page errors: {len(page_errors)}")
    print(f"console errors (app): {len(app_console)}")
    print(f"console errors (dev HMR channel, filtered): {len(hmr)}")
    for line in (page_errors + app_console)[:8]:
        print("  !", line[:200])
    if page_errors or app_console:
        ok = False

    print("VERIFY PASS: zip-size exact, tree stats true, pins render clean"
          if ok else "VERIFY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
