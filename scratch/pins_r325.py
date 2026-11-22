#!/usr/bin/env python3
"""round-325 pins pair — the web-app page render at the trued fixed
point (the r313/r321/r324 convention): desktop + 390px mobile, zero
page/console errors, the stats line carrying this round's pins
(parameterized: files / lines / tests / component tests / version).

Usage: pins_r325.py <FILES> <LINES> <TESTS> <COMPONENT>
e.g.     pins_r325.py 1706 567706 8555 110
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = Path("/home/z/my-project/pharma-ml-toolkit/docs/gui/screenshots")
WEB = "http://localhost:3000"

FILES, LINES, TESTS, COMPONENT = (int(a) for a in sys.argv[1:5])
FMT_FILES = f"{FILES:,}"
FMT_LINES = f"{LINES:,}"
FMT_TESTS = f"{TESTS:,}"


def main() -> int:
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.goto(WEB, wait_until="domcontentloaded")
        page.wait_for_selector(f"text={FMT_FILES}", timeout=60_000)
        page.wait_for_selector(f"text={FMT_LINES}", timeout=30_000)
        page.wait_for_selector(f"text={FMT_TESTS}", timeout=30_000)
        page.wait_for_selector(f"text={COMPONENT}", timeout=30_000)
        page.wait_for_timeout(800)
        page.screenshot(
            path=str(SHOTS / "r325-pins-desktop.png"), full_page=False
        )
        print(f"desktop: the trued pins rendered ({FMT_FILES} / {FMT_LINES} / {FMT_TESTS} / {COMPONENT})")

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        mobile.on("pageerror", lambda err: page_errors.append(str(err)))
        mobile.goto(WEB, wait_until="domcontentloaded")
        mobile.wait_for_selector(f"text={FMT_FILES}", timeout=60_000)
        mobile.wait_for_timeout(600)
        mobile.screenshot(
            path=str(SHOTS / "r325-pins-mobile.png"), full_page=False
        )
        print("mobile: the trued pins rendered at 390px")
        browser.close()

    print(f"page errors: {len(page_errors)}")
    print(f"console errors: {len(console_errors)}")
    if page_errors or console_errors:
        for line in page_errors + console_errors:
            print("  !", line[:200])
        return 1
    print("PINS PASS: zero page errors, zero console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
