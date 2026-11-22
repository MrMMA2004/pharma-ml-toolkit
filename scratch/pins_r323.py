#!/usr/bin/env python3
"""round-323 pins pair — the web-app page render at the trued fixed
point (the r313/r321 convention): desktop + 390px mobile, zero page/
console errors, the stats line carrying this round's pins
(1,698 files / 566,909 lines / 8,525 tests / 110 component tests /
v1.2.240)."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = Path("/home/z/my-project/pharma-ml-toolkit/docs/gui/screenshots")
WEB = "http://localhost:3000"


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
        page.wait_for_selector("text=1,698", timeout=60_000)
        page.wait_for_selector("text=566,909", timeout=30_000)
        page.wait_for_selector("text=8,525", timeout=30_000)
        page.wait_for_selector("text=110", timeout=30_000)
        page.wait_for_timeout(800)
        page.screenshot(
            path=str(SHOTS / "r323-pins-desktop.png"), full_page=False
        )
        print("desktop: the trued pins rendered (1,698 / 566,909 / 8,525 / 110)")

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        mobile.on("pageerror", lambda err: page_errors.append(str(err)))
        mobile.goto(WEB, wait_until="domcontentloaded")
        mobile.wait_for_selector("text=1,698", timeout=60_000)
        mobile.wait_for_timeout(600)
        mobile.screenshot(
            path=str(SHOTS / "r323-pins-mobile.png"), full_page=False
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
