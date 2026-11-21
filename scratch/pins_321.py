#!/usr/bin/env python3
"""round-321 pins pair — the web console page render at the terminal
fixed point (1,680 files / 563,278 lines / 8,484 tests / 102 component
tests / v1.2.240): the live /api/toolkit/tree walk agrees with the
toolkit-stats pins, the page renders them, and the pair rides the
terminal build (the placeholders closed the walk first — the fixed-point
discipline; the r317/r318 convention)."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SCREENSHOTS = Path("/home/z/my-project/pharma-ml-toolkit/docs/gui/screenshots")
PAGE = "http://localhost:3000/"


def main() -> int:
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        # desktop 1440x900
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.goto(PAGE, wait_until="domcontentloaded")
        page.wait_for_selector("text=1,680", timeout=60_000)
        page.wait_for_selector("text=563,278", timeout=30_000)
        page.wait_for_selector("text=102", timeout=30_000)
        page.wait_for_selector("text=8,484", timeout=30_000)
        page.wait_for_timeout(900)
        page.screenshot(
            path=str(SCREENSHOTS / "r321-pins-desktop.png"), full_page=False
        )
        desktop_text = page.locator("text=1,680").first.inner_text()
        print(f"pins desktop: the hero stats rendered ({desktop_text.strip()})")
        page.close()

        # mobile 390x844
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        mobile.on("pageerror", lambda err: page_errors.append(str(err)))
        mobile.goto(PAGE, wait_until="domcontentloaded")
        mobile.wait_for_selector("text=1,680", timeout=60_000)
        mobile.wait_for_selector("text=563,278", timeout=30_000)
        mobile.wait_for_timeout(900)
        mobile.screenshot(
            path=str(SCREENSHOTS / "r321-pins-mobile.png"), full_page=False
        )
        print("pins mobile: the hero stats rendered at 390px")
        mobile.close()
        browser.close()

    print(f"page errors: {len(page_errors)}")
    print(f"console errors: {len(console_errors)}")
    if page_errors or console_errors:
        for line in page_errors + console_errors:
            print("  !", line[:200])
        return 1
    print("PINS PAIR PASS: zero page errors, zero console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
