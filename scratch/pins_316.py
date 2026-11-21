#!/usr/bin/env python3
"""round-316 pins pair -- the terminal three-way render evidence
(the walk = the live API = the page render; desktop + 390px mobile,
zero page/console errors), captured over the REAL Next.js web app."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SCREENSHOTS = Path("/home/z/my-project/pharma-ml-toolkit/docs/gui/screenshots")
WEB = "http://127.0.0.1:3000"


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
        page.wait_for_selector("text=1,657", timeout=60_000)
        page.wait_for_selector("text=559,328", timeout=60_000)
        page.wait_for_selector("text=8,438", timeout=60_000)
        body = page.inner_text("body")
        for needle in ("1,657", "559,328", "8,438", "78", "1.2.238"):
            if needle not in body:
                print(f"PIN MISSING: {needle}")
                return 1
        print("pins render: 1,657 files · 559,328 lines · 8,438 tests · 78 component · v1.2.238")
        page.wait_for_timeout(1800)
        page.screenshot(path=str(SCREENSHOTS / "r316-pins-desktop.png"))
        # the 390px mobile baseline: zero horizontal overflow
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        mobile.on("pageerror", lambda err: page_errors.append(str(err)))
        mobile.goto(WEB, wait_until="domcontentloaded")
        mobile.wait_for_selector("text=1,657", timeout=60_000)
        overflow = mobile.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        print(f"mobile 390px overflow: {overflow}")
        if overflow > 0:
            print("MOBILE OVERFLOW -- refusing the baseline")
            return 1
        mobile.screenshot(path=str(SCREENSHOTS / "r316-pins-mobile.png"))
        browser.close()

    print(f"page errors: {len(page_errors)}")
    # the Next.js dev server's own HMR websocket handshake failures
    # (_next/webpack-hmr) are dev-mode infrastructure noise, not app
    # errors -- filtered explicitly, never silently
    hmr_noise = [e for e in console_errors if "_next/webpack-hmr" in e]
    app_console = [e for e in console_errors if "_next/webpack-hmr" not in e]
    print(f"console errors (app): {len(app_console)}")
    print(f"console errors (dev HMR channel, filtered): {len(hmr_noise)}")
    if page_errors or app_console:
        for line in page_errors + app_console:
            print("  !", line[:200])
        return 1
    print("PINS PASS: zero page errors, zero app console errors, zero mobile overflow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
