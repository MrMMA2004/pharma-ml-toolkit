#!/usr/bin/env python3
"""round-314 browser E2E — the ADMET lane golden path over the REAL
backend (the r311/r313 pattern: the double-forked daemons on 8017/5174,
the dev-same-origin token channel, zero page/console errors, evidence
screenshots under docs/gui/screenshots/r314-*).

Lanes:
  * the app boot (the startup gate renders the backend-ready marker);
  * the Machine Learning workspace's ADMET Profile stage renders with
    the registry vocabulary line (the backend's own 52/6 numbers);
  * the score submission over the default aspirin+caffeine batch —
    the per-molecule cards with the physchem tables + the radial bars
    (the REAL summarize_molecule payloads);
  * the honest invalid-row verdict — an invalid SMILES line reports
    itself per-row (never dropped, never a 500);
  * ZERO page errors, ZERO console errors across the entire run.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SCREENSHOTS = Path("/home/z/my-project/pharma-ml-toolkit/docs/gui/screenshots")
VITE = "http://localhost:5174"


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

        # 1. the app boot (the startup gate + the nine-workspace registry)
        page.goto(VITE, wait_until="domcontentloaded")
        page.wait_for_selector("text=backend 1.2.236 ready", timeout=60_000)
        print("boot: the startup gate rendered the backend-ready marker")

        # 2. navigate to the Machine Learning workspace
        page.click("nav >> text=Machine Learning")
        page.wait_for_selector("text=ADMET Profile", timeout=30_000)
        admet_stage = page.locator("text=ADMET Profile").first
        admet_stage.scroll_into_view_if_needed()
        page.wait_for_selector(
            "text=/Registry: 52 endpoints across 6 categories/",
            timeout=30_000,
        )
        print("admet stage: the registry vocabulary line rendered (52/6)")

        # 3. the score submission over the default batch (aspirin + caffeine)
        page.click("button:has-text('Score ADMET')")
        page.wait_for_selector("text=/2 valid · 0 invalid/", timeout=60_000)
        page.wait_for_selector("text=Blood-Brain Barrier Safe", timeout=30_000)
        mw = page.locator("text=MW (Da)").first
        mw.scroll_into_view_if_needed()
        card = page.locator("text=CC(=O)Oc1ccccc1C(=O)O").first
        card.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.screenshot(
            path=str(SCREENSHOTS / "r314-admet-card-desktop.png"),
            full_page=False,
        )
        print("score: the per-molecule cards rendered (aspirin + caffeine)")

        # 4. the honest invalid-row verdict
        textarea = page.locator("textarea[aria-label='ADMET SMILES batch']")
        textarea.fill("CC(=O)Oc1ccccc1C(=O)O\nnot-a-molecule")
        page.click("button:has-text('Score ADMET')")
        page.wait_for_selector(
            "text=/1 valid · 1 invalid/", timeout=60_000
        )
        page.wait_for_selector(
            "text=invalid SMILES — reported per-row", timeout=30_000
        )
        invalid_card = page.locator("text=not-a-molecule").first
        invalid_card.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(
            path=str(SCREENSHOTS / "r314-admet-invalid-desktop.png"),
            full_page=False,
        )
        print("invalid row: the per-row verdict rendered honestly")

        browser.close()

    print(f"page errors: {len(page_errors)}")
    print(f"console errors: {len(console_errors)}")
    if page_errors or console_errors:
        for line in page_errors + console_errors:
            print("  !", line[:200])
        return 1
    print("E2E PASS: zero page errors, zero console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
