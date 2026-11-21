#!/usr/bin/env python3
"""round-315 browser E2E — the DTA lane golden path over the REAL
backend (the r311/r313/r314 pattern: the double-forked daemons on
8017/5174, the dev-same-origin token channel, zero page/console
errors, evidence screenshots under docs/gui/screenshots/r315-*).

Lanes:
  * the app boot (the startup gate renders the backend-ready marker
    at 1.2.237);
  * the Machine Learning workspace's "Drug–Target Interaction" stage
    renders with the registry vocabulary line (the backend's own
    25/64/62 + 2048+23→2071 arithmetic);
  * the featurization submission over the default three-pair batch —
    the per-pair cards (the kept badges, the affinity values, the top
    AAC components, the residue counts) over the REAL
    paired_dta_features payloads;
  * the honest invalid-row verdict — an invalid SMILES pair reports
    itself per-row (dropped together with its affinity, never silent);
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
        page.wait_for_selector("text=backend 1.2.237 ready", timeout=60_000)
        print("boot: the startup gate rendered the backend-ready marker")

        # 2. navigate to the Machine Learning workspace's DTA stage
        page.click("nav >> text=Machine Learning")
        page.wait_for_selector("text=Drug–Target Interaction", timeout=30_000)
        dta_stage = page.locator("text=Drug–Target Interaction").first
        dta_stage.scroll_into_view_if_needed()
        page.wait_for_selector(
            "text=/Alphabets: 25 protein symbols/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/64 isomeric \\/ 62 canonical SMILES symbols/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/2048-bit ECFP \\+ 23 AAC columns → 2071 features/",
            timeout=30_000,
        )
        print("dta stage: the registry vocabulary line rendered (25/64/62, 2071)")

        # 3. the featurization over the default three-pair batch
        page.click("button:has-text('Featurize pairs')")
        page.wait_for_selector(
            "text=/3 pairs → 3 kept · 0 dropped/", timeout=60_000
        )
        page.wait_for_selector(
            "text=/the matrix: 2071 features = 2048 ECFP \\+ 23/",
            timeout=30_000,
        )
        # the per-pair cards: kept badges, residues, top AAC components
        page.wait_for_selector("text=/22 residues/", timeout=30_000)
        kept_card = page.locator("text=top AAC components").first
        kept_card.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.screenshot(
            path=str(SCREENSHOTS / "r315-dta-card-desktop.png"),
            full_page=False,
        )
        print("features: the per-pair cards rendered (3 kept, AAC panels)")

        # 4. the honest invalid-row verdict
        textarea = page.locator("textarea[aria-label='DTA pairs batch']")
        textarea.fill(
            "CCO,MKTAYIAKQRQISFVKSHFSRQ,5.0\n"
            "not-a-molecule,MKTAYIAKQRQISFVKSHFSRQ,7.0"
        )
        page.click("button:has-text('Featurize pairs')")
        page.wait_for_selector(
            "text=/2 pairs → 1 kept · 1 dropped/", timeout=60_000
        )
        page.wait_for_selector(
            "text=/invalid SMILES — dropped together with its protein/",
            timeout=30_000,
        )
        dropped_card = page.locator("text=not-a-molecule").first
        dropped_card.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(
            path=str(SCREENSHOTS / "r315-dta-invalid-desktop.png"),
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
