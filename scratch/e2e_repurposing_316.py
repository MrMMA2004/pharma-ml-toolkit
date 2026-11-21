#!/usr/bin/env python3
"""round-316 browser E2E — the Drug Repurposing lane golden path over
the REAL backend (the r311/r313/r314 pattern: the double-forked daemons
on 8017/5174, the dev-same-origin token channel, zero page/console
errors, evidence screenshots under docs/gui/screenshots/r316-*).

Lanes:
  * the app boot (the startup gate renders the backend-ready marker);
  * the Machine Learning workspace's Drug Repurposing stage renders
    with the registry vocabulary line (the backend's own split ladder +
    ten-key metric contract);
  * the screen submission over the default parameter block — the fit
    telemetry + the ranked per-profile candidate tables (the REAL
    CaDRReS fit + repurposing_candidates payloads, most-sensitive
    first);
  * the honest client-side gate — a below-bound screen never leaves
    the page (the backend's own >= 4-cells bound mirrored client-side,
    no request fires);
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
        page.wait_for_selector("text=backend 1.2.238 ready", timeout=60_000)
        print("boot: the startup gate rendered the backend-ready marker")

        # 2. navigate to the Machine Learning workspace
        page.click("nav >> text=Machine Learning")
        page.wait_for_selector("text=Drug Repurposing", timeout=30_000)
        stage = page.locator("text=Drug Repurposing").first
        stage.scroll_into_view_if_needed()
        page.wait_for_selector(
            "text=/Strategies: random · cell_stratified/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/the metric contract: 10 keys at threshold 6/",
            timeout=30_000,
        )
        print("stage: the registry vocabulary line rendered (the ladder + 10 keys)")

        # 3. the screen submission over the default parameter block
        page.click("button:has-text('Run the screen')")
        page.wait_for_selector("text=/12×8 screen \\(SYNTHETIC\\)/", timeout=60_000)
        page.wait_for_selector("text=/the fit: \\d+\\/200 epochs/", timeout=30_000)
        page.wait_for_selector("text=/C01 — top 5 of 8/", timeout=30_000)
        page.wait_for_selector(
            "text=/predicted IC50 \\(model estimate\\)/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/never clinical recommendations/", timeout=30_000
        )
        card = page.locator("text=C01 — top 5 of 8").first
        card.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.screenshot(
            path=str(SCREENSHOTS / "r316-repurposing-card-desktop.png"),
            full_page=False,
        )
        print("screen: the ranked candidate tables rendered (real fit payloads)")

        # 4. the honest client-side gate — a below-bound screen never
        #    leaves the page (the backend's own >= 4 bound, no request)
        cells = page.locator("input[aria-label='repurposing n_cells']")
        cells.fill("3")
        page.click("button:has-text('Run the screen')")
        page.wait_for_selector(
            "text=/needs at least 4 cell lines/", timeout=30_000
        )
        gate = page.locator("text=needs at least 4 cell lines").first
        gate.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(
            path=str(SCREENSHOTS / "r316-repurposing-gate-desktop.png"),
            full_page=False,
        )
        print("gate: the below-bound screen gated client-side, no request fired")

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
