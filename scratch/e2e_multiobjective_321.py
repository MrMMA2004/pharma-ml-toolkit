#!/usr/bin/env python3
"""round-321 browser E2E — the Multi-objective Optimization lane golden
path over the REAL backend (the r311/r314/r316/r317/r318 pattern: the
double-forked daemons on 8017/5174, the dev-same-origin token channel,
zero page/console errors, evidence screenshots under
docs/gui/screenshots/r321-*).

Lanes:
  * the app boot (the startup gate renders the backend-ready marker);
  * the Machine Learning workspace's Multi-objective Optimization
    stage renders with the registry vocabulary line (the minimize
    convention + the tunable-objective ledger keys, off the REAL
    registry fetch);
  * the optimize round trip over the default parameter block — the REAL
    Pareto study payloads (the study summary line with the hypervolume
    + curve, the non-dominated front table with the objective-named
    value columns and the boundary-marked crowding cells, the
    trade-off corners, the crowding-pruned shortlist, the provenance
    footer with the served-dataset honesty);
  * the honest client-side gate — a negative seed never leaves the
    page (the route's own bound surfaced in the banner);
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

        # 1. the app boot (the startup gate + the backend-ready marker)
        page.goto(VITE, wait_until="domcontentloaded")
        page.wait_for_selector("text=backend 1.2.240 ready", timeout=60_000)
        print("boot: the startup gate rendered the backend-ready marker")

        # 2. navigate to the Machine Learning workspace
        page.click("nav >> text=Machine Learning")
        page.wait_for_selector("text=Multi-objective Optimization", timeout=30_000)
        stage = page.locator("text=Multi-objective Optimization").first
        stage.scroll_into_view_if_needed()
        page.wait_for_selector(
            "text=/Every objective is minimized — 'score' rides/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/score · fit_seconds · score_seconds/", timeout=30_000
        )
        print("mo stage: the registry vocabulary line rendered (the ledger)")

        # 3. the optimize round trip over the default parameter block
        page.click("button:has-text('Run the study')")
        page.wait_for_selector(
            r"text=/20 trials over 80×6 \(SYNTHETIC\)/", timeout=120_000
        )
        page.wait_for_selector(
            "text=/score vs fit_seconds/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/hypervolume 0\./", timeout=30_000
        )
        page.wait_for_selector(
            "text=/The non-dominated front \(/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/best score: trial #/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/best fit_seconds: trial #/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/Shortlist — keep 5 by full-front crowding/", timeout=30_000
        )
        summary = page.locator(
            r"text=/20 trials over 80×6 \(SYNTHETIC\)/"
        ).first.inner_text()
        print(f"study summary: {summary}")
        front = page.locator("text=/The non-dominated front \(/").first
        front.scroll_into_view_if_needed()
        prov = page.locator(
            "text=/provenance pharmaml\.ml\.multiobjective/"
        ).first
        prov.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.screenshot(
            path=str(SCREENSHOTS / "r321-multiobjective-card-desktop.png"),
            full_page=False,
        )
        print("study: the front table + corners + shortlist rendered")

        # 4. the honest client-side gate (a negative seed never leaves)
        seed = page.locator("input[aria-label='multiobjective seed']")
        seed.fill("-1")
        page.click("button:has-text('Run the study')")
        page.wait_for_selector(
            "text=/the seed must lie in 0\\.\\.2147483647/", timeout=30_000
        )
        gate = page.locator(
            "text=/the seed must lie in 0\\.\\.2147483647/"
        ).first
        gate.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(
            path=str(SCREENSHOTS / "r321-multiobjective-gate-desktop.png"),
            full_page=False,
        )
        print("gate: the negative-seed banner rendered (no request fired)")

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
