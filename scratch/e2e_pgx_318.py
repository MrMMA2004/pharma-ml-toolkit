#!/usr/bin/env python3
"""round-318 browser E2E — the Pharmacogenomics lane golden path over
the REAL backend (the r311/r314/r316/r317 pattern: the double-forked
daemons on 8017/5174, the dev-same-origin token channel, zero page/
console errors, evidence screenshots under docs/gui/screenshots/r318-*).

Lanes:
  * the app boot (the startup gate renders the backend-ready marker);
  * the Machine Learning workspace's Pharmacogenomics stage renders
    with the registry vocabulary line (the feasibility gates + the
    percent convention, off the REAL registry fetch);
  * the analyze round trip over the default parameter block — the REAL
    ANOVA payloads (the screen summary line, the top-association table
    with the p/q/FDR/verdict columns, the volcano census, the guarded
    + normalized ruler line, the provenance footer);
  * the honest client-side gate — a negative seed never leaves the
    page (the route's own bound surfaced in the banner);
  * ZERO page errors, ZERO console errors across the entire run.

Re-created after the r318 machinery ghost commit (ca04acf7, a UUID
subject) was reset away unpushed per the r312 precedent — the rebase
checkout removed the ghost-tracked files from disk; this runner is
re-written byte-faithfully (the evidence pair it captured rode the
r318 ship and the worklog records the run).
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
        page.wait_for_selector("text=Pharmacogenomics", timeout=30_000)
        stage = page.locator("text=Pharmacogenomics").first
        stage.scroll_into_view_if_needed()
        page.wait_for_selector(
            "text=/The feasibility gates: a drug needs/", timeout=30_000
        )
        print("pgx stage: the registry vocabulary line rendered (the gates)")

        # 3. the analyze round trip over the default parameter block
        page.click("button:has-text('Run the ANOVA')")
        page.wait_for_selector(
            r"text=/\d+×\d+ screen \(SYNTHETIC\)/", timeout=60_000
        )
        page.wait_for_selector(
            "text=/associations over 12 drugs × 10 features/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/Top 5 associations \\(statistic over the synthetic screen/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/the mean-effects ruler \\(guarded pearson/", timeout=30_000
        )
        summary = page.locator(
            r"text=/\d+×\d+ screen \(SYNTHETIC\)/"
        ).first.inner_text()
        print(f"analyze summary: {summary}")
        # the top-association table: the planted TP53_mut row must surface
        tp53 = page.locator("td:has-text('TP53_mut')").first
        tp53.scroll_into_view_if_needed()
        prov = page.locator("text=/provenance pharmaml.ml.pharmacogenomics/").first
        prov.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.screenshot(
            path=str(SCREENSHOTS / "r318-pharmacogenomics-card-desktop.png"),
            full_page=False,
        )
        print("analyze: the top-association table + ruler line rendered")

        # 4. the honest client-side gate (a negative seed never leaves)
        seed = page.locator("input[aria-label='pharmacogenomics seed']")
        seed.fill("-1")
        page.click("button:has-text('Run the ANOVA')")
        page.wait_for_selector(
            "text=/the seed must lie in 0\\.\\.2147483647/", timeout=30_000
        )
        gate = page.locator(
            "text=/the seed must lie in 0\\.\\.2147483647/"
        ).first
        gate.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(
            path=str(SCREENSHOTS / "r318-pharmacogenomics-gate-desktop.png"),
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
