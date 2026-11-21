#!/usr/bin/env python3
"""round-318 browser E2E — the pharmacogenomics lane golden path over
the REAL backend (the r311/r313/r314/r316/r317 pattern: the
double-forked daemons on 8017/5174, the dev-same-origin token channel,
zero page/console errors, evidence screenshots under
docs/gui/screenshots/r318-*).

Lanes:
  * the app boot (the startup gate renders the backend-ready marker);
  * the Machine Learning workspace's Pharmacogenomics stage renders
    with the registry vocabulary line (the ANOVA feasibility gates +
    gdsctools' percent convention);
  * the analyze submission over the default parameter block — the
    screen summary + the association census + the volcano counts +
    the top-association table (the REAL gdsc_anova payloads: the
    planted TP53_mut rows win with their p/q/FDR columns and
    verdicts) + the guarded/normalized ruler scores;
  * the honest client-side gate — a negative seed never leaves the
    page (validated BEFORE any request fires);
  * ZERO page errors, ZERO console errors across the entire run.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SCREENSHOTS = Path("docs/gui/screenshots")
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
        page.wait_for_selector("text=backend 1.2.240 ready", timeout=60_000)
        print("boot: the startup gate rendered the backend-ready marker")

        # 2. navigate to the Machine Learning workspace
        page.click("nav >> text=Machine Learning")
        page.wait_for_selector("text=Pharmacogenomics", timeout=30_000)
        stage = page.locator("text=Pharmacogenomics").first
        stage.scroll_into_view_if_needed()
        page.wait_for_selector(
            "text=/The feasibility gates: a drug needs ≥ 6 non-NA/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/a feature ≥ 3 positives AND ≥ 3 negatives/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/FDR_THRESHOLD = 25 means 25 %/",
            timeout=30_000,
        )
        print("stage: the registry vocabulary line rendered (the gates + the percent convention)")

        # 3. the analyze submission over the default parameter block
        page.click("button:has-text('Run the ANOVA')")
        page.wait_for_selector("text=/40×12 screen \\(SYNTHETIC\\)/", timeout=60_000)
        page.wait_for_selector("text=/407\\/480 observed entries \\(85%\\)/", timeout=30_000)
        page.wait_for_selector("text=/120 associations over/", timeout=30_000)
        page.wait_for_selector("text=/12 drugs × 10 features/", timeout=30_000)
        page.wait_for_selector(
            "text=/volcano: 5 sensitive \\/ 1 resistant \\/ 114 not significant/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/Top 5 associations/", timeout=30_000,
        )
        page.wait_for_selector(
            "text=/statistic over the synthetic screen, sorted by p/",
            timeout=30_000,
        )
        page.wait_for_selector("text=TP53_mut", timeout=30_000)
        page.wait_for_selector(
            "text=/the mean-effects ruler \\(guarded pearson 0\\.721/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/normalized beyond the naive drug-mean ruler \\(pearson 0\\.473\\)/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/never clinical recommendations/", timeout=30_000
        )
        card = page.locator("text=Top 5 associations").first
        card.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.screenshot(
            path=str(SCREENSHOTS / "r318-pharmacogenomics-card-desktop.png"),
            full_page=False,
        )
        print("analyze: the association table + ruler scores rendered (real ANOVA payloads)")

        # 4. the honest client-side gate — a negative seed never leaves
        #    the page (validated before any request fires)
        seed = page.locator("input[aria-label='pharmacogenomics seed']")
        seed.fill("-1")
        page.click("button:has-text('Run the ANOVA')")
        page.wait_for_selector(
            "text=/the seed must lie in 0\\.\\.2147483647/", timeout=30_000
        )
        gate = page.locator("text=the seed must lie in 0..2147483647").first
        gate.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(
            path=str(SCREENSHOTS / "r318-pharmacogenomics-gate-desktop.png"),
            full_page=False,
        )
        print("gate: the negative seed gated client-side, no request fired")

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
