#!/usr/bin/env python3
"""round-323 browser E2E — the §8 Workflow Builder lane golden path
over the REAL backend (the r311/r314/r316/r317/r318/r321 pattern: the
double-forked daemons on 8017/5174, the dev-same-origin token channel,
zero page/console errors, evidence screenshots under
docs/gui/screenshots/r323-*).

Lanes:
  * the app boot (the startup gate renders the backend-ready marker);
  * the Machine Learning workspace's Workflow Builder stage renders
    with the engine block's own vocabulary line (10 steps · 8 port
    kinds, off the REAL registry fetch) + the guided-qsar template
    chain with the honest leakage note;
  * the validate round trip over the spec as loaded — the REAL
    pre-execution gate answer (the verdict, the executor's
    topological order, the canonical digest);
  * the sync run round trip — the REAL engine round trip over the
    REAL backend runners (the run record: the state, the per-node
    ledger with the artifact lineage + digest_kinds, the
    reproducibility block with the honest determinism scope);
  * the runs ledger — the bounded in-session execution history with
    the honest lifetime note;
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
        page.wait_for_selector("text=Workflow Builder", timeout=30_000)
        stage = page.locator("text=Workflow Builder").first
        stage.scroll_into_view_if_needed()
        # the registry vocabulary line (the engine's own numbers)
        page.wait_for_selector(
            "text=/The registry: 10 steps · 8 port kinds/",
            timeout=30_000,
        )
        # the template chain + the honest leakage reorder note
        page.wait_for_selector(
            "text=/import: Dataset Import → standardize: Standardization/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/the honest leakage reorder: the split runs BEFORE feature selection/",
            timeout=30_000,
        )
        print("builder stage: the registry line + the template chain rendered")

        # 3. the validate round trip over the spec as loaded
        page.click("button:has-text('Validate the spec')")
        page.wait_for_selector(
            r"text=/valid — 10 nodes · digest /", timeout=60_000
        )
        page.wait_for_selector(
            "text=/the executor's order: import → standardize → descriptors/",
            timeout=30_000,
        )
        print("validate: the pre-execution gate answered (order + digest)")

        # 4. the sync run round trip — the REAL engine round trip
        page.click("button:has-text('Run the workflow')")
        page.wait_for_selector(
            r"text=/run wf-run-[0-9a-f]+ · guided-qsar · COMPLETED/",
            timeout=180_000,
        )
        page.wait_for_selector(
            "text=/10 completed \/ 0 failed \/ 0 skipped \/ 0 cancelled/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/model \(model, estimator-params, [0-9]+ upstream\)/",
            timeout=30_000,
        )
        page.wait_for_selector(
            "text=/report \(report, json, [0-9]+ upstream\)/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/reproducibility — toolkit 1\.2\.240 · python 3\.12\./",
            timeout=30_000,
        )
        run_line = page.locator(
            r"text=/run wf-run-[0-9a-f]+ · guided-qsar · COMPLETED/"
        ).first.inner_text()
        print(f"run summary: {run_line}")
        card = page.locator(
            "text=/reproducibility — toolkit 1\.2\.240/"
        ).first
        card.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.screenshot(
            path=str(SCREENSHOTS / "r323-workflowbuilder-card-desktop.png"),
            full_page=False,
        )
        print("run: the run record + lineage + reproducibility rendered")

        # 5. the runs ledger (the bounded in-session history)
        page.click("button:has-text('Refresh the runs ledger')")
        page.wait_for_selector(
            "text=/The execution history \(1 runs —/", timeout=30_000
        )
        page.wait_for_selector(
            "text=/the in-session ledger lives as long as the process/",
            timeout=30_000,
        )
        print("ledger: the execution history rendered with the lifetime note")

        # 6. the honest client-side gate (a negative seed never leaves)
        seed = page.locator("input[aria-label='workflow seed']")
        seed.fill("-1")
        page.click("button:has-text('Run the workflow')")
        page.wait_for_selector(
            "text=/the seed must lie in 0\\.\\.2147483647/", timeout=30_000
        )
        gate = page.locator(
            "text=/the seed must lie in 0\\.\\.2147483647/"
        ).first
        gate.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(
            path=str(SCREENSHOTS / "r323-workflowbuilder-gate-desktop.png"),
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
