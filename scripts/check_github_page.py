#!/usr/bin/env python3
"""Validate the repository landing page and citation metadata without dependencies.

Run from anywhere in the checkout::

    python scripts/check_github_page.py

This intentionally checks release *carriers*, not the latest GitHub API state:
CI must remain deterministic and work without network access or credentials.
"""

from __future__ import annotations

import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "pharma-ml-toolkit/VERSION"


def fail(message: str) -> None:
    raise ValueError(message)


def png_size(path: Path) -> tuple[int, int]:
    """Read PNG dimensions from the IHDR header without Pillow."""
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        fail(f"not a valid PNG with an IHDR header: {path.relative_to(ROOT)}")
    return struct.unpack(">II", data[16:24])


def version_from(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        fail(f"version carrier not found in {path.relative_to(ROOT)}")
    return match.group(1)


def _without_markdown_code(text: str) -> str:
    """Remove fenced and inline code so examples cannot masquerade as links."""
    visible: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence_char is None:
            opening = re.match(r"(`{3,}|~{3,})", stripped)
            if opening:
                fence_char = opening.group(1)[0]
                fence_length = len(opening.group(1))
                continue
            visible.append(line)
            continue
        closing = re.match(rf"{re.escape(fence_char)}{{{fence_length},}}\s*$", stripped)
        if closing:
            fence_char = None
            fence_length = 0
    outside_fences = "\n".join(visible)
    return re.sub(r"(?<!`)`+[^`\n]*`+(?!`)", "", outside_fences)


def check_local_links(path: Path) -> int:
    """Check relative Markdown and inline-HTML links, including image sources."""
    text = path.read_text(encoding="utf-8")
    link_text = _without_markdown_code(text)
    refs = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", link_text)
    refs += re.findall(r'(?:src|href)="([^"]+)"', link_text)
    checked = 0
    for ref in refs:
        if ref.startswith(("http://", "https://", "mailto:")):
            continue
        if ref.startswith("#"):
            anchor = ref[1:]
            headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
            anchors = [
                re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
                for heading in headings
            ]
            if anchor not in anchors:
                fail(f"missing heading anchor {ref} in {path.relative_to(ROOT)}")
            continue
        target, _, fragment = ref.partition("#")
        if not target:
            continue
        resolved = path.parent / target
        if not resolved.exists():
            fail(f"missing local link in {path.relative_to(ROOT)}: {ref}")
        if fragment and resolved.suffix.lower() == ".md":
            headings = re.findall(
                r"^#{1,6}\s+(.+)$", resolved.read_text(encoding="utf-8"), re.MULTILINE
            )
            anchors = [
                re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
                for heading in headings
            ]
            if fragment not in anchors:
                fail(f"missing heading anchor in {path.relative_to(ROOT)}: {ref}")
        checked += 1
    return checked


def check_image_alternatives(text: str) -> None:
    """Require readable alternatives on the page's meaningful HTML images.

    Markdown badges carry their own alt text. Parse HTML attributes rather
    than mistaking a data-alt attribute for an actual image alternative.
    """
    class ImageParser(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "img":
                alt = dict(attrs).get("alt")
                if alt is None or not alt.strip():
                    fail("README HTML images must have nonempty alternative text")

    parser = ImageParser()
    parser.feed(text)
    parser.close()


def main() -> None:
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        fail(f"unexpected release version: {version!r}")

    carriers = {
        ROOT / "pharma-ml-toolkit/pyproject.toml": r'^version\s*=\s*"([^"]+)"',
        ROOT / "pharma-ml-toolkit/src/pharmaml/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
        ROOT / "src/lib/toolkit-version.ts": r'SHIPPED_TOOLKIT_VERSION\s*=\s*"([^"]+)"',
    }
    for path, pattern in carriers.items():
        actual = version_from(path, pattern)
        if actual != version:
            fail(f"{path.relative_to(ROOT)} has {actual}, expected {version}")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r"^version:[ \t]*([^\n]*)", citation, re.MULTILINE)
    if citation_version:
        cited = citation_version.group(1).split("#", 1)[0].strip().strip("\"'")
        if cited != version:
            fail(f"CITATION.cff has {cited}, expected {version}")
    print(f"release carriers: {version} (VERSION, package and web; citation version optional)")

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    required_claims = {
        "research/clinical limitation": r"not clinical decision support",
        "source-install status": r"not currently published on PyPI",
        "supported desktop interface": r"React \+ Tauri 2",
        "separate Next.js console": r"separate Next\.js",
        "clean-machine release caveat": r"not recorded",
    }
    for label, pattern in required_claims.items():
        if not re.search(pattern, text, re.IGNORECASE):
            fail(f"README must state the {label}")
    if re.search(r"(?m)^\s*python\s+(?:legacy/)?studio\.py\b", text):
        fail("README must not present the retired PySide6 launcher as a command")
    if "](pharma-ml-toolkit/CHANGELOG.md)" not in text:
        fail("README must link the source changelog")
    source_version = re.search(
        r"\*\*Source version in this checkout:\*\*\s+`v?(\d+\.\d+\.\d+)`", text
    )
    if not source_version or source_version.group(1) != version:
        reported = source_version.group(1) if source_version else "missing"
        fail(f"README source version is {reported}, expected {version}")
    if f"source%20version-{version}-" not in text:
        fail("README source-version badge does not match the checkout")

    capture_path = "pharma-ml-toolkit/docs/gui/screenshots/r311-boot-introduction-desktop.png"
    if capture_path not in text:
        fail("README must include the labeled PharmaML Studio interface capture")
    if not re.search(r"round 311 browser capture", text, re.IGNORECASE):
        fail("README must label the Studio screenshot as a round 311 browser capture")
    if not re.search(r"not evidence of a packaged Windows launch", text, re.IGNORECASE):
        fail("README must distinguish the browser capture from packaged-install evidence")
    screenshot = ROOT / capture_path
    if not screenshot.is_file():
        fail(f"Studio capture is missing: {screenshot.relative_to(ROOT)}")

    archive = ROOT / f"download/pharma-ml-toolkit-v{version}.zip"
    if not archive.is_file():
        fail(f"current release archive is missing: {archive.relative_to(ROOT)}")

    hero_path = "pharma-ml-toolkit/docs/assets/github-hero.svg"
    if not re.search(rf'<img\s+[^>]*src="{re.escape(hero_path)}"[^>]*alt="[^"\s][^"]*"', text):
        fail("README must embed the hero illustration with alternative text")
    check_image_alternatives(text)
    try:
        hero = ET.parse(ROOT / hero_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"hero illustration is invalid SVG: {exc}") from exc
    if hero.tag != "{http://www.w3.org/2000/svg}svg" or not hero.get("viewBox"):
        fail("hero illustration must be a responsive SVG with a viewBox")

    social = ROOT / "pharma-ml-toolkit/docs/assets/github-social-preview.png"
    if png_size(social) != (1280, 640):
        fail("GitHub social preview must be exactly 1280x640")
    capture_size = png_size(screenshot)
    if capture_size[0] < 1000 or capture_size[1] < 600:
        fail(f"Studio capture is too small for the README: {capture_size}")
    print(f"images: Studio browser capture {capture_size[0]}x{capture_size[1]}, social 1280x640")

    manual_path = ROOT / "pharma-ml-toolkit/README.md"
    frontend_readme = ROOT / "pharma-ml-toolkit/frontend/README.md"
    workflow_guide = ROOT / "pharma-ml-toolkit/docs/RESEARCH_WORKFLOWS.md"
    documents = [
        readme,
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        ROOT / "CODE_OF_CONDUCT.md",
        manual_path,
        frontend_readme,
        workflow_guide,
    ]
    total = sum(check_local_links(path) for path in documents)
    print(f"local links: {total} across {len(documents)} community and user documents")

    manual = manual_path.read_text(encoding="utf-8")
    for section in ("Installation", "Testing"):
        if not re.search(rf"^#+\s+{section}\b", manual, re.IGNORECASE | re.MULTILINE):
            fail(f"nested user guide is missing its {section} section")
    if not re.search(r"not\s+currently\s+published\s+on\s+PyPI", manual, re.IGNORECASE):
        fail("package guide must disclose that PharmaML is not currently published on PyPI")
    for extra in ("chem", "deep", "reports"):
        if f"python -m pip install -c requirements.txt -e '.[{extra}]'" not in manual:
            fail(f"package guide must include a constrained source-local .[{extra}] installation example")
    if re.search(r"pip install\s+['\"]?pharmaml\[", manual):
        fail("package guide must not present pharmaml extras as PyPI install targets")

    web_install_path = ROOT / "src/components/explorer/download-panel.tsx"
    web_install = web_install_path.read_text(encoding="utf-8")
    if "python -m pip install -c requirements.txt -e '${name}'" not in web_install:
        fail("web-console extra chips must constrain source-local installs to the pinned core")
    for extra in ("chem", "docking", "boosting", "deep", "reports", "selfies", "explain"):
        if f'name: ".[{extra}]"' not in web_install:
            fail(f"web-console install guide must list the local .[{extra}] extra")
    if not re.search(r"React \+ TypeScript application hosted by Tauri 2", web_install):
        fail("web-console install guide must identify the React/Tauri desktop app")
    if "Next.js site is a separate web console" not in web_install:
        fail("web-console install guide must identify the separate Next.js console")
    if "retired PySide6 launcher" not in web_install or "do not install or launch the current desktop app" not in web_install:
        fail("web-console install guide must mark the PySide6 launcher as retired")
    playground_sources = ROOT / "src/components/explorer/playground"
    for source in playground_sources.rglob("*.tsx"):
        if re.search(
            r"(?m)\bpython\s+(?:legacy/)?studio\.py\b",
            source.read_text(encoding="utf-8"),
        ):
            fail(
                "web-console card must not present the retired launcher as current: "
                f"{source.relative_to(ROOT)}"
            )

    workflow_text = workflow_guide.read_text(encoding="utf-8")
    if not re.search(r"supported desktop is the React\s*\+\s*TypeScript/Tauri 2", workflow_text, re.IGNORECASE):
        fail("workflow guide must identify the supported React/Tauri desktop")
    if not re.search(r"retired PySide6 implementation", workflow_text, re.IGNORECASE):
        fail("workflow guide must label historical PySide6 interface instructions")
    if not re.search(r"do not imply feature parity", workflow_text, re.IGNORECASE):
        fail("workflow guide must distinguish Python/CLI coverage from current app scope")
    if re.search(r"(?m)^## .*\bin Studio\b", workflow_text, re.IGNORECASE):
        fail("workflow guide must not label retired PySide6 procedures as current Studio sections")
    print("GitHub page validation passed")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        message = f"GitHub page check failed: {exc}"
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::error title=GitHub page validation::{message}", file=sys.stderr)
            sys.exit(1)
        sys.exit(message)
