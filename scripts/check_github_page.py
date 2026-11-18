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


def check_local_links(path: Path) -> int:
    """Check relative Markdown and inline-HTML links, including image sources."""
    text = path.read_text(encoding="utf-8")
    refs = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text)
    refs += re.findall(r'(?:src|href)="([^"]+)"', text)
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
    if "](pharma-ml-toolkit/CHANGELOG.md)" not in text:
        fail("README must link the source changelog")
    capture = re.search(
        r'<img\s+[^>]*src="(pharma-ml-toolkit/docs/assets/web-console-v(\d+\.\d+\.\d+)\.png)"',
        text,
    )
    if not capture:
        fail("README must embed a versioned application capture")
    screenshot = ROOT / capture.group(1)
    captured_version = capture.group(2)
    if not screenshot.is_file():
        fail(f"application capture is missing: {screenshot.relative_to(ROOT)}")
    if tuple(map(int, captured_version.split("."))) > tuple(map(int, version.split("."))):
        fail("application capture is newer than the checkout")
    if captured_version != version and not re.search(
        rf"historical\s+v{re.escape(captured_version)}\s+capture", text, re.IGNORECASE
    ):
        fail(f"older capture v{captured_version} must be labeled historical")
    archive = ROOT / f"download/pharma-ml-toolkit-v{version}.zip"
    if not archive.is_file():
        fail(f"current release archive is missing: {archive.relative_to(ROOT)}")

    hero_path = "pharma-ml-toolkit/docs/assets/github-hero.svg"
    if not re.search(rf'<img\s+[^>]*src="{re.escape(hero_path)}"[^>]*alt="[^"\s][^"]*"', text):
        fail("README must embed the hero illustration with alternative text")
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
        fail(f"application capture is too small for the README: {capture_size}")
    print(f"images: application v{captured_version} {capture_size[0]}x{capture_size[1]}, social 1280x640")

    documents = [readme, ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md", ROOT / "CODE_OF_CONDUCT.md"]
    total = sum(check_local_links(path) for path in documents)
    print(f"local links: {total} across {len(documents)} community documents")

    manual = (ROOT / "pharma-ml-toolkit/README.md").read_text(encoding="utf-8")
    for section in ("Installation", "Testing"):
        if not re.search(rf"^#+\s+{section}\b", manual, re.IGNORECASE | re.MULTILINE):
            fail(f"nested user guide is missing its {section} section")
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
