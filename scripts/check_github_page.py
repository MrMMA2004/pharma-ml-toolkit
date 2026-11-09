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
        target = ref.split("#", 1)[0]
        if not target:
            continue
        resolved = path.parent / target
        if not resolved.exists():
            fail(f"missing local link in {path.relative_to(ROOT)}: {ref}")
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
        ROOT / "CITATION.cff": r"^version:\s*(\S+)",
    }
    for path, pattern in carriers.items():
        actual = version_from(path, pattern)
        if actual != version:
            fail(f"{path.relative_to(ROOT)} has {actual}, expected {version}")
    print(f"release carriers: {version} (VERSION, package, web and CITATION.cff)")

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    expected_record = f"The [v{version} release record](pharma-ml-toolkit/CHANGELOG.md)"
    if expected_record not in text:
        fail(f"README release evidence is not linked to v{version}")
    screenshot = ROOT / f"pharma-ml-toolkit/docs/assets/web-console-v{version}.png"
    if not screenshot.is_file():
        fail(f"current application capture is missing: {screenshot.relative_to(ROOT)}")
    if f"pharma-ml-toolkit/docs/assets/web-console-v{version}.png" not in text:
        fail("README does not embed the current application capture")
    archive = ROOT / f"download/pharma-ml-toolkit-v{version}.zip"
    if not archive.is_file():
        fail(f"current release archive is missing: {archive.relative_to(ROOT)}")

    social = ROOT / "pharma-ml-toolkit/docs/assets/github-social-preview.png"
    if png_size(social) != (1280, 640):
        fail("GitHub social preview must be exactly 1280x640")
    capture_size = png_size(screenshot)
    if capture_size[0] < 1000 or capture_size[1] < 600:
        fail(f"application capture is too small for the README: {capture_size}")
    print(f"images: application {capture_size[0]}x{capture_size[1]}, social 1280x640")

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
