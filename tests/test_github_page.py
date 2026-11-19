"""Regression tests for the offline GitHub landing-page validator."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/check_github_page.py"
spec = importlib.util.spec_from_file_location("github_page_check", MODULE_PATH)
assert spec and spec.loader
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class GitHubPageValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("pharma-ml-toolkit/VERSION", "1.2.172\n")
        self.write("pharma-ml-toolkit/pyproject.toml", 'version = "1.2.172"\n')
        self.write("pharma-ml-toolkit/src/pharmaml/__init__.py", '__version__ = "1.2.172"\n')
        self.write("src/lib/toolkit-version.ts", 'SHIPPED_TOOLKIT_VERSION = "1.2.172"\n')
        self.write("CITATION.cff", 'cff-version: 1.2.0\ntitle: "PharmaML"\n')
        self.write("pharma-ml-toolkit/README.md", "# Installation\n# Testing\n")
        self.write("pharma-ml-toolkit/CHANGELOG.md", "# Changelog\n")
        for name in ("CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md"):
            self.write(name, "# Document\n")
        self.write("download/pharma-ml-toolkit-v1.2.172.zip", "test")
        self.png("pharma-ml-toolkit/docs/assets/github-social-preview.png", 1280, 640)
        self.png("pharma-ml-toolkit/docs/assets/web-console-v1.2.161.png", 1280, 800)
        self.write("pharma-ml-toolkit/docs/assets/github-hero.svg",
                   '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 340"/>')
        self.readme = (
            '# Home\n<img src="pharma-ml-toolkit/docs/assets/github-hero.svg" alt="Hero">\n'
            '[Changelog](pharma-ml-toolkit/CHANGELOG.md)\n'
            '[Install](pharma-ml-toolkit/README.md#installation)\n'
            'Historical v1.2.161 capture\n'
            '<img src="pharma-ml-toolkit/docs/assets/web-console-v1.2.161.png" alt="Historical web console">\n'
        )
        self.write("README.md", self.readme)
        self.root_patch = patch.object(check, "ROOT", self.root)
        self.version_patch = patch.object(check, "VERSION_FILE", self.root / "pharma-ml-toolkit/VERSION")
        self.root_patch.start()
        self.version_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.version_patch.stop)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def png(self, name, width, height):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height))

    def validate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            check.main()

    def test_unversioned_citation_and_historical_capture_are_valid(self):
        self.validate()
        self.write("CITATION.cff", 'version: "1.2.172"\n')
        self.validate()

    def test_stale_versioned_citation_is_rejected(self):
        self.write("CITATION.cff", "version: 1.2.161\n")
        with self.assertRaisesRegex(ValueError, "CITATION.cff has 1.2.161"):
            self.validate()

    def test_citation_version_cannot_be_blank(self):
        self.write("CITATION.cff", "version: \n")
        with self.assertRaisesRegex(ValueError, "CITATION.cff has"):
            self.validate()

    def test_current_capture_requires_no_historical_label(self):
        self.png("pharma-ml-toolkit/docs/assets/web-console-v1.2.172.png", 1280, 800)
        current = self.readme.replace("Historical v1.2.161 capture", "Current capture")
        current = current.replace("web-console-v1.2.161.png", "web-console-v1.2.172.png")
        self.write("README.md", current)
        self.validate()

    def test_future_capture_is_rejected(self):
        self.png("pharma-ml-toolkit/docs/assets/web-console-v1.2.173.png", 1280, 800)
        future = self.readme.replace("web-console-v1.2.161.png", "web-console-v1.2.173.png")
        self.write("README.md", future)
        with self.assertRaisesRegex(ValueError, "newer than the checkout"):
            self.validate()

    def test_old_capture_must_be_honestly_labeled(self):
        self.write("README.md", self.readme.replace("Historical v1.2.161 capture", "Current capture"))
        with self.assertRaisesRegex(ValueError, "must be labeled historical"):
            self.validate()

    def test_missing_and_too_small_capture_fail(self):
        self.png("pharma-ml-toolkit/docs/assets/web-console-v1.2.161.png", 400, 300)
        with self.assertRaisesRegex(ValueError, "application capture is too small"):
            self.validate()
        (self.root / "pharma-ml-toolkit/docs/assets/web-console-v1.2.161.png").unlink()
        with self.assertRaisesRegex(ValueError, "application capture is missing"):
            self.validate()

    def test_invalid_or_inaccessible_hero_is_rejected(self):
        self.write("README.md", self.readme.replace('alt="Hero"', 'alt=""'))
        with self.assertRaisesRegex(ValueError, "alternative text"):
            self.validate()
        self.write("README.md", self.readme)
        self.write("pharma-ml-toolkit/docs/assets/github-hero.svg", "<svg>")
        with self.assertRaisesRegex(ValueError, "invalid SVG"):
            self.validate()

    def test_all_screenshot_images_require_alternative_text(self):
        self.write("README.md", self.readme.replace(' alt="Historical web console"', ""))
        with self.assertRaisesRegex(ValueError, "HTML images must have nonempty alternative text"):
            self.validate()
        self.write("README.md", self.readme.replace('alt="Historical web console"', 'alt="  "'))
        with self.assertRaisesRegex(ValueError, "HTML images must have nonempty alternative text"):
            self.validate()
        self.write("README.md", self.readme.replace('alt="Historical web console"', 'data-alt="Not an alternative"'))
        with self.assertRaisesRegex(ValueError, "HTML images must have nonempty alternative text"):
            self.validate()
        self.write("README.md", self.readme + '<img src="pharma-ml-toolkit/docs/assets/github-social-preview.png">\n')
        with self.assertRaisesRegex(ValueError, "HTML images must have nonempty alternative text"):
            self.validate()

    def test_bad_cross_document_anchor_is_rejected(self):
        self.write("README.md", self.readme.replace("#installation", "#missing"))
        with self.assertRaisesRegex(ValueError, "missing heading anchor"):
            self.validate()

    def test_version_carriers_and_archive_are_still_checked(self):
        self.write("src/lib/toolkit-version.ts", 'SHIPPED_TOOLKIT_VERSION = "1.2.171"\n')
        with self.assertRaisesRegex(ValueError, "expected 1.2.172"):
            self.validate()
        self.write("src/lib/toolkit-version.ts", 'SHIPPED_TOOLKIT_VERSION = "1.2.172"\n')
        (self.root / "download/pharma-ml-toolkit-v1.2.172.zip").unlink()
        with self.assertRaisesRegex(ValueError, "current release archive is missing"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
