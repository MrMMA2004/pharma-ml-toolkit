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
        self.write(
            "pharma-ml-toolkit/README.md",
            "# Installation\n# Testing\n"
            "Not currently published on PyPI. From the package root run "
            "`python -m pip install -c requirements.txt -e '.[chem]'`, "
            "`python -m pip install -c requirements.txt -e '.[deep]'`, and "
            "`python -m pip install -c requirements.txt -e '.[reports]'`.\n",
        )
        self.write("pharma-ml-toolkit/frontend/README.md", "# Frontend\n")
        self.write(
            "src/components/explorer/download-panel.tsx",
            "const command = `python -m pip install -c requirements.txt -e '${name}'`;\n"
            'name: ".[chem]" name: ".[docking]" name: ".[boosting]" '
            'name: ".[deep]" name: ".[reports]" name: ".[selfies]" name: ".[explain]"\n'
            "PharmaML Studio is a React + TypeScript application hosted by Tauri 2. "
            "This Next.js site is a separate web console. The retired PySide6 launcher "
            "and the .[studio] compatibility extra do not install or launch the current desktop app.\n",
        )
        self.write(
            "pharma-ml-toolkit/docs/RESEARCH_WORKFLOWS.md",
            "# Workflows\n\n"
            "The supported desktop is the React + TypeScript/Tauri 2 application. "
            "UI walkthroughs describe only the retired PySide6 implementation. "
            "They do not imply feature parity.\n",
        )
        self.write("pharma-ml-toolkit/CHANGELOG.md", "# Changelog\n")
        for name in ("CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md"):
            self.write(name, "# Document\n")
        self.write("download/pharma-ml-toolkit-v1.2.172.zip", "test")
        self.png("pharma-ml-toolkit/docs/assets/github-social-preview.png", 1280, 640)
        self.png("pharma-ml-toolkit/docs/gui/screenshots/r311-boot-introduction-desktop.png", 1280, 800)
        self.write(
            "pharma-ml-toolkit/docs/assets/github-hero.svg",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 340"/>',
        )
        self.readme = (
            '# Home\n'
            '<img src="pharma-ml-toolkit/docs/assets/github-hero.svg" alt="Hero">\n'
            '[Changelog](pharma-ml-toolkit/CHANGELOG.md)\n'
            '[Install](pharma-ml-toolkit/README.md#installation)\n'
            '[![Source version](https://img.shields.io/badge/source%20version-1.2.172-087f6a)](pharma-ml-toolkit/VERSION)\n'
            '**Source version in this checkout:** `v1.2.172`\n'
            'This is not clinical decision support. It is not currently published on PyPI.\n'
            'The supported desktop is React + Tauri 2. The separate Next.js console is distinct.\n'
            'The independent clean-machine check is not recorded.\n'
            'Round 311 browser capture; not evidence of a packaged Windows launch.\n'
            '<img src="pharma-ml-toolkit/docs/gui/screenshots/r311-boot-introduction-desktop.png" alt="Studio browser preview">\n'
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
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">II", width, height)
        )

    def validate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            check.main()

    def test_unversioned_citation_is_valid(self):
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

    def test_readme_source_version_must_match_checkout(self):
        stale = self.readme.replace("v1.2.172", "v1.2.171")
        self.write("README.md", stale)
        with self.assertRaisesRegex(ValueError, "README source version is 1.2.171"):
            self.validate()

    def test_stale_source_version_badge_is_rejected(self):
        stale = self.readme.replace("source%20version-1.2.172-", "source%20version-1.2.171-")
        self.write("README.md", stale)
        with self.assertRaisesRegex(ValueError, "source-version badge"):
            self.validate()

    def test_studio_capture_must_exist_and_be_large_enough(self):
        path = self.root / "pharma-ml-toolkit/docs/gui/screenshots/r311-boot-introduction-desktop.png"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Studio capture is missing"):
            self.validate()
        self.png("pharma-ml-toolkit/docs/gui/screenshots/r311-boot-introduction-desktop.png", 400, 300)
        with self.assertRaisesRegex(ValueError, "Studio capture is too small"):
            self.validate()

    def test_screenshot_must_be_labeled_as_a_browser_preview(self):
        self.write("README.md", self.readme.replace("Round 311 browser capture", "Current app"))
        with self.assertRaisesRegex(ValueError, "round 311 browser capture"):
            self.validate()
        self.write("README.md", self.readme.replace("not evidence of a packaged Windows launch", "verified installer"))
        with self.assertRaisesRegex(ValueError, "distinguish the browser capture"):
            self.validate()

    def test_research_and_interface_status_claims_are_required(self):
        self.write("README.md", self.readme.replace("not clinical decision support", "clinical decision support"))
        with self.assertRaisesRegex(ValueError, "research/clinical limitation"):
            self.validate()
        self.write("README.md", self.readme.replace("not recorded", "verified"))
        with self.assertRaisesRegex(ValueError, "clean-machine release caveat"):
            self.validate()

    def test_retired_launcher_cannot_be_presented_as_a_command(self):
        self.write("README.md", self.readme + "\n```bash\npython studio.py\n```\n")
        with self.assertRaisesRegex(ValueError, "must not present the retired"):
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
        self.write("README.md", self.readme.replace(' alt="Studio browser preview"', ""))
        with self.assertRaisesRegex(ValueError, "HTML images must have nonempty alternative text"):
            self.validate()
        self.write("README.md", self.readme.replace('alt="Studio browser preview"', 'alt="  "'))
        with self.assertRaisesRegex(ValueError, "HTML images must have nonempty alternative text"):
            self.validate()
        self.write("README.md", self.readme.replace('alt="Studio browser preview"', 'data-alt="Not an alternative"'))
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

    def test_social_preview_must_have_github_dimensions(self):
        self.png("pharma-ml-toolkit/docs/assets/github-social-preview.png", 1200, 600)
        with self.assertRaisesRegex(ValueError, "exactly 1280x640"):
            self.validate()

    def test_link_checker_ignores_code_examples_but_checks_real_links(self):
        examples = (
            "\n`[inline example](missing-inline.md)`\n"
            "```markdown\n[code example](missing-code.md)\n```\n"
        )
        self.write("README.md", self.readme + examples)
        self.validate()
        self.write("README.md", self.readme + "\n[broken](missing-real.md)\n")
        with self.assertRaisesRegex(ValueError, "missing local link.*missing-real.md"):
            self.validate()

    def test_package_install_guidance_is_source_local(self):
        path = "pharma-ml-toolkit/README.md"
        manual = (self.root / path).read_text(encoding="utf-8")
        self.validate()
        self.write(path, manual.replace("-c requirements.txt -e '.[chem]'", "-e '.[chem]'"))
        with self.assertRaisesRegex(ValueError, "constrained source-local"):
            self.validate()
        self.write(path, manual.replace("Not currently published on PyPI", "Published on PyPI"))
        with self.assertRaisesRegex(ValueError, "not currently published on PyPI"):
            self.validate()
        self.write(path, manual + "\nExample: pip install 'pharmaml[chem]'\n")
        with self.assertRaisesRegex(ValueError, "must not present pharmaml extras"):
            self.validate()

    def test_web_console_install_guidance_uses_local_extras_and_app_boundaries(self):
        path = "src/components/explorer/download-panel.tsx"
        panel = (self.root / path).read_text(encoding="utf-8")
        self.validate()
        self.write(
            path,
            panel.replace(
                "python -m pip install -c requirements.txt -e '${name}'",
                "pip install pharmaml[chem]",
            ),
        )
        with self.assertRaisesRegex(ValueError, "constrain source-local installs"):
            self.validate()
        self.write(path, panel.replace("Next.js site is a separate web console", "desktop application"))
        with self.assertRaisesRegex(ValueError, "separate Next.js console"):
            self.validate()
        self.write(path, panel)
        self.write(
            "src/components/explorer/playground/feature.tsx",
            'Use the retired interface with `python studio.py`.\n',
        )
        with self.assertRaisesRegex(ValueError, "must not present the retired launcher"):
            self.validate()

    def test_workflow_guide_marks_legacy_ui_and_no_parity(self):
        path = "pharma-ml-toolkit/docs/RESEARCH_WORKFLOWS.md"
        guide = (self.root / path).read_text(encoding="utf-8")
        self.write(path, guide.replace("do not imply feature parity", ""))
        with self.assertRaisesRegex(ValueError, "must distinguish Python/CLI coverage"):
            self.validate()
        self.write(path, guide + "\n## Data quality in Studio\n")
        with self.assertRaisesRegex(ValueError, "must not label retired PySide6"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
