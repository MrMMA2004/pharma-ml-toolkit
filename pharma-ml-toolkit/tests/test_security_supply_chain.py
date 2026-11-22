"""part13 §17's local half — the offline supply-chain hygiene guard.

The CI-side half rides ``.github/workflows/security-scan.yml`` (pip-audit
over the pinned stack + npm audit over the locked frontend tree + this
guard on GitHub's runners), honouring §17's own rule verbatim: "Do not
add network-dependent security services to the runtime application.
Security scanning may run in CI." This file is the offline half that
runs inside the ordinary pytest suite with zero network: the SBOM's
coverage of the pinned core stack, the exact-pin discipline of
``requirements.txt``, the shipped tree's secret-pattern hygiene (the
strict full-pattern tier — mirroring the ship ritual's leak scan), the
credential quarantine, and the workflow file's own existence + its
CI-only trigger posture.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent
SBOM = REPO / "packaging" / "sbom" / "sbom.cdx.json"
REQUIREMENTS = REPO / "requirements.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "security-scan.yml"

# The strict full-pattern tier (a bare "github_pat_" mention in a doc
# never matches these; the short documented needles stay the ship
# ritual builder's own allowlisted concern).
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".json"}


def _active_pins() -> dict[str, str]:
    """The uncommented ``name==version`` lines of requirements.txt."""
    pins: dict[str, str] = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        name, _, version = stripped.partition("==")
        pins[name.strip().lower().replace("-", "_")] = version.strip()
    return pins


class TestSbomSupplyChain:
    def test_sbom_is_a_valid_cyclonedx_document(self):
        assert SBOM.is_file(), "packaging/sbom/sbom.cdx.json is gone"
        bom = json.loads(SBOM.read_text(encoding="utf-8"))
        assert bom["bomFormat"] == "CycloneDX"
        assert bom["specVersion"].startswith("1.")
        components = bom["components"]
        assert isinstance(components, list) and components
        assert bom.get("metadata"), "SBOM metadata block is missing"

    def test_sbom_covers_the_pinned_core_stack(self):
        bom = json.loads(SBOM.read_text(encoding="utf-8"))
        names = {
            str(c["name"]).lower().replace("-", "_")
            for c in bom["components"]
        }
        missing = set(_active_pins()) - names
        assert not missing, (
            f"SBOM lost pinned core components: {sorted(missing)}"
        )

    def test_requirements_active_lines_are_exact_pins(self):
        for name, version in _active_pins().items():
            assert version, f"unpinned active requirement: {name}"
            assert not version.startswith((">", "<", "=")), (
                f"loose pin in the verified snapshot: {name}=={version}"
            )


class TestTreeHygiene:
    def test_no_secret_patterns_in_the_shipped_tree(self):
        offenders: list[str] = []
        bases = [
            REPO / "src",
            REPO / "tests",
            REPO / "docs",
            REPO / "tools",
            ROOT / ".github" / "workflows",
        ]
        for base in bases:
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if (
                    not path.is_file()
                    or path.suffix not in TEXT_SUFFIXES
                    or path.stat().st_size > 1_000_000
                ):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for pattern in SECRET_PATTERNS:
                    if pattern.search(text):
                        offenders.append(
                            f"{path.relative_to(ROOT)}: {pattern.pattern}"
                        )
        assert not offenders, f"secret-pattern hits: {offenders}"

    def test_credentials_stay_quarantined_outside_the_tree(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".credentials/" in ignore, (
            ".gitignore no longer quarantines .credentials/"
        )
        assert not (REPO / ".credentials").exists()


class TestSecurityScanWorkflow:
    def test_the_workflow_exists_and_declares_its_lanes(self):
        assert WORKFLOW.is_file(), (
            ".github/workflows/security-scan.yml is missing"
        )
        text = WORKFLOW.read_text(encoding="utf-8")
        for token in (
            "pip-audit",
            "npm audit",
            "tests/test_security_supply_chain.py",
            "schedule:",
            "workflow_dispatch:",
        ):
            assert token in text, f"lane missing from the workflow: {token}"

    def test_the_workflow_stays_ci_only(self):
        # §17: scanning may run in CI — never as a runtime service, and
        # never as a per-push blocker of the ordinary CI lane.
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "push:" not in text
        assert "pull_request:" not in text

    def test_the_workflow_never_serves_the_runtime_application(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "uvicorn" not in text
        assert "pharmaml.api" not in text
