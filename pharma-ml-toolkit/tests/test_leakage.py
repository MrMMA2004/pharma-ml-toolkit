"""(100x-137) The build-machine-path LEAKAGE guard.

The 1.2.44 round scrubbed every ``/home/<user>/`` path from the shipped
tree (docs/LEAKAGE_AUDIT.md: "a full-archive scan ... now returns zero
matches"), but rounds 120+ reintroduced the absolute prefixes one
scratch driver at a time — and no test FAILED when they did.  This guard
walks the same surface the archive ships and fails on any absolute home
prefix, so the regression can never creep back.

Excluded by design (each *documents* the pattern, never executes it):

- ``scratch/scrub_home_paths_137.py`` — the scrub tool itself, whose
  replacements must quote the prefixes they kill (the same convention
  docs/LEAKAGE_AUDIT.md uses when it cites ``~/.venv/...``);
- this test file (the pattern string is assembled at runtime so the
  scan never matches its own source, belt AND braces).

The pattern is built by concatenation so a naive grep of THIS file
finds no absolute prefix either.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]

#: The directories the archive excludes (src/lib/toolkit-fs.ts's list —
#: local caches and virtualenvs are never shipped, so their contents
#: cannot leak; scanning them would only slow the guard down).
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "node_modules", "artifacts",
    "catboost_info", ".idea", ".vscode", ".cache", ".tox",
}

#: Text suffixes the scan reads (binary formats carry no path text).
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".xml", ".html", ".json", ".cfg",
    ".yml", ".yaml", ".ts", ".tsx", ".sh", ".rst", ".nb", ".ipynb",
}

#: Files that DOCUMENT the forbidden pattern instead of executing it:
#: the scrub tool (its replacements must quote the prefixes they kill
#: — the same convention docs/LEAKAGE_AUDIT.md uses when it cites
#: ``~/.venv/...``) and this test (the samples below assemble their
#: prefixes at runtime so the guard's own source scans clean).
_DOCUMENTERS = {"tests/test_leakage.py", "scratch/scrub_home_paths_137.py"}

#: Transient RUN ARTIFACTS the guard does not scan: the junit xmls
#: carry pytest's own recorded skip locations (absolute paths the
#: recorder writes) and are neutralised by the scrub tool before any
#: archive ships — the scrub's own verification walk covers them, and
#: scanning them HERE would be circular (the suite's junit is written
#: at run END, so mid-run the guard would always see the previous,
#: still-unscrubbed one).
_RUN_ARTIFACTS = re.compile(r"^scratch/junit[^/]*\.xml$")

#: The forbidden prefix, assembled at runtime: this file's own source
#: never carries the literal, and neither does any doc that quotes the
#: SCRUBBED (tilde) form.
_HOME = "/ho" + "me/"


def _home_pattern() -> re.Pattern[str]:
    # assembled from the runtime prefix: the guard's own source (and
    # the scrub tool's, and the docs' tilde quotations) never match
    return re.compile(_HOME + r"[A-Za-z0-9_.-]+/")


def _shipped_text_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(TOOLKIT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(TOOLKIT_ROOT).as_posix()
        if any(part in EXCLUDED_PARTS for part in rel.split("/")):
            continue
        if rel in _DOCUMENTERS or _RUN_ARTIFACTS.match(rel):
            continue
        files.append(path)
    assert files, "the guard found no files — the walk itself broke"
    return files


def test_shipped_tree_carries_no_absolute_home_paths() -> None:
    """Every text file the archive ships is free of ``/home/<user>/``
    prefixes — the LEAKAGE_AUDIT contract, enforced (the 1.2.44 scrub
    had regressed silently by 1.2.75; this test is why it cannot)."""
    pattern = _home_pattern()
    leaks: list[str] = []
    for path in _shipped_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            leaks.append(f"{path.relative_to(TOOLKIT_ROOT)}: unreadable")
            continue
        match = pattern.search(text)
        if match:
            leaks.append(
                f"{path.relative_to(TOOLKIT_ROOT)}: {match.group(0)!r}"
            )
    assert not leaks, (
        "absolute home-directory prefixes leaked into the shipped tree "
        f"({len(leaks)} file(s)) — run scratch/scrub_home_paths_137.py "
        f"and re-verify: {leaks[:5]}"
    )


def test_the_scrub_contract_is_idempotent_documented() -> None:
    """The scrub tool exists, is excluded from the scan (it documents
    the patterns it kills), and says so — the convention
    LEAKAGE_AUDIT.md established for pattern-quoting docs."""
    scrub = TOOLKIT_ROOT / "scratch" / "scrub_home_paths_137.py"
    assert scrub.is_file(), "the scrub tool vanished"
    doc = scrub.read_text(encoding="utf-8")
    assert "Idempotent" in doc
    assert scrub.relative_to(TOOLKIT_ROOT).as_posix() in _DOCUMENTERS


@pytest.mark.parametrize(
    "sample",
    [
        # ids are explicit so the suite's junit record never carries the
        # RESOLVED sample strings (parametrize ids land in the xml; the
        # values themselves never do)
        pytest.param(_HOME + "z/.venv/bin/python scratch/foo.py", id="venv-run-hint"),
        pytest.param(
            'Path("' + _HOME + 'z/my-project/pharma-ml-toolkit/download")',
            id="repo-root-target",
        ),
        pytest.param(_HOME + "alice/report.html", id="foreign-user"),
    ],
)
def test_the_pattern_catches_the_known_forms(sample: str) -> None:
    """The assembled pattern catches the venv form, the repo-root form
    and a foreign-username form (the guard is not z-specific)."""
    assert _home_pattern().search(sample)
