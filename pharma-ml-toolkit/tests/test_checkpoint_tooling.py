"""The round-260 §4 integrity suite (MASTER_PROMPT part4, Phase 1).

Every defect pinned here was verified live at v1.2.166 before the fix
landed (round 259's read-only audit found all five §4 items):

* §4.1 -- the ship ritual still gated on the retired ``playground.py``
  (the ruff gate fed it a missing file; the GUI self-test ran
  ``playground.py --self-test``); the honest gates are real test
  commands over files that exist, plus a notebook-version consistency
  check (part4's own release checklist).
* §4.2 -- ``restore()`` assumed the ``pharma-ml-toolkit/`` archive
  prefix while every shipped zip since round 210 is ROOT-LEVEL, so the
  rollback-recovery path raised ``ValueError`` on the very archive it
  ships with.  The rewrite handles BOTH layouts, refuses absolute /
  traversal / backslash members, supports ``--dry-run``, stages the
  extraction, and verifies the version before touching the tree.
* §4.3 -- the release builder shipped ``scratch/`` + ``download/`` +
  ``dist-archive/`` (129 evidence PNGs = 8.7 MB = 37% of the archive);
  the canonical build utilities move to ``tools/`` with scratch/
  forwarders keeping the historical invocation paths alive.
* §4.4 -- README said "seventeen" pages (the Studio has eighteen: the
  page list even omitted ``molecular_dynamics``) and pinned the version
  example at 1.2.125; pyproject carried a placeholder homepage.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parents[1]
REPO = TOOLKIT.parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cp = _load_module(TOOLKIT / "tools" / "checkpoint.py", "checkpoint_under_test")


def _make_zip(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return path


# ------------------------------------------------------------------ §4.2


class TestRestoreDualLayout:
    """The recovery path must consume both archive layouts safely."""

    def test_root_level_archive_lands_files(self, tmp_path: Path) -> None:
        # THE round-259 regression: the live v1.2.166 layout raised
        # ValueError('pharma-ml-toolkit' is not in the relative path...)
        # on every member -- --restore was broken against the archive
        # it shipped with.
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {
            "VERSION": "9.9.9\n",
            "src/pharmaml/__init__.py": "__version__ = '9.9.9'\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        rc = cp._restore_archive(arc, target)
        assert rc == 0
        assert (target / "VERSION").read_text() == "9.9.9\n"
        assert (target / "src" / "pharmaml" / "__init__.py").exists()

    def test_prefixed_archive_strips_the_prefix(self, tmp_path: Path) -> None:
        # the pre-round-210 layout (and checkpoint.py's own historical
        # builder) -- every member under pharma-ml-toolkit/.
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.8.zip", {
            "pharma-ml-toolkit/VERSION": "9.9.8\n",
            "pharma-ml-toolkit/src/pharmaml/__init__.py": "x = 1\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        rc = cp._restore_archive(arc, target)
        assert rc == 0
        assert (target / "VERSION").read_text() == "9.9.8\n"
        assert (target / "src" / "pharmaml" / "__init__.py").exists()
        # the prefix must NOT become a literal directory
        assert not (target / "pharma-ml-toolkit").exists()

    def test_mixed_prefix_layout_refused(self, tmp_path: Path) -> None:
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.7.zip", {
            "pharma-ml-toolkit/VERSION": "9.9.7\n",
            "src/stray.py": "x = 1\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        assert cp._restore_archive(arc, target) == 4
        assert not (target / "VERSION").exists()

    def test_absolute_member_refused(self, tmp_path: Path) -> None:
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {
            "VERSION": "9.9.9\n",
            "/etc/pharmaml-evil": "no\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        assert cp._restore_archive(arc, target) == 4
        assert not Path("/etc/pharmaml-evil").exists()

    def test_traversal_member_refused(self, tmp_path: Path) -> None:
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {
            "VERSION": "9.9.9\n",
            "src/../../../../evil.py": "no\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        assert cp._restore_archive(arc, target) == 4
        assert not (tmp_path / "evil.py").exists()

    def test_backslash_member_refused(self, tmp_path: Path) -> None:
        # a Windows-style separator would bypass a PurePosixPath ".."
        # check and traverse on win32 -- refused outright on every
        # platform (part4 §4.2: reject path-traversal entries).
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {
            "VERSION": "9.9.9\n",
            "src\\..\\..\\evil.py": "no\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        assert cp._restore_archive(arc, target) == 4

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {
            "VERSION": "9.9.9\n",
            "src/pharmaml/__init__.py": "x = 1\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        assert cp._restore_archive(arc, target, dry_run=True) == 0
        assert list(target.iterdir()) == []  # preview only, zero writes

    def test_staged_version_mismatch_refused(self, tmp_path: Path) -> None:
        # part4 §4.2: verify the restored project version -- an archive
        # whose VERSION member disagrees with its own filename is
        # malformed and must never reach the tree.
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {
            "VERSION": "1.0.0\n",  # lies about itself
            "src/pharmaml/__init__.py": "x = 1\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        assert cp._restore_archive(arc, target) == 4
        assert not (target / "VERSION").exists()

    def test_user_data_survives(self, tmp_path: Path) -> None:
        # part4 §4.2: preserve existing user data -- restore overwrites
        # archive members only, it NEVER deletes anything else.
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {
            "VERSION": "9.9.9\n",
        })
        target = tmp_path / "tree"
        target.mkdir()
        (target / "data").mkdir()
        keeper = target / "data" / "user_experiments.csv"
        keeper.write_text("smiles,ic50\nCCO,1.5\n")
        assert cp._restore_archive(arc, target) == 0
        assert keeper.read_text().startswith("smiles,ic50")

    def test_empty_archive_refused(self, tmp_path: Path) -> None:
        arc = _make_zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", {})
        target = tmp_path / "tree"
        target.mkdir()
        assert cp._restore_archive(arc, target) == 4


# ------------------------------------------------------------------ §4.1


class TestShipGates:
    """The ship ritual must run real, supported commands (§4.1)."""

    def test_commands_reference_real_files(self) -> None:
        cmds = cp._ship_gate_commands()
        assert cmds, "the ship ritual must have gates"
        flat = " ".join(arg for _, command in cmds for arg in command)
        # the retired Tkinter entry point must be gone from every gate
        assert "playground" not in flat, flat
        # no invented studio.py CLI either (part4 §4.1: do not invent a
        # command the application does not implement -- studio.py has
        # no argparse surface; the GUI gate is the offscreen suite)
        assert "--self-test" not in flat
        # every path argument the gates name must exist on the tree
        for label, command in cmds:
            for arg in command:
                if arg.startswith("-") or "/" not in arg and arg not in {"src", "tests", "tools", "ui"}:
                    continue
                if arg in {sys.executable, "pytest", "ruff", "mypy", "run"}:
                    continue
                assert (TOOLKIT / arg).exists(), (label, arg)

    def test_gui_gate_is_the_offscreen_suite(self) -> None:
        flat = {label: command for label, command in cp._ship_gate_commands()}
        gui = [command for label, command in flat.items() if "gui" in label]
        assert gui, "a GUI gate must exist"
        assert any("tests/ui" in arg for command in gui for arg in command)

    def test_notebook_freshness_green_on_the_live_tree(self) -> None:
        # the release checklist item: notebook version consistency
        assert cp._notebook_freshness_problems() == []

    def test_notebook_freshness_catches_stale_prints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "VERSION").write_text("9.9.9\n")
        examples = tmp_path / "examples"
        examples.mkdir()
        stale = {
            "cells": [{
                "cell_type": "code",
                "outputs": [{"output_type": "stream", "text": "pharmaml 1.2.99\n"}],
            }],
            "metadata": {},
        }
        (examples / "demo.ipynb").write_text(json.dumps(stale))
        monkeypatch.setattr(cp, "ROOT", tmp_path)
        problems = cp._notebook_freshness_problems()
        assert any("1.2.99" in line for line in problems)


# ------------------------------------------------------------------ §4.3


class TestZipPolicy:
    """Release archives ship the toolkit, not the research history."""

    def test_checkpoint_builder_excludes_history_dirs(self) -> None:
        assert {"scratch", "download", "dist-archive"} <= cp.ZIP_EXCLUDE_DIRS

    def test_ritual_builder_excludes_history_dirs(self) -> None:
        bz = _load_module(TOOLKIT / "tools" / "build_zip.py", "build_zip_under_test")
        assert {"scratch", "download", "dist-archive"} <= bz.EXCLUDE_DIRS

    def test_checkpoint_builder_writes_root_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # the two builders must agree on layout: root-level since the
        # round-210 four-home ritual (restore handles both, but the
        # shipped layout is root-level and checkpoint's own builder may
        # no longer drift back to the prefixed one).
        fake = tmp_path / "tk"
        (fake / "src").mkdir(parents=True)
        (fake / "src" / "a.py").write_text("x = 1\n")
        (fake / "scratch").mkdir()
        (fake / "scratch" / "probe.py").write_text("p = 1\n")
        (fake / "download").mkdir()
        (fake / "download" / "evidence.png").write_bytes(b"\x89PNG\r\n")
        (fake / "VERSION").write_text("9.9.9\n")
        monkeypatch.setattr(cp, "ROOT", fake)
        monkeypatch.setattr(cp, "REPO", tmp_path)
        monkeypatch.setattr(cp, "DOWNLOAD", tmp_path / "download-out")
        out = cp._build_zip("9.9.9")
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "src/a.py" in names
        assert "VERSION" in names
        assert not any(n.startswith("scratch/") for n in names)
        assert not any(n.startswith("download/") for n in names)
        assert not any(n.startswith("pharma-ml-toolkit/") for n in names)
        # the staging dir outside every home is cleaned up
        assert not (tmp_path / ".zip-staging-9.9.9").exists()

    def test_canonical_utilities_live_in_tools(self) -> None:
        # part4 §4.3: canonical build utilities move out of scratch/
        for name in ("build_zip.py", "reexec_notebooks.py", "run_suite_chunks.py",
                     "venv_repair.sh", "scrub_home_paths.py"):
            assert (TOOLKIT / "tools" / name).is_file(), name

    def test_scratch_forwarders_keep_the_historical_paths(self) -> None:
        # the cron ritual + muscle memory invoke the scratch/ paths;
        # the shims must forward to the canonical tools/ copies.
        forwarding = {
            "build_zip_210.py": "build_zip.py",
            "reexec_notebooks_209.py": "reexec_notebooks.py",
            "run_suite_chunks_225.py": "run_suite_chunks.py",
            "run_suite_chunks_250.py": "run_suite_chunks.py",
            "venv_repair.sh": "venv_repair.sh",
        }
        for shim, target in forwarding.items():
            text = (TOOLKIT / "scratch" / shim).read_text(encoding="utf-8")
            assert target in text, (shim, "must name its tools/ target")
            assert "tools" in text, (shim, "must point at tools/")


# ------------------------------------------------------------------ §4.4


class TestMetadataDrift:
    """README + pyproject must describe the real project (§4.4)."""

    def test_readme_counts_eighteen_pages(self) -> None:
        text = (TOOLKIT / "README.md").read_text(encoding="utf-8")
        assert "seventeen" not in text.lower()
        assert "eighteen" in text.lower()
        # the 18th page rides the documented page list
        assert "molecular_dynamics" in text

    def test_readme_version_example_not_pinned_to_the_past(self) -> None:
        text = (TOOLKIT / "README.md").read_text(encoding="utf-8")
        assert "(1.2.125)" not in text

    def test_pyproject_homepage_is_the_real_remote(self) -> None:
        text = (TOOLKIT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'Homepage\s*=\s*"([^"]+)"', text)
        assert match, "pyproject must declare a project Homepage"
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(REPO), capture_output=True, text=True, check=False,
        ).stdout.strip()
        assert remote, "the origin remote must exist (it is the survivor)"
        assert match.group(1) in remote.replace(".git", ""), (
            match.group(1), remote,
        )
