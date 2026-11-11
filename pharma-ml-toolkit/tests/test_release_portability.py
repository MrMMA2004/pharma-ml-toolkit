"""Exercise the shipped files without scratch, Git, or the companion web tree."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_docs_generator_matches_shipped_index() -> None:
    assert _load_tool("build_docs_index").render_index() == (ROOT / "docs/INDEX.md").read_text()


def test_standalone_archive_profile(tmp_path: Path) -> None:
    # Build an isolated *test* ZIP using the release file selection, not the
    # destructive four-home publishing ritual. Never touch real release copies.
    builder = _load_tool("build_zip")
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in builder.iter_files():
            zf.write(path, path.relative_to(ROOT).as_posix())
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        assert not any(n.startswith(("scratch/", "download/", "dist-archive/")) for n in names)
        for name in ("build_example_data.py", "build_docs_index.py", "strip_notebook_outputs.py"):
            assert f"tools/{name}" in names
        zf.extractall(standalone)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    # Prevent a surrounding checkout from supplying Git metadata to this test.
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=-q --tb=short"]
    checkout_tests = [
        "tests/test_checkpoint_tooling.py::TestZipPolicy::test_scratch_forwarders_keep_the_historical_paths",
        "tests/test_checkpoint_tooling.py::TestMetadataDrift::test_pyproject_homepage_is_the_real_remote",
    ]
    full = subprocess.run(base + checkout_tests, cwd=standalone, env=env, text=True, capture_output=True, check=False)
    assert full.returncode == 1, full.stdout + full.stderr
    assert "2 failed" in full.stdout, full.stdout + full.stderr
    result = subprocess.run(
        base + ["--standalone-release", "tests/test_data_provenance.py", "tests/test_checkpoint_tooling.py"],
        cwd=standalone, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 skipped" in result.stdout, result.stdout + result.stderr
