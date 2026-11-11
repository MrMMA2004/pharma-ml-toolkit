"""Release restore: hostile ZIPs and fault-injected rollback on isolated trees."""
from __future__ import annotations

import importlib.util
import os
import stat
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "restore_checkpoint", Path(__file__).resolve().parents[1] / "tools/checkpoint.py"
)
cp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cp)


def make_archive(path, extra=(), prefix=""):
    entries = [
        (prefix + "VERSION", "1.2.3\n"),
        (prefix + "pyproject.toml", '[project]\nversion = "1.2.3"\n'),
        (prefix + "src/pharmaml/__init__.py", '__version__ = "1.2.3"\n'),
        (prefix + "a.txt", "new A"),
        (prefix + "new/deep/file.txt", "new file"),
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in entries + list(extra):
            zf.writestr(name, text)
    return path


def prepare_tree(tmp_path):
    root = tmp_path / "target"
    (root / "src/pharmaml").mkdir(parents=True)
    (root / "VERSION").write_text("old version\n")
    (root / "pyproject.toml").write_text("old metadata\n")
    (root / "src/pharmaml/__init__.py").write_text("old package\n")
    (root / "a.txt").write_bytes(b"private previous contents\x00")
    (root / "user_data.csv").write_bytes(b"do not replace\x00")
    (root / "a.txt").chmod(0o640)
    return root


def tree_state(root):
    return {str(p.relative_to(root)): ("dir" if p.is_dir() else (p.read_bytes(), stat.S_IMODE(p.stat().st_mode), p.stat().st_mtime_ns))
            for p in root.rglob("*")}


def test_success_preserves_unrepresented_files_and_both_layouts(tmp_path):
    for prefix, suffix in (("", "root"), ("pharma-ml-toolkit/", "prefixed")):
        base = tmp_path / suffix
        base.mkdir()
        root = prepare_tree(base)
        arc = make_archive(base / "pharma-ml-toolkit-v1.2.3.zip", prefix=prefix)
        assert cp._restore_archive(arc, root) == 0
        assert (root / "VERSION").read_text() == "1.2.3\n"
        assert (root / "a.txt").read_text() == "new A"
        assert (root / "new/deep/file.txt").read_text() == "new file"
        assert (root / "user_data.csv").read_bytes() == b"do not replace\x00"
        assert not list(base.glob(".restore-recovery-*"))


@pytest.mark.parametrize("fault", ["first-install", "later-install", "mkdir", "backup", "verification"])
def test_faults_restore_preexisting_tree_exactly(tmp_path, monkeypatch, fault):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip")
    if fault in ("first-install", "later-install"):
        original = cp._restore_install
        calls = [0]

        def fail_install(staged, target):
            calls[0] += 1
            if fault == "first-install" and calls[0] == 1:
                raise OSError("before first replacement")
            original(staged, target)
            if fault == "later-install" and calls[0] == 4:
                raise OSError("after replacements")
        monkeypatch.setattr(cp, "_restore_install", fail_install)
    elif fault == "mkdir":
        original_mkdir = cp._restore_mkdir

        def fail_mkdir(path):
            if path.name == "deep":
                raise OSError("cannot create directory")
            original_mkdir(path)
        monkeypatch.setattr(cp, "_restore_mkdir", fail_mkdir)
    elif fault == "backup":
        original_backup = cp._backup_restore_file
        calls = [0]

        def fail_backup(source, backup):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError("backup failed")
            original_backup(source, backup)
        monkeypatch.setattr(cp, "_backup_restore_file", fail_backup)
    else:
        def fail_verification(root, expected_hashes, archive_path):
            (root / "a.txt").write_text("corrupted during verification")
            raise OSError("verification failed")
        monkeypatch.setattr(cp, "_verify_restore_tree", fail_verification)
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before
    assert not list(tmp_path.glob(".restore-recovery-*"))


def test_incomplete_rollback_keeps_backups_and_journal(tmp_path, monkeypatch, capsys):
    root = prepare_tree(tmp_path)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip")
    original_replace = os.replace

    def fail_restore(src, dst):
        if "rollback" in Path(src).parts and Path(dst).name == "VERSION":
            raise OSError("restore destination inaccessible")
        original_replace(src, dst)

    def fail_verify(*args):
        raise OSError("injected verification failure")

    monkeypatch.setattr(cp.os, "replace", fail_restore)
    monkeypatch.setattr(cp, "_verify_restore_tree", fail_verify)
    assert cp._restore_archive(arc, root) == 4
    assert "ROLLBACK INCOMPLETE" in capsys.readouterr().out
    recovery, = tmp_path.glob(".restore-recovery-*")
    assert (recovery / "journal.json").is_file()
    assert (recovery / "backups/VERSION").read_text() == "old version\n"
    assert (root / "user_data.csv").read_bytes() == b"do not replace\x00"


@pytest.mark.parametrize("name", ["../evil", "/etc/evil", "C:/evil", "dir/C:evil", "//host/share",
                                  "a/./b", "a//b", "a/../b", "a\\..\\b", "./x",
                                  "dir/con.txt", "name.", "name "])
def test_unsafe_names_are_rejected_before_mutation(tmp_path, name):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip", [(name, "bad")])
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before


@pytest.mark.parametrize("extra", [
    [("a.txt", "duplicate")],
    [("A.TXT", "case collision")],
    [("dir", "file"), ("dir/file", "child")],
    [("caf\u00e9", "one"), ("cafe\u0301", "two")],
    [("dir/", ""), ("DIR/other", "case collision")],
])
def test_collisions_are_rejected(tmp_path, extra):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    with pytest.warns(UserWarning) if extra[0][0] == "a.txt" else nullcontext():
        arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip", extra)
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before


def test_symlinks_in_archive_and_target_are_refused(tmp_path):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip")
    info = zipfile.ZipInfo("bad-link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(arc, "a") as zf:
        zf.writestr(info, "somewhere")
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before
    arc = make_archive(arc)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "new").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("platform does not allow symlink creation")
    assert cp._restore_archive(arc, root) == 4
    assert not list(outside.iterdir())


@pytest.mark.parametrize("limits", [{"max_members": 2}, {"max_file_size": 2},
                                    {"max_total_size": 10}, {"max_ratio": 0.1},
                                    {"max_ratio": float("nan")}])
def test_limits_refuse_before_mutation(tmp_path, limits):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip")
    assert cp._restore_archive(arc, root, **limits) == 4
    assert tree_state(root) == before


def test_metadata_required_even_for_dry_run(tmp_path):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = tmp_path / "pharma-ml-toolkit-v1.2.3.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("VERSION", "1.2.3")
    assert cp._restore_archive(arc, root, dry_run=True) == 4
    assert tree_state(root) == before
    assert not list(tmp_path.glob(".restore-recovery-*"))


@pytest.mark.parametrize("field, value", [
    ("VERSION", "1.2.4\n"),
    ("VERSION", "not-a-version\n"),
    ("pyproject.toml", '[project]\nversion = "1.2.4"\n'),
    ("pyproject.toml", '[tool.other]\nversion = "1.2.3"\n'),
    ("src/pharmaml/__init__.py", '__version__ = "1.2.4"\n'),
    ("src/pharmaml/__init__.py", "value = 123\n"),
    ("src/pharmaml/__init__.py", '__version__ = "1.2.3"\n__version__ = "1.2.4"\n'),
    ("pyproject.toml", '[project]\nversion = "1.2.3"\nversion = "1.2.3"\n'),
])
def test_invalid_version_carriers_are_rejected_for_real_and_dry_run(tmp_path, field, value):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = tmp_path / "pharma-ml-toolkit-v1.2.3.zip"
    entries = {"VERSION": "1.2.3\n", "pyproject.toml": '[project]\nversion = "1.2.3"\n',
               "src/pharmaml/__init__.py": '__version__ = "1.2.3"\n'}
    entries[field] = value
    with zipfile.ZipFile(arc, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    for dry_run in (False, True):
        assert cp._restore_archive(arc, root, dry_run=dry_run) == 4
        assert tree_state(root) == before


def test_crc_corruption_is_rejected(tmp_path):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = tmp_path / "pharma-ml-toolkit-v1.2.3.zip"
    with zipfile.ZipFile(arc, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("VERSION", "1.2.3\n")
        zf.writestr("pyproject.toml", '[project]\nversion = "1.2.3"\n')
        zf.writestr("src/pharmaml/__init__.py", '__version__ = "1.2.3"\n')
        zf.writestr("payload.txt", "unique-payload-for-crc")
    raw = arc.read_bytes().replace(b"unique-payload-for-crc", b"unique-Xayload-for-crc", 1)
    arc.write_bytes(raw)
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before


def test_journal_write_failure_rolls_back(tmp_path, monkeypatch):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip")
    original = cp._write_restore_journal
    calls = [0]

    def fail_once(path, journal):
        calls[0] += 1
        if calls[0] == 4:
            raise OSError("cannot update journal")
        original(path, journal)
    monkeypatch.setattr(cp, "_write_restore_journal", fail_once)
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before


def test_dry_run_never_materialises_missing_tracked_archive(tmp_path, monkeypatch):
    archive = tmp_path / "pharma-ml-toolkit-v1.2.3.zip"
    monkeypatch.setattr(cp, "_newest_zip", lambda: archive)
    monkeypatch.setattr(cp, "_git", lambda *args, **kw: pytest.fail("dry run invoked git"))
    assert cp.restore(dry_run=True) == 4
    assert not archive.exists()


def test_prefixed_root_directory_symlink_is_rejected(tmp_path):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip", prefix="pharma-ml-toolkit/")
    info = zipfile.ZipInfo("pharma-ml-toolkit/")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(arc, "a") as zf:
        zf.writestr(info, "")
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before


def test_directory_members_cannot_contain_data(tmp_path):
    root = prepare_tree(tmp_path)
    before = tree_state(root)
    arc = make_archive(tmp_path / "pharma-ml-toolkit-v1.2.3.zip", [("fake-dir/", "payload")])
    assert cp._restore_archive(arc, root) == 4
    assert tree_state(root) == before
