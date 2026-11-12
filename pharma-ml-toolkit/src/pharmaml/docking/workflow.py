"""Docking execution workflows (AutoDock4 and Vina).

Upgrades ``others/Docking/main.py``, the batch-run conventions, and
``others/Vina/vinar.py`` / ``vinar_multiple.py`` (config-driven Vina
batch docking).

Design changes vs. the originals:

* No ``os.chdir``: each run gets an explicit working directory passed
  to subprocess via ``cwd`` -- thread/process safe.
* No ``shell=True``: commands are argument lists, immune to shell
  injection and paths with spaces/unicode.
* Availability checks (``shutil.which``) with actionable errors before
  burning hours of queue time.
* Idempotency: existing output files are skipped unless overwritten.
* Parallelism via ``ThreadPoolExecutor`` instead of spawning one CMD
  console per ligand and polling ``.done`` signal files.

Examples
--------
>>> from pathlib import Path
>>> from pharmaml.docking.workflow import DockingRunResult
>>> DockingRunResult(Path("/tmp"), True, None, 1.5)   # a healthy single run
DockingRunResult(directory=PosixPath('/tmp'), ok=True, log_file=None, seconds=1.5, error=None, skipped=False)

>>> from pharmaml.docking.workflow import parse_vina_config
>>> conf = Path("/tmp/pharmaml_doctest_vina.conf")
>>> _ = conf.write_text(
...     "# Vina config\\n"
...     "receptor = r.pdbqt\\n"
...     "\\n"
...     "center_x = 1.5   # angstrom\\n"
...     "size_x = 20\\n"
...     "exhaustiveness = 8\\n"
... )
>>> parse_vina_config(conf)   # comments/blanks stripped, values stay strings
{'receptor': 'r.pdbqt', 'center_x': '1.5', 'size_x': '20', 'exhaustiveness': '8'}
>>> conf.unlink()
>>> parse_vina_config(conf)
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: config file not found: /tmp/pharmaml_doctest_vina.conf

>>> from pharmaml.docking.workflow import run_autodock4
>>> run_autodock4("/tmp/pharmaml_doctest_no_such_dir")
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: docking directory not found: /tmp/pharmaml_doctest_no_such_dir
>>> work = Path("/tmp/pharmaml_doctest_run")
>>> work.mkdir()
>>> run_autodock4(work)    # no r.gpf -> typed error before anything executes
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: missing r.gpf in /tmp/pharmaml_doctest_run
>>> _ = (work / "r.gpf").write_text("# stub\\n")
>>> _ = (work / "r.dpf").write_text("# stub\\n")
>>> result = run_autodock4(work, timeout_s=10.0)  # engine absent -> not-ok
>>> (result.ok, result.log_file, result.error is not None)
(False, None, True)
>>> (work / "r.gpf").unlink()
>>> (work / "r.dpf").unlink()
>>> work.rmdir()

>>> from pharmaml.docking.workflow import batch_run_directories, run_vina_batch
>>> batch_run_directories("/tmp/pharmaml_doctest_base", engine="vina")
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: batch engine must be 'autodock4', got 'vina'
>>> base = Path("/tmp/pharmaml_doctest_base")
>>> base.mkdir()
>>> batch_run_directories(base)    # no ligand sub-directories inside
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: no ligand sub-directories found under /tmp/pharmaml_doctest_base
>>> base.rmdir()
>>> conf = Path("/tmp/pharmaml_doctest_batch.conf")
>>> _ = conf.write_text("receptor = pharmaml_doctest_rec.pdbqt\\n")
>>> _ = (conf.parent / "pharmaml_doctest_rec.pdbqt").write_text("REMARK stub\\n")
>>> run_vina_batch(conf, "/tmp/pharmaml_doctest_missing_ligs")
Traceback (most recent call last):
...
pharmaml.exceptions.DockingError: ligands directory not found: /tmp/pharmaml_doctest_missing_ligs
>>> conf.unlink()
>>> (conf.parent / "pharmaml_doctest_rec.pdbqt").unlink()
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import struct
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pharmaml.exceptions import DockingError
from pharmaml.logger import get_logger
from pharmaml.utils.io import safe_filename

_LOG = get_logger(__name__)

__all__ = [
    "DockingRunResult",
    "VinaBatchProgress",
    "batch_run_directories",
    "engine_argv",
    "locate_engine_binary",
    "parse_vina_config",
    "run_autodock4",
    "run_vina",
    "run_vina_batch",
    "vina_batch_progress",
]


@dataclass(frozen=True, slots=True)
class DockingRunResult:
    """Outcome of one docking run.

    Attributes
    ----------
    directory:
        Working directory of the run.
    ok:
        Whether the run completed successfully.
    log_file:
        Primary log/output file produced.
    seconds:
        Wall-clock duration.
    error:
        Error message when ``ok`` is False.
    skipped:
        True when the run was not started at all because its output
        already existed (``run_vina_batch(skip_existing=True)`` -- the
        resume behaviour of ``Vina_contol/batch.py`` and
        ``niggavina.py``); such rows count as ``ok`` with zero seconds.
    """

    directory: Path
    ok: bool
    log_file: Path | None
    seconds: float
    error: str | None = None
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class VinaBatchProgress:
    """Pre-flight resume census for :func:`run_vina_batch`.

    The answer to "what would a resumed run actually do?" before the
    Dock button is pressed -- the same skip rule
    ``run_vina_batch(skip_existing=True)`` applies (output PDBQT AND
    non-empty log, the zero-byte log marking a crashed job), computed
    with the SAME collision-disambiguated output names the batch will
    write, so the census and the run can never disagree.

    Attributes
    ----------
    total:
        Ligand PDBQTs found under the tree (recursively).
    docked:
        Ligands whose ``<stem>_out.pdbqt`` AND non-empty ``<stem>.log``
        already exist in the output folder -- the rows a resumed batch
        skips in seconds.
    remaining:
        ``total - docked`` -- the ligands a resumed run would dock.
    collisions:
        Same-stem ligands in different sub-folders whose flat output
        names were disambiguated (the "<parent>_<stem>" prefix rule).

    Examples
    --------
    >>> import shutil, tempfile
    >>> from pathlib import Path
    >>> from pharmaml.docking.workflow import vina_batch_progress
    >>> root = Path(tempfile.mkdtemp())
    >>> ligs, out = root / "ligs", root / "out"
    >>> ligs.mkdir(); out.mkdir()
    >>> _ = (ligs / "a.pdbqt").write_text("LIGAND\\n")
    >>> _ = (ligs / "b.pdbqt").write_text("LIGAND\\n")
    >>> vina_batch_progress(ligs, out)   # nothing docked yet
    VinaBatchProgress(total=2, docked=0, remaining=2, collisions=0)
    >>> _ = (out / "a_out.pdbqt").write_text("POSE\\n")
    >>> _ = (out / "a.log").write_text("Estimated\\n")
    >>> vina_batch_progress(ligs, out)   # 'a' complete, 'b' still to go
    VinaBatchProgress(total=2, docked=1, remaining=1, collisions=0)
    >>> _ = (out / "b_out.pdbqt").write_text("POSE\\n")
    >>> _ = (out / "b.log").write_text("")   # zero-byte: crashed job
    >>> vina_batch_progress(ligs, out)   # 'b' is honestly NOT docked
    VinaBatchProgress(total=2, docked=1, remaining=1, collisions=0)
    >>> sub = ligs / "sub"; sub.mkdir()
    >>> _ = (sub / "a.pdbqt").write_text("LIGAND\\n")   # stem collision
    >>> vina_batch_progress(ligs, out)
    VinaBatchProgress(total=3, docked=1, remaining=2, collisions=1)
    >>> shutil.rmtree(root)
    >>> vina_batch_progress(root / "ligs", out)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DockingError: ligands directory not found: .../ligs
    """

    total: int
    docked: int
    remaining: int
    collisions: int


def _require_executable(name: str) -> str:
    located = shutil.which(name)
    if located is None:
        raise DockingError(
            f"executable {name!r} not found on PATH -- install AutoDock Tools "
            "(autogrid4/autodock4) or AutoDock Vina first"
        )
    return located


# Hashes identify the executables in the existing v1.2.172 release. An
# intentional binary update must update these references and be reviewed as a
# separate provenance/licensing change; explicit user engines are not pinned.
_BUNDLED_ENGINE_SHA256 = {
    "vina.exe": "e0c4b2715e0c1a74f6e92d0f3be0328ac97542eafbc111e6b1efad897a73cce5",
    "autogrid4.exe": "797efce687d1ae82df59726461e0e1966b3d8edb0f8b187f982fa1ab0c12da9e",
    "autodock4.exe": "36c0b16c04d7df8e6225737bae65bb04058d1f4c90accfaf2d230dbc913954bf",
}
_BUNDLED_ENGINE_MACHINE = {"vina.exe": 0x8664, "autogrid4.exe": 0x14C, "autodock4.exe": 0x14C}


def _bundled_engine_dirs() -> tuple[Path, ...]:
    """Known resource locations only; never guess an unrelated site-packages sibling."""
    source = Path(__file__).resolve()
    package = source.parents[1]
    paths = [package / "engines"]  # if a future wheel explicitly includes package data
    if source.parents[2].name == "src":
        root = source.parents[3]
        if (root / "VERSION").is_file() and (root / "pyproject.toml").is_file():
            paths.insert(0, root / "engines")
    return tuple(paths)


def _pe_machine(path: Path) -> int | None:
    """Return the PE machine field, or None for a non-PE file; reject malformed MZ files."""
    try:
        with path.open("rb") as handle:
            header = handle.read(64)
            if header[:2] != b"MZ":
                return None
            if len(header) != 64:
                raise DockingError(f"invalid or truncated PE header: {path}")
            offset = struct.unpack_from("<I", header, 0x3C)[0]
            # Bound the seek as well as checking the signature; no full binary read.
            if offset < 64 or offset > 16 * 1024 * 1024:
                raise DockingError(f"invalid PE header offset: {path}")
            handle.seek(offset)
            coff = handle.read(6)
            if len(coff) != 6 or coff[:4] != b"PE\0\0":
                raise DockingError(f"invalid or truncated PE signature: {path}")
            return struct.unpack_from("<H", coff, 4)[0]
    except OSError as exc:
        raise DockingError(f"cannot inspect engine binary {path}: {exc}") from exc


def _bundled_engine(name: str) -> Path | None:
    if "/" in name or "\\" in name:
        return None
    filename = name.lower() if name.lower().endswith(".exe") else name.lower() + ".exe"
    expected = _BUNDLED_ENGINE_SHA256.get(filename)
    if expected is None:
        return None
    for folder in _bundled_engine_dirs():
        candidate = folder / filename
        if not candidate.is_file():
            continue
        try:
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise DockingError(f"cannot inspect bundled engine {candidate}: {exc}") from exc
        if digest.hexdigest() != expected:
            raise DockingError(f"bundled engine checksum mismatch: {candidate}")
        if _pe_machine(candidate) != _BUNDLED_ENGINE_MACHINE[filename]:
            raise DockingError(f"bundled engine PE architecture mismatch: {candidate}")
        return candidate
    return None


def locate_engine_binary(
    name: str, engine_dir: str | Path | None = None
) -> Path | None:
    """Locate an engine, but do not claim it is runnable on the current host.

    Explicit binary paths win, then an explicitly supplied folder, then PATH,
    then a checksum- and PE-validated bundled Windows engine. PATH precedes
    bundled binaries so existing native installations are not silently replaced.
    Source trees / extracted ZIPs use their toolkit-root ``engines/``; wheels
    can only use binaries explicitly packaged in ``pharmaml/engines/``. Use
    :func:`engine_argv` to validate host architecture and Wine availability.

    >>> locate_engine_binary("definitely_not_a_real_engine_anywhere") is None
    True
    """
    if "/" in name or "\\" in name:
        candidate = Path(name)
        return candidate.resolve() if candidate.is_file() else None
    spellings = (name,) if name.lower().endswith(".exe") else (name, f"{name}.exe")
    if engine_dir is not None:
        folder = Path(engine_dir)
        for spelling in spellings:
            candidate = folder / spelling
            if candidate.is_file():
                return candidate.resolve()
    for spelling in spellings:
        located = shutil.which(spelling)
        if located is not None:
            return Path(located).resolve()
    return _bundled_engine(name)


def _host_platform() -> tuple[str, str]:
    return os.name, platform.machine().lower()


def _pe_argv(binary: Path | str) -> list[str]:
    """Construct an argv, validating PE machine and Wine presence where required.

    Discovery and a Wine binary do not prove that the engine or Wine's 32-bit
    support works; subprocess execution remains the actual runtime test.
    """
    path = Path(binary)
    if not path.is_file():  # preserve callers that inject an executable resolver
        return [str(binary)]
    machine = _pe_machine(path)
    if machine is None:
        return [str(binary)]
    if machine not in (0x14C, 0x8664):
        raise DockingError(f"unsupported PE architecture 0x{machine:04x}: {path}")
    host_os, arch = _host_platform()
    if arch not in {"x86_64", "amd64", "x64", "x86", "i386", "i486", "i586", "i686"}:
        raise DockingError(f"cannot assume x86 PE engine compatibility on host architecture {arch!r}: {path}")
    if machine == 0x8664 and arch in {"x86", "i386", "i486", "i586", "i686"}:
        raise DockingError(f"64-bit PE engine requires a 64-bit x86 host: {path}")
    if host_os == "nt":
        return [str(binary)]
    wine = shutil.which("wine") or shutil.which("wine64")
    if wine is None:
        raise DockingError(
            f"{path.name} is a Windows PE executable and no wine was found on "
            "PATH -- run it on Windows, install compatible wine, or use a native engine build"
        )
    wine_path = Path(wine)
    return [str(wine_path.resolve() if wine_path.is_file() else wine), str(binary)]


def engine_argv(name: str, engine_dir: str | Path | None = None) -> list[str]:
    """Resolve an engine and validate its launch prefix; no execution occurs.

    Resolution is explicit path, explicit folder, PATH, then a verified bundled
    binary. Native PATH installations are preferred over bundled Windows builds.
    Wine must exist for a PE on non-Windows; its presence does not prove the
    executable will actually run. An explicit but unusable selection fails
    rather than silently changing engines.
    """
    if engine_dir is None:
        try:
            resolved = _require_executable(name)  # preserve existing PATH hook
        except DockingError:
            binary = locate_engine_binary(name)
            if binary is None:
                raise
            return _pe_argv(binary)
        resolved_path = Path(resolved)
        return _pe_argv(resolved_path.resolve() if resolved_path.is_file() else resolved)
    binary = locate_engine_binary(name, engine_dir)
    if binary is None:
        raise DockingError(
            f"executable {name!r} not found in engine folder {engine_dir}, on PATH, "
            "or among verified bundled engines"
        )
    return _pe_argv(binary)


#: Sentinel AutoDock4 writes at the very end of a healthy ``.dlg`` --
#: the same string :func:`~pharmaml.docking.parsing.check_docking_completion`
#: audits per directory.
_COMPLETION_SENTINEL = "Successful Completion"


def _log_complete(log: Path) -> bool:
    """True when an AutoDock log exists AND carries the completion sentinel.

    A log's *presence* says nothing: interrupted runs leave partial logs
    behind.  This is the single-file form of the rule
    :func:`~pharmaml.docking.parsing.check_docking_completion` applies
    per directory.
    """
    if not log.is_file():
        return False
    content = log.read_text(encoding="utf-8", errors="ignore")
    return _COMPLETION_SENTINEL in content


def run_autodock4(
    directory: str | Path,
    grid_field: str = "r.gpf",
    dock_param: str = "r.dpf",
    autogrid: str = "autogrid4",
    autodock: str = "autodock4",
    skip_existing: bool = True,
    timeout_s: float = 3600.0,
    engine_dir: str | Path | None = None,
) -> DockingRunResult:
    """Run autogrid4 + autodock4 for one ligand directory.

    The original changed into each subfolder globally and ran
    ``"autogrid4 -p r.gpf -l r.glg"`` through a shell; here the
    directory is an explicit argument and outputs are validated.

    Parameters
    ----------
    directory:
        Folder containing ``r.gpf``/``r.dpf`` (or where they will be
        generated by :func:`~pharmaml.docking.preparation.write_gpf`).
    grid_field:
        Grid parameter file basename.
    dock_param:
        Docking parameter file basename.
    skip_existing:
        Reuse existing ``.glg`` output when present, and an existing
        ``.dlg`` only when it carries AutoDock's ``Successful
        Completion`` sentinel -- an interrupted run is re-run, never
        trusted just because the file exists.
    engine_dir:
        Folder holding the engine binaries (``autogrid4`` /
        ``autogrid4.exe`` / ``autodock4`` / ``autodock4.exe``), searched
        before PATH -- the Windows ``.exe`` builds the toolkit ships.

    Returns
    -------
    DockingRunResult
    """
    import time

    work = Path(directory)
    if not work.is_dir():
        raise DockingError(f"docking directory not found: {work}")

    glg = work / "r.glg"
    dlg = work / "r.dlg"
    started = time.perf_counter()

    def _failure(error: str) -> DockingRunResult:
        return DockingRunResult(
            work, False, None, time.perf_counter() - started, error
        )

    if not (skip_existing and glg.is_file()):
        if not (work / grid_field).is_file():
            raise DockingError(f"missing {grid_field} in {work}")
        try:
            autogrid_prefix = engine_argv(autogrid, engine_dir)
            subprocess.run(
                [*autogrid_prefix, "-p", grid_field, "-l", "r.glg"],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except DockingError as exc:
            # missing engine: a clean not-ok result (the batch wrapper
            # reports it per ligand) instead of an exception mid-run
            return _failure(str(exc))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            return _failure(str(exc))

    if not (skip_existing and _log_complete(dlg)):
        if not (work / dock_param).is_file():
            raise DockingError(f"missing {dock_param} in {work}")
        try:
            autodock_prefix = engine_argv(autodock, engine_dir)
            subprocess.run(
                [*autodock_prefix, "-p", dock_param, "-l", "r.dlg"],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except DockingError as exc:
            return _failure(str(exc))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            return _failure(str(exc))

    if not _log_complete(dlg):
        return _failure("autodock4 produced no complete .dlg output")
    return DockingRunResult(
        work, True, dlg, time.perf_counter() - started
    )


def run_vina(
    receptor: str | Path,
    ligand: str | Path,
    out: str | Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int = 8,
    num_modes: int = 9,
    cpu: int = 0,
    seed: int | None = None,
    vina_bin: str = "vina",
    timeout_s: float = 3600.0,
    engine_dir: str | Path | None = None,
) -> DockingRunResult:
    """Run a single AutoDock Vina docking.

    Parameters
    ----------
    receptor, ligand:
        PDBQT input paths.
    out:
        Output PDBQT path (poses appended with affinity REMARKs).
    center, size:
        Search box centre and edge lengths in Å.
    exhaustiveness:
        Vina search depth (8 default, 16+ for production).
    num_modes:
        Maximum number of poses to write.
    cpu:
        CPU count (0 = let Vina decide).
    seed:
        Random seed for reproducible docking.
    engine_dir:
        Folder holding the engine binary (``vina`` / ``vina.exe``),
        searched before PATH -- the Windows ``.exe`` build the toolkit
        ships.

    Returns
    -------
    DockingRunResult
        A zero exit is not sufficient: a new non-empty output must exist.
        Output is staged next to the destination and only replaces an older
        file after successful execution. This does not validate the pose or
        score scientifically and does not provide crash durability.
    """
    import time

    prefix = engine_argv(vina_bin, engine_dir)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *prefix,
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", f"{center[0]:.4f}",
        "--center_y", f"{center[1]:.4f}",
        "--center_z", f"{center[2]:.4f}",
        "--size_x", f"{size[0]:.4f}",
        "--size_y", f"{size[1]:.4f}",
        "--size_z", f"{size[2]:.4f}",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]
    if cpu:
        cmd += ["--cpu", str(cpu)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=".vina-run-", dir=out_path.parent) as folder:
            staged = Path(folder) / "pose.pdbqt"
            subprocess.run([*cmd, "--out", str(staged)], check=True, capture_output=True, timeout=timeout_s)
            if not staged.is_file() or staged.is_symlink() or staged.stat().st_size == 0:
                return DockingRunResult(out_path.parent, False, None, time.perf_counter() - started,
                                        "Vina produced no new non-empty pose file")
            os.replace(staged, out_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return DockingRunResult(out_path.parent, False, None, time.perf_counter() - started, str(exc))
    return DockingRunResult(out_path.parent, True, out_path, time.perf_counter() - started)


def _run_one(directory: str, engine_dir: str | Path | None = None) -> DockingRunResult:
    """Top-level helper so ProcessPoolExecutor can pickle it."""
    try:
        return run_autodock4(directory, engine_dir=engine_dir)
    except DockingError as exc:
        return DockingRunResult(Path(directory), False, None, 0.0, str(exc))


def parse_vina_config(conf_path: str | Path) -> dict[str, str]:
    """Parse an AutoDock Vina config file into a ``key -> value`` dict.

    Upgrades the ``get_receptor_from_config`` helper of
    ``others/Vina/vinar.py`` into a general reader: lines of the form
    ``key = value`` are collected with ``#`` comments and blank lines
    stripped, and values stay strings (Vina itself parses them).  Keys
    typically include ``receptor``, ``center_x/y/z``, ``size_x/y/z``,
    ``exhaustiveness`` and ``num_modes``.

    Parameters
    ----------
    conf_path:
        Path to the config file.

    Returns
    -------
    dict of str to str

    Raises
    ------
    DockingError
        If the config file is missing.
    """
    path = Path(conf_path)
    if not path.is_file():
        raise DockingError(f"config file not found: {path}")
    config: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and value:
            config[key] = value
    return config


def _batch_output_names(ligands: list[Path]) -> dict[Path, str]:
    """The flat output stem per ligand, collision-disambiguated.

    Flat output names ("l_out.pdbqt") collide when two sub-folders hold
    ligands with the same stem -- the second run would silently
    overwrite the first.  Colliding stems get a "<parent>_<stem>"
    prefix (and a running counter if even that repeats), so no result
    is ever lost.  One rule shared by :func:`run_vina_batch` and
    :func:`vina_batch_progress` (100x-142) -- the census and the run can
    never disagree about which file a ligand owns.
    """
    used_stems: set[str] = set()
    output_names: dict[Path, str] = {}
    for ligand in ligands:
        stem = ligand.stem
        if stem in used_stems:
            stem = f"{safe_filename(ligand.parent.name)}_{ligand.stem}"
            counter = 2
            while stem in used_stems:
                stem = f"{safe_filename(ligand.parent.name)}_{ligand.stem}_{counter}"
                counter += 1
            _LOG.warning(
                "output name collision for %s -- writing %s_out.pdbqt instead",
                ligand,
                stem,
            )
        used_stems.add(stem)
        output_names[ligand] = stem
    return output_names


def vina_batch_progress(
    ligands_dir: str | Path,
    out_dir: str | Path,
) -> VinaBatchProgress:
    """Count what a resumed Vina batch would skip and what it would dock.

    The pre-flight twin of ``run_vina_batch(skip_existing=True)``: the
    SAME skip rule (``<stem>_out.pdbqt`` AND non-empty ``<stem>.log``, a
    zero-byte log counting as a crashed job to redo) applied with the
    SAME collision-disambiguated output names -- so "N ligands already
    docked, M to go" on a button is a promise the run keeps.

    Parameters
    ----------
    ligands_dir:
        Tree with ligand PDBQTs (searched recursively).
    out_dir:
        The batch's output folder (poses and logs live there).

    Returns
    -------
    VinaBatchProgress
        total / docked / remaining / collisions census.

    Raises
    ------
    DockingError
        If the ligands directory is missing (the run's own first
        check, fired early for the same honest reason).
    """
    ligands_root = Path(ligands_dir)
    if not ligands_root.is_dir():
        raise DockingError(f"ligands directory not found: {ligands_dir}")
    out = Path(out_dir)
    ligands = sorted(p for p in ligands_root.rglob("*.pdbqt") if p.is_file())
    output_names = _batch_output_names(ligands)
    collisions = sum(
        1 for ligand, stem in output_names.items() if stem != ligand.stem
    )
    docked = 0
    for stem in output_names.values():
        out_pdbqt = out / f"{stem}_out.pdbqt"
        log_file = out / f"{stem}.log"
        if (
            out_pdbqt.is_file()
            and out_pdbqt.stat().st_size > 0
            and log_file.is_file()
            and log_file.stat().st_size > 0
        ):
            docked += 1
    return VinaBatchProgress(
        total=len(ligands),
        docked=docked,
        remaining=len(ligands) - docked,
        collisions=collisions,
    )


def run_vina_batch(
    conf_path: str | Path,
    ligands_dir: str | Path,
    out_dir: str | Path = "vina_results",
    max_parallel: int = 1,
    vina_bin: str = "vina",
    timeout_s: float = 3600.0,
    skip_existing: bool = False,
    engine_dir: str | Path | None = None,
) -> list[DockingRunResult]:
    """Dock every ``.pdbqt`` under a directory with one Vina config.

    Upgrades ``others/Vina/vinar.py`` (sequential) and
    ``vinar_multiple.py`` (six detached CMD consoles polled via
    ``.done`` signal files): the receptor is read from the config,
    resolved relative to the config's own directory, and every ligand
    found recursively under ``ligands_dir`` is docked with the config
    file itself (``--config``), producing ``<name>_out.pdbqt`` and
    ``<name>.log`` in ``out_dir``. The log contains captured engine output;
    Vina 1.2 does not accept the old ``--log`` flag. Output is staged and
    checked for nonempty content, not scientific validity. Promotion of the
    pose and log is not a crash-atomic two-file transaction. ``max_parallel > 1`` runs the
    subprocesses through a ``ThreadPoolExecutor`` (subprocesses
    release the GIL, so threads are enough).

    ``skip_existing=True`` adds the resume behaviour of the 2026-10
    "new folder" upload's ``Vina_contol/batch.py`` and ``niggavina.py``:
    a ligand whose ``<stem>_out.pdbqt`` AND ``<stem>.log`` both exist
    and are non-empty is not started again (a zero-byte log marks a
    crashed job, exactly like ``niggavina.py``'s empty-log cleanup) --
    an interrupted campaign re-runs in seconds, docking only the
    remainder. The skip check is only a file-presence heuristic: it does not
    prove a completed docking, validate content, or match the current inputs,
    configuration, engine, or seed. Use a fresh output directory when those
    change and independently inspect results.

    Parameters
    ----------
    conf_path:
        Vina config file (must contain a ``receptor = ...`` line).
    ligands_dir:
        Tree with ligand PDBQTs (searched recursively).
    out_dir:
        Output folder for poses and logs (created when missing).
    max_parallel:
        Maximum concurrent Vina processes (1 = sequential).
    vina_bin:
        Vina executable name or path.
    timeout_s:
        Per-ligand subprocess timeout.
    skip_existing:
        Resume mode -- skip ligands whose output PDBQT and non-empty
        log already exist (``batch.py``'s "already docked" rule).
    engine_dir:
        Folder holding the engine binary (``vina`` / ``vina.exe``),
        searched before PATH -- the Windows ``.exe`` build the toolkit
        ships.

    Returns
    -------
    list of DockingRunResult
        One result per ligand, in ligand-sorted order; skipped
        ligands carry ``skipped=True`` with ``ok=True``.

    Raises
    ------
    DockingError
        If the config lacks a ``receptor`` key (the same error the
        original raised), the receptor file is not next to the config,
        or the ligands directory is missing.
    """
    import time

    conf = Path(conf_path)
    config = parse_vina_config(conf)
    if "receptor" not in config:
        raise DockingError(
            f"no receptor found in config file {conf} (missing 'receptor = ...')"
        )
    receptor = conf.parent / config["receptor"]
    if not receptor.is_file():
        raise DockingError(
            f"receptor '{config['receptor']}' not found -- expected next to "
            f"the config ({conf.parent}) or give a full path"
        )
    ligands_root = Path(ligands_dir)
    if not ligands_root.is_dir():
        raise DockingError(f"ligands directory not found: {ligands_root}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ligands = sorted(p for p in ligands_root.rglob("*.pdbqt") if p.is_file())
    if not ligands:
        _LOG.warning("no .pdbqt ligands found under %s", ligands_root)
        return []
    prefix = engine_argv(vina_bin, engine_dir)

    # (100x-142) the collision-disambiguated stems live in ONE shared
    # helper -- vina_batch_progress applies the identical naming, so
    # the pre-flight census and this run can never disagree.
    output_names = _batch_output_names(ligands)

    def _dock_one(ligand: Path) -> DockingRunResult:
        started = time.perf_counter()
        stem = output_names[ligand]
        out_pdbqt = out / f"{stem}_out.pdbqt"
        log_file = out / f"{stem}.log"
        if skip_existing and out_pdbqt.is_file() and out_pdbqt.stat().st_size > 0 \
                and log_file.is_file() and log_file.stat().st_size > 0:
            _LOG.info("skipping %s (already docked)", ligand.name)
            return DockingRunResult(out, True, log_file, 0.0, None, True)
        try:
            with tempfile.TemporaryDirectory(prefix=".vina-batch-", dir=out.resolve()) as folder:
                stage = Path(folder)
                staged_pose = stage / out_pdbqt.name
                staged_log = stage / log_file.name
                cmd = [
                    *prefix,
                    "--config", str(conf.resolve()),
                    "--ligand", str(ligand.resolve()),
                    "--out", str(staged_pose),
                ]
                process = subprocess.run(
                    cmd, check=True, capture_output=True, timeout=timeout_s, cwd=conf.parent,
                )
                if not staged_pose.is_file() or staged_pose.is_symlink() or staged_pose.stat().st_size == 0:
                    return DockingRunResult(out, False, None, time.perf_counter() - started,
                                            "Vina produced no new non-empty pose file")
                captured = (process.stdout or b"") + (process.stderr or b"")
                staged_log.write_bytes(captured or b"Vina exited successfully with no console output.\n")
                os.replace(staged_pose, out_pdbqt)
                os.replace(staged_log, log_file)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            return DockingRunResult(
                out, False, None, time.perf_counter() - started, str(exc)
            )
        return DockingRunResult(out, True, log_file, time.perf_counter() - started)

    _LOG.info(
        "Vina batch: %d ligand(s) from %s with config %s (parallel=%d)",
        len(ligands),
        ligands_root,
        conf,
        max_parallel,
    )
    if max_parallel > 1:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(_dock_one, ligand) for ligand in ligands]
            results = [future.result() for future in futures]
    else:
        results = [_dock_one(ligand) for ligand in ligands]

    failures = [r for r in results if not r.ok]
    if failures:
        _LOG.warning("%d/%d Vina runs failed", len(failures), len(results))
    return results


def batch_run_directories(
    base_directory: str | Path,
    workers: int = 4,
    engine: str = "autodock4",
    engine_dir: str | Path | None = None,
) -> list[DockingRunResult]:
    """Dock every prepared ligand sub-directory in parallel.

    Upgrades ``main.py``: instead of an ``input()`` prompt for the
    worker count and a batched ``executor.map``, the count is a
    parameter, failures are collected (not swallowed), and per-run
    timings are returned for pipeline monitoring.

    Parameters
    ----------
    base_directory:
        Folder whose *sub-directories* each hold one ligand's
        ``r.gpf``/``r.dpf`` pair.
    workers:
        Parallel processes (cap at CPU count - 1 as the original
        comment rightly suggested).
    engine:
        ``'autodock4'`` (batch folders) -- Vina runs are single-shot
        via :func:`run_vina`.
    engine_dir:
        Folder holding the engine binaries, searched before PATH --
        threaded through to every per-folder :func:`run_autodock4`
        child process.

    Returns
    -------
    list of DockingRunResult
        One result per ligand directory, completion order.
    """
    import os

    if engine != "autodock4":
        raise DockingError(f"batch engine must be 'autodock4', got {engine!r}")
    base = Path(base_directory)
    if not base.is_dir():
        raise DockingError(f"base directory not found: {base}")

    workers = max(1, min(workers, max(1, (os.cpu_count() or 2) - 1)))
    directories = sorted(p for p in base.iterdir() if p.is_dir())
    if not directories:
        raise DockingError(f"no ligand sub-directories found under {base}")

    _LOG.info("Batch docking %d ligands with %d workers", len(directories), workers)
    results: list[DockingRunResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one, str(d), engine_dir): d for d in directories
        }
        for future in as_completed(futures):
            results.append(future.result())

    failures = [r for r in results if not r.ok]
    if failures:
        _LOG.warning("%d/%d docking runs failed", len(failures), len(results))
    return results
