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

import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pharmaml.exceptions import DockingError
from pharmaml.logger import get_logger
from pharmaml.utils.io import safe_filename

_LOG = get_logger(__name__)

__all__ = [
    "DockingRunResult",
    "batch_run_directories",
    "parse_vina_config",
    "run_autodock4",
    "run_vina",
    "run_vina_batch",
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


def _require_executable(name: str) -> str:
    located = shutil.which(name)
    if located is None:
        raise DockingError(
            f"executable {name!r} not found on PATH -- install AutoDock Tools "
            "(autogrid4/autodock4) or AutoDock Vina first"
        )
    return located


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
            autogrid_bin = _require_executable(autogrid)
            subprocess.run(
                [autogrid_bin, "-p", grid_field, "-l", "r.glg"],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except DockingError as exc:
            # missing engine: a clean not-ok result (the batch wrapper
            # reports it per ligand) instead of an exception mid-run
            return _failure(str(exc))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return _failure(str(exc))

    if not (skip_existing and _log_complete(dlg)):
        if not (work / dock_param).is_file():
            raise DockingError(f"missing {dock_param} in {work}")
        try:
            autodock_bin = _require_executable(autodock)
            subprocess.run(
                [autodock_bin, "-p", dock_param, "-l", "r.dlg"],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except DockingError as exc:
            return _failure(str(exc))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
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

    Returns
    -------
    DockingRunResult
    """
    import time

    binary = _require_executable(vina_bin)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary,
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--out", str(out_path),
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
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_s)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return DockingRunResult(out_path.parent, False, None, time.perf_counter() - started, str(exc))
    return DockingRunResult(out_path.parent, True, out_path, time.perf_counter() - started)


def _run_one(directory: str) -> DockingRunResult:
    """Top-level helper so ProcessPoolExecutor can pickle it."""
    try:
        return run_autodock4(directory)
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


def run_vina_batch(
    conf_path: str | Path,
    ligands_dir: str | Path,
    out_dir: str | Path = "vina_results",
    max_parallel: int = 1,
    vina_bin: str = "vina",
    timeout_s: float = 3600.0,
    skip_existing: bool = False,
) -> list[DockingRunResult]:
    """Dock every ``.pdbqt`` under a directory with one Vina config.

    Upgrades ``others/Vina/vinar.py`` (sequential) and
    ``vinar_multiple.py`` (six detached CMD consoles polled via
    ``.done`` signal files): the receptor is read from the config,
    resolved relative to the config's own directory, and every ligand
    found recursively under ``ligands_dir`` is docked with the config
    file itself (``--config``), producing ``<name>_out.pdbqt`` and
    ``<name>.log`` in ``out_dir``.  ``max_parallel > 1`` runs the
    subprocesses through a ``ThreadPoolExecutor`` (subprocesses
    release the GIL, so threads are enough).

    ``skip_existing=True`` adds the resume behaviour of the 2026-10
    "new folder" upload's ``Vina_contol/batch.py`` and ``niggavina.py``:
    a ligand whose ``<stem>_out.pdbqt`` AND ``<stem>.log`` both exist
    and are non-empty is not started again (a zero-byte log marks a
    crashed job, exactly like ``niggavina.py``'s empty-log cleanup) --
    an interrupted campaign re-runs in seconds, docking only the
    remainder.

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
    binary = _require_executable(vina_bin)

    # Flat output names ("l_out.pdbqt") collide when two sub-folders hold
    # ligands with the same stem -- the second run would silently overwrite
    # the first.  Colliding stems get a "<parent>_<stem>" prefix (and a
    # running counter if even that repeats), so no result is ever lost.
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

    def _dock_one(ligand: Path) -> DockingRunResult:
        started = time.perf_counter()
        stem = output_names[ligand]
        out_pdbqt = out / f"{stem}_out.pdbqt"
        log_file = out / f"{stem}.log"
        if skip_existing and out_pdbqt.is_file() and out_pdbqt.stat().st_size > 0 \
                and log_file.is_file() and log_file.stat().st_size > 0:
            _LOG.info("skipping %s (already docked)", ligand.name)
            return DockingRunResult(out, True, log_file, 0.0, None, True)
        cmd = [
            binary,
            "--config", str(conf),
            "--ligand", str(ligand),
            "--out", str(out_pdbqt),
            "--log", str(log_file),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=timeout_s,
                cwd=conf.parent,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
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
        futures = {executor.submit(_run_one, str(d)): d for d in directories}
        for future in as_completed(futures):
            results.append(future.result())

    failures = [r for r in results if not r.ok]
    if failures:
        _LOG.warning("%d/%d docking runs failed", len(failures), len(results))
    return results
