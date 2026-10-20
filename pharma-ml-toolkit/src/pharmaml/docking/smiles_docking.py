"""Dock a pasted SMILES array with Vina or AutoDock4, all at once.

The campaign workflow (``prepare_ligand_library`` → distribute → grid →
run) is built around a CSV library and pre-made folders; when the
question is just "dock these ten SMILES against this receptor NOW",
that pipeline is five manual steps.  :func:`dock_smiles_array` is the
one-call version of the whole story:

1. every SMILES is embedded and typed into its own campaign folder
   (``<out>/<ii>_<name>/l.pdbqt``, Meeko + RDKit, fixed seed);
2. the engine runs over every prepared ligand -- Vina directly, or the
   full AutoDock4 stack (receptor distributed, GPF + DPF written,
   ``autogrid4`` + ``autodock4`` executed per folder);
3. the best affinity of every run is parsed back out of the engine's
   own output (``REMARK VINA RESULT`` poses / the DLG cluster table)
   into one ranked DataFrame.

``workers > 1`` runs the engine subprocesses simultaneously (a
``ThreadPoolExecutor`` -- subprocesses release the GIL), the ligand
preparation itself stays sequential because RDKit embedding is
in-process CPU work.

Examples
--------
>>> from pharmaml.docking.smiles_docking import _slug_name, _vina_best
>>> _slug_name("CC(=O)Oc1ccccc1C(=O)O")
'aspirin'
>>> _slug_name("not a valid ??? smiles")
'notavalidsmiles'
>>> _slug_name("CC(C)CC(=O)O")
'cccccoo'
>>> _vina_best("REMARK VINA RESULT:    -7.3      0.000      0.000\\n"
...            "REMARK VINA RESULT:    -6.1      1.923      2.941\\n")
(-7.3, 2)
>>> _vina_best("no poses here")
(None, 0)
>>> from pharmaml.docking.smiles_docking import _AUTODOCK_ENGINES, _ENGINES
>>> sorted(_ENGINES)
['autodock4', 'vina']
>>> _AUTODOCK_ENGINES == frozenset({"autodock4"})
True
"""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pharmaml.docking.parsing import parse_dlg_cluster_table, parse_vina_output
from pharmaml.docking.preparation import GridBox, smiles_to_pdbqt, write_dpf, write_gpf
from pharmaml.docking.receptor import distribute_receptor
from pharmaml.docking.workflow import run_autodock4, run_vina
from pharmaml.exceptions import DataValidationError, DockingError
from pharmaml.logger import get_logger
from pharmaml.utils.io import ensure_directory

__all__ = ["dock_smiles_array"]

_LOG = get_logger(__name__)

#: Engine tokens accepted by :func:`dock_smiles_array`.
_ENGINES = frozenset({"vina", "autodock4"})

#: The engines that need the AutoDock4 grid map stack.
_AUTODOCK_ENGINES = frozenset({"autodock4"})

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9]+")

# A short, molecule-identifying folder name from a SMILES: known
# trivial patterns map to their names (the bundled example library's
# own molecules), anything else keeps only its safe alphanumeric core
# and falls back to "ligand" when nothing survives.
_TRIVIAL_NAMES: dict[str, str] = {
    "CC(=O)Oc1ccccc1C(=O)O": "aspirin",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C": "caffeine",
    "CC(=O)Nc1ccc(O)cc1": "paracetamol",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O": "ibuprofen",
    "COc1cc2nc(CC(C)C)sc2c(O)cc1": "naproxen",
    "Cc1c(O)c(C(=O)O)c(O)nc1N": "theophylline",
    "O=C(NCc1ccc(O)cc1)NCCN": "metformin",
    "COc1ccc(cc1OC)C(C)C(=O)OC": "propranolol",
    "CCOC(=O)c1ccc2[nH]c(=O)ccc2c1": "coumarin",
    "COc1cc(C=O)ccc1O": "vanillin",
    "CN(C)CC(Oc1cccc2ccccc12)CCN": "lidocaine",
    "c1ccc(cc1)C": "benzocaine",
    "OC(=O)C=CCc1ccc(O)cc1": "caffeic_acid",
}


def _slug_name(smiles: str) -> str:
    """A short, molecule-identifying slug for one SMILES."""
    slug = _TRIVIAL_NAMES.get(smiles.strip())
    if not slug:
        slug = _SLUG_SAFE.sub("", smiles.strip())[:24].lower()
    return slug or "ligand"


def _vina_best(text: str) -> tuple[float | None, int]:
    """Best affinity + pose count from Vina output text (pose 1 = best).

    A text with no pose rows (interrupted run, empty output) yields
    ``(None, 0)`` instead of the parser's exception -- the batch caller
    reports it as a failed row, not a crashed run.
    """
    from pharmaml.exceptions import DockingParseError

    try:
        poses = parse_vina_output(text)
    except DockingParseError:
        return None, 0
    if not poses:
        return None, 0
    best = min(poses, key=lambda pose: pose.affinity)
    return best.affinity, len(poses)


def _autodock_best(dlg_path: Path) -> tuple[float | None, int]:
    """Best cluster energy + cluster count from a DLG file.

    An unreadable or pose-less DLG yields ``(None, 0)`` -- the batch
    caller reports it as a failed row instead of crashing the run.
    """
    from pharmaml.exceptions import DockingParseError

    try:
        clusters = parse_dlg_cluster_table(dlg_path)
    except DockingParseError:
        return None, 0
    if not clusters:
        return None, 0
    best = min(clusters, key=lambda cluster: cluster.lowest_energy)
    return best.lowest_energy, len(clusters)


def _progress_sink(
    progress: Callable[..., Any] | None,
) -> Callable[[int, int | None, str], None]:
    """Normalise the optional ``progress`` argument into a no-arg-safe call."""

    def _report(done: int, total: int | None = None, note: str = "") -> None:
        if progress is not None:
            progress(done, total, note)

    return _report


def dock_smiles_array(
    smiles: Sequence[str],
    receptor: str | Path,
    out_dir: str | Path,
    *,
    engine: str = "vina",
    center: Sequence[float] = (0.0, 0.0, 0.0),
    size: Sequence[float] = (22.5, 22.5, 22.5),
    spacing: float = 0.375,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    workers: int = 1,
    names: Sequence[str] | None = None,
    seed: int = 0,
    ga_run: int = 100,
    timeout_s: float = 3600.0,
    keep_folders: bool = True,
    progress: Callable[..., Any] | None = None,
) -> Any:
    """Dock every SMILES of an array against one receptor, simultaneously.

    One campaign folder per molecule is created under ``out_dir``
    (``01_aspirin``, ``02_caffeine``, ...), the ligand is embedded to
    3D PDBQT (Meeko + RDKit, deterministic ``seed``), and the chosen
    engine docks it:

    * ``engine="vina"`` -- one ``vina`` process per ligand (parallel
      across ``workers`` threads), poses written to ``out.pdbqt`` in
      each folder, best affinity parsed from the ``REMARK VINA
      RESULT`` block;
    * ``engine="autodock4"`` -- the full classic stack per folder: the
      receptor is distributed in as ``r.pdbqt``, ``r.gpf`` / ``r.dpf``
      are generated from the box, and ``autogrid4`` + ``autodock4``
      run (also parallel across ``workers``); the best cluster energy
      is parsed from the ``r.dlg`` cluster table.

    Parameters
    ----------
    smiles:
        The SMILES array to dock (invalid entries are reported as
        failed rows, never abort the batch).
    receptor:
        Prepared receptor PDBQT.
    out_dir:
        Campaign output root (created when missing).
    engine:
        ``"vina"`` or ``"autodock4"``.
    center:
        Search box centre ``(x, y, z)`` in Å.
    size:
        Search box edge lengths in Å (Vina); for AutoDock4 the box is
        quantised to ``npts = round(size / spacing) + 1`` grid points.
    spacing:
        AutoDock4 grid spacing (Å).
    exhaustiveness:
        Vina search depth.
    num_modes:
        Vina pose cap.
    workers:
        Concurrent engine subprocesses (subprocesses release the GIL;
        the RDKit ligand preparation itself stays sequential).
    names:
        Optional per-SMILES names (``len(smiles)`` entries); defaults
        to the SMILES slug / trivial-name table.
    seed:
        ETKDG embedding seed (deterministic by default).
    ga_run:
        AutoDock4 GA runs per ligand.
    timeout_s:
        Per-ligand engine subprocess timeout.
    keep_folders:
        Keep the prepared campaign folders (default).  ``False``
        removes them after the run -- only the result table survives.
    progress:
        Optional ``report(done, total, note)`` callback (worker-thread
        safe); called once per finished ligand.

    Returns
    -------
    pandas.DataFrame
        One row per input SMILES: ``index, name, smiles, ok,
        best_score, poses, seconds, folder, output, error`` -- ranked
        best-first among the successful rows, failures at the end.

    Raises
    ------
    DataValidationError
        Empty SMILES array, ``names`` length mismatch, or a bad box.
    DockingError
        Unknown engine, missing receptor, or missing engine binaries
        (surfaced before any work starts).

    Notes
    -----
    Real docking takes minutes to hours per ligand: ``workers``-wide
    parallelism is the difference between "tomorrow" and "tonight".
    """
    import pandas as pd

    engine = engine.strip().lower()
    if engine not in _ENGINES:
        raise DockingError(
            f"unknown engine {engine!r}; valid engines are: "
            + ", ".join(sorted(_ENGINES))
        )
    rows = [str(entry).strip() for entry in smiles]
    rows = [row for row in rows if row and not row.startswith("#")]
    if not rows:
        raise DataValidationError("the SMILES array is empty -- nothing to dock")
    receptor_path = Path(receptor)
    if not receptor_path.is_file():
        raise DockingError(f"receptor PDBQT not found: {receptor_path}")
    if len(center) != 3 or len(size) != 3:
        raise DataValidationError(
            "center and size must be (x, y, z) triples"
        )
    if names is not None and len(names) != len(rows):
        raise DataValidationError(
            f"names has {len(names)} entries for {len(rows)} SMILES -- they "
            "must match"
        )
    binaries = ("vina",) if engine == "vina" else ("autogrid4", "autodock4")
    missing = [name for name in binaries if shutil.which(name) is None]
    if missing:
        raise DockingError(
            f"the {engine} engine needs {', '.join(missing)} on PATH -- "
            "install AutoDock Tools / Vina first"
        )
    workers = max(1, min(workers, len(rows)))
    report = _progress_sink(progress)

    root = Path(out_dir)
    ensure_directory(root)

    # -- stage 1: ligand preparation (sequential; RDKit in-process) --- #
    prepared: list[dict[str, Any]] = []
    total = len(rows)
    for position, (index, smiles_text) in enumerate(zip(range(1, total + 1), rows)):
        name = str(names[index - 1]) if names is not None else _slug_name(
            smiles_text
        )
        name = _SLUG_SAFE.sub("_", name)[:40] or f"ligand_{index}"
        folder_name = f"{index:02d}_{name}"
        folder = root / folder_name
        entry: dict[str, Any] = {
            "index": index,
            "name": name,
            "smiles": smiles_text,
            "ok": False,
            "best_score": None,
            "poses": 0,
            "seconds": 0.0,
            "folder": str(folder),
            "output": "",
            "error": "",
        }
        try:
            ligand = smiles_to_pdbqt(smiles_text, folder, name="l", seed=seed)
            entry["pdbqt"] = str(ligand)
        except Exception as exc:  # noqa: BLE001 - per-ligand failure, keep going
            entry["error"] = f"preparation failed: {exc}"
            _LOG.warning("SMILES %r failed preparation: %s", smiles_text, exc)
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
        prepared.append(entry)
        report(position + 1, total, "preparing ligands")

    dockable = [entry for entry in prepared if entry.get("pdbqt")]

    # -- stage 2: engine setup ----------------------------------------- #
    center_tuple = (float(center[0]), float(center[1]), float(center[2]))
    size_tuple = (float(size[0]), float(size[1]), float(size[2]))
    if engine in _AUTODOCK_ENGINES:
        # the grid box Vina sizes in Å, AutoDock in points
        npts = tuple(
            max(4, round(edge / spacing) + 1) for edge in size_tuple
        )
        box = GridBox(
            center_x=center_tuple[0],
            center_y=center_tuple[1],
            center_z=center_tuple[2],
            npts=npts,  # type: ignore[arg-type]
            spacing=spacing,
        )
        receptor_targets = distribute_receptor(
            receptor_path, root, filename="r.pdbqt", overwrite=True
        )
        _LOG.info(
            "AutoDock4 batch: receptor distributed to %d folder(s)", len(receptor_targets)
        )

    def _dock_one(entry: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        ligand = Path(entry["pdbqt"])
        folder = ligand.parent
        try:
            if engine == "vina":
                out = folder / "out.pdbqt"
                result = run_vina(
                    receptor_path,
                    ligand,
                    out=out,
                    center=center_tuple,
                    size=size_tuple,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes,
                    timeout_s=timeout_s,
                )
                entry["seconds"] = result.seconds
                if not result.ok:
                    entry["error"] = result.error or "vina failed"
                    return entry
                best, poses = _vina_best(out.read_text(encoding="utf-8", errors="ignore"))
                entry["best_score"] = best
                entry["poses"] = poses
                entry["output"] = str(out)
                entry["ok"] = best is not None
                if not entry["ok"]:
                    entry["error"] = "vina produced no scoreable poses"
            else:
                write_gpf(ligand, receptor_file="r.pdbqt", box=box)
                write_dpf(ligand, receptor_file="r.pdbqt", ga_run=ga_run)
                result = run_autodock4(folder, timeout_s=timeout_s)
                entry["seconds"] = result.seconds
                if not result.ok:
                    entry["error"] = result.error or "autodock4 failed"
                    return entry
                dlg = folder / "r.dlg"
                best, poses = _autodock_best(dlg)
                entry["best_score"] = best
                entry["poses"] = poses
                entry["output"] = str(dlg)
                entry["ok"] = best is not None
                if not entry["ok"]:
                    entry["error"] = "no clusters in the DLG output"
        except Exception as exc:  # noqa: BLE001 - engine failure per ligand
            entry["seconds"] = time.perf_counter() - started
            entry["error"] = str(exc)
        return entry

    # -- stage 3: the simultaneous engine run -------------------------- #
    done_count = 0
    if dockable:
        _LOG.info(
            "docking %d ligand(s) with %s (workers=%d)", len(dockable), engine, workers
        )
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_dock_one, entry) for entry in dockable]
                for future in futures:
                    future.result()
                    done_count += 1
                    report(done_count, len(dockable), f"docking ({engine})")
        else:
            for entry in dockable:
                _dock_one(entry)
                done_count += 1
                report(done_count, len(dockable), f"docking ({engine})")

    if not keep_folders:
        shutil.rmtree(root, ignore_errors=True)
        for entry in prepared:
            entry["folder"] = "" if not entry["ok"] else entry["folder"]

    frame = pd.DataFrame(
        [
            {key: entry.get(key, "") for key in (
                "index", "name", "smiles", "ok", "best_score", "poses",
                "seconds", "folder", "output", "error",
            )}
            for entry in prepared
        ]
    )
    ranked = frame.sort_values(
        by=["ok", "best_score"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    ok_count = int(frame["ok"].sum())
    _LOG.info(
        "SMILES array docking finished: %d/%d docked (%s)",
        ok_count,
        len(frame),
        engine,
    )
    return ranked
