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

import hashlib
import json
import os
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
from pharmaml.docking.workflow import engine_argv, run_autodock4, run_vina
from pharmaml.exceptions import DataValidationError, DockingError, PersistenceError
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
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O": "naproxen",
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


def _entries_frame(prepared: list[dict[str, Any]]) -> Any:
    """The result frame from per-ligand entries: one row per input
    SMILES, ranked best-first among the successes, failures at the
    end (the shared shape of the final and the §13 partial frame)."""
    import pandas as pd

    frame = pd.DataFrame(
        [
            {key: entry.get(key, "") for key in (
                "index", "name", "smiles", "ok", "best_score", "poses",
                "seconds", "folder", "output", "error",
            )}
            for entry in prepared
        ]
    )
    return frame.sort_values(
        by=["ok", "best_score"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


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
    engine_dir: str | Path | None = None,
    progress: Callable[..., Any] | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    checkpoint_every: int = 1,
    checkpoint_resume: bool = True,
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
        quantised to ``npts = round(size / spacing) + 1`` as the legacy
        GPF directive. Bundled AutoGrid4 was observed to interpret even npts
        as intervals, yielding one more grid point; the actual AutoGrid span
        can therefore differ from the requested Vina size. Inspect the grid
        log and specify geometry deliberately rather than assuming equivalence.
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
    engine_dir:
        Folder holding the engine binaries (``vina`` / ``vina.exe`` /
        ``autogrid4.exe`` / ``autodock4.exe``), searched before PATH --
        the Windows ``.exe`` builds the toolkit ships in ``engines/``.
        progress:
            Optional ``report(done, total, note)`` callback (worker-thread
            safe); called once per finished ligand.
        checkpoint_dir:
            (part5-continuation round 10, §13) When given, the batch is a
            resumable §13 operation: a v2-envelope checkpoint is written
            every ``checkpoint_every`` finished ligands (and at each stage
            boundary) carrying the per-ligand completion ledger -- every
            finished entry's full row.  A later call with the same
            fingerprint resumes: finished ligands are NEVER re-run and
            their rows are preserved verbatim (a real docking engine is
            stochastic by nature; "preserve", not re-execute).  A
            cooperative stop (``progress`` returning ``False``) raises
            :class:`~pharmaml.exceptions.JobCancelled` with the partial
            result frame on ``partial_results``.  Keep this OUTSIDE
            ``out_dir`` when ``keep_folders=False`` (the campaign root is
            removed then).
        checkpoint_every:
            Save every N finished ligands (default 1; >= 1 enforced).
        checkpoint_resume:
            Resume from the newest valid checkpoint (default).  ``False``
            runs fresh but still writes checkpoints (the fresh-run twin
            protocol of the proof suite).

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
    # Check launch prerequisites before ligand preparation. Finding a Windows
    # executable alone does not establish that Wine is available or functional.
    for name in binaries:
        engine_argv(name, engine_dir)
    workers = max(1, min(workers, len(rows)))
    report = _progress_sink(progress)

    root = Path(out_dir)
    ensure_directory(root)

    # -- (round 10, §13) the resumable docking batch ---------------------- #
    #
    # The resume unit is the LIGAND: the per-ligand entry dict is the
    # complete state (score, poses, seconds, output/error, folder), so
    # the checkpoint is the ledger of finished entries.  Finished
    # ligands are never re-run -- their rows are preserved verbatim
    # (a real engine is stochastic by nature; the §13 "preserve"
    # policy, not re-execute).  The fingerprint closes both gaps the
    # R9 audit flagged for this round: the FULL smiles list + names
    # (the dataset) and the receptor PDBQT content checksum (the
    # "environment"), besides every engine knob.
    if (
        not isinstance(checkpoint_every, int)
        or isinstance(checkpoint_every, bool)
        or checkpoint_every < 1
    ):
        raise ValueError(
            f"checkpoint_every must be a positive int, got {checkpoint_every!r}"
        )

    manager: Any = None
    spec: Any = None
    config_fp = ""
    ledger: dict[str, dict[str, Any]] = {}

    def _ledger_key(entry: dict[str, Any]) -> str:
        # index + smiles: position AND molecule (a reordered list is a
        # different dataset and refuses to resume)
        return f"{entry['index']}:{entry['smiles']}"

    def _save_checkpoint(step: int, note: str) -> None:
        if manager is None or spec is None:
            return
        try:
            manager.save(
                step=step,
                total=total,
                state={
                    "config_fp": config_fp,
                    "ledger": ledger,
                    "prepared_count": len(prepared),
                },
                note=note,
                operation=spec,
            )
        except (PersistenceError, OSError, TypeError, ValueError) as exc:
            _LOG.warning(
                "checkpoint write failed at %s (%s); continuing without it",
                note,
                exc,
            )

    if checkpoint_dir is not None:
        from pharmaml.utils.checkpoint_schema import OperationSpec
        from pharmaml.utils.checkpoints import CheckpointManager

        try:
            manager = CheckpointManager(
                checkpoint_dir, name="docking-batch"
            )
            spec = OperationSpec(
                operation_type="docking_batch",
                operation_id=f"batch-{Path(receptor_path).name}",
            )
        except PersistenceError as exc:
            _LOG.warning(
                "checkpoint home unusable (%s); docking without checkpoints",
                exc,
            )
            manager = None
            spec = None

        receptor_digest = hashlib.sha256(
            receptor_path.read_bytes()
        ).hexdigest()
        config_fp = hashlib.sha256(
            json.dumps(
                {
                    "engine": engine,
                    "center": [float(c) for c in center],
                    "size": [float(s) for s in size],
                    "spacing": spacing,
                    "exhaustiveness": exhaustiveness,
                    "num_modes": num_modes,
                    "workers": workers,
                    "seed": seed,
                    "ga_run": ga_run,
                    "timeout_s": timeout_s,
                    "keep_folders": keep_folders,
                    "smiles": rows,
                    "names": list(names) if names is not None else None,
                    "receptor_sha256": receptor_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        if manager is not None and checkpoint_resume:
            try:
                latest = manager.latest()
            except OSError:  # pragma: no cover - scan failure
                latest = None
                _LOG.warning("checkpoint scan failed; starting fresh")
            if latest is not None and latest.resumable:
                st = latest.state
                if st.get("config_fp") != config_fp:
                    _LOG.warning(
                        "docking-batch checkpoint at step %d is for a "
                        "different dataset/config/receptor; starting fresh "
                        "(exact restore or nothing)",
                        latest.step,
                    )
                else:
                    try:
                        ledger = {
                            str(k): dict(v)
                            for k, v in st["ledger"].items()
                        }
                        _LOG.info(
                            "docking batch resumed from checkpoint step %d "
                            "(%d ligand(s) already finished)",
                            latest.step,
                            len(ledger),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        _LOG.warning(
                            "checkpoint payload unreadable (%s); starting "
                            "fresh",
                            exc,
                        )
                        ledger = {}

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
        # (round 10, §13) a finished ligand's row is preserved verbatim:
        # neither preparation nor engine run repeats.  A resume with
        # keep_folders=True re-lists the folder path from the ledger so
        # the row is byte-identical; with keep_folders=False the folders
        # were already swept (folder/output read back as the swept "" /
        # preserved path recorded at sweep time).
        prior = ledger.get(_ledger_key(entry))
        if prior is not None:
            entry.update(prior)
            prepared.append(entry)
            report(position + 1, total, "resumed ligands")
            continue
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

    # a preparation failure is FINAL for that ligand (the row is
    # complete with its error) -- it goes into the ledger too
    for entry in prepared:
        key = _ledger_key(entry)
        if key not in ledger and (entry.get("pdbqt") is None):
            ledger[key] = dict(entry)

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
                    engine_dir=engine_dir,
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
                result = run_autodock4(folder, timeout_s=timeout_s, engine_dir=engine_dir)
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
    # (round 10, §13) finished ligands drop out of the queue; the ledger
    # fills as entries complete; checkpoints land every
    # checkpoint_every finished ligands; a cooperative stop raises
    # JobCancelled with the partial frame.  The threaded path runs in
    # waves of `workers` so a stop request lands at a clean barrier
    # (never mid-wave -- submitted subprocesses always finish).
    done_count = 0
    pending = [
        entry for entry in dockable if _ledger_key(entry) not in ledger
    ]
    already = len(dockable) - len(pending)
    if already:
        _LOG.info(
            "%d ligand(s) already finished (checkpoint); docking the "
            "remaining %d",
            already,
            len(pending),
        )
    cancelled = False

    def _finish_one(entry: dict[str, Any]) -> None:
        nonlocal done_count
        ledger[_ledger_key(entry)] = dict(entry)
        done_count += 1
        report(already + done_count, len(dockable), f"docking ({engine})")
        if (
            manager is not None
            and spec is not None
            and done_count % checkpoint_every == 0
        ):
            _save_checkpoint(
                already + done_count,
                f"docking batch {already + done_count}/{len(dockable)}",
            )

    if pending:
        _LOG.info(
            "docking %d ligand(s) with %s (workers=%d)",
            len(pending),
            engine,
            workers,
        )
        if workers > 1:
            wave_start = 0
            while wave_start < len(pending) and not cancelled:
                wave = pending[wave_start : wave_start + workers]
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(_dock_one, entry) for entry in wave
                    ]
                    for future in futures:
                        future.result()
                for entry in wave:
                    _finish_one(entry)
                wave_start += len(wave)
                if (
                    progress is not None
                    and wave_start < len(pending)
                    and progress(
                        already + done_count,
                        len(dockable),
                        f"docking ({engine})",
                    )
                    is False
                ):
                    cancelled = True
        else:
            for entry in pending:
                if (
                    progress is not None
                    and progress(
                        already + done_count,
                        len(dockable),
                        f"docking ({engine})",
                    )
                    is False
                ):
                    cancelled = True
                    break
                _dock_one(entry)
                _finish_one(entry)

    if manager is not None and spec is not None and not cancelled:
        _save_checkpoint(
            len(dockable) if dockable else total,
            f"docking batch complete ({len(ledger)} ledger entr(ies))",
        )

    if cancelled:
        from pharmaml.exceptions import JobCancelled

        partial_frame = _entries_frame(prepared)
        raise JobCancelled(
            f"docking batch cancelled after {already + done_count}/"
            f"{len(dockable)} ligand(s) -- stopped at a checkpoint by the "
            "caller",
            partial_results=partial_frame,
        )

    if not keep_folders:
        shutil.rmtree(root, ignore_errors=True)
        for entry in prepared:
            entry["folder"] = "" if not entry["ok"] else entry["folder"]

    ranked = _entries_frame(prepared)
    frame = ranked
    ok_count = int(frame["ok"].sum())
    _LOG.info(
        "SMILES array docking finished: %d/%d docked (%s)",
        ok_count,
        len(frame),
        engine,
    )
    return ranked
