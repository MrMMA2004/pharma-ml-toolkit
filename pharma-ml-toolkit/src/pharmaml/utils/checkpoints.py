"""Durable checkpoints for long-running jobs (MASTER_PROMPT Part-3 §20).

A checkpoint is a small JSON file a compute loop writes after each
completed unit of work (one feature family, one docking batch, one
generative chunk).  The file records *how far* the job got (``step``
of ``total``) and enough *state* (the rows collected so far) for a
later run of the SAME job -- same data, same settings -- to skip the
finished units and resume exactly where the earlier one stopped, even
from a different process, days later.

Why JSON and not pickle: the checkpoint file is data the toolkit
promises to be able to re-read across versions and machines, and a
malicious ``.json`` file cannot execute anything on load.  States that
need binary payloads (arrays, models) are the caller's business -- the
manager records metadata and small JSON-serialisable state; large
artefacts belong in the caller's own files, referenced by path in the
state dict.

The contract, in order of importance:

1. **Atomic** -- every save is write-to-temp, ``fsync``, ``os.replace``
   (crash-safe: the previous checkpoint survives a kill mid-write),
   then the directory entry is ``fsync``ed too.
2. **Self-verifying** -- each file carries a SHA-256 over its canonical
   content; a truncated or tampered file is QUARANTINED (renamed
   ``*.json.corrupt``) and the manager falls back to the previous
   checkpoint instead of failing the resuming run.
3. **Rotated** -- only the newest ``keep`` checkpoints survive; a
   40-hour job checkpoints thousands of times without filling the disk.
4. **Never fatal on write** -- callers checkpoint inside loops whose
   failure would destroy minutes of work; ``save()`` failures are the
   caller's to catch, and the benchmark wiring in
   :mod:`pharmaml.ml.training` degrades them to log warnings.

Examples
--------
>>> from pharmaml.utils.checkpoints import CheckpointManager
>>> mgr = CheckpointManager(_tmp_dir(), "demo", keep=2)
>>> first = mgr.save(step=1, total=3, state={"rows": [1]}, note="unit 1")
>>> first.step, first.total, first.note
(1, 3, 'unit 1')
>>> mgr.save(step=2, total=3, state={"rows": [1, 2]}, note="unit 2").step
2
>>> info = mgr.latest()
>>> info.step, info.state["rows"]
(2, [1, 2])
>>> mgr.clear() >= 1
True
>>> mgr.latest() is None
True
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharmaml.exceptions import PersistenceError
from pharmaml.utils.checkpoint_schema import (
    ENVELOPE_KEY,
    OperationSpec,
)
from pharmaml.utils.checkpoint_schema import (
    build_envelope as _build_envelope,
)

SCHEMA = "pharmaml-checkpoint/1"


@dataclass(frozen=True)
class CheckpointInfo:
    """One validated checkpoint record (the contents of one JSON file)."""

    step: int
    total: int | None
    state: dict[str, Any]
    note: str
    saved_at: str
    elapsed_s: float
    path: Path
    integrity: str
    envelope: dict[str, Any] | None = None

    @property
    def resumable(self) -> bool:
        """True only when the §13 v2 envelope is present and intact.

        Legacy checkpoints (envelope is ``None``) are readable and their
        state is returned unchanged, but they are not resumable under
        the Part-5 §13 contract.
        """
        return self.envelope is not None


def default_directory(name: str) -> Path:
    """The GUI's per-job checkpoint home: ``~/.pharmaml/checkpoints/<name>/``.

    The Studio pages use this so Stop keeps its progress for the next
    Start without any configuration.  Tests and scripts pass their own
    directory to :class:`CheckpointManager` instead.
    """
    from pharmaml.utils.io import safe_filename

    return Path.home() / ".pharmaml" / "checkpoints" / safe_filename(name, "checkpoints")


def _canonical(record: dict[str, Any]) -> bytes:
    """The exact bytes the integrity digest covers (sorted keys, UTF-8)."""
    return json.dumps(
        record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


class CheckpointManager:
    """Write, validate, rotate and clear checkpoint files for one job.

    Parameters
    ----------
    directory:
        Where the checkpoint files live (created on first save).
    name:
        The job's checkpoint stem; files are
        ``<name>.step-NNNNNN.json``.  Must be a plain filename (no
        path separators) so one directory cannot be escaped.
    keep:
        How many of the newest checkpoints to retain (at least 1).
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        name: str,
        *,
        keep: int = 3,
    ) -> None:
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
            raise PersistenceError(
                f"keep must be a positive int (how many checkpoints to "
                f"retain), got {keep!r}"
            )
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            or name != name.strip()
        ):
            raise PersistenceError(
                f"checkpoint name must be a plain filename without path "
                f"separators, got {name!r}"
            )
        self._dir = Path(directory)
        self._name = name
        self._keep = keep
        self._t0 = time.monotonic()

    # -- properties ------------------------------------------------------------
    @property
    def directory(self) -> Path:
        """Where the checkpoint files live."""
        return self._dir

    @property
    def name(self) -> str:
        """The job's checkpoint stem."""
        return self._name

    # -- paths -------------------------------------------------------------------
    def _path(self, step: int) -> Path:
        return self._dir / f"{self._name}.step-{step:06d}.json"

    def _files(self) -> list[Path]:
        """This job's checkpoint files, NEWEST STEP first."""
        if not self._dir.is_dir():
            return []
        return sorted(self._dir.glob(f"{self._name}.step-*.json"), reverse=True)

    # -- writing -----------------------------------------------------------------
    def save(
        self,
        step: int,
        total: int | None = None,
        state: dict[str, Any] | None = None,
        note: str = "",
        *,
        operation: OperationSpec | None = None,
        inputs: tuple[Path | str, ...] | list[Path | str] = (),
    ) -> CheckpointInfo:
        """Atomically persist one checkpoint and rotate older ones.

        With ``operation`` (Part-5 §13), the record additionally carries
        the v2 resumable-operation envelope: registered operation class,
        input fingerprints, implementation fingerprint, toolkit version and
        ``resume_unit == step``.  Without it the save is a legacy checkpoint
        — identical to before, and reported as not §13-resumable.

        Raises
        ------
        PersistenceError
            If ``step`` is not a non-negative int, ``state`` is not a
            dict, the state claims the reserved ``envelope`` key, the
            operation is not registry-backed, or the state is not
            JSON-serialisable -- the caller decides whether that is fatal
            (the toolkit's own wiring degrades it to a log warning and
            keeps computing).
        """
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise PersistenceError(
                f"checkpoint step must be a non-negative int, got {step!r}"
            )
        if state is None:
            state = {}
        if not isinstance(state, dict):
            raise PersistenceError(
                f"checkpoint state must be a dict, got {type(state).__name__}"
            )
        if ENVELOPE_KEY in state:
            raise PersistenceError(
                f"state key {ENVELOPE_KEY!r} is reserved for the §13 "
                "envelope; pass operation= to save() instead"
            )
        envelope: dict[str, Any] | None = None
        if operation is not None:
            envelope = _build_envelope(operation, step=step, inputs=inputs)
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "name": self._name,
            "step": step,
            "total": total,
            "note": str(note),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "elapsed_s": round(time.monotonic() - self._t0, 3),
            "state": state,
        }
        if envelope is not None:
            record[ENVELOPE_KEY] = envelope
        try:
            record["integrity"] = hashlib.sha256(_canonical(record)).hexdigest()
            blob = json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                indent=1,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise PersistenceError(
                "checkpoint state must be JSON-serialisable "
                f"({exc}) -- store large/binary artefacts in the "
                "caller's own files and reference them by path"
            ) from exc
        self._dir.mkdir(parents=True, exist_ok=True)
        final = self._path(step)
        tmp = self._dir / f"{final.name}.tmp-{os.getpid()}"
        with open(tmp, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, final)  # atomic: readers see all-or-nothing
        if os.name == "posix":
            # POSIX-only durability: fsyncing the directory handle makes
            # the rename itself durable. Windows cannot open a directory
            # with plain O_RDONLY (EACCES on every os.open of a dir) and
            # offers no directory fsync; NTFS metadata journaling carries
            # the rename's durability there instead -- the same
            # platform split the freeze tooling already documents. (The
            # Windows runner proved the crash: run 44487380952 step 12.)
            dir_fd = os.open(self._dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        self._rotate()
        return self._info(record, final)

    def _rotate(self) -> int:
        """Delete every checkpoint beyond the newest ``keep``; returns count."""
        files = self._files()
        removed = 0
        for stale in files[self._keep :]:
            try:
                stale.unlink()
                removed += 1
            except OSError:  # pragma: no cover -- racing reader, keep going
                pass
        return removed

    # -- reading -----------------------------------------------------------------
    def _info(self, record: dict[str, Any], path: Path) -> CheckpointInfo:
        return CheckpointInfo(
            step=int(record["step"]),
            total=record["total"],
            state=record["state"],
            note=str(record.get("note", "")),
            saved_at=str(record.get("saved_at", "")),
            elapsed_s=float(record.get("elapsed_s", 0.0)),
            path=path,
            integrity=str(record.get("integrity", "")),
            envelope=record.get(ENVELOPE_KEY),
        )

    def _validate(self, path: Path) -> CheckpointInfo:
        """Read and verify one checkpoint file.

        Raises ``PersistenceError`` (or ``json.JSONDecodeError`` /
        ``OSError``) for any file that is unreadable, not our schema,
        or fails its integrity digest.
        """
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise PersistenceError(f"{path.name}: unknown checkpoint schema")
        if record.get("name") != self._name:
            raise PersistenceError(f"{path.name}: foreign checkpoint name")
        digest = record.pop("integrity", None)
        if digest != hashlib.sha256(_canonical(record)).hexdigest():
            raise PersistenceError(f"{path.name}: integrity digest mismatch")
        return self._info(record, path)

    def _quarantine(self, path: Path) -> Path:
        """Rename a corrupt file out of the resume scan's way."""
        target = path.with_name(path.name + ".corrupt")
        counter = 1
        while target.exists():
            target = path.with_name(f"{path.name}.corrupt-{counter}")
            counter += 1
        try:
            path.rename(target)
        except OSError:  # pragma: no cover -- unwritable dir, leave the file
            target = path
        return target

    def list(self) -> list[CheckpointInfo]:
        """Every VALID checkpoint, newest step first.

        Invalid files (truncated, tampered, foreign) are quarantined to
        ``*.json.corrupt`` on sight and skipped, so a damaged newest
        checkpoint degrades the resume to the previous one instead of
        failing the run.
        """
        valid: list[CheckpointInfo] = []
        for path in self._files():
            try:
                valid.append(self._validate(path))
            except PersistenceError:
                self._quarantine(path)
            except (ValueError, OSError):
                self._quarantine(path)
        return valid

    def latest(self) -> CheckpointInfo | None:
        """The newest valid checkpoint, or ``None`` when there is none."""
        checkpoints = self.list()
        return checkpoints[0] if checkpoints else None

    def clear(self) -> int:
        """Delete every checkpoint and quarantine file; returns the count."""
        removed = 0
        if not self._dir.is_dir():
            return removed
        for path in list(self._dir.glob(f"{self._name}.step-*.json*")):
            try:
                path.unlink()
                removed += 1
            except OSError:  # pragma: no cover
                pass
        return removed


def _tmp_dir() -> str:
    """Doctest helper: a throwaway directory for the module examples."""
    import tempfile

    return tempfile.mkdtemp(prefix="pharmaml-checkpoints-doctest-")
