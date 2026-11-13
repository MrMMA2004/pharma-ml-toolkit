"""The shared project/session workspace (part6 §8, Phase 4).

A file-based registry of the session's real inputs and outputs — the
datasets opened, the model bundles saved, the runs recorded, the
artifacts exported — so cards can share state without behaving like
unrelated standalone applications.

§8 rules this module follows:

* file-based persistence under one JSON file
  (``$PHARML_STUDIO_WORKSPACE`` or ``~/.pharmaml/studio-workspace.json``)
  — no database, because file-based persistence is sufficient and the
  directive forbids forcing one in;
* working with independent files stays first-class: a project name is
  OPTIONAL (``current_project`` may be empty forever) and every entry
  records what it is regardless;
* entries are recorded by the pages that produce real results, never
  fabricated here: ``note_dataset`` is called by the Data Explorer's
  load path, ``note_bundle`` by the bundle save path, ``note_artifact``
  by export paths — each with the REAL path and, where cheap, a sha-256
  via the canonical ``input_fingerprint`` helper;
* best-effort throughout: an unwritable home loses the side-registry
  but never the user's actual results (the same contract as
  ``pharmaml.studio.runs``).

The registry is capped (most-recent first) so a long session cannot
grow the file without bound.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_WORKSPACE_ENV = "PHARML_STUDIO_WORKSPACE"
_SCHEMA = "studio-workspace/1"
_MAX_PER_KIND = 24
_LOG = None  # lazily: logging import stays out of the hot import path

#: kind -> (label for the GUI, what makes an entry "the same")
_KINDS = ("dataset", "bundle", "artifact")


def workspace_path() -> Path:
    """The workspace file: env override or ``~/.pharmaml/studio-workspace.json``."""
    override = os.environ.get(_WORKSPACE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".pharmaml" / "studio-workspace.json"


@dataclass
class WorkspaceEntry:
    """One recorded input/output the session can share across cards."""

    kind: str            # "dataset" | "bundle" | "artifact"
    path: str            # the real filesystem path
    name: str            # the basename the GUI shows
    recorded: str        # ISO-8601 UTC
    sha256: str = ""     # from the canonical fingerprint helper ("" if skipped)
    project: str = ""    # optional §8 project tag — never required
    note: str = ""       # free-form provenance from the recording page


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load() -> dict[str, Any]:
    path = workspace_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": _SCHEMA, "entries": [], "current_project": ""}
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        return {"schema": _SCHEMA, "entries": [], "current_project": ""}
    return payload


def _save(payload: dict[str, Any]) -> bool:
    path = workspace_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        return False  # best-effort: the user's results are unaffected
    return True


def _fingerprint(path: Path) -> str:
    """The canonical sha-256 for a single file ("" when unreadable)."""
    from pharmaml.utils.checkpoint_schema import _sha256_file

    try:
        return _sha256_file(path)
    except OSError:
        return ""


def note(kind: str, path: str | Path, *, project: str = "", note: str = "",
         with_hash: bool = True) -> bool:
    """Record one entry (best-effort; returns whether it was written).

    The kinds: ``dataset`` (a table the session opened), ``bundle``
    (a saved model bundle), ``artifact`` (an exported figure/report/
    table).  Duplicate paths are re-recorded (moved to most-recent);
    each kind keeps at most :data:`_MAX_PER_KIND` entries.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown workspace kind {kind!r}")
    real = Path(path)
    entry = WorkspaceEntry(
        kind=kind,
        path=str(real),
        name=real.name,
        recorded=_now(),
        sha256=_fingerprint(real) if with_hash and real.is_file() else "",
        project=project or current_project(),
        note=note,
    )
    payload = _load()
    entries = [e for e in payload["entries"] if not (
        e.get("kind") == kind and e.get("path") == entry.path
    )]
    entries.insert(0, asdict(entry))
    payload["entries"] = _cap_per_kind(entries)
    return _save(payload)


def _cap_per_kind(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the newest _MAX_PER_KIND entries of each kind."""
    kept: list[dict[str, Any]] = []
    counts = {kind: 0 for kind in _KINDS}
    for entry in entries:  # entries are most-recent first
        kind = entry.get("kind", "")
        if kind in counts and counts[kind] >= _MAX_PER_KIND:
            continue
        if kind in counts:
            counts[kind] += 1
        kept.append(entry)
    return kept


def entries(kind: str | None = None) -> list[WorkspaceEntry]:
    """The recorded entries, most-recent first (optionally one kind)."""
    payload = _load()
    out: list[WorkspaceEntry] = []
    for raw in payload.get("entries", []):
        if kind is not None and raw.get("kind") != kind:
            continue
        try:
            out.append(WorkspaceEntry(
                kind=str(raw["kind"]),
                path=str(raw["path"]),
                name=str(raw.get("name", Path(str(raw["path"])).name)),
                recorded=str(raw.get("recorded", "")),
                sha256=str(raw.get("sha256", "")),
                project=str(raw.get("project", "")),
                note=str(raw.get("note", "")),
            ))
        except (KeyError, TypeError):
            continue  # a corrupt row is skipped, never guessed
    return out


def current_project() -> str:
    """The optional project tag ("" when the user never set one)."""
    return str(_load().get("current_project", ""))


def set_current_project(name: str) -> bool:
    """Set (or clear, with "") the optional project tag."""
    payload = _load()
    payload["current_project"] = str(name)
    return _save(payload)


def summary_counts() -> dict[str, int]:
    """Per-kind entry counts for the Dashboard's workspace card."""
    payload = _load()
    counts = {kind: 0 for kind in _KINDS}
    for raw in payload.get("entries", []):
        kind = raw.get("kind")
        if kind in counts:
            counts[kind] += 1
    return counts
