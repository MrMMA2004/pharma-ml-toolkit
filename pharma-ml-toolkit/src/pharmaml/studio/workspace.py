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

import hashlib
import json
import os
from dataclasses import asdict, dataclass
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
    # (part8 §14.2 Proposal A, round 6) dataset-level provenance: a
    # monotone per-path content version (same bytes → same version;
    # changed bytes under the same path → +1) and the column
    # signature (names + dtypes hash) the loading page recorded.
    # Bundles/artifacts carry them too when their loader supplies a
    # frame — one registry, one rule.
    content_version: int = 1
    column_signature: str = ""


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


def _column_signature(columns: Any) -> str:
    """The Proposal A column signature: names (+ dtypes when a frame).

    Accepts a pandas DataFrame (names + dtypes — the dtype IS part of
    the dataset's meaning: an int64 target read back as float64 is a
    different dataset to a classifier), any ``columns``-like iterable
    of names (names only), or ``None``/``""`` (no signature, "").  The
    signature is a sha-256 hex prefix; it is descriptive, never a
    compatibility promise.
    """
    if columns is None or (isinstance(columns, str) and not columns):
        return ""
    names: list[str]
    dtypes: list[str] | None = None
    try:
        frame_columns = getattr(columns, "columns", None)
        if frame_columns is not None:  # a DataFrame (or anything shaped like one)
            names = [str(c) for c in columns.columns]
            dtypes = [str(t) for t in columns.dtypes]
        else:
            names = [str(c) for c in columns]
    except Exception:  # noqa: BLE001 - a weird frame must never fail the note
        return ""
    parts = ["|".join(names)]
    if dtypes is not None:
        parts.append("|".join(dtypes))
    joined = "\n".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(joined).hexdigest()[:16]


def note(kind: str, path: str | Path, *, project: str = "", note: str = "",
         with_hash: bool = True,
         columns: Any = None) -> bool:
    """Record one entry (best-effort; returns whether it was written).

    The kinds: ``dataset`` (a table the session opened), ``bundle``
    (a saved model bundle), ``artifact`` (an exported figure/report/
    table).  Duplicate paths are re-recorded (moved to most-recent);
    each kind keeps at most :data:`_MAX_PER_KIND` entries.

    ``columns`` (Proposal A): the loaded frame's column names or a
    ``pandas`` frame/index — recorded as a sha-256 ``column_signature``
    (names + dtypes when a frame is given).  The per-path
    ``content_version`` is monotone: re-recording byte-identical
    content keeps the existing version; changed content under the
    same path bumps it (the OLD entry's version is what run records
    already wrote — they keep it, by design).
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown workspace kind {kind!r}")
    real = Path(path)
    payload = _load()
    previous = next((
        e for e in payload["entries"]
        if e.get("kind") == kind and e.get("path") == str(real)
    ), None)
    digest = _fingerprint(real) if with_hash and real.is_file() else ""
    if previous is not None and digest and previous.get("sha256") == digest:
        version = int(previous.get("content_version", 1))  # same bytes, same version
    elif previous is not None and previous.get("sha256"):
        version = int(previous.get("content_version", 1)) + 1  # changed content
    else:
        version = 1
    entry = WorkspaceEntry(
        kind=kind,
        path=str(real),
        name=real.name,
        recorded=_now(),
        sha256=digest,
        project=project or current_project(),
        note=note,
        content_version=version,
        column_signature=_column_signature(columns),
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
                content_version=int(raw.get("content_version", 1) or 1),
                column_signature=str(raw.get("column_signature", "")),
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


# -- §17G: favorites, export presets, layout preferences ----------------------
# All three ride the SAME workspace file as additive top-level keys:
# an older build reading a newer file ignores the unknown keys, and
# `_load` never rejects a payload for having extras.

def toggle_favorite(path: str | Path) -> bool:
    """Star/unstar one recorded entry by path (returns the NEW state).

    A favorite survives the per-kind recency cap: capping drops the
    OLDEST rows first, so a starred path is kept as long as it stays
    recorded at all.
    """
    real = str(Path(path))
    payload = _load()
    favorites = set(payload.get("favorites", []))
    now_starred = real not in favorites
    if now_starred:
        favorites.add(real)
    else:
        favorites.discard(real)
    payload["favorites"] = sorted(favorites)
    _save(payload)  # best-effort; the new state is returned regardless
    return now_starred


def is_favorite(path: str | Path) -> bool:
    """Whether ``path`` is currently starred."""
    real = str(Path(path))
    return real in set(_load().get("favorites", []))


def favorite_entries(kind: str | None = None) -> list[WorkspaceEntry]:
    """The starred entries (most-recent first)."""
    favorites = set(_load().get("favorites", []))
    return [e for e in entries(kind) if e.path in favorites]


def last_export_folder(template: str = "") -> str:
    """The remembered export folder (per template, or the shared one)."""
    payload = _load()
    folders = payload.get("export_folders", {})
    if not isinstance(folders, dict):
        return ""
    if template:
        return str(folders.get(template, ""))
    return str(folders.get("_shared", ""))


def remember_export_folder(folder: str | Path, template: str = "") -> bool:
    """Persist the export folder for next time (best-effort)."""
    payload = _load()
    folders = payload.get("export_folders", {})
    if not isinstance(folders, dict):
        folders = {}
    folders[template or "_shared"] = str(Path(folder))
    payload["export_folders"] = folders
    return _save(payload)


def layout_preference(key: str, default: str = "") -> str:
    """One named layout preference ("" when never set)."""
    prefs = _load().get("layout", {})
    if not isinstance(prefs, dict):
        return default
    return str(prefs.get(key, default))


def set_layout_preference(key: str, value: str) -> bool:
    """Set one named layout preference (best-effort)."""
    payload = _load()
    prefs = payload.get("layout", {})
    if not isinstance(prefs, dict):
        prefs = {}
    prefs[str(key)] = str(value)
    payload["layout"] = prefs
    return _save(payload)
