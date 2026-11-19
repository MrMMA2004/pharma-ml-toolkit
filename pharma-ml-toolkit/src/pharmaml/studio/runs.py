"""Persisted Studio run records — part5 §11's "actual saved runs" home.

One JSON file per finished Studio job, under a per-family directory::

    ~/.pharmaml/studio-runs/<family>/<UTC-timestamp>-<slug>.json

The record is deliberately tiny and stdlib-only:

.. code-block:: json

    {
      "schema": "pharmaml-studio-run/1",
      "family": "dl-training",
      "label": "NumpyMLPRegressor on pIC50",
      "created": "2026-11-12T09:41:02+00:00",
      "toolkit_version": "1.2.174",
      "summary": {"epochs_run": 40, "history": [{"epoch": 1, ...}, ...]}
    }

Contract rules (pinned by ``tests/test_studio_evidence.py``):

* :func:`record_run` is **best-effort and never raises** — a full disk or an
  unwritable home degrades to ``None`` plus a warning log, exactly like the
  §20 checkpoint writes it mirrors.  A failed record must never fail the
  job that produced the results.
* Files are written atomically (temp file + ``os.replace``) with UTC-second
  filenames plus microseconds so two same-label records never collide.
* Readers skip anything unreadable, foreign-schema, or wrongly nested —
  the honest answer to a corrupt record is its absence, never a guess.
* ``$PHARML_STUDIO_RUNS_DIR`` overrides the root (tests; never set it in
  production code).

The §11 families written today: ``ml-evaluation`` (CV benchmark boards),
``dl-training`` (training histories), ``generative`` (SA batches and Pareto
fronts), ``docking`` (campaign scans), ``pkpd`` (regimen simulations) and
``viz-export`` (Image Lab and MD figure exports).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any

from pharmaml.logger import get_logger

__all__ = [
    "RUNS_DIR_ENV",
    "RUN_SCHEMA",
    "latest_runs",
    "record_run",
    "run_families",
    "runs_root",
]

#: The record schema stamp — readers refuse anything else.
RUN_SCHEMA = "pharmaml-studio-run/1"

#: Environment override for the runs root (tests only).
RUNS_DIR_ENV = "PHARML_STUDIO_RUNS_DIR"

_LOG = get_logger("pharmaml.studio.runs")

_SLUG_DASH = re.compile(r"[^a-z0-9]+")


def runs_root() -> Path:
    """The runs home: ``$PHARML_STUDIO_RUNS_DIR`` or ``~/.pharmaml/studio-runs``."""
    override = os.environ.get(RUNS_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".pharmaml" / "studio-runs"


def slugify(text: str, fallback: str = "run") -> str:
    """A filesystem-safe lowercase slug (``"My Run #2!" -> "my-run-2"``).

    >>> slugify("family benchmark · pIC50 (5-fold CV)")
    'family-benchmark-pic50-5-fold-cv'
    >>> slugify("!!!"), slugify("")
    ('run', 'run')
    """
    slug = _SLUG_DASH.sub("-", text.strip().lower()).strip("-")
    return slug or fallback


def record_run(
    family: str, label: str, summary: dict[str, Any] | None = None
) -> Path | None:
    """Persist one finished job; best-effort, never raises.

    Returns the written path, or ``None`` when the record could not be
    written (unwritable home, non-JSON-able summary value).  The caller's
    results are already real — losing the side-record must never lose them.
    """
    root = runs_root() / slugify(family, "run")
    created = _dt.datetime.now(_dt.timezone.utc)
    payload = {
        "schema": RUN_SCHEMA,
        "family": family,
        "label": str(label),
        "created": created.isoformat(timespec="microseconds"),
        "toolkit_version": _toolkit_version(),
        "summary": dict(summary or {}),
    }
    # r297: Windows' Python 3.12 clock (time.time() over
    # GetSystemTimeAsFileTime) resolves only to ~15.6 ms, so two records
    # written inside one tick stamp IDENTICAL names and the second
    # os.replace() silently erases the first (run 44496937643: the v1
    # record vanished from latest_runs -- [2] where [2, 1] was pinned).
    # Bumping the stamp's microsecond tail until the name is fresh keeps
    # the filename a strictly monotonic ordering key while staying inside
    # the clock's own resolution (the payload keeps the true timestamp).
    slug = slugify(label)
    stamp = created.strftime("%Y%m%dT%H%M%S%f")
    path = root / f"{stamp}-{slug}.json"
    while path.exists():
        created += _dt.timedelta(microseconds=1)
        stamp = created.strftime("%Y%m%dT%H%M%S%f")
        path = root / f"{stamp}-{slug}.json"
    temp = path.with_name(path.name + ".tmp")
    try:
        root.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, sort_keys=True, default=_jsonable)
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    except (OSError, TypeError, ValueError) as exc:
        _LOG.warning("Studio run record for %r could not be written: %s", family, exc)
        try:  # a half-written temp file must not linger as a fake record
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return path


def latest_runs(family: str, limit: int = 5) -> list[dict[str, Any]]:
    """The newest parsed records for ``family`` (newest first).

    Unreadable or foreign-schema files are skipped — a corrupt record is
    reported by its absence, never by a guessed payload.
    """
    root = runs_root() / slugify(family, "run")
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("schema") != RUN_SCHEMA:
            continue
        records.append(payload)
        if len(records) >= limit:
            break
    return records


def run_families() -> dict[str, int]:
    """Record counts per family directory that exists on disk."""
    root = runs_root()
    if not root.is_dir():
        return {}
    counts: dict[str, int] = {}
    for family_dir in sorted(root.iterdir()):
        if family_dir.is_dir():
            n = len(list(family_dir.glob("*.json")))
            if n:
                counts[family_dir.name] = n
    return counts


def _toolkit_version() -> str:
    try:
        import pharmaml

        return str(pharmaml.__version__)
    except Exception:  # noqa: BLE001 - the stamp is cosmetic, never fatal
        return ""


def _jsonable(obj: Any) -> Any:
    """Coerce numpy/pandas scalars and arrays into plain JSON types."""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        # as_posix (r297): run records are cross-platform stable
        # documents -- str() would freeze Windows backslashes into
        # them (the studio-evidence pin read '\\tmp\\x.png' where
        # '/tmp/x.png' was recorded); POSIX renderings are identical
        # to str() so no Linux record changes.
        return obj.as_posix()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")
