"""SQLite persistence for the local backend (part11 Phase Eleven).

The part10 stores were session-scoped by design — the WinUI host
re-uploaded everything after a restart.  part11 replaces that contract:

* SQLite is the persistent local application database (the Python
  backend owns it; the frontend never touches the file);
* large payloads (dataset source bytes, pickled models, artifact
  bytes) live in the backend-managed **file vault** — SQLite stores
  identifiers, paths, hashes and metadata (§16.2: no big BLOBs by
  default);
* the database lives in a writable user-data directory —
  ``%LOCALAPPDATA%\\PharmaML Studio`` on Windows, the XDG data home
  elsewhere — never inside the installation directory (§16.5);
* schema changes go through **versioned migrations**
  (``schema_migrations`` ledger, idempotent, transactional);
* connections are opened per operation with ``foreign_keys``,
  ``journal_mode=WAL`` and a busy timeout (§16.4's
  connection-per-worker *or* -per-operation strategy — no connection
  is ever shared across threads).

Activation contract (honest and test-friendly):

* ``create_app()`` without ``data_dir`` stays **ephemeral** — the
  documented library/test default, fresh per app, nothing written to
  the user's disk (the 178 existing api tests rely on it);
* ``create_app(data_dir=...)`` (or the ``PHARMAML_STUDIO_DATA``
  environment variable) turns persistence on at that directory —
  stores write through, ``create_app`` recovers the prior registry,
  jobs keep their history across restarts;
* the real host entry (``python -m pharmaml.api`` /
  :func:`pharmaml.api.server.serve`) always passes the platform
  user-data directory, so the *shipped application* is durable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .stores import json_safe

if TYPE_CHECKING:  # pragma: no cover - annotation-only import
    import zipfile

__all__ = [
    "APP_DIR_NAME",
    "FileVault",
    "Persistence",
    "default_data_dir",
    "resolve_data_dir",
]

#: The per-application directory name under the platform data home.
APP_DIR_NAME = "PharmaML Studio"

#: SQLite pragras applied to EVERY connection (§16.4 safety rules).
_BUSY_TIMEOUT_MS = 5000

#: Migration 1 — the base registry schema (v1).
_V1_SCHEMA = [
    """
    CREATE TABLE datasets (
        dataset_id     TEXT PRIMARY KEY,
        name           TEXT NOT NULL,
        source_filename TEXT NOT NULL,
        size_bytes     INTEGER NOT NULL,
        sha256         TEXT NOT NULL,
        target         TEXT,
        task           TEXT,
        n_rows         INTEGER,
        n_columns      INTEGER,
        created_at     TEXT NOT NULL,
        storage_rel    TEXT
    )
    """,
    """
    CREATE TABLE trained_models (
        model_id       TEXT PRIMARY KEY,
        name           TEXT NOT NULL,
        task           TEXT NOT NULL,
        dataset_id     TEXT REFERENCES datasets(dataset_id) ON DELETE SET NULL,
        target         TEXT,
        features_json  TEXT NOT NULL DEFAULT '[]',
        n_train_rows   INTEGER,
        best_params_json TEXT NOT NULL DEFAULT '{}',
        cv_scores_json TEXT,
        cv_mean        REAL,
        cv_std         REAL,
        fit_seconds    REAL,
        created_at     TEXT NOT NULL,
        storage_rel    TEXT
    )
    """,
    """
    CREATE TABLE jobs (
        job_id         TEXT PRIMARY KEY,
        operation      TEXT NOT NULL,
        workspace      TEXT NOT NULL,
        state          TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        started_at     TEXT,
        completed_at   TEXT,
        progress       INTEGER NOT NULL DEFAULT 0,
        error          TEXT,
        request_json   TEXT
    )
    """,
    """
    CREATE TABLE artifacts (
        artifact_id    TEXT PRIMARY KEY,
        kind           TEXT NOT NULL,
        filename       TEXT NOT NULL,
        content_type   TEXT NOT NULL,
        size_bytes     INTEGER NOT NULL,
        sha256         TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        meta_json      TEXT,
        storage_rel    TEXT
    )
    """,
    """
    CREATE TABLE user_preferences (
        key            TEXT PRIMARY KEY,
        value_json     TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    )
    """,
]

#: Migration 2 — the reporting indexes + the generative priors table
#: (added when the generative workspace gained durable priors).
_V2_SCHEMA = [
    "CREATE INDEX idx_datasets_created ON datasets (created_at)",
    "CREATE INDEX idx_trained_models_created ON trained_models (created_at)",
    "CREATE INDEX idx_jobs_created ON jobs (created_at)",
    "CREATE INDEX idx_artifacts_created ON artifacts (created_at)",
    """
    CREATE TABLE generative_priors (
        prior_id       TEXT PRIMARY KEY,
        name           TEXT NOT NULL,
        kind           TEXT NOT NULL DEFAULT 'smiles_rnn',
        params_json    TEXT NOT NULL DEFAULT '{}',
        n_train_smiles INTEGER,
        losses_json    TEXT NOT NULL DEFAULT '[]',
        corpus_source  TEXT,
        created_at     TEXT NOT NULL,
        storage_rel    TEXT
    )
    """,
]

#: Migration 3 — part16 §15.2 (Phase 12): the experiment-run lineage
#: table.  ``payload_json`` holds the full §15.2 record; the indexed
#: columns mirror the list-view projection for SQL-side filtering.
_V3_SCHEMA = [
    """
    CREATE TABLE experiment_runs (
        run_id         TEXT PRIMARY KEY,
        family         TEXT NOT NULL,
        status         TEXT NOT NULL,
        job_id         TEXT,
        mode           TEXT,
        dataset_id     TEXT,
        model_id       TEXT,
        model_name     TEXT,
        created_at     TEXT NOT NULL,
        completed_at   TEXT,
        payload_json   TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_experiment_runs_created ON experiment_runs (created_at)",
]

#: The ordered, versioned migration ledger (§16.4).
MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, _V1_SCHEMA),
    (2, _V2_SCHEMA),
    (3, _V3_SCHEMA),
]

#: The highest migration version known to this build.
SCHEMA_VERSION = MIGRATIONS[-1][0]


def default_data_dir() -> Path:
    """The writable platform user-data directory (§16.5).

    Windows: ``%LOCALAPPDATA%\\PharmaML Studio``; everywhere else the
    XDG data home (``$XDG_DATA_HOME`` or ``~/.local/share``).  The
    ``PHARMAML_STUDIO_DATA`` environment variable overrides both —
    tests and custom hosts use it.
    """
    override = os.environ.get("PHARMAML_STUDIO_DATA")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":  # pragma: no cover - Windows host path
        base = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
        return Path(base) / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(
        Path.home() / ".local" / "share"
    )
    return Path(base) / APP_DIR_NAME


def resolve_data_dir(data_dir: Any) -> Path | None:
    """Normalise a caller-supplied data dir (``None`` stays ``None``)."""
    if data_dir is None:
        return None
    return Path(data_dir).expanduser()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class FileVault:
    """The backend-managed file store for large payloads (§16.2).

    Files are written under ``<root>/files/<category>/<uuid><ext>``
    atomically (temp file + ``os.replace``), referenced from SQLite
    by their **relative** path + sha256, and read back with explicit
    traversal protection (no ``..``, no absolute paths, resolved
    strictly inside the vault root).
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._files_root = root / "files"
        self._files_root.mkdir(parents=True, exist_ok=True)

    # -- writes -------------------------------------------------------

    def save(self, category: str, data: bytes, ext: str = "") -> dict:
        """Store bytes; return ``storage_rel``/``sha256``/``size_bytes``."""
        if ext and not ext.startswith("."):
            ext = "." + ext
        category = _safe_segment(category)
        category_dir = self._files_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        rel = f"files/{category}/{uuid.uuid4().hex}{ext}"
        target = self.root / rel
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return {
            "storage_rel": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    # -- reads --------------------------------------------------------

    def read(self, storage_rel: str | None) -> bytes:
        """Read a vault file back (traversal-checked)."""
        path = self._resolve(storage_rel)
        return path.read_bytes()

    def exists(self, storage_rel: str | None) -> bool:
        if not storage_rel:
            return False
        try:
            path = self._resolve(storage_rel)
        except ValueError:
            return False
        # the payload must actually BE on disk: the orphan sweep and
        # the recovery report both rely on this to detect vanished
        # files honestly (a resolvable-but-absent path is a missing
        # payload, not a present one)
        return path.is_file()

    # -- deletes ------------------------------------------------------

    def delete(self, storage_rel: str | None) -> None:
        if not storage_rel:
            return
        try:
            path = self._resolve(storage_rel)
        except ValueError:
            return
        path.unlink(missing_ok=True)

    # -- internals ----------------------------------------------------

    def _resolve(self, storage_rel: str | None) -> Path:
        if not storage_rel or not isinstance(storage_rel, str):
            raise ValueError("storage_rel is required")
        candidate = (self.root / storage_rel).resolve()
        vault_root = self._files_root.resolve()
        if vault_root not in candidate.parents:
            raise ValueError(f"path escapes the vault: {storage_rel!r}")
        return candidate


def _safe_segment(name: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(name)
    ).strip("_")
    return cleaned or "misc"


class Persistence:
    """The SQLite database + file vault pair (one per app instance).

    Every operation opens its own connection with the safety pragmas
    applied (foreign keys, WAL, busy timeout), commits and closes —
    the §16.4 connection-per-operation strategy: no connection is
    ever shared across the FastAPI event loop and the job threads.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.data_dir / "studio.db"
        self.vault = FileVault(self.data_dir)
        self.apply_migrations()

    # -- connection plumbing ------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Open a configured connection (caller commits/closes)."""
        conn = sqlite3.connect(str(self.database_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        conn.row_factory = sqlite3.Row
        return conn

    def _execute(
        self, sql: str, params: tuple | list = ()
    ) -> list[sqlite3.Row]:
        conn = self.connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            conn.commit()
            return rows
        finally:
            conn.close()

    def _execute_write(
        self, sql: str, params: tuple | list = ()
    ) -> int:
        conn = self.connect()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    # -- migrations ----------------------------------------------------

    def apply_migrations(self) -> dict:
        """Apply pending migrations (idempotent, transactional)."""
        conn = self.connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            conn.commit()
            applied = {
                r["version"]
                for r in conn.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
            for version, statements in MIGRATIONS:
                if version in applied:
                    continue
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for statement in statements:
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations VALUES (?, ?)",
                        (version, _now_iso()),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            applied_after = {
                r["version"]
                for r in conn.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
            return {
                "schema_version": max(applied_after or {0}),
                "applied": sorted(
                    v for v in applied_after if v not in applied
                ),
            }
        finally:
            conn.close()

    def schema_version(self) -> int:
        rows = self._execute("SELECT MAX(version) AS v FROM schema_migrations")
        return int(rows[0]["v"] or 0)

    def integrity_check(self) -> str:
        rows = self._execute("PRAGMA integrity_check")
        return str(rows[0][0]) if rows else "unknown"

    # -- datasets ------------------------------------------------------

    def save_dataset(self, view: dict, source_bytes: bytes) -> dict:
        ref = self.vault.save("datasets", source_bytes, ".bin")
        self._execute_write(
            """
            INSERT OR REPLACE INTO datasets (
                dataset_id, name, source_filename, size_bytes, sha256,
                target, task, n_rows, n_columns, created_at, storage_rel
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view["dataset_id"],
                view["name"],
                view["source_filename"],
                int(view["size_bytes"]),
                ref["sha256"],
                view.get("target"),
                view.get("task"),
                int(view["shape"][0]) if view.get("shape") else None,
                int(view["shape"][1]) if view.get("shape") else None,
                view["created_at"],
                ref["storage_rel"],
            ),
        )
        return ref

    def delete_dataset(self, dataset_id: str) -> None:
        rows = self._execute(
            "SELECT storage_rel FROM datasets WHERE dataset_id = ?",
            (dataset_id,),
        )
        self._execute_write(
            "DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,)
        )
        for row in rows:
            self.vault.delete(row["storage_rel"])

    def load_dataset_source(self, dataset_id: str) -> bytes | None:
        rows = self._execute(
            "SELECT storage_rel FROM datasets WHERE dataset_id = ?",
            (dataset_id,),
        )
        if not rows or not rows[0]["storage_rel"]:
            return None
        return self.vault.read(rows[0]["storage_rel"])

    # -- trained models --------------------------------------------------

    def save_model(self, view: dict, payload: bytes) -> dict:
        ref = self.vault.save("models", payload, ".joblib")
        self._execute_write(
            """
            INSERT OR REPLACE INTO trained_models (
                model_id, name, task, dataset_id, target,
                features_json, n_train_rows, best_params_json,
                cv_scores_json, cv_mean, cv_std, fit_seconds,
                created_at, storage_rel
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view["model_id"],
                view["name"],
                view["task"],
                view.get("dataset_id"),
                view.get("target"),
                json.dumps(json_safe(view.get("features") or [])),
                view.get("n_train_rows"),
                json.dumps(json_safe(view.get("best_params") or {})),
                json.dumps(json_safe(view.get("cv_scores") or None)),
                view.get("cv_mean"),
                view.get("cv_std"),
                view.get("fit_seconds"),
                view["created_at"],
                ref["storage_rel"],
            ),
        )
        return ref

    def delete_model(self, model_id: str) -> None:
        rows = self._execute(
            "SELECT storage_rel FROM trained_models WHERE model_id = ?",
            (model_id,),
        )
        self._execute_write(
            "DELETE FROM trained_models WHERE model_id = ?", (model_id,)
        )
        for row in rows:
            self.vault.delete(row["storage_rel"])

    def load_model_payload(self, model_id: str) -> bytes | None:
        rows = self._execute(
            "SELECT storage_rel FROM trained_models WHERE model_id = ?",
            (model_id,),
        )
        if not rows or not rows[0]["storage_rel"]:
            return None
        return self.vault.read(rows[0]["storage_rel"])

    # -- generative priors ------------------------------------------------

    def save_prior(self, view: dict, payload: bytes) -> dict:
        ref = self.vault.save("priors", payload, ".joblib")
        self._execute_write(
            """
            INSERT OR REPLACE INTO generative_priors (
                prior_id, name, kind, params_json, n_train_smiles,
                losses_json, corpus_source, created_at, storage_rel
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view["prior_id"],
                view["name"],
                view.get("kind", "smiles_rnn"),
                json.dumps(json_safe(view.get("params") or {})),
                view.get("n_train_smiles"),
                json.dumps(json_safe(view.get("losses") or [])),
                # corpus_source is a dict ({"source": ..., "n": ...}) —
                # the TEXT column needs the JSON encoding, never the
                # raw dict (sqlite binds scalars only; the E2E caught
                # the ProgrammingError at exactly this parameter)
                json.dumps(json_safe(view.get("corpus_source") or {})),
                view["created_at"],
                ref["storage_rel"],
            ),
        )
        return ref

    def delete_prior(self, prior_id: str) -> None:
        rows = self._execute(
            "SELECT storage_rel FROM generative_priors WHERE prior_id = ?",
            (prior_id,),
        )
        self._execute_write(
            "DELETE FROM generative_priors WHERE prior_id = ?", (prior_id,)
        )
        for row in rows:
            self.vault.delete(row["storage_rel"])

    def load_prior_payload(self, prior_id: str) -> bytes | None:
        rows = self._execute(
            "SELECT storage_rel FROM generative_priors WHERE prior_id = ?",
            (prior_id,),
        )
        if not rows or not rows[0]["storage_rel"]:
            return None
        return self.vault.read(rows[0]["storage_rel"])

    # -- artifacts -------------------------------------------------------

    def save_artifact(self, view: dict, data: bytes) -> dict:
        suffix = Path(view.get("filename", "artifact")).suffix
        ref = self.vault.save("artifacts", data, suffix or ".bin")
        self._execute_write(
            """
            INSERT OR REPLACE INTO artifacts (
                artifact_id, kind, filename, content_type, size_bytes,
                sha256, created_at, meta_json, storage_rel
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view["artifact_id"],
                view["kind"],
                view["filename"],
                view["content_type"],
                int(view["size_bytes"]),
                view["sha256"],
                view["created_at"],
                json.dumps(json_safe(view.get("meta") or {})),
                ref["storage_rel"],
            ),
        )
        return ref

    def delete_artifact(self, artifact_id: str) -> None:
        rows = self._execute(
            "SELECT storage_rel FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        self._execute_write(
            "DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )
        for row in rows:
            self.vault.delete(row["storage_rel"])

    def load_artifact_data(self, artifact_id: str) -> bytes | None:
        rows = self._execute(
            "SELECT storage_rel FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        if not rows or not rows[0]["storage_rel"]:
            return None
        return self.vault.read(rows[0]["storage_rel"])

    # -- jobs --------------------------------------------------------------

    def record_job(self, view: dict) -> None:
        """Upsert one job row (called on every state transition)."""
        self._execute_write(
            """
            INSERT OR REPLACE INTO jobs (
                job_id, operation, workspace, state, created_at,
                started_at, completed_at, progress, error, request_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view["job_id"],
                view["operation"],
                view["workspace"],
                view["state"],
                view["created_at"],
                view.get("started_at"),
                view.get("completed_at"),
                int(view.get("progress") or 0),
                view.get("error"),
                json.dumps(json_safe(view.get("configuration") or {})),
            ),
        )

    def prune_jobs(self, keep_last: int = 500) -> int:
        """Retention rule (§16.4): keep the newest ``keep_last`` rows."""
        return self._execute_write(
            """
            DELETE FROM jobs WHERE job_id NOT IN (
                SELECT job_id FROM jobs
                ORDER BY created_at DESC, job_id DESC
                LIMIT ?
            )
            """,
            (int(keep_last),),
        )

    def list_job_history(
        self, workspace: str | None = None, limit: int = 100
    ) -> list[dict]:
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if workspace:
            sql += " WHERE workspace = ?"
            params.append(workspace)
        sql += " ORDER BY created_at DESC, job_id DESC LIMIT ?"
        params.append(int(limit))
        rows = self._execute(sql, params)
        return [dict(r) for r in rows]

    # -- experiment runs (part16 §15.2) ------------------------------------

    def record_run(self, view: dict) -> None:
        """Upsert one experiment-run row (payload = the full record)."""
        model_artifact = view.get("model_artifact") or {}
        self._execute_write(
            """
            INSERT OR REPLACE INTO experiment_runs (
                run_id, family, status, job_id, mode, dataset_id,
                model_id, model_name, created_at, completed_at,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view["run_id"],
                view["family"],
                view["status"],
                view.get("job_id"),
                view.get("mode"),
                view.get("dataset", {}).get("dataset_id"),
                model_artifact.get("model_id"),
                view.get("algorithm", {}).get("name"),
                view["created_at"],
                view.get("completed_at"),
                json.dumps(json_safe(view)),
            ),
        )

    def list_runs(
        self,
        family: str | None = None,
        dataset_id: str | None = None,
        model_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """The durable lineage query (the summary columns, newest first)."""
        sql = (
            "SELECT run_id, family, status, job_id, mode, dataset_id,"
            " model_id, model_name, created_at, completed_at"
            " FROM experiment_runs"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if family is not None:
            clauses.append("family = ?")
            params.append(family)
        if dataset_id is not None:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, run_id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self._execute(sql, params)]

    def load_run(self, run_id: str) -> dict | None:
        """The full §15.2 payload for one run."""
        rows = self._execute(
            "SELECT payload_json FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        )
        if not rows:
            return None
        return json.loads(rows[0]["payload_json"])

    # -- preferences -----------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        self._execute_write(
            """
            INSERT OR REPLACE INTO user_preferences (key, value_json, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, json.dumps(json_safe(value)), _now_iso()),
        )

    def get_preference(self, key: str, default: Any = None) -> Any:
        rows = self._execute(
            "SELECT value_json FROM user_preferences WHERE key = ?", (key,)
        )
        if not rows:
            return default
        return json.loads(rows[0]["value_json"])

    def list_preferences(self) -> list[dict]:
        rows = self._execute(
            "SELECT key, value_json, updated_at FROM user_preferences"
            " ORDER BY key"
        )
        return [
            {
                "key": r["key"],
                "value": json.loads(r["value_json"]),
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def delete_preference(self, key: str) -> bool:
        return self._execute_write(
            "DELETE FROM user_preferences WHERE key = ?", (key,)
        ) > 0

    # -- reporting / backup ----------------------------------------------

    def status(self) -> dict:
        counts: dict[str, int] = {}
        for table in (
            "datasets",
            "trained_models",
            "generative_priors",
            "jobs",
            "artifacts",
            "user_preferences",
        ):
            rows = self._execute(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(rows[0]["n"])
        return {
            "persistent": True,
            "data_dir": str(self.data_dir),
            "database": "studio.db",
            "schema_version": self.schema_version(),
            "journal_mode": "wal",
            "integrity": self.integrity_check(),
            "tables": counts,
        }

    def backup_bytes(self) -> bytes:
        """A consistent snapshot of the database core (§16.4).

        Uses sqlite3's online backup API into a private temporary
        file — no locks held on the live WAL database beyond the copy,
        safe to call while jobs are writing.  (The file route, not
        :memory: serialize: a WAL source's backup image carries the
        WAL header flag, and an in-memory deserialize of such an
        image then fails to open its nonexistent sidecar — "unable
        to open database file".  The temp-file image is a plain
        journal-mode snapshot.)
        """
        import tempfile

        with tempfile.TemporaryDirectory(prefix="pharmaml-bkp-") as tmp:
            target = Path(tmp) / "backup.db"
            source = self.connect()
            try:
                dest = sqlite3.connect(str(target))
                try:
                    source.backup(dest)
                finally:
                    dest.close()
            finally:
                source.close()
            return target.read_bytes()

    # -- the backup bundle (DB + referenced vault payloads) ----------

    _BUNDLE_DB_NAME = "studio.db"
    _BUNDLE_FILES_PREFIX = "files/"

    def _referenced_storage_rels(self) -> list[str]:
        rels: list[str] = []
        for table, key in (
            ("datasets", "dataset_id"),
            ("trained_models", "model_id"),
            ("generative_priors", "prior_id"),
            ("artifacts", "artifact_id"),
        ):
            rows = self._execute(
                f"SELECT storage_rel FROM {table}"
                " WHERE storage_rel IS NOT NULL"
            )
            rels.extend(r["storage_rel"] for r in rows)
        return sorted(set(rels))

    def backup_bundle(self) -> bytes:
        """The full backup artifact: SQLite image + vault payloads.

        A zip container holding ``studio.db`` (the online-backup
        snapshot) plus every vault file referenced by a registry
        row, stored under its vault-relative path — so a restore
        brings back the *complete* registry (the DB-only snapshot
        would honestly degrade to rows whose payloads survived).
        """
        import io as _io
        import zipfile

        db_image = self.backup_bytes()
        buffer = _io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(self._BUNDLE_DB_NAME, db_image)
            for rel in self._referenced_storage_rels():
                try:
                    payload = self.vault.read(rel)
                except ValueError:  # pragma: no cover - dropped rows
                    continue  # are pruned before this loop in practice
                zf.writestr(rel, payload)
        return buffer.getvalue()

    def _validate_bundle_db(self, zf: zipfile.ZipFile) -> bytes:
        """Read + integrity-validate the bundle's database image."""
        db_image = zf.read(self._BUNDLE_DB_NAME)
        import tempfile

        with tempfile.TemporaryDirectory(prefix="pharmaml-rst-") as tmp:
            probe_path = Path(tmp) / "probe.db"
            probe_path.write_bytes(db_image)
            probe = sqlite3.connect(str(probe_path))
            try:
                try:
                    check = probe.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()
                except sqlite3.DatabaseError:
                    # garbage bytes raise at execute time, not connect
                    # time — normalise to the documented ValueError so
                    # both direct callers and the REST surface (422)
                    # see one honest error type
                    raise ValueError(
                        "backup failed the integrity check"
                    )
                if not check or str(check[0]) != "ok":
                    raise ValueError(
                        "backup failed the integrity check"
                    )
                versions = [
                    r[0]
                    for r in probe.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                ]
                if not versions:
                    raise ValueError(
                        "backup has no schema_migrations ledger"
                    )
                if max(versions) > SCHEMA_VERSION:
                    raise ValueError(
                        f"backup schema v{max(versions)} is newer than"
                        f" this build's v{SCHEMA_VERSION}"
                    )
            finally:
                probe.close()
        return db_image

    def restore_bytes(self, data: bytes) -> dict:
        """Restore the database core from a validated image (internal).

        The image is validated (integrity + schema version <= the
        build's) on a temp-file probe, then swapped in with an atomic
        replace; stale WAL sidecars of the old database are removed
        with it (their frames must never touch the new file).  The
        live connection pool is per-operation so the swap needs no
        coordination.
        """
        db_image = data
        import tempfile

        with tempfile.TemporaryDirectory(prefix="pharmaml-rst-") as tmp:
            probe_path = Path(tmp) / "probe.db"
            probe_path.write_bytes(db_image)
            probe = sqlite3.connect(str(probe_path))
            try:
                try:
                    check = probe.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()
                except sqlite3.DatabaseError:
                    raise ValueError(
                        "backup failed the integrity check"
                    )
                if not check or str(check[0]) != "ok":
                    raise ValueError("backup failed the integrity check")
                versions = [
                    r[0]
                    for r in probe.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                ]
                if not versions:
                    raise ValueError(
                        "backup has no schema_migrations ledger"
                    )
                if max(versions) > SCHEMA_VERSION:
                    raise ValueError(
                        f"backup schema v{max(versions)} is newer than"
                        f" this build's v{SCHEMA_VERSION}"
                    )
            finally:
                probe.close()
        swap_path = self.database_path.with_suffix(".db.restore.tmp")
        swap_path.write_bytes(db_image)
        os.replace(swap_path, self.database_path)
        # the old database's WAL sidecars die with it: their frames
        # must never be replayed onto the restored file
        for sidecar in ("-wal", "-shm"):
            sidecar_path = self.data_dir / (self.database_path.name + sidecar)
            sidecar_path.unlink(missing_ok=True)
        # re-run migrations (no-op at equal version, upgrades if older)
        self.apply_migrations()
        orphans = self._drop_orphaned_rows()
        return {
            "restored": True,
            "schema_version": self.schema_version(),
            "orphaned_rows_dropped": orphans,
        }

    def restore_bundle(self, data: bytes) -> dict:
        """Restore from a full backup bundle (DB + vault payloads).

        Validates the container's database image first, then (1)
        writes every bundled vault payload back under its
        vault-relative path (traversal-checked), (2) swaps the
        database in atomically, (3) re-runs migrations and the
        orphan sweep, and reports what was restored — the *complete*
        registry comes back, payloads included.
        """
        import io as _io
        import zipfile

        try:
            zf = zipfile.ZipFile(_io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"not a backup bundle: {exc}") from exc
        with zf:
            names = zf.namelist()
            if self._BUNDLE_DB_NAME not in names:
                raise ValueError("backup bundle has no studio.db entry")
            # safety: only the db entry and files/ payloads are legal
            for name in names:
                if name == self._BUNDLE_DB_NAME:
                    continue
                if not name.startswith(self._BUNDLE_FILES_PREFIX):
                    raise ValueError(f"unexpected bundle entry: {name!r}")
                if name.startswith("/") or ".." in name.split("/"):
                    raise ValueError(f"unsafe bundle entry: {name!r}")
            db_image = self._validate_bundle_db(zf)
            payload_names = [
                n for n in names if n != self._BUNDLE_DB_NAME
            ]
            # 1) payloads first: the orphan sweep runs after the swap
            #    and must find every referenced file present
            files_root = self.vault._files_root.resolve()
            for name in payload_names:
                target = (self.vault.root / name).resolve()
                if files_root not in target.parents:
                    raise ValueError(f"unsafe bundle entry: {name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
        # 2) the atomic database swap (+ sidecar cleanup + migrations)
        result = self.restore_bytes(db_image)
        result["payloads_restored"] = len(payload_names)
        return result

    def _drop_orphaned_rows(self) -> dict[str, int]:
        """Remove rows whose vault payloads are gone (honest repair)."""
        dropped: dict[str, int] = {}
        for table, key in (
            ("datasets", "dataset_id"),
            ("trained_models", "model_id"),
            ("generative_priors", "prior_id"),
            ("artifacts", "artifact_id"),
        ):
            rows = self._execute(
                f"SELECT {key}, storage_rel FROM {table}"
                " WHERE storage_rel IS NOT NULL"
            )
            missing = [
                r[key]
                for r in rows
                if not self.vault.exists(r["storage_rel"])
            ]
            for row_id in missing:
                self._execute_write(
                    f"DELETE FROM {table} WHERE {key} = ?", (row_id,)
                )
            if missing:
                dropped[table] = len(missing)
        return dropped


def recover_registry(
    persistence: Persistence,
    dataset_store: Any,
    model_store: Any,
    prior_store: Any,
    artifact_store: Any,
    run_store: Any | None = None,
) -> dict:
    """Rebuild the live stores from the persisted registry.

    Called by :func:`~pharmaml.api.create_app` when persistence is
    enabled — the part11 "preserve user data across restarts"
    contract.  Dataset frames are re-parsed from the vault's source
    bytes through the REAL loader; models/priors are unpickled;
    artifact bytes are read back.  Rows whose payloads fail to load
    (version drift, corruption) are dropped from both the DB and the
    live registry and *reported* — never silently skipped.  The
    part16 §15.2 lane adds the experiment-run recovery: the full
    §15.2 records rehydrate from ``payload_json`` (JSON only — no
    pickles, no estimators to rebuild).
    """
    import pickle
    import tempfile
    from pathlib import Path as _Path

    from pharmaml.data import load_dataset

    from .runs import ExperimentRunRecord
    from .stores import (
        ArtifactRecord,
        DatasetRecord,
        PriorRecord,
        TrainedModelRecord,
    )

    report: dict[str, Any] = {
        "datasets": 0,
        "models": 0,
        "priors": 0,
        "artifacts": 0,
        "runs": 0,
        "dropped": [],
    }

    # -- datasets: re-parse the source bytes with the REAL loader ---
    rows = persistence._execute(
        "SELECT * FROM datasets ORDER BY created_at, dataset_id"
    )
    for row in rows:
        try:
            raw = persistence.load_dataset_source(row["dataset_id"])
            if raw is None:
                raise ValueError("vault payload missing")
            suffix = _Path(row["source_filename"]).suffix or ".csv"
            with tempfile.TemporaryDirectory(prefix="pharmaml-rec-") as tmp:
                path = _Path(tmp) / f"recover{suffix}"
                path.write_bytes(raw)
                dataset = load_dataset(
                    path, target=row["target"], task=row["task"]
                )
            record = DatasetRecord(
                row["dataset_id"],
                dataset,
                row["name"],
                row["size_bytes"],
                row["source_filename"],
            )
            record.created_at = row["created_at"]
            dataset_store._items[row["dataset_id"]] = record
            report["datasets"] += 1
        except Exception as exc:  # noqa: BLE001 - honest degradation
            persistence.delete_dataset(row["dataset_id"])
            report["dropped"].append(
                {"table": "datasets", "id": row["dataset_id"],
                 "reason": f"{type(exc).__name__}: {exc}"}
            )

    # -- trained models: unpickle + rebuild the record -----------------
    rows = persistence._execute(
        "SELECT * FROM trained_models ORDER BY created_at, model_id"
    )
    for row in rows:
        try:
            payload = persistence.load_model_payload(row["model_id"])
            if payload is None:
                raise ValueError("vault payload missing")
            model = pickle.loads(payload)
            model_record = TrainedModelRecord(
                model_id=row["model_id"],
                model=model,
                name=row["name"],
                task=row["task"],
                dataset_id=row["dataset_id"],
                target=row["target"],
                features=json.loads(row["features_json"] or "[]"),
                n_train_rows=row["n_train_rows"] or 0,
                best_params=json.loads(row["best_params_json"] or "{}"),
                cv_scores=json.loads(row["cv_scores_json"] or "null"),
                cv_mean=row["cv_mean"],
                cv_std=row["cv_std"],
                fit_seconds=row["fit_seconds"] or 0.0,
            )
            model_record.created_at = row["created_at"]
            model_store._items[row["model_id"]] = model_record
            report["models"] += 1
        except Exception as exc:  # noqa: BLE001 - honest degradation
            persistence.delete_model(row["model_id"])
            report["dropped"].append(
                {"table": "trained_models", "id": row["model_id"],
                 "reason": f"{type(exc).__name__}: {exc}"}
            )

    # -- generative priors ------------------------------------------------
    rows = persistence._execute(
        "SELECT * FROM generative_priors ORDER BY created_at, prior_id"
    )
    for row in rows:
        try:
            payload = persistence.load_prior_payload(row["prior_id"])
            if payload is None:
                raise ValueError("vault payload missing")
            model = pickle.loads(payload)
            prior_record = PriorRecord(
                prior_id=row["prior_id"],
                model=model,
                name=row["name"],
                params=json.loads(row["params_json"] or "{}"),
                n_train_smiles=row["n_train_smiles"] or 0,
                losses=json.loads(row["losses_json"] or "[]"),
                corpus_source=row["corpus_source"] or "",
            )
            prior_record.created_at = row["created_at"]
            prior_store._items[row["prior_id"]] = prior_record
            report["priors"] += 1
        except Exception as exc:  # noqa: BLE001 - honest degradation
            persistence.delete_prior(row["prior_id"])
            report["dropped"].append(
                {"table": "generative_priors", "id": row["prior_id"],
                 "reason": f"{type(exc).__name__}: {exc}"}
            )

    # -- artifacts ----------------------------------------------------
    rows = persistence._execute(
        "SELECT * FROM artifacts ORDER BY created_at, artifact_id"
    )
    for row in rows:
        try:
            data = persistence.load_artifact_data(row["artifact_id"])
            if data is None:
                raise ValueError("vault payload missing")
            artifact_record = ArtifactRecord(
                artifact_id=row["artifact_id"],
                kind=row["kind"],
                filename=row["filename"],
                content_type=row["content_type"],
                data=data,
                meta=json.loads(row["meta_json"] or "{}"),
            )
            artifact_record.created_at = row["created_at"]
            artifact_store._items[row["artifact_id"]] = artifact_record
            report["artifacts"] += 1
        except Exception as exc:  # noqa: BLE001 - honest degradation
            persistence.delete_artifact(row["artifact_id"])
            report["dropped"].append(
                {"table": "artifacts", "id": row["artifact_id"],
                 "reason": f"{type(exc).__name__}: {exc}"}
            )

    # -- experiment runs (part16 §15.2): JSON rehydration, newest-first --
    if run_store is not None:
        rows = persistence._execute(
            "SELECT * FROM experiment_runs ORDER BY created_at, run_id"
        )
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
                run_record = ExperimentRunRecord.from_payload(payload)
                run_store._items[row["run_id"]] = run_record
                report["runs"] += 1
            except Exception as exc:  # noqa: BLE001 - honest degradation
                persistence._execute_write(
                    "DELETE FROM experiment_runs WHERE run_id = ?",
                    (row["run_id"],),
                )
                report["dropped"].append(
                    {"table": "experiment_runs", "id": row["run_id"],
                     "reason": f"{type(exc).__name__}: {exc}"}
                )

    return report
