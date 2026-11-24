"""``/api/v1/samples`` — the part16 §8 (Phase 5) demo-data lane.

The directive's Phase 5: "Expose the existing sample-data functionality
in every relevant workflow … Use the actual sample-data services and
files already available in the backend. Do not invent sample datasets
and present them as existing project resources."

This route is exactly that discipline made mechanical:

* the catalog is a **curated metadata table over the REAL files** in
  ``examples/data/`` — every entry names an on-disk file that shipped
  with the repository (the examples scripts and walkthrough notebooks
  consume these same files); the runtime guard marks an entry
  ``available: false`` when its file is missing, and loading refuses;
* **preview-before-load**: the tabular preview is the RAW file head
  (``pd.read_csv`` — what the file actually contains), never the
  post-``load_dataset`` view, so the dialog never shows a shape the
  import will not deliver;
* **one-click load, four payload kinds**: ``tabular`` samples go
  through the SAME import path as a manual upload —
  :func:`pharmaml.data.load_dataset` (target auto-detection, task
  inference, non-numeric column policy) + ``DatasetStore.register``
  — so a loaded sample is an ordinary dataset record usable by every
  workspace that consumes ``dataset_id``; ``series`` samples (the
  PK concentration–time profile) return the two numeric arrays the
  PK/PD forms take verbatim (``times``/``conc`` — the routes' own
  request shape, NOT a forced dataset import the real loader would
  rightly reject: a time-series has no target column); ``library``
  samples (the SMILES compound lists) deliver the name+SMILES columns
  the chemistry/generative/docking lanes take verbatim (a SMILES
  library is not a features+target table — the real loader rightly
  refuses it, and the routes consume ``smiles: list[str]``); and
  ``text_file`` samples (the docking receptor PDB, the demo ligand
  PDBQT, the campaign DLG) return the bounded file content itself,
  which the docking/MD/visualization inputs accept verbatim;
* **no-overwrite is structural**: ``register`` always allocates a fresh
  ``ds-<uuid>`` — loading a demo sample never replaces, mutates or
  evicts user data (the eviction policy is the store's own bounded
  oldest-first rule, unchanged here);
* **cleanup where safe**: loaded samples are ordinary dataset records,
  so the existing ``DELETE /api/v1/datasets/{id}`` lane removes them —
  the catalog entry states this hint and the GUI surfaces it.

Offline by construction: everything is file reads + the local science
stack — no external API access anywhere in the lane.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from pharmaml.data import load_dataset
from pharmaml.exceptions import DataValidationError

from ..stores import DatasetStore, frame_payload

router = APIRouter(prefix="/samples", tags=["samples"])

#: The shipped sample-data root — resolved relative to the editable
#: source tree (``src/pharmaml/api/routes/samples.py`` ->
#: ``<repo>/examples/data``), exactly the files the example scripts
#: and walkthrough notebooks consume.
SAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "data"

#: Preview rows served for the raw-head dialog (bounded, §17.3-style).
MAX_PREVIEW_ROWS = 20

#: Text-sample content cap (the largest shipped sample is ~116 KB; the
#: cap is the guard, never the truncation-in-practice).
MAX_TEXT_CONTENT_CHARS = 1_000_000


class LoadRequest(BaseModel):
    """Optional knobs for the tabular import (mirroring the upload form).

    The curated defaults (target/task hints per sample) are applied
    when the request omits them; ``None`` explicitly defers to the
    REAL loader's auto-detection, exactly like an upload that leaves
    the fields blank.
    """

    target: str | None = Field(default=None, description=(
        "Target column override; None = the curated hint (if any) "
        "then the loader's auto-detection."
    ))
    task: str | None = Field(default=None, description=(
        "Task hint override ('classification' / 'regression'); "
        "None = the curated hint (if any) then auto-inference."
    ))


# ---------------------------------------------------------------------------
# The curated catalog — metadata over the REAL shipped files
# ---------------------------------------------------------------------------

#: One catalog entry. ``filename`` is relative to ``SAMPLES_DIR`` and
#: MUST exist on disk (the runtime guard re-verifies per request).
_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "qsar-classification",
        "title": "QSAR classification set",
        "filename": "qsar_classification.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "Precomputed molecular descriptors plus a binary activity "
            "label — the standard QSAR classification shape."
        ),
        "scientific_context": (
            "Structure–activity screening: descriptor space -> active/"
            "inactive decision boundary."
        ),
        "workflows": ("datasets", "machine-learning"),
        "target": "activity",
        "task": "classification",
    },
    {
        "id": "qsar-regression",
        "title": "QSAR regression set (pIC50)",
        "filename": "qsar_regression.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "Descriptor rows with a continuous pIC50 potency target — "
            "the regression counterpart of the classification set."
        ),
        "scientific_context": (
            "Potency modelling: predicting -log(IC50) from descriptor "
            "space."
        ),
        "workflows": ("datasets", "machine-learning"),
        "target": "pIC50",
        "task": "regression",
    },
    {
        "id": "qsar-multiclass",
        "title": "QSAR multiclass set",
        "filename": "qsar_multiclass.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "Descriptor rows with a multi-class Target column — the "
            "multiclass classification lane."
        ),
        "scientific_context": (
            "Multi-category activity/property assignment from "
            "descriptor space."
        ),
        "workflows": ("datasets", "machine-learning"),
        "target": "Target",
        "task": "classification",
    },
    {
        "id": "walkthrough-classification",
        "title": "Walkthrough classification table",
        "filename": "walkthrough_classification.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "The walkthrough notebook's own classification table — "
            "small, fast to train, ideal for a first end-to-end run."
        ),
        "scientific_context": (
            "The guided QSAR walkthrough's demo table."
        ),
        "workflows": ("datasets", "machine-learning"),
        "target": "Target",
        "task": "classification",
    },
    {
        "id": "walkthrough-multiclass",
        "title": "Walkthrough multiclass table",
        "filename": "walkthrough_multiclass.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "The walkthrough's multiclass variant of the demo table."
        ),
        "scientific_context": (
            "The guided multiclass walkthrough's demo table."
        ),
        "workflows": ("datasets", "machine-learning"),
        "target": "Target",
        "task": "classification",
    },
    {
        "id": "walkthrough-compounds",
        "title": "Walkthrough compound set (SMILES + activity)",
        "filename": "walkthrough_compounds.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "Compounds carrying both a SMILES column (chemistry-ready) "
            "and an activity label — one table feeds the descriptor "
            "stage AND the model stage."
        ),
        "scientific_context": (
            "SMILES + descriptors + activity: the chemistry-to-model "
            "bridge in one file."
        ),
        "workflows": ("datasets", "machine-learning", "chemistry"),
        "target": "activity",
        "task": "classification",
    },
    {
        "id": "descriptor-library",
        "title": "Screening library with descriptors",
        "filename": "library_with_descriptors.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "A small compound library already carrying computed "
            "physicochemical descriptors (MolWt, LogP, TPSA, H-bond "
            "counts, LabuteASA, ring counts …)."
        ),
        "scientific_context": (
            "The descriptor/fingerprint stage's demo input: a library "
            "ready for inspection, filtering and model features."
        ),
        "workflows": ("datasets", "machine-learning", "chemistry"),
        "target": None,
        "task": None,
    },
    {
        "id": "screening-library",
        "title": "Virtual screening library (SMILES)",
        "filename": "screening_library.csv",
        "kind": "library",
        "file_type": "csv",
        "library_columns": {"name": "name", "smiles": "SMILES"},
        "description": (
            "Named compounds as SMILES — the seed input for screening, "
            "generative exploration and ligand-list docking lanes."
        ),
        "scientific_context": (
            "A virtual screening library: name + SMILES pairs."
        ),
        "workflows": ("chemistry", "generative-ai", "molecular-docking"),
        "target": None,
        "task": None,
    },
    {
        "id": "docking-ligand-library",
        "title": "Docking ligand library",
        "filename": "ligand_library.csv",
        "kind": "library",
        "file_type": "csv",
        "library_columns": {"name": "name", "smiles": "SMILES"},
        "description": (
            "A compact ligand list (name + SMILES) sized for a first "
            "docking campaign — feeds ligand preparation."
        ),
        "scientific_context": (
            "Ligand-side input for receptor docking runs."
        ),
        "workflows": ("molecular-docking",),
        "target": None,
        "task": None,
    },
    {
        "id": "campaign-ligands",
        "title": "Demo campaign ligands (14 compounds)",
        "filename": "docking_campaign/ligands.csv",
        "kind": "library",
        "file_type": "csv",
        "library_columns": {"name": "name", "smiles": "SMILES"},
        "description": (
            "The 14-ligand list of the demo docking campaign (aspirin … "
            "vanillin) whose scored results ship alongside it."
        ),
        "scientific_context": (
            "The campaign's ligand roster: aspirin, ibuprofen, "
            "naproxen, paracetamol, caffeine, theophylline, metformin, "
            "benzocaine, propranolol, nicotine, atenolol, lidocaine, "
            "coumarin, vanillin."
        ),
        "workflows": ("molecular-docking",),
        "target": None,
        "task": None,
    },
    {
        "id": "seed-molecules",
        "title": "Generative seed molecules",
        "filename": "seed_molecules.csv",
        "kind": "library",
        "file_type": "csv",
        "library_columns": {"name": "name", "smiles": "smiles"},
        "description": (
            "A small named seed set (name + SMILES) for the "
            "generative-design workflow's seed stage."
        ),
        "scientific_context": (
            "Seed structures for molecule generation campaigns."
        ),
        "workflows": ("generative-ai",),
        "target": None,
        "task": None,
    },
    {
        "id": "cellline-expression",
        "title": "Cell-line expression matrix",
        "filename": "cellline_expression.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "Cell lines x gene-expression feature matrix — the deep-"
            "learning / drug-response feature side."
        ),
        "scientific_context": (
            "Transcriptomic features for cell-line response modelling."
        ),
        "workflows": ("datasets", "deep-learning", "drugresponse"),
        "target": None,
        "task": None,
    },
    {
        "id": "cellline-screen",
        "title": "Cell-line sensitivity screen (LN_IC50)",
        "filename": "cellline_screen.csv",
        "kind": "tabular",
        "file_type": "csv",
        "description": (
            "Cell-line / tissue / compound rows with the LN_IC50 "
            "response value — the response (target) side of the "
            "drug-response lane."
        ),
        "scientific_context": (
            "Sensitivity labels (natural-log IC50) joining the "
            "expression matrix on cell_line."
        ),
        "workflows": ("datasets", "deep-learning", "drugresponse"),
        "target": "LN_IC50",
        "task": "regression",
    },
    {
        "id": "pk-profile",
        "title": "PK concentration–time profile",
        "filename": "pk_profile.csv",
        "kind": "series",
        "file_type": "csv",
        "series_columns": {"time": "time_h", "value": "conc_ng_ml"},
        "description": (
            "A time (h) / concentration (ng/mL) series — the exact "
            "input shape of the PK/PD workspace's NCA and compartment "
            "lanes (times + conc arrays)."
        ),
        "scientific_context": (
            "A single-subject concentration–time profile for "
            "non-compartmental and compartmental analysis."
        ),
        "workflows": ("pkpd",),
        "target": None,
        "task": None,
    },
    {
        "id": "lysozyme-receptor",
        "title": "Lysozyme receptor structure (1AKI)",
        "filename": "lysozyme_1aki.pdb",
        "kind": "text_file",
        "file_type": "pdb",
        "description": (
            "The classic hen egg-white lysozyme crystal structure "
            "(PDB 1AKI) — a real receptor for docking grid-box "
            "configuration, MD-style inspection and 3D visualization."
        ),
        "scientific_context": (
            "A complete protein receptor structure: the docking "
            "workspace's binding-site/grid-box reference and the "
            "molecular viewer's demo structure."
        ),
        "workflows": ("molecular-docking", "molecular-dynamics",
                      "visualization"),
        "target": None,
        "task": None,
    },
    {
        "id": "demo-ligand-pdbqt",
        "title": "Demo ligand (PDBQT)",
        "filename": "contacts_demo_ligand.pdbqt",
        "kind": "text_file",
        "file_type": "pdbqt",
        "description": (
            "A prepared ligand in AutoDock PDBQT form — feeds the "
            "contacts/interaction analysis and the 3D viewers."
        ),
        "scientific_context": (
            "The ligand-side demo for pose inspection and contact "
            "analysis."
        ),
        "workflows": ("molecular-docking", "visualization"),
        "target": None,
        "task": None,
    },
    {
        "id": "aspirin-docking-log",
        "title": "Demo campaign docking log (aspirin DLG)",
        "filename": "docking_campaign/01_aspirin/r.dlg",
        "kind": "text_file",
        "file_type": "dlg",
        "description": (
            "The aspirin run of the demo docking campaign — an "
            "AutoDock4-style result log with a clustering histogram, "
            "ready for the DLG parsing / scoring lanes."
        ),
        "scientific_context": (
            "Docked-result inspection: cluster energies and the "
            "campaign's scored outcome."
        ),
        "workflows": ("molecular-docking",),
        "target": None,
        "task": None,
    },
)

_INDEX: dict[str, dict[str, Any]] = {e["id"]: e for e in _CATALOG}

#: The workflow tags the GUI filter accepts (the nine §10.1 workspace
#: ids + the two capability lanes that name their consumers).
_WORKFLOW_TAGS: frozenset[str] = frozenset(
    {
        "introduction", "datasets", "machine-learning", "deep-learning",
        "generative-ai", "molecular-docking", "pkpd",
        "molecular-dynamics", "visualization", "chemistry",
        "drugresponse",
    }
)

#: File-content enrichment cache: (mtime_ns, size) -> dict — the files
#: are static per session; a touched file re-hashes.
_stats_lock = threading.Lock()
_stats_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}


def _file_stats(path: Path, key: str) -> dict[str, Any]:
    """size + sha256 for a sample file, cached by (mtime_ns, size)."""
    try:
        st = path.stat()
    except OSError:
        return {"size_bytes": None, "sha256": None}
    stamp = (st.st_mtime_ns, st.st_size)
    with _stats_lock:
        cached = _stats_cache.get(key)
        if cached is not None and cached[:2] == stamp:
            return cached[2]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"size_bytes": int(st.st_size), "sha256": digest}
    with _stats_lock:
        _stats_cache[key] = (stamp[0], stamp[1], payload)
    return payload


def clear_stats_cache() -> int:
    """Drop every cached sample-file stat (part16 §15.5 cleanup lane).

    The ONLY derived cache this backend holds: the catalog's
    size/sha256 enrichment entries, keyed by the shipped sample
    files' (mtime_ns, size) stamps — re-derived on the next catalog
    read, never user data. Returns the number of entries dropped.
    """
    with _stats_lock:
        dropped = len(_stats_cache)
        _stats_cache.clear()
    return dropped


def _resolve(entry: dict[str, Any]) -> Path:
    return SAMPLES_DIR / str(entry["filename"])


def _entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    """The JSON catalog view of one curated entry + runtime guard."""
    path = _resolve(entry)
    available = path.is_file()
    stats = _file_stats(path, str(entry["id"])) if available else {
        "size_bytes": None, "sha256": None,
    }
    return {
        "sample_id": entry["id"],
        "title": entry["title"],
        "filename": str(entry["filename"]),
        "kind": entry["kind"],
        "file_type": entry["file_type"],
        "series_columns": entry.get("series_columns"),
        "library_columns": entry.get("library_columns"),
        "description": entry["description"],
        "scientific_context": entry["scientific_context"],
        "workflows": list(entry["workflows"]),
        "target": entry["target"],
        "task": entry["task"],
        "size_bytes": stats["size_bytes"],
        "sha256": stats["sha256"],
        "available": available,
        "cleanup": (
            "Loaded demo datasets are ordinary dataset records — "
            "removable via DELETE /api/v1/datasets/{dataset_id}."
        ),
    }


def _get_entry(sample_id: str) -> dict[str, Any]:
    entry = _INDEX.get(sample_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sample_id {sample_id!r}.",
        )
    if not _resolve(entry).is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Sample {sample_id!r} is listed but its file "
                f"{entry['filename']!r} is missing from the shipped "
                "examples/data directory."
            ),
        )
    return entry


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------

@router.get("")
def list_samples(workflow: str | None = Query(default=None)) -> dict:
    """The catalog — optionally filtered to one workflow tag.

    Every entry re-verifies its file on disk (``available``), so the
    GUI can never offer an invented dataset: the listing IS the
    shipped ``examples/data`` reality.
    """
    if workflow is not None and workflow not in _WORKFLOW_TAGS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown workflow tag {workflow!r}; known: "
                + ", ".join(sorted(_WORKFLOW_TAGS))
            ),
        )
    entries = [
        _entry_view(e) for e in _CATALOG
        if workflow is None or workflow in e["workflows"]
    ]
    return {
        "count": len(entries),
        "workflow": workflow,
        "samples": entries,
    }


@router.get("/{sample_id}/preview")
def sample_preview(
    sample_id: str,
    limit: int = Query(default=8, ge=1, le=MAX_PREVIEW_ROWS),
) -> dict:
    """Preview-before-load: the RAW file, never the imported view.

    Tabular: the raw CSV head (columns, dtypes, full row count,
    bounded rows). Text files: the first ``limit`` lines with the true
    line count. A preview never registers anything.
    """
    entry = _get_entry(sample_id)
    path = _resolve(entry)
    if entry["kind"] in {"tabular", "series", "library"}:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # malformed file = a clean 422
            raise HTTPException(
                status_code=422,
                detail=f"Sample file unreadable: {exc}",
            ) from exc
        payload = frame_payload(frame, limit)
        return {
            "sample_id": sample_id,
            "kind": entry["kind"],
            **payload,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return {
        "sample_id": sample_id,
        "kind": "text_file",
        "columns": [],
        "n_rows": len(lines),
        "returned": min(limit, len(lines)),
        "truncated": len(lines) > limit,
        "lines": lines[:limit],
        "size_bytes": path.stat().st_size,
    }


@router.post("/{sample_id}/load")
def load_sample(
    sample_id: str,
    request: Request,
    req: LoadRequest | None = None,
) -> dict:
    """One-click load into the relevant workflow.

    ``tabular``: the SAME import path as a manual upload — the REAL
    :func:`pharmaml.data.load_dataset` (target auto-detection, task
    inference, non-numeric policy) + ``DatasetStore.register``, always
    a FRESH record (no overwrite of user data, structurally).

    ``series``: the two numeric arrays (``times``/``conc``) the PK/PD
    forms take verbatim. ``library``: the name+SMILES columns the
    chemistry/generative/docking lanes take verbatim.

    ``text_file``: the bounded file content itself (sha256-pinned) —
    the docking/MD/visualization inputs take it verbatim.
    """
    entry = _get_entry(sample_id)
    path = _resolve(entry)
    if entry["kind"] == "library":
        # the SMILES compound lists: the chemistry/generative/docking
        # lanes consume ``smiles: list[str]`` — delivering the name+
        # SMILES columns IS the one-click load (the real loader rightly
        # refuses these as features+target tables; forcing the dataset
        # import would be the dishonest shortcut).
        columns = entry["library_columns"]
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Sample file unreadable: {exc}",
            ) from exc
        missing = [
            name for name in columns.values() if name not in frame.columns
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Library columns missing from the file: {missing}",
            )
        names = [str(v) for v in frame[columns["name"]].tolist()]
        smiles = [str(v) for v in frame[columns["smiles"]].tolist()]
        return {
            "mode": "sync",
            "kind": "library",
            "sample_id": sample_id,
            "columns": {"name": columns["name"], "smiles": columns["smiles"]},
            "names": names,
            "smiles": smiles,
            "n_molecules": len(smiles),
        }
    if entry["kind"] == "series":
        # the PK/PD lanes take times/conc arrays directly — delivering
        # the two parsed columns IS the one-click load for this kind
        # (the real loader rightly refuses a no-target import; forcing
        # it would be the dishonest shortcut).
        columns = entry["series_columns"]
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Sample file unreadable: {exc}",
            ) from exc
        missing = [
            name for name in columns.values() if name not in frame.columns
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Series columns missing from the file: {missing}",
            )
        times = [float(v) for v in frame[columns["time"]].tolist()]
        conc = [float(v) for v in frame[columns["value"]].tolist()]
        return {
            "mode": "sync",
            "kind": "series",
            "sample_id": sample_id,
            "columns": {"time": columns["time"], "value": columns["value"]},
            "times": times,
            "conc": conc,
            "n_points": len(times),
        }
    if entry["kind"] == "tabular":
        target = req.target if req is not None and req.target else (
            entry["target"]
        )
        task = req.task if req is not None and req.task else entry["task"]
        raw = path.read_bytes()
        store: DatasetStore = request.app.state.dataset_store
        try:
            dataset = load_dataset(path, target=target, task=task)
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = store.register(
            dataset,
            name=f"Demo — {entry['title']}",
            source_bytes=len(raw),
            source_filename=str(entry["filename"]),
            raw_bytes=raw,
        )
        return {
            "mode": "sync",
            "kind": "tabular",
            "sample_id": sample_id,
            "dataset": record.view(preview_rows=10),
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > MAX_TEXT_CONTENT_CHARS
    return {
        "mode": "sync",
        "kind": "text_file",
        "sample_id": sample_id,
        "filename": str(entry["filename"]),
        "file_type": entry["file_type"],
        "size_bytes": path.stat().st_size,
        "sha256": _file_stats(path, str(entry["id"]))["sha256"],
        "n_lines": len(text.splitlines()),
        "content": text[:MAX_TEXT_CONTENT_CHARS],
        "truncated": truncated,
    }
