"""Conservative identity for supported genetic-search checkpoints.

Only known, stateless objectives have an auditable configuration here. Matching
identity is not authentication or a cross-machine reproducibility guarantee.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from pharmaml.exceptions import PersistenceError
from pharmaml.generative.scoring import (
    LogPBandObjective,
    MolecularWeightBandObjective,
    MultiObjective,
    QedObjective,
    SAScoreObjective,
    TpsaBandObjective,
)


def _objective_identity(objective: Any, *, active: set[int] | None = None) -> dict[str, Any]:
    """Fingerprint a whitelist of exact built-in objective types, never user objects."""
    active = set() if active is None else active
    if id(objective) in active:
        raise PersistenceError("cyclic objectives cannot use genetic checkpoints")
    active.add(id(objective))
    try:
        cls = type(objective)
        counters = {"_n_scored", "_n_invalid"}
        if cls is QedObjective:
            fields: tuple[str, ...] = ()
        elif cls in (LogPBandObjective, MolecularWeightBandObjective, TpsaBandObjective):
            fields = ("lo", "hi")
        elif cls is SAScoreObjective:
            fields = ("easy", "hard")
        elif cls is MultiObjective:
            if set(vars(objective)) != counters | {"objectives", "weights"}:
                raise PersistenceError("modified multi-objective cannot use genetic checkpoints")
            children, weights = objective.objectives, objective.weights
            if (not isinstance(children, list) or not isinstance(weights, list) or len(children) < 2
                    or len(children) != len(weights)
                    or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights)):
                raise PersistenceError("invalid multi-objective configuration for genetic checkpoints")
            return {"type": "multi", "weights": weights,
                    "objectives": [_objective_identity(child, active=active) for child in children]}
        else:
            raise PersistenceError("genetic checkpoints support only the documented built-in objectives; "
                                   "custom and stateful objectives must run without checkpoint_dir")
        if set(vars(objective)) != counters | set(fields):
            raise PersistenceError("modified objective cannot use genetic checkpoints")
        values = {key: getattr(objective, key) for key in fields}
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
            raise PersistenceError("invalid objective parameter for genetic checkpoints")
        return {"type": cls.__name__, "parameters": values}
    finally:
        active.remove(id(objective))


def genetic_config_fingerprint(*, objective: Any, config: dict[str, Any],
                               canonical_pool: list[str], kept_seeds: list[str]) -> str:
    """Hash effective inputs, known objective parameters, code and runtime metadata."""
    import pharmaml

    root = Path(pharmaml.__file__).resolve().parent
    digest = hashlib.sha256()
    try:
        for path in sorted(root.rglob("*.py")):
            name = path.relative_to(root).as_posix().encode("utf-8")
            data = path.read_bytes()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
        rdkit_version = importlib.metadata.version("rdkit")
    except (OSError, importlib.metadata.PackageNotFoundError) as exc:
        raise PersistenceError(f"cannot fingerprint genetic checkpoint environment: {exc}") from exc
    record = {"schema": "genetic-checkpoint-identity/2", "objective": _objective_identity(objective),
              "config": config, "canonical_pool": canonical_pool, "kept_seeds": kept_seeds,
              "package_source_sha256": digest.hexdigest(), "rdkit": rdkit_version,
              "python": sys.version, "platform": platform.platform(), "machine": platform.machine()}
    try:
        blob = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"cannot fingerprint genetic checkpoint setup: {exc}") from exc
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
