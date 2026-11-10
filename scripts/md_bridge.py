#!/usr/bin/env python3
"""Stdin -> stdout JSON bridge for the Next.js ``/api/md/*`` routes.

Runs inside the toolkit virtualenv and wires ONLY the public
``pharmaml.md`` API (the round-249 molecular-dynamics package):

    op "trajectory" -> synthetic_trajectory -> rmsd_trajectory / rmsf /
        radius_of_gyration_trajectory / native_contact_fraction /
        contact_map / trajectory_pca / anm_fluctuations
    op "hbond"      -> synthetic_hbond_trajectory -> hbond_timeline +
        the per-bond donor-acceptor distance series (gmx hbond -num)
    op "saltbridge" -> synthetic_salt_bridge_trajectory ->
        salt_bridge_timeline + the per-bridge cation-anion distance
        series (getcontacts' ionic interaction)
    op "xvg"        -> read_ligand_folder semantics over pasted text
        (tempdir) -> series + stats + metric_correlations + summary_csv

Request (stdin, JSON)::

    {"op": "trajectory", "nFrames": 80, "seed": 0, "noise": 0.25,
     "contactCutoff": 4.5}
    {"op": "hbond", "nFrames": 60, "seed": 0, "noise": 0.03}
    {"op": "saltbridge", "nFrames": 60, "seed": 0, "noise": 0.03}
    {"op": "xvg", "files": [{"name": "ligand_rmsd.xvg", "content": "..."}],
     "smoothHbonds": true}

Response (stdout, JSON, camelCase keys for the TypeScript routes).
Errors print a one-line ``{"error": "..."}`` JSON on stdout with exit
code 2 so the route can surface them as HTTP 400/500 cleanly.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pharmaml.md import LigandResult

EXIT_OK = 0
EXIT_ERROR = 2

MAX_FILES = 12
MAX_FILE_CHARS = 200_000
MAX_FRAMES = 200


def _clean(value: Any) -> Any:
    """Recursively convert numpy scalars + NaN/inf to strict-JSON values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)  # numpy scalars, Paths, etc.


def _error(message: str) -> int:
    print(json.dumps({"error": message}))
    return EXIT_ERROR


def _finite_list(arr: Any) -> list[float]:
    return [round(float(v), 6) for v in arr]


def run_trajectory(request: dict[str, Any]) -> dict[str, Any]:
    from pharmaml.md import (
        anm_fluctuations,
        contact_map,
        native_contact_fraction,
        radius_of_gyration_trajectory,
        rmsd_trajectory,
        rmsf,
        synthetic_trajectory,
        trajectory_pca,
    )

    n_frames = request.get("nFrames", 80)
    seed = request.get("seed", 0)
    noise = request.get("noise", 0.25)
    contact_cutoff = request.get("contactCutoff", 4.5)

    if not isinstance(n_frames, int) or isinstance(n_frames, bool):
        return {"error": "nFrames must be an integer"}
    if not 4 <= n_frames <= MAX_FRAMES:
        return {"error": f"nFrames must be in [4, {MAX_FRAMES}]"}
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        return {"error": "seed must be an integer in [0, 2^31)"}
    if not isinstance(noise, (int, float)) or isinstance(noise, bool):
        return {"error": "noise must be a number"}
    if not 0.0 <= float(noise) <= 1.0:
        return {"error": "noise must be in [0, 1]"}
    if not isinstance(contact_cutoff, (int, float)) or isinstance(contact_cutoff, bool):
        return {"error": "contactCutoff must be a number"}
    if not 1.0 <= float(contact_cutoff) <= 12.0:
        return {"error": "contactCutoff must be in [1, 12] Å"}

    started = time.perf_counter()
    traj = synthetic_trajectory(
        n_frames=n_frames, seed=seed, noise=float(noise)
    )
    rmsd = rmsd_trajectory(traj)
    rg = radius_of_gyration_trajectory(traj)
    q = native_contact_fraction(traj, cutoff=float(contact_cutoff))
    fl = rmsf(traj)
    n_residues = traj.n_atoms // 3
    rmsf_residues = [
        float(mean) for mean in (
            fl[3 * r : 3 * r + 3].mean() for r in range(n_residues)
        )
    ]
    cmap = contact_map(traj, cutoff=float(contact_cutoff))
    pca = trajectory_pca(traj, n_components=2)
    projection = pca.transform_trajectory(traj)
    anm = anm_fluctuations(traj.coords[0])

    payload: dict[str, Any] = {
        "nFrames": n_frames,
        "nAtoms": traj.n_atoms,
        "nResidues": n_residues,
        "seed": seed,
        "noise": float(noise),
        "contactCutoff": float(contact_cutoff),
        "time": _finite_list(traj.time),
        "rmsd": _finite_list(rmsd),
        "rg": _finite_list(rg),
        "q": _finite_list(q),
        "rmsf": _finite_list(fl),
        "rmsfResidues": [round(v, 6) for v in rmsf_residues],
        "contactMap": [[round(float(v), 4) for v in row] for row in cmap],
        "pca": {
            "projection": [
                [round(float(a), 4), round(float(b), 4)]
                for a, b in projection
            ],
            "explainedVarianceRatio": [
                round(float(v), 6) for v in pca.explained_variance_ratio
            ],
        },
        "anmFluctuations": _finite_list(anm),
        "stats": {
            "rmsdMean": float(rmsd.mean()),
            "rmsdFinal": float(rmsd[-1]),
            "rgMean": float(rg.mean()),
            "qMean": float(q.mean()),
            "contactPairs": int((cmap > 0.05).sum() // 2),
        },
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "method": (
            "pharmaml.md — the deterministic alpha-helix demo "
            "(seeded noise + slow drift); every number computed by the "
            "real toolkit: rmsd_trajectory, radius_of_gyration_trajectory, "
            "native_contact_fraction, rmsf, contact_map, trajectory_pca, "
            "anm_fluctuations"
        ),
    }
    return payload


def run_xvg(request: dict[str, Any]) -> dict[str, Any]:
    from pharmaml.md import metric_correlations, read_ligand_folder
    from pharmaml.md.xvg import classify_metric

    files = request.get("files")
    smooth = request.get("smoothHbonds", True)
    if not isinstance(files, list) or not files:
        return {"error": "files must be a non-empty list"}
    if not isinstance(smooth, bool):
        return {"error": "smoothHbonds must be a boolean"}
    if len(files) > MAX_FILES:
        return {"error": f"at most {MAX_FILES} files per run"}
    for entry in files:
        if not isinstance(entry, dict):
            return {"error": "each file entry must be an object"}
        name = entry.get("name")
        content = entry.get("content")
        if not isinstance(name, str) or not name:
            return {"error": "each file needs a non-empty name"}
        if Path(name).name != name:
            return {"error": f"file name {name!r} must be a plain file name"}
        if not isinstance(content, str):
            return {"error": f"file {name!r} needs text content"}
        if len(content) > MAX_FILE_CHARS:
            return {"error": f"file {name!r} exceeds {MAX_FILE_CHARS} characters"}
        if classify_metric(name) is None:
            return {"error": (
                f"{name!r} is not a recognised MD export (expected e.g. "
                "ligand_rmsd.xvg, hbnum_intermolecular.ndx, "
                "binding_free_energy.xvg)"
            )}

    warnings: list[str] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pharmaml-md-") as tmp:
        folder = Path(tmp) / "ligand"
        folder.mkdir()
        for entry in files:
            (folder / str(entry["name"])).write_text(str(entry["content"]))
        try:
            ligand: LigandResult = read_ligand_folder(
                folder, smooth_hbonds=smooth
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the web UI
            return {"error": f"no parsable metric in the pasted files: {exc}"}
        for path in sorted(folder.iterdir()):
            if path.stem not in ligand.series and classify_metric(path.name):
                # Present in input, absent from the result: it was skipped.
                warnings.append(f"skipped {path.name} (unparseable or empty)")

    correlation = metric_correlations(ligand.series)
    payload: dict[str, Any] = {
        "ligand": ligand.to_dict(),
        "correlations": (
            None
            if correlation is None
            else {
                "labels": correlation[0],
                "matrix": [
                    [round(float(v), 4) for v in row] for row in correlation[1]
                ],
            }
        ),
        "summaryCsv": _summary_csv(ligand),
        "warnings": warnings,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "method": (
            "pharmaml.md.study.read_ligand_folder — the uploaded "
            "MD_analysis.py pipeline: .xvg/.ndx parsing, kJ→kcal, ps→ns, "
            "3-point H-bond smoothing"
        ),
    }
    return payload


def _summary_csv(ligand: LigandResult) -> str:
    lines = ["File,Mean,Stdev,Coefficient_of_Variation"]
    for key, series in ligand.series.items():
        stats = series.stats
        mean = "" if stats["mean"] is None else f"{stats['mean']:.6g}"
        std = "" if stats["std"] is None else f"{stats['std']:.6g}"
        cv = "" if stats["cv"] is None else f"{stats['cv']:.6g}"
        lines.append(f"{key},{mean},{std},{cv}")
    return "\n".join(lines) + "\n"


def run_hbond(request: dict[str, Any]) -> dict[str, Any]:
    """`gmx hbond -num` on the designed demo: two phase-shifted
    donor-H...acceptor pairs oscillating across the 3.5 A cutoff."""
    import numpy as np
    from pharmaml.md.hbond import (
        HBOND_ANGLE_CUTOFF,
        HBOND_DISTANCE_CUTOFF,
        hbond_timeline,
    )
    from pharmaml.md.trajectory import synthetic_hbond_trajectory

    n_frames = request.get("nFrames", 60)
    seed = request.get("seed", 0)
    noise = request.get("noise", 0.03)
    if not isinstance(n_frames, int) or isinstance(n_frames, bool):
        return {"error": "nFrames must be an integer"}
    if not 4 <= n_frames <= MAX_FRAMES:
        return {"error": f"nFrames must be in [4, {MAX_FRAMES}]"}
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        return {"error": "seed must be a non-negative integer"}
    if not isinstance(noise, (int, float)) or isinstance(noise, bool):
        return {"error": "noise must be a number"}
    if not 0.0 <= float(noise) <= 1.0:
        return {"error": "noise must be in [0, 1]"}

    started = time.perf_counter()
    traj = synthetic_hbond_trajectory(
        n_frames=n_frames, seed=seed, noise=float(noise)
    )
    donors = [(0, 1), (3, 4)]
    acceptors = [2, 5]
    counts = hbond_timeline(traj, donors=donors, acceptors=acceptors)
    distances = [
        [
            float(np.linalg.norm(traj.coords[f, a] - traj.coords[f, d]))
            for f in range(n_frames)
        ]
        for d, a in ((0, 2), (3, 5))
    ]

    payload: dict[str, Any] = {
        "nFrames": n_frames,
        "seed": seed,
        "noise": float(noise),
        "time": _finite_list(traj.time),
        "counts": _finite_list(counts),
        "distances": [
            [round(v, 4) for v in series] for series in distances
        ],
        "cutoffs": {
            "distance": HBOND_DISTANCE_CUTOFF,
            "angle": HBOND_ANGLE_CUTOFF,
        },
        "stats": {
            "mean": float(counts.mean()),
            "max": int(counts.max()),
            "bondedFraction": float((counts > 0).mean()),
            "bond0Fraction": float((counts >= 1).mean()),
            "bond1LeadsBond0": bool(
                sum(1 for c in counts if c == 2) > 0
            ),
        },
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "method": (
            "pharmaml.md — the designed two-bond demo (seeded, "
            "phase-shifted across the 3.5 A cutoff; each hydrogen "
            "rides its donor-acceptor axis so the angle criterion "
            "stays ~180 degrees and the distance gate is isolated); "
            "every number computed by the real toolkit: "
            "synthetic_hbond_trajectory + hbond_timeline"
        ),
    }
    return payload


def run_salt_bridge(request: dict[str, Any]) -> dict[str, Any]:
    """`getcontacts` ionic contacts on the designed demo: two
    phase-shifted cation-anion pairs oscillating across the 4.0 A
    cutoff (salt bridges are distance-only — no angle gate)."""
    import numpy as np
    from pharmaml.md.hbond import SALT_BRIDGE_CUTOFF, salt_bridge_timeline
    from pharmaml.md.trajectory import synthetic_salt_bridge_trajectory

    n_frames = request.get("nFrames", 60)
    seed = request.get("seed", 0)
    noise = request.get("noise", 0.03)
    if not isinstance(n_frames, int) or isinstance(n_frames, bool):
        return {"error": "nFrames must be an integer"}
    if not 4 <= n_frames <= MAX_FRAMES:
        return {"error": f"nFrames must be in [4, {MAX_FRAMES}]"}
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        return {"error": "seed must be a non-negative integer"}
    if not isinstance(noise, (int, float)) or isinstance(noise, bool):
        return {"error": "noise must be a number"}
    if not 0.0 <= float(noise) <= 1.0:
        return {"error": "noise must be in [0, 1]"}

    started = time.perf_counter()
    traj = synthetic_salt_bridge_trajectory(
        n_frames=n_frames, seed=seed, noise=float(noise)
    )
    positive = [0, 2]
    negative = [1, 3]
    counts = salt_bridge_timeline(
        traj, positive=positive, negative=negative
    )
    distances = [
        [
            float(np.linalg.norm(traj.coords[f, a] - traj.coords[f, p]))
            for f in range(n_frames)
        ]
        for p, a in ((0, 1), (2, 3))
    ]

    def _first_bonded(series: list[float]) -> int:
        for f, d in enumerate(series):
            if d <= SALT_BRIDGE_CUTOFF:
                return f
        return len(series)

    bridge_fractions = [
        float(sum(1 for d in series if d <= SALT_BRIDGE_CUTOFF)) / n_frames
        for series in distances
    ]
    payload: dict[str, Any] = {
        "nFrames": n_frames,
        "seed": seed,
        "noise": float(noise),
        "time": _finite_list(traj.time),
        "counts": _finite_list(counts),
        "distances": [
            [round(v, 4) for v in series] for series in distances
        ],
        "cutoffs": {
            "distance": SALT_BRIDGE_CUTOFF,
        },
        "stats": {
            "mean": float(counts.mean()),
            "max": int(counts.max()),
            "bondedFraction": float((counts > 0).mean()),
            "bridge1Fraction": bridge_fractions[0],
            "bridge2Fraction": bridge_fractions[1],
            "bridge2LeadsBridge1": bool(
                _first_bonded(distances[1]) < _first_bonded(distances[0])
            ),
        },
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "method": (
            "pharmaml.md — the designed two-bridge ionic demo (seeded, "
            "phase-shifted across the 4.0 A cutoff; salt bridges are "
            "distance-only, so unlike the H-bond demo there is no angle "
            "gate to isolate — this is exactly getcontacts' ionic "
            "criterion); every number computed by the real toolkit: "
            "synthetic_salt_bridge_trajectory + salt_bridge_timeline"
        ),
    }
    return payload


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _error(f"bridge request is not valid JSON: {exc}")

    if not isinstance(request, dict):
        return _error("bridge request must be a JSON object")

    op = request.get("op")
    if op == "trajectory":
        payload = run_trajectory(request)
    elif op == "hbond":
        payload = run_hbond(request)
    elif op == "saltbridge":
        payload = run_salt_bridge(request)
    elif op == "xvg":
        payload = run_xvg(request)
    else:
        return _error("op must be 'trajectory', 'hbond', 'saltbridge' or 'xvg'")

    if "error" in payload:
        _error_failed = payload["error"]
        if isinstance(_error_failed, str):
            return _error(_error_failed)
        return _error("unknown bridge failure")
    print(json.dumps(_clean(payload)))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
