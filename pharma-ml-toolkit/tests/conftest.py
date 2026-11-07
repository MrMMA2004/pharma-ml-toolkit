"""Shared pytest fixtures: sample docking outputs and synthetic QSAR data."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


def pdbqt_atom(serial: int, name: str, x: float, y: float, z: float, atom_type: str) -> str:
    """Column-aligned PDBQT ATOM record (cols 31-54 = xyz)."""
    return (
        f"ATOM  {serial:5d} {name:>4s} LIG A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}    0.000 {atom_type:>2s}"
    )


@pytest.fixture(scope="session")
def sample_ligand_pdbqt() -> str:
    """Four-atom ligand with known coordinates."""
    return "\n".join(
        [
            "REMARK  Name = sample_ligand",
            "ROOT",
            pdbqt_atom(1, "C", 2.345, -0.802, 0.645, "C"),
            pdbqt_atom(2, "O", 3.456, -1.234, 1.789, "OA"),
            pdbqt_atom(3, "N", 1.234, 0.123, -0.456, "N"),
            pdbqt_atom(4, "HD", 0.987, 1.012, -1.234, "HD"),
            "ENDROOT",
            "TORSDOF 2",
        ]
    ) + "\n"


@pytest.fixture(scope="session")
def sample_dlg() -> str:
    """Synthetic AutoDock4 DLG with a clustering histogram table."""
    return """\
AutoDock4 log file (synthetic)

________
CLUSTERING HISTOGRAM
__________________
______________________________________________________________________
Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster
_____________|_______________|_____|_____________|________________
     1       |     -11.5     |  3  |    -11.2    |      7 |
     2       |     -10.8     |  9  |    -10.5    |     34 |
     3       |      -9.9     | 17  |     -9.7    |     52 |
     4       |      -9.1     | 22  |     -9.0    |     12 |
____________|_______________|_____|_____________|________________

Successful Completion
"""


@pytest.fixture(scope="session")
def sample_vina_output() -> str:
    """Synthetic Vina stdout table."""
    return """\
Estimated Free Energy of Binding   = -10.3 kcal/mol
Modes | Affinity | dist from best mode
      |          | rmsd l.b. | rmsd u.b.
------+----------+----------+----------
   1      -10.3          0          0
   2       -9.6      1.842      2.301
   3       -9.4      2.105      2.777
   4       -8.9      2.733      3.418
------+----------+----------+----------
"""


@pytest.fixture(scope="session")
def classification_frame() -> pd.DataFrame:
    """Small synthetic QSAR classification table."""
    X, y = make_classification(
        n_samples=140, n_features=6, n_informative=4, random_state=42
    )
    frame = pd.DataFrame(X, columns=[f"descr_{i:02d}" for i in range(6)])
    frame["Target"] = y
    frame["SMILES"] = ["CCO"] * len(frame)  # non-numeric passthrough column
    return frame


@pytest.fixture(scope="session")
def regression_frame() -> pd.DataFrame:
    """Small synthetic pIC50 regression table."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(140, 6))
    weight = np.array([1.5, -2.0, 0.0, 3.0, 0.5, -1.0])
    frame = pd.DataFrame(X, columns=[f"descr_{i:02d}" for i in range(6)])
    frame["pIC50"] = 6.0 + X @ weight + rng.normal(scale=0.3, size=140)
    return frame


@pytest.fixture(scope="session")
def xvfb_display() -> Any:
    """ONE virtual display per pytest session (the round-209 collapse fix).

    The round-209 forensics: a fresh Xvfb per class/module combined with
    a stale Tk display connection from a destroyed app means the NEXT Tk
    event pump touches the broken connection and Tk 9's X error handler
    terminates the whole process (``X connection to :N broken`` ->
    ``exit(1)``, rc=1, zero output).  A session-scoped server is torn
    down exactly once -- after the last test, when nothing pumps Tk any
    more -- so a mid-session server death cannot happen, and the stale
    connection an app leaves behind stays harmless for the whole run.
    """
    import os as _os
    import socket as _socket
    import subprocess as _subprocess
    import time as _time

    display = ":96"
    process = _subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x800x24"],
        stdout=_subprocess.DEVNULL,
        stderr=_subprocess.DEVNULL,
    )
    socket_path = Path(f"/tmp/.X11-unix/X{display[1:]}")
    deadline = _time.time() + 10.0
    ready = False
    while _time.time() < deadline:
        if socket_path.exists():
            probe = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            try:
                probe.connect(str(socket_path))
                ready = True
            except OSError:
                pass
            finally:
                probe.close()
        if ready:
            break
        _time.sleep(0.1)
    assert ready, "Xvfb did not come up on :96 within 10 s"
    _os.environ["DISPLAY"] = display
    try:
        yield display
    finally:
        process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=5)
