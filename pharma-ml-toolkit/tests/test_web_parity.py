"""Web-mirror parity guards (100x-115).

The shipped TS modules under ``<repo>/src/lib`` mirror library math:

- ``bayes-acquisition.ts`` (the GP band's EI/UCB overlay) vs
  ``scipy.stats.norm`` — the 100x-114 pair;
- ``docking-configs.ts`` (GPF/DPF writers, grid box, Vina log scan) vs
  ``pharmaml.docking`` — the 100x-112 pair.

The paired scratch probes verify them, but until now nothing FAILED the
pytest suite when they broke.  These guards run BOTH sides of each pair
from inside the suite — the python twin (via ``sys.executable``), then
the shipped TS module (via a ``bun`` subprocess) — so a regression in
either language fails CI with the probe's own diagnostics.

Skipped honestly when the checkout is the zip layout (web side absent —
the zip ships the toolkit only) or bun is not on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOLKIT_ROOT.parent
ACQUISITION_TS = REPO_ROOT / "src" / "lib" / "bayes-acquisition.ts"
BUN = shutil.which("bun")

needs_web_side = pytest.mark.skipif(
    not ACQUISITION_TS.is_file() or BUN is None,
    reason="web checkout (<repo>/src/lib TS mirrors) + bun runtime required; "
    "the zip-only tree skips — its probes live outside the shipped tree",
)


def _run(command: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{command} exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc.stdout


class TestAcquisitionParity:
    """bayes-acquisition.ts vs scipy.stats.norm (the 100x-114 pair)."""

    @needs_web_side
    def test_shipped_ts_module_matches_scipy(self) -> None:
        # 1) python twin: emits the FIXED grid + scipy gold, asserts its
        #    own line-by-line mirror honours the AS-26.2.17 bound.
        _run(
            [sys.executable, str(TOOLKIT_ROOT / "scratch" / "verify_acquisition_parity_114.py")],
            TOOLKIT_ROOT,
        )
        # 2) bun twin: runs the SHIPPED src/lib/bayes-acquisition.ts on
        #    the same grid (imports the module the browser imports, not
        #    a mirror) and exits non-zero outside tolerance.
        _run([BUN, "run", "scratch/verify_acquisition_parity_114.ts"], REPO_ROOT)
        # 3) cross-language verdict with the SAME tolerances the twins
        #    enforce — restated here so a failure names the quantity.
        ts = json.loads(Path("/tmp/acq_ts_114.json").read_text())
        ref = json.loads(Path("/tmp/acq_ref_114.json").read_text())
        tolerances = {"cdf": 1.5e-7, "pdf": 1e-12, "ei": 1e-6, "ucb": 1e-12}
        for quantity, tol in tolerances.items():
            worst = max(abs(a - b) for a, b in zip(ts[quantity], ref[quantity]))
            assert worst <= tol, (
                f"{quantity}: shipped TS vs scipy max|d| = {worst:.3e} > {tol:.1e}"
            )


class TestDockingConfigsParity:
    """docking-configs.ts vs pharmaml.docking (the 100x-112 pair)."""

    @needs_web_side
    def test_python_twin_still_asserts(self) -> None:
        """The library side: real write_config_batch on the shared example."""
        _run(
            [sys.executable, str(TOOLKIT_ROOT / "scratch" / "probe_batch_parity_112.py")],
            TOOLKIT_ROOT,
        )

    @needs_web_side
    def test_ts_mirror_byte_parity(self) -> None:
        """The web side: GPF/DPF byte-parity, box math, Vina scan ranking.

        The probe carries the Python writers' output INLINE, so it fails
        loudly when the library's config formats change — the developer
        then updates both twins, which is exactly the contract.
        """
        _run([BUN, "run", "scratch/probe_batch_parity_112.ts"], REPO_ROOT)
