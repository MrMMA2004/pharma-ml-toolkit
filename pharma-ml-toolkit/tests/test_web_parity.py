"""Web-mirror parity guards (100x-115, extended 100x-117).

The shipped TS modules under ``<repo>/src/lib`` mirror library math:

- ``bayes-acquisition.ts`` (the GP band's EI/UCB overlay) vs
  ``scipy.stats.norm`` — the 100x-114 pair;
- ``docking-configs.ts`` (GPF/DPF writers, grid box, Vina log scan) vs
  ``pharmaml.docking`` — the 100x-112 pair;
- ``hyperband-rungs.ts`` (the rung-scatter row fold) vs its own
  pinned contract — the 100x-117 pair;
- ``ml-ad.ts`` (the AD card's distance domain) vs
  ``pharmaml.ml.applicability`` — the 100x-118 pair.

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
RUNG_FOLD_TS = REPO_ROOT / "src" / "lib" / "hyperband-rungs.ts"
BUN = shutil.which("bun")

needs_web_side = pytest.mark.skipif(
    not ACQUISITION_TS.is_file() or BUN is None,
    reason="web checkout (<repo>/src/lib TS mirrors) + bun runtime required; "
    "the zip-only tree skips — its probes live outside the shipped tree",
)

needs_rung_fold = pytest.mark.skipif(
    not RUNG_FOLD_TS.is_file() or BUN is None,
    reason="web checkout (<repo>/src/lib/hyperband-rungs.ts) + bun runtime "
    "required; the zip-only tree skips — its probes live outside the "
    "shipped tree",
)
AD_TS_PATH = REPO_ROOT / "src" / "lib" / "ml-ad.ts"
needs_ad_parity = pytest.mark.skipif(
    not AD_TS_PATH.is_file() or BUN is None,
    reason="web checkout (<repo>/src/lib/ml-ad.ts) + bun runtime required; "
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
            assert worst <= tol, f"{quantity}: shipped TS vs scipy max|d| = {worst:.3e} > {tol:.1e}"


class TestHyperbandRungFold:
    """hyperband-rungs.ts vs its pinned fold contract (100x-117).

    The bracket-best fold (100x-115/116, extracted from
    ml-playground.tsx this round) is pure and mirrorable: the probe
    pins the winner-first contract, the exactly-one-bracket-best
    invariant, the tie-first-occurrence rule, the filter defaults,
    the NaN-resource fix, and row-parity with the original quadratic
    algorithm on a 241-trial seeded grid — all against the SHIPPED
    module the browser imports.
    """

    @needs_rung_fold
    def test_fold_contract(self) -> None:
        _run([BUN, "run", "scratch/probe_rung_fold_117.ts"], REPO_ROOT)


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


class TestApplicabilityDomainParity:
    """ml-ad.ts vs pharmaml.ml.applicability (the 100x-118 pair).

    The web AD card (Data & ML playground) mirrors the uploaded
    workflow's distance half in TS; this guard runs BOTH sides of the
    pair from inside the suite so a regression in either language
    fails CI with the probe's own diagnostics.
    """

    AD_TS = REPO_ROOT / "src" / "lib" / "ml-ad.ts"

    @needs_ad_parity
    def test_ad_knn_parity(self) -> None:
        """The python twin writes the numeric readout (train/test mean
        distances, threshold, verdicts) from ``knn_applicability_domain``
        on the shared minstd dataset; the bun twin runs the SHIPPED
        ``ml-ad.ts`` (fitStandardizer + knnDomain) on the identical rows
        and asserts agreement to 1e-9 plus the self-exclusion invariant.
        """
        _run(
            [sys.executable, str(TOOLKIT_ROOT / "scratch" / "probe_ad_parity_118.py")],
            TOOLKIT_ROOT,
        )
        _run([BUN, "run", "scratch/probe_ad_parity_118.ts"], REPO_ROOT)


class TestUnitConverterParity:
    """unit-converter.ts vs pharmaml.pharma.units (the 100x-123 pair).

    The web Calculators card's unit converter mirrors the desktop's
    dimension-aware converter in TS; this guard runs BOTH sides on the
    same matrix (every same-dimension unit pair of every dimension at
    the sample value 7.5, the dimension_of answers, the sorted unit
    lists, and both error paths) and asserts field-by-field agreement:
    conversions compare as EXACT floats (both sides compute
    ``value * from / to`` in the same IEEE order), the cross-dimension
    message byte-identically, and the unknown-unit message modulo the
    JSON-vs-repr quoting of the supported-units dictionary.
    """

    UNITS_TS = REPO_ROOT / "src" / "lib" / "unit-converter.ts"

    @pytest.mark.skipif(
        not UNITS_TS.is_file() or BUN is None,
        reason="web checkout (<repo>/src/lib/unit-converter.ts) + bun runtime "
        "required; the zip-only tree skips — its probes live outside the "
        "shipped tree",
    )
    def test_units_matrix_parity(self) -> None:
        _run(
            [sys.executable, str(TOOLKIT_ROOT / "scratch" / "probe_units_parity_123.py")],
            TOOLKIT_ROOT,
        )
        _run([BUN, "run", "scratch/probe_units_parity_123.ts"], REPO_ROOT)
        ts = json.loads(Path("/tmp/units_ts_123.json").read_text())
        ref = json.loads(Path("/tmp/units_ref_123.json").read_text())
        assert len(ts["conversions"]) == len(ref["conversions"]) > 150
        for a, b in zip(ts["conversions"], ref["conversions"]):
            assert (a["from"], a["to"]) == (b["from"], b["to"]), (a, b)
            assert a["value"] == b["value"], (a, b)  # exact IEEE parity
        assert ts["dimensions"] == ref["dimensions"]
        assert ts["unitLists"] == ref["unitLists"]
        assert ts["errors"]["crossDimension"] == ref["errors"]["crossDimension"]
        # unknown-unit: identical prefix + identical parsed payload (the
        # quoting of the supported-units dict differs by language --
        # Python repr vs JSON -- so parse both sides back to structures)
        import ast

        prefix = "unknown unit 'bogus'; supported units by dimension: "
        payloads = {}
        for side, message in (
            ("ts", ts["errors"]["unknownUnit"]),
            ("ref", ref["errors"]["unknownUnit"]),
        ):
            assert message.startswith(prefix), message
            payloads[side] = message[len(prefix) :]
        supported_ts = json.loads(payloads["ts"])
        supported_ref = ast.literal_eval(payloads["ref"])
        assert {k: sorted(v) for k, v in supported_ts.items()} == {
            k: sorted(v) for k, v in supported_ref.items()
        }
