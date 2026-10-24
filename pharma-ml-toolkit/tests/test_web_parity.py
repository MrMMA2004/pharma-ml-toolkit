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


class TestVariableGenesParity:
    """variable-genes.ts vs pharmaml.biology.most_variable_genes (the
    100x-124 pair).

    The web Cell Line tab's "Rank variable genes (MAD)" disclosure
    mirrors the desktop's MAD pre-filter in TS; this guard runs BOTH
    sides on the same hand-crafted matrices (a 9-line odd panel and an
    8-line even panel, each with MAD ties, a constant gene, negatives
    and decimals) and asserts row-by-row agreement: the gene ORDER is
    exact (ties broken by name ascending on both sides — the library
    contract) and the MAD values compare as floats (both sides compute
    median and median-of-|deviations| with the average-of-two-middles
    convention in the same IEEE order).
    """

    MAD_TS = REPO_ROOT / "src" / "lib" / "variable-genes.ts"

    @pytest.mark.skipif(
        not MAD_TS.is_file() or BUN is None,
        reason="web checkout (<repo>/src/lib/variable-genes.ts) + bun runtime "
        "required; the zip-only tree skips — its probes live outside the "
        "shipped tree",
    )
    def test_mad_ranking_parity(self) -> None:
        _run(
            [sys.executable, str(TOOLKIT_ROOT / "scratch" / "probe_mad_parity_124.py")],
            TOOLKIT_ROOT,
        )
        _run([BUN, "run", "scratch/probe_mad_parity_124.ts"], REPO_ROOT)
        ts = json.loads(Path("/tmp/mad_ts_124.json").read_text())
        ref = json.loads(Path("/tmp/mad_ref_124.json").read_text())
        assert ts["topN"] == ref["topN"]
        for panel in ("odd", "even"):
            ts_rows, ref_rows = ts[panel], ref[panel]
            assert len(ts_rows) == len(ref_rows) > 0, panel
            for ts_row, ref_row in zip(ts_rows, ref_rows):
                assert ts_row["rank"] == ref_row["rank"], (panel, ts_row, ref_row)
                assert ts_row["gene"] == ref_row["gene"], (panel, ts_row, ref_row)
                assert abs(ts_row["mad"] - ref_row["mad"]) <= 1e-12, (panel, ts_row, ref_row)
        # the hand-crafted fixtures pin the semantics: the odd panel's
        # runaway-spread gene leads, the constant gene is last, and the
        # MAD ties really are ties ranked alphabetically
        odd = ref["odd"]
        assert odd[0]["gene"] == "GENE_I" and odd[0]["mad"] == 200.0
        assert odd[-1]["gene"] == "GENE_C" and odd[-1]["mad"] == 0.0
        tie_genes = [row["gene"] for row in odd if row["mad"] == 2.0]
        assert tie_genes == ["GENE_A", "GENE_B", "GENE_E", "GENE_F", "GENE_J"], tie_genes


class TestLearningCurveContract:
    """learning-curve.ts vs pharmaml.viz.plot_learning_curve (the
    100x-125 pair).

    The web ML Training tab's "Learning curve" card mirrors the desktop
    ML card's button.  The per-point numbers are engine-specific by
    design (sklearn vs the TS twin), so this guard pins the CONTRACT
    the two cards share: the default size schedule (numpy
    ``linspace(0.1, 1.0, 8)`` — the TS linspace must reproduce numpy's
    arithmetic bit-for-bit), the gap threshold (0.15, read from the
    shipped Python source), and the two verdict strings — the Python
    side reads them off REAL figures the shipped library produced (one
    overfitting tree, one converging OLS), the TS side formats the SAME
    parsed gaps, and the strings compare byte-for-byte.  The reduction
    engine itself is pinned on the TS side against a hand-computed
    fixture (prefix subsampling of each fold's train part, 1-based fold
    slots, population std).
    """

    LC_TS = REPO_ROOT / "src" / "lib" / "learning-curve.ts"

    @pytest.mark.skipif(
        not LC_TS.is_file() or BUN is None,
        reason="web checkout (<repo>/src/lib/learning-curve.ts) + bun runtime "
        "required; the zip-only tree skips — its probes live outside the "
        "shipped tree",
    )
    def test_learning_curve_contract(self) -> None:
        _run(
            [sys.executable, str(TOOLKIT_ROOT / "scratch" / "probe_lc_parity_125.py")],
            TOOLKIT_ROOT,
        )
        ref = json.loads(Path("/tmp/lc_ref_125.json").read_text())
        # the TS twin formats the very gaps the Python figures printed
        gaps = [str(case["gap"]) for case in ref["cases"] if case["gap"] >= 0]
        _run(
            [BUN, "run", "scratch/probe_lc_parity_125.ts", *gaps],
            REPO_ROOT,
        )
        ts = json.loads(Path("/tmp/lc_ts_125.json").read_text())

        # 1. the default schedule: same 8 fractions, bit-for-bit (both
        #    compute start + step*i with the last element pinned to 1)
        assert len(ts["fractions"]) == len(ref["fractions"]) == 8
        for ts_fraction, ref_fraction in zip(ts["fractions"], ref["fractions"]):
            assert abs(ts_fraction - ref_fraction) <= 1e-12, (ts_fraction, ref_fraction)
        assert ts["fractions"][-1] == 1.0 and ref["fractions"][-1] == 1.0
        assert ref["fractions"][0] == 0.1

        # 2. the gap threshold: both sides agree on the 0.15 cut
        assert ts["threshold"] == ref["threshold"] == 0.15

        # 3. the verdict strings: the TS twin formats the SAME gap the
        #    Python figure printed -> byte-identical strings
        for case in ref["cases"]:
            if case["gap"] < 0:
                continue
            key = str(case["gap"])
            assert key in ts["verdicts"], (key, sorted(ts["verdicts"]))
            assert ts["verdicts"][key] == case["verdict"], (key, ts["verdicts"][key], case["verdict"])
        # the boundary: the shared 0.15 cut flips the verdict between
        # 0.149999 and 0.150001 (both sides exclusive at the threshold)
        converging = "curves converging — additional data has diminishing returns"
        assert ts["verdicts"]["0.149999"] == converging
        assert ts["verdicts"]["0.150001"].startswith("gap ≈ 0.15")
        assert ts["verdicts"]["0.150001"] != converging

        # 4. the reduction engine: the hand-computed fixture (prefix
        #    cut, 1-based slots, population std) — see the probe source
        engine = ts["engine"]
        assert [(p["size"], p["trainMean"], p["trainStd"], p["valMean"], p["valStd"]) for p in engine] == [
            (2, 17, 5, 101.5, 0.5),
            (4, 19, 5, 101.5, 0.5),
        ]


class TestSessionReportContract:
    """session-report.ts (the web session collector) → the repo-root
    report bridge → a REAL pharmaml.viz.HTMLReport (the 100x-127 pair).

    The web twin's "Session report" card collects the persisted
    playground state into a section payload; the bridge composes real
    matplotlib figures and assembles them through the SAME
    self-contained themed HTMLReport the desktop playground saves.
    This guard runs the round trip from inside the suite: the SHIPPED
    TS collector over a fixed fixture snapshot, its payload piped
    through the shipped bridge script, and the composed document
    asserted against the contract the desktop report pins — theme
    baked as the ``data-theme`` attribute, the per-reader toggle with
    its localStorage persistence, base64-embedded figures, zero
    external references."""

    COLLECTOR_TS = REPO_ROOT / "src" / "lib" / "session-report.ts"
    BRIDGE_PY = REPO_ROOT / "scripts" / "report_bridge.py"

    @pytest.mark.skipif(
        not COLLECTOR_TS.is_file() or not BRIDGE_PY.is_file() or BUN is None,
        reason="web checkout (<repo>/src/lib/session-report.ts + "
        "<repo>/scripts/report_bridge.py) + bun runtime required; the "
        "zip-only tree skips — its probes live outside the shipped tree",
    )
    def test_session_report_round_trip(self) -> None:
        # 1. the TS side: the SHIPPED collector over the fixed fixture
        _run([BUN, "run", "scratch/probe_session_report_127.ts"], REPO_ROOT)
        ts = json.loads(Path("/tmp/sr_ts_127.json").read_text())
        payload = json.loads(Path("/tmp/sr_payload_127.json").read_text())

        # 2. the census: the fixture session ships 4 figures (parity,
        #    learning curve, PK profile, PD effect), 2 tables (NCA rows,
        #    run history) and 2 metric cards (training + curve)
        assert ts["census"] == {"figures": 4, "tables": 2, "metrics": 2}
        assert "Predicted vs observed" in ts["sectionTitles"]
        assert "Learning curve — R² vs training samples" in ts["sectionTitles"]
        assert "PK regimen — concentration vs time" in ts["sectionTitles"]
        assert "PD effect — emax model" in ts["sectionTitles"]
        assert "NCA observed data" in ts["sectionTitles"]
        assert "ML run history" in ts["sectionTitles"]
        # the parity figure's trace contract: model line + ±band +
        # test scatter + train scatter, in collector order
        assert ts["parityTraceStyles"] == [
            "model prediction:line",
            "± band:band",
            "test observations:scatter",
            "train observations:scatter",
        ]
        # metric values format through the collector's fmtCell
        assert ["model", "ridge"] in ts["firstMetrics"]
        assert ["R²", "0.9123"] in ts["firstMetrics"]
        assert ["RMSE", "0.2111"] in ts["firstMetrics"]

        # 3. the bridge: the collector's payload composes into a real
        #    HTMLReport (the same builder the desktop report walks)
        result = subprocess.run(
            [sys.executable, str(self.BRIDGE_PY)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        composed = json.loads(result.stdout)
        assert composed["figures"] == 4
        assert composed["tables"] == 2
        assert composed["theme"] == "dark"
        html = composed["html"]
        assert html.startswith("<!DOCTYPE html>")
        assert "<html lang='en' data-theme='dark'>" in html
        assert "id='theme-toggle'" in html
        assert "localStorage" in html and "pharmaml-report-theme" in html
        assert html.count("<img") == 4
        assert html.count("<table") == 2
        assert "data:image/png;base64," in html
        # self-contained: no external stylesheet/script/image refs
        assert "<link" not in html
        assert "src='http" not in html and "href='http" not in html
        assert "PharmaML web playground session" in html

    @pytest.mark.skipif(
        not BRIDGE_PY.is_file(),
        reason="web checkout (<repo>/scripts/report_bridge.py) required",
    )
    def test_bridge_rejects_invalid_theme(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.BRIDGE_PY)],
            input=json.dumps({"theme": "neon", "sections": []}),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 2
        error = json.loads(result.stdout)
        assert "theme" in error["error"]
