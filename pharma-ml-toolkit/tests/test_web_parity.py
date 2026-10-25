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
        # (100x-131) the census gained the images axis (the surgery
        # grids); this no-surgery fixture pins images: 0 — the 127
        # contract otherwise unchanged
        assert ts["census"] == {
            "figures": 4, "tables": 2, "metrics": 2, "images": 0,
        }
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


class TestSessionReportSections:
    """Round 100x-131: the session report's SECTION CONTROL + the
    SURGERY feed, the full contract pair.

    The TS side (the SHIPPED collector over a fixed fixture with one
    topology surgery run) pins the per-group options semantics —
    ``{pkpd: false}`` drops the two regimen figures, ``{surgery:
    false}`` drops the three surgery sections, a surgery-only report
    ships exactly those three, ``sessionGroupsAvailable`` tells the UI
    which chips to render — and the Python side pipes the all-on
    payload (WITH the surgery grid) through the SHIPPED report bridge,
    pinning the new ``image`` section kind: decoded, PNG-magic-checked,
    size-capped, re-composed as a full-bleed matplotlib figure and
    embedded through the SAME ``add_figure`` contract every chart
    rides.  The static wiring pins (store persistence + the cards'
    feed + the UI's toggle chips + the clear-recent affordance) keep
    the round's surfaces from silently disappearing between rounds."""

    COLLECTOR_TS = REPO_ROOT / "src" / "lib" / "session-report.ts"
    BRIDGE_PY = REPO_ROOT / "scripts" / "report_bridge.py"
    STORE_TS = REPO_ROOT / "src" / "lib" / "store.ts"
    SURGERY_TSX = (
        REPO_ROOT / "src" / "components" / "explorer" / "playground" / "surgery-cards.tsx"
    )
    REPORT_TSX = (
        REPO_ROOT / "src" / "components" / "explorer" / "playground" / "session-report.tsx"
    )
    ROUTE_TS = REPO_ROOT / "src" / "app" / "api" / "report" / "session" / "route.ts"
    CARD_FINDER_TSX = (
        REPO_ROOT / "src" / "components" / "explorer" / "playground" / "card-finder.tsx"
    )

    needs_round = pytest.mark.skipif(
        not COLLECTOR_TS.is_file() or not BRIDGE_PY.is_file() or BUN is None,
        reason="web checkout (<repo>/src + <repo>/scripts) + bun runtime "
        "required; the zip-only tree skips — its probes live outside the "
        "shipped tree",
    )

    @needs_round
    def test_collector_section_control_pins(self) -> None:
        # the TS probe exits non-zero on drift and prints its censuses
        _run([BUN, "run", "scratch/probe_session_report_131.ts"], REPO_ROOT)
        ts = json.loads(Path("/tmp/sr_ts_131.json").read_text())
        # all-on WITH the topology surgery run: the 127 census + the
        # surgery's metrics card + mutants table + structure grid
        assert ts["allOnCensus"] == {
            "figures": 4, "tables": 3, "metrics": 3, "images": 1,
        }
        # pkpd off drops exactly the two regimen figures
        assert ts["noPkpd"] == {
            "figures": 2, "tables": 3, "metrics": 3, "images": 1,
        }
        # surgery off drops the metrics + table + image (the 127 census)
        assert ts["noSurgery"] == {
            "figures": 4, "tables": 2, "metrics": 2, "images": 0,
        }
        # a surgery-only report ships exactly the three sections
        assert ts["surgeryOnlyCensus"] == {
            "figures": 0, "tables": 1, "metrics": 1, "images": 1,
        }
        assert ts["surgeryOnlyTitles"] == [
            "Ring topology moves",
            "Ring topology mutants",
            "Ring topology mutants — 3 structure panel(s)",
        ]
        # the honest availability map: a fresh session shows only PK / PD
        assert ts["availableEmpty"] == {
            "ml": False, "learningCurve": False, "pkpd": True,
            "nca": False, "history": False, "surgery": False,
        }
        assert ts["available"]["surgery"] is True
        # the image section carries the grid PNG data URI verbatim
        assert ts["imageSection"]["kind"] == "image"
        assert ts["imageSection"]["dataUri"].startswith("data:image/png;base64,")
        # the mutants table pins the piperidine set (the bridge contract)
        assert ts["mutantRows"] == [
            [1, "C1CCCC1"], [2, "C1CCCNCC1"], [3, "C1CCNC1"],
        ]

    @needs_round
    def test_bridge_composes_the_surgery_image_section(self) -> None:
        payload = json.loads(Path("/tmp/sr_payload_131.json").read_text())
        result = subprocess.run(
            [sys.executable, str(self.BRIDGE_PY)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        composed = json.loads(result.stdout)
        # 4 chart figures + the surgery grid = 5 embedded <img> blocks
        assert composed["figures"] == 5
        assert composed["tables"] == 3
        html = composed["html"]
        assert html.count("<img") == 5
        assert html.count("<table") == 3
        # the surgery caption rides as the image's own figcaption
        assert "Ring topology mutants — 3 structure panel(s)" in html
        # self-contained: no external references
        assert "<link" not in html
        assert "src='http" not in html and "href='http" not in html

    @pytest.mark.skipif(
        not BRIDGE_PY.is_file(),
        reason="web checkout (<repo>/scripts/report_bridge.py) required",
    )
    def test_bridge_rejects_broken_image_sections(self) -> None:
        base = {
            "kind": "image",
            "title": "Surgery — ring topology mutants",
        }
        for data_uri, fragment in (
            ("data:image/png;base64,not-valid-!!", "not valid base64"),
            ("data:image/jpeg;base64,aGVsbG8=", "dataUri"),
            ("data:image/png;base64," + "A" * 64, "PNG payload"),
        ):
            result = subprocess.run(
                [sys.executable, str(self.BRIDGE_PY)],
                input=json.dumps(
                    {"theme": "light", "sections": [dict(base, dataUri=data_uri)]}
                ),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert result.returncode == 2, data_uri[:40]
            error = json.loads(result.stdout)["error"]
            assert fragment in error, (data_uri[:40], error)

    # -- the static wiring pins ------------------------------------------- #
    @pytest.mark.skipif(
        not STORE_TS.is_file(),
        reason="web checkout (<repo>/src/lib/store.ts) required",
    )
    def test_store_persists_the_surgery_result(self) -> None:
        store = self.STORE_TS.read_text(encoding="utf-8")
        assert 'const SURGERY_RESULT_STORAGE_KEY = "pharmaml.surgeryResult";' in store
        assert "surgeryResult: loadSurgeryResult()," in store
        assert "setSurgeryResult" in store
        assert "validateSurgeryResult" in store

    @pytest.mark.skipif(
        not SURGERY_TSX.is_file(),
        reason="web checkout (<repo>/src components) required",
    )
    def test_surgery_cards_feed_the_session(self) -> None:
        surgery = self.SURGERY_TSX.read_text(encoding="utf-8")
        # both cards persist a successful run (the latest run wins)
        assert surgery.count("setSurgeryResult(data);") == 2
        assert 's.setSurgeryResult' in surgery

    @pytest.mark.skipif(
        not REPORT_TSX.is_file(),
        reason="web checkout (<repo>/src components) required",
    )
    def test_report_card_carries_the_toggle_chips(self) -> None:
        report = self.REPORT_TSX.read_text(encoding="utf-8")
        assert "sessionGroupsAvailable" in report
        assert "Include in report" in report
        assert "aria-pressed" in report
        assert 'const SECTION_TOGGLE_STORAGE_KEY = "pharmaml.reportSections";' in report
        assert "surgeryResult" in report
        assert "Nothing would ship" in report

    @pytest.mark.skipif(
        not ROUTE_TS.is_file(),
        reason="web checkout (<repo>/src route) required",
    )
    def test_route_accepts_the_image_kind(self) -> None:
        route = self.ROUTE_TS.read_text(encoding="utf-8")
        assert 'z.literal("image")' in route
        assert "startsWith" in route
        assert "dataUri" in route

    @pytest.mark.skipif(
        not CARD_FINDER_TSX.is_file(),
        reason="web checkout (<repo>/src components) required",
    )
    def test_web_finder_clear_recent_pins(self) -> None:
        finder = self.CARD_FINDER_TSX.read_text(encoding="utf-8")
        assert "clearRecentIds" in finder
        assert "setRecentCleared(true);" in finder
        assert "recentCleared" in finder
        # the affordance rides INSIDE the row (no row, no Clear)
        assert "Forget every remembered jump" in finder


class TestSessionReportOrder:
    """Round 100x-132: the session report's WALKING ORDER (the web
    twin's reorderable chips) + the desktop picklist's wiring pins.

    The TS side (the SHIPPED collector over the fixed 131 fixture)
    pins the reorder semantics: WHOLE groups move (a group's internal
    order never re-cuts), unknown/duplicate ids drop, missing groups
    still ship (appended canonical-last), and the census is
    order-invariant — reordering changes how the report reads, never
    what ships.  The static pins keep the round's surfaces (the
    chips' drag + arrows + the persisted order + the reset chip + the
    desktop picklist both menus consult) from silently disappearing
    between rounds."""

    COLLECTOR_TS = REPO_ROOT / "src" / "lib" / "session-report.ts"
    REPORT_TSX = (
        REPO_ROOT / "src" / "components" / "explorer" / "playground" / "session-report.tsx"
    )
    PLAYGROUND_PY = TOOLKIT_ROOT / "playground.py"

    needs_round = pytest.mark.skipif(
        not COLLECTOR_TS.is_file() or BUN is None,
        reason="web checkout (<repo>/src + <repo>/scratch) + bun runtime "
        "required; the zip-only tree skips — its probes live outside the "
        "shipped tree",
    )

    @needs_round
    def test_collector_walking_order_pins(self) -> None:
        # the TS probe exits non-zero on drift and prints its censuses
        _run([BUN, "run", "scratch/probe_session_report_132.ts"], REPO_ROOT)
        ts = json.loads(Path("/tmp/sr_ts_132.json").read_text())
        canonical = [
            "ml", "learningCurve", "pkpd", "nca", "history", "surgery",
        ]
        assert ts["canonical"] == canonical
        # unknown ids dropped; missing groups appended canonical-last
        assert ts["sanitizedUnknown"] == [
            "surgery", "ml", "learningCurve", "pkpd", "nca", "history",
        ]
        # duplicates collapse (first occurrence wins)
        assert ts["sanitizedDupes"] == [
            "ml", "pkpd", "learningCurve", "nca", "history", "surgery",
        ]
        # a partial order completes (surgery stays first, the missing
        # groups appended canonical-last); empty/absent falls back
        assert ts["sanitizedPartial"] == ["surgery"] + canonical[:5]
        assert ts["sanitizedEmpty"] == canonical
        assert ts["sanitizedUndefined"] == canonical
        # the ordered walk moves WHOLE groups: surgery's three sections
        # first, then ML, then the canonical remainder — the group's
        # internal order never re-cuts
        assert ts["orderedTitles"][:3] == [
            "Ring topology moves",
            "Ring topology mutants",
            "Ring topology mutants — 3 structure panel(s)",
        ]
        assert ts["orderedTitles"][3:] == ts["defaultTitles"][:8]
        assert sorted(ts["orderedTitles"]) == sorted(ts["defaultTitles"])
        # the census is order-invariant — the 131 baseline byte-identical
        assert ts["censusDefault"] == ts["censusReordered"] == {
            "figures": 4, "tables": 3, "metrics": 3, "images": 1,
        }
        # order never resurrects a disabled group
        assert ts["censusSurgeryFirstOnly"] == {
            "figures": 2, "tables": 1, "metrics": 1, "images": 1,
        }
        # the availability map is unchanged (the 131 contract)
        assert ts["available"]["surgery"] is True

    @pytest.mark.skipif(
        not COLLECTOR_TS.is_file(),
        reason="web checkout (<repo>/src/lib/session-report.ts) required",
    )
    def test_collector_exports_the_order_contract(self) -> None:
        collector = self.COLLECTOR_TS.read_text(encoding="utf-8")
        assert "export function sanitizeSessionOrder" in collector
        assert "export const SESSION_GROUP_ORDER" in collector
        assert "order?: SessionGroupId[]" in collector
        # the sanitize contract is documented, not incidental
        assert "unknown ids dropped" in collector

    @pytest.mark.skipif(
        not REPORT_TSX.is_file(),
        reason="web checkout (<repo>/src components) required",
    )
    def test_report_card_carries_the_reorder_affordances(self) -> None:
        report = self.REPORT_TSX.read_text(encoding="utf-8")
        # the persisted order (its own key, tolerant restore)
        assert (
            'const SECTION_ORDER_STORAGE_KEY = "pharmaml.reportSectionOrder";'
            in report
        )
        assert "loadSectionOrder" in report
        assert "saveSectionOrder" in report
        # the drag path + the arrow path + the reset chip
        assert "draggable" in report
        assert "moveGroup" in report
        assert "dropOn" in report
        assert "Reset walking order" in report
        assert "GripVertical" in report
        assert "RotateCcw" in report
        # the affordances are labelled (a11y)
        assert "aria-roledescription" in report
        assert "earlier in the report" in report
        # the census line confesses the custom order
        assert "in your chosen walking order" in report

    @pytest.mark.skipif(
        not PLAYGROUND_PY.is_file(),
        reason="toolkit checkout (<toolkit>/playground.py) required",
    )
    def test_desktop_picklist_wiring_pins(self) -> None:
        source = self.PLAYGROUND_PY.read_text(encoding="utf-8")
        assert "def _session_report_picklist" in source
        # BOTH report menus consult it (PDF + HTML)
        assert source.count("_session_report_picklist(live, tables)") == 2
        # the honest zero (the web twin's census line, verbatim kin)
        assert "Nothing would ship — tick at least one page." in source
        # the dialog's own title + the guarded confirm
        assert '"Session report contents"' in source
        assert "zero pages never saves" in source


class TestCardFinderParity:
    """Round 100x-129: the web card finder's catalog cannot drift.

    The finder jumps by matching each catalog entry's ``match`` string
    against the live DOM, so a renamed card heading would silently
    break its jump.  The paired TS probe imports the SHIPPED
    ``card-directory.ts`` and checks every entry against the component
    source that renders it (JSX entities normalized); this guard fails
    the suite when the catalog and the components disagree.  The live
    jump itself (every entry through the real finder UI) is walked by
    ``scratch/probe_card_finder_129.sh`` — the E2E layer on top of
    this static layer."""

    CARD_DIRECTORY_TS = REPO_ROOT / "src" / "components" / "explorer" / "playground" / "card-directory.ts"
    CARD_FINDER_TS = REPO_ROOT / "src" / "components" / "explorer" / "playground" / "card-finder.tsx"
    PLAYGROUND_TSX = REPO_ROOT / "src" / "components" / "explorer" / "playground.tsx"
    PAGE_TSX = REPO_ROOT / "src" / "app" / "page.tsx"

    needs_card_finder = pytest.mark.skipif(
        not CARD_DIRECTORY_TS.is_file() or BUN is None,
        reason="web checkout (<repo>/src card finder) + bun runtime required; "
        "the zip-only tree skips — its probes live outside the shipped tree",
    )

    @needs_card_finder
    def test_catalog_matches_the_component_sources(self) -> None:
        # the TS probe exits non-zero AND prints its problem list on drift
        result = subprocess.run(
            [BUN, "run", "scratch/probe_card_directory_129.ts"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "CARD_DIRECTORY_OK" in result.stdout
        # the census: 26 cards (24 + the two Surgery entries of
        # 100x-130), 26 unique ids, zero problems
        start = result.stdout.index("{")
        end = result.stdout.index("}", start) + 1
        report = json.loads(result.stdout[start:end])
        assert report["entries"] == 26, report
        assert report["ids"] == 26, report
        assert report["problems"] == [], report

    @needs_card_finder
    def test_finder_wiring_pins(self) -> None:
        # the dialog is mounted at the PAGE level (hotkey from any tab)
        page = self.PAGE_TSX.read_text(encoding="utf-8")
        assert 'from "@/components/explorer/playground/card-finder"' in page
        assert "<CardFinder />" in page
        # the trigger rides beside the Playground sub-tab strip
        playground = self.PLAYGROUND_TSX.read_text(encoding="utf-8")
        assert "<CardFinderTrigger />" in playground
        # the hotkey is Ctrl/Cmd+Shift+F — never plain Ctrl+F (the
        # browser owns that one), pinned by the finder's own footer
        finder = self.CARD_FINDER_TS.read_text(encoding="utf-8")
        assert 'e.key.toLowerCase() === "f"' in finder
        assert "e.ctrlKey || e.metaKey" in finder
        assert "the browser owns Ctrl+F" in finder
        # the jump resolves against the shadcn card contract
        assert '[data-slot="card"]' in finder
        assert '[data-slot=card-title]' in finder
        # the flash class exists in the stylesheet (globals.css)
        styles = (REPO_ROOT / "src" / "app" / "globals.css").read_text(encoding="utf-8")
        assert ".pg-card-flash" in styles
        assert "@keyframes pg-card-flash" in styles


class TestSurgeryBridge:
    """Round 100x-130: the molecule-surgery bridge contract.

    The web twin's Surgery sub-tab mirrors the desktop playground's
    two generative surgery cards ("Ring topology moves" + "Breed two
    molecules") over the repo-root bridge pattern: one POST runs the
    REAL pharmaml.generative operators with real RDKit and returns the
    SMILES list PLUS the captioned 2D structure grid as a base64 PNG
    data URI — the same panels the desktop's "Draw mutants" / "Draw
    litter" buttons compose.  These guards pipe fixed requests through
    the SHIPPED bridge script and pin the contract: the piperidine
    mutant set, seeded determinism, honest null draws, the PNG grid,
    and the client-error exits."""

    BRIDGE_PY = REPO_ROOT / "scripts" / "surgery_bridge.py"

    needs_bridge = pytest.mark.skipif(
        not BRIDGE_PY.is_file(),
        reason="web checkout (<repo>/scripts/surgery_bridge.py) required; "
        "the zip-only tree skips — its probes live outside the shipped tree",
    )

    def _bridge(self, payload: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.BRIDGE_PY)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

    @staticmethod
    def _png_magic(data_uri: str) -> bytes:
        import base64

        assert data_uri.startswith("data:image/png;base64,"), data_uri[:40]
        return base64.b64decode(data_uri.split(",", 1)[1])[:8]

    @needs_bridge
    def test_topology_round_trip_pins_the_piperidine_set(self) -> None:
        result = self._bridge({"mode": "topology", "smiles": "C1CCNCC1"})
        assert result.returncode == 0, result.stdout + result.stderr
        response = json.loads(result.stdout)
        # the library's own enumeration for piperidine: two contractions
        # and one expansion, sorted — pinned so the twin cannot drift
        assert response["mutants"] == ["C1CCCC1", "C1CCCNCC1", "C1CCNC1"]
        assert response["source"] == "C1CCNCC1"
        assert "distinct one-step ring-size mutant" in response["summary"]
        assert response["gridPanels"] == 3
        assert self._png_magic(response["gridPng"]) == b"\x89PNG\r\n\x1a\n"
        assert response["rdkitVersion"]

    @needs_bridge
    def test_topology_acyclic_is_an_honest_empty(self) -> None:
        result = self._bridge({"mode": "topology", "smiles": "CCO"})
        assert result.returncode == 0, result.stdout + result.stderr
        response = json.loads(result.stdout)
        assert response["mutants"] == []
        assert "no qualifying saturated ring" in response["summary"]
        # the grid still renders — the classic "nothing to draw" panel
        assert self._png_magic(response["gridPng"]) == b"\x89PNG\r\n\x1a\n"

    @needs_bridge
    def test_breed_round_trip_is_seeded_and_honest_about_nulls(self) -> None:
        result = self._bridge(
            {
                "mode": "breed",
                "parentA": "CC(=O)Oc1ccccc1C(=O)O",
                "parentB": "CC(C)Cc1ccc(O)cc1",
                "children": 4,
                "seed": 42,
            }
        )
        assert result.returncode == 0, result.stdout + result.stderr
        response = json.loads(result.stdout)
        assert response["mode"] == "breed"
        assert len(response["litter"]) == 4, "one row per draw, failed or not"
        assert response["viable"] == sum(1 for child in response["litter"] if child)
        assert "viable children" in response["summary"]
        assert "seed 42" in response["summary"]
        # parents + viable children panels
        assert response["gridPanels"] == 2 + response["viable"]
        assert self._png_magic(response["gridPng"]) == b"\x89PNG\r\n\x1a\n"

    @needs_bridge
    def test_breed_is_deterministic_per_seed(self) -> None:
        payload = {
            "mode": "breed",
            "parentA": "CC(=O)Oc1ccccc1C(=O)O",
            "parentB": "CC(C)Cc1ccc(O)cc1",
            "children": 6,
            "seed": 7,
        }
        first = self._bridge(payload)
        second = self._bridge(payload)
        assert first.returncode == 0 and second.returncode == 0
        litter_a = json.loads(first.stdout)["litter"]
        litter_b = json.loads(second.stdout)["litter"]
        assert litter_a == litter_b, "the same seed must breed the same litter"

    @needs_bridge
    def test_bridge_rejects_bad_requests_as_client_errors(self) -> None:
        for payload, fragment in (
            ({"mode": "neon"}, "mode"),
            ({"mode": "topology", "smiles": "not-a-smiles"}, "did not parse"),
            ({"mode": "topology", "smiles": ""}, "non-empty"),
            (
                {
                    "mode": "breed",
                    "parentA": "CCO",
                    "parentB": "CCO",
                    "children": 9,
                    "seed": 0,
                },
                "children",
            ),
            (
                {
                    "mode": "breed",
                    "parentA": "CCO",
                    "parentB": "CCO",
                    "children": 2,
                    "seed": -1,
                },
                "seed",
            ),
        ):
            result = self._bridge(payload)
            assert result.returncode == 2, (payload, result.stdout)
            assert fragment in json.loads(result.stdout)["error"], payload


class TestSurgeryWebWiring:
    """Round 100x-130: the Surgery sub-tab's static wiring pins.

    The live E2E (agent-browser) clicks the real buttons; these pins
    keep the wiring from silently disappearing between rounds: the
    sub-tab strip carries the generative trigger, the two cards ride
    on the /api/gen/surgery route, and the finder's catalog gained
    the two entries (their drift probe lives in
    scratch/probe_card_directory_129.ts, pinned by
    TestCardFinderParity below)."""

    PLAYGROUND_TSX = REPO_ROOT / "src" / "components" / "explorer" / "playground.tsx"
    SURGERY_TSX = (
        REPO_ROOT / "src" / "components" / "explorer" / "playground" / "surgery-cards.tsx"
    )
    ROUTE_TS = REPO_ROOT / "src" / "app" / "api" / "gen" / "surgery" / "route.ts"
    CARD_DIRECTORY_TS = (
        REPO_ROOT / "src" / "components" / "explorer" / "playground" / "card-directory.ts"
    )
    CARD_FINDER_TSX = (
        REPO_ROOT / "src" / "components" / "explorer" / "playground" / "card-finder.tsx"
    )

    needs_wiring = pytest.mark.skipif(
        not PLAYGROUND_TSX.is_file(),
        reason="web checkout (<repo>/src components) required; the zip-only "
        "tree skips — its probes live outside the shipped tree",
    )

    @needs_wiring
    def test_sub_tab_and_cards_are_wired(self) -> None:
        playground = self.PLAYGROUND_TSX.read_text(encoding="utf-8")
        assert 'value="generative"' in playground
        assert "<SurgeryCards />" in playground
        assert "surgery-cards" in playground

    @needs_wiring
    def test_cards_post_to_the_surgery_route(self) -> None:
        surgery = self.SURGERY_TSX.read_text(encoding="utf-8")
        assert '"/api/gen/surgery"' in surgery
        # the twin's honest null row and download-figure export
        assert "(no viable crossover on this draw)" in surgery
        assert "Download PNG" in surgery
        # the desktop defaults ride over
        assert "C1CCNCC1" in surgery  # piperidine
        assert "CC(=O)Oc1ccccc1C(=O)O" in surgery  # aspirin
        assert "CC(C)Cc1ccc(O)cc1" in surgery  # ibuprofen core

    @needs_wiring
    def test_route_validates_through_the_bridge(self) -> None:
        route = self.ROUTE_TS.read_text(encoding="utf-8")
        assert "surgery_bridge.py" in route
        assert "discriminatedUnion" in route

    @needs_wiring
    def test_catalog_carries_the_two_surgery_entries(self) -> None:
        catalog = self.CARD_DIRECTORY_TS.read_text(encoding="utf-8")
        assert '"generative"' in catalog
        assert 'id: "gen-topology"' in catalog
        assert 'id: "gen-breed"' in catalog
        assert 'match: "Ring topology moves"' in catalog
        assert 'match: "Breed two molecules"' in catalog
        finder = self.CARD_FINDER_TSX.read_text(encoding="utf-8")
        assert 'generative: "Surgery"' in finder

    @needs_wiring
    def test_web_finder_recency_wiring_pins(self) -> None:
        finder = self.CARD_FINDER_TSX.read_text(encoding="utf-8")
        # the recency row's storage contract + the honest-landing rule
        assert 'const RECENT_STORAGE_KEY = "pharmaml.cardFinder.recent";' in finder
        assert "const RECENT_CAP = 6;" in finder
        assert "rememberJump(entry.id)" in finder
        assert "if (found) {" in finder
        assert "Recently jumped" in finder
