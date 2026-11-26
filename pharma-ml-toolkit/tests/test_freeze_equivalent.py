"""Focused tests for the part9 freeze-equivalent campaign (r273+r274).

``tools/freeze_smoke_driver.py`` is the §14 evidence driver that runs
INSIDE the frozen bundle (and unfrozen, pinned here so API drift fails
at gate time, not freeze time); ``tools/freeze_equivalent.py`` is the
builder that captures the Windows spec's policy verbatim. These tests
pin both: the full unfrozen flow run, the offline guard's blocking
semantics, the §7 user-dir isolation contract, and the captured
Windows-spec policy (hiddenimports/datas/excludes) that the
Linux-equivalent build inherits unchanged. The phase-3 tier (r274:
deep-learning / semi-supervised / generative / docking ligand prep /
PK-PD / MD / SHAP / uncertainty / plot metadata) is pinned the same
way — registry shape plus the full-run record.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

TOOLKIT = Path(__file__).resolve().parents[1]
if str(TOOLKIT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT))

from tools import freeze_equivalent as fe
from tools import freeze_smoke_driver as fsd


@pytest.fixture()
def restore_socket_connect():
    """The driver arms the offline proof on the CLASS; restore after."""
    original = socket.socket.connect
    yield original
    socket.socket.connect = original


class TestSmokeDriver:
    def test_flows_registry_shape(self) -> None:
        names = [name for name, _ in fsd.FLOWS]
        # the §14.2 canon in execution order (qt bootstraps first so the
        # QFontDatabase-dependent fonts flow has an application object)
        assert names[:3] == ["environment", "qt_offscreen", "fonts"]
        assert "import_csv" in names and "bundled_engines" in names
        assert "model_save_reload" in names and "applicability_domain" in names
        # the phase-3 (r274) deep-feature tier — all nine, registered
        phase3 = {
            "deep_learning", "semi_supervised", "explainability",
            "uncertainty", "generative", "docking_ligand_prep",
            "pkpd_ode", "md_analysis", "plot_metadata",
        }
        assert phase3 <= set(names)
        assert len(names) == 22
        # every flow has the uniform (record, user, bundle) contract
        for _, flow in fsd.FLOWS:
            assert callable(flow)

    def test_synth_frame_deterministic(self) -> None:
        import pandas as pd

        first = fsd._synth_frame()
        second = fsd._synth_frame()
        pd.testing.assert_frame_equal(first, second)
        assert first.shape == (96, 10)
        assert set(first.columns[-2:]) == {"y_cls", "y_reg"}
        assert first["y_cls"].nunique() == 2

    def test_offline_guard_blocks_and_records(
        self, restore_socket_connect: None
    ) -> None:
        fsd.block_network()  # arm the §14.3 proof in-process
        fsd._NET_ATTEMPTS.clear()
        with pytest.raises(OSError, match="offline proof"):
            socket.socket().connect(("127.0.0.1", 1))
        assert _addr_in(fsd._NET_ATTEMPTS, "127.0.0.1")
        fsd._NET_ATTEMPTS.clear()

    def test_full_unfrozen_run_verdict_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        restore_socket_connect: None,
    ) -> None:
        monkeypatch.setenv("FREEZE_SMOKE_UNFROZEN_OK", "1")
        user_dir = tmp_path / "user"
        rc = fsd.main(["--user-dir", str(user_dir)])
        verdict = __import__("json").loads(
            (user_dir / "freeze_smoke_verdict.json").read_text()
        )
        assert rc == 0
        assert verdict["verdict"] == "PASS"
        assert verdict["failures"] == []
        assert verdict["offline"] is True
        assert verdict["net_attempts"] == []
        flows = verdict["flows"]
        assert len(flows) == 22
        assert all(f["status"] == "PASS" for f in flows.values())
        # §7: everything the flows wrote lives under the user dir
        assert user_dir.joinpath("smoke_dataset.csv").is_file()
        assert user_dir.joinpath("parity_plot.png").is_file()
        assert user_dir.joinpath("parity_plot.tiff").is_file()
        assert user_dir.joinpath("smoke_rf.joblib").is_file()
        # §7 (phase 3): the docking campaign tree is a USER-dir artifact
        campaign = user_dir / "docking_campaign"
        folders = sorted(p.name for p in campaign.iterdir())
        assert len(folders) == 3
        assert all((campaign / f / "l.pdbqt").is_file() for f in folders)
        # §14.2 science sanity: the frozen flows' regression parity
        assert verdict["ml"]["regression_r2"] > 0.9
        assert verdict["ml"]["classification_accuracy"] > 0.8
        # the phase-3 (r274) deep-feature tier's own numbers
        assert verdict["deep_learning"]["accuracy"] > 0.7
        assert verdict["semi_supervised"]["accuracy_held_out"] > 0.7
        assert verdict["semi_supervised"]["pseudo_label_accuracy"] > 0.5
        assert verdict["generative"]["best_score"] > 0.1
        assert verdict["explainability"]["top_feature"] in ("f1", "f2")
        assert 0.0 <= verdict["uncertainty"]["coverage"] <= 1.0
        assert verdict["pkpd_ode"]["c_peak"] > 0
        assert verdict["md_analysis"]["rmsf_loop_mean"] > verdict["md_analysis"]["rmsf_helix_mean"]
        assert verdict["plot_metadata"]["hovertext_len"] == len(fsd.SMILES)
        assert verdict["docking_ligand_prep"]["ok_rows"] == 3

    def test_environment_flow_refuses_silent_unfrozen_runs(self) -> None:


        record: dict = {}
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("FREEZE_SMOKE_UNFROZEN_OK", raising=False)
            with pytest.raises(AssertionError, match="frozen flag"):
                fsd.flow_environment(record, Path("/tmp"), {})


def _addr_in(attempts: list[str], needle: str) -> bool:
    return any(needle in a for a in attempts)


class TestFreezeEquivalentBuilder:
    def test_capture_carries_the_windows_spec_policy(self) -> None:
        policy = fe.capture_spec()
        hidden = set(policy["hiddenimports"])
        # the freeze traps the spec exists to catch (part9 §3.1)
        assert "mordred.Descriptor" in hidden or any(
            h.startswith("mordred.") for h in hidden
        )
        assert "shap" in hidden and "numba" in hidden and "llvmlite" in hidden
        assert "matplotlib.backends.backend_agg" in hidden
        assert "PySide6.QtSvg" in hidden
        assert "docx" in hidden and "selfies" in hidden
        # the datas contract: fonts + engines + LICENSE/VERSION ride the
        # bundle where the app's __file__ math expects them
        assert any(dst == "fonts" for _, dst in policy["datas"])
        assert any(
            dst == "pharmaml/engines" for _, dst in policy["datas"]
        )
        assert any(dst == "." for _, dst in policy["datas"])
        # §12: dev/test tooling never enters the installer
        excludes = set(policy["excludes"])
        assert {"pytest", "mypy", "ruff", "tkinter"} <= excludes
        # the entry is the real application, not a surrogate
        # part18 §10: the legacy GUI entry is quarantined at legacy/studio.py
        assert policy["scripts"] == [str(TOOLKIT / "legacy" / "studio.py")]

    def test_generated_spec_two_entries_one_collect(self, tmp_path: Path) -> None:
        spec = fe.write_equivalent_spec(
            {
                "scripts": [str(TOOLKIT / "legacy" / "studio.py")],
                "pathex": [str(TOOLKIT), str(TOOLKIT / "src")],
                "binaries": [],
                "datas": [(str(TOOLKIT / "fonts"), "fonts")],
                "hiddenimports": ["shap"],
                "excludes": ["pytest"],
            }
        )
        text = spec.read_text()
        assert "studio.py" in text and "freeze_smoke_driver.py" in text
        assert 'name="PharmaMLStudio"' in text
        assert 'name="FreezeSmokeDriver"' in text
        assert 'name="freeze-equivalent"' in text
        assert "_entry_for(a.scripts" in text
        # the documented Linux delta: the EGL stub the app's own
        # ui.headless preloads on libEGL-less Linux hosts
        assert "'ui/native'" in text
        assert text.count("EXE(") == 2 and text.count("COLLECT(") == 1
