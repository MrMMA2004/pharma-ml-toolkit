"""Tests for the round-205 heavy-work safety kit (the collapse report).

The owner's report -- "playground.py collapses during heavy work" -- is
the OOM story: a heavy batch (or a many-worker CV fan-out) outruns the
RAM, the kernel kills the biggest process (the Tk app) and the window
vanishes with every loaded input.  Three layers answer it, each pinned
here:

* the cooperative-stop channel -- :class:`pharmaml.exceptions.JobCancelled`
  raised at the checkpoint loops (feature factory, recalculation,
  progressive table reads, both CV leaderboards) when the progress
  callback answers ``False``; legacy callbacks (``None``/truthy
  returns) never cancel;
* the playground's Stop button + live memory watchdog + pre-flight
  batch guard -- the app-level machinery that flips the flag and
  refuses impossible batches with their own arithmetic;
* the ML n_jobs memory cap -- loky CV workers are budgeted 768 MB of
  available memory each, and a bite rides the job label honestly.
"""

from __future__ import annotations

import gc
import importlib.util
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_PATH = TOOLKIT_ROOT / "playground.py"


def _load_playground() -> Any:
    """Import playground.py by path (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("playground_safety_under_test", PLAYGROUND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys

    sys.modules["playground_safety_under_test"] = module
    return module


@pytest.fixture(scope="module")
def playground() -> Any:
    return _load_playground()


def _xvfb_available() -> bool:
    return shutil.which("Xvfb") is not None


XVFB_SKIP = pytest.mark.skipif(
    importlib.util.find_spec("tkinter") is None or not _xvfb_available(),
    reason="needs tkinter + Xvfb for a headless display",
)


# --------------------------------------------------------------------- #
# the cooperative-stop channel (library layer)
# --------------------------------------------------------------------- #
class TestJobCancelledChannel:
    def test_the_error_is_a_pharmaml_error(self) -> None:
        from pharmaml.exceptions import JobCancelled, PharmaMLError

        assert issubclass(JobCancelled, PharmaMLError)
        assert str(JobCancelled("stopped")) == "stopped"

    def test_factory_legacy_callbacks_never_cancel(self) -> None:
        """``None``-returning callbacks keep the historical behaviour."""
        from pharmaml.chem.features import compute_feature_frame

        seen: list[tuple[int, int, str]] = []
        frame, _report = compute_feature_frame(
            ["CCO"] * 30, families=("ecfp",), progress=lambda d, t, n: seen.append((d, t, n))
        )
        assert frame.shape == (30, 2048)
        assert seen and seen[-1][0] == 30

    def test_factory_false_stops_at_the_boundary(self) -> None:
        from pharmaml.chem.features import compute_feature_frame
        from pharmaml.exceptions import JobCancelled

        with pytest.raises(JobCancelled) as caught:
            compute_feature_frame(
                ["CCO"] * 60,
                families=("ecfp",),
                # False = stop; the factory checkpoints at 25-molecule
                # chunks, so the batch halts exactly at 25/60.
                progress=lambda done, total, note: done < 25,
            )
        assert "25/60 molecules" in str(caught.value)

    def test_recalculation_inherits_the_channel(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features
        from pharmaml.exceptions import JobCancelled

        metadata = {
            "selected_features": ["desc_amw", "desc_tpsa"],
            "family_counts": {"descriptors": 2},
            "fingerprint_sizes": {"descriptors": 43},
            "families_computed": {"descriptors": 43},
            "k_requested": 2,
            "method": "variance",
        }
        with pytest.raises(JobCancelled) as caught:
            recalculate_selected_features(
                ["CCO"] * 60, metadata, progress=lambda done, total, note: False
            )
        assert "25/60 molecules" in str(caught.value)

    def test_table_reader_false_stops_mid_stream(self, tmp_path: Path) -> None:
        from pharmaml.exceptions import JobCancelled
        from pharmaml.utils.io import read_table_with_progress

        path = tmp_path / "big.csv"
        lines = ["a,b"] + [f"{i},{i * 2}" for i in range(300)]
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(JobCancelled) as caught:
            read_table_with_progress(path, lambda d, t, n: False, size_threshold=64)
        assert "cancelled after" in str(caught.value)
        assert "rows" in str(caught.value)

    def test_table_reader_legacy_report_completes(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import read_table_with_progress

        path = tmp_path / "legacy.csv"
        lines = ["a,b"] + [f"{i},{i * 2}" for i in range(300)]
        path.write_text("\n".join(lines), encoding="utf-8")
        seen: list[tuple[int, Any, str]] = []
        frame = read_table_with_progress(
            path, lambda d, t, n: seen.append((d, t, n)), size_threshold=64
        )
        assert len(frame) == 300
        assert seen and seen[-1][0] == 300

    def test_indexed_reader_false_stops(self, tmp_path: Path) -> None:
        from pharmaml.exceptions import JobCancelled
        from pharmaml.utils.io import read_indexed_table_with_progress

        path = tmp_path / "indexed.csv"
        lines = ["name,a"] + [f"cell{i},{i}" for i in range(300)]
        path.write_text("\n".join(lines), encoding="utf-8")
        with pytest.raises(JobCancelled) as caught:
            read_indexed_table_with_progress(
                path,
                lambda d, t, n: False,
                index_col="name",
                size_threshold=64,
            )
        assert "cancelled after" in str(caught.value)

    def test_compare_models_false_stops_after_the_row(self) -> None:
        import numpy as np
        import pandas as pd

        from pharmaml.exceptions import JobCancelled
        from pharmaml.ml import compare_models

        rng = np.random.default_rng(205)
        frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=["x1", "x2", "x3"])
        y = pd.Series(frame["x1"] * 2.0 + 1.0, name="y")
        with pytest.raises(JobCancelled) as caught:
            compare_models(
                frame,
                y,
                "regression",
                models=["ridge", "lasso"],
                cv=2,
                on_model=lambda info: False,
            )
        assert "1/2 models" in str(caught.value)

    def test_compare_models_legacy_heartbeat_completes(self) -> None:
        import numpy as np
        import pandas as pd

        from pharmaml.ml import compare_models

        rng = np.random.default_rng(205)
        frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=["x1", "x2", "x3"])
        y = pd.Series(frame["x1"] * 2.0 + 1.0, name="y")
        seen: list[dict[str, Any]] = []
        board = compare_models(
            frame,
            y,
            "regression",
            models=["ridge", "lasso"],
            cv=2,
            on_model=lambda info: seen.append(info) or None,
        )
        assert len(board) == 2 and [info["total"] for info in seen] == [2, 2]

    def test_family_benchmark_false_stops(self) -> None:
        import numpy as np
        import pandas as pd

        from pharmaml.exceptions import JobCancelled
        from pharmaml.ml import benchmark_feature_families

        smiles = ["CCO", "CCC", "CCN", "c1ccccc1", "CC(=O)O", "C1CCCCC1"] * 5
        rng = np.random.default_rng(205)
        y = pd.Series(rng.normal(size=len(smiles)))
        with pytest.raises(JobCancelled) as caught:
            benchmark_feature_families(
                smiles,
                y,
                "regression",
                families=("ecfp", "maccs"),
                cv=2,
                on_family=lambda info: False,
            )
        assert "1/2 families" in str(caught.value)


# --------------------------------------------------------------------- #
# the memory maths (module layer, no display needed)
# --------------------------------------------------------------------- #
class TestMemoryMaths:
    def test_matrix_bytes_arithmetic(self, playground: Any) -> None:
        assert playground._feature_matrix_bytes(10, 100) == 10 * 100 * 8 * 3
        assert playground._feature_matrix_bytes(0, 100) == 0

    def test_factory_columns_use_family_widths(self, playground: Any) -> None:
        columns = playground._factory_batch_columns(["descriptors", "ecfp", "maccs"])
        assert columns == 210 + 2048 + 167
        assert playground._factory_batch_columns(["mystery"]) == 2048  # the safe default

    def test_guard_passes_when_it_fits(self, playground: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(playground, "_meminfo_mb", lambda field: 4096.0)
        assert playground._guard_batch("Batch too large", 237, 27_926) is None

    def test_guard_refuses_with_its_own_arithmetic(
        self, playground: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(playground, "_meminfo_mb", lambda field: 200.0)
        message = playground._guard_batch("Batch too large", 237, 27_926)
        assert message is not None
        assert "Batch too large" in message
        assert "237 molecules" in message and "27,926 columns" in message
        assert "0.2 GB of memory is available" in message

    def test_guard_stands_down_off_linux(self, playground: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(playground, "_meminfo_mb", lambda field: None)
        assert playground._guard_batch("Batch too large", 10**9, 10**6) is None

    def test_is_job_cancelled_matches_the_name(self, playground: Any) -> None:
        from pharmaml.exceptions import JobCancelled

        assert playground._is_job_cancelled(JobCancelled("x"))
        assert not playground._is_job_cancelled(ValueError("x"))

        class JobCancelled:
            pass

        assert playground._is_job_cancelled(JobCancelled())

    def test_meminfo_reads_the_real_proc(self, playground: Any) -> None:
        value = playground._meminfo_mb("MemAvailable")
        if value is None:  # non-Linux CI: the guards stand down there
            pytest.skip("no /proc/meminfo on this platform")
        assert value > 0.0
        assert playground._rss_mb() > 0.0

    def test_ml_njobs_cap_math(self, playground: Any) -> None:
        """Pure arithmetic over the module constant (no monkeypatching)."""
        available = 1600.0
        capped = max(1, min(8, int(available / playground._ML_WORKER_MEM_MB)))
        assert capped == 2
        assert playground._ML_WORKER_MEM_MB == 768.0


# --------------------------------------------------------------------- #
# the GUI machinery (Xvfb)
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestGuiSafetyMachinery:
    # (round 209) the display fixture moved to tests/conftest.py as ONE
    # session-scoped Xvfb: per-class fresh servers + a stale display
    # connection from a destroyed app = the next Tk event pump hits the
    # broken connection and Tk 9's X error handler exits the whole
    # process (the round-209 collapse forensics).

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        """Neutralise modal dialogs so failures assert instead of blocking."""
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the Stop button ------------------------------------------------ #

    def test_stop_button_exists_and_idles_disabled(self, app: Any) -> None:
        assert bool(app._stop_button.instate(["disabled"]))

    def test_idle_cancel_request_is_a_noop(self, app: Any) -> None:
        app._request_cancel()
        assert not app._cancel_requested

    def test_reporter_answers_false_once_the_flag_is_set(self, app: Any) -> None:
        assert app._report_progress(1, 10) is True
        app._cancel_requested = True
        assert app._report_progress(2, 10) is False
        app._cancel_requested = False

    def test_a_cancellable_job_lands_cancelled(self, app: Any, quiet_dialogs: list) -> None:
        """The full round trip: work → Stop flag → JobCancelled → poller."""
        from pharmaml.exceptions import JobCancelled

        def work(notify: Any) -> str:
            # 60 x 0.03 s ~= 1.8 s: the first Tk update() calls are slow
            # (font/Tk warm-up can eat several hundred ms each), so a
            # short job finishes INSIDE the warm-up and the Stop click
            # races a finished job.  This length keeps the stop landing
            # mid-job deterministic under the slowest warm-up.
            for index in range(60):
                time.sleep(0.03)
                if notify(index, 60, "tick") is False:
                    raise JobCancelled("stopped mid-tick")
            return "all done"

        app._start_job("tick loop", work, None)
        for _ in range(3):  # let the worker thread start before asserting
            app.update()
            time.sleep(0.02)
        assert app._busy and app._job_cancellable
        assert not bool(app._stop_button.instate(["disabled"]))
        app._request_cancel()
        assert self._drain(app)
        label = str(app._job_label.cget("text"))
        assert "cancelled" in label, label
        assert not app._busy
        assert bool(app._stop_button.instate(["disabled"]))
        assert not app._cancel_requested  # reset for the next job
        assert any(name == "showinfo" for name, _args in quiet_dialogs), quiet_dialogs

    def test_a_checkpointless_job_is_honestly_labelled(self, app: Any) -> None:
        def work() -> str:
            time.sleep(0.6)  # outlive the slow first update() cycles
            return "uncancellable"

        app._start_job("silent job", work, None)
        assert app._busy and not app._job_cancellable
        app._request_cancel()
        assert app._cancel_requested
        label = str(app._job_label.cget("text"))
        assert "no checkpoints" in label, label
        assert self._drain(app)
        assert not app._cancel_requested  # the reset retires the stale stop

    def test_watchdog_trips_on_the_ceiling(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list
    ) -> None:
        def work(notify: Any) -> str:
            for index in range(200):
                time.sleep(0.02)
                if notify(index, 200, "grind") is False:
                    from pharmaml.exceptions import JobCancelled

                    raise JobCancelled("ground to a halt")
            return "done"

        app._start_job("grinder", work, None)
        ceiling = app._job_mem_floor_mb + app._job_mem_budget_mb + 64.0
        real_rss, real_meminfo = playground._rss_mb, playground._meminfo_mb
        monkeypatch.setattr(playground, "_rss_mb", lambda: ceiling)
        monkeypatch.setattr(
            playground, "_meminfo_mb", lambda field: 4096.0 if field == "MemAvailable" else real_meminfo(field)
        )
        tripped = False
        for _ in range(20):  # ~3 watchdog polls
            app.update()
            time.sleep(0.16)
            if app._cancel_requested:
                tripped = True
                break
        # NB: restore by re-setting, NEVER monkeypatch.undo() here -- the
        # quiet_dialogs fixture rides the SAME monkeypatch instance, and
        # an undo() would re-arm the real modal dialogs mid-test (the
        # JobCancelled info box would then block Xvfb forever).
        monkeypatch.setattr(playground, "_rss_mb", real_rss)
        monkeypatch.setattr(playground, "_meminfo_mb", real_meminfo)
        assert tripped, "the watchdog never tripped over the ceiling"
        assert app._cancel_reason == "memory ceiling"
        assert self._drain(app)
        assert "cancelled" in str(app._job_label.cget("text"))

    def test_factory_card_guard_refuses_oversized_batches(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list
    ) -> None:
        real_meminfo = playground._meminfo_mb
        monkeypatch.setattr(playground, "_meminfo_mb", lambda field: 200.0)
        app._feat_pasted_text.delete("1.0", "end")
        # 250 molecules x ~28k columns x 8 B x 3 copies ~= 168 MB, well
        # over the 0.5 x 200 MB budget the fake MemAvailable grants
        # (50 molecules would estimate only ~34 MB and pass the guard).
        app._feat_pasted_text.insert("1.0", "\n".join(["CCO"] * 250))
        for family, _label in playground._FEATURE_FAMILY_LABELS:
            app._feat_family_vars[family].set(True)
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_njobs_spin.set("1")
        app._feat_compute()
        # restore by re-setting (never undo(): the dialogs must stay quiet)
        monkeypatch.setattr(playground, "_meminfo_mb", real_meminfo)
        for _ in range(6):
            app.update()
            time.sleep(0.05)
        assert not app._busy, "a job started despite the guard"
        assert any(
            name == "showwarning" and "Batch too large" in " ".join(map(str, args))
            for name, args in quiet_dialogs
        ), quiet_dialogs

    def test_ml_njobs_cap_is_honest(
        self, app: Any, playground: Any, monkeypatch: Any
    ) -> None:
        real_meminfo = playground._meminfo_mb
        monkeypatch.setattr(playground, "_meminfo_mb", lambda field: 1600.0)
        capped, note = app._effective_ml_n_jobs(8)
        untouched, empty = app._effective_ml_n_jobs(2)
        monkeypatch.setattr(playground, "_meminfo_mb", real_meminfo)
        assert (capped, note) == (2, " — n_jobs 8→2 (memory cap)")
        assert (untouched, empty) == (2, "")

    # -- the round-209 registry leak (the second collapse mode) ---------- #

    def test_destroy_releases_the_module_card_registries(
        self, playground: Any, xvfb_display: str
    ) -> None:
        """destroy() drops the module-level card registries.

        _CARD_INDEX/_COLLAPSE_CARDS held every shell widget past the
        app's death, keeping the Tcl interpreter (and its X connection)
        alive; when the fixture's Xvfb then went away, the next Tk event
        pump hit the broken connection and Tk 9's X error handler
        terminated the whole process (exit(1), zero output).  Own
        destroy (not the app fixture) so the teardown's destroy stays
        the only re-destroy.
        """
        application = playground.PlaygroundApp(persist_session=False)
        try:
            assert playground._CARD_INDEX, "a built app indexes no cards"
            assert playground._COLLAPSE_CARDS, "a built app registers no collapse state"
        finally:
            application.destroy()
            del application
            gc.collect()
        assert playground._CARD_INDEX == []
        assert playground._COLLAPSE_CARDS == {}

    def test_a_rebuilt_app_replaces_the_registry(
        self, playground: Any, xvfb_display: str
    ) -> None:
        """Two sequential apps in one process: the second owns the index.

        The same build-destroy-rebuild shape that terminated the pytest
        process pre-fix -- now the rebuild starts from a clean directory
        and every indexed shell belongs to the live app.
        """
        first = playground.PlaygroundApp(persist_session=False)
        try:
            assert len(playground._CARD_INDEX) >= 130
        finally:
            first.destroy()
            del first
            gc.collect()
        second = playground.PlaygroundApp(persist_session=False)
        try:
            live = second._live_cards()
            assert len(live) >= 130
            for _title, _subtitle, shell, _index in live:
                assert shell.winfo_toplevel() is second, (
                    "the rebuilt app indexed a stale shell of its predecessor"
                )
        finally:
            second.destroy()
            del second
            gc.collect()


# --------------------------------------------------------------------- #
# (round 215) the file-load jobs stream too
# --------------------------------------------------------------------- #
class TestTheFileLoadJobsStream:
    """The three user-file loads ride ``read_table_with_progress`` now.

    Round 205's kit guarded the factory/recalc batches, the CV fan-outs
    and the progressive reads the Docking tab used -- but three file
    loaders still called the plain ``read_table`` inside a ZERO-argument
    ``work``: the job runner classified them checkpointless, so the Stop
    button stayed disabled and the memory watchdog had nothing to flip
    while a multi-hundred-megabyte library CSV streamed in.  Exactly the
    "huge CSV load" story the round-205 header names.  Each test here
    starts the real job against a small temp CSV and pins BOTH halves of
    the repair: the job is cancellable while it runs (the reporter is
    wired), and the render still lands the loaded frame.
    """

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _assay_csv(tmp_path: Path, name: str = "assay.csv") -> Path:
        path = tmp_path / name
        rows = ["SMILES,IC50 (nM)"]
        rows += [f"{'C' * (index + 1)}O,{100.0 + index}" for index in range(12)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_dock_rescore_load_is_cancellable(self, app: Any, tmp_path: Path) -> None:
        path = self._assay_csv(tmp_path, "summary.csv")
        app._dock_load_rescore_csv(path)
        assert app._busy
        # THE pin: the reporter is wired, so Stop / the watchdog apply
        assert app._job_cancellable
        assert self._drain(app)
        assert app._dock_results_source == "csv"
        assert len(app._dock_results_tree.get_children()) == 12
        assert not app._busy

    def test_library_scan_load_is_cancellable(
        self, playground: Any, app: Any, tmp_path: Path, monkeypatch: Any, quiet_dialogs: list
    ) -> None:
        path = self._assay_csv(tmp_path, "library.csv")
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **k: str(path)
        )
        app._scan_pick_library()
        assert app._busy and app._job_cancellable
        assert self._drain(app)
        assert "12 SMILES loaded" in str(app._scan_lib_info.cget("text"))
        assert not app._busy

    def test_pic50_append_load_is_cancellable(self, app: Any, tmp_path: Path) -> None:
        path = self._assay_csv(tmp_path, "pic50_in.csv")
        app._conv_mode.set("append pIC50 (from IC50 nM)")
        app._conv_input = path
        app._conv_run()
        assert app._busy and app._job_cancellable
        assert self._drain(app)
        frame = app._conv_frame
        assert frame is not None and len(frame) == 12
        assert "pIC50" in frame.columns
        assert not app._busy
