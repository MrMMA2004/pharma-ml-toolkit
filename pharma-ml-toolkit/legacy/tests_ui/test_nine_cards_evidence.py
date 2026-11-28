"""part5 §11 — the rendered evidence wiring: same labels, real text.

The pure collectors are pinned in ``tests/test_studio_evidence.py``; this
suite pins the Qt half: every card's preview label is addressable and
STABLE across refreshes (the §8 widget-reuse contract extended to the
evidence text), a live page context flows into the rendered grid, and the
existing part-4 contracts (order, navigation, 18 pages) survive the §11
rework untouched.
"""

from __future__ import annotations

import pandas as pd
from ui.nine_cards import NINE_CARDS, build_nine_card_grid, live_preview


def _preview_label(frame):
    from PySide6.QtWidgets import QLabel

    labels = [
        widget for widget in frame.findChildren(QLabel) if widget.objectName() == "nineCardLivePreview"
    ]
    assert len(labels) == 1, "each card carries exactly one live-preview label"
    return labels[0]


class TestPreviewLabels:
    def test_every_card_has_one_addressable_preview_label(self, studio_app) -> None:
        _host, frames = build_nine_card_grid()
        for card in NINE_CARDS:
            _preview_label(frames[card.card_id])

    def test_refresh_reuses_the_same_label_objects(self, studio_app) -> None:
        # the r262 id-reuse lesson: hold the REFERENCES, compare identity
        host, _frames = build_nine_card_grid()
        before = host.preview_labels()
        host.refresh_evidence()
        after = host.preview_labels()
        for card in NINE_CARDS:
            assert after[card.card_id] is before[card.card_id]

    def test_bare_grid_shows_the_precise_empty_states(self, studio_app, monkeypatch) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", tmp)
            _host, frames = build_nine_card_grid()
            for card in NINE_CARDS:
                text = _preview_label(frames[card.card_id]).text()
                if card.card_id == "introduction":
                    assert "pharmaml" in text
                    continue
                assert text.startswith("No ") or "model families" in text, (
                    f"{card.card_id}: {text[:60]!r} is neither real evidence nor an empty state"
                )

    def test_context_feeds_the_datasets_card(self, studio_app, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path / "runs"))
        frame = pd.DataFrame({"smiles": ["CCO", "CCC"], "pic50": [5.1, 6.2]})
        _host, frames = build_nine_card_grid(
            context_provider=lambda: {
                "data_frame": frame,
                "data_source": "qsar.csv",
                "data_column": "pic50",
            }
        )
        text = _preview_label(frames["datasets"]).text()
        assert "qsar.csv" in text and "2 rows x 2 columns" in text
        assert "min 5.1" in text

    def test_live_preview_accepts_a_context_mapping(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path / "runs"))
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        text = live_preview("datasets", {"data_frame": frame, "data_source": "x.csv"})
        assert "3 rows x 1 columns" in text

    def test_a_raising_provider_degrades_to_the_empty_states(self, studio_app, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path / "runs"))

        def broken():
            raise RuntimeError("page exploded")

        _host, frames = build_nine_card_grid(context_provider=broken)
        text = _preview_label(frames["datasets"]).text()
        assert text.startswith("No dataset loaded")

    def test_set_card_context_refreshes_in_place(self, studio_app, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path / "runs"))
        host, frames = build_nine_card_grid()
        label = _preview_label(frames["datasets"])
        assert label.text().startswith("No dataset loaded")
        frame = pd.DataFrame({"a": [1, 2]})
        host.set_card_context(lambda: {"data_frame": frame, "data_source": "y.csv"})
        assert "y.csv" in label.text() and "2 rows x 1 columns" in label.text()


class TestDashboardIntegration:
    def test_main_window_wires_the_live_context(self, main_window) -> None:
        grid = main_window.dashboard_page.findChildren(type(main_window.dashboard_page._cards_host))
        assert len(grid) == 1
        # the context provider is wired: the Datasets card reflects the
        # (empty) Data Explorer state, i.e. the precise empty state
        previews = grid[0].preview_labels()
        assert previews["datasets"].text().startswith("No dataset loaded")

    def test_loaded_dataset_reaches_the_dashboard_card(self, main_window, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path / "runs"))

        from PySide6.QtWidgets import QApplication

        main_window.data_page.load_path(
            main_window.data_page._TK / "examples" / "data" / "screening_library.csv"
            if hasattr(main_window.data_page, "_TK")
            else _examples_library()
        )
        # load_path is async (worker); pump events until the frame lands
        for _ in range(200):
            QApplication.processEvents()
            if main_window.data_page.frame() is not None:
                break
        assert main_window.data_page.frame() is not None
        grid = main_window.dashboard_page.findChildren(type(main_window.dashboard_page._cards_host))[0]
        grid.refresh_evidence()
        text = grid.preview_labels()["datasets"].text()
        assert "rows x" in text and "missing" in text
        assert not text.startswith("No dataset loaded")

    def test_show_event_refreshes_the_evidence(self, main_window, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path / "runs"))
        grid = main_window.dashboard_page.findChildren(type(main_window.dashboard_page._cards_host))[0]
        main_window.show()
        main_window.go_to("models")
        main_window.go_to("dashboard")
        # no exception and the labels are still addressable after the round trip
        assert set(grid.preview_labels()) == {card.card_id for card in NINE_CARDS}


def _examples_library():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "examples" / "data" / "screening_library.csv"


class TestContractsSurvive:
    """The part-4 pins that §11 must not disturb."""

    def test_ml_preview_still_starts_with_the_registry_count(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path / "runs"))
        preview = live_preview("machine-learning")
        assert preview.split()[0].isdigit(), "model-registry preview must stay numeric"

    def test_page_stack_stays_eighteen(self, main_window) -> None:
        assert main_window.pages.count() == 18
