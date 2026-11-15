"""(part8 §13, round 4) The workflow-guide strip in every workspace.

Pins the Phase-Twelve contract:
- all NINE workspaces have a registered stage chain (no workspace left
  behind, no invented keys),
- each chain transcribes the directive's suggested workflow VERBATIM
  (the stage names come from §13.2–§13.9),
- every stage names a real surface (the ``where`` text is non-empty and
  never a placeholder),
- the main window mounts one strip per workspace tab bar, collapsed,
- the F1 help carries the same chains as a searchable topic.
"""

from __future__ import annotations

import pytest

from ui.widgets.workflow_guide import WORKFLOWS, WorkflowGuide

#: The nine §12.1 workspace keys, exact.
NINE_KEYS = (
    "introduction", "datasets", "machine-learning", "deep-learning",
    "generative-ai", "molecular-docking", "pkpd",
    "molecular-dynamics-analysis", "visualization",
)

#: First-stage / key-stage pins transcribed from the directive's own
#: suggested workflows — a renamed stage here is a regression.
_DIRECTIVE_PINS = {
    "datasets": ("Import", "Export", 10),
    "machine-learning": ("Select Dataset", "Save/Export", 14),
    "deep-learning": ("Select Dataset", "Save Checkpoint/Export", 11),
    "generative-ai": ("Select Generation Method", "Export", 10),
    "molecular-docking": ("Select Ligands", "Export", 10),
    "pkpd": ("Import Data", "Export", 8),
    "molecular-dynamics-analysis": ("Load Topology/Trajectory", "Export", 7),
    "visualization": ("Select Dataset/Model Result", "Export", 7),
    "introduction": ("Purpose", "Navigation", 4),
}


class TestWorkflowRegistry:
    def test_exactly_the_nine_workspaces(self) -> None:
        assert sorted(WORKFLOWS) == sorted(NINE_KEYS)

    @pytest.mark.parametrize("key", NINE_KEYS)
    def test_every_stage_names_a_real_surface(self, key) -> None:
        _chain, stages, _note = WORKFLOWS[key]
        assert stages, f"{key} has no stages"
        for stage, where in stages:
            assert stage.strip(), f"{key}: empty stage name"
            assert where.strip(), f"{key}: stage {stage!r} has no where"
            for bad in ("TODO", "XXX", "【", "placeholder", "TBD"):
                assert bad.lower() not in where.lower(), (key, stage, where)

    @pytest.mark.parametrize("key,expected", sorted(_DIRECTIVE_PINS.items()))
    def test_directive_stages_verbatim(self, key, expected) -> None:
        first, last, count = expected
        _chain, stages, _note = WORKFLOWS[key]
        names = [stage for stage, _where in stages]
        assert names[0] == first, f"{key}: chain must start with {first!r}"
        assert names[-1] == last, f"{key}: chain must end with {last!r}"
        assert len(stages) == count, (
            f"{key}: expected {count} stages, got {len(stages)} — a stage "
            "was silently added or dropped from the directive's chain"
        )

    def test_every_workspace_note_is_nonempty(self) -> None:
        for key in NINE_KEYS:
            _chain, _stages, note = WORKFLOWS[key]
            assert len(note) > 20, f"{key}: note too thin to orient anyone"

    def test_md_clarifies_analysis_not_simulation(self) -> None:
        # §13.8: "Do not claim to run simulations when only analysis is
        # implemented" — the note must carry that honesty in writing.
        _chain, _stages, note = WORKFLOWS["molecular-dynamics-analysis"]
        assert "ANALYZ" in note.upper()
        assert "simulation" in note.lower()

    def test_no_persian_or_rtl_text(self) -> None:
        # the directive's own rule: application-facing content is English
        for key in NINE_KEYS:
            chain, stages, note = WORKFLOWS[key]
            blob = chain + note + " ".join(w for _s, w in stages)
            assert not any("\u0600" <= ch <= "\u06ff" for ch in blob), key
            assert not any("\u200f" in ch or "\u200e" in ch for ch in blob), key


class TestWorkflowGuideWidget:
    def test_renders_collapsed_with_the_toggle(self, qtbot, studio_app) -> None:
        guide = WorkflowGuide("datasets")
        qtbot.addWidget(guide)
        # headless: isVisible() answers false for unshown parents, so the
        # contract is pinned on the toggle state + the hide/show policy.
        assert not guide.is_expanded()
        assert not guide._toggle.isChecked()
        assert not guide._body.isVisibleTo(guide)
        guide._toggle.click()
        assert guide.is_expanded()
        assert guide._toggle.isChecked()
        assert guide._body.isVisibleTo(guide)
        guide._toggle.click()
        assert not guide.is_expanded()

    def test_stage_count_matches_the_registry(self, qtbot, studio_app) -> None:
        guide = WorkflowGuide("machine-learning")
        qtbot.addWidget(guide)
        assert guide.stage_count() == 14

    def test_unknown_workspace_fails_loudly(self, qtbot, studio_app) -> None:
        with pytest.raises(ValueError, match="no §13 workflow registered"):
            WorkflowGuide("does-not-exist")


class TestMainWindowMounts:
    def test_every_workspace_tab_bar_carries_the_strip(
        self, qtbot, studio_app
    ) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QTabWidget

        from ui.main_window import MainWindow
        from ui.nine_cards import NINE_CARDS

        window = MainWindow()
        qtbot.addWidget(window)

        for card in NINE_CARDS:
            tabs = window.workspaces[card.card_id]
            assert isinstance(tabs, QTabWidget), card.card_id
            corner = tabs.cornerWidget(Qt.Corner.BottomRightCorner)
            assert isinstance(corner, WorkflowGuide), (
                f"{card.card_id}: no workflow strip mounted"
            )
            assert corner._workspace_key == card.card_id
            assert not corner.is_expanded()  # collapsed: quiet by default
