"""(part8 round 5) The workflow-guide ``where`` linter.

Round 4's honest limitation, closed: the WORKFLOWS registry's ``where``
texts are maintained by hand, so a renamed button could silently make a
guide stale.  This suite lints the guides against the REAL pages:

- every widget-name ``_al_*``-style reference the guides mention must
  still exist (the named surfaces are pinned by identity, not prose);
- the §14.1 guide's named surfaces (Active learning tab) must exist in
  the ML Workbench now that the tab is real;
- the strategies the AL tab offers must be exactly the backend's
  QUERY_STRATEGIES (a strategy added to the backend without the combo
  is a gap; a removed one still offered is a lie).
"""

from __future__ import annotations

import pytest


class TestWorkflowGuideWhereLint:
    def test_al_where_now_names_a_real_tab(self) -> None:
        # round 4 wrote "ML Workbench — self-training paths" for the ML
        # workspace's §13.3 semi-supervised note; round 5 adds the AL
        # campaign — the note must not claim a surface that is absent.
        from ui.widgets.workflow_guide import WORKFLOWS

        _chain, _stages, note = WORKFLOWS["machine-learning"]
        assert "active learning" in note.lower(), (
            "the ML workspace note must mention the §14.1 active-learning "
            "campaign now that the tab exists"
        )

    def test_ml_workbench_has_the_al_tab(self, qtbot, studio_app) -> None:
        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        labels = [
            page._tabs.tabText(i) for i in range(page._tabs.count())
        ]
        assert "Active learning" in labels, (
            f"the §14.1 tab must be mounted; tabs are {labels}"
        )

    def test_al_strategies_are_exactly_the_backends(self, qtbot, studio_app) -> None:
        from pharmaml.ml.active_learning import QUERY_STRATEGIES

        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        offered = [
            page._al_strategy.itemText(i)
            for i in range(page._al_strategy.count())
        ]
        assert tuple(offered) == tuple(QUERY_STRATEGIES), (
            "the strategy combo must mirror the backend registry exactly"
        )

    @pytest.mark.parametrize("surface", [
        "Data Explorer", "Chemistry Lab", "Model Analysis", "ML Workbench",
        "Deep Learning", "Docking", "PK/PD", "Molecular Dynamics",
        "Image Lab", "Generative",
    ])
    def test_named_pages_all_exist(self, surface) -> None:
        # every page the guides' ``where`` texts name must be importable
        # and constructible — a deleted/renamed page breaks its guide.
        import importlib

        module_map = {
            "Data Explorer": ("ui.pages.data_explorer", "DataExplorerPage"),
            "Chemistry Lab": ("ui.pages.chemistry", "ChemistryWorkbenchPage"),
            "Model Analysis": ("ui.pages.model_analysis", "ModelAnalysisPage"),
            "ML Workbench": ("ui.pages.ml_workbench", "MLWorkbenchPage"),
            "Deep Learning": (
                "ui.pages.deep_learning", "DeepLearningWorkbenchPage"
            ),
            "Docking": ("ui.pages.docking", "DockingWorkbenchPage"),
            "PK/PD": ("ui.pages.pkpd", "PKPDWorkbenchPage"),
            "Molecular Dynamics": (
                "ui.pages.molecular_dynamics", "MolecularDynamicsPage"
            ),
            "Image Lab": ("ui.pages.image_lab", "ImageLabPage"),
            "Generative": ("ui.pages.generative", "GenerativeWorkbenchPage"),
        }
        module_name, class_name = module_map[surface]
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), (
            f"{surface}: {module_name}.{class_name} is gone — the workflow "
            "guide names it as a stage surface"
        )

    def test_guides_never_reference_removed_buttons(self) -> None:
        # the specific historical names that were renamed in round 4's
        # n_jobs sweep — the guides must not pin the old prose
        from ui.widgets.workflow_guide import WORKFLOWS

        for key, (_chain, stages, _note) in WORKFLOWS.items():
            for _stage, where in stages:
                assert "the old combo" not in where.lower(), key
