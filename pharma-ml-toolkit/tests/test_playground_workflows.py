"""Tests for the playground's complementary-card workflows (round 202).

The owner's two-part ask: every card carries a short description of
what it does, and cards that complete a process together are grouped.
The machinery under test:

* :data:`WORKFLOWS` -- the declarative registry of pipelines, each
  step addressed by ``(tab title, card title, blurb)``;
* ``make_card(workflow=...)`` -- the ``\u25c7 pipeline \u00b7 step i/n``
  badge on every participating card's title row;
* the Home tab's *Complementary cards -- the workflows* map, whose
  step chips jump through ``_jump_to_workflow_step``;
* the Ctrl+F finder matching workflow names (``workflow_for_card``).

The registry is validated against the BUILT application (under Xvfb):
every step must resolve to a live card on the named tab, every live
card must carry a non-empty subtitle, and every badge must exist with
the exact expected text.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_PATH = TOOLKIT_ROOT / "playground.py"

#: The 13 notebook tab titles in BUILD order -- the registry's step
#: addresses must come from THIS set (a typo in a step's tab name would
#: otherwise silently never resolve), and the built notebook must match
#: it exactly (the round-201 jump-chip lesson: positions drift, titles
#: do not).
TAB_TITLES = (
    "Home",
    "Chemistry",
    "PK / PD",
    "Cell Lines",
    "Docking",
    "Data & ML",
    "Unsupervised",
    "Deep Learning",
    "Foundations",
    "Best-of ML",
    "Response AI",
    "Generative",
    "Utilities",
)


def _load_playground() -> Any:
    """Import playground.py by path (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("playground_wf_under_test", PLAYGROUND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["playground_wf_under_test"] = module
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


# (round 209) the display fixture moved to tests/conftest.py as ONE
# session-scoped Xvfb: a module-scoped server torn down between modules
# left a destroyed app's stale display connection pointing at a dead
# server -- the next Tk event pump hit it and Tk 9's X error handler
# exited the whole process (the round-209 collapse forensics).


# --------------------------------------------------------------------- #
# the registry itself -- pure data, no GUI needed
# --------------------------------------------------------------------- #
class TestWorkflowRegistry:
    def test_registry_exists_with_named_pipelines(self, playground: Any) -> None:
        assert isinstance(playground.WORKFLOWS, list) and len(playground.WORKFLOWS) >= 10
        for workflow in playground.WORKFLOWS:
            assert workflow["name"].strip(), "every workflow needs a name"
            assert workflow["purpose"].strip(), "every workflow needs a purpose line"

    def test_workflow_names_are_unique(self, playground: Any) -> None:
        names = [w["name"] for w in playground.WORKFLOWS]
        assert len(names) == len(set(names)), f"duplicate workflow names: {names}"

    def test_every_workflow_has_at_least_three_steps(self, playground: Any) -> None:
        for workflow in playground.WORKFLOWS:
            assert len(workflow["steps"]) >= 3, f"{workflow['name']} is too short to be a process"

    def test_every_step_is_a_tab_title_card_blurb_triple(self, playground: Any) -> None:
        for workflow in playground.WORKFLOWS:
            for step in workflow["steps"]:
                assert isinstance(step, tuple) and len(step) == 3, (workflow["name"], step)
                tab, card, blurb = step
                assert tab in TAB_TITLES, f"{workflow['name']}: unknown tab {tab!r}"
                assert card.strip(), f"{workflow['name']}: empty card title"
                assert blurb.strip(), f"{workflow['name']}: {card} has no step blurb"

    def test_steps_are_unique_within_a_workflow(self, playground: Any) -> None:
        for workflow in playground.WORKFLOWS:
            pairs = [(tab, card) for tab, card, _blurb in workflow["steps"]]
            assert len(pairs) == len(set(pairs)), f"{workflow['name']} repeats a step"

    def test_badge_text_for_registered_and_unregistered_cases(self, playground: Any) -> None:
        badge = playground.workflow_step_badge
        assert badge("QSAR end-to-end", "Dataset") == "QSAR end-to-end \u00b7 step 1/8"
        assert badge("QSAR end-to-end", "Leaderboard") == "QSAR end-to-end \u00b7 step 5/8"
        # a card joining a pipeline without being an official step
        assert badge("Docking campaign", "No such card") == "Docking campaign"
        # an unknown workflow passes the name through unchanged
        assert badge("Not a workflow", "Dataset") == "Not a workflow"

    def test_chain_text_lists_the_whole_pipeline(self, playground: Any) -> None:
        chain = playground.workflow_chain_text("Docking campaign")
        assert "Prepare, dock, rescore and audit" in chain
        assert "Ligand library \u2192 PDBQT" in chain
        assert "Receptor preparation" in chain
        assert "Consensus rescore" in chain
        # unknown workflow falls back to the bare name
        assert playground.workflow_chain_text("Nope") == "Nope"

    def test_workflow_for_card_resolves_by_tab_and_title(self, playground: Any) -> None:
        resolve = playground.workflow_for_card
        assert resolve("Docking", "Campaign runner") == "Docking campaign"
        assert resolve("Data & ML", "Campaign runner") == ""  # right title, wrong tab
        assert resolve("Utilities", "Reindex by number") == "Table preparation"
        assert resolve("Best-of ML", "The catalogue") == ""  # not in any workflow

    def test_every_registry_step_addressed_once(self, playground: Any) -> None:
        """No (tab, card) may appear in two different workflows."""
        seen: dict[tuple[str, str], str] = {}
        for workflow in playground.WORKFLOWS:
            for tab, card, _blurb in workflow["steps"]:
                key = (tab, card)
                assert key not in seen, (
                    f"{card!r} on {tab!r} is claimed by both "
                    f"{seen[key]!r} and {workflow['name']!r}"
                )
                seen[key] = workflow["name"]


# --------------------------------------------------------------------- #
# the built application -- under Xvfb
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestWorkflowsInBuiltApp:
    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    def test_every_live_card_carries_a_subtitle(
        self, playground: Any, app: Any, xvfb_display: str
    ) -> None:
        """The owner's first ask: every card says what it does."""
        live = app._live_cards()
        assert len(live) >= 130, f"expected the full card population, saw {len(live)}"
        bare = [title for title, subtitle, _shell, _index in live if not subtitle.strip()]
        assert bare == [], f"cards without a description: {bare}"

    def test_every_workflow_step_resolves_to_a_live_card(
        self, playground: Any, app: Any, xvfb_display: str
    ) -> None:
        tabs = [str(app._book.tab(tab, "text")).strip() for tab in app._book.tabs()]
        assert tuple(tabs) == TAB_TITLES, f"tab set drifted: {tabs}"
        tab_pos = {name: index for index, name in enumerate(tabs)}
        live_keys = {(index, title) for title, _s, _sh, index in app._live_cards()}
        for workflow in playground.WORKFLOWS:
            for step_tab, step_card, _blurb in workflow["steps"]:
                assert (tab_pos[step_tab], step_card) in live_keys, (
                    f"{workflow['name']}: no live card {step_card!r} on {step_tab!r}"
                )

    def test_every_step_wears_its_badge_with_exact_text(
        self, playground: Any, app: Any, xvfb_display: str
    ) -> None:
        badges = playground._WORKFLOW_BADGES
        for workflow in playground.WORKFLOWS:
            for _tab, card, _blurb in workflow["steps"]:
                key = (workflow["name"], card)
                assert key in badges, f"{workflow['name']}: {card!r} has no badge"
                assert badges[key] == playground.workflow_step_badge(
                    workflow["name"], card
                ), f"{card!r}: badge text drifted"

    def test_home_tab_renders_the_workflow_map_card(
        self, playground: Any, app: Any, xvfb_display: str
    ) -> None:
        titles = {title for title, _subtitle, _shell, index in app._live_cards() if index == 0}
        assert "Complementary cards — the workflows" in titles

    def test_workflow_step_jump_selects_the_right_tab(
        self, playground: Any, app: Any, xvfb_display: str
    ) -> None:
        app._jump_to_workflow_step("Data & ML", "Applicability domain & prediction reliability")
        current = str(app._book.tab(app._book.select(), "text")).strip()
        assert current == "Data & ML"
        app._jump_to_workflow_step("Generative", "Pareto front")
        current = str(app._book.tab(app._book.select(), "text")).strip()
        assert current == "Generative"

    def test_workflow_step_jump_is_a_silent_noop_for_unknown_cards(
        self, playground: Any, app: Any, xvfb_display: str
    ) -> None:
        before = str(app._book.tab(app._book.select(), "text")).strip()
        app._jump_to_workflow_step("Data & ML", "No Such Card Ever")
        app._jump_to_workflow_step("No Such Tab", "Dataset")
        after = str(app._book.tab(app._book.select(), "text")).strip()
        assert before == after

    def test_finder_haystack_includes_workflow_names(
        self, playground: Any, app: Any, xvfb_display: str
    ) -> None:
        """Typing a workflow name in Ctrl+F must find the chained cards."""
        resolve = playground.workflow_for_card
        tabs = [str(app._book.tab(tab, "text")).strip() for tab in app._book.tabs()]
        chained = 0
        for title, _subtitle, _shell, index in app._live_cards():
            if resolve(tabs[index], title) == "Docking campaign":
                chained += 1
        assert chained == 7, f"the docking campaign chain should badge 7 cards, saw {chained}"

    def test_module_docstring_documents_the_grouping_guarantee(
        self, playground: Any
    ) -> None:
        assert "every card is described and grouped" in (playground.__doc__ or "")
        assert "WORKFLOWS" in (playground.__doc__ or "")
