"""The capability registry contract (part6 §6.2, Phase 2).

The eleven validation rules, pinned against the pure-Python registry
under ``src/pharmaml/studio/capability_registry.py``.  No Qt anywhere:
the registry must import and validate headlessly (rule: the GUI
consumes the registry, never the reverse).  These tests ARE the
automated coverage summary §6.2 demands — documentation alone is
never a substitute.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pharmaml.studio.capability_registry import (
    CAPABILITIES,
    CARD_ORDER,
    OTHER_CARD_PRESENT,
    SECTION_KEYS,
    Capability,
    capabilities_for_card,
    card_route_target,
    find,
    summary,
)

_TK = Path(__file__).resolve().parents[1]

#: The class-C backend-only modules still awaiting their GUI workflow.
#: EMPTY since v1.2.190 (part6 Phase 3, batch 4 — the closer): the
#: classification-set sibling, SAR matched pairs and the split-overlap
#: audit landed (ML Workbench Statistical tools + the Data Explorer's
#: two new panels), so every backend module now has a GUI workflow.
C_LIST_ASSIGNMENTS: dict[str, str] = {}


class TestRule1ValidIdentifiers:
    def test_every_card_id_is_a_valid_identifier(self) -> None:
        for card_id in CARD_ORDER:
            assert card_id and card_id.replace("-", "").isalnum(), card_id

    def test_every_capability_id_is_unique_and_stable(self) -> None:
        ids = [c.capability_id for c in CAPABILITIES]
        assert len(ids) == len(set(ids)), "capability ids must be unique"
        for cap_id in ids:
            assert "." in cap_id and " " not in cap_id, cap_id


class TestRule2DirectiveOrder:
    def test_the_nine_primary_cards_in_the_required_order(self) -> None:
        assert CARD_ORDER == (
            "introduction", "datasets", "machine-learning", "deep-learning",
            "generative-ai", "molecular-docking", "pkpd",
            "molecular-dynamics-analysis", "visualization",
        )


class TestRule3OtherPosition:
    def test_no_other_card_and_the_decision_is_pinned(self) -> None:
        # §2.3 rule 8: every genuine user-facing capability has a home
        # (the 19 C-list modules included), so no Other card exists.
        # If this pin fails, a new assignment audit must justify it.
        assert OTHER_CARD_PRESENT is False
        assert "other" not in CARD_ORDER
        assert summary().other_card_present is False


class TestRule4RoutesResolve:
    def test_every_route_points_to_a_real_section(self) -> None:
        for cap in CAPABILITIES:
            assert cap.route in SECTION_KEYS, (cap.capability_id, cap.route)

    def test_every_section_is_routed_by_some_capability(self) -> None:
        routed = {cap.route for cap in CAPABILITIES}
        unrouted = set(SECTION_KEYS) - routed - {"settings"}
        assert not unrouted, f"sections with no capability: {sorted(unrouted)}"


class TestRule5BackendReferences:
    def test_every_feature_has_a_backend_reference_or_blocked_status(self) -> None:
        for cap in CAPABILITIES:
            assert cap.backend_module, cap.capability_id
            assert cap.public_api, cap.capability_id
            if cap.status == "blocked":
                assert cap.status_detail, "blocked rows must name the blocker"

    def test_backend_modules_really_exist(self) -> None:
        import importlib

        for cap in CAPABILITIES:
            if cap.backend_module.startswith("pharmaml."):
                importlib.import_module(cap.backend_module)  # raises if fake

    def test_public_api_symbols_really_exist(self) -> None:
        import importlib
        import re

        for cap in CAPABILITIES:
            if not cap.backend_module.startswith("pharmaml."):
                continue
            module = importlib.import_module(cap.backend_module)
            for symbol in re.split(r",\s*", cap.public_api):
                symbol = symbol.strip()
                assert hasattr(module, symbol), (cap.capability_id, symbol)


class TestRule6NoOrphans:
    def test_no_capability_is_unintentionally_orphaned(self) -> None:
        s = summary()
        assert s.orphans == ()
        assert s.duplicate_ownership == ()  # no routeless cards either

    def test_every_card_owns_at_least_one_capability(self) -> None:
        s = summary()
        for card in CARD_ORDER:
            assert s.by_card.get(card, 0) >= 1, f"{card} owns nothing"


class TestRule7UniqueOwnership:
    def test_no_duplicate_primary_ownership(self) -> None:
        # §6.1: one primary card per capability — the (capability, card)
        # pairs are unique by construction; the audit target is the
        # reverse: a capability appearing twice with different cards.
        seen: dict[str, str] = {}
        for cap in CAPABILITIES:
            if cap.capability_id in seen:
                assert seen[cap.capability_id] == cap.card, cap.capability_id
            seen[cap.capability_id] = cap.card

    def test_the_c_list_assignment_is_exactly_as_audited(self) -> None:
        for module, card in C_LIST_ASSIGNMENTS.items():
            rows = [c for c in CAPABILITIES if c.backend_module == module]
            assert rows, f"{module} missing from the registry"
            for row in rows:
                assert row.card == card, (module, row.card, card)
        # and the reverse: nothing else claims to be class-C api status
        api_rows = {c.backend_module for c in CAPABILITIES if c.status == "api"}
        assert api_rows == set(C_LIST_ASSIGNMENTS)


class TestRule8SpecializedPagesReachable:
    def test_every_specialized_page_remains_reachable(self) -> None:
        # The 18 sections the Studio ships; the registry's routes plus
        # Settings must cover them all (§5.1: nothing is discarded).
        reachable = {cap.route for cap in CAPABILITIES} | {"settings"}
        assert reachable == set(SECTION_KEYS)

    def test_the_ui_card_registry_agrees_with_the_backend(self) -> None:
        # ui/nine_cards.py guards this at import time; assert it directly
        # so a headless run also proves the single source of truth.  The
        # subprocess bootstraps the same EGL-stub the root conftest uses
        # (ui.headless.ensure_headless) so Qt imports in a bare sandbox.
        code = (
            "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')\n"
            "from ui.headless import ensure_headless\n"
            "ensure_headless()\n"
            "from ui import nine_cards\n"
            "from pharmaml.studio.capability_registry import CARD_ORDER\n"
            "assert tuple(c.card_id for c in nine_cards.NINE_CARDS) == CARD_ORDER\n"
            "print('AGREE')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=_TK, capture_output=True,
            text=True, timeout=120, check=False,
        )
        assert out.stdout.strip().endswith("AGREE"), out.stderr[-400:]


class TestRule9OptionalDependencies:
    def test_optional_dependencies_are_declared_not_implied(self) -> None:
        # Rows with optional deps must name them; the docker/vina engine
        # rows carry their engines.  Today no row needs a soft dep that
        # pandas/rdkit/scipy (hard deps) do not already cover — the
        # registry says so explicitly rather than staying silent.
        for cap in CAPABILITIES:
            if cap.engines:
                assert all(isinstance(e, str) and e for e in cap.engines)

    def test_engine_limitations_are_disclosed(self) -> None:
        # The one external-engine row: Vina docking names its engine.
        vina = find("dock.vina")
        assert vina is not None and "vina" in vina.engines
        # Everything else runs on the package's own dependencies.
        engine_rows = [c for c in CAPABILITIES if c.engines]
        assert [c.capability_id for c in engine_rows] == ["dock.vina"]


class TestRule10StatusHonesty:
    def test_statuses_are_the_allowed_vocabulary(self) -> None:
        for cap in CAPABILITIES:
            assert cap.status in ("integrated", "cli", "api", "blocked"), cap.status

    def test_api_rows_carry_a_phase3_target_detail(self) -> None:
        for cap in CAPABILITIES:
            if cap.status == "api":
                assert cap.status_detail, cap.capability_id


class TestRule11TestReferences:
    def test_every_feature_has_tests_or_an_explicit_gap_record(self) -> None:
        s = summary()
        assert s.test_gaps == ()

    def test_referenced_test_files_exist(self) -> None:
        for cap in CAPABILITIES:
            for ref in cap.tests:
                assert (_TK / ref.split("::")[0]).is_file(), (cap.capability_id, ref)


class TestCoverageSummary:
    def test_the_summary_counts_are_self_consistent(self) -> None:
        s = summary()
        assert s.total == len(CAPABILITIES)
        assert sum(s.by_status.values()) == s.total
        assert sum(s.by_card.values()) == s.total
        assert set(s.by_card) <= set(CARD_ORDER)

    def test_capabilities_for_card_filters_correctly(self) -> None:
        for card in CARD_ORDER:
            for cap in capabilities_for_card(card):
                assert cap.card == card
        assert len(capabilities_for_card("no-such-card")) == 0

    def test_find_round_trips(self) -> None:
        for cap in CAPABILITIES:
            assert find(cap.capability_id) is cap
        assert find("no.such") is None

    def test_card_route_targets_are_sections(self) -> None:
        for card in CARD_ORDER:
            assert card_route_target(card) in SECTION_KEYS

    def test_registry_is_qt_free(self) -> None:
        # The §6 architecture rule: importable without a QApplication,
        # without importing PySide6 at all (subprocess = clean sys.modules).
        code = (
            "import sys\n"
            "import pharmaml.studio.capability_registry as r\n"
            "assert not [m for m in sys.modules if m.startswith('PySide6')]\n"
            "print('QTFREE')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=_TK / "src", capture_output=True,
            text=True, timeout=120, check=False,
        )
        assert out.stdout.strip().endswith("QTFREE"), out.stderr[-400:]


class TestCapabilityDataclass:
    def test_frozen_and_iterable(self) -> None:
        cap = CAPABILITIES[0]
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            cap.status = "integrated"  # type: ignore[misc]
        assert isinstance(cap, Capability)
