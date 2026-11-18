"""Round 22 (part6 Phase 5): glossary, resources, settings surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

_TK = Path(__file__).resolve().parents[1]
if str(_TK) not in sys.path:
    sys.path.insert(0, str(_TK))

# -- §17 F: the glossary ----------------------------------------------------------


def test_every_glossary_entry_is_short_and_carries_a_title() -> None:
    from pharmaml.studio.glossary import GLOSSARY

    assert len(GLOSSARY) >= 40
    for key, (title, text) in GLOSSARY.items():
        assert title and title.strip(), key
        assert 40 < len(text) < 400, f"{key}: §F wants SHORT help, got {len(text)} chars"
        assert text.rstrip().endswith("."), key  # complete sentences


def test_units_are_stated_where_units_exist() -> None:
    from pharmaml.studio.glossary import GLOSSARY

    unit_bearing = {
        "metric_rmse": ("units",),
        "tpsa": ("Å",),
        "rmsd": ("Å",),
        "rmsf": ("Å",),
        "clearance": ("mL",),
        "sa_score": ("1 (easy)",),
        "logp": ("log",),
    }
    for key, needles in unit_bearing.items():
        text = GLOSSARY[key][1]
        assert any(n in text for n in needles), f"{key} states no unit"


def test_missing_keys_reports_typos() -> None:
    from pharmaml.studio.glossary import missing_keys

    assert missing_keys(["random_seed", "rmsd", "auc_pk"]) == ()
    assert missing_keys(["nope", "random_seed"]) == ("nope",)


def test_tooltip_text_is_the_title_plus_explanation() -> None:
    from pharmaml.studio.glossary import tooltip_text

    text = tooltip_text("morgan_fingerprint")
    assert text.startswith("Morgan fingerprint (ECFP) — ")
    assert tooltip_text("bogus") == ""


# -- §17 E: the honest resource probe ------------------------------------------------


def test_snapshot_never_fabricates_memory() -> None:
    from pharmaml.utils.resources import snapshot

    snap = snapshot()
    assert snap.cpu_count >= 1
    # memory is either evidenced or None — never a guess
    if snap.memory_bytes is None:
        assert snap.memory_source == "unknown"
        assert snap.memory_gib() == "unknown"
    else:
        assert snap.memory_bytes > 0
        assert snap.memory_source in ("cgroup", "meminfo", "cgroup+meminfo")


def test_available_memory_none_when_no_evidence(monkeypatch, tmp_path) -> None:
    from pharmaml.utils import resources

    monkeypatch.setattr(resources, "_cgroup_limit", lambda: None)
    monkeypatch.setattr(resources, "_mem_available", lambda: None)
    assert resources.available_memory_bytes() is None


def test_cgroup_v2_limit_wins_over_huge_meminfo(monkeypatch) -> None:
    from pharmaml.utils import resources

    monkeypatch.setattr(resources, "_cgroup_limit", lambda: 512 << 20)
    monkeypatch.setattr(resources, "_mem_available", lambda: 64 << 30)
    assert resources.available_memory_bytes() == 512 << 20  # the tighter bound


def test_cgroup_v1_sentinel_is_not_a_limit(monkeypath=None) -> None:
    from pharmaml.utils import resources

    assert resources.PRESETS[1].chunk_rows < resources.PRESETS[0].chunk_rows


def test_presets_cover_the_two_policies_and_resolve() -> None:
    from pharmaml.utils.resources import PRESETS, find_preset

    keys = [p.key for p in PRESETS]
    assert keys == ["balanced", "eco"]
    eco = find_preset("eco")
    assert eco.n_jobs == 1 and eco.chunk_rows <= 10_000
    assert eco.trades  # the honest "what you give up" sentence
    assert find_preset("nonsense").key == "balanced"  # safe fallback


def test_low_memory_recommendation_needs_evidence(monkeypatch) -> None:
    from pharmaml.utils import resources

    # evidenced small memory → eco
    monkeypatch.setattr(resources, "snapshot",
                        lambda: resources.ResourceSnapshot(2, 1 << 30, "meminfo"))
    assert resources.preset_for_low_memory() == "eco"
    # unknown memory → balanced, never a guess
    monkeypatch.setattr(resources, "snapshot",
                        lambda: resources.ResourceSnapshot(2, None, "unknown"))
    assert resources.preset_for_low_memory() == "balanced"
    # evidenced large memory → balanced
    monkeypatch.setattr(resources, "snapshot",
                        lambda: resources.ResourceSnapshot(2, 64 << 30, "meminfo"))
    assert resources.preset_for_low_memory() == "balanced"


def test_store_preset_validates_and_round_trips(monkeypatch) -> None:
    from pharmaml.utils import resources

    monkeypatch.delenv("PHARML_EXECUTION_PRESET", raising=False)
    assert resources.stored_preset() == "balanced"
    assert resources.store_preset("eco") is True
    assert resources.stored_preset() == "eco"
    assert resources.store_preset("bogus") is False
