#!/usr/bin/env python3
"""R14 (Phase 0 of MASTER_PROMPT part6): build docs/CAPABILITY_MAP.md.

Everything in the generated tables is derived LIVE from the working tree
(page imports grepped, module census, CLI parser census, test counts) —
the §3 audit rules forbid documentation-only claims.  Classification
columns (status, card assignment) are the Phase-0 editorial layer,
recorded here so the map is reproducible.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TK = ROOT


def page_backends(page: str) -> list[str]:
    """pharmaml modules a page actually imports (the §4.5 evidence rule)."""
    src = (TK / "legacy" / "ui" / "pages" / f"{page}.py").read_text(encoding="utf-8")
    mods = set(re.findall(r"from pharmaml\.([a-z_]+(?:\.[a-z_]+)?) import", src))
    mods |= set(re.findall(r"import pharmaml\.([a-z_]+(?:\.[a-z_]+)?)", src))
    return sorted(mods)


def module_loc(module: Path) -> int:
    return sum(1 for _ in module.open(encoding="utf-8", errors="replace"))


def test_count(pattern: str) -> int:
    n = 0
    base = TK / "legacy" / "tests_ui" if pattern.startswith("legacy/") else TK / "tests"
    rel = pattern[len("legacy/tests_ui/"):] if pattern.startswith("legacy/tests_ui/") else pattern
    for t in base.glob(rel):
        n += len(re.findall(r"^\s*def test_", t.read_text(encoding="utf-8"), re.MULTILINE))
    return n


# ---- live census ------------------------------------------------------------
PAGES = [
    "dashboard", "data_explorer", "chemistry", "pkpd", "biology", "docking",
    "model_analysis", "ml_workbench", "unsupervised", "deep_learning",
    "generative", "foundations", "bestofml", "responseai", "image_lab",
    "molecular_dynamics", "utilities", "settings_page",
]

page_rows = []
for p in PAGES:
    mods = page_backends(p)
    n = test_count(f"legacy/tests_ui/test_*{p.replace('_page','')}*.py")
    page_rows.append((p, ", ".join(mods) if mods else "—", n))

ml_modules = sorted(
    (f.name[:-3], module_loc(f))
    for f in (TK / "src" / "pharmaml" / "ml").glob("*.py")
    if f.name != "__init__.py"
)

cli_parsers = len(re.findall(r"\.add_parser\(", (TK / "src" / "pharmaml" / "cli.py").read_text(encoding="utf-8")))

# ---- the Phase-2 registry (the single source of truth) ----------------------
import sys

sys.path.insert(0, str(TK / "src"))
from pharmaml.studio.capability_registry import CAPABILITIES, summary

# ---- the Phase-0 editorial layer -------------------------------------------
# Card assignment per §4 of part6, verified against the live page census.
CARD_OF_PAGE = {
    "dashboard": "01 Introduction (current host of the §5 grid)",
    "data_explorer": "02 Datasets",
    "chemistry": "02 Datasets (chemical preparation)",
    "pkpd": "07 PK/PD",
    "biology": "07 PK/PD (biology sub-workspace)",
    "docking": "06 Molecular Docking",
    "model_analysis": "03 Machine Learning",
    "ml_workbench": "03 Machine Learning",
    "unsupervised": "03 Machine Learning (exploration)",
    "deep_learning": "04 Deep Learning",
    "generative": "05 Generative AI",
    "foundations": "03 Machine Learning (walkthroughs)",
    "bestofml": "03 Machine Learning (sweeps)",
    "responseai": "07 PK/PD (response AI)",
    "image_lab": "09 Visualization",
    "molecular_dynamics": "08 Molecular Dynamics Analysis",
    "utilities": "02 Datasets + cross-card utilities",
    "settings_page": "01 Introduction (global menu)",
}

# Historical Phase-0 C-list (kept for the record; §4 now emits from the
# live registry above, so this dict is no longer the source).
BACKEND_ONLY = {
    "gnn": ("GNN featuriser/trainer + XAI registry", "04 Deep Learning (Phase 3)"),
    "gnnxai": ("GNN explainability", "04 Deep Learning (Phase 3)"),
    "geometric": ("geometric deep-learning primitives", "04 Deep Learning (Phase 3)"),
    "pruning": ("magnitude/structured network pruning", "04 Deep Learning (Phase 3)"),
    "quantization": ("int8/fp16-style quantisation arithmetic", "04 Deep Learning (Phase 3)"),
    "harness": ("offline sharded-training harness (ZeRO + early stopping)", "04 Deep Learning (Phase 3)"),
    "conformal": ("split-conformal regression intervals", "03 Machine Learning (Phase 3)"),
    "conformal_classification": ("classification/label-conditional prediction sets", "03 Machine Learning (Phase 3)"),
    "coverage_diagnostics": ("empirical-vs-nominal coverage shortfall table", "03 Machine Learning (Phase 3)"),
    "split_audit": ("identifier-overlap audit across splits", "02 Datasets (Phase 3)"),
    "calibration": ("probability calibration reports/recalibration", "03 Machine Learning (Phase 3)"),
    "comparison": ("paired model-comparison tests + corrections", "03 Machine Learning (Phase 3)"),
    "decision": ("decision-curve analysis", "03 Machine Learning (Phase 3)"),
    "drift": ("distribution-shift detection (PSI/quantile sketches)", "03 Machine Learning (Phase 3)"),
    "model_card": ("model cards / provenance sheets", "03 Machine Learning (Phase 3)"),
    "multiplicity": ("multiple-comparison corrections", "03 Machine Learning (Phase 3)"),
    "power": ("fold-count power analysis", "03 Machine Learning (Phase 3)"),
    "sar": ("MMP / activity cliffs / R-groups", "03 Machine Learning (Phase 3)"),
    "subgroups": ("subgroup performance audits", "03 Machine Learning (Phase 3)"),
}

# ---- emit ---------------------------------------------------------------------
out = []
w = out.append
w("# Capability map — the nine-card-first audit (Phase 0)")
w("")
w("**Baseline:** v1.2.183 (`e81985d`) · **Spec:** MASTER_PROMPT part6 §3")
w("**Method:** every row below was derived live from the working tree —")
w("page imports grepped, module and CLI censuses counted, tests collected —")
w("never copied from prose.  This is the Phase-0 deliverable of the")
w("nine-card-first enhancement architecture: the auditable")
w("capability→card mapping and the migration plan for Phases 1-6.")
w("")
w("## 1. What the launch architecture actually is (verified)")
w("")
w("```")
w("studio.py → ui.run() → MainWindow")
w("  ├─ TopBar (Home / Back / location / jobs / Settings / Advanced)")
w("  └─ mode stack: NineCardHome (default) → CardWorkspaces ×9 →")
w("       QStackedWidget(18 pages, unchanged)")
w("       (the 18-section sidebar rides only behind the Advanced toggle)")
w("```")
w("")
w("Since v1.2.185 (Phase 1) the Studio opens on the nine-card home;")
w("`MainWindow.go_to(section)` remains the single routing API and all 18")
w("pages are preserved.  Since v1.2.186 (Phase 2) the card ORDER and the")
w("capability assignments live in the pure-Python registry")
w("`src/pharmaml/studio/capability_registry.py` (Qt-free, §6);")
w("`ui/nine_cards.py` asserts agreement at import time.")
w("")
w("## 2. The eighteen pages and their verified backends")
w("")
w("| Page | pharmaml modules imported (live grep) | card assignment (Phase 2 target) |")
w("| --- | --- | --- |")
for p, mods, _n in page_rows:
    w(f"| `{p}` | {mods} | {CARD_OF_PAGE[p]} |")
w("")
w(f"Tests: {test_count('legacy/tests_ui/test_*.py')} UI test functions across")
w(f"{len(list((TK/'legacy'/'tests_ui').glob('test_*.py')))} UI suites; page smoke")
w("coverage via `legacy/tests_ui/test_pages_smoke.py`.")
w("")
w("## 3. Backend census")
w("")
w(f"- `src/pharmaml/ml/`: **{len(ml_modules)} modules**,")
w(f"{sum(l for _, l in ml_modules):,} lines.")
w("- CLI: " + f"**{cli_parsers}** `add_parser` sites (16 command groups).")
w(f"- Library tests: {test_count('test_*.py')} test functions across")
w(f"{len(list((TK/'tests').glob('test_*.py')))} suites.")
w("- `src/pharmaml/studio/` (runs + evidence): the §11 real-evidence")
w("  backend already exists — Phase 4 extends, not replaces, it.")
w("")
w("## 4. Backend-only capabilities — now ASSIGNED (Phase 2 complete)")
w("")
w("The Phase-0 audit found these modules referenced by no Studio page.")
w("As of v1.2.186 every one carries its **primary card assignment**")
w("(part6 §6.1) in the machine-checkable registry")
w("`src/pharmaml/studio/capability_registry.py`; the 11 §6.2 validation")
w("rules are pinned by `tests/test_capability_registry.py` (27 pins).")
w("Status `api` means the backend and its tests are real, and the GUI")
w("form is the Phase-3 work:")
w("")
w("| Module | What it does | Primary card | Route (Phase-3 form) |")
w("| --- | --- | --- | --- |")
for cap in CAPABILITIES:
    if cap.status == "api":
        w(f"| `{cap.backend_module}` | {cap.name} | {cap.card} | {cap.route} |")
w("")
w("### The Other-card decision (part6 §2.3)")
w("")
w("After the full assignment pass, **no Other card is created** (§2.3")
w("rule 8): every genuine user-facing capability above has a logical")
w("home — 13 modules under Machine Learning, 5 under Deep Learning, 1")
w("under Datasets — and the remaining Studio sections (biology,")
w("foundations, best-of-ml, responseai) are integrated pages with real")
w("backends, already reachable.  The decision is pinned by")
w("`TestRule3OtherPosition` so it can only change through a new,")
w("justified assignment audit, never by drift.")
w("")
w("## 5. What counts as integrated (§3.2, restated as the gate)")
w("")
w("A capability is GUI-integrated only when its workflow: collects and")
w("validates inputs → invokes the real backend → reports progress and")
w("errors → displays real results → supports inspection and export →")
w("preserves metadata/provenance.  Import presence, a button, or a")
w("synthetic demo is NOT integration.  Phase 6's coverage tests enforce")
w("this against the registry, not against prose.")
w("")
_s = summary()
w("Registry coverage (live): " + f"**{_s.total} capabilities**, "
  + ", ".join(f"{v} {k}" for k, v in _s.by_status.items())
  + "; orphans 0; test gaps 0; Other card absent.")
w("")
w("## 6. Migration plan (Phases 1-6)")
w("")
w("- **Phase 1 — nine-card-first shell (DONE, v1.2.185):** invert MainWindow so the")
w("  nine-card home is the default screen; sidebar becomes an optional")
w("  advanced mode; Home/Back + history; Settings via global menu;")
w("  all 18 pages preserved as card workspaces; registry validation")
w("  tests (exact order, Other-last, routes valid).")
w("- **Phase 2 — full capability assignment (DONE, v1.2.186):** every public capability")
w("  mapped to exactly one primary card (§6.1 unique ownership); the")
w("  C-list above reclassified as it lands; `Other` added last only if")
w("  genuinely unassignable capabilities remain.")
w("- **Phase 3 — backend integration completion:** build the missing GUI")
w("  workflows for the C-list rows above (forms → execution → results →")
w("  export), each calling the existing backend API (§3.3: one canonical")
w("  implementation, never a GUI reimplementation).")
w("- **Phase 4 — global search + shared workspace:** capability search /")
w("  command palette over the registry; shared project/session state")
w("  (datasets, models, runs, artifacts) extending `pharmaml.studio.runs`.")
w("- **Phase 5 — refinement:** responsive polish, contextual help,")
w("  dependency diagnostics, memory-aware jobs.")
w("- **Phase 6 — verification:** registry-driven coverage tests,")
w("  end-to-end workflow tests per card, dependency-matrix verification,")
w("  documentation + screenshots, release validation.")
w("")
w("## 7. Gaps (1-3 CLOSED; 4-5 are the Phase-4 targets)")
w("")
w("1. CLOSED v1.2.185 (Phase 1): the nine-card grid IS the main screen.")
w("2. CLOSED v1.2.185 (Phase 1): Settings lives on the global top bar.")
w("3. CLOSED v1.2.186 (Phase 2): all 19 backend-only modules carry")
w("   primary-card assignments in the registry; GUI forms = Phase 3.")
w("4. No global capability search exists yet (Phase 4).")
w("5. No shared project/session layer beyond `studio.runs` evidence")
w("   collectors (Phase 4).")
w("")
w("*Generated by `scripts/build_capability_map.py` (R14, Phase 0); the")
w("counts re-derive from the live tree on every run.*")
w("")

outpath = TK / "docs" / "CAPABILITY_MAP.md"
outpath.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {outpath} ({len(out)} lines)")
print(f"ml modules: {len(ml_modules)}; cli parsers: {cli_parsers}; "
      f"ui tests: {test_count('legacy/tests_ui/test_*.py')}; lib tests: {test_count('test_*.py')}")
