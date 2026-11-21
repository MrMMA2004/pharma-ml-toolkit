"""Round-304 (part12 Phase-3) — the architecture-hygiene guard.

Pins the A-6/B-7 map from ``docs/part12_audit/ARCHITECTURE_REVIEW.md``
so the two documented hygiene facts cannot drift silently:

* the 9 library-surface modules (``fih``, ``pkstats``, ``odesim``,
  ``association``, ``admet``, ``lime``, ``dataset``, ``surrogate``,
  ``ebm``) are intentionally importable public API with no internal
  importers — their coverage is the doctest lane (``--doctest-modules``
  collects them; the ten-chunk gate's chunk-10 leg runs it). If one of
  them gains an internal consumer, this guard fires so the census of
  "doctest-only" surfaces is re-trued;
* the duplicate-symbol families — ``gdsc_anova`` x2,
  ``mabel_dose``/``noael_hed`` x3, ``scaffold_split`` x2 — are
  intentional interface variants (see the docstring map at every
  definition site): the exact definition-site sets are pinned, the
  mg-chain ``mabel_dose`` pair is pinned NUMERICALLY (fih <-> pbpk),
  the unitless ``pkpd.pbpk`` variant's relationship to the mg chain is
  pinned, and the ``noael_hed`` trio's default parity is pinned.

The round also removed the retired Tk ``playground.py`` and its
``tests/test_playground.py`` suite — the definition-site pins below
double as the tree's "no new duplicates" tripwire for the post-
playground tree.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "pharmaml"

#: The doctest-only library-surface modules (A-6, ground-truthed by
#: the AST import scan on the round-304 tree; re-trued round 315:
#: ml.admet gained its internal consumer pharmaml.api.routes.admet at
#: r314 and left the doctest-only tier; the census documents the
#: remaining set).
LIBRARY_SURFACE_MODULES: tuple[str, ...] = (
    "pharmaml.pharma.fih",
    "pharmaml.pharma.pkstats",
    "pharmaml.pharma.odesim",
    "pharmaml.biology.association",
    "pharmaml.ml.lime",
    "pharmaml.ml.dataset",
    "pharmaml.ml.surrogate",
    "pharmaml.ml.ebm",
)

#: The duplicate-symbol families -> the exact module set where each is
#: defined (B-7's map; every site carries the docstring map note).
DUPLICATE_SYMBOL_MAP: dict[str, frozenset[str]] = {
    "gdsc_anova": frozenset(
        {"pharmaml.biology.association", "pharmaml.ml.pharmacogenomics"}
    ),
    "mabel_dose": frozenset(
        {"pharmaml.pharma.fih", "pharmaml.pharma.pbpk", "pharmaml.pkpd.pbpk"}
    ),
    "noael_hed": frozenset(
        {"pharmaml.pharma.fih", "pharmaml.pharma.pbpk", "pharmaml.pkpd.pbpk"}
    ),
    "scaffold_split": frozenset(
        {"pharmaml.ml.benchmark", "pharmaml.ml.dataset"}
    ),
}


def _dotted(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["pharmaml", *parts])


def _imported_targets(path: Path) -> set[str]:
    """Every dotted import target a file names, relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _dotted(path).rsplit(".", 1)[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                root = package.split(".")
                up = node.level - 1
                base = ".".join(
                    root[: len(root) - up]
                    + ((node.module or "").split(".") if node.module else [])
                )
            out.add(base)
            for alias in node.names:
                out.add(f"{base}.{alias.name}" if base else alias.name)
    return out


def _definition_sites(symbol: str) -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - malformed files fail elsewhere
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == symbol:
                found.add(_dotted(path))
    return found


_GRAPH_CACHE: dict[str, set[str]] | None = None


def _import_graph() -> dict[str, set[str]]:
    """Import target -> the modules importing it (AST truth, computed once).

    The library's own zero-AST-importer surface is broader than A-6's
    nine (API routes are dynamically mounted, CLI entries resolve at
    run time) — this guard only pins the nine the audit named.
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        graph: dict[str, set[str]] = {}
        for path in SRC.rglob("*.py"):
            if path == SRC / "__init__.py":
                continue
            own = _dotted(path)
            for target in _imported_targets(path):
                graph.setdefault(target, set()).add(own)
        _GRAPH_CACHE = graph
    return _GRAPH_CACHE


class TestLibrarySurfaceModules:
    """A-6's modules: importable public API, zero internal importers.

    (The broader zero-importer surface — routes, CLI entries — is the
    architecture's dynamic-registration reality; only the audit's
    named set is pinned, because those are the ones whose
    "doctest-only coverage" tier the census documents. Re-trued at
    r315: ml.admet gained its internal consumer routes/admet.py at
    r314 and left the tier.)
    """

    @pytest.mark.parametrize("module", LIBRARY_SURFACE_MODULES)
    def test_module_imports(self, module: str) -> None:
        importlib.import_module(module)

    def test_no_internal_importers(self) -> None:
        graph = _import_graph()
        got = {m: graph.get(m, set()) for m in LIBRARY_SURFACE_MODULES}
        offenders = {m: sorted(imp) for m, imp in got.items() if imp}
        assert not offenders, (
            "the library-surface census changed (a doctest-only module "
            "gained an internal consumer -- re-true A-6 and this guard): "
            f"{offenders}"
        )


class TestDuplicateSymbolMap:
    """B-7's families: the definition-site sets, exactly as documented."""

    @pytest.mark.parametrize("symbol,expected", sorted(DUPLICATE_SYMBOL_MAP.items()))
    def test_definition_sites(self, symbol: str, expected: frozenset[str]) -> None:
        got = _definition_sites(symbol)
        assert got == set(expected), (
            f"{symbol} definition sites moved: got {sorted(got)}, "
            f"pinned {sorted(expected)} -- a 4th duplicate or a removal "
            "needs the docstring maps re-trued"
        )

    @pytest.mark.parametrize(
        "module,symbol",
        [
            ("pharmaml.biology.association", "gdsc_anova"),
            ("pharmaml.ml.pharmacogenomics", "gdsc_anova"),
            ("pharmaml.pharma.fih", "mabel_dose"),
            ("pharmaml.pharma.pbpk", "mabel_dose"),
            ("pharmaml.pkpd.pbpk", "mabel_dose"),
            ("pharmaml.pharma.fih", "noael_hed"),
            ("pharmaml.pharma.pbpk", "noael_hed"),
            ("pharmaml.pkpd.pbpk", "noael_hed"),
            ("pharmaml.ml.benchmark", "scaffold_split"),
            ("pharmaml.ml.dataset", "scaffold_split"),
        ],
    )
    def test_docstring_map_present(self, module: str, symbol: str) -> None:
        fn = getattr(importlib.import_module(module), symbol)
        doc = inspect.getdoc(fn) or ""
        assert "Duplicate-symbol map (round 304" in doc, (
            f"{module}.{symbol} lost its duplicate-symbol map note"
        )

    def test_registry_references_live(self) -> None:
        """The registry claims in the docstring maps stay true."""
        from pharmaml.studio.capability_registry import CAPABILITIES

        pkpd_pbpk = [
            c for c in CAPABILITIES if c.backend_module == "pharmaml.pkpd.pbpk"
        ]
        assert pkpd_pbpk, "the pkpd.pbpk registry entry is gone"
        assert any("mabel_dose" in c.public_api for c in pkpd_pbpk), (
            "the pkpd.pbpk registry entry no longer names mabel_dose"
        )
        ml_bench = [
            c for c in CAPABILITIES if c.backend_module == "pharmaml.ml.benchmark"
        ]
        assert ml_bench, "the ml.benchmark registry entry is gone"
        assert any("scaffold_split" in c.public_api for c in ml_bench), (
            "the ml.benchmark registry entry no longer names scaffold_split"
        )


class TestMabelParity:
    """The mg-chain pair (fih <-> pharma.pbpk) is one math, pinned."""

    GRID = tuple(
        (ec50, mw, fu, vd, ro)
        for ec50 in (0.1, 1.0, 10.0, 100.0)
        for mw in (150.0, 400.0, 800.0)
        for fu in (0.01, 0.1, 0.5, 1.0)
        for vd in (5.0, 50.0, 500.0)
        for ro in (0.05, 0.10, 0.20)
    )

    @pytest.mark.parametrize("ec50,mw,fu,vd,ro", GRID)
    def test_mg_chain_pair(self, ec50: float, mw: float, fu: float, vd: float, ro: float) -> None:
        from pharmaml.pharma.fih import mabel_dose as fih_mabel
        from pharmaml.pharma.pbpk import mabel_dose as pbpk_mabel

        assert fih_mabel(ec50, mw, fu, vd, target_ro=ro) == pytest.approx(
            pbpk_mabel(ec50, mw, fu, vd, target_occupancy=ro), rel=1e-12
        )

    def test_published_doctest_anchors(self) -> None:
        from pharmaml.pharma.fih import mabel_dose as fih_mabel
        from pharmaml.pharma.pbpk import mabel_dose as pbpk_mabel

        assert round(fih_mabel(ec50_nm=1.0, mw_da=400.0, fu=0.10, vd_l=100.0), 4) == 0.0444
        assert round(pbpk_mabel(10.0, 400.0, 0.10, 120.0), 2) == 0.53

    @pytest.mark.parametrize(
        "ec50,mw,fu,vd",
        [(0.5, 400.0, 0.1, 20.0), (10.0, 250.0, 0.05, 80.0), (2.0, 800.0, 0.5, 5.0)],
    )
    def test_unitless_relationship(self, ec50: float, mw: float, fu: float, vd: float) -> None:
        """pkpd's unitless dose x MW/1e6 == fih's mg dose (default ro)."""
        from pharmaml.pharma.fih import mabel_dose as fih_mabel
        from pharmaml.pkpd.pbpk import mabel_dose as unitless

        mg = fih_mabel(ec50, mw, fu, vd)
        raw = unitless(ec50=ec50, fu_plasma=fu, v_volume=vd)
        assert mg == pytest.approx(raw * mw / 1e6, rel=1e-12)


class TestNoaelParity:
    """The noael_hed trio computes one formula at the shared defaults."""

    @pytest.mark.parametrize(
        "noael,bw_animal",
        [(100.0, 0.25), (50.0, 0.02), (250.0, 4.0), (10.0, 0.5)],
    )
    def test_default_parity(self, noael: float, bw_animal: float) -> None:
        from pharmaml.pharma.fih import noael_hed as fih_hed
        from pharmaml.pharma.pbpk import noael_hed as pbpk_hed
        from pharmaml.pkpd.pbpk import noael_hed as pkpd_hed

        expected = fih_hed(noael, bw_animal)
        assert expected == pytest.approx(
            pbpk_hed(noael, bw_animal), rel=1e-12
        )
        assert expected == pytest.approx(
            pkpd_hed(noael_mgkg=noael, animal_bw=bw_animal), rel=1e-12
        )

    def test_published_doctest_anchor(self) -> None:
        from pharmaml.pharma.fih import noael_hed as fih_hed

        assert round(fih_hed(100.0, 0.25), 1) == 102.3


class TestScaffoldVariants:
    """Both scaffold_split variants run and cover their inputs disjointly."""

    def test_benchmark_variant_smoke(self) -> None:
        from pharmaml.ml.benchmark import scaffold_split

        smiles = ["C", "CC", "CCC", "C1CC1", "C1CCC1"]
        got = scaffold_split(smiles, 0.4, 0.3, 0.3)
        seen = sorted(
            int(i) for key in ("train", "val", "test") for i in got[key]
        )
        assert seen == [0, 1, 2, 3, 4]
        assert set(got) >= {"train", "val", "test"}

    def test_dataset_variant_smoke(self) -> None:
        import numpy as np

        from pharmaml.ml.dataset import Dataset, scaffold_split

        smiles = [
            "CCO", "CCCO", "c1ccccc1O", "Cc1ccccc1N", "C1CCCCC1O", "CC1CCCCC1",
        ]
        ds = Dataset(X=np.arange(6.0).reshape(6, 1), y=np.ones(6), ids=smiles)
        train, valid, test = scaffold_split(ds, fracs=(0.5, 0.25, 0.25))
        total = len(train) + len(valid) + len(test)
        assert total == 6
        assert not (set(train.ids) & set(valid.ids))
        assert not (set(train.ids) & set(test.ids))
        assert not (set(valid.ids) & set(test.ids))


class TestSbomArtifact:
    """FEATURE_GAP #8: the shipped CycloneDX SBOM stays a real artifact.

    ``packaging/sbom/sbom.cdx.json`` is generated by
    ``scratch/build_sbom_304.py`` (locally) / the Windows workflow's
    step 22b (on the release runner) from the release stack:
    ``packaging/windows/installer-requirements.txt`` + the step-7
    extras (torch, shap, pyinstaller, pytest).
    """

    SBOM = Path(__file__).resolve().parent.parent / "packaging" / "sbom" / "sbom.cdx.json"

    def test_artifact_exists_and_parses(self) -> None:
        assert self.SBOM.is_file(), "packaging/sbom/sbom.cdx.json is gone"
        bom = json.loads(self.SBOM.read_text(encoding="utf-8"))
        assert bom["bomFormat"] == "CycloneDX"
        assert bom["specVersion"] == "1.6"

    def test_root_component_is_the_product(self) -> None:
        bom = json.loads(self.SBOM.read_text(encoding="utf-8"))
        root = bom["metadata"]["component"]
        version = (
            Path(__file__).resolve().parent.parent / "VERSION"
        ).read_text().strip()
        assert root["name"] == "pharmaml"
        assert root["type"] == "application"
        assert root["version"] == version

    def test_release_stack_coverage(self) -> None:
        bom = json.loads(self.SBOM.read_text(encoding="utf-8"))
        components = bom["components"]
        names = {c["name"].lower() for c in components}
        # the pinned core + the scientific stack + the step-7 extras
        required = {
            "numpy", "pandas", "scikit-learn", "scipy", "joblib", "openpyxl",
            "tqdm", "rdkit", "mordred", "meeko", "gemmi", "xgboost",
            "lightgbm", "catboost", "matplotlib", "seaborn", "fpdf2",
            "pyside6", "fastapi", "uvicorn", "websockets", "torch", "shap",
            "pyinstaller", "pytest",
        }
        missing = sorted(required - names)
        assert not missing, f"SBOM lost release-stack components: {missing}"
        assert len(components) >= 40, (
            f"the release stack is 37 pinned + 4 extras; got {len(components)}"
        )

    def test_every_component_is_versioned(self) -> None:
        bom = json.loads(self.SBOM.read_text(encoding="utf-8"))
        names = [c["name"] for c in bom["components"]]
        # one entry per package: the builder's pyinstaller dedupe
        # (the -r file's range line superseded by the exact pin)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, f"duplicate SBOM components: {duplicates}"
        # exact pins carry their version; the two documented exceptions
        # are honest SPEC CARRIERS (a range pin and an unpinned tool
        # the workflow installs -- their version is resolved at build
        # time and recorded in RELEASE_MANIFEST.json, not by the SBOM):
        exceptions = {"pyinstaller-hooks-contrib", "pytest"}
        bare = sorted(
            c["name"] for c in bom["components"] if not c.get("version")
        )
        unexpected = sorted(set(bare) - exceptions)
        assert not unexpected, f"unversioned SBOM components: {unexpected}"
        # the exact-pinned tooling carries its version
        by_name = {c["name"]: c for c in bom["components"]}
        assert by_name["pyinstaller"]["version"] == "6.22.3"
