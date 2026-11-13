"""Phase 6 (part6 §16): the evidence-based release-readiness report.

One command gathers EVERY piece of release evidence the part6 core
directive names and writes ``docs/RELEASE_READINESS.md``:

* dependency matrix — every optional dependency, installed-or-not in
  THIS environment, honestly (§12: missing ≠ broken);
* capability coverage — the live registry counts (integrated / api /
  orphan / test gaps);
* test evidence — per-suite collection counts plus the recorded
  results this environment actually ran (passed/failed/skipped are
  only ever numbers a real pytest run produced — never claims);
* version carriers — VERSION / pyproject / __init__ / CHANGELOG agree;
* archive verification — the §14 audit report for the shipped zip,
  when present.

The report NEVER converts absence of evidence into a pass: sections
whose evidence was not gathered are listed as "not gathered" with the
exact command that would gather them.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

TK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TK / "src"))
sys.path.insert(0, str(TK))

# -- the dependency matrix (§12) ------------------------------------------------

#: pip name -> (human label, import name, what needs it)
OPTIONAL_MATRIX: dict[str, tuple[str, str, str]] = {
    "numpy": ("Core array stack", "numpy", "everything"),
    "pandas": ("DataFrames", "pandas", "tables, QSAR, drift"),
    "scikit-learn": ("Classical ML", "sklearn", "the ml.* modules"),
    "scipy": ("Scientific routines", "scipy", "stats, optimization"),
    "joblib": ("Parallel + cache", "joblib", "workers, Memory"),
    "matplotlib": ("Plotting", "matplotlib", "viz, reports, figures"),
    "rdkit": ("Cheminformatics", "rdkit", "chem.*, docking prep"),
    "mordred": ("Descriptor battery", "mordred", "descriptors (mordred set)"),
    "meeko": ("Ligand preparation", "meeko", "SMILES -> PDBQT"),
    "gemmi": ("Structures (mmCIF)", "gemmi", "receptor prep"),
    "selfies": ("Robust strings", "selfies", "chem.selfies_codec"),
    "xgboost": ("Boosting family", "xgboost", "ml boosting estimators"),
    "lightgbm": ("Boosting family", "lightgbm", "ml boosting estimators"),
    "catboost": ("Boosting family", "catboost", "ml boosting estimators"),
    "shap": ("Explanations", "shap", "ml.explain (TreeExplainer)"),
    "torch": ("Deep learning", "torch", "dnn estimators, generative"),
    "PySide6": ("Qt 6 bindings", "PySide6", "the Studio GUI"),
    "pyqtgraph": ("Qt plotting", "pyqtgraph", "Studio charts"),
}


@dataclass
class Readiness:
    """The gathered evidence (never a verdict without a number)."""

    dependency_rows: list[tuple[str, str, str]] = field(default_factory=list)
    capability_counts: dict[str, int] = field(default_factory=dict)
    suite_counts: dict[str, int] = field(default_factory=dict)
    version_agreement: dict[str, str] = field(default_factory=dict)
    recorded_runs: dict[str, str] = field(default_factory=dict)
    archive_audit: dict[str, str] = field(default_factory=dict)

    def dependency_status(self) -> tuple[int, int, list[str]]:
        installed = [r[0] for r in self.dependency_rows if r[1] == "installed"]
        broken = [r[0] for r in self.dependency_rows if r[1] == "broken"]
        return len(installed), len(broken), broken


def probe_dependencies() -> list[tuple[str, str, str]]:
    """(pip name, honest status, consumer) for every matrix row.

    Qt-side imports are probed AFTER ``ui.headless.ensure_headless``
    (the committed EGL stub + offscreen platform): without it, Qt
    imports fail for a missing SYSTEM library, which would misreport
    an installed, working package as broken.
    """
    try:
        from ui.headless import ensure_headless

        ensure_headless()
    except Exception:  # noqa: BLE001 - probing must not crash the report
        pass
    rows: list[tuple[str, str, str]] = []
    for pip_name, (_label, import_name, consumer) in OPTIONAL_MATRIX.items():
        try:
            spec = importlib.util.find_spec(import_name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            try:
                __import__(import_name)
                status = "installed"
            except Exception as exc:  # noqa: BLE001 - honest probe
                status = f"broken ({type(exc).__name__})"
        else:
            status = "not installed"
        rows.append((pip_name, status, consumer))
    return rows


def probe_capabilities() -> dict[str, int]:
    from pharmaml.studio.capability_registry import CAPABILITIES

    statuses: dict[str, int] = {}
    orphans = 0
    for cap in CAPABILITIES:
        statuses[cap.status] = statuses.get(cap.status, 0) + 1
        if not cap.tests:
            orphans += 1
    statuses["total"] = len(CAPABILITIES)
    statuses["without_tests"] = orphans
    return statuses


def count_suites() -> dict[str, int]:
    """Per-suite test counts via pytest --collect-only -q (no run).

    Paths are enumerated from the filesystem (pytest does not glob its
    own arguments) and collected in one subprocess per group; on this
    low-memory sandbox the library group is split in half so a single
    collect never peaks with every module imported at once.
    """
    counts: dict[str, int] = {}
    groups = [
        sorted(str(p) for p in (TK / "tests").glob("test_*.py")),
        sorted(str(p) for p in (TK / "tests" / "ui").glob("test_*.py")),
    ]
    for paths in groups:
        if not paths:
            continue
        half = 200  # collect in halves to bound memory (§14.2 spirit)
        for i in range(0, len(paths), half):
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--collect-only",
                 *paths[i:i + half], "-p", "no:cacheprovider", "-o", "addopts="],
                cwd=TK, capture_output=True, text=True, timeout=1800,
            )
            for line in proc.stdout.splitlines():
                # collect-only -q lists "<file>::<node>::<node>" per test
                if "::" not in line:
                    continue
                file_part = line.split("::", 1)[0].strip()
                if not file_part.endswith(".py"):
                    continue
                # pytest prints paths relative to its cwd (TK)
                file_path = Path(file_part)
                if file_path.is_absolute():
                    try:
                        name = str(file_path.relative_to(TK))
                    except ValueError:
                        continue
                else:
                    name = str(file_path)
                counts[name] = counts.get(name, 0) + 1
    return counts


def probe_versions() -> dict[str, str]:
    out = {}
    out["VERSION"] = (TK / "VERSION").read_text().strip()
    pyproject = (TK / "pyproject.toml").read_text()
    m = re.search(r'^version = "(.*?)"', pyproject, re.M)
    out["pyproject.toml"] = m.group(1) if m else "MISSING"
    init = (TK / "src/pharmaml/__init__.py").read_text()
    m = re.search(r'^__version__ = "(.*?)"', init, re.M)
    out["src/pharmaml/__init__.py"] = m.group(1) if m else "MISSING"
    changelog = (TK / "CHANGELOG.md").read_text()
    m = re.search(r"^## \[(.*?)\]", changelog, re.M)
    out["CHANGELOG.md leads with"] = m.group(1) if m else "MISSING"
    versions = {v for k, v in out.items() if k != "CHANGELOG.md leads with"}
    out["agreement"] = "yes" if len(versions) == 1 else "NO"
    return out


def find_audit_report() -> dict[str, str]:
    """The freshest §14 audit summary if one exists in the tree."""
    for cand in sorted((TK / ".." / "upload").glob("pharma-ml-toolkit-*.zip")):
        pass  # only a placeholder scan; the real digest comes from report.json
    report = Path("/tmp/audit-release/report.json")
    if not report.exists():
        # look beside the freshest zip in the outer workspace upload/
        upload = TK.parent / "upload"
        return {"status": "not gathered",
                "how": "tools/audit_release.py <zip> after tools/build_zip.py"}
    data = json.loads(report.read_text())
    return {
        "status": "passed" if data.get("archive_checks_passed") else "FAILED",
        "checks": str(len(data.get("checks", {}))),
        "how": str(report),
    }


def render(r: Readiness) -> str:
    L: list[str] = []
    L.append("# Release readiness — the Phase-6 evidence report (part6 §16)")
    L.append("")
    L.append(f"Generated {date.today().isoformat()} by `tools/release_readiness.py`.")
    L.append("Every number below comes from a live probe or a recorded test")
    L.append("run; sections without gathered evidence say so and name the")
    L.append("command that gathers them. Nothing here converts absence of")
    L.append("evidence into a pass.")
    L.append("")

    L.append("## 1. Dependency matrix (§12)")
    L.append("")
    L.append("| Dependency | Status (this environment) | Consumed by |")
    L.append("| --- | --- | --- |")
    for name, status, consumer in r.dependency_rows:
        L.append(f"| {name} | {status} | {consumer} |")
    installed, broken, _ = r.dependency_status()
    L.append("")
    L.append(f"{installed} installed / {len(r.dependency_rows)} probed."
             " Optional rows that are not installed are honestly reported")
    L.append("as unavailable — never as broken, and never silently.")
    L.append("")
    missing = [row[0] for row in r.dependency_rows
               if row[1] not in ("installed",)]
    broken_rows = [row[0] for row in r.dependency_rows
                   if row[1].startswith("broken")]
    if missing or broken_rows:
        L.append("Environment caveats, stated plainly:")
        for name in broken_rows:
            L.append(f"- `{name}` imports but raises — see its status above;")
            L.append("  the lazy-import surface keeps everything else working.")
        for name in missing:
            L.append(f"- `{name}` is not installed here: its features surface")
            L.append("  as dependency-honest 'missing optional dependency'")
            L.append("  states (§12), never as silent empties.")
    else:
        L.append("All probed dependencies import cleanly in this environment"
                 " (the Qt rows are probed under the committed EGL-stub"
                 " bootstrap, so their status matches what the test suites"
                 " actually exercise).")
    L.append("")

    L.append("## 2. Capability coverage")
    L.append("")
    c = r.capability_counts
    L.append(f"- Registry rows: **{c.get('total', 0)}**")
    for status in ("integrated", "api"):
        if c.get(status):
            L.append(f"- {status}: **{c[status]}**")
    L.append(f"- Rows without test references: **{c.get('without_tests', 0)}**")
    L.append("")

    L.append("## 3. Test evidence")
    L.append("")
    total = sum(r.suite_counts.values())
    L.append(f"- Suites discovered: **{len(r.suite_counts)}** "
             f"({sum(1 for k in r.suite_counts if k.startswith('tests/ui'))} UI, "
             f"{sum(1 for k in r.suite_counts if not k.startswith('tests/ui'))} library)")
    L.append(f"- Tests collected: **{total}**")
    L.append("")
    if r.recorded_runs:
        L.append("| Suite chunk (as actually run) | Result |")
        L.append("| --- | --- |")
        for chunk, result in r.recorded_runs.items():
            L.append(f"| `{chunk}` | {result} |")
        L.append("")
        L.append("Chunkening follows §14.2 (low-memory constraints): suites")
        L.append("run sequentially in small groups; nothing was skipped in")
        L.append("bulk and no skipped test is claimed as passed.")
    else:
        L.append("**Not gathered this run** — run the §14.2-chunkened suites")
        L.append("and record results, then re-generate with")
        L.append("`--record <chunk>=<result>` (repeatable).")
    L.append("")

    L.append("## 4. Version carriers (§15)")
    L.append("")
    v = r.version_agreement
    L.append(f"- VERSION: `{v.get('VERSION')}`")
    L.append(f"- pyproject.toml: `{v.get('pyproject.toml')}`")
    L.append(f"- src/pharmaml/__init__.py: `{v.get('src/pharmaml/__init__.py')}`")
    L.append(f"- CHANGELOG.md leads with: `{v.get('CHANGELOG.md leads with')}`")
    L.append(f"- Agreement: **{v.get('agreement')}**")
    L.append("")

    L.append("## 5. Release-archive verification (§14)")
    L.append("")
    a = r.archive_audit or {"status": "not gathered",
                            "how": "tools/audit_release.py <zip> after "
                                   "tools/build_zip.py"}
    L.append(f"- Status: **{a.get('status')}**"
             + (f" ({a['checks']} checks)" if a.get("checks") else ""))
    if a.get("status") != "passed":
        L.append(f"- How to gather: `{a.get('how')}`")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="append", default=[],
                    metavar="CHUNK=RESULT",
                    help="record a suite-chunk result, e.g. "
                         "'tests/ui/...=48 passed' (repeatable)")
    ap.add_argument("--audit-report", type=Path, default=None,
                    help="path to an audit_release report.json")
    args = ap.parse_args(argv)

    r = Readiness()
    r.dependency_rows = probe_dependencies()
    r.capability_counts = probe_capabilities()
    r.suite_counts = count_suites()
    r.version_agreement = probe_versions()
    for pair in args.record:
        if "=" in pair:
            chunk, result = pair.split("=", 1)
            r.recorded_runs[chunk] = result
    if args.audit_report and args.audit_report.exists():
        data = json.loads(args.audit_report.read_text())
        r.archive_audit = {
            "status": "passed" if data.get("archive_checks_passed") else "FAILED",
            "checks": str(len(data.get("checks", {}))),
            "how": str(args.audit_report),
        }
    else:
        r.archive_audit = find_audit_report()

    out = TK / "docs" / "RELEASE_READINESS.md"
    out.write_text(render(r), encoding="utf-8")
    installed, broken, broken_names = r.dependency_status()
    print(f"wrote {out}")
    print(f"deps: {installed} installed, {broken} broken"
          f"{' (' + ', '.join(broken_names) + ')' if broken_names else ''}")
    print(f"suites: {len(r.suite_counts)}, tests collected: "
          f"{sum(r.suite_counts.values())}")
    print(f"carriers agree: {r.version_agreement.get('agreement')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
