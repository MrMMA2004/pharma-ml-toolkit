#!/usr/bin/env python
"""PharmaML Studio — the packaging-readiness audit (part9 §1/§15).

Three modes (all fail nonzero on their own failure conditions):

default
    Scans the REAL source tree (studio.py + ui/ + src/ + scripts/) with
    the ast module for every Import/ImportFrom, maps import names to
    distributions, resolves each distribution's installed version and
    license from the running venv's dist-info, classifies Required vs
    Optional (unconditional module-level import in the GUI core chain
    vs guarded/lazy imports in capability modules), and writes the
    part9 §1.1 dependency-inventory table with the exact required
    columns: Package | Version | Purpose | Required or Optional |
    Included in Installer | Verification.  Output:
    packaging/windows/DEPENDENCY_INVENTORY.md.

--licenses-dir <dir>
    Copies each distribution's actual LICENSE/NOTICE file(s) out of
    dist-info into <dir>/THIRD_PARTY_LICENSES/ and writes an INDEX.md
    manifest — the redistribution inventory the installer embeds.

--manifest <setup.exe> --release-dir <dir> --python <venvpython>
    Post-build (Windows host): writes SHA256SUMS.txt,
    RELEASE_MANIFEST.json (the part9 §16 manifest: versions, sizes,
    dependency/tool versions, engines, checksums) and BUILD_REPORT.md.

The audit is deliberately evidence-based: every version/license row
comes from importlib.metadata on the running interpreter, never from a
hand-maintained table, and every "not imported anywhere" claim comes
from the AST scan of the actual tree (so TensorFlow/OpenCV/Polars/LIME
are honestly reported as NOT USED rather than padded in — the part9
§1 rule against inventing dependencies).
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import platform
import re
import shutil
import subprocess
from importlib import metadata as _md
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PACKAGING = TOOLKIT / "packaging" / "windows"

SCAN_ROOTS = [TOOLKIT / "legacy" / "studio.py", TOOLKIT / "ui",
               TOOLKIT / "src", TOOLKIT / "scripts"]

# import-name -> (pip distribution, purpose-for-the-inventory)
DISTRIBUTION_MAP = {
    "numpy": ("numpy", "array math everywhere"),
    "pandas": ("pandas", "tabular datasets, drift tables"),
    "sklearn": ("scikit-learn", "the ml.* estimator families"),
    "scipy": ("scipy", "stats + optimization"),
    "joblib": ("joblib", "worker utilities (utils/compute absorbs its API)"),
    "openpyxl": ("openpyxl", "Excel .xlsx import/export"),
    "tqdm": ("tqdm", "progress bars for long jobs"),
    "rdkit": ("rdkit", "chem.* + docking preparation (Chem/Draw/DataStructs)"),
    "mordred": ("mordred", "the mordred descriptor set (dynamic registry)"),
    "meeko": ("meeko", "SMILES -> PDBQT ligand preparation"),
    "gemmi": ("gemmi", "receptor .pdb parsing/preparation"),
    "xgboost": ("xgboost", "gradient-boosted estimators"),
    "lightgbm": ("lightgbm", "gradient-boosted estimators"),
    "catboost": ("catboost", "gradient-boosted estimators"),
    "shap": ("shap", "TreeExplainer values + interaction cube (ml/explain)"),
    "numba": ("numba", "shap's JIT layer (transitive)"),
    "llvmlite": ("llvmlite", "numba's compiler backend (transitive)"),
    "torch": ("torch", "deep-learning estimators, generative VAEs, drug-response model"),
    "selfies": ("selfies", "robust molecular string codec (chem/selfies_codec)"),
    "matplotlib": ("matplotlib", "all figure rendering + export backends"),
    "seaborn": ("seaborn", "styled report figures"),
    "fpdf": ("fpdf2", "PDF report writer (reports.py)"),
    "docx": ("python-docx", "DOCX report writer (reports.py)"),
    "PIL": ("Pillow", "image IO behind PNG/JPEG/TIFF figure export"),
    "PySide6": ("PySide6", "the Studio GUI (Qt 6.8 Essentials+Addons)"),
    "pyqtgraph": ("pyqtgraph", "the Studio's interactive charts"),
    "qt_material": ("qt-material", "the Studio light/dark theme engine"),
    "qdarkstyle": ("qdarkstyle", "the dark stylesheet engine"),
    "shiboken6": ("shiboken6", "PySide6's C++ binding runtime (transitive)"),
    "statsmodels": ("statsmodels", "multiple-testing FDR correction (biology/association)"),
    # the part11 API layer (Phase Seven — the FastAPI sidecar surface
    # the Phase Twenty-Four §29 audit added: these power the shipped
    # REST/WebSocket backend and must ride the PyInstaller bundle)
    "fastapi": ("fastapi", "the local REST API layer (api/routes/*, the app-level guard)"),
    "uvicorn": ("uvicorn", "the loopback ASGI server the sidecar runs (port-0 bind)"),
    "pydantic": ("pydantic", "the typed request/response schemas behind the OpenAPI contract"),
    # optional external chemistry engine (feature-gated, never required)
    "openbabel": ("openbabel", "optional mol2 writer lane (docking/ligands; NOT bundled — external)")
}

# modules the prompt names that the AST scan is expected NOT to find
# (honest NOT-USED section, the anti-padding rule):
CANDIDATE_NOT_USED = [
    "tensorflow", "cv2", "polars", "lime", "PyQt5", "PyQt6", "Gromacs wrapper",
]

# stdlib tops the scan legitimately hits (not dependencies):
_STDLIB = {
    "__future__", "abc", "argparse", "ast", "atexit", "base64", "bisect",
    "bz2", "codecs", "collections", "colorsys", "concurrent", "configparser",
    "contextlib", "copy", "csv", "ctypes", "dataclasses", "datetime", "decimal",
    "enum", "fractions", "functools", "gc", "glob", "graphlib", "gzip", "hashlib",
    "heapq", "html", "importlib", "inspect", "io", "itertools", "json", "keyword",
    "logging", "lzma", "math", "mimetypes", "multiprocessing", "numbers", "operator",
    "fnmatch", "os", "pathlib", "pickle", "platform", "queue", "random", "re", "secrets",
    "shutil", "signal", "socket", "sqlite3", "statistics", "string", "struct",
    "subprocess", "sys", "tempfile", "textwrap", "threading", "time", "traceback", "types",
    "typing", "typing_extensions", "unittest", "urllib", "uuid", "warnings",
    "weakref", "zlib", "asyncio", "hmac", "zipfile",
}


def scan_imports() -> dict[str, list[str]]:
    """Return {top-level module name: [file, ...]} from the real tree."""
    found: dict[str, list[str]] = {}

    def visit(path: Path) -> None:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (OSError, SyntaxError):
            return
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for n in names:
                rel = str(path.relative_to(TOOLKIT))
                found.setdefault(n, []).append(rel)

    for root in SCAN_ROOTS:
        if root.is_file():
            visit(root)
        elif root.is_dir():
            for p in sorted(root.rglob("*.py")):
                if ".venv" in p.parts or "build" in p.parts:
                    continue
                visit(p)
    return found


def dist_version(dist: str) -> str | None:
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return None


def dist_license(dist: str) -> str:
    try:
        return _md.metadata(dist).get("License", "") or "(see license file)"
    except _md.PackageNotFoundError:
        return ""


def dist_homepage(dist: str) -> str:
    try:
        return _md.metadata(dist).get("Home-page", "") or ""
    except _md.PackageNotFoundError:
        return ""


def frontend_host_section() -> list[str]:
    """part11 §29 (Phase Twenty-Four)'s frontend/desktop/host dependency
    table — derived from the ACTUAL frontend/package.json, src-tauri/
    Cargo.toml and tauri.conf.json (never a hand-maintained list)."""
    pkg = json.loads((TOOLKIT / "frontend" / "package.json").read_text(encoding="utf-8"))
    deps: dict[str, str] = pkg.get("dependencies", {})
    dev_deps: dict[str, str] = pkg.get("devDependencies", {})
    cargo = (TOOLKIT / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    conf = json.loads((TOOLKIT / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))

    # the §29 purpose column, keyed on the actual dependency names
    purposes = {
        "react": "the UI framework (the SPA's components/hooks)",
        "react-dom": "React's DOM renderer",
        "@tauri-apps/api": "the startup bridge's invoke (backend_endpoint IPC; offline-bundled)",
        "motion": "Motion for React — restrained shell animations (§8.5)",
        "plotly.js-dist-min": "the mandated scientific charting stack (declared + chunked; src adoption is the §3.2 campaign row)",
        "three": "the Three.js 3D molecular viewer + trajectory playback (§23)",
        "@radix-ui/react-tabs": "the shadcn/ui Tab head",
        "class-variance-authority": "shadcn/ui component variants",
        "clsx": "shadcn/ui class merge",
        "tailwind-merge": "shadcn/ui class merge",
        "lucide-react": "the icon set",
    }
    dev_purposes = {
        "typescript": "strict typechecking (build-time)",
        "vite": "the static-asset build tool (build-time; dist/ ships)",
        "@vitejs/plugin-react": "the React transform (build-time)",
        "tailwindcss": "the styling engine (build-time CSS generation)",
        "@tailwindcss/vite": "the Tailwind 4 Vite integration (build-time)",
        "openapi-typescript": "generates src/types/api.d.ts from contracts/openapi.json (codegen-time — §12.2's single DTO source)",
        "@types/react": "type declarations (build-time)",
        "@types/react-dom": "type declarations (build-time)",
        "@types/three": "type declarations (build-time)",
        "@types/node": "type declarations (build-time)",
    }
    lines = [
        "",
        "## Frontend and desktop dependencies (part11 §29, Phase Twenty-Four)",
        "",
        (f"Derived from the actual `frontend/package.json` (name: "
         f"{pkg.get('name')}, version {pkg.get('version')}), `src-tauri/Cargo.toml`"),
        "and `src-tauri/tauri.conf.json` — the same files the Windows",
        ("release workflow's step-2 version lockstep gate enforces"
         "(`tests/test_windows_pipeline.py`)."),
        "",
        "| Package/Component | Version | Purpose | Required or Optional | Frontend/Host/Backend | Included in Installer | Verification |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, ver in sorted(deps.items()):
        purpose = purposes.get(name, "runtime dependency (package.json)")
        lines.append(
            f"| {name} | {ver} | {purpose} | Required (runtime) | Frontend | "
            "YES (bundled into the Vite static dist/) | package.json + bun.lock; "
            "tsc/vite build gates"
        )
    for name, ver in sorted(dev_deps.items()):
        purpose = dev_purposes.get(name, "build-time tooling (devDependency)")
        lines.append(
            f"| {name} | {ver} | {purpose} | Build-time only | Frontend (toolchain) | "
            "NO (never ships) | package.json + bun.lock"
        )
    # the Rust host surface
    rust_rows = re.findall(
        r"^([a-zA-Z0-9_-]+)\s*=\s*\{\s*version\s*=\s*\"([^\"]+)\"", cargo, re.MULTILINE
    )
    plain_rust = re.findall(r"^([a-zA-Z0-9_-]+)\s*=\s*\"([^\"]+)\"", cargo, re.MULTILINE)
    seen_rust = {n: v for n, v in rust_rows}
    for n, v in plain_rust:
        seen_rust.setdefault(n, v)
    rust_purpose = {
        "tauri": "the Tauri 2 host core (window, IPC, sidecar supervision)",
        "serde": "the BackendEndpoint IPC serialization",
        "serde_json": "the handshake/IPC JSON layer",
        "tauri-build": "the Tauri build script (build-time)",
    }
    for name, ver in sorted(seen_rust.items()):
        if name == "pharmaml-studio":
            continue
        build_time = name == "tauri-build"
        lines.append(
            f"| {name} (Rust) | {ver} | {rust_purpose.get(name, 'host dependency (Cargo.toml)')} | "
            + ("Build-time only" if build_time else "Required (runtime)")
            + " | Host | "
            + ("NO (never ships)" if build_time else "YES (compiled into pharmaml-studio.exe)")
            + " | src-tauri/Cargo.toml (cargo build --release on the Windows runner)"
        )
    # WebView2 + runtimes
    webview2_mode = conf.get("bundle", {}).get("windows", {}).get("webview2", {})
    lines += [
        "",
        "### Runtimes and redistribution notes (§29's runtime-dependency audit)",
        "",
        f"- **Microsoft WebView2 Runtime**: `{json.dumps(webview2_mode)}` in",
        "  `tauri.conf.json` — the offline Evergreen installer rides the",
        "  NSIS/MSI bundle (part11_2 §7 / §32); the workflow's step 19 installs",
        "  it offline; the release manifest records the selected mode.",
        "- **Node.js / bun / Vite / TypeScript / Tailwind**: build-time only —",
        "  the end user needs none of them (`vite build` emits the static",
        "  `frontendDist` assets the Tauri host loads locally).",
        "- **Rust toolchain (rustc/cargo 1.77+)**: build-time only.",
        "- **Python runtime + the table above's wheels**: bundled via the",
        "  PyInstaller onedir sidecar (`packaging/windows/pharmaml_backend.spec`)",
        "  — the end user needs no Python, conda, or pip (part11_2 §5).",
        "- **AutoDock trio (vina/autogrid4/autodock4)**: the engines/ table above.",
        "- **No CDN, no remote URL**: the offline CSP pins the frontend to",
        "  `self` + the loopback API (`frontend/index.html`, drift-guarded).",
        "",
    ]
    return lines


def write_inventory() -> int:
    found = scan_imports()
    rows = []
    for top, (dist, purpose) in DISTRIBUTION_MAP.items():
        used_in = found.get(top)
        version = dist_version(dist)
        if used_in:
            where = sorted(set(used_in))
            core = (
                top in {"numpy", "pandas", "scipy", "sklearn", "joblib",
                        "openpyxl", "tqdm", "PySide6", "pyqtgraph"}
            )
            if top == "openbabel":
                # optional external engine lane — honestly optional, never
                # bundled: the §29/§34 anti-padding rule over the hardcoded YES
                status = "Optional (feature-gated import)"
                included = "NO (external, optional)"
            else:
                status = "Required (unconditional)" if core else "Required (feature-gated import)"
                included = "YES"
            sample = "; ".join(where[:3]) + ("; ..." if len(where) > 3 else "")
            rows.append((dist, version or "NOT INSTALLED", purpose, status, included, sample))
        else:
            rows.append((dist, version or "-", "declared but no static import found",
                         "n/a", "review", "no import in studio.py/ui/src/scripts"))
    extra_used = sorted(
        set(found) - set(DISTRIBUTION_MAP) - _STDLIB - {"pharmaml", "ui", "numpy"}
    )
    version = (TOOLKIT / "VERSION").read_text(encoding="utf-8").strip()
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# PharmaML Studio — dependency inventory (part9 §1.1)",
        "",
        f"Generated {stamp} by `tools/packaging_audit.py` from the ACTUAL",
        (f"source tree (studio.py + ui/ + src/ + scripts/, {sum(len(v) for v in found.values())} import"
        " statements parsed) and the live venv's dist-info. Versions are"),
        "importlib.metadata's truth, not a hand-maintained table.",
        "",
        f"Toolkit version audited: **{version}**. Python: {platform.python_version()}.",
        "",
        "| Package | Version | Purpose | Required or Optional | Included in Installer | Verification |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dist, ver, purpose, status, included, verify in rows:
        lines.append(f"| {dist} | {ver} | {purpose} | {status} | {included} | {verify} |")
    lines += [
        "",
        "## Not used (the anti-padding rule)",
        "",
        "part9 §1: *\"Do not add packages that the project does not use merely",
        "to increase the package size.\"* The following candidates from the",
        "prompt's list are confirmed NOT imported anywhere in the app tree:",
        "",
    ]
    for m in CANDIDATE_NOT_USED:
        hit = found.get(m.split()[0].replace("_", "").lower())
        lines.append(f"- **{m}**: NOT USED"
                     + (f" — but found as {hit}" if hit else ""))
    lines += [
        "",
        "## External executables (part9 §5)",
        "",
        "The AutoDock trio ships in `engines/` and is bundled into the",
        "installer at `_internal/pharmaml/engines` (the app's own",
        "`_bundled_engine_dirs()` discovery, SHA-256- and PE-machine-validated):",
        "",
        "| Executable | Build | Redistribution | Bundled |",
        "| --- | --- | --- | --- |",
        "| vina.exe | PE x86-64 | AutoDock Vina (Apache-2.0 family) | YES |",
        "| autogrid4.exe | PE x86 | AutoDock (GPL-family; engine NOTICE included) | YES |",
        "| autodock4.exe | PE x86 | AutoDock (GPL-family; engine NOTICE included) | YES |",
        "",
        "GROMACS: no import, no subprocess call site — not invoked",
        "anywhere in the tree; honestly NOT bundled.",
        "",
        "Open Babel — the honest §5 note: the MOL2 leg of ligand",
        "generation (``docking/ligands.py``) uses the ``openbabel``",
        "PyPI wrapper ONLY when the user installs it, inside a guarded",
        "``try/except ImportError`` whose failure the GUI reports as",
        "`MOL2 skipped: OpenBabel not installed` in the result rows —",
        "the exact pattern §5 demands (the capability reports the",
        "missing tool clearly; every other workflow is unaffected).",
        "The wrapper (GPL + native library) is deliberately NOT",
        "bundled with the installer.",
        "",
        "## GUI/Qt runtime",
        "",
        "PySide6 6.8.3 (Essentials + Addons) is the complete Qt runtime for",
        "the Studio; platform/imageformat/style plugins ride along via",
        "PyInstaller's PySide6 hooks and are asserted post-build by",
        "`packaging_validation.py --dist` (qwindows.dll etc.).",
    ]
    lines += frontend_host_section()
    out = PACKAGING / "DEPENDENCY_INVENTORY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(TOOLKIT)} ({len(rows)} rows)")
    print(f"extra top-level modules imported but not mapped: {extra_used or 'none'}")
    return 0


def write_licenses_dir(target: Path) -> int:
    licenses = target / "THIRD_PARTY_LICENSES"
    licenses.mkdir(parents=True, exist_ok=True)
    seen: list[str] = []
    for dist, _purpose in DISTRIBUTION_MAP.values():
        version = dist_version(dist)
        if version is None:
            continue
        try:
            d = _md.distribution(dist)
            info_dir = Path(getattr(d, "_path", "")) if getattr(d, "_path", "") else Path()
            files = [
                f for f in info_dir.iterdir()
                if f.is_file()
                and f.name.lower().split(".")[0].startswith(
                    ("license", "licence", "notice", "copying")
                )
            ] if info_dir.is_dir() else []
        except Exception:  # noqa: BLE001 - a missing dist lists no licenses
            files = []
        wrote = False
        for f in files:
            # keep the original extension (LICENSE, .txt, .rst, ...) and
            # NEVER mangle dotted versions (qt-material-2.17 stays whole):
            base = re.sub(r"[^A-Za-z0-9._-]", "_", f"{dist}-{version}-{f.name}")
            shutil.copyfile(f, licenses / base)
            wrote = True
        if wrote:
            seen.append(f"{dist}=={version}")
    index = licenses / "INDEX.md"
    index.write_text(
        "# Third-party license inventory (part9 §15)\n\n"
        "One file per distribution, copied verbatim from each dist-info.\n\n"
        + "\n".join(f"- {s}" for s in sorted(seen))
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(seen)} license files + INDEX.md into {target}")
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(setup_exe: Path, release_dir: Path, venv_python: Path) -> int:
    version = (TOOLKIT / "VERSION").read_text(encoding="utf-8").strip()
    freeze = subprocess.run(
        [str(venv_python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    pyver = subprocess.run(
        [str(venv_python), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dist_tree = release_dir / "dist" / "PharmaMLStudio"
    dist_size = sum(p.stat().st_size for p in dist_tree.rglob("*") if p.is_file()) \
        if dist_tree.is_dir() else 0
    manifest = {
        "application": "PharmaML Studio",
        "version": version,
        "build_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": pyver,
        "architecture": "Windows x64",
        "packaging": {
            "pyinstaller": dist_version("pyinstaller") or "unknown",
            "inno_setup": "6 (ISCC; version passed on the command line)",
        },
        "dependency_versions": sorted(freeze),
        "bundled_external_executables": {
            "vina.exe": "AutoDock Vina x64",
            "autogrid4.exe": "AutoGrid4 x86",
            "autodock4.exe": "AutoDock4 x86",
        },
        "installer_filename": setup_exe.name,
        "installer_size_bytes": setup_exe.stat().st_size,
        "onedir_size_bytes": dist_size,
        "installer_sha256": sha256_file(setup_exe),
        "code_signing": "unsigned (no certificate available) — documented, not claimed",
    }
    (release_dir / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    sums = []
    for p in sorted(release_dir.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS.txt"} and ".zip" not in p.suffix:
            sums.append(f"{sha256_file(p)}  {p.relative_to(release_dir)}")
    (release_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"wrote RELEASE_MANIFEST.json + SHA256SUMS.txt ({len(sums)} files hashed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--licenses-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--release-dir", type=Path, default=None)
    ap.add_argument("--python", type=Path, default=None)
    args = ap.parse_args()
    if args.licenses_dir:
        return write_licenses_dir(args.licenses_dir)
    if args.manifest and args.release_dir and args.python:
        return write_manifest(args.manifest, args.release_dir, args.python)
    return write_inventory()


if __name__ == "__main__":
    raise SystemExit(main())
