#!/usr/bin/env python
"""PharmaML Studio — packaging validation (part9 §16 steps 8-9).

Two modes, both failing with a nonzero exit code on any broken check:

``--static``
    Runs ON ANY OS (this is the mode the Linux-side release evidence
    runs): validates that everything the spec references actually
    exists in the source tree, that the bundled engines match their
    pinned SHA-256/PE-machine contract, that the version quadruple in
    ``version_info.txt`` agrees with the toolkit ``VERSION`` file,
    that the .iss/.ps1/.bat config files carry their load-bearing
    structure, and that the pinned stack in
    ``installer-requirements.txt`` is importable in the running venv.

``--dist <dir> [--frozen-smoke]``
    Runs on the Windows build machine against the frozen onedir
    output: the exe exists, the Qt platform plugins landed
    (``platforms/qwindows.dll``), the image-format plugins landed,
    the fonts and engines landed exactly where the app's own
    ``__file__``-relative resource math expects them, no test/dev
    packages leaked into the bundle, no secrets ride along, and
    (with ``--frozen-smoke``) the frozen ``PharmaMLStudio.exe
    --doctor`` exits 0.  Also prints the top-20 size report (part9
    §12: the size report explaining the largest components).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

SPEC_DIR = Path(__file__).resolve().parent
TOOLKIT = SPEC_DIR.parents[1]

# The pinned bundled-engine contract, mirrored from
# pharmaml/docking/workflow.py's own table (single source of truth is
# the workflow module; this table is validated against it in --static).
BUNDLED_ENGINE_SHA256 = {
    "vina.exe": "e0c4b2715e0c1a74f6e92d0f3be0328ac97542eafbc111e6b1efad897a73cce5",
    "autogrid4.exe": "797efce687d1ae82df59726461e0e1966b3d8edb0f8b187f982fa1ab0c12da9e",
    "autodock4.exe": "36c0b16c04d7df8e6225737bae65bb04058d1f4c90accfaf2d230dbc913954bf",
}
BUNDLED_ENGINE_MACHINE = {"vina.exe": 0x8664, "autogrid4.exe": 0x14C, "autodock4.exe": 0x14C}

SECRET_PATTERNS = (
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"ghs_[A-Za-z0-9]{20,}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def ok(self, label: str) -> None:
        self.checks += 1
        print(f"  PASS  {label}")

    def fail(self, label: str) -> None:
        self.checks += 1
        self.failures.append(label)
        print(f"  FAIL  {label}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pe_machine(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(64)
            if header[:2] != b"MZ":
                return None
            offset = int.from_bytes(header[0x3C:0x40], "little")
            handle.seek(offset)
            coff = handle.read(6)
            if coff[:4] != b"PE\x00\x00":
                return None
            return int.from_bytes(coff[4:6], "little")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# --static: source-tree / config validation (any OS)
# ---------------------------------------------------------------------------
def validate_static() -> int:
    print("== packaging validation -- static (source + config) ==")
    r = Report()

    # 1. the entry point and the app resources the spec references:
    for rel in ("studio.py", "LICENSE", "VERSION", "pyproject.toml"):
        p = TOOLKIT / rel
        if p.is_file():
            r.ok(f"source file {rel}")
        else:
            r.fail(f"source file {rel} missing")

    fonts = sorted((TOOLKIT / "fonts").glob("*.ttf"))
    if len(fonts) >= 4 and any("OFL" in f.name for f in (TOOLKIT / "fonts").iterdir()):
        r.ok(f"fonts bundle present ({len(fonts)} TTFs + the OFL notice)")
    else:
        r.fail("fonts bundle incomplete (expect the IBM Plex set + OFL.txt)")

    # 2. the bundled engines against the workflow.py contract:
    src = (TOOLKIT / "src" / "pharmaml" / "docking" / "workflow.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r"_BUNDLED_ENGINE_SHA256 = \{(.*?)\}", src, re.DOTALL)
    if m:
        table = dict(re.findall(r'"([a-z0-9.]+)": "([0-9a-f]{64})"', m.group(1)))
        if table == BUNDLED_ENGINE_SHA256:
            r.ok("engine SHA table mirrors pharmaml/docking/workflow.py")
        else:
            r.fail("engine SHA table DRIFTED from workflow.py (update this script)")
    else:
        r.fail("could not read workflow.py's engine SHA table")

    for name, expected in BUNDLED_ENGINE_SHA256.items():
        p = TOOLKIT / "engines" / name
        if not p.is_file():
            r.fail(f"bundled engine {name} missing")
            continue
        got = sha256_file(p)
        machine = pe_machine(p)
        if got != expected:
            r.fail(f"engine {name} sha mismatch ({got[:12]})")
        elif machine != BUNDLED_ENGINE_MACHINE[name]:
            r.fail(f"engine {name} PE machine {machine:#x} != pin")
        else:
            r.ok(f"engine {name} sha+PE pinned ({got[:12]}, {machine:#x})")

    # 3. version quadruple agreement (VERSION <-> version_info.txt):
    version = (TOOLKIT / "VERSION").read_text(encoding="utf-8").strip()
    vi = (SPEC_DIR / "version_info.txt").read_text(encoding="utf-8")
    # the FixedFileInfo quadruple as space-free text -- a tuple's repr
    # carries ", " separators, which can never match the space-stripped
    # haystack (the r272 fix; the old tuple-f-string needle always failed)
    quad_txt = "(" + version.replace(".", ",") + ",0)"
    if f"filevers={quad_txt}" in vi.replace(" ", "") and f"prodvers={quad_txt}" in vi.replace(" ", ""):
        r.ok(f"version_info.txt quadruple agrees with VERSION ({version})")
    else:
        r.fail(f"version_info.txt quadruple disagrees with VERSION ({version})")

    # 4. spec file structure (compiles + load-bearing members):
    spec = (SPEC_DIR / "pharmaml.spec").read_text(encoding="utf-8")
    try:
        compile(spec, "pharmaml.spec", "exec")
        r.ok("pharmaml.spec compiles as Python")
    except SyntaxError as exc:
        r.fail(f"pharmaml.spec syntax error: {exc}")
    for needle, label in (
        ("collect_submodules(\"mordred\")", "mordred dynamic-registry collection"),
        (', "fonts")', "the fonts datas target"),
        ('"pharmaml/engines"', "the engines datas target"),
        ("console=False", "the GUI (no-console) exe flag"),
        ("exclude_binaries=True", "onedir mode (COLLECT follows)"),
        ("upx=False", "UPX disabled for Qt/science DLLs"),
        ("strip=False", "strip disabled for scientific binaries"),
    ):
        if needle in spec:
            r.ok(f"spec: {label}")
        else:
            r.fail(f"spec missing: {label}")

    # 5. the Inno script's load-bearing structure:
    iss = (SPEC_DIR / "installer.iss").read_text(encoding="utf-8")
    for needle, label in (
        ("AppId={", "stable AppId"),
        ("LicenseFile=", "the MIT license page"),
        ("recursesubdirs", "recursive onedir payload"),
        ("SetupIconFile=", "the installer icon"),
        ("ArchitecturesAllowed=x64compatible", "x64-only install"),
        ("DelTree(DataRoot", "the explicit user-data removal question"),
    ):
        if needle in iss:
            r.ok(f"installer.iss: {label}")
        else:
            r.fail(f"installer.iss missing: {label}")

    # 6. the build scripts exist and fail loudly:
    ps1 = (SPEC_DIR / "build_windows.ps1").read_text(encoding="utf-8")
    if "$ErrorActionPreference = 'Stop'" in ps1 and "throw" in ps1:
        r.ok("build_windows.ps1 fails loudly (Stop + throw)")
    else:
        r.fail("build_windows.ps1 lost its fail-loud posture")

    # 7. the pinned stack importable in the RUNNING venv (the same pins
    #    installer-requirements.txt freezes — the strongest statement a
    #    non-Windows host can make about the stack):
    req = (SPEC_DIR / "installer-requirements.txt").read_text(encoding="utf-8")
    pins = [
        line.split("==")[0].strip()
        for line in req.splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    ]
    import importlib

    # distribution names differ from import names for the six classics
    # (r297: python-multipart renamed its module to python_multipart --
    # the dash form was never importable, which the runner's first real
    # validation run surfaced):
    DIST_TO_IMPORT = {
        "scikit-learn": "sklearn",
        "python-docx": "docx",
        "Pillow": "PIL",
        "qt-material": "qt_material",
        "fpdf2": "fpdf",
        "python-multipart": "python_multipart",
    }
    bad = []
    for dist in pins:
        if dist in {"pyinstaller", "pyinstaller-hooks-contrib"}:
            continue  # build tools, probed at build time on Windows
        top = DIST_TO_IMPORT.get(dist, dist)
        try:
            importlib.import_module(top)
        except Exception:  # noqa: BLE001 — the probe's design: ANY import
            # failure (missing, renamed, or broken-at-import) marks the pin
            # bad here; narrowing the catch would lie about broken pins.
            bad.append(dist)
    if bad:
        r.fail(f"pinned stack NOT importable here: {', '.join(bad)}")
    else:
        r.ok(f"all {len(pins)} pinned runtime packages import in this venv")

    print(f"== static: {r.checks - len(r.failures)}/{r.checks} checks passed")
    return 1 if r.failures else 0


# ---------------------------------------------------------------------------
# --dist: frozen-output validation (Windows build machine)
# ---------------------------------------------------------------------------
def validate_dist(dist: Path, frozen_smoke: bool) -> int:
    print(f"== packaging validation -- dist ({dist}) ==")
    if sys.platform != "win32":
        print("  NOTE  --dist is meaningful on the Windows build host; "
              "on this OS it only checks the tree layout.")
    r = Report()

    exe = dist / "PharmaMLStudio.exe"
    if exe.is_file():
        r.ok(f"the application exe ({exe.name})")
    else:
        r.fail("PharmaMLStudio.exe missing at the dist root")

    # the PyInstaller 6 onedir layout puts binaries under _internal:
    for internal in (dist / "_internal", dist):
        if internal.is_dir():
            break
    else:
        internal = dist

    qt_platform = internal / "PySide6" / "plugins" / "platforms" / "qwindows.dll"
    if qt_platform.is_file():
        r.ok("Qt platform plugin qwindows.dll present")
    else:
        r.fail(f"Qt platform plugin missing: {qt_platform}")

    img_formats = internal / "PySide6" / "plugins" / "imageformats"
    if img_formats.is_dir() and any(img_formats.glob("*.dll")):
        r.ok(f"Qt image-format plugins present ({len(list(img_formats.glob('*.dll')))} dlls)")
    else:
        r.fail("Qt image-format plugins missing")

    fonts = sorted(internal.glob("fonts/*.ttf"))
    if len(fonts) >= 4:
        r.ok(f"the bundled font set landed at _internal/fonts ({len(fonts)} TTFs)")
    else:
        r.fail("the bundled font set did not land at _internal/fonts")

    engines_dir = internal / "pharmaml" / "engines"
    for name, expected in BUNDLED_ENGINE_SHA256.items():
        p = engines_dir / name
        if not p.is_file():
            r.fail(f"bundled engine {name} not at _internal/pharmaml/engines")
            continue
        got = sha256_file(p)
        if got != expected:
            r.fail(f"frozen engine {name} sha drift ({got[:12]})")
        else:
            r.ok(f"frozen engine {name} sha-pinned ({got[:12]})")

    # dev/test leakage + secrets (part9 §12/§15):
    leaked = [
        d.name
        for d in internal.iterdir()
        if d.is_dir() and d.name.lower() in {"pytest", "_pytest", "tests", "mypy"}
    ]
    if leaked:
        r.fail(f"dev/test packages leaked into the bundle: {', '.join(leaked)}")
    else:
        r.ok("no dev/test packages in the bundle")

    secret_hits: list[str] = []
    for p in dist.rglob("*"):
        if p.is_file() and p.stat().st_size < 40 * 1024 * 1024 and p.suffix.lower() in {
            ".txt", ".md", ".json", ".py", ".pyc", ".cfg", ".ini", ".iss", ".ps1", ".bat", ".spec", ".qml", ".qss"
        }:
            try:
                blob = p.read_bytes()
            except OSError:
                continue
            for pat in SECRET_PATTERNS:
                if pat.search(blob):
                    secret_hits.append(str(p.relative_to(dist)))
    if secret_hits:
        r.fail(f"secret material found in: {', '.join(secret_hits[:5])}")
    else:
        r.ok("no token/key secret material in the bundle")

    # the size report (part9 §12):
    files = [p for p in dist.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    top = sorted(files, key=lambda p: -p.stat().st_size)[:20]
    print(f"  dist size: {total / (1024 * 1024):,.1f} MiB across {len(files)} files")
    print("  top-20 largest components:")
    for p in top:
        print(f"    {p.stat().st_size / (1024 * 1024):>8.1f} MiB  {p.relative_to(dist)}")

    if frozen_smoke and sys.platform == "win32":
        proc = subprocess.run(
            [str(exe), "--doctor"], capture_output=True, timeout=300, check=False
        )
        if proc.returncode == 0:
            r.ok("frozen --doctor smoke test exited 0")
        else:
            r.fail(f"frozen --doctor smoke test rc={proc.returncode}")

    print(f"== dist: {r.checks - len(r.failures)}/{r.checks} checks passed")
    return 1 if r.failures else 0


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] == "--static":
        return validate_static()
    if args[0] == "--dist":
        if len(args) < 2:
            print("usage: packaging_validation.py --dist <dir> [--frozen-smoke]")
            return 2
        return validate_dist(Path(args[1]), "--frozen-smoke" in args[2:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
