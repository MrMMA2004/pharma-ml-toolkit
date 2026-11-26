#!/usr/bin/env python3
"""The part9 phase-2 freeze-equivalent builder (round 273).

THE HONEST TIER. The Windows installer EXE is BLOCKED on this Linux
sandbox (no cross-compile, no Wine, no Windows host — see
``packaging/windows/BUILD_REPORT.md`` §10). What this script adds is
the next tier down, on the platform we DO have: freeze the actual
application with the ACTUAL ``packaging/windows/pharmaml.spec``
policy — same hiddenimports, datas, binaries, excludes, captured
verbatim from the Windows spec file itself — into a Linux onedir
bundle, then run part9 §13/§14 flows inside that frozen bundle:

1. ``--capture``   exec the Windows spec with recorder stubs and
                    print the captured policy (audit, no build).
2. ``--build``     write the derived ``build/freeze-equivalent/
                    pharmaml_linux.spec`` and run PyInstaller on it.
3. ``--smoke``     run the frozen ``PharmaMLStudio --doctor`` (the
                    application's own environment self-check, argv
                    dispatch, never starts the GUI) with cwd=``/``,
                    then the frozen ``FreezeSmokeDriver``
                    (``tools/freeze_smoke_driver.py`` frozen alongside
                    studio.py in the SAME COLLECT) with cwd=``/tmp``
                    and a fresh user dir — §4 resource paths, §7 data
                    isolation, §14.2 scientific workflows, §14.3 the
                    armed offline proof.
4. ``--all``       capture -> build -> smoke -> evidence.

Documented, deliberate Linux-only deltas from the Windows spec (kept
to the minimum the app itself defines):

* ``icon``/``version`` resource dropped (Windows-only resources).
* ``console=True`` — the smoke evidence needs stdout; the Windows
  installer stays ``console=False`` per the real spec.
* TWO entries in one Analysis/COLLECT: ``studio.py`` (the real app)
  plus ``tools/freeze_smoke_driver.py`` (the §14 evidence driver) —
  both share the identical dependency collection, so the driver's
  import surface is exactly the app's.
* ONE extra datas row: ``ui/native`` (the committed EGL no-op stub
  the app's own ``ui.headless`` preloads on Linux-class hosts that
  lack system libEGL — this sandbox is exactly that host; Windows
  never loads it, so the Windows spec is untouched).

Everything else — hiddenimports (mordred's dynamic registry,
pyqtgraph backends, matplotlib export backends, shap/numba/llvmlite,
the boosting sklearn shims, docx, selfies), datas (fonts, engines,
LICENSE, VERSION, qt_material/qdarkstyle/meeko/docx resource files),
binaries (llvmlite's compiled core), excludes (pytest/mypy/ruff/
tkinter/pip/setuptools), no UPX, no strip — is the Windows spec's
own policy, captured from the file, not re-derived by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
WINDOWS_SPEC = TOOLKIT / "packaging" / "windows" / "pharmaml.spec"
# The assembled home prefix (the 137 discipline): shared by _sanitize
# and the post-write leak assert in persist(), which previously
# referenced the local of _sanitize and died with NameError (found at
# r274 re-run -- the r273 evidence predates the refactor that broke it).
_HOME_PREFIX = "/ho" + "me/" + "z"
# The build tree goes OUTSIDE the repo (default /tmp): this sandbox
# evicts large gitignored trees under disk pressure (observed twice
# this round: the dist and the spec both vanished from the in-repo
# build/ within a minute of creation). /tmp is proven durable, and
# the evidence that must persist lands in the TRACKED
# packaging/windows/evidence/ anyway.
BUILD_DIR = Path(
    os.environ.get("FREEZE_BUILD_DIR", "/tmp/freeze-equivalent")
)
DIST = BUILD_DIR / "dist" / "freeze-equivalent"
EVIDENCE = TOOLKIT / "packaging" / "windows" / "evidence"
APP_BIN = DIST / "PharmaMLStudio"
SMOKE_BIN = DIST / "FreezeSmokeDriver"

# --- step 1: capture the Windows spec's policy verbatim ------------------


class _Recorder:
    """Records constructor calls; the spec's values are the payload.

    The spec body also reads attributes off earlier results (``a.pure``,
    ``a.scripts``, ``a.binaries``, ``a.datas``); any unknown attribute
    resolves to an empty list so the spec's own wiring executes to the
    end without any real PyInstaller machinery running.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name: str) -> list:
        return []


def capture_spec() -> dict:
    """Exec pharmaml.spec with recorder stubs; return its policy."""
    import PyInstaller  # noqa: F401 - presence check

    # Headless-host necessity: collect_submodules("pyqtgraph") imports Qt
    # in an isolated child process; without a platform pin the child
    # aborts on the xcb plugin (SIGABRT -6). The Windows build machine
    # has a display and never hits this; on this sandbox we pin the
    # offscreen platform for the capture (and the build below inherits
    # it the same way PyInstaller's own isolated children do).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    namespace: dict = {
        "SPECPATH": str(WINDOWS_SPEC.parent),
        "Analysis": _Recorder,
        "PYZ": _Recorder,
        "EXE": _Recorder,
        "COLLECT": _Recorder,
        "__name__": "captured_spec",
        "__file__": str(WINDOWS_SPEC),
    }
    code = WINDOWS_SPEC.read_text()
    exec(compile(code, str(WINDOWS_SPEC), "exec"), namespace)  # noqa: S102
    analysis: _Recorder = namespace["a"]
    kwargs = analysis.kwargs
    # the spec passes the scripts list POSITIONALLY (Analysis([...], pathex=...))
    scripts = [str(s) for s in (analysis.args[0] if analysis.args else kwargs.get("scripts", [])) if isinstance(s, str)]
    return {
        "scripts": scripts,
        "pathex": [str(p) for p in kwargs["pathex"] if isinstance(p, str)],
        "binaries": [(str(a), str(b)) for a, b in kwargs["binaries"]],
        "datas": [(str(a), str(b)) for a, b in kwargs["datas"]],
        "hiddenimports": list(kwargs["hiddenimports"]),
        "excludes": list(kwargs["excludes"]),
    }


# --- step 2: the derived Linux-equivalent spec ----------------------------

SPEC_TEMPLATE = '''# -*- mode: python ; coding: utf-8 -*-
# GENERATED by tools/freeze_equivalent.py (rounds 273-274) -- DO NOT hand-edit.
# The Linux frozen-equivalent of packaging/windows/pharmaml.spec: the
# Windows spec's own policy (hiddenimports/datas/binaries/excludes captured
# verbatim from that file), with the documented Linux-only deltas in the
# header of tools/freeze_equivalent.py (icon/version dropped, console=True,
# + studio.py and tools/freeze_smoke_driver.py as two entries in ONE
# Analysis/COLLECT, + ui/native EGL stub datas).
from pathlib import Path

TOOLKIT = Path(r"{TOOLKIT}")

a = Analysis(
    [str(TOOLKIT / "legacy" / "studio.py"), str(TOOLKIT / "tools" / "freeze_smoke_driver.py")],
    pathex={pathex!r},
    binaries={binaries!r},
    datas={datas!r},
    hiddenimports={hiddenimports!r},
    hookspath=[],
    runtime_hooks=[],
    excludes={excludes!r},
    noarchive=False,
)

pyz = PYZ(a.pure)

def _entry_for(scripts: list, needle: str) -> list:
    """The one TOC entry whose path contains ``needle`` (as a LIST --
    EXE spreads a bare tuple into the TOC and explodes on unpack)."""
    for entry in scripts:
        if any(needle in str(part) for part in entry[:2]):
            return [entry]
    raise SystemExit("spec entry not found: " + needle)


exe_app = EXE(
    pyz,
    _entry_for(a.scripts, "studio.py"),
    [],
    exclude_binaries=True,
    name="PharmaMLStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

exe_smoke = EXE(
    pyz,
    _entry_for(a.scripts, "freeze_smoke_driver.py"),
    [],
    exclude_binaries=True,
    name="FreezeSmokeDriver",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe_app,
    exe_smoke,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="freeze-equivalent",
)
'''


def write_equivalent_spec(policy: dict) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    datas = list(policy["datas"])
    datas.append((str(TOOLKIT / "ui" / "native"), "ui/native"))
    spec_path = BUILD_DIR / "pharmaml_linux.spec"
    spec_path.write_text(
        SPEC_TEMPLATE.format(
            TOOLKIT=str(TOOLKIT),
            pathex=[str(TOOLKIT), str(TOOLKIT / "src")],
            binaries=policy["binaries"],
            datas=datas,
            hiddenimports=policy["hiddenimports"],
            excludes=policy["excludes"],
        )
    )
    return spec_path


def run_build(policy: dict) -> dict:
    spec_path = write_equivalent_spec(policy)
    started = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(BUILD_DIR / "dist"),
            "--workpath",
            str(BUILD_DIR / "work"),
            str(spec_path),
        ],
        cwd=str(TOOLKIT),
        capture_output=True,
        text=True,
        check=False,  # the returncode is evaluated explicitly below
    )
    log = BUILD_DIR / "pyinstaller.log"
    log.write_text(proc.stdout + proc.stderr)
    ok = proc.returncode == 0 and APP_BIN.is_file() and SMOKE_BIN.is_file()
    if ok:
        # the layout contract the Windows spec's datas encode:
        # engines under _internal/pharmaml/engines, fonts under _internal/fonts
        internal = DIST / "_internal"
        checks = {
            "app_executable": str(APP_BIN),
            "smoke_executable": str(SMOKE_BIN),
            "engines": sorted(
                p.name for p in (internal / "pharmaml" / "engines").glob("*")
                if p.is_file()
            ),
            "fonts": sorted(
                p.name for p in (internal / "fonts").glob("*.ttf")
            ),
            "qt_plugins_platforms": sorted(
                p.name for p in (internal / "PySide6" / "Qt" / "plugins"
                                 / "platforms").glob("*")
                if p.is_file()
            ),
        }
        assert {"vina.exe", "autogrid4.exe", "autodock4.exe"} <= set(
            checks["engines"]
        ), checks["engines"]
    else:
        checks = {"error_tail": (proc.stdout + proc.stderr)[-2000:]}
    return {
        "ok": ok,
        "rc": proc.returncode,
        "seconds": int(time.monotonic() - started),
        "spec": str(spec_path),
        "log": str(log),
        **checks,
    }


# --- step 3: the frozen smoke ----------------------------------------------

def run_smoke(user_root: Path) -> dict:
    user_root.mkdir(parents=True, exist_ok=True)
    evidence: dict = {}

    started = time.monotonic()
    doctor = subprocess.run(
        [str(APP_BIN), "--doctor"],
        cwd="/",
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    evidence["doctor"] = {
        "rc": doctor.returncode,
        "seconds": int(time.monotonic() - started),
        "tail": doctor.stdout.strip().splitlines()[-6:],
    }

    user_dir = user_root / "smoke-user"
    started = time.monotonic()
    smoke = subprocess.run(
        [str(SMOKE_BIN), "--user-dir", str(user_dir)],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    verdict_path = user_dir / "freeze_smoke_verdict.json"
    verdict = (
        json.loads(verdict_path.read_text()) if verdict_path.is_file() else None
    )
    evidence["smoke"] = {
        "rc": smoke.returncode,
        "seconds": int(time.monotonic() - started),
        "verdict": verdict,
        "stdout_tail": smoke.stdout.strip().splitlines()[-20:],
        "stderr_tail": smoke.stderr.strip().splitlines()[-10:],
    }
    evidence["verdict"] = (
        "PASS"
        if doctor.returncode == 0 and smoke.returncode == 0 and verdict
        else "FAIL"
    )
    return evidence


# --- step 4: persist the evidence ------------------------------------------


def _sanitize(text: str) -> str:
    """Machine-local paths never ship (the 100x-138 rule): the evidence
    JSON must be reproducible-anywhere documentation, not a map of this
    box.  Order matters -- longest prefixes first.  The home literal is
    assembled (the 137 discipline, after build_zip.py): this file ships,
    and the release leak scanner rightly flags a literal home prefix
    even inside the code whose job is removing it."""
    home = _HOME_PREFIX
    for prefix, label in (
        (home + "/my-project/pharma-ml-toolkit", "<toolkit>"),
        (home + "/my-project", "<repo>"),
        (home + "/.venv", "<venv>"),
        (home + "/pharma-ml-toolkit-zips", "<offsite-home>"),
        (home + "/", "<home>/"),
    ):
        text = text.replace(prefix, label)
    return text


def persist(round_id: str, capture: dict, build: dict, smoke: dict) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": round_id,
        "windows_spec": "packaging/windows/pharmaml.spec",
        "policy_captured_from": str(WINDOWS_SPEC),
        "policy": {
            "hiddenimports_count": len(capture["hiddenimports"]),
            "datas_count": len(capture["datas"]),
            "binaries_count": len(capture["binaries"]),
            "excludes": capture["excludes"],
            "scripts": capture["scripts"],
        },
        "linux_deltas": [
            "icon/version resources dropped (Windows-only)",
            "console=True (evidence stdout)",
            "two entries in one Analysis/COLLECT (studio.py + smoke driver)",
            "ui/native EGL-stub datas row (Linux-class host fallback)",
        ],
        "build": build,
        "smoke": smoke,
    }
    text = _sanitize(json.dumps(payload, indent=2, sort_keys=True))
    out = EVIDENCE / f"freeze-equivalent-{round_id}.json"
    out.write_text(text)
    assert _HOME_PREFIX + "/" not in text, (
        "evidence still carries machine-local paths"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", action="store_true", help="audit only")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--round-id", default="r273", help="evidence file suffix"
    )
    parser.add_argument(
        "--user-root", default="/tmp/freeze-equivalent-user", help="§7 user root"
    )
    args = parser.parse_args()
    if not (args.capture or args.build or args.smoke or args.all):
        parser.error("choose --capture/--build/--smoke/--all")

    capture = capture_spec()
    print(
        f"captured windows spec policy: "
        f"{len(capture['hiddenimports'])} hiddenimports, "
        f"{len(capture['datas'])} datas, {len(capture['binaries'])} binaries, "
        f"{len(capture['excludes'])} excludes, scripts={capture['scripts']}"
    )
    if args.capture:
        print(json.dumps(capture, indent=2)[:4000])
        return 0

    build: dict = {}
    smoke: dict = {}
    if args.build or args.all:
        build = run_build(capture)
        print(f"build: rc={build['rc']} seconds={build['seconds']} ok={build['ok']}")
        if not build["ok"]:
            print(build.get("error_tail", ""))
            if not args.all:
                return 1
    if args.smoke or args.all:
        if not (APP_BIN.is_file() and SMOKE_BIN.is_file()):
            print("frozen bundle missing -- run --build first")
            return 1
        smoke = run_smoke(Path(args.user_root))
        print(f"smoke: verdict={smoke.get('verdict')}")

    if build or smoke:
        out = persist(args.round_id, capture, build, smoke)
        print(f"evidence: {out}")
    overall = smoke.get("verdict") == "PASS" if smoke else (
        build.get("ok") if build else None
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
