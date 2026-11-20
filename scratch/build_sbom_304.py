"""round-304 (part12 Phase-3) -- the SBOM builder (FEATURE_GAP item #8).

Generates ``packaging/sbom/sbom.cdx.json`` -- the CycloneDX 1.6 SBOM of
the RELEASE stack, the audit's "release venv lockfile" surface:

  * ``packaging/windows/installer-requirements.txt`` -- the pinned
    stack the Windows release's step 7 installs (37 components), PLUS
  * the step-7 extras the workflow installs outside the -r file:
    torch==2.14.1 (the CPU wheel rides its own index at the same
    version), shap==0.54.1 (--no-deps, numpy-1.26-compatible),
    pyinstaller==6.22.3 (the freeze tool, version recorded in the
    RELEASE_MANIFEST) and pytest (the gate runner).

The output is validated (--validate) and made reproducible
(--output-reproducible: fixed serial, no timestamp) so the shipped
artifact re-verifies byte-identically in every home.

The Windows workflow's step 22b runs the SAME command on the runner
against the checkout's files and attaches the fresh SBOM to the
release -- this local artifact is the shipped reference + the pin
(tests/test_architecture_hygiene.py::TestSbomArtifact).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLKIT = Path("/home/z/my-project/pharma-ml-toolkit")
PY = "/home/z/.venv/bin/python"
CYCLONEDX = "/home/z/.venv/bin/cyclonedx-py"
OUT = TOOLKIT / "packaging" / "sbom" / "sbom.cdx.json"

#: the step-7 extras (installer-requirements does not carry them --
#: torch rides its own CPU index, shap rides --no-deps, pyinstaller +
#: pytest are the tooling pins the workflow installs explicitly).
EXTRAS = [
    "torch==2.14.1",
    "shap==0.54.1",
    "pyinstaller==6.22.3",
    "pytest",
]


def main() -> int:
    installer_lines = (
        TOOLKIT / "packaging" / "windows" / "installer-requirements.txt"
    ).read_text(encoding="utf-8").splitlines()
    # dedupe: the installer file's pyinstaller RANGE line is superseded
    # by the workflow's exact pin (6.22.3, the RELEASE_MANIFEST's own
    # record) carried in EXTRAS -- one component, one version.
    installer_lines = [
        line for line in installer_lines
        if not line.startswith("pyinstaller>=") and not line.startswith("pyinstaller==")
    ]
    combined = (
        "\n".join(installer_lines)
        + "\n\n# -- round-304 SBOM: the workflow step-7 extras "
        "(torch rides the CPU index at the same version, shap rides\n"
        "# --no-deps; pyinstaller==6.22.3 replaces the -r file's range "
        "line -- the exact pin the RELEASE_MANIFEST records)\n"
        + "\n".join(EXTRAS)
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(combined)
        tmp = Path(fh.name)
    proc = subprocess.run(
        [
            CYCLONEDX, "requirements", str(tmp),
            "--pyproject", str(TOOLKIT / "pyproject.toml"),
            "--sv", "1.6", "--of", "Json",
            "--output-reproducible", "--validate",
            "-o", str(OUT),
        ],
        capture_output=True, text=True,
    )
    tmp.unlink()
    if proc.returncode != 0:
        print("cyclonedx-py failed:", proc.returncode)
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return 1
    bom = json.loads(OUT.read_text(encoding="utf-8"))
    comps = bom.get("components", [])
    root = bom.get("metadata", {}).get("component", {})
    print(f"bomFormat={bom.get('bomFormat')} specVersion={bom.get('specVersion')}")
    print(f"root={root.get('name')} {root.get('version')} ({root.get('type')})")
    print(f"components={len(comps)}")
    names = sorted(c.get("name", "") for c in comps)
    print("sample:", ", ".join(names[:10]), "...")
    for needle in ("rdkit", "torch", "shap", "PySide6", "pyinstaller", "fastapi"):
        print(f"  {needle}: {'OK' if any(n.lower() == needle.lower() or n.lower().startswith(needle.lower() + '-') for n in names) else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
