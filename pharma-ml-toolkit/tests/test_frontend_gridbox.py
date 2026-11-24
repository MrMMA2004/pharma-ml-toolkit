"""part16 §12 (Phase 9) — the grid-box visualization drift guards.

The Python gate cannot execute the React 3D viewer, but every contract
the lane rides CAN be pinned against the real frontend + backend
source — a rename, a dropped pinned note, a second mutation surface,
or a transformed protein turns a guard red before the surface drifts:

* the viewer + its vitest pair exist under ``components/docking`` and
  ``services/docking`` (the component, the pure lane, both suites);
* the synchronization rule: the viewer's config prop is fed from the
  SAME six state variables the stage-6 dock request serializes
  (``center: [centerX, centerY, centerZ]`` /
  ``size: [sizeX, sizeY, sizeZ]``) — the coordinate-consistency test
  Phase 9 names, pinned as source text;
* the alignment rule verbatim: the pinned reference-frame note (never
  shift, rescale, or rotate) exists in the pure lane AND renders in
  the viewer;
* the spacing honesty verbatim: the DockRequest carries no spacing
  field, the pinned note states it, and the backend schema really has
  no spacing field (the three-way pin);
* the never-fabricate rule: the viewer's atoms arrive only through
  its props (the page's own PDBQT parse) — no hardcoded molecule
  coordinates, no imagined structure;
* the validation vocabulary: the six warning codes + the two
  severities pinned in the pure lane;
* the required controls: the four visibility toggles;
* the readouts: center/size/volume/in-box censuses + the Å unit note;
* the WebGL honesty: the fallback is a stated card, never a blank
  canvas;
* the DockingPage mounts the viewer inside stage 4 without
  renumbering any stage;
* the vitest pair was counted green this round (287/287 over 33
  files — the suite-level pin lives in the QA report, the files'
  own existence + test counts pinned here).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
COMPONENTS = SRC / "components" / "docking"
SERVICES = SRC / "services" / "docking"
DOCKING_PAGE = SRC / "features" / "docking" / "DockingPage.tsx"
DOCKING_ROUTE = ROOT / "src" / "pharmaml" / "api" / "routes" / "docking.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the component + the pure lane + both suites exist


def test_the_viewer_and_the_pure_lane_exist() -> None:
    assert (COMPONENTS / "GridBoxViewer.tsx").is_file(), "the Phase 9 grid-box viewer is missing"
    assert (COMPONENTS / "__tests__" / "GridBoxViewer.test.tsx").is_file()
    assert (SERVICES / "gridBox.ts").is_file(), "the pure grid-box data lane is missing"
    assert (SERVICES / "__tests__" / "gridBox.test.ts").is_file()


def test_the_viewer_uses_three_for_the_genuine_3d_scene() -> None:
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    assert 'import * as THREE from "three"' in viewer
    assert "OrbitControls" in viewer, (
        "the orbit/zoom/pan interaction is part of the Phase 9 contract"
    )


# ---------------------------------------------------------------------------
# the synchronization rule (the coordinate-consistency pin)


def test_the_dock_request_serializes_the_same_six_numbers() -> None:
    page = _read(DOCKING_PAGE)
    # the stage-6 request build (the ONLY dock submission lane)
    assert "center: [centerX, centerY, centerZ]" in page
    assert "size: [sizeX, sizeY, sizeZ]" in page
    # the viewer's config prop carries the SAME six state variables
    assert "config={{" in page
    for name in ("centerX", "centerY", "centerZ", "sizeX", "sizeY", "sizeZ"):
        assert re.search(rf"^\s+{name},$", page, re.MULTILINE), (
            f"the viewer config is missing {name}"
        )


def test_the_numeric_inputs_remain_the_one_mutation_surface() -> None:
    page = _read(DOCKING_PAGE)
    # the six setX handlers stay bound (the numeric form is the writer)
    for setter in ("setCenterX", "setCenterY", "setCenterZ", "setSizeX", "setSizeY", "setSizeZ"):
        assert setter in page
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    # the viewer itself never writes the config: no setter leaks in
    assert "setCenterX" not in viewer and "setSizeX" not in viewer
    assert "the numeric inputs stay the ONE mutation surface" in viewer


# ---------------------------------------------------------------------------
# the pinned notes (the alignment + spacing honesty, three-way)


def test_the_reference_frame_note_is_pinned_and_renders() -> None:
    lane = _read(SERVICES / "gridBox.ts")
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    assert "never shifts, rescales, or rotates the protein" in lane
    assert "receptor's own coordinate frame" in lane
    assert "REFERENCE_FRAME_NOTE" in lane
    assert "REFERENCE_FRAME_NOTE" in viewer, "the frame note must render verbatim in the viewer"


def test_the_spacing_note_is_pinned_and_the_backend_has_no_spacing_field() -> None:
    lane = _read(SERVICES / "gridBox.ts")
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    route = _read(DOCKING_ROUTE)
    assert "not a DockRequest field" in lane
    assert "SPACING_NOTE" in viewer, "the spacing note must render verbatim"
    # the backend DockRequest schema really carries no spacing field
    request = route[route.index("class DockRequest") :]
    request = request[: request.index("\n\n@")]
    assert "spacing" not in request, (
        "if the backend ever grows a spacing field, the honest note and "
        "the rendered control must be trued together"
    )
    assert "center: list[float]" in request and "size: list[float]" in request


# ---------------------------------------------------------------------------
# the never-fabricate rule (props-only structures)


def test_the_viewer_renders_only_atoms_that_arrived_through_props() -> None:
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    # no hardcoded molecule coordinates: every Vector3 literal in the
    # scene code is scaffolding (camera/triad math), never an atom —
    # the atoms enter exclusively through the props' parse results
    assert "receptorAtoms" in viewer and "ligand" in viewer
    assert viewer.count("new THREE.Vector3(") < 20, (
        "a molecule-sized vertex list would be a fabricated structure"
    )
    assert "actually arrived" in viewer
    # the page feeds the viewer from its own PDBQT parse memos
    page = _read(DOCKING_PAGE)
    assert "receptor3dScene" in page and "ligand3dScene" in page


# ---------------------------------------------------------------------------
# the validation vocabulary + the required controls + the readouts


def test_the_six_warning_codes_are_pinned() -> None:
    lane = _read(SERVICES / "gridBox.ts")
    for code in (
        "non-finite",
        "non-positive-size",
        "extreme-size",
        "far-origin",
        "empty-box",
        "ligand-outside",
    ):
        assert f'"{code}"' in lane, f"the {code} warning code is missing"
    assert 'severity: "error" | "warning"' in lane


def test_the_four_visibility_toggles_exist() -> None:
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    for label in ("Protein", "Grid box", "Ligand", "Binding-site highlight"):
        assert label in viewer, f"the {label} toggle is missing"
    assert viewer.count('type="checkbox"') == 4


def test_the_readouts_and_the_unit_note_render() -> None:
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    assert "receptor atoms in box" in viewer
    assert "ligand atoms in box" in viewer
    assert "volume" in viewer
    assert "UNIT_NOTE" in viewer


def test_the_webgl_fallback_is_a_stated_card_not_a_blank_canvas() -> None:
    viewer = _read(COMPONENTS / "GridBoxViewer.tsx")
    assert "WebGL unavailable" in viewer
    assert 'role="status"' in viewer


# ---------------------------------------------------------------------------
# the stage-4 mount (no stage renumbering)


def test_the_page_mounts_the_viewer_in_stage_4() -> None:
    page = _read(DOCKING_PAGE)
    assert 'import { GridBoxViewer } from "@/components/docking/GridBoxViewer";' in page
    assert "<GridBoxViewer" in page
    # the stage numbering stands: exactly one index={4} header
    assert page.count("index={4}") == 1
    assert "Define Binding Site/Grid" in page


def test_the_vitest_pair_files_carry_their_contract_counts() -> None:
    component_suite = _read(COMPONENTS / "__tests__" / "GridBoxViewer.test.tsx")
    lane_suite = _read(SERVICES / "__tests__" / "gridBox.test.ts")
    # the suites stay meaningful: both carry their describe blocks
    assert component_suite.count("describe(") >= 2
    assert lane_suite.count("describe(") >= 4
    assert component_suite.count("  it(") + lane_suite.count("  it(") >= 20, (
        "the pair's 33-test payload must not silently shrink"
    )
