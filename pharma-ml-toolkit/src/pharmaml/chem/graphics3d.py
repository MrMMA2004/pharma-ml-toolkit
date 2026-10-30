r"""The nglview and 3Dmol.js concepts, absorbed natively: the 3D
geometry layer (coordinate builder, cameras, representations, atom
selections, the Shrake-Rupley surface) in pure Python, with an ASCII
depth renderer standing in for WebGL.

Two instructions, one round: *"Check the contents and files in this
GitHub link completely and one by one and add the missing items to
playground.py: github.com/nglviewer/nglview"* and *"...:
github.com/3dmol/3Dmol.js"*.  Both repositories were cloned and
reviewed file by file.

**nglview** (the Python side, ``nglview/*.py``): ``widget.py`` (the
ipywidgets bridge -- every method funnels into ``_remote_call``, a JS
message), ``component.py`` (``ComponentViewer``: per-component
representations, hide/show, set_position/rotation/scale),
``representation.py`` (the parameter panel: opacity, colorScheme, the
surface types ``vws/sas/ms/ses``, probeRadius), ``adaptor.py`` and
``show.py`` (the ``show_*`` dispatchers wrapping thirty structure
backends), ``color.py`` (``COLOR_SCHEMES``, the ColormakerRegistry),
``shape.py``, ``stage.py``, ``viewer_control.py``, plus the ``js/``
TypeScript engine (NGL itself, ~200k lines of WebGL).

**3Dmol.js** (``src/*.ts``): ``GLViewer``/``GLModel`` (the viewer and
model), ``specs.ts`` (``AtomSpec`` and the ``AtomSelectionSpec`` DSL:
``elem``, ``index``, ``bonds``, ``invert``, ``and``/``or``/``not``,
``within``), ``colors.ts`` (the Jmol element palette),
``GLModel.vdwRadii`` (the Bondi 1964 table), ``ProteinSurface4.ts``
(the EDTSurf surface), ``Label.ts``, ``GLShape.ts``, ``parsers/``,
and the ``py3Dmol`` wrapper (a 363-line ``view`` class that emits JS).

What both repositories *are* is a browser 3D engine -- and the engine
(WebGL, shaders, the NGL TypeScript core, ipywidgets) is exactly what
this toolkit does not vendor.  What transfers natively is the geometry
layer both engines sit on, and all of it landed here:

* **A rule-based 3D coordinate builder**
  (:func:`build_coordinates_3d`) -- nglview's ``RdkitStructure``
  adaptor calls RDKit's ``EmbedMolecule``; this is the pure-Python
  stand-in: rings from the planar layout, chains grown by BFS with
  hybridisation-aware directions (the tetrahedral completion
  ``-0.5(v1+v2) +- sqrt(2/3) n``, the 120-degree cone for sp2, the
  exact ``-(v1+v2+v3)`` for the last tetrahedral slot), ideal bond
  lengths from the covalent radii table, collision-aware staggered
  rotation from a deterministic candidate order.  Honest scope: rule
  geometry, not force-field refinement -- chair/fused-ring
  puckering is not modelled (rings are planar), and stereochemistry
  is not interpreted (the parser records but never reads ``@``/``@@``).
* **The two selection dialects** -- 3Dmol's dict-based
  :func:`select_spec` (``elem``, ``index``, ``bonds``, ``invert``,
  ``and``/``or``/``not``, ``within`` with its ``distance``/``sel``/
  ``invert`` fields, straight from ``specs.ts``) and NGL's string
  grammar :func:`select_string` (``all``, ``none``, ``@1-5,7``,
  ``element C``, ``not``/``and``/``or``, ``within(4.0, ...)`` with
  parentheses), both resolved to atom indices.
* **The representation registry** (:data:`REPRESENTATIONS`) -- the
  NGL names (``licorice``, ``ball+stick``, ``spacefill``, ``line``)
  and the 3Dmol styles (``stick``, ``sphere``, ``cross``) as one
  table of atom scales and bond switches, each with its provenance.
  ``cartoon``/``ribbon`` are absent on purpose: they need the protein
  residue/secondary-structure records a SMILES-scale toolkit does not
  have -- stated, not faked.
* **The camera** (:class:`Camera`) -- yaw/pitch/zoom, the orthographic
  projection both engines' ``setRotation``/``zoomTo`` wrap; the
  perspective camera is not modelled and said so.
* **The surface as Shrake-Rupley SASA** (:func:`sasa`) -- the concept
  behind 3Dmol's ``SurfaceType.SAS``/``VDW`` and NGL's ``sas``/
  ``vws`` surface types: a golden-spiral point set per atom, points
  occluded by any neighbour's probe-expanded sphere are buried, the
  survivors integrate the accessible area.  Per-atom and total, with
  ``probe_radius=0`` reducing to the van der Waals surface.
* **The Jmol palette and Bondi radii** (:data:`ELEMENT_COLORS`) --
  3Dmol's ``elementColors.Jmol`` as ``#rrggbb`` strings for the GUI
  canvas, the radii already in :mod:`pharmaml.chem.molgraph`'s
  element table (same Bondi source as ``GLModel.vdwRadii``).
* **The renderer** (:func:`render_ascii`, :func:`render_primitives`)
  -- the painter's algorithm over depth-sorted primitives: atoms as
  element symbols (``spacefill`` fills the van der Waals disc),
  bonds as slope-chosen glyphs, selected atoms bracketed, labels
  placed at projected positions; :func:`render_primitives` emits the
  same primitives as typed records for the playground's tkinter
  canvas (the stand-in for the WebGL canvas both repos render into).

Attribution: nglview (MIT) and 3Dmol.js (BSD-3-Clause); the data
tables follow their sources, and every piece of browser machinery
(ipywidgets, WebGL, shaders, click/hover callbacks, ``spin``) is
stated here as deliberately absent rather than quietly re-implemented.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmaml.chem.molgraph import (
    ELEMENTS,
    Molecule,
    find_rings,
    generate_2d_coordinates,
    molecular_formula,
)
from pharmaml.exceptions import DataValidationError

__all__ = [
    "ELEMENT_COLORS",
    "REPRESENTATIONS",
    "AtomPrimitive",
    "BondPrimitive",
    "Camera",
    "RepresentationSpec",
    "SasaReport",
    "build_coordinates_3d",
    "render_ascii",
    "render_primitives",
    "sasa",
    "select_spec",
    "select_string",
    "view_report",
]


#: 3Dmol.js ``elementColors.Jmol`` (``src/colors.ts``), organic subset
#: plus the common ions, as ``#rrggbb`` strings for canvas drawing.
ELEMENT_COLORS: dict[str, str] = {
    "H": "#ffffff",
    "B": "#ffb5b5",
    "C": "#909090",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "F": "#90e050",
    "Na": "#ab5cf2",
    "Mg": "#8aff00",
    "Si": "#f0c8a0",
    "P": "#ff8000",
    "S": "#ffff30",
    "Cl": "#1ff01f",
    "K": "#8f40d4",
    "Ca": "#3dff00",
    "Fe": "#e06633",
    "Zn": "#7d80b0",
    "Br": "#a62929",
    "I": "#940094",
}
#: 3Dmol.js ``defaultColor`` -- unknown elements.
_DEFAULT_COLOR = "#ff1493"


@dataclass(frozen=True)
class RepresentationSpec:
    """One representation: atom scale, bond switch, provenance."""

    atom_scale: float
    bonds: bool
    cross: bool = False
    label: str = ""


#: The representation registry -- NGL's names and 3Dmol's styles, one table.
REPRESENTATIONS: dict[str, RepresentationSpec] = {
    "line": RepresentationSpec(0.0, True, label="3Dmol 'line' style"),
    "licorice": RepresentationSpec(0.18, True, label="NGL 'licorice'"),
    "stick": RepresentationSpec(0.25, True, label="3Dmol 'stick' style"),
    "ball+stick": RepresentationSpec(
        0.33, True, label="NGL 'ball+stick' / 3Dmol ball-and-stick"
    ),
    "spacefill": RepresentationSpec(1.0, False, label="NGL 'spacefill' / 3Dmol 'sphere'"),
    "cross": RepresentationSpec(0.0, False, cross=True, label="3Dmol 'cross' style"),
}
REPRESENTATIONS["sphere"] = REPRESENTATIONS["spacefill"]


@dataclass(frozen=True)
class Camera:
    """Yaw/pitch/zoom camera -- the ``setRotation``/``zoomTo`` concept.

    Orthographic: the rotation is ``Rx(pitch) @ Ry(yaw)`` applied to
    the centroid-centred coordinates; ``zoom`` scales the fit.  The
    perspective cameras of NGL/3Dmol are not modelled.

    >>> camera = Camera(yaw=90.0, pitch=0.0)
    >>> camera.rotation_matrix().round(3).tolist()
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    """

    yaw: float = 25.0
    pitch: float = 20.0
    zoom: float = 1.0

    def rotation_matrix(self) -> np.ndarray:
        """``Rx(pitch) @ Ry(yaw)`` as a 3x3 array."""
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        matrix = np.array(
            [
                [cy, -sy * sp, sy * cp],
                [0.0, cp, sp],
                [-sy, -cy * sp, cy * cp],
            ]
        )
        # + 0.0 folds the -0.0 entries the products leave behind
        # into plain 0.0 so reprs stay doctest-stable
        return matrix + 0.0



#: Bond-length factors by bond order (aromatic bonds carry the
#: perceived 1/2 order with ``is_aromatic`` set).
_ORDER_SCALE = {1.0: 1.0, 2.0: 0.87, 3.0: 0.78}


def _bond_length(mol: Molecule, begin: int, end: int) -> float:
    """Ideal bond length: covalent radii sum x order factor."""
    bond = next(
        b
        for b in mol.bonds
        if {b.begin, b.end} == {begin, end}
    )
    r_a = ELEMENTS[mol.atoms[begin].symbol].covalent_radius
    r_b = ELEMENTS[mol.atoms[end].symbol].covalent_radius
    if bond.is_aromatic:
        return 0.90 * (r_a + r_b)
    factor = _ORDER_SCALE.get(float(bond.order), 1.0)
    return factor * (r_a + r_b)


def _hybridization(mol: Molecule, index: int) -> str:
    """``"sp3"``/``"sp2"``/``"sp"`` from pi-bond count and aromaticity."""
    atom = mol.atoms[index]
    if atom.is_aromatic:
        return "sp2"
    pi = 0.0
    for bond in mol.atom_bonds(index):
        if not bond.is_aromatic and bond.order > 1:
            pi += bond.order - 1
    if pi >= 2:
        return "sp"
    if pi == 1:
        return "sp2"
    return "sp3"


_IDEAL_ANGLE = {"sp3": math.radians(109.47), "sp2": math.radians(120.0), "sp": math.pi}


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise DataValidationError("cannot normalise a zero-length direction")
    return vec / norm


def _ortho_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to ``axis`` (deterministic)."""
    ref = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(axis[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(axis, ref)
    u = u / float(np.linalg.norm(u))
    w = np.cross(axis, u)
    return u, w


def _cone_directions(axis: np.ndarray, theta: float, n_az: int = 6) -> list[np.ndarray]:
    """Directions making angle ``theta`` with ``axis`` (n_az azimuths)."""
    u, w = _ortho_basis(axis)
    out = []
    for k in range(n_az):
        phi = 2.0 * math.pi * k / n_az
        out.append(
            axis * math.cos(theta)
            + (u * math.cos(phi) + w * math.sin(phi)) * math.sin(theta)
        )
    return out


_AXIS_CANDIDATES = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, -1.0]),
]


def build_coordinates_3d(mol: Molecule) -> np.ndarray:
    """Rule-based 3D coordinates for ``mol`` -- ``(n_atoms, 3)`` array.

    The concept of nglview's structure adaptors (which call RDKit's
    ``EmbedMolecule``) as a pure builder: rings take the planar layout
    in the ``z=0`` plane (scaled to ideal bond lengths); every other
    atom is grown by BFS from its placed neighbours with
    hybridisation-aware directions -- one neighbour: the ideal-angle
    cone at six azimuths; two: the in-plane ``-(v1+v2)`` bisector for
    sp2 or the two tetrahedral completions ``-0.5(v1+v2) +- sqrt(2/3)n``
    for sp3; three: the exact ``-(v1+v2+v3)`` slot -- and the candidate
    that maximises the distance to every placed atom wins, which makes
    chains stagger and substituents alternate above/below rings without
    any randomness.  Deterministic; not force-field refined (rings stay
    planar, stereo markers are not interpreted -- the honest limits).

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> coords = build_coordinates_3d(parse_smiles("CCO"))
    >>> round(float(np.linalg.norm(coords[0] - coords[1])), 2)   # C-C
    1.52
    >>> round(float(np.linalg.norm(coords[1] - coords[2])), 2)   # C-O
    1.42
    >>> import numpy as np
    >>> v1 = coords[0] - coords[1]
    >>> v2 = coords[2] - coords[1]
    >>> cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    >>> round(math.degrees(math.acos(cosang)), 1)                # C-C-O angle
    109.5
    """
    n_atoms = len(mol.atoms)
    coords = np.zeros((n_atoms, 3), dtype=float)
    if n_atoms <= 1:
        return coords
    adj = mol.adjacency()
    rings = find_rings(mol)
    ring_atoms: set[int] = set()
    for ring in rings:
        ring_atoms.update(ring)

    placed: set[int] = set()
    if ring_atoms:
        planar = np.asarray(generate_2d_coordinates(mol), dtype=float)
        ring_bond_lengths = []
        for bond in mol.bonds:
            if bond.begin in ring_atoms and bond.end in ring_atoms:
                ring_bond_lengths.append(
                    _bond_length(mol, bond.begin, bond.end)
                    / max(
                        float(
                            np.linalg.norm(planar[bond.begin] - planar[bond.end])
                        ),
                        1e-9,
                    )
                )
        scale = float(np.mean(ring_bond_lengths)) if ring_bond_lengths else 1.0
        for atom in ring_atoms:
            coords[atom, 0] = planar[atom, 0] * scale
            coords[atom, 1] = planar[atom, 1] * scale
        placed.update(ring_atoms)

    order_key = [len(adj[i]) for i in range(n_atoms)]
    if not placed:
        first = min(range(n_atoms), key=lambda i: (order_key[i], i))
        placed.add(first)
        queue: list[int] = [first]
    else:
        queue = sorted(placed)
    while True:
        # drain the BFS frontier; only when it is exhausted does the
        # next disconnected component get its own root (min degree,
        # lowest index), so every atom is grown from a neighbour
        while queue:
            parent = queue.pop(0)
            for child in sorted(adj[parent], key=lambda k: (order_key[k], k)):
                if child in placed:
                    continue
                length = _bond_length(mol, parent, child)
                neighbours = [k for k in adj[parent] if k in placed]
                vectors = [_unit(coords[k] - coords[parent]) for k in neighbours]
                hybrid = _hybridization(mol, parent)
                candidates: list[np.ndarray] = []
                if not vectors:
                    candidates = list(_AXIS_CANDIDATES)
                elif len(vectors) == 1:
                    theta = _IDEAL_ANGLE[hybrid]
                    if hybrid == "sp":
                        candidates = [-vectors[0]]
                    else:
                        candidates = _cone_directions(vectors[0], theta)
                elif len(vectors) == 2:
                    total = vectors[0] + vectors[1]
                    if hybrid == "sp2":
                        candidates = [_unit(-total)]
                    else:
                        normal = np.cross(vectors[0], vectors[1])
                        norm = float(np.linalg.norm(normal))
                        if norm < 1e-6:
                            candidates = _cone_directions(
                                vectors[0], _IDEAL_ANGLE["sp3"]
                            )
                        else:
                            normal = normal / norm
                            in_plane = -0.5 * total
                            perp = math.sqrt(
                                max(0.0, 1.0 - float(np.dot(in_plane, in_plane)))
                            )
                            candidates = [
                                _unit(in_plane + normal * perp),
                                _unit(in_plane - normal * perp),
                            ]
                else:
                    total = vectors[0] + vectors[1] + vectors[2]
                    if float(np.linalg.norm(total)) < 1e-6:
                        candidates = list(_AXIS_CANDIDATES)
                    else:
                        candidates = [_unit(-total)]
                best: tuple[float, np.ndarray] | None = None
                for direction in candidates:
                    trial = coords[parent] + length * direction
                    clear = min(
                        (
                            float(np.linalg.norm(trial - coords[k]))
                            for k in placed
                            if k != parent
                        ),
                        default=1e9,
                    )
                    if best is None or clear > best[0] + 1e-9:
                        best = (clear, direction)
                assert best is not None
                coords[child] = coords[parent] + length * best[1]
                placed.add(child)
                queue.append(child)
        unplaced = [i for i in range(n_atoms) if i not in placed]
        if not unplaced:
            break
        root = min(unplaced, key=lambda i: (order_key[i], i))
        placed.add(root)
        queue = [root]

    # disconnected components: lay them out along x with a 6 A gap
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(n_atoms):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(sorted(component))
    if len(components) > 1:
        offset = 0.0
        for component in components:
            block = coords[component]
            block = block - block.mean(axis=0)
            span = float(block[:, 0].max() - block[:, 0].min())
            coords[component] = block
            coords[component, 0] += offset + span / 2.0
            offset += span + 6.0
    return coords


# ------------------------------------------------------------------ #
# Atom selections: 3Dmol's dict spec and NGL's string grammar          #
# ------------------------------------------------------------------ #


def _match_base(mol: Molecule, index: int, spec: Mapping[str, Any]) -> bool:
    """One atom against the scalar keys of an AtomSelectionSpec."""
    atom = mol.atoms[index]
    for key, value in spec.items():
        if key in ("invert", "and", "or", "not", "within", "byres", "expand"):
            continue
        if key == "elem":
            if atom.symbol not in (value if isinstance(value, list) else [value]):
                return False
        elif key == "index":
            if index not in (value if isinstance(value, list) else [value]):
                return False
        elif key == "bonds":
            if len(mol.atom_bonds(index)) != value:
                return False
        elif key == "aromatic":
            if bool(atom.is_aromatic) != bool(value):
                return False
        else:
            raise DataValidationError(
                f"unsupported selection key {key!r}; supported: elem, index, "
                "bonds, aromatic, invert, and, or, not, within"
            )
    return True


def select_spec(
    mol: Molecule,
    spec: Mapping[str, Any],
    coords: np.ndarray | None = None,
) -> list[int]:
    """3Dmol.js's ``AtomSelectionSpec`` resolved to atom indices.

    The dict dialect of ``src/specs.ts``: scalar fields match
    individually (lists mean any-of), ``invert`` flips, ``and``
    intersects, ``or`` unions, ``not`` complements, and ``within``
    (``{"distance": d, "sel": spec}``) keeps the atoms of this spec
    within ``d`` Angstrom of the other selection.  ``within`` needs
    coordinates; pass them or get the accounted error.

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    >>> select_spec(aspirin, {"elem": "O"})
    [2, 3, 11, 12]
    >>> select_spec(aspirin, {"elem": "O", "invert": True})
    [0, 1, 4, 5, 6, 7, 8, 9, 10]
    >>> select_spec(aspirin, {"or": [{"elem": "N"}, {"elem": "S"}]})
    []
    >>> select_spec(aspirin, {"bonds": 1})
    [0, 2, 11, 12]
    """
    selected = [i for i in range(len(mol.atoms)) if _match_base(mol, i, spec)]
    # "and"/"or"/"not"/"within" are fields of the spec: an atom must
    # satisfy every present field, exactly as 3Dmol intersects them
    if "and" in spec:
        for other in spec["and"]:
            keep = set(select_spec(mol, other, coords))
            selected = [i for i in selected if i in keep]
    if "or" in spec:
        union: set[int] = set()
        for other in spec["or"]:
            union.update(select_spec(mol, other, coords))
        selected = [i for i in selected if i in union]
    if "not" in spec:
        drop = set(select_spec(mol, spec["not"], coords))
        selected = [i for i in selected if i not in drop]
    if "within" in spec:
        within = spec["within"]
        distance = float(within.get("distance", 5.0))
        other = select_spec(mol, within.get("sel", {}), coords)
        if coords is None:
            raise DataValidationError(
                "within(...) needs coordinates: pass the built 3D coordinates"
            )
        if not other:
            return []
        near = np.asarray([coords[j] for j in other])
        selected = [
            i
            for i in selected
            if float(np.min(np.linalg.norm(near - coords[i], axis=1))) <= distance
        ]
    if spec.get("invert"):
        selected = [i for i in range(len(mol.atoms)) if i not in selected]
    if "within" in spec and spec["within"].get("invert"):
        selected = [i for i in range(len(mol.atoms)) if i not in selected]
    return selected


def _tokenize(expression: str) -> list[str]:
    """Split an NGL selection string into tokens."""
    tokens: list[str] = []
    current = ""
    for char in expression:
        if char in "(),":
            if current:
                tokens.append(current)
                current = ""
            if char in "(),":
                tokens.append(char)
        elif char.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens


class _StringParser:
    """Recursive-descent parser for the NGL selection grammar."""

    def __init__(
        self, mol: Molecule, tokens: list[str], coords: np.ndarray | None
    ) -> None:
        self.mol = mol
        self.tokens = tokens
        self.pos = 0
        self.coords = coords

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self) -> list[int]:
        result = self.parse_or()
        if self.peek() is not None:
            raise DataValidationError(
                f"unexpected token {self.peek()!r} in the selection string"
            )
        return result

    def parse_or(self) -> list[int]:
        result = set(self.parse_and())
        while True:
            token = self.peek()
            if token is None or token.lower() != "or":
                return sorted(result)
            self.take()
            result.update(self.parse_and())

    def parse_and(self) -> list[int]:
        result = set(self.parse_unary())
        while True:
            token = self.peek()
            if token is None or token.lower() != "and":
                return sorted(result)
            self.take()
            result.intersection_update(self.parse_unary())

    def parse_unary(self) -> list[int]:
        head = self.peek()
        if head is not None and head.lower() == "not":
            self.take()
            inner = set(self.parse_unary())
            return [i for i in range(len(self.mol.atoms)) if i not in inner]
        if self.peek() == "(":
            self.take()
            result = self.parse_or()
            if self.take() != ")":
                raise DataValidationError("missing ')' in the selection string")
            return result
        return self.parse_primary()

    def parse_primary(self) -> list[int]:
        token = self.take()
        lowered = token.lower()
        if lowered == "all":
            return list(range(len(self.mol.atoms)))
        if lowered == "none":
            return []
        if token.startswith("@"):
            indices: list[int] = []
            for part in token[1:].split(","):
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    indices.extend(range(int(lo), int(hi) + 1))
                else:
                    indices.append(int(part))
            valid = set(range(len(self.mol.atoms)))
            return [i for i in indices if i in valid]
        if lowered == "element":
            symbol = self.take()
            return [
                i
                for i, atom in enumerate(self.mol.atoms)
                if atom.symbol == symbol
            ]
        if lowered == "within":
            if self.take() != "(":
                raise DataValidationError("within expects '('")
            distance = float(self.take())
            if self.take() != ",":
                raise DataValidationError("within expects ','")
            inner = self.parse_or()
            if self.take() != ")":
                raise DataValidationError("within expects ')'")
            if self.coords is None:
                raise DataValidationError(
                    "within(...) needs coordinates: pass the built 3D coordinates"
                )
            if not inner:
                return []
            near = np.asarray([self.coords[j] for j in inner])
            return [
                i
                for i in range(len(self.mol.atoms))
                if float(np.min(np.linalg.norm(near - self.coords[i], axis=1)))
                <= distance
            ]
        raise DataValidationError(
            f"unrecognised selection token {token!r}; the grammar is "
            "all | none | @i-j,k | element SYM | not E | E and E | E or E | "
            "( E ) | within(d, E)"
        )


def select_string(
    mol: Molecule,
    expression: str,
    coords: np.ndarray | None = None,
) -> list[int]:
    """NGL's selection string grammar resolved to atom indices.

    The string dialect nglview forwards to the engine: ``all``,
    ``none``, ``@`` index lists with ranges (``@1-3,7``), ``element``
    tests, ``not``/``and``/``or`` with parentheses, and
    ``within(distance, selection)`` -- the last needs coordinates.

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    >>> select_string(aspirin, "@0-1")
    [0, 1]
    >>> select_string(aspirin, "element O")
    [2, 3, 11, 12]
    >>> select_string(aspirin, "not element C and not element O")
    []
    >>> select_string(aspirin, "(@0 or @2) and element O")
    [2]
    """
    parser = _StringParser(mol, _tokenize(expression), coords)
    return parser.parse()


# ------------------------------------------------------------------ #
# Shrake-Rupley SASA: the surface behind SurfaceType.SAS / 'sas'      #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class SasaReport:
    """Solvent-accessible surface areas, per atom and total."""

    total: float
    per_atom: tuple[float, ...]
    probe_radius: float
    n_points: int


def _sphere_points(n_points: int) -> np.ndarray:
    """Golden-spiral (Fibonacci) points on the unit sphere."""
    offsets = np.arange(n_points, dtype=float) + 0.5
    z = 1.0 - 2.0 * offsets / n_points
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden * offsets
    return np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), z], axis=1
    )


def sasa(
    mol: Molecule,
    coords: np.ndarray,
    *,
    probe_radius: float = 1.4,
    n_points: int = 92,
) -> SasaReport:
    """Shrake-Rupley solvent-accessible surface area (A^2), per atom.

    The concept behind 3Dmol's ``SurfaceType.SAS``/``VDW`` and NGL's
    ``sas``/``vws`` surface representations: each atom carries its
    Bondi van der Waals radius plus the probe; ``n_points``
    golden-spiral points are scattered on the expanded sphere, every
    point buried inside another atom's expanded sphere is discarded,
    and the survivors scale ``4 pi R^2``.  ``probe_radius=0`` reduces
    to the van der Waals surface.  Deterministic (no sampling noise
    beyond the fixed point set).

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> carbon = parse_smiles("C")
    >>> report = sasa(carbon, build_coordinates_3d(carbon))
    >>> round(report.total, 2)      # 4 pi (1.7 + 1.4)^2
    120.76
    >>> ethanol = parse_smiles("CCO")
    >>> coords = build_coordinates_3d(ethanol)
    >>> bare = sasa(ethanol, coords, probe_radius=0.0)
    >>> probe = sasa(ethanol, coords, probe_radius=1.4)
    >>> bare.total < probe.total    # the probe reaches into crevices
    True
    >>> bool(round(sum(report.per_atom), 6) == round(report.total, 6))
    True
    """
    n_atoms = len(mol.atoms)
    if n_atoms == 0:
        return SasaReport(0.0, (), probe_radius, n_points)
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (n_atoms, 3):
        raise DataValidationError(
            f"coordinates must be ({n_atoms}, 3), got {coords.shape}"
        )
    radii = np.array(
        [
            ELEMENTS[atom.symbol].vdw_radius + probe_radius
            for atom in mol.atoms
        ]
    )
    points = _sphere_points(n_points)
    per_atom: list[float] = []
    for i in range(n_atoms):
        sphere = coords[i] + radii[i] * points
        free = np.ones(n_points, dtype=bool)
        for j in range(n_atoms):
            if j == i:
                continue
            gap = float(np.linalg.norm(coords[j] - coords[i]))
            if gap > radii[i] + radii[j]:
                continue  # cannot occlude any point
            blocked = np.linalg.norm(sphere - coords[j], axis=1) < radii[j]
            free &= ~blocked
        fraction = float(free.sum()) / n_points
        per_atom.append(4.0 * math.pi * radii[i] ** 2 * fraction)
    return SasaReport(
        total=float(sum(per_atom)),
        per_atom=tuple(per_atom),
        probe_radius=probe_radius,
        n_points=n_points,
    )


# ------------------------------------------------------------------ #
# The renderer: camera projection, depth-sorted primitives, ASCII      #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class AtomPrimitive:
    """One drawable atom: screen position, depth, radius, colour."""

    index: int
    x: float
    y: float
    depth: float
    radius: float
    symbol: str
    color: str
    selected: bool


@dataclass(frozen=True)
class BondPrimitive:
    """One drawable bond: both endpoints' screen positions and colours."""

    begin: int
    end: int
    x1: float
    y1: float
    x2: float
    y2: float
    depth: float
    color_begin: str
    color_end: str
    order: int


def _projection(
    mol: Molecule,
    coords: np.ndarray,
    camera: Camera,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Centred, rotated, scaled: screen xy, depth, and the scale/AA."""
    coords = np.asarray(coords, dtype=float)
    centred = coords - coords.mean(axis=0)
    rotated = centred @ camera.rotation_matrix().T
    # fit the coordinate span PLUS the largest van der Waals radius,
    # so spacefill discs (3Dmol "sphere", NGL "spacefill") stay on the
    # canvas instead of blowing past it -- the engines fit the view to
    # the drawn geometry, and the discs are part of it
    vdw_pad = max(
        (ELEMENTS[atom.symbol].vdw_radius for atom in mol.atoms),
        default=1.7,
    )
    span = np.abs(rotated[:, :2]).max(axis=0) + vdw_pad
    span = np.where(span < 1e-6, 1.0, span)
    scale = (
        min(
            0.46 * width / float(span[0]),
            0.46 * height / float(span[1]),
        )
        * camera.zoom
    )
    screen = rotated[:, :2] * scale
    screen[:, 1] *= -1.0
    screen += np.array([width / 2.0, height / 2.0])
    return screen, rotated[:, 2], scale


def render_primitives(
    mol: Molecule,
    coords: np.ndarray,
    *,
    representation: str = "ball+stick",
    camera: Camera | None = None,
    width: float = 640.0,
    height: float = 480.0,
    selection: Iterable[int] | None = None,
) -> list[AtomPrimitive | BondPrimitive]:
    """Depth-sorted drawing primitives (painter's algorithm order).

    The records the playground's canvas consumes -- the stand-in for
    the WebGL draw list both engines build: atoms (with their Jmol
    colour and spacefill-scaled radius) and bonds (half-coloured by
    endpoint), sorted far-to-near so back-to-front drawing occludes
    correctly.  A bond draws when both endpoints are in the active
    selection (3Dmol's rule); atoms outside the selection are dropped.

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> water = parse_smiles("O")
    >>> prims = render_primitives(
    ...     water, build_coordinates_3d(water), width=100, height=100
    ... )
    >>> [(type(p).__name__, p.symbol) for p in prims]
    [('AtomPrimitive', 'O')]
    """
    if representation not in REPRESENTATIONS:
        raise DataValidationError(
            f"unknown representation {representation!r}; known: "
            f"{sorted(REPRESENTATIONS)}"
        )
    spec = REPRESENTATIONS[representation]
    camera = camera or Camera()
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (len(mol.atoms), 3):
        raise DataValidationError(
            f"coordinates must be ({len(mol.atoms)}, 3), got {coords.shape}"
        )
    selected = (
        set(range(len(mol.atoms))) if selection is None else set(selection)
    )
    screen, depth, scale = _projection(mol, coords, camera, width, height)
    primitives: list[AtomPrimitive | BondPrimitive] = []
    for atom_index in sorted(selected):
        atom = mol.atoms[atom_index]
        primitives.append(
            AtomPrimitive(
                index=atom_index,
                x=float(screen[atom_index, 0]),
                y=float(screen[atom_index, 1]),
                depth=float(depth[atom_index]),
                radius=spec.atom_scale
                * ELEMENTS[atom.symbol].vdw_radius
                * scale,
                symbol=atom.symbol,
                color=ELEMENT_COLORS.get(atom.symbol, _DEFAULT_COLOR),
                selected=selection is not None,
            )
        )
    if spec.bonds:
        for bond in mol.bonds:
            if bond.begin not in selected or bond.end not in selected:
                continue
            primitives.append(
                BondPrimitive(
                    begin=bond.begin,
                    end=bond.end,
                    x1=float(screen[bond.begin, 0]),
                    y1=float(screen[bond.begin, 1]),
                    x2=float(screen[bond.end, 0]),
                    y2=float(screen[bond.end, 1]),
                    depth=float(
                        (depth[bond.begin] + depth[bond.end]) / 2.0
                    ),
                    color_begin=ELEMENT_COLORS.get(
                        mol.atoms[bond.begin].symbol, _DEFAULT_COLOR
                    ),
                    color_end=ELEMENT_COLORS.get(
                        mol.atoms[bond.end].symbol, _DEFAULT_COLOR
                    ),
                    order=int(bond.order),
                )
            )
    primitives.sort(key=lambda p: p.depth)
    return primitives


def _ascii_glyph(dx: float, dy: float) -> str:
    """Bond glyph by screen direction (y grows downward)."""
    if abs(dx) >= 3.0 * abs(dy):
        return "-"
    if abs(dy) >= 3.0 * abs(dx):
        return "|"
    return "/" if dx * dy < 0 else chr(92)


def render_ascii(
    mol: Molecule,
    coords: np.ndarray,
    *,
    representation: str = "ball+stick",
    camera: Camera | None = None,
    width: int = 64,
    height: int = 26,
    selection: Iterable[int] | None = None,
    labels: Iterable[tuple[int, str]] = (),
) -> str:
    """Render one molecule as ASCII -- the WebGL canvas as a char grid.

    The painter's algorithm over :func:`render_primitives`: bonds
    rasterised as slope glyphs, atoms as their element symbols
    (``spacefill`` fills the van der Waals disc with the symbol),
    ``cross`` draws ``+`` markers, selected atoms are bracketed, and
    labels are written beside their atoms.  Character cells are twice
    as tall as wide, so the projection compensates.  Deterministic
    under a fixed camera.

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> water = parse_smiles("O")
    >>> print(render_ascii(water, build_coordinates_3d(water), width=20, height=7))
    ####################
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
              O
    <BLANKLINE>
    <BLANKLINE>
    ####################
    <BLANKLINE>
    """
    if width < 10 or height < 5:
        raise DataValidationError(
            f"ascii canvas must be at least 10x5 cells, got {width}x{height}"
        )
    if representation not in REPRESENTATIONS:
        raise DataValidationError(
            f"unknown representation {representation!r}; known: "
            f"{sorted(REPRESENTATIONS)}"
        )
    spec = REPRESENTATIONS[representation]
    pixel_height = height * 2.2
    primitives = render_primitives(
        mol,
        coords,
        representation=representation,
        camera=camera,
        width=float(width),
        height=pixel_height,
        selection=selection,
    )
    grid: list[list[str]] = [[" "] * width for _ in range(height)]

    def to_cell(x: float, y: float) -> tuple[int, int]:
        return round(float(y) / 2.2), round(float(x))

    for primitive in primitives:
        if isinstance(primitive, BondPrimitive):
            glyph = _ascii_glyph(
                primitive.x2 - primitive.x1, primitive.y2 - primitive.y1
            )
            row_a, col_a = to_cell(primitive.x1, primitive.y1)
            row_b, col_b = to_cell(primitive.x2, primitive.y2)
            if abs(row_b - row_a) >= abs(col_b - col_a):
                span = row_b - row_a
                for row in range(min(row_a, row_b), max(row_a, row_b) + 1):
                    t = 0.0 if span == 0 else (row - row_a) / span
                    col = round(col_a + t * (col_b - col_a))
                    if 0 <= row < height and 0 <= col < width and grid[row][col] == " ":
                        grid[row][col] = glyph
            else:
                span = col_b - col_a
                for col in range(min(col_a, col_b), max(col_a, col_b) + 1):
                    t = 0.0 if span == 0 else (col - col_a) / span
                    row = round(row_a + t * (row_b - row_a))
                    if 0 <= row < height and 0 <= col < width and grid[row][col] == " ":
                        grid[row][col] = glyph
            continue
        row, col = to_cell(primitive.x, primitive.y)
        if spec.cross:
            for d_row, d_col in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                r2, c2 = row + d_row, col + d_col
                if 0 <= r2 < height and 0 <= c2 < width:
                    grid[r2][c2] = "+"
            continue
        text = f"[{primitive.symbol}]" if primitive.selected else primitive.symbol
        if spec.atom_scale >= 0.9:  # spacefill: fill the vdw disc
            radius_cells = max(1.0, primitive.radius)
            for d_row in range(height):
                for d_col in range(width):
                    if grid[d_row][d_col] != " ":
                        continue
                    dx = (d_col - col) / max(radius_cells, 1e-9)
                    dy = (d_row - row) / max(radius_cells / 2.2, 1e-9)
                    if dx * dx + dy * dy <= 1.0:
                        grid[d_row][d_col] = primitive.symbol
        col -= len(text) // 2
        for offset, char in enumerate(text):
            if 0 <= row < height and 0 <= col + offset < width:
                grid[row][col + offset] = char

    for atom_index, text in labels:
        if not 0 <= atom_index < len(mol.atoms):
            raise DataValidationError(
                f"label refers to atom {atom_index} of a {len(mol.atoms)}-atom molecule"
            )
        row, col = to_cell(*_projection(
            mol,
            np.asarray(coords, dtype=float),
            camera or Camera(),
            float(width),
            pixel_height,
        )[0][atom_index])
        col += 2
        for offset, char in enumerate(text):
            if 0 <= row < height and 0 <= col + offset < width:
                grid[row][col + offset] = char

    border = "#" * width
    lines = [border]
    lines.extend("".join(line).rstrip() for line in grid)
    lines.append(border)
    return "\n".join(lines) + "\n"


def view_report(
    mol: Molecule,
    coords: np.ndarray,
    *,
    representation: str = "ball+stick",
    camera: Camera | None = None,
    selection: Iterable[int] | None = None,
    probe_radius: float = 1.4,
) -> str:
    """The Stage as text: what the viewer would show, in numbers.

    >>> from pharmaml.chem.molgraph import parse_smiles
    >>> ethanol = parse_smiles("CCO")
    >>> print(view_report(ethanol, build_coordinates_3d(ethanol)))
    formula C2H6O, 3 atoms
    representation ball+stick (NGL 'ball+stick' / 3Dmol ball-and-stick)
    camera yaw 25.0 pitch 20.0 zoom 1.0 (orthographic)
    selection all (3 of 3)
    SASA 171.56 A^2 (probe 1.4)
    <BLANKLINE>
    """
    camera = camera or Camera()
    rep = REPRESENTATIONS[representation]
    n_atoms = len(mol.atoms)
    selected_count = n_atoms if selection is None else len(set(selection))
    report = sasa(mol, coords, probe_radius=probe_radius)
    lines = [
        f"formula {molecular_formula(mol)}, {n_atoms} atoms",
        f"representation {representation} ({rep.label})",
        (f"camera yaw {camera.yaw:.1f} pitch {camera.pitch:.1f} "
        f"zoom {camera.zoom:.1f} (orthographic)"),
        (f"selection {'all' if selection is None else 'spec'} "
        f"({selected_count} of {n_atoms})"),
        f"SASA {report.total:.2f} A^2 (probe {probe_radius})",
    ]
    return "\n".join(lines) + "\n"
