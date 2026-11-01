"""Round 168 redo: generate exact doctest fixtures for chem/structure.py.

Builds an idealised 8-residue polyalanine alpha-helix (NeRF placement,
phi=-57, psi=-47, omega=180) plus two waters with high B-factors,
emits the PDB block, and prints the real values every doctest needs.
"""

from __future__ import annotations

import numpy as np

BOND = {("N", "CA"): 1.458, ("CA", "C"): 1.525, ("C", "N"): 1.329, ("C", "O"): 1.231}
ANG = {
    ("N", "CA", "C"): 111.2,
    ("CA", "C", "N"): 116.2,
    ("C", "N", "CA"): 121.7,
    ("CA", "C", "O"): 120.8,
}


def place(a, b, c, bond, angle_deg, tors_deg):
    """NeRF: place d at bond from c, angle(b,c,d), torsion(a,b,c,d)."""
    ang = np.radians(angle_deg)
    tor = np.radians(tors_deg)
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n = n / np.linalg.norm(n)
    m = np.cross(n, bc)
    d_local = np.array(
        [-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)]
    )
    frame = np.column_stack([bc, m, n])
    return c + frame @ d_local


def build_helix(n_res=8, phi=-57.0, psi=-47.0, seed_xyz=(0.0, 0.0, 0.0)):
    atoms = []  # (name, xyz)
    n1 = np.array(seed_xyz, dtype=float)
    ca1 = n1 + np.array([BOND[("N", "CA")], 0.0, 0.0])
    c1 = place(n1, ca1, n1 + np.array([0.0, 0.0, 0.0]), 0.1, 90.0, 0.0)  # placeholder
    # place C1 with angle N-CA-C in the xy-plane (torsion irrelevant for first)
    back = n1 - ca1
    bc = ca1 - n1
    # construct C1 explicitly: angle(n1, ca1, c1) = 111.2, dihedral undefined -> use place with dummy a
    a_dummy = n1 + np.array([0.0, 1.0, 0.0])
    c1 = place(a_dummy, n1, ca1, BOND[("CA", "C")], ANG[("N", "CA", "C")], 180.0)
    o1 = place(n1, ca1, c1, BOND[("C", "O")], ANG[("CA", "C", "O")], psi + 180.0)
    atoms += [("N", n1), ("CA", ca1), ("C", c1), ("O", o1)]
    prev = [n1, ca1, c1, o1]
    for res in range(2, n_res + 1):
        n_prev, ca_prev, c_prev, o_prev = prev
        n_new = place(
            n_prev, ca_prev, c_prev, BOND[("C", "N")], ANG[("CA", "C", "N")], psi
        )
        ca_new = place(
            ca_prev, c_prev, n_new, BOND[("N", "CA")], ANG[("C", "N", "CA")], 180.0
        )
        c_new = place(
            c_prev, n_new, ca_new, BOND[("CA", "C")], ANG[("N", "CA", "C")], phi
        )
        o_new = place(
            n_new, ca_new, c_new, BOND[("C", "O")], ANG[("CA", "C", "O")], psi + 180.0
        )
        atoms += [("N", n_new), ("CA", ca_new), ("C", c_new), ("O", o_new)]
        prev = [n_new, ca_new, c_new, o_new]
    return atoms


def pdb_block(atoms, resname="ALA", chain="A", bfac=30.0):
    lines = []
    serial = 1
    for i, (name, xyz) in enumerate(atoms):
        resseq = i // 4 + 1
        element = "C" if name in ("CA", "C") else "N" if name == "N" else "O"
        lines.append(
            f"ATOM  {serial:5d} {name:>4s} {resname:>3s} {chain:1s}{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.00:6.2f}{bfac:6.2f}"
            f"          {element:>2s}  "
        )
        serial += 1
    # two waters, chain B, B=55
    waters = [(20.0, 10.0, 5.0), (21.5, 10.4, 5.9)]
    for j, w in enumerate(waters, start=1):
        lines.append(
            f"HETATM{serial:5d}  O   HOH B{ j:4d}    "
            f"{w[0]:8.3f}{w[1]:8.3f}{w[2]:8.3f}{1.00:6.2f}{55.00:6.2f}"
            f"          O  "
        )
        serial += 1
    return "\n".join(lines) + "\nEND\n"


def dihedral(p0, p1, p2, p3):
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


atoms = build_helix()
block = pdb_block(atoms)
print(block)
print("N_ATOMS", len(atoms))
# phi of residue 3: (C2, N3, CA3, C3) -- indices: res r atoms at (r-1)*4
phi3 = dihedral(atoms[4][1], atoms[8][1], atoms[9][1], atoms[10][1])
print("PHI3", round(float(phi3), 1))
psi3 = dihedral(atoms[9][1], atoms[10][1], atoms[11][1], atoms[12][1])
print("PSI3", round(float(psi3), 1))
# helix radius / H-bond check: distance N(i+4) ... O(i)
for i in range(5):
    o_i = atoms[i * 4 + 3][1]
    n_i4 = atoms[(i + 4) * 4][1]
    print(f"HBOND O{i+1}-N{i+5}", round(float(np.linalg.norm(o_i - n_i4)), 2))
with open("scratch/helix_168.pdb", "w") as fh:
    fh.write(block)
