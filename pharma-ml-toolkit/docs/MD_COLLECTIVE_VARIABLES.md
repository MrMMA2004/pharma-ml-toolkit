# Collective variables (PLUMED-style)

Enhanced-sampling methods — metadynamics, umbrella sampling, steered MD —
bias a system along a small number of *collective variables* (CVs):
named functions of atom coordinates whose time series is the bias target.
The `plumed/plumed2` stack defines the field's vocabulary: a `plumed.dat`
input declaring one CV per line, and a `COLVAR` table carrying the CV
values per frame. `pharmaml.md.cv` is that vocabulary in plain NumPy over
the toolkit's own `Trajectory` frames.

```python
from pharmaml.md import cv_distance, cv_table
from pharmaml.md.trajectory import synthetic_trajectory

traj = synthetic_trajectory(n_frames=64, seed=0)
distance = cv_distance(traj, atoms=[0, 1])   # the DISTANCE CV (Å per frame)
```

## The six CV actions

| Action | Function | Notes |
|---|---|---|
| `DISTANCE` | `cv_distance(traj, atoms=[i, j])` | Two distinct atoms, Å per frame. |
| `ANGLE` | `cv_angle(traj, atoms=[i, j, k])` | Angle at the middle atom `j`, degrees. Coincident atoms are refused — the angle is undefined. |
| `DIHEDRAL` | `cv_dihedral(traj, atoms=[i, j, k, l])` | Signed torsion of the `j–k` bond (0° eclipsed, ±180° trans). |
| `COORDINATION` | `cv_coordination(traj, group_a, group_b, r_0=…)` | Fraction of groupA–groupB pairs within `r_0` — the honest hard-switch subset; PLUMED's rational switching keywords are refused, not approximated. |
| `RMSD` | `cv_rmsd(traj, atoms, reference=None)` | Kabsch-aligned RMSD; the reference defaults to the trajectory's own first frame. |
| `GYRATION` | `cv_gyration(traj, atoms, masses=None)` | Radius of gyration (`gmx gyrate`); mass-weighted via `masses`. |

All six refuse invalid input loudly (out-of-range indices, repeated atoms,
non-positive `r_0`, empty groups) and never import matplotlib — a CV
feed is analysis input, not a figure.

## The COLVAR table

`cv_table` assembles named CV series onto one validated time axis — every
series must match the trajectory's frame count exactly (the same
axes-must-match rule the correlation twins enforce; a ragged COLVAR table
is a broken export). The time axis defaults to the trajectory's own
`time` when it carries one, else the frame index.

```python
from pharmaml.md import cv_angle, cv_distance, cv_table

table = cv_table(traj, {
    "d": cv_distance(traj, atoms=[0, 1]),
    "ang": cv_angle(traj, atoms=[0, 1, 2]),
})
table.to_colvar()        # '#! FIELDS time d ang\n...' — PLUMED's COLVAR text
table.to_dict()          # JSON-ready payload for bridges
table.xvg_series("d")    # one CV as an XvgSeries
```

CV tables must be finite: NaN or ±inf in a CV series is refused outright.
Figures may tolerate-and-disclose NaN; a bias target a metadynamics run
steers on may not be NaN.

## The `plumed.dat` mini-parser

`parse_plumed_cvs` reads a real run's declaration block and computes
every CV over the trajectory. Atom numbering is PLUMED's 1-based
convention — `ATOMS=1,2` means the toolkit's `(0, 1)` — and
`ATOMS=1,3-5` expands to atoms 1, 3, 4, 5.

```python
from pharmaml.md import parse_plumed_cvs
from pharmaml.md.trajectory import synthetic_trajectory

traj = synthetic_trajectory(n_frames=64, seed=0)
table = parse_plumed_cvs(
    traj,
    "\n".join(
        [
            "# a real-ish plumed.dat",
            "d: DISTANCE ATOMS=1,2",
            "coord: COORDINATION GROUPA=1-3 GROUPB=4-6 R_0=5.0",
            "rms: RMSD ATOMS=1-10",
            "rg: GYRATION ATOMS=1-30 MASS",
        ]
    ),
)
```

`MASS` (GYRATION's flag) weights by the topology's masses and is refused
when the trajectory carries none. Everything unsupported is refused with
a message that names what is supported — `METAD`, `REFERENCE=` files,
the rational switching keywords — never silently ignored.

## Plotting a CV

A CV column flows into the shipped MD viz twins exactly like any parsed
`.xvg` metric series:

```python
from pharmaml.viz import plot_md_series

stats = plot_md_series(table.xvg_series("d"), path="cv_distance.png")
assert stats["n"] == 64
```

See `docs/MD_VISUALIZATION.md` for the full series and correlation twin
API.
