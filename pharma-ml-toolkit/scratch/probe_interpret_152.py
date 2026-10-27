"""100x-152 probe — the GAP 2 interpretability bundle, verified live.

Reproduces the D2.py ``analyze_correlations`` story end to end with the
new library calls:

1. the pinned decode facts (caffeine's carbonyl bit 314, the FCFP
   flavour bit 418, aspirin's radius-0 bit 650) + bit-identity with the
   legacy ``GetMorganFingerprintAsBitVect`` bitInfo route;
2. a 6-molecule ecfp feature frame whose environment bits separate
   the high-target (carbonyl) molecules -> ``signed_feature_correlations``
   -> ``plot_signed_correlations`` (top-10 signed bar PNG) ->
   ``plot_correlation_heatmap`` (top features + target PNG);
3. the upload's third act: decoding the top correlated Morgan bit back
   to its atom environment (the ``morgan_atom_environments.txt`` story),
   plus the RDKit bit-legend SVG artifact.

Run from the repository root:  .venv/bin/python scratch/probe_interpret_152.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from pharmaml.chem.features import compute_feature_frame
from pharmaml.chem.interpret import decode_morgan_bit, signed_feature_correlations
from pharmaml.viz import save_figure
from pharmaml.viz.ml_plots import (
    plot_correlation_heatmap,
    plot_signed_correlations,
)

OUT = ROOT / "download"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)O2)C"

#: 3 carbonyl-free molecules (low target) + 3 carbonyl carriers (high).
MOLECULES: list[tuple[str, float]] = [
    ("CCO", 1.0),
    ("CCCO", 1.2),
    ("OCCO", 1.1),
    ("CC(=O)C", 7.0),
    ("CC(=O)O", 7.3),
    (CAFFEINE, 8.0),
]


def probe_pinned_bits() -> bool:
    """Act 0: the doctest facts, verified live."""
    info = decode_morgan_bit(CAFFEINE, 314, with_drawing=False)
    assert [(h.atom_index, h.radius) for h in info.hits] == [(7, 1), (10, 1)]
    assert {h.environment_smiles for h in info.hits} == {"c=O"}

    feat = decode_morgan_bit(CAFFEINE, 418, use_features=True, with_drawing=False)
    assert [(h.atom_index, h.radius) for h in feat.hits] == [(4, 2)]

    aspirin = decode_morgan_bit("CC(=O)Oc1ccccc1C(=O)O", 650, with_drawing=False)
    assert [(h.atom_index, h.radius) for h in aspirin.hits] == [(2, 0), (11, 0)]
    assert {h.environment_smiles for h in aspirin.hits} == {"O"}

    # bit-identity with the D2.py original's own legacy call
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(CAFFEINE)
    legacy: dict = {}
    rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=2048, bitInfo=legacy)
    for bit, hits in legacy.items():
        decoded = decode_morgan_bit(CAFFEINE, int(bit), with_drawing=False)
        assert [(h.atom_index, h.radius) for h in decoded.hits] == [
            (int(a), int(r)) for a, r in hits
        ]
    print(f"[1] pinned bits + legacy bitInfo identity OK ({len(legacy)} bits)")
    return True


def probe_bundle() -> bool:
    """Acts 1-3: correlations -> bars -> heatmap -> bit decoding."""
    smiles = [smi for smi, _ in MOLECULES]
    target = [value for _, value in MOLECULES]
    frame, report = compute_feature_frame(smiles, families=("ecfp",))
    assert report.n_valid == 6 and frame.shape[1] == 2048

    corr = signed_feature_correlations(frame, target)
    top = corr.head(10)
    assert abs(float(top.iloc[0])) > 0.9  # near-separating environment bits
    magnitudes = sorted(abs(float(value)) for value in top)
    print(f"[2] signed correlations: {len(corr)} of 2048 bit column(s) ranked "
          f"({2048 - len(corr)} constant dropped); top-10 |r| from "
          f"{magnitudes[0]:.3f} to {magnitudes[-1]:.3f}")

    bar_path = save_figure(
        plot_signed_correlations(top, title="Top 10 features by correlation with target"),
        OUT / "pg152-signed-correlations.png",
    )

    heat_frame = frame[list(top.index[:3])].copy()
    heat_frame["target"] = target
    heat_path = save_figure(
        plot_correlation_heatmap(
            heat_frame, columns=list(top.index[:3]) + ["target"],
            title="Feature correlation heatmap (top 3 + target)"),
        OUT / "pg152-correlation-heatmap.png",
    )

    # the upload's third act: decode the top correlated Morgan bit that
    # the first carbonyl carrier actually sets (the top-|r| bit can be a
    # negative one the carrier never sets)
    carrier = smiles[3]  # CC(=O)C
    carrier_row = frame.iloc[3]
    best_column = next(
        column for column in top.index if float(carrier_row[column]) == 1.0
    )
    bit_index = int(str(best_column).rsplit("_", 1)[1])
    decoded = decode_morgan_bit(carrier, bit_index)
    print(f"[3] top bit {best_column} (r={float(top[best_column]):+.3f}) decodes on "
          f"{carrier!r}: {[(h.atom_index, h.radius) for h in decoded.hits]} "
          f"env={{{', '.join(sorted({h.environment_smiles for h in decoded.hits}))}}}")
    assert decoded.hits
    (OUT / "pg152-morgan-bit-legend.svg").write_text(decoded.drawing or "", encoding="utf-8")
    print(f"[4] artifacts: {bar_path.name}, {heat_path.name}, "
          f"pg152-morgan-bit-legend.svg ({len(decoded.drawing or '')} bytes SVG)")
    return True


if __name__ == "__main__":
    ok = probe_pinned_bits() and probe_bundle()
    print("PROBE152_OK" if ok else "PROBE152_FAILED")
    sys.exit(0 if ok else 1)
