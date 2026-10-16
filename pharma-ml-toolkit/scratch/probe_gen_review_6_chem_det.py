"""100x-21-b probe 6: chemistry edges + determinism hunts.

Own hunts: similarity with stereo/aromatic-variant seeds, charged /
zwitterion scoring, n_layers=2 determinism (GRU dropout check), GA
determinism under torch intra-op threads, MultiObjective breakdown key
collision when a child is literally named 'name_2'.
"""
from __future__ import annotations

import torch

from pharmaml.generative import (
    DEFAULT_CORPUS,
    LogPBandObjective,
    MolecularWeightBandObjective,
    MultiObjective,
    QedObjective,
    ScoringObjective,
    SimilarityObjective,
    SmilesRnn,
    TpsaBandObjective,
)

print("== similarity seeds that sanitize differently ==")
aspirin = "CC(=O)Oc1ccccc1C(=O)O"
for seed in [
    aspirin,
    "O=C(O)c1ccccc1OC(C)=O",           # same molecule, different spelling
    "CC(=O)Oc1ccccc1C(=O)O",           # identical
    "CC(=O)O[C@H]...",                  # invalid -> constructor error expected
    "C/C=C/C",                           # stereo retained by canonicalization
    "C/C=C\\C",                           # the other stereoisomer
]:
    try:
        obj = SimilarityObjective(seed)
        print(f"seed {seed!r:40s} -> canonical {obj.seed_smiles!r} "
              f"self-sim={obj.score(obj.seed_smiles):.3f}")
    except Exception as exc:
        print(f"seed {seed!r:40s} -> {type(exc).__name__}: {exc}")

print("\n== charged / zwitterion scoring ==")
charged = [
    "C[N+](C)(C)CC(=O)[O-]",       # betaine zwitterion (in corpus)
    "CC(=O)[O-]",                  # acetate anion
    "C[NH3+]",                     # methylammonium
    "N[C@@H](CCC(=O)[O-])C(=O)[O-]",  # aspartate dianion
    "[O-]C(=O)[CH2]1CC[C@H](N)CC1",   # charged ring amine
    "C[N+](C)(C)CCO",              # choline cation (in corpus)
]
for smi in charged:
    qed = QedObjective().score(smi)
    logp = LogPBandObjective(1.0, 3.0).score(smi)
    mw = MolecularWeightBandObjective(150, 450).score(smi)
    tpsa = TpsaBandObjective(30, 90).score(smi)
    print(f"{smi:38s} qed={qed:.3f} logp_band={logp:.3f} "
          f"mw_band={mw:.3f} tpsa_band={tpsa:.3f}")

print("\n== MultiObjective breakdown key collision 'name_2' ==")


class NamedObjective(ScoringObjective):
    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value

    def score(self, smiles):
        return self.value


obj = MultiObjective([NamedObjective("dup", 0.5), NamedObjective("dup", 0.5),
                      NamedObjective("dup_2", 0.4)])
total, breakdown = obj.score_with_breakdown("CCO")
print("breakdown:", breakdown, "total:", round(total, 4))

print("\n== n_layers=2 determinism (GRU dropout check) ==")
torch.manual_seed(0)
a = SmilesRnn(n_layers=2, seed=7)
a.train_prior(DEFAULT_CORPUS[:60], epochs=1)
s1 = a.sample(8, temperature=1.0, seed=99)
s2 = a.sample(8, temperature=1.0, seed=99)
print("same-process same-seed identical:", s1 == s2)

# cross-process style: fresh interpreter simulation via a second object
# trained identically (dropout would consume global RNG if active)
b = SmilesRnn(n_layers=2, seed=7)
b.train_prior(DEFAULT_CORPUS[:60], epochs=1)
s3 = b.sample(8, temperature=1.0, seed=99)
print("two identically-seeded n_layers=2 models sample identically:", s1 == s3)
print("GRU dropout attr:", a._model.gru.dropout)

print("\n== torch intra-op thread determinism (GA-free check) ==")
torch.set_num_threads(2)
c = SmilesRnn(seed=7)
c.train_prior(DEFAULT_CORPUS[:60], epochs=1)
t1 = c.sample(8, temperature=1.0, seed=99)
torch.set_num_threads(1)
d = SmilesRnn(seed=7)
d.train_prior(DEFAULT_CORPUS[:60], epochs=1)
t2 = d.sample(8, temperature=1.0, seed=99)
print("sample identical across torch thread counts:", t1 == t2)
torch.set_num_threads(2)
print("probe 6 complete")
