# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:49:43 2025

@author: Dr. Razzaghi
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter

# Load SMILES file
df = pd.read_csv("yyy.csv")  # assumes a column named "SMILES"

# Create SDF writer
sdf_writer = SDWriter("ligands.sdf")

for i, smi in enumerate(df["SMILES"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"Invalid SMILES at index {i}: {smi}")
        continue

    mol = Chem.AddHs(mol)  # add hydrogens
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # 3D coordinates
    AllChem.UFFOptimizeMolecule(mol)  # energy minimization

    # Optionally set molecule name for clarity in PyRx
    mol.SetProp("_Name", f"ligand_{i+1}")

    # Write molecule to SDF
    sdf_writer.write(mol)
    print(f"Added ligand_{i+1} to ligands.sdf")

sdf_writer.close()
print("All molecules saved to ligands.sdf")


