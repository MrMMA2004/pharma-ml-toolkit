import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from openbabel import pybel
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ------------------- CONFIG -------------------
csv_path = r"C:\Users\alimo\Desktop\Docking\ligands.csv"
num_confs = 50
pdb_out_dir = 'output_pdb'
mol2_out_dir = 'output_mol2'
log_file = 'unsupported_molecules.log'
default_b_factor = 20.0
unsupported_atoms = {'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Ni', 'Mo', 'W'}
num_processes = max(1, cpu_count() - 1)  # leave one core free

# Create output directories
for d in [pdb_out_dir, mol2_out_dir]:
    os.makedirs(d, exist_ok=True)

# Clear log
with open(log_file, 'w') as log:
    log.write("Unsupported molecules:\n\n")

# ------------------- FUNCTIONS -------------------
def has_unsupported_atoms(mol):
    return any(atom.GetSymbol() in unsupported_atoms for atom in mol.GetAtoms())

def generate_pdb_block(mol, name="LIGAND", conf_id=0, default_b_factor=20.0):
    """Generate PDB block with atom names and connectivity."""
    conf = mol.GetConformer(conf_id)
    pdb_lines = []

    # HEADER and COMPND
    pdb_lines.append(
        "HEADER    LIGAND                                  " +
        datetime.now().strftime("%d-%b-%y").upper() + "   " + name[:3].upper()
    )
    pdb_lines.append(f"COMPND    {name}")

    # ATOM lines
    res_name = name[:3].upper()
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_name = f"{atom.GetSymbol()}{i+1}"
        pdb_lines.append(
            "ATOM  {:5d} {:>4s} {:3s} A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00 {:5.2f}           {:>2s}".format(
                i+1, atom_name, res_name, 1,
                pos.x, pos.y, pos.z, default_b_factor, atom.GetSymbol()
            )
        )

    # Connectivity
    conect_records = {}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx() + 1
        a2 = bond.GetEndAtomIdx() + 1
        conect_records.setdefault(a1, set()).add(a2)
        conect_records.setdefault(a2, set()).add(a1)

    for atom_idx, bonded_atoms in conect_records.items():
        pdb_lines.append("CONECT{:>5d}{}".format(
            atom_idx,
            "".join(f"{b:>5d}" for b in sorted(bonded_atoms))
        ))

    pdb_lines.append("END")
    return "\n".join(pdb_lines) + "\n"

# ------------------- PROCESS SINGLE LIGAND -------------------
def process_ligand(args):
    i, smi, name = args
    enumerator = rdMolStandardize.TautomerEnumerator()
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        mol.SetProp('_Name', name)
    except Exception as e:
        return f"{i}: {name} ({smi}) --> SMILES parse/sanitize failed: {str(e)}"

    # Canonical tautomer
    try:
        mol = enumerator.Canonicalize(mol)
    except Exception as e:
        return f"{i}: {name} ({smi}) --> Tautomer canonicalization failed: {str(e)}"

    # Skip unsupported atoms
    if has_unsupported_atoms(mol):
        return f"{i}: {name} ({smi}) --> Skipped due to unsupported atoms"

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Embed conformers
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDG())
        if not cids:
            raise ValueError("Embedding returned no conformers")
    except Exception as e:
        return f"{i}: {name} ({smi}) --> Embedding failed: {str(e)}"

    # Force field optimization
    forcefield_used = None
    optimized = False

    # MMFF
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        energies = []
        for cid in cids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                ff.Minimize()
                energies.append((cid, ff.CalcEnergy()))
            except:
                continue
        if energies:
            lowest = min(energies, key=lambda x: x[1])[0]
            forcefield_used = "RDKit MMFF"
            optimized = True

    # UFF fallback
    if not optimized:
        energies = []
        for cid in cids:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=cid)
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                energies.append((cid, ff.CalcEnergy()))
            except:
                continue
        if energies:
            lowest = min(energies, key=lambda x: x[1])[0]
            forcefield_used = "RDKit UFF"
            optimized = True

    if not optimized:
        return f"{i}: {name} ({smi}) --> Optimization failed for all conformers"

    # Save lowest-energy conformer
    conf = mol.GetConformer(lowest)
    best_conf_mol = Chem.Mol(mol)
    best_conf_mol.RemoveAllConformers()
    best_conf_mol.AddConformer(conf, assignId=True)
    best_conf_mol.SetProp('_Name', name)

    # ✅ Sanitize name for filenames
    safe_name = name.replace(" ", "_")

    # PDB via RDKit
    pdb_path = os.path.join(pdb_out_dir, f'{safe_name}.pdb')
    pdb_block = generate_pdb_block(best_conf_mol, safe_name, conf_id=0, default_b_factor=default_b_factor)
    with open(pdb_path, 'w') as f:
        f.write(pdb_block)

    # MOL2 via OpenBabel
    try:
        mol_block = Chem.MolToMolBlock(best_conf_mol)
        ob_mol = pybel.readstring("mol", mol_block)
        ob_mol.addh()  # ensure hydrogens
        ob_mol.calccharges(model="gasteiger")
        mol2_path = os.path.join(mol2_out_dir, f'{safe_name}.mol2')
        ob_mol.write("mol2", mol2_path, overwrite=True)
    except Exception as e:
        return f"{i}: {name} ({smi}) --> MOL2 generation failed: {str(e)}"

    return f"[{safe_name}] tautomer fixed, optimized with {forcefield_used}, saved as PDB & MOL2."

# ------------------- LOAD CSV -------------------
df = pd.read_csv(csv_path)
if 'SMILES' not in df.columns or 'name' not in df.columns:
    raise ValueError("CSV must contain 'SMILES' and 'name' columns.")

# ------------------- PARALLEL PROCESSING -------------------
if __name__ == "__main__":
    ligand_args = list(enumerate(zip(df['SMILES'], df['name']), start=1))
    ligand_args = [(i, smi, name) for i, (smi, name) in ligand_args]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_ligand, ligand_args)

    # ------------------- LOG RESULTS -------------------
    with open(log_file, 'a') as log:
        for res in results:
            if "tautomer fixed" not in res:   # Only log failed/unsupported molecules
                log.write(res + "\n")
            print(res)
