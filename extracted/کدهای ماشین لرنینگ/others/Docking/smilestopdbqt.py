import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy

# --- تعریف نگاشت تایپ‌ها ---
TYPE_MAP = {
    "CG0": "C", "CG": "C", "G0": "C", "G": "C",
    "OG0": "OA", "OG": "OA",
    "NG0": "N", "NG": "N"
}

# تایپ‌های مجاز در AutoDock
VALID_AD4_TYPES = {
    "C", "A", "N", "OA", "SA", "HD", "NA",
    "F", "Cl", "Br", "I", "P", "S", "Zn", "Fe", "Mg", "Ca", "Mn"
}

def ligand_prep(ligmol, out_path):
    try:
        mol_Hs = Chem.AddHs(ligmol, addCoords=True)
        AllChem.EmbedMolecule(mol_Hs, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol_Hs)

        preparator = MoleculePreparation(merge_these_atom_types=("H",))
        mol_setup = preparator.prepare(mol_Hs)[0]
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setup)

        if not is_ok:
            return False, f"Meeko Error: {error_msg}"

        lines = pdbqt_string.splitlines()
        fixed_lines = []
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_type = line[77:].strip()
                if atom_type not in VALID_AD4_TYPES:
                    if atom_type in TYPE_MAP:
                        corrected = TYPE_MAP[atom_type]
                        print(f"ℹ️ {atom_type} → {corrected}")
                        line = line[:77] + corrected.rjust(3)
                    else:
                        print(f"⚠️ Unknown atom type {atom_type} skipped.")
                        continue
            fixed_lines.append(line)

        with open(os.path.join(out_path, "l.pdbqt"), 'w') as f:
            f.write("\n".join(fixed_lines) + "\n")

        return True, "OK"

    except Exception as e:
        return False, str(e)

# === اجرای کلی برای CSV ===
print("🚀 SMILES to PDBQT batch converter (AutoDock compatible)")
text_or_csv = int(input("Are SMILES in a text file or csv? Enter 0 or 1 respectively: "))

SMILES_COL = 'SMILES'
failed = []

if text_or_csv == 1:
    df = pd.read_csv(r"./ligands.csv")
    df['mol'] = df[SMILES_COL].apply(Chem.MolFromSmiles)
    mol_index = list(df.columns).index('mol')

    for idx, row in enumerate(df.values):
        folder = str(idx)
        try:
            os.makedirs(folder, exist_ok=True)
            success, message = ligand_prep(row[mol_index], out_path=folder)
            if not success:
                failed.append((idx, message))
                os.rmdir(folder)
        except Exception as e:
            failed.append((idx, str(e)))
            if os.path.exists(folder):
                os.rmdir(folder)
else:
    print("❌ فقط CSV پشتیبانی می‌شود.")

# === گزارش نهایی ===
if failed:
    print("\n❌ لیست SMILESهایی که تبدیل نشدند:")
    for idx, msg in failed:
        print(f"  - Index {idx}: {msg}")
else:
    print("\n✅ همه لیگاندها با موفقیت تبدیل شدند.")
