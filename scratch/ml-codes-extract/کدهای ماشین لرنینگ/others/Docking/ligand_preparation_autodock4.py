import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges
import subprocess
import tempfile

def ligand_to_pdbqt(mol, output_dir, ligand_name):
    """
    Prepare a ligand for AutoDock 4.2 using OpenBabel or RDKit fallback, setting vdW to 1.00 and temp factor to 0.00.
    
    Args:
        mol: RDKit molecule object
        output_dir: Directory to save the output files
        ligand_name: Name of the ligand (used for file naming)
    
    Returns:
        tuple: (success_flag, error_message if any)
    """
    try:
        # Add hydrogens and generate 3D coordinates
        mol_h = Chem.AddHs(mol, addCoords=True)
        embed_result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if embed_result < 0:
            return 0, "Failed to generate 3D coordinates"
        AllChem.MMFFOptimizeMolecule(mol_h)
        
        # Create temporary PDB file
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_pdb:
            Chem.MolToPDBFile(mol_h, temp_pdb.name)
        
        # Prepare PDBQT using OpenBabel
        pdbqt_path = os.path.join(output_dir, f"{ligand_name}.pdbqt")
        obabel_cmd = [
            "obabel",
            "-ipdb", temp_pdb.name,
            "-opdbqt", "-O", pdbqt_path,
            "-xh",  # Add hydrogens
            "-xn",  # Neutralize charges
            "-p", "7.4",  # pH 7.4 for charge neutralization
            "--partialcharge", "gasteiger"  # Compute Gasteiger charges
        ]
        
        result = subprocess.run(obabel_cmd, capture_output=True, text=True)
        
        # Verify and set vdW to 1.00 and temp factor to 0.00
        if result.returncode == 0 and os.path.exists(pdbqt_path):
            with open(pdbqt_path, 'r') as f:
                lines = f.readlines()
            
            with open(pdbqt_path, 'w') as f_out:
                for line in lines:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        # PDBQT format has fixed-width columns
                        # Occupancy is at positions 54-59 (1-based), temp factor at 60-65
                        if len(line) >= 66:  # Ensure line is long enough
                            # Modify the line by replacing specific character positions
                            line_list = list(line)
                            
                            # Set occupancy (vdW) to 1.00 (positions 54-59, 0-based: 53-58)
                            line_list[54:60] = list(" 1.00 ")
                            
                            # Set temp factor to 0.00 (positions 60-65, 0-based: 59-64)
                            line_list[60:66] = list(" 0.00 ")
                            
                            line = ''.join(line_list)
                            print(f"Debug: Modified line - {line.strip()}")  # Debug output
                    
                    f_out.write(line)
            
            # Clean up temporary file
            try:
                os.unlink(temp_pdb.name)
            except:
                pass
                
            return 1, None
        else:
            return 0, f"Error in OpenBabel: {result.stderr}"
    
    except Exception as e:
        return 0, f"Exception during ligand preparation: {str(e)}"

def main():
    print("This script converts SMILES from a CSV or Excel file to PDBQT files for AutoDock 4.2 using OpenBabel.")
    try:
        file_type = int(input("Is the input file CSV (1) or Excel (0)? Enter 0 or 1: "))
    except ValueError:
        print("Error: Please enter 0 for Excel or 1 for CSV.")
        return
    
    SMILES_COL = 'SMILES'  # Column name for SMILES strings
    failed = []
    
    # Read input file
    try:
        if file_type == 1:
            df = pd.read_csv("ligands.csv")
        else:
            df = pd.read_excel("ligands.xlsx")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Ensure SMILES column exists
    if SMILES_COL not in df.columns:
        print(f"Error: '{SMILES_COL}' column not found in the input file.")
        return
    
    # Initialize list for failed SMILES
    failed_smiles = []
    
    # Convert SMILES to RDKit molecules, handling invalid entries
    def safe_mol_from_smiles(smiles):
        if not isinstance(smiles, str) or pd.isna(smiles) or smiles.strip() == "":
            return None, "Invalid or empty SMILES"
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return None, "Failed to parse SMILES"
            return mol, None
        except Exception as e:
            return None, f"SMILES parsing error: {str(e)}"
    
    try:
        df[['mol', 'error']] = df[SMILES_COL].apply(lambda x: pd.Series(safe_mol_from_smiles(x)))
    except Exception as e:
        print(f"Error processing SMILES: {e}")
        return
    
    # Log invalid SMILES
    invalid_smiles = df[df['mol'].isna()]
    if not invalid_smiles.empty:
        print(f"Warning: Invalid SMILES found in rows: {invalid_smiles.index.tolist()}")
        for idx, row in invalid_smiles.iterrows():
            failed_smiles.append({
                'Index': idx,
                'SMILES': row[SMILES_COL],
                'Error': row['error']
            })
            failed.append(str(idx + 1))
    
    # Process each valid molecule
    for idx, row in df.iterrows():
        if row['mol'] is None:
            continue  # Skip invalid SMILES, already logged
        
        # Create output directory
        output_dir = str(idx + 1)  # Folders start from 1
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory {output_dir}: {e}")
            failed_smiles.append({
                'Index': idx,
                'SMILES': row[SMILES_COL],
                'Error': f"Directory creation error: {str(e)}"
            })
            failed.append(output_dir)
            continue
        
        # Prepare ligand
        try:
            success, error_msg = ligand_to_pdbqt(row['mol'], output_dir, output_dir)
        except Exception as e:
            success, error_msg = 0, f"Unexpected error in ligand_to_pdbqt: {str(e)}"
        
        if success:
            print(f"Successfully prepared ligand {output_dir}.pdbqt")
        else:
            print(f"Failed to prepare ligand {output_dir}: {error_msg}")
            failed_smiles.append({
                'Index': idx,
                'SMILES': row[SMILES_COL],
                'Error': error_msg or "Failed during PDBQT conversion"
            })
            failed.append(output_dir)
            try:
                os.removedirs(output_dir)  # Remove empty directory
            except:
                pass
    
    # Save failed SMILES to CSV
    if failed_smiles:
        try:
            failed_df = pd.DataFrame(failed_smiles)
            failed_df.to_csv("failed_smiles.csv", index=False)
            print(f"Failed SMILES saved to 'failed_smiles.csv'")
        except Exception as e:
            print(f"Error saving failed_smiles.csv: {e}")
    
    if failed:
        print(f"Failed ligands: {failed}")
    else:
        print("All valid ligands processed successfully!")

if __name__ == "__main__":
    main()