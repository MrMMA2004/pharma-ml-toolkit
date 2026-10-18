import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.DataStructs import TanimotoSimilarity
from tqdm import tqdm
import os

# ---------------- Config ----------------
REFERENCE_FILE = r"C:\Users\lenovo\new\Drugs.csv"
TARGET_FILE = r"C:\Users\lenovo\new\zinc3.csv"
OUTPUT_CSV = r"C:\Users\lenovo\new\tanimatoedzinc3.csv"
CHUNK_LOG = r"C:\Users\lenovo\new\chunk_log.txt"

TARGET_CHUNK_SIZE = 50_000
TANIMOTO_THRESHOLD = 0.9
# ----------------------------------------

# ----------- Load reference SMILES -----------
df_ref = pd.read_csv(REFERENCE_FILE, usecols=["SMILES_REF"])
df_ref.columns = df_ref.columns.str.strip()

if "SMILES_REF" not in df_ref.columns:
    raise KeyError("Column 'SMILES_REF' not found in reference file!")

reference_list = df_ref["SMILES_REF"].dropna().astype(str).tolist()

# ---------- Resume: Read last processed chunk ----------
start_chunk = 1
if os.path.exists(CHUNK_LOG):
    with open(CHUNK_LOG, "r") as f:
        last_chunk = f.read().strip()
        if last_chunk.isdigit():
            start_chunk = int(last_chunk) + 1

# ----------- Read targets in chunks -----------
target_reader = pd.read_csv(TARGET_FILE, usecols=["SMILES"], chunksize=TARGET_CHUNK_SIZE)

# ----------- Skip already processed chunks -----------
for _ in range(start_chunk - 1):
    try:
        next(target_reader)
    except StopIteration:
        print("✅ All chunks have already been processed.")
        exit()

print(f"📌 Starting from chunk #{start_chunk}")

# ----------- Process remaining chunks -----------
chunk_num = start_chunk
for df_chunk in target_reader:
    print(f"\n🔁 Processing chunk #{chunk_num}…")
    df_chunk.columns = df_chunk.columns.str.strip()

    target_list = df_chunk["SMILES"].dropna().astype(str).tolist()

    # Precompute target fingerprints
    target_fps = []
    for smi in target_list:
        mol = Chem.MolFromSmiles(smi)
        fp = RDKFingerprint(mol) if mol else None
        target_fps.append(fp)

    results = []

    # Compare each reference with each target in chunk
    for ref_smi in tqdm(reference_list, desc=f"Chunk {chunk_num}: refs", unit="ref"):
        ref_mol = Chem.MolFromSmiles(ref_smi)
        if ref_mol is None:
            continue
        ref_fp = RDKFingerprint(ref_mol)

        for tgt_smi, tgt_fp in zip(target_list, target_fps):
            if tgt_fp is None:
                continue
            sim = TanimotoSimilarity(ref_fp, tgt_fp)
            if sim >= TANIMOTO_THRESHOLD:
                results.append({
                    "SMILES_REF": ref_smi,
                    "SMILES_TARGET": tgt_smi,
                    "Tanimoto": sim
                })

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_CSV, mode='a', index=False, header=not os.path.exists(OUTPUT_CSV))

    # Log current chunk
    with open(CHUNK_LOG, "w") as f:
        f.write(str(chunk_num))

    print(f"✅ Finished chunk #{chunk_num}. Results appended.")

    chunk_num += 1

print(f"\n🎉 Done! All filtered results saved to:\n{OUTPUT_CSV}")
print("💾 You can resume from the last successful chunk if interrupted.")
