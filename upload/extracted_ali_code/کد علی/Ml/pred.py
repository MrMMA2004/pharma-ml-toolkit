#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for D2 pEC50 prediction model (ENHANCED FEATURE VERSION)
Supports both base models AND meta-learners (loads base models automatically)
Compatible with models trained using enhanced D2EC.py (6,164 raw features)
"""
import pandas as pd
import numpy as np
import pickle
import argparse
import sys
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, MolSurf, Fragments

warnings.filterwarnings('ignore')

def calculate_enhanced_features(smiles_list):
    """
    Generate EXACT features matching enhanced training pipeline:
    - Morgan R2: 4096 bits (radius=2)
    - Morgan R3: 2048 bits (radius=3)
    - 20 extended descriptors (exact order matching training)
    Returns: (feature_matrix, valid_indices, invalid_smiles_list)
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    invalid_smiles = [smiles_list[i] for i, m in enumerate(mols) if m is None]
    
    if invalid_smiles:
        print(f"\n⚠️  Warning: {len(invalid_smiles)} invalid SMILES skipped (first 5 shown):")
        for smi in invalid_smiles[:5]:
            print(f"    {smi}")
    
    mols = [m for m in mols if m is not None]
    
    # Morgan fingerprints (MUST match training parameters exactly)
    print("  → Generating Morgan R2 fingerprints (radius=2, 4096 bits)...")
    fps_r2 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096) for m in mols]
    
    print("  → Generating Morgan R3 fingerprints (radius=3, 2048 bits)...")
    fps_r3 = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mols]
    
    # Extended descriptors (EXACT order matching training pipeline)
    def get_extended_descriptors(mol):
        try:
            return [
                # Core physicochemical
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol), Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol),
                # Lipinski & complexity
                Lipinski.NHOHCount(mol), Lipinski.NOCount(mol),
                Descriptors.NumHeteroatoms(mol), Descriptors.MolMR(mol),
                # Surface properties
                MolSurf.LabuteASA(mol), MolSurf.TPSA(mol),
                # Fragment counts
                Fragments.fr_benzene(mol), Fragments.fr_pyridine(mol),
                Fragments.fr_piperdine(mol), Fragments.fr_morpholine(mol)
            ]
        except Exception as e:
            print(f"    ⚠️  Descriptor calculation failed for molecule: {str(e)}")
            return [0.0] * 20
    
    print("  → Calculating 20 extended molecular descriptors...")
    desc_arr = np.array([get_extended_descriptors(m) for m in mols], dtype=np.float32)
    
    # Stack all features in EXACT training order
    X = np.hstack([
        np.array(fps_r2, dtype=np.float32),
        np.array(fps_r3, dtype=np.float32),
        desc_arr
    ])
    
    print(f"✓ Generated {X.shape[1]} raw features (4096 R2 + 2048 R3 + 20 descriptors)")
    return X, valid_indices, invalid_smiles

def predict_pEC50(input_csv,
                 output_csv="predictions.csv",
                 model_info_pkl="model_info.pkl",
                 model_pkl="best_model.pkl",
                 scaler_pkl="scaler.pkl",
                 selector_pkl="variance_selector.pkl"):
    """
    Predict pEC50 values with rigorous feature validation
    Handles both base models and meta-learners
    """
    # Load model metadata to determine type and expected features
    try:
        with open(model_info_pkl, 'rb') as f:
            model_info = pickle.load(f)
        print(f"\n✅ Loaded model info: {model_info['model_name']}")
        print(f"   Trained on {model_info.get('n_features', 'unknown')} selected features")
        print(f"   Hardware used: {model_info.get('hardware', 'Unknown')}")
        is_meta = model_info.get('is_meta_learner', False)
        expected_raw_features = 6164  # Critical: Must match training pipeline
    except FileNotFoundError:
        print(f"\n❌ Error: {model_info_pkl} not found!")
        print("   Required for feature validation and model type detection")
        return None
    except Exception as e:
        print(f"\n❌ Error loading model info: {str(e)}")
        return None

    # Load input data
    print(f"\n📥 Loading input data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

    # Detect SMILES column
    smiles_col = None
    for candidate in ['SMILES', 'Ligand SMILES', 'smiles', 'ligand_smiles']:
        if candidate in df.columns:
            smiles_col = candidate
            break
    
    if smiles_col is None:
        print(f"❌ Error: CSV must contain a SMILES column. Found columns: {list(df.columns)}")
        print("   Supported column names: 'SMILES', 'Ligand SMILES', 'smiles', 'ligand_smiles'")
        return None
    
    print(f"✓ Using SMILES column: '{smiles_col}'")
    smiles_list = df[smiles_col].astype(str).dropna().tolist()
    print(f"✓ Processing {len(smiles_list)} compounds...")

    # Generate RAW features (MUST match training exactly)
    X_raw, valid_idx, invalid_smiles = calculate_enhanced_features(smiles_list)
    
    # CRITICAL VALIDATION: Check raw feature count matches training
    if X_raw.shape[1] != expected_raw_features:
        print(f"\n❌ FATAL ERROR: Feature dimension mismatch!")
        print(f"   Generated: {X_raw.shape[1]} raw features")
        print(f"   Expected:  {expected_raw_features} raw features (from training pipeline)")
        print("\n   This means your inference script does NOT match the training pipeline.")
        print("   Possible causes:")
        print("     • Using old predEC.py with new model")
        print("     • Training used different fingerprint parameters")
        print("     • RDKit version mismatch between training/inference")
        return None
    
    print(f"✓ Raw feature count validated: {X_raw.shape[1]} features")

    # Load preprocessing objects
    try:
        print("\n⚙️  Loading preprocessing artifacts...")
        with open(selector_pkl, 'rb') as f:
            selector = pickle.load(f)
        with open(scaler_pkl, 'rb') as f:
            scaler = pickle.load(f)
        with open(model_pkl, 'rb') as f:
            final_model = pickle.load(f)
        print("✓ Loaded variance selector, scaler, and prediction model")
    except FileNotFoundError as e:
        print(f"\n❌ Error loading pickle files: {e}")
        print("\nRequired files:")
        print("  - best_model.pkl")
        print("  - scaler.pkl")
        print("  - variance_selector.pkl")
        if is_meta:
            print("  - base_model_SVR.pkl")
            print("  - base_model_GradientBoosting.pkl")
            print("  - base_model_XGBoost.pkl")
            print("  - base_model_LGBM.pkl")
            print("  - base_model_CatBoost.pkl")
        return None
    except Exception as e:
        print(f"\n❌ Error loading artifacts: {str(e)}")
        return None

    # Apply preprocessing pipeline (MUST match training order)
    print("\n🔧 Applying preprocessing pipeline...")
    X_sel = selector.transform(X_raw)
    print(f"  → After variance threshold: {X_sel.shape[1]} features")
    
    X_scaled = scaler.transform(X_sel)
    print(f"  → After standardization: {X_scaled.shape[1]} features")
    
    # Verify feature count matches model expectation
    expected_model_features = model_info.get('n_features')
    if expected_model_features and X_scaled.shape[1] != expected_model_features:
        print(f"\n❌ FATAL ERROR: Preprocessed feature mismatch!")
        print(f"   Model expects: {expected_model_features} features")
        print(f"   Got:           {X_scaled.shape[1]} features")
        print("\n   Check variance_selector.pkl matches training version")
        return None

    # PREDICTION LOGIC: Branch based on model type
    if is_meta:
        print(f"\n🧠 Meta-learner detected ({model_info['model_name']}). Generating meta-features...")
        base_model_names = model_info['base_models']
        meta_features = np.zeros((X_scaled.shape[0], len(base_model_names)))
        
        # Load each base model and generate predictions
        for i, name in enumerate(base_model_names):
            try:
                with open(f'base_model_{name}.pkl', 'rb') as f:
                    base_model = pickle.load(f)
                meta_features[:, i] = base_model.predict(X_scaled)
                print(f"  ✓ Generated predictions from {name}")
            except FileNotFoundError:
                print(f"\n❌ ERROR: Base model file 'base_model_{name}.pkl' not found!")
                print("   Meta-learners require ALL base models used during training.")
                print("\n   Solution options:")
                print("     1. Ensure all base_model_*.pkl files are in working directory")
                print("     2. Re-run training to regenerate missing base models")
                print("     3. Use a base model directly (check model_info.pkl for alternatives)")
                return None
            except Exception as e:
                print(f"\n❌ Error loading/predicting with base model {name}: {str(e)}")
                return None
        
        # Predict using meta-learner
        print(f"\n🎯 Predicting with meta-learner using {meta_features.shape[1]} meta-features...")
        predictions = final_model.predict(meta_features)
        print(f"✓ Meta-learner prediction shape: {predictions.shape}")
    else:
        # Base model: predict directly from chemical features
        print(f"\n🎯 Base model detected ({model_info.get('model_name', 'unknown')}). Predicting directly...")
        predictions = final_model.predict(X_scaled)
        print(f"✓ Prediction shape: {predictions.shape}")

    # Create results dataframe
    results_df = df.iloc[valid_idx].copy()
    results_df['pEC50_pred'] = predictions
    results_df['EC50_pred_nM'] = 10**(9 - predictions)
    
    # Add invalid SMILES entries with error flags
    if invalid_smiles:
        invalid_df = pd.DataFrame({
            smiles_col: invalid_smiles,
            'pEC50_pred': [np.nan] * len(invalid_smiles),
            'EC50_pred_nM': [np.nan] * len(invalid_smiles),
            'Error': ['Invalid SMILES'] * len(invalid_smiles)
        })
        # Preserve original columns where possible
        for col in df.columns:
            if col not in invalid_df.columns and col != smiles_col:
                invalid_df[col] = [np.nan] * len(invalid_smiles)
        results_df = pd.concat([results_df, invalid_df], ignore_index=True)
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to: {output_csv}")
    
    # Print summary statistics (only valid predictions)
    valid_mask = ~np.isnan(predictions)
    if np.any(valid_mask):
        valid_preds = predictions[valid_mask]
        print(f"\n📊 Prediction Summary (n={len(valid_preds)} valid compounds):")
        print(f"   Mean pEC50:  {valid_preds.mean():.3f} ± {valid_preds.std():.3f}")
        print(f"   Range:       [{valid_preds.min():.3f}, {valid_preds.max():.3f}]")
        print(f"   Mean EC50:   {10**(9 - valid_preds.mean()):,.1f} nM")
        print(f"   EC50 Range:  [{10**(9 - valid_preds.max()):,.1f}, {10**(9 - valid_preds.min()):,.1f}] nM")
    else:
        print("\n⚠️  No valid predictions generated!")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(
        description='Predict pEC50 values for D2 receptor ligands (ENHANCED FEATURE VERSION)',
        epilog='Note: Requires models trained with enhanced D2EC.py (6,164 raw features)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_csv', help='Input CSV file with SMILES column')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file (default: predictions.csv)')
    parser.add_argument('--model_info', default='model_info.pkl', help='Model info pickle file (default: model_info.pkl)')
    parser.add_argument('--model', default='best_model.pkl', help='Model pickle file (default: best_model.pkl)')
    parser.add_argument('--scaler', default='scaler.pkl', help='Scaler pickle file (default: scaler.pkl)')
    parser.add_argument('--selector', default='variance_selector.pkl', help='Variance selector pickle file (default: variance_selector.pkl)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("D2 RECEPTOR pEC50 PREDICTION PIPELINE (ENHANCED FEATURES)")
    print("="*70)
    print(f"Input file:  {args.input_csv}")
    print(f"Output file: {args.output}")
    print("="*70)
    
    results = predict_pEC50(
        input_csv=args.input_csv,
        output_csv=args.output,
        model_info_pkl=args.model_info,
        model_pkl=args.model,
        scaler_pkl=args.scaler,
        selector_pkl=args.selector
    )
    
    if results is not None:
        print("\n" + "="*70)
        print("✅ INFERENCE COMPLETED SUCCESSFULLY")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ INFERENCE FAILED - CHECK ERROR MESSAGES ABOVE")
        print("="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()