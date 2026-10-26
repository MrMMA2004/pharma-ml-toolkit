import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, MolSurf, Fragments
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_validate
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm.auto import tqdm
import warnings
import logging
import sys
import pickle
import os
import time

# Hardware Optimization
N_JOBS = 16  # Matches your 8-core/16-thread CPU
warnings.filterwarnings('ignore')
tqdm.pandas()
np.random.seed(42)
sns.set_style("whitegrid")

# Logging Setup
log_filename = "model_performance.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

def get_extended_descriptors(mol):
    """Compute comprehensive molecular descriptors with error handling"""
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
        logging.warning(f"Descriptor calculation failed: {str(e)}")
        return [0.0] * 20

def calculate_enhanced_features(smiles_list):
    """Generate high-resolution molecular representations"""
    logging.info("Generating Enhanced Molecular Features (Morgan 4096 + Extended Descriptors)...")
    start_time = time.time()
    
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    mols = [m for m in mols if m is not None]
    valid_smiles = np.array(smiles_list)[valid_indices]
    
    # High-resolution fingerprints (2x representations)
    logging.info("  → Generating Morgan fingerprints (radius 2, 4096 bits)...")
    fps_morgan_r2 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096) for m in mols]
    
    logging.info("  → Generating Morgan fingerprints (radius 3, 2048 bits)...")
    fps_morgan_r3 = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mols]
    
    # Extended descriptors
    logging.info("  → Calculating extended molecular descriptors...")
    desc_list = [get_extended_descriptors(m) for m in mols]
    
    # Stack all features
    X = np.hstack([
        np.array(fps_morgan_r2, dtype=np.float32),
        np.array(fps_morgan_r3, dtype=np.float32),
        np.array(desc_list, dtype=np.float32)
    ])
    
    logging.info(f"Feature generation completed in {time.time()-start_time:.1f}s | "
                 f"Total features: {X.shape[1]:,} (4096 + 2048 + 20)")
    return X, valid_indices, valid_smiles

def decode_morgan_bit(smiles_list, bit_index, radius=2, nBits=4096):
    """Decode Morgan fingerprint bit to atom environment examples"""
    examples = []
    for smiles in smiles_list[:100]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=bit_info)
        if bit_index in bit_info and len(bit_info[bit_index]) > 0:
            atom_idx, _ = bit_info[bit_index][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            if submol.GetNumAtoms() > 0:
                examples.append({
                    'smiles': smiles,
                    'atom_idx': atom_idx,
                    'environment_smarts': Chem.MolToSmarts(submol),
                    'center_atom': mol.GetAtomWithIdx(atom_idx).GetSymbol()
                })
                if len(examples) >= 3:
                    break
    return examples

def process_data(filename):
    logging.info(f"Loading dataset: {filename}")
    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return None, None
    
    required_cols = ['Ligand SMILES', 'EC50 (nM)']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing required columns. Found: {df.columns.tolist()}")
        return None, None

    df = df.dropna(subset=required_cols)
    
    def clean_EC50(val):
        try:
            val = str(val).replace('>', '').replace('<', '').strip()
            val = float(val)
            return val if 0 < val <= 1e9 else None
        except:
            return None
    
    df['EC50_clean'] = df['EC50 (nM)'].apply(clean_EC50)
    df = df.dropna(subset=['EC50_clean'])
    df = df.groupby('Ligand SMILES')['EC50_clean'].median().reset_index()
    df['pEC50'] = 9 - np.log10(df['EC50_clean'])
    
    logging.info(f"Dataset processed: {len(df):,} unique compounds | pEC50 range: [{df['pEC50'].min():.2f}, {df['pEC50'].max():.2f}]")
    return df['Ligand SMILES'].values, df['pEC50'].values

def get_raw_feature_names():
    morgan_r2 = [f'MorganR2_Bit_{i}' for i in range(4096)]
    morgan_r3 = [f'MorganR3_Bit_{i}' for i in range(2048)]
    desc_names = [
        'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 
        'NumRotBonds', 'RingCount', 'NumAromaticRings', 'FractionCSP3', 'HeavyAtomCount',
        'NHOHCount', 'NOCount', 'NumHeteroatoms', 'MolMR', 'LabuteASA',
        'TPSA_MolSurf', 'fr_benzene', 'fr_pyridine', 'fr_piperdine', 'fr_morpholine'
    ]
    return np.array(morgan_r2 + morgan_r3 + desc_names)

def analyze_correlations(X, y, feature_names, valid_smiles):
    logging.info("Analyzing Feature-Target Correlations...")
    correlations = []
    for i in range(min(X.shape[1], 1000)):  # Limit to first 1000 features for speed
        try:
            corr, _ = pearsonr(X[:, i], y)
            correlations.append(corr)
        except:
            correlations.append(0.0)
    
    corr_df = pd.DataFrame({'Feature': feature_names[:len(correlations)], 'R': correlations})
    corr_df['Abs_R'] = corr_df['R'].abs()
    top_10 = corr_df.sort_values('Abs_R', ascending=False).head(10)
    
    # Directional bar plot
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in top_10['R']]
    plt.barh(top_10['Feature'], top_10['R'], color=colors, edgecolor='black')
    plt.xlabel('Pearson Correlation (R)', fontsize=12)
    plt.title('Top 10 Features by Correlation with pEC50', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Top10_Correlation_Direction.svg", format='svg', bbox_inches='tight')
    plt.close()
    logging.info("Saved: Top10_Correlation_Direction.svg")

    # Correlation heatmap with target
    top_indices = [list(feature_names).index(f) for f in top_10['Feature']]
    df_heatmap = pd.DataFrame(X[:, top_indices], columns=top_10['Feature'].values)
    df_heatmap['pEC50'] = y
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df_heatmap.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", 
                center=0, square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5)
    plt.title('Feature Correlation Heatmap (Top 10 + Target)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Feature_Correlation_Heatmap_with_pEC50.svg", format='svg', bbox_inches='tight')
    plt.close()
    logging.info("Saved: Feature_Correlation_Heatmap_with_pEC50.svg")

    # Decode top Morgan bits
    logging.info("\n=== Top Morgan Fingerprint Bits - Chemical Interpretation ===")
    morgan_env_data = []
    for _, row in top_10.iterrows():
        feat_name = row['Feature']
        if 'Morgan' in feat_name:
            bit_type = 'r2' if 'R2' in feat_name else 'r3'
            bit_idx = int(feat_name.split('_')[-1])
            radius = 2 if bit_type == 'r2' else 3
            nBits = 4096 if bit_type == 'r2' else 2048
            examples = decode_morgan_bit(valid_smiles, bit_idx, radius=radius, nBits=nBits)
            if examples:
                logging.info(f"\n{feat_name} (R={row['R']:.3f}):")
                for ex in examples[:2]:
                    logging.info(f"  • Central atom: {ex['center_atom']} (index {ex['atom_idx']})")
                    logging.info(f"    Environment: {ex['environment_smarts']}")
                morgan_env_data.append({'Feature': feat_name, 'R': row['R'], 'Examples': examples[:2]})
    
    if morgan_env_data:
        with open('morgan_atom_environments.txt', 'w', encoding='utf-8') as f:
            f.write("Top Morgan Fingerprint Bits - Atom Environment Analysis\n")
            f.write("="*70 + "\n\n")
            for item in morgan_env_data:
                f.write(f"{item['Feature']} (Correlation R={item['R']:.3f})\n")
                f.write("-"*70 + "\n")
                for ex in item['Examples']:
                    f.write(f"Central atom: {ex['center_atom']} (index {ex['atom_idx']})\n")
                    f.write(f"SMARTS pattern: {ex['environment_smarts']}\n")
                    f.write(f"Example molecule: {ex['smiles'][:60]}...\n\n")
        logging.info("Saved detailed atom environments: morgan_atom_environments.txt")
    
    return top_10

def evaluate_model_cv(model, name, X, y, cv=5, n_jobs=N_JOBS):
    """Perform rigorous k-fold cross-validation evaluation"""
    logging.info(f"\n{'='*70}")
    logging.info(f"PERFORMING {cv}-FOLD CROSS-VALIDATION: {name}")
    logging.info(f"{'='*70}")
    
    scoring = {
        'r2': 'r2',
        'rmse': make_scorer(mean_squared_error, squared=False, greater_is_better=False),
        'mae': 'neg_mean_absolute_error'
    }
    
    try:
        cv_results = cross_validate(
            model, X, y, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0,
            return_train_score=True
        )
        
        # Format results
        r2_mean = cv_results['test_r2'].mean()
        r2_std = cv_results['test_r2'].std()
        rmse_mean = -cv_results['test_rmse'].mean()
        rmse_std = cv_results['test_rmse'].std()
        mae_mean = -cv_results['test_mae'].mean()
        mae_std = cv_results['test_mae'].std()
        
        # Log results
        logging.info(f"\n{name} CROSS-VALIDATION RESULTS ({cv}-fold):")
        logging.info(f"  R²:    {r2_mean:.4f} ± {r2_std:.4f}  [Train: {cv_results['train_r2'].mean():.4f}]")
        logging.info(f"  RMSE:  {rmse_mean:.4f} ± {rmse_std:.4f}  [Train: {-cv_results['train_rmse'].mean():.4f}]")
        logging.info(f"  MAE:   {mae_mean:.4f} ± {mae_std:.4f}  [Train: {-cv_results['train_mae'].mean():.4f}]")
        logging.info(f"  Overfitting gap (R²): {cv_results['train_r2'].mean() - r2_mean:.4f}")
        
        return {
            'cv_r2_mean': r2_mean, 'cv_r2_std': r2_std,
            'cv_rmse_mean': rmse_mean, 'cv_rmse_std': rmse_std,
            'cv_mae_mean': mae_mean, 'cv_mae_std': mae_std,
            'overfit_gap_r2': cv_results['train_r2'].mean() - r2_mean
        }
    except Exception as e:
        logging.warning(f"CV evaluation failed for {name}: {str(e)}")
        return None

def evaluate_and_log(model, name, X_train, y_train, X_test, y_test, feature_names=None):
    logging.info(f"\n{'─'*70}")
    logging.info(f"TRAINING MODEL: {name}")
    logging.info(f"{'─'*70}")
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logging.info(f"Training completed in {train_time:.1f}s")
    
    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_r': pearsonr(y_test, y_pred_test)[0]
    }
    
    # Log metrics
    logging.info(f"TRAIN SET -> R²: {metrics['train_r2']:.4f} | RMSE: {metrics['train_rmse']:.4f} | MAE: {metrics['train_mae']:.4f}")
    logging.info(f"TEST SET  -> R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.4f} | MAE: {metrics['test_mae']:.4f} | R: {metrics['test_r']:.4f}")
    
    # Save model-native feature importance
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        pd.DataFrame({'Feature': feature_names, 'Importance': imp}).sort_values(
            'Importance', key=abs, ascending=False
        ).to_excel(f"{name}_feature_importance.xlsx", index=False)
        logging.info(f"Saved model-native feature importance: {name}_feature_importance.xlsx")
    
    # Prediction vs Actual plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors='k', color='#1f77b4', s=60)
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='Ideal Prediction')
    plt.xlabel('Experimental pEC50', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted pEC50', fontsize=12, fontweight='bold')
    plt.title(f'{name}: Prediction Performance', fontsize=14, fontweight='bold')
    plt.text(0.05, 0.95, 
             f"Test R² = {metrics['test_r2']:.3f}\n"
             f"Pearson R = {metrics['test_r']:.3f}\n"
             f"RMSE = {metrics['test_rmse']:.3f}",
             transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{name}_pred_vs_exp.svg", format='svg', bbox_inches='tight')
    plt.close()
    logging.info(f"Saved prediction plot: {name}_pred_vs_exp.svg")
    
    # Permutation importance (critical for reliable interpretation)
    perm_importance = None
    if feature_names is not None:
        try:
            logging.info("  → Computing permutation importance (5 repeats)...")
            perm_start = time.time()
            perm_result = permutation_importance(
                model, X_test, y_test,
                n_repeats=5,
                random_state=42,
                n_jobs=1,  # Prevent nested parallelism issues
                scoring='r2'
            )
            perm_time = time.time() - perm_start
            perm_importance = perm_result.importances_mean
            logging.info(f"  ✓ Permutation importance computed in {perm_time:.1f}s")
        except Exception as e:
            logging.warning(f"  ✗ Permutation importance failed: {str(e)}")
    
    return {
        'metrics': metrics,
        'preds_test': y_pred_test,
        'Permutation_Importance': perm_importance,
        'training_time': train_time
    }

def generate_permutation_heatmaps(base_results, meta_results, selected_feat_names, base_model_names):
    """Generate publication-quality permutation importance heatmaps"""
    logging.info("\n" + "="*70)
    logging.info("GENERATING PERMUTATION IMPORTANCE HEATMAPS")
    logging.info("="*70)
    
    # BASE MODELS HEATMAP
    base_perm_data = []
    for name, res in base_results.items():
        if res.get('Permutation_Importance') is not None:
            for i, feat in enumerate(selected_feat_names):
                base_perm_data.append({
                    'Model': name,
                    'Feature': feat,
                    'Permutation_Importance': res['Permutation_Importance'][i]
                })
    
    if base_perm_data:
        base_perm_df = pd.DataFrame(base_perm_data)
        base_perm_df['Abs_Importance'] = base_perm_df['Permutation_Importance'].abs()
        
        # Select top 25 features by max absolute importance across all models
        top_features = base_perm_df.groupby('Feature')['Abs_Importance'].max().nlargest(25).index.tolist()
        base_perm_df_top = base_perm_df[base_perm_df['Feature'].isin(top_features)]
        
        # Pivot for heatmap
        heatmap_data = base_perm_df_top.pivot(index='Feature', columns='Model', values='Permutation_Importance')
        
        # Plot
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".3f", 
            cmap="coolwarm", 
            center=0,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Permutation Importance (ΔR²)', 'shrink': 0.8},
            annot_kws={"size": 9}
        )
        plt.title('Permutation Feature Importance: Base Models (Top 25 Features)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Molecular Feature', fontsize=13, fontweight='bold')
        plt.xlabel('Model', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("Permutation_Heatmap_Base_Models.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()
        logging.info("✓ Saved: Permutation_Heatmap_Base_Models.svg")
        
        # Save raw data
        base_perm_df_top.to_excel("Permutation_Importance_Base_Models.xlsx", index=False)
        logging.info("✓ Saved: Permutation_Importance_Base_Models.xlsx")
    else:
        logging.warning("⚠ No permutation importance data available for base models")
    
    # META-LEARNERS HEATMAP
    meta_perm_data = []
    for name, res in meta_results.items():
        if res.get('Permutation_Importance') is not None:
            for i, base_name in enumerate(base_model_names):
                meta_perm_data.append({
                    'Meta_Model': name,
                    'Base_Model': base_name,
                    'Permutation_Importance': res['Permutation_Importance'][i]
                })
    
    if meta_perm_data:
        meta_perm_df = pd.DataFrame(meta_perm_data)
        heatmap_meta = meta_perm_df.pivot(index='Base_Model', columns='Meta_Model', values='Permutation_Importance')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_meta,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Permutation Importance (ΔR²)', 'shrink': 0.8}
        )
        plt.title('Permutation Importance: Meta-Learners', fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Base Model (as Feature)', fontsize=12, fontweight='bold')
        plt.xlabel('Meta-Learner', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("Permutation_Heatmap_Meta_Learners.svg", format='svg', bbox_inches='tight')
        plt.close()
        logging.info("✓ Saved: Permutation_Heatmap_Meta_Learners.svg")
        
        meta_perm_df.to_excel("Permutation_Importance_Meta_Learners.xlsx", index=False)
        logging.info("✓ Saved: Permutation_Importance_Meta_Learners.xlsx")
    else:
        logging.warning("⚠ No permutation importance data available for meta-learners")

def generate_shap_analysis(best_model, model_name, X_test, feature_names, is_tree_based):
    """Generate SHAP interpretability plots for tree-based models"""
    if not is_tree_based:
        logging.info("SHAP analysis skipped (only supported for tree-based models)")
        return
    
    try:
        import shap
        logging.info("\n" + "="*70)
        logging.info(f"GENERATING SHAP ANALYSIS FOR: {model_name}")
        logging.info("="*70)
        
        # Sample test set if too large
        max_samples = 1000
        if X_test.shape[0] > max_samples:
            idx = np.random.choice(X_test.shape[0], max_samples, replace=False)
            X_display = X_test[idx]
            logging.info(f"Sampling {max_samples} instances for SHAP analysis")
        else:
            X_display = X_test
        
        # Create explainer and compute SHAP values
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_display)
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_display, feature_names=feature_names, 
                         show=False, plot_size=(12, 10), color_bar_label="Feature Value")
        plt.title(f'SHAP Feature Importance: {model_name}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig("SHAP_Summary_Plot.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()
        logging.info("✓ Saved: SHAP_Summary_Plot.svg")
        
        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(0)
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_SHAP': mean_shap
        }).sort_values('Mean_SHAP', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Mean_SHAP', y='Feature', data=shap_df, palette="viridis")
        plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        plt.title(f'Top 20 Features by SHAP Impact: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("SHAP_Top_Features.svg", format='svg', bbox_inches='tight')
        plt.close()
        logging.info("✓ Saved: SHAP_Top_Features.svg")
        
        # Save SHAP values
        shap_df.to_excel("SHAP_Feature_Importance.xlsx", index=False)
        logging.info("✓ Saved: SHAP_Feature_Importance.xlsx")
        
    except Exception as e:
        logging.warning(f"SHAP analysis failed: {str(e)}. Install with: pip install shap")

def generate_uncertainty_quantification(X_train, y_train, X_test, y_test, best_model_name):
    """Generate prediction intervals using quantile regression"""
    logging.info("\n" + "="*70)
    logging.info("GENERATING UNCERTAINTY QUANTIFICATION (90% PREDICTION INTERVALS)")
    logging.info("="*70)
    
    try:
        # Train three quantile regressors
        models = {
            'lower': RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=N_JOBS, random_state=42),
            'median': RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=N_JOBS, random_state=42),
            'upper': RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=N_JOBS, random_state=42)
        }
        
        logging.info("Training quantile regression models...")
        models['lower'].fit(X_train, y_train)
        models['median'].fit(X_train, y_train)
        models['upper'].fit(X_train, y_train)
        
        # Generate predictions
        y_pred_lower = models['lower'].predict(X_test)
        y_pred_median = models['median'].predict(X_test)
        y_pred_upper = models['upper'].predict(X_test)
        
        # Calculate coverage
        covered = np.sum((y_test >= y_pred_lower) & (y_test <= y_pred_upper))
        coverage = covered / len(y_test) * 100
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, y_pred_median, alpha=0.7, color='#1f77b4', s=60, label='Median Prediction')
        plt.fill_between(np.sort(y_test), 
                        y_pred_lower[np.argsort(y_test)], 
                        y_pred_upper[np.argsort(y_test)], 
                        alpha=0.3, color='#ff7f0e', label='90% Prediction Interval')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
        plt.xlabel('Experimental pEC50', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted pEC50', fontsize=12, fontweight='bold')
        plt.title(f'Uncertainty Quantification: {best_model_name}\nCoverage: {coverage:.1f}% of points within 90% PI', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("Uncertainty_Quantification.svg", format='svg', bbox_inches='tight')
        plt.close()
        logging.info(f"✓ Saved: Uncertainty_Quantification.svg (Coverage: {coverage:.1f}%)")
        
        # Save interval data
        interval_df = pd.DataFrame({
            'Experimental_pEC50': y_test,
            'Predicted_Median': y_pred_median,
            'Predicted_Lower_5%': y_pred_lower,
            'Predicted_Upper_95%': y_pred_upper,
            'Interval_Width': y_pred_upper - y_pred_lower
        })
        interval_df.to_excel("Prediction_Intervals.xlsx", index=False)
        logging.info("✓ Saved: Prediction_Intervals.xlsx")
        
    except Exception as e:
        logging.warning(f"Uncertainty quantification failed: {str(e)}")

def run_pipeline():
    # ======================
    # DATA LOADING & PREP
    # ======================
    smiles, targets = process_data('D2.csv')
    if smiles is None:
        logging.error("Pipeline terminated due to data loading error")
        return
    
    # Generate enhanced features
    X_all, valid_idx, valid_smiles = calculate_enhanced_features(smiles)
    y_all = targets[valid_idx]
    all_feat_names = get_raw_feature_names()
    
    # Train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    logging.info(f"\nDataset split: {X_train_raw.shape[0]:,} train | {X_test_raw.shape[0]:,} test samples")
    
    # Preprocessing
    logging.info("\nPreprocessing data...")
    sel_var = VarianceThreshold(threshold=0.01)
    scaler = StandardScaler()
    
    X_train_sel = sel_var.fit_transform(X_train_raw)
    X_test_sel = sel_var.transform(X_test_raw)
    selected_feat_names = all_feat_names[sel_var.get_support()]
    
    X_train = scaler.fit_transform(X_train_sel)
    X_test = scaler.transform(X_test_sel)
    
    logging.info(f"Features after variance threshold: {X_train.shape[1]:,} / {len(all_feat_names):,}")
    
    # Save processed datasets
    logging.info("Saving processed datasets...")
    pd.DataFrame(X_train, columns=selected_feat_names).assign(Target=y_train).to_excel("Final_Train_Set.xlsx", index=False)
    pd.DataFrame(X_test, columns=selected_feat_names).assign(Target=y_test).to_excel("Final_Test_Set.xlsx", index=False)
    logging.info("✓ Saved train/test datasets to Excel")
    
    # Correlation analysis
    analyze_correlations(X_train, y_train, selected_feat_names, valid_smiles)
    
    # ======================
    # MODEL TRAINING
    # ======================
    base_models = {
        'SVR': SVR(kernel='rbf', C=10, epsilon=0.1, cache_size=2000),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            n_jobs=N_JOBS, tree_method='hist', random_state=42, verbosity=0
        ),
        'LGBM': LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            n_jobs=N_JOBS, random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            thread_count=N_JOBS, verbose=0, random_state=42
        )
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features_train = pd.DataFrame()
    meta_features_test = pd.DataFrame()
    base_results = {}
    cv_results_all = {}
    
    # Train base models with rigorous CV evaluation
    for name, model in base_models.items():
        logging.info(f"\n{'='*70}")
        logging.info(f"BASE MODEL: {name}")
        logging.info(f"{'='*70}")
        
        # 1. Perform rigorous 5-fold CV evaluation on TRAINING SET
        cv_res = evaluate_model_cv(
            model.__class__(**model.get_params()),  # Fresh instance
            name, 
            X_train, 
            y_train, 
            cv=5, 
            n_jobs=1 if name in ['CatBoost', 'LGBM', 'XGBoost'] else N_JOBS
        )
        if cv_res:
            cv_results_all[name] = cv_res
        
        # 2. Train on full training set and evaluate on TEST SET
        res = evaluate_and_log(model, name, X_train, y_train, X_test, y_test, selected_feat_names)
        base_results[name] = res
        meta_features_test[name] = res['preds_test']
        
        # 3. Generate meta-features via cross_val_predict (for stacking)
        n_jobs_cv = 1 if name in ['CatBoost', 'LGBM', 'XGBoost'] else N_JOBS
        meta_features_train[name] = cross_val_predict(
            model, X_train, y_train, cv=kf, n_jobs=n_jobs_cv, verbose=0
        )
        
        # Save base model
        with open(f'base_model_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"✓ Saved base model: base_model_{name}.pkl")
    
    # Train meta-learners
    meta_learners = {
        'Meta_Ridge': Ridge(alpha=1.0),
        'Meta_Huber': HuberRegressor(epsilon=1.1),
        'Meta_LGBM': LGBMRegressor(n_estimators=500, n_jobs=N_JOBS, verbose=-1, random_state=42)
    }
    meta_results = {}
    
    for name, model in meta_learners.items():
        logging.info(f"\n{'='*70}")
        logging.info(f"META-LEARNER: {name}")
        logging.info(f"{'='*70}")
        meta_results[name] = evaluate_and_log(
            model, name, 
            meta_features_train, y_train, 
            meta_features_test, y_test,
            np.array(list(base_models.keys()))
        )
    
    # ======================
    # POST-PROCESSING
    # ======================
    # Generate permutation importance heatmaps
    generate_permutation_heatmaps(base_results, meta_results, selected_feat_names, list(base_models.keys()))
    
    # Determine winner
    all_results = {}
    for name, res in base_results.items():
        all_results[name] = res['metrics']['test_r2']
    for name, res in meta_results.items():
        all_results[name] = res['metrics']['test_r2']
    
    winner_name = max(all_results, key=all_results.get)
    winner_score = all_results[winner_name]
    is_meta = 'Meta' in winner_name
    winner_model = meta_learners[winner_name] if is_meta else base_models[winner_name]
    
    logging.info(f"\n{'='*70}")
    logging.info(f"🏆 WINNER: {winner_name} | Test R²: {winner_score:.4f} | Type: {'Meta-Learner' if is_meta else 'Base Model'}")
    logging.info(f"{'='*70}")
    
    # Retrain winner on full data
    if is_meta:
        winner_model.fit(meta_features_train, y_train)
        logging.info(f"Retrained meta-learner {winner_name} on full meta-feature training set")
    else:
        winner_model.fit(X_train, y_train)
        logging.info(f"Retrained base model {winner_name} on full training set")
    
    # SHAP analysis (if tree-based)
    is_tree_based = any(x in winner_name for x in ['LGBM', 'XGBoost', 'CatBoost', 'GradientBoosting'])
    if is_tree_based:
        generate_shap_analysis(winner_model, winner_name, X_test, selected_feat_names, is_tree_based)
    
    # Uncertainty quantification
    generate_uncertainty_quantification(X_train, y_train, X_test, y_test, winner_name)
    
    # Save artifacts
    artifacts = [
        ('best_model.pkl', winner_model),
        ('scaler.pkl', scaler),
        ('variance_selector.pkl', sel_var),
        ('selected_features.pkl', selected_feat_names.tolist()),
    ]
    
    for fname, obj in artifacts:
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
        logging.info(f"✓ Saved: {fname}")
    
    # Save comprehensive model info
    model_info = {
        'model_name': winner_name,
        'is_meta_learner': is_meta,
        'base_models': list(base_models.keys()) if is_meta else None,
        'test_r2': winner_score,
        'test_rmse': base_results[winner_name]['metrics']['test_rmse'] if not is_meta else meta_results[winner_name]['metrics']['test_rmse'],
        'test_mae': base_results[winner_name]['metrics']['test_mae'] if not is_meta else meta_results[winner_name]['metrics']['test_mae'],
        'test_r': base_results[winner_name]['metrics']['test_r'] if not is_meta else meta_results[winner_name]['metrics']['test_r'],
        'cv_results': cv_results_all.get(winner_name.split('_')[-1] if is_meta else winner_name),
        'n_features': len(selected_feat_names),
        'feature_names': selected_feat_names.tolist(),
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': 'Ryzen 7 7800X3D + 32GB DDR5'
    }
    
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    logging.info("✓ Saved comprehensive model info: model_info.pkl")
    
    # Final summary
    logging.info("\n" + "="*70)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    logging.info("="*70)
    logging.info("Key outputs generated:")
    logging.info("  • Permutation_Heatmap_Base_Models.svg")
    logging.info("  • Permutation_Heatmap_Meta_Learners.svg")
    logging.info("  • SHAP_Summary_Plot.svg (if tree-based winner)")
    logging.info("  • Uncertainty_Quantification.svg")
    logging.info("  • Final_Train_Set.xlsx / Final_Test_Set.xlsx")
    logging.info("  • All model artifacts (best_model.pkl, scalers, etc.)")
    logging.info("\nNext steps:")
    logging.info("  1. Examine permutation heatmaps to identify robust features")
    logging.info("  2. Review SHAP plots for chemical interpretability")
    logging.info("  3. Use prediction intervals for experimental validation planning")
    logging.info("  4. Deploy best_model.pkl with supporting artifacts for inference")
    logging.info("="*70)
    

if __name__ == "__main__":
    total_start = time.time()
    logging.info("="*70)
    logging.info("STARTING D2 EC50 PREDICTION PIPELINE")
    logging.info(f"Hardware: AMD Ryzen 7 7800X3D | {N_JOBS} parallel jobs")
    logging.info(f"Enhanced features: 4096 + 2048 Morgan bits + 20 extended descriptors")
    logging.info("="*70)
    
    try:
        run_pipeline()
        logging.info(f"\n✅ FULL PIPELINE COMPLETED IN {time.time()-total_start:.1f} SECONDS")
    except Exception as e:
        logging.error(f"\n❌ PIPELINE FAILED WITH ERROR:\n{str(e)}", exc_info=True)
        raise