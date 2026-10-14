# train_and_save_voting_with_tuning.py
import pandas as pd
import joblib
import json
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sns.set()


def load_data(file_path, target_col='Activity'):
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(f"ستون '{target_col}' یافت نشد.")
    return df


def prepare_X_y(df, target_col='Activity', smiles_col='SMILES', drop_cols=None):
    drop_list = [target_col] + (drop_cols or [])
    if smiles_col in df.columns and smiles_col not in (drop_cols or []):
        drop_list.append(smiles_col)
    X = df.drop(columns=drop_list)
    y = df[target_col]
    return X, y


def split_train_test_with_df(df, X, y, test_size=0.2, random_state=42, stratify=False):
    strat = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    df_test = df.loc[X_test.index].copy().reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, df_test


def _default_param_grids():
    # Parameter grids for each base estimator (referencing pipeline step names)
    svr_grid = {
        'svr__kernel': ['rbf'],
        'svr__C': [1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__gamma': ['scale', 'auto']
    }
    rf_grid = {
        'rf__n_estimators': [100, 300],
        'rf__max_depth': [None, 10, 30],
        'rf__min_samples_leaf': [1, 2]
    }
    gbr_grid = {
        'gbr__n_estimators': [100, 300],
        'gbr__learning_rate': [0.05, 0.1],
        'gbr__max_depth': [3, 5]
    }
    return {'svr': svr_grid, 'rf': rf_grid, 'gbr': gbr_grid}


def tune_base_estimators(X_train, y_train, param_grids=None, cv=5, n_jobs=-1, scoring='r2', verbose=1):
    """
    Tune each base estimator separately using GridSearchCV.
    Returns dict with keys:
      - 'estimators': {name: best_pipeline}
      - 'best_params': {name: best_params}
      - 'best_cv_scores': {name: best_cv_score}
      - 'grid_search_objects': {name: GridSearchCV instance}
    """
    if param_grids is None:
        param_grids = _default_param_grids()

    results = {
        'estimators': {},
        'best_params': {},
        'best_cv_scores': {},
        'grid_search_objects': {}
    }

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Define pipelines
    svr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    gbr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ])

    pipeline_map = {'svr': svr_pipeline, 'rf': rf_pipeline, 'gbr': gbr_pipeline}

    for name, pipeline in pipeline_map.items():
        grid = None
        if name in param_grids and param_grids[name]:
            print(f"🔧 Tuning '{name}' with GridSearchCV ...")
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[name],
                cv=kf,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_
            best_params = grid.best_params_
            best_score = grid.best_score_
            print(f" -> {name} best CV {scoring}: {best_score:.6f}")
        else:
            # If no grid provided, just fit default pipeline
            print(f"ℹ️ No param grid provided for '{name}'. Fitting default pipeline.")
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = None
            # attempt to estimate CV score using KFold manually
            scores = []
            for train_idx, val_idx in kf.split(X_train):
                pipeline_clone = Pipeline(pipeline.steps)
                pipeline_clone.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                scores.append(r2_score(y_train.iloc[val_idx], pipeline_clone.predict(X_train.iloc[val_idx])))
            best_score = float(np.mean(scores))

        results['estimators'][name] = best_pipeline
        results['best_params'][name] = best_params
        results['best_cv_scores'][name] = float(best_score)
        results['grid_search_objects'][name] = grid

    return results


def compute_weights_from_scores(score_dict):
    """
    Convert CV scores to non-negative weights for VotingRegressor.
    If all scores <= 0, return equal weights.
    """
    scores = np.array([score_dict[k] for k in sorted(score_dict.keys())])  # order by key
    # Map keys order to names to retrieve corresponding later
    names = sorted(score_dict.keys())
    # shift to positive
    min_score = np.min(scores)
    if np.all(np.isfinite(scores)):
        if min_score <= 0:
            # shift up so smallest becomes small positive
            shift = abs(min_score) + 1e-6
            pos_scores = scores + shift
        else:
            pos_scores = scores
    else:
        pos_scores = np.ones_like(scores)

    if pos_scores.sum() == 0:
        weights = np.ones_like(pos_scores) / len(pos_scores)
    else:
        weights = (pos_scores / pos_scores.sum()).tolist()

    # return mapping name -> weight
    return {name: float(w) for name, w in zip(names, weights)}


def build_and_fit_voting(estimators_dict, X_train, y_train, weights_map=None, n_jobs=-1):
    """
    estimators_dict: {'svr': pipeline, 'rf': pipeline, ...}
    weights_map: {'svr': 0.5, 'gbr': 0.3, 'rf': 0.2} or None
    """
    # Create list of (name, estimator) in a stable order
    order = sorted(estimators_dict.keys())
    estimators_list = [(name, estimators_dict[name]) for name in order]

    if weights_map is not None:
        weights = [weights_map.get(name, 1.0) for name in order]
    else:
        weights = None

    voting = VotingRegressor(estimators=estimators_list, n_jobs=n_jobs, weights=weights)
    # Fit the ensemble (will clone & fit each pipeline with the tuned params)
    voting.fit(X_train, y_train)
    return voting


def evaluate_and_save(model, X_test, y_test, df_test,
                      prefix='voting_bundle', save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)

    # prediction
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    # metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2 (test): {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    # results DataFrame
    df_res = X_test.copy().reset_index(drop=True)
    df_res['Actual'] = y_test.reset_index(drop=True)
    df_res['Predicted'] = pd.Series(y_pred)
    df_res['Residual'] = df_res['Actual'] - df_res['Predicted']
    if 'SMILES' in df_test.columns:
        df_res['SMILES'] = df_test['SMILES'].values

    cols = []
    if 'SMILES' in df_res.columns:
        cols.append('SMILES')
    cols += ['Actual', 'Predicted', 'Residual']
    df_res = df_res[cols + [c for c in df_res.columns if c not in cols]]

    out_xlsx = os.path.join(save_dir, f'{prefix}_results.xlsx')
    df_res.to_excel(out_xlsx, index=False)
    print(f"✅ نتایج نمونه‌ای در '{out_xlsx}' ذخیره شد.")

    images = []

    # Parity plot
    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(df_res['Actual'], df_res['Predicted'], alpha=0.6)
        minv = min(df_res['Actual'].min(), df_res['Predicted'].min())
        maxv = max(df_res['Actual'].max(), df_res['Predicted'].max())
        plt.plot([minv, maxv], [minv, maxv], '--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Parity plot')
        plt.tight_layout()
        parity_file = os.path.join(save_dir, f'{prefix}_parity.jpg')
        plt.savefig(parity_file, format='jpg', dpi=300)
        plt.close()
        images.append(parity_file)
    except Exception as e:
        print("Warning: failed to draw/save parity plot:", e)

    # Residual histogram
    try:
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True)
        plt.title('Residuals distribution')
        plt.xlabel('Residual')
        plt.tight_layout()
        res_hist = os.path.join(save_dir, f'{prefix}_residuals_hist.jpg')
        plt.savefig(res_hist, format='jpg', dpi=300)
        plt.close()
        images.append(res_hist)
    except Exception as e:
        print("Warning: failed to draw/save residuals histogram:", e)

    print("📊 نمودارها ذخیره شدند (JPG).")

    # PDF report
    try:
        pdf_file = os.path.join(save_dir, f'{prefix}_report.pdf')
        pdf = FPDF()
        pdf.set_auto_page_break(True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Voting Regressor Report', ln=True, align='C')
        pdf.ln(5)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 8, f'R2: {r2:.6f}')
        pdf.multi_cell(0, 8, f'MSE: {mse:.6f}')
        pdf.multi_cell(0, 8, f'RMSE: {rmse:.6f}')
        pdf.multi_cell(0, 8, f'MAE: {mae:.6f}')
        pdf.ln(4)
        for img in images:
            if img and os.path.exists(img):
                pdf.add_page()
                pdf.image(img, x=15, y=30, w=180)
        pdf.output(pdf_file)
        print(f"📄 گزارش PDF در '{pdf_file}' ذخیره شد.")
    except Exception as e:
        print("Warning: failed to create PDF report:", e)

    return {
        'r2': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'results_xlsx': out_xlsx,
        'images': images,
        'pdf': pdf_file if 'pdf_file' in locals() else None
    }


def _to_python_native(x):
    import numpy as _np
    import pandas as _pd

    if isinstance(x, (list, tuple)):
        return [_to_python_native(v) for v in x]
    if isinstance(x, _pd.Index):
        return [ _to_python_native(v) for v in list(x) ]
    if isinstance(x, _np.ndarray):
        return [_to_python_native(v) for v in x.tolist()]
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_ , bool)):
        return bool(x)
    try:
        if hasattr(x, 'item'):
            py = x.item()
            if isinstance(py, (int, float, bool, str)):
                return py
    except Exception:
        pass
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    return str(x)


def save_bundle(model_pipeline, X_columns, per_estimator_info,
                target_col='Activity', smiles_col='SMILES',
                bundle_prefix='voting_bundle', save_dir='outputs'):
    """
    per_estimator_info: dict with keys = estimator names and values:
       {'best_params': ..., 'best_cv_score': ...}
    """
    os.makedirs(save_dir, exist_ok=True)
    pipeline_path = os.path.join(save_dir, f'{bundle_prefix}_pipeline.joblib')
    metadata_path = os.path.join(save_dir, f'{bundle_prefix}_metadata.json')

    joblib.dump(model_pipeline, pipeline_path)

    metadata = {
        'feature_columns': _to_python_native(list(X_columns)),
        'target_col': _to_python_native(target_col),
        'smiles_col': _to_python_native(smiles_col),
        'bundle_version': '1.0',
        'notes': 'VotingRegressor built from tuned base estimators',
        'per_estimator': _to_python_native(per_estimator_info)
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Bundle saved:")
    print(f" - {pipeline_path}")
    print(f" - {metadata_path}")
    return pipeline_path, metadata_path


def main():
    # user-editable
    input_csv = '1000selected_features_reg.csv'  # change as needed
    target_col = 'Activity'
    smiles_col = 'SMILES'
    bundle_prefix = 'voting_bundle'
    save_dir = 'outputs'

    print("Loading data...")
    df = load_data(input_csv, target_col=target_col)

    print("Preparing X and y...")
    X, y = prepare_X_y(df, target_col=target_col, smiles_col=smiles_col)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test, df_test = split_train_test_with_df(
        df, X, y, test_size=0.2, random_state=42, stratify=False
    )

    # -------------- tune individual estimators --------------
    print("Tuning base estimators (SVR, RandomForest, GradientBoosting)...")
    param_grids = _default_param_grids()  # change if you want custom grids
    tuned = tune_base_estimators(X_train, y_train, param_grids, cv=5, n_jobs=-1, scoring='r2', verbose=1)

    per_info = {}
    for name in tuned['estimators'].keys():
        per_info[name] = {
            'best_params': tuned['best_params'].get(name),
            'best_cv_score': tuned['best_cv_scores'].get(name)
        }

    # -------------- compute weights from CV scores --------------
    weights_map = compute_weights_from_scores(per_info and {k: per_info[k]['best_cv_score'] for k in per_info})
    print("Computed voting weights (from CV scores):", weights_map)

    # -------------- build/fits voting regressor --------------
    print("Building & fitting VotingRegressor ensemble...")
    voting_model = build_and_fit_voting(tuned['estimators'], X_train, y_train, weights_map=weights_map, n_jobs=-1)
    print("VotingRegressor fit complete.")

    # -------------- evaluate & save --------------
    print("Evaluating on test set and saving reports...")
    eval_info = evaluate_and_save(voting_model, X_test, y_test, df_test,
                                  prefix=bundle_prefix, save_dir=save_dir)

    pipeline_path, metadata_path = save_bundle(voting_model, X.columns, per_info,
                                              target_col=target_col, smiles_col=smiles_col,
                                              bundle_prefix=bundle_prefix, save_dir=save_dir)

    print("All done.")


if __name__ == '__main__':
    main()
