"""
Model 1: XGBoost
-----------------
Baseline model using biological features only.
Trains fast (~2-5 min), produces SHAP values.

Usage:
    pip install xgboost shap scikit-learn
    python train_xgboost.py

Output:
    model_results/xgboost_model.json
    model_results/xgboost_results.json
    model_results/xgboost_shap_values.npy
    model_results/xgboost_shap_plot.png
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve)

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH  = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'
OUTPUT_DIR  = '/Users/moana/Downloads/model_results/'
RANDOM_SEED = 42
TEST_SIZE   = 0.2
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

CELL_LINES = ['HEK293T', 'HEK293', 'K562', 'U2OS', 'U2OS_exp1', 'U2OS_exp2']

BIO_FEATURES = [
    'gc_content_target', 'gc_content_offtarget',
    'shannon_entropy_target', 'shannon_entropy_offt',
    'mismatch_count', 'seed_mismatches', 'nonseed_mismatches',
    'pam_proximity_score', 'guide_length', 'mfe',
    'has_bulge', 'atac_accessible',
]

def load_data(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows | {df['is_cut'].sum():,} positives")

    # Cell line one-hot encoding
    df['cell_line_clean'] = df['cell_line'].fillna('other').apply(
        lambda x: x if x in CELL_LINES else 'other'
    )
    cell_dummies = pd.get_dummies(df['cell_line_clean'], prefix='cl')
    for c in CELL_LINES + ['other']:
        col = f'cl_{c}'
        if col not in cell_dummies.columns:
            cell_dummies[col] = 0

    # Fill missing features with median
    for col in BIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    X = pd.concat([df[BIO_FEATURES], cell_dummies], axis=1)
    X.columns = X.columns.astype(str)
    y = df['is_cut'].values

    # ── GUIDE-AWARE SPLIT ────────────────────────────────────────────────────
    # All pairs from the same guide go into train OR test — never both
    # This prevents the model from memorizing guide-specific patterns
    unique_guides = df['target_seq'].unique().tolist()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_guides)
    n_test_guides = int(len(unique_guides) * TEST_SIZE)
    test_guides   = set(unique_guides[:n_test_guides])

    test_mask  = df['target_seq'].isin(test_guides)
    train_mask = ~test_mask

    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_test  = y[test_mask]

    print(f"  Guide-aware split: {train_mask.sum():,} train / {test_mask.sum():,} test rows")
    print(f"  Train guides: {(~test_mask).sum()} rows from {len(unique_guides)-n_test_guides} guides")
    print(f"  Test guides:  {test_mask.sum()} rows from {n_test_guides} guides")

    return X_train, X_test, y_train, y_test, list(X.columns)

def evaluate(y_true, y_prob, y_pred):
    auc   = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(auc_roc=round(auc,4), auc_pr=round(auprc,4),
                precision=round(prec,4), recall=round(rec,4),
                f1=round(f1,4), tp=int(tp), fp=int(fp),
                tn=int(tn), fn=int(fn))

def main():
    print("=" * 55)
    print("   Model 1: XGBoost")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_cols = load_data(INPUT_PATH)
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # Class weight for imbalance
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    results = evaluate(y_test, y_prob, y_pred)
    results['model'] = 'XGBoost'

    print(f"\n{'─'*55}")
    print("  RESULTS")
    print(f"{'─'*55}")
    for k, v in results.items():
        if k != 'model':
            print(f"  {k:20s}: {v}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'xgboost_model.json')
    model.save_model(model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'xgboost_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    np.save(os.path.join(OUTPUT_DIR, 'xgboost_roc.npy'),
            np.array([fpr, tpr]))

    # Print top feature importances
    importances = model.feature_importances_
    feat_names  = list(X_test.columns)
    top_idx     = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features by XGBoost importance:")
    for i in top_idx:
        print(f"  {feat_names[i]:35s}: {importances[i]:.4f}")

    # ── SHAP VALUES ──────────────────────────────────────────────────────────
    print("\nComputing SHAP values (this may take a few minutes)...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_SEED)
    shap_values = explainer.shap_values(X_sample)

    np.save(os.path.join(OUTPUT_DIR, 'xgboost_shap_values.npy'), shap_values)

    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=list(X_test.columns),
                      show=False, max_display=15)
    plt.title("XGBoost — Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    shap_plot_path = os.path.join(OUTPUT_DIR, 'xgboost_shap_plot.png')
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP plot saved to {shap_plot_path}")

    print(f"\n✅ XGBoost complete | AUC-ROC: {results['auc_roc']} | AUC-PR: {results['auc_pr']}")
    print("Next: run train_cnn.py")

if __name__ == "__main__":
    main()
