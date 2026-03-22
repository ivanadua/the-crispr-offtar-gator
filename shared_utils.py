"""
shared_utils.py
----------------
Shared functions used by all training scripts.
Import this in each training script.

Key design decisions:
- Cell line is EXCLUDED from XGBoost features (prevents source leakage)
- Cell line IS included in hybrid model via ATAC-seq (real experimental data)
- Guide-aware train/test split (no guide appears in both train and test)
- Synthetic negatives are NOT identifiable by any feature
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve)

RANDOM_SEED = 42
TEST_SIZE   = 0.2
SEQ_LEN     = 20
BASES       = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# ── FEATURE SETS ──────────────────────────────────────────────────────────────

# Pure biology features — no cell line, no source, no tier info
# Used by XGBoost and CNN biological branch
BIO_FEATURES = [
    'gc_content_target',
    'gc_content_offtarget',
    'shannon_entropy_target',
    'shannon_entropy_offt',
    'mismatch_count',
    'seed_mismatches',
    'nonseed_mismatches',
    'pam_proximity_score',
    'guide_length',
    'mfe',
    'has_bulge',
    'atac_accessible',   # This is the only cell-line-aware feature
                         # but it's tied to actual chromatin data, not source
]

# ── SEQUENCE ENCODING ─────────────────────────────────────────────────────────

def encode_pair(target, offtarget, length=SEQ_LEN):
    """
    Encode a target+offtarget pair as a (length x 8) matrix.
    First 4 channels = target one-hot (A,C,G,T)
    Next  4 channels = offtarget one-hot (A,C,G,T)
    """
    def one_hot(seq):
        m = np.zeros((length, 4), dtype=np.float32)
        for i, b in enumerate(str(seq)[:length].upper()):
            if b in BASES:
                m[i, BASES[b]] = 1.0
        return m
    return np.concatenate([one_hot(target), one_hot(offtarget)], axis=1)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_features(path):
    """
    Load dataset and return:
      X_bio  — biological feature matrix (no cell line, no source)
      y      — labels
      df     — full dataframe (for guide-aware split)
    """
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows | {int(df['is_cut'].sum()):,} positives | "
          f"{int((df['is_cut']==0).sum()):,} negatives")

    # Fill missing with median — do NOT impute with source-aware values
    for col in BIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            df[col] = 0.0

    X_bio = df[BIO_FEATURES].values.astype(np.float32)
    y     = df['is_cut'].values.astype(np.float32)

    return X_bio, y, df

def guide_aware_split(df, X_bio=None, X_seq=None, test_size=TEST_SIZE):
    """
    Split data so that ALL pairs from a given guide RNA go into
    either train OR test — never both.

    This prevents the model from memorising guide-specific patterns
    and forces it to generalise to unseen guides.
    """
    unique_guides = df['target_seq'].unique().tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_guides)

    n_test  = int(len(unique_guides) * test_size)
    test_guides = set(unique_guides[:n_test])

    test_mask  = df['target_seq'].isin(test_guides).values
    train_mask = ~test_mask

    print(f"  Guide-aware split:")
    print(f"    Train: {train_mask.sum():,} rows | {len(unique_guides)-n_test} unique guides")
    print(f"    Test:  {test_mask.sum():,} rows  | {n_test} unique guides")

    result = {}
    if X_bio is not None:
        result['X_bio_train'] = X_bio[train_mask]
        result['X_bio_test']  = X_bio[test_mask]
    if X_seq is not None:
        result['X_seq_train'] = X_seq[train_mask]
        result['X_seq_test']  = X_seq[test_mask]

    y = df['is_cut'].values.astype(np.float32)
    result['y_train']     = y[train_mask]
    result['y_test']      = y[test_mask]
    result['train_mask']  = train_mask
    result['test_mask']   = test_mask

    return result

# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_prob, model_name):
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr, tpr, _    = roc_curve(y_true, y_prob)

    results = dict(
        model=model_name,
        auc_roc=round(float(auc), 4),
        auc_pr=round(float(auprc), 4),
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1=round(float(f1), 4),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)
    )

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Test Results")
    print(f"{'─'*50}")
    for k, v in results.items():
        if k not in ('model', 'tp', 'fp', 'tn', 'fn'):
            print(f"  {k:20s}: {v}")
    print(f"  TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")

    return results, fpr, tpr
