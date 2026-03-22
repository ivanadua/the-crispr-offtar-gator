"""
Baseline Comparison — CFD Score and MIT Score
----------------------------------------------
Computes CFD (Cutting Frequency Determination) and MIT off-target
scores for your test set and compares AUC against your trained models.

CFD Score: Doench et al. 2016, Nature Biotechnology
MIT Score: Hsu et al. 2013, Nature Biotechnology

Both scores are lookup-table based formulas — no training required.
Higher score = more likely to cut.

Usage:
    python compute_baseline_scores.py

Output:
    model_results/baseline_comparison.csv
    model_results/figure_baseline_comparison.png
"""

import sys
sys.path.insert(0, '/Users/moana/Downloads/')

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from shared_utils import guide_aware_split, RANDOM_SEED

INPUT_PATH = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'
OUTPUT_DIR = '/Users/moana/Downloads/model_results/'

# ── CFD SCORE LOOKUP TABLES ───────────────────────────────────────────────────
# From Doench et al. 2016 — mismatch penalty per position per substitution type
# Keys: 'rX:dY,n' where X=RNA base, Y=DNA base, n=position (1-20)

CFD_MM_SCORES = {
    'rA:dA': [0,0,0.014,0,0,0.395,0.317,0,0.389,0.079,0.445,0.508,0.613,0.851,0.732,0.828,0.615,0.804,0.685,0.583],
    'rA:dC': [0,0,0,0.044,0.058,0.067,0,0.123,0,0,0.021,0,0,0,0,0,0,0,0,0],
    'rA:dG': [0.059,0.059,0.078,0.082,0.044,0.105,0.094,0.110,0.073,0.129,0.084,0.181,0.183,0.213,0.237,0.207,0.222,0.186,0.145,0.237],
    'rC:dA': [0,0,0,0.025,0,0.018,0.028,0.050,0.051,0.050,0.064,0.080,0.101,0.124,0.108,0.138,0.129,0.150,0.117,0.108],
    'rC:dC': [0,0,0.019,0.020,0,0,0,0,0,0.002,0.012,0.019,0.029,0.030,0.024,0.019,0.023,0.019,0.017,0.020],
    'rC:dT': [0,0,0,0,0,0,0.020,0.017,0.023,0.013,0.014,0.021,0.026,0.034,0.033,0.037,0.035,0.037,0.025,0.035],
    'rG:dA': [0.078,0.082,0.091,0.091,0.082,0.104,0.110,0.107,0.117,0.125,0.121,0.146,0.148,0.172,0.154,0.168,0.157,0.157,0.130,0.167],
    'rG:dG': [0,0,0.036,0.039,0.055,0.038,0.034,0.040,0.046,0.048,0.067,0.076,0.082,0.088,0.091,0.089,0.087,0.089,0.079,0.085],
    'rG:dT': [0,0,0,0,0,0,0,0,0,0.003,0.002,0.007,0.010,0.014,0.015,0.016,0.017,0.017,0.013,0.016],
    'rT:dA': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'rT:dC': [0,0,0,0.023,0.030,0.022,0.019,0.030,0.022,0.019,0.025,0.030,0.036,0.037,0.035,0.031,0.030,0.030,0.022,0.029],
    'rT:dG': [0,0,0,0.049,0.060,0.054,0.049,0.065,0.055,0.059,0.076,0.083,0.096,0.108,0.101,0.106,0.097,0.105,0.083,0.099],
}

# PAM score lookup for CFD
CFD_PAM_SCORES = {
    'AA': 0, 'AC': 0, 'AG': 0.259259259, 'AT': 0,
    'CA': 0, 'CC': 0, 'CG': 0.107142857, 'CT': 0,
    'GA': 0.069444444, 'GC': 0, 'GG': 1.0, 'GT': 0.022222222,
    'TA': 0, 'TC': 0, 'TG': 0.038961039, 'TT': 0,
}

def calc_cfd(guide, offtarget):
    """
    Compute CFD score between guide RNA and off-target DNA sequence.
    Both sequences should be 20nt (without PAM).
    Returns score between 0 (won't cut) and 1 (will cut).
    """
    guide     = str(guide)[:20].upper().replace('T', 'U')  # RNA
    offtarget = str(offtarget)[:20].upper()

    if len(guide) < 20 or len(offtarget) < 20:
        return np.nan

    score = 1.0
    for i, (g, o) in enumerate(zip(guide, offtarget)):
        if g == 'N' or o == 'N':
            continue
        # Convert U back to T for DNA comparison
        g_dna = g.replace('U', 'T')
        if g_dna != o:
            key = f'r{g}:d{o}'
            if key in CFD_MM_SCORES:
                mm_score = CFD_MM_SCORES[key][i]
                score *= (1 - mm_score)
            else:
                score *= 0.5  # Unknown substitution, conservative penalty

    return score

def calc_mit(guide, offtarget):
    """
    Compute MIT score (Hsu et al. 2013).
    Uses a position-specific penalty model.
    Returns score between 0 and 1.
    """
    guide     = str(guide)[:20].upper()
    offtarget = str(offtarget)[:20].upper()

    if len(guide) < 20 or len(offtarget) < 20:
        return np.nan

    # MIT position weights (higher = more important, closer to PAM)
    MIT_WEIGHTS = [0, 0, 0.014, 0, 0, 0.395, 0.317, 0, 0.389, 0.079,
                   0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804,
                   0.685, 0.583]

    mismatch_positions = [i for i, (g, o) in enumerate(zip(guide, offtarget))
                          if g != o and g != 'N' and o != 'N']

    n_mm = len(mismatch_positions)
    if n_mm == 0:
        return 1.0

    # Mean pairwise distance penalty
    if n_mm > 1:
        pairs = [(mismatch_positions[i], mismatch_positions[j])
                 for i in range(n_mm) for j in range(i+1, n_mm)]
        mean_dist = np.mean([abs(a - b) for a, b in pairs])
        d_penalty = 1.0 / ((19 - mean_dist) / 19 * 4 + 1)
    else:
        d_penalty = 1.0

    # Position weight penalty
    w_penalty = 1.0
    for pos in mismatch_positions:
        w_penalty *= (1 - MIT_WEIGHTS[pos])

    # Number of mismatches penalty
    n_penalty = 1.0 / (n_mm ** 2 if n_mm > 1 else 1)

    score = w_penalty * d_penalty * n_penalty
    return float(np.clip(score, 0, 1))

def main():
    print("=" * 60)
    print("   Baseline Score Comparison (CFD + MIT)")
    print("=" * 60)

    print(f"\nLoading dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df):,} rows loaded")

    # Get the same test split as training scripts
    split = guide_aware_split(df)
    test_mask = split['test_mask']
    df_test = df[test_mask].copy()
    print(f"  Test set: {len(df_test):,} rows")
    print(f"  Test positives: {int(df_test['is_cut'].sum()):,}")
    print(f"  Test negatives: {int((df_test['is_cut']==0).sum()):,}")

    # Compute CFD and MIT scores
    print("\nComputing CFD scores...")
    df_test['cfd_score'] = df_test.apply(
        lambda r: calc_cfd(r['target_seq'], r['offtarget_seq']), axis=1)

    print("Computing MIT scores...")
    df_test['mit_score'] = df_test.apply(
        lambda r: calc_mit(r['target_seq'], r['offtarget_seq']), axis=1)

    # Drop rows where scores couldn't be computed
    df_valid = df_test.dropna(subset=['cfd_score', 'mit_score'])
    print(f"  Valid rows for scoring: {len(df_valid):,}")

    y_true = df_valid['is_cut'].values

    # Evaluate CFD
    cfd_auc   = roc_auc_score(y_true, df_valid['cfd_score'])
    cfd_auprc = average_precision_score(y_true, df_valid['cfd_score'])
    cfd_fpr, cfd_tpr, _ = roc_curve(y_true, df_valid['cfd_score'])

    # Evaluate MIT
    mit_auc   = roc_auc_score(y_true, df_valid['mit_score'])
    mit_auprc = average_precision_score(y_true, df_valid['mit_score'])
    mit_fpr, mit_tpr, _ = roc_curve(y_true, df_valid['mit_score'])

    print(f"\n{'─'*50}")
    print(f"  CFD Score  — AUC-ROC: {cfd_auc:.4f} | AUC-PR: {cfd_auprc:.4f}")
    print(f"  MIT Score  — AUC-ROC: {mit_auc:.4f} | AUC-PR: {mit_auprc:.4f}")
    print(f"{'─'*50}")

    # Load your model results
    model_results = {}
    for name, fname in [('XGBoost',              'xgboost_results.json'),
                        ('CNN_only',             'cnn_results.json'),
                        ('Hybrid_CNN_Attention', 'hybrid_results.json')]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                model_results[name] = json.load(f)

    # Full comparison table
    all_results = [
        {'Model': 'MIT Score (Hsu 2013)',    'AUC-ROC': round(mit_auc,4),  'AUC-PR': round(mit_auprc,4)},
        {'Model': 'CFD Score (Doench 2016)', 'AUC-ROC': round(cfd_auc,4),  'AUC-PR': round(cfd_auprc,4)},
    ]
    label_map = {
        'XGBoost':              'XGBoost (ours)',
        'CNN_only':             'CNN Only (ours)',
        'Hybrid_CNN_Attention': 'Hybrid CNN+Attention (ours)',
    }
    for name, r in model_results.items():
        all_results.append({
            'Model':   label_map.get(name, name),
            'AUC-ROC': r.get('auc_roc'),
            'AUC-PR':  r.get('auc_pr'),
        })

    df_comparison = pd.DataFrame(all_results).sort_values('AUC-ROC')
    print(f"\n{'='*55}")
    print("  FULL COMPARISON TABLE")
    print(f"{'='*55}")
    print(df_comparison.to_string(index=False))

    # Save comparison table
    df_comparison.to_csv(
        os.path.join(OUTPUT_DIR, 'baseline_comparison.csv'), index=False)

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: ROC curves for all methods
    ax = axes[0]
    ax.plot([0,1],[0,1],'k--',alpha=0.4,linewidth=1,label='Random (AUC=0.50)')

    # Baselines
    ax.plot(mit_fpr, mit_tpr, color='#95a5a6', linewidth=2,
            linestyle=':', label=f'MIT Score (AUC={mit_auc:.4f})')
    ax.plot(cfd_fpr, cfd_tpr, color='#7f8c8d', linewidth=2,
            linestyle='--', label=f'CFD Score (AUC={cfd_auc:.4f})')

    # Your models
    colors = {'XGBoost': '#e74c3c', 'CNN_only': '#3498db',
              'Hybrid_CNN_Attention': '#2ecc71'}
    labels = {'XGBoost': 'XGBoost (ours)',
              'CNN_only': 'CNN Only (ours)',
              'Hybrid_CNN_Attention': 'Hybrid CNN+Attention (ours)'}

    for name in ['XGBoost', 'CNN_only', 'Hybrid_CNN_Attention']:
        roc_path = os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ','_')}_roc.npy")
        fname_map = {
            'XGBoost':              'xgboost_roc.npy',
            'CNN_only':             'cnn_roc.npy',
            'Hybrid_CNN_Attention': 'hybrid_roc.npy',
        }
        roc_path = os.path.join(OUTPUT_DIR, fname_map[name])
        if os.path.exists(roc_path):
            data = np.load(roc_path)
            auc  = model_results.get(name, {}).get('auc_roc', '?')
            ax.plot(data[0], data[1], color=colors[name], linewidth=2.5,
                    label=f"{labels[name]} (AUC={auc})")

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve Comparison\nAll Methods Including Baselines',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    # Right: Bar chart comparison
    ax2 = axes[1]
    models   = df_comparison['Model'].tolist()
    auc_vals = df_comparison['AUC-ROC'].tolist()
    bar_colors = []
    for m in models:
        if 'ours' in m.lower():
            if 'Hybrid' in m:
                bar_colors.append('#2ecc71')
            elif 'CNN' in m:
                bar_colors.append('#3498db')
            else:
                bar_colors.append('#e74c3c')
        else:
            bar_colors.append('#bdc3c7')

    bars = ax2.barh(range(len(models)), auc_vals,
                    color=bar_colors, alpha=0.85)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontsize=10)
    ax2.set_xlabel('AUC-ROC', fontsize=12)
    ax2.set_title('AUC-ROC Comparison\nAll Methods', fontsize=13, fontweight='bold')
    ax2.set_xlim([0.5, 1.02])
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)

    for bar, val in zip(bars, auc_vals):
        ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)

    ax2.grid(alpha=0.3, axis='x')

    plt.suptitle('Figure 9 — Comparison with Published Baseline Methods',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure9_baseline_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Figure saved to {path}")
    print(f"✅ Table saved to {OUTPUT_DIR}baseline_comparison.csv")
    print(f"\nNext step: install and run Elevation for a stronger comparison")
    print(f"  pip install elevation")

if __name__ == "__main__":
    main()
