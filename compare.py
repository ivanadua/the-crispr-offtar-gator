"""
Compare Models + Generate Paper Figures
-----------------------------------------
Run after all training scripts complete.

Generates:
  figure_roc_comparison.png   ← Paper Figure 1
  figure_shap_importance.png  ← Paper Figure 2
  figure_ablation.png         ← Paper Figure 3
  final_comparison_table.csv  ← Paper Table 1

Usage:
    python compare_models.py
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
import matplotlib.patches as mpatches
import shap
import xgboost as xgb

from shared_utils import BIO_FEATURES

OUTPUT_DIR = '/Users/moana/Downloads/model_results/'

COLORS = {
    'XGBoost':              '#e74c3c',
    'CNN_only':             '#3498db',
    'Hybrid_CNN_Attention': '#2ecc71',
}
LABELS = {
    'XGBoost':              'XGBoost (baseline)',
    'CNN_only':             'CNN Only (sequence)',
    'Hybrid_CNN_Attention': 'Hybrid CNN + Attention (ours)',
}

def load_results():
    results = {}
    for name, fname in [('XGBoost',              'xgboost_results.json'),
                        ('CNN_only',             'cnn_results.json'),
                        ('Hybrid_CNN_Attention', 'hybrid_results.json')]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"  ⚠️  Missing: {fname}")
    return results

def plot_roc(results):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0,1],[0,1],'k--',alpha=0.4,linewidth=1,label='Random (AUC=0.50)')
    for name in ['XGBoost','CNN_only','Hybrid_CNN_Attention']:
        roc_path = os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ','_')}_roc.npy")
        # Try alternative naming
        for candidate in [
            os.path.join(OUTPUT_DIR, 'xgboost_roc.npy'),
            os.path.join(OUTPUT_DIR, 'cnn_roc.npy'),
            os.path.join(OUTPUT_DIR, 'hybrid_roc.npy'),
        ]:
            pass

        fname_map = {
            'XGBoost':              'xgboost_roc.npy',
            'CNN_only':             'cnn_roc.npy',
            'Hybrid_CNN_Attention': 'hybrid_roc.npy',
        }
        roc_path = os.path.join(OUTPUT_DIR, fname_map[name])
        if not os.path.exists(roc_path):
            continue
        data = np.load(roc_path)
        auc  = results.get(name, {}).get('auc_roc', '?')
        ax.plot(data[0], data[1],
                color=COLORS[name], linewidth=2.5,
                label=f"{LABELS[name]} (AUC={auc})")

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve Comparison\nCRISPR Off-Target Cleavage Prediction',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure_roc_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC comparison → {path}")

def plot_shap():
    shap_path  = os.path.join(OUTPUT_DIR, 'xgboost_shap_values.npy')
    if not os.path.exists(shap_path):
        print("  ⚠️  SHAP values missing — skipping")
        return

    shap_values = np.load(shap_path)
    mean_shap   = np.abs(shap_values).mean(axis=0)
    n           = min(len(mean_shap), len(BIO_FEATURES))
    mean_shap   = mean_shap[:n]
    names       = BIO_FEATURES[:n]

    idx    = np.argsort(mean_shap)[::-1][:12]
    s_vals = mean_shap[idx]
    s_names = [names[i] for i in idx]

    name_map = {
        'gc_content_target':      'GC content (target)',
        'gc_content_offtarget':   'GC content (off-target)',
        'shannon_entropy_target': 'Shannon entropy (target)',
        'shannon_entropy_offt':   'Shannon entropy (off-target)',
        'mismatch_count':         'Total mismatches',
        'seed_mismatches':        'Seed region mismatches (pos 17-20)',
        'nonseed_mismatches':     'Non-seed mismatches',
        'pam_proximity_score':    'PAM proximity score',
        'guide_length':           'Guide RNA length',
        'mfe':                    'MFE (RNA stability)',
        'has_bulge':              'Has bulge',
        'atac_accessible':        'Chromatin accessibility (ATAC)',
    }
    display = [name_map.get(n, n) for n in s_names]
    colors  = ['#e74c3c' if any(k in n for k in ['seed','pam','mismatch'])
               else '#3498db' for n in s_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(display)), s_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(display[::-1], fontsize=11)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Feature Importance (SHAP) — XGBoost\nCRISPR Off-Target Prediction',
                 fontsize=13, fontweight='bold')
    r = mpatches.Patch(color='#e74c3c', label='Mismatch/positional')
    b = mpatches.Patch(color='#3498db', label='Sequence/structural')
    ax.legend(handles=[r, b], fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure_shap_importance.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP plot → {path}")

def plot_ablation():
    path = os.path.join(OUTPUT_DIR, 'ablation_table.csv')
    if not os.path.exists(path):
        print("  ⚠️  Ablation results missing — skipping")
        return

    df      = pd.read_csv(path)
    labels  = ['Run A\n(Hard only)', 'Run B\n(Hard+Medium)', 'Run C\n(All tiers)']
    metrics = ['auc_roc', 'auc_pr', 'f1']
    mlabels = ['AUC-ROC', 'AUC-PR', 'F1']
    colors  = ['#e67e22', '#9b59b6', '#1abc9c']
    x       = np.arange(len(labels))
    w       = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (m, ml, c) in enumerate(zip(metrics, mlabels, colors)):
        vals = df[m].values
        bars = ax.bar(x + i*w, vals, w, label=ml, color=c, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Negative Set', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study — Effect of Negative Set Quality',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'figure_ablation.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Ablation plot → {out}")

def make_table(results):
    rows = []
    for name, r in results.items():
        rows.append({
            'Model':     LABELS.get(name, name),
            'AUC-ROC':   r.get('auc_roc'),
            'AUC-PR':    r.get('auc_pr'),
            'Precision': r.get('precision'),
            'Recall':    r.get('recall'),
            'F1':        r.get('f1'),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'final_comparison_table.csv')
    df.to_csv(path, index=False)
    print(f"\n{'='*55}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*55}")
    print(df.to_string(index=False))
    print(f"\n✅ Table saved → {path}")

def main():
    print("=" * 55)
    print("   Compare Models + Paper Figures")
    print("=" * 55)

    results = load_results()
    if not results:
        print("\nNo results found. Run training scripts first.")
        return

    plot_roc(results)
    plot_shap()
    plot_ablation()
    make_table(results)

    print(f"\n{'='*55}")
    print("  ALL DONE — check model_results/ for figures")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
