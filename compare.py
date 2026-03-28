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

from shared_utils import BIO_FEATURES

OUTPUT_DIR = '/Users/moana/Downloads/model_results/'

DARK  = '#2C2C2A'
MID   = '#5F5E5A'
SPINE = '#D3D1C7'

COLORS = {
    'LogReg':               '#7F77DD',
    'Elevation':            '#D4537E',
    'XGBoost':              '#D85A30',
    'CNN_only':             '#378ADD',
    'Hybrid_CNN_Attention': '#1D9E75',
}
LABELS = {
    'LogReg':               'Logistic Regression (baseline)',
    'Elevation':            'Elevation-equivalent (Listgarten 2018)',
    'XGBoost':              'XGBoost (baseline)',
    'CNN_only':             'CNN Only (sequence)',
    'Hybrid_CNN_Attention': 'Hybrid CNN + Attention (ours)',
}

def load_results():
    results = {}
    for name, fname in [('LogReg',               'logreg_results.json'),
                        ('Elevation',            'elevation_results.json'),
                        ('XGBoost',              'xgboost_results.json'),
                        ('CNN_only',             'cnn_results.json'),
                        ('Hybrid_CNN_Attention', 'hybrid_results.json')]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"  Missing: {fname}")
    return results

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color(SPINE)
    ax.tick_params(colors=MID)
    ax.xaxis.label.set_color(DARK)
    ax.yaxis.label.set_color(DARK)
    ax.title.set_color(DARK)

def plot_roc(results):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.plot([0,1],[0,1], color=SPINE, linewidth=1, linestyle='--',
            label='Random (AUC = 0.50)')

    fname_map = {
        'LogReg':               'logreg_roc.npy',
        'Elevation':            'elevation_roc.npy',
        'XGBoost':              'xgboost_roc.npy',
        'CNN_only':             'cnn_roc.npy',
        'Hybrid_CNN_Attention': 'hybrid_roc.npy',
    }
    for name in ['LogReg', 'Elevation', 'CNN_only', 'XGBoost',
                 'Hybrid_CNN_Attention']:
        roc_path = os.path.join(OUTPUT_DIR, fname_map[name])
        if not os.path.exists(roc_path):
            print(f"  Missing ROC file: {fname_map[name]}")
            continue
        data = np.load(roc_path)
        auc  = results.get(name, {}).get('auc_roc', '?')
        lw   = 3.0 if name == 'Hybrid_CNN_Attention' else 2.0
        ax.plot(data[0], data[1],
                color=COLORS[name], linewidth=lw,
                label=f"{LABELS[name]}  (AUC = {auc:.4f})")

    ax.set_xlabel('False positive rate', fontsize=12)
    ax.set_ylabel('True positive rate', fontsize=12)
    ax.set_title('ROC curve comparison', fontsize=13, pad=10)
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.yaxis.grid(True, color=SPINE, linewidth=0.5)
    ax.set_axisbelow(True)
    style_ax(ax)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure_roc_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")

def plot_shap():
    shap_path = os.path.join(OUTPUT_DIR, 'xgboost_shap_values.npy')
    if not os.path.exists(shap_path):
        print("  SHAP values missing — skipping")
        return

    shap_values = np.load(shap_path)
    mean_shap   = np.abs(shap_values).mean(axis=0)
    n           = min(len(mean_shap), len(BIO_FEATURES))
    mean_shap   = mean_shap[:n]
    names       = BIO_FEATURES[:n]
    idx         = np.argsort(mean_shap)[::-1][:12]
    s_vals      = mean_shap[idx]
    s_names     = [names[i] for i in idx]

    name_map = {
        'gc_content_target':      'GC content (target)',
        'gc_content_offtarget':   'GC content (off-target)',
        'shannon_entropy_target': 'Shannon entropy (target)',
        'shannon_entropy_offt':   'Shannon entropy (off-target)',
        'mismatch_count':         'Total mismatches',
        'seed_mismatches':        'Seed region mismatches',
        'nonseed_mismatches':     'Non-seed mismatches',
        'pam_proximity_score':    'PAM proximity score',
        'guide_length':           'Guide RNA length',
        'mfe':                    'MFE (RNA stability)',
        'has_bulge':              'Has bulge',
        'atac_accessible':        'Chromatin accessibility (ATAC)',
    }
    display = [name_map.get(n, n) for n in s_names]
    colors  = ['#D85A30' if any(k in n for k in ['seed', 'pam', 'mismatch'])
               else '#378ADD' for n in s_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    ax.barh(range(len(display)), s_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(display[::-1], fontsize=11)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Feature importance (SHAP) — XGBoost baseline',
                 fontsize=13, pad=10)
    r = mpatches.Patch(color='#D85A30', label='Mismatch / positional features')
    b = mpatches.Patch(color='#378ADD', label='Sequence / structural features')
    ax.legend(handles=[r, b], fontsize=10, frameon=False)
    ax.xaxis.grid(True, color=SPINE, linewidth=0.5)
    ax.set_axisbelow(True)
    style_ax(ax)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure_shap_importance.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")

def plot_ablation():
    path = os.path.join(OUTPUT_DIR, 'ablation_table.csv')
    if not os.path.exists(path):
        print("  Ablation results missing — skipping")
        return

    df      = pd.read_csv(path)
    labels  = ['Run A\n(Hard only)', 'Run B\n(Hard + medium)', 'Run C\n(All tiers)']
    metrics = ['auc_roc', 'auc_pr', 'f1']
    mlabels = ['AUC-ROC', 'AUC-PR', 'F1']
    colors  = ['#D85A30', '#7F77DD', '#1D9E75']
    x       = np.arange(len(labels))
    w       = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    for i, (m, ml, c) in enumerate(zip(metrics, mlabels, colors)):
        vals = df[m].values
        bars = ax.bar(x + i*w, vals, w, label=ml, color=c, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=9, color=MID)

    ax.set_xlabel('Negative set', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation study — effect of negative set quality',
                 fontsize=13, pad=10)
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=10, frameon=False)
    ax.yaxis.grid(True, color=SPINE, linewidth=0.5)
    ax.set_axisbelow(True)
    style_ax(ax)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'figure_ablation.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")

def make_table(results):
    order = ['LogReg', 'Elevation', 'XGBoost', 'CNN_only',
             'Hybrid_CNN_Attention']
    rows = []
    for name in order:
        r = results.get(name, {})
        if r:
            rows.append({
                'Model':     LABELS.get(name, name),
                'AUC-ROC':   r.get('auc_roc'),
                'AUC-PR':    r.get('auc_pr'),
                'Precision': r.get('precision'),
                'Recall':    r.get('recall'),
                'F1':        r.get('f1'),
            })
    df   = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'final_comparison_table.csv')
    df.to_csv(path, index=False)
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved: {path}")

def main():
    print("=" * 60)
    print("   Compare Models + Paper Figures")
    print("=" * 60)
    results = load_results()
    if not results:
        print("No results found. Run training scripts first.")
        return
    plot_roc(results)
    plot_shap()
    plot_ablation()
    make_table(results)
    print(f"\n{'='*60}")
    print("  ALL DONE — check model_results/ for figures")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
