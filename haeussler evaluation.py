"""
Haeussler 2016 Benchmark Evaluation
--------------------------------------
Evaluates your hybrid model against the Haeussler 2016 benchmark dataset
(Additional file 2 from Genome Biology 2016).

This dataset contains 718 experimentally measured off-target cleavage
frequencies (readFraction) for 26 guide RNAs. It is the standard
benchmark used to compare CRISPR off-target scoring tools.

Two analyses:
1. AUC-ROC — binary classification (readFraction > 0 = cut)
2. Spearman correlation — how well predicted probability tracks
   actual cleavage frequency (continuous)

Usage:
    python evaluate_haeussler.py

Output:
    model_results/haeussler_evaluation.csv
    model_results/figure10_haeussler_benchmark.png
"""

import sys
sys.path.insert(0, '/Users/moana/Downloads/')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import os

from shared_utils import encode_pair, BIO_FEATURES, RANDOM_SEED
from scipy.stats import entropy as scipy_entropy

# ── PATHS ─────────────────────────────────────────────────────────────────────
HAEUSSLER_PATH = '/Users/moana/Downloads/13059_2016_1012_MOESM2_ESM.tsv'
MODEL_PATH     = '/Users/moana/Downloads/model_results/hybrid_model.pt'
OUTPUT_DIR     = '/Users/moana/Downloads/model_results/'
# ─────────────────────────────────────────────────────────────────────────────

POS_WEIGHTS = torch.tensor([
    0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.10,
    0.12, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00
], dtype=torch.float32)

BASES         = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
SEED_POSITIONS = set(range(16, 20))

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

class HybridModel(nn.Module):
    def __init__(self, n_bio=12):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.attn  = nn.MultiheadAttention(128, num_heads=4, batch_first=True, dropout=0.1)
        self.register_buffer('pos_weights', POS_WEIGHTS)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_fc   = nn.Linear(128, 64)
        self.bio_fc1  = nn.Linear(n_bio, 64)
        self.bio_fc2  = nn.Linear(64, 32)
        self.bio_bn   = nn.BatchNorm1d(64)
        self.fuse1 = nn.Linear(96, 64)
        self.fuse2 = nn.Linear(64, 32)
        self.fuse3 = nn.Linear(32, 1)
        self.drop  = nn.Dropout(0.3)
        self.relu  = nn.ReLU()

    def forward(self, x_seq, x_bio):
        x = x_seq.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x = x * self.pos_weights.unsqueeze(0).unsqueeze(-1)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.seq_pool(x).squeeze(-1)
        x_s = self.drop(self.relu(self.seq_fc(x)))
        x_b = self.relu(self.bio_bn(self.bio_fc1(x_bio)))
        x_b = self.drop(self.relu(self.bio_fc2(x_b)))
        x_f = torch.cat([x_s, x_b], dim=1)
        x_f = self.drop(self.relu(self.fuse1(x_f)))
        x_f = self.drop(self.relu(self.fuse2(x_f)))
        return self.fuse3(x_f).squeeze(-1)

def clean_seq(seq):
    return str(seq).upper().strip().replace('-', '')

def gc_content(seq):
    seq = seq[:20].upper()
    return (seq.count('G') + seq.count('C')) / len(seq) if seq else 0.0

def shannon_entropy(seq):
    seq = seq[:20].upper()
    counts = [seq.count(b) / len(seq) for b in 'ACGT']
    counts = [c for c in counts if c > 0]
    return float(scipy_entropy(counts, base=2)) if counts else 0.0

def get_mismatch_info(target, offtarget):
    t = str(target)[:20].upper()
    o = str(offtarget)[:20].upper()
    seed_mm = nonseed_mm = 0
    weighted_score = 0.0
    for i, (tb, ob) in enumerate(zip(t, o)):
        if tb != ob and tb != 'N':
            w = POS_WEIGHTS[i].item()
            weighted_score += w
            if i in SEED_POSITIONS:
                seed_mm += 1
            else:
                nonseed_mm += 1
    return seed_mm, nonseed_mm, round(weighted_score, 4)

def get_mfe(seq):
    try:
        import RNA
        rna = seq[:20].upper().replace('T', 'U').replace('N', 'A')
        _, mfe = RNA.fold(rna)
        return round(mfe, 3)
    except:
        return -2.0

def compute_features(target, offtarget):
    target    = clean_seq(target)
    offtarget = clean_seq(offtarget)
    seed_mm, nonseed_mm, pam_score = get_mismatch_info(target, offtarget)
    mm_count = seed_mm + nonseed_mm
    return np.array([
        gc_content(target),
        gc_content(offtarget),
        shannon_entropy(target),
        shannon_entropy(offtarget),
        float(mm_count),
        float(seed_mm),
        float(nonseed_mm),
        float(pam_score),
        float(len(target.replace('N', ''))),
        float(get_mfe(target)),
        0.0,   # has_bulge — set to 0, no dash info in this dataset
        0.0,   # atac_accessible — unknown for this dataset
    ], dtype=np.float32)

# CFD score for comparison
CFD_MM_SCORES = {
    'rA:dA': [0,0,0.014,0,0,0.395,0.317,0,0.389,0.079,0.445,0.508,0.613,0.851,0.732,0.828,0.615,0.804,0.685,0.583],
    'rA:dG': [0.059,0.059,0.078,0.082,0.044,0.105,0.094,0.110,0.073,0.129,0.084,0.181,0.183,0.213,0.237,0.207,0.222,0.186,0.145,0.237],
    'rC:dA': [0,0,0,0.025,0,0.018,0.028,0.050,0.051,0.050,0.064,0.080,0.101,0.124,0.108,0.138,0.129,0.150,0.117,0.108],
    'rC:dC': [0,0,0.019,0.020,0,0,0,0,0,0.002,0.012,0.019,0.029,0.030,0.024,0.019,0.023,0.019,0.017,0.020],
    'rC:dT': [0,0,0,0,0,0,0.020,0.017,0.023,0.013,0.014,0.021,0.026,0.034,0.033,0.037,0.035,0.037,0.025,0.035],
    'rG:dA': [0.078,0.082,0.091,0.091,0.082,0.104,0.110,0.107,0.117,0.125,0.121,0.146,0.148,0.172,0.154,0.168,0.157,0.157,0.130,0.167],
    'rG:dG': [0,0,0.036,0.039,0.055,0.038,0.034,0.040,0.046,0.048,0.067,0.076,0.082,0.088,0.091,0.089,0.087,0.089,0.079,0.085],
    'rG:dT': [0,0,0,0,0,0,0,0,0,0.003,0.002,0.007,0.010,0.014,0.015,0.016,0.017,0.017,0.013,0.016],
    'rT:dC': [0,0,0,0.023,0.030,0.022,0.019,0.030,0.022,0.019,0.025,0.030,0.036,0.037,0.035,0.031,0.030,0.030,0.022,0.029],
    'rT:dG': [0,0,0,0.049,0.060,0.054,0.049,0.065,0.055,0.059,0.076,0.083,0.096,0.108,0.101,0.106,0.097,0.105,0.083,0.099],
}

def calc_cfd(guide, offtarget):
    guide     = str(guide)[:20].upper().replace('T', 'U')
    offtarget = str(offtarget)[:20].upper()
    if len(guide) < 20 or len(offtarget) < 20:
        return np.nan
    score = 1.0
    for i, (g, o) in enumerate(zip(guide, offtarget)):
        if g == 'N' or o == 'N':
            continue
        g_dna = g.replace('U', 'T')
        if g_dna != o:
            key = f'r{g}:d{o}'
            if key in CFD_MM_SCORES:
                score *= (1 - CFD_MM_SCORES[key][i])
            else:
                score *= 0.5
    return score

def calc_mit(guide, offtarget):
    guide     = str(guide)[:20].upper()
    offtarget = str(offtarget)[:20].upper()
    if len(guide) < 20 or len(offtarget) < 20:
        return np.nan
    MIT_W = [0,0,0.014,0,0,0.395,0.317,0,0.389,0.079,0.445,0.508,0.613,0.851,0.732,0.828,0.615,0.804,0.685,0.583]
    mm_pos = [i for i,(g,o) in enumerate(zip(guide,offtarget)) if g!=o and g!='N' and o!='N']
    n = len(mm_pos)
    if n == 0: return 1.0
    if n > 1:
        pairs = [(mm_pos[i],mm_pos[j]) for i in range(n) for j in range(i+1,n)]
        mean_dist = np.mean([abs(a-b) for a,b in pairs])
        d_pen = 1.0 / ((19 - mean_dist) / 19 * 4 + 1)
    else:
        d_pen = 1.0
    w_pen = 1.0
    for p in mm_pos:
        w_pen *= (1 - MIT_W[p])
    n_pen = 1.0 / (n**2 if n > 1 else 1)
    return float(np.clip(w_pen * d_pen * n_pen, 0, 1))

def main():
    print("=" * 60)
    print("   Haeussler 2016 Benchmark Evaluation")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(HAEUSSLER_PATH, sep='\t')
    print(f"\nLoaded {len(df)} off-target pairs")
    print(f"Guides: {df['name'].nunique()}")
    print(f"readFraction range: {df['readFraction'].min():.4f} — {df['readFraction'].max():.4f}")

    # Clean sequences
    df['guide_clean'] = df['guideSeq'].apply(lambda s: clean_seq(str(s))[:20])
    df['ot_clean']    = df['otSeq'].apply(lambda s: clean_seq(str(s))[:20])

    # Binary label: readFraction > 0 means it cuts
    df['is_cut'] = (df['readFraction'] > 0).astype(int)
    print(f"\nBinary labels: {df['is_cut'].sum()} cuts, {(df['is_cut']==0).sum()} non-cuts")

    # Load model
    print("\nLoading hybrid model...")
    model = HybridModel(n_bio=12)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()

    # Compute model predictions
    print("Computing hybrid model predictions...")
    hybrid_probs = []
    for _, row in df.iterrows():
        try:
            features = compute_features(row['guide_clean'], row['ot_clean'])
            x_seq = torch.tensor(
                encode_pair(row['guide_clean'], row['ot_clean']),
                dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_bio = torch.tensor(features).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(model(x_seq, x_bio)).item()
        except:
            prob = np.nan
        hybrid_probs.append(prob)
    df['hybrid_prob'] = hybrid_probs

    # Compute CFD and MIT scores
    print("Computing CFD and MIT scores...")
    df['cfd_score'] = df.apply(lambda r: calc_cfd(r['guide_clean'], r['ot_clean']), axis=1)
    df['mit_score'] = df.apply(lambda r: calc_mit(r['guide_clean'], r['ot_clean']), axis=1)

    # Drop NaN rows
    df_valid = df.dropna(subset=['hybrid_prob', 'cfd_score', 'mit_score'])
    print(f"Valid rows: {len(df_valid)}")

    y_true  = df_valid['is_cut'].values
    y_freq  = df_valid['readFraction'].values

    # ── AUC EVALUATION ───────────────────────────────────────────────────────
    results = {}
    for name, scores in [
        ('MIT Score',              df_valid['mit_score'].values),
        ('CFD Score',              df_valid['cfd_score'].values),
        ('Hybrid CNN+Attention',   df_valid['hybrid_prob'].values),
        ('otScore (Haeussler)',    df_valid['otScore'].values),
    ]:
        try:
            auc   = roc_auc_score(y_true, scores)
            auprc = average_precision_score(y_true, scores)
            sp_r, sp_p = spearmanr(scores, y_freq)
            results[name] = {
                'AUC-ROC': round(auc, 4),
                'AUC-PR':  round(auprc, 4),
                'Spearman_r': round(sp_r, 4),
                'Spearman_p': round(sp_p, 6),
            }
        except Exception as e:
            print(f"  {name} failed: {e}")

    print(f"\n{'='*60}")
    print("  HAEUSSLER BENCHMARK RESULTS")
    print(f"{'='*60}")
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())

    # Save
    df_valid.to_csv(os.path.join(OUTPUT_DIR, 'haeussler_evaluation.csv'), index=False)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'haeussler_results.csv'))

    # ── FIGURES ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: ROC curves
    ax = axes[0]
    ax.plot([0,1],[0,1],'k--',alpha=0.4,linewidth=1,label='Random')
    colors = {'MIT Score': '#95a5a6', 'CFD Score': '#7f8c8d',
              'Hybrid CNN+Attention': '#2ecc71', 'otScore (Haeussler)': '#e74c3c'}
    for name, scores in [
        ('MIT Score',           df_valid['mit_score'].values),
        ('CFD Score',           df_valid['cfd_score'].values),
        ('otScore (Haeussler)', df_valid['otScore'].values),
        ('Hybrid CNN+Attention',df_valid['hybrid_prob'].values),
    ]:
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = results[name]['AUC-ROC']
            ax.plot(fpr, tpr, color=colors[name], linewidth=2.5,
                    label=f'{name} (AUC={auc})')
        except:
            pass
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves\nHaeussler 2016 Benchmark', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    # Middle: Spearman correlation scatter
    ax2 = axes[1]
    ax2.scatter(df_valid['hybrid_prob'], df_valid['readFraction'],
                alpha=0.5, color='#2ecc71', s=20)
    sp_r = results.get('Hybrid CNN+Attention', {}).get('Spearman_r', 0)
    ax2.set_xlabel('Predicted Cleavage Probability', fontsize=12)
    ax2.set_ylabel('Measured Read Fraction', fontsize=12)
    ax2.set_title(f'Predicted vs Measured Cleavage\nSpearman r = {sp_r:.4f}',
                  fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Right: AUC comparison bar chart
    ax3 = axes[2]
    names = list(results.keys())
    aucs  = [results[n]['AUC-ROC'] for n in names]
    bar_colors = ['#2ecc71' if 'Hybrid' in n else
                  '#e74c3c' if 'Haeussler' in n else '#bdc3c7'
                  for n in names]
    bars = ax3.barh(range(len(names)), aucs, color=bar_colors, alpha=0.85)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=10)
    ax3.set_xlabel('AUC-ROC', fontsize=12)
    ax3.set_title('AUC Comparison\nHaeussler Benchmark', fontsize=12, fontweight='bold')
    ax3.set_xlim([0.4, 1.02])
    for bar, val in zip(bars, aucs):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)
    ax3.grid(alpha=0.3, axis='x')

    plt.suptitle('Figure 10 — Validation on Haeussler 2016 Benchmark Dataset',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure10_haeussler_benchmark.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Figure saved to {path}")
    print(f"✅ Results saved to {OUTPUT_DIR}haeussler_results.csv")

if __name__ == "__main__":
    main()
