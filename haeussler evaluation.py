import sys
sys.path.insert(0, '/Users/moana/Downloads/')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import joblib
from itertools import combinations
from scipy.stats import spearmanr
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from shared_utils import encode_pair, BIO_FEATURES

HAEUSSLER_PATH = '/Users/moana/Downloads/13059_2016_1012_MOESM2_ESM.tsv'
MODEL_PATH     = '/Users/moana/Downloads/model_results/hybrid_model.pt'
OUTPUT_DIR     = '/Users/moana/Downloads/model_results/'
TRAIN_DATA     = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'

DARK  = '#2C2C2A'
MID   = '#5F5E5A'
SPINE = '#D3D1C7'

POS_WEIGHTS   = torch.tensor([
    0.05,0.05,0.05,0.07,0.07,0.07,0.08,0.08,0.08,0.10,
    0.12,0.15,0.20,0.30,0.40,0.55,0.70,0.85,0.95,1.00
], dtype=torch.float32)
SEED_POSITIONS = set(range(16, 20))
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── Model definition ──────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, n_bio=12):
        super().__init__()
        self.conv1    = nn.Conv1d(8,  64,  kernel_size=3, padding=1)
        self.conv2    = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1      = nn.BatchNorm1d(64)
        self.bn2      = nn.BatchNorm1d(128)
        self.attn     = nn.MultiheadAttention(128, num_heads=4, batch_first=True, dropout=0.1)
        self.register_buffer('pos_weights', POS_WEIGHTS)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_fc   = nn.Linear(128, 64)
        self.bio_fc1  = nn.Linear(n_bio, 64)
        self.bio_fc2  = nn.Linear(64, 32)
        self.bio_bn   = nn.BatchNorm1d(64)
        self.fuse1    = nn.Linear(96, 64)
        self.fuse2    = nn.Linear(64, 32)
        self.fuse3    = nn.Linear(32, 1)
        self.drop     = nn.Dropout(0.3)
        self.relu     = nn.ReLU()

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

# ── Sequence utilities ────────────────────────────────────────────────────────
def clean_seq(seq):
    return str(seq).upper().strip().replace('-', '')

def gc_content(seq):
    seq = seq[:20].upper()
    return (seq.count('G') + seq.count('C')) / len(seq) if seq else 0.0

def shannon_entropy(seq):
    seq    = seq[:20].upper()
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

def compute_bio_features(target, offtarget):
    target    = clean_seq(target)
    offtarget = clean_seq(offtarget)
    seed_mm, nonseed_mm, pam_score = get_mismatch_info(target, offtarget)
    mm_count = seed_mm + nonseed_mm
    return np.array([
        gc_content(target), gc_content(offtarget),
        shannon_entropy(target), shannon_entropy(offtarget),
        float(mm_count), float(seed_mm), float(nonseed_mm),
        float(pam_score),
        float(len(target.replace('N', ''))),
        float(get_mfe(target)),
        0.0, 0.0,   # has_bulge, atac_accessible unknown
    ], dtype=np.float32)

# ── CFD / MIT scores ──────────────────────────────────────────────────────────
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
MIT_W = [0,0,0.014,0,0,0.395,0.317,0,0.389,0.079,
         0.445,0.508,0.613,0.851,0.732,0.828,0.615,0.804,0.685,0.583]

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
            score *= (1 - CFD_MM_SCORES.get(key, [0.5]*20)[i])
    return float(score)

def calc_mit(guide, offtarget):
    guide     = str(guide)[:20].upper()
    offtarget = str(offtarget)[:20].upper()
    if len(guide) < 20 or len(offtarget) < 20:
        return np.nan
    mm_pos = [i for i, (g, o) in enumerate(zip(guide, offtarget))
              if g != o and g != 'N' and o != 'N']
    n = len(mm_pos)
    if n == 0:
        return 1.0
    d_pen = (1.0 / ((19 - np.mean([abs(a-b) for a,b in combinations(mm_pos,2)])) / 19 * 4 + 1)
             if n > 1 else 1.0)
    w_pen = 1.0
    for p in mm_pos:
        w_pen *= (1 - MIT_W[p])
    return float(np.clip(w_pen * d_pen / (n**2 if n > 1 else 1), 0, 1))

# ── Elevation features ────────────────────────────────────────────────────────
def elevation_features(target, offtarget):
    t = str(target)[:20].upper()
    o = str(offtarget)[:20].upper()
    mm_vec = np.zeros(20, dtype=np.float32)
    for i, (tb, ob) in enumerate(zip(t, o)):
        if tb != ob and tb != 'N' and ob != 'N':
            mm_vec[i] = 1.0
    mm_pos   = np.where(mm_vec)[0].tolist()
    n_mm     = len(mm_pos)
    pair_vec = np.zeros(190, dtype=np.float32)
    pair_idx = {(i,j): k for k,(i,j) in enumerate(combinations(range(20), 2))}
    for i, j in combinations(mm_pos, 2):
        pair_vec[pair_idx[(i,j)]] = 1.0
    mean_dist = (np.mean([abs(a-b) for a,b in combinations(mm_pos,2)])
                 if n_mm > 1 else 0.0)
    return np.concatenate([
        [calc_cfd(target, offtarget)],
        [float(n_mm)],
        mm_vec,
        pair_vec,
        [mean_dist],
    ]).astype(np.float32)

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color(SPINE)
    ax.tick_params(colors=MID)
    ax.xaxis.label.set_color(DARK)
    ax.yaxis.label.set_color(DARK)
    ax.title.set_color(DARK)

def main():
    print("=" * 60)
    print("   Haeussler 2016 Benchmark Evaluation")
    print("=" * 60)

    df = pd.read_csv(HAEUSSLER_PATH, sep='\t')
    print(f"\nLoaded {len(df)} off-target pairs")
    print(f"Guides: {df['name'].nunique()}")
    print(f"readFraction range: {df['readFraction'].min():.4f} — "
          f"{df['readFraction'].max():.4f}")

    df['guide_clean'] = df['guideSeq'].apply(lambda s: clean_seq(str(s))[:20])
    df['ot_clean']    = df['otSeq'].apply(lambda s: clean_seq(str(s))[:20])
    df['is_cut']      = (df['readFraction'] > 0).astype(int)
    print(f"Binary labels: {df['is_cut'].sum()} cuts, "
          f"{(df['is_cut']==0).sum()} non-cuts")

    # ── Hybrid model predictions ──────────────────────────────────────────────
    print("\nLoading hybrid model...")
    model = HybridModel(n_bio=12)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.to(DEVICE).eval()

    print("Computing hybrid predictions...")
    hybrid_probs = []
    for _, row in df.iterrows():
        try:
            bio  = compute_bio_features(row['guide_clean'], row['ot_clean'])
            xc   = torch.tensor(
                       encode_pair(row['guide_clean'], row['ot_clean']),
                       dtype=torch.float32).unsqueeze(0).to(DEVICE)
            xf   = torch.tensor(bio).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = torch.sigmoid(model(xc, xf)).item()
        except:
            p = np.nan
        hybrid_probs.append(p)
    df['hybrid_prob'] = hybrid_probs

    # ── CFD / MIT scores ──────────────────────────────────────────────────────
    print("Computing CFD and MIT scores...")
    df['cfd_score'] = [calc_cfd(r.guide_clean, r.ot_clean)
                       for r in df.itertuples()]
    df['mit_score'] = [calc_mit(r.guide_clean, r.ot_clean)
                       for r in df.itertuples()]

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("Computing logistic regression predictions...")
    print("  (retraining on full training data — takes ~1 min)")
    df_train = pd.read_csv(TRAIN_DATA, low_memory=False)
    from shared_utils import guide_aware_split
    split    = guide_aware_split(df_train)
    df_tr    = df_train[split['train_mask']]
    X_tr     = df_tr[BIO_FEATURES].fillna(0).values.astype(np.float32)
    y_tr     = df_tr['is_cut'].values
    pos      = y_tr.sum(); neg = len(y_tr) - pos
    cw       = {0: pos/len(y_tr), 1: neg/len(y_tr)}
    logreg   = Pipeline([('sc', StandardScaler()),
                         ('clf', LogisticRegression(max_iter=1000,
                                                     class_weight=cw,
                                                     solver='lbfgs',
                                                     C=1.0,
                                                     random_state=42))])
    logreg.fit(X_tr, y_tr)

    logreg_probs = []
    for _, row in df.iterrows():
        bio = compute_bio_features(row['guide_clean'], row['ot_clean'])
        p   = logreg.predict_proba(bio.reshape(1, -1))[0, 1]
        logreg_probs.append(float(p))
    df['logreg_prob'] = logreg_probs

    # ── Elevation ─────────────────────────────────────────────────────────────
    print("Computing Elevation-equivalent predictions...")
    print("  (retraining GBM on full training data — takes ~5 min)")
    from sklearn.ensemble import GradientBoostingClassifier
    X_el     = np.array([elevation_features(r.target_seq, r.offtarget_seq)
                         for r in df_tr.itertuples()])
    sw       = np.where(y_tr == 1, neg/pos, 1.0)
    elev_clf = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                           learning_rate=0.05, subsample=0.8,
                                           min_samples_leaf=20, random_state=42)
    elev_clf.fit(X_el, y_tr, sample_weight=sw)

    elev_probs = []
    for _, row in df.iterrows():
        feats = elevation_features(row['guide_clean'], row['ot_clean'])
        p     = elev_clf.predict_proba(feats.reshape(1, -1))[0, 1]
        elev_probs.append(float(p))
    df['elevation_prob'] = elev_probs

    # ── Evaluate all methods ──────────────────────────────────────────────────
    df_valid = df.dropna(subset=['hybrid_prob', 'cfd_score', 'mit_score',
                                 'logreg_prob', 'elevation_prob'])
    print(f"\nValid rows: {len(df_valid)}")

    y_true = df_valid['is_cut'].values
    y_freq = df_valid['readFraction'].values

    score_cols = {
        'MIT Score':                df_valid['mit_score'].values,
        'CFD Score':                df_valid['cfd_score'].values,
        'otScore (Haeussler)':      df_valid['otScore'].values,
        'Logistic Regression':      df_valid['logreg_prob'].values,
        'Elevation-equivalent':     df_valid['elevation_prob'].values,
        'Hybrid CNN+Attention':     df_valid['hybrid_prob'].values,
    }

    results = {}
    for name, scores in score_cols.items():
        try:
            auc   = roc_auc_score(y_true, scores)
            auprc = average_precision_score(y_true, scores)
            sp_r, sp_p = spearmanr(scores, y_freq)
            results[name] = {
                'AUC-ROC':    round(auc,  4),
                'AUC-PR':     round(auprc,4),
                'Spearman_r': round(sp_r, 4),
                'Spearman_p': round(sp_p, 6),
            }
        except Exception as e:
            print(f"  {name} failed: {e}")

    print(f"\n{'='*60}")
    print("  HAEUSSLER BENCHMARK RESULTS")
    print(f"{'='*60}")
    df_res = pd.DataFrame(results).T
    print(df_res.to_string())

    df_valid.to_csv(os.path.join(OUTPUT_DIR, 'haeussler_evaluation.csv'),
                    index=False)
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'haeussler_results.csv'))

    # ── Figure ────────────────────────────────────────────────────────────────
    COLORS = {
        'MIT Score':            '#888780',
        'CFD Score':            '#5F5E5A',
        'otScore (Haeussler)':  '#D85A30',
        'Logistic Regression':  '#7F77DD',
        'Elevation-equivalent': '#D4537E',
        'Hybrid CNN+Attention': '#1D9E75',
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('white')

    # Left: ROC curves
    ax = axes[0]
    ax.plot([0,1],[0,1], color=SPINE, linewidth=1, linestyle='--',
            label='Random (AUC = 0.50)')
    order = ['MIT Score', 'CFD Score', 'otScore (Haeussler)',
             'Logistic Regression', 'Elevation-equivalent',
             'Hybrid CNN+Attention']
    for name in order:
        scores = score_cols[name]
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = results[name]['AUC-ROC']
            lw  = 3.0 if 'Hybrid' in name else 1.8
            ax.plot(fpr, tpr, color=COLORS[name], linewidth=lw,
                    label=f'{name}  (AUC = {auc:.4f})')
        except:
            pass
    ax.set_xlabel('False positive rate', fontsize=12)
    ax.set_ylabel('True positive rate', fontsize=12)
    ax.set_title('ROC curves\nHaeussler 2016 benchmark', fontsize=12, pad=8)
    ax.legend(fontsize=7.5, loc='lower right', frameon=False)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.yaxis.grid(True, color=SPINE, linewidth=0.5)
    ax.set_axisbelow(True)
    style_ax(ax)

    # Middle: scatter hybrid vs measured
    ax2 = axes[1]
    ax2.scatter(df_valid['hybrid_prob'], df_valid['readFraction'],
                alpha=0.5, color='#1D9E75', s=20)
    sp_r = results.get('Hybrid CNN+Attention', {}).get('Spearman_r', 0)
    sp_p = results.get('Hybrid CNN+Attention', {}).get('Spearman_p', 0)
    ax2.set_xlabel('Predicted cleavage probability', fontsize=12)
    ax2.set_ylabel('Measured read fraction', fontsize=12)
    ax2.set_title(f'Predicted vs measured cleavage\n'
                  f'Spearman ρ = {sp_r:.4f}  (p = {sp_p:.4f})',
                  fontsize=12, pad=8)
    ax2.yaxis.grid(True, color=SPINE, linewidth=0.5)
    ax2.set_axisbelow(True)
    style_ax(ax2)

    # Right: AUC bar chart
    ax3      = axes[2]
    names    = list(results.keys())
    auc_vals = [results[n]['AUC-ROC'] for n in names]
    bars     = ax3.barh(range(len(names)), auc_vals,
                        color=[COLORS.get(n, '#B4B2A9') for n in names],
                        alpha=0.85)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=9.5)
    ax3.set_xlabel('AUC-ROC', fontsize=12)
    ax3.set_title('AUC-ROC comparison\nHaeussler benchmark', fontsize=12, pad=8)
    ax3.set_xlim([0, 1.05])
    ax3.axvline(x=0.5, color=SPINE, linestyle='--', linewidth=1)
    for bar, val in zip(bars, auc_vals):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9, color=MID)
    ax3.xaxis.grid(True, color=SPINE, linewidth=0.5)
    ax3.set_axisbelow(True)
    style_ax(ax3)

    fig.suptitle('Figure 10 — Validation on Haeussler 2016 benchmark dataset',
                 fontsize=13, color=DARK, fontweight='500')
    plt.tight_layout(pad=2.0)
    path = os.path.join(OUTPUT_DIR, 'figure10_haeussler_benchmark.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved figure: {path}")
    print(f"Saved results: {OUTPUT_DIR}haeussler_results.csv")

if __name__ == "__main__":
    main()
