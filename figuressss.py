"""
Paper Interpretability Figures
--------------------------------
Generates biological insight figures from the trained Hybrid CNN+Attention model.

Figures produced:
  oogaboogafig4_positional_sensitivity.png  — How mismatch position affects cleavage
  oogaboogafig5_attention_heatmap.png       — What the attention layer learned
  oogaboogafig6_mismatch_count_effect.png   — Cleavage vs mismatch count (seed vs non-seed)
  oogaboogafig7_chromatin_effect.png        — Open vs closed chromatin cleavage rates
  oogaboogafig8_mismatch_type_analysis.png  — Transition vs transversion tolerance

Usage:
    python generate_paper_figures.py
    (Run after train_hybrid.py)
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
import matplotlib.gridspec as gridspec
from collections import defaultdict

from shared_utils import encode_pair, BIO_FEATURES, RANDOM_SEED, SEQ_LEN

INPUT_PATH  = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'
MODEL_PATH  = '/Users/moana/Downloads/model_results/hybrid_model.pt'
OUTPUT_DIR  = '/Users/moana/Downloads/model_results/'

BASES    = ['A', 'C', 'G', 'T']
BASES_D  = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

POS_WEIGHTS = torch.tensor([
    0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.10,
    0.12, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00
], dtype=torch.float32)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# ── MODEL DEFINITION (must match train_hybrid.py) ─────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, n_bio):
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

    def forward(self, x_seq, x_bio, return_attn=False):
        x = x_seq.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x = x * self.pos_weights.unsqueeze(0).unsqueeze(-1)
        x, attn_weights = self.attn(x, x, x)
        if return_attn:
            return attn_weights
        x = x.permute(0, 2, 1)
        x = self.seq_pool(x).squeeze(-1)
        x_s = self.drop(self.relu(self.seq_fc(x)))
        x_b = self.relu(self.bio_bn(self.bio_fc1(x_bio)))
        x_b = self.drop(self.relu(self.bio_fc2(x_b)))
        x_f = torch.cat([x_s, x_b], dim=1)
        x_f = self.drop(self.relu(self.fuse1(x_f)))
        x_f = self.drop(self.relu(self.fuse2(x_f)))
        return self.fuse3(x_f).squeeze(-1)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model(n_bio):
    model = HybridModel(n_bio=n_bio)
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

def get_bio_features(df):
    for col in BIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
    return df[BIO_FEATURES].values.astype(np.float32)

def predict(model, seq_pair, bio_features):
    """Get cleavage probability for a single sequence pair."""
    x_seq = torch.tensor(encode_pair(*seq_pair), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_bio = torch.tensor(bio_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(x_seq, x_bio)
        prob  = torch.sigmoid(logit).item()
    return prob

# ── FIGURE 4: POSITIONAL SENSITIVITY ─────────────────────────────────────────
def figure4_positional_sensitivity(model, df):
    print("\nFigure 4: Positional Sensitivity...")

    # Use multiple real on-target guides for robustness
    on_targets = df[df['mismatch_count'] == 0]['target_seq'].unique()[:10].tolist()
    if not on_targets:
        on_targets = df['target_seq'].unique()[:10].tolist()

    all_position_scores = defaultdict(list)

    for guide in on_targets:
        # Get median bio features for this guide
        guide_rows = df[df['target_seq'] == guide]
        bio = get_bio_features(guide_rows)
        median_bio = bio.mean(axis=0)

        for pos in range(20):
            pos_scores = []
            orig_base = guide[pos].upper()
            for base in BASES:
                if base == orig_base or orig_base == 'N':
                    continue
                # Create mutant
                mutant = list(guide[:20])
                mutant[pos] = base
                mutant_seq = ''.join(mutant) + guide[20:]

                # Adjust bio features for this mutation
                bio_copy = median_bio.copy()
                # Update mismatch-related features
                bio_copy[BIO_FEATURES.index('mismatch_count')]    = 1.0
                bio_copy[BIO_FEATURES.index('seed_mismatches')]   = 1.0 if pos >= 16 else 0.0
                bio_copy[BIO_FEATURES.index('nonseed_mismatches')]= 0.0 if pos >= 16 else 1.0
                bio_copy[BIO_FEATURES.index('pam_proximity_score')] = float(POS_WEIGHTS[pos].item())

                prob = predict(model, (guide, mutant_seq), bio_copy)
                pos_scores.append(prob)

            if pos_scores:
                all_position_scores[pos + 1].append(np.mean(pos_scores))

    positions = list(range(1, 21))
    mean_scores = [np.mean(all_position_scores[p]) if all_position_scores[p] else 0
                   for p in positions]
    std_scores  = [np.std(all_position_scores[p])  if all_position_scores[p] else 0
                   for p in positions]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(positions, mean_scores, 'o-', color='#2c3e50', linewidth=2.5, markersize=6)
    ax.fill_between(positions,
                    np.array(mean_scores) - np.array(std_scores),
                    np.array(mean_scores) + np.array(std_scores),
                    alpha=0.2, color='#2c3e50', label='±1 SD across guides')
    ax.fill_between(range(13, 21), 0, 1.1, color='#e74c3c', alpha=0.12, label='Seed Region (pos 13-20)')
    ax.set_xlabel('Position (1=Distal, 20=Proximal to PAM)', fontsize=13)
    ax.set_ylabel('Mean Predicted Cleavage Probability', fontsize=13)
    ax.set_title('Figure 4 — Positional Sensitivity of Cas9 Cleavage\nHybrid CNN+Attention Model',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(positions)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}oogaboogafig4_positional_sensitivity.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to {path}")

# ── FIGURE 5: ATTENTION HEATMAP ───────────────────────────────────────────────
def figure5_attention_heatmap(model, df):
    print("\nFigure 5: Attention Heatmap...")

    # Sample real positive pairs
    pos_rows = df[df['is_cut'] == 1].sample(n=min(500, len(df)), random_state=RANDOM_SEED)
    bio = get_bio_features(pos_rows)

    all_attn = []
    for i, (_, row) in enumerate(pos_rows.iterrows()):
        x_seq = torch.tensor(
            encode_pair(row['target_seq'], row['offtarget_seq']),
            dtype=torch.float32).unsqueeze(0).to(DEVICE)
        x_bio = torch.tensor(bio[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            attn = model(x_seq, x_bio, return_attn=True)
            # attn shape can be (1, heads, seq, seq) or (1, seq, seq)
            attn_np = attn.squeeze(0).cpu().numpy()
            if attn_np.ndim == 3:
                attn_np = attn_np.mean(axis=0)  # average over heads → (20, 20)
            elif attn_np.ndim == 1:
                attn_np = np.outer(attn_np, attn_np)  # fallback
            all_attn.append(attn_np)

    mean_attn = np.mean(all_attn, axis=0)  # (20, 20)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mean_attn, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    ax.set_xlabel('Key Position (attended to)', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title('Figure 5 — Self-Attention Weights\nHybrid Model (averaged over 500 positive pairs)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.set_xticklabels([str(i+1) for i in range(20)], fontsize=8)
    ax.set_yticklabels([str(i+1) for i in range(20)], fontsize=8)

    # Highlight seed region
    for i in range(12, 20):
        ax.axhline(y=i-0.5, color='red', linewidth=0.5, alpha=0.4)
        ax.axvline(x=i-0.5, color='red', linewidth=0.5, alpha=0.4)

    ax.text(15.5, -1.5, 'Seed Region', color='red', fontsize=9, ha='center')
    plt.tight_layout()
    path = f"{OUTPUT_DIR}oogaboogafig5_attention_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to {path}")

# ── FIGURE 6: MISMATCH COUNT EFFECT ──────────────────────────────────────────
def figure6_mismatch_count(model, df):
    print("\nFigure 6: Mismatch Count Effect...")

    # Use experimental data only, sample from both positives and negatives
    exp_data = df[df['source'] != 'synthetic'].copy()
    syn_data = df[df['source'] == 'synthetic'].copy()

    # Sample positives and negatives separately
    pos_sample = exp_data[exp_data['is_cut'] == 1].sample(
        n=min(3000, len(exp_data[exp_data['is_cut']==1])), random_state=RANDOM_SEED)
    neg_sample = syn_data.sample(
        n=min(2000, len(syn_data)), random_state=RANDOM_SEED)

    def get_probs(subset):
        bio = get_bio_features(subset)
        probs = []
        for i, (_, row) in enumerate(subset.iterrows()):
            prob = predict(model, (row['target_seq'], row['offtarget_seq']), bio[i])
            probs.append(prob)
        subset = subset.copy()
        subset['pred_prob'] = probs
        return subset

    pos_sample = get_probs(pos_sample)
    neg_sample = get_probs(neg_sample)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean predicted probability by mismatch count
    # Positives (green) vs Negatives (red) — should separate clearly
    ax = axes[0]
    mm_range = range(0, 10)

    pos_means = []
    neg_means = []
    pos_sems  = []
    neg_sems  = []
    valid_mm  = []

    for mm in mm_range:
        p = pos_sample[pos_sample['mismatch_count'] == mm]['pred_prob'].values
        n = neg_sample[neg_sample['mismatch_count'] == mm]['pred_prob'].values
        if len(p) >= 5 and len(n) >= 3:
            pos_means.append(np.mean(p))
            neg_means.append(np.mean(n))
            pos_sems.append(np.std(p) / np.sqrt(len(p)))
            neg_sems.append(np.std(n) / np.sqrt(len(n)))
            valid_mm.append(mm)

    ax.errorbar(valid_mm, pos_means, yerr=pos_sems, fmt='o-',
                color='#2ecc71', linewidth=2.5, markersize=7,
                label='Confirmed cuts (is_cut=1)', capsize=4)
    ax.errorbar(valid_mm, neg_means, yerr=neg_sems, fmt='s--',
                color='#e74c3c', linewidth=2.5, markersize=7,
                label='Synthetic negatives (is_cut=0)', capsize=4)
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Decision boundary')
    ax.set_xlabel('Mismatch Count', fontsize=12)
    ax.set_ylabel('Mean Predicted Cleavage Probability', fontsize=12)
    ax.set_title('Positives vs Negatives\nby Mismatch Count', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Right: seed vs non-seed effect on positives only
    # Shows that even among confirmed cuts, seed mismatches reduce probability
    ax2 = axes[1]
    seed_pos    = pos_sample[pos_sample['seed_mismatches'] >= 1]
    nonseed_pos = pos_sample[pos_sample['seed_mismatches'] == 0]

    for subset, label, color in [
        (seed_pos,    'Has seed mismatch\n(pos 17-20)',    '#e74c3c'),
        (nonseed_pos, 'No seed mismatch\n(pos 1-16 only)', '#2ecc71'),
    ]:
        mm_vals = sorted(subset['mismatch_count'].unique())
        means   = [subset[subset['mismatch_count']==m]['pred_prob'].mean()
                   for m in mm_vals if len(subset[subset['mismatch_count']==m]) >= 5]
        mm_filt = [m for m in mm_vals
                   if len(subset[subset['mismatch_count']==m]) >= 5]
        if means:
            ax2.plot(mm_filt, means, 'o-', color=color,
                     linewidth=2.5, markersize=7, label=label)

    ax2.set_xlabel('Mismatch Count', fontsize=12)
    ax2.set_ylabel('Mean Predicted Cleavage Probability', fontsize=12)
    ax2.set_title('Seed vs Non-Seed Mismatches\n(Confirmed Cuts Only)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.suptitle('Figure 6 — Effect of Mismatch Count and Position on Predicted Cleavage',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}oogaboogafig6_mismatch_count_effect.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to {path}")

# ── FIGURE 7: CHROMATIN ACCESSIBILITY ────────────────────────────────────────
def figure7_chromatin(model, df):
    print("\nFigure 7: Chromatin Accessibility Effect...")

    annotated = df[df['atac_accessible'].notna()].copy()
    if len(annotated) < 100:
        print("  ⚠️  Not enough ATAC data — skipping")
        return

    sample = annotated.sample(n=min(3000, len(annotated)), random_state=RANDOM_SEED)
    bio    = get_bio_features(sample)

    probs = []
    for i, (_, row) in enumerate(sample.iterrows()):
        prob = predict(model, (row['target_seq'], row['offtarget_seq']), bio[i])
        probs.append(prob)

    sample = sample.copy()
    sample['pred_prob'] = probs

    open_chrom   = sample[sample['atac_accessible'] == 1]['pred_prob']
    closed_chrom = sample[sample['atac_accessible'] == 0]['pred_prob']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violin plot
    ax = axes[0]
    parts = ax.violinplot([closed_chrom.values, open_chrom.values],
                          positions=[1, 2], showmedians=True)
    colors = ['#95a5a6', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Closed Chromatin', 'Open Chromatin'], fontsize=11)
    ax.set_ylabel('Predicted Cleavage Probability', fontsize=12)
    ax.set_title('Cleavage Probability by\nChromatin State', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Add mean annotations
    for x, data, color in [(1, closed_chrom, '#7f8c8d'), (2, open_chrom, '#c0392b')]:
        ax.text(x, data.mean() + 0.02, f'μ={data.mean():.3f}',
                ha='center', fontsize=10, color=color, fontweight='bold')

    # Bar chart by cell line
    ax2 = axes[1]
    cell_lines = sample['cell_line'].dropna().unique()
    x_pos = np.arange(len(cell_lines))
    open_means   = []
    closed_means = []
    for cl in cell_lines:
        cl_data = sample[sample['cell_line'] == cl]
        open_means.append(cl_data[cl_data['atac_accessible']==1]['pred_prob'].mean())
        closed_means.append(cl_data[cl_data['atac_accessible']==0]['pred_prob'].mean())

    w = 0.35
    ax2.bar(x_pos - w/2, closed_means, w, label='Closed', color='#95a5a6', alpha=0.8)
    ax2.bar(x_pos + w/2, open_means,   w, label='Open',   color='#e74c3c', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cell_lines, fontsize=10, rotation=15)
    ax2.set_ylabel('Mean Predicted Cleavage Probability', fontsize=11)
    ax2.set_title('By Cell Line', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    plt.suptitle('Figure 7 — Chromatin Accessibility Effect on Predicted Cleavage',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}oogaboogafig7_chromatin_effect.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to {path}")

# ── FIGURE 8: MISMATCH TYPE ANALYSIS ─────────────────────────────────────────
def figure8_mismatch_types(model, df):
    print("\nFigure 8: Mismatch Type Analysis...")

    # Transitions: A↔G, C↔T (same purine/pyrimidine class)
    # Transversions: A↔C, A↔T, G↔C, G↔T (different classes)
    TRANSITIONS    = {'A>G','G>A','C>T','T>C'}
    TRANSVERSIONS  = {'A>C','C>A','A>T','T>A','G>C','C>G','G>T','T>G'}

    sample = df[df['mismatch_count'] == 1].sample(
        n=min(2000, len(df[df['mismatch_count']==1])), random_state=RANDOM_SEED)
    bio = get_bio_features(sample)

    type_probs = defaultdict(list)
    for i, (_, row) in enumerate(sample.iterrows()):
        types_str = str(row.get('mismatch_types', '[]'))
        try:
            types = eval(types_str)
        except:
            continue
        if not types:
            continue
        prob = predict(model, (row['target_seq'], row['offtarget_seq']), bio[i])
        mm_type = types[0]
        if mm_type in TRANSITIONS:
            type_probs['Transition\n(A↔G, C↔T)'].append(prob)
        elif mm_type in TRANSVERSIONS:
            type_probs['Transversion\n(A↔C/T, G↔C/T)'].append(prob)
        type_probs[mm_type].append(prob)

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: transitions vs transversions
    ax = axes[0]
    groups = ['Transition\n(A↔G, C↔T)', 'Transversion\n(A↔C/T, G↔C/T)']
    means  = [np.mean(type_probs[g]) if type_probs[g] else 0 for g in groups]
    sems   = [np.std(type_probs[g])/np.sqrt(len(type_probs[g])+1) for g in groups]
    colors = ['#3498db', '#e74c3c']
    bars   = ax.bar(groups, means, color=colors, alpha=0.8, yerr=sems, capsize=5)
    ax.set_ylabel('Mean Predicted Cleavage Probability', fontsize=12)
    ax.set_title('Transition vs Transversion\nMismatch Tolerance', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, mean+0.02,
                f'{mean:.3f}', ha='center', fontweight='bold')

    # Right: all mismatch types
    ax2 = axes[1]
    all_types = {k: v for k, v in type_probs.items()
                 if '>' in k and len(v) >= 10}
    sorted_types  = sorted(all_types.keys(),
                           key=lambda k: np.mean(all_types[k]), reverse=True)
    type_means    = [np.mean(all_types[t]) for t in sorted_types]
    type_colors   = ['#3498db' if t in TRANSITIONS else '#e74c3c'
                     for t in sorted_types]

    ax2.barh(range(len(sorted_types)), type_means,
             color=type_colors, alpha=0.8)
    ax2.set_yticks(range(len(sorted_types)))
    ax2.set_yticklabels(sorted_types, fontsize=10)
    ax2.set_xlabel('Mean Predicted Cleavage Probability', fontsize=12)
    ax2.set_title('By Specific Mismatch Type\n(single mismatch pairs)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend = [Patch(color='#3498db', label='Transition'),
              Patch(color='#e74c3c', label='Transversion')]
    ax2.legend(handles=legend, fontsize=10)

    plt.suptitle('Figure 8 — Mismatch Type Tolerance Analysis',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}oogaboogafig8_mismatch_type_analysis.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved to {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   Paper Interpretability Figures")
    print("=" * 60)

    print(f"\nLoading dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df):,} rows loaded")

    n_bio = len(BIO_FEATURES)
    print(f"\nLoading hybrid model from {MODEL_PATH}...")
    model = load_model(n_bio)
    print(f"  Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

    figure4_positional_sensitivity(model, df)
    figure5_attention_heatmap(model, df)
    figure6_mismatch_count(model, df)
    figure7_chromatin(model, df)
    figure8_mismatch_types(model, df)

    print(f"\n{'='*60}")
    print("  ALL FIGURES COMPLETE")
    print(f"{'='*60}")
    print(f"\nFiles saved to {OUTPUT_DIR}:")
    for f in sorted([f for f in __import__('os').listdir(OUTPUT_DIR)
                     if f.startswith('figure')]):
        size = __import__('os').path.getsize(f"{OUTPUT_DIR}{f}")
        print(f"  {f:50s} {size/1024:.1f} KB")

if __name__ == "__main__":
    main()
