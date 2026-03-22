"""
Ablation Study — Negative Tier Quality
-----------------------------------------
Trains the Hybrid model 3 times with different negative sets:

  Run A: Positives + Tier 1 only  (hard negatives,  ~3k rows)
  Run B: Positives + Tiers 1+2    (hard+medium,    ~33k rows)
  Run C: Positives + All tiers    (full set,       ~48k rows)

Shows that richer negative sets improve model performance.
This justifies the synthetic negative methodology in the paper.

Usage:
    python train_ablation.py
    (Run after train_hybrid.py)

Output:
    model_results/ablation_table.csv
    model_results/ablation_results.json
"""

import sys
sys.path.insert(0, '/Users/moana/Downloads/')

import numpy as np
import pandas as pd
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

from shared_utils import (load_features, guide_aware_split,
                          encode_pair, BIO_FEATURES, RANDOM_SEED, SEQ_LEN)

INPUT_PATH = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'
OUTPUT_DIR = '/Users/moana/Downloads/model_results/'
BATCH_SIZE = 512
EPOCHS     = 30   # Fewer epochs to save time
LR         = 0.001
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

POS_WEIGHTS = torch.tensor([
    0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.10,
    0.12, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00
], dtype=torch.float32)

class HybridDataset(Dataset):
    def __init__(self, X_seq, X_bio, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_bio = torch.tensor(X_bio, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X_seq[i], self.X_bio[i], self.y[i]

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

class FocalLoss(nn.Module):
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt  = torch.exp(-bce)
        return (0.8 * (1 - pt) ** 2.0 * bce).mean()

def run_one(df_sub, run_name):
    print(f"\n{'='*55}")
    print(f"  {run_name}")
    print(f"  Positives: {int(df_sub['is_cut'].sum()):,} | "
          f"Negatives: {int((df_sub['is_cut']==0).sum()):,}")
    print(f"{'='*55}")

    for col in BIO_FEATURES:
        if col in df_sub.columns:
            df_sub[col] = pd.to_numeric(df_sub[col], errors='coerce').fillna(0)
        else:
            df_sub[col] = 0.0

    X_bio = df_sub[BIO_FEATURES].values.astype(np.float32)
    X_seq = np.array([
        encode_pair(row['target_seq'], row['offtarget_seq'])
        for _, row in df_sub.iterrows()
    ], dtype=np.float32)

    split = guide_aware_split(df_sub, X_bio=X_bio, X_seq=X_seq)
    X_seq_tr, X_seq_te = split['X_seq_train'], split['X_seq_test']
    X_bio_tr, X_bio_te = split['X_bio_train'], split['X_bio_test']
    y_tr, y_te         = split['y_train'],     split['y_test']

    loader      = DataLoader(HybridDataset(X_seq_tr, X_bio_tr, y_tr),
                             batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HybridDataset(X_seq_te, X_bio_te, y_te),
                             batch_size=BATCH_SIZE, shuffle=False)

    model     = HybridModel(n_bio=X_bio.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = FocalLoss()
    best_auc, best_state = 0, None

    for epoch in range(EPOCHS):
        model.train()
        for xs, xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xs.to(DEVICE), xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            probs = []
            with torch.no_grad():
                for xs, xb, _ in test_loader:
                    probs.extend(torch.sigmoid(
                        model(xs.to(DEVICE), xb.to(DEVICE))).cpu().numpy())
            val_auc = roc_auc_score(y_te, probs)
            print(f"  Epoch {epoch+1}/{EPOCHS} | AUC: {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for xs, xb, _ in test_loader:
            probs.extend(torch.sigmoid(
                model(xs.to(DEVICE), xb.to(DEVICE))).cpu().numpy())

    y_prob = np.array(probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'run':          run_name,
        'n_positives':  int(df_sub['is_cut'].sum()),
        'n_negatives':  int((df_sub['is_cut']==0).sum()),
        'auc_roc':      round(float(roc_auc_score(y_te, y_prob)), 4),
        'auc_pr':       round(float(average_precision_score(y_te, y_prob)), 4),
        'precision':    round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        'recall':       round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        'f1':           round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
    }

def main():
    print("=" * 55)
    print("   Ablation Study — Negative Tier Quality")
    print("=" * 55)

    df = pd.read_csv(INPUT_PATH)
    pos = df[df['is_cut'] == 1].copy()
    neg = df[df['is_cut'] == 0].copy()

    print(f"\nPositives: {len(pos):,}")
    print(f"Negatives: {len(neg):,}")
    print(f"Tiers: {neg['negative_tier'].value_counts().to_dict()}")

    tier1   = neg[neg['negative_tier'] == 'hard_synthetic']
    tier1_2 = neg[neg['negative_tier'].isin(['hard_synthetic','medium_synthetic'])]
    all_neg = neg

    results = []
    for neg_subset, name in [
        (tier1,   'Run_A — Tier 1 only (hard)'),
        (tier1_2, 'Run_B — Tiers 1+2 (hard+medium)'),
        (all_neg, 'Run_C — All tiers (full)'),
    ]:
        df_sub = pd.concat([pos, neg_subset]).sample(
            frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        results.append(run_one(df_sub, name))

    df_results = pd.DataFrame(results)
    print(f"\n{'='*55}")
    print("  ABLATION TABLE")
    print(f"{'='*55}")
    print(df_results[['run','n_negatives','auc_roc','auc_pr','f1']].to_string(index=False))

    df_results.to_csv(os.path.join(OUTPUT_DIR, 'ablation_table.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Ablation study complete")
    print("Next: python compare_models.py")

if __name__ == "__main__":
    main()
