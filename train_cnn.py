"""
Model 2: CNN Only (Sequence-Aware)
------------------------------------
Learns directly from raw DNA sequences.
No hand-crafted features — the model discovers patterns itself.
Guide-aware train/test split.

Usage:
    python train_cnn.py

Output:
    model_results/cnn_model.pt
    model_results/cnn_results.json
    model_results/cnn_roc.npy
"""

import sys
sys.path.insert(0, '/Users/moana/Downloads/')

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from shared_utils import (load_features, guide_aware_split,
                          evaluate, encode_pair, RANDOM_SEED,
                          SEQ_LEN)

INPUT_PATH = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'
OUTPUT_DIR = '/Users/moana/Downloads/model_results/'
BATCH_SIZE = 512
EPOCHS     = 50
LR         = 0.001
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8,   64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64,  128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(64, 32)
        self.fc2   = nn.Linear(32, 1)
        self.drop  = nn.Dropout(0.3)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt   = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

def main():
    print("=" * 55)
    print("   Model 2: CNN Only")
    print("=" * 55)

    import pandas as pd
    _, y, df = load_features(INPUT_PATH)

    print("\nEncoding sequences...")
    X_seq = np.array([
        encode_pair(row['target_seq'], row['offtarget_seq'])
        for _, row in df.iterrows()
    ], dtype=np.float32)

    split = guide_aware_split(df, X_seq=X_seq)
    X_tr, X_te = split['X_seq_train'], split['X_seq_test']
    y_tr, y_te = split['y_train'], split['y_test']

    train_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(SeqDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    model     = CNNModel().to(DEVICE)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_auc, best_state = 0, None

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            probs = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            val_auc = roc_auc_score(y_te, probs)
            scheduler.step(1 - val_auc)
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | AUC: {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())

    results, fpr, tpr = evaluate(y_te, np.array(probs), 'CNN_only')

    torch.save(best_state, os.path.join(OUTPUT_DIR, 'cnn_model.pt'))
    with open(os.path.join(OUTPUT_DIR, 'cnn_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, 'cnn_roc.npy'), np.array([fpr, tpr]))

    print(f"\n✅ CNN complete | AUC: {results['auc_roc']} | AUC-PR: {results['auc_pr']}")
    print("Next: python train_hybrid.py")

if __name__ == "__main__":
    main()
