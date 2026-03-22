"""
CRISPR Off-Target Cleavage Predictor
--------------------------------------
Predicts the probability that Cas9 will cleave at an off-target site
given a guide RNA sequence and a candidate off-target sequence.

Usage:
    # Single pair
    python predict_offtarget.py \\
        --target  GGGTGGGGGGAGTTTGCTCCNGG \\
        --offtarget GGATGGGGGGAGTTTGCTCCNGG

    # Batch mode (CSV file with target_seq and offtarget_seq columns)
    python predict_offtarget.py --input sequences.csv --output predictions.csv

    # With chromatin accessibility
    python predict_offtarget.py \\
        --target GGGTGGGGGGAGTTTGCTCCNGG \\
        --offtarget GGATGGGGGGAGTTTGCTCCNGG \\
        --atac 1

Requirements:
    pip install torch pandas numpy scipy ViennaRNA
    Model file: hybrid_model.pt (in same directory or specify with --model)

Output:
    Cleavage probability (0-1)
    Risk level (LOW / MEDIUM / HIGH)
    Mismatch analysis
    Feature breakdown
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import entropy as scipy_entropy

# ── MODEL DEFINITION ──────────────────────────────────────────────────────────
POS_WEIGHTS = torch.tensor([
    0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.10,
    0.12, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00
], dtype=torch.float32)

BASES      = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
SEQ_LEN    = 20
N_BIO      = 12  # Number of biological features

SEED_POSITIONS    = set(range(16, 20))
POSITION_WEIGHTS  = POS_WEIGHTS.numpy()

class HybridModel(nn.Module):
    def __init__(self, n_bio=N_BIO):
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

# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def clean_seq(seq):
    return str(seq).upper().strip().replace('-', '')

def encode_pair(target, offtarget, length=SEQ_LEN):
    def one_hot(seq):
        m = np.zeros((length, 4), dtype=np.float32)
        for i, b in enumerate(str(seq)[:length].upper()):
            if b in BASES:
                m[i, BASES[b]] = 1.0
        return m
    return np.concatenate([one_hot(target), one_hot(offtarget)], axis=1)

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
    positions, types = [], []
    seed_mm = nonseed_mm = 0
    weighted_score = 0.0
    for i, (tb, ob) in enumerate(zip(t, o)):
        if tb != ob and tb != 'N':
            positions.append(i + 1)
            types.append(f"{tb}>{ob}")
            weighted_score += POSITION_WEIGHTS[i]
            if i in SEED_POSITIONS:
                seed_mm += 1
            else:
                nonseed_mm += 1
    return positions, seed_mm, nonseed_mm, round(weighted_score, 4), types

def get_mfe(seq):
    try:
        import RNA
        rna = seq[:20].upper().replace('T', 'U').replace('N', 'A')
        _, mfe = RNA.fold(rna)
        return round(mfe, 3)
    except ImportError:
        return -2.0   # Median fallback if ViennaRNA not installed

def has_bulge(target, offtarget):
    t = str(target)[:20].upper().replace('-', '')
    o = str(offtarget)[:20].upper().replace('-', '')
    return int(len(t) != len(o))

def compute_features(target, offtarget, atac=None):
    """Compute all 12 biological features for a sequence pair."""
    target    = clean_seq(target)
    offtarget = clean_seq(offtarget)

    positions, seed_mm, nonseed_mm, pam_score, types = get_mismatch_info(target, offtarget)
    mismatch_count = len(positions)

    features = np.array([
        gc_content(target),
        gc_content(offtarget),
        shannon_entropy(target),
        shannon_entropy(offtarget),
        float(mismatch_count),
        float(seed_mm),
        float(nonseed_mm),
        float(pam_score),
        float(len(target.replace('N', ''))),
        float(get_mfe(target)),
        float(has_bulge(target, offtarget)),
        float(atac) if atac is not None else 0.0,
    ], dtype=np.float32)

    return features, positions, types, seed_mm, nonseed_mm, pam_score

def risk_level(prob):
    if prob >= 0.7:
        return 'HIGH   ⚠️'
    elif prob >= 0.4:
        return 'MEDIUM ⚡'
    else:
        return 'LOW    ✅'

# ── PREDICTION ────────────────────────────────────────────────────────────────

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Make sure hybrid_model.pt is in the same directory.")
        sys.exit(1)
    model = HybridModel(n_bio=N_BIO)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

def predict_single(model, target, offtarget, atac=None, verbose=True):
    target    = clean_seq(target)
    offtarget = clean_seq(offtarget)

    features, positions, types, seed_mm, nonseed_mm, pam_score = \
        compute_features(target, offtarget, atac)

    x_seq = torch.tensor(
        encode_pair(target, offtarget), dtype=torch.float32).unsqueeze(0)
    x_bio = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(model(x_seq, x_bio)).item()

    if verbose:
        print(f"\n{'='*55}")
        print(f"  CRISPR Off-Target Prediction")
        print(f"{'='*55}")
        print(f"  Target:      {target[:20]} + {target[20:] if len(target)>20 else 'NGG'}")
        print(f"  Off-target:  {offtarget[:20]} + {offtarget[20:] if len(offtarget)>20 else ''}")
        print(f"{'─'*55}")
        print(f"  Cleavage probability: {prob:.4f}")
        print(f"  Risk level:           {risk_level(prob)}")
        print(f"{'─'*55}")
        print(f"  Mismatch count:       {len(positions)}")
        print(f"  Mismatch positions:   {positions} (1=distal, 20=PAM-proximal)")
        print(f"  Mismatch types:       {types}")
        print(f"  Seed mismatches:      {seed_mm} (positions 17-20)")
        print(f"  Non-seed mismatches:  {nonseed_mm}")
        print(f"  PAM proximity score:  {pam_score:.4f}")
        print(f"  GC content (target):  {features[0]:.3f}")
        print(f"  MFE (RNA stability):  {features[9]:.2f} kcal/mol")
        if atac is not None:
            print(f"  Chromatin access.:    {'Open' if atac else 'Closed'}")
        print(f"{'='*55}\n")

    return prob

def predict_batch(model, input_csv, output_csv):
    df = pd.read_csv(input_csv)
    required = ['target_seq', 'offtarget_seq']
    for col in required:
        if col not in df.columns:
            print(f"ERROR: CSV must have columns: {required}")
            sys.exit(1)

    print(f"Processing {len(df):,} sequence pairs...")
    probs = []
    for i, row in df.iterrows():
        atac = row.get('atac_accessible', None)
        try:
            prob = predict_single(model, row['target_seq'],
                                  row['offtarget_seq'], atac, verbose=False)
        except Exception as e:
            prob = np.nan
        probs.append(prob)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)} done...")

    df['cleavage_probability'] = probs
    df['risk_level'] = df['cleavage_probability'].apply(
        lambda p: 'HIGH' if p >= 0.7 else ('MEDIUM' if p >= 0.4 else 'LOW')
        if not np.isnan(p) else 'UNKNOWN'
    )
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to {output_csv}")
    print(f"   HIGH risk:   {(df['risk_level']=='HIGH').sum():,} sites")
    print(f"   MEDIUM risk: {(df['risk_level']=='MEDIUM').sum():,} sites")
    print(f"   LOW risk:    {(df['risk_level']=='LOW').sum():,} sites")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CRISPR Off-Target Cleavage Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single pair:
    python predict_offtarget.py \\
        --target GGGTGGGGGGAGTTTGCTCCNGG \\
        --offtarget GGATGGGGGGAGTTTGCTCCNGG

  With chromatin accessibility:
    python predict_offtarget.py \\
        --target GGGTGGGGGGAGTTTGCTCCNGG \\
        --offtarget GGATGGGGGGAGTTTGCTCCNGG \\
        --atac 1

  Batch mode:
    python predict_offtarget.py \\
        --input my_sequences.csv \\
        --output predictions.csv
        """)

    parser.add_argument('--target',    type=str, help='Guide RNA target sequence (20-23nt)')
    parser.add_argument('--offtarget', type=str, help='Off-target sequence (20-23nt)')
    parser.add_argument('--atac',      type=int, default=None,
                        help='Chromatin accessibility: 1=open, 0=closed (optional)')
    parser.add_argument('--input',     type=str, help='Input CSV for batch mode')
    parser.add_argument('--output',    type=str, default='predictions.csv',
                        help='Output CSV for batch mode (default: predictions.csv)')
    parser.add_argument('--model',     type=str,
                        default=os.path.join(os.path.dirname(__file__), 'hybrid_model.pt'),
                        help='Path to model weights (default: hybrid_model.pt)')

    args = parser.parse_args()

    model = load_model(args.model)

    if args.input:
        predict_batch(model, args.input, args.output)
    elif args.target and args.offtarget:
        predict_single(model, args.target, args.offtarget, args.atac)
    else:
        parser.print_help()
        print("\nERROR: Provide either --target + --offtarget or --input CSV file")
        sys.exit(1)

if __name__ == "__main__":
    main()
