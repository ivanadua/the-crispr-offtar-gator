"""
Feature Engineering for CRISPR Off-Target Prediction
------------------------------------------------------
Adds biological feature columns to master_dataset.csv.

Features added:
    gc_content_target       — GC% of target sequence
    gc_content_offtarget    — GC% of off-target sequence
    shannon_entropy_target  — Shannon entropy of target (sequence complexity)
    shannon_entropy_offt    — Shannon entropy of off-target
    mismatch_positions      — list of positions where mismatches occur
    seed_mismatches         — number of mismatches in seed region (pos 17-20)
    nonseed_mismatches      — number of mismatches outside seed region
    pam_proximity_score     — weighted mismatch score (higher = closer to PAM)
    mismatch_types          — types of substitutions e.g. A>T, G>C
    has_bulge               — 1 if sequences have different lengths (indel)
    guide_length            — length of target sequence
    mfe                     — minimum free energy of target RNA (RNAfold)
                              Falls back to None if ViennaRNA not installed

Usage:
    pip install ViennaRNA   # optional but recommended
    python feature_engineering.py

Output:
    master_dataset_features.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy

# ── FILE PATHS ────────────────────────────────────────────────────────────────
INPUT_PATH  = '/Users/moana/Downloads/final dataset masterio.csv'
OUTPUT_PATH = '/Users/moana/Downloads/final dataset masterio_features.csv'
# ─────────────────────────────────────────────────────────────────────────────

try:
    import RNA
    VIENNA_AVAILABLE = True
    print("ViennaRNA found — MFE will be computed.")
except ImportError:
    VIENNA_AVAILABLE = False
    print("ViennaRNA not installed — MFE will be skipped.")
    print("To install: pip install ViennaRNA")

# Positional weight vector — higher = closer to PAM = more important
POSITION_WEIGHTS = np.array([
    0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.10,
    0.12, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00
])

SEED_POSITIONS = set(range(16, 20))

def gc_content(seq):
    seq = seq[:20].upper()
    if len(seq) == 0:
        return None
    return round((seq.count('G') + seq.count('C')) / len(seq), 4)

def shannon_entropy(seq):
    seq = seq[:20].upper()
    if len(seq) == 0:
        return None
    counts = [seq.count(b) / len(seq) for b in 'ACGT']
    counts = [c for c in counts if c > 0]
    return round(float(scipy_entropy(counts, base=2)), 4)

def get_mismatch_info(target, offtarget):
    t = str(target)[:20].upper()
    o = str(offtarget)[:20].upper()
    if len(t) < 20 or len(o) < 20:
        return [], 0, 0, 0.0, []

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
    if not VIENNA_AVAILABLE:
        return None
    rna = seq[:20].upper().replace('T', 'U').replace('N', 'A')
    try:
        _, mfe = RNA.fold(rna)
        return round(mfe, 3)
    except Exception:
        return None

def has_bulge(target, offtarget):
    # Compare only the 20nt protospacer, not the PAM
    # A bulge = insertion or deletion in the protospacer only
    t = str(target)[:20].upper().replace('-', '')
    o = str(offtarget)[:20].upper().replace('-', '')
    return int(len(t) != len(o))

def process_batch(df_batch):
    results = []
    for _, row in df_batch.iterrows():
        target    = str(row['target_seq'])
        offtarget = str(row['offtarget_seq'])

        gc_t  = gc_content(target)
        gc_o  = gc_content(offtarget)
        ent_t = shannon_entropy(target)
        ent_o = shannon_entropy(offtarget)

        positions, seed_mm, nonseed_mm, pam_score, types = get_mismatch_info(target, offtarget)

        mfe       = get_mfe(target)
        # has_bulge already populated from master builder (CIRCLE-seq raw data)
        guide_len = len(target.replace('N', ''))

        results.append({
            'gc_content_target':      gc_t,
            'gc_content_offtarget':   gc_o,
            'shannon_entropy_target': ent_t,
            'shannon_entropy_offt':   ent_o,
            'mismatch_positions':     str(positions),
            'seed_mismatches':        seed_mm,
            'nonseed_mismatches':     nonseed_mm,
            'pam_proximity_score':    pam_score,
            'mismatch_types':         str(types),
            'guide_length':           guide_len,
            'mfe':                    mfe,
        })
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("   Feature Engineering")
    print("=" * 60)

    print(f"\nLoading {INPUT_PATH}...")
    master = pd.read_csv(INPUT_PATH)
    print(f"  {len(master):,} rows loaded")

    BATCH_SIZE = 10000
    n_batches  = (len(master) // BATCH_SIZE) + 1
    feature_dfs = []

    print(f"\nComputing features in {n_batches} batches of {BATCH_SIZE:,}...")
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(master))
        if start >= len(master):
            break
        feature_dfs.append(process_batch(master.iloc[start:end]))
        print(f"  Batch {i+1}/{n_batches} done ({end:,}/{len(master):,})")

    features = pd.concat(feature_dfs, ignore_index=True)
    result   = pd.concat([master.reset_index(drop=True), features], axis=1)

    print(f"\n{'=' * 60}")
    print("FEATURE SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nGC content (target)   — mean: {result['gc_content_target'].mean():.3f}")
    print(f"Shannon entropy       — mean: {result['shannon_entropy_target'].mean():.3f}")
    print(f"\nPAM proximity score:")
    print(f"  Positives mean: {result[result['is_cut']==1]['pam_proximity_score'].mean():.3f}")
    print(f"  Negatives mean: {result[result['is_cut']==0]['pam_proximity_score'].mean():.3f}")
    print(f"\nSeed mismatches (is_cut=1 vs 0):")
    print(f"  Positives mean: {result[result['is_cut']==1]['seed_mismatches'].mean():.3f}")
    print(f"  Negatives mean: {result[result['is_cut']==0]['seed_mismatches'].mean():.3f}")
    print(f"\nRows with bulges (from CIRCLE-seq): {result['has_bulge'].sum():,}")
    print(f"Bulge types: {result['bulge_type'].value_counts().to_dict()}")
    if VIENNA_AVAILABLE:
        print(f"\nMFE — mean: {result['mfe'].mean():.3f} kcal/mol")

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved to: {OUTPUT_PATH}")
    print(f"   Total columns: {len(result.columns)}")
    print(f"   Columns: {result.columns.tolist()}")
    print(f"\nNext step: add ATAC-seq chromatin accessibility from ENCODE")

if __name__ == "__main__":
    main()
