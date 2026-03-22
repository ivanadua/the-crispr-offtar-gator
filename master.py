"""
Master Dataset Builder
-----------------------
Merges all positive and negative datasets into one clean CSV
ready for feature engineering and model training.

New in this version:
    - Bulge detection preserved from raw CIRCLE-seq sequences
    - Output named 'final dataset masterio.csv'

Output columns:
    target_seq        — 20-23nt guide RNA sequence
    offtarget_seq     — 20-23nt off-target sequence (dashes removed)
    is_cut            — 1 = confirmed cut, 0 = negative
    mismatch_count    — number of mismatches between target and offtarget
    genomic_location  — chromosome:start-end (if available)
    reads             — experimental read count (if available)
    cell_line         — cell line (if available)
    distance          — edit distance (if available)
    source            — which dataset this row came from
    negative_tier     — for synthetic negatives only (hard/medium/easy)
    has_bulge         — 1 if off-target has an insertion or deletion
    bulge_type        — RNA_bulge / DNA_bulge / None
    bulge_position    — position of bulge in sequence (1-indexed)

Usage:
    python build_master_dataset.py

Output:
    final dataset masterio.csv
"""

import pandas as pd
import numpy as np

# ── FILE PATHS — update if needed ────────────────────────────────────────────
GUIDE_SEQ_PATH      = '/Users/moana/Downloads/guide_seq_positives.csv'
CHANGE_SEQ_PATH     = '/Users/moana/Downloads/change_seq_negatives.csv'
CIRCLE_SEQ_PATH     = '/Users/moana/Downloads/CIRCLE_SEQ. off targets.csv'
SITESEQ_PATH        = '/Users/moana/Downloads/siteseq_master.csv'
SYNTHETIC_NEG_PATH  = '/Users/moana/Downloads/synthetic_negatives.csv'
OUTPUT_PATH         = '/Users/moana/Downloads/final dataset masterio.csv'
# ─────────────────────────────────────────────────────────────────────────────

def clean_seq(seq):
    """Uppercase, strip whitespace, remove dashes (after bulge info extracted)."""
    return str(seq).upper().strip().replace('-', '')

def detect_bulge(offtarget_raw):
    """
    Detect bulge from raw off-target sequence BEFORE dashes are removed.

    Dash in sequence = RNA bulge (deletion in the genomic DNA strand)
    Example: GGGAAAGACC-AGCATCCATAGG → dash at position 11

    Returns: (has_bulge, bulge_type, bulge_position)
    """
    raw = str(offtarget_raw).upper().strip()
    if '-' in raw:
        pos = raw.index('-') + 1   # 1-indexed
        return 1, 'RNA_bulge', pos
    # DNA bulge (extra base) would show as length > 23 — handled separately
    cleaned_len = len(raw.replace('-', ''))
    if cleaned_len > 23:
        return 1, 'DNA_bulge', None
    return 0, None, None

def count_mismatches(target, offtarget):
    t = str(target)[:20].upper()
    o = str(offtarget)[:20].upper()
    if len(t) < 20 or len(o) < 20:
        return None
    return sum(1 for a, b in zip(t, o) if a != b and a != 'N')

def load_guide_seq():
    print("Loading GUIDE-seq...")
    df = pd.read_csv(GUIDE_SEQ_PATH)
    out = pd.DataFrame({
        'target_seq':       df['target'].apply(clean_seq),
        'offtarget_seq':    df['offtarget_sequence'].apply(clean_seq),
        'is_cut':           1,
        'reads':            df['GUIDEseq_reads'],
        'genomic_location': df['genomic_coordinate'],
        'distance':         df['distance'],
        'cell_line':        'U2OS',
        'source':           'GUIDE-seq',
        'negative_tier':    None,
        'has_bulge':        0,
        'bulge_type':       None,
        'bulge_position':   None,
    })
    out = out.dropna(subset=['target_seq', 'offtarget_seq'])
    print(f"  {len(out):,} rows")
    return out

def load_change_seq():
    print("Loading CHANGE-seq...")
    df = pd.read_csv(CHANGE_SEQ_PATH)
    out = pd.DataFrame({
        'target_seq':       df['target'].apply(clean_seq),
        'offtarget_seq':    df['offtarget_sequence'].apply(clean_seq),
        'is_cut':           1,
        'reads':            df['CHANGEseq_reads'],
        'genomic_location': df['chromStart:chromEnd'],
        'distance':         df['distance'],
        'cell_line':        'HEK293T',
        'source':           'CHANGE-seq',
        'negative_tier':    None,
        'has_bulge':        0,
        'bulge_type':       None,
        'bulge_position':   None,
    })
    out = out.dropna(subset=['target_seq', 'offtarget_seq'])
    print(f"  {len(out):,} rows")
    return out

def load_circle_seq():
    print("Loading CIRCLE-seq...")
    df = pd.read_csv(CIRCLE_SEQ_PATH, encoding='latin1')

    # Detect bulges BEFORE cleaning sequences
    bulge_info = df['Off-target Sequence'].apply(
        lambda s: pd.Series(detect_bulge(s), index=['has_bulge', 'bulge_type', 'bulge_position'])
    )

    out = pd.DataFrame({
        'target_seq':       df['TargetSequence'].apply(clean_seq),
        'offtarget_seq':    df['Off-target Sequence'].apply(clean_seq),
        'is_cut':           1,
        'reads':            df['Read'],
        'genomic_location': df['Summary'],
        'distance':         df['Distance'],
        'cell_line':        df['Cell'],
        'source':           'CIRCLE-seq',
        'negative_tier':    None,
        'has_bulge':        bulge_info['has_bulge'].values,
        'bulge_type':       bulge_info['bulge_type'].values,
        'bulge_position':   bulge_info['bulge_position'].values,
    })
    out = out.dropna(subset=['target_seq', 'offtarget_seq'])
    bulge_count = out['has_bulge'].sum()
    print(f"  {len(out):,} rows | {bulge_count} bulge-containing off-targets")
    return out

def load_site_seq():
    print("Loading SITE-seq...")
    df = pd.read_csv(SITESEQ_PATH)
    df = df[df['is_cut'] == 1].copy()
    out = pd.DataFrame({
        'target_seq':       df['target_seq'].apply(clean_seq),
        'offtarget_seq':    df['offtarget_seq'].apply(clean_seq),
        'is_cut':           1,
        'reads':            None,
        'genomic_location': df['genomic_location'],
        'distance':         None,
        'cell_line':        None,
        'source':           'SITE-seq',
        'negative_tier':    None,
        'has_bulge':        0,
        'bulge_type':       None,
        'bulge_position':   None,
    })
    out = out.dropna(subset=['target_seq', 'offtarget_seq'])
    print(f"  {len(out):,} rows")
    return out

def load_synthetic_negatives():
    print("Loading synthetic negatives...")
    df = pd.read_csv(SYNTHETIC_NEG_PATH)
    out = pd.DataFrame({
        'target_seq':       df['target_seq'].apply(clean_seq),
        'offtarget_seq':    df['offtarget_seq'].apply(clean_seq),
        'is_cut':           0,
        'reads':            0,
        'genomic_location': None,
        'distance':         df['mismatch_count'],
        'cell_line':        None,
        'source':           'synthetic',
        'negative_tier':    df['negative_tier'],
        'has_bulge':        0,
        'bulge_type':       None,
        'bulge_position':   None,
    })
    out = out.dropna(subset=['target_seq', 'offtarget_seq'])
    print(f"  {len(out):,} rows")
    return out

def main():
    print("=" * 60)
    print("   Master Dataset Builder")
    print("=" * 60)

    dfs = []
    for loader in [load_guide_seq, load_change_seq, load_circle_seq,
                   load_site_seq, load_synthetic_negatives]:
        try:
            dfs.append(loader())
        except Exception as e:
            print(f"  WARNING: {loader.__name__} failed — {e}")

    master = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal before cleaning: {len(master):,}")

    # Compute mismatch counts
    print("Computing mismatch counts...")
    master['mismatch_count'] = master.apply(
        lambda r: count_mismatches(r['target_seq'], r['offtarget_seq']), axis=1
    )

    # Remove invalid sequences
    valid_bases = set('ACGTN')
    master = master[master['target_seq'].apply(
        lambda s: len(s) >= 20 and all(b in valid_bases for b in s[:20])
    )]
    master = master[master['offtarget_seq'].apply(
        lambda s: len(s) >= 20 and all(b in valid_bases for b in s[:20])
    )]

    # Deduplicate — keep experimental over synthetic
    master = master.sort_values('source')
    master = master.drop_duplicates(subset=['target_seq', 'offtarget_seq'], keep='first')

    # Shuffle
    master = master.sample(frac=1, random_state=42).reset_index(drop=True)

    # Final column order
    master = master[[
        'target_seq', 'offtarget_seq', 'is_cut', 'mismatch_count',
        'reads', 'genomic_location', 'distance', 'cell_line',
        'source', 'negative_tier',
        'has_bulge', 'bulge_type', 'bulge_position'
    ]]

    # Summary
    print(f"\nTotal after cleaning: {len(master):,}")
    print(f"\nis_cut distribution:")
    print(master['is_cut'].value_counts().to_string())
    print(f"\nSource breakdown:")
    print(master['source'].value_counts().to_string())
    print(f"\nBulge summary:")
    print(f"  Total bulge rows: {master['has_bulge'].sum():,}")
    print(f"  Bulge types: {master['bulge_type'].value_counts().to_dict()}")
    print(f"\nCell line distribution (positives only):")
    print(master[master['is_cut']==1]['cell_line'].value_counts().to_string())

    master.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved to: {OUTPUT_PATH}")
    print(f"\nNext step: run feature_engineering.py")

if __name__ == "__main__":
    main()
