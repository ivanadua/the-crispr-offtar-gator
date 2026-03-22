"""
Synthetic Negative Generator — 7-Tier Method
----------------------------------------------
Generates synthetic negatives across 7 biologically justified tiers.

Tier 1: Hard          — 1-2 mismatches in seed region (pos 17-20)
Tier 2: Medium        — 2-3 mismatches mixed seed+non-seed
Tier 3: Easy          — 4-5 mismatches in non-seed region
Tier 4: Gap-fill      — exactly 4, 5, 6 mismatches (fixes imbalance)
Tier 5: Multi-seed    — 2-3 mismatches ALL in seed (pos 17-20)
                        Fixes: model scored 3 seed mismatches as 0.95
Tier 6: Mid-seed      — 2-3 mismatches ALL in mid-seed (pos 13-16)
                        Teaches: positions 13-16 also matter
Tier 7: Clustered     — 6-7 mismatches clustered in positions 1-7 only
                        Teaches: extreme distal clustering eventually prevents cutting

NOT included (and why):
  - GC-based negatives: GC thresholds apply to on-target efficiency (Doench 2016),
    not off-target cleavage. No published experimental data supports GC-based
    off-target non-cut thresholds. Including them would introduce unverified biology.
  - Bulge non-cuts: No published data on which bulge configurations don't cut.

Usage:
    python generate_synthetic_negatives.py

Output:
    synthetic_negatives.csv
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
GUIDE_SEQ_PATH  = '/Users/moana/Downloads/guide_seq_positives.csv'
CHANGE_SEQ_PATH = '/Users/moana/Downloads/change_seq_negatives.csv'
CIRCLE_SEQ_PATH = '/Users/moana/Downloads/CIRCLE_SEQ. off targets.csv'
OUTPUT_PATH     = '/Users/moana/Downloads/synthetic_negatives.csv'

TIER1_COUNT = 4000    # Hard — 1-2 seed mismatches
TIER2_COUNT = 30000   # Medium — 2-3 mixed
TIER3_COUNT = 15000   # Easy — 4-5 non-seed
TIER4_COUNT = 15000   # Gap-fill — 5k each at mismatch 4, 5, 6
TIER5_COUNT = 6000    # Multi-seed — 2-3 mismatches ALL in seed (pos 17-20)
TIER6_COUNT = 6000    # Mid-seed — 2-3 mismatches ALL in mid-seed (pos 13-16)
TIER7_COUNT = 5000    # Clustered distal — 6-7 mismatches in positions 1-7 only
# ─────────────────────────────────────────────────────────────────────────────

BASES             = ['A', 'C', 'G', 'T']
SEED_POSITIONS    = list(range(16, 20))  # pos 17-20 (0-indexed 16-19)
MIDSEED_POSITIONS = list(range(12, 16))  # pos 13-16
NONSEED_POSITIONS = list(range(0, 12))   # pos 1-12
DISTAL_POSITIONS  = list(range(0, 7))    # pos 1-7 only (very distal)

DISRUPTIVE_SUBS = {
    'A': ['C', 'G'],
    'T': ['G', 'A'],
    'G': ['T', 'C'],
    'C': ['A', 'G'],
    'N': ['A', 'C', 'G', 'T'],
}

def clean_seq(seq):
    return str(seq).upper().strip().replace('-', '')

def mutate_positions(seq, positions, n_mutations, disruptive=True):
    seq = list(seq)
    chosen = random.sample(positions, min(n_mutations, len(positions)))
    for pos in chosen:
        if pos >= len(seq):
            continue
        orig = seq[pos]
        if disruptive:
            options = DISRUPTIVE_SUBS.get(orig, [b for b in BASES if b != orig])
        else:
            options = [b for b in BASES if b != orig]
        if options:
            seq[pos] = random.choice(options)
    return ''.join(seq)

def count_mismatches(t, o):
    return sum(1 for a, b in zip(t[:20], o[:20]) if a != b and a != 'N')

def count_region_mismatches(t, o, positions):
    return sum(1 for i, (a, b) in enumerate(zip(t[:20], o[:20]))
               if a != b and a != 'N' and i in set(positions))

def load_guide_sequences():
    pairs = []
    print("\nLoading sequences from positive datasets...")

    try:
        df = pd.read_csv(GUIDE_SEQ_PATH)
        for _, row in df[['target', 'offtarget_sequence']].dropna().iterrows():
            t = clean_seq(row['target'])
            o = clean_seq(row['offtarget_sequence'])
            if 20 <= len(t) <= 23 and 20 <= len(o) <= 23:
                pairs.append((t, o))
        print(f"  GUIDE-seq:  {len(df):,} rows, {df['target'].nunique()} unique targets")
    except Exception as e:
        print(f"  GUIDE-seq failed: {e}")

    try:
        df = pd.read_csv(CHANGE_SEQ_PATH)
        sample = df[['target', 'offtarget_sequence']].dropna().sample(
            n=min(50000, len(df)), random_state=42)
        for _, row in sample.iterrows():
            t = clean_seq(row['target'])
            o = clean_seq(row['offtarget_sequence'])
            if 20 <= len(t) <= 23 and 20 <= len(o) <= 23:
                pairs.append((t, o))
        print(f"  CHANGE-seq: {len(df):,} rows, sampled 50k for diversity")
    except Exception as e:
        print(f"  CHANGE-seq failed: {e}")

    try:
        df = pd.read_csv(CIRCLE_SEQ_PATH, encoding='latin1')
        for _, row in df[['TargetSequence', 'Off-target Sequence']].dropna().iterrows():
            t = clean_seq(row['TargetSequence'])
            o = clean_seq(row['Off-target Sequence'])
            if 20 <= len(t) <= 23 and 20 <= len(o) <= 23:
                pairs.append((t, o))
        print(f"  CIRCLE-seq: {len(df):,} rows loaded")
    except Exception as e:
        print(f"  CIRCLE-seq failed: {e}")

    print(f"  Total (target, template) pairs available: {len(pairs):,}")
    return pairs

def generate_tier(pairs, tier, count):
    tier_labels = {
        1: 'hard_synthetic',
        2: 'medium_synthetic',
        3: 'easy_synthetic',
        4: 'gapfill_synthetic',
        5: 'multiseed_synthetic',
        6: 'midseed_synthetic',
        7: 'clustered_distal_synthetic',
    }
    tier_descriptions = {
        1: '1-2 seed mismatches',
        2: '2-3 mixed seed+non-seed',
        3: '4-5 non-seed mismatches',
        4: 'exactly 4/5/6 mismatches (gap-fill)',
        5: '2-3 mismatches ALL in seed (pos 17-20)',
        6: '2-3 mismatches ALL in mid-seed (pos 13-16)',
        7: '6-7 mismatches clustered in pos 1-7 only',
    }
    label = tier_labels[tier]
    desc  = tier_descriptions[tier]
    print(f"\nGenerating Tier {tier} — {label} ({count:,} negatives)")
    print(f"  Strategy: {desc}")

    rows = []
    seen = set()
    attempts = 0
    max_attempts = count * 20

    # Tier 4 specific tracking
    target_mm_cycle = [4, 5, 6]
    mm_counts       = {4: 0, 5: 0, 6: 0}
    per_mm_target   = count // 3

    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        target, template = random.choice(pairs)
        target_20   = target[:20]
        template_20 = template[:20]
        pam = target[20:] if len(target) > 20 else 'NGG'

        # ── TIER 1: 1-2 disruptive seed mismatches ───────────────────────────
        if tier == 1:
            n = random.choice([1, 2])
            offtarget_20 = mutate_positions(target_20, SEED_POSITIONS,
                                            n, disruptive=True)

        # ── TIER 2: 1 seed + 1-2 non-seed mismatches ─────────────────────────
        elif tier == 2:
            offtarget_20 = mutate_positions(target_20, SEED_POSITIONS,
                                            1, disruptive=True)
            offtarget_20 = mutate_positions(offtarget_20, NONSEED_POSITIONS,
                                            random.choice([1, 2]), disruptive=False)

        # ── TIER 3: 4-5 non-seed+midseed mismatches ──────────────────────────
        elif tier == 3:
            n = random.choice([4, 5])
            offtarget_20 = mutate_positions(template_20,
                                            NONSEED_POSITIONS + MIDSEED_POSITIONS,
                                            n, disruptive=False)

        # ── TIER 4: exactly 4, 5, or 6 total mismatches (gap-fill) ───────────
        elif tier == 4:
            needed     = {mm: per_mm_target - mm_counts[mm] for mm in target_mm_cycle}
            target_mm  = max(needed, key=needed.get)
            if needed[target_mm] <= 0:
                break
            all_non_seed = NONSEED_POSITIONS + MIDSEED_POSITIONS
            offtarget_20 = mutate_positions(target_20, all_non_seed,
                                            target_mm, disruptive=False)

        # ── TIER 5: 2-3 mismatches ALL in seed region (pos 17-20) ────────────
        elif tier == 5:
            n = random.choice([2, 3])
            offtarget_20 = mutate_positions(target_20, SEED_POSITIONS,
                                            n, disruptive=True)
            # Verify at least 2 mismatches landed in seed
            if count_region_mismatches(target_20, offtarget_20, SEED_POSITIONS) < 2:
                continue

        # ── TIER 6: 2-3 mismatches ALL in mid-seed (pos 13-16) ───────────────
        elif tier == 6:
            n = random.choice([2, 3])
            offtarget_20 = mutate_positions(target_20, MIDSEED_POSITIONS,
                                            n, disruptive=True)
            # Verify at least 2 mismatches landed in mid-seed
            if count_region_mismatches(target_20, offtarget_20, MIDSEED_POSITIONS) < 2:
                continue
            # Ensure no seed region mismatches (keep it pure mid-seed)
            if count_region_mismatches(target_20, offtarget_20, SEED_POSITIONS) > 0:
                continue

        # ── TIER 7: 6-7 mismatches clustered in positions 1-7 only ──────────
        elif tier == 7:
            n = random.choice([6, 7])
            # Can only mutate 7 positions max — use all 7 distal positions
            offtarget_20 = mutate_positions(target_20, DISTAL_POSITIONS,
                                            n, disruptive=False)
            # Verify all mismatches are distal (pos 1-7 only)
            non_distal_mm = count_region_mismatches(
                target_20, offtarget_20,
                MIDSEED_POSITIONS + SEED_POSITIONS)
            if non_distal_mm > 0:
                continue
            # Must have at least 5 distal mismatches to be meaningful
            if count_mismatches(target_20, offtarget_20) < 5:
                continue

        # ── SHARED VALIDATION ─────────────────────────────────────────────────
        mm = count_mismatches(target_20, offtarget_20)
        if mm == 0:
            continue

        if tier == 4:
            if mm != target_mm:
                continue
            mm_counts[mm] = mm_counts.get(mm, 0) + 1

        key = (target_20, offtarget_20)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            'target_seq':     target,
            'offtarget_seq':  offtarget_20 + pam,
            'mismatch_count': mm,
            'is_cut':         0,
            'negative_tier':  label,
            'source':         'synthetic',
        })

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df):,} | Mismatch dist: "
          f"{df['mismatch_count'].value_counts().sort_index().to_dict()}")
    return df

def main():
    print("=" * 60)
    print("   CRISPR Synthetic Negative Generator — 7-Tier Method")
    print("=" * 60)
    print("""
Tiers:
  1. Hard          — 1-2 seed mismatches (pos 17-20)
  2. Medium        — 2-3 mixed seed+non-seed
  3. Easy          — 4-5 non-seed mismatches
  4. Gap-fill      — exactly 4/5/6 total mismatches
  5. Multi-seed    — 2-3 mismatches ALL in seed
  6. Mid-seed      — 2-3 mismatches ALL in mid-seed (pos 13-16)
  7. Clustered     — 6-7 mismatches clustered in pos 1-7 only
""")

    pairs = load_guide_sequences()
    if not pairs:
        print("\nERROR: No pairs loaded. Check your file paths above.")
        return

    tier1 = generate_tier(pairs, tier=1, count=TIER1_COUNT)
    tier2 = generate_tier(pairs, tier=2, count=TIER2_COUNT)
    tier3 = generate_tier(pairs, tier=3, count=TIER3_COUNT)
    tier4 = generate_tier(pairs, tier=4, count=TIER4_COUNT)
    tier5 = generate_tier(pairs, tier=5, count=TIER5_COUNT)
    tier6 = generate_tier(pairs, tier=6, count=TIER6_COUNT)
    tier7 = generate_tier(pairs, tier=7, count=TIER7_COUNT)

    all_neg = pd.concat([tier1, tier2, tier3, tier4,
                         tier5, tier6, tier7]).reset_index(drop=True)
    all_neg = all_neg.drop_duplicates(subset=['target_seq', 'offtarget_seq'])

    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY")
    print(f"Total synthetic negatives: {len(all_neg):,}")
    print(f"\nBy tier:")
    print(all_neg['negative_tier'].value_counts().to_string())
    print(f"\nOverall mismatch distribution:")
    print(all_neg['mismatch_count'].value_counts().sort_index().to_string())

    all_neg.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved to: {OUTPUT_PATH}")
    print(f"\nNext: build_master_dataset.py → feature_engineering.py → "
          f"add_atac_features.py → train_hybrid.py → biological_sanity_check.py")

if __name__ == "__main__":
    main()
