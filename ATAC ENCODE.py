"""
ATAC-seq Chromatin Accessibility Intersection
-----------------------------------------------
Intersects CRISPR off-target genomic coordinates with ENCODE
chromatin accessibility peaks using bedtools.

Steps:
1. Extract genomic coordinates from your dataset
2. Convert to BED format
3. Intersect with ATAC/DNase-seq peaks per cell line
4. Add atac_score column to your feature dataset

Requirements:
    bedtools installed (bedtools --version to check)

Usage:
    python add_atac_features.py

Output:
    final dataset masterio_features_atac.csv
"""

import pandas as pd
import numpy as np
import subprocess
import os
import tempfile

# ── FILE PATHS ────────────────────────────────────────────────────────────────
INPUT_PATH    = '/Users/moana/Downloads/final dataset masterio_features.csv'
OUTPUT_PATH   = '/Users/moana/Downloads/final dataset masterio_features_atac.csv'
ENCODE_DIR    = '/Users/moana/Downloads/ENCODE_ATAC/'

# Cell line → ATAC/DNase BED file mapping
ATAC_FILES = {
    'HEK293T':   os.path.join(ENCODE_DIR, 'ENCFF285OXK.bed.gz'),
    'HEK293':    os.path.join(ENCODE_DIR, 'ENCFF285OXK.bed.gz'),  # proxy
    'K562':      os.path.join(ENCODE_DIR, 'ENCFF333TAT.bed.gz'),
}
# ─────────────────────────────────────────────────────────────────────────────

def parse_genomic_location(loc):
    """
    Parse genomic location string into (chrom, start, end).
    Handles formats like:
        chr19:55115745-55115767:+
        chr4:121343303-121343325:+
        12:5555189-5555211
    """
    if pd.isna(loc) or str(loc).strip() == '':
        return None, None, None
    loc = str(loc).strip()
    try:
        # Remove strand info if present
        loc = loc.split(':')[0] + ':' + loc.split(':')[1] if ':' in loc else loc
        # Handle chr prefix
        parts = loc.replace(':', '-').split('-')
        chrom = parts[0]
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        start = int(parts[1])
        end   = int(parts[2]) if len(parts) > 2 else start + 23
        return chrom, start, end
    except Exception:
        return None, None, None

def create_bed_file(df, cell_line, tmp_dir):
    """Create a BED file for rows matching a specific cell line."""
    subset = df[df['cell_line'] == cell_line].copy()
    subset = subset[subset['genomic_location'].notna()]

    rows = []
    indices = []
    for idx, row in subset.iterrows():
        chrom, start, end = parse_genomic_location(row['genomic_location'])
        if chrom and start and end:
            rows.append(f"{chrom}\t{start}\t{end}\t{idx}")
            indices.append(idx)

    if not rows:
        print(f"  No valid coordinates for {cell_line}")
        return None, []

    bed_path = os.path.join(tmp_dir, f"{cell_line}.bed")
    with open(bed_path, 'w') as f:
        f.write('\n'.join(rows) + '\n')

    print(f"  {cell_line}: {len(rows):,} coordinates written to BED")
    return bed_path, indices

def run_bedtools_intersect(query_bed, atac_bed, tmp_dir, cell_line):
    """
    Run bedtools intersect and return set of indices that overlap peaks.
    -wa: write original query entry
    -u:  report each query only once
    """
    out_path = os.path.join(tmp_dir, f"{cell_line}_intersect.bed")
    cmd = f"/opt/homebrew/bin/bedtools intersect -a {query_bed} -b {atac_bed} -wa -u > {out_path}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  bedtools error: {result.stderr}")
        return set()

    # Read intersected rows and extract original indices
    accessible_indices = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    try:
                        accessible_indices.add(int(parts[3]))
                    except ValueError:
                        pass

    print(f"  {cell_line}: {len(accessible_indices):,} sites in accessible chromatin")
    return accessible_indices

def main():
    print("=" * 60)
    print("   ATAC-seq Chromatin Accessibility Intersection")
    print("=" * 60)

    print(f"\nLoading dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df):,} rows loaded")

    # Initialize atac columns
    df['atac_accessible'] = np.nan   # 1 = accessible, 0 = not, NaN = no data
    df['atac_source']     = None     # which file was used

    tmp_dir = tempfile.mkdtemp()
    print(f"\nTemp directory: {tmp_dir}")

    total_annotated = 0

    for cell_line, atac_file in ATAC_FILES.items():
        print(f"\nProcessing {cell_line}...")

        if not os.path.exists(atac_file):
            print(f"  ⚠️  ATAC file not found: {atac_file}")
            continue

        # Get rows for this cell line that have coordinates
        cell_mask = (df['cell_line'] == cell_line) & (df['genomic_location'].notna())
        n_rows = cell_mask.sum()
        if n_rows == 0:
            print(f"  No rows with coordinates for {cell_line}")
            continue

        print(f"  {n_rows:,} rows with coordinates")

        # Create BED file
        query_bed, valid_indices = create_bed_file(df, cell_line, tmp_dir)
        if query_bed is None:
            continue

        # Run bedtools intersect
        accessible_indices = run_bedtools_intersect(query_bed, atac_file, tmp_dir, cell_line)

        # Annotate rows
        for idx in valid_indices:
            df.at[idx, 'atac_accessible'] = 1 if idx in accessible_indices else 0
            df.at[idx, 'atac_source'] = os.path.basename(atac_file)

        n_annotated = len(valid_indices)
        n_accessible = len(accessible_indices)
        total_annotated += n_annotated
        print(f"  ✅ {n_accessible:,}/{n_annotated:,} ({n_accessible/max(n_annotated,1)*100:.1f}%) sites accessible")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total rows annotated: {total_annotated:,}")
    print(f"Rows with no ATAC data: {df['atac_accessible'].isna().sum():,}")
    print(f"\nAccessibility by cell line (annotated rows only):")
    annotated = df[df['atac_accessible'].notna()]
    if len(annotated) > 0:
        print(annotated.groupby('cell_line')['atac_accessible'].agg(['sum','count','mean']).round(3).to_string())

    print(f"\nAccessibility vs cleavage (is_cut):")
    print(annotated.groupby(['is_cut','atac_accessible']).size().unstack(fill_value=0).to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved to: {OUTPUT_PATH}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"\nNext step: train the model!")

if __name__ == "__main__":
    main()
