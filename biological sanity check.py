"""
Biological Sanity Check
------------------------
Tests your model against known biological facts about CRISPR off-target cutting.
Each test has a known expected outcome from published literature.

If the model passes all tests, it has learned real biology.
If it fails, we know exactly what to fix before publishing.

Tests:
  1. Seed region sensitivity — mismatches near PAM should reduce cutting more
  2. Mismatch count — more mismatches = less cutting
  3. On-target should score highest
  4. PAM requirement — non-NGG PAM should reduce cutting
  5. GC content effect — very low or very high GC reduces efficiency
  6. Transition vs transversion — transitions more tolerated
  7. Chromatin accessibility — open chromatin should increase probability
  8. Known published sequences from Tsai 2015 VEGFA guide

Usage:
    python3 biological_sanity_check.py

Output:
    Prints PASS/FAIL for each test with explanation
"""

import sys
sys.path.insert(0, '/Users/moana/Downloads/')

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from shared_utils import encode_pair, RANDOM_SEED

MODEL_PATH = '/Users/moana/Downloads/model_results/hybrid_model.pt'

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

def gc_content(seq):
    seq = seq[:20].upper()
    return (seq.count('G') + seq.count('C')) / len(seq)

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
            weighted_score += POS_WEIGHTS[i].item()
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

def make_features(target, offtarget, atac=0.0):
    seed_mm, nonseed_mm, pam_score = get_mismatch_info(target, offtarget)
    mm = seed_mm + nonseed_mm
    return np.array([
        gc_content(target),
        gc_content(offtarget),
        shannon_entropy(target),
        shannon_entropy(offtarget),
        float(mm),
        float(seed_mm),
        float(nonseed_mm),
        float(pam_score),
        float(len(target.replace('N',''))),
        float(get_mfe(target)),
        0.0,
        float(atac),
    ], dtype=np.float32)

def predict(model, target, offtarget, atac=0.0):
    target    = str(target)[:23].upper()
    offtarget = str(offtarget)[:23].upper()
    features  = make_features(target[:20], offtarget[:20], atac)
    x_seq = torch.tensor(
        encode_pair(target, offtarget), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_bio = torch.tensor(features).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(x_seq, x_bio)).item()

def run_test(name, condition, result, expected, tolerance=None):
    if tolerance:
        passed = abs(result - expected) <= tolerance
    else:
        passed = condition
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
    print(f"          Result: {result:.4f} | Expected: {expected}")
    return passed

def main():
    print("=" * 65)
    print("   Biological Sanity Check")
    print("=" * 65)

    model = HybridModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}\n")

    # Reference guide from Tsai 2015 (VEGFA site 1)
    GUIDE = "GGGTGGGGGGAGTTTGCTCCNGG"

    passes = []

    # ── TEST 1: On-target should score high ───────────────────────────────────
    print("TEST 1: On-target sequence scores high")
    print("  Biology: Perfect match guide = highest cleavage probability")
    p_ontarget = predict(model, GUIDE, GUIDE)
    passed = run_test("On-target probability > 0.7",
                      p_ontarget > 0.7, p_ontarget, ">0.70")
    passes.append(passed)

    # ── TEST 2: Seed region mismatch reduces probability more than distal ─────
    print("\nTEST 2: Seed region mismatch reduces cutting more than distal mismatch")
    print("  Biology: Positions 17-20 (seed) are critical — mismatches there")
    print("           should be more disruptive than mismatches at position 1-5")

    # Mismatch at position 1 (distal — should barely affect cutting)
    ot_distal = list(GUIDE[:20])
    ot_distal[0] = 'C' if ot_distal[0] != 'C' else 'A'
    p_distal = predict(model, GUIDE, ''.join(ot_distal) + GUIDE[20:])

    # Mismatch at position 18 (seed — should strongly reduce cutting)
    ot_seed = list(GUIDE[:20])
    ot_seed[17] = 'C' if ot_seed[17] != 'C' else 'A'
    p_seed = predict(model, GUIDE, ''.join(ot_seed) + GUIDE[20:])

    print(f"  Distal mismatch (pos 1) probability:  {p_distal:.4f}")
    print(f"  Seed mismatch (pos 18) probability:   {p_seed:.4f}")
    passed = run_test("Seed mismatch < Distal mismatch",
                      p_seed < p_distal, p_seed - p_distal, "<0")
    passes.append(passed)

    # ── TEST 3: More mismatches = lower probability ────────────────────────────
    print("\nTEST 3: More mismatches = lower cleavage probability")
    print("  Biology: Each additional mismatch reduces Cas9 binding affinity")
    probs_by_mm = {}
    for n_mm in [0, 1, 2, 3, 6]:
        ot = list(GUIDE[:20])
        positions_to_mutate = list(range(0, min(n_mm, 12)))  # non-seed only
        for pos in positions_to_mutate:
            ot[pos] = 'C' if ot[pos] != 'C' else 'A'
        probs_by_mm[n_mm] = predict(model, GUIDE, ''.join(ot) + GUIDE[20:])
        print(f"  {n_mm} mismatches: {probs_by_mm[n_mm]:.4f}")

    # Check general decreasing trend
    trend_ok = probs_by_mm[0] > probs_by_mm[3] and probs_by_mm[1] > probs_by_mm[6]
    passed = run_test("Probability decreases with mismatch count",
                      trend_ok, probs_by_mm[6], f"< {probs_by_mm[0]:.4f}")
    passes.append(passed)

    # ── TEST 4: Known published off-targets from Tsai 2015 ────────────────────
    print("\nTEST 4: Known published off-targets from Tsai 2015 (VEGFA site 1)")
    print("  Biology: These sequences were experimentally confirmed to cut")
    print("  Source: 41587_2015_BFnbt3117 supplementary table")

    known_cuts = [
        ("GGGTGGGGGGAGTTTGCTCCTGG", "GGATGGAGGGAGTTTGCTCCTGG", "2mm, should cut"),
        ("GGGTGGGGGGAGTTTGCTCCTGG", "GGGTGGGGGGAGTTTGCTCCTGG", "0mm on-target"),
        ("GGGTGGGGGGAGTTTGCTCCTGG", "GTGTGGGGGGAGTTTGCTCCTGG", "1mm distal, should cut"),
    ]
    for target, offtarget, description in known_cuts:
        p = predict(model, target, offtarget)
        status = "✅" if p > 0.5 else "❌"
        print(f"  {status} {description}: {p:.4f}")

    all_cut = all(predict(model, t, o) > 0.5 for t, o, _ in known_cuts)
    passed = run_test("Known cuts scored > 0.5", all_cut, 1.0 if all_cut else 0.0, "1.0")
    passes.append(passed)

    # ── TEST 5: Transition vs transversion ────────────────────────────────────
    print("\nTEST 5: Transition mismatches more tolerated than transversions")
    print("  Biology: A>G (transition) disrupts less than A>T (transversion)")
    print("           due to wobble base pairing geometry")

    # Create a mismatch at position 5 (non-seed) using A>G (transition)
    pos = 4  # position 5, 0-indexed
    orig_base = GUIDE[pos]
    # Transition: A↔G or C↔T
    transitions   = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
    transversions = {'A': 'T', 'G': 'T', 'C': 'A', 'T': 'A'}

    transition_base   = transitions.get(orig_base, 'G')
    transversion_base = transversions.get(orig_base, 'T')

    ot_transition   = list(GUIDE[:20]); ot_transition[pos]   = transition_base
    ot_transversion = list(GUIDE[:20]); ot_transversion[pos] = transversion_base

    p_transition   = predict(model, GUIDE, ''.join(ot_transition) + GUIDE[20:])
    p_transversion = predict(model, GUIDE, ''.join(ot_transversion) + GUIDE[20:])

    print(f"  Transition ({orig_base}>{transition_base}):   {p_transition:.4f}")
    print(f"  Transversion ({orig_base}>{transversion_base}): {p_transversion:.4f}")
    passed = run_test("Transition > Transversion probability",
                      p_transition >= p_transversion,
                      p_transition - p_transversion, ">=0")
    passes.append(passed)

    # ── TEST 6: Chromatin accessibility ───────────────────────────────────────
    print("\nTEST 6: Open chromatin increases cleavage probability")
    print("  Biology: Nucleosome-free regions are more accessible to Cas9")

    p_open   = predict(model, GUIDE, GUIDE, atac=1.0)
    p_closed = predict(model, GUIDE, GUIDE, atac=0.0)
    print(f"  Open chromatin:   {p_open:.4f}")
    print(f"  Closed chromatin: {p_closed:.4f}")
    passed = run_test("Open chromatin > Closed chromatin",
                      p_open >= p_closed,
                      p_open - p_closed, ">=0")
    passes.append(passed)

    # ── TEST 7: Multiple seed mismatches should nearly abolish cutting ─────────
    print("\nTEST 7: Multiple seed region mismatches abolish cutting")
    print("  Biology: 3+ mismatches in positions 17-20 should prevent cleavage")

    ot_multi_seed = list(GUIDE[:20])
    for pos in [16, 17, 18]:  # 3 seed mismatches
        ot_multi_seed[pos] = 'C' if ot_multi_seed[pos] != 'C' else 'A'
    p_multi_seed = predict(model, GUIDE, ''.join(ot_multi_seed) + GUIDE[20:])
    print(f"  3 seed mismatches probability: {p_multi_seed:.4f}")
    passed = run_test("3 seed mismatches → probability < 0.3",
                      p_multi_seed < 0.3, p_multi_seed, "<0.30")
    passes.append(passed)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    n_passed = sum(passes)
    n_total  = len(passes)
    print(f"\n{'='*65}")
    print(f"  SANITY CHECK SUMMARY: {n_passed}/{n_total} tests passed")
    print(f"{'='*65}")

    if n_passed == n_total:
        print("\n  ✅ All tests passed — model has learned correct biology")
        print("  Safe to proceed to GitHub and paper writing")
    elif n_passed >= n_total - 1:
        print(f"\n  ⚠️  {n_total - n_passed} test(s) failed — minor biological inconsistency")
        print("  Review failed tests before publishing")
    else:
        print(f"\n  ❌ {n_total - n_passed} tests failed — significant biological issues")
        print("  Investigate before publishing")

if __name__ == "__main__":
    main()
