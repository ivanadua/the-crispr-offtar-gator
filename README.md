# CRISPR Off-Target Cleavage Predictor

A hybrid CNN + self-attention model for predicting Cas9 off-target cleavage probability, incorporating sequence features, biological properties, and chromatin accessibility.

## Requirements

```bash
pip install torch pandas numpy scipy ViennaRNA
```

## Setup

1. Download `hybrid_model.pt` and `predict_offtarget.py` to the same folder
2. Optionally download `shared_utils.py` if retraining

## Usage

### Single Sequence Pair

```bash
python predict_offtarget.py \
    --target  GGGTGGGGGGAGTTTGCTCCNGG \
    --offtarget GGATGGGGGGAGTTTGCTCCNGG
```

Output:
```
=======================================================
  CRISPR Off-Target Prediction
=======================================================
  Target:      GGGTGGGGGGAGTTTGCTCC + NGG
  Off-target:  GGATGGGGGGAGTTTGCTCC
-------------------------------------------------------
  Cleavage probability: 0.847
  Risk level:           HIGH   ⚠️
-------------------------------------------------------
  Mismatch count:       1
  Mismatch positions:   [1] (1=distal, 20=PAM-proximal)
  Mismatch types:       ['G>A']
  Seed mismatches:      0
  Non-seed mismatches:  1
  PAM proximity score:  0.05
  GC content (target):  0.650
  MFE (RNA stability):  -2.10 kcal/mol
=======================================================
```

### With Chromatin Accessibility

```bash
python predict_offtarget.py \
    --target  GGGTGGGGGGAGTTTGCTCCNGG \
    --offtarget GGATGGGGGGAGTTTGCTCCNGG \
    --atac 1
```

### Batch Mode (CSV Input)

Create a CSV with columns `target_seq` and `offtarget_seq`:

```csv
target_seq,offtarget_seq,atac_accessible
GGGTGGGGGGAGTTTGCTCCNGG,GGATGGGGGGAGTTTGCTCCNGG,1
GGGTGGGGGGAGTTTGCTCCNGG,GTGTGGGGGGAGTTTGCTCCNGG,0
```

Then run:

```bash
python predict_offtarget.py --input sequences.csv --output predictions.csv
```

### Custom Model Path

```bash
python predict_offtarget.py \
    --target GGGTGGGGGGAGTTTGCTCCNGG \
    --offtarget GGATGGGGGGAGTTTGCTCCNGG \
    --model /path/to/hybrid_model.pt
```

## Risk Levels

| Level | Probability | Meaning |
|---|---|---|
| HIGH ⚠️ | ≥ 0.70 | Likely off-target cut — investigate |
| MEDIUM ⚡ | 0.40–0.69 | Possible off-target — use caution |
| LOW ✅ | < 0.40 | Unlikely to cut |

## Model Architecture

The predictor uses a Hybrid CNN + Self-Attention model with two branches:

**Sequence branch** — CNN layers learn local mismatch patterns, followed by multi-head self-attention with biologically-motivated positional weights (positions 17-20 near PAM weighted highest)

**Biological features branch** — Dense layers process 12 hand-crafted features:
- GC content (target and off-target)
- Shannon entropy (target and off-target)
- Mismatch count, seed mismatches, non-seed mismatches
- PAM proximity score
- Guide RNA length
- Minimum free energy (RNA stability)
- Bulge detection
- Chromatin accessibility (ATAC-seq)

## Training Data

Trained on 210,774 sequence pairs from:
- GUIDE-seq (Tsai et al. 2015)
- CHANGE-seq (Lazzarotto et al. 2020)
- CIRCLE-seq
- SITE-seq (Cameron et al. 2017)
- Synthetic negatives (7-tier generation method)

## Citation

If you use this tool, please cite:
```
[Your paper citation here]
```

## Performance

| Model | AUC-ROC | AUC-PR |
|---|---|---|
| MIT Score (Hsu 2013) | 0.2750 | 0.5865 |
| CFD Score (Doench 2016) | 0.3358 | 0.6200 |
| CNN Only | 0.9381 | 0.9731 |
| XGBoost | 0.9471 | 0.9626 |
| Logistic Regression (baseline) | 0.7146 | 0.7572 |
| Elevation-equivalent (Listgarten 2018)| 0.9418 | 0.9565 |
| **Hybrid CNN+Attention (this model)** | **0.9681** | **0.9772** |

Limitations
Model performance drops on independent datasets (AUC-ROC ~0.63), indicating limited generalization
Likely causes include dataset heterogeneity and reliance on sequence-derived features
Future work will focus on improving cross-dataset robustness and incorporating additional biological context
## License

MIT License
