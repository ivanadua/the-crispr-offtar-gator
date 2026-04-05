"""
CRISPR Off-Tar-Gator — Web Interface
========================================
Streamlit app for the hybrid CNN+Attention off-target cleavage predictor.

Installation:
    pip install streamlit torch pandas numpy scipy ViennaRNA

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    Push to GitHub, connect repo at share.streamlit.io
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import io
import sys
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRISPR Off-Tar-Gator",
    page_icon="🐊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #39d353;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}

.sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #8b949e;
    margin-top: 4px;
    margin-bottom: 32px;
}

.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
}

.risk-high {
    color: #f85149;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}

.risk-medium {
    color: #e3b341;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}

.risk-low {
    color: #39d353;
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}

.prob-display {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}

.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.9rem;
}

.feature-label {
    color: #8b949e;
}

.feature-value {
    color: #e6edf3;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.seq-display {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px 14px;
    letter-spacing: 0.08em;
    color: #58a6ff;
    word-break: break-all;
}

.mismatch-highlight {
    color: #f85149;
    font-weight: 700;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}

.badge-green { background: #1a4722; color: #39d353; border: 1px solid #2ea043; }
.badge-yellow { background: #3d2e00; color: #e3b341; border: 1px solid #9e6a03; }
.badge-red { background: #3d0f0f; color: #f85149; border: 1px solid #da3633; }

.stTextInput > div > div > input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
}

.stButton > button {
    background-color: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background-color: #2ea043 !important;
}

.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 24px 0;
}

.info-box {
    background: #161b22;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #8b949e;
    margin: 8px 0;
}

.stFileUploader {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model definition (must match hybrid1.py) ─────────────────────────────────
POS_WEIGHTS = torch.tensor([
    0.05,0.05,0.05,0.07,0.07,0.07,0.08,0.08,0.08,0.10,
    0.12,0.15,0.20,0.30,0.40,0.55,0.70,0.85,0.95,1.00
], dtype=torch.float32)

BASES        = ['A','C','G','T']
BIO_FEATURES = [
    'gc_content_target','gc_content_offtarget',
    'shannon_entropy_target','shannon_entropy_offt',
    'mismatch_count','seed_mismatches','nonseed_mismatches',
    'pam_proximity_score','guide_length','mfe','has_bulge','atac_accessible'
]

class HybridModel(nn.Module):
    def __init__(self, n_bio=12):
        super().__init__()
        self.conv1    = nn.Conv1d(8,  64,  kernel_size=3, padding=1)
        self.conv2    = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1      = nn.BatchNorm1d(64)
        self.bn2      = nn.BatchNorm1d(128)
        self.attn     = nn.MultiheadAttention(128, num_heads=4, batch_first=True, dropout=0.1)
        self.register_buffer('pos_weights', POS_WEIGHTS)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_fc   = nn.Linear(128, 64)
        self.bio_fc1  = nn.Linear(n_bio, 64)
        self.bio_fc2  = nn.Linear(64, 32)
        self.bio_bn   = nn.BatchNorm1d(64)
        self.fuse1    = nn.Linear(96, 64)
        self.fuse2    = nn.Linear(64, 32)
        self.fuse3    = nn.Linear(32, 1)
        self.drop     = nn.Dropout(0.3)
        self.relu     = nn.ReLU()

    def forward(self, x_seq, x_bio):
        x = x_seq.permute(0,2,1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0,2,1)
        x = x * self.pos_weights.unsqueeze(0).unsqueeze(-1)
        x, _ = self.attn(x,x,x)
        x = x.permute(0,2,1)
        x = self.seq_pool(x).squeeze(-1)
        x_s = self.drop(self.relu(self.seq_fc(x)))
        x_b = self.relu(self.bio_bn(self.bio_fc1(x_bio)))
        x_b = self.drop(self.relu(self.bio_fc2(x_b)))
        x_f = torch.cat([x_s,x_b],dim=1)
        x_f = self.drop(self.relu(self.fuse1(x_f)))
        x_f = self.drop(self.relu(self.fuse2(x_f)))
        return self.fuse3(x_f).squeeze(-1)

# ── Feature computation ───────────────────────────────────────────────────────
def encode_pair(target, offtarget, length=20):
    base_idx = {'A':0,'C':1,'G':2,'T':3,'N':-1}
    enc = np.zeros((length, 8), dtype=np.float32)
    for i, (t, o) in enumerate(zip(target[:length], offtarget[:length])):
        ti = base_idx.get(t.upper(), -1)
        oi = base_idx.get(o.upper(), -1)
        if ti >= 0: enc[i, ti]   = 1.0
        if oi >= 0: enc[i, oi+4] = 1.0
    return enc

def gc_content(seq):
    seq = seq[:20].upper().replace('N','')
    return (seq.count('G')+seq.count('C'))/len(seq) if seq else 0.0

def shannon_entropy(seq):
    from scipy.stats import entropy as sp_entropy
    seq    = seq[:20].upper().replace('N','')
    counts = [seq.count(b)/len(seq) for b in 'ACGT']
    counts = [c for c in counts if c > 0]
    return float(sp_entropy(counts, base=2)) if counts else 0.0

def get_mfe(seq):
    try:
        import RNA
        rna = seq[:20].upper().replace('T','U').replace('N','A')
        _, mfe = RNA.fold(rna)
        return round(float(mfe), 3)
    except:
        return -2.0

def compute_features(target, offtarget):
    t = target[:20].upper()
    o = offtarget[:20].upper()
    pw = POS_WEIGHTS.numpy()
    seed_mm = nonseed_mm = 0
    pam_score = 0.0
    mm_positions = []
    mm_types = []
    for i, (tb, ob) in enumerate(zip(t, o)):
        if tb != ob and tb != 'N' and ob != 'N':
            mm_positions.append(i+1)
            mm_types.append(f"{tb}→{ob}")
            pam_score += pw[i]
            if i >= 16: seed_mm += 1
            else:       nonseed_mm += 1
    mm_count = seed_mm + nonseed_mm
    bio = np.array([
        gc_content(t), gc_content(o),
        shannon_entropy(t), shannon_entropy(o),
        float(mm_count), float(seed_mm), float(nonseed_mm),
        float(pam_score),
        float(len(t.replace('N',''))),
        get_mfe(t),
        0.0, 0.0,
    ], dtype=np.float32)
    return bio, mm_positions, mm_types, mm_count, seed_mm, nonseed_mm, round(float(pam_score),3)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'hybrid_model.pt')
    if not os.path.exists(model_path):
        return None
    device = torch.device('cpu')
    model  = HybridModel(n_bio=12)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, target, offtarget):
    bio, mm_pos, mm_types, mm_count, seed_mm, nonseed_mm, pam_score = \
        compute_features(target, offtarget)
    enc = encode_pair(target, offtarget)
    xc  = torch.tensor(enc[np.newaxis],  dtype=torch.float32)
    xf  = torch.tensor(bio[np.newaxis],  dtype=torch.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(xc, xf)).item()
    return p, mm_pos, mm_types, mm_count, seed_mm, nonseed_mm, pam_score

def risk_label(p):
    if p >= 0.70: return "HIGH", "risk-high", "badge-red"
    if p >= 0.40: return "MEDIUM", "risk-medium", "badge-yellow"
    return "LOW", "risk-low", "badge-green"

def clean_seq(s):
    s = s.upper().strip()
    for suffix in ['NGG','NAG','NGA','NCG']:
        if s.endswith(suffix):
            s = s[:-3]
    return s[:20]

def highlight_mismatches(target, offtarget):
    t = target[:20].upper()
    o = offtarget[:20].upper()
    html_t = html_o = ""
    for tb, ob in zip(t, o):
        if tb != ob:
            html_t += f'<span class="mismatch-highlight">{tb}</span>'
            html_o += f'<span class="mismatch-highlight">{ob}</span>'
        else:
            html_t += tb
            html_o += ob
    return html_t, html_o

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🐊 CRISPR Off-Tar-Gator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid CNN + Attention model for Cas9 off-target cleavage prediction</div>', unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("Model file (hybrid_model.pt) not found. Place it in the same directory as app.py.")
    st.stop()

tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])

# ── Tab 1: Single prediction ──────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Input sequences")
        st.markdown('<div class="info-box">Enter 20-nt sequences. PAM (NGG) will be stripped automatically if present.</div>', unsafe_allow_html=True)

        target_input = st.text_input(
            "Guide RNA target sequence",
            placeholder="e.g. GGGTGGGGGGAGTTTGCTCC",
            key="target"
        )
        offtarget_input = st.text_input(
            "Off-target sequence",
            placeholder="e.g. GGATGGGGGGAGTTTGCTCC",
            key="offtarget"
        )
        atac = st.checkbox("Site is in accessible chromatin (ATAC-seq peak)", value=False)
        predict_btn = st.button("Predict cleavage risk", key="predict_single")

    with col2:
        if predict_btn and target_input and offtarget_input:
            target_clean    = clean_seq(target_input)
            offtarget_clean = clean_seq(offtarget_input)

            if len(target_clean) < 15 or len(offtarget_clean) < 15:
                st.error("Sequences must be at least 15 nucleotides.")
            else:
                p, mm_pos, mm_types, mm_count, seed_mm, nonseed_mm, pam_score = \
                    predict(model, target_clean, offtarget_clean)

                # override atac feature if checked
                if atac:
                    bio, *_ = compute_features(target_clean, offtarget_clean)
                    bio[11] = 1.0
                    enc = encode_pair(target_clean, offtarget_clean)
                    xc  = torch.tensor(enc[np.newaxis], dtype=torch.float32)
                    xf  = torch.tensor(bio[np.newaxis], dtype=torch.float32)
                    with torch.no_grad():
                        p = torch.sigmoid(model(xc, xf)).item()

                risk, risk_cls, badge_cls = risk_label(p)
                html_t, html_o = highlight_mismatches(target_clean, offtarget_clean)

                st.markdown("#### Prediction result")
                st.markdown(f"""
                <div class="result-card">
                  <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:16px">
                    <div class="{risk_cls} prob-display">{p:.3f}</div>
                    <div>
                      <span class="badge {badge_cls}">{risk} RISK</span>
                      <div style="color:#8b949e;font-size:0.8rem;margin-top:4px">cleavage probability</div>
                    </div>
                  </div>

                  <div style="margin-bottom:16px">
                    <div style="font-size:0.75rem;color:#8b949e;margin-bottom:4px;font-family:'Space Mono',monospace">TARGET</div>
                    <div class="seq-display">{html_t}</div>
                    <div style="font-size:0.75rem;color:#8b949e;margin:6px 0 4px;font-family:'Space Mono',monospace">OFF-TARGET</div>
                    <div class="seq-display">{html_o}</div>
                  </div>

                  <hr class="divider" style="margin:12px 0">

                  <div class="feature-row">
                    <span class="feature-label">Total mismatches</span>
                    <span class="feature-value">{mm_count}</span>
                  </div>
                  <div class="feature-row">
                    <span class="feature-label">Seed region mismatches (pos 17–20)</span>
                    <span class="feature-value">{seed_mm}</span>
                  </div>
                  <div class="feature-row">
                    <span class="feature-label">Non-seed mismatches</span>
                    <span class="feature-value">{nonseed_mm}</span>
                  </div>
                  <div class="feature-row">
                    <span class="feature-label">PAM proximity score</span>
                    <span class="feature-value">{pam_score:.3f}</span>
                  </div>
                  <div class="feature-row">
                    <span class="feature-label">Mismatch positions</span>
                    <span class="feature-value">{', '.join(map(str, mm_pos)) if mm_pos else '—'}</span>
                  </div>
                  <div class="feature-row" style="border-bottom:none">
                    <span class="feature-label">Substitution types</span>
                    <span class="feature-value">{', '.join(mm_types) if mm_types else '—'}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        elif predict_btn:
            st.warning("Please enter both sequences.")
        else:
            st.markdown("""
            <div class="info-box" style="margin-top:40px;text-align:center;padding:32px">
                <div style="font-size:2rem;margin-bottom:8px">🐊</div>
                <div style="color:#e6edf3;font-family:'Space Mono',monospace;font-size:0.9rem">
                    Enter sequences and click predict
                </div>
                <div style="color:#8b949e;font-size:0.8rem;margin-top:6px">
                    Red mismatches highlighted in sequence display
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Batch prediction ───────────────────────────────────────────────────
with tab2:
    st.markdown("#### Batch prediction")
    st.markdown('<div class="info-box">Upload a CSV with columns <code>target_seq</code> and <code>offtarget_seq</code>. Optionally include <code>atac_accessible</code> (0 or 1).</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_in = pd.read_csv(uploaded)
        if 'target_seq' not in df_in.columns or 'offtarget_seq' not in df_in.columns:
            st.error("CSV must contain columns: target_seq, offtarget_seq")
        else:
            st.write(f"Loaded {len(df_in):,} rows")
            run_batch = st.button("Run batch prediction", key="batch_run")

            if run_batch:
                results = []
                progress = st.progress(0)
                for idx, row in df_in.iterrows():
                    t = clean_seq(str(row['target_seq']))
                    o = clean_seq(str(row['offtarget_seq']))
                    try:
                        bio, mm_pos, mm_types, mm_count, seed_mm, nonseed_mm, pam_score = \
                            compute_features(t, o)
                        if 'atac_accessible' in df_in.columns:
                            bio[11] = float(row['atac_accessible'])
                        enc = encode_pair(t, o)
                        xc  = torch.tensor(enc[np.newaxis], dtype=torch.float32)
                        xf  = torch.tensor(bio[np.newaxis], dtype=torch.float32)
                        with torch.no_grad():
                            p = torch.sigmoid(model(xc, xf)).item()
                        risk, _, _ = risk_label(p)
                        results.append({
                            'target_seq':       t,
                            'offtarget_seq':    o,
                            'cleavage_prob':    round(p, 4),
                            'risk_level':       risk,
                            'mismatch_count':   mm_count,
                            'seed_mismatches':  seed_mm,
                            'nonseed_mismatches': nonseed_mm,
                            'pam_proximity_score': pam_score,
                            'mismatch_positions': ';'.join(map(str, mm_pos)),
                            'substitution_types': ';'.join(mm_types),
                        })
                    except Exception as e:
                        results.append({'target_seq': t, 'offtarget_seq': o,
                                        'cleavage_prob': None, 'risk_level': 'ERROR'})
                    progress.progress((idx+1)/len(df_in))

                df_out = pd.DataFrame(results).sort_values('cleavage_prob', ascending=False)
                st.success(f"Done. {len(df_out):,} predictions computed.")
                st.dataframe(df_out.head(50), use_container_width=True)

                csv_bytes = df_out.to_csv(index=False).encode()
                st.download_button(
                    label="Download full results CSV",
                    data=csv_bytes,
                    file_name="offtarget_predictions.csv",
                    mime="text/csv",
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#21262d;margin-top:48px'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#8b949e;font-size:0.8rem;padding:16px 0">
    Hybrid CNN + Attention model · 
    <a href="https://github.com/ivanadua/the-crispr-offtar-gator" style="color:#58a6ff">
        github.com/ivanadua/the-crispr-offtar-gator
    </a>
</div>
""", unsafe_allow_html=True)
