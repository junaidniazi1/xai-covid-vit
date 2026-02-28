import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
import numpy as np
import cv2
from PIL import Image
import sqlite3
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.main { background: #0a0e1a; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1628 50%, #0a0e1a 100%);
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #00d4ff !important;
}

.header-box {
    background: linear-gradient(90deg, rgba(0,212,255,0.1) 0%, rgba(0,255,136,0.05) 100%);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}

.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: -1px;
    margin: 0;
}

.header-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.9rem;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}

.result-card {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}

.class-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.badge-covid    { background: rgba(255,60,60,0.2);   border: 1px solid #ff3c3c; color: #ff3c3c; }
.badge-normal   { background: rgba(0,255,136,0.2);   border: 1px solid #00ff88; color: #00ff88; }
.badge-opacity  { background: rgba(255,165,0,0.2);   border: 1px solid #ffa500; color: #ffa500; }
.badge-pneumonia{ background: rgba(200,100,255,0.2); border: 1px solid #c864ff; color: #c864ff; }

.metric-box {
    flex: 1;
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4ff;
}

.metric-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.4);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #00ff88) !important;
    color: #0a0e1a !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
}

[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.4) !important;
    border-right: 1px solid rgba(0,212,255,0.1) !important;
}

.sidebar-title {
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(0,212,255,0.2);
    margin-bottom: 12px;
}

.warning-box {
    background: rgba(255,60,60,0.1);
    border: 1px solid rgba(255,60,60,0.4);
    border-radius: 8px;
    padding: 12px 16px;
    color: #ff9999;
    font-size: 0.85rem;
}

.info-box {
    background: rgba(0,212,255,0.07);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 8px;
    padding: 12px 16px;
    color: rgba(255,255,255,0.6);
    font-size: 0.82rem;
    line-height: 1.6;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASSES     = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = 4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CNN_PATH = r"C:\Users\kashif-pc\Downloads\rsearchpaper computer vison\New folder\New folder\CNN_best.pth"
DEFAULT_VIT_PATH = r"C:\Users\kashif-pc\Downloads\rsearchpaper computer vison\New folder\New folder\ViT_best.pth"

CLASS_BADGES = {
    'COVID':           'badge-covid',
    'Lung_Opacity':    'badge-opacity',
    'Normal':          'badge-normal',
    'Viral Pneumonia': 'badge-pneumonia',
}

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, filename TEXT,
            cnn_prediction TEXT, cnn_confidence REAL,
            vit_prediction TEXT, vit_confidence REAL,
            agreement INTEGER
        )
    """)
    conn.commit(); conn.close()

def save_prediction(filename, cnn_pred, cnn_conf, vit_pred, vit_conf):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (timestamp,filename,cnn_prediction,cnn_confidence,vit_prediction,vit_confidence,agreement)
        VALUES (?,?,?,?,?,?,?)
    """, (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          filename, cnn_pred, round(cnn_conf,4),
          vit_pred, round(vit_conf,4), int(cnn_pred==vit_pred)))
    conn.commit(); conn.close()

def fetch_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall(); conn.close(); return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions");            total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE agreement=1"); agree = c.fetchone()[0]
    c.execute("SELECT cnn_prediction,COUNT(*) cnt FROM predictions GROUP BY cnn_prediction ORDER BY cnt DESC LIMIT 1")
    top = c.fetchone(); conn.close(); return total, agree, top

init_db()

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone    = models.resnet50(weights=None)   # FIX: pretrained→weights
        self.feature_maps = None
        self.gradients    = None
        self.backbone.fc  = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.layer4.register_forward_hook(self._save_feature_maps)
        self.backbone.layer4.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, inp, out):  self.feature_maps = out.detach()
    def _save_gradients   (self, module, gi,  go):   self.gradients    = go[0].detach()

    def forward(self, x): return self.backbone(x)

    def get_gradcam(self):
        if self.gradients is None or self.feature_maps is None:
            return None
        w   = torch.mean(self.gradients, dim=[2,3], keepdim=True)
        cam = F.relu(torch.sum(w * self.feature_maps, dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


class VisionTransformerExplainable(nn.Module):
    """
    Hooks into timm ViT attention.
    timm's Attention module returns the projected output (not raw attn weights)
    from its forward().  We must hook into the softmax attention weights instead,
    which live inside Attention.forward as a local variable.
    We do this by subclassing / monkey-patching or — more robustly — by reading
    the q/k dot-product result via a custom forward hook on the attn_drop layer.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.vit           = timm.create_model('vit_base_patch16_224',
                                               pretrained=False,
                                               num_classes=num_classes)
        self.attention_maps = []   # each entry: [heads, tokens, tokens]
        self._register_hooks()

    def _register_hooks(self):
        """
        Hook onto each block's attn.attn_drop.
        At that point the tensor is already softmax'd: shape [B, heads, N, N].
        """
        def make_hook(idx):
            def hook(module, inp, out):
                # out: [B, heads, N, N]  — grab batch-0, keep on cpu
                self.attention_maps.append(out[0].detach().cpu())  # [heads, N, N]
            return hook

        for i, blk in enumerate(self.vit.blocks):
            # timm ViT block → blk.attn.attn_drop  (nn.Dropout applied after softmax)
            blk.attn.attn_drop.register_forward_hook(make_hook(i))

    def forward(self, x):
        self.attention_maps = []
        return self.vit(x)

    def get_attention_map(self, img_size=224, head_fusion='mean', layer_idx=-1):
        if not self.attention_maps:
            return None

        attn = self.attention_maps[layer_idx]   # [heads, N, N]

        # head fusion
        if head_fusion == 'mean':
            attn = attn.mean(dim=0)             # [N, N]
        elif head_fusion == 'max':
            attn = attn.max(dim=0)[0]           # [N, N]
        else:
            attn = attn[0]                      # first head

        # CLS token → patch tokens: row 0, columns 1:
        cls_attn = attn[0, 1:]                  # [N-1]  = [196] for 224px / patch16

        num_patches = int(np.sqrt(cls_attn.shape[0]))  # 14
        attn_map    = cls_attn.reshape(num_patches, num_patches).numpy()

        # resize to image size
        attn_map = cv2.resize(attn_map, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # return as [1, H, W] so caller can do attn[0]
        return attn_map[np.newaxis, ...]


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_models(cnn_path, vit_path):
    errors = []
    cnn_model = vit_model = None

    try:
        cnn_model = CNNModel(num_classes=NUM_CLASSES).to(DEVICE)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        cnn_model.eval()
    except Exception as e:
        errors.append(f"CNN load error: {e}")

    try:
        vit_model = VisionTransformerExplainable(num_classes=NUM_CLASSES).to(DEVICE)
        vit_model.load_state_dict(torch.load(vit_path, map_location=DEVICE))
        vit_model.eval()
    except Exception as e:
        errors.append(f"ViT load error: {e}")

    return cnn_model, vit_model, errors

# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────
def predict_cnn(model, tensor):
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs

def predict_vit(model, tensor):
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs

def generate_gradcam(model, tensor, pred_class):
    model.eval()
    t   = tensor.clone().requires_grad_(True)
    out = model(t)
    model.zero_grad()
    out[0, pred_class].backward()
    cam = model.get_gradcam()
    if cam is None:
        return None
    return cv2.resize(cam, (224, 224))

def overlay_heatmap(img_np, heatmap, alpha=0.45):
    colored = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    return cv2.addWeighted(np.uint8(255*img_np), 1-alpha, colored, alpha, 0)

def tensor_to_img(tensor):
    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    img  = tensor.squeeze().cpu().numpy().transpose(1,2,0)
    return np.clip(std*img + mean, 0, 1)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Model Config</div>', unsafe_allow_html=True)
    cnn_path = st.text_input("CNN Model Path (.pth)", value=DEFAULT_CNN_PATH)
    vit_path = st.text_input("ViT Model Path (.pth)", value=DEFAULT_VIT_PATH)

    st.markdown('<div class="sidebar-title" style="margin-top:20px;">🧠 Classes</div>', unsafe_allow_html=True)
    for c in CLASSES:
        st.markdown(f'<span class="class-badge {CLASS_BADGES[c]}">{c}</span>', unsafe_allow_html=True)
        st.markdown("")

    st.markdown('<div class="sidebar-title" style="margin-top:20px;">💾 Device</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">Running on: <b style="color:#00d4ff">{DEVICE}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title" style="margin-top:20px;">ℹ Disclaimer</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">⚠️ For research purposes only. Not a substitute for professional medical diagnosis.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
cnn_model, vit_model, load_errors = load_models(cnn_path, vit_path)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <p class="header-title">🫁 COVID-19 X-Ray Classifier</p>
  <p class="header-sub">CNN (ResNet50 + Grad-CAM) &nbsp;|&nbsp; ViT (Vision Transformer + Attention) &nbsp;|&nbsp; 4-Class Chest X-Ray Analysis</p>
</div>
""", unsafe_allow_html=True)

for err in load_errors:
    st.error(err)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📊 History", "📈 Statistics"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    col_upload, col_results = st.columns([1, 2], gap="large")

    with col_upload:
        st.markdown("### Upload X-Ray Image")
        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=["png","jpg","jpeg","bmp","tiff"],
            label_visibility="collapsed"
        )
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, caption="Uploaded Image", width='stretch')
            st.markdown(
                f'<div class="info-box">📄 File: <b>{uploaded.name}</b><br>📐 Size: {img_pil.width}×{img_pil.height}</div>',
                unsafe_allow_html=True
            )
            run_btn = st.button("🚀 Run Prediction")
        else:
            st.markdown('<div class="info-box">Upload a chest X-ray image (PNG, JPG) to begin analysis.</div>', unsafe_allow_html=True)
            run_btn = False

    with col_results:
        if uploaded and run_btn:
            if cnn_model is None and vit_model is None:
                st.error("No models loaded. Check model paths in the sidebar.")
            else:
                with st.spinner("Analyzing X-ray..."):
                    tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
                    img_np = tensor_to_img(tensor)
                    results = {}

                    # ── CNN
                    if cnn_model is not None:
                        try:
                            cnn_pred, cnn_probs = predict_cnn(cnn_model, tensor)
                            gradcam = generate_gradcam(cnn_model, tensor, cnn_pred)
                            results['cnn'] = {
                                'class':   CLASSES[cnn_pred],
                                'probs':   cnn_probs,
                                'conf':    float(cnn_probs[cnn_pred]),
                                'heatmap': gradcam,
                            }
                        except Exception as e:
                            st.warning(f"CNN prediction error: {e}")

                    # ── ViT
                    if vit_model is not None:
                        try:
                            vit_pred, vit_probs = predict_vit(vit_model, tensor)
                            # attention maps are collected during predict_vit forward pass
                            attn = vit_model.get_attention_map()
                            results['vit'] = {
                                'class':   CLASSES[vit_pred],
                                'probs':   vit_probs,
                                'conf':    float(vit_probs[vit_pred]),
                                'heatmap': attn[0] if attn is not None else None,
                            }
                        except Exception as e:
                            st.warning(f"ViT prediction error: {e}")

                    # Save to DB
                    cnn_res = results.get('cnn', {})
                    vit_res = results.get('vit', {})
                    save_prediction(
                        uploaded.name,
                        cnn_res.get('class','N/A'), cnn_res.get('conf',0.0),
                        vit_res.get('class','N/A'), vit_res.get('conf',0.0),
                    )

                # ── Agreement banner
                if 'cnn' in results and 'vit' in results:
                    if results['cnn']['class'] == results['vit']['class']:
                        st.success(f"✅ Both models agree: **{results['cnn']['class']}**")
                    else:
                        st.warning("⚠️ Models disagree — review both predictions below.")

                # ── Result cards
                model_keys = list(results.keys())
                for col, key in zip(st.columns(len(results)), model_keys):
                    r     = results[key]
                    badge = CLASS_BADGES.get(r['class'], 'badge-normal')
                    label = "🧱 CNN (ResNet50)" if key == 'cnn' else "🤖 ViT"
                    with col:
                        st.markdown(f"**{label}**")
                        st.markdown(f'<span class="class-badge {badge}">{r["class"]}</span>', unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** {r['conf']*100:.1f}%")
                        st.progress(r['conf'])
                        st.markdown("**Class probabilities:**")
                        for ci, cls in enumerate(CLASSES):
                            st.markdown(f"<small style='color:rgba(255,255,255,0.5)'>{cls}</small>", unsafe_allow_html=True)
                            st.progress(float(r['probs'][ci]))

                # ── Explainability maps
                st.markdown("### 🔍 Explainability Maps")
                viz_cols = st.columns(1 + len(results))
                with viz_cols[0]:
                    st.markdown("**Original**")
                    st.image(img_np, width='stretch')

                for i, key in enumerate(model_keys):
                    r = results[key]
                    with viz_cols[i+1]:
                        st.markdown(f"**{'Grad-CAM (CNN)' if key=='cnn' else 'Attention (ViT)'}**")
                        if r['heatmap'] is not None:
                            st.image(overlay_heatmap(img_np, r['heatmap']), width='stretch')
                        else:
                            st.info("Heatmap unavailable")

        elif not uploaded:
            st.markdown(
                '<div style="margin-top:60px;text-align:center;color:rgba(255,255,255,0.2);'
                'font-family:Space Mono,monospace;font-size:1.1rem;">← Upload an image to see results</div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Prediction History")
    rows = fetch_history(50)
    if not rows:
        st.info("No predictions yet. Upload an X-ray to get started.")
    else:
        st.markdown(f"**{len(rows)} recent predictions**")
        for col, h in zip(
            st.columns([1,2,2,2,2,2,1]),
            ["ID","Timestamp","File","CNN Pred","ViT Pred","Confidence","Agree"]
        ):
            col.markdown(f"<small style='color:#00d4ff;font-family:Space Mono'><b>{h}</b></small>", unsafe_allow_html=True)
        st.divider()
        for row in rows:
            id_, ts, fn, cnn_p, cnn_c, vit_p, vit_c, agree = row
            cols = st.columns([1,2,2,2,2,2,1])
            cols[0].markdown(f"<small style='color:rgba(255,255,255,0.4)'>{id_}</small>", unsafe_allow_html=True)
            cols[1].markdown(f"<small>{ts}</small>", unsafe_allow_html=True)
            cols[2].markdown(f"<small>{fn[:20]}</small>", unsafe_allow_html=True)
            cols[3].markdown(f'<span class="class-badge {CLASS_BADGES.get(cnn_p,"badge-normal")}" style="font-size:0.7rem;padding:3px 8px">{cnn_p}</span>', unsafe_allow_html=True)
            cols[4].markdown(f'<span class="class-badge {CLASS_BADGES.get(vit_p,"badge-normal")}" style="font-size:0.7rem;padding:3px 8px">{vit_p}</span>', unsafe_allow_html=True)
            cols[5].markdown(f"<small>{(cnn_c+vit_c)/2*100:.1f}%</small>", unsafe_allow_html=True)
            cols[6].markdown("✅" if agree else "❌")

# ══════════════════════════════════════════════
# TAB 3 — STATISTICS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Database Statistics")
    total, agree_count, top_class = get_stats()
    agree_pct = (agree_count/total*100) if total > 0 else 0
    top_name  = top_class[0] if top_class else "N/A"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div><div class="metric-label">Total Predictions</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{agree_pct:.0f}%</div><div class="metric-label">Model Agreement</div></div>', unsafe_allow_html=True)
    with m3:
        badge = CLASS_BADGES.get(top_name, 'badge-normal')
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size:1rem"><span class="class-badge {badge}">{top_name}</span></div><div class="metric-label">Most Frequent Class</div></div>', unsafe_allow_html=True)

    if total > 0:
        import pandas as pd
        conn     = sqlite3.connect(DB_PATH)
        df_stats = pd.read_sql("SELECT cnn_prediction as Class, COUNT(*) as Count FROM predictions GROUP BY cnn_prediction", conn)
        conn.close()

        st.markdown("#### CNN Prediction Distribution")
        colors = {'COVID':'#ff3c3c','Lung_Opacity':'#ffa500','Normal':'#00ff88','Viral Pneumonia':'#c864ff'}
        fig, ax = plt.subplots(figsize=(7,3))
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#0f1628')
        bars = ax.bar(df_stats['Class'], df_stats['Count'],
                      color=[colors.get(c,'#00d4ff') for c in df_stats['Class']],
                      edgecolor='none', width=0.5)
        # FIX: matplotlib color tuples, not CSS rgba()
        ax.set_xlabel("Class", color=(1,1,1,0.5), fontsize=9)
        ax.set_ylabel("Count", color=(1,1,1,0.5), fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor((1,1,1,0.1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, count in zip(bars, df_stats['Count']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    str(count), ha='center', va='bottom', color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    col_clear, _ = st.columns([1,3])
    with col_clear:
        if st.button("🗑 Clear History"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM predictions"); conn.commit(); conn.close()
            st.success("History cleared.")
            st.rerun()