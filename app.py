"""Veritas demo UI.  Run:  streamlit run app.py"""
from __future__ import annotations
import os, tempfile
import cv2, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from veritas.config import Config, REGIONS, PAIRS
from veritas import roi, signals as S, features as F
from veritas.classify import majority_vote

st.set_page_config(page_title="Veritas", page_icon="🫀", layout="wide")
st.title("🫀 Veritas — deepfake detection via local blood-flow consistency")
st.caption("Checks whether pulse signals from forehead / left cheek / right cheek agree with each other, "
           "not just whether a pulse exists.")

MODEL_PATH = "models/veritas_model.joblib"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

with st.sidebar:
    method = st.selectbox("Pulse extraction", ["green", "chrom", "pos"])
    window = st.slider("Window (s)", 5, 20, 10)
    stride = st.slider("Stride (s)", 1, 10, 5)
    max_sec = st.slider("Max video length to analyse (s)", 10, 120, 40)
    st.markdown("---")
    st.write("Trained model:", f"✅ {bundle['name']}" if bundle else "❌ none (run train.py)")

up = st.file_uploader("Upload a face video (mp4 / avi / mov, frontal face, ≥10 s)", type=["mp4", "avi", "mov", "mkv"])
if not up:
    st.info("Upload a video to start."); st.stop()

cfg = Config(method=method, window_sec=window, stride_sec=stride)
with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
    f.write(up.read()); path = f.name

cap = cv2.VideoCapture(path); fps0 = cap.get(cv2.CAP_PROP_FPS) or 30; cap.release()
bar = st.progress(0.0, text="Step 1-2: tracking face & extracting colour traces…")
try:
    traces, fps, valid, frame, masks = roi.extract_rgb_traces(path, int(max_sec * fps0), progress=lambda p: bar.progress(p))
except RuntimeError as e:
    st.error(str(e)); st.stop()
bar.empty()

if valid.mean() < 0.5 or frame is None:
    st.error(f"Face found in only {valid.mean()*100:.0f}% of frames — need a clear, frontal face."); st.stop()
if len(valid) / fps < window:
    st.error("Video shorter than one analysis window."); st.stop()

sigs = S.process_regions(traces, fps, cfg)
wfs = F.video_features(sigs, fps, cfg)
cols = st.columns([1, 1.4])

# ---- Explainability: per-region consistency heatmap ----
region_score = {r: 0.0 for r in REGIONS}
for wf in wfs:
    pc = wf["_pair_coh"]
    for r in REGIONS:
        pr = [v for k, v in pc.items() if r[0].upper() in k.split("-")]  # F/L/R letters
        region_score[r] += np.mean(pr) / len(wfs)
cmap = plt.get_cmap("RdYlGn")
colors = {r: tuple(int(255 * c) for c in cmap(region_score[r])[:3][::-1]) for r in REGIONS}  # BGR
overlay = roi.draw_rois(frame, masks, 0.5, colors)

with cols[0]:
    st.subheader("Regions & consistency heatmap")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="green = region agrees with the others · red = inconsistent")
    st.write({r: round(v, 2) for r, v in region_score.items()})

with cols[1]:
    st.subheader("Cleaned pulse signals (first 10 s)")
    n = int(10 * fps); t = np.arange(n) / fps
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    for r in REGIONS:
        ax[0].plot(t, sigs[r][:n], label=r)
        _, fr, p, _ = S.fft_hr(sigs[r], fps)
        m = (fr > 0.5) & (fr < 4); ax[1].plot(fr[m] * 60, p[m] / p[m].max(), label=r)
    ax[0].set_xlabel("s"); ax[0].legend(); ax[1].set_xlabel("BPM"); ax[1].set_ylabel("norm. power")
    plt.tight_layout(); st.pyplot(fig)

st.subheader("Heart rate per region")
hr = {r: S.fft_hr(sigs[r], fps)[0] for r in REGIONS}
snr = {r: S.snr_db(sigs[r], fps) for r in REGIONS}
st.dataframe(pd.DataFrame({"HR (BPM)": hr, "SNR (dB)": snr}).round(1).T)

st.subheader("Local consistency features (per window)")
tbl = pd.DataFrame([{k: v for k, v in w.items() if not k.startswith("_")} for w in wfs]).set_index("t_start")
st.dataframe(tbl.round(3))
cm = np.eye(3)
idx = {r: i for i, r in enumerate(REGIONS)}
mean_pc = {k: np.mean([w["_pair_coh"][k] for w in wfs]) for k in ["F-L", "F-R", "L-R"]}
for (a, b), k in zip(PAIRS, mean_pc):
    cm[idx[a], idx[b]] = cm[idx[b], idx[a]] = mean_pc[k]
fig2, ax2 = plt.subplots(figsize=(3.6, 3))
im = ax2.imshow(cm, vmin=0, vmax=1, cmap="viridis"); ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
ax2.set_xticklabels(["F", "L", "R"]); ax2.set_yticklabels(["F", "L", "R"]); plt.colorbar(im)
ax2.set_title("Cross-region coherence @ pulse freq")
st.pyplot(fig2)

st.subheader("Verdict")
if bundle:
    X = F.to_matrix(wfs, bundle["cols"])
    pred = bundle["model"].predict(X); prob = bundle["model"].predict_proba(X)[:, 1]
    final = majority_vote(pred)
    st.metric("Result (majority vote over windows)", "🚩 LIKELY FAKE" if final else "✅ LIKELY REAL",
              f"{int(pred.sum())}/{len(pred)} windows flagged · mean P(fake)={prob.mean():.2f}")
else:
    st.warning("No trained classifier yet — showing features only. Train with `python train.py --manifest …` "
               "to get a real/fake verdict. Nothing here is a verdict until the model is validated on held-out people.")
st.caption("Limitations: motion, lighting, heavy compression and skin tone affect rPPG quality. "
           "Report results by person/source split, never random clip split.")
