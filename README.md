# Veritas — deepfake detection via local rPPG consistency

Pipeline (matches the 7 steps in the project doc):
1. `veritas/roi.py` — OpenCV + MediaPipe Face Landmarker → forehead / left cheek / right cheek ROIs
2. `veritas/signals.py` — mean RGB per ROI per frame → green / CHROM / POS pulse signal
3. `veritas/signals.py` — detrend, 0.7–3.0 Hz band-pass, FFT → BPM; `validate_ubfc.py` → MAE on UBFC-rPPG
4. `veritas/features.py` — cross-region coherence, phase lag, SNR agreement per window
5. `veritas/classify.py` — Logistic Regression / SVM per window, majority vote per video
6. Models A (global HR) vs B (local only) vs C (combined) — same splits
7. GroupKFold by person / source video, optional compression test (`--compress-crf`)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py            # first run auto-downloads models/face_landmarker.task
```
If the auto-download fails, download the file manually (URL printed in the error) into `models/`.

## Validate pulse extraction (Step 3)
```bash
python validate_ubfc.py --root /path/to/UBFC --method green     # then chrom, pos
```

## Train + hypothesis test (Steps 5–7)
`data/manifest.csv`:
```
path,label,person,source
videos/real_01.mp4,real,p01,s01
videos/fake_01_from_s01.mp4,fake,p01,s01
```
```bash
python train.py --manifest data/manifest.csv --method green --group-by person
python train.py --manifest data/manifest.csv --group-by source
python train.py --manifest data/manifest.csv --compress-crf 28     # robustness
```
Prints the A/B/C comparison table and saves `models/veritas_model.joblib`, which the app then uses for its verdict.

Datasets: UBFC-rPPG (pulse validation); FaceForensics++ / Celeb-DF / DFDC (real vs fake).
Use clips ≥ 10 s with a frontal face.
