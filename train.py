"""Build features from a labelled dataset, run the A/B/C hypothesis test with GroupKFold, save a model.

Manifest CSV columns:  path,label,person,source
  label  : real / fake  (or 0 / 1)
  person : identity in the video (clips of one person never split across train/test)
  source : id of the original real video (a real clip and the fakes made from it share this id)

Usage:
  python train.py --manifest data/manifest.csv --method green --group-by person
  python train.py --manifest data/manifest.csv --compress-crf 28     # robustness check
"""
from __future__ import annotations
import argparse, os, subprocess, tempfile
import joblib
import pandas as pd
from tqdm import tqdm

from veritas.config import Config
from veritas import roi, signals as S, features as F
from veritas.classify import evaluate, fit_final, MODELS


def compress(src, crf):
    out = os.path.join(tempfile.mkdtemp(), "c.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-crf", str(crf), out], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--method", default="green", choices=["green", "chrom", "pos"])
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "svm"])
    ap.add_argument("--group-by", default="person", choices=["person", "source"])
    ap.add_argument("--window", type=float, default=10.0, help="window length in s (use 5 for ~11 s clips)")
    ap.add_argument("--stride", type=float, default=5.0, help="stride in s (use 2.5 with --window 5)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--compress-crf", type=int, default=None, help="re-encode every video with H.264 at this CRF first")
    ap.add_argument("--save-model", default="best", choices=["best", "A_global", "B_local", "C_combined"],
                    help="which model to save; A_global is the baseline (global heart-rate features only)")
    ap.add_argument("--out", default="models/veritas_model.joblib")
    a = ap.parse_args()

    cfg = Config(method=a.method, max_frames=a.max_frames, window_sec=a.window, stride_sec=a.stride)
    if not os.path.exists(a.manifest):
        raise SystemExit(f"Manifest not found: {a.manifest}")
    man = pd.read_csv(a.manifest)
    if len(man) == 0:
        raise SystemExit("Manifest is empty - no videos to train on.")
    man["label"] = man["label"].map(lambda x: 1 if str(x).lower() in ("fake", "1") else 0)
    rows = []
    for _, r in tqdm(man.iterrows(), total=len(man), desc="extracting"):
        path = compress(r.path, a.compress_crf) if a.compress_crf else r.path
        try:
            tr, fps, valid, _, _ = roi.extract_rgb_traces(path, cfg.max_frames)
            if valid.mean() < 0.5: print("skip (face lost):", r.path); continue
            sigs = S.process_regions(tr, fps, cfg)
            for wf in F.video_features(sigs, fps, cfg):
                d = {k: v for k, v in wf.items() if not k.startswith("_")}
                d.update(video=r.path, label=r.label, group=r[a.group_by])
                rows.append(d)
        except Exception as e:
            print("skip", r.path, e)
    df = pd.DataFrame(rows)
    if df.empty or df.label.nunique() < 2:
        raise SystemExit("Need windows from both real AND fake videos to train "
                         f"(got {len(df)} windows). Check the skip messages above.")
    df.to_csv("features.csv", index=False)
    print(f"\n{len(df)} windows from {df.video.nunique()} videos")
    res = evaluate(df, a.classifier)
    print("\n=== Hypothesis test (GroupKFold by %s) ===" % a.group_by)
    print(res.round(3).to_string(index=False))
    res.to_csv("results.csv", index=False)

    best = (res.sort_values("video_auc", ascending=False).iloc[0].model
            if a.save_model == "best" else a.save_model)
    cols = MODELS[best]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump({"model": fit_final(df, cols, a.classifier), "cols": cols, "name": best, "cfg": cfg}, a.out)
    print(f"\nSaved '{best}' -> {a.out}  (report all three models regardless)")


if __name__ == "__main__":
    main()
