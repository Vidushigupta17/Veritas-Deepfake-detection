"""PILOT / PLUMBING TEST ONLY - this is NOT a deepfake detector.

UBFC contains only REAL videos, so a real/fake classifier can't be learned from subjects 1, 3, 4 alone.
This script makes SYNTHETIC 'fakes' from them: it takes a real video and replaces ONE face region's pulse
with the same region from a DIFFERENT person (a 'spliced pulse'). That breaks cross-region consistency by
construction, so any classifier will find it easy. It only proves that the pipeline runs end to end:
    video -> regions -> signals -> features -> A/B/C models -> evaluation -> saved model.

Do NOT report these scores as detection accuracy. Real evidence needs real deepfakes (Celeb-DF, or
face-swaps you make yourself) via train.py.

Usage:  python train_pilot.py --root C:\\data\\UBFC --method pos --max-frames 1800
"""
from __future__ import annotations
import argparse, glob, os
import joblib
import pandas as pd

from veritas.config import Config, REGIONS
from veritas import roi, signals as S, features as F
from veritas.classify import evaluate, fit_final, MODELS

CACHE = "models/ubfc_traces.joblib"


def load_traces(root, max_frames):
    if os.path.exists(CACHE):
        c = joblib.load(CACHE)
        if c["max_frames"] == max_frames:
            print(f"Using cached traces ({CACHE})"); return c["traces"], c["fps"]
    traces, fps = {}, {}
    vids = sorted(glob.glob(os.path.join(root, "**", "vid.avi"), recursive=True))
    if not vids: raise SystemExit(f"No vid.avi under {root}")
    for v in vids:
        name = os.path.basename(os.path.dirname(v))
        print(f"{name}: tracking face...", flush=True)
        tr, f, valid, _, _ = roi.extract_rgb_traces(v, max_frames)
        if valid.mean() < 0.5: print("  skipped (face lost)"); continue
        traces[name], fps[name] = tr, f
    os.makedirs("models", exist_ok=True)
    joblib.dump({"traces": traces, "fps": fps, "max_frames": max_frames}, CACHE)
    return traces, fps


def window_rows(traces, fps, cfg, video, label, group):
    sigs = S.process_regions(traces, fps, cfg)
    rows = []
    for wf in F.video_features(sigs, fps, cfg):
        d = {k: v for k, v in wf.items() if not k.startswith("_")}
        d.update(video=video, label=label, group=group); rows.append(d)
    return rows


def build_pilot_dataset(traces, fps, cfg):
    names = sorted(traces)
    rows = []
    for i, host in enumerate(names):
        donor = names[(i + 1) % len(names)]
        T = min(len(traces[host]["forehead"]), len(traces[donor]["forehead"]))
        f = fps[host]
        real = {r: traces[host][r][:T] for r in REGIONS}
        rows += window_rows(real, f, cfg, f"{host}_real", 0, host)
        for region in REGIONS:                       # splice the donor's pulse into one region
            fake = dict(real); fake[region] = traces[donor][region][:T]
            rows += window_rows(fake, f, cfg, f"{host}_fake_{region}_from_{donor}", 1, host)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--method", default="pos", choices=["green", "chrom", "pos"])
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "svm"])
    ap.add_argument("--max-frames", type=int, default=1800)
    ap.add_argument("--out", default="models/pilot_model.joblib")
    a = ap.parse_args()
    cfg = Config(method=a.method)
    traces, fps = load_traces(a.root, a.max_frames)
    if len(traces) < 3: raise SystemExit("Need at least 3 subjects for leave-one-subject-out evaluation.")
    df = build_pilot_dataset(traces, fps, cfg)
    print(f"\n{len(df)} windows | real videos: {(df.drop_duplicates('video').label==0).sum()} "
          f"| synthetic fakes: {(df.drop_duplicates('video').label==1).sum()}")
    res = evaluate(df, a.classifier)
    print("\n=== PILOT RESULT (synthetic fakes, leave-one-subject-out) - plumbing check, NOT detection accuracy ===")
    print(res.round(3).to_string(index=False))
    best = res.sort_values("video_auc", ascending=False).iloc[0].model
    joblib.dump({"model": fit_final(df, MODELS[best], a.classifier), "cols": MODELS[best],
                 "name": f"PILOT-{best}", "cfg": cfg}, a.out)
    print(f"\nSaved {a.out}. The app only loads models/veritas_model.joblib, so it ignores this file "
          "unless you rename it. A verdict from a pilot model is NOT a real detection.")

if __name__ == "__main__":
    main()
