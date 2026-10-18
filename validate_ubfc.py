"""Step 3 validation: rPPG heart rate vs UBFC-rPPG ground truth (MAE in BPM).

Expects UBFC folders each containing vid.avi + ground_truth.txt
(rows: PPG signal, heart rate, timestamps).
Usage: python validate_ubfc.py --root /path/to/UBFC_DATASET --method green
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
from veritas.config import Config, REGIONS
from veritas import roi, signals as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--method", default="green", choices=["green", "chrom", "pos"])
    ap.add_argument("--max-frames", type=int, default=1800)
    a = ap.parse_args()
    cfg = Config(method=a.method)
    errs = []
    vids = sorted(glob.glob(os.path.join(a.root, "**", "vid.avi"), recursive=True))
    print(f"Found {len(vids)} vid.avi under {a.root}")
    if not vids:
        print("Nothing to do. Check --root: it must contain subject folders with vid.avi + ground_truth.txt "
              "(search is recursive). Names are case-sensitive on some systems.")
        return
    for vid in vids:
        name = os.path.basename(os.path.dirname(vid))
        gt = os.path.join(os.path.dirname(vid), "ground_truth.txt")
        if not os.path.exists(gt):
            print(f"{name}: SKIP, no ground_truth.txt next to vid.avi"); continue
        try:
            rows = [np.array(l.split(), float) for l in open(gt) if l.strip()]
            hr_gt, t_gt = rows[1], rows[2]
            print(f"{name}: tracking face (this takes a few minutes)...", flush=True)
            last = [-1]
            def prog(p):
                k = int(p * 10)
                if k != last[0]:
                    last[0] = k; print(f"   {k*10}%", end="", flush=True)
            tr, fps, valid, _, _ = roi.extract_rgb_traces(vid, a.max_frames, progress=prog)
            print()
            if valid.mean() < 0.5:
                print(f"{name}: SKIP, face found in only {valid.mean()*100:.0f}% of frames"); continue
            sig = S.process_regions(tr, fps, cfg)
            g = np.mean([sig[r] for r in REGIONS], axis=0)
            L, step = int(cfg.window_sec * fps), int(cfg.stride_sec * fps)
            if len(g) < L:
                print(f"{name}: SKIP, only {len(g)/fps:.1f}s processed, need >= {cfg.window_sec}s "
                      f"(raise --max-frames)"); continue
            est, ref = [], []
            for s0 in range(0, len(g) - L + 1, step):
                est.append(S.fft_hr(g[s0:s0 + L], fps)[0])
                m = (t_gt >= s0 / fps) & (t_gt < (s0 + L) / fps)
                ref.append(hr_gt[m].mean() if m.any() else np.nan)
            est, ref = np.array(est), np.array(ref); ok = ~np.isnan(ref)
            if not ok.any():
                print(f"{name}: SKIP, ground-truth timestamps do not overlap windows"); continue
            e = S.mae(est[ok], ref[ok]); errs.append(e)
            print(f"{name}: MAE {e:.2f} BPM  (est {est[ok].mean():.1f} vs ref {ref[ok].mean():.1f}, {ok.sum()} windows)")
        except Exception as ex:
            print(f"{name}: ERROR {type(ex).__name__}: {ex}")
    if errs: print(f"\nOverall MAE ({a.method}): {np.mean(errs):.2f} BPM over {len(errs)} videos")
    else: print("\nNo video produced a result - see messages above.")


if __name__ == "__main__":
    main()
