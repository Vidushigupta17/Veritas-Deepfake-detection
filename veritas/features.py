"""Step 4: local consistency features per window + global baseline features."""
from __future__ import annotations
import numpy as np
from scipy import signal as sg

from .config import Config, REGIONS, PAIRS
from . import signals as S

GLOBAL_COLS = ["g_hr", "g_snr", "g_hr_std_regions"]
LOCAL_COLS = (["coh_mean", "coh_peak", "coh_min", "lag_abs_mean", "lag_std", "lag_max",
               "snr_std", "snr_range", "snr_min", "hr_range"])


def _pair_stats(a, b, fps, pk, cfg):
    nper = int(min(len(a), max(fps * 4, 32)))
    f, coh = sg.coherence(a, b, fs=fps, nperseg=nper, noverlap=nper // 2)
    band = (f >= cfg.low_hz) & (f <= cfg.high_hz)
    if not band.any():
        return 0.0, 0.0, 0.0
    c_mean = coh[band].mean()
    i = np.argmin(np.abs(f - pk))
    c_peak = coh[i]
    # phase lag at the dominant frequency via cross-spectral density
    fc, Pxy = sg.csd(a, b, fs=fps, nperseg=nper, noverlap=nper // 2)
    phase = np.angle(Pxy[np.argmin(np.abs(fc - pk))])
    lag_ms = phase / (2 * np.pi * pk) * 1000.0
    return c_mean, c_peak, lag_ms


def window_features(win, fps, cfg: Config):
    """win{region:(L,)} cleaned signals of one window -> dict of features."""
    hr, snr = {}, {}
    for r in REGIONS:
        hr[r], _, _, _ = S.fft_hr(win[r], fps, cfg.low_hz, cfg.high_hz)
        snr[r] = S.snr_db(win[r], fps, cfg.low_hz, cfg.high_hz)
    # global: average of the 3 region signals
    g = np.mean([win[r] for r in REGIONS], axis=0)
    g_hr, _, _, g_pk = S.fft_hr(g, fps, cfg.low_hz, cfg.high_hz)
    feats = {"g_hr": g_hr, "g_snr": S.snr_db(g, fps, cfg.low_hz, cfg.high_hz),
             "g_hr_std_regions": float(np.std(list(hr.values())))}
    cm, cp, lags = [], [], []
    for a, b in PAIRS:
        c1, c2, lg = _pair_stats(win[a], win[b], fps, g_pk, cfg)
        cm.append(c1); cp.append(c2); lags.append(lg)
    sv = np.array(list(snr.values())); hv = np.array(list(hr.values()))
    feats.update({
        "coh_mean": float(np.mean(cm)), "coh_peak": float(np.mean(cp)), "coh_min": float(np.min(cp)),
        "lag_abs_mean": float(np.mean(np.abs(lags))), "lag_std": float(np.std(lags)),
        "lag_max": float(np.max(np.abs(lags))),
        "snr_std": float(sv.std()), "snr_range": float(sv.max() - sv.min()), "snr_min": float(sv.min()),
        "hr_range": float(hv.max() - hv.min()),
    })
    # per-pair / per-region detail kept for explainability (not used by the classifier)
    feats["_pair_coh"] = dict(zip(["F-L", "F-R", "L-R"], cp))
    feats["_region_snr"] = snr
    feats["_region_hr"] = hr
    return feats


def video_features(sigs, fps, cfg: Config):
    """Slide windows over cleaned region signals -> list of per-window feature dicts (+ time)."""
    T = len(next(iter(sigs.values())))
    L, step = int(cfg.window_sec * fps), int(cfg.stride_sec * fps)
    out = []
    for s in range(0, T - L + 1, max(step, 1)):
        w = {r: v[s:s + L] for r, v in sigs.items()}
        # re-normalise inside window
        w = {r: (x - x.mean()) / (x.std() + 1e-9) for r, x in w.items()}
        ft = window_features(w, fps, cfg)
        ft["t_start"] = s / fps
        out.append(ft)
    return out


def to_matrix(win_feats, cols):
    return np.array([[wf[c] for c in cols] for wf in win_feats], dtype=float)
