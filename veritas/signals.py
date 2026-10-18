"""Steps 2-3: signal extraction, cleaning, FFT heart-rate."""
from __future__ import annotations
import numpy as np
from scipy import signal as sg

from .config import Config


def fill_nans(x):
    """Linear-interpolate NaN rows (frames with no face). x: (T,) or (T,C)."""
    x = np.array(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]; squeeze = True
    else:
        squeeze = False
    t = np.arange(len(x))
    for c in range(x.shape[1]):
        bad = np.isnan(x[:, c])
        if bad.all():
            x[:, c] = 0
        elif bad.any():
            x[bad, c] = np.interp(t[bad], t[~bad], x[~bad, c])
    return x[:, 0] if squeeze else x


# ---- Step 2: RGB trace -> 1D pulse signal ----
def green(rgb):
    return rgb[:, 1]


def chrom(rgb, fps, win_sec=1.6):
    """De Haan & Jeanne (2013) CHROM, overlap-added in short windows."""
    T = len(rgb)
    L = max(int(win_sec * fps), 8)
    out = np.zeros(T); wsum = np.zeros(T)
    hann = np.hanning(L)
    for s in range(0, max(T - L, 0) + 1, max(L // 2, 1)):
        seg = rgb[s:s + L]
        if len(seg) < L: break
        n = seg / (seg.mean(axis=0) + 1e-9)
        X = 3 * n[:, 0] - 2 * n[:, 1]
        Y = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
        alpha = X.std() / (Y.std() + 1e-9)
        h = X - alpha * Y
        h = (h - h.mean()) * hann
        out[s:s + L] += h; wsum[s:s + L] += hann
    return out / np.maximum(wsum, 1e-9)


def pos(rgb, fps, win_sec=1.6):
    """Wang et al. (2017) POS, overlap-added."""
    T = len(rgb)
    L = max(int(win_sec * fps), 8)
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)
    out = np.zeros(T)
    for s in range(0, T - L + 1):
        seg = rgb[s:s + L]
        n = seg / (seg.mean(axis=0) + 1e-9)
        S = P @ n.T
        h = S[0] + (S[0].std() / (S[1].std() + 1e-9)) * S[1]
        out[s:s + L] += h - h.mean()
    return out


def extract_pulse(rgb, fps, method="green"):
    rgb = fill_nans(rgb)
    if method == "chrom": return chrom(rgb, fps)
    if method == "pos": return pos(rgb, fps)
    return green(rgb)


# ---- Step 3: cleaning ----
def clean(x, fps, low=0.7, high=3.0, order=3):
    x = sg.detrend(np.asarray(x, float))
    nyq = fps / 2
    high = min(high, nyq * 0.95)
    b, a = sg.butter(order, [low / nyq, high / nyq], btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x - x.mean()
    x = sg.filtfilt(b, a, x)
    return (x - x.mean()) / (x.std() + 1e-9)


def fft_hr(x, fps, low=0.7, high=3.0, pad=8):
    """Dominant frequency in band -> (bpm, freqs, power, peak_hz)."""
    n = len(x) * pad
    f = np.fft.rfftfreq(n, 1 / fps)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    band = (f >= low) & (f <= high)
    pk = f[band][np.argmax(p[band])]
    return pk * 60.0, f, p, pk       # BPM = Hz * 60


def snr_db(x, fps, low=0.7, high=3.0, tol=0.1):
    """Pulse SNR: power near the peak (+2nd harmonic) vs the rest of the band."""
    _, f, p, pk = fft_hr(x, fps, low, high)
    band = (f >= low) & (f <= high)
    sig = ((np.abs(f - pk) <= tol) | (np.abs(f - 2 * pk) <= tol)) & band
    s, n = p[sig].sum(), p[band & ~sig].sum()
    return 10 * np.log10((s + 1e-12) / (n + 1e-12))


def process_regions(traces, fps, cfg: Config):
    """traces{region:(T,3)} -> cleaned 1D signals{region:(T,)}"""
    return {r: clean(extract_pulse(v, fps, cfg.method), fps, cfg.low_hz, cfg.high_hz)
            for r, v in traces.items()}


def mae(est_bpm, ref_bpm):
    return float(np.mean(np.abs(np.asarray(est_bpm) - np.asarray(ref_bpm))))
