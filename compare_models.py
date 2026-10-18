"""Fair comparison of A (baseline), B and C with uncertainty. Run AFTER train.py (needs features.csv).

For small datasets a single accuracy number is misleading. This prints video-level AUC with 95% bootstrap
confidence intervals, and the paired difference B-A and C-A. If a difference's interval includes 0, you
can NOT claim the model is better. Report whatever comes out.

Usage: python compare_models.py --features features.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from veritas.classify import MODELS, make_clf


def video_probs(df, cols, kind, n_splits):
    X, y, g = df[cols].values, df["label"].values, df["group"].values
    prob = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        prob[te] = make_clf(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    v = df.assign(p=prob).groupby("video").agg(label=("label", "first"), p=("p", "mean"))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "svm"])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    df = pd.read_csv(a.features)
    n_splits = min(5, df["group"].nunique())
    vids = {n: video_probs(df, c, a.classifier, n_splits) for n, c in MODELS.items()}
    names = list(vids); idx = vids[names[0]].index
    y = vids[names[0]].loc[idx, "label"].values
    P = {n: vids[n].loc[idx, "p"].values for n in names}
    rng = np.random.default_rng(0); N = len(idx)
    aucs = {n: [] for n in names}; diffs = {n: [] for n in names if n != "A_global"}
    for _ in range(a.boot):
        s = rng.integers(0, N, N)
        if len(np.unique(y[s])) < 2: continue
        r = {n: roc_auc_score(y[s], P[n][s]) for n in names}
        for n in names: aucs[n].append(r[n])
        for n in diffs: diffs[n].append(r[n] - r["A_global"])
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    print(f"\n{N} videos ({int(y.sum())} fake / {int((1-y).sum())} real), group-wise CV\n")
    print("Video-level AUC (95% bootstrap CI)")
    for n in names:
        lo, hi = ci(aucs[n]); print(f"  {n:11s} {roc_auc_score(y, P[n]):.3f}  [{lo:.3f}, {hi:.3f}]")
    print("\nDifference vs baseline A (95% CI)")
    for n in diffs:
        lo, hi = ci(diffs[n]); mean = roc_auc_score(y, P[n]) - roc_auc_score(y, P["A_global"])
        verdict = "interval excludes 0 -> evidence of improvement" if lo > 0 else \
                  "interval includes 0 -> NOT shown to be better"
        print(f"  {n:11s} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")

if __name__ == "__main__":
    main()
