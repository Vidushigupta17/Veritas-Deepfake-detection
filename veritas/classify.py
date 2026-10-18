"""Steps 5-7: classifier, 3-model hypothesis test, honest evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .features import GLOBAL_COLS, LOCAL_COLS

MODELS = {"A_global": GLOBAL_COLS, "B_local": LOCAL_COLS, "C_combined": GLOBAL_COLS + LOCAL_COLS}


def make_clf(kind="logreg"):
    clf = (LogisticRegression(max_iter=2000, class_weight="balanced") if kind == "logreg"
           else SVC(kernel="rbf", probability=True, class_weight="balanced"))
    return make_pipeline(StandardScaler(), clf)


def majority_vote(preds):
    """1 = fake, 0 = real. Ties -> fake (conservative)."""
    preds = np.asarray(preds)
    return int(preds.mean() >= 0.5)


def evaluate(df, kind="logreg", n_splits=5):
    """df: one row per WINDOW with columns: video, label(0 real/1 fake), group, + feature cols.
    GroupKFold keeps a group (person / source video) entirely in train OR test -> no leakage."""
    groups = df["group"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for name, cols in MODELS.items():
        X, y = df[cols].values, df["label"].values
        win_pred = np.zeros(len(df), int); win_prob = np.zeros(len(df))
        for tr, te in gkf.split(X, y, groups):
            m = make_clf(kind).fit(X[tr], y[tr])
            win_pred[te] = m.predict(X[te]); win_prob[te] = m.predict_proba(X[te])[:, 1]
        d = df.assign(pred=win_pred, prob=win_prob)
        v = d.groupby("video").agg(label=("label", "first"), pred=("pred", majority_vote), prob=("prob", "mean"))
        rows.append({
            "model": name, "n_windows": len(d), "n_videos": len(v),
            "window_acc": accuracy_score(d.label, d.pred),
            "video_acc": accuracy_score(v.label, v.pred),
            "video_f1": f1_score(v.label, v.pred, zero_division=0),
            "video_auc": roc_auc_score(v.label, v.prob) if v.label.nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def fit_final(df, cols, kind="logreg"):
    return make_clf(kind).fit(df[cols].values, df["label"].values)
