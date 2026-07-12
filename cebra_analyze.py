"""Diagnostic evaluation of cached CEBRA embeddings (no retraining).

Compares two cross-validation schemes for decoding reach / movement / wrist-speed
from each representation (raw high-gamma, CEBRA-Time, CEBRA-Behavior):

  * single split  : train on first 70% of time, test on last 30%
                    (sensitive to non-stationarity / electrode drift)
  * blocked 5-fold: hold out each contiguous 1/5 of time in turn, average
                    (isolates "is the signal decodable?" from "does it drift?")

If blocked-CV scores are good but the single split is poor, the limitation is
temporal drift, not absence of signal.

Usage:
    python cebra_analyze.py --dir cebra_out
"""
import argparse
import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def blocked_folds(n, k=5):
    b = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        te = np.zeros(n, bool)
        te[b[i]:b[i + 1]] = True
        yield ~te, te


def single_split(n, frac=0.7):
    k = int(n * frac)
    tr = np.zeros(n, bool); tr[:k] = True
    return [(tr, ~tr)]


def _auc(Z, y, folds, raw):
    out = []
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = (LogisticRegression(max_iter=1000, class_weight="balanced") if raw
               else KNeighborsClassifier(n_neighbors=25))
        clf.fit(Z[tr], y[tr])
        out.append(roc_auc_score(y[te], clf.predict_proba(Z[te])[:, 1]))
    return np.nanmean(out) if out else float("nan")


def _r2(Z, spd, folds, raw):
    out = []
    for tr, te in folds:
        m = Ridge(alpha=1.0) if raw else KNeighborsRegressor(n_neighbors=25)
        m.fit(Z[tr], spd[tr])
        out.append(r2_score(spd[te], m.predict(Z[te])))
    return np.nanmean(out) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cebra_out")
    args = ap.parse_args()

    emb = np.load(os.path.join(args.dir, "embeddings.npz"), allow_pickle=True)
    et, eb = emb["emb_time"], emb["emb_beh"]
    reach = emb["reach"].astype(int)
    spd = emb["speed"][:, 0].astype(float)
    n = len(et)

    # Match the raw high-gamma stream to the embedding length (files in a dir may
    # be from different-length runs).
    reps = [("CEBRA-Time", et, False), ("CEBRA-Behavior", eb, False)]
    for sp in glob.glob(os.path.join(args.dir, "stream_*.npz")):
        Xs = np.load(sp, allow_pickle=True)["X"].astype(np.float32)
        if len(Xs) == n:
            reps.insert(0, ("raw high-gamma", Xs, True))
            break
    else:
        print("(no matching raw stream found; comparing embeddings only)")
    thr = np.percentile(spd, 75)
    moving = (spd > thr).astype(int)

    print("dir={}  T={}  reach_frac={:.3f}  moving_frac={:.3f}\n".format(
        args.dir, n, reach.mean(), moving.mean()))

    for scheme, folds in [("single split (train early/test late)", single_split(n)),
                          ("blocked 5-fold CV", list(blocked_folds(n, 5)))]:
        print("== {} ==".format(scheme))
        print("{:16s} | reach AUC | move AUC | speed R^2".format("representation"))
        print("-" * 56)
        for name, Z, raw in reps:
            ra = _auc(Z, reach, folds, raw)
            ma = _auc(Z, moving, folds, raw)
            r2 = _r2(Z, spd, folds, raw)
            print("{:16s} |   {:.3f}   |  {:.3f}   |  {:+.3f}".format(name, ra, ma, r2))
        print()


if __name__ == "__main__":
    main()
