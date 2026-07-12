"""Label-efficiency experiment (no retraining) — the test where CEBRA should shine.

Idea: CEBRA learns its representation from the *whole unlabeled* stream. So if we
then train a decoder on only a *small fraction* of labels, a good self-supervised
embedding should reach high accuracy with fewer labels than raw features.

For each representation (raw high-gamma, CEBRA-Time, CEBRA-Behavior) we:
  - hold out the last 30% of time as a fixed test set,
  - draw a **stratified** subsample of the first 70% as the labeled training set,
    sweeping the fraction from a few % up to 100%,
  - train a decoder (kNN for embeddings, logistic for raw) and score test AUC,
  - average over several random seeds,
and plot AUC vs number of labeled samples, for two targets:
  - reach (annotated, sparse)            - movement vs rest (top-25% wrist speed).

Usage:
    python cebra_label_efficiency.py --dir cebra_out
"""
import argparse
import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

FRACTIONS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def stratified_subsample(y, frac, rng):
    """Indices for a class-balanced fraction of the pool (>=1 per present class)."""
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        m = max(1, int(round(len(ci) * frac)))
        idx.append(rng.choice(ci, size=min(m, len(ci)), replace=False))
    return np.concatenate(idx)


def curve(Z, y, k_test, raw):
    """AUC vs fraction for one representation/target. Returns (n_labels, aucs)."""
    tr_pool = np.arange(k_test)
    te = np.arange(k_test, len(Z))
    ytr_pool, yte = y[tr_pool], y[te]
    if len(np.unique(yte)) < 2:
        return None
    n_labels, aucs = [], []
    for frac in FRACTIONS:
        seed_auc, n_used = [], 0
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            sub = tr_pool[stratified_subsample(ytr_pool, frac, rng)]
            if len(np.unique(y[sub])) < 2:
                continue
            if raw:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            else:
                clf = KNeighborsClassifier(n_neighbors=min(25, max(1, len(sub) // 2)))
            clf.fit(Z[sub], y[sub])
            seed_auc.append(roc_auc_score(yte, clf.predict_proba(Z[te])[:, 1]))
            n_used = len(sub)
        if seed_auc:
            n_labels.append(n_used)
            aucs.append(np.mean(seed_auc))
    return np.array(n_labels), np.array(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cebra_out")
    args = ap.parse_args()

    emb = np.load(os.path.join(args.dir, "embeddings.npz"), allow_pickle=True)
    et, eb = emb["emb_time"], emb["emb_beh"]
    reach = emb["reach"].astype(int)
    spd = emb["speed"][:, 0].astype(float)
    n = len(et)
    k_test = int(n * 0.7)

    reps = [("CEBRA-Time", et, False), ("CEBRA-Behavior", eb, False)]
    for sp in glob.glob(os.path.join(args.dir, "stream_*.npz")):
        Xs = np.load(sp, allow_pickle=True)["X"].astype(np.float32)
        if len(Xs) == n:
            reps.insert(0, ("raw high-gamma", Xs, True))
            break

    thr = np.percentile(spd[:k_test], 75)
    targets = {"reach": reach, "move (top-25% speed)": (spd > thr).astype(int)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    rows = []
    for ax, (tname, y) in zip(axes, targets.items()):
        for name, Z, raw in reps:
            res = curve(Z, y, k_test, raw)
            if res is None:
                continue
            nl, au = res
            ax.plot(nl, au, marker="o", label=name)
            for a, b in zip(nl, au):
                rows.append("{},{},{},{:.4f}".format(tname, name, a, b))
        ax.set_xscale("log")
        ax.set_xlabel("# labeled training samples")
        ax.set_ylabel("test AUC")
        ax.set_title("Label efficiency — target: {}".format(tname))
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Does CEBRA reach accuracy with fewer labels? (test = last 30% of time)")
    fig.tight_layout()
    out_png = os.path.join(args.dir, "label_efficiency.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    with open(os.path.join(args.dir, "label_efficiency.csv"), "w") as fh:
        fh.write("target,representation,n_labels,test_auc\n")
        fh.write("\n".join(rows) + "\n")

    # console summary
    print("target,representation,n_labels,test_auc")
    for r in rows:
        print(r)
    print("\nWrote", out_png)


if __name__ == "__main__":
    main()
