"""Decisive test for the focused direction: does an interpretable arousal biomarker
transfer to a NEW patient without retraining?

The cross-subject geometry test failed, but the biomarker's alpha/beta spectral signature
was shared across all three patients. So we test transfer at the *feature* level, using
channel-agnostic spectral summaries (per-band statistics pooled over whatever electrodes a
patient has) -- identical feature layout for every patient regardless of montage. Then
leave-one-patient-out: train the sleep-vs-active readout on 2 patients, test on the third.

If LOSO AUC clears chance (and approaches the within-patient ~0.97), that is first evidence
of a patient-agnostic arousal biomarker -- the novel, focused result.

Run (dbs-ml env):
  python phase4_biomarker_transfer.py
"""

import argparse
import os

import numpy as np

from nwb_dataset import good_channel_indices
from phase3_eval import build_epoch_dataset, find_default_file
from phase3_crosssubject import find_subject_files
from phase3_manifold import STATE_BANDS, T5_LABELS, drop_outliers

PCTS = (10, 25, 50, 75, 90)


def spectral_summary(X, n_ch, n_bd):
    """(windows, n_ch*n_bd) log band-power -> channel-agnostic per-band summary features.
    For each band: mean, std, and percentiles across channels. Same layout for every
    patient regardless of electrode count."""
    T = X.shape[0]
    cube = X.reshape(T, n_ch, n_bd)
    feats, names = [], []
    for bi, b in enumerate(STATE_BANDS):
        band = cube[:, :, bi]  # (T, n_ch)
        feats.append(band.mean(axis=1)); names.append(b + "_mean")
        feats.append(band.std(axis=1)); names.append(b + "_std")
        for p in PCTS:
            feats.append(np.percentile(band, p, axis=1)); names.append("{}_p{}".format(b, p))
    return np.column_stack(feats), names


def load_patient(path):
    with __import__("h5py").File(path, "r") as f:
        good = good_channel_indices(f)
    X, y, _ = build_epoch_dataset(path, good, STATE_BANDS, window_sec=10.0,
                                  max_per_label=150, label_set=set(T5_LABELS),
                                  seed=0, verbose=False)
    keep = drop_outliers(X)
    X, y = X[keep], np.asarray(y)[keep]
    yb = (y == "Sleep/rest").astype(int)
    if yb.sum() < 10 or (yb == 0).sum() < 10:
        return None
    feats, names = spectral_summary(X, len(good), len(STATE_BANDS))
    return feats, yb, names


def within_auc(F, y, seed=0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(F)); cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(F[tr])
    lda = LinearDiscriminantAnalysis().fit(sc.transform(F[tr]), y[tr])
    return float(roc_auc_score(y[te], lda.decision_function(sc.transform(F[te]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--extra-root", default="ajile12-nwb-data")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    files = find_subject_files(args.extra_root, args.nwb or find_default_file())
    data = {}
    for sid, p in files.items():
        print("[load] sub-{}".format(sid))
        d = load_patient(p)
        if d is not None:
            data[sid] = d
            print("  {} windows, sleep frac {:.2f}, {} features".format(
                len(d[1]), d[1].mean(), d[0].shape[1]))
        else:
            print("  excluded (needs both sleep and active windows)")

    subs = list(data)
    if len(subs) < 3:
        raise SystemExit("Need >=3 usable patients for leave-one-patient-out.")

    print("\n=== within-patient (upper bound) ===")
    for s in subs:
        print("  sub-{}: AUC {:.3f}".format(s, within_auc(*data[s][:2])))

    print("\n=== LEAVE-ONE-PATIENT-OUT transfer (train on others, test on held-out) ===")
    loso = []
    for held in subs:
        Ftr = np.vstack([data[s][0] for s in subs if s != held])
        ytr = np.concatenate([data[s][1] for s in subs if s != held])
        Fte, yte = data[held][0], data[held][1]
        sc = StandardScaler().fit(Ftr)
        lda = LinearDiscriminantAnalysis().fit(sc.transform(Ftr), ytr)
        auc = float(roc_auc_score(yte, lda.decision_function(sc.transform(Fte))))
        loso.append(auc)
        print("  train on {:<12s} -> test sub-{}: AUC {:.3f}".format(
            ",".join(s for s in subs if s != held), held, auc))

    print("\n================ TRANSFER VERDICT ================")
    print("mean LOSO transfer AUC = {:.3f}  (chance 0.5; within-patient ~0.97)".format(np.mean(loso)))
    m = np.mean(loso)
    verdict = ("TRANSFERS (well above chance)" if m > 0.75
               else "PARTIAL (above chance, below within-patient)" if m > 0.6
               else "DOES NOT TRANSFER")
    print("verdict: {}".format(verdict))
    print("==================================================")


if __name__ == "__main__":
    main()
