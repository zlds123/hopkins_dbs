"""Does the arousal biomarker transfer across patients on a MULTI-STATE target, not just
binary sleep-vs-active?

The binary transfer result (mean LOSO AUC 0.72) could be dismissed as a sleep detector.
This tests the stronger claim: train a multi-class state readout on channel-agnostic
spectral-summary features from N-1 patients and decode the held-out patient's behavioral
STATE (up to 6 classes), scored by one-vs-rest macro-AUC.

Patients do not share all states (e.g. sub-07 has no TV), so transfer is only defined over
states present in every participating patient. The script reports the shared-state set it
used, the leave-one-patient-out macro-AUC, and, because the three local patients differ in
coverage, a richer pairwise transfer for the two patients that share the most states.

Run (dbs-ml env):
  python phase4_multistate_transfer.py
"""

import argparse
import os

import numpy as np

from nwb_dataset import good_channel_indices
from phase3_eval import build_epoch_dataset, find_default_file
from phase3_crosssubject import find_subject_files
from phase3_manifold import STATE_BANDS, T5_LABELS, drop_outliers
from phase4_biomarker_transfer import spectral_summary

MIN_PER_STATE = 15


def load(path):
    with __import__("h5py").File(path, "r") as f:
        good = good_channel_indices(f)
    X, y, _ = build_epoch_dataset(path, good, STATE_BANDS, window_sec=10.0,
                                  max_per_label=150, label_set=set(T5_LABELS),
                                  seed=0, verbose=False)
    keep = drop_outliers(X)
    X, y = X[keep], np.asarray(y)[keep]
    feats, _ = spectral_summary(X, len(good), len(STATE_BANDS))
    return feats, y


def well_sampled(y, min_n=MIN_PER_STATE):
    return {s for s in T5_LABELS if np.sum(y == s) >= min_n}


def macro_auc(train_feats, train_y, test_feats, test_y, states):
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    states = sorted(states)
    mtr, mte = np.isin(train_y, states), np.isin(test_y, states)
    if len(set(test_y[mte])) < len(states) or len(set(train_y[mtr])) < len(states):
        return float("nan")
    sc = StandardScaler().fit(train_feats[mtr])
    lda = LinearDiscriminantAnalysis().fit(sc.transform(train_feats[mtr]), train_y[mtr])
    order = list(lda.classes_)
    proba = lda.predict_proba(sc.transform(test_feats[mte]))
    proba = proba[:, [order.index(s) for s in states]]
    return float(roc_auc_score(test_y[mte], proba, multi_class="ovr", average="macro", labels=states))


def within(feats, y, states, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); cut = int(0.7 * len(idx))
    return macro_auc(feats[idx[:cut]], y[idx[:cut]], feats[idx[cut:]], y[idx[cut:]], states)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--extra-root", default="ajile12-nwb-data")
    args = ap.parse_args()

    files = find_subject_files(args.extra_root, args.nwb or find_default_file())
    data, states_by = {}, {}
    for sid, p in files.items():
        print("[load] sub-{}".format(sid))
        feats, y = load(p)
        ws = well_sampled(y)
        if len(ws) >= 3:
            data[sid] = (feats, y)
            states_by[sid] = ws
            print("  well-sampled states ({}+ windows): {}".format(MIN_PER_STATE, sorted(ws)))
        else:
            print("  <3 well-sampled states -> excluded (insufficient for multi-class transfer)")

    subs = list(data)

    # ---- leave-one-patient-out over states shared by ALL patients ---------- #
    common_all = set.intersection(*[states_by[s] for s in subs]) if subs else set()
    print("\n=== states shared across ALL {} patients: {} ({} classes) ===".format(
        len(subs), sorted(common_all), len(common_all)))
    if len(common_all) >= 2 and len(subs) >= 3:
        st = sorted(common_all)
        print("within-patient macro-AUC (upper bound):")
        for s in subs:
            print("  sub-{}: {:.3f}".format(s, within(*data[s], st)))
        print("LEAVE-ONE-PATIENT-OUT ({}-way {}):".format(len(st), st))
        aucs = []
        for held in subs:
            Ftr = np.vstack([data[s][0] for s in subs if s != held])
            ytr = np.concatenate([data[s][1] for s in subs if s != held])
            a = macro_auc(Ftr, ytr, data[held][0], data[held][1], st)
            aucs.append(a)
            print("  test sub-{}: macro-AUC {:.3f}".format(held, a))
        print("  mean LOSO macro-AUC = {:.3f}  (chance 0.5)".format(np.nanmean(aucs)))
    else:
        print("(<3 patients or <2 shared states; full LOSO not defined)")

    # ---- richer pairwise transfer for the best-matched patient pair -------- #
    best = None
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            shared = states_by[subs[i]] & states_by[subs[j]]
            if best is None or len(shared) > len(best[2]):
                best = (subs[i], subs[j], shared)
    if best and len(best[2]) >= 3:
        a, b, st = best[0], best[1], sorted(best[2])
        print("\n=== richest pair: sub-{} <-> sub-{}  ({}-way {}) ===".format(a, b, len(st), st))
        ab = macro_auc(data[a][0], data[a][1], data[b][0], data[b][1], st)
        ba = macro_auc(data[b][0], data[b][1], data[a][0], data[a][1], st)
        print("  train sub-{} -> test sub-{}: macro-AUC {:.3f}".format(a, b, ab))
        print("  train sub-{} -> test sub-{}: macro-AUC {:.3f}".format(b, a, ba))
        print("  within sub-{}: {:.3f}   within sub-{}: {:.3f}".format(
            a, within(*data[a], st), b, within(*data[b], st)))
        print("  mean pairwise transfer macro-AUC = {:.3f}".format(np.nanmean([ab, ba])))

    print("\nDone.")


if __name__ == "__main__":
    main()
