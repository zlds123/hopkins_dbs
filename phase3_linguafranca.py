"""Firm up the "neural lingua franca": is the geometry of behavioral states shared across
patients, beyond chance?

The single Procrustes number (0.37 for sub-01↔07) needs two things to be trustworthy:
(1) every available subject pair, not just one, and (2) a null -- how low a disparity would
we get if the state geometry were random? We compute all pairwise Procrustes disparities on
the states each pair shares, plus a label-shuffle null per pair, and report whether the
observed geometry is shared beyond chance.

Run (dbs-ml env, after downloading extra subjects):
  python phase3_linguafranca.py --out-dir phase3_manifold
"""

import argparse
import csv
import os

import numpy as np

from nwb_dataset import good_channel_indices
from phase3_eval import build_epoch_dataset, find_default_file
from phase3_crosssubject import find_subject_files
from phase3_manifold import STATE_BANDS, T5_LABELS, drop_outliers, robust_scale, fit_pca


def centroids(X, y, states, k):
    Z = fit_pca(robust_scale(X), k=k)[1]
    return np.vstack([Z[np.asarray(y) == s].mean(axis=0) for s in states])


def procrustes_disparity(A, B):
    from scipy.spatial import procrustes
    return float(procrustes(A, B)[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--extra-root", default="ajile12-nwb-data")
    ap.add_argument("--out-dir", default="phase3_manifold")
    ap.add_argument("--epoch-window-sec", type=float, default=10.0)
    ap.add_argument("--epoch-max-per-label", type=int, default=150)
    ap.add_argument("--min-per-state", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    primary = args.nwb or find_default_file()
    os.makedirs(args.out_dir, exist_ok=True)
    files = find_subject_files(args.extra_root, primary)
    print("subjects:", list(files))

    # build per-subject state epochs
    subj = {}
    for sid, path in files.items():
        print("[state epochs] sub-{}".format(sid))
        try:
            with __import__("h5py").File(path, "r") as f:
                good = good_channel_indices(f)
            X, y, _ = build_epoch_dataset(path, good, STATE_BANDS, window_sec=args.epoch_window_sec,
                                          max_per_label=args.epoch_max_per_label,
                                          label_set=set(T5_LABELS), seed=args.seed, verbose=False)
            keep = drop_outliers(X)
            X, y = X[keep], np.asarray(y)[keep]
            present = [s for s in T5_LABELS if np.sum(y == s) >= args.min_per_state]
            if len(present) >= 3:
                subj[sid] = (X, y, present)
                print("  states: {}".format(present))
            else:
                print("  <3 well-sampled states; excluded")
        except Exception as e:  # noqa: BLE001
            print("  skip ({})".format(type(e).__name__))

    subs = list(subj)
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b = subs[i], subs[j]
            Xa, ya, pa = subj[a]
            Xb, yb, pb = subj[b]
            common = [s for s in T5_LABELS if s in pa and s in pb]
            if len(common) < 3:
                continue
            k = min(len(common) - 1, 6)
            Ca, Cb = centroids(Xa, ya, common, k), centroids(Xb, yb, common, k)
            obs = procrustes_disparity(Ca, Cb)
            # null: shuffle state labels within each subject
            null = []
            for _ in range(args.n_null):
                ya_s = rng.permutation(ya)
                yb_s = rng.permutation(yb)
                try:
                    null.append(procrustes_disparity(
                        centroids(Xa, ya_s, common, k), centroids(Xb, yb_s, common, k)))
                except Exception:  # noqa: BLE001
                    pass
            null = np.array(null)
            pval = float((null <= obs).mean()) if len(null) else float("nan")
            rows.append({"pair": "{}-{}".format(a, b), "n_states": len(common),
                        "disparity": round(obs, 3), "null_median": round(float(np.median(null)), 3),
                        "p_value": round(pval, 3),
                        "shared": "yes" if pval < 0.05 else "no"})
            print("  {}-{}: disparity={:.3f}  null median={:.3f}  p={:.3f}  [{}]".format(
                a, b, obs, np.median(null), pval, "SHARED" if pval < 0.05 else "n.s."))

    with open(os.path.join(args.out_dir, "linguafranca_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n================ LINGUA-FRANCA (firmed up) ================")
    if rows:
        shared = [r for r in rows if r["shared"] == "yes"]
        md = float(np.mean([r["disparity"] for r in rows]))
        print("{} subject pair(s) tested; {} shared beyond chance (p<0.05).".format(len(rows), len(shared)))
        print("mean observed disparity {:.3f} (lower = more shared geometry).".format(md))
    else:
        print("No subject pair had >=3 shared well-sampled states; cannot test (need more data).")
    print("==========================================================")
    print("\nDone. Outputs in", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
